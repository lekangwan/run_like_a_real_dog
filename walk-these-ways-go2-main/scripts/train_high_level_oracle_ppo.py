import argparse
import csv
import gc
import json
from pathlib import Path
import pickle as pkl
from types import SimpleNamespace
import time

import isaacgym

assert isaacgym
import torch

from go2_gym import MINI_GYM_ROOT_DIR
from go2_gym.envs.base.legged_robot_config import Cfg
from go2_gym.envs.go2.go2_config import config_go2
from go2_gym.envs.go2.velocity_tracking import VelocityTrackingEasyEnv
from go2_gym.envs.wrappers.history_wrapper import HistoryWrapper
from go2_gym.envs.wrappers.high_level_gait_wrapper import HighLevelGaitWrapper
from gait_conditions import DIRECTED_PUSH_INTERVAL_S, apply_condition_cfg
from gait_project_config import (
    MAINLINE_TASK_MAP,
    TRAIN_EDGE_RESET_MARGIN,
    TRAIN_MESH_TYPE,
    TRAIN_TELEPORT_THRESH,
    TRAIN_TERRAIN_SIZE,
)
from train_high_level_ppo import (
    ActorCritic,
    RolloutBuffer,
    append_metrics,
    find_logdir,
    load_low_level_policy,
    save_checkpoint,
)


GAIT_NAMES = ("pronking", "trotting", "bounding", "pacing")
GAIT_SHORT_NAMES = {
    "pronking": "pronk",
    "trotting": "trot",
    "bounding": "bound",
    "pacing": "pace",
}
STYLE_COEFS = {
    "none": 0.0,
    "mild": 0.15,
    "medium": 0.6,
}
BASE_METRIC_WEIGHTS = {
    "progress": 1.0,
    "yaw_tracking": 0.3,
    "orientation": 0.3,
    "lateral_drift": 0.8,
    "gait_stability": 0.4,
    "action_smoothness": 0.7,
    "action_magnitude": 0.6,
    "action_boundary_margin": 0.8,
    "survival": 2.0,
}
TASK_REWARD_FOCUS_WEIGHTS = {
    # ── v4: data-driven reward focus ──
    # progress / recovery_progress deliberately REMOVED from focus
    #   (base 1.0 is enough; 2.0 was causing pace collapse)
    # trot_contact_style / pace_contact_style / bound_contact_style
    #   deliberately REMOVED (were silently dropped before, now clean)
    "low_slip": {"slip": 1.2},               # was 0.8 — strongest anti-pace signal
    "low_vertical_bounce": {"vertical_bounce": 0.8},  # was 0.6
    "low_lateral_drift": {"lateral_drift": 0.8},
    # orientation_stability — split into 3 levels
    "orientation_stability":        {"orientation": 0.5},   # total 0.8  (mild)
    "orientation_stability_strong": {"orientation": 0.9},   # total 1.2  (rough, push)
    "orientation_stability_mild":   {"orientation": 0.3},   # total 0.6  (stones)
    "pitch_control": {"pitch_rate": 0.8, "orientation": 0.4},
    "low_roll_pitch_rate": {"roll_rate": 0.6, "pitch_rate": 0.6},
    "low_roll_rate": {"roll_rate": 0.8},
    "low_yaw_rate": {"yaw_rate": 0.6},                      # was 0.8
    "low_done_rate": {"survival": 1.0},                     # kept but unused in v4
    "foot_clearance": {"clearance": 0.6},                   # was 0.35
    "low_scuffing": {"clearance": 0.15},
    # retained but unused in v4 focus:
    "low_energy": {"energy": 0.6},
}


def task_reward_weights_from_focus(reward_focus):
    weights = {name: 0.0 for name in HighLevelGaitWrapper.TASK_REWARD_NAMES}
    for name, value in BASE_METRIC_WEIGHTS.items():
        weights[name] += value
    for token in str(reward_focus).split(","):
        token = token.strip()
        for name, value in TASK_REWARD_FOCUS_WEIGHTS.get(token, {}).items():
            weights[name] += value
    for name in weights:
        weights[name] = min(weights[name], 2.0)
    return [weights[name] for name in HighLevelGaitWrapper.TASK_REWARD_NAMES]


def read_task_specs(task_map_path, style_reward_scale=0.0):
    grouped = {}
    with open(task_map_path, newline="") as file:
        for row in csv.DictReader(file):
            if row["use_for_training"] != "yes":
                continue
            task_id = row["task_id"]
            group = grouped.setdefault(
                task_id,
                {
                    "task_id": task_id,
                    "condition": row["condition"],
                    "target_gait": row["target_gait"],
                    "style_reward_strength": row["style_reward_strength"],
                    "reward_focus": row.get("reward_focus", "progress,orientation_stability"),
                    "vx_values": [],
                },
            )
            group["vx_values"].append(float(row["vx"]))

    specs = []
    for task_id, item in grouped.items():
        vx_values = sorted(item["vx_values"])
        if len(vx_values) == 1:
            center = vx_values[0]
            if item["condition"] == "push_lateral":
                vx_low, vx_high = 1.2, 1.8
            elif item["condition"] == "stepping_stones_easy":
                vx_low, vx_high = 1.7, 2.0
            else:
                vx_low, vx_high = max(0.2, center - 0.2), min(2.0, center + 0.2)
        else:
            vx_low, vx_high = min(vx_values), max(vx_values)

        specs.append(
            SimpleNamespace(
                task_id=task_id,
                condition=item["condition"],
                target_gait=item["target_gait"],
                target_gait_id=GAIT_NAMES.index(item["target_gait"]),
                style_reward_strength=item["style_reward_strength"],
                selector_reference_coef=style_reward_scale * STYLE_COEFS[item["style_reward_strength"]],
                reward_focus=item["reward_focus"],
                task_reward_weights=task_reward_weights_from_focus(item["reward_focus"]),
                vx_values=vx_values,
                vx_low=vx_low,
                vx_high=vx_high,
            )
        )
    return specs


def build_env_assignment(specs, num_envs, device):
    if num_envs < len(specs):
        raise ValueError(f"num_envs={num_envs} is smaller than num_tasks={len(specs)}")

    base = num_envs // len(specs)
    remainder = num_envs % len(specs)
    task_ids = []
    conditions = []
    target_gait_ids = []
    selector_coefs = []
    vx_lows = []
    vx_highs = []
    push_axes = []
    task_reward_weights = []

    for task_index, spec in enumerate(specs):
        count = base + (1 if task_index < remainder else 0)
        task_ids.extend([task_index] * count)
        conditions.extend([spec.condition] * count)
        target_gait_ids.extend([spec.target_gait_id] * count)
        selector_coefs.extend([spec.selector_reference_coef] * count)
        vx_lows.extend([spec.vx_low] * count)
        vx_highs.extend([spec.vx_high] * count)
        push_axes.extend([1 if spec.condition == "push_lateral" else -1] * count)
        task_reward_weights.extend([spec.task_reward_weights] * count)

    return SimpleNamespace(
        task_ids=torch.tensor(task_ids, device=device, dtype=torch.long),
        conditions=conditions,
        target_gait_ids=torch.tensor(target_gait_ids, device=device, dtype=torch.long),
        selector_coefs=torch.tensor(selector_coefs, device=device, dtype=torch.float),
        vx_lows=torch.tensor(vx_lows, device=device, dtype=torch.float),
        vx_highs=torch.tensor(vx_highs, device=device, dtype=torch.float),
        push_axes=push_axes,
        task_reward_weights=torch.tensor(task_reward_weights, device=device, dtype=torch.float),
    )


def load_mixed_low_level_env(
    logdir,
    num_envs,
    render,
    conditions,
    push_axes,
    terrain_size,
    edge_reset_margin,
    teleport_thresh,
    mesh_type,
):
    config_go2(Cfg)
    with open(Path(logdir) / "parameters.pkl", "rb") as file:
        pkl_cfg = pkl.load(file)
        loaded_cfg = pkl_cfg["Cfg"]
        for key, value in loaded_cfg.items():
            if hasattr(Cfg, key):
                for key2, value2 in value.items():
                    setattr(getattr(Cfg, key), key2, value2)

    apply_condition_cfg(Cfg, "flat")
    Cfg.env.num_envs = num_envs
    Cfg.env.num_recording_envs = 0
    Cfg.terrain.min_init_terrain_level = 0
    Cfg.terrain.max_init_terrain_level = 0
    Cfg.terrain.num_rows = 1
    Cfg.terrain.num_cols = max(1, num_envs)
    Cfg.terrain.border_size = 0
    Cfg.terrain.center_robots = False
    Cfg.terrain.measure_heights = True
    Cfg.terrain.mesh_type = mesh_type
    Cfg.terrain.terrain_length = terrain_size
    Cfg.terrain.terrain_width = terrain_size
    Cfg.terrain.teleport_thresh = teleport_thresh
    Cfg.terrain.edge_reset_robots = True
    Cfg.terrain.edge_reset_margin = edge_reset_margin
    Cfg.terrain.teleport_robots = False
    Cfg.terrain.env_conditions = list(conditions)
    Cfg.asset.flip_visual_attachments = True

    Cfg.domain_rand.push_robots = any(axis >= 0 for axis in push_axes)
    Cfg.domain_rand.push_interval_s = DIRECTED_PUSH_INTERVAL_S
    Cfg.domain_rand.max_push_vel_xy = 1.5
    Cfg.domain_rand.push_axis_by_env = list(push_axes)

    print(
        f"Creating mixed oracle env: num_envs={num_envs}, mesh_type={mesh_type}, "
        f"terrain_size={terrain_size:.1f}m, edge_reset_margin={edge_reset_margin:.1f}m"
    )
    env = VelocityTrackingEasyEnv(sim_device="cuda:0", headless=not render, cfg=Cfg)
    return HistoryWrapper(env)


def sample_vx(lows, highs):
    return torch.rand_like(lows) * (highs - lows) + lows


class OracleConditionHighLevelEnv:
    def __init__(
        self,
        specs,
        logdir,
        low_policy,
        num_envs,
        render,
        oracle_condition_obs=True,
        terrain_size=TRAIN_TERRAIN_SIZE,
        edge_reset_margin=TRAIN_EDGE_RESET_MARGIN,
        teleport_thresh=TRAIN_TELEPORT_THRESH,
        mesh_type=TRAIN_MESH_TYPE,
    ):
        self.specs = specs
        self.oracle_condition_obs = oracle_condition_obs

        placeholder_device = "cpu"
        self.assignment = build_env_assignment(specs, num_envs, placeholder_device)
        low_env = load_mixed_low_level_env(
            logdir,
            num_envs,
            render,
            self.assignment.conditions,
            self.assignment.push_axes,
            terrain_size,
            edge_reset_margin,
            teleport_thresh,
            mesh_type,
        )
        self.env = HighLevelGaitWrapper(
            low_env,
            low_policy,
            record_reward_terms=True,
            selector_reference_coef=0.0,
        )
        self.device = self.env.device
        self.assignment = build_env_assignment(specs, num_envs, self.device)
        self.env.set_target_gait(self.assignment.target_gait_ids, self.assignment.selector_coefs)
        self.env.set_task_reward_weights(self.assignment.task_reward_weights)
        self.vx_cmd = sample_vx(self.assignment.vx_lows, self.assignment.vx_highs)

        self.num_envs = num_envs
        self.num_tasks = len(specs)
        self.num_gaits = self.env.num_gaits
        self.num_behavior_actions = self.env.num_behavior_actions
        self.num_high_level_actions = self.env.num_high_level_actions
        self.base_obs_dim = self.env.num_high_level_obs_history
        self.obs_dim = self.base_obs_dim + (self.num_tasks if oracle_condition_obs else 0)
        self.condition_one_hot = torch.nn.functional.one_hot(
            self.assignment.task_ids,
            num_classes=self.num_tasks,
        ).to(device=self.device, dtype=torch.float)

    def reset(self):
        obs = self.env.reset()
        self.env.set_velocity_command(self.vx_cmd, 0.0, 0.0)
        return self._augment_obs(obs)

    def step(self, actions):
        obs, reward, done, info = self.env.step(actions)
        done_ids = done.nonzero(as_tuple=False).flatten()
        if len(done_ids) > 0:
            self.vx_cmd[done_ids] = sample_vx(
                self.assignment.vx_lows[done_ids],
                self.assignment.vx_highs[done_ids],
            )
        self.env.set_velocity_command(self.vx_cmd, 0.0, 0.0)
        return self._augment_obs(obs), reward, done, info

    def _augment_obs(self, obs):
        if not self.oracle_condition_obs:
            return obs
        return torch.cat((obs, self.condition_one_hot), dim=-1)

    def get_base_obs(self):
        """Return raw proprioceptive history WITHOUT task one-hot (510D)."""
        return self.env.obs_history.detach()

    def get_high_level_privileged_obs(self):
        """Delegate to HighLevelGaitWrapper for privileged environment info (14D)."""
        return self.env.get_high_level_privileged_obs()

    def command_vx(self):
        return self.env.commands[:, 0]

    def measured_vx(self):
        return self.env.base_lin_vel[:, 0]

    def mapped_action_stats(self, actions):
        mapped = self.env._map_action(actions)
        return {
            "frequency": mapped["frequency"].detach(),
            "duration": mapped["duration"].detach(),
            "footswing_height": mapped["footswing_height"].detach(),
            "stance_width": mapped["stance_width"].detach(),
            "body_pitch": mapped["body_pitch"].detach(),
        }


def add_gait_metrics(metrics, env, actions):
    gait_ids = torch.argmax(actions[:, : env.num_gaits], dim=-1)
    residual = actions[:, env.num_gaits :]
    if actions.shape[0] % env.num_envs != 0:
        raise ValueError(f"Expected actions to be a multiple of num_envs={env.num_envs}")
    repeats = actions.shape[0] // env.num_envs
    action_seq = actions.reshape(repeats, env.num_envs, -1)
    gait_seq = torch.argmax(action_seq[:, :, : env.num_gaits], dim=-1)
    task_ids = env.assignment.task_ids.repeat(repeats)
    last_actions = actions[-env.num_envs :]

    for gait_id, gait_name in enumerate(GAIT_NAMES):
        metrics[f"gait_{GAIT_SHORT_NAMES[gait_name]}_ratio"] = (gait_ids == gait_id).float().mean().item()

    if repeats > 1:
        switch_seq = (gait_seq[1:] != gait_seq[:-1]).float()
        metrics["gait_switch_rate"] = switch_seq.mean().item()
    else:
        switch_seq = None
        metrics["gait_switch_rate"] = 0.0

    for task_index, spec in enumerate(env.specs):
        mask = task_ids == task_index
        task_gait_ids = gait_ids[mask]
        env_mask = env.assignment.task_ids == task_index
        short_target = GAIT_SHORT_NAMES[spec.target_gait]
        metrics[f"{spec.task_id}_{short_target}_ratio"] = (
            task_gait_ids == spec.target_gait_id
        ).float().mean().item()
        metrics[f"{spec.task_id}_count"] = int(mask.sum().item() / repeats)
        if switch_seq is not None and torch.any(env_mask):
            metrics[f"{spec.task_id}_gait_switch_rate"] = switch_seq[:, env_mask].mean().item()
        else:
            metrics[f"{spec.task_id}_gait_switch_rate"] = 0.0
        for gait_id, gait_name in enumerate(GAIT_NAMES):
            metrics[f"{spec.task_id}_{GAIT_SHORT_NAMES[gait_name]}_ratio"] = (
                task_gait_ids == gait_id
            ).float().mean().item()

    mapped = env.mapped_action_stats(last_actions)
    for key, value in mapped.items():
        metrics[f"{key}_mean"] = value.mean().item()
    metrics["action_clip_rate"] = (torch.abs(residual) > 0.98).float().mean().item()

    rollout_mapped = env.mapped_action_stats(actions)
    for task_index, spec in enumerate(env.specs):
        mask = task_ids == task_index
        if not torch.any(mask):
            continue
        task_residual = residual[mask]
        metrics[f"{spec.task_id}_action_clip_rate"] = (
            torch.abs(task_residual) > 0.98
        ).float().mean().item()
        for key, value in rollout_mapped.items():
            metrics[f"{spec.task_id}_{key}_mean"] = value[mask].mean().item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", default="gait-conditioned-agility/pretrain-go2/train")
    parser.add_argument("--run-index", type=int, default=0)
    parser.add_argument("--task-map", default=str(MAINLINE_TASK_MAP))
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--num-steps", type=int, default=32)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--mini-batches", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lam", type=float, default=0.95)
    parser.add_argument("--clip", type=float, default=0.2)
    parser.add_argument("--entropy-coef", type=float, default=0.003)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--save-dir", default=str(Path(MINI_GYM_ROOT_DIR) / "runs" / "high_level_oracle_gait"))
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--save-interval", type=int, default=50)
    parser.add_argument("--no-oracle-condition-obs", action="store_true")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--terrain-size", type=float, default=TRAIN_TERRAIN_SIZE)
    parser.add_argument("--edge-reset-margin", type=float, default=TRAIN_EDGE_RESET_MARGIN)
    parser.add_argument("--teleport-thresh", type=float, default=TRAIN_TELEPORT_THRESH)
    parser.add_argument("--mesh-type", default=TRAIN_MESH_TYPE, choices=["heightfield", "trimesh"])
    parser.add_argument(
        "--task-reward-coef",
        type=float,
        default=1.0,
        help="Deprecated compatibility option. Reward is now a normalized weighted metric average.",
    )
    parser.add_argument(
        "--style-reward-scale",
        type=float,
        default=0.0,
        help="Scale for gait-label selector reward. Default 0 disables hard gait-label shaping.",
    )
    parser.add_argument("--z-dim", type=int, default=16, help="Environment latent dimension for RMA distillation.")
    parser.add_argument(
        "--priv-dim",
        type=int,
        default=14,
        help="Dimension of high-level privileged observation (teacher input).",
    )
    parser.add_argument(
        "--adaptation-coef",
        type=float,
        default=0.1,
        help="Weight of adaptation MSE loss in total training loss.",
    )
    parser.add_argument(
        "--selector-only",
        action="store_true",
        help="Train only the gait categorical head; execute zero continuous residuals and exclude residual log-probs.",
    )
    args = parser.parse_args()

    logdir = find_logdir(args.label, args.run_index)
    specs = read_task_specs(args.task_map, style_reward_scale=args.style_reward_scale)
    low_policy = load_low_level_policy(logdir)
    env = OracleConditionHighLevelEnv(
        specs,
        logdir,
        low_policy,
        args.num_envs,
        args.render,
        oracle_condition_obs=not args.no_oracle_condition_obs,
        terrain_size=args.terrain_size,
        edge_reset_margin=args.edge_reset_margin,
        teleport_thresh=args.teleport_thresh,
        mesh_type=args.mesh_type,
    )
    device = env.device

    model = ActorCritic(
        env.obs_dim,
        env.num_gaits,
        env.num_behavior_actions,
        base_obs_dim=env.base_obs_dim,
        priv_dim=args.priv_dim,
        z_dim=args.z_dim,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    use_rma = args.z_dim > 0
    obs = env.reset()
    base_obs = env.get_base_obs() if use_rma else None

    # ── observation dims after RMA augmentation ──
    aug_obs_dim = env.obs_dim + args.z_dim  # = env.obs_dim when z_dim == 0

    run_name = args.run_name or time.strftime("%Y%m%d_%H%M%S_oracle")
    run_dir = Path(args.save_dir) / run_name
    metrics_path = run_dir / "metrics.csv"
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "args.json", "w") as file:
        args_dict = vars(args).copy()
        args_dict["tasks"] = [vars(spec).copy() for spec in specs]
        args_dict["obs_dim"] = env.obs_dim
        args_dict["base_obs_dim"] = env.base_obs_dim
        args_dict["aug_obs_dim"] = aug_obs_dim
        json.dump(args_dict, file, indent=2)

    print(f"Saving oracle high-level checkpoints to: {run_dir}")
    print(
        f"obs_dim={env.obs_dim} (+z={args.z_dim} → aug_obs_dim={aug_obs_dim}) "
        f"base_obs_dim={env.base_obs_dim} "
        f"priv_dim={args.priv_dim} "
        f"action_dim={env.num_high_level_actions} "
        f"selector_only={args.selector_only}"
    )
    for task_index, spec in enumerate(specs):
        count = int((env.assignment.task_ids == task_index).sum().item())
        print(
            f"task={spec.task_id} condition={spec.condition} target={spec.target_gait} "
            f"envs={count} vx=[{spec.vx_low:.2f},{spec.vx_high:.2f}] "
            f"style_coef={spec.selector_reference_coef:.2f} focus={spec.reward_focus}"
        )

    for iteration in range(args.iterations):
        progress = iteration / max(1, args.iterations - 1)

        # ── phase-dependent alpha: z_input = α * z_teacher + (1-α) * z_student ──
        if progress < 0.25:
            alpha_val = 1.0  # teacher-only
        elif progress < 0.75:
            alpha_val = 1.0 - (progress - 0.25) / 0.5  # linear anneal 1→0
        else:
            alpha_val = 0.0  # student-only

        buffer = RolloutBuffer(
            args.num_steps,
            env.num_envs,
            aug_obs_dim,
            env.num_high_level_actions,
            device,
        )
        reward_sum = 0.0
        done_sum = 0.0
        edge_reset_sum = 0.0
        vx_error_sum = 0.0
        weighted_metric_reward_sum = 0.0
        selector_reference_penalty_sum = 0.0
        orientation_penalty_sum = 0.0
        slip_penalty_sum = 0.0
        lateral_velocity_penalty_sum = 0.0
        lateral_position_penalty_sum = 0.0
        clearance_reward_sum = 0.0
        gait_switch_penalty_sum = 0.0
        action_boundary_penalty_sum = 0.0
        adaptation_loss_sum = 0.0
        metric_score_sums = {name: 0.0 for name in HighLevelGaitWrapper.TASK_REWARD_NAMES}
        actual_actions = []

        for _ in range(args.num_steps):
            # ── RMA: compute z_input (skip when z_dim == 0) ──
            if use_rma:
                priv_obs = env.get_high_level_privileged_obs()
                base_obs_step = env.get_base_obs()
                with torch.inference_mode():
                    z_teacher = model.encode_teacher(priv_obs)
                    z_student = model.encode_student(base_obs_step)
                z_input = alpha_val * z_teacher + (1.0 - alpha_val) * z_student
                obs_with_z = torch.cat((obs, z_input), dim=-1)
            else:
                obs_with_z = obs

            with torch.inference_mode():
                if args.selector_only:
                    action, log_prob, value = model.act_selector_only(obs_with_z)
                else:
                    action, log_prob, value = model.act(obs_with_z)
                next_obs, reward, done, info = env.step(action)

            buffer.add(obs_with_z, action, log_prob, reward, done, value)
            executed_action = info.get("executed_high_level_action", env.env.high_level_action)
            actual_actions.append(executed_action.detach().clone())
            reward_sum += reward.mean().item()
            done_sum += done.float().mean().item()
            terms = info.get("high_level_reward_terms", {})
            if "edge_reset" in terms:
                edge_reset_sum += terms["edge_reset"].mean().item()
            if "weighted_metric_reward" in terms:
                weighted_metric_reward_sum += terms["weighted_metric_reward"].mean().item()
            if "selector_reference_penalty" in terms:
                selector_reference_penalty_sum += terms["selector_reference_penalty"].mean().item()
            if "orientation_penalty" in terms:
                orientation_penalty_sum += terms["orientation_penalty"].mean().item()
            if "slip_penalty" in terms:
                slip_penalty_sum += terms["slip_penalty"].mean().item()
            if "lateral_velocity_penalty" in terms:
                lateral_velocity_penalty_sum += terms["lateral_velocity_penalty"].mean().item()
            if "lateral_position_penalty" in terms:
                lateral_position_penalty_sum += terms["lateral_position_penalty"].mean().item()
            if "clearance_reward" in terms:
                clearance_reward_sum += terms["clearance_reward"].mean().item()
            if "gait_switch_penalty" in terms:
                gait_switch_penalty_sum += terms["gait_switch_penalty"].mean().item()
            if "action_boundary_penalty" in terms:
                action_boundary_penalty_sum += terms["action_boundary_penalty"].mean().item()
            for name in HighLevelGaitWrapper.TASK_REWARD_NAMES:
                key = f"score_{name}"
                if key in terms:
                    metric_score_sums[name] += terms[key].mean().item()
            vx_error_sum += torch.abs(env.measured_vx() - env.command_vx()).mean().item()
            obs = next_obs
            if use_rma:
                base_obs = base_obs_step

        with torch.inference_mode():
            if use_rma:
                priv_obs_final = env.get_high_level_privileged_obs()
                base_obs_final = env.get_base_obs()
                z_student_final = model.encode_student(base_obs_final)
                if alpha_val > 0.0:
                    z_teacher_final = model.encode_teacher(priv_obs_final)
                    z_final = alpha_val * z_teacher_final + (1.0 - alpha_val) * z_student_final
                else:
                    z_final = z_student_final
                obs_final_with_z = torch.cat((obs, z_final), dim=-1)
            else:
                obs_final_with_z = obs
            last_value = model.critic(obs_final_with_z).squeeze(-1)

        buffer.compute_returns(last_value, args.gamma, args.lam)

        flat_obs, flat_actions, flat_log_probs, flat_returns, flat_advantages, _ = buffer.flat()
        actual_actions_flat = torch.stack(actual_actions, dim=0).reshape(-1, env.num_high_level_actions)
        batch_size = flat_obs.shape[0]
        mini_batch_size = max(1, batch_size // args.mini_batches)

        # Pre-compute privileged obs for the full batch once per iteration
        # (privileged obs are static per env, so repeat for num_steps)
        priv_obs_flat = env.get_high_level_privileged_obs().repeat(args.num_steps, 1) if use_rma else None

        value_loss_epoch = 0.0
        policy_loss_epoch = 0.0
        entropy_epoch = 0.0
        adaptation_loss_epoch = 0.0
        updates = 0

        for _ in range(args.epochs):
            indices = torch.randperm(batch_size, device=device)
            for start in range(0, batch_size, mini_batch_size):
                idx = indices[start : start + mini_batch_size]
                if args.selector_only:
                    new_log_prob, entropy, value = model.evaluate_actions_selector_only(
                        flat_obs[idx],
                        flat_actions[idx],
                    )
                else:
                    new_log_prob, entropy, value = model.evaluate_actions(flat_obs[idx], flat_actions[idx])
                ratio = torch.exp(new_log_prob - flat_log_probs[idx])
                surrogate_1 = ratio * flat_advantages[idx]
                surrogate_2 = torch.clamp(ratio, 1.0 - args.clip, 1.0 + args.clip) * flat_advantages[idx]
                policy_loss = -torch.min(surrogate_1, surrogate_2).mean()
                value_loss = (flat_returns[idx] - value).pow(2).mean()
                entropy_loss = entropy.mean()

                # ── RMA adaptation loss (on the same mini-batch, skip when z_dim == 0) ──
                if use_rma:
                    mini_base_obs = flat_obs[idx, : env.base_obs_dim]
                    mini_priv_obs = priv_obs_flat[idx]
                    z_teacher_mini = model.encode_teacher(mini_priv_obs)
                    z_student_mini = model.encode_student(mini_base_obs)
                    adaptation_loss = torch.nn.functional.mse_loss(z_student_mini, z_teacher_mini.detach())
                else:
                    adaptation_loss = torch.tensor(0.0, device=device)

                loss = (
                    policy_loss
                    + args.value_coef * value_loss
                    - args.entropy_coef * entropy_loss
                    + args.adaptation_coef * adaptation_loss
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                value_loss_epoch += value_loss.item()
                policy_loss_epoch += policy_loss.item()
                entropy_epoch += entropy_loss.item()
                adaptation_loss_epoch += adaptation_loss.item()
                updates += 1

        # ── compute z statistics for logging (skip when z_dim == 0) ──
        if use_rma:
            base_obs_all = flat_obs[:, : env.base_obs_dim]
            priv_obs_all = priv_obs_flat
            with torch.inference_mode():
                z_teacher_all = model.encode_teacher(priv_obs_all)
                z_student_all = model.encode_student(base_obs_all)
            z_teacher_mean = z_teacher_all.mean().item()
            z_teacher_std = z_teacher_all.std().item()
            z_student_mean = z_student_all.mean().item()
            z_student_std = z_student_all.std().item()
            z_error = torch.nn.functional.mse_loss(z_student_all, z_teacher_all).item()
        else:
            z_teacher_mean = z_teacher_std = z_student_mean = z_student_std = z_error = float("nan")

        metrics = {
            "iteration": iteration,
            "reward": reward_sum / args.num_steps,
            "done_rate": done_sum / args.num_steps,
            "edge_reset_rate": edge_reset_sum / args.num_steps,
            "vx_err": vx_error_sum / args.num_steps,
            "weighted_metric_reward": weighted_metric_reward_sum / args.num_steps,
            "selector_reference_penalty": selector_reference_penalty_sum / args.num_steps,
            "orientation_penalty": orientation_penalty_sum / args.num_steps,
            "slip_penalty": slip_penalty_sum / args.num_steps,
            "lateral_velocity_penalty": lateral_velocity_penalty_sum / args.num_steps,
            "lateral_position_penalty": lateral_position_penalty_sum / args.num_steps,
            "clearance_reward": clearance_reward_sum / args.num_steps,
            "gait_switch_penalty": gait_switch_penalty_sum / args.num_steps,
            "action_boundary_penalty": action_boundary_penalty_sum / args.num_steps,
            "policy_loss": policy_loss_epoch / updates,
            "value_loss": value_loss_epoch / updates,
            "entropy": entropy_epoch / updates,
            "adaptation_loss": adaptation_loss_epoch / updates,
            "alpha": alpha_val,
            "z_teacher_mean": z_teacher_mean,
            "z_teacher_std": z_teacher_std,
            "z_student_mean": z_student_mean,
            "z_student_std": z_student_std,
            "z_error": z_error,
            "log_std_mean": model.log_std.detach().mean().item(),
        }
        for name, value in metric_score_sums.items():
            metrics[f"score_{name}"] = value / args.num_steps
        add_gait_metrics(metrics, env, actual_actions_flat.detach())
        append_metrics(metrics_path, metrics)

        if iteration % args.save_interval == 0 or iteration == args.iterations - 1:
            save_checkpoint(run_dir / "checkpoints" / f"high_level_{iteration:06d}.pt", model, optimizer, iteration)

        print(
            f"iter={iteration:04d} reward={metrics['reward']:.3f} "
            f"done={metrics['done_rate']:.3f} edge={metrics['edge_reset_rate']:.3f} "
            f"vx_err={metrics['vx_err']:.3f} "
            f"lat_pos={metrics['lateral_position_penalty']:.3f} "
            f"switch={metrics['gait_switch_rate']:.3f} "
            f"metric={metrics['weighted_metric_reward']:.3f} "
            f"adapt={metrics['adaptation_loss']:.4f} "
            f"z_err={metrics['z_error']:.4f} "
            f"flat_trot={metrics.get('flat_trot_efficiency_trot_ratio', float('nan')):.2f} "
            f"push_pace={metrics.get('push_lateral_pace_recovery_pace_ratio', float('nan')):.2f} "
            f"stone_bound={metrics.get('stepping_stones_easy_bound_highspeed_bound_ratio', float('nan')):.2f}"
        )

        del (
            buffer,
            flat_obs,
            flat_actions,
            flat_log_probs,
            flat_returns,
            flat_advantages,
            actual_actions_flat,
            actual_actions,
        )
        if use_rma:
            del base_obs_all, priv_obs_all, priv_obs_flat
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize(device)
            torch.cuda.empty_cache()
            torch.cuda.synchronize(device)


if __name__ == "__main__":
    main()
