import argparse
import csv
import json
from pathlib import Path
import random
import subprocess
import sys
import time

import isaacgym

assert isaacgym
import numpy as np
import torch

from go2_gym.envs.wrappers.high_level_reward_metrics import CANONICAL_REWARD_NAMES
from gait_project_config import (
    MAINLINE_TASK_MAP,
    TRAIN_EDGE_RESET_MARGIN,
    TRAIN_MESH_TYPE,
    TRAIN_TELEPORT_THRESH,
    TRAIN_TERRAIN_SIZE,
)
from train_high_level_oracle_ppo import (
    GAIT_NAMES,
    GAIT_SHORT_NAMES,
    OracleConditionHighLevelEnv,
    REWARD_PROFILE_CHOICES,
    parse_residual_action_mask,
    residual_mask_description,
    read_task_specs,
)
from train_high_level_ppo import ActorCritic, find_logdir, load_low_level_policy


DEFAULT_EVAL = (
    "flat_trot_efficiency:0.5,"
    "flat_trot_efficiency:1.0,"
    "flat_trot_efficiency:1.5,"
    "flat_trot_efficiency:2.0,"
    "ramp_up_trot_robustness:0.5,"
    "ramp_up_trot_robustness:1.0,"
    "ramp_up_trot_robustness:1.5,"
    "ramp_up_trot_robustness:2.0,"
    "rough_slope_trot_robustness:0.5,"
    "rough_slope_trot_robustness:1.0,"
    "rough_slope_trot_robustness:1.5,"
    "rough_slope_trot_robustness:2.0,"
    "push_lateral_pace_recovery:1.2,"
    "push_lateral_pace_recovery:1.5,"
    "push_lateral_pace_recovery:1.8,"
    "stepping_stones_easy_bound_highspeed:1.7,"
    "stepping_stones_easy_bound_highspeed:2.0"
)

QUICK_EVAL = (
    "flat_trot_efficiency:1.0,"
    "ramp_up_trot_robustness:1.0,"
    "rough_slope_trot_robustness:1.0,"
    "push_lateral_pace_recovery:1.5,"
    "stepping_stones_easy_bound_highspeed:2.0"
)

SCORE_KEYS = tuple(f"score_{name}" for name in CANONICAL_REWARD_NAMES)
RAW_TERM_KEYS = (
    "velocity_reward",
    "yaw_reward",
    "orientation_penalty",
    "pitch_rate_penalty",
    "roll_rate_penalty",
    "yaw_rate_penalty",
    "lateral_velocity_penalty",
    "lateral_position_penalty",
    "vertical_velocity_penalty",
    "slip_penalty",
    "contact_slip_penalty",
    "torque_penalty",
    "mechanical_power_abs",
    "transport_cost_proxy",
    "impact_velocity_rms",
    "scuffing_ratio",
    "clearance_reward",
    "gait_switch_penalty",
    "action_delta_penalty",
    "continuous_action_penalty",
    "action_boundary_penalty",
    "fall_penalty",
    "weighted_metric_reward",
)


def latest_checkpoint(run_dir):
    checkpoints = sorted(Path(run_dir).glob("checkpoints/high_level_*.pt"))
    if not checkpoints:
        raise FileNotFoundError(f"No high_level_*.pt checkpoints found under: {run_dir}")
    return checkpoints[-1]


def checkpoint_iteration_from_path(path):
    stem = Path(path).stem
    try:
        return int(stem.rsplit("_", 1)[1])
    except (IndexError, ValueError):
        return -1


def load_run_args(run_dir):
    args_path = Path(run_dir) / "args.json"
    if not args_path.exists():
        return {}
    with args_path.open() as file:
        return json.load(file)


def load_model(checkpoint_path, env, run_args):
    checkpoint = torch.load(checkpoint_path, map_location=env.device)
    obs_dim = int(run_args.get("obs_dim", env.obs_dim))
    model = ActorCritic(
        obs_dim,
        env.num_gaits,
        env.num_behavior_actions,
        base_obs_dim=env.base_obs_dim,
        priv_dim=int(run_args.get("priv_dim", 14)),
        z_dim=int(run_args.get("z_dim", 16)),
        selector_latent_cmd_only=bool(run_args.get("selector_latent_cmd_only", False)),
        physical_aux_dim=int(run_args.get("physical_aux_dim", 0)),
        selector_physical_state_input=bool(run_args.get("selector_physical_state_input", False)),
        adaptation_temporal_summary=bool(run_args.get("adaptation_temporal_summary", False)),
        gait_conditioned_residuals=bool(run_args.get("gait_conditioned_residuals", False)),
        gait_input_residuals=bool(run_args.get("gait_input_residuals", False)),
    ).to(env.device)
    model.load_state_dict(checkpoint["model"])
    residual_mask = run_args.get("residual_action_mask")
    if residual_mask is None:
        residual_mask = parse_residual_action_mask(run_args.get("residual_train_dims", "all"), device=env.device)
    else:
        residual_mask = torch.tensor(residual_mask, device=env.device)
    model.set_residual_action_mask(residual_mask)
    model.eval()
    return model, int(checkpoint.get("iteration", -1))


def parse_eval_items(text, specs):
    by_task = {spec.task_id: spec for spec in specs}
    items = []
    for raw_item in text.split(","):
        raw_item = raw_item.strip()
        if not raw_item:
            continue
        if ":" in raw_item:
            task_id, vx_text = raw_item.split(":", 1)
            vx = float(vx_text)
        else:
            task_id = raw_item
            if task_id not in by_task:
                raise ValueError(f"Unknown task_id={task_id!r}. Choices: {sorted(by_task)}")
            spec = by_task[task_id]
            vx = 0.5 * (spec.vx_low + spec.vx_high)
        if task_id not in by_task:
            raise ValueError(f"Unknown task_id={task_id!r}. Choices: {sorted(by_task)}")
        items.append((by_task[task_id], specs.index(by_task[task_id]), vx))
    if not items:
        raise ValueError("No eval items requested")
    return items


def set_fixed_vx(env, vx):
    env.vx_cmd[:] = vx
    env.env.set_velocity_command(env.vx_cmd, 0.0, 0.0)


def force_gait_action(env, gait_name):
    """Build the default-template action for one fixed gait family."""
    gait_id = GAIT_NAMES.index(gait_name)
    action = torch.zeros(env.num_envs, env.num_high_level_actions, device=env.device)
    action[:, gait_id] = 1.0
    return action


def augment_for_checkpoint(obs, task_index, num_tasks, use_task_onehot):
    if not use_task_onehot:
        return obs
    one_hot = torch.zeros(obs.shape[0], num_tasks, device=obs.device, dtype=obs.dtype)
    one_hot[:, task_index] = 1.0
    return torch.cat((obs, one_hot), dim=-1)


def append_command_vx_obs(obs, cmd_vx, enabled):
    if not enabled:
        return obs
    return torch.cat((obs, cmd_vx[:, None].to(dtype=obs.dtype)), dim=-1)


def make_stats(num_envs):
    return {
        "steps": 0,
        "reward_sum": 0.0,
        "done_sum": 0.0,
        "vx_err_sum": 0.0,
        "lateral_offset_sum": 0.0,
        "action_clip_sum": 0.0,
        "requested_residual_abs_sum": 0.0,
        "requested_residual_sq_sum": 0.0,
        "requested_residual_boundary_sum": 0.0,
        "executed_residual_abs_sum": 0.0,
        "executed_residual_sq_sum": 0.0,
        "executed_residual_boundary_sum": 0.0,
        "frequency_sum": 0.0,
        "duration_sum": 0.0,
        "footswing_height_sum": 0.0,
        "stance_width_sum": 0.0,
        "body_pitch_sum": 0.0,
        "score_sums": {key: 0.0 for key in SCORE_KEYS},
        "raw_term_sums": {key: 0.0 for key in RAW_TERM_KEYS},
        "gait_counts": {name: 0 for name in GAIT_NAMES},
        "last_gait_ids": torch.full((num_envs,), -1, dtype=torch.long),
        "dwell_lengths": [],
        "current_dwell": torch.zeros(num_envs, dtype=torch.long),
        "switch_count": 0,
    }


def update_dwell(stats, gait_ids):
    gait_ids_cpu = gait_ids.detach().cpu()
    last = stats["last_gait_ids"]
    current = stats["current_dwell"]
    first = last < 0
    same = gait_ids_cpu == last
    switched = (~first) & (~same)
    if torch.any(switched):
        stats["dwell_lengths"].extend(current[switched].tolist())
        stats["switch_count"] += int(switched.sum().item())
    current[first | switched] = 1
    current[(~first) & same] += 1
    last[:] = gait_ids_cpu


def add_step_stats(stats, env, action, reward, done, info, requested_action=None):
    actual = info.get("executed_high_level_action", action)
    requested = action if requested_action is None else requested_action
    mapped = env.env._map_action(actual)
    gait_ids = torch.argmax(mapped["selector_weights"], dim=-1)
    update_dwell(stats, gait_ids)

    for gait_id, gait_name in enumerate(GAIT_NAMES):
        stats["gait_counts"][gait_name] += int((gait_ids == gait_id).sum().item())

    terms = info.get("high_level_reward_terms", {})
    lateral_offset = torch.abs(env.env._compute_lateral_offset())
    vx_err = torch.abs(env.measured_vx() - env.command_vx())
    executed_residual = actual[:, env.num_gaits :]
    requested_residual = requested[:, env.num_gaits :]

    n = env.num_envs
    stats["steps"] += n
    stats["reward_sum"] += float(reward.sum().item())
    stats["done_sum"] += float(done.float().sum().item())
    stats["vx_err_sum"] += float(vx_err.sum().item())
    stats["lateral_offset_sum"] += float(lateral_offset.sum().item())
    stats["action_clip_sum"] += float((torch.abs(executed_residual) > 0.98).float().mean(dim=1).sum().item())
    stats["requested_residual_abs_sum"] += float(torch.abs(requested_residual).mean(dim=1).sum().item())
    stats["requested_residual_sq_sum"] += float((requested_residual**2).mean(dim=1).sum().item())
    stats["requested_residual_boundary_sum"] += float(
        (torch.abs(requested_residual) > 0.85).float().mean(dim=1).sum().item()
    )
    stats["executed_residual_abs_sum"] += float(torch.abs(executed_residual).mean(dim=1).sum().item())
    stats["executed_residual_sq_sum"] += float((executed_residual**2).mean(dim=1).sum().item())
    stats["executed_residual_boundary_sum"] += float(
        (torch.abs(executed_residual) > 0.85).float().mean(dim=1).sum().item()
    )
    stats["frequency_sum"] += float(mapped["frequency"].sum().item())
    stats["duration_sum"] += float(mapped["duration"].sum().item())
    stats["footswing_height_sum"] += float(mapped["footswing_height"].sum().item())
    stats["stance_width_sum"] += float(mapped["stance_width"].sum().item())
    stats["body_pitch_sum"] += float(mapped["body_pitch"].sum().item())
    for key in SCORE_KEYS:
        if key in terms:
            stats["score_sums"][key] += float(terms[key].sum().item())
    for key in RAW_TERM_KEYS:
        if key in terms:
            stats["raw_term_sums"][key] += float(terms[key].sum().item())


def finalize_stats(stats, task_id, condition, target_gait, vx, forced_gait=None):
    if torch.any(stats["last_gait_ids"] >= 0):
        stats["dwell_lengths"].extend(stats["current_dwell"][stats["last_gait_ids"] >= 0].tolist())
    steps = max(1, stats["steps"])
    dwell = stats["dwell_lengths"]
    row = {
        "task_id": task_id,
        "condition": condition,
        "target_gait": target_gait,
        "control_mode": f"forced_{forced_gait}" if forced_gait else "adaptive",
        "forced_gait": forced_gait or "",
        "cmd_vx": vx,
        "samples": stats["steps"],
        "reward_mean": stats["reward_sum"] / steps,
        "done_rate": stats["done_sum"] / steps,
        "vx_err_mean": stats["vx_err_sum"] / steps,
        "lateral_offset_mean": stats["lateral_offset_sum"] / steps,
        "action_clip_rate": stats["action_clip_sum"] / steps,
        "requested_residual_abs_mean": stats["requested_residual_abs_sum"] / steps,
        "requested_residual_sq_mean": stats["requested_residual_sq_sum"] / steps,
        "requested_residual_boundary_rate": stats["requested_residual_boundary_sum"] / steps,
        "executed_residual_abs_mean": stats["executed_residual_abs_sum"] / steps,
        "executed_residual_sq_mean": stats["executed_residual_sq_sum"] / steps,
        "executed_residual_boundary_rate": stats["executed_residual_boundary_sum"] / steps,
        "gait_switch_rate": stats["switch_count"] / max(1, stats["steps"] - 1),
        "mean_gait_dwell_steps": sum(dwell) / len(dwell) if dwell else float("nan"),
        "frequency_mean": stats["frequency_sum"] / steps,
        "duration_mean": stats["duration_sum"] / steps,
        "footswing_height_mean": stats["footswing_height_sum"] / steps,
        "stance_width_mean": stats["stance_width_sum"] / steps,
        "body_pitch_mean": stats["body_pitch_sum"] / steps,
    }
    for gait_name in GAIT_NAMES:
        row[f"{GAIT_SHORT_NAMES[gait_name]}_ratio"] = stats["gait_counts"][gait_name] / steps
    for key in SCORE_KEYS:
        row[key] = stats["score_sums"][key] / steps
    for key in RAW_TERM_KEYS:
        row[key] = stats["raw_term_sums"][key] / steps
    return row


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_summary(path, rows, checkpoint_path, iteration):
    decision_interval = int(float(rows[0].get("decision_interval", 1))) if rows else 1
    lines = [
        "# Independent High-Level Policy Evaluation",
        "",
        f"- checkpoint: `{checkpoint_path}`",
        f"- checkpoint_iteration: {iteration}",
        f"- decision_interval: {decision_interval}",
        "",
        "| task | vx | mode | target | pronk | trot | bound | pace | switch | dwell | vx_err | progress | slip | orientation | clearance | foot | clip | req_res | exec_res |",
        "|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['task_id']} | {row['cmd_vx']:.2f} | {row['control_mode']} | {row['target_gait']} "
            f"| {row['pronk_ratio']:.3f} | {row['trot_ratio']:.3f} "
            f"| {row['bound_ratio']:.3f} | {row['pace_ratio']:.3f} "
            f"| {row['gait_switch_rate']:.3f} | {row['mean_gait_dwell_steps']:.2f} "
            f"| {row['vx_err_mean']:.3f} | {row['score_progress']:.3f} "
            f"| {row['score_slip']:.3f} | {row['score_orientation']:.3f} "
            f"| {row['score_clearance']:.3f} | {row['footswing_height_mean']:.3f} "
            f"| {row['action_clip_rate']:.3f} | {row['requested_residual_abs_mean']:.3f} "
            f"| {row['executed_residual_abs_mean']:.3f} |"
        )
    path.write_text("\n".join(lines) + "\n")


def run_child_evals(args, eval_items, output_dir):
    rows = []
    for item_index, (spec, _task_index, vx) in enumerate(eval_items):
        child_dir = output_dir / f"{spec.task_id}_vx{vx:.2f}".replace(".", "p")
        child_seed = args.seed + item_index
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--run-dir",
            str(args.run_dir),
            "--label",
            args.label,
            "--run-index",
            str(args.run_index),
            "--task-map",
            args.task_map,
            "--eval",
            f"{spec.task_id}:{vx}",
            "--seed",
            str(child_seed),
            "--num-envs",
            str(args.num_envs),
            "--steps",
            str(args.steps),
            "--warmup-steps",
            str(args.warmup_steps),
            "--terrain-size",
            str(args.terrain_size),
            "--edge-reset-margin",
            str(args.edge_reset_margin),
            "--teleport-thresh",
            str(args.teleport_thresh),
            "--mesh-type",
            args.mesh_type,
            "--output-dir",
            str(child_dir),
            "--no-spawn",
        ]
        if args.checkpoint:
            cmd += ["--checkpoint", args.checkpoint]
        if args.reward_profile:
            cmd += ["--reward-profile", args.reward_profile]
        if args.decision_interval is not None:
            cmd += ["--decision-interval", str(args.decision_interval)]
        if args.render:
            cmd.append("--render")
        if args.force_zero_residuals:
            cmd.append("--force-zero-residuals")
        if args.force_gait:
            cmd += ["--force-gait", args.force_gait]
        print(f"\nRunning independent eval: {spec.task_id} vx={vx:.2f}")
        subprocess.run(cmd, check=True)
        child_summary = child_dir / "independent_eval_summary.csv"
        with child_summary.open(newline="") as file:
            rows.extend(csv.DictReader(file))
    return rows


def print_row(row):
    print(
        f"task={row['task_id']} vx={row['cmd_vx']:.2f} mode={row['control_mode']} "
        f"target={row['target_gait']} "
        f"gaits[p/t/b/pa]={row['pronk_ratio']:.2f}/"
        f"{row['trot_ratio']:.2f}/{row['bound_ratio']:.2f}/{row['pace_ratio']:.2f} "
        f"switch={row['gait_switch_rate']:.3f} dwell={row['mean_gait_dwell_steps']:.2f} "
        f"vx_err={row['vx_err_mean']:.3f} slip={row['score_slip']:.3f} "
        f"clear={row['score_clearance']:.3f} foot={row['footswing_height_mean']:.3f} "
        f"clip={row['action_clip_rate']:.3f} req_res={row['requested_residual_abs_mean']:.3f} "
        f"exec_res={row['executed_residual_abs_mean']:.3f}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--run-dir", default="runs/high_level_oracle_gait/20260610_rma_notask_reward_v4")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--label", default="gait-conditioned-agility/pretrain-go2/train")
    parser.add_argument("--run-index", type=int, default=0)
    parser.add_argument("--task-map", default=str(MAINLINE_TASK_MAP))
    parser.add_argument("--eval", default=QUICK_EVAL)
    parser.add_argument("--full", action="store_true", help="Evaluate all recommended speeds instead of quick set.")
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument(
        "--decision-interval",
        type=int,
        default=None,
        help=(
            "High-level action hold length during evaluation. Defaults to the "
            "training decision_interval stored in args.json."
        ),
    )
    parser.add_argument("--terrain-size", type=float, default=TRAIN_TERRAIN_SIZE)
    parser.add_argument("--edge-reset-margin", type=float, default=TRAIN_EDGE_RESET_MARGIN)
    parser.add_argument("--teleport-thresh", type=float, default=TRAIN_TELEPORT_THRESH)
    parser.add_argument("--mesh-type", default=TRAIN_MESH_TYPE, choices=["heightfield", "trimesh"])
    parser.add_argument(
        "--reward-profile",
        default=None,
        choices=REWARD_PROFILE_CHOICES,
        help=(
            "Reward profile used for evaluation metrics. Defaults to the "
            "profile stored in the run args.json, falling back to task_focus_v4 "
            "for old runs."
        ),
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--render", action="store_true")
    parser.add_argument(
        "--force-zero-residuals",
        action="store_true",
        help=(
            "Diagnostic: keep the model's gait choice but force all continuous "
            "residual actions to zero before stepping the environment."
        ),
    )
    parser.add_argument(
        "--force-gait",
        choices=GAIT_NAMES,
        default=None,
        help=(
            "Diagnostic baseline: override the policy's gait choice with one "
            "fixed gait family and execute its default continuous parameters."
        ),
    )
    parser.add_argument("--no-spawn", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    run_dir = Path(args.run_dir)
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else latest_checkpoint(run_dir)
    run_args = load_run_args(run_dir)
    decision_interval = int(
        args.decision_interval
        if args.decision_interval is not None
        else run_args.get("decision_interval", 1)
    )
    if decision_interval < 1:
        raise ValueError("--decision-interval must be >= 1")
    args.decision_interval = decision_interval
    reward_profile = args.reward_profile or run_args.get("reward_profile", "task_focus_v4")
    oracle_condition_obs = not bool(run_args.get("no_oracle_condition_obs", False))
    selector_only = bool(run_args.get("selector_only", False))
    selector_latent_cmd_only = bool(run_args.get("selector_latent_cmd_only", False))
    eval_text = DEFAULT_EVAL if args.full else args.eval
    all_specs = read_task_specs(args.task_map, style_reward_scale=0.0, reward_profile=reward_profile)
    eval_items = parse_eval_items(eval_text, all_specs)

    logdir = find_logdir(args.label, args.run_index)
    low_policy = load_low_level_policy(logdir)
    output_dir = Path(args.output_dir) if args.output_dir else run_dir / "independent_eval" / time.strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)

    if len(eval_items) > 1 and not args.no_spawn:
        rows = run_child_evals(args, eval_items, output_dir)
        checkpoint_path = Path(args.checkpoint) if args.checkpoint else latest_checkpoint(run_dir)
        checkpoint_iteration = checkpoint_iteration_from_path(checkpoint_path)
        summary_path = output_dir / "independent_eval_summary.csv"
        markdown_path = output_dir / "summary.md"
        for row in rows:
            for key, value in list(row.items()):
                try:
                    row[key] = float(value)
                except (TypeError, ValueError):
                    pass
        write_csv(summary_path, rows)
        write_summary(markdown_path, rows, checkpoint_path, checkpoint_iteration)
        print(f"\nWrote combined: {summary_path}")
        print(f"Wrote combined: {markdown_path}")
        return

    rows = []
    checkpoint_iteration = -1
    print(f"Checkpoint: {checkpoint_path}")
    print(
        f"oracle_condition_obs={oracle_condition_obs}, "
        f"selector_only={selector_only}, force_zero_residuals={args.force_zero_residuals}, "
        f"force_gait={args.force_gait}, "
        f"eval_items={len(eval_items)}, reward_profile={reward_profile}, "
        f"decision_interval={decision_interval}"
    )
    residual_mask = run_args.get("residual_action_mask")
    if residual_mask is None:
        residual_mask = parse_residual_action_mask(run_args.get("residual_train_dims", "all"))
    else:
        residual_mask = torch.tensor(residual_mask)
    print(f"residual_train_dims={residual_mask_description(residual_mask)}")

    for spec, task_index, vx in eval_items:
        env = OracleConditionHighLevelEnv(
            [spec],
            logdir,
            low_policy,
            args.num_envs,
            render=args.render,
            oracle_condition_obs=False,
            terrain_size=args.terrain_size,
            edge_reset_margin=args.edge_reset_margin,
            teleport_thresh=args.teleport_thresh,
            mesh_type=args.mesh_type,
            selector_hold_steps=int(run_args.get("selector_hold_steps", 3)),
        )
        model, checkpoint_iteration = load_model(checkpoint_path, env, run_args)
        obs = augment_for_checkpoint(env.reset(), task_index, len(all_specs), oracle_condition_obs)
        set_fixed_vx(env, vx)
        obs = append_command_vx_obs(obs, env.command_vx(), selector_latent_cmd_only)
        stats = make_stats(env.num_envs)

        action = None
        requested_action = None
        with torch.inference_mode():
            for step in range(args.steps + args.warmup_steps):
                if action is None or step % decision_interval == 0:
                    if args.force_gait:
                        requested_action = force_gait_action(env, args.force_gait)
                    elif selector_only:
                        requested_action = model.act_student_selector_only(obs)
                    else:
                        requested_action = model.act_student(obs)
                    action = requested_action
                    if args.force_zero_residuals:
                        action = action.clone()
                        action[:, env.num_gaits :] = 0.0
                next_obs, reward, done, info = env.step(action)
                set_fixed_vx(env, vx)
                obs = augment_for_checkpoint(next_obs, task_index, len(all_specs), oracle_condition_obs)
                obs = append_command_vx_obs(obs, env.command_vx(), selector_latent_cmd_only)
                if step >= args.warmup_steps:
                    add_step_stats(stats, env, action, reward, done, info, requested_action=requested_action)

        row = finalize_stats(
            stats,
            spec.task_id,
            spec.condition,
            spec.target_gait,
            vx,
            forced_gait=args.force_gait,
        )
        row["seed"] = args.seed
        row["decision_interval"] = decision_interval
        rows.append(row)
        print_row(row)

        del env, model, obs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary_path = output_dir / "independent_eval_summary.csv"
    markdown_path = output_dir / "summary.md"
    write_csv(summary_path, rows)
    write_summary(markdown_path, rows, checkpoint_path, checkpoint_iteration)
    print(f"\nWrote: {summary_path}")
    print(f"Wrote: {markdown_path}")


if __name__ == "__main__":
    main()
