import argparse
import csv
import glob
import itertools
import json
from pathlib import Path
import pickle as pkl

import isaacgym

assert isaacgym
import torch

from go2_gym import MINI_GYM_ROOT_DIR
from go2_gym.envs.base.legged_robot_config import Cfg
from go2_gym.envs.go2.go2_config import config_go2
from go2_gym.envs.go2.velocity_tracking import VelocityTrackingEasyEnv
from go2_gym.envs.wrappers.history_wrapper import HistoryWrapper


GAIT_PARAMS = {
    # This follows the command convention used by scripts/play.py.
    # Values fill command indices [5:8] = phase, offset, bound.
    "pronking": (0.0, 0.0, 0.0),
    "trotting": (0.5, 0.0, 0.0),
    "bounding": (0.0, 0.5, 0.0),
    "pacing": (0.0, 0.0, 0.5),
}


def parse_floats(value):
    return [float(v.strip()) for v in value.split(",") if v.strip()]


def parse_strings(value):
    values = [v.strip() for v in value.split(",") if v.strip()]
    unknown = [v for v in values if v not in GAIT_PARAMS]
    if unknown:
        raise argparse.ArgumentTypeError(f"Unknown gait(s): {unknown}. Choices: {sorted(GAIT_PARAMS)}")
    return values


def find_logdir(label, run_index):
    dirs = sorted(glob.glob(str(Path(MINI_GYM_ROOT_DIR) / "runs" / label / "*")))
    if not dirs:
        raise FileNotFoundError(f"No runs found for label: {label}")
    return dirs[run_index]


def load_policy(logdir):
    body = torch.jit.load(str(Path(logdir) / "checkpoints" / "body_latest.jit"))
    adaptation_module = torch.jit.load(str(Path(logdir) / "checkpoints" / "adaptation_module_latest.jit"))

    def policy(obs, info=None):
        if info is None:
            info = {}
        obs_history = obs["obs_history"].to("cpu")
        latent = adaptation_module.forward(obs_history)
        action = body.forward(torch.cat((obs_history, latent), dim=-1))
        info["latent"] = latent
        return action

    return policy


def load_env(logdir, num_envs, headless):
    config_go2(Cfg)
    with open(Path(logdir) / "parameters.pkl", "rb") as file:
        pkl_cfg = pkl.load(file)
        cfg = pkl_cfg["Cfg"]
        for key, value in cfg.items():
            if hasattr(Cfg, key):
                for key2, value2 in value.items():
                    setattr(getattr(Cfg, key), key2, value2)

    # Disable domain randomization during scans so parameter effects are easier to read.
    Cfg.domain_rand.push_robots = False
    Cfg.domain_rand.randomize_friction = False
    Cfg.domain_rand.randomize_gravity = False
    Cfg.domain_rand.randomize_restitution = False
    Cfg.domain_rand.randomize_motor_offset = False
    Cfg.domain_rand.randomize_motor_strength = False
    Cfg.domain_rand.randomize_friction_indep = False
    Cfg.domain_rand.randomize_ground_friction = False
    Cfg.domain_rand.randomize_base_mass = False
    Cfg.domain_rand.randomize_Kd_factor = False
    Cfg.domain_rand.randomize_Kp_factor = False
    Cfg.domain_rand.randomize_joint_friction = False
    Cfg.domain_rand.randomize_com_displacement = False

    Cfg.env.num_envs = num_envs
    Cfg.env.num_recording_envs = 0
    Cfg.terrain.curriculum = False
    Cfg.terrain.selected = False
    Cfg.terrain.min_init_terrain_level = 0
    Cfg.terrain.max_init_terrain_level = 0
    Cfg.terrain.num_rows = 1
    Cfg.terrain.num_cols = max(1, num_envs)
    Cfg.terrain.border_size = 0
    Cfg.terrain.center_robots = False
    Cfg.terrain.teleport_robots = True
    Cfg.asset.flip_visual_attachments = True

    env = VelocityTrackingEasyEnv(sim_device="cuda:0", headless=headless, cfg=Cfg)
    return HistoryWrapper(env)


def build_grid(args):
    keys = [
        ("vx", args.vx),
        ("vy", args.vy),
        ("yaw", args.yaw),
        ("gait", args.gaits),
        ("frequency", args.frequencies),
        ("footswing_height", args.footswing_heights),
        ("body_pitch", args.body_pitches),
        ("stance_width", args.stance_widths),
    ]
    names, values = zip(*keys)
    return [dict(zip(names, combo)) for combo in itertools.product(*values)]


def set_commands(env, batch, batch_size):
    commands = torch.zeros(batch_size, env.commands.shape[1], device=env.commands.device)
    for i, params in enumerate(batch):
        phase, offset, bound = GAIT_PARAMS[params["gait"]]
        commands[i, 0] = params["vx"]
        commands[i, 1] = params["vy"]
        commands[i, 2] = params["yaw"]
        commands[i, 3] = 0.0
        commands[i, 4] = params["frequency"]
        commands[i, 5] = phase
        commands[i, 6] = offset
        commands[i, 7] = bound
        commands[i, 8] = 0.5
        commands[i, 9] = params["footswing_height"]
        commands[i, 10] = params["body_pitch"]
        commands[i, 11] = 0.0
        commands[i, 12] = params["stance_width"]
        if commands.shape[1] > 13:
            commands[i, 13] = 0.40
    env.commands[:, : commands.shape[1]] = commands


def empty_metrics(n):
    return {
        "vx_abs_error": torch.zeros(n),
        "vy_abs_error": torch.zeros(n),
        "yaw_abs_error": torch.zeros(n),
        "roll_pitch_rms_proxy": torch.zeros(n),
        "torque_sq": torch.zeros(n),
        "action_delta_sq": torch.zeros(n),
        "feet_slip_proxy": torch.zeros(n),
        "fall_count": torch.zeros(n),
    }


def update_metrics(metrics, env, actions, prev_actions, batch, active_n, dones):
    device = env.base_lin_vel.device
    vx_cmd = torch.tensor([p["vx"] for p in batch], device=device)
    vy_cmd = torch.tensor([p["vy"] for p in batch], device=device)
    yaw_cmd = torch.tensor([p["yaw"] for p in batch], device=device)

    metrics["vx_abs_error"] += torch.abs(env.base_lin_vel[:active_n, 0] - vx_cmd).cpu()
    metrics["vy_abs_error"] += torch.abs(env.base_lin_vel[:active_n, 1] - vy_cmd).cpu()
    metrics["yaw_abs_error"] += torch.abs(env.base_ang_vel[:active_n, 2] - yaw_cmd).cpu()
    metrics["roll_pitch_rms_proxy"] += torch.sum(env.projected_gravity[:active_n, :2] ** 2, dim=1).cpu()
    metrics["torque_sq"] += torch.mean(env.torques[:active_n] ** 2, dim=1).cpu()
    metrics["action_delta_sq"] += torch.mean((actions[:active_n].cpu() - prev_actions[:active_n].cpu()) ** 2, dim=1)

    contacts = env.contact_forces[:active_n, env.feet_indices, 2] > 1.0
    foot_xy_vel = torch.sum(env.foot_velocities[:active_n, :, :2] ** 2, dim=2)
    metrics["feet_slip_proxy"] += torch.mean(contacts * foot_xy_vel, dim=1).cpu()
    metrics["fall_count"] += dones[:active_n].float().cpu()


def run_scan(env, policy, grid, args):
    rows = []
    batch_size = args.batch_size
    num_batches = (len(grid) + batch_size - 1) // batch_size

    for batch_idx in range(num_batches):
        start = batch_idx * batch_size
        batch = grid[start : start + batch_size]
        active_n = len(batch)
        obs = env.reset()
        set_commands(env, batch, batch_size)

        metrics = empty_metrics(active_n)
        prev_actions = torch.zeros(batch_size, env.num_actions)
        eval_steps = 0

        for step in range(args.warmup_steps + args.eval_steps):
            with torch.no_grad():
                actions = policy(obs)
            set_commands(env, batch, batch_size)
            obs, _, dones, _ = env.step(actions)

            if step >= args.warmup_steps:
                update_metrics(metrics, env, actions, prev_actions, batch, active_n, dones)
                eval_steps += 1
            prev_actions = actions.detach().clone()

        for i, params in enumerate(batch):
            row = dict(params)
            for key, values in metrics.items():
                value = values[i].item()
                row[key] = value if key == "fall_count" else value / max(eval_steps, 1)
            rows.append(row)

        print(f"Finished batch {batch_idx + 1}/{num_batches} ({len(rows)}/{len(grid)} configs)")

    return rows


def save_results(rows, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "scan_results.csv"
    json_path = output_dir / "scan_results.json"

    with open(csv_path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    with open(json_path, "w") as file:
        json.dump(rows, file, indent=2)

    return csv_path, json_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", default="gait-conditioned-agility/pretrain-go2/train")
    parser.add_argument("--run-index", type=int, default=0)
    parser.add_argument("--render", action="store_true", help="Open the Isaac Gym viewer instead of running headless.")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--eval-steps", type=int, default=300)
    parser.add_argument("--vx", type=parse_floats, default=parse_floats("0.2,0.6,1.0,1.4"))
    parser.add_argument("--vy", type=parse_floats, default=parse_floats("0.0"))
    parser.add_argument("--yaw", type=parse_floats, default=parse_floats("0.0"))
    parser.add_argument("--gaits", type=parse_strings, default=parse_strings("trotting,bounding"))
    parser.add_argument("--frequencies", type=parse_floats, default=parse_floats("2.0,3.0,4.0"))
    parser.add_argument("--footswing-heights", type=parse_floats, default=parse_floats("0.06,0.10,0.14"))
    parser.add_argument("--body-pitches", type=parse_floats, default=parse_floats("0.0,-0.08"))
    parser.add_argument("--stance-widths", type=parse_floats, default=parse_floats("0.25,0.33"))
    parser.add_argument("--output-dir", default=str(Path(MINI_GYM_ROOT_DIR) / "logs" / "gait_param_scan"))
    args = parser.parse_args()

    logdir = find_logdir(args.label, args.run_index)
    grid = build_grid(args)
    print(f"Loaded run: {logdir}")
    print(f"Scanning {len(grid)} parameter configs with batch size {args.batch_size}")

    env = load_env(logdir, args.batch_size, headless=not args.render)
    policy = load_policy(logdir)
    rows = run_scan(env, policy, grid, args)
    csv_path, json_path = save_results(rows, Path(args.output_dir))
    print(f"Saved CSV: {csv_path}")
    print(f"Saved JSON: {json_path}")


if __name__ == "__main__":
    main()
