import argparse
import csv
import gc
import itertools
import json
from pathlib import Path

import isaacgym

assert isaacgym
import torch

from go2_gym import MINI_GYM_ROOT_DIR
from scan_gait_params import GAIT_PARAMS, find_logdir, load_env, load_policy, parse_floats, parse_strings


def build_grid(args):
    keys = [
        ("vx", args.vx),
        ("gait", args.gaits),
        ("frequency", args.frequencies),
        ("footswing_height", args.footswing_heights),
        ("body_pitch", args.body_pitches),
        ("stance_width", args.stance_widths),
    ]
    names, values = zip(*keys)
    return [dict(zip(names, combo)) for combo in itertools.product(*values)]


def build_command_tensor(env, batch, batch_size):
    commands = torch.zeros(batch_size, env.commands.shape[1], device=env.commands.device)
    for i, params in enumerate(batch):
        phase, offset, bound = GAIT_PARAMS[params["gait"]]
        commands[i, 0] = params["vx"]
        commands[i, 1] = 0.0
        commands[i, 2] = 0.0
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
    return commands


def set_commands(env, commands):
    env.commands[:, : commands.shape[1]] = commands


def empty_metrics(n):
    metrics = {
        "measured_vx": torch.zeros(n),
        "vx_abs_error": torch.zeros(n),
        "vy_abs_error": torch.zeros(n),
        "yaw_abs_error": torch.zeros(n),
        "lateral_vel_abs": torch.zeros(n),
        "lateral_vel_sq": torch.zeros(n),
        "base_z_sum": torch.zeros(n),
        "base_z_sq_sum": torch.zeros(n),
        "base_z_vel_abs": torch.zeros(n),
        "base_z_vel_sq": torch.zeros(n),
        "roll_rate_abs": torch.zeros(n),
        "roll_rate_sq": torch.zeros(n),
        "pitch_rate_abs": torch.zeros(n),
        "pitch_rate_sq": torch.zeros(n),
        "yaw_rate_abs": torch.zeros(n),
        "yaw_rate_sq": torch.zeros(n),
        "gravity_x_abs": torch.zeros(n),
        "gravity_x_sq": torch.zeros(n),
        "gravity_y_abs": torch.zeros(n),
        "gravity_y_sq": torch.zeros(n),
        "velocity_reward": torch.zeros(n),
        "yaw_reward": torch.zeros(n),
        "orientation_penalty": torch.zeros(n),
        "torque_penalty": torch.zeros(n),
        "slip_penalty": torch.zeros(n),
        "vertical_velocity_penalty": torch.zeros(n),
        "action_delta_sq": torch.zeros(n),
        "fall_count": torch.zeros(n),
        "template_score": torch.zeros(n),
    }
    for foot_id in range(4):
        metrics[f"foot{foot_id}_duty"] = torch.zeros(n)
    for pair in ("01", "02", "03", "12", "13", "23"):
        metrics[f"contact_pair{pair}_sync"] = torch.zeros(n)
        metrics[f"contact_pair{pair}_co"] = torch.zeros(n)
    metrics["contact_count"] = torch.zeros(n)
    metrics["flight_ratio"] = torch.zeros(n)
    metrics["all_contact_ratio"] = torch.zeros(n)
    return metrics


def update_metrics(metrics, env, actions, prev_actions, command_vx, active_n, dones):
    vx_cmd = command_vx[:active_n]
    vx_error = env.base_lin_vel[:active_n, 0] - vx_cmd
    vy_error = env.base_lin_vel[:active_n, 1]
    yaw_error = env.base_ang_vel[:active_n, 2]
    lateral_vel = env.base_lin_vel[:active_n, 1]
    base_z = env.root_states[:active_n, 2]
    base_z_vel = env.base_lin_vel[:active_n, 2]
    roll_rate = env.base_ang_vel[:active_n, 0]
    pitch_rate = env.base_ang_vel[:active_n, 1]
    yaw_rate = env.base_ang_vel[:active_n, 2]
    gravity_x = env.projected_gravity[:active_n, 0]
    gravity_y = env.projected_gravity[:active_n, 1]

    velocity_reward = torch.exp(-(vx_error**2 + 0.25 * vy_error**2) / 0.25)
    yaw_reward = torch.exp(-(yaw_error**2) / 0.10)
    orientation_penalty = torch.sum(env.projected_gravity[:active_n, :2] ** 2, dim=1)
    torque_penalty = torch.mean(env.torques[:active_n] ** 2, dim=1) / 100.0

    contacts = env.contact_forces[:active_n, env.feet_indices, 2] > 1.0
    contacts_f = contacts.float()
    foot_xy_vel = torch.sum(env.foot_velocities[:active_n, :, :2] ** 2, dim=2)
    slip_penalty = torch.mean(contacts * foot_xy_vel, dim=1)
    vertical_velocity_penalty = base_z_vel**2
    vx_abs_error_penalty = torch.abs(vx_error)
    fall_penalty = dones[:active_n].float()

    template_score = (
        2.0 * velocity_reward
        + 0.5 * yaw_reward
        - orientation_penalty
        - 0.05 * torque_penalty
        - 0.5 * slip_penalty
        - 0.25 * vx_abs_error_penalty
        - 0.05 * vertical_velocity_penalty
        - 10.0 * fall_penalty
    )

    metrics["measured_vx"] += env.base_lin_vel[:active_n, 0].detach().cpu()
    metrics["vx_abs_error"] += vx_abs_error_penalty.detach().cpu()
    metrics["vy_abs_error"] += torch.abs(vy_error).detach().cpu()
    metrics["yaw_abs_error"] += torch.abs(yaw_error).detach().cpu()
    metrics["lateral_vel_abs"] += torch.abs(lateral_vel).detach().cpu()
    metrics["lateral_vel_sq"] += (lateral_vel**2).detach().cpu()
    metrics["base_z_sum"] += base_z.detach().cpu()
    metrics["base_z_sq_sum"] += (base_z**2).detach().cpu()
    metrics["base_z_vel_abs"] += torch.abs(base_z_vel).detach().cpu()
    metrics["base_z_vel_sq"] += (base_z_vel**2).detach().cpu()
    metrics["roll_rate_abs"] += torch.abs(roll_rate).detach().cpu()
    metrics["roll_rate_sq"] += (roll_rate**2).detach().cpu()
    metrics["pitch_rate_abs"] += torch.abs(pitch_rate).detach().cpu()
    metrics["pitch_rate_sq"] += (pitch_rate**2).detach().cpu()
    metrics["yaw_rate_abs"] += torch.abs(yaw_rate).detach().cpu()
    metrics["yaw_rate_sq"] += (yaw_rate**2).detach().cpu()
    metrics["gravity_x_abs"] += torch.abs(gravity_x).detach().cpu()
    metrics["gravity_x_sq"] += (gravity_x**2).detach().cpu()
    metrics["gravity_y_abs"] += torch.abs(gravity_y).detach().cpu()
    metrics["gravity_y_sq"] += (gravity_y**2).detach().cpu()
    metrics["velocity_reward"] += velocity_reward.detach().cpu()
    metrics["yaw_reward"] += yaw_reward.detach().cpu()
    metrics["orientation_penalty"] += orientation_penalty.detach().cpu()
    metrics["torque_penalty"] += torque_penalty.detach().cpu()
    metrics["slip_penalty"] += slip_penalty.detach().cpu()
    metrics["vertical_velocity_penalty"] += vertical_velocity_penalty.detach().cpu()
    action_delta_sq = torch.mean((actions[:active_n] - prev_actions[:active_n]) ** 2, dim=1)
    metrics["action_delta_sq"] += action_delta_sq.detach().cpu()
    metrics["fall_count"] += fall_penalty.detach().cpu()
    metrics["template_score"] += template_score.detach().cpu()
    for foot_id in range(4):
        metrics[f"foot{foot_id}_duty"] += contacts_f[:, foot_id].detach().cpu()
    for i, j in ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)):
        pair = f"{i}{j}"
        metrics[f"contact_pair{pair}_sync"] += (
            1.0 - torch.abs(contacts_f[:, i] - contacts_f[:, j])
        ).detach().cpu()
        metrics[f"contact_pair{pair}_co"] += (contacts_f[:, i] * contacts_f[:, j]).detach().cpu()
    contact_count = torch.sum(contacts_f, dim=1)
    metrics["contact_count"] += contact_count.detach().cpu()
    metrics["flight_ratio"] += (contact_count == 0).float().detach().cpu()
    metrics["all_contact_ratio"] += (contact_count == 4).float().detach().cpu()


def finalize_metrics(values, eval_steps):
    row = {}
    steps = max(eval_steps, 1)
    for key, value in values.items():
        if key in ("base_z_sum", "base_z_sq_sum"):
            continue
        row[key] = value if key == "fall_count" else value / steps

    base_z_mean = values["base_z_sum"] / steps
    base_z_sq_mean = values["base_z_sq_sum"] / steps
    row["base_z_mean"] = base_z_mean
    row["base_z_std"] = max(base_z_sq_mean - base_z_mean**2, 0.0) ** 0.5
    for src, dst in (
        ("lateral_vel_sq", "lateral_vel_rms"),
        ("base_z_vel_sq", "base_z_vel_rms"),
        ("roll_rate_sq", "roll_rate_rms"),
        ("pitch_rate_sq", "pitch_rate_rms"),
        ("yaw_rate_sq", "yaw_rate_rms"),
        ("gravity_x_sq", "gravity_x_rms"),
        ("gravity_y_sq", "gravity_y_rms"),
    ):
        row[dst] = max(row[src], 0.0) ** 0.5
    row["done_rate"] = row["fall_count"] / steps
    return row


def run_eval(env, policy, grid, args):
    rows = []
    batch_size = args.batch_size
    num_batches = (len(grid) + batch_size - 1) // batch_size

    for batch_idx in range(num_batches):
        start = batch_idx * batch_size
        batch = grid[start : start + batch_size]
        active_n = len(batch)
        obs = env.reset()
        commands = build_command_tensor(env, batch, batch_size)
        command_vx = commands[:, 0]
        set_commands(env, commands)

        metrics = empty_metrics(active_n)
        prev_actions = None
        eval_steps = 0

        for step in range(args.warmup_steps + args.eval_steps):
            with torch.inference_mode():
                actions = policy(obs)
            if prev_actions is None:
                prev_actions = torch.zeros_like(actions)
            set_commands(env, commands)
            obs, _, dones, _ = env.step(actions)

            if step >= args.warmup_steps:
                update_metrics(metrics, env, actions, prev_actions, command_vx, active_n, dones)
                eval_steps += 1
            prev_actions.copy_(actions.detach())

        for i, params in enumerate(batch):
            row = dict(params)
            phase, offset, bound = GAIT_PARAMS[params["gait"]]
            row["phase"] = phase
            row["offset"] = offset
            row["bound"] = bound
            finalized = finalize_metrics({key: values[i].item() for key, values in metrics.items()}, eval_steps)
            row.update(finalized)
            rows.append(row)

        del obs, actions, dones, commands, command_vx, prev_actions, metrics
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize(env.device)
            torch.cuda.empty_cache()
            torch.cuda.synchronize(env.device)

        memory_text = ""
        if args.log_memory and torch.cuda.is_available():
            memory_text = (
                f" cuda_alloc={torch.cuda.memory_allocated(env.device) / (1024**2):.0f}MB"
                f" cuda_reserved={torch.cuda.memory_reserved(env.device) / (1024**2):.0f}MB"
            )
        print(
            f"Finished batch {batch_idx + 1}/{num_batches} "
            f"({len(rows)}/{len(grid)} configs){memory_text}"
        )

    return rows


def best_rows(rows, group_keys, score_key="template_score"):
    grouped = {}
    for row in rows:
        key = tuple(row[name] for name in group_keys)
        if key not in grouped or row[score_key] > grouped[key][score_key]:
            grouped[key] = row
    return list(grouped.values())


def save_csv(path, rows):
    if not rows:
        raise ValueError("No rows to save")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_results(rows, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    save_csv(output_dir / "template_eval_results.csv", rows)
    save_csv(output_dir / "best_by_speed.csv", best_rows(rows, ["vx"]))
    save_csv(output_dir / "best_by_speed_gait.csv", best_rows(rows, ["vx", "gait"]))
    with open(output_dir / "template_eval_results.json", "w") as file:
        json.dump(rows, file, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", default="gait-conditioned-agility/pretrain-go2/train")
    parser.add_argument("--run-index", type=int, default=0)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--eval-steps", type=int, default=300)
    parser.add_argument("--log-memory", action="store_true")
    parser.add_argument("--vx", type=parse_floats, default=parse_floats("0.2,0.5,0.8,1.2,1.6,2.0"))
    parser.add_argument("--gaits", type=parse_strings, default=parse_strings("pronking,trotting,bounding,pacing"))
    parser.add_argument("--frequencies", type=parse_floats, default=parse_floats("3.0"))
    parser.add_argument("--footswing-heights", type=parse_floats, default=parse_floats("0.08"))
    parser.add_argument("--body-pitches", type=parse_floats, default=parse_floats("0.0"))
    parser.add_argument("--stance-widths", type=parse_floats, default=parse_floats("0.33"))
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-configs", type=int, default=None)
    parser.add_argument(
        "--output-dir",
        default=str(Path(MINI_GYM_ROOT_DIR) / "logs" / "gait_template_eval"),
    )
    args = parser.parse_args()

    logdir = find_logdir(args.label, args.run_index)
    full_grid = build_grid(args)
    end_index = len(full_grid) if args.max_configs is None else args.start_index + args.max_configs
    grid = full_grid[args.start_index : min(end_index, len(full_grid))]
    if not grid:
        raise ValueError(
            f"Empty grid slice: start_index={args.start_index}, "
            f"max_configs={args.max_configs}, total={len(full_grid)}"
        )
    print(f"Loaded run: {logdir}")
    print(
        f"Evaluating {len(grid)} fixed template configs "
        f"({args.start_index}:{args.start_index + len(grid)} of {len(full_grid)}) "
        f"with batch size {args.batch_size}"
    )

    env = load_env(logdir, args.batch_size, headless=not args.render)
    policy = load_policy(logdir)
    rows = run_eval(env, policy, grid, args)
    save_results(rows, Path(args.output_dir))
    print(f"Saved results to: {args.output_dir}")


if __name__ == "__main__":
    main()
