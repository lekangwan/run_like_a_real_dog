import argparse
import csv
from pathlib import Path
import subprocess
import sys
import time

import isaacgym

assert isaacgym
import torch

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
    read_task_specs,
)
from train_high_level_ppo import find_logdir, load_low_level_policy


QUICK_EVAL = (
    "flat_trot_efficiency:1.0,"
    "ramp_up_trot_robustness:1.0,"
    "rough_slope_trot_robustness:1.0,"
    "push_lateral_pace_recovery:1.5,"
    "stepping_stones_easy_bound_highspeed:2.0"
)

FULL_EVAL = (
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
    "push_lateral_pace_recovery:1.5,"
    "stepping_stones_easy_bound_highspeed:2.0"
)


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
        items.append((by_task[task_id], vx))
    if not items:
        raise ValueError("No eval items requested")
    return items


def parse_gaits(text):
    if text == "all":
        return list(GAIT_NAMES)
    values = [item.strip() for item in text.split(",") if item.strip()]
    unknown = [value for value in values if value not in GAIT_NAMES]
    if unknown:
        raise ValueError(f"Unknown gait(s): {unknown}. Choices: {list(GAIT_NAMES)}")
    return values


def set_fixed_vx(env, vx):
    env.vx_cmd[:] = vx
    env.env.set_velocity_command(env.vx_cmd, 0.0, 0.0)


def fixed_action(env, gait_name):
    gait_id = GAIT_NAMES.index(gait_name)
    action = torch.zeros(env.num_envs, env.num_high_level_actions, device=env.device)
    action[:, gait_id] = 1.0
    return action


def make_stats():
    return {
        "samples": 0,
        "reward_sum": 0.0,
        "weighted_metric_reward_sum": 0.0,
        "done_sum": 0.0,
        "vx_err_sum": 0.0,
        "lateral_offset_sum": 0.0,
        "frequency_sum": 0.0,
        "duration_sum": 0.0,
        "footswing_height_sum": 0.0,
        "stance_width_sum": 0.0,
        "body_pitch_sum": 0.0,
        "gait_counts": {name: 0 for name in GAIT_NAMES},
        "score_sums": {},
    }


def add_step_stats(stats, env, action, reward, done, info):
    actual = info.get("executed_high_level_action", action)
    mapped = env.env._map_action(actual)
    gait_ids = torch.argmax(mapped["selector_weights"], dim=-1)
    terms = info.get("high_level_reward_terms", {})
    n = env.num_envs

    stats["samples"] += n
    stats["reward_sum"] += float(reward.sum().item())
    stats["weighted_metric_reward_sum"] += float(
        terms.get("weighted_metric_reward", reward).sum().item()
    )
    stats["done_sum"] += float(done.float().sum().item())
    stats["vx_err_sum"] += float(torch.abs(env.measured_vx() - env.command_vx()).sum().item())
    stats["lateral_offset_sum"] += float(torch.abs(env.env._compute_lateral_offset()).sum().item())
    stats["frequency_sum"] += float(mapped["frequency"].sum().item())
    stats["duration_sum"] += float(mapped["duration"].sum().item())
    stats["footswing_height_sum"] += float(mapped["footswing_height"].sum().item())
    stats["stance_width_sum"] += float(mapped["stance_width"].sum().item())
    stats["body_pitch_sum"] += float(mapped["body_pitch"].sum().item())

    for gait_id, gait_name in enumerate(GAIT_NAMES):
        stats["gait_counts"][gait_name] += int((gait_ids == gait_id).sum().item())

    for key, value in terms.items():
        if key.startswith("score_"):
            stats["score_sums"][key] = stats["score_sums"].get(key, 0.0) + float(value.sum().item())


def finalize_stats(stats, spec, vx, requested_gait):
    samples = max(1, stats["samples"])
    row = {
        "task_id": spec.task_id,
        "condition": spec.condition,
        "cmd_vx": vx,
        "target_gait": spec.target_gait,
        "requested_gait": requested_gait,
        "samples": stats["samples"],
        "reward_mean": stats["reward_sum"] / samples,
        "weighted_metric_reward": stats["weighted_metric_reward_sum"] / samples,
        "done_rate": stats["done_sum"] / samples,
        "vx_err_mean": stats["vx_err_sum"] / samples,
        "lateral_offset_mean": stats["lateral_offset_sum"] / samples,
        "frequency_mean": stats["frequency_sum"] / samples,
        "duration_mean": stats["duration_sum"] / samples,
        "footswing_height_mean": stats["footswing_height_sum"] / samples,
        "stance_width_mean": stats["stance_width_sum"] / samples,
        "body_pitch_mean": stats["body_pitch_sum"] / samples,
    }
    for gait_name in GAIT_NAMES:
        short = GAIT_SHORT_NAMES[gait_name]
        row[f"{short}_ratio"] = stats["gait_counts"][gait_name] / samples
    row["requested_gait_actual_ratio"] = row[f"{GAIT_SHORT_NAMES[requested_gait]}_ratio"]
    for key, value in sorted(stats["score_sums"].items()):
        row[key] = value / samples
    return row


def write_csv(path, rows):
    if not rows:
        return
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with Path(path).open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value):
    return f"{float(value):.3f}"


def write_summary(path, rows):
    grouped = {}
    for row in rows:
        key = (row["task_id"], float(row["cmd_vx"]))
        grouped.setdefault(key, []).append(row)

    lines = [
        "# Fixed-Gait Live Reward Audit",
        "",
        "This audit uses the current training reward path, not the offline template objective.",
        "",
    ]
    for (task_id, vx), group in grouped.items():
        ranked = sorted(group, key=lambda row: float(row["weighted_metric_reward"]), reverse=True)
        target = ranked[0]["target_gait"]
        live_best = ranked[0]["requested_gait"]
        lines += [
            f"## {task_id} vx={vx:.2f}",
            "",
            f"- target_gait: `{target}`",
            f"- live_best_by_weighted_metric_reward: `{live_best}`",
            "",
            "| rank | gait | weighted | reward | vx_err | done | progress | slip | orientation | clearance | actual | foot |",
            "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for rank, row in enumerate(ranked, start=1):
            lines.append(
                f"| {rank} | {row['requested_gait']} "
                f"| {fmt(row['weighted_metric_reward'])} "
                f"| {fmt(row['reward_mean'])} "
                f"| {fmt(row['vx_err_mean'])} "
                f"| {fmt(row['done_rate'])} "
                f"| {fmt(row.get('score_progress', float('nan')))} "
                f"| {fmt(row.get('score_slip', float('nan')))} "
                f"| {fmt(row.get('score_orientation', float('nan')))} "
                f"| {fmt(row.get('score_clearance', float('nan')))} "
                f"| {fmt(row['requested_gait_actual_ratio'])} "
                f"| {fmt(row['footswing_height_mean'])} |"
            )
        lines.append("")
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def run_child_audits(args, eval_items, gaits, output_dir):
    rows = []
    for spec, vx in eval_items:
        for gait in gaits:
            child_dir = output_dir / f"{spec.task_id}_vx{vx:.2f}_{gait}".replace(".", "p")
            cmd = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--label",
                args.label,
                "--run-index",
                str(args.run_index),
                "--task-map",
                args.task_map,
                "--eval",
                f"{spec.task_id}:{vx}",
                "--gaits",
                gait,
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
                "--selector-hold-steps",
                str(args.selector_hold_steps),
                "--style-reward-scale",
                str(args.style_reward_scale),
                "--reward-profile",
                args.reward_profile,
                "--output-dir",
                str(child_dir),
                "--no-spawn",
            ]
            if args.render:
                cmd.append("--render")
            print(f"\nAuditing fixed gait: task={spec.task_id} vx={vx:.2f} gait={gait}")
            subprocess.run(cmd, check=True)
            with (child_dir / "fixed_gait_live_reward.csv").open(newline="") as file:
                rows.extend(csv.DictReader(file))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", default="gait-conditioned-agility/pretrain-go2/train")
    parser.add_argument("--run-index", type=int, default=0)
    parser.add_argument("--task-map", default=str(MAINLINE_TASK_MAP))
    parser.add_argument("--eval", default=QUICK_EVAL)
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--gaits", default="all")
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--terrain-size", type=float, default=TRAIN_TERRAIN_SIZE)
    parser.add_argument("--edge-reset-margin", type=float, default=TRAIN_EDGE_RESET_MARGIN)
    parser.add_argument("--teleport-thresh", type=float, default=TRAIN_TELEPORT_THRESH)
    parser.add_argument("--mesh-type", default=TRAIN_MESH_TYPE, choices=["heightfield", "trimesh"])
    parser.add_argument("--selector-hold-steps", type=int, default=0)
    parser.add_argument(
        "--style-reward-scale",
        type=float,
        default=0.0,
        help="Keep 0.0 to audit reward-only v4; set >0 only to audit explicit selector shaping.",
    )
    parser.add_argument(
        "--reward-profile",
        default="task_focus_v4",
        choices=REWARD_PROFILE_CHOICES,
        help="Use task_focus_v4 for legacy per-task weights or unified_* for one shared reward.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Defaults to runs/high_level_oracle_gait/fixed_gait_live_reward_audit/<timestamp>.",
    )
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--no-spawn", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    specs = read_task_specs(
        args.task_map,
        style_reward_scale=args.style_reward_scale,
        reward_profile=args.reward_profile,
    )
    eval_text = FULL_EVAL if args.full else args.eval
    eval_items = parse_eval_items(eval_text, specs)
    gaits = parse_gaits(args.gaits)
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else Path("runs/high_level_oracle_gait/fixed_gait_live_reward_audit") / time.strftime("%Y%m%d_%H%M%S")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    if (len(eval_items) > 1 or len(gaits) > 1) and not args.no_spawn:
        rows = run_child_audits(args, eval_items, gaits, output_dir)
        for row in rows:
            for key, value in list(row.items()):
                try:
                    row[key] = float(value)
                except (TypeError, ValueError):
                    pass
        csv_path = output_dir / "fixed_gait_live_reward.csv"
        summary_path = output_dir / "summary.md"
        write_csv(csv_path, rows)
        write_summary(summary_path, rows)
        print(f"\nWrote combined: {csv_path}")
        print(f"Wrote combined: {summary_path}")
        return

    logdir = find_logdir(args.label, args.run_index)
    low_policy = load_low_level_policy(logdir)
    rows = []
    for spec, vx in eval_items:
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
            selector_hold_steps=args.selector_hold_steps,
        )
        obs = env.reset()
        del obs
        set_fixed_vx(env, vx)
        for gait in gaits:
            env.reset()
            set_fixed_vx(env, vx)
            action = fixed_action(env, gait)
            stats = make_stats()
            with torch.inference_mode():
                for step in range(args.steps + args.warmup_steps):
                    set_fixed_vx(env, vx)
                    _obs, reward, done, info = env.step(action)
                    set_fixed_vx(env, vx)
                    if step >= args.warmup_steps:
                        add_step_stats(stats, env, action, reward, done, info)
            row = finalize_stats(stats, spec, vx, gait)
            rows.append(row)
            print(
                f"task={spec.task_id} vx={vx:.2f} gait={gait} "
                f"weighted={row['weighted_metric_reward']:.3f} "
                f"vx_err={row['vx_err_mean']:.3f} done={row['done_rate']:.3f} "
                f"actual={row['requested_gait_actual_ratio']:.3f}"
            )
        del env
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    csv_path = output_dir / "fixed_gait_live_reward.csv"
    summary_path = output_dir / "summary.md"
    write_csv(csv_path, rows)
    write_summary(summary_path, rows)
    print(f"\nWrote: {csv_path}")
    print(f"Wrote: {summary_path}")


if __name__ == "__main__":
    main()
