import argparse
import csv
import time
from pathlib import Path

import isaacgym

assert isaacgym
from isaacgym import gymtorch
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


DEFAULT_EVAL = "ramp_up_trot_robustness:1.25"


BASE_TENSOR_ATTRS = (
    "root_states",
    "dof_state",
    "last_actions",
    "last_last_actions",
    "actions",
    "last_dof_vel",
    "last_root_vel",
    "feet_air_time",
    "episode_length_buf",
    "reset_buf",
    "time_out_buf",
    "edge_reset_buf",
    "gait_indices",
    "commands",
    "commands_value",
    "base_lin_vel",
    "base_ang_vel",
    "projected_gravity",
    "last_joint_pos_target",
    "last_last_joint_pos_target",
    "joint_pos_target",
    "prev_base_pos",
    "prev_base_quat",
    "prev_base_lin_vel",
    "prev_foot_velocities",
)


def clone_value(value):
    if torch.is_tensor(value):
        return value.detach().clone()
    if isinstance(value, dict):
        return {key: clone_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clone_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(clone_value(item) for item in value)
    return value


def restore_value(target, value):
    if torch.is_tensor(target) and torch.is_tensor(value):
        target.copy_(value)
        return target
    if isinstance(target, dict) and isinstance(value, dict):
        for key in value:
            if key in target:
                restore_value(target[key], value[key])
            else:
                target[key] = clone_value(value[key])
        return target
    if isinstance(target, list) and isinstance(value, list):
        for idx, item in enumerate(value):
            if idx < len(target):
                restore_value(target[idx], item)
        return target
    return clone_value(value)


def parse_eval_item(text, specs):
    by_task = {spec.task_id: spec for spec in specs}
    if "," in text:
        raise ValueError("This paired diagnostic accepts exactly one --eval item")
    if ":" in text:
        task_id, vx_text = text.split(":", 1)
        vx = float(vx_text)
    else:
        task_id = text
        if task_id not in by_task:
            raise ValueError(f"Unknown task_id={task_id!r}. Choices: {sorted(by_task)}")
        spec = by_task[task_id]
        vx = 0.5 * (spec.vx_low + spec.vx_high)
    if task_id not in by_task:
        raise ValueError(f"Unknown task_id={task_id!r}. Choices: {sorted(by_task)}")
    return by_task[task_id], vx


def fixed_action(env, gait_name):
    gait_id = GAIT_NAMES.index(gait_name)
    action = torch.zeros(env.num_envs, env.num_high_level_actions, device=env.device)
    action[:, gait_id] = 1.0
    return action


def set_fixed_vx(env, vx):
    env.vx_cmd[:] = vx
    env.env.set_velocity_command(env.vx_cmd, 0.0, 0.0)


def snapshot_env(env):
    high = env.env
    base = high._get_base_env()
    hist = high.env

    state = {
        "vx_cmd": clone_value(env.vx_cmd),
        "high": {
            "high_level_action": clone_value(high.high_level_action),
            "prev_high_level_action": clone_value(high.prev_high_level_action),
            "selector_hold_counter": clone_value(high.selector_hold_counter),
            "velocity_command": clone_value(high.velocity_command),
            "obs_history": clone_value(high.obs_history),
            "obs_shift_buffer": clone_value(high.obs_shift_buffer),
            "low_level_obs": clone_value(high.low_level_obs),
            "prev_foot_contacts": clone_value(high.prev_foot_contacts),
        },
        "history": {
            "obs_history": clone_value(hist.obs_history),
            "obs_shift_buffer": clone_value(hist.obs_shift_buffer),
        },
        "base": {},
    }
    for name in BASE_TENSOR_ATTRS:
        if hasattr(base, name):
            state["base"][name] = clone_value(getattr(base, name))
    if hasattr(base, "lag_buffer"):
        state["base"]["lag_buffer"] = clone_value(base.lag_buffer)
    return state


def restore_env(env, state):
    high = env.env
    base = high._get_base_env()
    hist = high.env

    restore_value(env.vx_cmd, state["vx_cmd"])
    for name, value in state["high"].items():
        current = getattr(high, name)
        restored = restore_value(current, value)
        if not torch.is_tensor(current) and not isinstance(current, (dict, list)):
            setattr(high, name, restored)
    for name, value in state["history"].items():
        restore_value(getattr(hist, name), value)
    for name, value in state["base"].items():
        if name == "lag_buffer" and hasattr(base, "lag_buffer"):
            restore_value(base.lag_buffer, value)
        elif hasattr(base, name):
            restore_value(getattr(base, name), value)

    base.gym.set_actor_root_state_tensor(base.sim, gymtorch.unwrap_tensor(base.root_states))
    base.gym.set_dof_state_tensor(base.sim, gymtorch.unwrap_tensor(base.dof_state))
    base.gym.refresh_actor_root_state_tensor(base.sim)
    base.gym.refresh_dof_state_tensor(base.sim)
    high.env.commands[:, 0:3] = high.velocity_command


def init_metric_tensors(env):
    return {
        "samples": torch.zeros(env.num_envs, device=env.device),
        "reward": torch.zeros(env.num_envs, device=env.device),
        "weighted_metric_reward": torch.zeros(env.num_envs, device=env.device),
        "done": torch.zeros(env.num_envs, device=env.device),
        "vx_err": torch.zeros(env.num_envs, device=env.device),
        "lateral_offset": torch.zeros(env.num_envs, device=env.device),
        "metrics": {},
    }


def add_metric_tensor(stats, key, value):
    if key not in stats["metrics"]:
        stats["metrics"][key] = torch.zeros_like(stats["reward"])
    stats["metrics"][key] += value.detach()


def collect_step_stats(stats, env, reward, done, info):
    terms = info.get("high_level_reward_terms", {})
    n = env.num_envs
    stats["samples"] += 1.0
    stats["reward"] += reward.detach()
    stats["weighted_metric_reward"] += terms.get("weighted_metric_reward", reward).detach()
    stats["done"] += done.float().detach()
    stats["vx_err"] += torch.abs(env.measured_vx() - env.command_vx()).detach()
    stats["lateral_offset"] += torch.abs(env.env._compute_lateral_offset()).detach()

    for key, value in terms.items():
        if key.startswith("score_") or key in (
            "velocity_reward",
            "yaw_reward",
            "orientation_penalty",
            "torque_penalty",
            "slip_penalty",
            "contact_slip_penalty",
            "mechanical_power_abs",
            "transport_cost_proxy",
            "impact_velocity_rms",
            "scuffing_ratio",
            "vertical_velocity_penalty",
            "lateral_velocity_penalty",
            "lateral_position_penalty",
            "roll_rate_penalty",
            "pitch_rate_penalty",
            "yaw_rate_penalty",
            "clearance_reward",
            "edge_reset",
            "fall_penalty",
        ):
            add_metric_tensor(stats, key, value)


def finalize_metric_tensors(stats):
    samples = torch.clamp(stats["samples"], min=1.0)
    output = {
        "reward_mean": stats["reward"] / samples,
        "weighted_metric_reward": stats["weighted_metric_reward"] / samples,
        "done_rate": stats["done"] / samples,
        "vx_err_mean": stats["vx_err"] / samples,
        "lateral_offset_mean": stats["lateral_offset"] / samples,
    }
    for key, value in stats["metrics"].items():
        output[key] = value / samples
    return output


def run_fixed_gait(env, gait_name, vx, steps, warmup_steps, gamma, time_bin_steps=0):
    action = fixed_action(env, gait_name)
    stats = init_metric_tensors(env)
    bin_stats = init_metric_tensors(env) if time_bin_steps > 0 else None
    bin_outputs = []
    bin_start_step = 0
    ppo_option_return = torch.zeros(env.num_envs, device=env.device)
    ppo_active_steps = torch.zeros(env.num_envs, device=env.device)
    ppo_active = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    # Use no_grad instead of inference_mode because this script snapshots and
    # restores simulator tensors. Inference tensors reject later inplace restore.
    with torch.no_grad():
        for step in range(steps + warmup_steps):
            set_fixed_vx(env, vx)
            _obs, reward, done, info = env.step(action)
            set_fixed_vx(env, vx)
            if step >= warmup_steps:
                eval_step = step - warmup_steps
                active_float = ppo_active.to(dtype=reward.dtype)
                # Match train_high_level_oracle_ppo.py exactly: include the
                # terminal step, discount physical steps, then stop accumulating
                # this option after the first reset.
                ppo_option_return += (gamma ** eval_step) * reward * active_float
                ppo_active_steps += active_float
                ppo_active &= ~done.bool()
                collect_step_stats(stats, env, reward, done, info)
                if bin_stats is not None:
                    collect_step_stats(bin_stats, env, reward, done, info)
                    bin_complete = (eval_step + 1) % time_bin_steps == 0
                    eval_complete = eval_step + 1 == steps
                    if bin_complete or eval_complete:
                        bin_outputs.append(
                            {
                                "start_step": bin_start_step,
                                "end_step": eval_step + 1,
                                "metrics": finalize_metric_tensors(bin_stats),
                            }
                        )
                        bin_start_step = eval_step + 1
                        bin_stats = init_metric_tensors(env)
    output = finalize_metric_tensors(stats)
    output["ppo_option_return"] = ppo_option_return
    output["ppo_active_steps"] = ppo_active_steps
    return output, bin_outputs


def context_rollout(env, gait_name, vx, steps):
    if steps <= 0:
        return
    action = fixed_action(env, gait_name)
    # Keep restored env tensors writable; see run_fixed_gait.
    with torch.no_grad():
        for _ in range(steps):
            set_fixed_vx(env, vx)
            env.step(action)
            set_fixed_vx(env, vx)


def tensor_to_float(value, index):
    return float(value[index].detach().cpu().item())


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


def summarize_deltas(rows, gait_a, gait_b):
    metric_names = sorted(
        {
            key[: -len(f"_{gait_a}")]
            for row in rows
            for key in row
            if key.endswith(f"_{gait_a}") and f"{key[: -len(f'_{gait_a}')]}_{gait_b}" in row
        }
    )
    out = []
    for metric in metric_names:
        values_a = torch.tensor([float(row[f"{metric}_{gait_a}"]) for row in rows], dtype=torch.float)
        values_b = torch.tensor([float(row[f"{metric}_{gait_b}"]) for row in rows], dtype=torch.float)
        delta = values_a - values_b
        out.append(
            {
                "metric": metric,
                f"{gait_a}_mean": float(values_a.mean().item()),
                f"{gait_b}_mean": float(values_b.mean().item()),
                "delta_mean": float(delta.mean().item()),
                "delta_median": float(delta.median().item()),
                "delta_std": float(delta.std(unbiased=False).item()),
                "delta_positive_rate": float((delta > 0.0).float().mean().item()),
            }
        )
    return out


def summarize_time_bin_deltas(rows, gait_a, gait_b):
    grouped = {}
    for row in rows:
        key = (int(row["start_step"]), int(row["end_step"]))
        grouped.setdefault(key, []).append(row)

    out = []
    for (start_step, end_step), bin_rows in sorted(grouped.items()):
        for summary in summarize_deltas(bin_rows, gait_a, gait_b):
            out.append(
                {
                    "start_step": start_step,
                    "end_step": end_step,
                    **summary,
                }
            )
    return out


def write_summary(path, args, spec, vx, delta_rows):
    important = [
        "weighted_metric_reward",
        "reward_mean",
        "vx_err_mean",
        "done_rate",
        "score_progress",
        "score_orientation",
        "score_yaw_tracking",
        "score_contact_slip",
        "score_power_efficiency",
        "score_impact",
        "score_scuffing",
        "lateral_offset_mean",
        "mechanical_power_abs",
        "impact_velocity_rms",
        "scuffing_ratio",
    ]
    by_metric = {row["metric"]: row for row in delta_rows}
    lines = [
        "# Paired Fixed-Gait Live Reward Audit",
        "",
        f"- task: `{spec.task_id}`",
        f"- condition: `{spec.condition}`",
        f"- vx: `{vx:.2f}`",
        f"- gait_a: `{args.gait_a}`",
        f"- gait_b: `{args.gait_b}`",
        f"- delta: `{args.gait_a} - {args.gait_b}`",
        f"- context_gait: `{args.context_gait}`",
        f"- context_steps: `{args.context_steps}`",
        "",
        "| metric | gait_a | gait_b | delta mean | delta median | delta std | P(delta>0) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for metric in important:
        row = by_metric.get(metric)
        if row is None:
            continue
        lines.append(
            f"| {metric} | {row[f'{args.gait_a}_mean']:.6f} "
            f"| {row[f'{args.gait_b}_mean']:.6f} "
            f"| {row['delta_mean']:.6f} "
            f"| {row['delta_median']:.6f} "
            f"| {row['delta_std']:.6f} "
            f"| {row['delta_positive_rate']:.3f} |"
        )
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", default="gait-conditioned-agility/pretrain-go2/train")
    parser.add_argument("--run-index", type=int, default=0)
    parser.add_argument("--task-map", default=str(MAINLINE_TASK_MAP))
    parser.add_argument("--eval", default=DEFAULT_EVAL)
    parser.add_argument("--gait-a", default="pronking", choices=GAIT_NAMES)
    parser.add_argument("--gait-b", default="trotting", choices=GAIT_NAMES)
    parser.add_argument("--context-gait", default="trotting", choices=GAIT_NAMES)
    parser.add_argument("--context-steps", type=int, default=20)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument(
        "--time-bin-steps",
        type=int,
        default=0,
        help="Also write paired metric summaries for consecutive step bins; 0 disables it.",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help="Per-physical-step discount used for the PPO-matched option return.",
    )
    parser.add_argument("--terrain-size", type=float, default=TRAIN_TERRAIN_SIZE)
    parser.add_argument(
        "--terrain-length",
        type=float,
        default=None,
        help="Forward terrain length in metres; defaults to --terrain-size.",
    )
    parser.add_argument(
        "--terrain-width",
        type=float,
        default=None,
        help="Lateral terrain width in metres; defaults to --terrain-size.",
    )
    parser.add_argument("--edge-reset-margin", type=float, default=TRAIN_EDGE_RESET_MARGIN)
    parser.add_argument("--teleport-thresh", type=float, default=TRAIN_TELEPORT_THRESH)
    parser.add_argument("--mesh-type", default=TRAIN_MESH_TYPE, choices=["heightfield", "trimesh"])
    parser.add_argument("--selector-hold-steps", type=int, default=0)
    parser.add_argument(
        "--style-reward-scale",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--reward-profile",
        default="canonical_efficiency_v4_physical",
        choices=REWARD_PROFILE_CHOICES,
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()

    specs = read_task_specs(
        args.task_map,
        style_reward_scale=args.style_reward_scale,
        reward_profile=args.reward_profile,
    )
    spec, vx = parse_eval_item(args.eval, specs)
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else Path("runs/high_level_oracle_gait/paired_gait_live_reward_audit")
        / time.strftime("%Y%m%d_%H%M%S")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    logdir = find_logdir(args.label, args.run_index)
    low_policy = load_low_level_policy(logdir)
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
        terrain_length=args.terrain_length,
        terrain_width=args.terrain_width,
    )

    rows = []
    time_bin_rows = []
    try:
        for repeat in range(args.repeats):
            env.reset()
            set_fixed_vx(env, vx)
            context_rollout(env, args.context_gait, vx, args.context_steps)
            state = snapshot_env(env)

            restore_env(env, state)
            result_a, bins_a = run_fixed_gait(
                env,
                args.gait_a,
                vx,
                args.steps,
                args.warmup_steps,
                args.gamma,
                args.time_bin_steps,
            )
            restore_env(env, state)
            result_b, bins_b = run_fixed_gait(
                env,
                args.gait_b,
                vx,
                args.steps,
                args.warmup_steps,
                args.gamma,
                args.time_bin_steps,
            )

            metric_names = sorted(set(result_a) | set(result_b))
            for env_id in range(args.num_envs):
                row = {
                    "repeat": repeat,
                    "env_id": env_id,
                    "task_id": spec.task_id,
                    "condition": spec.condition,
                    "cmd_vx": vx,
                    "gait_a": args.gait_a,
                    "gait_b": args.gait_b,
                    "context_gait": args.context_gait,
                    "context_steps": args.context_steps,
                }
                for metric in metric_names:
                    if metric in result_a:
                        row[f"{metric}_{args.gait_a}"] = tensor_to_float(result_a[metric], env_id)
                    if metric in result_b:
                        row[f"{metric}_{args.gait_b}"] = tensor_to_float(result_b[metric], env_id)
                rows.append(row)
            if len(bins_a) != len(bins_b):
                raise RuntimeError("Paired gait runs produced different time-bin counts")
            for bin_a, bin_b in zip(bins_a, bins_b):
                if (bin_a["start_step"], bin_a["end_step"]) != (
                    bin_b["start_step"],
                    bin_b["end_step"],
                ):
                    raise RuntimeError("Paired gait runs produced mismatched time bins")
                bin_metric_names = sorted(set(bin_a["metrics"]) | set(bin_b["metrics"]))
                for env_id in range(args.num_envs):
                    bin_row = {
                        "repeat": repeat,
                        "env_id": env_id,
                        "task_id": spec.task_id,
                        "condition": spec.condition,
                        "cmd_vx": vx,
                        "gait_a": args.gait_a,
                        "gait_b": args.gait_b,
                        "context_gait": args.context_gait,
                        "context_steps": args.context_steps,
                        "start_step": bin_a["start_step"],
                        "end_step": bin_a["end_step"],
                    }
                    for metric in bin_metric_names:
                        if metric in bin_a["metrics"]:
                            bin_row[f"{metric}_{args.gait_a}"] = tensor_to_float(
                                bin_a["metrics"][metric], env_id
                            )
                        if metric in bin_b["metrics"]:
                            bin_row[f"{metric}_{args.gait_b}"] = tensor_to_float(
                                bin_b["metrics"][metric], env_id
                            )
                    time_bin_rows.append(bin_row)
            print(
                f"repeat={repeat} task={spec.task_id} vx={vx:.2f} "
                f"{args.gait_a} reward={result_a['weighted_metric_reward'].mean().item():.4f} "
                f"{args.gait_b} reward={result_b['weighted_metric_reward'].mean().item():.4f}"
            )
    finally:
        del env
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    delta_rows = summarize_deltas(rows, args.gait_a, args.gait_b)
    env_csv = output_dir / "paired_env_metrics.csv"
    delta_csv = output_dir / "paired_metric_deltas.csv"
    summary_path = output_dir / "summary.md"
    write_csv(env_csv, rows)
    write_csv(delta_csv, delta_rows)
    write_summary(summary_path, args, spec, vx, delta_rows)
    print(f"Wrote: {env_csv}")
    print(f"Wrote: {delta_csv}")
    print(f"Wrote: {summary_path}")
    if time_bin_rows:
        time_bin_csv = output_dir / "paired_time_bin_metrics.csv"
        time_bin_delta_csv = output_dir / "paired_time_bin_deltas.csv"
        write_csv(time_bin_csv, time_bin_rows)
        write_csv(
            time_bin_delta_csv,
            summarize_time_bin_deltas(time_bin_rows, args.gait_a, args.gait_b),
        )
        print(f"Wrote: {time_bin_csv}")
        print(f"Wrote: {time_bin_delta_csv}")


if __name__ == "__main__":
    main()
