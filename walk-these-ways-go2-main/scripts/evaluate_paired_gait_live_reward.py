import argparse
import csv
import random
import time
from pathlib import Path

import isaacgym

assert isaacgym
from isaacgym import gymtorch
import numpy as np
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


def parse_residuals(text):
    values = tuple(float(value.strip()) for value in text.split(","))
    if len(values) != 5:
        raise argparse.ArgumentTypeError(
            "Expected five comma-separated residuals: "
            "frequency,duration,footswing,stance_width,body_pitch"
        )
    if any(value < -1.0 or value > 1.0 for value in values):
        raise argparse.ArgumentTypeError("Every residual must be within [-1, 1]")
    return values


def parse_initial_phase(text):
    normalized = text.strip().lower()
    if normalized in ("preserve", "none"):
        return None
    value = float(normalized)
    if value < 0.0 or value >= 1.0:
        raise argparse.ArgumentTypeError(
            "Initial gait phase must be within [0, 1), or use 'preserve'"
        )
    return value


def phase_label(phase):
    return "preserve" if phase is None else f"{phase:.2f}"


def fixed_action(env, gait_name, residuals):
    gait_id = GAIT_NAMES.index(gait_name)
    action = torch.zeros(env.num_envs, env.num_high_level_actions, device=env.device)
    action[:, gait_id] = 1.0
    residual_start = len(GAIT_NAMES)
    residual_end = residual_start + len(residuals)
    if residual_end > action.shape[1]:
        raise ValueError(
            f"Action has {action.shape[1]} dimensions, but gait plus residuals "
            f"requires {residual_end}"
        )
    action[:, residual_start:residual_end] = torch.tensor(
        residuals,
        device=env.device,
        dtype=action.dtype,
    )
    return action


def set_initial_gait_phase(env, phase):
    if phase is None:
        return
    base = env.env._get_base_env()
    base.gait_indices.fill_(phase)


def set_gait_template_transition(env, source_gait, target_gait, step, transition_steps):
    high = env.env
    if transition_steps <= 0 or step >= transition_steps:
        high.set_gait_command_override(None)
        return
    source_id = GAIT_NAMES.index(source_gait)
    target_id = GAIT_NAMES.index(target_gait)
    alpha = float(step + 1) / float(transition_steps)
    gait_command = torch.lerp(
        high.gait_templates[source_id],
        high.gait_templates[target_id],
        alpha,
    )
    high.set_gait_command_override(gait_command.unsqueeze(0).expand(env.num_envs, -1))


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


def run_fixed_gait(
    env,
    gait_name,
    residuals,
    vx,
    steps,
    warmup_steps,
    gamma,
    time_bin_steps=0,
    initial_phase=None,
    transition_from_gait=None,
    transition_from_residuals=None,
    template_transition_steps=0,
    contact_aware_switch=False,
    contact_switch_min_feet=3,
    contact_switch_max_abs_vz=0.20,
    contact_switch_max_wait_steps=5,
    contact_switch_phase=0.0,
):
    set_initial_gait_phase(env, initial_phase)
    target_action = fixed_action(env, gait_name, residuals)
    source_action = fixed_action(
        env,
        transition_from_gait or gait_name,
        transition_from_residuals or residuals,
    )
    switched = torch.full(
        (env.num_envs,),
        not contact_aware_switch,
        dtype=torch.bool,
        device=env.device,
    )
    switch_step = torch.full((env.num_envs,), -1, dtype=torch.long, device=env.device)
    safe_trigger = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    forced_trigger = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    switch_contact_count = torch.full(
        (env.num_envs,), -1.0, dtype=torch.float, device=env.device
    )
    switch_abs_vz = torch.full((env.num_envs,), -1.0, dtype=torch.float, device=env.device)
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
            if contact_aware_switch:
                env.env.set_gait_command_override(None)
                contacts = env.env._current_foot_contacts()
                contact_count = contacts.sum(dim=1)
                abs_vz = torch.abs(env.env._get_base_env().base_lin_vel[:, 2])
                safe_now = (contact_count >= contact_switch_min_feet) & (
                    abs_vz <= contact_switch_max_abs_vz
                )
                force_now = torch.full_like(safe_now, step >= contact_switch_max_wait_steps)
                newly_switched = (~switched) & (safe_now | force_now)
                if torch.any(newly_switched):
                    switch_step[newly_switched] = step
                    safe_trigger[newly_switched] = safe_now[newly_switched]
                    forced_trigger[newly_switched] = ~safe_now[newly_switched]
                    switch_contact_count[newly_switched] = contact_count[newly_switched].float()
                    switch_abs_vz[newly_switched] = abs_vz[newly_switched]
                    base = env.env._get_base_env()
                    base.gait_indices[newly_switched] = contact_switch_phase
                    switched |= newly_switched
                action = torch.where(switched.unsqueeze(1), target_action, source_action)
            else:
                set_gait_template_transition(
                    env,
                    transition_from_gait or gait_name,
                    gait_name,
                    step,
                    template_transition_steps,
                )
                action = target_action
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
    env.env.set_gait_command_override(None)
    output = finalize_metric_tensors(stats)
    output["ppo_option_return"] = ppo_option_return
    output["ppo_active_steps"] = ppo_active_steps
    if contact_aware_switch:
        output["transition_switch_step"] = switch_step.float()
        output["transition_safe_trigger"] = safe_trigger.float()
        output["transition_forced_trigger"] = forced_trigger.float()
        output["transition_contact_count"] = switch_contact_count
        output["transition_abs_vz"] = switch_abs_vz
    return output, bin_outputs


def context_rollout(env, gait_name, residuals, vx, steps):
    if steps <= 0:
        return
    action = fixed_action(env, gait_name, residuals)
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
        f"- low_level_run_dir: `{args.resolved_low_level_run_dir}`",
        f"- task: `{spec.task_id}`",
        f"- condition: `{spec.condition}`",
        f"- vx: `{vx:.2f}`",
        f"- gait_a: `{args.gait_a}`",
        f"- gait_a_residuals: `{','.join(str(value) for value in args.gait_a_residuals)}`",
        f"- gait_a_initial_phase: `{phase_label(args.gait_a_initial_phase)}`",
        f"- gait_a_template_transition_steps: `{args.gait_a_template_transition_steps}`",
        f"- gait_a_contact_aware_switch: `{args.gait_a_contact_aware_switch}`",
        f"- gait_b: `{args.gait_b}`",
        f"- gait_b_residuals: `{','.join(str(value) for value in args.gait_b_residuals)}`",
        f"- gait_b_initial_phase: `{phase_label(args.gait_b_initial_phase)}`",
        f"- gait_b_template_transition_steps: `{args.gait_b_template_transition_steps}`",
        f"- gait_b_contact_aware_switch: `{args.gait_b_contact_aware_switch}`",
        f"- delta: `{args.gait_a} - {args.gait_b}`",
        f"- context_gait: `{args.context_gait}`",
        f"- context_residuals: `{','.join(str(value) for value in args.context_residuals)}`",
        f"- context_steps: `{args.context_steps}`",
        f"- high_level_dt: `{args.high_level_dt}`",
        f"- contact_switch_min_feet: `{args.contact_switch_min_feet}`",
        f"- contact_switch_max_abs_vz: `{args.contact_switch_max_abs_vz}`",
        f"- contact_switch_max_wait_steps: `{args.contact_switch_max_wait_steps}`",
        f"- contact_switch_phase: `{args.contact_switch_phase}`",
        f"- branch_order: `{args.branch_order}`",
        f"- seed: `{args.seed}`",
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
    parser.add_argument(
        "--low-level-run-dir",
        default=None,
        help=(
            "Load parameters.pkl and exported low-level policy modules from this "
            "run directory instead of resolving --label/--run-index."
        ),
    )
    parser.add_argument("--task-map", default=str(MAINLINE_TASK_MAP))
    parser.add_argument("--eval", default=DEFAULT_EVAL)
    parser.add_argument("--gait-a", default="pronking", choices=GAIT_NAMES)
    parser.add_argument("--gait-b", default="trotting", choices=GAIT_NAMES)
    parser.add_argument(
        "--gait-a-residuals",
        type=parse_residuals,
        default=parse_residuals("0,0,0,0,0"),
    )
    parser.add_argument(
        "--gait-b-residuals",
        type=parse_residuals,
        default=parse_residuals("0,0,0,0,0"),
    )
    parser.add_argument(
        "--gait-a-initial-phase",
        type=parse_initial_phase,
        default=None,
        metavar="PHASE|preserve",
        help=(
            "Set the base gait clock once before gait A starts. "
            "Use a value in [0, 1), or preserve the context clock (default)."
        ),
    )
    parser.add_argument(
        "--gait-b-initial-phase",
        type=parse_initial_phase,
        default=None,
        metavar="PHASE|preserve",
        help=(
            "Set the base gait clock once before gait B starts. "
            "Use a value in [0, 1), or preserve the context clock (default)."
        ),
    )
    parser.add_argument(
        "--gait-a-template-transition-steps",
        type=int,
        default=0,
        help=(
            "Interpolate gait phase/offset/bound from the context gait to gait A "
            "over this many high-level steps; 0 keeps the immediate switch."
        ),
    )
    parser.add_argument(
        "--gait-b-template-transition-steps",
        type=int,
        default=0,
        help=(
            "Interpolate gait phase/offset/bound from the context gait to gait B "
            "over this many high-level steps; 0 keeps the immediate switch."
        ),
    )
    parser.add_argument("--gait-a-contact-aware-switch", action="store_true")
    parser.add_argument("--gait-b-contact-aware-switch", action="store_true")
    parser.add_argument("--contact-switch-min-feet", type=int, default=3)
    parser.add_argument("--contact-switch-max-abs-vz", type=float, default=0.20)
    parser.add_argument("--contact-switch-max-wait-steps", type=int, default=5)
    parser.add_argument(
        "--contact-switch-phase",
        type=parse_initial_phase,
        default=0.0,
        metavar="PHASE",
    )
    parser.add_argument("--context-gait", default="trotting", choices=GAIT_NAMES)
    parser.add_argument(
        "--context-residuals",
        type=parse_residuals,
        default=None,
        help="Defaults to --gait-b-residuals.",
    )
    parser.add_argument("--context-steps", type=int, default=20)
    parser.add_argument(
        "--high-level-dt",
        type=float,
        default=0.10,
        help="Seconds per evaluator action/check step; training defaults to 0.10.",
    )
    parser.add_argument(
        "--branch-order",
        default="ab",
        choices=("ab", "ba"),
        help=(
            "Order used after the shared context: 'ab' runs gait A first; "
            "'ba' runs gait B first. Use both orders to diagnose simulator restore effects."
        ),
    )
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
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    if args.gait_a_template_transition_steps < 0 or args.gait_b_template_transition_steps < 0:
        parser.error("Gait template transition steps must be non-negative")
    if args.gait_a_contact_aware_switch and args.gait_a_template_transition_steps:
        parser.error("Gait A cannot use contact-aware and interpolated switching together")
    if args.gait_b_contact_aware_switch and args.gait_b_template_transition_steps:
        parser.error("Gait B cannot use contact-aware and interpolated switching together")
    if args.gait_a_contact_aware_switch and args.gait_a_initial_phase is not None:
        parser.error("Use --contact-switch-phase instead of --gait-a-initial-phase")
    if args.gait_b_contact_aware_switch and args.gait_b_initial_phase is not None:
        parser.error("Use --contact-switch-phase instead of --gait-b-initial-phase")
    if args.contact_switch_min_feet < 1 or args.contact_switch_min_feet > 4:
        parser.error("--contact-switch-min-feet must be between 1 and 4")
    if args.contact_switch_max_abs_vz < 0.0:
        parser.error("--contact-switch-max-abs-vz must be non-negative")
    if args.contact_switch_max_wait_steps < 0:
        parser.error("--contact-switch-max-wait-steps must be non-negative")
    if args.contact_switch_phase is None:
        parser.error("--contact-switch-phase requires a numeric phase")
    if args.high_level_dt <= 0.0:
        parser.error("--high-level-dt must be positive")
    if args.context_residuals is None:
        args.context_residuals = args.gait_b_residuals

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

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

    if args.low_level_run_dir is None:
        logdir = Path(find_logdir(args.label, args.run_index)).resolve()
    else:
        logdir = Path(args.low_level_run_dir).expanduser().resolve()
        required_files = (
            logdir / "parameters.pkl",
            logdir / "checkpoints" / "body_latest.jit",
            logdir / "checkpoints" / "adaptation_module_latest.jit",
        )
        missing = [str(path) for path in required_files if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                "Low-level run directory is incomplete; missing: " + ", ".join(missing)
            )
    args.resolved_low_level_run_dir = str(logdir)
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
        high_level_dt=args.high_level_dt,
        terrain_length=args.terrain_length,
        terrain_width=args.terrain_width,
    )

    rows = []
    time_bin_rows = []
    try:
        for repeat in range(args.repeats):
            env.reset()
            set_fixed_vx(env, vx)
            context_rollout(
                env,
                args.context_gait,
                args.context_residuals,
                vx,
                args.context_steps,
            )
            state = snapshot_env(env)

            branch_specs = {
                "a": (
                    args.gait_a,
                    args.gait_a_residuals,
                    args.gait_a_initial_phase,
                    args.gait_a_template_transition_steps,
                    args.gait_a_contact_aware_switch,
                ),
                "b": (
                    args.gait_b,
                    args.gait_b_residuals,
                    args.gait_b_initial_phase,
                    args.gait_b_template_transition_steps,
                    args.gait_b_contact_aware_switch,
                ),
            }
            branch_results = {}
            for branch in args.branch_order:
                gait_name, residuals, initial_phase, transition_steps, contact_switch = branch_specs[
                    branch
                ]
                restore_env(env, state)
                branch_results[branch] = run_fixed_gait(
                    env,
                    gait_name,
                    residuals,
                    vx,
                    args.steps,
                    args.warmup_steps,
                    args.gamma,
                    args.time_bin_steps,
                    initial_phase,
                    args.context_gait,
                    args.context_residuals,
                    transition_steps,
                    contact_switch,
                    args.contact_switch_min_feet,
                    args.contact_switch_max_abs_vz,
                    args.contact_switch_max_wait_steps,
                    args.contact_switch_phase,
                )
            result_a, bins_a = branch_results["a"]
            result_b, bins_b = branch_results["b"]

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
                    "gait_a_initial_phase": phase_label(args.gait_a_initial_phase),
                    "gait_b_initial_phase": phase_label(args.gait_b_initial_phase),
                    "gait_a_template_transition_steps": args.gait_a_template_transition_steps,
                    "gait_b_template_transition_steps": args.gait_b_template_transition_steps,
                    "gait_a_contact_aware_switch": args.gait_a_contact_aware_switch,
                    "gait_b_contact_aware_switch": args.gait_b_contact_aware_switch,
                    "branch_order": args.branch_order,
                    "first_branch": args.branch_order[0],
                    "context_gait": args.context_gait,
                    "context_steps": args.context_steps,
                    "high_level_dt": args.high_level_dt,
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
                        "gait_a_initial_phase": phase_label(args.gait_a_initial_phase),
                        "gait_b_initial_phase": phase_label(args.gait_b_initial_phase),
                        "gait_a_template_transition_steps": args.gait_a_template_transition_steps,
                        "gait_b_template_transition_steps": args.gait_b_template_transition_steps,
                        "gait_a_contact_aware_switch": args.gait_a_contact_aware_switch,
                        "gait_b_contact_aware_switch": args.gait_b_contact_aware_switch,
                        "branch_order": args.branch_order,
                        "first_branch": args.branch_order[0],
                        "context_gait": args.context_gait,
                        "context_steps": args.context_steps,
                        "high_level_dt": args.high_level_dt,
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
