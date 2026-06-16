import argparse
import csv
import json
from itertools import product
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

TRAINING_RANGE_EVAL = (
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
    # The trainer samples push_lateral from [1.2, 1.8] when only one training
    # map speed is active, so audit the lower edge, center, and upper edge.
    "push_lateral_pace_recovery:1.2,"
    "push_lateral_pace_recovery:1.5,"
    "push_lateral_pace_recovery:1.8,"
    # The trainer samples stepping_stones_easy from [1.7, 2.0].
    "stepping_stones_easy_bound_highspeed:1.7,"
    "stepping_stones_easy_bound_highspeed:2.0"
)

EXTENDED_EVAL = (
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
    "push_lateral_pace_recovery:0.5,"
    "push_lateral_pace_recovery:1.0,"
    "push_lateral_pace_recovery:1.5,"
    "push_lateral_pace_recovery:2.0,"
    "stepping_stones_easy_bound_highspeed:1.0,"
    "stepping_stones_easy_bound_highspeed:1.5,"
    "stepping_stones_easy_bound_highspeed:2.0"
)

BEHAVIOR_NAMES = (
    "frequency",
    "duration",
    "footswing_height",
    "stance_width",
    "body_pitch",
)

SCORE_OBJECTIVES = {
    "flat": (
        "progress_score",
        "survival_score",
        "orientation_score",
        "slip_score",
        "energy_score",
        "smoothness_score",
    ),
    "ramp_up": (
        "progress_score",
        "survival_score",
        "orientation_score",
        "slip_score",
        "energy_score",
        "impact_score",
    ),
    "rough_slope": (
        "progress_score",
        "survival_score",
        "orientation_score",
        "slip_score",
        "lateral_score",
        "impact_score",
    ),
    "push_lateral": (
        "progress_score",
        "survival_score",
        "lateral_score",
        "yaw_score",
        "orientation_score",
    ),
    "stepping_stones_easy": (
        "progress_score",
        "survival_score",
        "scuffing_score",
        "impact_score",
        "lateral_score",
        "orientation_score",
    ),
}

NEUTRAL_WEIGHTS = {
    "flat": {
        "progress_score": 2.0,
        "survival_score": 2.0,
        "orientation_score": 1.0,
        "slip_score": 1.0,
        "energy_score": 0.7,
        "smoothness_score": 0.5,
        "yaw_score": 0.3,
    },
    "ramp_up": {
        "progress_score": 2.0,
        "survival_score": 2.0,
        "orientation_score": 1.4,
        "slip_score": 1.0,
        "energy_score": 0.6,
        "impact_score": 0.5,
        "yaw_score": 0.3,
    },
    "rough_slope": {
        "progress_score": 1.8,
        "survival_score": 2.0,
        "orientation_score": 1.5,
        "slip_score": 1.0,
        "lateral_score": 0.8,
        "impact_score": 0.5,
    },
    "push_lateral": {
        "progress_score": 1.3,
        "survival_score": 2.0,
        "lateral_score": 2.0,
        "yaw_score": 1.0,
        "orientation_score": 1.2,
        "energy_score": 0.3,
    },
    "stepping_stones_easy": {
        "progress_score": 1.6,
        "survival_score": 2.0,
        "scuffing_score": 1.3,
        "impact_score": 0.8,
        "lateral_score": 0.8,
        "orientation_score": 0.8,
        "energy_score": 0.3,
    },
}


def parse_floats(text):
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def parse_strings(text):
    return [item.strip() for item in text.split(",") if item.strip()]


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
    values = parse_strings(text)
    unknown = [value for value in values if value not in GAIT_NAMES]
    if unknown:
        raise ValueError(f"Unknown gait(s): {unknown}. Choices: {list(GAIT_NAMES)}")
    return values


def clamp(value, low, high):
    return max(low, min(high, value))


def exp_score(value, scale):
    return float(torch.exp(torch.tensor(-float(value) / scale)).item())


def exp_sq_score(value, scale):
    return float(torch.exp(torch.tensor(-(float(value) ** 2) / scale)).item())


def set_fixed_vx(env, vx):
    env.vx_cmd[:] = vx
    env.env.set_velocity_command(env.vx_cmd, 0.0, 0.0)


def get_base_env(high_env):
    if hasattr(high_env, "_get_base_env"):
        return high_env._get_base_env()
    env = high_env
    while hasattr(env, "env"):
        env = env.env
    return env


def sample_ground_heights(base_env, foot_positions):
    if not hasattr(base_env, "height_samples") or not hasattr(base_env, "terrain"):
        return torch.zeros_like(foot_positions[:, :, 2])

    height_samples = base_env.height_samples
    terrain_cfg = base_env.terrain.cfg
    points = foot_positions[:, :, :2] + terrain_cfg.border_size
    px = torch.clamp(
        (points[:, :, 0] / terrain_cfg.horizontal_scale).long(),
        0,
        height_samples.shape[0] - 2,
    )
    py = torch.clamp(
        (points[:, :, 1] / terrain_cfg.horizontal_scale).long(),
        0,
        height_samples.shape[1] - 2,
    )
    heights1 = height_samples[px, py]
    heights2 = height_samples[px + 1, py]
    heights3 = height_samples[px, py + 1]
    return torch.minimum(torch.minimum(heights1, heights2), heights3) * terrain_cfg.vertical_scale


def behavior_ranges_for_gait(env, gait_name):
    gait_id = GAIT_NAMES.index(gait_name)
    base = env.env.gait_behavior_templates[gait_id].detach().cpu().tolist()
    delta = env.env.residual_delta_ranges.detach().cpu().tolist()
    lows = env.env.behavior_lows.detach().cpu().tolist()
    highs = env.env.behavior_highs.detach().cpu().tolist()
    ranges = {}
    for index, name in enumerate(BEHAVIOR_NAMES):
        low = clamp(base[index] + delta[index][0], lows[index], highs[index])
        high = clamp(base[index] + delta[index][1], lows[index], highs[index])
        ranges[name] = (low, high)
    return ranges


def residual_from_behavior(env, gait_name, values):
    gait_id = GAIT_NAMES.index(gait_name)
    base = env.env.gait_behavior_templates[gait_id].detach().cpu()
    delta = env.env.residual_delta_ranges.detach().cpu()
    residual = []
    clipped = {}
    for index, name in enumerate(BEHAVIOR_NAMES):
        low_delta, high_delta = float(delta[index, 0]), float(delta[index, 1])
        low_behavior = float(env.env.behavior_lows[index].item())
        high_behavior = float(env.env.behavior_highs[index].item())
        desired = clamp(float(values[name]), low_behavior, high_behavior)
        unit = (desired - float(base[index])) - low_delta
        unit = unit / max(high_delta - low_delta, 1e-6)
        action = clamp(2.0 * unit - 1.0, -1.0, 1.0)
        residual.append(action)
        clipped[name] = desired
    return residual, clipped


def build_action_space_grid(env, args, gaits):
    residual_lists = {
        "frequency": args.freq_residuals,
        "duration": args.duration_residuals,
        "footswing_height": args.footswing_residuals,
        "stance_width": args.stance_residuals,
        "body_pitch": args.body_pitch_residuals,
    }
    grid = []
    for gait in gaits:
        gait_id = GAIT_NAMES.index(gait)
        for values in product(*(residual_lists[name] for name in BEHAVIOR_NAMES)):
            residual = [clamp(value, -1.0, 1.0) for value in values]
            action = torch.zeros(1, env.num_high_level_actions, device=env.device)
            action[:, gait_id] = 1.0
            action[:, env.num_gaits :] = torch.tensor(residual, device=env.device)
            mapped = env.env._map_action(action)
            row = {
                "gait": gait,
                "gait_id": gait_id,
                "grid_mode": "action_space",
            }
            for index, name in enumerate(BEHAVIOR_NAMES):
                row[f"{name}_residual"] = residual[index]
                row[name] = float(mapped[name][0].item())
            grid.append(row)
    return grid


def build_physical_grid(env, args, gaits):
    value_lists = {
        "frequency": args.frequencies,
        "duration": args.durations,
        "footswing_height": args.footswing_heights,
        "stance_width": args.stance_widths,
        "body_pitch": args.body_pitches,
    }
    grid = []
    for gait in gaits:
        gait_id = GAIT_NAMES.index(gait)
        reachable = behavior_ranges_for_gait(env, gait)
        for values in product(*(value_lists[name] for name in BEHAVIOR_NAMES)):
            desired = {name: values[index] for index, name in enumerate(BEHAVIOR_NAMES)}
            if args.drop_unreachable:
                if any(
                    desired[name] < reachable[name][0] - 1e-6
                    or desired[name] > reachable[name][1] + 1e-6
                    for name in BEHAVIOR_NAMES
                ):
                    continue
            residual, clipped = residual_from_behavior(env, gait, desired)
            row = {
                "gait": gait,
                "gait_id": gait_id,
                "grid_mode": "physical",
            }
            for index, name in enumerate(BEHAVIOR_NAMES):
                row[f"{name}_requested"] = desired[name]
                row[f"{name}_residual"] = residual[index]
                row[name] = clipped[name]
            grid.append(row)
    return grid


def build_action(env, rows, repeats_per_config):
    actions = torch.zeros(
        len(rows) * repeats_per_config,
        env.num_high_level_actions,
        device=env.device,
    )
    for row_index, row in enumerate(rows):
        start = row_index * repeats_per_config
        end = start + repeats_per_config
        actions[start:end, int(row["gait_id"])] = 1.0
        residual = [float(row[f"{name}_residual"]) for name in BEHAVIOR_NAMES]
        actions[start:end, env.num_gaits :] = torch.tensor(residual, device=env.device)
    return actions


def pad_batch(batch, target_size):
    if not batch or len(batch) >= target_size:
        return batch
    padded = list(batch)
    while len(padded) < target_size:
        padded.append(dict(batch[-1]))
    return padded


def make_stats(num_envs, device):
    return {
        "steps": 0,
        "reward": torch.zeros(num_envs, device=device),
        "weighted_metric_reward": torch.zeros(num_envs, device=device),
        "done": torch.zeros(num_envs, device=device),
        "edge_reset": torch.zeros(num_envs, device=device),
        "fall": torch.zeros(num_envs, device=device),
        "measured_vx": torch.zeros(num_envs, device=device),
        "vx_abs_error": torch.zeros(num_envs, device=device),
        "vy_abs": torch.zeros(num_envs, device=device),
        "yaw_abs": torch.zeros(num_envs, device=device),
        "lateral_offset_abs": torch.zeros(num_envs, device=device),
        "lateral_vel_abs": torch.zeros(num_envs, device=device),
        "lateral_vel_sq": torch.zeros(num_envs, device=device),
        "base_z_vel_abs": torch.zeros(num_envs, device=device),
        "base_z_vel_sq": torch.zeros(num_envs, device=device),
        "roll_rate_abs": torch.zeros(num_envs, device=device),
        "roll_rate_sq": torch.zeros(num_envs, device=device),
        "pitch_rate_abs": torch.zeros(num_envs, device=device),
        "pitch_rate_sq": torch.zeros(num_envs, device=device),
        "yaw_rate_abs": torch.zeros(num_envs, device=device),
        "yaw_rate_sq": torch.zeros(num_envs, device=device),
        "gravity_x_sq": torch.zeros(num_envs, device=device),
        "gravity_y_sq": torch.zeros(num_envs, device=device),
        "torque_penalty": torch.zeros(num_envs, device=device),
        "slip_penalty": torch.zeros(num_envs, device=device),
        "mechanical_power_abs": torch.zeros(num_envs, device=device),
        "positive_mechanical_power": torch.zeros(num_envs, device=device),
        "contact_force_mean": torch.zeros(num_envs, device=device),
        "stance_contact_force_mean": torch.zeros(num_envs, device=device),
        "peak_contact_force": torch.zeros(num_envs, device=device),
        "swing_foot_clearance_mean": torch.zeros(num_envs, device=device),
        "scuffing_ratio": torch.zeros(num_envs, device=device),
        "foot_impact_vel_sum": torch.zeros(num_envs, device=device),
        "foot_impact_vel_sq_sum": torch.zeros(num_envs, device=device),
        "foot_impact_count": torch.zeros(num_envs, device=device),
        "forward_distance": torch.zeros(num_envs, device=device),
        "target_forward_distance": torch.zeros(num_envs, device=device),
        "actual_frequency": torch.zeros(num_envs, device=device),
        "actual_duration": torch.zeros(num_envs, device=device),
        "actual_footswing_height": torch.zeros(num_envs, device=device),
        "actual_stance_width": torch.zeros(num_envs, device=device),
        "actual_body_pitch": torch.zeros(num_envs, device=device),
        "score_sums": {},
    }


def add_step_stats(stats, env, action, reward, done, info, prev_contacts):
    high_env = env.env
    base_env = get_base_env(high_env)
    terms = info.get("high_level_reward_terms", {})
    actual = info.get("executed_high_level_action", action)
    mapped = high_env._map_action(actual)

    contacts = high_env.contact_forces[:, high_env.feet_indices, 2] > 1.0
    contacts_f = contacts.float()
    contact_count = torch.sum(contacts_f, dim=1)
    vx_error = high_env.base_lin_vel[:, 0] - high_env.commands[:, 0]
    vy_error = high_env.base_lin_vel[:, 1] - high_env.commands[:, 1]
    yaw_error = high_env.base_ang_vel[:, 2] - high_env.commands[:, 2]
    lateral_offset = high_env._compute_lateral_offset()
    edge_reset = high_env._get_edge_reset_buf()
    fall = done.bool() & ~edge_reset

    stats["steps"] += 1
    stats["reward"] += reward
    stats["weighted_metric_reward"] += terms.get("weighted_metric_reward", reward)
    stats["done"] += done.float()
    stats["edge_reset"] += edge_reset.float()
    stats["fall"] += fall.float()
    stats["measured_vx"] += high_env.base_lin_vel[:, 0]
    stats["vx_abs_error"] += torch.abs(vx_error)
    stats["vy_abs"] += torch.abs(vy_error)
    stats["yaw_abs"] += torch.abs(yaw_error)
    stats["lateral_offset_abs"] += torch.abs(lateral_offset)
    stats["lateral_vel_abs"] += torch.abs(high_env.base_lin_vel[:, 1])
    stats["lateral_vel_sq"] += high_env.base_lin_vel[:, 1] ** 2
    stats["base_z_vel_abs"] += torch.abs(high_env.base_lin_vel[:, 2])
    stats["base_z_vel_sq"] += high_env.base_lin_vel[:, 2] ** 2
    stats["roll_rate_abs"] += torch.abs(high_env.base_ang_vel[:, 0])
    stats["roll_rate_sq"] += high_env.base_ang_vel[:, 0] ** 2
    stats["pitch_rate_abs"] += torch.abs(high_env.base_ang_vel[:, 1])
    stats["pitch_rate_sq"] += high_env.base_ang_vel[:, 1] ** 2
    stats["yaw_rate_abs"] += torch.abs(high_env.base_ang_vel[:, 2])
    stats["yaw_rate_sq"] += high_env.base_ang_vel[:, 2] ** 2
    stats["gravity_x_sq"] += high_env.projected_gravity[:, 0] ** 2
    stats["gravity_y_sq"] += high_env.projected_gravity[:, 1] ** 2

    torque_penalty = torch.mean(high_env.torques**2, dim=1) / 100.0
    foot_xy_vel = torch.sum(high_env.foot_velocities[:, :, :2] ** 2, dim=2)
    slip_penalty = torch.mean(contacts * foot_xy_vel, dim=1)
    stats["torque_penalty"] += torque_penalty
    stats["slip_penalty"] += slip_penalty

    joint_power = high_env.torques * high_env.dof_vel[:, : high_env.torques.shape[1]]
    stats["mechanical_power_abs"] += torch.sum(torch.abs(joint_power), dim=1)
    stats["positive_mechanical_power"] += torch.sum(torch.clamp(joint_power, min=0.0), dim=1)

    contact_force_norm = torch.norm(high_env.contact_forces[:, high_env.feet_indices, :], dim=2)
    stats["contact_force_mean"] += torch.mean(contact_force_norm, dim=1)
    stance_force = torch.sum(contact_force_norm * contacts_f, dim=1) / torch.clamp(contact_count, min=1.0)
    stats["stance_contact_force_mean"] += stance_force
    stats["peak_contact_force"] = torch.maximum(
        stats["peak_contact_force"],
        torch.max(contact_force_norm, dim=1).values,
    )

    foot_positions = high_env.foot_positions
    ground_heights = sample_ground_heights(base_env, foot_positions)
    foot_clearance = foot_positions[:, :, 2] - ground_heights
    swing_mask = (~contacts).float()
    swing_count = torch.clamp(torch.sum(swing_mask, dim=1), min=1.0)
    stats["swing_foot_clearance_mean"] += torch.sum(swing_mask * foot_clearance, dim=1) / swing_count
    stats["scuffing_ratio"] += torch.sum(swing_mask * (foot_clearance < 0.035).float(), dim=1) / swing_count

    if prev_contacts is not None:
        new_contacts = contacts & (~prev_contacts)
        impact_vel = torch.clamp(-high_env.prev_foot_velocities[:, :, 2], min=0.0)
        stats["foot_impact_vel_sum"] += torch.sum(new_contacts * impact_vel, dim=1)
        stats["foot_impact_vel_sq_sum"] += torch.sum(new_contacts * impact_vel**2, dim=1)
        stats["foot_impact_count"] += torch.sum(new_contacts.float(), dim=1)

    dt = getattr(high_env, "dt", getattr(high_env.env, "dt", 0.02))
    stats["forward_distance"] += high_env.base_lin_vel[:, 0] * dt
    stats["target_forward_distance"] += high_env.commands[:, 0] * dt

    for name in BEHAVIOR_NAMES:
        stats[f"actual_{name}"] += mapped[name]

    for key, value in terms.items():
        if key.startswith("score_"):
            stats["score_sums"].setdefault(key, torch.zeros_like(reward))
            stats["score_sums"][key] += value

    return contacts.detach().clone()


def grouped_mean(values, num_configs, repeats):
    return values.reshape(num_configs, repeats).mean(dim=1)


def grouped_max(values, num_configs, repeats):
    return values.reshape(num_configs, repeats).max(dim=1).values


def finalize_batch(stats, rows, spec, vx, repeats_per_config, steps):
    num_configs = len(rows)
    denom = max(steps, 1)
    out_rows = []
    for row_index, row in enumerate(rows):
        result = dict(row)
        result["task_id"] = spec.task_id
        result["condition"] = spec.condition
        result["cmd_vx"] = vx
        result["target_gait"] = spec.target_gait
        result["samples"] = steps * repeats_per_config
        for key, value in stats.items():
            if key in ("steps", "score_sums"):
                continue
            if key == "peak_contact_force":
                env_values = grouped_max(value.detach(), num_configs, repeats_per_config)
            else:
                env_values = grouped_mean((value / denom).detach(), num_configs, repeats_per_config)
            result[key if key.endswith("_rate") else f"{key}_mean"] = float(env_values[row_index].item())

        for key, value in stats["score_sums"].items():
            env_values = grouped_mean((value / denom).detach(), num_configs, repeats_per_config)
            result[key] = float(env_values[row_index].item())

        result["done_rate"] = result.pop("done_mean")
        result["edge_reset_rate"] = result.pop("edge_reset_mean")
        result["fall_rate"] = result.pop("fall_mean")
        impact_count_total = grouped_mean(
            stats["foot_impact_count"].detach(), num_configs, repeats_per_config
        )[row_index].item()
        impact_vel_sum = grouped_mean(
            stats["foot_impact_vel_sum"].detach(), num_configs, repeats_per_config
        )[row_index].item()
        impact_vel_sq_sum = grouped_mean(
            stats["foot_impact_vel_sq_sum"].detach(), num_configs, repeats_per_config
        )[row_index].item()
        impact_count = max(impact_count_total, 1.0)
        result["foot_impact_rate"] = impact_count_total / denom
        result["foot_impact_vel_mean"] = impact_vel_sum / impact_count
        result["foot_impact_vel_rms"] = max(impact_vel_sq_sum / impact_count, 0.0) ** 0.5
        result["lateral_vel_rms"] = result["lateral_vel_sq_mean"] ** 0.5
        result["base_z_vel_rms"] = result["base_z_vel_sq_mean"] ** 0.5
        result["roll_rate_rms"] = result["roll_rate_sq_mean"] ** 0.5
        result["pitch_rate_rms"] = result["pitch_rate_sq_mean"] ** 0.5
        result["yaw_rate_rms"] = result["yaw_rate_sq_mean"] ** 0.5
        orientation_sq = result["gravity_x_sq_mean"] + result["gravity_y_sq_mean"]
        result["orientation_rms"] = max(orientation_sq, 0.0) ** 0.5
        target_distance = max(abs(result["target_forward_distance_mean"]), 0.2)
        result["forward_distance_ratio"] = result["forward_distance_mean"] / target_distance
        result["progress_deficit"] = max(
            result["target_forward_distance_mean"] - result["forward_distance_mean"], 0.0
        ) / target_distance
        result["transport_cost_proxy"] = result["mechanical_power_abs_mean"] / max(
            abs(result["measured_vx_mean"]), 0.2
        )
        add_neutral_scores(result)
        out_rows.append(result)
    return out_rows


def add_neutral_scores(row):
    row["progress_score"] = exp_sq_score(row["vx_abs_error_mean"], 0.25)
    row["yaw_score"] = exp_sq_score(row["yaw_abs_mean"], 0.10)
    row["orientation_score"] = exp_score(row["orientation_rms"] ** 2, 0.05)
    row["lateral_score"] = exp_score(
        row["lateral_offset_abs_mean"] + row["lateral_vel_rms"] ** 2 / 0.05,
        1.0,
    )
    row["slip_score"] = exp_score(row["slip_penalty_mean"], 0.05)
    row["energy_score"] = exp_score(row["torque_penalty_mean"], 0.50)
    row["impact_score"] = exp_sq_score(row["foot_impact_vel_rms"], 4.0)
    row["scuffing_score"] = exp_score(row["scuffing_ratio_mean"], 0.20)
    row["smoothness_score"] = row.get("score_action_smoothness", 1.0)
    row["survival_score"] = max(0.0, 1.0 - row["fall_rate"])

    weights = NEUTRAL_WEIGHTS.get(row["condition"], NEUTRAL_WEIGHTS["flat"])
    total = 0.0
    weight_sum = 0.0
    for key, weight in weights.items():
        total += weight * float(row.get(key, 0.0))
        weight_sum += weight
    row["neutral_score"] = total / max(weight_sum, 1e-6)


def dominates(a, b, objectives, eps=1e-9):
    better_or_equal = all(float(a[obj]) >= float(b[obj]) - eps for obj in objectives)
    strictly_better = any(float(a[obj]) > float(b[obj]) + eps for obj in objectives)
    return better_or_equal and strictly_better


def pareto_front(rows):
    by_group = {}
    for row in rows:
        by_group.setdefault((row["task_id"], float(row["cmd_vx"])), []).append(row)

    front = []
    for (_task_id, _vx), group in by_group.items():
        condition = group[0]["condition"]
        objectives = SCORE_OBJECTIVES.get(condition, SCORE_OBJECTIVES["flat"])
        for row in group:
            dominated = False
            for other in group:
                if other is row:
                    continue
                if dominates(other, row, objectives):
                    dominated = True
                    break
            if not dominated:
                item = dict(row)
                item["pareto_objectives"] = ",".join(objectives)
                front.append(item)
    return front


def best_rows(rows, group_keys, score_key="neutral_score"):
    grouped = {}
    for row in rows:
        key = tuple(row[name] for name in group_keys)
        if key not in grouped or float(row[score_key]) > float(grouped[key][score_key]):
            grouped[key] = row
    return list(grouped.values())


def write_csv(path, rows):
    if not rows:
        return
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def softmax_by_score(rows, temperature):
    if not rows:
        return {}
    scores = torch.tensor([float(row["neutral_score"]) for row in rows], dtype=torch.float)
    probs = torch.softmax((scores - torch.max(scores)) / max(temperature, 1e-6), dim=0)
    return {rows[i]["gait"]: float(probs[i].item()) for i in range(len(rows))}


def fmt(value):
    return f"{float(value):.3f}"


def write_summary(path, rows, temperature):
    grouped = {}
    for row in rows:
        grouped.setdefault((row["task_id"], float(row["cmd_vx"])), []).append(row)

    lines = [
        "# Fair Target-Gait Audit",
        "",
        "This audit gives every gait an equal continuous-parameter search budget.",
        "It reports neutral weighted scores, raw metrics, and Pareto candidates.",
        "",
    ]
    for (task_id, vx), group in sorted(grouped.items()):
        best_by_gait = best_rows(group, ["gait"])
        ranked = sorted(best_by_gait, key=lambda row: float(row["neutral_score"]), reverse=True)
        probs = softmax_by_score(ranked, temperature)
        lines += [
            f"## {task_id} vx={vx:.2f}",
            "",
            f"- target_gait_from_task_map: `{ranked[0]['target_gait']}`",
            f"- best_gait_by_neutral_score: `{ranked[0]['gait']}`",
            "- soft_distribution_from_best_per_gait: "
            + ", ".join(f"{gait}={probs[gait]:.3f}" for gait in probs),
            "",
            "| rank | gait | neutral | live_weighted | vx_err | fall | lateral | scuff | impact | energy | params |",
            "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
        for rank, row in enumerate(ranked, start=1):
            params = (
                f"f={float(row['actual_frequency_mean']):.2f}, "
                f"d={float(row['actual_duration_mean']):.2f}, "
                f"foot={float(row['actual_footswing_height_mean']):.3f}, "
                f"width={float(row['actual_stance_width_mean']):.3f}, "
                f"pitch={float(row['actual_body_pitch_mean']):.3f}"
            )
            lines.append(
                f"| {rank} | {row['gait']} "
                f"| {fmt(row['neutral_score'])} "
                f"| {fmt(row['weighted_metric_reward_mean'])} "
                f"| {fmt(row['vx_abs_error_mean'])} "
                f"| {fmt(row['fall_rate'])} "
                f"| {fmt(row['lateral_offset_abs_mean'])} "
                f"| {fmt(row['scuffing_ratio_mean'])} "
                f"| {fmt(row['foot_impact_vel_rms'])} "
                f"| {fmt(row['transport_cost_proxy'])} "
                f"| {params} |"
            )
        lines.append("")
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def save_outputs(output_dir, rows, args):
    output_dir.mkdir(parents=True, exist_ok=True)
    best_by_gait = best_rows(rows, ["task_id", "cmd_vx", "gait"])
    best_by_task = best_rows(rows, ["task_id", "cmd_vx"])
    front = pareto_front(rows)
    write_csv(output_dir / "fair_gait_grid_results.csv", rows)
    write_csv(output_dir / "best_by_task_speed_gait.csv", best_by_gait)
    write_csv(output_dir / "best_by_task_speed.csv", best_by_task)
    write_csv(output_dir / "pareto_front.csv", front)
    write_summary(output_dir / "summary.md", rows, args.softmax_temperature)
    config = vars(args).copy()
    config["output_dir"] = str(output_dir)
    with (output_dir / "run_config.json").open("w") as file:
        json.dump(config, file, indent=2)


def run_eval_item(args, spec, vx, output_dir):
    logdir = find_logdir(args.label, args.run_index)
    low_policy = load_low_level_policy(logdir)
    env = OracleConditionHighLevelEnv(
        [spec],
        logdir,
        low_policy,
        args.batch_size * args.repeats_per_config,
        render=args.render,
        oracle_condition_obs=False,
        terrain_size=args.terrain_size,
        edge_reset_margin=args.edge_reset_margin,
        teleport_thresh=args.teleport_thresh,
        mesh_type=args.mesh_type,
        selector_hold_steps=0,
    )

    gaits = parse_gaits(args.gaits)
    if args.grid_mode == "action-space":
        grid = build_action_space_grid(env, args, gaits)
    else:
        grid = build_physical_grid(env, args, gaits)
    end_index = len(grid) if args.max_configs is None else args.start_index + args.max_configs
    grid = grid[args.start_index : min(end_index, len(grid))]
    if not grid:
        raise ValueError("Empty parameter grid slice")

    rows = []
    num_batches = (len(grid) + args.batch_size - 1) // args.batch_size
    start_time = time.perf_counter()
    print(
        f"Fair gait audit: task={spec.task_id} vx={vx:.2f} configs={len(grid)} "
        f"batch_size={args.batch_size} repeats={args.repeats_per_config}"
    )
    for batch_idx in range(num_batches):
        batch_start_time = time.perf_counter()
        batch = grid[batch_idx * args.batch_size : (batch_idx + 1) * args.batch_size]
        real_batch_size = len(batch)
        padded_batch = pad_batch(batch, args.batch_size)
        env.reset()
        set_fixed_vx(env, vx)
        action = build_action(env, padded_batch, args.repeats_per_config)
        stats = make_stats(env.num_envs, env.device)
        prev_contacts = None
        # Do not use torch.inference_mode here. IsaacGym/WTW mutates persistent
        # env buffers during reset, and PyTorch 2.4 can reject later inplace
        # writes if those buffers were touched under inference_mode.
        with torch.no_grad():
            for step in range(args.steps + args.warmup_steps):
                set_fixed_vx(env, vx)
                _obs, reward, done, info = env.step(action)
                set_fixed_vx(env, vx)
                contacts = env.env.contact_forces[:, env.env.feet_indices, 2] > 1.0
                if step >= args.warmup_steps:
                    prev_contacts = add_step_stats(
                        stats,
                        env,
                        action,
                        reward,
                        done,
                        info,
                        prev_contacts,
                    )
                else:
                    prev_contacts = contacts.detach().clone()
        finalized = finalize_batch(
            stats,
            padded_batch,
            spec,
            vx,
            args.repeats_per_config,
            args.steps,
        )
        rows.extend(finalized[:real_batch_size])
        if args.save_each_batch:
            save_outputs(output_dir, rows, args)
        now = time.perf_counter()
        batch_seconds = now - batch_start_time
        elapsed_seconds = now - start_time
        batches_done = batch_idx + 1
        eta_seconds = (elapsed_seconds / batches_done) * max(num_batches - batches_done, 0)
        configs_per_second = real_batch_size / max(batch_seconds, 1e-6)
        print(
            f"Finished batch {batches_done}/{num_batches}: rows={len(rows)} "
            f"batch={batch_seconds:.1f}s elapsed={elapsed_seconds / 60.0:.1f}m "
            f"eta={eta_seconds / 60.0:.1f}m configs/s={configs_per_second:.2f}"
        )

    save_outputs(output_dir, rows, args)
    del env
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rows


def run_child_audits(args, eval_items, output_dir):
    all_rows = []
    for spec, vx in eval_items:
        child_dir = output_dir / f"{spec.task_id}_vx{vx:.2f}".replace(".", "p")
        csv_path = child_dir / "fair_gait_grid_results.csv"
        if args.skip_existing and csv_path.exists():
            print(f"Skipping existing child result: {csv_path}")
            with csv_path.open(newline="") as file:
                all_rows.extend(csv.DictReader(file))
            continue
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
            args.gaits,
            "--grid-mode",
            args.grid_mode,
            "--num-envs",
            str(args.num_envs),
            "--batch-size",
            str(args.batch_size),
            "--repeats-per-config",
            str(args.repeats_per_config),
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
            "--reward-profile",
            args.reward_profile,
            f"--freq-residuals={','.join(str(v) for v in args.freq_residuals)}",
            f"--duration-residuals={','.join(str(v) for v in args.duration_residuals)}",
            f"--footswing-residuals={','.join(str(v) for v in args.footswing_residuals)}",
            f"--stance-residuals={','.join(str(v) for v in args.stance_residuals)}",
            f"--body-pitch-residuals={','.join(str(v) for v in args.body_pitch_residuals)}",
            f"--frequencies={','.join(str(v) for v in args.frequencies)}",
            f"--durations={','.join(str(v) for v in args.durations)}",
            f"--footswing-heights={','.join(str(v) for v in args.footswing_heights)}",
            f"--stance-widths={','.join(str(v) for v in args.stance_widths)}",
            f"--body-pitches={','.join(str(v) for v in args.body_pitches)}",
            "--start-index",
            str(args.start_index),
            "--softmax-temperature",
            str(args.softmax_temperature),
            "--output-dir",
            str(child_dir),
            "--no-spawn",
        ]
        if args.max_configs is not None:
            cmd.extend(["--max-configs", str(args.max_configs)])
        if args.render:
            cmd.append("--render")
        if args.drop_unreachable:
            cmd.append("--drop-unreachable")
        if not args.save_each_batch:
            cmd.append("--no-save-each-batch")
        print(f"\nLaunching fair audit child: task={spec.task_id} vx={vx:.2f}")
        child_dir.mkdir(parents=True, exist_ok=True)
        child_log = child_dir / "child.log"
        with child_log.open("w") as log_file:
            try:
                subprocess.run(
                    cmd,
                    check=True,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                )
            except subprocess.CalledProcessError as exc:
                print(f"Child audit failed: task={spec.task_id} vx={vx:.2f}")
                print(f"See child log: {child_log}")
                raise exc
        with csv_path.open(newline="") as file:
            all_rows.extend(csv.DictReader(file))
    return all_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", default="gait-conditioned-agility/pretrain-go2/train")
    parser.add_argument("--run-index", type=int, default=0)
    parser.add_argument("--task-map", default=str(MAINLINE_TASK_MAP))
    parser.add_argument("--eval", default=QUICK_EVAL)
    parser.add_argument("--full", action="store_true")
    parser.add_argument(
        "--training-range",
        action="store_true",
        help="Audit representative speeds over each active task's actual sampled training range.",
    )
    parser.add_argument(
        "--extended",
        action="store_true",
        help="Audit extra push/stones speeds outside the current training range for diagnosis.",
    )
    parser.add_argument("--gaits", default="all")
    parser.add_argument("--grid-mode", choices=("action-space", "physical"), default="action-space")
    parser.add_argument("--num-envs", type=int, default=64, help="Kept for command metadata.")
    parser.add_argument("--batch-size", type=int, default=32, help="Parameter configs per sim batch.")
    parser.add_argument("--repeats-per-config", type=int, default=2)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--terrain-size", type=float, default=TRAIN_TERRAIN_SIZE)
    parser.add_argument("--edge-reset-margin", type=float, default=TRAIN_EDGE_RESET_MARGIN)
    parser.add_argument("--teleport-thresh", type=float, default=TRAIN_TELEPORT_THRESH)
    parser.add_argument("--mesh-type", default=TRAIN_MESH_TYPE, choices=["heightfield", "trimesh"])
    parser.add_argument(
        "--reward-profile",
        default="task_focus_v4",
        choices=REWARD_PROFILE_CHOICES,
        help=(
            "Reward profile used for live weighted_metric_reward logging. The fair "
            "grid's derived neutral scores are still computed from raw metrics."
        ),
    )
    parser.add_argument("--freq-residuals", type=parse_floats, default=parse_floats("-1,0,1"))
    parser.add_argument("--duration-residuals", type=parse_floats, default=parse_floats("0"))
    parser.add_argument("--footswing-residuals", type=parse_floats, default=parse_floats("-1,0,1"))
    parser.add_argument("--stance-residuals", type=parse_floats, default=parse_floats("-1,0,1"))
    parser.add_argument("--body-pitch-residuals", type=parse_floats, default=parse_floats("-1,0,1"))
    parser.add_argument("--frequencies", type=parse_floats, default=parse_floats("2.2,2.6,3.0,3.4"))
    parser.add_argument("--durations", type=parse_floats, default=parse_floats("0.5"))
    parser.add_argument("--footswing-heights", type=parse_floats, default=parse_floats("0.06,0.09,0.12"))
    parser.add_argument("--stance-widths", type=parse_floats, default=parse_floats("0.28,0.33,0.38"))
    parser.add_argument("--body-pitches", type=parse_floats, default=parse_floats("-0.04,0.0,0.04"))
    parser.add_argument("--drop-unreachable", action="store_true")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-configs", type=int, default=None)
    parser.add_argument("--softmax-temperature", type=float, default=0.03)
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Defaults to runs/high_level_oracle_gait/fair_target_gait_audit/<timestamp>.",
    )
    parser.add_argument("--render", action="store_true")
    parser.add_argument(
        "--no-save-each-batch",
        dest="save_each_batch",
        action="store_false",
        help="Disable incremental CSV/summary writes after each batch.",
    )
    parser.set_defaults(save_each_batch=True)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--no-spawn", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    specs = read_task_specs(
        args.task_map,
        style_reward_scale=0.0,
        reward_profile=args.reward_profile,
    )
    if args.extended:
        eval_text = EXTENDED_EVAL
    elif args.training_range:
        eval_text = TRAINING_RANGE_EVAL
    elif args.full:
        eval_text = FULL_EVAL
    else:
        eval_text = args.eval
    eval_items = parse_eval_items(eval_text, specs)
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else Path("runs/high_level_oracle_gait/fair_target_gait_audit") / time.strftime("%Y%m%d_%H%M%S")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    if len(eval_items) > 1 and not args.no_spawn:
        rows = run_child_audits(args, eval_items, output_dir)
        converted = []
        for row in rows:
            item = {}
            for key, value in row.items():
                try:
                    item[key] = float(value)
                except (TypeError, ValueError):
                    item[key] = value
            converted.append(item)
        save_outputs(output_dir, converted, args)
        print(f"\nWrote combined fair audit to: {output_dir}")
        return

    spec, vx = eval_items[0]
    run_eval_item(args, spec, vx, output_dir)
    print(f"\nWrote fair audit to: {output_dir}")


if __name__ == "__main__":
    main()
