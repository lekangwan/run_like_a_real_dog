import argparse
import csv
import gc
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
from scan_gait_params import GAIT_PARAMS, find_logdir, load_policy, parse_floats, parse_strings


CONDITIONS = {
    "flat",
    "low_friction",
    "very_low_friction",
    "rough",
    "rough_mid",
    "rough_hard",
    "rough_slope",
    "ramp_up",
    "slope",
    "stairs",
    "stairs_up_low",
    "stairs_down_low",
    "stairs_up",
    "stairs_down",
    "discrete_obstacles_low",
    "discrete_obstacles",
    "stepping_stones_easy",
    "stepping_stones",
    "push",
    "push_hard",
    "push_lateral",
    "push_longitudinal",
    "push_down",
    "push_up",
    "push_forward",
    "push_backward",
    "push_left",
    "push_right",
}


DIRECTIONAL_PUSH_VELOCITIES = {
    "push_forward": (1.5, 0.0, 0.0),
    "push_backward": (-1.5, 0.0, 0.0),
    "push_left": (0.0, 1.5, 0.0),
    "push_right": (0.0, -1.5, 0.0),
    "push_down": (0.0, 0.0, -1.5),
    "push_up": (0.0, 0.0, 1.5),
}


PUSH_AXIS_CONDITIONS = {
    "push_longitudinal": 0,
    "push_lateral": 1,
}


DIRECTED_PUSH_INTERVAL_S = 2.0


def build_grid(args):
    keys = [
        ("vx", args.vx),
        ("gait", args.gaits),
        ("frequency", args.frequencies),
        ("duration", args.durations),
        ("footswing_height", args.footswing_heights),
        ("body_pitch", args.body_pitches),
        ("stance_width", args.stance_widths),
    ]
    names, values = zip(*keys)
    return [dict(zip(names, combo)) for combo in itertools.product(*values)]


def apply_condition_cfg(condition):
    if condition not in CONDITIONS:
        raise ValueError(f"Unknown condition '{condition}'. Choices: {sorted(CONDITIONS)}")

    # Defaults: deterministic flat-ish evaluation without domain randomization.
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
    Cfg.domain_rand.randomize_rigids_after_start = False
    Cfg.domain_rand.push_vel_xy = None
    Cfg.domain_rand.push_vel_xyz = None
    Cfg.domain_rand.push_axis = None

    # Use a zero-height trimesh even for flat conditions. The plane branch in this
    # codebase builds CPU origin tensors and fails when assigning to CUDA env buffers.
    Cfg.terrain.mesh_type = "trimesh"
    Cfg.terrain.measure_heights = False
    Cfg.terrain.curriculum = False
    Cfg.terrain.selected = False
    Cfg.terrain.static_friction = 1.0
    Cfg.terrain.dynamic_friction = 1.0
    Cfg.terrain.restitution = 0.0
    Cfg.terrain.terrain_noise_magnitude = 0.0
    Cfg.terrain.terrain_proportions = [0, 0, 0, 0, 0, 0, 0, 0, 1.0]
    Cfg.terrain.terrain_length = 4.0
    Cfg.terrain.terrain_width = 4.0
    Cfg.terrain.x_init_range = 0.0
    Cfg.terrain.y_init_range = 0.0
    Cfg.terrain.teleport_robots = True
    Cfg.terrain.teleport_thresh = 1.0
    Cfg.terrain.center_robots = False
    Cfg.terrain.stair_step_height = None
    Cfg.terrain.discrete_obstacles_height = None
    Cfg.terrain.stepping_stones_size = None
    Cfg.terrain.stone_distance = None
    Cfg.terrain.stepping_stones_platform_size = 4.0
    Cfg.terrain.stepping_stones_max_height = 0.0
    Cfg.terrain.stepping_stones_depth = -10.0
    Cfg.terrain.ramp_slope = None

    if condition == "low_friction":
        Cfg.terrain.static_friction = 0.25
        Cfg.terrain.dynamic_friction = 0.25
        Cfg.domain_rand.randomize_friction = True
        Cfg.domain_rand.friction_range = [0.25, 0.26]
    elif condition == "very_low_friction":
        Cfg.terrain.static_friction = 0.12
        Cfg.terrain.dynamic_friction = 0.12
        Cfg.domain_rand.randomize_friction = True
        Cfg.domain_rand.friction_range = [0.12, 0.13]
    elif condition == "rough":
        Cfg.terrain.mesh_type = "trimesh"
        Cfg.terrain.measure_heights = True
        Cfg.terrain.terrain_noise_magnitude = 0.10
        Cfg.terrain.terrain_proportions = [0, 0, 0, 0, 0, 0, 0, 0, 1.0]
    elif condition == "rough_mid":
        Cfg.terrain.mesh_type = "trimesh"
        Cfg.terrain.measure_heights = True
        Cfg.terrain.terrain_noise_magnitude = 0.12
        Cfg.terrain.terrain_proportions = [0, 0, 0, 0, 0, 0, 0, 0, 1.0]
    elif condition == "rough_hard":
        Cfg.terrain.mesh_type = "trimesh"
        Cfg.terrain.measure_heights = True
        Cfg.terrain.terrain_noise_magnitude = 0.16
        Cfg.terrain.terrain_proportions = [0, 0, 0, 0, 0, 0, 0, 0, 1.0]
    elif condition == "rough_slope":
        Cfg.terrain.mesh_type = "trimesh"
        Cfg.terrain.measure_heights = True
        Cfg.terrain.terrain_proportions = [0, 1.0, 0, 0, 0, 0, 0, 0, 0]
    elif condition == "ramp_up":
        Cfg.terrain.mesh_type = "trimesh"
        Cfg.terrain.measure_heights = True
        Cfg.terrain.ramp_slope = 0.20
        Cfg.terrain.terrain_proportions = [1.0, 0, 0, 0, 0, 0, 0, 0, 0]
    elif condition == "slope":
        Cfg.terrain.mesh_type = "trimesh"
        Cfg.terrain.measure_heights = True
        Cfg.terrain.difficulty_scale = 0.6
        Cfg.terrain.terrain_proportions = [1.0, 0, 0, 0, 0, 0, 0, 0, 0]
    elif condition == "stairs":
        Cfg.terrain.mesh_type = "trimesh"
        Cfg.terrain.measure_heights = True
        Cfg.terrain.difficulty_scale = 0.35
        Cfg.terrain.terrain_proportions = [0, 0, 1.0, 0, 0, 0, 0, 0, 0]
    elif condition == "stairs_up_low":
        Cfg.terrain.mesh_type = "trimesh"
        Cfg.terrain.measure_heights = True
        Cfg.terrain.stair_step_height = 0.08
        Cfg.terrain.terrain_proportions = [0, 0, 0, 1.0, 0, 0, 0, 0, 0]
    elif condition == "stairs_down_low":
        Cfg.terrain.mesh_type = "trimesh"
        Cfg.terrain.measure_heights = True
        Cfg.terrain.stair_step_height = 0.08
        Cfg.terrain.terrain_proportions = [0, 0, 1.0, 0, 0, 0, 0, 0, 0]
    elif condition == "stairs_up":
        Cfg.terrain.mesh_type = "trimesh"
        Cfg.terrain.measure_heights = True
        Cfg.terrain.terrain_proportions = [0, 0, 0, 1.0, 0, 0, 0, 0, 0]
    elif condition == "stairs_down":
        Cfg.terrain.mesh_type = "trimesh"
        Cfg.terrain.measure_heights = True
        Cfg.terrain.terrain_proportions = [0, 0, 1.0, 0, 0, 0, 0, 0, 0]
    elif condition == "discrete_obstacles":
        Cfg.terrain.mesh_type = "trimesh"
        Cfg.terrain.measure_heights = True
        Cfg.terrain.terrain_proportions = [0, 0, 0, 0, 1.0, 0, 0, 0, 0]
    elif condition == "discrete_obstacles_low":
        Cfg.terrain.mesh_type = "trimesh"
        Cfg.terrain.measure_heights = True
        Cfg.terrain.discrete_obstacles_height = 0.08
        Cfg.terrain.terrain_proportions = [0, 0, 0, 0, 1.0, 0, 0, 0, 0]
    elif condition == "stepping_stones":
        Cfg.terrain.mesh_type = "trimesh"
        Cfg.terrain.measure_heights = True
        Cfg.terrain.stepping_stones_size = 0.55
        Cfg.terrain.stone_distance = 0.16
        Cfg.terrain.stepping_stones_platform_size = 1.0
        Cfg.terrain.stepping_stones_depth = -0.12
        Cfg.terrain.terrain_proportions = [0, 0, 0, 0, 0, 1.0, 0, 0, 0]
    elif condition == "stepping_stones_easy":
        Cfg.terrain.mesh_type = "trimesh"
        Cfg.terrain.measure_heights = True
        Cfg.terrain.stepping_stones_size = 0.80
        Cfg.terrain.stone_distance = 0.10
        Cfg.terrain.stepping_stones_platform_size = 1.2
        Cfg.terrain.stepping_stones_depth = -0.06
        Cfg.terrain.terrain_proportions = [0, 0, 0, 0, 0, 1.0, 0, 0, 0]
    elif condition == "push":
        Cfg.domain_rand.push_robots = True
        Cfg.domain_rand.push_interval_s = 0.5
        Cfg.domain_rand.max_push_vel_xy = 1.0
    elif condition == "push_hard":
        Cfg.domain_rand.push_robots = True
        Cfg.domain_rand.push_interval_s = 0.4
        Cfg.domain_rand.max_push_vel_xy = 1.5
    elif condition in DIRECTIONAL_PUSH_VELOCITIES:
        Cfg.domain_rand.push_robots = True
        Cfg.domain_rand.push_interval_s = DIRECTED_PUSH_INTERVAL_S
        Cfg.domain_rand.max_push_vel_xy = 1.5
        Cfg.domain_rand.push_vel_xyz = DIRECTIONAL_PUSH_VELOCITIES[condition]
    elif condition in PUSH_AXIS_CONDITIONS:
        Cfg.domain_rand.push_robots = True
        Cfg.domain_rand.push_interval_s = DIRECTED_PUSH_INTERVAL_S
        Cfg.domain_rand.max_push_vel_xy = 1.5
        Cfg.domain_rand.push_axis = PUSH_AXIS_CONDITIONS[condition]


def load_env(
    logdir,
    num_envs,
    headless,
    condition,
    terrain_length=None,
    terrain_width=None,
    teleport_thresh=None,
    edge_reset_margin=None,
):
    config_go2(Cfg)
    with open(Path(logdir) / "parameters.pkl", "rb") as file:
        pkl_cfg = pkl.load(file)
        cfg = pkl_cfg["Cfg"]
        for key, value in cfg.items():
            if hasattr(Cfg, key):
                for key2, value2 in value.items():
                    setattr(getattr(Cfg, key), key2, value2)

    apply_condition_cfg(condition)

    Cfg.env.num_envs = num_envs
    Cfg.env.num_recording_envs = 0
    Cfg.terrain.min_init_terrain_level = 0
    Cfg.terrain.max_init_terrain_level = 0
    Cfg.terrain.num_rows = 1
    Cfg.terrain.num_cols = max(1, num_envs)
    Cfg.terrain.border_size = 0
    Cfg.asset.flip_visual_attachments = True
    if terrain_length is not None:
        Cfg.terrain.terrain_length = terrain_length
    if terrain_width is not None:
        Cfg.terrain.terrain_width = terrain_width
    if teleport_thresh is not None:
        Cfg.terrain.teleport_thresh = teleport_thresh
    Cfg.terrain.edge_reset_robots = edge_reset_margin is not None
    if edge_reset_margin is not None:
        Cfg.terrain.edge_reset_margin = edge_reset_margin
        Cfg.terrain.teleport_robots = False

    env = VelocityTrackingEasyEnv(sim_device="cuda:0", headless=headless, cfg=Cfg)
    return HistoryWrapper(env)


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
        commands[i, 8] = params.get("duration", 0.5)
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
        "mechanical_power_abs": torch.zeros(n),
        "positive_mechanical_power": torch.zeros(n),
        "contact_force_mean": torch.zeros(n),
        "stance_contact_force_mean": torch.zeros(n),
        "peak_contact_force": torch.zeros(n),
        "swing_foot_clearance_mean": torch.zeros(n),
        "scuffing_ratio": torch.zeros(n),
        "phase_match_error": torch.zeros(n),
        "forward_distance": torch.zeros(n),
        "target_forward_distance": torch.zeros(n),
        "foot_impact_vel_sum": torch.zeros(n),
        "foot_impact_vel_sq_sum": torch.zeros(n),
        "foot_impact_count": torch.zeros(n),
        "fall_count": torch.zeros(n),
        "edge_reset_count": torch.zeros(n),
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


def get_policy_dt(env):
    return getattr(env, "dt", getattr(env.env, "dt", 0.02))


def sample_ground_heights(env, foot_positions):
    if not hasattr(env, "height_samples") or not hasattr(env, "terrain"):
        return torch.zeros_like(foot_positions[:, :, 2])

    height_samples = env.height_samples
    terrain_cfg = env.terrain.cfg
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


def update_metrics(metrics, env, actions, prev_actions, command_vx, active_n, dones, contacts, prev_contacts):
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

    contacts_f = contacts.float()
    contact_count = torch.sum(contacts_f, dim=1)
    foot_xy_vel = torch.sum(env.foot_velocities[:active_n, :, :2] ** 2, dim=2)
    slip_penalty = torch.mean(contacts * foot_xy_vel, dim=1)
    vertical_velocity_penalty = base_z_vel**2
    vx_abs_error_penalty = torch.abs(vx_error)
    edge_reset = getattr(env, "edge_reset_buf", None)
    if edge_reset is None:
        edge_reset = torch.zeros_like(dones, dtype=torch.bool)
    edge_reset = edge_reset[:active_n].bool()
    fall_penalty = (dones[:active_n].bool() & ~edge_reset).float()

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
    joint_power = env.torques[:active_n] * env.dof_vel[:active_n, : env.torques.shape[1]]
    mechanical_power_abs = torch.sum(torch.abs(joint_power), dim=1)
    positive_mechanical_power = torch.sum(torch.clamp(joint_power, min=0.0), dim=1)
    metrics["mechanical_power_abs"] += mechanical_power_abs.detach().cpu()
    metrics["positive_mechanical_power"] += positive_mechanical_power.detach().cpu()

    contact_force_norm = torch.norm(env.contact_forces[:active_n, env.feet_indices, :], dim=2)
    metrics["contact_force_mean"] += torch.mean(contact_force_norm, dim=1).detach().cpu()
    stance_force = torch.sum(contact_force_norm * contacts_f, dim=1) / torch.clamp(contact_count, min=1.0)
    metrics["stance_contact_force_mean"] += stance_force.detach().cpu()
    metrics["peak_contact_force"] = torch.maximum(
        metrics["peak_contact_force"],
        torch.max(contact_force_norm, dim=1).values.detach().cpu(),
    )

    foot_positions = env.foot_positions[:active_n]
    ground_heights = sample_ground_heights(env, foot_positions)
    foot_clearance = foot_positions[:, :, 2] - ground_heights
    swing_mask = (~contacts).float()
    swing_count = torch.clamp(torch.sum(swing_mask, dim=1), min=1.0)
    swing_clearance = torch.sum(swing_mask * foot_clearance, dim=1) / swing_count
    scuffing_ratio = torch.sum(swing_mask * (foot_clearance < 0.035).float(), dim=1) / swing_count
    metrics["swing_foot_clearance_mean"] += swing_clearance.detach().cpu()
    metrics["scuffing_ratio"] += scuffing_ratio.detach().cpu()

    desired_contacts = env.desired_contact_states[:active_n]
    metrics["phase_match_error"] += torch.mean(torch.abs(contacts_f - desired_contacts), dim=1).detach().cpu()

    dt = get_policy_dt(env)
    metrics["forward_distance"] += (env.base_lin_vel[:active_n, 0] * dt).detach().cpu()
    metrics["target_forward_distance"] += (vx_cmd * dt).detach().cpu()

    if prev_contacts is not None:
        new_contacts = contacts & (~prev_contacts[:active_n])
        impact_vel = torch.clamp(-env.prev_foot_velocities[:active_n, :, 2], min=0.0)
        metrics["foot_impact_vel_sum"] += torch.sum(new_contacts * impact_vel, dim=1).detach().cpu()
        metrics["foot_impact_vel_sq_sum"] += torch.sum(new_contacts * impact_vel**2, dim=1).detach().cpu()
        metrics["foot_impact_count"] += torch.sum(new_contacts.float(), dim=1).detach().cpu()

    metrics["fall_count"] += fall_penalty.detach().cpu()
    metrics["edge_reset_count"] += edge_reset.float().detach().cpu()
    metrics["template_score"] += template_score.detach().cpu()
    for foot_id in range(4):
        metrics[f"foot{foot_id}_duty"] += contacts_f[:, foot_id].detach().cpu()
    for i, j in ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)):
        pair = f"{i}{j}"
        metrics[f"contact_pair{pair}_sync"] += (
            1.0 - torch.abs(contacts_f[:, i] - contacts_f[:, j])
        ).detach().cpu()
        metrics[f"contact_pair{pair}_co"] += (contacts_f[:, i] * contacts_f[:, j]).detach().cpu()
    metrics["contact_count"] += contact_count.detach().cpu()
    metrics["flight_ratio"] += (contact_count == 0).float().detach().cpu()
    metrics["all_contact_ratio"] += (contact_count == 4).float().detach().cpu()


def finalize_metrics(values, eval_steps):
    row = {}
    steps = max(eval_steps, 1)
    for key, value in values.items():
        if key in (
            "base_z_sum",
            "base_z_sq_sum",
            "foot_impact_vel_sum",
            "foot_impact_vel_sq_sum",
        ):
            continue
        row[key] = value if key in (
            "fall_count",
            "edge_reset_count",
            "foot_impact_count",
            "forward_distance",
            "target_forward_distance",
            "peak_contact_force",
        ) else value / steps

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
    row["edge_reset_rate"] = row["edge_reset_count"] / steps
    impact_count = max(values["foot_impact_count"], 1.0)
    row["foot_impact_rate"] = values["foot_impact_count"] / steps
    row["foot_impact_vel_mean"] = values["foot_impact_vel_sum"] / impact_count
    row["foot_impact_vel_rms"] = max(values["foot_impact_vel_sq_sum"] / impact_count, 0.0) ** 0.5
    measured_speed = max(abs(row["measured_vx"]), 0.2)
    row["transport_cost_proxy"] = row["mechanical_power_abs"] / measured_speed
    target_distance = max(abs(row["target_forward_distance"]), 0.2)
    row["forward_distance_ratio"] = row["forward_distance"] / target_distance
    row["progress_deficit"] = max(row["target_forward_distance"] - row["forward_distance"], 0.0) / target_distance
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
        prev_contacts = None
        eval_steps = 0

        for step in range(args.warmup_steps + args.eval_steps):
            with torch.inference_mode():
                actions = policy(obs)
            if prev_actions is None:
                prev_actions = torch.zeros_like(actions)
            set_commands(env, commands)
            obs, _, dones, _ = env.step(actions)
            contacts = env.contact_forces[:active_n, env.feet_indices, 2] > 1.0

            if step >= args.warmup_steps:
                update_metrics(
                    metrics,
                    env,
                    actions,
                    prev_actions,
                    command_vx,
                    active_n,
                    dones,
                    contacts,
                    prev_contacts,
                )
                eval_steps += 1
            prev_actions.copy_(actions.detach())
            prev_contacts = contacts.detach().clone()

        for i, params in enumerate(batch):
            row = dict(params)
            row["condition"] = args.condition
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
    save_csv(output_dir / "best_by_condition_speed.csv", best_rows(rows, ["condition", "vx"]))
    save_csv(
        output_dir / "best_by_condition_speed_gait.csv",
        best_rows(rows, ["condition", "vx", "gait"]),
    )
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
    parser.add_argument("--condition", choices=sorted(CONDITIONS), default="flat")
    parser.add_argument("--vx", type=parse_floats, default=parse_floats("0.2,0.5,0.8,1.2,1.6,2.0"))
    parser.add_argument("--gaits", type=parse_strings, default=parse_strings("pronking,trotting,bounding,pacing"))
    parser.add_argument("--frequencies", type=parse_floats, default=parse_floats("3.0"))
    parser.add_argument("--durations", type=parse_floats, default=parse_floats("0.5"))
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
        f"Evaluating {len(grid)} fixed template configs under condition={args.condition} "
        f"({args.start_index}:{args.start_index + len(grid)} of {len(full_grid)}) "
        f"with batch size {args.batch_size}"
    )

    env = load_env(logdir, args.batch_size, headless=not args.render, condition=args.condition)
    policy = load_policy(logdir)
    rows = run_eval(env, policy, grid, args)
    save_results(rows, Path(args.output_dir))
    print(f"Saved results to: {args.output_dir}")


if __name__ == "__main__":
    main()
