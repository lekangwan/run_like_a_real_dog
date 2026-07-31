from pathlib import Path
import glob
import pickle

import torch

from go2_gym import MINI_GYM_ROOT_DIR
from go2_gym.envs.base.legged_robot_config import Cfg
from go2_gym.envs.go2.go2_config import config_go2
from go2_gym.envs.go2.velocity_tracking import VelocityTrackingEasyEnv
from go2_gym.envs.wrappers.history_wrapper import HistoryWrapper


def find_run(label, run_index=0):
    paths = sorted(glob.glob(str(Path(MINI_GYM_ROOT_DIR) / "runs" / label / "*")))
    if not paths:
        raise FileNotFoundError(f"No low-level run found for {label}")
    return Path(paths[run_index])


def load_policy(run_dir):
    body = torch.jit.load(str(run_dir / "checkpoints/body_latest.jit"))
    adaptation = torch.jit.load(str(run_dir / "checkpoints/adaptation_module_latest.jit"))
    body.eval()
    adaptation.eval()

    def policy(observation, info=None):
        del info
        with torch.inference_mode():
            history = observation["obs_history"].detach().cpu()
            latent = adaptation(history)
            return body(torch.cat((history, latent), dim=-1))

    return policy


def _restore_training_config(run_dir):
    config_go2(Cfg)
    with open(run_dir / "parameters.pkl", "rb") as file:
        saved = pickle.load(file)["Cfg"]
    for section_name, values in saved.items():
        if not hasattr(Cfg, section_name):
            continue
        section = getattr(Cfg, section_name)
        for name, value in values.items():
            if isinstance(section, dict):
                section[name] = value
            else:
                setattr(section, name, value)


def _disable_randomization():
    fields = (
        "push_robots",
        "randomize_friction",
        "randomize_gravity",
        "randomize_restitution",
        "randomize_motor_offset",
        "randomize_motor_strength",
        "randomize_friction_indep",
        "randomize_ground_friction",
        "randomize_base_mass",
        "randomize_Kd_factor",
        "randomize_Kp_factor",
        "randomize_joint_friction",
        "randomize_com_displacement",
        "randomize_rigids_after_start",
    )
    for name in fields:
        if hasattr(Cfg.domain_rand, name):
            setattr(Cfg.domain_rand, name, False)
    for name in ("push_vel_xy", "push_vel_xyz", "push_axis"):
        if hasattr(Cfg.domain_rand, name):
            setattr(Cfg.domain_rand, name, None)


def _reset_mixed_terrain_baseline():
    """Clear condition-specific values left in the saved low-level config."""
    Cfg.terrain.static_friction = 1.0
    Cfg.terrain.dynamic_friction = 1.0
    Cfg.terrain.restitution = 0.0
    Cfg.terrain.terrain_noise_magnitude = 0.0
    Cfg.terrain.terrain_proportions = [0, 0, 0, 0, 0, 0, 0, 0, 1.0]
    Cfg.terrain.x_init_range = 0.0
    Cfg.terrain.y_init_range = 0.0
    Cfg.terrain.stair_step_height = None
    Cfg.terrain.discrete_obstacles_height = None
    Cfg.terrain.stepping_stones_size = None
    Cfg.terrain.stone_distance = None
    Cfg.terrain.stepping_stones_platform_size = 4.0
    Cfg.terrain.stepping_stones_max_height = 0.0
    Cfg.terrain.stepping_stones_depth = -10.0
    Cfg.terrain.ramp_slope = None


def create_environment(
    run_dir,
    num_envs,
    conditions,
    push_axes,
    render,
    terrain_length,
    terrain_width,
    edge_reset_margin,
    teleport_threshold,
    recording_width=None,
    recording_height=None,
):
    _restore_training_config(run_dir)
    _disable_randomization()
    _reset_mixed_terrain_baseline()

    Cfg.env.num_envs = num_envs
    Cfg.env.num_recording_envs = 0
    if recording_width is not None:
        Cfg.env.recording_width_px = int(recording_width)
    if recording_height is not None:
        Cfg.env.recording_height_px = int(recording_height)
    Cfg.asset.flip_visual_attachments = True

    Cfg.terrain.curriculum = False
    Cfg.terrain.selected = False
    Cfg.terrain.min_init_terrain_level = 0
    Cfg.terrain.max_init_terrain_level = 0
    Cfg.terrain.num_rows = 1
    Cfg.terrain.num_cols = max(1, num_envs)
    Cfg.terrain.border_size = 0
    Cfg.terrain.center_robots = False
    Cfg.terrain.measure_heights = True
    Cfg.terrain.mesh_type = "trimesh"
    Cfg.terrain.terrain_length = terrain_length
    Cfg.terrain.terrain_width = terrain_width
    Cfg.terrain.teleport_robots = False
    Cfg.terrain.teleport_thresh = teleport_threshold
    Cfg.terrain.edge_reset_robots = True
    Cfg.terrain.edge_reset_margin = edge_reset_margin
    Cfg.terrain.env_conditions = list(conditions)

    Cfg.domain_rand.push_robots = any(axis >= 0 for axis in push_axes)
    Cfg.domain_rand.push_interval_s = 2.0
    Cfg.domain_rand.max_push_vel_xy = 1.5
    Cfg.domain_rand.push_axis_by_env = list(push_axes)

    print(
        f"Creating {num_envs} environments, "
        f"terrain={terrain_length:.1f}m x {terrain_width:.1f}m"
    )
    env = VelocityTrackingEasyEnv(sim_device="cuda:0", headless=not render, cfg=Cfg)
    return HistoryWrapper(env)
