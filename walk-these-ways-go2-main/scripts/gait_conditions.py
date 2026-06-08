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


def apply_condition_cfg(cfg, condition):
    if condition not in CONDITIONS:
        raise ValueError(f"Unknown condition '{condition}'. Choices: {sorted(CONDITIONS)}")

    # Defaults: deterministic flat-ish evaluation without domain randomization.
    cfg.domain_rand.push_robots = False
    cfg.domain_rand.randomize_friction = False
    cfg.domain_rand.randomize_gravity = False
    cfg.domain_rand.randomize_restitution = False
    cfg.domain_rand.randomize_motor_offset = False
    cfg.domain_rand.randomize_motor_strength = False
    cfg.domain_rand.randomize_friction_indep = False
    cfg.domain_rand.randomize_ground_friction = False
    cfg.domain_rand.randomize_base_mass = False
    cfg.domain_rand.randomize_Kd_factor = False
    cfg.domain_rand.randomize_Kp_factor = False
    cfg.domain_rand.randomize_joint_friction = False
    cfg.domain_rand.randomize_com_displacement = False
    cfg.domain_rand.randomize_rigids_after_start = False
    cfg.domain_rand.push_vel_xy = None
    cfg.domain_rand.push_vel_xyz = None
    cfg.domain_rand.push_axis = None

    # Use a zero-height trimesh even for flat conditions. The plane branch in this
    # codebase builds CPU origin tensors and fails when assigning to CUDA env buffers.
    cfg.terrain.mesh_type = "trimesh"
    cfg.terrain.measure_heights = False
    cfg.terrain.curriculum = False
    cfg.terrain.selected = False
    cfg.terrain.static_friction = 1.0
    cfg.terrain.dynamic_friction = 1.0
    cfg.terrain.restitution = 0.0
    cfg.terrain.terrain_noise_magnitude = 0.0
    cfg.terrain.terrain_proportions = [0, 0, 0, 0, 0, 0, 0, 0, 1.0]
    cfg.terrain.terrain_length = 4.0
    cfg.terrain.terrain_width = 4.0
    cfg.terrain.x_init_range = 0.0
    cfg.terrain.y_init_range = 0.0
    cfg.terrain.teleport_robots = True
    cfg.terrain.teleport_thresh = 1.0
    cfg.terrain.center_robots = False
    cfg.terrain.stair_step_height = None
    cfg.terrain.discrete_obstacles_height = None
    cfg.terrain.stepping_stones_size = None
    cfg.terrain.stone_distance = None
    cfg.terrain.stepping_stones_platform_size = 4.0
    cfg.terrain.stepping_stones_max_height = 0.0
    cfg.terrain.stepping_stones_depth = -10.0
    cfg.terrain.ramp_slope = None

    if condition == "low_friction":
        cfg.terrain.static_friction = 0.25
        cfg.terrain.dynamic_friction = 0.25
        cfg.domain_rand.randomize_friction = True
        cfg.domain_rand.friction_range = [0.25, 0.26]
    elif condition == "very_low_friction":
        cfg.terrain.static_friction = 0.12
        cfg.terrain.dynamic_friction = 0.12
        cfg.domain_rand.randomize_friction = True
        cfg.domain_rand.friction_range = [0.12, 0.13]
    elif condition == "rough":
        cfg.terrain.mesh_type = "trimesh"
        cfg.terrain.measure_heights = True
        cfg.terrain.terrain_noise_magnitude = 0.10
        cfg.terrain.terrain_proportions = [0, 0, 0, 0, 0, 0, 0, 0, 1.0]
    elif condition == "rough_mid":
        cfg.terrain.mesh_type = "trimesh"
        cfg.terrain.measure_heights = True
        cfg.terrain.terrain_noise_magnitude = 0.12
        cfg.terrain.terrain_proportions = [0, 0, 0, 0, 0, 0, 0, 0, 1.0]
    elif condition == "rough_hard":
        cfg.terrain.mesh_type = "trimesh"
        cfg.terrain.measure_heights = True
        cfg.terrain.terrain_noise_magnitude = 0.16
        cfg.terrain.terrain_proportions = [0, 0, 0, 0, 0, 0, 0, 0, 1.0]
    elif condition == "rough_slope":
        cfg.terrain.mesh_type = "trimesh"
        cfg.terrain.measure_heights = True
        cfg.terrain.terrain_proportions = [0, 1.0, 0, 0, 0, 0, 0, 0, 0]
    elif condition == "ramp_up":
        cfg.terrain.mesh_type = "trimesh"
        cfg.terrain.measure_heights = True
        cfg.terrain.ramp_slope = 0.20
        cfg.terrain.terrain_proportions = [1.0, 0, 0, 0, 0, 0, 0, 0, 0]
    elif condition == "slope":
        cfg.terrain.mesh_type = "trimesh"
        cfg.terrain.measure_heights = True
        cfg.terrain.difficulty_scale = 0.6
        cfg.terrain.terrain_proportions = [1.0, 0, 0, 0, 0, 0, 0, 0, 0]
    elif condition == "stairs":
        cfg.terrain.mesh_type = "trimesh"
        cfg.terrain.measure_heights = True
        cfg.terrain.difficulty_scale = 0.35
        cfg.terrain.terrain_proportions = [0, 0, 1.0, 0, 0, 0, 0, 0, 0]
    elif condition == "stairs_up_low":
        cfg.terrain.mesh_type = "trimesh"
        cfg.terrain.measure_heights = True
        cfg.terrain.stair_step_height = 0.08
        cfg.terrain.terrain_proportions = [0, 0, 0, 1.0, 0, 0, 0, 0, 0]
    elif condition == "stairs_down_low":
        cfg.terrain.mesh_type = "trimesh"
        cfg.terrain.measure_heights = True
        cfg.terrain.stair_step_height = 0.08
        cfg.terrain.terrain_proportions = [0, 0, 1.0, 0, 0, 0, 0, 0, 0]
    elif condition == "stairs_up":
        cfg.terrain.mesh_type = "trimesh"
        cfg.terrain.measure_heights = True
        cfg.terrain.terrain_proportions = [0, 0, 0, 1.0, 0, 0, 0, 0, 0]
    elif condition == "stairs_down":
        cfg.terrain.mesh_type = "trimesh"
        cfg.terrain.measure_heights = True
        cfg.terrain.terrain_proportions = [0, 0, 1.0, 0, 0, 0, 0, 0, 0]
    elif condition == "discrete_obstacles":
        cfg.terrain.mesh_type = "trimesh"
        cfg.terrain.measure_heights = True
        cfg.terrain.terrain_proportions = [0, 0, 0, 0, 1.0, 0, 0, 0, 0]
    elif condition == "discrete_obstacles_low":
        cfg.terrain.mesh_type = "trimesh"
        cfg.terrain.measure_heights = True
        cfg.terrain.discrete_obstacles_height = 0.08
        cfg.terrain.terrain_proportions = [0, 0, 0, 0, 1.0, 0, 0, 0, 0]
    elif condition == "stepping_stones":
        cfg.terrain.mesh_type = "trimesh"
        cfg.terrain.measure_heights = True
        cfg.terrain.stepping_stones_size = 0.55
        cfg.terrain.stone_distance = 0.16
        cfg.terrain.stepping_stones_platform_size = 1.0
        cfg.terrain.stepping_stones_depth = -0.12
        cfg.terrain.terrain_proportions = [0, 0, 0, 0, 0, 1.0, 0, 0, 0]
    elif condition == "stepping_stones_easy":
        cfg.terrain.mesh_type = "trimesh"
        cfg.terrain.measure_heights = True
        cfg.terrain.stepping_stones_size = 0.80
        cfg.terrain.stone_distance = 0.10
        cfg.terrain.stepping_stones_platform_size = 1.2
        cfg.terrain.stepping_stones_depth = -0.06
        cfg.terrain.terrain_proportions = [0, 0, 0, 0, 0, 1.0, 0, 0, 0]
    elif condition == "push":
        cfg.domain_rand.push_robots = True
        cfg.domain_rand.push_interval_s = 0.5
        cfg.domain_rand.max_push_vel_xy = 1.0
    elif condition == "push_hard":
        cfg.domain_rand.push_robots = True
        cfg.domain_rand.push_interval_s = 0.4
        cfg.domain_rand.max_push_vel_xy = 1.5
    elif condition in DIRECTIONAL_PUSH_VELOCITIES:
        cfg.domain_rand.push_robots = True
        cfg.domain_rand.push_interval_s = DIRECTED_PUSH_INTERVAL_S
        cfg.domain_rand.max_push_vel_xy = 1.5
        cfg.domain_rand.push_vel_xyz = DIRECTIONAL_PUSH_VELOCITIES[condition]
    elif condition in PUSH_AXIS_CONDITIONS:
        cfg.domain_rand.push_robots = True
        cfg.domain_rand.push_interval_s = DIRECTED_PUSH_INTERVAL_S
        cfg.domain_rand.max_push_vel_xy = 1.5
        cfg.domain_rand.push_axis = PUSH_AXIS_CONDITIONS[condition]
