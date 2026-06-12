import torch


class HighLevelGaitWrapper:
    """Wrap a WTW env + frozen low-level policy as a high-level gait-parameter env.

    High-level action shape [num_envs, 9]:
      0-3: one-hot gait selector for pronking/trotting/bounding/pacing
      4: residual gait frequency
      5: residual gait duration
      6: residual foot swing height
      7: residual stance width
      8: residual body pitch
    """

    TASK_REWARD_NAMES = (
        "progress",
        "yaw_tracking",
        "orientation",
        "pitch_rate",
        "roll_rate",
        "yaw_rate",
        "lateral_drift",
        "vertical_bounce",
        "slip",
        "energy",
        "clearance",
        "gait_stability",
        "action_smoothness",
        "action_magnitude",
        "action_boundary_margin",
        "survival",
    )

    def __init__(
        self,
        env,
        low_level_policy,
        high_level_dt=0.10,
        history_length=10,
        action_smoothing=0.6,
        record_reward_terms=False,
        freq_range=(2.0, 3.5),
        duration_range=(0.42, 0.58),
        phase_range=(0.0, 0.5),
        offset_range=(0.0, 0.5),
        bound_range=(0.0, 0.5),
        footswing_range=(0.04, 0.12),
        stance_width_range=(0.25, 0.38),
        body_pitch_range=(-0.10, 0.05),
        freq_delta_range=(-0.4, 0.4),
        duration_delta_range=(-0.08, 0.08),
        footswing_delta_range=(-0.03, 0.03),
        stance_width_delta_range=(-0.04, 0.04),
        body_pitch_delta_range=(-0.04, 0.04),
        selector_temperature=0.25,
        selector_hold_steps=3,
        velocity_tracking_sigma=0.25,
        selector_reference_coef=0.0,
    ):
        self.env = env
        self.low_level_policy = low_level_policy
        self.high_level_dt = high_level_dt
        self.low_level_steps = max(1, int(round(high_level_dt / self.env.dt)))
        self.history_length = history_length
        self.action_smoothing = action_smoothing
        self.record_reward_terms = record_reward_terms

        self.phase_range = phase_range
        self.offset_range = offset_range
        self.bound_range = bound_range
        self.freq_range = freq_range
        self.duration_range = duration_range
        self.footswing_range = footswing_range
        self.stance_width_range = stance_width_range
        self.body_pitch_range = body_pitch_range
        self.freq_delta_range = freq_delta_range
        self.duration_delta_range = duration_delta_range
        self.footswing_delta_range = footswing_delta_range
        self.stance_width_delta_range = stance_width_delta_range
        self.body_pitch_delta_range = body_pitch_delta_range
        self.selector_temperature = selector_temperature
        self.selector_hold_steps = max(0, int(selector_hold_steps))
        self.velocity_tracking_sigma = velocity_tracking_sigma
        self.selector_reference_coef = selector_reference_coef

        self.num_envs = self.env.num_envs
        self.device = self.env.device
        self.gait_names = ("pronking", "trotting", "bounding", "pacing")
        self.gait_templates = torch.tensor(
            (
                (0.0, 0.0, 0.0),
                (0.5, 0.0, 0.0),
                (0.0, 0.5, 0.0),
                (0.0, 0.0, 0.5),
            ),
            device=self.device,
            dtype=torch.float,
        )
        self.gait_behavior_templates = torch.tensor(
            (
                # frequency, duration, footswing_height, stance_width, body_pitch
                (3.0, 0.5, 0.08, 0.33, 0.0),
                (3.0, 0.5, 0.08, 0.33, 0.0),
                (3.0, 0.5, 0.12, 0.38, 0.0),
                (2.5, 0.5, 0.12, 0.38, 0.0),
            ),
            device=self.device,
            dtype=torch.float,
        )
        self.residual_delta_ranges = torch.tensor(
            (
                self.freq_delta_range,
                self.duration_delta_range,
                self.footswing_delta_range,
                self.stance_width_delta_range,
                self.body_pitch_delta_range,
            ),
            device=self.device,
            dtype=torch.float,
        )
        self.behavior_lows = torch.tensor(
            (
                self.freq_range[0],
                self.duration_range[0],
                self.footswing_range[0],
                self.stance_width_range[0],
                self.body_pitch_range[0],
            ),
            device=self.device,
            dtype=torch.float,
        )
        self.behavior_highs = torch.tensor(
            (
                self.freq_range[1],
                self.duration_range[1],
                self.footswing_range[1],
                self.stance_width_range[1],
                self.body_pitch_range[1],
            ),
            device=self.device,
            dtype=torch.float,
        )
        self.num_gaits = len(self.gait_names)
        self.num_behavior_actions = 5
        self.num_high_level_actions = self.num_gaits + self.num_behavior_actions
        self.num_high_level_obs = 42 + self.num_high_level_actions
        self.num_high_level_obs_history = self.num_high_level_obs * self.history_length

        self.high_level_action = torch.zeros(
            self.num_envs, self.num_high_level_actions, device=self.device, dtype=torch.float
        )
        self.default_high_level_action = torch.zeros_like(self.high_level_action)
        self.default_high_level_action[:, 1] = 1.0
        self.prev_high_level_action = torch.zeros_like(self.high_level_action)
        self.selector_hold_counter = torch.full(
            (self.num_envs,),
            self.selector_hold_steps,
            device=self.device,
            dtype=torch.long,
        )
        self.velocity_command = torch.zeros(self.num_envs, 3, device=self.device, dtype=torch.float)
        self.obs_history = torch.zeros(
            self.num_envs,
            self.num_high_level_obs_history,
            device=self.device,
            dtype=torch.float,
        )
        self.obs_shift_buffer = torch.zeros(
            self.num_envs,
            self.num_high_level_obs_history - self.num_high_level_obs,
            device=self.device,
            dtype=torch.float,
        )
        self.low_level_obs = None
        self.last_reward_terms = {}
        self.target_gait_ids = None
        self.selector_reference_coef_tensor = None
        self.task_reward_weights = None

    def reset(self):
        self.low_level_obs = self.env.reset()
        self.high_level_action.copy_(self.default_high_level_action)
        self.prev_high_level_action.copy_(self.default_high_level_action)
        self.selector_hold_counter.fill_(self.selector_hold_steps)
        self.obs_history.zero_()
        return self.get_observations()

    def set_velocity_command(self, vx, vy=0.0, yaw=0.0):
        self.velocity_command[:, 0] = torch.as_tensor(vx, device=self.device, dtype=torch.float)
        self.velocity_command[:, 1] = torch.as_tensor(vy, device=self.device, dtype=torch.float)
        self.velocity_command[:, 2] = torch.as_tensor(yaw, device=self.device, dtype=torch.float)
        self.env.commands[:, 0:3] = self.velocity_command

    def set_target_gait(self, gait_ids, selector_reference_coef=None):
        self.target_gait_ids = torch.as_tensor(gait_ids, device=self.device, dtype=torch.long)
        if selector_reference_coef is not None:
            self.selector_reference_coef_tensor = torch.as_tensor(
                selector_reference_coef,
                device=self.device,
                dtype=torch.float,
            )

    def set_task_reward_weights(self, weights):
        weights = torch.as_tensor(weights, device=self.device, dtype=torch.float)
        expected_shape = (self.num_envs, len(self.TASK_REWARD_NAMES))
        if tuple(weights.shape) != expected_shape:
            raise ValueError(f"Expected task reward weights shape {expected_shape}, got {tuple(weights.shape)}")
        self.task_reward_weights = weights

    def step(self, high_level_action):
        high_level_action = torch.clip(high_level_action.to(self.device), -1.0, 1.0).detach()
        self.prev_high_level_action[:] = self.high_level_action
        self.high_level_action[:, : self.num_gaits] = self._held_selector_weights(high_level_action)
        self.high_level_action[:, self.num_gaits :] = (
            self.action_smoothing * self.high_level_action[:, self.num_gaits :]
            + (1.0 - self.action_smoothing) * high_level_action[:, self.num_gaits :]
        )

        rewards = torch.zeros(self.num_envs, device=self.device)
        dones_accum = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        info = {}
        reward_terms = None
        if self.record_reward_terms:
            reward_terms = {
                "velocity_reward": torch.zeros(self.num_envs, device=self.device),
                "yaw_reward": torch.zeros(self.num_envs, device=self.device),
                "orientation_penalty": torch.zeros(self.num_envs, device=self.device),
                "torque_penalty": torch.zeros(self.num_envs, device=self.device),
                "slip_penalty": torch.zeros(self.num_envs, device=self.device),
                "action_delta_penalty": torch.zeros(self.num_envs, device=self.device),
                "continuous_action_penalty": torch.zeros(self.num_envs, device=self.device),
                "action_boundary_penalty": torch.zeros(self.num_envs, device=self.device),
                "selector_reference_penalty": torch.zeros(self.num_envs, device=self.device),
                "behavior_reference_penalty": torch.zeros(self.num_envs, device=self.device),
                "vertical_velocity_penalty": torch.zeros(self.num_envs, device=self.device),
                "lateral_velocity_penalty": torch.zeros(self.num_envs, device=self.device),
                "lateral_position_penalty": torch.zeros(self.num_envs, device=self.device),
                "roll_rate_penalty": torch.zeros(self.num_envs, device=self.device),
                "pitch_rate_penalty": torch.zeros(self.num_envs, device=self.device),
                "yaw_rate_penalty": torch.zeros(self.num_envs, device=self.device),
                "clearance_reward": torch.zeros(self.num_envs, device=self.device),
                "gait_switch_penalty": torch.zeros(self.num_envs, device=self.device),
                "weighted_metric_reward": torch.zeros(self.num_envs, device=self.device),
                "edge_reset": torch.zeros(self.num_envs, device=self.device),
                "fall_penalty": torch.zeros(self.num_envs, device=self.device),
            }
            for name in self.TASK_REWARD_NAMES:
                reward_terms[f"score_{name}"] = torch.zeros(self.num_envs, device=self.device)

        for _ in range(self.low_level_steps):
            self._apply_high_level_action()
            with torch.inference_mode():
                low_action = self.low_level_policy(self.low_level_obs)
            self.low_level_obs, _, dones, info = self.env.step(low_action.to(self.device))
            rewards += self._compute_high_level_reward(dones)
            if self.record_reward_terms:
                for key, value in self.last_reward_terms.items():
                    reward_terms[key] += value
            dones_accum |= dones.bool()
            del low_action

        info["executed_high_level_action"] = self.high_level_action.detach().clone()
        if torch.any(dones_accum):
            done_ids = dones_accum.nonzero(as_tuple=False).flatten()
            self.high_level_action[done_ids] = self.default_high_level_action[done_ids]
            self.prev_high_level_action[done_ids] = self.default_high_level_action[done_ids]
            self.selector_hold_counter[done_ids] = self.selector_hold_steps

        rewards /= self.low_level_steps
        if self.record_reward_terms:
            for key in reward_terms:
                reward_terms[key] /= self.low_level_steps
            info["high_level_reward_terms"] = reward_terms
        return self.get_observations(), rewards, dones_accum, info

    def get_observations(self):
        current_obs = self._get_current_proprioceptive_obs()
        self.obs_shift_buffer.copy_(self.obs_history[:, self.num_high_level_obs :])
        self.obs_history[:, : -self.num_high_level_obs].copy_(self.obs_shift_buffer)
        self.obs_history[:, -self.num_high_level_obs :].copy_(current_obs)
        return self.obs_history.detach()

    def _get_current_proprioceptive_obs(self):
        velocity_error = torch.cat(
            (
                self.env.base_lin_vel[:, 0:2] - self.env.commands[:, 0:2],
                (self.env.base_ang_vel[:, 2] - self.env.commands[:, 2]).unsqueeze(1),
            ),
            dim=-1,
        )
        dof_pos_error = self.env.dof_pos[:, : self.env.num_actuated_dof] - self.env.default_dof_pos[
            :, : self.env.num_actuated_dof
        ]
        dof_vel = self.env.dof_vel[:, : self.env.num_actuated_dof]
        contact_state = (self.env.contact_forces[:, self.env.feet_indices, 2] > 1.0).float()
        lateral_offset = self._compute_lateral_offset()
        lateral_offset_obs = torch.clamp(lateral_offset / 2.0, -2.0, 2.0).unsqueeze(1)
        lateral_offset_abs_obs = torch.clamp(torch.abs(lateral_offset) / 2.0, 0.0, 2.0).unsqueeze(1)
        obs = torch.cat(
            (
                velocity_error,
                self.env.base_lin_vel,
                self.env.base_ang_vel,
                self.env.projected_gravity,
                dof_pos_error,
                dof_vel,
                contact_state,
                lateral_offset_obs,
                lateral_offset_abs_obs,
                self.high_level_action,
            ),
            dim=-1,
        )
        return obs

    def _apply_high_level_action(self):
        mapped = self._map_action(self.high_level_action)
        commands = self.env.commands

        commands[:, 0:3] = self.velocity_command
        commands[:, 4] = mapped["frequency"]
        commands[:, 5] = mapped["phase"]
        commands[:, 6] = mapped["offset"]
        commands[:, 7] = mapped["bound"]
        commands[:, 8] = mapped["duration"]
        commands[:, 9] = mapped["footswing_height"]
        commands[:, 10] = mapped["body_pitch"]
        commands[:, 11] = 0.0
        commands[:, 12] = mapped["stance_width"]
        if commands.shape[1] > 13:
            commands[:, 13] = 0.40

    def _map_action(self, action):
        selector_weights = self._selector_weights(action)
        gait_command = selector_weights @ self.gait_templates
        behavior_command = self._behavior_from_residual(selector_weights, action[:, self.num_gaits :])
        return {
            "selector_weights": selector_weights,
            "phase": gait_command[:, 0],
            "offset": gait_command[:, 1],
            "bound": gait_command[:, 2],
            "frequency": behavior_command[:, 0],
            "duration": behavior_command[:, 1],
            "footswing_height": behavior_command[:, 2],
            "stance_width": behavior_command[:, 3],
            "body_pitch": behavior_command[:, 4],
        }

    def _selector_weights(self, action):
        gait_ids = torch.argmax(action[:, : self.num_gaits], dim=-1)
        return torch.nn.functional.one_hot(gait_ids, num_classes=self.num_gaits).to(
            device=self.device,
            dtype=torch.float,
        )

    def _held_selector_weights(self, requested_action):
        if self.selector_hold_steps == 0:
            return self._selector_weights(requested_action)

        requested_ids = torch.argmax(requested_action[:, : self.num_gaits], dim=-1)
        current_ids = torch.argmax(self.high_level_action[:, : self.num_gaits], dim=-1)
        can_switch = self.selector_hold_counter >= self.selector_hold_steps
        should_switch = can_switch & (requested_ids != current_ids)
        selected_ids = torch.where(should_switch, requested_ids, current_ids)
        self.selector_hold_counter = torch.where(
            should_switch,
            torch.zeros_like(self.selector_hold_counter),
            self.selector_hold_counter + 1,
        )
        return torch.nn.functional.one_hot(selected_ids, num_classes=self.num_gaits).to(
            device=self.device,
            dtype=torch.float,
        )

    def _behavior_from_residual(self, selector_weights, residual_action):
        base_behavior = selector_weights @ self.gait_behavior_templates
        residual_action = torch.clamp(residual_action, -1.0, 1.0)
        residual_unit = 0.5 * (residual_action + 1.0)
        deltas = self._lerp(
            residual_unit,
            self.residual_delta_ranges[:, 0],
            self.residual_delta_ranges[:, 1],
        )
        behavior = base_behavior + deltas
        return torch.maximum(torch.minimum(behavior, self.behavior_highs), self.behavior_lows)

    def _speed_conditioned_target(self):
        if self.target_gait_ids is not None:
            target_selector_weights = torch.nn.functional.one_hot(
                self.target_gait_ids,
                num_classes=self.num_gaits,
            ).to(device=self.device, dtype=torch.float)
            target_residual_actions = torch.zeros(
                self.num_envs,
                self.num_behavior_actions,
                device=self.device,
                dtype=torch.float,
            )
            return target_selector_weights, target_residual_actions

        speed = torch.clamp(torch.abs(self.env.commands[:, 0]), 0.0, 2.0)
        speed_unit = speed / 2.0
        high_speed_weight = torch.clamp((speed - 1.1) / 0.7, 0.0, 1.0)

        target_selector_weights = torch.zeros(
            self.num_envs, self.num_gaits, device=self.device, dtype=torch.float
        )
        target_selector_weights[:, 1] = 1.0 - high_speed_weight
        target_selector_weights[:, 2] = high_speed_weight

        del speed_unit
        target_residual_actions = torch.zeros(
            self.num_envs,
            self.num_behavior_actions,
            device=self.device,
            dtype=torch.float,
        )
        return target_selector_weights, target_residual_actions

    def _map_target(self, target_selector_weights, target_residual_actions):
        gait_command = target_selector_weights @ self.gait_templates
        behavior_command = self._behavior_from_residual(target_selector_weights, target_residual_actions)
        return {
            "selector_weights": target_selector_weights,
            "phase": gait_command[:, 0],
            "offset": gait_command[:, 1],
            "bound": gait_command[:, 2],
            "frequency": behavior_command[:, 0],
            "duration": behavior_command[:, 1],
            "footswing_height": behavior_command[:, 2],
            "stance_width": behavior_command[:, 3],
            "body_pitch": behavior_command[:, 4],
        }

    @staticmethod
    def _lerp(x, low, high):
        return low + (high - low) * x

    @staticmethod
    def _inv_lerp(x, low, high):
        return (x - low) / (high - low)

    def _compute_high_level_reward(self, dones):
        vx_error = self.env.base_lin_vel[:, 0] - self.env.commands[:, 0]
        vy_error = self.env.base_lin_vel[:, 1] - self.env.commands[:, 1]
        yaw_error = self.env.base_ang_vel[:, 2] - self.env.commands[:, 2]

        velocity_reward = torch.exp(
            -(vx_error**2 + 0.25 * vy_error**2) / self.velocity_tracking_sigma
        )
        yaw_reward = torch.exp(-(yaw_error**2) / 0.10)
        orientation_penalty = torch.sum(self.env.projected_gravity[:, :2] ** 2, dim=1)
        torque_penalty = torch.mean(self.env.torques**2, dim=1) / 100.0

        contacts = self.env.contact_forces[:, self.env.feet_indices, 2] > 1.0
        foot_xy_vel = torch.sum(self.env.foot_velocities[:, :, :2] ** 2, dim=2)
        slip_penalty = torch.mean(contacts * foot_xy_vel, dim=1)

        action_delta_penalty = torch.mean(
            (self.high_level_action - self.prev_high_level_action) ** 2, dim=1
        )
        continuous_action_penalty = torch.mean(
            self.high_level_action[:, self.num_gaits :] ** 2, dim=1
        )
        residual_abs = torch.abs(self.high_level_action[:, self.num_gaits :])
        action_boundary_penalty = torch.mean(
            torch.clamp((residual_abs - 0.85) / 0.15, min=0.0) ** 2,
            dim=1,
        )
        selector_weights = self._selector_weights(self.high_level_action)
        prev_selector_weights = self._selector_weights(self.prev_high_level_action)
        gait_switch_penalty = torch.mean(
            (selector_weights - prev_selector_weights) ** 2,
            dim=1,
        )
        target_selector_weights, target_residual_actions = self._speed_conditioned_target()
        residual_actions = self.high_level_action[:, self.num_gaits :]
        selector_reference_penalty = torch.mean(
            (selector_weights - target_selector_weights) ** 2, dim=1
        )
        behavior_reference_penalty = torch.mean(
            (residual_actions - target_residual_actions) ** 2, dim=1
        )
        vertical_velocity_penalty = self.env.base_lin_vel[:, 2] ** 2
        lateral_velocity_penalty = self.env.base_lin_vel[:, 1] ** 2
        lateral_position_penalty = self._compute_lateral_position_penalty()
        roll_rate_penalty = self.env.base_ang_vel[:, 0] ** 2
        pitch_rate_penalty = self.env.base_ang_vel[:, 1] ** 2
        yaw_rate_penalty = self.env.base_ang_vel[:, 2] ** 2
        mapped = self._map_action(self.high_level_action)
        clearance_reward = torch.clamp(
            (mapped["footswing_height"] - self.footswing_range[0])
            / (self.footswing_range[1] - self.footswing_range[0]),
            0.0,
            1.0,
        )

        edge_reset = self._get_edge_reset_buf()
        fall_penalty = (dones.bool() & ~edge_reset).float()
        selector_reference_coef = self._get_selector_reference_coef()
        metric_scores = self._compute_metric_scores(
            velocity_reward=velocity_reward,
            yaw_reward=yaw_reward,
            orientation_penalty=orientation_penalty,
            pitch_rate_penalty=pitch_rate_penalty,
            roll_rate_penalty=roll_rate_penalty,
            yaw_rate_penalty=yaw_rate_penalty,
            lateral_velocity_penalty=lateral_velocity_penalty,
            lateral_position_penalty=lateral_position_penalty,
            vertical_velocity_penalty=vertical_velocity_penalty,
            slip_penalty=slip_penalty,
            torque_penalty=torque_penalty,
            clearance_reward=clearance_reward,
            gait_switch_penalty=gait_switch_penalty,
            action_delta_penalty=action_delta_penalty,
            continuous_action_penalty=continuous_action_penalty,
            action_boundary_penalty=action_boundary_penalty,
            fall_penalty=fall_penalty,
        )
        weighted_metric_reward = self._compute_weighted_metric_reward(metric_scores)
        if self.record_reward_terms:
            self.last_reward_terms = {
                "velocity_reward": velocity_reward,
                "yaw_reward": yaw_reward,
                "orientation_penalty": orientation_penalty,
                "torque_penalty": torque_penalty,
                "slip_penalty": slip_penalty,
                "action_delta_penalty": action_delta_penalty,
                "continuous_action_penalty": continuous_action_penalty,
                "action_boundary_penalty": action_boundary_penalty,
                "selector_reference_penalty": selector_reference_penalty,
                "behavior_reference_penalty": behavior_reference_penalty,
                "vertical_velocity_penalty": vertical_velocity_penalty,
                "lateral_velocity_penalty": lateral_velocity_penalty,
                "lateral_position_penalty": lateral_position_penalty,
                "roll_rate_penalty": roll_rate_penalty,
                "pitch_rate_penalty": pitch_rate_penalty,
                "yaw_rate_penalty": yaw_rate_penalty,
                "clearance_reward": clearance_reward,
                "gait_switch_penalty": gait_switch_penalty,
                "weighted_metric_reward": weighted_metric_reward,
                "edge_reset": edge_reset.float(),
                "fall_penalty": fall_penalty,
            }
            for i, name in enumerate(self.TASK_REWARD_NAMES):
                self.last_reward_terms[f"score_{name}"] = metric_scores[:, i]

        return weighted_metric_reward - selector_reference_coef * selector_reference_penalty

    def _compute_metric_scores(
        self,
        velocity_reward,
        yaw_reward,
        orientation_penalty,
        pitch_rate_penalty,
        roll_rate_penalty,
        yaw_rate_penalty,
        lateral_velocity_penalty,
        lateral_position_penalty,
        vertical_velocity_penalty,
        slip_penalty,
        torque_penalty,
        clearance_reward,
        gait_switch_penalty,
        action_delta_penalty,
        continuous_action_penalty,
        action_boundary_penalty,
        fall_penalty,
    ):
        return torch.stack(
            (
                velocity_reward,
                yaw_reward,
                torch.exp(-orientation_penalty / 0.05),
                torch.exp(-pitch_rate_penalty / 0.25),
                torch.exp(-roll_rate_penalty / 0.25),
                torch.exp(-yaw_rate_penalty / 0.25),
                torch.exp(-lateral_velocity_penalty / 0.05 - lateral_position_penalty / 1.00),
                torch.exp(-vertical_velocity_penalty / 0.05),
                torch.exp(-slip_penalty / 0.05),
                torch.exp(-torque_penalty / 0.50),
                clearance_reward,
                torch.exp(-gait_switch_penalty / 0.25),
                torch.exp(-action_delta_penalty / 0.05),
                torch.exp(-continuous_action_penalty / 0.25),
                torch.exp(-action_boundary_penalty / 0.25),
                1.0 - fall_penalty,
            ),
            dim=1,
        )

    def _compute_weighted_metric_reward(self, scores):
        if self.task_reward_weights is None:
            return torch.mean(scores, dim=1)
        weight_sum = torch.sum(self.task_reward_weights, dim=1).clamp(min=1e-6)
        return torch.sum(self.task_reward_weights * scores, dim=1) / weight_sum

    def __getattr__(self, name):
        return getattr(self.env, name)

    def _get_base_env(self):
        env = self.env
        while hasattr(env, "env"):
            env = env.env
        return env

    def _compute_lateral_position_penalty(self):
        lateral_offset = self._compute_lateral_offset()
        return torch.clamp(torch.abs(lateral_offset) - 0.25, min=0.0) ** 2

    def _compute_lateral_offset(self):
        base_env = self._get_base_env()
        env_origins = getattr(base_env, "env_origins", None)
        if env_origins is None:
            return base_env.root_states[: self.num_envs, 1]
        return base_env.root_states[: self.num_envs, 1] - env_origins[: self.num_envs, 1]

    def _get_edge_reset_buf(self):
        edge_reset = getattr(self._get_base_env(), "edge_reset_buf", None)
        if edge_reset is None:
            return torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        return edge_reset.to(device=self.device, dtype=torch.bool)

    def _get_selector_reference_coef(self):
        if self.selector_reference_coef_tensor is not None:
            return self.selector_reference_coef_tensor
        return self.selector_reference_coef

    def get_high_level_privileged_obs(self):
        """Assemble privileged environment observations for high-level RMA training.

        Returns a 14-dim tensor per env containing terrain, dynamics, push,
        and body-state information that is available in simulation but not
        on the real robot.  Used as the teacher target for AdaptationModule.

        Layout:
          0  terrain_height_mean   mean of measured_heights, clamped
          1  terrain_height_std    std  of measured_heights, clamped
          2  terrain_height_range  max-min of measured_heights, clamped
          3  terrain_slope         front-back height gradient proxy
          4  friction              ground friction coefficient
          5  base_mass             payload / added mass
          6  com_dx                COM displacement x, normalised
          7  com_dy                COM displacement y, normalised
          8  com_dz                COM displacement z, normalised
          9  push_active           1.0 if push is enabled for this env
         10  push_axis             -1 none / 0 longitudinal / 1 lateral
         11  body_height           base height above ground, normalised
         12  pitch_proxy           projected-gravity x component
         13  roll_proxy            projected-gravity y component
        """
        base_env = self._get_base_env()

        # --- terrain height stats from measured_heights ---
        heights = getattr(base_env, "measured_heights", None)
        if heights is not None and heights.ndim == 2 and heights.shape[1] > 0:
            h_mean = torch.clamp(heights.mean(dim=1) / 0.5, -1.0, 1.0).unsqueeze(1)
            h_std = torch.clamp(heights.std(dim=1) / 0.3, 0.0, 1.0).unsqueeze(1)
            h_range = torch.clamp(
                (heights.max(dim=1).values - heights.min(dim=1).values) / 0.5, 0.0, 1.0
            ).unsqueeze(1)
            # slope proxy: difference between mean of front half and back half of height points
            mid = heights.shape[1] // 2
            front_mean = heights[:, mid:].mean(dim=1)
            back_mean = heights[:, :mid].mean(dim=1)
            h_slope = torch.clamp((front_mean - back_mean) / 0.3, -1.0, 1.0).unsqueeze(1)
        else:
            z = torch.zeros(self.num_envs, 1, device=self.device)
            h_mean = h_std = h_range = h_slope = z

        # --- friction ---
        friction = getattr(base_env, "friction_coeffs", None)
        if friction is not None and friction.ndim >= 1:
            friction_val = torch.clamp(friction[:, 0] / 1.5, 0.0, 1.0).unsqueeze(1)
        else:
            friction_val = torch.zeros(self.num_envs, 1, device=self.device)

        # --- base mass (payload) ---
        payloads = getattr(base_env, "payloads", None)
        if payloads is not None and payloads.ndim >= 1:
            # payload range is roughly [-1, 3] kg
            mass_val = torch.clamp((payloads.unsqueeze(1) + 1.0) / 4.0, 0.0, 1.0)
        else:
            mass_val = torch.zeros(self.num_envs, 1, device=self.device)

        # --- COM displacement ---
        com = getattr(base_env, "com_displacements", None)
        if com is not None and com.ndim == 2 and com.shape[1] >= 3:
            com_val = torch.clamp(com[:, :3] / 0.1, -1.0, 1.0)
        else:
            com_val = torch.zeros(self.num_envs, 3, device=self.device)

        # --- push info ---
        cfg = getattr(base_env, "cfg", None)
        push_axis_by_env = None
        if cfg is not None:
            push_axis_by_env = getattr(cfg.domain_rand, "push_axis_by_env", None)
        if push_axis_by_env is not None:
            push_axis_t = torch.tensor(push_axis_by_env, device=self.device, dtype=torch.float)
            push_active = (push_axis_t >= 0).float().unsqueeze(1)
            push_axis_norm = torch.clamp(push_axis_t / 1.0, -1.0, 1.0).unsqueeze(1)
        else:
            push_active = torch.zeros(self.num_envs, 1, device=self.device)
            push_axis_norm = -torch.ones(self.num_envs, 1, device=self.device)

        # --- body height ---
        # nominal height 0.34 m, scale so that 0.15-0.55 maps to roughly [-1,1]
        root_z = base_env.root_states[: self.num_envs, 2]
        body_height = torch.clamp((root_z - 0.34).unsqueeze(1) / 0.15, -1.0, 1.0)

        # --- pitch / roll proxies from projected gravity ---
        proj_grav = base_env.projected_gravity[: self.num_envs]
        pitch_proxy = proj_grav[:, 0:1]
        roll_proxy = proj_grav[:, 1:2]

        priv = torch.cat(
            (
                h_mean,
                h_std,
                h_range,
                h_slope,
                friction_val,
                mass_val,
                com_val,
                push_active,
                push_axis_norm,
                body_height,
                pitch_proxy,
                roll_proxy,
            ),
            dim=1,
        )
        return priv
