import torch
import torch.nn.functional as F

from go2_gym.envs.wrappers.high_level_reward_metrics import (
    CANONICAL_REWARD_NAMES,
    compute_metric_score_dict,
    compute_weighted_metric_reward,
    stack_metric_scores,
)


class MinimalGaitWrapper:
    """Turn gait choices and parameter residuals into commands for frozen WTW."""

    TASK_REWARD_NAMES = CANONICAL_REWARD_NAMES

    def __init__(
        self,
        env,
        low_level_policy,
        high_level_dt=0.10,
        history_length=10,
        action_smoothing=0.6,
        record_reward_terms=True,
    ):
        self.env = env
        self.low_level_policy = low_level_policy
        self.high_level_dt = high_level_dt
        self.low_level_steps = max(1, round(high_level_dt / env.dt))
        self.history_length = history_length
        self.action_smoothing = action_smoothing
        self.record_reward_terms = record_reward_terms

        self.num_envs = env.num_envs
        self.device = env.device
        self.num_gaits = 4
        self.num_behavior_actions = 5
        self.num_high_level_actions = self.num_gaits + self.num_behavior_actions

        # phase, offset and bound encode the four WTW gait families.
        self.gait_templates = torch.tensor(
            (
                (0.0, 0.0, 0.0),  # pronking
                (0.5, 0.0, 0.0),  # trotting
                (0.0, 0.5, 0.0),  # bounding
                (0.0, 0.0, 0.5),  # pacing
            ),
            device=self.device,
        )
        # frequency, duration, swing height, stance width and body pitch.
        self.behavior_templates = torch.tensor(
            (
                (3.0, 0.5, 0.08, 0.33, 0.0),
                (3.0, 0.5, 0.08, 0.33, 0.0),
                (3.0, 0.5, 0.12, 0.38, 0.0),
                (2.5, 0.5, 0.12, 0.38, 0.0),
            ),
            device=self.device,
        )
        self.residual_ranges = torch.tensor(
            (
                (-0.4, 0.4),
                (-0.08, 0.08),
                (-0.03, 0.03),
                (-0.04, 0.04),
                (-0.04, 0.04),
            ),
            device=self.device,
        )
        self.behavior_lows = torch.tensor(
            (2.0, 0.42, 0.04, 0.25, -0.10),
            device=self.device,
        )
        self.behavior_highs = torch.tensor(
            (3.5, 0.58, 0.12, 0.38, 0.05),
            device=self.device,
        )

        # One frame has 42 proprioceptive values plus the previous 9-D action.
        self.num_high_level_obs = 42 + self.num_high_level_actions
        self.num_high_level_obs_history = self.num_high_level_obs * history_length
        self.obs_history = torch.zeros(
            self.num_envs,
            self.num_high_level_obs_history,
            device=self.device,
        )

        self.high_level_action = torch.zeros(
            self.num_envs,
            self.num_high_level_actions,
            device=self.device,
        )
        self.default_action = torch.zeros_like(self.high_level_action)
        self.default_action[:, 1] = 1.0
        self.previous_action = self.default_action.clone()
        self.velocity_command = torch.zeros(self.num_envs, 3, device=self.device)
        self.previous_contacts = torch.zeros(
            self.num_envs,
            4,
            device=self.device,
            dtype=torch.bool,
        )
        self.low_level_obs = None
        self.task_reward_weights = None
        self.last_reward_terms = {}

    def reset(self):
        self.low_level_obs = self.env.reset()
        self.high_level_action.copy_(self.default_action)
        self.previous_action.copy_(self.default_action)
        self.previous_contacts = self._foot_contacts()
        self.obs_history.zero_()
        return self.get_observations()

    def set_velocity_command(self, vx, vy=0.0, yaw=0.0):
        self.velocity_command[:, 0] = torch.as_tensor(vx, device=self.device)
        self.velocity_command[:, 1] = torch.as_tensor(vy, device=self.device)
        self.velocity_command[:, 2] = torch.as_tensor(yaw, device=self.device)
        self.env.commands[:, :3] = self.velocity_command

    def set_task_reward_weights(self, weights):
        weights = torch.as_tensor(weights, device=self.device, dtype=torch.float)
        expected = (self.num_envs, len(self.TASK_REWARD_NAMES))
        if tuple(weights.shape) != expected:
            raise ValueError(f"Expected reward weights {expected}, got {tuple(weights.shape)}")
        self.task_reward_weights = weights

    def step(self, requested_action):
        requested_action = requested_action.to(self.device).clamp(-1.0, 1.0).detach()
        self.previous_action.copy_(self.high_level_action)
        self.high_level_action[:, : self.num_gaits] = self._one_hot_gait(requested_action)
        self.high_level_action[:, self.num_gaits :] = (
            self.action_smoothing * self.high_level_action[:, self.num_gaits :]
            + (1.0 - self.action_smoothing) * requested_action[:, self.num_gaits :]
        )

        reward_sum = torch.zeros(self.num_envs, device=self.device)
        done_any = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        edge_any = torch.zeros_like(done_any)
        timeout_any = torch.zeros_like(done_any)
        term_sums = {}
        info = {}

        for _ in range(self.low_level_steps):
            self._write_low_level_commands()
            with torch.inference_mode():
                low_action = self.low_level_policy(self.low_level_obs)
            self.low_level_obs, _, done, info = self.env.step(low_action.to(self.device))
            reward_sum += self._reward(done)
            done_any |= done.bool()
            edge_any |= self._reset_buffer("edge_reset_buf")
            timeout_any |= self._reset_buffer("time_out_buf")
            if self.record_reward_terms:
                for name, value in self.last_reward_terms.items():
                    term_sums.setdefault(name, torch.zeros_like(value)).add_(value)

        info["executed_high_level_action"] = self.high_level_action.detach().clone()
        info["high_level_edge_resets"] = edge_any
        info["high_level_timeouts"] = timeout_any
        info["high_level_physical_dones"] = done_any & ~edge_any & ~timeout_any

        if done_any.any():
            done_ids = done_any.nonzero(as_tuple=False).flatten()
            self.high_level_action[done_ids] = self.default_action[done_ids]
            self.previous_action[done_ids] = self.default_action[done_ids]

        if self.record_reward_terms:
            info["high_level_reward_terms"] = {
                name: value / self.low_level_steps
                for name, value in term_sums.items()
            }
        return (
            self.get_observations(),
            reward_sum / self.low_level_steps,
            done_any,
            info,
        )

    def get_observations(self):
        frame = self._proprioceptive_frame()
        self.obs_history[:, :-self.num_high_level_obs] = self.obs_history[
            :, self.num_high_level_obs :
        ].clone()
        self.obs_history[:, -self.num_high_level_obs :] = frame
        return self.obs_history.detach()

    def _proprioceptive_frame(self):
        velocity_error = torch.cat(
            (
                self.env.base_lin_vel[:, :2] - self.env.commands[:, :2],
                (self.env.base_ang_vel[:, 2] - self.env.commands[:, 2]).unsqueeze(1),
            ),
            dim=1,
        )
        actuated = self.env.num_actuated_dof
        joint_error = (
            self.env.dof_pos[:, :actuated]
            - self.env.default_dof_pos[:, :actuated]
        )
        joint_velocity = self.env.dof_vel[:, :actuated]
        contacts = self._foot_contacts().float()
        lateral = self._lateral_offset()
        return torch.cat(
            (
                velocity_error,
                self.env.base_lin_vel,
                self.env.base_ang_vel,
                self.env.projected_gravity,
                joint_error,
                joint_velocity,
                contacts,
                (lateral / 2.0).clamp(-2.0, 2.0).unsqueeze(1),
                (lateral.abs() / 2.0).clamp(0.0, 2.0).unsqueeze(1),
                self.high_level_action,
            ),
            dim=1,
        )

    def _one_hot_gait(self, action):
        gait_ids = action[:, : self.num_gaits].argmax(dim=1)
        return F.one_hot(gait_ids, self.num_gaits).to(self.device, torch.float)

    def _mapped_action(self):
        selector = self._one_hot_gait(self.high_level_action)
        gait = selector @ self.gait_templates
        behavior = selector @ self.behavior_templates
        residual = self.high_level_action[:, self.num_gaits :].clamp(-1.0, 1.0)
        residual_unit = 0.5 * (residual + 1.0)
        delta = self.residual_ranges[:, 0] + (
            self.residual_ranges[:, 1] - self.residual_ranges[:, 0]
        ) * residual_unit
        behavior = (behavior + delta).maximum(self.behavior_lows).minimum(
            self.behavior_highs
        )
        return gait, behavior

    def _write_low_level_commands(self):
        gait, behavior = self._mapped_action()
        commands = self.env.commands
        commands[:, :3] = self.velocity_command
        commands[:, 4:8] = torch.cat((behavior[:, :1], gait), dim=1)
        commands[:, 8] = behavior[:, 1]
        commands[:, 9] = behavior[:, 2]
        commands[:, 10] = behavior[:, 4]
        commands[:, 11] = 0.0
        commands[:, 12] = behavior[:, 3]
        if commands.shape[1] > 13:
            commands[:, 13] = 0.40

    def _reward(self, done):
        vx_error = self.env.base_lin_vel[:, 0] - self.env.commands[:, 0]
        vy_error = self.env.base_lin_vel[:, 1] - self.env.commands[:, 1]
        yaw_error = self.env.base_ang_vel[:, 2] - self.env.commands[:, 2]
        contacts = self._foot_contacts()
        contact_float = contacts.float()
        contact_count = contact_float.sum(dim=1).clamp_min(1.0)
        foot_xy_speed_sq = self.env.foot_velocities[:, :, :2].square().sum(dim=2)
        impact, scuffing = self._contact_safety(contacts)
        joint_power = self.env.torques * self.env.dof_vel[:, : self.env.torques.shape[1]]
        power = joint_power.abs().sum(dim=1)
        selector = self._one_hot_gait(self.high_level_action)
        previous_selector = self._one_hot_gait(self.previous_action)
        _, behavior = self._mapped_action()
        edge = self._reset_buffer("edge_reset_buf")
        timeout = self._reset_buffer("time_out_buf")

        primitive = {
            "velocity_reward": torch.exp(
                -(vx_error.square() + 0.25 * vy_error.square()) / 0.25
            ),
            "yaw_reward": torch.exp(-yaw_error.square() / 0.10),
            "orientation_penalty": self.env.projected_gravity[:, :2].square().sum(dim=1),
            "pitch_rate_penalty": self.env.base_ang_vel[:, 1].square(),
            "roll_rate_penalty": self.env.base_ang_vel[:, 0].square(),
            "yaw_rate_penalty": self.env.base_ang_vel[:, 2].square(),
            "lateral_velocity_penalty": self.env.base_lin_vel[:, 1].square(),
            "lateral_position_penalty": (self._lateral_offset().abs() - 0.25)
            .clamp_min(0.0)
            .square(),
            "vertical_velocity_penalty": self.env.base_lin_vel[:, 2].square(),
            "slip_penalty": (contacts * foot_xy_speed_sq).float().mean(dim=1),
            "contact_slip_penalty": (
                contact_float * foot_xy_speed_sq
            ).sum(dim=1)
            / contact_count,
            "torque_penalty": self.env.torques.square().mean(dim=1) / 100.0,
            "mechanical_power_abs": power,
            "transport_cost_proxy": power
            / self.env.base_lin_vel[:, 0].abs().clamp_min(0.3),
            "clearance_reward": (
                (behavior[:, 2] - 0.04) / (0.12 - 0.04)
            ).clamp(0.0, 1.0),
            "gait_switch_penalty": (selector - previous_selector).square().mean(dim=1),
            "action_delta_penalty": (
                self.high_level_action - self.previous_action
            ).square().mean(dim=1),
            "continuous_action_penalty": self.high_level_action[
                :, self.num_gaits :
            ].square().mean(dim=1),
            "action_boundary_penalty": (
                (self.high_level_action[:, self.num_gaits :].abs() - 0.85)
                .div(0.15)
                .clamp_min(0.0)
                .square()
                .mean(dim=1)
            ),
            "fall_penalty": (done.bool() & ~edge & ~timeout).float(),
            "impact_velocity_rms": impact,
            "scuffing_ratio": scuffing,
        }
        score_dict = compute_metric_score_dict(**primitive)
        scores = stack_metric_scores(score_dict, self.TASK_REWARD_NAMES)
        reward = compute_weighted_metric_reward(scores, self.task_reward_weights)

        if self.record_reward_terms:
            self.last_reward_terms = dict(primitive)
            self.last_reward_terms["weighted_metric_reward"] = reward
            for index, name in enumerate(self.TASK_REWARD_NAMES):
                self.last_reward_terms[f"score_{name}"] = scores[:, index]
        self.previous_contacts = contacts.detach().clone()
        return reward

    def _contact_safety(self, contacts):
        new_contacts = contacts & ~self.previous_contacts
        previous_velocity = getattr(
            self.env,
            "prev_foot_velocities",
            self.env.foot_velocities,
        )
        impact_velocity = (-previous_velocity[:, :, 2]).clamp_min(0.0)
        count = new_contacts.float().sum(dim=1).clamp_min(1.0)
        impact = (
            (new_contacts.float() * impact_velocity.square()).sum(dim=1) / count
        ).sqrt()

        positions = getattr(self.env, "foot_positions", None)
        if positions is None:
            return impact, torch.zeros(self.num_envs, device=self.device)
        clearance = positions[:, :, 2] - self._ground_height(positions)
        swing = (~contacts).float()
        scuffing = (
            swing * (clearance < 0.035).float()
        ).sum(dim=1) / swing.sum(dim=1).clamp_min(1.0)
        return impact, scuffing

    def _ground_height(self, foot_positions):
        base = self._base_env()
        if not hasattr(base, "height_samples") or not hasattr(base, "terrain"):
            return torch.zeros_like(foot_positions[:, :, 2])
        cfg = base.terrain.cfg
        points = foot_positions[:, :, :2] + cfg.border_size
        x = (points[:, :, 0] / cfg.horizontal_scale).long().clamp(
            0, base.height_samples.shape[0] - 2
        )
        y = (points[:, :, 1] / cfg.horizontal_scale).long().clamp(
            0, base.height_samples.shape[1] - 2
        )
        heights = torch.minimum(
            torch.minimum(base.height_samples[x, y], base.height_samples[x + 1, y]),
            base.height_samples[x, y + 1],
        )
        return heights * cfg.vertical_scale

    def get_high_level_privileged_obs(self):
        """Return the 14 generic simulation-only physical quantities."""
        base = self._base_env()
        heights = getattr(base, "measured_heights", None)
        if heights is not None and heights.ndim == 2 and heights.shape[1]:
            middle = heights.shape[1] // 2
            terrain = torch.stack(
                (
                    (heights.mean(dim=1) / 0.5).clamp(-1.0, 1.0),
                    (heights.std(dim=1) / 0.3).clamp(0.0, 1.0),
                    (
                        (heights.max(dim=1).values - heights.min(dim=1).values)
                        / 0.5
                    ).clamp(0.0, 1.0),
                    (
                        heights[:, middle:].mean(dim=1)
                        - heights[:, :middle].mean(dim=1)
                    ).div(0.3).clamp(-1.0, 1.0),
                ),
                dim=1,
            )
        else:
            terrain = torch.zeros(self.num_envs, 4, device=self.device)

        friction = getattr(base, "friction_coeffs", None)
        friction = (
            (friction[:, 0] / 1.5).clamp(0.0, 1.0).unsqueeze(1)
            if friction is not None
            else torch.zeros(self.num_envs, 1, device=self.device)
        )
        payload = getattr(base, "payloads", None)
        payload = (
            ((payload + 1.0) / 4.0).clamp(0.0, 1.0).reshape(-1, 1)
            if payload is not None
            else torch.zeros(self.num_envs, 1, device=self.device)
        )
        com = getattr(base, "com_displacements", None)
        com = (
            (com[:, :3] / 0.1).clamp(-1.0, 1.0)
            if com is not None
            else torch.zeros(self.num_envs, 3, device=self.device)
        )
        axes = getattr(base.cfg.domain_rand, "push_axis_by_env", None)
        if axes is None:
            push = torch.cat(
                (
                    torch.zeros(self.num_envs, 1, device=self.device),
                    -torch.ones(self.num_envs, 1, device=self.device),
                ),
                dim=1,
            )
        else:
            axes = torch.as_tensor(axes, device=self.device, dtype=torch.float)
            push = torch.stack(((axes >= 0).float(), axes.clamp(-1.0, 1.0)), dim=1)

        body_height = ((base.root_states[: self.num_envs, 2] - 0.34) / 0.15).clamp(
            -1.0, 1.0
        ).unsqueeze(1)
        return torch.cat(
            (
                terrain,
                friction,
                payload,
                com,
                push,
                body_height,
                base.projected_gravity[: self.num_envs, :2],
            ),
            dim=1,
        )

    def _base_env(self):
        env = self.env
        while hasattr(env, "env"):
            env = env.env
        return env

    def _foot_contacts(self):
        return self.env.contact_forces[:, self.env.feet_indices, 2] > 1.0

    def _lateral_offset(self):
        base = self._base_env()
        origins = getattr(base, "env_origins", None)
        if origins is None:
            return base.root_states[: self.num_envs, 1]
        return base.root_states[: self.num_envs, 1] - origins[: self.num_envs, 1]

    def _reset_buffer(self, name):
        value = getattr(self._base_env(), name, None)
        if value is None:
            return torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        return value.to(self.device, torch.bool)

    def __getattr__(self, name):
        return getattr(self.env, name)
