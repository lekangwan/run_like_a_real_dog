import torch

from .config import (
    EDGE_RESET_MARGIN,
    MESH_TYPE,
    TELEPORT_THRESHOLD,
    TERRAIN_LENGTH,
    TERRAIN_WIDTH,
)
from .gait_wrapper import MinimalGaitWrapper
from .low_level import create_environment, load_policy
from .tasks import build_assignment


def sample_uniform(low, high):
    return low + torch.rand_like(low) * (high - low)


def clean_privileged_observation(observation):
    """Remove direct push-condition flags while retaining generic physics."""
    cleaned = observation.clone()
    cleaned[:, 9:11] = 0.0
    return cleaned


class HighLevelEnvironment:
    """Mixed-terrain high-level environment with no task-id observation."""

    def __init__(
        self,
        specs,
        low_level_run,
        num_envs,
        render=False,
        terrain_length=TERRAIN_LENGTH,
        terrain_width=TERRAIN_WIDTH,
        recording_width=None,
        recording_height=None,
    ):
        self.specs = specs
        cpu_assignment = build_assignment(specs, num_envs, "cpu")
        low_env = create_environment(
            low_level_run,
            num_envs,
            cpu_assignment.conditions,
            cpu_assignment.push_axes,
            render,
            terrain_length,
            terrain_width,
            EDGE_RESET_MARGIN,
            TELEPORT_THRESHOLD,
            recording_width,
            recording_height,
        )
        self.env = MinimalGaitWrapper(
            low_env,
            load_policy(low_level_run),
            record_reward_terms=True,
        )
        self.device = self.env.device
        self.assignment = build_assignment(specs, num_envs, self.device)
        self.env.set_task_reward_weights(self.assignment.reward_weights)

        self.num_envs = num_envs
        self.num_gaits = self.env.num_gaits
        self.action_dim = self.env.num_high_level_actions
        self.base_obs_dim = self.env.num_high_level_obs_history
        self.policy_obs_dim = self.base_obs_dim + 1
        self.vx_command = sample_uniform(
            self.assignment.vx_lows,
            self.assignment.vx_highs,
        )

    def _policy_observation(self, history):
        return torch.cat((history, self.command_vx()[:, None]), dim=-1)

    def reset(self):
        history = self.env.reset()
        self.env.set_velocity_command(self.vx_command, 0.0, 0.0)
        return self._policy_observation(history)

    def step(self, action):
        history, reward, done, info = self.env.step(action)
        done_ids = done.nonzero(as_tuple=False).flatten()
        if done_ids.numel():
            self.vx_command[done_ids] = sample_uniform(
                self.assignment.vx_lows[done_ids],
                self.assignment.vx_highs[done_ids],
            )
        self.env.set_velocity_command(self.vx_command, 0.0, 0.0)
        return self._policy_observation(history), reward, done, info

    def base_history(self):
        return self.env.obs_history.detach()

    def privileged_observation(self):
        return clean_privileged_observation(
            self.env.get_high_level_privileged_obs()
        )

    def command_vx(self):
        return self.env.commands[:, 0]

    def measured_vx(self):
        return self.env.base_lin_vel[:, 0]
