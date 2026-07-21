from pathlib import Path
import sys
import types
import unittest

import torch
import torch.nn.functional as F
from torch.distributions import Categorical, Normal


def install_training_dependency_stubs():
    """Load the actor without importing Isaac Gym's simulator extension."""
    sys.modules["isaacgym"] = types.ModuleType("isaacgym")

    go2_gym = types.ModuleType("go2_gym")
    go2_gym.MINI_GYM_ROOT_DIR = Path(".")
    sys.modules["go2_gym"] = go2_gym
    sys.modules["go2_gym.envs"] = types.ModuleType("go2_gym.envs")
    sys.modules["go2_gym.envs.base"] = types.ModuleType("go2_gym.envs.base")
    sys.modules["go2_gym.envs.go2"] = types.ModuleType("go2_gym.envs.go2")
    sys.modules["go2_gym.envs.wrappers"] = types.ModuleType("go2_gym.envs.wrappers")

    config_module = types.ModuleType("go2_gym.envs.base.legged_robot_config")
    config_module.Cfg = object()
    sys.modules[config_module.__name__] = config_module

    go2_config_module = types.ModuleType("go2_gym.envs.go2.go2_config")
    go2_config_module.config_go2 = lambda _cfg: None
    sys.modules[go2_config_module.__name__] = go2_config_module

    velocity_module = types.ModuleType("go2_gym.envs.go2.velocity_tracking")
    velocity_module.VelocityTrackingEasyEnv = object
    sys.modules[velocity_module.__name__] = velocity_module

    history_module = types.ModuleType("go2_gym.envs.wrappers.history_wrapper")
    history_module.HistoryWrapper = object
    sys.modules[history_module.__name__] = history_module

    gait_wrapper_module = types.ModuleType("go2_gym.envs.wrappers.high_level_gait_wrapper")
    gait_wrapper_module.HighLevelGaitWrapper = object
    sys.modules[gait_wrapper_module.__name__] = gait_wrapper_module


install_training_dependency_stubs()

from train_high_level_ppo import ActorCritic


class GaitInputResidualTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        self.obs_dim = 6
        self.num_gaits = 4
        self.residual_dim = 3
        self.batch_size = 5
        self.obs = torch.randn(self.batch_size, self.obs_dim)

    def make_model(self, gait_input=False, additive_correction=False):
        return ActorCritic(
            self.obs_dim,
            self.num_gaits,
            self.residual_dim,
            z_dim=0,
            hidden_dims=(8,),
            init_std=0.2,
            gait_conditioned_residuals=additive_correction,
            gait_input_residuals=gait_input,
        )

    def test_modes_are_mutually_exclusive(self):
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            self.make_model(gait_input=True, additive_correction=True)

    def test_old_checkpoint_loads_and_new_network_starts_at_zero(self):
        old_model = self.make_model()
        new_model = self.make_model(gait_input=True)
        incompatible = new_model.load_state_dict(old_model.state_dict(), strict=False)

        self.assertEqual(incompatible.unexpected_keys, [])
        self.assertEqual(
            set(incompatible.missing_keys),
            {
                "actor.gait_input_residual_network.0.weight",
                "actor.gait_input_residual_network.0.bias",
                "actor.gait_input_residual_network.2.weight",
                "actor.gait_input_residual_network.2.bias",
            },
        )

        new_logits, all_means = new_model.actor.all_distribution_params(self.obs)
        old_logits, _old_mean = old_model.distribution_params(self.obs)
        torch.testing.assert_close(new_logits, old_logits)
        torch.testing.assert_close(all_means, torch.zeros_like(all_means))

    def test_one_network_uses_the_selected_gait_code(self):
        model = self.make_model(gait_input=True)
        network = model.actor.gait_input_residual_network
        with torch.no_grad():
            network[0].weight.zero_()
            network[0].bias.zero_()
            network[2].weight.zero_()
            network[2].bias.zero_()
            gait_start = network[0].in_features - self.num_gaits
            network[0].weight[0, gait_start + 2] = 1.0
            network[2].weight[1, 0] = 1.0

        gait_zero = torch.zeros(self.batch_size, dtype=torch.long)
        gait_two = torch.full((self.batch_size,), 2, dtype=torch.long)
        _, mean_zero = model.distribution_params(self.obs, gait_ids=gait_zero)
        _, mean_two = model.distribution_params(self.obs, gait_ids=gait_two)

        torch.testing.assert_close(mean_zero, torch.zeros_like(mean_zero))
        expected = torch.zeros_like(mean_two)
        expected[:, 1] = torch.tanh(torch.tensor(1.0))
        torch.testing.assert_close(mean_two, expected)

    def test_log_probability_uses_sampled_gait_condition(self):
        model = self.make_model(gait_input=True)
        network = model.actor.gait_input_residual_network
        with torch.no_grad():
            network[2].weight.fill_(0.1)
            network[2].bias.copy_(torch.tensor((0.2, -0.1, 0.3)))

        gait_ids = torch.ones(self.batch_size, dtype=torch.long)
        residual_actions = torch.full((self.batch_size, self.residual_dim), 0.15)
        actions = torch.cat(
            (
                F.one_hot(gait_ids, num_classes=self.num_gaits).float(),
                residual_actions,
            ),
            dim=-1,
        )
        actual_log_prob, _entropy, _value = model.evaluate_actions(self.obs, actions)

        gait_logits, residual_mean = model.distribution_params(self.obs, gait_ids=gait_ids)
        expected_log_prob = Categorical(logits=gait_logits).log_prob(gait_ids)
        expected_log_prob += Normal(
            residual_mean,
            torch.exp(model.log_std).expand_as(residual_mean),
        ).log_prob(residual_actions).sum(dim=-1)
        torch.testing.assert_close(actual_log_prob, expected_log_prob)

    def test_only_sampled_gait_input_column_receives_gradient(self):
        model = self.make_model(gait_input=True)
        network = model.actor.gait_input_residual_network
        with torch.no_grad():
            network[2].weight.fill_(0.1)

        gait_ids = torch.ones(self.batch_size, dtype=torch.long)
        residual_actions = torch.full((self.batch_size, self.residual_dim), 0.25)
        actions = torch.cat(
            (
                F.one_hot(gait_ids, num_classes=self.num_gaits).float(),
                residual_actions,
            ),
            dim=-1,
        )
        log_prob, _entropy, _value = model.evaluate_actions(self.obs, actions)
        (-log_prob.mean()).backward()

        gait_grads = network[0].weight.grad[:, -self.num_gaits :]
        self.assertGreater(gait_grads[:, 1].abs().sum().item(), 0.0)
        torch.testing.assert_close(gait_grads[:, 0], torch.zeros_like(gait_grads[:, 0]))
        torch.testing.assert_close(gait_grads[:, 2:], torch.zeros_like(gait_grads[:, 2:]))


if __name__ == "__main__":
    unittest.main()
