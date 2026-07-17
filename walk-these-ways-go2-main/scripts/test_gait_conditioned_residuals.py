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


class GaitConditionedResidualTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.obs_dim = 6
        self.num_gaits = 4
        self.residual_dim = 3
        self.batch_size = 5
        self.obs = torch.randn(self.batch_size, self.obs_dim)

    def make_model(self, conditioned):
        return ActorCritic(
            self.obs_dim,
            self.num_gaits,
            self.residual_dim,
            z_dim=0,
            hidden_dims=(8,),
            init_std=0.2,
            gait_conditioned_residuals=conditioned,
        )

    def test_old_checkpoint_preserves_initial_behavior(self):
        shared_model = self.make_model(conditioned=False)
        conditioned_model = self.make_model(conditioned=True)
        incompatible = conditioned_model.load_state_dict(shared_model.state_dict(), strict=False)

        self.assertEqual(incompatible.unexpected_keys, [])
        self.assertEqual(
            set(incompatible.missing_keys),
            {
                "actor.gait_residual_delta_head.weight",
                "actor.gait_residual_delta_head.bias",
            },
        )

        for gait_id in range(self.num_gaits):
            gait_ids = torch.full((self.batch_size,), gait_id, dtype=torch.long)
            old_logits, old_mean = shared_model.distribution_params(self.obs, gait_ids=gait_ids)
            new_logits, new_mean = conditioned_model.distribution_params(self.obs, gait_ids=gait_ids)
            torch.testing.assert_close(new_logits, old_logits)
            torch.testing.assert_close(new_mean, old_mean)

    def test_each_gait_selects_its_own_correction(self):
        model = self.make_model(conditioned=True)
        model.zero_init_residual_heads()
        with torch.no_grad():
            correction = model.actor.gait_residual_delta_head.bias.view(
                self.num_gaits,
                self.residual_dim,
            )
            correction[2, 1] = 0.4

        gait_zero = torch.zeros(self.batch_size, dtype=torch.long)
        gait_two = torch.full((self.batch_size,), 2, dtype=torch.long)
        _, mean_zero = model.distribution_params(self.obs, gait_ids=gait_zero)
        _, mean_two = model.distribution_params(self.obs, gait_ids=gait_two)

        torch.testing.assert_close(mean_zero, torch.zeros_like(mean_zero))
        expected = torch.zeros_like(mean_two)
        expected[:, 1] = torch.tanh(torch.tensor(0.4))
        torch.testing.assert_close(mean_two, expected)

    def test_log_probability_uses_sampled_gait_correction(self):
        model = self.make_model(conditioned=True)
        model.zero_init_residual_heads()
        with torch.no_grad():
            correction = model.actor.gait_residual_delta_head.bias.view(
                self.num_gaits,
                self.residual_dim,
            )
            correction[1] = torch.tensor((0.2, -0.1, 0.3))

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

    def test_only_selected_gait_correction_receives_gradient(self):
        model = self.make_model(conditioned=True)
        model.zero_init_residual_heads()
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
        correction_grad = model.actor.gait_residual_delta_head.bias.grad.view(
            self.num_gaits,
            self.residual_dim,
        )

        self.assertGreater(correction_grad[1].abs().sum().item(), 0.0)
        torch.testing.assert_close(correction_grad[0], torch.zeros_like(correction_grad[0]))
        torch.testing.assert_close(correction_grad[2], torch.zeros_like(correction_grad[2]))
        torch.testing.assert_close(correction_grad[3], torch.zeros_like(correction_grad[3]))

    def test_student_inference_uses_configured_temporal_summary(self):
        base_obs_dim = 20
        model = ActorCritic(
            base_obs_dim,
            self.num_gaits,
            self.residual_dim,
            base_obs_dim=base_obs_dim,
            priv_dim=4,
            z_dim=2,
            hidden_dims=(8,),
            adaptation_temporal_summary=True,
            adaptation_history_length=10,
            gait_conditioned_residuals=True,
        )
        obs = torch.randn(self.batch_size, base_obs_dim)

        expected_z = model.encode_student(obs)
        expected_action = model.act_inference(torch.cat((obs, expected_z), dim=-1))
        actual_action = model.act_student(obs)
        torch.testing.assert_close(actual_action, expected_action)

    def test_residual_mean_penalty_has_gait_specific_gradient(self):
        model = self.make_model(conditioned=True)
        model.zero_init_residual_heads()
        with torch.no_grad():
            correction = model.actor.gait_residual_delta_head.bias.view(
                self.num_gaits,
                self.residual_dim,
            )
            correction[3] = 0.2

        gait_ids = torch.full((self.batch_size,), 3, dtype=torch.long)
        _, residual_mean = model.distribution_params(self.obs, gait_ids=gait_ids)
        residual_mean.pow(2).mean().backward()
        correction_grad = model.actor.gait_residual_delta_head.bias.grad.view(
            self.num_gaits,
            self.residual_dim,
        )

        self.assertGreater(correction_grad[3].abs().sum().item(), 0.0)
        torch.testing.assert_close(correction_grad[:3], torch.zeros_like(correction_grad[:3]))


if __name__ == "__main__":
    unittest.main()
