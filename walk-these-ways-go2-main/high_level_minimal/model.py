import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal


def mlp(input_dim, hidden_dims, output_dim):
    layers = []
    last_dim = input_dim
    for hidden_dim in hidden_dims:
        layers.extend((nn.Linear(last_dim, hidden_dim), nn.ELU()))
        last_dim = hidden_dim
    layers.append(nn.Linear(last_dim, output_dim))
    return nn.Sequential(*layers)


class HybridActor(nn.Module):
    """Shared state backbone with discrete gait and continuous residual outputs."""

    def __init__(self, obs_dim, num_gaits=4, residual_dim=5, hidden_dims=(256, 256)):
        super().__init__()
        layers = []
        last_dim = obs_dim
        for hidden_dim in hidden_dims:
            layers.extend((nn.Linear(last_dim, hidden_dim), nn.ELU()))
            last_dim = hidden_dim
        self.backbone = nn.Sequential(*layers)
        self.gait_head = nn.Linear(last_dim, num_gaits)
        self.residual_head = nn.Linear(last_dim, residual_dim)
        self.num_gaits = num_gaits
        self.residual_dim = residual_dim

        # Stage two uses one shared network conditioned on the selected gait.
        self.gait_input_residual_network = nn.Sequential(
            nn.Linear(last_dim + num_gaits, last_dim),
            nn.ELU(),
            nn.Linear(last_dim, residual_dim),
        )
        nn.init.zeros_(self.gait_input_residual_network[-1].weight)
        nn.init.zeros_(self.gait_input_residual_network[-1].bias)

    def parameters_for_all_gaits(self, obs, use_gait_input):
        features = self.backbone(obs)
        gait_logits = self.gait_head(features)
        if not use_gait_input:
            return gait_logits, torch.tanh(self.residual_head(features))

        batch_size = features.shape[0]
        gait_codes = torch.eye(
            self.num_gaits,
            device=features.device,
            dtype=features.dtype,
        ).unsqueeze(0).expand(batch_size, -1, -1)
        feature_grid = features.unsqueeze(1).expand(-1, self.num_gaits, -1)
        network_input = torch.cat((feature_grid, gait_codes), dim=-1)
        residuals = self.gait_input_residual_network(
            network_input.reshape(batch_size * self.num_gaits, -1)
        ).reshape(batch_size, self.num_gaits, self.residual_dim)
        return gait_logits, torch.tanh(residuals)

    def zero_residual_output(self):
        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)
        nn.init.zeros_(self.gait_input_residual_network[-1].weight)
        nn.init.zeros_(self.gait_input_residual_network[-1].bias)


class HighLevelPolicy(nn.Module):
    """Teacher-student high-level policy used by both training and deployment."""

    def __init__(
        self,
        policy_obs_dim,
        base_obs_dim,
        num_gaits=4,
        residual_dim=5,
        privileged_dim=14,
        latent_dim=16,
        use_gait_input_residuals=False,
    ):
        super().__init__()
        self.policy_obs_dim = policy_obs_dim
        self.base_obs_dim = base_obs_dim
        self.num_gaits = num_gaits
        self.residual_dim = residual_dim
        self.latent_dim = latent_dim
        self.use_gait_input_residuals = use_gait_input_residuals

        augmented_dim = policy_obs_dim + latent_dim
        self.actor = HybridActor(augmented_dim, num_gaits, residual_dim)
        self.critic = mlp(augmented_dim, (256, 256), 1)

        self.terrain_encoder = mlp(privileged_dim, (128, 64), latent_dim)
        self.adaptation_module = mlp(base_obs_dim, (256, 128), latent_dim)
        self.physical_state_head = mlp(latent_dim, (64,), privileged_dim)

        # The selector intentionally sees only command, student/teacher latent,
        # and the physical state predicted from that latent.
        self.latent_cmd_selector = mlp(1 + latent_dim + privileged_dim, (256, 256), num_gaits)
        self.log_std = nn.Parameter(torch.log(torch.ones(residual_dim) * 0.5))
        self.register_buffer("residual_action_mask", torch.zeros(residual_dim), persistent=False)

    def encode_teacher(self, privileged_obs):
        return self.terrain_encoder(privileged_obs)

    def encode_student(self, history):
        return self.adaptation_module(history)

    def predict_physical_state(self, latent):
        return self.physical_state_head(latent)

    def augment_observation(self, policy_obs, latent):
        return torch.cat((policy_obs, latent), dim=-1)

    def distribution_parameters(self, augmented_obs, gait_ids=None):
        gait_logits_unused, residuals = self.actor.parameters_for_all_gaits(
            augmented_obs,
            self.use_gait_input_residuals,
        )
        del gait_logits_unused

        command = augmented_obs[:, self.base_obs_dim : self.base_obs_dim + 1]
        latent = augmented_obs[:, -self.latent_dim :]
        physical = self.predict_physical_state(latent).detach()
        gait_logits = self.latent_cmd_selector(torch.cat((command, latent, physical), dim=-1))

        if residuals.ndim == 3:
            if gait_ids is None:
                gait_ids = torch.argmax(gait_logits, dim=-1)
            rows = torch.arange(residuals.shape[0], device=residuals.device)
            residuals = residuals[rows, gait_ids]
        residuals = residuals * self.residual_action_mask
        return gait_logits, residuals

    def set_stage(self, stage):
        if stage == "gait":
            self.residual_action_mask.zero_()
            self.use_gait_input_residuals = False
            return
        if stage != "parameters":
            raise ValueError(f"Unknown training stage: {stage}")

        self.residual_action_mask.fill_(1.0)
        self.use_gait_input_residuals = True
        self.actor.zero_residual_output()
        with torch.no_grad():
            self.log_std.fill_(torch.log(torch.tensor(0.1, device=self.log_std.device)))

        # Stage two changes only the continuous parameter network and critic.
        for module in (
            self.latent_cmd_selector,
            self.terrain_encoder,
            self.adaptation_module,
            self.physical_state_head,
        ):
            for parameter in module.parameters():
                parameter.requires_grad_(False)

    def _distributions(self, obs, gait_ids=None):
        gait_logits, residual_mean = self.distribution_parameters(obs, gait_ids)
        return (
            Categorical(logits=gait_logits),
            Normal(residual_mean, torch.exp(self.log_std).expand_as(residual_mean)),
        )

    def act(self, obs, selector_only):
        gait_logits, residual_mean_all = self.actor.parameters_for_all_gaits(
            obs,
            self.use_gait_input_residuals,
        )
        del gait_logits

        command = obs[:, self.base_obs_dim : self.base_obs_dim + 1]
        latent = obs[:, -self.latent_dim :]
        physical = self.predict_physical_state(latent).detach()
        gait_dist = Categorical(
            logits=self.latent_cmd_selector(torch.cat((command, latent, physical), dim=-1))
        )
        gait_id = gait_dist.sample()
        gait_one_hot = F.one_hot(gait_id, self.num_gaits).to(dtype=obs.dtype)

        if selector_only:
            residual = torch.zeros(
                obs.shape[0], self.residual_dim, device=obs.device, dtype=obs.dtype
            )
            log_prob = gait_dist.log_prob(gait_id)
        else:
            if residual_mean_all.ndim == 3:
                rows = torch.arange(obs.shape[0], device=obs.device)
                residual_mean = residual_mean_all[rows, gait_id]
            else:
                residual_mean = residual_mean_all
            residual_mean = residual_mean * self.residual_action_mask
            residual_dist = Normal(
                residual_mean,
                torch.exp(self.log_std).expand_as(residual_mean),
            )
            raw_residual = torch.clamp(residual_dist.sample(), -1.0, 1.0)
            residual = raw_residual * self.residual_action_mask
            log_prob = gait_dist.log_prob(gait_id)
            log_prob += (
                residual_dist.log_prob(raw_residual) * self.residual_action_mask
            ).sum(dim=-1)

        action = torch.cat((gait_one_hot, residual), dim=-1)
        value = self.critic(obs).squeeze(-1)
        return action, log_prob, value

    def evaluate_actions(self, obs, actions, selector_only):
        gait_ids = torch.argmax(actions[:, : self.num_gaits], dim=-1)
        gait_dist, residual_dist = self._distributions(obs, gait_ids)
        log_prob = gait_dist.log_prob(gait_ids)
        entropy = gait_dist.entropy()
        if not selector_only:
            residual = actions[:, self.num_gaits :]
            log_prob += (
                residual_dist.log_prob(residual) * self.residual_action_mask
            ).sum(dim=-1)
            entropy += (
                residual_dist.entropy() * self.residual_action_mask
            ).sum(dim=-1)
        return log_prob, entropy, self.critic(obs).squeeze(-1)

    def act_student(self, policy_obs, selector_only):
        latent = self.encode_student(policy_obs[:, : self.base_obs_dim])
        augmented = self.augment_observation(policy_obs, latent)
        gait_logits, residual = self.distribution_parameters(augmented)
        gait_ids = torch.argmax(gait_logits, dim=-1)
        gait_one_hot = F.one_hot(gait_ids, self.num_gaits).to(dtype=policy_obs.dtype)
        if selector_only:
            residual.zero_()
        return torch.cat((gait_one_hot, residual), dim=-1)
