import torch
import torch.nn.functional as F


class RolloutBuffer:
    def __init__(self, steps, num_envs, obs_dim, action_dim, privileged_dim, device):
        shape = (steps, num_envs)
        self.obs = torch.zeros(*shape, obs_dim, device=device)
        self.privileged = torch.zeros(*shape, privileged_dim, device=device)
        self.actions = torch.zeros(*shape, action_dim, device=device)
        self.log_probs = torch.zeros(*shape, device=device)
        self.rewards = torch.zeros(*shape, device=device)
        self.dones = torch.zeros(*shape, device=device)
        self.values = torch.zeros(*shape, device=device)
        self.returns = torch.zeros(*shape, device=device)
        self.advantages = torch.zeros(*shape, device=device)
        self.index = 0

    def add(self, obs, privileged, action, log_prob, reward, done, value):
        index = self.index
        self.obs[index].copy_(obs)
        self.privileged[index].copy_(privileged)
        self.actions[index].copy_(action)
        self.log_probs[index].copy_(log_prob)
        self.rewards[index].copy_(reward)
        self.dones[index].copy_(done.float())
        self.values[index].copy_(value)
        self.index += 1

    def finish(self, last_value, gamma, gae_lambda):
        gae = 0.0
        for step in reversed(range(self.rewards.shape[0])):
            next_value = last_value if step == self.rewards.shape[0] - 1 else self.values[step + 1]
            active = 1.0 - self.dones[step]
            delta = self.rewards[step] + gamma * next_value * active - self.values[step]
            gae = delta + gamma * gae_lambda * active * gae
            self.advantages[step] = gae
        self.returns.copy_(self.advantages + self.values)
        self.advantages.sub_(self.advantages.mean()).div_(self.advantages.std() + 1e-8)

    def flattened(self):
        return tuple(
            value.flatten(0, 1)
            for value in (
                self.obs,
                self.privileged,
                self.actions,
                self.log_probs,
                self.returns,
                self.advantages,
            )
        )


class PPO:
    def __init__(
        self,
        model,
        stage,
        learning_rate=3e-4,
        epochs=4,
        mini_batches=4,
        clip=0.2,
        entropy_coef=0.003,
        value_coef=0.5,
        adaptation_coef=0.1,
        physical_coef=0.1,
        residual_coef=0.01,
    ):
        self.model = model
        self.selector_only = stage == "gait"
        self.epochs = epochs
        self.mini_batches = mini_batches
        self.clip = clip
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.adaptation_coef = adaptation_coef
        self.physical_coef = physical_coef
        self.residual_coef = residual_coef
        if stage == "parameters":
            self.adaptation_coef = 0.0
            self.physical_coef = 0.0
        parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        self.optimizer = torch.optim.Adam(parameters, lr=learning_rate)

    def update(self, buffer):
        obs, privileged, actions, old_log_prob, returns, advantages = buffer.flattened()
        batch_size = obs.shape[0]
        mini_batch_size = max(1, batch_size // self.mini_batches)
        totals = {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "adaptation_loss": 0.0,
            "physical_loss": 0.0,
            "updates": 0,
        }

        for _ in range(self.epochs):
            permutation = torch.randperm(batch_size, device=obs.device)
            for start in range(0, batch_size, mini_batch_size):
                indices = permutation[start : start + mini_batch_size]
                new_log_prob, entropy, value = self.model.evaluate_actions(
                    obs[indices],
                    actions[indices],
                    self.selector_only,
                )
                ratio = torch.exp(new_log_prob - old_log_prob[indices])
                surrogate = torch.minimum(
                    ratio * advantages[indices],
                    torch.clamp(ratio, 1.0 - self.clip, 1.0 + self.clip)
                    * advantages[indices],
                )
                policy_loss = -surrogate.mean()
                value_loss = F.mse_loss(value, returns[indices])

                if self.adaptation_coef or self.physical_coef:
                    history = obs[indices, : self.model.base_obs_dim]
                    teacher = self.model.encode_teacher(privileged[indices])
                    student = self.model.encode_student(history)
                    adaptation_loss = F.mse_loss(student, teacher.detach())
                    physical_loss = F.mse_loss(
                        self.model.predict_physical_state(student),
                        privileged[indices],
                    )
                else:
                    adaptation_loss = torch.zeros((), device=obs.device)
                    physical_loss = torch.zeros((), device=obs.device)

                if self.selector_only:
                    residual_loss = torch.zeros((), device=obs.device)
                else:
                    gait_ids = torch.argmax(actions[indices, : self.model.num_gaits], dim=-1)
                    _, residual_mean = self.model.distribution_parameters(
                        obs[indices],
                        gait_ids,
                    )
                    residual_loss = residual_mean.square().mean()

                loss = (
                    policy_loss
                    + self.value_coef * value_loss
                    - self.entropy_coef * entropy.mean()
                    + self.adaptation_coef * adaptation_loss
                    + self.physical_coef * physical_loss
                    + self.residual_coef * residual_loss
                )
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

                totals["policy_loss"] += policy_loss.item()
                totals["value_loss"] += value_loss.item()
                totals["adaptation_loss"] += adaptation_loss.item()
                totals["physical_loss"] += physical_loss.item()
                totals["updates"] += 1

        count = totals.pop("updates")
        return {name: value / count for name, value in totals.items()}
