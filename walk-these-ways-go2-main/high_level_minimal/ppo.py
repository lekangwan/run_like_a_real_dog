"""最小化实现的 PPO 轨迹缓存与参数更新。

除标准 PPO 策略/价值损失外，第一阶段还包含学生模仿教师和物理状态重建损失；
第二阶段关闭这两项，只优化连续修正并用小权重约束其不要远离零点。

PPO 文献把策略输出统称为 ``action``。在本文件中它指 9 维高层步态指令，
不是 WTW 输出给关节控制器的 12 维底层动作。
"""

import torch
import torch.nn.functional as F


class RolloutBuffer:
    """保存一批按“高层决策”组织的在线采样数据。"""

    def __init__(self, steps, num_envs, obs_dim, action_dim, privileged_dim, device):
        """预分配在线轨迹缓存。

        输入：高层决策步数、并行环境数、观测/动作/特权维度和设备。
        输出：构造函数无返回；创建形状 ``[steps,num_envs,...]`` 的全部缓存张量。
        内部逻辑：分别分配状态、特权量、动作、旧概率、奖励、终止、价值、回报和优势。
        作用：避免采样过程中频繁动态分配，并保持所有 PPO 数据严格按时间对齐。
        """
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
        """写入一个高层决策时刻的所有并行环境数据。

        输入：一批并行环境的状态、特权量、动作、旧概率、整段奖励、终止和价值。
        输出：无显式返回；写入当前 ``index`` 并将索引加一。
        内部逻辑：使用 ``copy_`` 写入预分配张量，避免保留原计算图。
        作用：把一次持续步态选项的起点信息和整段结果绑定为一个 PPO 转移。
        """
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
        """逆序计算广义优势估计，并对优势做标准化。

        输入：末状态价值、按决策周期换算后的折扣因子 ``gamma`` 和 GAE 系数。
        输出：无显式返回；填充 ``advantages`` 与 ``returns``。
        内部逻辑：从最后一步向前递推 TD 残差；终止处切断价值传播；最后计算回报并标准化优势。
        作用：把延迟奖励转换成每个动作的相对好坏信号，降低策略梯度方差。
        """
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
        """把时间和环境两个轴展平为 PPO 可随机分批的样本。

        输入：无。
        输出：观测、特权量、动作、旧概率、回报和优势六个展平张量组成的元组。
        内部逻辑：对每个张量执行 ``flatten(0,1)``，保留特征维。
        作用：让不同时间、不同机器人样本可以统一随机打乱并组成小批量。
        """
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
    """联合优化策略、价值、师生适应和物理状态预测。"""

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
        """根据训练阶段选择有效损失，并创建优化器。

        输入：模型、阶段、学习率、更新轮数、批次数、裁剪范围以及各损失系数。
        输出：构造函数无返回；保存超参数并创建只包含可训练参数的 Adam 优化器。
        内部逻辑：参数阶段关闭师生和物理重建损失，因为这些网络已被冻结。
        作用：使同一 PPO 实现能够支持“先学步态、后调参数”的两阶段课程。
        """
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
        """对一批在线数据执行多轮裁剪 PPO 更新。

        输入：已经完成回报与优势计算的 ``RolloutBuffer``。
        输出：字典，包含平均策略、价值、学生适应和物理预测损失。
        内部逻辑：
            展平并多轮随机分批；重算动作概率与价值；构造裁剪策略目标和价值损失；
            第一阶段增加学生模仿教师及物理重建，第二阶段增加连续均值零点正则；
            合并损失、裁剪梯度并更新参数，最后返回各项平均值。
        作用：利用在线轨迹稳定更新混合离散-连续策略及其条件表示。
        """
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
                # 新旧动作概率之比超过裁剪范围时，不允许继续放大策略更新。
                ratio = torch.exp(new_log_prob - old_log_prob[indices])
                surrogate = torch.minimum(
                    ratio * advantages[indices],
                    torch.clamp(ratio, 1.0 - self.clip, 1.0 + self.clip)
                    * advantages[indices],
                )
                policy_loss = -surrogate.mean()
                value_loss = F.mse_loss(value, returns[indices])

                # 第一阶段让学生隐变量接近教师，并能重建通用物理状态。
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

                # 第二阶段用零点正则限制连续参数只做小幅修正。
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
