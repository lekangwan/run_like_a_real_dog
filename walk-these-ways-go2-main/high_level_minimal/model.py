"""教师、学生、步态选择器和连续步态参数网络。

训练时，教师从仿真特权物理量得到条件表征，学生从本体历史模仿该表征。
部署时只保留学生。步态选择器输出四种步态概率；连续参数网络读取当前状态
和已选步态，在默认模板附近输出五个修正量。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal


def mlp(input_dim, hidden_dims, output_dim):
    """构造以 ELU 为激活函数的多层感知机。

    输入：输入维度、各隐藏层宽度和输出维度。
    输出：最后一层不带激活函数的 ``nn.Sequential``。
    处理：依次添加 ``Linear + ELU``，最后添加线性输出层。
    作用：统一教师、学生、选择器、价值网络和辅助预测头的写法。
    """
    layers = []
    last_dim = input_dim
    for hidden_dim in hidden_dims:
        layers.extend((nn.Linear(last_dim, hidden_dim), nn.ELU()))
        last_dim = hidden_dim
    layers.append(nn.Linear(last_dim, output_dim))
    return nn.Sequential(*layers)


class ContinuousParameterNetwork(nn.Module):
    """根据机器人状态和候选步态计算五个连续参数修正量。"""

    def __init__(self, obs_dim, num_gaits=4, residual_dim=5):
        """创建共享状态骨干和一套带步态条件的参数网络。

        输入：完整策略状态维度、步态数和连续参数数。
        输出：构造函数无返回；建立状态特征网络和参数输出网络。
        处理：先把状态压缩为 256 维特征，再拼接步态的一位编码，输出五维修正。
        作用：不同步态可以产生不同参数，但共享同一套网络，避免四套专家网络膨胀。
        """
        super().__init__()
        self.num_gaits = num_gaits
        self.residual_dim = residual_dim
        self.backbone = mlp(obs_dim, (256,), 256)
        self.parameter_head = mlp(256 + num_gaits, (256,), residual_dim)
        self.zero_output()

    def all_candidates(self, obs):
        """并行计算每种候选步态的参数修正。

        输入：``[batch, obs_dim]`` 的完整策略状态。
        输出：``[batch, 4, 5]``，第二维依次对应四种步态。
        处理：状态特征复制四份，分别拼接四种步态编码，再经同一参数网络计算。
        作用：采样或选定步态后，只需从结果中取对应的一行参数。
        """
        features = self.backbone(obs)
        batch_size = obs.shape[0]
        gait_codes = torch.eye(
            self.num_gaits,
            device=obs.device,
            dtype=obs.dtype,
        ).unsqueeze(0).expand(batch_size, -1, -1)
        feature_grid = features.unsqueeze(1).expand(-1, self.num_gaits, -1)
        network_input = torch.cat((feature_grid, gait_codes), dim=-1)
        residuals = self.parameter_head(
            network_input.reshape(batch_size * self.num_gaits, -1)
        )
        return torch.tanh(
            residuals.reshape(batch_size, self.num_gaits, self.residual_dim)
        )

    def zero_output(self):
        """把参数网络初始化为零修正。

        输入与输出：无；原地修改输出层权重和偏置。
        处理：只清零最后一层，前面的状态特征网络保留正常初始化。
        作用：第二阶段开始时继续执行可靠的默认模板，而不是随机扰动步态参数。
        """
        output_layer = self.parameter_head[-1]
        nn.init.zeros_(output_layer.weight)
        nn.init.zeros_(output_layer.bias)


class HighLevelPolicy(nn.Module):
    """训练与部署共用的师生高层策略。"""

    def __init__(
        self,
        policy_obs_dim,
        base_obs_dim,
        num_gaits=4,
        residual_dim=5,
        privileged_dim=14,
        latent_dim=16,
    ):
        """组装完整高层网络。

        输入：部署观测维度、本体历史维度、步态数、参数数、特权量维度和隐变量维度。
        输出：构造函数无返回；创建教师、学生、物理预测、步态选择、连续参数和价值网络。
        处理：教师与学生产生同维隐变量；选择器只读目标速度、隐变量和预测物理量；
        连续参数网络及价值网络读取部署观测与隐变量的拼接结果。
        作用：实现训练时教师指导、部署时只靠本体历史的条件控制。
        """
        super().__init__()
        self.policy_obs_dim = policy_obs_dim
        self.base_obs_dim = base_obs_dim
        self.num_gaits = num_gaits
        self.residual_dim = residual_dim
        self.privileged_dim = privileged_dim
        self.latent_dim = latent_dim

        augmented_dim = policy_obs_dim + latent_dim
        self.teacher_encoder = mlp(privileged_dim, (128, 64), latent_dim)
        self.student_encoder = mlp(base_obs_dim, (256, 128), latent_dim)
        self.physical_decoder = mlp(latent_dim, (64,), privileged_dim)
        self.gait_selector = mlp(
            1 + latent_dim + privileged_dim,
            (256, 256),
            num_gaits,
        )
        self.parameter_network = ContinuousParameterNetwork(
            augmented_dim,
            num_gaits,
            residual_dim,
        )
        self.critic = mlp(augmented_dim, (256, 256), 1)

        self.log_std = nn.Parameter(torch.log(torch.ones(residual_dim) * 0.5))
        self.register_buffer(
            "residual_mask",
            torch.zeros(residual_dim),
            persistent=False,
        )

    def encode_teacher(self, privileged_obs):
        """把 14 维仿真特权物理量压缩为 16 维教师表征。

        输入：``[batch, 14]`` 特权物理量。输出：``[batch, 16]`` 隐变量。
        作用：为策略提供信息充分的训练条件，并作为学生的模仿目标。
        """
        return self.teacher_encoder(privileged_obs)

    def encode_student(self, history):
        """从十帧本体历史估计教师表征。

        输入：``[batch, 510]`` 本体历史。输出：``[batch, 16]`` 隐变量。
        作用：部署时替代无法直接获得的仿真特权信息。
        """
        return self.student_encoder(history)

    def predict_physical_state(self, latent):
        """从隐变量重建通用物理量。

        输入：``[batch,16]`` 隐变量。输出：``[batch,14]`` 物理量预测。
        作用：通过重建损失约束隐变量具有物理含义，而不是任意策略特征。
        """
        return self.physical_decoder(latent)

    def augment_observation(self, policy_obs, latent):
        """拼接部署观测和条件隐变量。

        输入：``[batch,511]`` 部署观测与 ``[batch,16]`` 隐变量。
        输出：``[batch,527]`` 完整策略状态。
        作用：教师与学生只替换条件表征，不改变后续策略接口。
        """
        return torch.cat((policy_obs, latent), dim=-1)

    def gait_logits(self, augmented_obs):
        """计算四种步态的未归一化分数。

        输入：``[batch,527]`` 完整策略状态。输出：``[batch,4]`` 分数。
        处理：从完整状态中取目标速度和隐变量，再拼接由隐变量预测的物理量。
        作用：强制步态选择依赖紧凑条件信息，避免原始历史直接淹没隐变量。
        """
        command = augmented_obs[:, self.base_obs_dim : self.base_obs_dim + 1]
        latent = augmented_obs[:, -self.latent_dim :]
        physical = self.predict_physical_state(latent).detach()
        return self.gait_selector(torch.cat((command, latent, physical), dim=-1))

    def distribution_parameters(self, augmented_obs, gait_ids=None):
        """返回步态分数和对应步态的连续参数均值。

        输入：完整策略状态，以及可选的 ``[batch]`` 步态索引。
        输出：``[batch,4]`` 步态分数和 ``[batch,5]`` 参数均值。
        处理：未给索引时使用当前最高分步态；随后从四套候选结果中按行取值。
        作用：统一训练回放和确定性部署使用的输出计算。
        """
        logits = self.gait_logits(augmented_obs)
        if gait_ids is None:
            gait_ids = torch.argmax(logits, dim=-1)
        candidates = self.parameter_network.all_candidates(augmented_obs)
        rows = torch.arange(augmented_obs.shape[0], device=augmented_obs.device)
        residuals = candidates[rows, gait_ids] * self.residual_mask
        return logits, residuals

    def set_stage(self, stage):
        """设置“先学步态、后调连续参数”的训练阶段。

        输入：``gait`` 或 ``parameters``。输出：无。
        处理：步态阶段关闭连续参数；参数阶段开放参数、零初始化其输出、减小探索，
        并冻结教师、学生、物理预测和步态选择器。
        作用：隔离两类变量，避免随机连续参数污染早期步态奖励归因。
        """
        if stage == "gait":
            self.residual_mask.zero_()
            for parameter in self.parameter_network.parameters():
                parameter.requires_grad_(False)
            self.log_std.requires_grad_(False)
            return
        if stage != "parameters":
            raise ValueError(f"Unknown training stage: {stage}")

        self.residual_mask.fill_(1.0)
        for parameter in self.parameter_network.parameters():
            parameter.requires_grad_(True)
        self.log_std.requires_grad_(True)
        self.parameter_network.zero_output()
        with torch.no_grad():
            self.log_std.fill_(torch.log(torch.tensor(0.1, device=self.log_std.device)))
        for module in (
            self.teacher_encoder,
            self.student_encoder,
            self.physical_decoder,
            self.gait_selector,
        ):
            for parameter in module.parameters():
                parameter.requires_grad_(False)

    def _distributions(self, obs, gait_ids):
        """构造离散步态分布和连续参数高斯分布。

        输入：完整状态和已执行步态索引。
        输出：``Categorical`` 与 ``Normal`` 两个概率分布。
        作用：保证采样和 PPO 重算使用完全相同的概率定义。
        """
        logits, residual_mean = self.distribution_parameters(obs, gait_ids)
        residual_std = torch.exp(self.log_std).expand_as(residual_mean)
        return Categorical(logits=logits), Normal(residual_mean, residual_std)

    def act(self, obs, selector_only):
        """训练时随机采样高层步态指令。

        输入：完整状态和是否只训练步态。
        输出：九维高层指令、联合对数概率和价值估计。
        处理：先采样步态；参数阶段再从该步态的高斯分布采样五维修正。
        作用：产生在线探索数据，并保留 PPO 更新所需的概率与价值。
        """
        logits = self.gait_logits(obs)
        gait_dist = Categorical(logits=logits)
        gait_ids = gait_dist.sample()
        gait_one_hot = F.one_hot(gait_ids, self.num_gaits).to(obs.dtype)
        log_prob = gait_dist.log_prob(gait_ids)

        if selector_only:
            residuals = torch.zeros(
                obs.shape[0], self.residual_dim, device=obs.device, dtype=obs.dtype
            )
        else:
            _, residual_mean = self.distribution_parameters(obs, gait_ids)
            residual_dist = Normal(
                residual_mean,
                torch.exp(self.log_std).expand_as(residual_mean),
            )
            residuals = residual_dist.sample().clamp(-1.0, 1.0)
            log_prob += residual_dist.log_prob(residuals).sum(dim=-1)

        command = torch.cat((gait_one_hot, residuals), dim=-1)
        return command, log_prob, self.critic(obs).squeeze(-1)

    def evaluate_actions(self, obs, commands, selector_only):
        """为 PPO 重新计算已执行指令的概率、熵和价值。

        输入：完整状态、缓存的九维高层指令和阶段标志。
        输出：新策略对数概率、熵和价值估计。
        处理：从前四维恢复步态索引；参数阶段再计入后五维的高斯概率与熵。
        作用：构造 PPO 新旧策略概率比和熵正则。
        """
        gait_ids = torch.argmax(commands[:, : self.num_gaits], dim=-1)
        gait_dist, residual_dist = self._distributions(obs, gait_ids)
        log_prob = gait_dist.log_prob(gait_ids)
        entropy = gait_dist.entropy()
        if not selector_only:
            residuals = commands[:, self.num_gaits :]
            log_prob += residual_dist.log_prob(residuals).sum(dim=-1)
            entropy += residual_dist.entropy().sum(dim=-1)
        return log_prob, entropy, self.critic(obs).squeeze(-1)

    def act_student(self, policy_obs, selector_only):
        """部署时只用学生网络生成确定性高层指令。

        输入：部署观测和是否关闭连续参数。
        输出：``[batch,9]`` 高层步态指令。
        处理：学生编码历史，选择最高分步态，并使用对应参数均值。
        作用：这是最终不依赖仿真特权信息的部署路径。
        """
        latent = self.encode_student(policy_obs[:, : self.base_obs_dim])
        augmented_obs = self.augment_observation(policy_obs, latent)
        logits = self.gait_logits(augmented_obs)
        gait_ids = torch.argmax(logits, dim=-1)
        gait_one_hot = F.one_hot(gait_ids, self.num_gaits).to(policy_obs.dtype)
        _, residuals = self.distribution_parameters(augmented_obs, gait_ids)
        if selector_only:
            residuals.zero_()
        return torch.cat((gait_one_hot, residuals), dim=-1)
