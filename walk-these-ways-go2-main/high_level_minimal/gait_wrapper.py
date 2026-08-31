"""高层步态指令与冻结 WTW 底层策略之间的核心适配层。

一次高层步态指令共 9 维：前 4 维选择离散步态，后 5 维微调频率、支撑时长、
抬脚高度、站宽和身体俯仰。本类还构造 10 帧本体历史并计算统一物理奖励。
"""

import torch
import torch.nn.functional as F

from .reward import compute_unified_reward


class MinimalGaitWrapper:
    """将步态选择和参数修正转换为冻结 WTW 可执行的命令。"""

    def __init__(
        self,
        env,
        low_level_policy,
        high_level_dt=0.10,
        history_length=10,
        action_smoothing=0.6,
        record_reward_terms=True,
    ):
        """建立时间尺度、步态模板、参数范围和历史观测缓存。

        输入：
            env: 带底层观测历史的 Go2 Isaac Gym 环境。
            low_level_policy: 冻结 WTW 推理函数，输入底层观测字典并输出 12 维动作。
            high_level_dt: 一次高层环境步持续时间，默认 0.1 秒。
            history_length: 高层本体历史帧数，默认 10 帧。
            action_smoothing: 连续修正沿用旧值的比例；越大变化越平缓。
            record_reward_terms: 是否在 ``info`` 中保存逐项原始指标和分数。
        输出：
            构造函数无返回；创建步态模板、参数范围、状态缓存和奖励缓存。
        内部逻辑：
            根据 ``env.dt`` 算出一次高层步包含多少底层步，建立 4 种步态模板、
            5 维连续参数默认值与边界，并分配 ``[num_envs, 510]`` 历史缓存。
        作用：
            固定高层与 WTW 之间的接口约定，是指令、观测和奖励三条数据流的汇合点。
        """
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

        # WTW 用 phase、offset、bound 三个相位量编码四种步态，而不是直接接收步态编号。
        self.gait_templates = torch.tensor(
            (
                (0.0, 0.0, 0.0),  # pronking
                (0.5, 0.0, 0.0),  # trotting
                (0.0, 0.5, 0.0),  # bounding
                (0.0, 0.0, 0.5),  # pacing
            ),
            device=self.device,
        )
        # 每行依次为频率、支撑时长、抬脚高度、站宽和身体俯仰的默认值。
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

        # 单帧包含 42 维本体状态和当前 9 维高层步态指令；10 帧拼成 510 维历史。
        self.num_high_level_obs = 42 + self.num_high_level_actions
        self.num_high_level_obs_history = self.num_high_level_obs * history_length
        self.obs_history = torch.zeros(
            self.num_envs,
            self.num_high_level_obs_history,
            device=self.device,
        )

        self.high_level_command = torch.zeros(
            self.num_envs,
            self.num_high_level_actions,
            device=self.device,
        )
        self.default_command = torch.zeros_like(self.high_level_command)
        self.default_command[:, 1] = 1.0
        self.velocity_command = torch.zeros(self.num_envs, 3, device=self.device)
        self.previous_contacts = torch.zeros(
            self.num_envs,
            4,
            device=self.device,
            dtype=torch.bool,
        )
        self.low_level_obs = None
        self.last_reward_terms = {}

    def reset(self):
        """将机器人和高层缓存恢复为默认小跑、零修正状态。

        输入：无。
        输出：``[num_envs, 510]`` 的初始本体历史。
        内部逻辑：
            重置底层环境，将当前与上一高层步态指令都设为默认小跑，刷新足端接触并清空历史。
        作用：
            防止上一回合的动作、接触事件或历史状态泄漏到新回合。
        """
        self.low_level_obs = self.env.reset()
        self.high_level_command.copy_(self.default_command)
        self.previous_contacts = self._foot_contacts()
        self.obs_history.zero_()
        return self.get_observations()

    def set_velocity_command(self, vx, vy=0.0, yaw=0.0):
        """设置身体坐标系下的前进、横向和转向速度目标。

        输入：
            vx: 标量或 ``[num_envs]`` 前进速度，单位 m/s。
            vy: 标量或逐环境横向速度，默认 0。
            yaw: 标量或逐环境偏航角速度，默认 0，单位 rad/s。
        输出：无显式返回；原地更新高层缓存和底层 ``commands[:, :3]``。
        内部逻辑：将输入转换到仿真设备，并按 WTW 速度命令列写入。
        作用：保证高层任务速度不会被底层环境自己的命令重采样覆盖。
        """
        self.velocity_command[:, 0] = torch.as_tensor(vx, device=self.device)
        self.velocity_command[:, 1] = torch.as_tensor(vy, device=self.device)
        self.velocity_command[:, 2] = torch.as_tensor(yaw, device=self.device)
        self.env.commands[:, :3] = self.velocity_command

    def step(self, requested_command):
        """执行 0.1 秒高层步态指令，并汇总期间多个底层控制步的奖励。

        输入：
            requested_command: ``[num_envs, 9]``；前四维为步态分数，后五维为归一化修正。
        输出：
            observation: 更新后的 ``[num_envs, 510]`` 本体历史。
            reward: ``[num_envs]``，0.1 秒内底层步奖励的平均值。
            done: ``[num_envs]``，期间是否发生任一终止。
            info: 执行动作、终止类别和可选奖励分解。
        内部逻辑：
            裁剪动作，将步态离散化并平滑连续参数；随后循环若干次写 WTW 命令、
            调用冻结底层策略、推进仿真、计算奖励并累积终止状态；最后处理重置缓存。
        作用：
            把一次低频高层决策可靠地展开为多个 50 Hz 底层控制步。
        """
        requested_command = requested_command.to(self.device).clamp(-1.0, 1.0).detach()
        # 离散步态立即生效；连续修正使用指数平滑，避免参数瞬间跳变。
        self.high_level_command[:, : self.num_gaits] = self._one_hot_gait(requested_command)
        self.high_level_command[:, self.num_gaits :] = (
            self.action_smoothing * self.high_level_command[:, self.num_gaits :]
            + (1.0 - self.action_smoothing) * requested_command[:, self.num_gaits :]
        )

        reward_sum = torch.zeros(self.num_envs, device=self.device)
        done_any = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        edge_any = torch.zeros_like(done_any)
        timeout_any = torch.zeros_like(done_any)
        term_sums = {}
        info = {}

        # 高层步态指令保持不变，底层 WTW 在更高频率下连续输出 12 维关节动作。
        for _ in range(self.low_level_steps):
            self._write_low_level_commands()
            with torch.inference_mode():
                low_action = self.low_level_policy(self.low_level_obs)
            self.low_level_obs, _, done, info = self.env.step(low_action.to(self.device))
            reward_sum += self._reward(done)
            done_any |= done.bool()
            base = self._base_env()
            edge_any |= base.edge_reset_buf.bool()
            timeout_any |= base.time_out_buf.bool()
            if self.record_reward_terms:
                for name, value in self.last_reward_terms.items():
                    term_sums.setdefault(name, torch.zeros_like(value)).add_(value)

        info["executed_high_level_command"] = self.high_level_command.detach().clone()
        info["high_level_edge_resets"] = edge_any
        info["high_level_timeouts"] = timeout_any
        info["high_level_physical_dones"] = done_any & ~edge_any & ~timeout_any

        # 跌倒、越界或超时后，先恢复默认动作，防止旧动作污染新回合。
        if done_any.any():
            done_ids = done_any.nonzero(as_tuple=False).flatten()
            self.high_level_command[done_ids] = self.default_command[done_ids]

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
        """将最新本体帧压入定长滑动窗口，返回 510 维历史。

        输入：无；读取环境当前状态和当前高层步态指令。
        输出：``[num_envs, history_length × 51]`` 的分离张量，默认即 510 维。
        内部逻辑：左移旧历史一个 51 维帧位，再把 ``_proprioceptive_frame`` 写入末尾。
        作用：为学生编码器提供约一秒动态信息，使其能从响应过程推断隐藏物理条件。
        """
        frame = self._proprioceptive_frame()
        self.obs_history[:, :-self.num_high_level_obs] = self.obs_history[
            :, self.num_high_level_obs :
        ].clone()
        self.obs_history[:, -self.num_high_level_obs :] = frame
        return self.obs_history.detach()

    def _proprioceptive_frame(self):
        """构造一帧部署可获得的状态，不包含地形编号或仿真特权量。

        输入：无；读取当前机身、关节、足端接触、横向位置和高层步态指令。
        输出：``[num_envs, 51]`` 张量。
        内部逻辑：
            拼接 3 维速度误差、3 维线速度、3 维角速度、3 维重力投影、
            12 维关节位置误差、12 维关节速度、4 维接触、2 维横向偏移和 9 维步态指令。
        作用：
            构成可部署的本体感知帧；历史化后用于识别地形响应和机器人动力学变化。
        """
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
                self.high_level_command,
            ),
            dim=1,
        )

    def _one_hot_gait(self, action):
        """把前四维分数转换为严格的一位有效步态选择。

        输入：``[num_envs, >=4]`` 动作张量。
        输出：``[num_envs, 4]`` 浮点一位编码。
        内部逻辑：对前四列取 ``argmax``，再转为 one-hot。
        作用：确保 WTW 每次执行一个明确步态，而不是混合多个相位模板。
        """
        gait_ids = action[:, : self.num_gaits].argmax(dim=1)
        return F.one_hot(gait_ids, self.num_gaits).to(self.device, torch.float)

    def _mapped_action(self):
        """把 9 维归一化指令映射为 WTW 相位模板与五个物理参数。

        输入：无；使用当前 ``high_level_command``。
        输出：
            gait: ``[num_envs, 3]``，WTW 的 phase/offset/bound 相位编码。
            behavior: ``[num_envs, 5]``，实际频率、支撑时长、抬脚高度、站宽和俯仰。
        内部逻辑：
            先按离散步态选默认模板，再把后五维从 ``[-1,1]`` 映射到物理修正区间，
            与默认值相加，最后裁剪到低层允许的绝对边界。
        作用：
            将无量纲网络输出转换成物理可解释且不会超出 WTW 工作范围的命令。
        """
        selector = self._one_hot_gait(self.high_level_command)
        gait = selector @ self.gait_templates
        behavior = selector @ self.behavior_templates
        residual = self.high_level_command[:, self.num_gaits :].clamp(-1.0, 1.0)
        # 网络输出位于 [-1, 1]，先映射到各参数允许的修正区间。
        residual_unit = 0.5 * (residual + 1.0)
        delta = self.residual_ranges[:, 0] + (
            self.residual_ranges[:, 1] - self.residual_ranges[:, 0]
        ) * residual_unit
        behavior = (behavior + delta).maximum(self.behavior_lows).minimum(
            self.behavior_highs
        )
        return gait, behavior

    def _write_low_level_commands(self):
        """按 WTW 约定的 commands 列顺序写入速度、步态和连续参数。

        输入：无；读取速度缓存和 ``_mapped_action`` 结果。
        输出：无显式返回；原地更新底层 ``env.commands``。
        内部逻辑：
            将速度写入 0:3，频率和三维相位写入 4:8，其余行为参数写入固定列。
        作用：
            把高层步态指令准确翻译为原 WTW 策略训练时使用的命令格式。
        """
        gait, behavior = self._mapped_action()
        commands = self.env.commands
        commands[:, :3] = self.velocity_command
        commands[:, 4:8] = torch.cat((behavior[:, :1], gait), dim=1)
        commands[:, 8] = behavior[:, 1]
        commands[:, 9] = behavior[:, 2]
        commands[:, 10] = behavior[:, 4]
        commands[:, 11] = 0.0
        commands[:, 12] = behavior[:, 3]
        commands[:, 13] = 0.40

    def _reward(self, done):
        """从当前物理状态计算原始指标、归一化分数和统一加权奖励。

        输入：
            done: ``[num_envs]`` 当前底层步的终止标志。
        输出：
            ``[num_envs]`` 统一物理奖励。
        内部逻辑：
            先计算跟踪、姿态、侧漂、滑移、功率、冲击和擦碰等原始物理量，
            再调用 ``compute_unified_reward`` 完成归一化和固定权重聚合。
            若启用记录，还会保存原始量、归一化分数和总奖励。
        作用：
            用同一物理目标比较不同地形和步态，不直接奖励预设步态标签。
        """
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
        base = self._base_env()
        edge = base.edge_reset_buf.bool()
        timeout = base.time_out_buf.bool()

        reward, raw, scores = compute_unified_reward(
            velocity_reward=torch.exp(
                -(vx_error.square() + 0.25 * vy_error.square()) / 0.25
            ),
            yaw_reward=torch.exp(-yaw_error.square() / 0.10),
            orientation_penalty=self.env.projected_gravity[:, :2].square().sum(dim=1),
            lateral_velocity_penalty=self.env.base_lin_vel[:, 1].square(),
            lateral_position_penalty=(self._lateral_offset().abs() - 0.25)
            .clamp_min(0.0)
            .square(),
            contact_slip_penalty=(contact_float * foot_xy_speed_sq).sum(dim=1)
            / contact_count,
            mechanical_power=power,
            impact_velocity_rms=impact,
            scuffing_ratio=scuffing,
            fall_penalty=(done.bool() & ~edge & ~timeout).float(),
        )

        if self.record_reward_terms:
            self.last_reward_terms = raw
            self.last_reward_terms["weighted_metric_reward"] = reward
            for name, score in scores.items():
                self.last_reward_terms[f"score_{name}"] = score
        self.previous_contacts = contacts.detach().clone()
        return reward

    def _contact_safety(self, contacts):
        """按新触地事件计算冲击，并统计摆动脚过低造成的擦碰比例。

        输入：``[num_envs, 4]`` 布尔足端接触状态。
        输出：
            impact: ``[num_envs]``，新触地足端向下速度的均方根。
            scuffing: ``[num_envs]``，摆动脚低于 3.5 cm 的比例。
        内部逻辑：
            用当前与上一接触状态检测触地边沿；冲击只统计触地事件，擦碰则查询足端
            相对局部地面高度。
        作用：
            避免用全时间平均稀释短时落地冲击，也避免把接触占空比不同误当安全优势。
        """
        new_contacts = contacts & ~self.previous_contacts
        previous_velocity = self.env.prev_foot_velocities
        impact_velocity = (-previous_velocity[:, :, 2]).clamp_min(0.0)
        count = new_contacts.float().sum(dim=1).clamp_min(1.0)
        impact = (
            (new_contacts.float() * impact_velocity.square()).sum(dim=1) / count
        ).sqrt()

        positions = self.env.foot_positions
        clearance = positions[:, :, 2] - self._ground_height(positions)
        swing = (~contacts).float()
        scuffing = (
            swing * (clearance < 0.035).float()
        ).sum(dim=1) / swing.sum(dim=1).clamp_min(1.0)
        return impact, scuffing

    def _ground_height(self, foot_positions):
        """在高度场上查询足端下方地面高度，用于计算摆动间隙。

        输入：``[num_envs, 4, 3]`` 世界坐标足端位置。
        输出：``[num_envs, 4]`` 的局部地面高度，单位米。
        内部逻辑：
            将世界 XY 坐标换算为高度图索引，检查边界，并取邻近三个采样点的最小高度。
            地形创建时已启用高度图，因此这里直接读取底层高度数据。
        作用：
            在斜坡、粗糙地面和踏石上以真实局部地面为基准判断足端擦碰。
        """
        base = self._base_env()
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
        """返回教师可见、实机不可直接获得的 14 维通用物理量。

        组成：4 维地形统计、摩擦、载荷、3 维质心偏移、2 维推扰信息、
        身体高度和 2 维重力投影。部署时学生只能从本体历史中估计这些信息。

        输入：无；读取仿真器内部高度、动力学随机化和机器人状态。
        输出：``[num_envs, 14]``，各维已缩放或裁剪到大致 ``[-1,1]``。
        内部逻辑：
            从高度采样计算均值、标准差、极差和前后差；再拼接摩擦、载荷、
            质心偏移、推扰、身体高度和重力投影。
        作用：
            为训练教师提供比学生更直接的物理条件，使学生有明确的隐变量模仿目标。
        """
        base = self._base_env()
        heights = base.measured_heights
        middle = heights.shape[1] // 2
        terrain = torch.stack(
            (
                (heights.mean(dim=1) / 0.5).clamp(-1.0, 1.0),
                (heights.std(dim=1) / 0.3).clamp(0.0, 1.0),
                ((heights.max(dim=1).values - heights.min(dim=1).values) / 0.5)
                .clamp(0.0, 1.0),
                (heights[:, middle:].mean(dim=1) - heights[:, :middle].mean(dim=1))
                .div(0.3)
                .clamp(-1.0, 1.0),
            ),
            dim=1,
        )
        friction = (base.friction_coeffs[:, 0] / 1.5).clamp(0.0, 1.0).unsqueeze(1)
        payload = ((base.payloads + 1.0) / 4.0).clamp(0.0, 1.0).unsqueeze(1)
        com = (base.com_displacements[:, :3] / 0.1).clamp(-1.0, 1.0)
        axes = torch.as_tensor(
            base.cfg.domain_rand.push_axis_by_env,
            device=self.device,
            dtype=torch.float,
        )
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
        """逐层拆开包装器，取得最底层 Isaac Gym 环境。

        输入：无。
            输出：最底层的 Isaac Gym 环境对象。
            内部逻辑：当前固定结构只有 ``HistoryWrapper`` 一层，因此取其 ``env``。
        作用：访问高度图、重置缓存和环境原点等包装器未直接暴露的数据。
        """
        return self.env.env

    def _foot_contacts(self):
        """判断四只足端是否处于有效接触。

        输入：无；读取足端刚体的三维接触力。
        输出：``[num_envs, 4]`` 布尔张量。
        内部逻辑：选取四足竖直方向接触力，并与 1 N 阈值比较。
        作用：为本体观测、滑移归一化、触地冲击和擦碰计算提供统一接触判据。
        """
        return self.env.contact_forces[:, self.env.feet_indices, 2] > 1.0

    def _lateral_offset(self):
        """计算机器人相对各自地形中心线的横向偏移。

        输入：无；读取机器人世界 Y 坐标和逐环境原点。
        输出：``[num_envs]`` 米制有符号偏移张量。
            内部逻辑：用根状态 Y 减去对应并行环境原点 Y。
        作用：让不同网格位置的并行环境共享同一侧漂评价基准。
        """
        base = self._base_env()
        return base.root_states[: self.num_envs, 1] - base.env_origins[: self.num_envs, 1]

    def __getattr__(self, name):
        """将未定义属性转发给下一层环境。

        输入：当前对象找不到的属性名称。
        输出：底层 ``env`` 上的同名属性或方法。
        内部逻辑：直接调用 ``getattr(self.env, name)``；底层也不存在时自然抛出异常。
        作用：让上层代码可直接访问设备、速度、关节状态等底层字段，而无需重复写代理属性。
        """
        return getattr(self.env, name)
