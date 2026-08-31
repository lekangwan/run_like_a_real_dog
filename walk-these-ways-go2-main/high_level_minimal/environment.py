"""把底层仿真、任务分配和高层步态包装器组合为统一环境接口。

模块作用：
    向训练、评测和录像代码提供同一种 ``reset()/step()`` 接口，隐藏多层包装器细节。

主要输入：
    * ``TaskSpec`` 列表、底层 WTW 运行目录和并行环境数；
    * 可选的可视化开关、地形尺寸与录像分辨率；
    * 每一步来自高层策略的 9 维步态指令。

内部处理：
    1. 在 CPU 上预分配任务，用它创建对应地形；
    2. 加载冻结 WTW，并套上 ``MinimalGaitWrapper``；
    3. 在 GPU 上建立任务和速度范围张量；
    4. 每个回合从任务速度范围重新采样目标速度；
    5. 把 510 维本体历史和 1 维目标速度拼成 511 维部署观测；
    6. 训练时额外提供清理后的 14 维仿真特权物理量，但不拼入部署观测。

主要输出：
    * ``reset`` -> 511 维策略观测；
    * ``step`` -> 下一观测、统一物理奖励、结束标志和诊断信息；
    * ``privileged_observation`` -> 仅教师训练使用的 14 维物理量。

在整体中的作用：
    它是 PPO 与真实仿真之间的边界，保证训练和评测共享完全相同的观测、动作和奖励口径。
"""

import torch

from .config import (
    EDGE_RESET_MARGIN,
    TELEPORT_THRESHOLD,
    TERRAIN_LENGTH,
    TERRAIN_WIDTH,
)
from .gait_wrapper import MinimalGaitWrapper
from .low_level import create_environment, load_policy
from .tasks import build_assignment


def sample_uniform(low, high):
    """为每个环境独立地在给定速度区间内采样。

    输入：
        low/high: 形状相同的逐环境下界和上界张量。
    输出：
        同形状张量，每个元素均匀分布于对应的 ``[low, high]``。
    内部逻辑：
        生成 ``[0,1)`` 随机数并进行线性缩放。
    作用：
        让同一任务覆盖连续速度，而不是只记忆少数离散速度点。
    """
    return low + torch.rand_like(low) * (high - low)


def clean_privileged_observation(observation):
    """删除直接暴露推扰类别的标志，仅保留通用仿真物理量。

    输入：
        observation: 形状 ``[num_envs, 14]`` 的教师特权物理量。
    输出：
        同形状副本，其中第 9、10 维推扰是否激活及推扰轴被置零。
    内部逻辑：
        先克隆，避免修改环境原始缓存，再清零会泄露任务类别的两维。
    作用：
        教师可以利用通用物理状态，但不能通过显式任务标志把地形答案直接传给学生。
    """
    cleaned = observation.clone()
    cleaned[:, 9:11] = 0.0
    return cleaned


class HighLevelEnvironment:
    """不向策略提供任务编号的多地形高层环境。"""

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
        """创建底层环境，并为每个机器人绑定任务与速度范围。

        输入：
            specs: 训练或评测任务配置列表。
            low_level_run: 冻结 WTW 的运行目录。
            num_envs: 并行机器人数量。
            render: 是否显示 Isaac Gym 窗口。
            terrain_length/terrain_width: 每个环境的地形尺寸。
            recording_width/recording_height: 可选相机分辨率。
        输出：
            构造函数无返回；初始化环境、任务分配、观测维度和逐环境速度命令。
        内部逻辑：
            先在 CPU 生成地形分配来创建底层环境，再在实际设备重建张量分配；
            随后包裹 ``MinimalGaitWrapper`` 并加载冻结底层策略。
        作用：
            向训练器提供简洁的 ``reset/step`` 接口，隐藏多层底层包装细节。
        """
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

        self.num_envs = num_envs
        self.num_gaits = self.env.num_gaits
        self.action_dim = self.env.num_high_level_actions
        self.base_obs_dim = self.env.num_high_level_obs_history
        # 部署观测 = 10 帧本体历史 + 1 维目标前进速度，没有地形编号。
        self.policy_obs_dim = self.base_obs_dim + 1
        self.vx_command = sample_uniform(
            self.assignment.vx_lows,
            self.assignment.vx_highs,
        )

    def _policy_observation(self, history):
        """将目标速度追加到本体历史末尾，形成高层策略输入。

        输入：
            history: ``[num_envs, 510]`` 的十帧本体历史。
        输出：
            ``[num_envs, 511]`` 张量，最后一维为当前目标前进速度。
        内部逻辑：
            将 ``command_vx`` 扩展为列向量后与历史拼接。
        作用：
            让策略区分“机器人当前状态相同但目标速度不同”的控制需求。
        """
        return torch.cat((history, self.command_vx()[:, None]), dim=-1)

    def reset(self):
        """重置仿真，并立即恢复当前环境对应的速度命令。

        输入：无。
        输出：``[num_envs, 511]`` 的初始高层策略观测。
        内部逻辑：
            重置步态包装器，写入已有逐环境速度命令，再拼接目标速度。
        作用：
            保证底层随机重置命令不会覆盖高层任务指定的速度。
        """
        history = self.env.reset()
        self.env.set_velocity_command(self.vx_command, 0.0, 0.0)
        return self._policy_observation(history)

    def step(self, command):
        """执行一次高层步态指令；机器人重置后为它重新采样目标速度。

        输入：
            command: ``[num_envs, 9]`` 的高层步态指令，前四维为步态，后五维为参数修正。
        输出：
            ``(observation, reward, done, info)``；观测为 511 维，奖励和终止为逐环境张量。
        内部逻辑：
            调用步态包装器执行 0.1 秒；对已终止环境重新采样速度并覆盖底层命令。
        作用：
            维持连续训练分布，并统一高层训练器与评测器的交互协议。
        """
        history, reward, done, info = self.env.step(command)
        done_ids = done.nonzero(as_tuple=False).flatten()
        if done_ids.numel():
            self.vx_command[done_ids] = sample_uniform(
                self.assignment.vx_lows[done_ids],
                self.assignment.vx_highs[done_ids],
            )
        self.env.set_velocity_command(self.vx_command, 0.0, 0.0)
        return self._policy_observation(history), reward, done, info

    def base_history(self):
        """返回学生网络可见的本体感知历史。

        输入：无。
        输出：``[num_envs, 510]`` 的只读式分离张量。
        内部逻辑：从步态包装器取得滑动历史并 ``detach``。
        作用：为学生编码器提供部署时真实可获得的信息。
        """
        return self.env.obs_history.detach()

    def privileged_observation(self):
        """返回仅训练教师使用的 14 维通用仿真物理量。

        输入：无。
        输出：``[num_envs, 14]`` 的清理后特权观测。
        内部逻辑：从步态包装器提取仿真量，再调用 ``clean_privileged_observation`` 清除任务提示。
        作用：训练教师形成物理条件表示，同时限制其不能直接读取任务编号。
        """
        return clean_privileged_observation(
            self.env.get_high_level_privileged_obs()
        )

    def command_vx(self):
        """读取逐环境目标前进速度。

        输入：无；读取步态包装器维护的 WTW 命令矩阵。
        输出：``[num_envs]`` 张量，单位 m/s。
        内部逻辑：返回 ``commands`` 第 0 列，不创建副本。
        作用：用于构造策略观测，并与实测速度计算统一跟踪误差。
        """
        return self.env.commands[:, 0]

    def measured_vx(self):
        """读取逐环境身体坐标系实测前进速度。

        输入：无；读取底层环境最新机身线速度。
        输出：``[num_envs]`` 张量，单位 m/s。
        内部逻辑：返回 ``base_lin_vel`` 的前进轴分量。
        作用：训练日志和独立评测使用它计算速度绝对误差。
        """
        return self.env.base_lin_vel[:, 0]
