"""恢复冻结的 WTW 底层策略并创建 Isaac Gym 环境。

模块作用：
    隔离“高层研究代码”和“底层运动控制基础设施”。高层无需了解 Isaac Gym 的完整创建细节。

主要输入：
    * 底层训练标签或具体运行目录；
    * 并行环境数、各环境地形条件和推扰方向；
    * 是否显示窗口、地形长宽、越界阈值和录像分辨率。

内部处理：
    1. 加载底层 ``body_latest.jit`` 与 ``adaptation_module_latest.jit``；
    2. 恢复底层训练配置，保证观测维度、关节控制方式与检查点一致；
    3. 关闭无关随机化并清除存档中残留的特定地形参数；
    4. 创建一行多列的混合地形，每个机器人位于独立地形块；
    5. 用原 ``HistoryWrapper`` 保存底层 30 帧、50 Hz 的观测历史。

主要输出：
    * ``load_policy`` 返回冻结推理函数：底层历史字典 -> 12 维关节动作；
    * ``create_environment`` 返回带底层历史的 Isaac Gym 环境。

在整体中的作用：
    接收上游步态包装器写入的 WTW 命令，以 50 Hz 生成关节目标，并由 200 Hz 仿真执行。
    本模块不训练底层策略，也不决定高层步态。
"""

from pathlib import Path
import glob
import pickle

import torch

from go2_gym import MINI_GYM_ROOT_DIR
from go2_gym.envs.base.legged_robot_config import Cfg
from go2_gym.envs.go2.go2_config import config_go2
from go2_gym.envs.go2.velocity_tracking import VelocityTrackingEasyEnv
from go2_gym.envs.wrappers.history_wrapper import HistoryWrapper


def find_run(label, run_index=0):
    """按照训练标签寻找底层 WTW 运行目录。

    输入：
        label: ``runs/`` 下的底层实验标签路径。
        run_index: 排序后的运行目录索引，默认为第一个。
    输出：
        ``Path``，指向包含参数和底层检查点的具体运行目录。
    内部逻辑：
        使用通配符列举并排序目录；没有候选时立即报错。
    作用：
        将稳定的实验标签解析为高层环境真正需要加载的底层模型位置。
    """
    paths = sorted(glob.glob(str(Path(MINI_GYM_ROOT_DIR) / "runs" / label / "*")))
    if not paths:
        raise FileNotFoundError(f"No low-level run found for {label}")
    return Path(paths[run_index])


def load_policy(run_dir):
    """加载底层身体网络与底层适应模块，并组合为冻结推理函数。

    输入：
        run_dir: 底层 WTW 运行目录，必须包含两个 TorchScript 文件。
    输出：
        可调用的 ``policy(observation, info=None)``；输入底层观测字典，输出
        形状 ``[num_envs, 12]`` 的关节动作。
    内部逻辑：
        分别加载身体网络和底层历史适应模块，切换为评测模式，再由内部闭包串联二者。
    作用：
        向高层隐藏底层网络细节，并确保底层策略始终冻结、只做推理。
    """
    body = torch.jit.load(str(run_dir / "checkpoints/body_latest.jit"))
    adaptation = torch.jit.load(str(run_dir / "checkpoints/adaptation_module_latest.jit"))
    body.eval()
    adaptation.eval()

    def policy(observation, info=None):
        """根据底层历史观测生成 12 维关节动作。

        输入：
            observation: HistoryWrapper 返回的字典，主要使用 ``obs_history``。
            info: 为兼容策略接口保留，本实现不使用。
        输出：
            底层身体网络输出的 12 维归一化关节动作。
        内部逻辑：
            将历史复制到 CPU，先由底层适应模块生成隐变量，再与历史拼接送入身体网络。
        作用：
            把高层步态命令最终落实为可交给执行器网络的关节目标。
        """
        del info
        with torch.inference_mode():
            history = observation["obs_history"].detach().cpu()
            latent = adaptation(history)
            return body(torch.cat((history, latent), dim=-1))

    return policy


def _restore_training_config(run_dir):
    """恢复底层训练时的配置，保证网络与环境观测维度一致。

    输入：
        run_dir: 包含 ``parameters.pkl`` 的底层运行目录。
    输出：
        无显式返回；原地修改全局 ``Cfg``。
    内部逻辑：
        先载入 Go2 默认配置，再逐节覆盖存档中存在的字段，同时兼容字典和属性式配置。
    作用：
        复现底层训练时的观测、动作、控制和仿真设置，避免检查点维度不匹配。
    """
    config_go2(Cfg)
    with open(run_dir / "parameters.pkl", "rb") as file:
        saved = pickle.load(file)["Cfg"]
    for section_name, values in saved.items():
        if not hasattr(Cfg, section_name):
            continue
        section = getattr(Cfg, section_name)
        for name, value in values.items():
            if isinstance(section, dict):
                section[name] = value
            else:
                setattr(section, name, value)


def _disable_randomization():
    """关闭与当前条件无关的随机化，避免评测结果被额外变量污染。

    输入：
        无；操作全局 ``Cfg.domain_rand``。
    输出：
        无显式返回。
    内部逻辑：
        将摩擦、重力、电机、质量等随机化开关置为假，并清空定向推扰参数。
    作用：
        先建立可重复的环境基线，再由任务分配显式开启真正需要的推扰。
    """
    fields = (
        "push_robots",
        "randomize_friction",
        "randomize_gravity",
        "randomize_restitution",
        "randomize_motor_offset",
        "randomize_motor_strength",
        "randomize_friction_indep",
        "randomize_ground_friction",
        "randomize_base_mass",
        "randomize_Kd_factor",
        "randomize_Kp_factor",
        "randomize_joint_friction",
        "randomize_com_displacement",
        "randomize_rigids_after_start",
    )
    for name in fields:
        if hasattr(Cfg.domain_rand, name):
            setattr(Cfg.domain_rand, name, False)
    for name in ("push_vel_xy", "push_vel_xyz", "push_axis"):
        if hasattr(Cfg.domain_rand, name):
            setattr(Cfg.domain_rand, name, None)


def _reset_mixed_terrain_baseline():
    """清除底层存档中可能残留的特定地形参数。

    输入：
        无；操作全局 ``Cfg.terrain``。
    输出：
        无显式返回。
    内部逻辑：
        恢复标准摩擦和零噪声，清空台阶、踏石和斜坡等专用参数。
    作用：
        防止上一次环境或底层训练配置残留，保证后续地形只由本次任务条件决定。
    """
    Cfg.terrain.static_friction = 1.0
    Cfg.terrain.dynamic_friction = 1.0
    Cfg.terrain.restitution = 0.0
    Cfg.terrain.terrain_noise_magnitude = 0.0
    Cfg.terrain.terrain_proportions = [0, 0, 0, 0, 0, 0, 0, 0, 1.0]
    Cfg.terrain.x_init_range = 0.0
    Cfg.terrain.y_init_range = 0.0
    Cfg.terrain.stair_step_height = None
    Cfg.terrain.discrete_obstacles_height = None
    Cfg.terrain.stepping_stones_size = None
    Cfg.terrain.stone_distance = None
    Cfg.terrain.stepping_stones_platform_size = 4.0
    Cfg.terrain.stepping_stones_max_height = 0.0
    Cfg.terrain.stepping_stones_depth = -10.0
    Cfg.terrain.ramp_slope = None


def create_environment(
    run_dir,
    num_envs,
    conditions,
    push_axes,
    render,
    terrain_length,
    terrain_width,
    edge_reset_margin,
    teleport_threshold,
    recording_width=None,
    recording_height=None,
):
    """创建一行多列地形，每个并行环境占据一个独立地形块。

    输入：
        run_dir: 底层配置与检查点目录。
        num_envs: 并行环境数量。
        conditions: 长度为 ``num_envs`` 的逐环境地形名称。
        push_axes: 逐环境推扰轴，负值表示不推扰。
        render: 是否创建可视窗口。
        terrain_length/terrain_width: 单个地形块尺寸，单位米。
        edge_reset_margin/teleport_threshold: 边界重置相关阈值。
        recording_width/recording_height: 可选录像相机分辨率。
    输出：
        ``HistoryWrapper(VelocityTrackingEasyEnv)``，即带底层观测历史的 Isaac Gym 环境。
    内部逻辑：
        恢复底层配置、清理随机化、设置一行多列混合地形和按环境推扰，最后在 CUDA 创建仿真。
    作用：
        为高层策略提供与冻结 WTW 完全兼容、又能混合多任务并行采样的执行环境。
    """
    _restore_training_config(run_dir)
    _disable_randomization()
    _reset_mixed_terrain_baseline()

    Cfg.env.num_envs = num_envs
    Cfg.env.num_recording_envs = 0
    if recording_width is not None:
        Cfg.env.recording_width_px = int(recording_width)
    if recording_height is not None:
        Cfg.env.recording_height_px = int(recording_height)
    Cfg.asset.flip_visual_attachments = True

    Cfg.terrain.curriculum = False
    Cfg.terrain.selected = False
    Cfg.terrain.min_init_terrain_level = 0
    Cfg.terrain.max_init_terrain_level = 0
    # 前进方向只放一行；横向每列对应一个并行机器人。
    Cfg.terrain.num_rows = 1
    Cfg.terrain.num_cols = max(1, num_envs)
    Cfg.terrain.border_size = 0
    Cfg.terrain.center_robots = False
    Cfg.terrain.measure_heights = True
    Cfg.terrain.mesh_type = "trimesh"
    Cfg.terrain.terrain_length = terrain_length
    Cfg.terrain.terrain_width = terrain_width
    Cfg.terrain.teleport_robots = False
    Cfg.terrain.teleport_thresh = teleport_threshold
    Cfg.terrain.edge_reset_robots = True
    Cfg.terrain.edge_reset_margin = edge_reset_margin
    Cfg.terrain.env_conditions = list(conditions)

    # 只有任务分配中明确要求推扰的环境才启用定向外力。
    Cfg.domain_rand.push_robots = any(axis >= 0 for axis in push_axes)
    Cfg.domain_rand.push_interval_s = 2.0
    Cfg.domain_rand.max_push_vel_xy = 1.5
    Cfg.domain_rand.push_axis_by_env = list(push_axes)

    print(
        f"Creating {num_envs} environments, "
        f"terrain={terrain_length:.1f}m x {terrain_width:.1f}m"
    )
    env = VelocityTrackingEasyEnv(sim_device="cuda:0", headless=not render, cfg=Cfg)
    return HistoryWrapper(env)
