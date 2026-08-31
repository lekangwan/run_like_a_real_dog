"""定义训练场景，并把场景分配给并行仿真环境。

教学版不再读取历史 CSV。五类场景及其速度范围直接写在 ``TASK_CATALOG`` 中，
读者可以在一个地方看清训练分布。任务只决定地形、速度和是否施加横向推扰，
不会作为网络输入，也不会改变统一奖励。
"""

from dataclasses import dataclass
from types import SimpleNamespace

import torch

from .config import GAIT_NAMES, RESIDUAL_NAMES


@dataclass(frozen=True)
class TaskSpec:
    """一个仿真场景的最小描述。

    输入字段：任务名称、底层地形名称、目标前进速度上下界和采样权重。
    输出用途：供环境创建地形，并在每次重置时采样目标速度。
    """

    task_id: str
    condition: str
    vx_low: float
    vx_high: float
    sampling_weight: float = 1.0


# 这里写的是训练分布，不是“地形对应的正确步态”。
TASK_CATALOG = {
    "flat_trot_efficiency": TaskSpec(
        "flat_trot_efficiency", "flat", 0.5, 2.0
    ),
    "ramp_up_trot_robustness": TaskSpec(
        "ramp_up_trot_robustness", "ramp_up", 0.5, 2.0
    ),
    "rough_slope_trot_robustness": TaskSpec(
        "rough_slope_trot_robustness", "rough_slope", 0.5, 2.0
    ),
    "push_lateral_pace_recovery": TaskSpec(
        "push_lateral_pace_recovery", "push_lateral", 1.2, 1.8
    ),
    "stepping_stones_easy_bound_highspeed": TaskSpec(
        "stepping_stones_easy_bound_highspeed",
        "stepping_stones_easy",
        1.7,
        2.0,
    ),
}


def task_specs(task_ids):
    """按给定顺序取出训练场景。

    输入：任务名称序列。
    输出：对应的 ``TaskSpec`` 列表。
    处理：直接查询 ``TASK_CATALOG``；名称错误时由字典抛出 ``KeyError``。
    作用：保持命令行任务顺序与并行环境中的任务索引一致。
    """
    return [TASK_CATALOG[task_id] for task_id in task_ids]


def parse_eval_specs(text):
    """把 ``任务:速度`` 文本转换为固定速度评测场景。

    输入：逗号分隔文本，例如 ``flat_trot_efficiency:1.0``。
    输出：``TaskSpec`` 列表，其中每项的速度上下界相同。
    处理：查询内置任务目录以获得地形，再用命令行速度覆盖训练范围。
    作用：在不修改训练分布的情况下指定独立评测点。
    """
    specs = []
    for item in text.split(","):
        task_id, speed_text = item.strip().split(":", 1)
        base = TASK_CATALOG[task_id]
        speed = float(speed_text)
        specs.append(
            TaskSpec(
                task_id=f"{task_id}@{speed:g}",
                condition=base.condition,
                vx_low=speed,
                vx_high=speed,
            )
        )
    return specs


def _allocate_counts(specs, num_envs):
    """按权重分配并行环境，并保证每个场景至少有一个环境。

    输入：场景列表和并行环境总数。
    输出：与场景列表等长、总和为 ``num_envs`` 的整数列表。
    处理：先按比例向下取整，再按理论配额与当前数量的差补齐余数。
    作用：用有限并行环境近似目标训练分布。
    """
    if num_envs < len(specs):
        raise ValueError("num_envs must be at least the number of tasks")
    weights = torch.tensor([spec.sampling_weight for spec in specs])
    quotas = weights / weights.sum() * num_envs
    counts = torch.floor(quotas).long().clamp_min(1)
    while int(counts.sum()) < num_envs:
        counts[torch.argmax(quotas - counts)] += 1
    while int(counts.sum()) > num_envs:
        candidates = torch.where(counts > 1)[0]
        counts[candidates[torch.argmin(quotas[candidates] - counts[candidates])]] -= 1
    return counts.tolist()


def build_assignment(specs, num_envs, device):
    """把任务级配置展开成逐环境张量。

    输入：场景列表、并行环境数和张量设备。
    输出：包含任务索引、地形名、速度范围和推扰轴的命名空间。
    处理：按 ``_allocate_counts`` 的结果重复每个场景；横向推扰场景使用 Y 轴，
    其他场景用 ``-1`` 表示不启用定向推扰。
    作用：为每个并行机器人确定本次仿真的场景条件。
    """
    task_ids = []
    conditions = []
    vx_lows = []
    vx_highs = []
    push_axes = []

    counts = _allocate_counts(specs, num_envs)
    for task_index, (spec, count) in enumerate(zip(specs, counts)):
        task_ids.extend([task_index] * count)
        conditions.extend([spec.condition] * count)
        vx_lows.extend([spec.vx_low] * count)
        vx_highs.extend([spec.vx_high] * count)
        push_axes.extend([1 if spec.condition == "push_lateral" else -1] * count)

    return SimpleNamespace(
        task_ids=torch.tensor(task_ids, device=device, dtype=torch.long),
        conditions=conditions,
        vx_lows=torch.tensor(vx_lows, device=device),
        vx_highs=torch.tensor(vx_highs, device=device),
        push_axes=push_axes,
    )


def fixed_gait_action(num_envs, gait_name, device):
    """生成固定步态、零参数修正的高层指令。

    输入：环境数、步态名称和张量设备。
    输出：``[num_envs, 9]`` 张量，前四维只有指定步态为 1，后五维为 0。
    处理：将名称转换为固定步态索引并写入一位编码。
    作用：在同一环境接口下构造固定步态对照。
    """
    action = torch.zeros(
        num_envs,
        len(GAIT_NAMES) + len(RESIDUAL_NAMES),
        device=device,
    )
    action[:, GAIT_NAMES.index(gait_name)] = 1.0
    return action
