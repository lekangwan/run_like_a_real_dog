"""教学版使用的统一物理奖励。

所有地形都使用同一组公式和权重。奖励先保证速度跟踪与存活，再评价姿态、
接触安全和机械功率。安全项使用阈值区间，使“已经足够安全”的微小差别不会
压过核心跟踪任务；功率项只有在跟踪较好时才生效。
"""

import torch


REWARD_WEIGHTS = {
    "progress": 5.0,
    "yaw_tracking": 0.2,
    "survival": 2.0,
    "orientation": 0.8,
    "safety_lateral_drift": 0.25,
    "safety_contact_slip": 0.4,
    "safety_impact": 0.4,
    "safety_scuffing": 0.25,
    "strict_gated_power_efficiency": 0.8,
}


def threshold_score(score, low, high):
    """把原分数的可接受区间线性映射到 0 至 1。

    输入：原分数、不可接受阈值 ``low`` 和充分安全阈值 ``high``。
    输出：同形状的 0 至 1 分数。
    处理：低于 ``low`` 记 0，高于 ``high`` 记 1，中间线性插值。
    作用：防止安全指标内部无意义的小差异持续累积成较大奖励优势。
    """
    return ((score - low) / (high - low)).clamp(0.0, 1.0)


def compute_unified_reward(
    *,
    velocity_reward,
    yaw_reward,
    orientation_penalty,
    lateral_velocity_penalty,
    lateral_position_penalty,
    contact_slip_penalty,
    mechanical_power,
    impact_velocity_rms,
    scuffing_ratio,
    fall_penalty,
):
    """把原始物理量转换为统一奖励。

    输入：逐环境的速度跟踪、偏航、姿态、侧漂、滑移、功率、冲击、擦碰和摔倒量。
    输出：``(reward, raw, scores)``，分别为总奖励、原始量字典和归一化分数字典。
    处理：
        1. 将各物理代价转换为“越大越好”的 0 至 1 分数；
        2. 对侧漂、滑移、冲击和擦碰使用阈值化安全分数；
        3. 只有速度跟踪分数超过 0.75 时，机械功率奖励才逐渐开启；
        4. 按固定权重求加权平均。
    作用：为所有训练与评测场景提供同一评价口径，不包含地形标签或目标步态。
    """
    orientation = torch.exp(-orientation_penalty / 0.05)
    lateral = torch.exp(
        -lateral_velocity_penalty / 0.05 - lateral_position_penalty / 1.0
    )
    contact_slip = torch.exp(-contact_slip_penalty / 0.05)
    impact = torch.exp(-impact_velocity_rms.square() / 4.0)
    scuffing = torch.exp(-scuffing_ratio / 0.20)
    power = torch.exp(-mechanical_power / 250.0)
    tracking_gate = ((velocity_reward - 0.75) / 0.25).clamp(0.0, 1.0)

    scores = {
        "progress": velocity_reward,
        "yaw_tracking": yaw_reward,
        "survival": 1.0 - fall_penalty,
        "orientation": orientation,
        "safety_lateral_drift": threshold_score(lateral, 0.25, 0.50),
        "safety_contact_slip": threshold_score(contact_slip, 0.60, 0.85),
        "safety_impact": threshold_score(impact, 0.80, 0.92),
        "safety_scuffing": threshold_score(scuffing, 0.70, 0.90),
        "strict_gated_power_efficiency": tracking_gate * power,
    }
    weighted_sum = sum(REWARD_WEIGHTS[name] * scores[name] for name in REWARD_WEIGHTS)
    reward = weighted_sum / sum(REWARD_WEIGHTS.values())

    raw = {
        "velocity_reward": velocity_reward,
        "yaw_reward": yaw_reward,
        "orientation_penalty": orientation_penalty,
        "lateral_velocity_penalty": lateral_velocity_penalty,
        "lateral_position_penalty": lateral_position_penalty,
        "contact_slip_penalty": contact_slip_penalty,
        "mechanical_power": mechanical_power,
        "impact_velocity_rms": impact_velocity_rms,
        "scuffing_ratio": scuffing_ratio,
        "fall_penalty": fall_penalty,
    }
    return reward, raw, scores
