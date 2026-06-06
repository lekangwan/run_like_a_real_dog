import argparse
import math
from pathlib import Path

import pandas as pd


IDENTITY_COLUMNS = {
    "condition",
    "vx",
    "gait",
    "frequency",
    "footswing_height",
    "body_pitch",
    "stance_width",
    "phase",
    "offset",
    "bound",
}

REFERENCE_COLUMNS = {
    "measured_vx",
    "vx_abs_error",
    "vy_abs_error",
    "yaw_abs_error",
    "velocity_reward",
    "yaw_reward",
    "template_score",
}

FOOT_INDEX_TO_LEG = {
    0: "FL",
    1: "FR",
    2: "RL",
    3: "RR",
}

PAIR_TO_LEGS = {
    "01": "FL_FR",
    "02": "FL_RL",
    "03": "FL_RR",
    "12": "FR_RL",
    "13": "FR_RR",
    "23": "RL_RR",
}

GAIT_SYNC_PAIRS = {
    "bounding": ("01", "23"),
    "pacing": ("02", "13"),
    "trotting": ("03", "12"),
}

GAIT_SYNC_TARGETS = {
    "pronking": {"01": 1.0, "02": 1.0, "03": 1.0, "12": 1.0, "13": 1.0, "23": 1.0},
    "bounding": {"01": 1.0, "02": 0.0, "03": 0.0, "12": 0.0, "13": 0.0, "23": 1.0},
    "pacing": {"01": 0.0, "02": 1.0, "03": 0.0, "12": 0.0, "13": 1.0, "23": 0.0},
    "trotting": {"01": 0.0, "02": 0.0, "03": 1.0, "12": 1.0, "13": 0.0, "23": 0.0},
}

QUALITY_COST_METRICS = [
    "torque_penalty",
    "slip_penalty",
    "orientation_penalty",
    "action_delta_sq",
    "transport_cost_proxy",
    "foot_impact_vel_mean",
    "scuffing_ratio",
    "phase_match_error",
    "progress_deficit",
    "done_rate",
]

DEFAULT_QUALITY_WEIGHTS = {
    "done_rate": 4.0,
    "progress_deficit": 3.0,
    "slip_penalty": 2.0,
    "orientation_penalty": 2.0,
    "scuffing_ratio": 2.0,
    "foot_impact_vel_mean": 1.5,
    "phase_match_error": 1.5,
    "transport_cost_proxy": 1.0,
    "action_delta_sq": 1.0,
    "torque_penalty": 0.5,
}

CONDITION_GATE_WEIGHTS = {
    "done_rate": 2.0,
    "progress_deficit": 2.0,
}

CONDITION_GATE_PENALTY = 0.15
TARGET_SCORE_GAP_HIGH = 0.05
TARGET_SCORE_GAP_MEDIUM = 0.02

# Positive weights mean lower normalized metric value is better.
# Negative weights mean higher normalized metric value is better.
DEFAULT_CONDITION_SCORE_WEIGHTS = {
    "flat": {
        "torque_penalty": 1.5,
        "mechanical_power_abs": 1.5,
        "transport_cost_proxy": 1.0,
        "slip_penalty": 1.0,
        "orientation_penalty": 1.0,
        "base_z_vel_rms": 1.0,
        "roll_rate_rms": 0.75,
        "pitch_rate_rms": 0.75,
        "yaw_rate_rms": 0.5,
        "lateral_vel_rms": 1.25,
        "foot_impact_vel_mean": 0.75,
        "scuffing_ratio": 0.5,
    },
    "low_friction": {
        "slip_penalty": 3.0,
        "lateral_vel_rms": 2.0,
        "yaw_rate_rms": 1.5,
        "roll_rate_rms": 1.0,
        "gravity_y_rms": 1.0,
        "orientation_penalty": 1.0,
        "torque_penalty": 0.5,
        "foot_impact_vel_mean": 0.5,
    },
    "very_low_friction": {
        "slip_penalty": 3.5,
        "lateral_vel_rms": 2.0,
        "yaw_rate_rms": 1.5,
        "roll_rate_rms": 1.0,
        "gravity_y_rms": 1.0,
        "orientation_penalty": 1.0,
        "torque_penalty": 0.5,
        "foot_impact_vel_mean": 0.5,
    },
    "rough": {
        "scuffing_ratio": 2.0,
        "swing_foot_clearance_mean": -2.0,
        "foot_impact_vel_mean": 1.5,
        "foot_impact_vel_rms": 1.0,
        "slip_penalty": 1.0,
        "orientation_penalty": 1.0,
        "base_z_vel_rms": 0.75,
        "transport_cost_proxy": 0.5,
    },
    "rough_mid": {
        "scuffing_ratio": 2.0,
        "swing_foot_clearance_mean": -2.0,
        "foot_impact_vel_mean": 1.25,
        "foot_impact_vel_rms": 0.75,
        "slip_penalty": 1.0,
        "orientation_penalty": 1.0,
        "base_z_vel_rms": 0.75,
        "progress_deficit": 0.75,
        "transport_cost_proxy": 0.5,
    },
    "rough_hard": {
        "scuffing_ratio": 2.5,
        "swing_foot_clearance_mean": -2.5,
        "foot_impact_vel_mean": 1.5,
        "foot_impact_vel_rms": 1.0,
        "slip_penalty": 1.0,
        "orientation_penalty": 1.0,
        "base_z_vel_rms": 0.75,
        "transport_cost_proxy": 0.5,
    },
    "rough_slope": {
        "scuffing_ratio": 1.5,
        "swing_foot_clearance_mean": -1.5,
        "orientation_penalty": 1.5,
        "gravity_x_rms": 1.0,
        "pitch_rate_rms": 1.0,
        "slip_penalty": 1.0,
        "foot_impact_vel_mean": 1.0,
        "base_z_vel_rms": 0.75,
    },
    "stairs": {
        "swing_foot_clearance_mean": -2.5,
        "scuffing_ratio": 2.0,
        "foot_impact_vel_mean": 1.5,
        "foot_impact_vel_rms": 1.0,
        "progress_deficit": 1.0,
        "orientation_penalty": 1.0,
        "base_z_vel_rms": 1.0,
        "pitch_rate_rms": 0.5,
    },
    "stairs_up_low": {
        "swing_foot_clearance_mean": -2.0,
        "scuffing_ratio": 1.5,
        "foot_impact_vel_mean": 1.25,
        "foot_impact_vel_rms": 0.75,
        "progress_deficit": 1.5,
        "orientation_penalty": 1.0,
        "base_z_vel_rms": 0.75,
        "pitch_rate_rms": 0.5,
    },
    "stairs_down_low": {
        "swing_foot_clearance_mean": -1.5,
        "scuffing_ratio": 1.25,
        "foot_impact_vel_mean": 1.75,
        "foot_impact_vel_rms": 1.0,
        "progress_deficit": 1.5,
        "orientation_penalty": 1.0,
        "base_z_vel_rms": 0.75,
        "pitch_rate_rms": 0.75,
    },
    "stairs_up": {
        "swing_foot_clearance_mean": -2.5,
        "scuffing_ratio": 2.0,
        "foot_impact_vel_mean": 1.5,
        "foot_impact_vel_rms": 1.0,
        "progress_deficit": 1.0,
        "orientation_penalty": 1.0,
        "base_z_vel_rms": 1.0,
        "pitch_rate_rms": 0.5,
    },
    "stairs_down": {
        "swing_foot_clearance_mean": -2.0,
        "scuffing_ratio": 1.5,
        "foot_impact_vel_mean": 2.0,
        "foot_impact_vel_rms": 1.5,
        "progress_deficit": 1.0,
        "orientation_penalty": 1.0,
        "base_z_vel_rms": 1.0,
        "pitch_rate_rms": 0.75,
    },
    "discrete_obstacles": {
        "swing_foot_clearance_mean": -2.5,
        "scuffing_ratio": 2.0,
        "foot_impact_vel_mean": 1.0,
        "foot_impact_vel_rms": 1.0,
        "progress_deficit": 1.0,
        "orientation_penalty": 1.0,
        "base_z_vel_rms": 1.0,
        "transport_cost_proxy": 0.5,
    },
    "discrete_obstacles_low": {
        "swing_foot_clearance_mean": -2.0,
        "scuffing_ratio": 1.5,
        "foot_impact_vel_mean": 1.0,
        "foot_impact_vel_rms": 0.75,
        "progress_deficit": 1.5,
        "orientation_penalty": 1.0,
        "base_z_vel_rms": 0.75,
        "transport_cost_proxy": 0.5,
    },
    "stepping_stones": {
        "swing_foot_clearance_mean": -2.5,
        "scuffing_ratio": 2.0,
        "foot_impact_vel_mean": 1.5,
        "foot_impact_vel_rms": 1.0,
        "progress_deficit": 1.5,
        "orientation_penalty": 1.0,
        "lateral_vel_rms": 1.0,
        "base_z_vel_rms": 1.0,
    },
    "stepping_stones_easy": {
        "swing_foot_clearance_mean": -2.0,
        "scuffing_ratio": 1.5,
        "foot_impact_vel_mean": 1.25,
        "foot_impact_vel_rms": 0.75,
        "progress_deficit": 1.5,
        "orientation_penalty": 1.0,
        "lateral_vel_rms": 1.0,
        "base_z_vel_rms": 0.75,
    },
    "push": {
        "progress_deficit": 2.5,
        "orientation_penalty": 1.5,
        "lateral_vel_rms": 1.5,
        "yaw_rate_rms": 1.0,
        "slip_penalty": 1.0,
        "base_z_vel_rms": 1.0,
        "torque_penalty": 0.5,
        "foot_impact_vel_mean": 0.5,
    },
    "push_hard": {
        "progress_deficit": 3.0,
        "orientation_penalty": 1.5,
        "lateral_vel_rms": 1.5,
        "yaw_rate_rms": 1.0,
        "slip_penalty": 1.0,
        "base_z_vel_rms": 1.0,
        "torque_penalty": 0.5,
        "foot_impact_vel_mean": 0.5,
    },
    "ramp_up": {
        "orientation_penalty": 2.0,
        "gravity_x_rms": 1.5,
        "pitch_rate_rms": 1.5,
        "slip_penalty": 1.0,
        "progress_deficit": 1.0,
        "lateral_vel_rms": 0.75,
        "torque_penalty": 0.5,
        "foot_impact_vel_mean": 0.5,
    },
    "slope": {
        "orientation_penalty": 2.0,
        "gravity_x_rms": 1.5,
        "pitch_rate_rms": 1.5,
        "slip_penalty": 1.0,
        "progress_deficit": 1.0,
        "lateral_vel_rms": 0.75,
        "torque_penalty": 0.5,
        "foot_impact_vel_mean": 0.5,
    },
}

BODY_MOTION_METRICS = [
    "base_z_std",
    "base_z_vel_rms",
    "roll_rate_rms",
    "pitch_rate_rms",
    "yaw_rate_rms",
    "lateral_vel_rms",
]


PREFERRED_METRICS = [
    "base_z_std",
    "base_z_vel_rms",
    "roll_rate_rms",
    "pitch_rate_rms",
    "yaw_rate_rms",
    "gravity_x_rms",
    "gravity_y_rms",
    "lateral_vel_rms",
    "contact_count",
    "flight_ratio",
    "all_contact_ratio",
    "foot0_duty",
    "foot1_duty",
    "foot2_duty",
    "foot3_duty",
    "contact_pair01_sync",
    "contact_pair02_sync",
    "contact_pair03_sync",
    "contact_pair12_sync",
    "contact_pair13_sync",
    "contact_pair23_sync",
    "contact_pair01_co",
    "contact_pair02_co",
    "contact_pair03_co",
    "contact_pair12_co",
    "contact_pair13_co",
    "contact_pair23_co",
    "torque_penalty",
    "mechanical_power_abs",
    "positive_mechanical_power",
    "transport_cost_proxy",
    "slip_penalty",
    "orientation_penalty",
    "vertical_velocity_penalty",
    "action_delta_sq",
    "contact_force_mean",
    "stance_contact_force_mean",
    "peak_contact_force",
    "foot_impact_rate",
    "foot_impact_vel_mean",
    "foot_impact_vel_rms",
    "swing_foot_clearance_mean",
    "scuffing_ratio",
    "phase_match_error",
    "forward_distance",
    "target_forward_distance",
    "forward_distance_ratio",
    "progress_deficit",
    "done_rate",
]


METRIC_DOCUMENTATION = {
    "mechanical_power_abs": {
        "meaning": "关节机械功率绝对值的时间平均，反映动作整体能耗强度",
        "formula": "mean_t sum_j abs(torque_j * dof_vel_j)",
        "source": "env.torques, env.dof_vel",
        "direction": "lower_is_better",
    },
    "positive_mechanical_power": {
        "meaning": "只统计正机械功率的时间平均，近似主动做功强度",
        "formula": "mean_t sum_j max(torque_j * dof_vel_j, 0)",
        "source": "env.torques, env.dof_vel",
        "direction": "lower_is_better",
    },
    "transport_cost_proxy": {
        "meaning": "单位前进速度下的机械功率代理指标，近似比较不同 gait 的移动效率",
        "formula": "mechanical_power_abs / max(abs(mean_vx), 0.2)",
        "source": "mechanical_power_abs, measured_vx",
        "direction": "lower_is_better",
    },
    "contact_force_mean": {
        "meaning": "四只脚接触力范数的时间平均，包含未接触脚的零/小力",
        "formula": "mean_t mean_i norm(contact_force_i)",
        "source": "env.contact_forces[:, feet_indices, :]",
        "direction": "lower_is_better",
    },
    "stance_contact_force_mean": {
        "meaning": "只在接触脚上统计的平均接触力，反映支撑冲击/负载强度",
        "formula": "mean_t sum_i contact_i*norm(force_i) / max(num_contacts, 1)",
        "source": "env.contact_forces, contact mask",
        "direction": "lower_is_better",
    },
    "peak_contact_force": {
        "meaning": "评估窗口内单脚最大接触力，反映最坏落脚/撞击风险",
        "formula": "max_t max_i norm(contact_force_i)",
        "source": "env.contact_forces",
        "direction": "lower_is_better",
    },
    "foot_impact_rate": {
        "meaning": "每个控制步内平均新触地事件数，反映落脚事件频率",
        "formula": "touchdown_count / eval_steps",
        "source": "contact_t and contact_{t-1}",
        "direction": "context_feature",
    },
    "foot_impact_vel_mean": {
        "meaning": "新触地瞬间足端向下速度均值，反映落脚冲击速度",
        "formula": "mean over touchdowns max(-prev_foot_vel_z, 0)",
        "source": "env.prev_foot_velocities, contact transitions",
        "direction": "lower_is_better",
    },
    "foot_impact_vel_rms": {
        "meaning": "新触地瞬间足端向下速度 RMS，更强调大的冲击速度",
        "formula": "sqrt(mean over touchdowns max(-prev_foot_vel_z, 0)^2)",
        "source": "env.prev_foot_velocities, contact transitions",
        "direction": "lower_is_better",
    },
    "swing_foot_clearance_mean": {
        "meaning": "摆动脚离地高度均值，反映跨越粗糙地形/台阶的余量",
        "formula": "mean_t mean_i non_contact_i * (foot_z - terrain_height_at_foot)",
        "source": "env.foot_positions, env.height_samples",
        "direction": "higher_is_better",
    },
    "scuffing_ratio": {
        "meaning": "摆动脚离地过低的比例，反映擦地风险",
        "formula": "mean_t count(non_contact and clearance < 0.035) / max(num_swing_feet, 1)",
        "source": "foot clearance, contact mask",
        "direction": "lower_is_better",
    },
    "phase_match_error": {
        "meaning": "实际接触状态和命令 gait 模板期望接触状态的差异",
        "formula": "mean_t mean_i abs(actual_contact_i - desired_contact_state_i)",
        "source": "contact mask, env.desired_contact_states",
        "direction": "lower_is_better",
    },
    "forward_distance": {
        "meaning": "评估窗口内由机身前向速度积分得到的前进距离",
        "formula": "sum_t base_lin_vel_x * policy_dt",
        "source": "env.base_lin_vel, env.dt",
        "direction": "higher_is_better",
    },
    "target_forward_distance": {
        "meaning": "评估窗口内速度指令对应的目标前进距离",
        "formula": "sum_t vx_cmd * policy_dt",
        "source": "command vx, env.dt",
        "direction": "reference",
    },
    "forward_distance_ratio": {
        "meaning": "实际前进距离和目标前进距离的比值",
        "formula": "forward_distance / max(abs(target_forward_distance), 0.2)",
        "source": "forward_distance, target_forward_distance",
        "direction": "closer_to_1_is_better",
    },
    "progress_deficit": {
        "meaning": "未达到目标前进距离的比例，反映地形通过/前进能力不足",
        "formula": "max(target_forward_distance - forward_distance, 0) / max(abs(target_forward_distance), 0.2)",
        "source": "forward_distance, target_forward_distance",
        "direction": "lower_is_better",
    },
}


def numeric_metric_columns(df):
    numeric = set(df.select_dtypes(include="number").columns)
    candidates = [name for name in PREFERRED_METRICS if name in numeric]
    extras = sorted(
        numeric
        - IDENTITY_COLUMNS
        - REFERENCE_COLUMNS
        - set(candidates)
        - {"fall_count"}
    )
    return candidates + extras


def metric_ranges(df, metrics):
    rows = []
    for metric in metrics:
        values = df[metric].dropna()
        p05 = values.quantile(0.05)
        p25 = values.quantile(0.25)
        p50 = values.quantile(0.50)
        p75 = values.quantile(0.75)
        p95 = values.quantile(0.95)
        rows.append(
            {
                "metric": metric,
                "min": values.min(),
                "p05": p05,
                "p25": p25,
                "median": p50,
                "p75": p75,
                "p95": p95,
                "max": values.max(),
                "mean": values.mean(),
                "std": values.std(ddof=0),
                "iqr": p75 - p25,
                "robust_range_p95_p05": p95 - p05,
            }
        )
    return pd.DataFrame(rows)


def parse_weights(text):
    if text is None or text.strip() == "":
        return dict(DEFAULT_QUALITY_WEIGHTS)
    weights = {}
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"Invalid weight item '{item}', expected metric=weight")
        metric, value = item.split("=", 1)
        metric = metric.strip()
        if not metric:
            raise ValueError(f"Invalid empty metric in '{item}'")
        weights[metric] = float(value)
    if not weights:
        raise ValueError("No valid quality weights were provided")
    return weights


def weights_table(weights):
    return pd.DataFrame(
        [{"metric": metric, "weight": weight} for metric, weight in weights.items()]
    )


def condition_score_weights_table(condition_weights, gate_weights):
    rows = []
    for condition, weights in condition_weights.items():
        for metric, signed_weight in weights.items():
            rows.append(
                {
                    "condition": condition,
                    "stage": "condition_feature",
                    "metric": metric,
                    "weight": abs(signed_weight),
                    "direction": "higher_is_better" if signed_weight < 0 else "lower_is_better",
                }
            )
    for metric, weight in gate_weights.items():
        rows.append(
            {
                "condition": "all",
                "stage": "safety_gate",
                "metric": metric,
                "weight": weight,
                "direction": "lower_is_better",
            }
        )
    return pd.DataFrame(rows)


def metric_documentation(metrics):
    rows = []
    for metric in metrics:
        doc = METRIC_DOCUMENTATION.get(metric)
        if not doc:
            continue
        rows.append(
            {
                "metric": metric,
                "meaning": doc["meaning"],
                "formula": doc["formula"],
                "source": doc["source"],
                "direction": doc["direction"],
            }
        )
    return pd.DataFrame(rows)


def separability(df, metrics, group_cols):
    rows = []
    groups = [((), df)] if not group_cols else df.groupby(group_cols, dropna=False)
    for keys, group in groups:
        if not isinstance(keys, tuple):
            keys = (keys,)
        key_values = dict(zip(group_cols, keys))
        for metric in metrics:
            total_var = group[metric].var(ddof=0)
            gait_means = group.groupby("gait")[metric].mean()
            between_var = gait_means.var(ddof=0)
            rows.append(
                {
                    **key_values,
                    "metric": metric,
                    "total_var": total_var,
                    "between_gait_mean_var": between_var,
                    "separability": 0.0 if total_var == 0 else between_var / total_var,
                }
            )
    return pd.DataFrame(rows)


def normalized_summary(df, ranges, metrics):
    group_cols = grouping_columns(df)
    normalized = robust_normalized_frame(df, ranges, metrics, keep_columns=group_cols)
    return normalized.groupby(group_cols, as_index=False)[metrics].mean()


def grouping_columns(df, include_gait=True):
    columns = []
    if "condition" in df:
        columns.append("condition")
    columns.append("vx")
    if include_gait:
        columns.append("gait")
    return columns


def robust_normalized_frame(df, ranges, metrics, keep_columns=None):
    range_by_metric = ranges.set_index("metric")
    normalized = df[list(keep_columns or [])].copy()
    for metric in metrics:
        p05 = range_by_metric.loc[metric, "p05"]
        p95 = range_by_metric.loc[metric, "p95"]
        scale = p95 - p05
        if scale == 0:
            normalized[metric] = 0.0
        else:
            normalized[metric] = ((df[metric] - p05) / scale).clip(0.0, 1.0)
    return normalized


def gait_signature(df, metrics):
    agg = {metric: "mean" for metric in metrics}
    reference = [name for name in ("measured_vx", "vx_abs_error", "template_score", "done_rate") if name in df]
    for name in reference:
        agg[name] = "mean"
    return df.groupby(grouping_columns(df), as_index=False).agg(agg)


def best_by_metric(df, metrics):
    rows = []
    group_cols = grouping_columns(df, include_gait=False)
    groups = df.groupby(group_cols, dropna=False)
    for keys, speed_group in groups:
        if not isinstance(keys, tuple):
            keys = (keys,)
        key_values = dict(zip(group_cols, keys))
        for metric in metrics:
            means = speed_group.groupby("gait")[metric].mean().sort_values()
            low_gait = means.index[0]
            high_gait = means.index[-1]
            rows.append(
                {
                    **key_values,
                    "metric": metric,
                    "lowest_gait": low_gait,
                    "lowest_mean": means.iloc[0],
                    "highest_gait": high_gait,
                    "highest_mean": means.iloc[-1],
                    "range_across_gaits": means.iloc[-1] - means.iloc[0],
                }
            )
    return pd.DataFrame(rows)


def add_contact_identity_scores(df):
    scored = df.copy()
    sync_cols = [f"contact_pair{pair}_sync" for pair in PAIR_TO_LEGS if f"contact_pair{pair}_sync" in scored]
    co_cols = [f"contact_pair{pair}_co" for pair in PAIR_TO_LEGS if f"contact_pair{pair}_co" in scored]
    if sync_cols:
        scored["pronk_sync_signal"] = scored[sync_cols].mean(axis=1)
    if co_cols:
        scored["pronk_co_signal"] = scored[co_cols].mean(axis=1)

    contact_signal_cols = []
    for gait, pairs in GAIT_SYNC_PAIRS.items():
        gait_sync_cols = [f"contact_pair{pair}_sync" for pair in pairs if f"contact_pair{pair}_sync" in scored]
        gait_co_cols = [f"contact_pair{pair}_co" for pair in pairs if f"contact_pair{pair}_co" in scored]
        if gait_sync_cols:
            name = f"{gait}_sync_signal"
            scored[name] = scored[gait_sync_cols].mean(axis=1)
            contact_signal_cols.append(name)
        if gait_co_cols:
            scored[f"{gait}_co_signal"] = scored[gait_co_cols].mean(axis=1)

    if "pronk_sync_signal" in scored:
        contact_signal_cols.append("pronk_sync_signal")
    if contact_signal_cols:
        scored["dominant_contact_signal"] = scored[contact_signal_cols].idxmax(axis=1).str.replace(
            "_sync_signal", "", regex=False
        )

    match_cols = []
    available_pairs = [pair for pair in PAIR_TO_LEGS if f"contact_pair{pair}_sync" in scored]
    for gait, target in GAIT_SYNC_TARGETS.items():
        if not available_pairs:
            continue
        distance_terms = []
        for pair in available_pairs:
            distance_terms.append((scored[f"contact_pair{pair}_sync"] - target[pair]).abs())
        name = f"{gait}_contact_match"
        scored[name] = 1.0 - pd.concat(distance_terms, axis=1).mean(axis=1)
        match_cols.append(name)
    if match_cols:
        scored["dominant_contact_match"] = scored[match_cols].idxmax(axis=1).str.replace(
            "_contact_match", "", regex=False
        )
    return scored


def intrinsic_scores(df, ranges, metrics):
    keep = [
        name
        for name in (
            "condition",
            "vx",
            "gait",
            "frequency",
            "footswing_height",
            "body_pitch",
            "stance_width",
            "phase",
            "offset",
            "bound",
        )
        if name in df
    ]
    normalized = robust_normalized_frame(df, ranges, metrics, keep_columns=keep)

    quality_costs = [metric for metric in QUALITY_COST_METRICS if metric in normalized]
    body_motion = [metric for metric in BODY_MOTION_METRICS if metric in normalized]
    if quality_costs:
        normalized["quality_cost_norm"] = normalized[quality_costs].mean(axis=1)
        normalized["quality_score"] = 1.0 - normalized["quality_cost_norm"]
    if body_motion:
        normalized["body_motion_intensity_norm"] = normalized[body_motion].mean(axis=1)
        normalized["flatness_score"] = 1.0 - normalized["body_motion_intensity_norm"]

    raw_with_contact = add_contact_identity_scores(df)
    contact_cols = [
        col
        for col in raw_with_contact.columns
        if col.endswith("_sync_signal")
        or col.endswith("_co_signal")
        or col.endswith("_contact_match")
        or col == "dominant_contact_signal"
        or col == "dominant_contact_match"
    ]
    return pd.concat([normalized, raw_with_contact[contact_cols]], axis=1)


def add_weighted_quality_score(scores, weights):
    scored = scores.copy()
    present_weights = {metric: weight for metric, weight in weights.items() if metric in scored}
    missing = sorted(set(weights) - set(present_weights))
    if missing:
        print(f"Warning: ignoring missing quality weight metrics: {', '.join(missing)}")
    total_weight = sum(present_weights.values())
    if total_weight <= 0:
        raise ValueError("Sum of present quality weights must be positive")

    weighted_cost = 0.0
    for metric, weight in present_weights.items():
        contribution_name = f"{metric}_weighted_contribution"
        scored[contribution_name] = scored[metric] * weight / total_weight
        weighted_cost = weighted_cost + scored[contribution_name]
    scored["weighted_quality_cost_norm"] = weighted_cost
    scored["weighted_quality_score"] = 1.0 - scored["weighted_quality_cost_norm"]
    return scored


def quality_component_summary(scores, weights):
    present_metrics = [metric for metric in weights if metric in scores]
    contribution_cols = [
        f"{metric}_weighted_contribution"
        for metric in present_metrics
        if f"{metric}_weighted_contribution" in scores
    ]
    value_cols = [
        "weighted_quality_score",
        "weighted_quality_cost_norm",
    ] + present_metrics + contribution_cols
    return scores.groupby(grouping_columns(scores), as_index=False)[value_cols].mean()


def quality_component_dominance(component_summary, weights):
    contribution_cols = [
        f"{metric}_weighted_contribution"
        for metric in weights
        if f"{metric}_weighted_contribution" in component_summary
    ]
    rows = []
    for _, row in component_summary.iterrows():
        if not contribution_cols:
            continue
        contributions = row[contribution_cols].astype(float)
        largest = contributions.idxmax()
        rows.append(
            {
                "vx": row["vx"],
                **({"condition": row["condition"]} if "condition" in row else {}),
                "gait": row["gait"],
                "dominant_quality_component": largest.replace("_weighted_contribution", ""),
                "dominant_contribution": contributions[largest],
                "weighted_quality_cost_norm": row["weighted_quality_cost_norm"],
                "weighted_quality_score": row["weighted_quality_score"],
            }
        )
    return pd.DataFrame(rows)


def best_weighted_by_speed_gait(scores):
    idx = scores.groupby(grouping_columns(scores))["weighted_quality_score"].idxmax()
    return scores.loc[idx].sort_values(grouping_columns(scores)).reset_index(drop=True)


def best_weighted_gait_by_speed(best_by_speed_gait):
    group_cols = grouping_columns(best_by_speed_gait, include_gait=False)
    idx = best_by_speed_gait.groupby(group_cols)["weighted_quality_score"].idxmax()
    return best_by_speed_gait.loc[idx].sort_values(group_cols).reset_index(drop=True)


def target_distribution_by_speed(best_by_speed_gait, temperature):
    if temperature <= 0:
        raise ValueError("--target-temperature must be positive")

    rows = []
    group_cols = grouping_columns(best_by_speed_gait, include_gait=False)
    for keys, group in best_by_speed_gait.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        key_values = dict(zip(group_cols, keys))
        scores = group["weighted_quality_score"]
        max_score = scores.max()
        logits = [(score - max_score) / temperature for score in scores]
        exp_logits = [math.exp(value) for value in logits]
        denom = sum(exp_logits)
        for (_, row), prob in zip(group.iterrows(), exp_logits):
            out = row.to_dict()
            out.update(key_values)
            out["target_temperature"] = temperature
            out["target_prob"] = prob / denom if denom > 0 else 0.0
            rows.append(out)
    return pd.DataFrame(rows).sort_values(
        group_cols + ["target_prob"],
        ascending=[True] * len(group_cols) + [False],
    )


def add_condition_adapt_score(
    scores,
    condition_weights,
    gate_weights,
    gate_penalty=CONDITION_GATE_PENALTY,
):
    scored = scores.copy()
    if "condition" not in scored:
        scored["condition"] = "flat"

    feature_costs = []
    gate_costs = []
    total_costs = []
    adapt_scores = []

    all_feature_metrics = sorted(
        {metric for weights in condition_weights.values() for metric in weights}
    )
    all_gate_metrics = sorted(gate_weights)
    feature_contrib_cols = [f"condition_feature_{metric}_contribution" for metric in all_feature_metrics]
    gate_contrib_cols = [f"condition_gate_{metric}_contribution" for metric in all_gate_metrics]
    for col in feature_contrib_cols + gate_contrib_cols:
        scored[col] = 0.0

    for idx, row in scored.iterrows():
        condition = row.get("condition", "flat")
        weights = condition_weights.get(condition, condition_weights.get("flat", {}))

        feature_total = 0.0
        feature_cost = 0.0
        for metric, signed_weight in weights.items():
            if metric not in scored:
                continue
            weight = abs(signed_weight)
            metric_cost = 1.0 - row[metric] if signed_weight < 0 else row[metric]
            contribution = metric_cost * weight
            scored.at[idx, f"condition_feature_{metric}_contribution"] = contribution
            feature_cost += contribution
            feature_total += weight
        if feature_total > 0:
            for metric in weights:
                col = f"condition_feature_{metric}_contribution"
                if col in scored:
                    scored.at[idx, col] = scored.at[idx, col] / feature_total
            feature_cost /= feature_total

        gate_total = 0.0
        gate_cost = 0.0
        for metric, weight in gate_weights.items():
            if metric not in scored:
                continue
            contribution = row[metric] * weight
            scored.at[idx, f"condition_gate_{metric}_contribution"] = contribution
            gate_cost += contribution
            gate_total += weight
        if gate_total > 0:
            for metric in gate_weights:
                col = f"condition_gate_{metric}_contribution"
                if col in scored:
                    scored.at[idx, col] = gate_penalty * scored.at[idx, col] / gate_total
            gate_cost /= gate_total

        total_cost = feature_cost + gate_penalty * gate_cost
        feature_costs.append(feature_cost)
        gate_costs.append(gate_cost)
        total_costs.append(total_cost)
        adapt_scores.append(max(0.0, min(1.0, 1.0 - total_cost)))

    scored["condition_feature_cost_norm"] = feature_costs
    scored["condition_gate_cost_norm"] = gate_costs
    scored["condition_total_cost_norm"] = total_costs
    scored["condition_adapt_score"] = adapt_scores
    return scored


def condition_component_summary(scores):
    contribution_cols = [
        col
        for col in scores.columns
        if col.startswith("condition_feature_") and col.endswith("_contribution")
        or col.startswith("condition_gate_") and col.endswith("_contribution")
    ]
    value_cols = [
        "condition_adapt_score",
        "condition_feature_cost_norm",
        "condition_gate_cost_norm",
        "condition_total_cost_norm",
    ] + contribution_cols
    return scores.groupby(grouping_columns(scores), as_index=False)[value_cols].mean()


def condition_component_dominance(component_summary):
    contribution_cols = [
        col
        for col in component_summary.columns
        if col.startswith("condition_feature_") and col.endswith("_contribution")
        or col.startswith("condition_gate_") and col.endswith("_contribution")
    ]
    rows = []
    for _, row in component_summary.iterrows():
        if not contribution_cols:
            continue
        contributions = row[contribution_cols].astype(float)
        largest = contributions.idxmax()
        component = largest
        component = component.replace("condition_feature_", "feature:")
        component = component.replace("condition_gate_", "gate:")
        component = component.replace("_contribution", "")
        rows.append(
            {
                "vx": row["vx"],
                **({"condition": row["condition"]} if "condition" in row else {}),
                "gait": row["gait"],
                "dominant_condition_component": component,
                "dominant_contribution": contributions[largest],
                "condition_total_cost_norm": row["condition_total_cost_norm"],
                "condition_adapt_score": row["condition_adapt_score"],
            }
        )
    return pd.DataFrame(rows)


def best_condition_score_by_speed_gait(scores):
    idx = scores.groupby(grouping_columns(scores))["condition_adapt_score"].idxmax()
    return scores.loc[idx].sort_values(grouping_columns(scores)).reset_index(drop=True)


def best_condition_score_gait_by_speed(best_by_speed_gait):
    group_cols = grouping_columns(best_by_speed_gait, include_gait=False)
    idx = best_by_speed_gait.groupby(group_cols)["condition_adapt_score"].idxmax()
    return best_by_speed_gait.loc[idx].sort_values(group_cols).reset_index(drop=True)


def condition_target_distribution_by_speed(best_by_speed_gait, temperature):
    renamed = best_by_speed_gait.drop(columns=["condition_adapt_score"]).copy()
    renamed["weighted_quality_score"] = best_by_speed_gait["condition_adapt_score"].values
    out = target_distribution_by_speed(renamed, temperature)
    out = out.rename(
        columns={
            "weighted_quality_score": "condition_adapt_score",
            "target_temperature": "condition_target_temperature",
            "target_prob": "condition_target_prob",
        }
    )
    return out


def condition_target_confidence(best_by_speed_gait, target_distribution):
    group_cols = grouping_columns(best_by_speed_gait, include_gait=False)
    prob_col = "condition_target_prob"
    rows = []
    for keys, group in best_by_speed_gait.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        key_values = dict(zip(group_cols, keys))
        ranked = group.sort_values("condition_adapt_score", ascending=False).reset_index(drop=True)
        target_group = target_distribution.copy()
        for col, value in key_values.items():
            target_group = target_group[target_group[col] == value]
        target_probs = target_group.set_index("gait")[prob_col].to_dict()

        best = ranked.iloc[0]
        second = ranked.iloc[1] if len(ranked) > 1 else ranked.iloc[0]
        score_gap = best["condition_adapt_score"] - second["condition_adapt_score"]
        best_prob = target_probs.get(best["gait"], 0.0)
        second_prob = target_probs.get(second["gait"], 0.0)
        if score_gap >= TARGET_SCORE_GAP_HIGH:
            confidence = "high"
            training_label_mode = "hard"
            use_for_training = True
        elif score_gap >= TARGET_SCORE_GAP_MEDIUM:
            confidence = "medium"
            training_label_mode = "soft"
            use_for_training = True
        else:
            confidence = "low"
            training_label_mode = "re_evaluate_or_skip"
            use_for_training = False
        rows.append(
            {
                **key_values,
                "best_gait": best["gait"],
                "best_condition_adapt_score": best["condition_adapt_score"],
                "second_gait": second["gait"],
                "second_condition_adapt_score": second["condition_adapt_score"],
                "score_gap": score_gap,
                "best_target_prob": best_prob,
                "second_target_prob": second_prob,
                "target_prob_gap": best_prob - second_prob,
                "confidence": confidence,
                "training_label_mode": training_label_mode,
                "use_for_training": use_for_training,
            }
        )
    return pd.DataFrame(rows).sort_values(group_cols)


def add_target_confidence_to_distribution(target_distribution, confidence):
    group_cols = grouping_columns(target_distribution, include_gait=False)
    cols = group_cols + [
        "confidence",
        "training_label_mode",
        "use_for_training",
        "score_gap",
        "target_prob_gap",
    ]
    return target_distribution.merge(confidence[cols], on=group_cols, how="left")


def gait_selection_summary(best_gait_by_speed):
    group_cols = ["condition"] if "condition" in best_gait_by_speed else []
    rows = []
    grouped = [(("all",), best_gait_by_speed)] if not group_cols else best_gait_by_speed.groupby(group_cols)
    for keys, group in grouped:
        if not isinstance(keys, tuple):
            keys = (keys,)
        key_values = dict(zip(group_cols or ["scope"], keys))
        total = len(group)
        counts = group["gait"].value_counts()
        for gait, count in counts.items():
            rows.append(
                {
                    **key_values,
                    "gait": gait,
                    "count": count,
                    "fraction": count / total if total else 0.0,
                }
            )
    return pd.DataFrame(rows)


def foot_pair_mapping():
    rows = [
        {
            "foot_index": index,
            "leg": leg,
            "evidence": "legged_robot.py desired_contact_states order FL,FR,RL,RR; go2.urdf foot links appear FL,FR,RL,RR",
        }
        for index, leg in FOOT_INDEX_TO_LEG.items()
    ]
    for pair, legs in PAIR_TO_LEGS.items():
        rows.append(
            {
                "foot_index": pair,
                "leg": legs,
                "evidence": "derived from foot index mapping",
            }
        )
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--quality-weights",
        default=",".join(f"{key}={value}" for key, value in DEFAULT_QUALITY_WEIGHTS.items()),
        help="Comma-separated normalized quality weights, e.g. done_rate=4,slip_penalty=2",
    )
    parser.add_argument(
        "--target-temperature",
        type=float,
        default=0.10,
        help="Softmax temperature for converting best gait scores at each speed into target probabilities",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir) if args.output_dir else input_path.parent / "intrinsic_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    metrics = numeric_metric_columns(df)
    if not metrics:
        raise ValueError("No intrinsic metric columns found")
    quality_weights = parse_weights(args.quality_weights)

    ranges = metric_ranges(df, metrics)
    sep_by_speed = separability(df, metrics, grouping_columns(df, include_gait=False))
    sep_global = separability(df, metrics, [])
    norm_summary = normalized_summary(df, ranges, metrics)
    signature = gait_signature(df, metrics)
    best_metric = best_by_metric(df, metrics)
    scores = intrinsic_scores(df, ranges, metrics)
    scores = add_weighted_quality_score(scores, quality_weights)
    scores = add_condition_adapt_score(
        scores,
        DEFAULT_CONDITION_SCORE_WEIGHTS,
        CONDITION_GATE_WEIGHTS,
        CONDITION_GATE_PENALTY,
    )
    component_summary = quality_component_summary(scores, quality_weights)
    component_dominance = quality_component_dominance(component_summary, quality_weights)
    weighted_best_by_speed_gait = best_weighted_by_speed_gait(scores)
    weighted_best_gait_by_speed = best_weighted_gait_by_speed(weighted_best_by_speed_gait)
    target_distribution = target_distribution_by_speed(
        weighted_best_by_speed_gait, args.target_temperature
    )
    condition_component = condition_component_summary(scores)
    condition_dominance = condition_component_dominance(condition_component)
    condition_best_by_speed_gait = best_condition_score_by_speed_gait(scores)
    condition_best_gait_by_speed = best_condition_score_gait_by_speed(
        condition_best_by_speed_gait
    )
    condition_target_distribution = condition_target_distribution_by_speed(
        condition_best_by_speed_gait, args.target_temperature
    )
    condition_confidence = condition_target_confidence(
        condition_best_by_speed_gait, condition_target_distribution
    )
    condition_target_distribution = add_target_confidence_to_distribution(
        condition_target_distribution, condition_confidence
    )
    condition_selection_summary = gait_selection_summary(condition_best_gait_by_speed)
    numeric_score_columns = [
        col
        for col in scores.select_dtypes(include="number").columns
        if col
        not in {
            "vx",
            "frequency",
            "footswing_height",
            "body_pitch",
            "stance_width",
            "phase",
            "offset",
            "bound",
        }
    ]
    score_summary = scores.groupby(grouping_columns(scores), as_index=False)[numeric_score_columns].mean()

    ranges.to_csv(output_dir / "metric_ranges.csv", index=False)
    sep_sort_cols = grouping_columns(df, include_gait=False) + ["separability"]
    sep_by_speed.sort_values(
        sep_sort_cols,
        ascending=[True] * (len(sep_sort_cols) - 1) + [False],
    ).to_csv(
        output_dir / "gait_separability_by_speed.csv", index=False
    )
    sep_global.sort_values("separability", ascending=False).to_csv(
        output_dir / "gait_separability_global.csv", index=False
    )
    norm_summary.to_csv(output_dir / "normalized_metric_summary.csv", index=False)
    signature.to_csv(output_dir / "gait_signature_by_speed.csv", index=False)
    best_metric.to_csv(output_dir / "best_gait_by_metric.csv", index=False)
    scores.to_csv(output_dir / "normalized_intrinsic_scores.csv", index=False)
    score_summary.to_csv(output_dir / "score_summary_by_speed_gait.csv", index=False)
    foot_pair_mapping().to_csv(output_dir / "foot_pair_mapping.csv", index=False)
    weights_table(quality_weights).to_csv(output_dir / "weighted_quality_weights.csv", index=False)
    condition_score_weights_table(
        DEFAULT_CONDITION_SCORE_WEIGHTS, CONDITION_GATE_WEIGHTS
    ).to_csv(output_dir / "condition_score_weights.csv", index=False)
    metric_documentation(metrics).to_csv(output_dir / "metric_documentation.csv", index=False)
    component_summary.to_csv(
        output_dir / "weighted_quality_components_by_speed_gait.csv", index=False
    )
    component_dominance.to_csv(
        output_dir / "weighted_quality_component_dominance.csv", index=False
    )
    weighted_best_by_speed_gait.to_csv(
        output_dir / "best_weighted_by_speed_gait.csv", index=False
    )
    weighted_best_gait_by_speed.to_csv(
        output_dir / "best_weighted_gait_by_speed.csv", index=False
    )
    target_distribution.to_csv(
        output_dir / "target_gait_distribution_by_speed.csv", index=False
    )
    condition_component.to_csv(
        output_dir / "condition_score_components_by_speed_gait.csv", index=False
    )
    condition_dominance.to_csv(
        output_dir / "condition_score_component_dominance.csv", index=False
    )
    condition_best_by_speed_gait.to_csv(
        output_dir / "best_condition_score_by_speed_gait.csv", index=False
    )
    condition_best_gait_by_speed.to_csv(
        output_dir / "best_condition_score_gait_by_speed.csv", index=False
    )
    condition_target_distribution.to_csv(
        output_dir / "target_condition_score_distribution_by_speed.csv", index=False
    )
    condition_confidence.to_csv(
        output_dir / "condition_target_confidence_by_speed.csv", index=False
    )
    condition_selection_summary.to_csv(
        output_dir / "condition_gait_selection_summary.csv", index=False
    )

    print(f"Analyzed {len(df)} rows and {len(metrics)} intrinsic metrics")
    print(f"Saved analysis to: {output_dir}")
    print("\nTop global gait-separating metrics:")
    print(
        sep_global.sort_values("separability", ascending=False)
        .head(12)[["metric", "separability", "between_gait_mean_var", "total_var"]]
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
