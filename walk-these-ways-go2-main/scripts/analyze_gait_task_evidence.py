import argparse
from pathlib import Path

import pandas as pd


METRIC_CATALOG = [
    {
        "metric": "forward_distance_ratio",
        "category": "viability",
        "direction": "higher",
        "meaning": "实际前进距离 / 目标前进距离。这里主要作为通过性检查，不作为步态风格本身的证据。",
    },
    {
        "metric": "progress_deficit",
        "category": "viability",
        "direction": "lower",
        "meaning": "目标前进距离缺口。越小表示越能在该地形上走出去。",
    },
    {
        "metric": "done_rate",
        "category": "viability",
        "direction": "lower",
        "meaning": "评估窗口内 reset/done 比例。越小表示越不容易失败。",
    },
    {
        "metric": "slip_penalty",
        "category": "terrain_interaction",
        "direction": "lower",
        "meaning": "接触脚切向滑动惩罚。低摩擦地形尤其重要。",
    },
    {
        "metric": "scuffing_ratio",
        "category": "terrain_interaction",
        "direction": "lower",
        "meaning": "摆动脚离地过低、容易擦地的比例。粗糙地形和障碍物上重要。",
    },
    {
        "metric": "swing_foot_clearance_mean",
        "category": "terrain_interaction",
        "direction": "higher",
        "meaning": "摆动脚平均离地/离地形高度。越高表示跨越余量越大，但过高也可能浪费能量。",
    },
    {
        "metric": "foot_impact_vel_mean",
        "category": "impact",
        "direction": "lower",
        "meaning": "新触地瞬间足端向下速度均值。越小表示落脚冲击更柔和。",
    },
    {
        "metric": "foot_impact_vel_rms",
        "category": "impact",
        "direction": "lower",
        "meaning": "新触地瞬间足端向下速度 RMS，更强调大的冲击。",
    },
    {
        "metric": "foot_impact_rate",
        "category": "impact",
        "direction": "lower",
        "meaning": "单位评估步数的新触地事件数。用于辅助判断接触是否频繁跳变。",
    },
    {
        "metric": "peak_contact_force",
        "category": "impact",
        "direction": "lower",
        "meaning": "评估窗口内最大单脚接触力。越小表示最坏冲击风险更低。",
    },
    {
        "metric": "orientation_penalty",
        "category": "stability",
        "direction": "lower",
        "meaning": "机身姿态偏离惩罚。越小表示身体姿态更稳。",
    },
    {
        "metric": "roll_rate_rms",
        "category": "stability",
        "direction": "lower",
        "meaning": "横滚角速度 RMS。pace 是否导致左右晃动，主要看它。",
    },
    {
        "metric": "pitch_rate_rms",
        "category": "stability",
        "direction": "lower",
        "meaning": "俯仰角速度 RMS。pronk/bound 的前后弹跳，主要看它。",
    },
    {
        "metric": "yaw_rate_rms",
        "category": "stability",
        "direction": "lower",
        "meaning": "偏航角速度 RMS。越小表示不容易扭转跑偏。",
    },
    {
        "metric": "gravity_x_rms",
        "category": "stability",
        "direction": "lower",
        "meaning": "重力在机体系 x 方向分量 RMS，近似反映俯仰姿态扰动。",
    },
    {
        "metric": "gravity_y_rms",
        "category": "stability",
        "direction": "lower",
        "meaning": "重力在机体系 y 方向分量 RMS，近似反映横滚姿态扰动。",
    },
    {
        "metric": "lateral_vel_rms",
        "category": "stability",
        "direction": "lower",
        "meaning": "机身侧向速度 RMS。越小表示不容易横向漂移。",
    },
    {
        "metric": "base_z_vel_rms",
        "category": "stability",
        "direction": "lower",
        "meaning": "机身竖直速度 RMS。越小表示上下弹跳少。",
    },
    {
        "metric": "torque_penalty",
        "category": "energy",
        "direction": "lower",
        "meaning": "力矩惩罚。越小表示关节输出负担更小。",
    },
    {
        "metric": "mechanical_power_abs",
        "category": "energy",
        "direction": "lower",
        "meaning": "关节机械功率绝对值均值。越小表示运动整体能耗强度更低。",
    },
    {
        "metric": "transport_cost_proxy",
        "category": "energy",
        "direction": "lower",
        "meaning": "单位前进速度下的机械功率代理指标。越小表示移动效率更好。",
    },
    {
        "metric": "flight_ratio",
        "category": "gait_style",
        "direction": "higher",
        "meaning": "四脚都离地的时间比例。用于识别跳跃/腾空类步态，不一定越高越好。",
    },
    {
        "metric": "all_contact_ratio",
        "category": "gait_style",
        "direction": "higher",
        "meaning": "四脚同时接触地面的比例。用于识别保守支撑或 pronk 落地特征。",
    },
    {
        "metric": "contact_count",
        "category": "gait_style",
        "direction": "higher",
        "meaning": "平均接触脚数量。用于判断支撑面积大小，不直接表示优劣。",
    },
    {
        "metric": "phase_match_error",
        "category": "gait_style",
        "direction": "lower",
        "meaning": "实际接触节律和命令步态模板的差异。越小表示该模板更容易被底层策略实现。",
    },
    {
        "metric": "trotting_contact_match",
        "category": "gait_style",
        "direction": "higher",
        "meaning": "实际接触模式接近 trot 的程度。",
    },
    {
        "metric": "pacing_contact_match",
        "category": "gait_style",
        "direction": "higher",
        "meaning": "实际接触模式接近 pace 的程度。",
    },
    {
        "metric": "bounding_contact_match",
        "category": "gait_style",
        "direction": "higher",
        "meaning": "实际接触模式接近 bound 的程度。",
    },
    {
        "metric": "pronking_contact_match",
        "category": "gait_style",
        "direction": "higher",
        "meaning": "实际接触模式接近 pronk 的程度。",
    },
]


TASK_OBJECTIVES = [
    {
        "objective": "flat_trot_efficiency",
        "condition": "flat",
        "expected_gait": "trotting",
        "intent": "平地默认高效稳定行走，验证 trot 是否确实是合理基线。",
        "weights": {
            "transport_cost_proxy": 1.5,
            "mechanical_power_abs": 1.0,
            "base_z_vel_rms": 1.0,
            "lateral_vel_rms": 1.0,
            "slip_penalty": 0.75,
            "phase_match_error": 0.75,
            "forward_distance_ratio": 0.5,
        },
    },
    {
        "objective": "slippery_pronk_or_sync",
        "condition": "very_low_friction",
        "expected_gait": "pronking",
        "intent": "低摩擦地面重点看低滑移和少横向漂移，检验同步支撑类步态是否有优势。",
        "weights": {
            "slip_penalty": 2.5,
            "lateral_vel_rms": 1.25,
            "yaw_rate_rms": 1.0,
            "orientation_penalty": 1.0,
            "progress_deficit": 1.0,
            "done_rate": 0.75,
            "pronking_contact_match": 0.5,
        },
    },
    {
        "objective": "ramp_up_stability",
        "condition": "ramp_up",
        "expected_gait": "trotting",
        "intent": "连续上坡坡道优先看姿态稳定、俯仰控制、少滑移、能耗和可通过性；坡面无台阶、无中心平台，作为 to-real 必需鲁棒场景。",
        "weights": {
            "orientation_penalty": 1.5,
            "gravity_x_rms": 1.25,
            "pitch_rate_rms": 1.25,
            "slip_penalty": 1.0,
            "progress_deficit": 1.0,
            "transport_cost_proxy": 0.75,
            "mechanical_power_abs": 0.75,
            "base_z_vel_rms": 0.75,
            "phase_match_error": 0.5,
            "trotting_contact_match": 0.5,
        },
    },
    {
        "objective": "rough_slope_stability",
        "condition": "rough_slope",
        "expected_gait": "",
        "intent": "粗糙斜坡优先看姿态稳定、少滑移、少擦地和可通过性，用来检验 pace 是否真的站得住。",
        "weights": {
            "orientation_penalty": 1.5,
            "pitch_rate_rms": 1.0,
            "roll_rate_rms": 1.0,
            "gravity_x_rms": 1.0,
            "slip_penalty": 1.0,
            "scuffing_ratio": 0.75,
            "swing_foot_clearance_mean": 0.75,
            "progress_deficit": 1.0,
        },
    },
    {
        "objective": "rough_mid_clearance",
        "condition": "rough_mid",
        "expected_gait": "",
        "intent": "中等粗糙地面重点看摆动脚余量、擦地风险、冲击和可通过性。",
        "weights": {
            "swing_foot_clearance_mean": 1.5,
            "scuffing_ratio": 1.5,
            "foot_impact_vel_mean": 1.0,
            "base_z_vel_rms": 0.75,
            "progress_deficit": 1.0,
            "phase_match_error": 0.5,
        },
    },
    {
        "objective": "obstacle_bound_hypothesis",
        "condition": "discrete_obstacles_low",
        "expected_gait": "bounding",
        "intent": "显式检验“障碍物适合 bound”这个假设是否被数据支持。",
        "weights": {
            "swing_foot_clearance_mean": 1.75,
            "scuffing_ratio": 1.5,
            "foot_impact_vel_mean": 0.75,
            "progress_deficit": 1.0,
            "done_rate": 0.75,
            "flight_ratio": 0.5,
            "bounding_contact_match": 0.5,
        },
    },
    {
        "objective": "stepping_stones_easy_bound_clearance",
        "condition": "stepping_stones_easy",
        "expected_gait": "bounding",
        "intent": "较温和离散踏石地形的高速通过测试。重点看能否保持前进、少摔倒、足端清障、少擦地和姿态稳定；只在高速差距明确时作为 bound 标签。",
        "weights": {
            "forward_distance_ratio": 2.0,
            "done_rate": 1.5,
            "swing_foot_clearance_mean": 1.5,
            "scuffing_ratio": 1.5,
            "orientation_penalty": 1.0,
            "lateral_vel_rms": 1.0,
            "base_z_vel_rms": 0.75,
            "foot_impact_vel_mean": 0.75,
            "phase_match_error": 0.5,
        },
    },
    {
        "objective": "stairs_up_clearance",
        "condition": "stairs_up_low",
        "expected_gait": "",
        "intent": "低台阶上楼梯的泛化测试。由于底层策略未明确针对楼梯训练，这里只做证据观察。",
        "weights": {
            "swing_foot_clearance_mean": 1.75,
            "scuffing_ratio": 1.5,
            "foot_impact_vel_mean": 1.0,
            "progress_deficit": 1.25,
            "orientation_penalty": 1.0,
            "base_z_vel_rms": 0.75,
        },
    },
    {
        "objective": "stairs_down_soft_impact",
        "condition": "stairs_down_low",
        "expected_gait": "",
        "intent": "低台阶下楼梯的泛化测试。重点看落脚冲击和姿态稳定，不直接做 hard label。",
        "weights": {
            "foot_impact_vel_mean": 1.75,
            "foot_impact_vel_rms": 1.25,
            "peak_contact_force": 1.0,
            "orientation_penalty": 1.0,
            "progress_deficit": 1.25,
            "base_z_vel_rms": 0.75,
        },
    },
    {
        "objective": "push_bound_recovery",
        "condition": "push_hard",
        "expected_gait": "bounding",
        "intent": "强推扰恢复测试，检验 bound 是否在恢复前进和抑制横向漂移上有优势。",
        "weights": {
            "progress_deficit": 1.75,
            "lateral_vel_rms": 1.25,
            "orientation_penalty": 1.0,
            "roll_rate_rms": 1.0,
            "yaw_rate_rms": 0.75,
            "done_rate": 1.0,
            "bounding_contact_match": 0.5,
        },
    },
    {
        "objective": "push_recovery_general",
        "condition": "directed_push",
        "expected_gait": "",
        "intent": "定向推扰的中性恢复指标，不预设某种步态；重点看恢复前进、少摔倒、少漂移、姿态稳定、少弹跳和低滑移。",
        "weights": {
            "forward_distance_ratio": 3.0,
            "done_rate": 3.0,
            "lateral_vel_rms": 1.5,
            "orientation_penalty": 1.5,
            "roll_rate_rms": 1.0,
            "pitch_rate_rms": 1.0,
            "yaw_rate_rms": 1.0,
            "base_z_vel_rms": 1.0,
            "slip_penalty": 1.0,
            "transport_cost_proxy": 1.0,
            "foot_impact_rate": 0.5,
        },
    },
    {
        "objective": "push_longitudinal_recovery",
        "condition": "push_longitudinal",
        "expected_gait": "",
        "intent": "纵向推扰恢复测试，混合前向和后向速度冲击。前后方向对前进任务不完全对称，因此需要先观察再决定标签。",
        "weights": {
            "progress_deficit": 1.75,
            "lateral_vel_rms": 1.0,
            "orientation_penalty": 1.0,
            "roll_rate_rms": 1.0,
            "yaw_rate_rms": 0.75,
            "done_rate": 1.0,
        },
    },
    {
        "objective": "push_lateral_recovery",
        "condition": "push_lateral",
        "expected_gait": "",
        "intent": "横向推扰恢复测试，混合左/右侧速度冲击。重点看横向漂移、roll/yaw 扰动和恢复前进能力。",
        "weights": {
            "progress_deficit": 1.5,
            "lateral_vel_rms": 1.5,
            "orientation_penalty": 1.0,
            "roll_rate_rms": 1.25,
            "yaw_rate_rms": 1.0,
            "done_rate": 1.0,
        },
    },
    {
        "objective": "push_down_recovery",
        "condition": "push_down",
        "expected_gait": "",
        "intent": "正上方向下冲击测试。重点看垂向弹跳、姿态稳定、摔倒率和恢复前进能力。",
        "weights": {
            "base_z_vel_rms": 1.5,
            "orientation_penalty": 1.25,
            "progress_deficit": 1.0,
            "roll_rate_rms": 1.0,
            "pitch_rate_rms": 1.0,
            "done_rate": 1.0,
        },
    },
    {
        "objective": "push_forward_recovery",
        "condition": "push_forward",
        "expected_gait": "",
        "intent": "定向前向推扰恢复测试。先观察哪种步态在前向速度扰动后恢复更好，不预设 hard label。",
        "weights": {
            "progress_deficit": 1.75,
            "lateral_vel_rms": 1.25,
            "orientation_penalty": 1.0,
            "roll_rate_rms": 1.0,
            "yaw_rate_rms": 0.75,
            "done_rate": 1.0,
        },
    },
    {
        "objective": "push_backward_recovery",
        "condition": "push_backward",
        "expected_gait": "",
        "intent": "定向后向推扰恢复测试。重点看被向后扰动后恢复前进和姿态稳定的能力。",
        "weights": {
            "progress_deficit": 1.75,
            "lateral_vel_rms": 1.25,
            "orientation_penalty": 1.0,
            "roll_rate_rms": 1.0,
            "yaw_rate_rms": 0.75,
            "done_rate": 1.0,
        },
    },
    {
        "objective": "push_left_recovery",
        "condition": "push_left",
        "expected_gait": "",
        "intent": "定向左侧推扰恢复测试。重点看横向漂移、roll/yaw 扰动和恢复前进能力。",
        "weights": {
            "progress_deficit": 1.5,
            "lateral_vel_rms": 1.5,
            "orientation_penalty": 1.0,
            "roll_rate_rms": 1.25,
            "yaw_rate_rms": 1.0,
            "done_rate": 1.0,
        },
    },
    {
        "objective": "push_right_recovery",
        "condition": "push_right",
        "expected_gait": "",
        "intent": "定向右侧推扰恢复测试。与左侧推扰成对验证，观察是否存在方向相关的步态优势。",
        "weights": {
            "progress_deficit": 1.5,
            "lateral_vel_rms": 1.5,
            "orientation_penalty": 1.0,
            "roll_rate_rms": 1.25,
            "yaw_rate_rms": 1.0,
            "done_rate": 1.0,
        },
    },
    {
        "objective": "push_up_recovery",
        "condition": "push_up",
        "expected_gait": "",
        "intent": "正下方向上冲击测试。该场景偏仿真诊断，重点看垂向弹跳、姿态稳定、摔倒率和恢复前进能力。",
        "weights": {
            "base_z_vel_rms": 1.5,
            "orientation_penalty": 1.25,
            "progress_deficit": 1.0,
            "roll_rate_rms": 1.0,
            "pitch_rate_rms": 1.0,
            "done_rate": 1.0,
        },
    },
]


def available_catalog(df):
    rows = [row for row in METRIC_CATALOG if row["metric"] in df.columns]
    return pd.DataFrame(rows)


def normalize_scores(values, direction):
    lo = values.min()
    hi = values.max()
    if pd.isna(lo) or pd.isna(hi) or abs(hi - lo) < 1e-12:
        return pd.Series(0.5, index=values.index)
    if direction == "higher":
        return (values - lo) / (hi - lo)
    if direction == "lower":
        return (hi - values) / (hi - lo)
    raise ValueError(f"Unknown direction: {direction}")


def metric_ranges(df, catalog):
    rows = []
    for _, spec in catalog.iterrows():
        metric = spec["metric"]
        values = df[metric].dropna()
        rows.append(
            {
                "metric": metric,
                "category": spec["category"],
                "direction": spec["direction"],
                "min": values.min(),
                "p05": values.quantile(0.05),
                "p25": values.quantile(0.25),
                "median": values.quantile(0.50),
                "p75": values.quantile(0.75),
                "p95": values.quantile(0.95),
                "max": values.max(),
                "robust_range_p95_p05": values.quantile(0.95) - values.quantile(0.05),
                "std": values.std(ddof=0),
            }
        )
    return pd.DataFrame(rows)


def group_metric_means(df, catalog):
    group_cols = ["condition", "vx", "gait"]
    rows = []
    for _, spec in catalog.iterrows():
        metric = spec["metric"]
        direction = spec["direction"]
        means = df.groupby(group_cols)[metric].mean().reset_index(name="metric_mean")
        best_values = (
            df.groupby(group_cols)[metric]
            .agg(lambda x: x.max() if direction == "higher" else x.min())
            .reset_index(name="metric_best_config_value")
        )
        merged = means.merge(best_values, on=group_cols)
        merged["metric"] = metric
        merged["category"] = spec["category"]
        merged["direction"] = direction
        rows.append(merged)
    return pd.concat(rows, ignore_index=True)


def metric_rank_table(metric_means):
    ranked_parts = []
    keys = ["condition", "vx", "metric"]
    for _, part in metric_means.groupby(keys):
        direction = part["direction"].iloc[0]
        ascending = direction == "lower"
        ranked = part.copy()
        ranked["rank"] = ranked["metric_mean"].rank(method="min", ascending=ascending)
        ranked["metric_score_norm"] = normalize_scores(ranked["metric_mean"], direction)

        best_row = ranked.sort_values("rank").iloc[0]
        second_score = ranked["metric_score_norm"].nlargest(2).iloc[-1]
        ranked["best_gait"] = best_row["gait"]
        ranked["best_metric_mean"] = best_row["metric_mean"]
        ranked["best_score_gap_to_second"] = best_row["metric_score_norm"] - second_score
        ranked["gap_to_best_value"] = (ranked["metric_mean"] - best_row["metric_mean"]).abs()
        ranked_parts.append(ranked)
    return pd.concat(ranked_parts, ignore_index=True).sort_values(
        ["condition", "vx", "metric", "rank", "gait"]
    )


def objective_rows(df, catalog):
    metric_to_direction = dict(zip(catalog["metric"], catalog["direction"]))
    metric_to_category = dict(zip(catalog["metric"], catalog["category"]))
    base_means = df.groupby(["condition", "vx", "gait"]).mean(numeric_only=True).reset_index()
    component_rows = []
    summary_rows = []

    for objective in TASK_OBJECTIVES:
        condition = objective["condition"]
        sub = base_means[base_means["condition"] == condition].copy()
        if sub.empty:
            continue
        usable_metrics = [
            metric for metric in objective["weights"] if metric in sub.columns and metric in metric_to_direction
        ]
        if not usable_metrics:
            continue

        for vx, vx_part in sub.groupby("vx"):
            gait_scores = pd.DataFrame(
                {
                    "gait": vx_part["gait"].values,
                    "condition": condition,
                    "vx": vx,
                    "objective": objective["objective"],
                }
            )
            gait_scores["weighted_score_sum"] = 0.0
            gait_scores["weight_sum"] = 0.0

            for metric in usable_metrics:
                direction = metric_to_direction[metric]
                scores = normalize_scores(vx_part[metric], direction)
                weight = objective["weights"][metric]
                values = vx_part[metric].reset_index(drop=True)
                metric_rank = values.rank(method="min", ascending=(direction == "lower"))
                best_idx = scores.reset_index(drop=True).idxmax()
                second_score = scores.reset_index(drop=True).nlargest(2).iloc[-1]
                metric_range = values.max() - values.min()
                for i, gait in enumerate(vx_part["gait"].values):
                    component_rows.append(
                        {
                            "objective": objective["objective"],
                            "condition": condition,
                            "vx": vx,
                            "gait": gait,
                            "metric": metric,
                            "category": metric_to_category[metric],
                            "direction": direction,
                            "weight": weight,
                            "metric_mean": values.iloc[i],
                            "metric_range_across_gaits": metric_range,
                            "metric_score_norm": scores.reset_index(drop=True).iloc[i],
                            "metric_rank": int(metric_rank.iloc[i]),
                            "metric_best_gait": vx_part["gait"].reset_index(drop=True).iloc[best_idx],
                            "metric_best_score_gap_to_second": scores.reset_index(drop=True).iloc[best_idx] - second_score,
                        }
                    )
                gait_scores["weighted_score_sum"] += weight * scores.reset_index(drop=True)
                gait_scores["weight_sum"] += weight

            gait_scores["objective_score"] = gait_scores["weighted_score_sum"] / gait_scores["weight_sum"].clip(lower=1e-12)
            gait_scores["rank"] = gait_scores["objective_score"].rank(method="min", ascending=False).astype(int)
            best = gait_scores.sort_values(["rank", "gait"]).iloc[0]
            second_score = gait_scores["objective_score"].nlargest(2).iloc[-1]
            expected_gait = objective["expected_gait"]
            metric_components = pd.DataFrame(component_rows)
            current_components = metric_components[
                (metric_components["objective"] == objective["objective"])
                & (metric_components["condition"] == condition)
                & (metric_components["vx"] == vx)
            ]

            for _, row in gait_scores.iterrows():
                gait_components = current_components[current_components["gait"] == row["gait"]]
                strong_wins = gait_components[
                    (gait_components["metric_rank"] == 1)
                    & (gait_components["metric_best_score_gap_to_second"] >= 0.10)
                ]["metric"].tolist()
                weak_wins = gait_components[gait_components["metric_rank"] == 1]["metric"].tolist()
                summary_rows.append(
                    {
                        "objective": objective["objective"],
                        "condition": condition,
                        "vx": vx,
                        "gait": row["gait"],
                        "objective_score": row["objective_score"],
                        "rank": row["rank"],
                        "best_gait": best["gait"],
                        "best_score": best["objective_score"],
                        "best_score_gap_to_second": best["objective_score"] - second_score,
                        "expected_gait": expected_gait,
                        "is_expected_gait": bool(expected_gait and row["gait"] == expected_gait),
                        "metric_win_count": int((gait_components["metric_rank"] == 1).sum()),
                        "strong_metric_wins": ",".join(strong_wins),
                        "all_metric_wins": ",".join(weak_wins),
                        "intent": objective["intent"],
                    }
                )

    return pd.DataFrame(component_rows), pd.DataFrame(summary_rows)


def hypothesis_support(summary):
    rows = []
    for _, row in summary[summary["is_expected_gait"]].iterrows():
        if row["rank"] == 1 and row["best_score_gap_to_second"] >= 0.05:
            support = "supported"
        elif row["rank"] == 1:
            support = "weak_supported"
        elif row["rank"] == 2 and (row["best_score"] - row["objective_score"]) < 0.05:
            support = "ambiguous"
        else:
            support = "not_supported"
        rows.append(
            {
                "objective": row["objective"],
                "condition": row["condition"],
                "vx": row["vx"],
                "expected_gait": row["expected_gait"],
                "expected_rank": row["rank"],
                "expected_score": row["objective_score"],
                "best_gait": row["best_gait"],
                "best_score": row["best_score"],
                "score_gap_expected_to_best": row["best_score"] - row["objective_score"],
                "best_score_gap_to_second": row["best_score_gap_to_second"],
                "metric_win_count": row["metric_win_count"],
                "strong_metric_wins": row["strong_metric_wins"],
                "all_metric_wins": row["all_metric_wins"],
                "support": support,
                "intent": row["intent"],
            }
        )
    return pd.DataFrame(rows)


def write_readme(output_dir):
    text = """# Gait Task Evidence

This folder compares gait families by interpretable task metrics, instead of trusting one
weighted quality score.

Files:

- `metric_catalog.csv`: metric meaning, category, and direction.
- `metric_ranges.csv`: global value range for each metric.
- `gait_metric_summary.csv`: mean and best-config value for each condition / speed / gait / metric.
- `metric_rank_by_condition_speed.csv`: per-metric gait ranking inside each condition and speed.
- `objective_metric_components.csv`: normalized component scores for each task objective.
- `objective_score_by_condition_speed_gait.csv`: weighted objective score for each gait.
- `hypothesis_support.csv`: whether a manually proposed task -> gait hypothesis is supported.

Use `hypothesis_support.csv` as a warning table, not as a final label table. A hypothesis
should only become a hard training label when it is supported by several meaningful metrics
and the corresponding terrain is within the low-level policy's capability.
"""
    (output_dir / "README.md").write_text(text)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="logs/gait_condition_eval_v6_selected/template_eval_results.csv",
        help="Path to template_eval_results.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="logs/gait_condition_eval_v6_selected/task_metric_evidence",
        help="Directory for evidence tables",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    catalog = available_catalog(df)
    catalog.to_csv(output_dir / "metric_catalog.csv", index=False)
    metric_ranges(df, catalog).to_csv(output_dir / "metric_ranges.csv", index=False)

    means = group_metric_means(df, catalog)
    means.to_csv(output_dir / "gait_metric_summary.csv", index=False)

    ranked = metric_rank_table(means)
    ranked.to_csv(output_dir / "metric_rank_by_condition_speed.csv", index=False)

    components, summary = objective_rows(df, catalog)
    components.to_csv(output_dir / "objective_metric_components.csv", index=False)
    summary.to_csv(output_dir / "objective_score_by_condition_speed_gait.csv", index=False)

    best = summary.sort_values(["objective", "condition", "vx", "rank", "gait"]).groupby(
        ["objective", "condition", "vx"], as_index=False
    ).first()
    best.to_csv(output_dir / "objective_best_gait_by_condition_speed.csv", index=False)

    hypothesis_support(summary).to_csv(output_dir / "hypothesis_support.csv", index=False)
    write_readme(output_dir)

    print(f"Wrote gait task evidence to: {output_dir}")


if __name__ == "__main__":
    main()
