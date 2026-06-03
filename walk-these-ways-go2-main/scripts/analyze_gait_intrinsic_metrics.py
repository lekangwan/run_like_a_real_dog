import argparse
import math
from pathlib import Path

import pandas as pd


IDENTITY_COLUMNS = {
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
    "done_rate",
]

DEFAULT_QUALITY_WEIGHTS = {
    "done_rate": 4.0,
    "slip_penalty": 2.0,
    "orientation_penalty": 2.0,
    "action_delta_sq": 1.0,
    "torque_penalty": 1.0,
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
    "slip_penalty",
    "orientation_penalty",
    "vertical_velocity_penalty",
    "action_delta_sq",
    "done_rate",
]


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
    normalized = robust_normalized_frame(df, ranges, metrics, keep_columns=["vx", "gait"])
    return normalized.groupby(["vx", "gait"], as_index=False)[metrics].mean()


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
    return df.groupby(["vx", "gait"], as_index=False).agg(agg)


def best_by_metric(df, metrics):
    rows = []
    for vx, speed_group in df.groupby("vx"):
        for metric in metrics:
            means = speed_group.groupby("gait")[metric].mean().sort_values()
            low_gait = means.index[0]
            high_gait = means.index[-1]
            rows.append(
                {
                    "vx": vx,
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
        weighted_cost = weighted_cost + scored[metric] * weight
    scored["weighted_quality_cost_norm"] = weighted_cost / total_weight
    scored["weighted_quality_score"] = 1.0 - scored["weighted_quality_cost_norm"]
    return scored


def best_weighted_by_speed_gait(scores):
    idx = scores.groupby(["vx", "gait"])["weighted_quality_score"].idxmax()
    return scores.loc[idx].sort_values(["vx", "gait"]).reset_index(drop=True)


def best_weighted_gait_by_speed(best_by_speed_gait):
    idx = best_by_speed_gait.groupby("vx")["weighted_quality_score"].idxmax()
    return best_by_speed_gait.loc[idx].sort_values("vx").reset_index(drop=True)


def target_distribution_by_speed(best_by_speed_gait, temperature):
    if temperature <= 0:
        raise ValueError("--target-temperature must be positive")

    rows = []
    for vx, group in best_by_speed_gait.groupby("vx"):
        scores = group["weighted_quality_score"]
        max_score = scores.max()
        logits = [(score - max_score) / temperature for score in scores]
        exp_logits = [math.exp(value) for value in logits]
        denom = sum(exp_logits)
        for (_, row), prob in zip(group.iterrows(), exp_logits):
            out = row.to_dict()
            out["target_temperature"] = temperature
            out["target_prob"] = prob / denom if denom > 0 else 0.0
            rows.append(out)
    return pd.DataFrame(rows).sort_values(["vx", "target_prob"], ascending=[True, False])


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
    sep_by_speed = separability(df, metrics, ["vx"])
    sep_global = separability(df, metrics, [])
    norm_summary = normalized_summary(df, ranges, metrics)
    signature = gait_signature(df, metrics)
    best_metric = best_by_metric(df, metrics)
    scores = intrinsic_scores(df, ranges, metrics)
    scores = add_weighted_quality_score(scores, quality_weights)
    weighted_best_by_speed_gait = best_weighted_by_speed_gait(scores)
    weighted_best_gait_by_speed = best_weighted_gait_by_speed(weighted_best_by_speed_gait)
    target_distribution = target_distribution_by_speed(
        weighted_best_by_speed_gait, args.target_temperature
    )
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
    score_summary = scores.groupby(["vx", "gait"], as_index=False)[numeric_score_columns].mean()

    ranges.to_csv(output_dir / "metric_ranges.csv", index=False)
    sep_by_speed.sort_values(["vx", "separability"], ascending=[True, False]).to_csv(
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
    weighted_best_by_speed_gait.to_csv(
        output_dir / "best_weighted_by_speed_gait.csv", index=False
    )
    weighted_best_gait_by_speed.to_csv(
        output_dir / "best_weighted_gait_by_speed.csv", index=False
    )
    target_distribution.to_csv(
        output_dir / "target_gait_distribution_by_speed.csv", index=False
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
