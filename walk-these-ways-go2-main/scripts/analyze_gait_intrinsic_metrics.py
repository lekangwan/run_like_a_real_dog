import argparse
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
    range_by_metric = ranges.set_index("metric")
    normalized = df[["vx", "gait"]].copy()
    for metric in metrics:
        p05 = range_by_metric.loc[metric, "p05"]
        p95 = range_by_metric.loc[metric, "p95"]
        scale = p95 - p05
        if scale == 0:
            normalized[metric] = 0.0
        else:
            normalized[metric] = ((df[metric] - p05) / scale).clip(0.0, 1.0)

    return normalized.groupby(["vx", "gait"], as_index=False)[metrics].mean()


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir) if args.output_dir else input_path.parent / "intrinsic_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    metrics = numeric_metric_columns(df)
    if not metrics:
        raise ValueError("No intrinsic metric columns found")

    ranges = metric_ranges(df, metrics)
    sep_by_speed = separability(df, metrics, ["vx"])
    sep_global = separability(df, metrics, [])
    norm_summary = normalized_summary(df, ranges, metrics)
    signature = gait_signature(df, metrics)
    best_metric = best_by_metric(df, metrics)

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
