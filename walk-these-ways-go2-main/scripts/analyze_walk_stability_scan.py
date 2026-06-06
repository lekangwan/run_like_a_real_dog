import argparse
from pathlib import Path

import pandas as pd


STABILITY_WEIGHTS = {
    "done_rate": 3.0,
    "progress_deficit": 2.0,
    "orientation_penalty": 2.0,
    "lateral_vel_rms": 2.0,
    "base_z_vel_rms": 1.5,
    "roll_rate_rms": 1.0,
    "pitch_rate_rms": 1.0,
    "yaw_rate_rms": 1.0,
    "slip_penalty": 1.0,
    "scuffing_ratio": 1.0,
    "foot_impact_vel_mean": 0.75,
}


LOWER_IS_BETTER = set(STABILITY_WEIGHTS)


def normalize(values, lower_is_better=True):
    lo = values.min()
    hi = values.max()
    if pd.isna(lo) or pd.isna(hi) or abs(hi - lo) < 1e-12:
        return pd.Series(0.5, index=values.index)
    if lower_is_better:
        return (hi - values) / (hi - lo)
    return (values - lo) / (hi - lo)


def score_walk_rows(df):
    scored = df.copy()
    score = pd.Series(0.0, index=scored.index)
    weight_sum = 0.0
    used_metrics = []

    for metric, weight in STABILITY_WEIGHTS.items():
        if metric not in scored.columns:
            continue
        component = normalize(scored[metric].astype(float), metric in LOWER_IS_BETTER)
        scored[f"component_{metric}"] = component
        score += weight * component
        weight_sum += weight
        used_metrics.append(metric)

    if weight_sum <= 0:
        raise ValueError("No stability metrics were found in the input CSV.")

    scored["walk_stability_score"] = score / weight_sum
    scored["walk_stability_metric_count"] = len(used_metrics)
    return scored, used_metrics


def best_rows(scored):
    group_keys = ["condition", "vx"]
    idx = scored.groupby(group_keys)["walk_stability_score"].idxmax()
    return scored.loc[idx].sort_values(group_keys).reset_index(drop=True)


def play_command(row, args):
    command = [
        "CUDA_VISIBLE_DEVICES=0",
        "python3",
        "scripts/play_walk_gait_compare.py",
        "--condition",
        str(row["condition"]),
        "--vx",
        f"{float(row['vx']):.3g}",
        "--frequency",
        f"{float(row['frequency']):.3g}",
        "--duration",
        f"{float(row['duration']):.3g}",
        "--footswing-height",
        f"{float(row['footswing_height']):.3g}",
        "--body-pitch",
        f"{float(row['body_pitch']):.3g}",
        "--stance-width",
        f"{float(row['stance_width']):.3g}",
        "--steps",
        str(args.play_steps),
    ]
    if args.walk_only:
        command.extend(["--gaits", "walking"])
    return " ".join(command)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="logs/walk_stability_scan/template_eval_results.csv")
    parser.add_argument("--output-dir", default="logs/walk_stability_scan")
    parser.add_argument("--gait", default="walking")
    parser.add_argument("--min-forward-ratio", type=float, default=0.20)
    parser.add_argument("--max-done-rate", type=float, default=0.05)
    parser.add_argument("--play-steps", type=int, default=2000)
    parser.add_argument("--walk-only", action="store_true")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    if "gait" in df.columns:
        df = df[df["gait"] == args.gait].copy()
    if df.empty:
        raise ValueError(f"No rows found for gait={args.gait} in {input_path}")

    if "forward_distance_ratio" in df.columns:
        df = df[df["forward_distance_ratio"].astype(float) >= args.min_forward_ratio].copy()
    if "done_rate" in df.columns:
        df = df[df["done_rate"].astype(float) <= args.max_done_rate].copy()
    if df.empty:
        raise ValueError(
            "No rows passed the viability gate. Try lowering --min-forward-ratio or raising --max-done-rate."
        )

    scored, used_metrics = score_walk_rows(df)
    scored = scored.sort_values("walk_stability_score", ascending=False).reset_index(drop=True)
    best = best_rows(scored)
    top = scored.head(20).copy()

    scored.to_csv(output_dir / "walk_stability_scored.csv", index=False)
    best.to_csv(output_dir / "best_walk_stability.csv", index=False)
    top.to_csv(output_dir / "top_walk_stability.csv", index=False)

    best_row = best.sort_values("walk_stability_score", ascending=False).iloc[0]
    command = play_command(best_row, args)
    (output_dir / "play_best_walk_command.txt").write_text(command + "\n")

    print(f"Read: {input_path}")
    print(f"Scored rows: {len(scored)} using metrics: {', '.join(used_metrics)}")
    print(f"Wrote: {output_dir / 'walk_stability_scored.csv'}")
    print(f"Wrote: {output_dir / 'best_walk_stability.csv'}")
    print(f"Wrote: {output_dir / 'top_walk_stability.csv'}")
    print("\nBest walking candidate:")
    summary_cols = [
        "condition",
        "vx",
        "frequency",
        "duration",
        "footswing_height",
        "body_pitch",
        "stance_width",
        "walk_stability_score",
        "measured_vx",
        "forward_distance_ratio",
        "done_rate",
        "orientation_penalty",
        "lateral_vel_rms",
        "base_z_vel_rms",
        "progress_deficit",
    ]
    for column in summary_cols:
        if column in best_row:
            print(f"  {column}: {best_row[column]}")
    print("\nPlayback command:")
    print(command)


if __name__ == "__main__":
    main()
