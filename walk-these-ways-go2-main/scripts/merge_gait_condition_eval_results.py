import argparse
import json
from pathlib import Path

import pandas as pd


DEFAULT_KEEP_CONDITIONS = [
    "flat",
    "rough_slope",
    "push_hard",
    "push_forward",
    "push_backward",
    "push_left",
    "push_right",
    "push_lateral",
    "push_longitudinal",
    "push_down",
    "push_up",
    "ramp_up",
    "stepping_stones_easy",
]


def parse_conditions(value):
    return [item.strip() for item in value.split(",") if item.strip()]


def best_rows(df, group_keys, score_key="template_score"):
    idx = df.groupby(group_keys)[score_key].idxmax()
    return df.loc[idx].sort_values(group_keys).reset_index(drop=True)


def read_results(path, keep_conditions):
    df = pd.read_csv(path)
    if "condition" not in df.columns:
        raise ValueError(f"{path} does not contain a condition column")
    return df[df["condition"].isin(keep_conditions)].copy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base",
        default="logs/gait_condition_eval_v5_smooth_slope/template_eval_results.csv",
        help="Existing evaluation CSV to reuse.",
    )
    parser.add_argument(
        "--append",
        action="append",
        default=[],
        help="Additional evaluation CSV to append. Can be passed multiple times.",
    )
    parser.add_argument(
        "--keep-conditions",
        default=",".join(DEFAULT_KEEP_CONDITIONS),
        help="Comma-separated condition list to keep in the merged output.",
    )
    parser.add_argument(
        "--output-dir",
        default="logs/gait_condition_eval_v6_selected",
        help="Output directory for merged evaluation tables.",
    )
    args = parser.parse_args()

    keep_conditions = parse_conditions(args.keep_conditions)
    frames = [read_results(Path(args.base), keep_conditions)]
    for path in args.append:
        frames.append(read_results(Path(path), keep_conditions))

    merged = pd.concat(frames, ignore_index=True)
    if "duration" in merged.columns:
        merged["duration"] = merged["duration"].fillna(0.5)
    merged = merged.drop_duplicates(
        subset=[
            column
            for column in [
                "condition",
                "vx",
                "gait",
                "frequency",
                "duration",
                "footswing_height",
                "body_pitch",
                "stance_width",
            ]
            if column in merged.columns
        ],
        keep="last",
    )
    merged = merged.sort_values(["condition", "vx", "gait"]).reset_index(drop=True)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_dir / "template_eval_results.csv", index=False)
    best_rows(merged, ["vx"]).to_csv(output_dir / "best_by_speed.csv", index=False)
    best_rows(merged, ["vx", "gait"]).to_csv(output_dir / "best_by_speed_gait.csv", index=False)
    best_rows(merged, ["condition", "vx"]).to_csv(output_dir / "best_by_condition_speed.csv", index=False)
    best_rows(merged, ["condition", "vx", "gait"]).to_csv(
        output_dir / "best_by_condition_speed_gait.csv", index=False
    )
    with open(output_dir / "template_eval_results.json", "w") as file:
        json.dump(merged.to_dict(orient="records"), file, indent=2)

    print(f"Merged {len(merged)} rows into: {output_dir}")
    print(f"Kept conditions: {','.join(sorted(merged['condition'].unique()))}")


if __name__ == "__main__":
    main()
