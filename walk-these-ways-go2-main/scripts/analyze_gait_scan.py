import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


NUMERIC_COLUMNS = [
    "vx",
    "vy",
    "yaw",
    "frequency",
    "footswing_height",
    "body_pitch",
    "stance_width",
    "vx_abs_error",
    "vy_abs_error",
    "yaw_abs_error",
    "roll_pitch_rms_proxy",
    "torque_sq",
    "action_delta_sq",
    "feet_slip_proxy",
    "fall_count",
]


DEFAULT_WEIGHTS = {
    "vx_abs_error": 1.2,
    "vy_abs_error": 0.35,
    "yaw_abs_error": 0.25,
    "roll_pitch_rms_proxy": 1.0,
    "torque_sq": 0.25,
    "action_delta_sq": 0.35,
    "feet_slip_proxy": 0.6,
    "fall_count": 3.0,
}


def load_rows(path):
    rows = []
    with open(path, newline="") as file:
        for row in csv.DictReader(file):
            for column in NUMERIC_COLUMNS:
                row[column] = float(row[column])
            rows.append(row)
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows


def add_scores(rows, weights):
    mins = {name: min(row[name] for row in rows) for name in weights}
    maxs = {name: max(row[name] for row in rows) for name in weights}

    for row in rows:
        score = 0.0
        for name, weight in weights.items():
            scale = maxs[name] - mins[name]
            normalized = 0.0 if scale == 0 else (row[name] - mins[name]) / scale
            score += weight * normalized
        row["score"] = score


def group_by(rows, key):
    groups = defaultdict(list)
    for row in rows:
        groups[row[key]].append(row)
    return dict(groups)


def mean(rows, key):
    return sum(row[key] for row in rows) / len(rows)


def summarize_param(rows, param):
    groups = group_by(rows, param)
    summary = []
    for value, group in sorted(groups.items(), key=lambda item: str(item[0])):
        summary.append(
            {
                "value": value,
                "count": len(group),
                "mean_score": mean(group, "score"),
                "mean_vx_abs_error": mean(group, "vx_abs_error"),
                "mean_roll_pitch_rms_proxy": mean(group, "roll_pitch_rms_proxy"),
                "mean_torque_sq": mean(group, "torque_sq"),
                "mean_action_delta_sq": mean(group, "action_delta_sq"),
                "mean_feet_slip_proxy": mean(group, "feet_slip_proxy"),
                "fall_sum": sum(row["fall_count"] for row in group),
            }
        )
    return sorted(summary, key=lambda item: item["mean_score"])


def compact_row(row):
    keys = [
        "vx",
        "gait",
        "frequency",
        "footswing_height",
        "body_pitch",
        "stance_width",
        "vx_abs_error",
        "vy_abs_error",
        "yaw_abs_error",
        "roll_pitch_rms_proxy",
        "torque_sq",
        "action_delta_sq",
        "feet_slip_proxy",
        "fall_count",
        "score",
    ]
    return {key: row[key] for key in keys}


def build_summary(rows, top_k):
    by_vx = group_by(rows, "vx")
    summary = {
        "total_rows": len(rows),
        "no_fall_rows": sum(1 for row in rows if row["fall_count"] == 0),
        "by_vx": {},
    }

    for vx, group in sorted(by_vx.items()):
        no_fall = [row for row in group if row["fall_count"] == 0]
        vx_summary = {
            "count": len(group),
            "no_fall_count": len(no_fall),
            "fall_sum": sum(row["fall_count"] for row in group),
            "top_score_no_fall": [compact_row(row) for row in sorted(no_fall, key=lambda row: row["score"])[:top_k]],
            "top_vx_error_no_fall": [
                compact_row(row) for row in sorted(no_fall, key=lambda row: row["vx_abs_error"])[:top_k]
            ],
            "parameter_means_no_fall": {},
        }

        for param in ["gait", "frequency", "footswing_height", "stance_width", "body_pitch"]:
            vx_summary["parameter_means_no_fall"][param] = summarize_param(no_fall, param) if no_fall else []

        summary["by_vx"][str(vx)] = vx_summary

    return summary


def write_csv(summary, path):
    rows = []
    for vx, vx_summary in summary["by_vx"].items():
        for rank, row in enumerate(vx_summary["top_score_no_fall"], start=1):
            output = {"vx_group": vx, "rank": rank, **row}
            rows.append(output)

    if not rows:
        return

    with open(path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def print_summary(summary, top_k):
    print(f"Total rows: {summary['total_rows']}")
    print(f"No-fall rows: {summary['no_fall_rows']}")
    for vx, vx_summary in summary["by_vx"].items():
        print(f"\nVX = {vx}")
        print(
            f"  no-fall: {vx_summary['no_fall_count']}/{vx_summary['count']}, "
            f"fall_sum: {vx_summary['fall_sum']:.1f}"
        )
        print(f"  top {top_k} no-fall by score:")
        for row in vx_summary["top_score_no_fall"]:
            print(
                "    "
                f"{row['gait']:8s} "
                f"freq={row['frequency']:.2f} "
                f"h={row['footswing_height']:.2f} "
                f"w={row['stance_width']:.2f} "
                f"vxerr={row['vx_abs_error']:.4f} "
                f"ori={row['roll_pitch_rms_proxy']:.5f} "
                f"slip={row['feet_slip_proxy']:.5f} "
                f"score={row['score']:.4f}"
            )

        print("  best mean score by parameter:")
        for param, items in vx_summary["parameter_means_no_fall"].items():
            if not items:
                continue
            best = items[0]
            print(f"    {param}: {best['value']} ({best['count']} rows, mean_score={best['mean_score']:.4f})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="logs/gait_param_scan/scan_results.csv")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--top-k", type=int, default=8)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir) if args.output_dir else input_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(input_path)
    add_scores(rows, DEFAULT_WEIGHTS)
    summary = build_summary(rows, args.top_k)

    json_path = output_dir / "scan_summary.json"
    csv_path = output_dir / "scan_top_score_no_fall.csv"
    with open(json_path, "w") as file:
        json.dump(summary, file, indent=2)
    write_csv(summary, csv_path)

    print_summary(summary, args.top_k)
    print(f"\nSaved JSON summary: {json_path}")
    print(f"Saved top-score CSV: {csv_path}")


if __name__ == "__main__":
    main()
