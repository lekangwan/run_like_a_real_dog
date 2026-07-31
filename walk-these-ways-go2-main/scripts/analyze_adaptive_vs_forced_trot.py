"""Aggregate repeated adaptive-selector versus forced-trot evaluations."""

import argparse
import csv
from pathlib import Path
from statistics import mean, stdev


METRICS = (
    "reward_mean",
    "vx_err_mean",
    "done_rate",
    "mechanical_power_abs",
    "slip_penalty",
    "contact_slip_penalty",
    "impact_velocity_rms",
    "scuffing_ratio",
    "orientation_penalty",
    "lateral_position_penalty",
)


def parse_seeds(text):
    seeds = [int(item.strip()) for item in text.split(",") if item.strip()]
    if not seeds:
        raise ValueError("Provide at least one seed")
    return seeds


def parse_items(text):
    if not text:
        return None
    items = set()
    for item in text.split(","):
        task_id, speed = item.strip().rsplit(":", 1)
        items.add((task_id, float(speed)))
    return items


def read_rows(path):
    with Path(path).open(newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return {(row["task_id"], float(row["cmd_vx"])): row for row in rows}


def population_std(values):
    return stdev(values) if len(values) > 1 else 0.0


def summarize(root, seeds, adaptive_template, baseline_template, include_items=None):
    paired = {}
    for seed in seeds:
        adaptive_path = root / adaptive_template.format(seed=seed) / "independent_eval_summary.csv"
        baseline_path = root / baseline_template.format(seed=seed) / "independent_eval_summary.csv"
        adaptive = read_rows(adaptive_path)
        baseline = read_rows(baseline_path)
        if include_items is not None:
            adaptive = {key: row for key, row in adaptive.items() if key in include_items}
            baseline = {key: row for key, row in baseline.items() if key in include_items}
            if not adaptive:
                raise ValueError(f"No requested task-speed rows found for seed {seed}")
        if adaptive.keys() != baseline.keys():
            raise ValueError(f"Task-speed mismatch for seed {seed}")
        for key in adaptive:
            paired.setdefault(key, []).append((seed, adaptive[key], baseline[key]))

    rows = []
    for key in sorted(paired):
        values = paired[key]
        first = values[0][1]
        row = {
            "task_id": key[0],
            "condition": first["condition"],
            "target_gait": first["target_gait"],
            "cmd_vx": key[1],
            "num_seeds": len(values),
        }
        for metric in METRICS:
            adaptive_values = [float(adaptive[metric]) for _, adaptive, _ in values]
            baseline_values = [float(baseline[metric]) for _, _, baseline in values]
            deltas = [adaptive_value - baseline_value for adaptive_value, baseline_value in zip(adaptive_values, baseline_values)]
            row[f"adaptive_{metric}_mean"] = mean(adaptive_values)
            row[f"forced_trot_{metric}_mean"] = mean(baseline_values)
            row[f"delta_{metric}_mean"] = mean(deltas)
            row[f"delta_{metric}_std"] = population_std(deltas)
            row[f"delta_{metric}_positive_count"] = sum(delta > 0.0 for delta in deltas)
        rows.append(row)
    return rows


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path, rows, seeds):
    lines = [
        "# Adaptive Selector Versus Forced Trot",
        "",
        f"- seeds: {', '.join(str(seed) for seed in seeds)}",
        "- delta convention: adaptive selector minus forced trot",
        "- lower is better for velocity error, done rate, power, slip, impact, and scuffing.",
        "",
        "| task | vx | reward delta | vx error delta | done delta | power delta | contact slip delta | impact delta | scuff delta |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['task_id']} | {row['cmd_vx']:.2f} "
            f"| {row['delta_reward_mean_mean']:+.6f} "
            f"| {row['delta_vx_err_mean_mean']:+.6f} "
            f"| {row['delta_done_rate_mean']:+.6f} "
            f"| {row['delta_mechanical_power_abs_mean']:+.3f} "
            f"| {row['delta_contact_slip_penalty_mean']:+.6f} "
            f"| {row['delta_impact_velocity_rms_mean']:+.6f} "
            f"| {row['delta_scuffing_ratio_mean']:+.6f} |"
        )
    Path(path).write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="Directory containing one subdirectory per seed and control mode.")
    parser.add_argument("--seeds", required=True, help="Comma-separated random seeds.")
    parser.add_argument("--adaptive-template", default="seed{seed}_adaptive")
    parser.add_argument("--baseline-template", default="seed{seed}_forced_trot")
    parser.add_argument(
        "--include",
        default=None,
        help="Optional comma-separated task_id:cmd_vx rows to aggregate.",
    )
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--output-markdown", default=None)
    args = parser.parse_args()

    root = Path(args.root)
    seeds = parse_seeds(args.seeds)
    rows = summarize(
        root,
        seeds,
        args.adaptive_template,
        args.baseline_template,
        parse_items(args.include),
    )
    csv_path = Path(args.output_csv) if args.output_csv else root / "paired_summary.csv"
    markdown_path = Path(args.output_markdown) if args.output_markdown else root / "paired_summary.md"
    write_csv(csv_path, rows)
    write_markdown(markdown_path, rows, seeds)
    print(f"Wrote: {csv_path}")
    print(f"Wrote: {markdown_path}")


if __name__ == "__main__":
    main()
