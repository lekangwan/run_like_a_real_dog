import argparse
import csv
from collections import defaultdict
from pathlib import Path


BEHAVIOR_NAMES = (
    "frequency",
    "duration",
    "footswing_height",
    "stance_width",
    "body_pitch",
)


DEFAULT_EVAL = (
    "flat_trot_efficiency:1.0,"
    "ramp_up_trot_robustness:1.0,"
    "rough_slope_trot_robustness:1.0,"
    "push_lateral_pace_recovery:1.5,"
    "stepping_stones_easy_bound_highspeed:2.0"
)


def to_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def read_rows(path):
    with Path(path).open(newline="") as file:
        return list(csv.DictReader(file))


def write_csv(path, rows):
    if not rows:
        return
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_eval(text):
    items = set()
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        task_id, vx = item.split(":")
        items.add((task_id, round(float(vx), 6)))
    return items


def config_key(row):
    parts = [
        row.get("task_id", ""),
        f"{to_float(row.get('cmd_vx')):.6f}",
        row.get("gait", ""),
    ]
    for name in BEHAVIOR_NAMES:
        parts.append(f"{to_float(row.get(f'{name}_residual')):.6f}")
    return "|".join(parts)


def compact_config(row, selected_by):
    item = {
        "config_key": config_key(row),
        "task_id": row.get("task_id", ""),
        "condition": row.get("condition", ""),
        "cmd_vx": row.get("cmd_vx", ""),
        "target_gait": row.get("target_gait", ""),
        "gait": row.get("gait", ""),
        "gait_id": row.get("gait_id", ""),
        "grid_mode": row.get("grid_mode", ""),
        "selected_by": ";".join(selected_by),
    }
    for name in BEHAVIOR_NAMES:
        item[f"{name}_residual"] = row.get(f"{name}_residual", "")
        item[name] = row.get(name, "")
        mean_key = f"actual_{name}_mean"
        if mean_key in row:
            item[mean_key] = row.get(mean_key, "")
    for key, value in row.items():
        if key.startswith("score_") or key in (
            "neutral_score",
            "weighted_metric_reward_mean",
            "vx_abs_error_mean",
            "torque_penalty_mean",
            "slip_penalty_mean",
            "reward_impact_velocity_rms_mean",
            "scuffing_ratio_mean",
            "fall_rate",
            "transport_cost_proxy",
        ):
            item[f"source_{key}"] = value
    return item


def select_topk(rows, score_key, top_k):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["task_id"], round(to_float(row["cmd_vx"]), 6), row["gait"])].append(row)

    selected = []
    for key, group in grouped.items():
        if score_key not in group[0]:
            raise ValueError(f"Missing score column {score_key!r}")
        ranked = sorted(group, key=lambda row: to_float(row.get(score_key)), reverse=True)
        for rank, row in enumerate(ranked[:top_k], start=1):
            selected.append((row, f"{score_key}:rank{rank}:score{to_float(row.get(score_key)):.6f}"))
    return selected


def write_summary(path, configs, eval_items, score_keys, top_k):
    by_task = defaultdict(int)
    by_gait = defaultdict(int)
    for row in configs:
        by_task[row["task_id"]] += 1
        by_gait[row["gait"]] += 1
    lines = [
        "# Metric Sanity Config Selection",
        "",
        "This file selects a small representative config set for metric sanity audits.",
        "It is not a full fair grid and should not be used as final gait ranking evidence.",
        "",
        f"- eval_items: `{', '.join(f'{task}:{vx:g}' for task, vx in sorted(eval_items))}`",
        f"- score_keys: `{', '.join(score_keys)}`",
        f"- top_k_per_task_speed_gait_per_score: `{top_k}`",
        f"- selected_unique_configs: `{len(configs)}`",
        "",
        "## By Gait",
        "",
    ]
    for gait, count in sorted(by_gait.items()):
        lines.append(f"- {gait}: {count}")
    lines += ["", "## By Task", ""]
    for task, count in sorted(by_task.items()):
        lines.append(f"- {task}: {count}")
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Select a small representative config CSV for metric sanity audits. "
            "The output can be passed to evaluate_gait_target_fairness.py with "
            "--config-csv ... --eval-from-config."
        )
    )
    parser.add_argument("--input", required=True, help="fair_gait_grid_results.csv or held-out CSV")
    parser.add_argument("--eval", default=DEFAULT_EVAL)
    parser.add_argument(
        "--score-keys",
        default="weighted_metric_reward_mean,neutral_score",
        help="Comma-separated score columns used to select candidate configs.",
    )
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    eval_items = parse_eval(args.eval)
    score_keys = [item.strip() for item in args.score_keys.split(",") if item.strip()]
    rows = [
        row
        for row in read_rows(args.input)
        if (row.get("task_id"), round(to_float(row.get("cmd_vx")), 6)) in eval_items
    ]
    if not rows:
        raise ValueError("No rows matched --eval task-speed pairs")

    selected_by = defaultdict(list)
    by_key = {}
    for score_key in score_keys:
        for row, label in select_topk(rows, score_key, args.top_k):
            key = config_key(row)
            by_key.setdefault(key, row)
            selected_by[key].append(label)

    configs = [
        compact_config(row, selected_by[key])
        for key, row in sorted(by_key.items(), key=lambda item: item[0])
    ]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "metric_sanity_config_requests.csv", configs)
    write_summary(output_dir / "summary.md", configs, eval_items, score_keys, args.top_k)


if __name__ == "__main__":
    main()
