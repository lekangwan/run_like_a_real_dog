import argparse
import csv
from collections import defaultdict
from pathlib import Path


TASK_REWARD_NAMES = (
    "progress",
    "yaw_tracking",
    "orientation",
    "pitch_rate",
    "roll_rate",
    "yaw_rate",
    "lateral_drift",
    "vertical_bounce",
    "slip",
    "energy",
    "clearance",
    "gait_stability",
    "action_smoothness",
    "action_magnitude",
    "action_boundary_margin",
    "survival",
)

BASE_METRIC_WEIGHTS = {
    "progress": 1.0,
    "yaw_tracking": 0.3,
    "orientation": 0.3,
    "lateral_drift": 0.8,
    "gait_stability": 0.4,
    "action_smoothness": 0.7,
    "action_magnitude": 0.6,
    "action_boundary_margin": 0.8,
    "survival": 2.0,
}

TASK_REWARD_FOCUS_WEIGHTS = {
    "low_slip": {"slip": 1.2},
    "low_vertical_bounce": {"vertical_bounce": 0.8},
    "low_lateral_drift": {"lateral_drift": 0.8},
    "orientation_stability": {"orientation": 0.5},
    "orientation_stability_strong": {"orientation": 0.9},
    "orientation_stability_mild": {"orientation": 0.3},
    "pitch_control": {"pitch_rate": 0.8, "orientation": 0.4},
    "low_roll_pitch_rate": {"roll_rate": 0.6, "pitch_rate": 0.6},
    "low_roll_rate": {"roll_rate": 0.8},
    "low_yaw_rate": {"yaw_rate": 0.6},
    "low_done_rate": {"survival": 1.0},
    "foot_clearance": {"clearance": 0.6},
    "low_scuffing": {"clearance": 0.15},
    "low_energy": {"energy": 0.6},
}

DEFAULT_AUDIT_CSV = (
    "runs/high_level_oracle_gait/fixed_gait_live_reward_audit/"
    "20260612_221845/fixed_gait_live_reward.csv"
)
DEFAULT_TASK_MAP = (
    "logs/gait_condition_eval_v8_mainline/training_task_map/"
    "training_task_map_by_speed.csv"
)


def task_reward_weights_from_focus(reward_focus):
    weights = {name: 0.0 for name in TASK_REWARD_NAMES}
    for name, value in BASE_METRIC_WEIGHTS.items():
        weights[name] += value
    for token in str(reward_focus).split(","):
        token = token.strip()
        for name, value in TASK_REWARD_FOCUS_WEIGHTS.get(token, {}).items():
            weights[name] += value
    return {name: min(value, 2.0) for name, value in weights.items()}


def read_task_focus(task_map_path):
    focus_by_task = {}
    with Path(task_map_path).open(newline="") as file:
        for row in csv.DictReader(file):
            if row["use_for_training"] != "yes":
                continue
            focus_by_task.setdefault(row["task_id"], row.get("reward_focus", ""))
    return focus_by_task


def read_audit_rows(path):
    rows = []
    with Path(path).open(newline="") as file:
        for row in csv.DictReader(file):
            for key, value in list(row.items()):
                try:
                    row[key] = float(value)
                except (TypeError, ValueError):
                    pass
            rows.append(row)
    return rows


def group_by_task_speed(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["task_id"], float(row["cmd_vx"]))].append(row)
    return grouped


def choose_comparisons(grouped):
    comparisons = []
    for (task_id, vx), rows in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1])):
        ranked = sorted(rows, key=lambda row: row["weighted_metric_reward"], reverse=True)
        if task_id == "flat_trot_efficiency":
            target = "trotting"
            competitor = next(row["requested_gait"] for row in ranked if row["requested_gait"] != target)
        elif task_id == "ramp_up_trot_robustness":
            target, competitor = "trotting", "pronking"
        elif task_id == "rough_slope_trot_robustness":
            target, competitor = "trotting", "pronking"
        elif task_id == "push_lateral_pace_recovery":
            target, competitor = "pacing", "trotting"
        elif task_id == "stepping_stones_easy_bound_highspeed":
            target, competitor = "bounding", "pacing"
        else:
            continue
        comparisons.append((task_id, vx, target, competitor))
    return comparisons


def build_gap_rows(grouped, focus_by_task, comparisons):
    output = []
    for task_id, vx, target_gait, competitor_gait in comparisons:
        rows = grouped[(task_id, vx)]
        by_gait = {row["requested_gait"]: row for row in rows}
        target = by_gait[target_gait]
        competitor = by_gait[competitor_gait]
        weights = task_reward_weights_from_focus(focus_by_task[task_id])
        weight_sum = sum(weights.values())
        target_reward = float(target["weighted_metric_reward"])
        competitor_reward = float(competitor["weighted_metric_reward"])

        for metric in TASK_REWARD_NAMES:
            weight = weights[metric]
            if weight <= 0.0:
                continue
            target_score = float(target.get(f"score_{metric}", 0.0))
            competitor_score = float(competitor.get(f"score_{metric}", 0.0))
            raw_gap = target_score - competitor_score
            weighted_gap = weight * raw_gap / weight_sum
            output.append(
                {
                    "task_id": task_id,
                    "cmd_vx": vx,
                    "target_gait": target_gait,
                    "competitor_gait": competitor_gait,
                    "metric": metric,
                    "weight": weight,
                    "weight_fraction": weight / weight_sum,
                    "target_score": target_score,
                    "competitor_score": competitor_score,
                    "raw_gap": raw_gap,
                    "weighted_gap": weighted_gap,
                    "target_reward": target_reward,
                    "competitor_reward": competitor_reward,
                    "reward_gap": target_reward - competitor_reward,
                }
            )
    return output


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def fmt(value):
    return f"{float(value):.4f}"


def write_summary(path, rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["task_id"], row["cmd_vx"], row["target_gait"], row["competitor_gait"])].append(row)

    lines = [
        "# Fixed-Gait Reward Gap Decomposition",
        "",
        "Positive weighted_gap means the metric helps the target gait. "
        "Negative weighted_gap means it helps the competitor.",
        "",
    ]
    for key, group in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1])):
        task_id, vx, target, competitor = key
        group_sorted = sorted(group, key=lambda row: abs(row["weighted_gap"]), reverse=True)
        reward_gap = group_sorted[0]["reward_gap"]
        lines += [
            f"## {task_id} vx={vx:.2f}: {target} vs {competitor}",
            "",
            f"- target_reward: {fmt(group_sorted[0]['target_reward'])}",
            f"- competitor_reward: {fmt(group_sorted[0]['competitor_reward'])}",
            f"- target_minus_competitor: {fmt(reward_gap)}",
            "",
            "| metric | weight | target_score | competitor_score | raw_gap | weighted_gap |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for row in group_sorted:
            lines.append(
                f"| {row['metric']} "
                f"| {fmt(row['weight'])} "
                f"| {fmt(row['target_score'])} "
                f"| {fmt(row['competitor_score'])} "
                f"| {fmt(row['raw_gap'])} "
                f"| {fmt(row['weighted_gap'])} |"
            )
        lines.append("")
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-csv", default=DEFAULT_AUDIT_CSV)
    parser.add_argument("--task-map", default=DEFAULT_TASK_MAP)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    audit_csv = Path(args.audit_csv)
    output_dir = Path(args.output_dir) if args.output_dir else audit_csv.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    focus_by_task = read_task_focus(args.task_map)
    rows = read_audit_rows(audit_csv)
    grouped = group_by_task_speed(rows)
    comparisons = choose_comparisons(grouped)
    gap_rows = build_gap_rows(grouped, focus_by_task, comparisons)

    csv_path = output_dir / "weighted_gap_decomposition.csv"
    summary_path = output_dir / "weighted_gap_decomposition.md"
    write_csv(csv_path, gap_rows)
    write_summary(summary_path, gap_rows)
    print(f"Wrote: {csv_path}")
    print(f"Wrote: {summary_path}")


if __name__ == "__main__":
    main()
