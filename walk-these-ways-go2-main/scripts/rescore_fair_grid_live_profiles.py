import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import torch

from go2_gym.envs.wrappers.high_level_reward_metrics import (
    CANONICAL_REWARD_NAMES,
    UNIFIED_REWARD_PROFILES,
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


def profile_weights(profile_name):
    if profile_name not in UNIFIED_REWARD_PROFILES:
        raise ValueError(f"Unknown live reward profile: {profile_name}")
    weights = {name: 0.0 for name in CANONICAL_REWARD_NAMES}
    weights.update(UNIFIED_REWARD_PROFILES[profile_name])
    return weights


def score_row(row, weights):
    total = 0.0
    weight_sum = 0.0
    parts = {}
    missing = []
    for metric in CANONICAL_REWARD_NAMES:
        weight = float(weights.get(metric, 0.0))
        if weight <= 0.0:
            continue
        column = f"score_{metric}"
        if column not in row:
            missing.append(column)
            continue
        value = to_float(row.get(column), 0.0)
        parts[f"weighted_{metric}"] = weight * value
        total += weight * value
        weight_sum += weight
    if missing:
        raise ValueError(f"Missing required score columns: {sorted(set(missing))}")
    return total / max(weight_sum, 1e-9), parts


def add_profile_scores(rows, profile_names):
    output = []
    weight_table = {name: profile_weights(name) for name in profile_names}
    for row in rows:
        item = dict(row)
        for profile_name, weights in weight_table.items():
            score, parts = score_row(row, weights)
            item[f"{profile_name}_score"] = score
            for key, value in parts.items():
                item[f"{profile_name}_{key}"] = value
        output.append(item)
    return output


def best_rows(rows, keys, score_key):
    best = {}
    for row in rows:
        key = tuple(row[k] for k in keys)
        if key not in best or to_float(row[score_key]) > to_float(best[key][score_key]):
            best[key] = row
    return list(best.values())


def softmax(values, temperature):
    values = [float(v) for v in values]
    max_value = max(values)
    exps = [math.exp((value - max_value) / max(temperature, 1e-9)) for value in values]
    total = sum(exps)
    return [value / max(total, 1e-12) for value in exps]


def compact_best_row(row, profile_name):
    keys = [
        "profile",
        "task_id",
        "condition",
        "cmd_vx",
        "gait",
        "gait_id",
        f"{profile_name}_score",
        "weighted_metric_reward_mean",
        "neutral_score",
        "vx_abs_error_mean",
        "fall_rate",
        "lateral_offset_abs_mean",
        "scuffing_ratio_mean",
        "foot_impact_vel_rms",
        "transport_cost_proxy",
        "actual_frequency_mean",
        "actual_duration_mean",
        "actual_footswing_height_mean",
        "actual_stance_width_mean",
        "actual_body_pitch_mean",
    ]
    item = {"profile": profile_name}
    for key in keys:
        if key == "profile":
            continue
        item[key] = row.get(key, "")
    return item


def summarize_profile(rows, profile_name, temperature):
    score_key = f"{profile_name}_score"
    best_by_gait = best_rows(rows, ["task_id", "cmd_vx", "gait"], score_key)
    best_by_task = best_rows(best_by_gait, ["task_id", "cmd_vx"], score_key)

    decisions = []
    soft_rows = []
    grouped = defaultdict(list)
    for row in best_by_gait:
        grouped[(row["task_id"], row["cmd_vx"])].append(row)

    for (task_id, vx), group in sorted(grouped.items()):
        ranked = sorted(group, key=lambda row: to_float(row[score_key]), reverse=True)
        top = ranked[0]
        second = ranked[1] if len(ranked) > 1 else ranked[0]
        margin = to_float(top[score_key]) - to_float(second[score_key])
        decisions.append(
            {
                "profile": profile_name,
                "task_id": task_id,
                "condition": top.get("condition", ""),
                "cmd_vx": vx,
                "best_gait": top["gait"],
                "second_gait": second["gait"],
                "best_score": to_float(top[score_key]),
                "second_score": to_float(second[score_key]),
                "score_margin": margin,
            }
        )
        probs = softmax([to_float(row[score_key]) for row in ranked], temperature)
        for row, prob in zip(ranked, probs):
            soft_rows.append(
                {
                    "profile": profile_name,
                    "task_id": task_id,
                    "condition": row.get("condition", ""),
                    "cmd_vx": vx,
                    "gait": row["gait"],
                    "score": to_float(row[score_key]),
                    "probability": prob,
                }
            )

    return best_by_gait, best_by_task, decisions, soft_rows


def write_summary(path, profile_names, decisions):
    lines = [
        "# Live Profile Fair-Grid Re-Score",
        "",
        "Rows are re-scored from saved `score_<metric>` columns in a completed",
        "fair gait grid. No IsaacGym rollout is performed here.",
        "",
    ]
    grouped = defaultdict(list)
    for row in decisions:
        grouped[row["profile"]].append(row)
    for profile_name in profile_names:
        rows = grouped[profile_name]
        gait_counts = defaultdict(int)
        for row in rows:
            gait_counts[row["best_gait"]] += 1
        lines += [
            f"## {profile_name}",
            "",
            "- top gait counts: "
            + ", ".join(f"{gait}={count}" for gait, count in sorted(gait_counts.items())),
            "",
            "| task | vx | best | second | margin |",
            "|---|---:|---|---|---:|",
        ]
        for row in sorted(rows, key=lambda item: (item["task_id"], float(item["cmd_vx"]))):
            lines.append(
                f"| {row['task_id']} | {float(row['cmd_vx']):.2f} | "
                f"{row['best_gait']} | {row['second_gait']} | {row['score_margin']:.4f} |"
            )
        lines.append("")
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(
        description="Re-score a completed fair gait grid with live reward profiles."
    )
    parser.add_argument("--input", required=True, help="fair_gait_grid_results.csv path")
    parser.add_argument(
        "--profiles",
        default="canonical_efficiency_candidate,canonical_balanced_candidate",
        help="Comma-separated names from train_high_level_oracle_ppo.UNIFIED_REWARD_PROFILES.",
    )
    parser.add_argument("--softmax-temperature", type=float, default=0.03)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    profile_names = [item.strip() for item in args.profiles.split(",") if item.strip()]
    rows = read_rows(args.input)
    scored_rows = add_profile_scores(rows, profile_names)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "rescored_all_rows.csv", scored_rows)

    all_best_by_gait = []
    all_best_by_task = []
    all_decisions = []
    all_soft = []
    for profile_name in profile_names:
        best_by_gait, best_by_task, decisions, soft_rows = summarize_profile(
            scored_rows,
            profile_name,
            args.softmax_temperature,
        )
        all_best_by_gait.extend(compact_best_row(row, profile_name) for row in best_by_gait)
        all_best_by_task.extend(compact_best_row(row, profile_name) for row in best_by_task)
        all_decisions.extend(decisions)
        all_soft.extend(soft_rows)

    write_csv(output_dir / "best_by_task_speed_gait.csv", all_best_by_gait)
    write_csv(output_dir / "best_by_task_speed.csv", all_best_by_task)
    write_csv(output_dir / "profile_decisions.csv", all_decisions)
    write_csv(output_dir / "soft_distribution.csv", all_soft)
    write_summary(output_dir / "summary.md", profile_names, all_decisions)
    with (output_dir / "run_config.json").open("w") as file:
        json.dump(vars(args), file, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
