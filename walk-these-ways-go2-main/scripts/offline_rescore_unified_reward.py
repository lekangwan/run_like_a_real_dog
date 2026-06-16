import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


DEFAULT_FAIR_GRID = (
    "runs/high_level_oracle_gait/fair_target_gait_audit/"
    "20260614_training_range_action_grid/fair_gait_grid_results.csv"
)

SCORE_COLUMNS = {
    "tracking": "progress_score",
    "yaw": "yaw_score",
    "orientation": "orientation_score",
    "lateral": "lateral_score",
    "slip": "slip_score",
    "energy": "energy_score",
    "impact": "impact_score",
    "scuff": "scuffing_score",
    "smoothness": "smoothness_score",
    "survival": "survival_score",
    "clearance": "score_clearance",
    "boundary": "score_action_boundary_margin",
}

RAW_METRICS = (
    "vx_abs_error_mean",
    "fall_rate",
    "lateral_offset_abs_mean",
    "scuffing_ratio_mean",
    "foot_impact_vel_sum_mean",
    "transport_cost_proxy",
    "foot_impact_vel_rms",
    "orientation_rms",
    "lateral_vel_rms",
)

CANDIDATE_WEIGHTS = {
    "balanced": {
        "tracking": 1.5,
        "yaw": 0.3,
        "orientation": 1.0,
        "lateral": 0.8,
        "slip": 1.0,
        "energy": 1.0,
        "impact": 1.0,
        "scuff": 1.0,
        "smoothness": 0.5,
        "survival": 2.0,
        "clearance": 0.5,
        "boundary": 0.4,
    },
    "efficiency": {
        "tracking": 2.0,
        "yaw": 0.3,
        "orientation": 0.8,
        "lateral": 0.4,
        "slip": 0.7,
        "energy": 2.0,
        "impact": 1.5,
        "scuff": 0.5,
        "smoothness": 0.8,
        "survival": 1.5,
        "clearance": 0.2,
        "boundary": 0.5,
    },
    "robustness": {
        "tracking": 1.0,
        "yaw": 0.5,
        "orientation": 1.5,
        "lateral": 1.5,
        "slip": 1.2,
        "energy": 0.5,
        "impact": 0.8,
        "scuff": 1.0,
        "smoothness": 0.4,
        "survival": 3.0,
        "clearance": 0.5,
        "boundary": 0.4,
    },
    "contact_safety": {
        "tracking": 0.8,
        "yaw": 0.2,
        "orientation": 0.8,
        "lateral": 0.8,
        "slip": 1.5,
        "energy": 0.6,
        "impact": 2.0,
        "scuff": 2.0,
        "smoothness": 0.4,
        "survival": 2.0,
        "clearance": 1.5,
        "boundary": 0.4,
    },
}


def to_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def read_rows(path):
    rows = []
    with Path(path).open(newline="") as file:
        for row in csv.DictReader(file):
            rows.append(row)
    return rows


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


def score_row(row, weights):
    total = 0.0
    weight_sum = 0.0
    parts = {}
    for metric, weight in weights.items():
        if weight <= 0.0:
            continue
        column = SCORE_COLUMNS[metric]
        value = to_float(row.get(column), 0.0)
        parts[f"score_{metric}"] = value
        parts[f"weighted_{metric}"] = weight * value
        total += weight * value
        weight_sum += weight
    score = total / max(weight_sum, 1e-9)
    return score, parts


def add_candidate_scores(rows):
    output = []
    for row in rows:
        item = dict(row)
        for candidate, weights in CANDIDATE_WEIGHTS.items():
            score, parts = score_row(row, weights)
            item[f"{candidate}_score"] = score
            for key, value in parts.items():
                item[f"{candidate}_{key}"] = value
        output.append(item)
    return output


def group_rows(rows, keys):
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in keys)].append(row)
    return grouped


def best_rows(rows, keys, score_key):
    best = {}
    for row in rows:
        key = tuple(row[k] for k in keys)
        if key not in best or to_float(row[score_key]) > to_float(best[key][score_key]):
            best[key] = row
    return list(best.values())


def softmax(values, temperature):
    max_value = max(values)
    exps = [math.exp((v - max_value) / max(temperature, 1e-9)) for v in values]
    total = sum(exps)
    return [v / total for v in exps]


def margin_label(margin):
    if margin < 0.01:
        return "tie_or_noise"
    if margin < 0.03:
        return "weak_advantage"
    return "clear_advantage"


def compact_row(row, candidate):
    keys = [
        "candidate",
        "task_id",
        "condition",
        "cmd_vx",
        "gait",
        "gait_id",
        f"{candidate}_score",
        "neutral_score",
        "weighted_metric_reward_mean",
        "vx_abs_error_mean",
        "fall_rate",
        "lateral_offset_abs_mean",
        "scuffing_ratio_mean",
        "foot_impact_vel_sum_mean",
        "transport_cost_proxy",
        "actual_frequency_mean",
        "actual_duration_mean",
        "actual_footswing_height_mean",
        "actual_stance_width_mean",
        "actual_body_pitch_mean",
    ]
    item = {"candidate": candidate}
    for key in keys:
        if key == "candidate":
            continue
        item[key] = row.get(key, "")
    return item


def build_outputs(rows, temperature):
    all_best_by_gait = []
    all_best_by_task = []
    soft_rows = []
    decision_rows = []
    gap_rows = []

    for candidate in CANDIDATE_WEIGHTS:
        score_key = f"{candidate}_score"
        best_by_gait = best_rows(
            rows,
            ("task_id", "cmd_vx", "gait"),
            score_key,
        )
        for row in best_by_gait:
            all_best_by_gait.append(compact_row(row, candidate))

        grouped = group_rows(best_by_gait, ("task_id", "cmd_vx"))
        for (task_id, cmd_vx), group in sorted(grouped.items(), key=lambda x: (x[0][0], float(x[0][1]))):
            ranked = sorted(group, key=lambda row: to_float(row[score_key]), reverse=True)
            top = ranked[0]
            second = ranked[1]
            margin = to_float(top[score_key]) - to_float(second[score_key])
            all_best_by_task.append(compact_row(top, candidate))
            decision_rows.append(
                {
                    "candidate": candidate,
                    "task_id": task_id,
                    "condition": top["condition"],
                    "cmd_vx": cmd_vx,
                    "top_gait": top["gait"],
                    "second_gait": second["gait"],
                    "score_margin": margin,
                    "margin_label": margin_label(margin),
                    "top_score": to_float(top[score_key]),
                    "second_score": to_float(second[score_key]),
                    "top_neutral_score": to_float(top.get("neutral_score")),
                    "top_live_weighted_reward": to_float(top.get("weighted_metric_reward_mean")),
                    "top_vx_abs_error": to_float(top.get("vx_abs_error_mean")),
                    "top_fall_rate": to_float(top.get("fall_rate")),
                    "top_lateral_offset_abs": to_float(top.get("lateral_offset_abs_mean")),
                    "top_scuffing_ratio": to_float(top.get("scuffing_ratio_mean")),
                    "top_foot_impact_vel_sum": to_float(top.get("foot_impact_vel_sum_mean")),
                    "top_transport_cost_proxy": to_float(top.get("transport_cost_proxy")),
                    "top_frequency": to_float(top.get("actual_frequency_mean")),
                    "top_duration": to_float(top.get("actual_duration_mean")),
                    "top_footswing_height": to_float(top.get("actual_footswing_height_mean")),
                    "top_stance_width": to_float(top.get("actual_stance_width_mean")),
                    "top_body_pitch": to_float(top.get("actual_body_pitch_mean")),
                }
            )

            scores = [to_float(row[score_key]) for row in ranked]
            probs = softmax(scores, temperature)
            for rank, (row, prob) in enumerate(zip(ranked, probs), start=1):
                soft_rows.append(
                    {
                        "candidate": candidate,
                        "task_id": task_id,
                        "condition": row["condition"],
                        "cmd_vx": cmd_vx,
                        "rank": rank,
                        "gait": row["gait"],
                        "score": to_float(row[score_key]),
                        "soft_prob": prob,
                    }
                )

            for metric, column in SCORE_COLUMNS.items():
                top_value = to_float(top.get(column), 0.0)
                second_value = to_float(second.get(column), 0.0)
                gap_rows.append(
                    {
                        "candidate": candidate,
                        "task_id": task_id,
                        "cmd_vx": cmd_vx,
                        "top_gait": top["gait"],
                        "second_gait": second["gait"],
                        "metric": metric,
                        "top_score": top_value,
                        "second_score": second_value,
                        "top_minus_second": top_value - second_value,
                    }
                )
            for metric in RAW_METRICS:
                top_value = to_float(top.get(metric), 0.0)
                second_value = to_float(second.get(metric), 0.0)
                gap_rows.append(
                    {
                        "candidate": candidate,
                        "task_id": task_id,
                        "cmd_vx": cmd_vx,
                        "top_gait": top["gait"],
                        "second_gait": second["gait"],
                        "metric": metric,
                        "top_score": top_value,
                        "second_score": second_value,
                        "top_minus_second": top_value - second_value,
                    }
                )

    return all_best_by_gait, all_best_by_task, soft_rows, decision_rows, gap_rows


def candidate_stats(decision_rows):
    stats = []
    by_candidate = group_rows(decision_rows, ("candidate",))
    for (candidate,), rows in sorted(by_candidate.items()):
        gait_counts = Counter(row["top_gait"] for row in rows)
        margin_counts = Counter(row["margin_label"] for row in rows)
        mean_margin = sum(to_float(row["score_margin"]) for row in rows) / len(rows)
        mean_vx_error = sum(to_float(row["top_vx_abs_error"]) for row in rows) / len(rows)
        mean_fall = sum(to_float(row["top_fall_rate"]) for row in rows) / len(rows)
        mean_energy = sum(to_float(row["top_transport_cost_proxy"]) for row in rows) / len(rows)
        mean_impact = sum(to_float(row["top_foot_impact_vel_sum"]) for row in rows) / len(rows)
        stats.append(
            {
                "candidate": candidate,
                "top_gait_counts": " ".join(f"{gait}:{count}" for gait, count in sorted(gait_counts.items())),
                "margin_counts": " ".join(f"{label}:{count}" for label, count in sorted(margin_counts.items())),
                "mean_margin": mean_margin,
                "mean_top_vx_abs_error": mean_vx_error,
                "mean_top_fall_rate": mean_fall,
                "mean_top_transport_cost_proxy": mean_energy,
                "mean_top_foot_impact_vel_sum": mean_impact,
            }
        )
    return stats


def fmt(value, digits=3):
    return f"{to_float(value):.{digits}f}"


def write_summary(path, decision_rows, stats):
    lines = [
        "# Offline Unified-Reward Re-Score",
        "",
        "This analysis re-scores the completed fair gait grid with unified, terrain-agnostic reward candidates.",
        "Each gait is first allowed to use its own best continuous parameters under the selected candidate score.",
        "",
        "## Candidate Summary",
        "",
        "| candidate | top gait counts | margin counts | mean margin | mean vx_err | mean fall | mean energy | mean impact |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in stats:
        lines.append(
            f"| {row['candidate']} "
            f"| {row['top_gait_counts']} "
            f"| {row['margin_counts']} "
            f"| {fmt(row['mean_margin'])} "
            f"| {fmt(row['mean_top_vx_abs_error'])} "
            f"| {fmt(row['mean_top_fall_rate'])} "
            f"| {fmt(row['mean_top_transport_cost_proxy'], 1)} "
            f"| {fmt(row['mean_top_foot_impact_vel_sum'])} |"
        )

    by_candidate = group_rows(decision_rows, ("candidate",))
    for (candidate,), rows in sorted(by_candidate.items()):
        lines += [
            "",
            f"## {candidate}",
            "",
            "| task | vx | top | second | margin | label | score | vx_err | fall | lateral | scuff | impact | energy | params |",
            "|---|---:|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
        for row in sorted(rows, key=lambda r: (r["task_id"], to_float(r["cmd_vx"]))):
            params = (
                f"f={fmt(row['top_frequency'], 2)}, "
                f"foot={fmt(row['top_footswing_height'])}, "
                f"width={fmt(row['top_stance_width'])}, "
                f"pitch={fmt(row['top_body_pitch'])}"
            )
            lines.append(
                f"| {row['task_id']} "
                f"| {fmt(row['cmd_vx'], 2)} "
                f"| {row['top_gait']} "
                f"| {row['second_gait']} "
                f"| {fmt(row['score_margin'])} "
                f"| {row['margin_label']} "
                f"| {fmt(row['top_score'])} "
                f"| {fmt(row['top_vx_abs_error'])} "
                f"| {fmt(row['top_fall_rate'])} "
                f"| {fmt(row['top_lateral_offset_abs'])} "
                f"| {fmt(row['top_scuffing_ratio'])} "
                f"| {fmt(row['top_foot_impact_vel_sum'])} "
                f"| {fmt(row['top_transport_cost_proxy'], 1)} "
                f"| {params} |"
            )
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fair-grid-csv", default=DEFAULT_FAIR_GRID)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--softmax-temperature", type=float, default=0.03)
    return parser.parse_args()


def main():
    args = parse_args()
    fair_grid_csv = Path(args.fair_grid_csv)
    output_dir = Path(args.output_dir) if args.output_dir else fair_grid_csv.parent / "unified_reward_rescore"
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = read_rows(fair_grid_csv)
    scored_rows = add_candidate_scores(rows)
    best_by_gait, best_by_task, soft_rows, decision_rows, gap_rows = build_outputs(
        scored_rows,
        args.softmax_temperature,
    )
    stats = candidate_stats(decision_rows)

    write_csv(output_dir / "unified_reward_rescore_all.csv", scored_rows)
    write_csv(output_dir / "unified_reward_best_by_task_speed_gait.csv", best_by_gait)
    write_csv(output_dir / "unified_reward_best_by_task_speed.csv", best_by_task)
    write_csv(output_dir / "unified_reward_soft_distribution.csv", soft_rows)
    write_csv(output_dir / "unified_reward_decisions.csv", decision_rows)
    write_csv(output_dir / "unified_reward_top1_top2_metric_gaps.csv", gap_rows)
    write_csv(output_dir / "unified_reward_candidate_stats.csv", stats)
    write_summary(output_dir / "summary.md", decision_rows, stats)

    with (output_dir / "candidate_weights.json").open("w") as file:
        json.dump(CANDIDATE_WEIGHTS, file, indent=2)

    print(f"Wrote unified reward re-score outputs to: {output_dir}")


if __name__ == "__main__":
    main()
