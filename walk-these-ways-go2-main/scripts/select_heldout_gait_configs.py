import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from go2_gym.envs.wrappers.high_level_reward_metrics import (
    CANONICAL_REWARD_NAMES,
    UNIFIED_REWARD_PROFILES,
)


BEHAVIOR_NAMES = (
    "frequency",
    "duration",
    "footswing_height",
    "stance_width",
    "body_pitch",
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
        raise ValueError(f"Unknown reward profile {profile_name!r}")
    weights = {name: 0.0 for name in CANONICAL_REWARD_NAMES}
    weights.update(UNIFIED_REWARD_PROFILES[profile_name])
    return weights


def score_profile(row, weights):
    total = 0.0
    weight_sum = 0.0
    for metric, weight in weights.items():
        weight = float(weight)
        if weight <= 0.0:
            continue
        column = f"score_{metric}"
        if column not in row:
            raise ValueError(f"Missing required score column {column!r}")
        total += weight * to_float(row[column])
        weight_sum += weight
    return total / max(weight_sum, 1e-9)


def add_scores(rows, profile_names):
    weights_by_profile = {name: profile_weights(name) for name in profile_names}
    for row in rows:
        for profile_name, weights in weights_by_profile.items():
            row[f"{profile_name}_score"] = score_profile(row, weights)


def config_key(row):
    parts = [
        row.get("task_id", ""),
        f"{to_float(row.get('cmd_vx')):.6f}",
        row.get("gait", ""),
    ]
    for name in BEHAVIOR_NAMES:
        parts.append(f"{to_float(row.get(f'{name}_residual')):.6f}")
    return "|".join(parts)


def select_topk(rows, score_key, top_k):
    grouped = defaultdict(list)
    for row in rows:
        key = (row["task_id"], row["cmd_vx"], row["gait"])
        grouped[key].append(row)

    selected = []
    for key, group in grouped.items():
        ranked = sorted(group, key=lambda item: to_float(item[score_key]), reverse=True)
        for rank, row in enumerate(ranked[:top_k], start=1):
            item = dict(row)
            item["_selection_score_key"] = score_key
            item["_selection_rank"] = rank
            item["_selection_score"] = to_float(row[score_key])
            selected.append(item)
    return selected


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
            "fall_rate",
            "lateral_offset_abs_mean",
            "scuffing_ratio_mean",
            "foot_impact_vel_rms",
            "transport_cost_proxy",
        ):
            item[f"source_{key}"] = value
        elif key.endswith("_score") and key not in item:
            item[f"source_{key}"] = value
    return item


def read_cached_keys(paths):
    cached = set()
    for path in paths:
        if not path:
            continue
        for row in read_rows(path):
            if row.get("config_key"):
                cached.add(row["config_key"])
            else:
                cached.add(config_key(row))
    return cached


def write_summary(path, rows, new_rows, top_k, score_keys):
    by_task = defaultdict(int)
    by_gait = defaultdict(int)
    for row in rows:
        by_task[row["task_id"]] += 1
        by_gait[row["gait"]] += 1
    lines = [
        "# Held-Out Config Selection",
        "",
        f"- top_k_per_task_speed_gait: `{top_k}`",
        f"- score_keys: `{', '.join(score_keys)}`",
        f"- selected_unique_configs: `{len(rows)}`",
        f"- new_uncached_configs: `{len(new_rows)}`",
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
        description="Select a deduplicated union of top-k fair-grid configs for held-out validation."
    )
    parser.add_argument("--input", required=True, help="fair_gait_grid_results.csv")
    parser.add_argument(
        "--profiles",
        default="canonical_efficiency_candidate,canonical_balanced_candidate",
        help="Comma-separated reward profiles to score from saved score_<metric> columns.",
    )
    parser.add_argument(
        "--score-keys",
        default="",
        help="Extra existing score columns to select by, e.g. neutral_score,weighted_metric_reward_mean.",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--cache-csv", action="append", default=[])
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    rows = read_rows(args.input)
    profile_names = [item.strip() for item in args.profiles.split(",") if item.strip()]
    add_scores(rows, profile_names)
    score_keys = [f"{profile}_score" for profile in profile_names]
    score_keys.extend(item.strip() for item in args.score_keys.split(",") if item.strip())

    selected = []
    for score_key in score_keys:
        selected.extend(select_topk(rows, score_key, args.top_k))

    by_key = {}
    selected_by = defaultdict(list)
    for row in selected:
        key = config_key(row)
        by_key.setdefault(key, row)
        selected_by[key].append(
            f"{row['_selection_score_key']}:rank{row['_selection_rank']}:score{row['_selection_score']:.6f}"
        )

    configs = [
        compact_config(row, selected_by[key])
        for key, row in sorted(by_key.items(), key=lambda item: item[0])
    ]

    cached_keys = read_cached_keys(args.cache_csv)
    new_configs = [row for row in configs if row["config_key"] not in cached_keys]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "heldout_config_requests.csv", configs)
    write_csv(output_dir / "new_heldout_config_requests.csv", new_configs)
    write_summary(output_dir / "summary.md", configs, new_configs, args.top_k, score_keys)
    with (output_dir / "run_config.json").open("w") as file:
        json.dump(vars(args), file, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
