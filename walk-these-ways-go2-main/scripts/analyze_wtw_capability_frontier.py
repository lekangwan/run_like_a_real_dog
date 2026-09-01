#!/usr/bin/env python3
"""Analyze whether tuned non-trot gaits extend beyond tuned-trot capability."""

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path


PARAMETER_NAMES = (
    "frequency_residual",
    "duration_residual",
    "footswing_height_residual",
    "stance_width_residual",
    "body_pitch_residual",
)

BASE_METRICS = (
    "vx_abs_error_mean",
    "fall_rate",
    "orientation_rms",
)

SECONDARY_METRICS = (
    "mechanical_power_abs_mean",
    "contact_slip_penalty_mean",
    "foot_impact_vel_rms",
    "scuffing_ratio_mean",
    "lateral_vel_rms",
)

IDENTITY_COLUMNS = (
    "task_id",
    "condition",
    "cmd_vx",
    "gait",
    *PARAMETER_NAMES,
)

REPORT_COLUMNS = (
    *IDENTITY_COLUMNS,
    "is_default_parameters",
    *BASE_METRICS,
    *SECONDARY_METRICS,
    "secondary_metric",
    "outside_tuned_trot_frontier",
    "on_all_gait_frontier",
    "validated_seed_count",
    "validated_seeds",
)

VALIDATION_CONFIG_COLUMNS = (
    "config_key",
    "task_id",
    "condition",
    "cmd_vx",
    "target_gait",
    "gait",
    "gait_id",
    "grid_mode",
    "selected_by",
    "frequency_residual",
    "frequency",
    "duration_residual",
    "duration",
    "footswing_height_residual",
    "footswing_height",
    "stance_width_residual",
    "stance_width",
    "body_pitch_residual",
    "body_pitch",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compare every gait's raw physical metrics against the equally tuned "
            "trotting frontier. No weighted reward is used."
        )
    )
    parser.add_argument("--grid-csv", required=True)
    parser.add_argument(
        "--validation-csv",
        action="append",
        default=[],
        help="Independent validation CSV. Repeat this argument for multiple seeds.",
    )
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def read_rows(path):
    with open(path, newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows


def to_float(row, name):
    value = row.get(name, "")
    if value in ("", None):
        raise ValueError(f"Missing required column {name!r}")
    return float(value)


def normalized_number(value):
    return f"{float(value):.8g}"


def config_key(row):
    return "|".join(
        (
            row["task_id"],
            normalized_number(row["cmd_vx"]),
            row["gait"],
            *(normalized_number(row[name]) for name in PARAMETER_NAMES),
        )
    )


def is_default_parameters(row, tolerance=1e-8):
    return all(abs(to_float(row, name)) <= tolerance for name in PARAMETER_NAMES)


def dominates(a, b, metrics, tolerances=None):
    """Return true when a is non-inferior throughout and meaningfully better."""
    tolerances = tolerances or {}
    a_values = [to_float(a, metric) for metric in metrics]
    b_values = [to_float(b, metric) for metric in metrics]
    no_worse = all(
        left <= right + tolerances.get(metric, 1e-12)
        for metric, left, right in zip(metrics, a_values, b_values)
    )
    better = any(
        left < right - tolerances.get(metric, 1e-12)
        for metric, left, right in zip(metrics, a_values, b_values)
    )
    return no_worse and better


def frontier(rows, metrics, tolerances=None):
    result = []
    for candidate in rows:
        if not any(
            other is not candidate
            and dominates(other, candidate, metrics, tolerances)
            for other in rows
        ):
            result.append(candidate)
    return result


def percentile(values, fraction):
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int(round(fraction * (len(ordered) - 1)))))
    return ordered[index]


def validation_data(paths):
    seeds_by_config = defaultdict(set)
    rows_by_config = defaultdict(list)
    for path in paths:
        for row in read_rows(path):
            seed = row.get("validation_seed", "")
            if seed:
                seeds_by_config[config_key(row)].add(seed)
            rows_by_config[config_key(row)].append(row)

    differences = defaultdict(lambda: defaultdict(list))
    for config_rows in rows_by_config.values():
        if len(config_rows) < 2:
            continue
        group_key = (
            config_rows[0]["task_id"],
            normalized_number(config_rows[0]["cmd_vx"]),
        )
        for metric in (*BASE_METRICS, *SECONDARY_METRICS):
            values = [to_float(row, metric) for row in config_rows]
            differences[group_key][metric].append(max(values) - min(values))

    noise_rows = []
    noise_by_group = {}
    for group_key, metric_differences in sorted(differences.items()):
        tolerances = {}
        for metric in (*BASE_METRICS, *SECONDARY_METRICS):
            values = metric_differences.get(metric, [])
            tolerances[metric] = percentile(values, 0.90)
            noise_rows.append(
                {
                    "task_id": group_key[0],
                    "cmd_vx": group_key[1],
                    "metric": metric,
                    "matched_config_count": len(values),
                    "median_abs_seed_difference": percentile(values, 0.50),
                    "p90_abs_seed_difference": tolerances[metric],
                }
            )
        noise_by_group[group_key] = tolerances
    return seeds_by_config, noise_by_group, noise_rows


def group_rows(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["task_id"], normalized_number(row["cmd_vx"]))].append(row)
    return grouped


def candidate_row(row, secondary_metric, outside_trot, on_all, validation):
    item = {name: row.get(name, "") for name in IDENTITY_COLUMNS}
    item["is_default_parameters"] = "yes" if is_default_parameters(row) else "no"
    for name in (*BASE_METRICS, *SECONDARY_METRICS):
        item[name] = row.get(name, "")
    seeds = sorted(validation.get(config_key(row), set()), key=float)
    item["secondary_metric"] = secondary_metric
    item["outside_tuned_trot_frontier"] = "yes" if outside_trot else "no"
    item["on_all_gait_frontier"] = "yes" if on_all else "no"
    item["validated_seed_count"] = len(seeds)
    item["validated_seeds"] = ",".join(seeds)
    return item


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def analyze(rows, validation, noise_by_group):
    candidates = []
    summaries = []
    coverage = []
    validation_union = {}

    def add_validation_row(row, reason):
        key = config_key(row)
        if key not in validation_union:
            validation_union[key] = {
                name: row.get(name, "")
                for name in VALIDATION_CONFIG_COLUMNS
                if name not in ("config_key", "selected_by")
            }
            validation_union[key]["config_key"] = key
            validation_union[key]["selected_by"] = set()
        validation_union[key]["selected_by"].add(reason)

    for (task_id, speed), group in sorted(group_rows(rows).items()):
        tolerances = noise_by_group.get((task_id, speed), {})
        gait_counts = Counter(row["gait"] for row in group)
        default_counts = Counter(
            row["gait"] for row in group if is_default_parameters(row)
        )
        coverage.append(
            {
                "task_id": task_id,
                "condition": group[0]["condition"],
                "cmd_vx": speed,
                "total_configs": len(group),
                "pronking_configs": gait_counts["pronking"],
                "trotting_configs": gait_counts["trotting"],
                "bounding_configs": gait_counts["bounding"],
                "pacing_configs": gait_counts["pacing"],
                "pronking_default_configs": default_counts["pronking"],
                "trotting_default_configs": default_counts["trotting"],
                "bounding_default_configs": default_counts["bounding"],
                "pacing_default_configs": default_counts["pacing"],
            }
        )

        trot_rows = [row for row in group if row["gait"] == "trotting"]
        if not trot_rows:
            raise ValueError(f"No trotting baseline for {task_id} vx={speed}")

        for secondary_metric in SECONDARY_METRICS:
            metrics = (*BASE_METRICS, secondary_metric)
            trot_front = frontier(trot_rows, metrics, tolerances)
            all_front = frontier(group, metrics, tolerances)
            for row in trot_front:
                add_validation_row(row, f"tuned_trot:{secondary_metric}")
            all_front_ids = {id(row) for row in all_front}
            outside_rows = [
                row
                for row in group
                if row["gait"] != "trotting"
                and not any(
                    dominates(trot, row, metrics, tolerances)
                    for trot in trot_front
                )
            ]
            candidate_ids = {
                id(row) for row in outside_rows if id(row) in all_front_ids
            }

            gait_counter = Counter(
                row["gait"] for row in outside_rows if id(row) in candidate_ids
            )
            validated_counter = Counter(
                row["gait"]
                for row in outside_rows
                if id(row) in candidate_ids and validation.get(config_key(row))
            )
            summaries.append(
                {
                    "task_id": task_id,
                    "condition": group[0]["condition"],
                    "cmd_vx": speed,
                    "secondary_metric": secondary_metric,
                    "trot_frontier_size": len(trot_front),
                    "all_gait_frontier_size": len(all_front),
                    "non_trot_frontier_size": len(candidate_ids),
                    "pronking_candidates": gait_counter["pronking"],
                    "bounding_candidates": gait_counter["bounding"],
                    "pacing_candidates": gait_counter["pacing"],
                    "pronking_validated": validated_counter["pronking"],
                    "bounding_validated": validated_counter["bounding"],
                    "pacing_validated": validated_counter["pacing"],
                    "noise_threshold_source": (
                        "heldout_p90" if tolerances else "exact_zero"
                    ),
                }
            )

            for row in outside_rows:
                on_all = id(row) in all_front_ids
                if not on_all:
                    continue
                add_validation_row(row, f"non_trot:{secondary_metric}")
                candidates.append(
                    candidate_row(
                        row,
                        secondary_metric,
                        outside_trot=True,
                        on_all=True,
                        validation=validation,
                    )
                )

    validation_rows = []
    for row in validation_union.values():
        row["selected_by"] = ";".join(sorted(row["selected_by"]))
        validation_rows.append(row)
    return coverage, summaries, candidates, validation_rows


def write_markdown(
    path,
    coverage,
    summaries,
    candidates,
    validation_paths,
    noise_rows,
):
    task_speed_count = len(coverage)
    total_configs = sum(int(row["total_configs"]) for row in coverage)
    candidate_configs = {config_key(row) for row in candidates}
    validated_configs = {
        config_key(row)
        for row in candidates
        if int(row["validated_seed_count"]) > 0
    }
    by_gait = Counter(row["gait"] for row in candidates)
    by_metric = Counter(row["secondary_metric"] for row in candidates)

    lines = [
        "# WTW Capability Frontier Audit",
        "",
        "This analysis uses raw physical metrics only. It does not rank configurations",
        "with a weighted reward. A non-trot candidate is retained only when it lies on",
        "the joint all-gait frontier and no equally tuned trotting configuration",
        "dominates it for tracking error, fall rate, orientation, and the named",
        "secondary metric.",
        "",
        "When the same configuration exists in two independent validation files, the",
        "90th percentile absolute seed difference is used as a task-speed-specific",
        "non-inferiority and meaningful-improvement tolerance. Missing groups fall",
        "back to exact numerical dominance.",
        "",
        "## Coverage",
        "",
        f"- task-speed points: `{task_speed_count}`",
        f"- total grid rows: `{total_configs}`",
        f"- independent validation files: `{len(validation_paths)}`",
        f"- repeat-noise estimates: `{len(noise_rows)}`",
        f"- unique non-trot candidate configurations: `{len(candidate_configs)}`",
        f"- candidates already present in any validation file: `{len(validated_configs)}`",
        "",
        "## Candidate Rows By Gait",
        "",
    ]
    for gait in ("pronking", "bounding", "pacing"):
        lines.append(f"- {gait}: `{by_gait[gait]}`")

    lines += ["", "## Candidate Rows By Secondary Metric", ""]
    for metric in SECONDARY_METRICS:
        lines.append(f"- {metric}: `{by_metric[metric]}`")

    lines += [
        "",
        "## Interpretation",
        "",
        "- A candidate is not automatically better than trotting. It may represent a",
        "  safety-energy or tracking-energy trade-off.",
        "- Rows not retained here are covered by another gait configuration under the",
        "  selected raw metrics, or by the tuned-trot frontier.",
        "- The repeat-noise threshold is conservative but is estimated from only two",
        "  validation seeds and historically score-selected configurations.",
        "- Existing held-out runs cover only candidates selected by historical score",
        "  rankings. Newly identified raw-frontier candidates may need targeted runs.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    args = parse_args()
    rows = read_rows(args.grid_csv)
    required = set(IDENTITY_COLUMNS) | set(BASE_METRICS) | set(SECONDARY_METRICS)
    missing = sorted(required.difference(rows[0]))
    if missing:
        raise ValueError(f"Grid CSV is missing required columns: {missing}")

    validation, noise_by_group, noise_rows = validation_data(args.validation_csv)
    coverage, summaries, candidates, validation_rows = analyze(
        rows,
        validation,
        noise_by_group,
    )
    output_dir = Path(args.output_dir)
    write_csv(output_dir / "coverage.csv", coverage, coverage[0].keys())
    write_csv(
        output_dir / "frontier_summary.csv",
        summaries,
        summaries[0].keys(),
    )
    write_csv(output_dir / "candidate_configs.csv", candidates, REPORT_COLUMNS)
    write_csv(
        output_dir / "validation_configs.csv",
        validation_rows,
        VALIDATION_CONFIG_COLUMNS,
    )
    if noise_rows:
        write_csv(
            output_dir / "repeat_noise.csv",
            noise_rows,
            noise_rows[0].keys(),
        )
    write_markdown(
        output_dir / "summary.md",
        coverage,
        summaries,
        candidates,
        args.validation_csv,
        noise_rows,
    )
    print(f"Wrote capability audit to {output_dir}")
    print(f"task-speed points: {len(coverage)}")
    print(f"candidate rows: {len(candidates)}")
    print(f"unique candidate configs: {len({config_key(row) for row in candidates})}")
    print(f"validation union configs: {len(validation_rows)}")


if __name__ == "__main__":
    main()
