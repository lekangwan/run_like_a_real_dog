import argparse
from pathlib import Path

import pandas as pd

from analyze_gait_task_evidence import METRIC_CATALOG, TASK_OBJECTIVES


SUMMARY_COLUMNS = [
    "task_id",
    "condition",
    "vx",
    "target_gait",
    "speed_label_type",
    "use_for_training",
    "source_vx",
    "frequency",
    "duration",
    "footswing_height",
    "body_pitch",
    "stance_width",
    "phase",
    "offset",
    "bound",
    "template_objective_score",
    "template_score",
    "measured_vx",
    "forward_distance_ratio",
    "progress_deficit",
    "done_rate",
    "slip_penalty",
    "scuffing_ratio",
    "swing_foot_clearance_mean",
    "foot_impact_vel_mean",
    "orientation_penalty",
    "lateral_vel_rms",
    "base_z_vel_rms",
    "roll_rate_rms",
    "pitch_rate_rms",
    "phase_match_error",
    "transport_cost_proxy",
]


def metric_directions():
    return {row["metric"]: row["direction"] for row in METRIC_CATALOG}


def objective_weights():
    return {task["objective"]: task["weights"] for task in TASK_OBJECTIVES}


def normalize(values, direction):
    lo = values.min()
    hi = values.max()
    if pd.isna(lo) or pd.isna(hi) or abs(hi - lo) < 1e-12:
        return pd.Series(0.5, index=values.index)
    if direction == "higher":
        return (values - lo) / (hi - lo)
    if direction == "lower":
        return (hi - values) / (hi - lo)
    raise ValueError(f"Unknown direction: {direction}")


def score_candidates(candidates, weights, directions):
    scored = candidates.copy()
    score = pd.Series(0.0, index=scored.index)
    weight_sum = 0.0
    component_columns = []
    for metric, weight in weights.items():
        if metric not in scored.columns or metric not in directions:
            continue
        component = normalize(scored[metric], directions[metric])
        column = f"component_{metric}"
        scored[column] = component
        score += abs(weight) * component
        weight_sum += abs(weight)
        component_columns.append(column)

    if weight_sum <= 0:
        scored["template_objective_score"] = scored.get("template_score", 0.0)
    else:
        scored["template_objective_score"] = score / weight_sum
    scored["template_component_count"] = len(component_columns)
    return scored


def viability_filter(candidates, min_forward_ratio, max_done_rate):
    if "forward_distance_ratio" not in candidates.columns or "done_rate" not in candidates.columns:
        return candidates, False
    viable = candidates[
        (candidates["forward_distance_ratio"] >= min_forward_ratio)
        & (candidates["done_rate"] <= max_done_rate)
    ].copy()
    if viable.empty:
        return candidates, False
    return viable, True


def select_templates(eval_results, speed_map, min_forward_ratio, max_done_rate):
    directions = metric_directions()
    weights_by_objective = objective_weights()
    library_rows = []
    candidate_rows = []

    usable = speed_map[
        (speed_map["use_for_training"] == "yes")
        & speed_map["target_gait"].notna()
        & (speed_map["target_gait"].astype(str).str.len() > 0)
    ].copy()

    for _, task in usable.iterrows():
        condition = task["condition"]
        vx = float(task["vx"])
        gait = task["target_gait"]
        objective = task.get("reward_focus", "")
        candidates = eval_results[
            (eval_results["condition"] == condition)
            & (eval_results["vx"] == vx)
            & (eval_results["gait"] == gait)
        ].copy()
        if candidates.empty:
            continue

        task_objective = task.get("task_id", "")
        objective_name = None
        # The by-speed map does not carry objective, so recover it from task_id
        # through the parent task map if present in the merged data.
        if "objective" in task and pd.notna(task["objective"]):
            objective_name = task["objective"]
        else:
            objective_name = task.get("objective_name", None)
        if not objective_name or objective_name not in weights_by_objective:
            # Fall back from task_id naming to known objective columns when the
            # input only contains the by-speed map.
            objective_name = {
                "flat_trot_efficiency": "flat_trot_efficiency",
                "rough_mid_trot_robustness": "rough_mid_clearance",
                "ramp_up_trot_robustness": "ramp_up_stability",
                "rough_slope_trot_robustness": "rough_slope_stability",
                "slippery_pronk_sync": "slippery_pronk_or_sync",
                "push_bound_recovery": "push_bound_recovery",
                "push_lateral_pace_recovery": "push_lateral_recovery",
                "stepping_stones_easy_bound_highspeed": "stepping_stones_easy_bound_clearance",
            }.get(task_objective)
        weights = weights_by_objective.get(objective_name, {})
        filtered, was_filtered = viability_filter(candidates, min_forward_ratio, max_done_rate)
        scored = score_candidates(filtered, weights, directions)
        scored["task_id"] = task["task_id"]
        scored["target_gait"] = gait
        scored["speed_label_type"] = task["speed_label_type"]
        scored["use_for_training"] = task["use_for_training"]
        scored["source_vx"] = scored["vx"]
        scored["objective_name"] = objective_name
        scored["viability_filtered"] = was_filtered
        scored["candidate_rank"] = scored["template_objective_score"].rank(
            method="first", ascending=False
        ).astype(int)
        candidate_rows.append(scored)

        best = scored.sort_values(
            ["template_objective_score", "template_score"], ascending=False
        ).iloc[0].to_dict()
        best["source_vx"] = best["vx"]
        best["vx"] = vx
        library_rows.append(best)

    library = pd.DataFrame(library_rows)
    candidates = pd.concat(candidate_rows, ignore_index=True) if candidate_rows else pd.DataFrame()
    return library, candidates


def write_readme(output_dir):
    text = """# Gait Template Library

This folder contains concrete low-level gait command templates selected from the
previous template evaluation results.

Files:

- `gait_template_library.csv`: one selected template for each trainable task and speed.
- `gait_template_candidates_scored.csv`: all candidate templates with objective scores.

Selection rule:

1. Read trainable rows from `training_task_map_by_speed.csv`.
2. Filter `template_eval_results.csv` by condition, speed, and target gait.
3. Score candidate parameter sets with the task objective metrics.
4. If viable candidates exist, keep only templates whose forward-distance ratio and
   done rate pass the gate.
5. Choose the highest-scoring concrete template.

This library is meant for oracle playback first. It should not be treated as final
learned behavior until the oracle visual check passes.
"""
    (output_dir / "README.md").write_text(text)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--eval-results",
        default="logs/gait_condition_eval_v6_selected/template_eval_results.csv",
    )
    parser.add_argument(
        "--task-map",
        default="logs/gait_condition_eval_v6_selected/training_task_map/training_task_map_by_speed.csv",
    )
    parser.add_argument(
        "--task-map-summary",
        default="logs/gait_condition_eval_v6_selected/training_task_map/training_task_map.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="logs/gait_condition_eval_v6_selected/gait_template_library",
    )
    parser.add_argument("--min-forward-ratio", type=float, default=0.45)
    parser.add_argument("--max-done-rate", type=float, default=0.02)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    eval_results = pd.read_csv(args.eval_results)
    speed_map = pd.read_csv(args.task_map)
    summary_path = Path(args.task_map_summary)
    if summary_path.exists():
        summary = pd.read_csv(summary_path)[["task_id", "objective"]]
        speed_map = speed_map.merge(summary, on="task_id", how="left")

    library, candidates = select_templates(
        eval_results,
        speed_map,
        min_forward_ratio=args.min_forward_ratio,
        max_done_rate=args.max_done_rate,
    )
    if library.empty:
        raise ValueError("No templates were selected. Check task map and eval result paths.")

    columns = [column for column in SUMMARY_COLUMNS if column in library.columns]
    extra_columns = [column for column in library.columns if column not in columns]
    library[columns + extra_columns].to_csv(output_dir / "gait_template_library.csv", index=False)
    if not candidates.empty:
        candidates.to_csv(output_dir / "gait_template_candidates_scored.csv", index=False)
    write_readme(output_dir)

    print(f"Wrote gait template library to: {output_dir}")


if __name__ == "__main__":
    main()
