import argparse
from pathlib import Path

import pandas as pd


TASKS = [
    {
        "task_id": "flat_trot_efficiency",
        "condition": "flat",
        "target_gait": "trotting",
        "label_type": "hard",
        "use_for_training": "yes",
        "speed_range": "0.5-2.0",
        "style_reward_strength": "medium",
        "role": "baseline,sim_to_real",
        "objective": "flat_trot_efficiency",
        "reward_focus": "progress,low_energy,low_slip,low_vertical_bounce,low_lateral_drift,trot_contact_style",
        "evidence_summary": "trotting is supported at all evaluated speeds",
        "notes": "Use as the default efficient stable gait on flat ground.",
    },
    {
        "task_id": "ramp_up_trot_robustness",
        "condition": "ramp_up",
        "target_gait": "trotting",
        "label_type": "weak",
        "use_for_training": "yes",
        "speed_range": "0.5-2.0",
        "style_reward_strength": "mild",
        "role": "sim_to_real,robustness",
        "objective": "ramp_up_stability",
        "reward_focus": "progress,orientation_stability,pitch_control,low_slip,low_energy,mild_trot_contact_style",
        "evidence_summary": "continuous uphill ramp without steps or center platform; evidence should come from the ramp_up supplement merged into v6_selected",
        "notes": "Use as a to-real robustness terrain. Keep gait-style reward mild so uphill traversal and stability dominate.",
    },
    {
        "task_id": "rough_slope_trot_robustness",
        "condition": "rough_slope",
        "target_gait": "trotting",
        "label_type": "weak",
        "use_for_training": "yes",
        "speed_range": "0.5-2.0",
        "style_reward_strength": "mild",
        "role": "sim_to_real,robustness",
        "objective": "rough_slope_stability",
        "reward_focus": "progress,orientation_stability,low_roll_pitch_rate,low_slip,low_scuffing,mild_trot_contact_style",
        "evidence_summary": "trotting is best at 0.5, 1.5, and 2.0 m/s; pronking slightly wins at 1.0 m/s",
        "notes": "Select trotting for consistency and sim-to-real conservatism; do not use strong style forcing.",
    },
    {
        "task_id": "push_bound_recovery",
        "condition": "push_hard",
        "target_gait": "bounding",
        "label_type": "evaluation_only",
        "use_for_training": "no",
        "speed_range": "0.5-2.0",
        "style_reward_strength": "none",
        "role": "legacy_demo_candidate,recovery",
        "objective": "push_bound_recovery",
        "reward_focus": "recovery_progress,low_lateral_drift,low_roll_rate,low_yaw_rate,low_done_rate,bound_contact_style",
        "evidence_summary": "old recovery objective favors bounding, but raw template score and visual stability are mixed; keep as a candidate, not a default hard label.",
        "notes": "Use only for visual comparison or ablation until the push_hard objective is revalidated.",
    },
    {
        "task_id": "push_lateral_pace_recovery",
        "condition": "push_lateral",
        "target_gait": "pacing",
        "label_type": "conditional_hard",
        "use_for_training": "yes",
        "speed_range": "1.5",
        "style_reward_strength": "medium",
        "role": "demo,recovery",
        "objective": "push_lateral_recovery",
        "reward_focus": "recovery_progress,low_lateral_drift,low_roll_rate,low_yaw_rate,low_done_rate,pace_contact_style",
        "evidence_summary": "pacing clearly wins the directed lateral-push recovery score at 1.5 m/s; 1.0 m/s remains trotting/ambiguous.",
        "notes": "Use as the main non-trot push-disturbance demo task after visual confirmation.",
    },
    {
        "task_id": "stepping_stones_easy_bound_highspeed",
        "condition": "stepping_stones_easy",
        "target_gait": "bounding",
        "label_type": "conditional_hard",
        "use_for_training": "yes",
        "speed_range": "2.0",
        "style_reward_strength": "medium",
        "role": "demo,obstacle",
        "objective": "stepping_stones_easy_bound_clearance",
        "reward_focus": "progress,foot_clearance,low_scuffing,low_done_rate,orientation_stability,bound_contact_style",
        "evidence_summary": "bounding wins the easy stepping-stone scan clearly at 2.0 m/s; 1.5 m/s is pace/bound ambiguous.",
        "notes": "Use only after the terrain visual check confirms the stepping-stone spacing/depth is reasonable.",
    },
]


def support_for_task(task, hypothesis_support, objective_best):
    objective = task["objective"]
    condition = task["condition"]
    target = task["target_gait"]

    if target:
        support = hypothesis_support[
            (hypothesis_support["objective"] == objective)
            & (hypothesis_support["condition"] == condition)
            & (hypothesis_support["expected_gait"] == target)
        ].copy()
        if not support.empty:
            return support[
                [
                    "objective",
                    "condition",
                    "vx",
                    "expected_gait",
                    "support",
                    "expected_rank",
                    "best_gait",
                    "score_gap_expected_to_best",
                    "best_score_gap_to_second",
                    "strong_metric_wins",
                    "all_metric_wins",
                ]
            ]

    best = objective_best[
        (objective_best["objective"] == objective) & (objective_best["condition"] == condition)
    ].copy()
    if best.empty:
        return pd.DataFrame()
    best["expected_gait"] = target
    best["support"] = "reference_only" if not target else "not_in_hypothesis_table"
    best["expected_rank"] = ""
    best["score_gap_expected_to_best"] = ""
    return best[
        [
            "objective",
            "condition",
            "vx",
            "expected_gait",
            "support",
            "expected_rank",
            "best_gait",
            "score_gap_expected_to_best",
            "best_score_gap_to_second",
            "strong_metric_wins",
            "all_metric_wins",
        ]
    ]


def build_speed_rows(task_map, hypothesis_support, objective_best):
    rows = []
    for task in TASKS:
        support = support_for_task(task, hypothesis_support, objective_best)
        if support.empty:
            continue
        for _, row in support.iterrows():
            speed_label = task["label_type"]
            use_for_training = task["use_for_training"]
            if task["condition"] == "push_lateral" and abs(float(row["vx"]) - 1.5) > 1e-6:
                speed_label = "evaluation_only"
                use_for_training = "no"
            if task["condition"] == "stepping_stones_easy" and abs(float(row["vx"]) - 2.0) > 1e-6:
                speed_label = "evaluation_only"
                use_for_training = "no"
            rows.append(
                {
                    "task_id": task["task_id"],
                    "condition": task["condition"],
                    "vx": row["vx"],
                    "target_gait": task["target_gait"],
                    "speed_label_type": speed_label,
                    "use_for_training": use_for_training,
                    "evidence_support": row["support"],
                    "best_gait_by_objective": row["best_gait"],
                    "expected_rank": row["expected_rank"],
                    "score_gap_expected_to_best": row["score_gap_expected_to_best"],
                    "best_score_gap_to_second": row["best_score_gap_to_second"],
                    "strong_metric_wins": row["strong_metric_wins"],
                    "all_metric_wins": row["all_metric_wins"],
                    "style_reward_strength": task["style_reward_strength"],
                    "reward_focus": task["reward_focus"],
                    "notes": task["notes"],
                }
            )
    return pd.DataFrame(rows)


def write_readme(output_dir):
    text = """# Training Task Map

This folder defines the current terrain/task to gait-label plan for high-level selector
training. It separates visually differentiated demo tasks from sim-to-real robustness tasks.

Files:

- `training_task_map.csv`: one row per task/terrain.
- `training_task_map_by_speed.csv`: evidence and label type per speed.

Label types:

- `hard`: strong enough to use as a gait target.
- `conditional_hard`: usable only in the listed speed range; other speeds may be evaluation only.
- `weak`: use with mild gait-style reward. Do not overrule traversal and stability.
- `evaluation_only`: do not train a hard gait label from this condition yet.

Current design:

- Flat terrain uses trotting as the efficient stable baseline.
- Continuous uphill ramp and rough slope also use trotting, but only as weak sim-to-real robustness labels.
- Rough-mid is intentionally removed from the main training map because visual
  checks showed that it is too difficult for the current low-level checkpoint and
  overlaps with rough_slope.
- Very low friction is intentionally excluded from training labels for now because
  visual checks and raw template scores favored trotting over the slip-biased
  pronking objective.
- The old high-frequency random push_hard -> bounding hypothesis is kept as an
  evaluation-only demo candidate, not as a default hard training label.
- Directed lateral push at 1.5 m/s uses pacing as a conditional hard label.
- Easy stepping stones at 2.0 m/s uses bounding as a conditional hard label.
- Stairs and discrete obstacles are intentionally excluded from the default task map because the current WTW checkpoint was not trained for those terrains.
"""
    (output_dir / "README.md").write_text(text)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--evidence-dir",
        default="logs/gait_condition_eval_v6_selected/task_metric_evidence",
        help="Directory containing hypothesis_support.csv and objective_best_gait_by_condition_speed.csv.",
    )
    parser.add_argument(
        "--output-dir",
        default="logs/gait_condition_eval_v6_selected/training_task_map",
        help="Output directory for the training task map.",
    )
    args = parser.parse_args()

    evidence_dir = Path(args.evidence_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    task_map = pd.DataFrame(TASKS)
    hypothesis_support = pd.read_csv(evidence_dir / "hypothesis_support.csv")
    objective_best = pd.read_csv(evidence_dir / "objective_best_gait_by_condition_speed.csv")

    speed_map = build_speed_rows(task_map, hypothesis_support, objective_best)

    task_map.to_csv(output_dir / "training_task_map.csv", index=False)
    speed_map.to_csv(output_dir / "training_task_map_by_speed.csv", index=False)
    write_readme(output_dir)

    print(f"Wrote training task map to: {output_dir}")


if __name__ == "__main__":
    main()
