import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from go2_gym.envs.wrappers.high_level_reward_metrics import UNIFIED_REWARD_PROFILES


LOWER_BETTER_METRICS = {
    "vx_abs_error_mean": "tracking_error",
    "torque_penalty_mean": "wrapper_energy_penalty",
    "transport_cost_proxy": "transport_cost_proxy",
    "slip_penalty_mean": "wrapper_slip_penalty",
    "reward_impact_velocity_rms_mean": "wrapper_step_impact_rms",
    "foot_impact_vel_rms": "event_impact_rms",
    "scuffing_ratio_mean": "wrapper_scuffing_ratio",
    "fall_rate": "fall_rate",
    "lateral_offset_abs_mean": "lateral_offset",
}

HIGHER_BETTER_METRICS = {
    "weighted_metric_reward_mean": "live_weighted_reward",
    "neutral_score": "neutral_score",
    "score_progress": "score_progress",
    "score_energy": "score_energy",
    "score_slip": "score_slip",
    "score_impact": "score_impact",
    "score_scuffing": "score_scuffing",
    "score_survival": "score_survival",
}

KEY_COLUMNS = ("task_id", "cmd_vx", "gait")

METRIC_RAW_KEYS = {
    "progress": ("vx_abs_error_mean", "lower"),
    "yaw_tracking": ("yaw_abs_mean", "lower"),
    "tracking_gate": ("vx_abs_error_mean", "lower"),
    "tracking_gate_strict": ("vx_abs_error_mean", "lower"),
    "orientation": ("orientation_rms", "lower"),
    "lateral_drift": ("lateral_offset_abs_mean", "lower"),
    "slip": ("slip_penalty_mean", "lower"),
    "contact_slip": ("contact_slip_penalty_mean", "lower"),
    "energy": ("torque_penalty_mean", "lower"),
    "power_efficiency": ("reward_mechanical_power_abs_mean", "lower"),
    "transport_efficiency": ("reward_transport_cost_proxy_mean", "lower"),
    "impact": ("reward_impact_velocity_rms_mean", "lower"),
    "scuffing": ("scuffing_ratio_mean", "lower"),
    "survival": ("fall_rate", "lower"),
    "safety_lateral_drift": ("lateral_offset_abs_mean", "lower"),
    "safety_contact_slip": ("contact_slip_penalty_mean", "lower"),
    "safety_impact": ("reward_impact_velocity_rms_mean", "lower"),
    "safety_scuffing": ("scuffing_ratio_mean", "lower"),
    "gated_orientation": ("orientation_rms", "lower"),
    "gated_lateral_drift": ("lateral_offset_abs_mean", "lower"),
    "gated_contact_slip": ("contact_slip_penalty_mean", "lower"),
    "gated_power_efficiency": ("reward_mechanical_power_abs_mean", "lower"),
    "gated_transport_efficiency": ("reward_transport_cost_proxy_mean", "lower"),
    "gated_impact": ("reward_impact_velocity_rms_mean", "lower"),
    "gated_scuffing": ("scuffing_ratio_mean", "lower"),
    "gated_action_smoothness": ("action_delta_penalty_mean", "lower"),
    "strict_gated_power_efficiency": ("reward_mechanical_power_abs_mean", "lower"),
    "action_smoothness": ("action_delta_penalty_mean", "lower"),
    "action_magnitude": ("continuous_action_penalty_mean", "lower"),
    "action_boundary_margin": ("action_boundary_penalty_mean", "lower"),
    "gait_stability": ("gait_switch_penalty_mean", "lower"),
}


def to_float(value, default=0.0):
    try:
        if value == "":
            return default
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


def best_rows(rows, keys, score_key, higher=True):
    best = {}
    for row in rows:
        key = tuple(row[k] for k in keys)
        value = to_float(row.get(score_key))
        if key not in best:
            best[key] = row
            continue
        best_value = to_float(best[key].get(score_key))
        if (higher and value > best_value) or ((not higher) and value < best_value):
            best[key] = row
    return list(best.values())


def rank_rows(rows, score_key, higher=True):
    return sorted(rows, key=lambda row: to_float(row.get(score_key)), reverse=higher)


def group_rows(rows, keys):
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row[k] for k in keys)].append(row)
    return grouped


def maybe_get(row, key):
    return row.get(key, "")


def compact_row(row, prefix=""):
    item = {}
    for key in (
        "task_id",
        "condition",
        "cmd_vx",
        "gait",
        "weighted_metric_reward_mean",
        "neutral_score",
        "vx_abs_error_mean",
        "measured_vx_mean",
        "torque_penalty_mean",
        "reward_mechanical_power_abs_mean",
        "reward_transport_cost_proxy_mean",
        "transport_cost_proxy",
        "slip_penalty_mean",
        "contact_slip_penalty_mean",
        "reward_impact_velocity_rms_mean",
        "foot_impact_vel_rms",
        "scuffing_ratio_mean",
        "fall_rate",
        "score_progress",
        "score_energy",
        "score_power_efficiency",
        "score_transport_efficiency",
        "score_slip",
        "score_contact_slip",
        "score_impact",
        "score_scuffing",
        "score_survival",
        "actual_frequency_mean",
        "actual_footswing_height_mean",
        "actual_stance_width_mean",
        "actual_body_pitch_mean",
    ):
        if key in row:
            item[f"{prefix}{key}"] = row[key]
    return item


def score_raw_alignment(rows):
    checks = []
    pairs = [
        ("torque_penalty_mean", "score_energy", "energy"),
        ("reward_mechanical_power_abs_mean", "score_power_efficiency", "power_efficiency"),
        ("reward_transport_cost_proxy_mean", "score_transport_efficiency", "transport_efficiency"),
        ("slip_penalty_mean", "score_slip", "slip"),
        ("contact_slip_penalty_mean", "score_contact_slip", "contact_slip"),
        ("reward_impact_velocity_rms_mean", "score_impact", "impact"),
        ("scuffing_ratio_mean", "score_scuffing", "scuffing"),
    ]
    grouped = group_rows(rows, ["task_id", "cmd_vx"])
    for (task_id, vx), group in sorted(grouped.items()):
        for raw_key, score_key, name in pairs:
            if raw_key not in group[0] or score_key not in group[0]:
                continue
            raw_best = rank_rows(group, raw_key, higher=False)[0]
            score_best = rank_rows(group, score_key, higher=True)[0]
            raw_values = [to_float(row.get(raw_key)) for row in group]
            score_values = [to_float(row.get(score_key)) for row in group]
            raw_range = max(raw_values) - min(raw_values)
            score_range = max(score_values) - min(score_values)
            checks.append(
                {
                    "task_id": task_id,
                    "cmd_vx": vx,
                    "metric": name,
                    "raw_key": raw_key,
                    "score_key": score_key,
                    "raw_best_gait": raw_best["gait"],
                    "score_best_gait": score_best["gait"],
                    "raw_best_value": to_float(raw_best.get(raw_key)),
                    "score_best_raw_value": to_float(score_best.get(raw_key)),
                    "score_best_value": to_float(score_best.get(score_key)),
                    "raw_range": raw_range,
                    "score_range": score_range,
                    "direction_agrees": raw_best["gait"] == score_best["gait"],
                }
            )
    return checks


def score_compensation_flags(rows, score_key):
    flags = []
    grouped = group_rows(rows, ["task_id", "cmd_vx"])
    for (task_id, vx), group in sorted(grouped.items()):
        best = rank_rows(group, score_key, higher=True)[0]
        tracking_best = rank_rows(group, "vx_abs_error_mean", higher=False)[0]
        energy_key = "reward_transport_cost_proxy_mean"
        if energy_key not in best:
            energy_key = "torque_penalty_mean"
        energy_best = rank_rows(group, energy_key, higher=False)[0]
        slip_key = "contact_slip_penalty_mean"
        if slip_key not in best:
            slip_key = "slip_penalty_mean"
        slip_best = rank_rows(group, slip_key, higher=False)[0]
        impact_key = "reward_impact_velocity_rms_mean"
        if impact_key not in best:
            impact_key = "foot_impact_vel_rms"
        impact_best = rank_rows(group, impact_key, higher=False)[0]

        best_tracking = to_float(best.get("vx_abs_error_mean"))
        min_tracking = to_float(tracking_best.get("vx_abs_error_mean"))
        best_energy = to_float(best.get(energy_key))
        min_energy = to_float(energy_best.get(energy_key))
        best_slip = to_float(best.get(slip_key))
        min_slip = to_float(slip_best.get(slip_key))
        best_impact = to_float(best.get(impact_key))
        min_impact = to_float(impact_best.get(impact_key))

        flags.append(
            {
                "task_id": task_id,
                "cmd_vx": vx,
                "score_key": score_key,
                "score_best_gait": best["gait"],
                "tracking_best_gait": tracking_best["gait"],
                "energy_best_gait": energy_best["gait"],
                "slip_best_gait": slip_best["gait"],
                "impact_best_gait": impact_best["gait"],
                "score_best_vx_abs_error": best_tracking,
                "tracking_best_vx_abs_error": min_tracking,
                "tracking_gap": best_tracking - min_tracking,
                "energy_key": energy_key,
                "score_best_energy": best_energy,
                "energy_best_energy": min_energy,
                "energy_gap": best_energy - min_energy,
                "slip_key": slip_key,
                "score_best_slip": best_slip,
                "slip_best_slip": min_slip,
                "slip_gap": best_slip - min_slip,
                "impact_key": impact_key,
                "score_best_impact": best_impact,
                "impact_best_impact": min_impact,
                "impact_gap": best_impact - min_impact,
                "fall_rate": to_float(best.get("fall_rate")),
                "tracking_warning": best_tracking > max(0.08, min_tracking * 1.35),
                "energy_warning": best_energy > max(0.02, min_energy * 1.50),
                "slip_warning": best_slip > max(0.02, min_slip * 1.50),
                "impact_warning": best_impact > max(0.15, min_impact * 1.50),
            }
        )
    return flags


def contribution_rows(rows, profile_name):
    if not profile_name:
        return []
    if profile_name not in UNIFIED_REWARD_PROFILES:
        raise ValueError(f"Unknown reward profile {profile_name!r}")

    weights = UNIFIED_REWARD_PROFILES[profile_name]
    weight_sum = sum(weights.values())
    if weight_sum <= 0:
        raise ValueError(f"Reward profile {profile_name!r} has non-positive weight sum")

    out = []
    for row in rows:
        tracking_gate = to_float(row.get("score_tracking_gate"), default=math.nan)
        for metric, weight in weights.items():
            score_key = f"score_{metric}"
            score = to_float(row.get(score_key), default=math.nan)
            raw_key, raw_direction = METRIC_RAW_KEYS.get(metric, ("", ""))
            raw_value = to_float(row.get(raw_key), default=math.nan) if raw_key else math.nan
            out.append(
                {
                    "task_id": row["task_id"],
                    "cmd_vx": row["cmd_vx"],
                    "gait": row["gait"],
                    "profile": profile_name,
                    "metric": metric,
                    "weight": weight,
                    "weight_fraction": weight / weight_sum,
                    "score_key": score_key,
                    "score": score,
                    "weighted_contribution": score * weight / weight_sum,
                    "raw_key": raw_key,
                    "raw_direction": raw_direction,
                    "raw_value": raw_value,
                    "tracking_gate": tracking_gate,
                    "weighted_metric_reward_mean": to_float(row.get("weighted_metric_reward_mean")),
                }
            )
    return out


def top_vs_second_contribution_gaps(rows, score_key, profile_name):
    if not profile_name:
        return []
    if profile_name not in UNIFIED_REWARD_PROFILES:
        raise ValueError(f"Unknown reward profile {profile_name!r}")

    weights = UNIFIED_REWARD_PROFILES[profile_name]
    weight_sum = sum(weights.values())
    out = []
    grouped = group_rows(rows, ["task_id", "cmd_vx"])
    for (task_id, vx), group in sorted(grouped.items()):
        ranked = rank_rows(group, score_key, higher=True)
        if len(ranked) < 2:
            continue
        top = ranked[0]
        second = ranked[1]
        reward_gap = to_float(top.get(score_key)) - to_float(second.get(score_key))
        for metric, weight in weights.items():
            score_key_metric = f"score_{metric}"
            top_score = to_float(top.get(score_key_metric), default=math.nan)
            second_score = to_float(second.get(score_key_metric), default=math.nan)
            raw_key, raw_direction = METRIC_RAW_KEYS.get(metric, ("", ""))
            top_raw = to_float(top.get(raw_key), default=math.nan) if raw_key else math.nan
            second_raw = to_float(second.get(raw_key), default=math.nan) if raw_key else math.nan
            out.append(
                {
                    "task_id": task_id,
                    "cmd_vx": vx,
                    "profile": profile_name,
                    "top_gait": top["gait"],
                    "second_gait": second["gait"],
                    "ranking_score_key": score_key,
                    "top_score_total": to_float(top.get(score_key)),
                    "second_score_total": to_float(second.get(score_key)),
                    "reward_gap": reward_gap,
                    "metric": metric,
                    "weight": weight,
                    "weight_fraction": weight / weight_sum,
                    "top_metric_score": top_score,
                    "second_metric_score": second_score,
                    "metric_score_gap": top_score - second_score,
                    "weighted_gap": (top_score - second_score) * weight / weight_sum,
                    "raw_key": raw_key,
                    "raw_direction": raw_direction,
                    "top_raw_value": top_raw,
                    "second_raw_value": second_raw,
                    "raw_value_gap": top_raw - second_raw,
                    "top_tracking_gate": to_float(top.get("score_tracking_gate"), default=math.nan),
                    "second_tracking_gate": to_float(second.get("score_tracking_gate"), default=math.nan),
                }
            )
    return out


def write_summary(path, rows, score_key, ranking, alignment, flags):
    by_gait = defaultdict(int)
    for row in ranking:
        by_gait[row["gait"]] += 1

    warning_count = sum(
        bool(flag["tracking_warning"])
        + bool(flag["energy_warning"])
        + bool(flag["slip_warning"])
        + bool(flag["impact_warning"])
        for flag in flags
    )
    lines = [
        "# Metric Sanity Audit",
        "",
        "This is a small representative-config audit. It checks whether reward scores",
        "and raw physical metrics move in the expected direction before any full fair grid.",
        "",
        f"- rows: `{len(rows)}`",
        f"- ranking_score_key: `{score_key}`",
        f"- task_speed_points: `{len({(row['task_id'], row['cmd_vx']) for row in rows})}`",
        f"- score_best_warning_count: `{warning_count}`",
        "",
        "## Score-Best Gait Counts",
        "",
    ]
    for gait, count in sorted(by_gait.items()):
        lines.append(f"- {gait}: {count}")

    lines += [
        "",
        "## Score-Best Ranking",
        "",
        "| task | vx | best gait | score | vx_err | torque | slip | impact | scuff | fall |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in ranking:
        impact = row.get("reward_impact_velocity_rms_mean", row.get("foot_impact_vel_rms", ""))
        lines.append(
            "| {task} | {vx:.2f} | {gait} | {score:.4f} | {vxerr:.4f} | "
            "{torque:.4f} | {slip:.4f} | {impact:.4f} | {scuff:.4f} | {fall:.4f} |".format(
                task=row["task_id"],
                vx=to_float(row["cmd_vx"]),
                gait=row["gait"],
                score=to_float(row.get(score_key)),
                vxerr=to_float(row.get("vx_abs_error_mean")),
                torque=to_float(row.get("torque_penalty_mean")),
                slip=to_float(row.get("slip_penalty_mean")),
                impact=to_float(impact),
                scuff=to_float(row.get("scuffing_ratio_mean")),
                fall=to_float(row.get("fall_rate")),
            )
        )

    disagree = [row for row in alignment if not row["direction_agrees"]]
    lines += [
        "",
        "## Raw/Score Direction Checks",
        "",
        f"- direction_disagreements: `{len(disagree)}`",
        "",
    ]
    for row in disagree[:20]:
        lines.append(
            "- {task} vx={vx} {metric}: raw-best={raw}, score-best={score}".format(
                task=row["task_id"],
                vx=row["cmd_vx"],
                metric=row["metric"],
                raw=row["raw_best_gait"],
                score=row["score_best_gait"],
            )
        )
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Analyze a small metric sanity audit CSV. This does not validate a "
            "reward for PPO; it checks raw metric/score direction and compensation."
        )
    )
    parser.add_argument("--input", required=True, help="fair_gait_grid_results.csv from a small config audit")
    parser.add_argument("--score-key", default="weighted_metric_reward_mean")
    parser.add_argument(
        "--reward-profile",
        default=None,
        choices=tuple(UNIFIED_REWARD_PROFILES),
        help=(
            "Optional profile used to write weighted contribution decomposition "
            "tables for best-by-gait rows and top-vs-second gaps."
        ),
    )
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    rows = read_rows(args.input)
    if not rows:
        raise ValueError("Input CSV has no rows")
    if args.score_key not in rows[0]:
        raise ValueError(f"Missing score key {args.score_key!r}")

    best_by_gait = best_rows(rows, ["task_id", "cmd_vx", "gait"], args.score_key, higher=True)
    ranking = best_rows(best_by_gait, ["task_id", "cmd_vx"], args.score_key, higher=True)
    ranking = sorted(ranking, key=lambda row: (row["task_id"], to_float(row["cmd_vx"])))
    alignment = score_raw_alignment(best_by_gait)
    flags = score_compensation_flags(best_by_gait, args.score_key)
    contributions = contribution_rows(best_by_gait, args.reward_profile)
    contribution_gaps = top_vs_second_contribution_gaps(
        best_by_gait,
        args.score_key,
        args.reward_profile,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "best_by_task_speed_gait.csv", best_by_gait)
    write_csv(output_dir / "score_best_by_task_speed.csv", ranking)
    write_csv(output_dir / "raw_score_direction_checks.csv", alignment)
    write_csv(output_dir / "score_best_compensation_flags.csv", flags)
    write_csv(output_dir / "weighted_contribution_decomposition.csv", contributions)
    write_csv(output_dir / "top_vs_second_contribution_gaps.csv", contribution_gaps)
    write_summary(output_dir / "summary.md", best_by_gait, args.score_key, ranking, alignment, flags)


if __name__ == "__main__":
    main()
