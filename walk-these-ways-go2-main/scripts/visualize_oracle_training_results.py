import argparse
import csv
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


KEY_METRICS = (
    "reward",
    "weighted_metric_reward",
    "vx_err",
    "lateral_position_penalty",
    "gait_switch_penalty",
    "gait_switch_rate",
    "score_progress",
    "score_clearance",
    "score_gait_stability",
    "score_action_smoothness",
    "score_action_magnitude",
    "score_action_boundary_margin",
    "action_clip_rate",
    "footswing_height_mean",
    "stance_width_mean",
    "body_pitch_mean",
)

TASK_IDS = (
    "flat_trot_efficiency",
    "ramp_up_trot_robustness",
    "rough_slope_trot_robustness",
    "push_lateral_pace_recovery",
    "stepping_stones_easy_bound_highspeed",
)

TARGET_RATIO_KEYS = {
    "flat_trot_efficiency": "flat_trot_efficiency_trot_ratio",
    "ramp_up_trot_robustness": "ramp_up_trot_robustness_trot_ratio",
    "rough_slope_trot_robustness": "rough_slope_trot_robustness_trot_ratio",
    "push_lateral_pace_recovery": "push_lateral_pace_recovery_pace_ratio",
    "stepping_stones_easy_bound_highspeed": "stepping_stones_easy_bound_highspeed_bound_ratio",
}

GAIT_RATIO_KEYS = ("pronk_ratio", "trot_ratio", "bound_ratio", "pace_ratio")
ROUTE_HEALTH_KEYS = (
    "vx_err_mean",
    "lateral_offset_mean",
    "done_rate",
    "action_clip_rate",
    "gait_switch_rate",
)
ROUTE_ACTION_KEYS = (
    "frequency_mean",
    "duration_mean",
    "footswing_height_mean",
    "stance_width_mean",
    "body_pitch_mean",
)


def read_metrics(path):
    rows = []
    with Path(path).open(newline="") as file:
        for row in csv.DictReader(file):
            parsed = {}
            for key, value in row.items():
                if value == "":
                    continue
                try:
                    parsed[key] = float(value)
                except ValueError:
                    parsed[key] = value
            rows.append(parsed)
    if not rows:
        raise ValueError(f"No rows found in metrics file: {path}")
    return rows


def latest_route_test_dir(run_dir):
    route_root = Path(run_dir) / "route_tests"
    if not route_root.exists():
        return None
    candidates = [
        path
        for path in route_root.iterdir()
        if path.is_dir()
        and (path / "route_summary.csv").exists()
        and (path / "route_timeseries.csv").exists()
    ]
    if not candidates:
        return None
    return sorted(candidates, key=lambda path: path.stat().st_mtime)[-1]


def values(rows, key):
    return [row[key] for row in rows if key in row]


def mean(items):
    items = list(items)
    return sum(items) / len(items) if items else float("nan")


def add_legend_if_present(ax):
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc="best", fontsize=8)


def window_mean(rows, key, window, end=False):
    subset = rows[-window:] if end else rows[:window]
    return mean(row[key] for row in subset if key in row)


def summarize(rows, window):
    summary = {}
    keys = [key for key in KEY_METRICS if key in rows[0]]
    for task_id in TASK_IDS:
        for suffix in (
            "action_clip_rate",
            "gait_switch_rate",
            "footswing_height_mean",
            "frequency_mean",
            "stance_width_mean",
            "body_pitch_mean",
        ):
            key = f"{task_id}_{suffix}"
            if key in rows[0]:
                keys.append(key)
    for task_id, key in TARGET_RATIO_KEYS.items():
        if key in rows[0]:
            keys.append(key)

    for key in keys:
        early = window_mean(rows, key, window, end=False)
        late = window_mean(rows, key, window, end=True)
        summary[key] = {
            "early": early,
            "late": late,
            "delta": late - early,
        }
    return summary


def plot_lines(rows, groups, output_path, title):
    fig, axes = plt.subplots(len(groups), 1, figsize=(11, 3.1 * len(groups)), sharex=True)
    if len(groups) == 1:
        axes = [axes]
    x = values(rows, "iteration")
    for ax, (ylabel, keys) in zip(axes, groups):
        for key in keys:
            if key not in rows[0]:
                continue
            ax.plot(x, values(rows, key), label=key, linewidth=1.8)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        add_legend_if_present(ax)
    axes[-1].set_xlabel("iteration")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_summary_bars(summary, keys, output_path, title):
    keys = [key for key in keys if key in summary]
    if not keys:
        return
    early = [summary[key]["early"] for key in keys]
    late = [summary[key]["late"] for key in keys]
    x = list(range(len(keys)))
    width = 0.36

    fig, ax = plt.subplots(figsize=(max(10, 0.7 * len(keys)), 5.5))
    ax.bar([i - width / 2 for i in x], early, width, label="early")
    ax.bar([i + width / 2 for i in x], late, width, label="late")
    ax.set_xticks(x)
    ax.set_xticklabels(keys, rotation=40, ha="right")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_compare(current_summary, baseline_summary, keys, output_path, title):
    keys = [key for key in keys if key in current_summary and key in baseline_summary]
    if not keys:
        return
    current_late = [current_summary[key]["late"] for key in keys]
    baseline_late = [baseline_summary[key]["late"] for key in keys]
    x = list(range(len(keys)))
    width = 0.36

    fig, ax = plt.subplots(figsize=(max(10, 0.7 * len(keys)), 5.5))
    ax.bar([i - width / 2 for i in x], baseline_late, width, label="baseline late")
    ax.bar([i + width / 2 for i in x], current_late, width, label="current late")
    ax.set_xticks(x)
    ax.set_xticklabels(keys, rotation=40, ha="right")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_route_summary(route_rows, output_path):
    labels = [f"{int(row['segment'])}:{row['condition']}" for row in route_rows]
    x = list(range(len(route_rows)))

    fig, axes = plt.subplots(3, 1, figsize=(12, 11), sharex=True)
    width = 0.18
    for offset, key in zip((-1.5, -0.5, 0.5, 1.5), GAIT_RATIO_KEYS):
        axes[0].bar(
            [i + offset * width for i in x],
            [row.get(key, 0.0) for row in route_rows],
            width,
            label=key,
        )
    axes[0].set_ylabel("gait ratio")
    add_legend_if_present(axes[0])
    axes[0].grid(True, axis="y", alpha=0.25)

    for key in ROUTE_HEALTH_KEYS:
        if key in route_rows[0]:
            axes[1].plot(x, [row.get(key, 0.0) for row in route_rows], marker="o", label=key)
    axes[1].set_ylabel("route health")
    add_legend_if_present(axes[1])
    axes[1].grid(True, alpha=0.25)

    for key in ROUTE_ACTION_KEYS:
        if key in route_rows[0]:
            axes[2].plot(x, [row.get(key, 0.0) for row in route_rows], marker="o", label=key)
    axes[2].set_ylabel("continuous actions")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels, rotation=25, ha="right")
    add_legend_if_present(axes[2])
    axes[2].grid(True, alpha=0.25)

    fig.suptitle("Route test summary")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_route_timeseries(route_rows, output_path):
    if not route_rows:
        return
    fig, axes = plt.subplots(4, 1, figsize=(12, 11), sharex=True)
    steps = values(route_rows, "step")

    axes[0].plot(steps, values(route_rows, "x"), label="x", linewidth=1.7)
    axes[0].set_ylabel("x position")
    add_legend_if_present(axes[0])
    axes[0].grid(True, alpha=0.25)

    if "lateral_offset" in route_rows[0]:
        axes[1].plot(steps, values(route_rows, "lateral_offset"), label="lateral_offset", linewidth=1.7)
    axes[1].set_ylabel("lateral offset")
    add_legend_if_present(axes[1])
    axes[1].grid(True, alpha=0.25)

    for key in ("cmd_vx", "measured_vx"):
        if key in route_rows[0]:
            axes[2].plot(steps, values(route_rows, key), label=key, linewidth=1.5)
    axes[2].set_ylabel("velocity")
    add_legend_if_present(axes[2])
    axes[2].grid(True, alpha=0.25)

    gait_to_id = {"pronking": 0, "trotting": 1, "bounding": 2, "pacing": 3}
    if "gait" in route_rows[0]:
        gait_ids = [gait_to_id.get(row["gait"], -1) for row in route_rows]
        axes[3].step(steps, gait_ids, where="post", label="gait_id", linewidth=1.5)
        axes[3].set_yticks(list(gait_to_id.values()))
        axes[3].set_yticklabels(list(gait_to_id.keys()))
    axes[3].set_ylabel("gait")
    axes[3].set_xlabel("step")
    axes[3].grid(True, alpha=0.25)

    fig.suptitle("Route test timeseries")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def write_summary(
    path,
    run_dir,
    rows,
    summary,
    baseline_summary=None,
    route_dir=None,
    route_summary_rows=None,
    route_timeseries_rows=None,
):
    lines = []
    lines.append(f"# Oracle Training Metrics Summary")
    lines.append("")
    lines.append(f"- run_dir: `{run_dir}`")
    lines.append(f"- rows: {len(rows)}")
    lines.append(f"- iterations: {int(rows[0]['iteration'])} to {int(rows[-1]['iteration'])}")
    lines.append("")
    lines.append("## Early vs Late")
    lines.append("")
    lines.append("| metric | early | late | delta |")
    lines.append("|---|---:|---:|---:|")
    for key in KEY_METRICS:
        if key not in summary:
            continue
        item = summary[key]
        lines.append(f"| {key} | {item['early']:.6f} | {item['late']:.6f} | {item['delta']:.6f} |")

    lines.append("")
    lines.append("## Per-Task Action Health")
    lines.append("")
    lines.append("| task | clip early | clip late | switch early | switch late | footswing early | footswing late |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for task_id in TASK_IDS:
        clip_key = f"{task_id}_action_clip_rate"
        switch_key = f"{task_id}_gait_switch_rate"
        foot_key = f"{task_id}_footswing_height_mean"
        if clip_key not in summary or foot_key not in summary:
            continue
        switch_early = summary[switch_key]["early"] if switch_key in summary else float("nan")
        switch_late = summary[switch_key]["late"] if switch_key in summary else float("nan")
        lines.append(
            f"| {task_id} | {summary[clip_key]['early']:.6f} | {summary[clip_key]['late']:.6f} "
            f"| {switch_early:.6f} | {switch_late:.6f} "
            f"| {summary[foot_key]['early']:.6f} | {summary[foot_key]['late']:.6f} |"
        )

    lines.append("")
    lines.append("## Target Gait Ratios")
    lines.append("")
    lines.append("| task | target ratio early | target ratio late | delta |")
    lines.append("|---|---:|---:|---:|")
    for task_id, key in TARGET_RATIO_KEYS.items():
        if key not in summary:
            continue
        item = summary[key]
        lines.append(f"| {task_id} | {item['early']:.6f} | {item['late']:.6f} | {item['delta']:.6f} |")

    if baseline_summary is not None:
        lines.append("")
        lines.append("## Baseline Late vs Current Late")
        lines.append("")
        lines.append("| metric | baseline late | current late | delta |")
        lines.append("|---|---:|---:|---:|")
        for key in KEY_METRICS:
            if key not in baseline_summary or key not in summary:
                continue
            baseline = baseline_summary[key]["late"]
            current = summary[key]["late"]
            lines.append(f"| {key} | {baseline:.6f} | {current:.6f} | {current - baseline:.6f} |")

    if route_summary_rows is not None:
        lines.append("")
        lines.append("## Route Test")
        lines.append("")
        lines.append(f"- route_dir: `{route_dir}`")
        if route_timeseries_rows is not None:
            lines.append(f"- route steps recorded: {len(route_timeseries_rows)}")
        lines.append("")
        lines.append(
            "| segment | condition | target | steps | reward | vx_err | lateral_offset | done | clip | switch | "
            "pronk | trot | bound | pace |"
        )
        lines.append("|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for row in route_summary_rows:
            lines.append(
                f"| {int(row['segment'])} | {row['condition']} | {row['target_gait']} "
                f"| {int(row['steps'])} | {row['reward_mean']:.6f} | {row['vx_err_mean']:.6f} "
                f"| {row.get('lateral_offset_mean', 0.0):.6f} | {row['done_rate']:.6f} "
                f"| {row['action_clip_rate']:.6f} | {row.get('gait_switch_rate', 0.0):.6f} "
                f"| {row['pronk_ratio']:.6f} "
                f"| {row['trot_ratio']:.6f} | {row['bound_ratio']:.6f} | {row['pace_ratio']:.6f} |"
            )

    path.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--baseline-run-dir", default=None)
    parser.add_argument("--window", type=int, default=10)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--route-test-dir",
        default=None,
        help="Route-test directory containing route_summary.csv and route_timeseries.csv. Defaults to latest route_tests/*.",
    )
    parser.add_argument("--no-route", action="store_true")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    metrics_path = run_dir / "metrics.csv"
    rows = read_metrics(metrics_path)
    window = min(args.window, len(rows))
    output_dir = Path(args.output_dir) if args.output_dir else run_dir / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = summarize(rows, window)
    baseline_summary = None
    if args.baseline_run_dir:
        baseline_rows = read_metrics(Path(args.baseline_run_dir) / "metrics.csv")
        baseline_summary = summarize(baseline_rows, min(args.window, len(baseline_rows)))

    route_dir = None
    route_summary_rows = None
    route_timeseries_rows = None
    if not args.no_route:
        route_dir = Path(args.route_test_dir) if args.route_test_dir else latest_route_test_dir(run_dir)
        if route_dir is not None:
            route_summary_rows = read_metrics(route_dir / "route_summary.csv")
            route_timeseries_rows = read_metrics(route_dir / "route_timeseries.csv")

    plot_lines(
        rows,
        (
            ("reward", ("reward", "weighted_metric_reward")),
            ("tracking", ("vx_err", "score_progress", "lateral_position_penalty")),
            ("gait stability", ("gait_switch_rate", "gait_switch_penalty", "score_gait_stability")),
            ("clearance/action", ("score_clearance", "footswing_height_mean", "action_clip_rate")),
            (
                "action health",
                ("score_action_smoothness", "score_action_magnitude", "score_action_boundary_margin"),
            ),
        ),
        output_dir / "training_overview.png",
        "Oracle training overview",
    )
    plot_lines(
        rows,
        (
            ("global gait ratios", ("gait_pronk_ratio", "gait_trot_ratio", "gait_bound_ratio", "gait_pace_ratio")),
            ("task target ratios", tuple(TARGET_RATIO_KEYS.values())),
            ("per-task switch rate", tuple(f"{task}_gait_switch_rate" for task in TASK_IDS)),
        ),
        output_dir / "gait_ratios.png",
        "Gait selection ratios",
    )
    plot_lines(
        rows,
        (
            ("per-task clip rate", tuple(f"{task}_action_clip_rate" for task in TASK_IDS)),
            ("per-task footswing", tuple(f"{task}_footswing_height_mean" for task in TASK_IDS)),
            ("per-task body pitch", tuple(f"{task}_body_pitch_mean" for task in TASK_IDS)),
        ),
        output_dir / "per_task_actions.png",
        "Per-task continuous actions",
    )
    plot_summary_bars(
        summary,
        KEY_METRICS,
        output_dir / "early_late_key_metrics.png",
        f"Early vs late window={window}",
    )
    if baseline_summary is not None:
        plot_compare(
            summary,
            baseline_summary,
            KEY_METRICS,
            output_dir / "baseline_compare_key_metrics.png",
            "Baseline late vs current late",
        )
    if route_summary_rows is not None:
        plot_route_summary(route_summary_rows, output_dir / "route_summary.png")
    if route_timeseries_rows is not None:
        plot_route_timeseries(route_timeseries_rows, output_dir / "route_timeseries.png")

    write_summary(
        output_dir / "summary.md",
        run_dir,
        rows,
        summary,
        baseline_summary,
        route_dir,
        route_summary_rows,
        route_timeseries_rows,
    )
    print(f"Wrote analysis to: {output_dir}")
    print(f"Summary: {output_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
