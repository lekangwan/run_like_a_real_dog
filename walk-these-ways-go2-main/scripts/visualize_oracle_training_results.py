import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


KEY_METRICS = (
    "reward",
    "weighted_metric_reward",
    "vx_err",
    "score_progress",
    "score_clearance",
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


def values(rows, key):
    return [row[key] for row in rows if key in row]


def mean(items):
    items = list(items)
    return sum(items) / len(items) if items else float("nan")


def window_mean(rows, key, window, end=False):
    subset = rows[-window:] if end else rows[:window]
    return mean(row[key] for row in subset if key in row)


def summarize(rows, window):
    summary = {}
    keys = [key for key in KEY_METRICS if key in rows[0]]
    for task_id in TASK_IDS:
        for suffix in (
            "action_clip_rate",
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
        ax.legend(loc="best", fontsize=8)
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


def write_summary(path, run_dir, rows, summary, baseline_summary=None):
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
    lines.append("| task | clip early | clip late | footswing early | footswing late |")
    lines.append("|---|---:|---:|---:|---:|")
    for task_id in TASK_IDS:
        clip_key = f"{task_id}_action_clip_rate"
        foot_key = f"{task_id}_footswing_height_mean"
        if clip_key not in summary or foot_key not in summary:
            continue
        lines.append(
            f"| {task_id} | {summary[clip_key]['early']:.6f} | {summary[clip_key]['late']:.6f} "
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

    path.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--baseline-run-dir", default=None)
    parser.add_argument("--window", type=int, default=10)
    parser.add_argument("--output-dir", default=None)
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

    plot_lines(
        rows,
        (
            ("reward", ("reward", "weighted_metric_reward")),
            ("tracking", ("vx_err", "score_progress")),
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

    write_summary(output_dir / "summary.md", run_dir, rows, summary, baseline_summary)
    print(f"Wrote analysis to: {output_dir}")
    print(f"Summary: {output_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
