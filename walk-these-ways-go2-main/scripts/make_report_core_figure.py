#!/usr/bin/env python3
"""Create the report-stage flat/ramp gait-selection result figure."""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


TASK_LABELS = {
    "flat_trot_efficiency": "平地",
    "ramp_up_trot_robustness": "上坡",
}
TASK_COLORS = {
    "flat_trot_efficiency": "#287271",
    "ramp_up_trot_robustness": "#D05A47",
}


def read_csv(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def load_gait_ratios(eval_root, seeds):
    values = {}
    for seed in seeds:
        path = eval_root / f"20260720_seed{seed}_adaptive" / "independent_eval_summary.csv"
        for row in read_csv(path):
            task = row["task_id"]
            if task not in TASK_LABELS:
                continue
            key = (task, float(row["cmd_vx"]))
            values.setdefault(key, []).append(float(row["pronk_ratio"]))
    return values


def load_paired_rows(path):
    rows = {}
    for row in read_csv(path):
        task = row["task_id"]
        if task in TASK_LABELS:
            rows[(task, float(row["cmd_vx"]))] = row
    return rows


def plot_task_lines(ax, speeds, getter, ylabel, title, zero_line=False):
    for task in TASK_LABELS:
        means = []
        errors = []
        for speed in speeds:
            mean, error = getter(task, speed)
            means.append(mean)
            errors.append(error)
        ax.errorbar(
            speeds,
            means,
            yerr=errors,
            marker="o",
            markersize=5,
            linewidth=2,
            capsize=3,
            color=TASK_COLORS[task],
            label=TASK_LABELS[task],
        )
    if zero_line:
        ax.axhline(0.0, color="#666666", linewidth=1, linestyle="--")
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("目标速度 (m/s)")
    ax.set_ylabel(ylabel)
    ax.set_xticks(speeds)
    ax.grid(True, alpha=0.22)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--paired-csv", type=Path, required=True)
    parser.add_argument("--seeds", default="22650,22750,22850")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    speeds = [0.5, 1.0, 1.5, 2.0]
    gait_ratios = load_gait_ratios(args.eval_root, seeds)
    paired = load_paired_rows(args.paired_csv)

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Noto Sans CJK JP", "Droid Sans Fallback", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "figure.facecolor": "white",
            "axes.facecolor": "#FAFAF8",
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.4), constrained_layout=True)

    def gait_getter(task, speed):
        samples = np.asarray(gait_ratios[(task, speed)], dtype=float)
        return 100.0 * samples.mean(), 100.0 * samples.std(ddof=0)

    def paired_getter(mean_key, std_key):
        def getter(task, speed):
            row = paired[(task, speed)]
            return float(row[mean_key]), float(row[std_key])

        return getter

    plot_task_lines(
        axes[0, 0],
        speeds,
        gait_getter,
        "双脚跳选择比例 (%)",
        "A. 学到的步态选择",
    )
    axes[0, 0].set_ylim(-3, 103)
    axes[0, 0].legend(frameon=False, loc="best")

    plot_task_lines(
        axes[0, 1],
        speeds,
        paired_getter("delta_reward_mean_mean", "delta_reward_mean_std"),
        "统一物理奖励差",
        "B. 相对固定小跑的奖励变化",
        zero_line=True,
    )
    plot_task_lines(
        axes[1, 0],
        speeds,
        paired_getter("delta_vx_err_mean_mean", "delta_vx_err_mean_std"),
        "速度误差差（负值更好）",
        "C. 速度跟踪变化",
        zero_line=True,
    )
    plot_task_lines(
        axes[1, 1],
        speeds,
        paired_getter(
            "delta_mechanical_power_abs_mean",
            "delta_mechanical_power_abs_std",
        ),
        "机械功率差（正值代价更高）",
        "D. 能耗代价",
        zero_line=True,
    )

    fig.suptitle(
        "本体感知高层步态选择器：相对固定小跑的三随机种子结果",
        fontsize=15,
        fontweight="bold",
    )
    fig.text(
        0.5,
        -0.012,
        "误差线表示三个评测随机种子的标准差；除步态比例外，差值均为自适应选择器减去固定小跑。",
        ha="center",
        fontsize=9,
        color="#444444",
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    png_path = args.output_dir / "core_result_figure.png"
    pdf_path = args.output_dir / "core_result_figure.pdf"
    fig.savefig(png_path, dpi=240, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    print(f"Wrote: {png_path}")
    print(f"Wrote: {pdf_path}")


if __name__ == "__main__":
    main()
