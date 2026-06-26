import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
import time


GAIT_NAMES = ("pronking", "trotting", "bounding", "pacing")


def _to_float(value, default=float("nan")):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _mean(values):
    values = list(values)
    if not values:
        return float("nan")
    return sum(values) / len(values)


def _softmax(scores, temperature):
    best = max(scores)
    scale = max(float(temperature), 1e-8)
    values = [math.exp((score - best) / scale) for score in scores]
    total = sum(values)
    if total <= 0.0:
        return [1.0 / len(scores)] * len(scores)
    return [value / total for value in values]


def read_source(path, score_key, top_k_per_gait):
    grouped = defaultdict(list)
    with open(path, newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            gait = row.get("gait")
            if gait not in GAIT_NAMES:
                continue
            score = _to_float(row.get(score_key))
            if not math.isfinite(score):
                continue
            task_id = row.get("task_id", "")
            condition = row.get("condition", "")
            cmd_vx = round(_to_float(row.get("cmd_vx")), 6)
            grouped[(task_id, condition, cmd_vx, gait)].append(
                {
                    "score": score,
                    "progress": _to_float(row.get("score_progress")),
                    "vx_err": _to_float(row.get("vx_abs_error_mean")),
                }
            )

    by_task_speed = defaultdict(dict)
    for (task_id, condition, cmd_vx, gait), rows in grouped.items():
        rows = sorted(rows, key=lambda item: item["score"], reverse=True)[:top_k_per_gait]
        by_task_speed[(task_id, condition, cmd_vx)][gait] = {
            "score": _mean(row["score"] for row in rows),
            "progress": _mean(row["progress"] for row in rows if math.isfinite(row["progress"])),
            "vx_err": _mean(row["vx_err"] for row in rows if math.isfinite(row["vx_err"])),
            "count": len(rows),
        }
    return by_task_speed


def build_targets(
    input_paths,
    score_key,
    top_k_per_gait,
    temperature,
    confidence_margin,
    progress_min,
    progress_good,
):
    merged = defaultdict(lambda: {gait: [] for gait in GAIT_NAMES})
    meta = {}
    for input_path in input_paths:
        source = read_source(input_path, score_key, top_k_per_gait)
        for task_speed, gait_items in source.items():
            task_id, condition, cmd_vx = task_speed
            meta[task_speed] = {"task_id": task_id, "condition": condition, "cmd_vx": cmd_vx}
            for gait in GAIT_NAMES:
                if gait in gait_items:
                    merged[task_speed][gait].append(gait_items[gait])

    rows = []
    for task_speed in sorted(merged, key=lambda item: (item[0], item[2])):
        gait_values = merged[task_speed]
        if any(not gait_values[gait] for gait in GAIT_NAMES):
            continue

        scores = [_mean(item["score"] for item in gait_values[gait]) for gait in GAIT_NAMES]
        progresses = [_mean(item["progress"] for item in gait_values[gait]) for gait in GAIT_NAMES]
        vx_errs = [_mean(item["vx_err"] for item in gait_values[gait]) for gait in GAIT_NAMES]
        probs = _softmax(scores, temperature)
        order = sorted(range(len(GAIT_NAMES)), key=lambda index: scores[index], reverse=True)
        top_index, second_index = order[0], order[1]
        margin = scores[top_index] - scores[second_index]
        margin_conf = max(0.0, min(1.0, margin / max(confidence_margin, 1e-8)))
        progress_conf = max(
            0.0,
            min(1.0, (progresses[top_index] - progress_min) / max(progress_good - progress_min, 1e-8)),
        )
        confidence = margin_conf * progress_conf
        item = meta[task_speed]
        row = {
            "task_id": item["task_id"],
            "condition": item["condition"],
            "cmd_vx": f"{item['cmd_vx']:.6f}",
            "top_gait": GAIT_NAMES[top_index],
            "second_gait": GAIT_NAMES[second_index],
            "score_margin": f"{margin:.9f}",
            "confidence": f"{confidence:.9f}",
            "top_progress": f"{progresses[top_index]:.9f}",
            "source_count": sum(len(gait_values[gait]) for gait in GAIT_NAMES),
        }
        for gait, prob, score, progress, vx_err in zip(GAIT_NAMES, probs, scores, progresses, vx_errs):
            row[gait] = f"{prob:.9f}"
            row[f"score_{gait}"] = f"{score:.9f}"
            row[f"progress_{gait}"] = f"{progress:.9f}"
            row[f"vx_err_{gait}"] = f"{vx_err:.9f}"
        rows.append(row)
    return rows


def write_summary(rows, path, args):
    counts = defaultdict(int)
    low_conf = 0
    for row in rows:
        counts[row["top_gait"]] += 1
        if float(row["confidence"]) < 0.25:
            low_conf += 1

    lines = [
        "# Soft Selector Targets",
        "",
        "这张表的用途：把已经复测过的行走结果，转成训练时可以读取的步态参考概率。",
        "它不是旧的硬标签，也不是直接替代统一物理奖励；它只给步态选择器一个很小的参考信号。",
        "",
        "## Inputs",
        "",
    ]
    for input_path in args.input:
        lines.append(f"- `{input_path}`")
    lines += [
        "",
        "## Settings",
        "",
        f"- score_key: `{args.score_key}`",
        f"- top_k_per_gait: `{args.top_k_per_gait}`",
        f"- temperature: `{args.temperature}`",
        f"- confidence_margin: `{args.confidence_margin}`",
        f"- progress_min/progress_good: `{args.progress_min}` / `{args.progress_good}`",
        "",
        "## Top Gait Counts",
        "",
    ]
    for gait in GAIT_NAMES:
        lines.append(f"- {gait}: {counts[gait]}")
    lines += [
        f"- low_confidence_rows(<0.25): {low_conf}",
        "",
        "## Files",
        "",
        f"- `{path.name}`",
    ]
    path.with_name("summary.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        action="append",
        default=None,
        help="Validation CSV. Pass multiple times to average across seeds.",
    )
    parser.add_argument("--score-key", default="weighted_metric_reward_mean")
    parser.add_argument("--top-k-per-gait", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.03)
    parser.add_argument("--confidence-margin", type=float, default=0.03)
    parser.add_argument("--progress-min", type=float, default=0.40)
    parser.add_argument("--progress-good", type=float, default=0.80)
    parser.add_argument(
        "--output",
        default=None,
        help="Output CSV path. Defaults to runs/high_level_oracle_gait/selector_targets/<timestamp>/selector_targets.csv",
    )
    args = parser.parse_args()
    if args.input is None:
        args.input = [
            "runs/high_level_oracle_gait/heldout_validation/20260621_v4_training_range_topk_k3_seed208/fair_gait_grid_results.csv",
            "runs/high_level_oracle_gait/heldout_validation/20260621_v4_training_range_topk_k3_seed209/fair_gait_grid_results.csv",
        ]

    if args.output is None:
        output_dir = Path("runs/high_level_oracle_gait/selector_targets") / time.strftime(
            "%Y%m%d_%H%M%S_v4_training_range"
        )
        output_path = output_dir / "selector_targets.csv"
    else:
        output_path = Path(args.output)
        output_dir = output_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = build_targets(
        [Path(path) for path in args.input],
        args.score_key,
        args.top_k_per_gait,
        args.temperature,
        args.confidence_margin,
        args.progress_min,
        args.progress_good,
    )
    if not rows:
        raise RuntimeError("No selector target rows were generated.")

    fieldnames = list(rows[0].keys())
    with open(output_path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    write_summary(rows, output_path, args)
    print(f"Wrote {len(rows)} selector target rows to {output_path}")
    print(f"Wrote summary to {output_path.with_name('summary.md')}")


if __name__ == "__main__":
    main()
