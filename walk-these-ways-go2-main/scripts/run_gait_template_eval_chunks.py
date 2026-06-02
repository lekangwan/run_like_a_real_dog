import argparse
import csv
import subprocess
import sys
from pathlib import Path


def parse_csv_values(value):
    return [item.strip() for item in value.split(",") if item.strip()]


def grid_size(args):
    total = 1
    for value in (
        args.vx,
        args.gaits,
        args.frequencies,
        args.footswing_heights,
        args.body_pitches,
        args.stance_widths,
    ):
        total *= len(parse_csv_values(value))
    return total


def best_rows(rows, group_keys, score_key="template_score"):
    grouped = {}
    for row in rows:
        key = tuple(row[name] for name in group_keys)
        if key not in grouped or float(row[score_key]) > float(grouped[key][score_key]):
            grouped[key] = row
    return list(grouped.values())


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def merge_chunks(output_dir):
    rows = []
    fieldnames = None
    chunk_paths = sorted((output_dir / "chunks").glob("chunk_*/template_eval_results.csv"))
    if not chunk_paths:
        raise FileNotFoundError(f"No chunk CSV files found under {output_dir / 'chunks'}")

    for path in chunk_paths:
        with open(path, newline="") as file:
            reader = csv.DictReader(file)
            if fieldnames is None:
                fieldnames = reader.fieldnames
            rows.extend(reader)

    write_csv(output_dir / "template_eval_results.csv", rows, fieldnames)
    write_csv(output_dir / "best_by_speed.csv", best_rows(rows, ["vx"]), fieldnames)
    write_csv(output_dir / "best_by_speed_gait.csv", best_rows(rows, ["vx", "gait"]), fieldnames)
    return len(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--label", default="gait-conditioned-agility/pretrain-go2/train")
    parser.add_argument("--run-index", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--warmup-steps", type=int, default=80)
    parser.add_argument("--eval-steps", type=int, default=200)
    parser.add_argument("--chunk-configs", type=int, default=128)
    parser.add_argument("--vx", default="0.2,0.5,0.8,1.2,1.6,2.0")
    parser.add_argument("--gaits", default="pronking,trotting,bounding,pacing")
    parser.add_argument("--frequencies", default="2.0,2.5,3.0,3.5")
    parser.add_argument("--footswing-heights", default="0.06,0.08,0.10")
    parser.add_argument("--body-pitches", default="-0.06,0.0")
    parser.add_argument("--stance-widths", default="0.28,0.33,0.38")
    parser.add_argument("--output-dir", default="logs/gait_template_eval_v2_grid_chunked")
    parser.add_argument("--log-memory", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    total = grid_size(args)
    script_path = Path(__file__).with_name("evaluate_gait_templates.py")
    print(
        f"Running {total} configs in chunks of {args.chunk_configs}; "
        f"output_dir={output_dir}",
        flush=True,
    )

    for start in range(0, total, args.chunk_configs):
        max_configs = min(args.chunk_configs, total - start)
        chunk_id = start // args.chunk_configs
        chunk_dir = output_dir / "chunks" / f"chunk_{chunk_id:04d}"
        cmd = [
            args.python,
            str(script_path),
            "--label",
            args.label,
            "--run-index",
            str(args.run_index),
            "--batch-size",
            str(args.batch_size),
            "--warmup-steps",
            str(args.warmup_steps),
            "--eval-steps",
            str(args.eval_steps),
            "--vx",
            args.vx,
            "--gaits",
            args.gaits,
            "--frequencies",
            args.frequencies,
            "--footswing-heights",
            args.footswing_heights,
            f"--body-pitches={args.body_pitches}",
            "--stance-widths",
            args.stance_widths,
            "--start-index",
            str(start),
            "--max-configs",
            str(max_configs),
            "--output-dir",
            str(chunk_dir),
        ]
        if args.log_memory:
            cmd.append("--log-memory")

        print(
            f"\n=== chunk {chunk_id:04d}: configs {start}:{start + max_configs}/{total} ===",
            flush=True,
        )
        subprocess.run(cmd, check=True)

    merged = merge_chunks(output_dir)
    print(f"\nMerged {merged} rows into {output_dir}", flush=True)


if __name__ == "__main__":
    main()
