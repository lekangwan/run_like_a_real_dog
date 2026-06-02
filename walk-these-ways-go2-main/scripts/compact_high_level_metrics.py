import argparse
import csv
from pathlib import Path


GAITS = (
    ("pronking", "pronk"),
    ("trotting", "trot"),
    ("bounding", "bound"),
    ("pacing", "pace"),
)


BASE_COLUMNS = (
    "iteration",
    "reward",
    "done",
    "vx",
    "vx_err",
    "policy_loss",
    "value_loss",
    "entropy",
    "log_std_mean",
)


ACTION_COLUMNS = (
    ("cmd_phase", "action/phase_mean"),
    ("cmd_offset", "action/offset_mean"),
    ("cmd_bound", "action/bound_mean"),
    ("cmd_freq", "action/frequency_mean"),
    ("cmd_swing", "action/footswing_height_mean"),
    ("cmd_stance", "action/stance_width_mean"),
    ("cmd_pitch", "action/body_pitch_mean"),
)


def first_present(row, *names):
    for name in names:
        if name in row:
            return row[name]
    return ""


def old_bin_prefix(row, bin_id):
    prefix = f"vxbin{bin_id}_"
    matches = [key.split("/")[0] for key in row if key.startswith(prefix)]
    return matches[0] if matches else None


def compact_row(row):
    out = {name: first_present(row, name) for name in BASE_COLUMNS}

    for gait_name, short_name in GAITS:
        out[f"sel_{short_name}"] = first_present(
            row, f"sel_{short_name}", f"selector/{gait_name}_mean"
        )
        out[f"target_sel_{short_name}"] = first_present(
            row, f"target_sel_{short_name}", f"target_selector/{gait_name}_mean"
        )

    for compact_name, verbose_name in ACTION_COLUMNS:
        out[compact_name] = first_present(row, compact_name, verbose_name)

    for bin_id in range(4):
        compact_prefix = f"b{bin_id}"
        verbose_prefix = old_bin_prefix(row, bin_id)
        out[f"{compact_prefix}_count"] = first_present(
            row,
            f"{compact_prefix}_count",
            f"{verbose_prefix}/count" if verbose_prefix else "",
        )
        for gait_name, short_name in GAITS:
            out[f"{compact_prefix}_sel_{short_name}"] = first_present(
                row,
                f"{compact_prefix}_sel_{short_name}",
                f"{verbose_prefix}/selector_{gait_name}" if verbose_prefix else "",
            )
        out[f"{compact_prefix}_target_trot"] = first_present(
            row,
            f"{compact_prefix}_target_trot",
            f"{verbose_prefix}/target_selector_trotting" if verbose_prefix else "",
        )
        out[f"{compact_prefix}_target_bound"] = first_present(
            row,
            f"{compact_prefix}_target_bound",
            f"{verbose_prefix}/target_selector_bounding" if verbose_prefix else "",
        )
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("metrics_csv")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    input_path = Path(args.metrics_csv)
    output_path = Path(args.output) if args.output else input_path.with_name("metrics_compact.csv")

    with open(input_path, newline="") as input_file:
        reader = csv.DictReader(input_file)
        rows = [compact_row(row) for row in reader]

    if not rows:
        raise ValueError(f"No rows found in {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote compact metrics: {output_path}")


if __name__ == "__main__":
    main()
