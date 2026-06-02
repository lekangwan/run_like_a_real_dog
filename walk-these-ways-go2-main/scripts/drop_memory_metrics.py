import argparse
import csv
from pathlib import Path


MEMORY_COLUMNS = {
    "cuda_allocated_mb",
    "cuda_reserved_mb",
    "cuda_max_allocated_mb",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("metrics_csv")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    input_path = Path(args.metrics_csv)
    output_path = Path(args.output) if args.output else input_path.with_name("metrics_no_memory.csv")

    with open(input_path, newline="") as input_file:
        reader = csv.DictReader(input_file)
        fieldnames = [name for name in reader.fieldnames or [] if name not in MEMORY_COLUMNS]
        rows = [{name: row[name] for name in fieldnames} for row in reader]

    if not fieldnames:
        raise ValueError(f"No columns found in {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    removed = sorted(MEMORY_COLUMNS & set((reader.fieldnames or [])))
    print(f"Wrote metrics without memory columns: {output_path}")
    print(f"Removed columns: {removed}")


if __name__ == "__main__":
    main()
