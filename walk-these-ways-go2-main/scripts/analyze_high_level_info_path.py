import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


def load_data(input_dir):
    input_dir = Path(input_dir)
    data = np.load(input_dir / "info_path_samples.npz", allow_pickle=True)
    metadata = json.loads((input_dir / "metadata.json").read_text())
    return data, metadata


def standardize(train_x, test_x):
    mean = train_x.mean(dim=0, keepdim=True)
    std = train_x.std(dim=0, keepdim=True).clamp_min(1e-6)
    return (train_x - mean) / std, (test_x - mean) / std


def train_linear_probe(features, labels, num_classes, seed=0, epochs=250, lr=0.05):
    valid = labels >= 0
    features = features[valid]
    labels = labels[valid]
    if features.shape[0] < 20 or num_classes < 2 or features.shape[1] == 0:
        return None

    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(features.shape[0], generator=generator)
    split = max(1, int(0.7 * features.shape[0]))
    train_idx = perm[:split]
    test_idx = perm[split:]
    if test_idx.numel() == 0:
        test_idx = train_idx

    train_x = features[train_idx]
    test_x = features[test_idx]
    train_y = labels[train_idx]
    test_y = labels[test_idx]
    train_x, test_x = standardize(train_x, test_x)

    model = nn.Linear(train_x.shape[1], num_classes)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    for _ in range(epochs):
        logits = model(train_x)
        loss = loss_fn(logits, train_y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        train_pred = torch.argmax(model(train_x), dim=-1)
        test_logits = model(test_x)
        test_pred = torch.argmax(test_logits, dim=-1)
        train_acc = (train_pred == train_y).float().mean().item()
        test_acc = (test_pred == test_y).float().mean().item()
        confusion = torch.zeros(num_classes, num_classes, dtype=torch.long)
        for true, pred in zip(test_y.cpu(), test_pred.cpu()):
            confusion[int(true), int(pred)] += 1

    return {
        "samples": int(features.shape[0]),
        "train_acc": train_acc,
        "test_acc": test_acc,
        "confusion": confusion.numpy(),
    }


def make_speed_labels(cmd_vx):
    rounded = np.round(cmd_vx.astype(np.float64), 2)
    values = sorted(float(value) for value in np.unique(rounded))
    index = {value: i for i, value in enumerate(values)}
    labels = np.array([index[float(value)] for value in rounded], dtype=np.int64)
    return labels, values


def mean_abs_prob_diff(a, b):
    return float(np.mean(np.abs(a - b)))


def kl_divergence(p, q):
    p = np.clip(p, 1e-8, 1.0)
    q = np.clip(q, 1e-8, 1.0)
    return float(np.mean(np.sum(p * (np.log(p) - np.log(q)), axis=1)))


def summarize_probability_sensitivity(data):
    student = data["gait_probs_student"]
    teacher = data["gait_probs_teacher"]
    zero_z = data["gait_probs_zero_z"]
    shuffled = data["gait_probs_shuffled_z"]
    rows = [
        {
            "comparison": "student_vs_teacher",
            "mean_abs_prob_diff": mean_abs_prob_diff(student, teacher),
            "mean_kl": kl_divergence(student, teacher),
        },
        {
            "comparison": "student_vs_zero_latent",
            "mean_abs_prob_diff": mean_abs_prob_diff(student, zero_z),
            "mean_kl": kl_divergence(student, zero_z),
        },
        {
            "comparison": "student_vs_shuffled_latent",
            "mean_abs_prob_diff": mean_abs_prob_diff(student, shuffled),
            "mean_kl": kl_divergence(student, shuffled),
        },
    ]
    return rows


def summarize_by_task_speed(data, metadata):
    gait_names = metadata["gait_names"]
    task_names = {int(k): v for k, v in metadata["task_id_by_index"].items()}
    rows = []
    task_index = data["task_index"].astype(np.int64)
    cmd_vx = np.round(data["cmd_vx"].astype(np.float64), 2)
    target_top = data["selector_target_top_gait"].astype(np.int64)
    weight = data["selector_target_weight"]
    probs = data["gait_probs_student"]
    selected = data["executed_gait"].astype(np.int64)

    for task in sorted(np.unique(task_index)):
        for vx in sorted(np.unique(cmd_vx[task_index == task])):
            mask = (task_index == task) & (cmd_vx == vx)
            if not np.any(mask):
                continue
            mean_probs = probs[mask].mean(axis=0)
            top = int(np.argmax(mean_probs))
            target_values = target_top[mask]
            valid_targets = target_values[target_values >= 0]
            ref_top = int(valid_targets[0]) if valid_targets.size else -1
            selected_counts = np.bincount(selected[mask], minlength=len(gait_names))
            row = {
                "task_id": task_names.get(int(task), str(task)),
                "cmd_vx": float(vx),
                "samples": int(mask.sum()),
                "reference_top_gait": gait_names[ref_top] if ref_top >= 0 else "",
                "reference_weight_mean": float(weight[mask].mean()),
                "student_top_gait": gait_names[top],
                "student_top_prob": float(mean_probs[top]),
            }
            for gait_id, gait_name in enumerate(gait_names):
                row[f"prob_{gait_name}"] = float(mean_probs[gait_id])
                row[f"executed_ratio_{gait_name}"] = float(selected_counts[gait_id] / max(1, mask.sum()))
            rows.append(row)
    return rows


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_summary(path, probe_rows, sensitivity_rows, task_rows):
    lines = [
        "# High-Level Information-Path Probe",
        "",
        "This diagnostic checks whether condition information is visible in the history,",
        "preserved in the RMA latents, and used by the gait-selection output.",
        "",
        "## Simple Diagnostic Accuracy",
        "",
        "| feature | target | samples | train_acc | test_acc |",
        "|---|---|---:|---:|---:|",
    ]
    for row in probe_rows:
        lines.append(
            f"| {row['feature']} | {row['target']} | {row['samples']} "
            f"| {row['train_acc']:.3f} | {row['test_acc']:.3f} |"
        )
    lines += [
        "",
        "## Latent Sensitivity",
        "",
        "| comparison | mean_abs_prob_diff | mean_kl |",
        "|---|---:|---:|",
    ]
    for row in sensitivity_rows:
        lines.append(
            f"| {row['comparison']} | {row['mean_abs_prob_diff']:.4f} | {row['mean_kl']:.4f} |"
        )
    lines += [
        "",
        "## Task-Speed Gait Probabilities",
        "",
        "| task | vx | reference | student_top | top_prob | pronk | trot | bound | pace |",
        "|---|---:|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in task_rows:
        lines.append(
            f"| {row['task_id']} | {row['cmd_vx']:.2f} | {row['reference_top_gait']} "
            f"| {row['student_top_gait']} | {row['student_top_prob']:.3f} "
            f"| {row.get('prob_pronking', 0.0):.3f} | {row.get('prob_trotting', 0.0):.3f} "
            f"| {row.get('prob_bounding', 0.0):.3f} | {row.get('prob_pacing', 0.0):.3f} |"
        )
    Path(path).write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    data, metadata = load_data(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else Path(args.input_dir) / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    features = {
        "history": torch.tensor(data["history"], dtype=torch.float32),
        "z_student": torch.tensor(data["z_student"], dtype=torch.float32),
        "z_teacher": torch.tensor(data["z_teacher"], dtype=torch.float32),
    }
    task_labels = torch.tensor(data["task_index"].astype(np.int64), dtype=torch.long)
    speed_labels_np, speed_values = make_speed_labels(data["cmd_vx"])
    speed_labels = torch.tensor(speed_labels_np, dtype=torch.long)
    target_top = torch.tensor(data["selector_target_top_gait"].astype(np.int64), dtype=torch.long)

    target_specs = [
        ("task", task_labels, int(task_labels.max().item()) + 1),
        ("speed", speed_labels, int(speed_labels.max().item()) + 1),
        ("reference_top_gait", target_top, len(metadata["gait_names"])),
    ]

    probe_rows = []
    for feature_name, feature in features.items():
        for target_name, labels, num_classes in target_specs:
            result = train_linear_probe(
                feature,
                labels,
                num_classes,
                seed=args.seed,
                epochs=args.epochs,
            )
            if result is None:
                continue
            probe_rows.append(
                {
                    "feature": feature_name,
                    "target": target_name,
                    "samples": result["samples"],
                    "train_acc": result["train_acc"],
                    "test_acc": result["test_acc"],
                }
            )
            np.savetxt(
                output_dir / f"confusion_{feature_name}_to_{target_name}.csv",
                result["confusion"],
                fmt="%d",
                delimiter=",",
            )

    sensitivity_rows = summarize_probability_sensitivity(data)
    task_rows = summarize_by_task_speed(data, metadata)

    write_csv(output_dir / "probe_results.csv", probe_rows)
    write_csv(output_dir / "latent_sensitivity.csv", sensitivity_rows)
    write_csv(output_dir / "task_speed_gait_probabilities.csv", task_rows)
    (output_dir / "speed_label_values.json").write_text(json.dumps(speed_values, indent=2) + "\n")
    write_summary(output_dir / "summary.md", probe_rows, sensitivity_rows, task_rows)

    print(f"Wrote: {output_dir / 'probe_results.csv'}")
    print(f"Wrote: {output_dir / 'latent_sensitivity.csv'}")
    print(f"Wrote: {output_dir / 'task_speed_gait_probabilities.csv'}")
    print(f"Wrote: {output_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
