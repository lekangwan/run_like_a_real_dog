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


def split_history(history, history_length):
    if history.ndim != 2 or history.shape[1] % history_length != 0:
        raise ValueError(
            f"history shape {tuple(history.shape)} cannot be split into "
            f"{history_length} time steps"
        )
    return history.reshape(history.shape[0], history_length, -1)


def make_temporal_summary(history, history_length):
    sequence = split_history(history, history_length)
    if history_length > 1:
        mean_abs_step_change = torch.diff(sequence, dim=1).abs().mean(dim=1)
    else:
        mean_abs_step_change = torch.zeros_like(sequence[:, 0])
    return torch.cat(
        [
            sequence[:, -1],
            sequence.mean(dim=1),
            sequence.std(dim=1, unbiased=False),
            sequence.amin(dim=1),
            sequence.amax(dim=1),
            sequence[:, -1] - sequence[:, 0],
            mean_abs_step_change,
        ],
        dim=1,
    )


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


def summarize_temporal_stability(metadata, num_envs):
    if not num_envs or not metadata.get("child_dirs"):
        return []
    task_names = {int(k): v for k, v in metadata["task_id_by_index"].items()}
    rows = []
    for child_dir_text in metadata["child_dirs"]:
        child_dir = Path(child_dir_text)
        data_path = child_dir / "info_path_samples.npz"
        if not data_path.exists():
            continue
        data = np.load(data_path, allow_pickle=True)
        samples = int(data["task_index"].shape[0])
        time_steps = samples // num_envs
        if time_steps < 2:
            continue
        usable = time_steps * num_envs

        z = data["z_student"][:usable].reshape(time_steps, num_envs, -1)
        probs = data["gait_probs_student"][:usable].reshape(time_steps, num_envs, -1)
        done = data["done"][:usable].reshape(time_steps, num_envs) > 0.5
        valid = ~(done[:-1] | done[1:])
        if not np.any(valid):
            continue

        z_delta = z[1:] - z[:-1]
        prob_delta = probs[1:] - probs[:-1]
        top_gait = np.argmax(probs, axis=-1)
        switch = top_gait[1:] != top_gait[:-1]
        sorted_probs = np.sort(probs.reshape(-1, probs.shape[-1]), axis=-1)

        task_index = int(data["task_index"][0])
        rows.append(
            {
                "task_id": task_names.get(task_index, str(task_index)),
                "cmd_vx": float(data["cmd_vx"][0]),
                "valid_pairs": int(valid.sum()),
                "z_mean_abs_delta": float(np.abs(z_delta).mean(axis=-1)[valid].mean()),
                "z_l2_delta": float(np.sqrt(np.square(z_delta).sum(axis=-1))[valid].mean()),
                "gait_prob_mean_abs_delta": float(
                    np.abs(prob_delta).mean(axis=-1)[valid].mean()
                ),
                "top_gait_switch_rate": float(switch[valid].mean()),
                "gait_prob_margin_mean": float(
                    np.mean(sorted_probs[:, -1] - sorted_probs[:, -2])
                ),
            }
        )
    return rows


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_summary(path, probe_rows, sensitivity_rows, task_rows, stability_rows=None):
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
    if stability_rows:
        lines += [
            "",
            "## Temporal Stability",
            "",
            "Adjacent samples refer to the same simulated robot and exclude pairs whose",
            "sampled endpoints contain a reset.",
            "",
            "| task | vx | pairs | z_abs_delta | prob_abs_delta | top_switch | margin |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for row in stability_rows:
            lines.append(
                f"| {row['task_id']} | {row['cmd_vx']:.2f} | {row['valid_pairs']} "
                f"| {row['z_mean_abs_delta']:.4f} "
                f"| {row['gait_prob_mean_abs_delta']:.4f} "
                f"| {row['top_gait_switch_rate']:.3f} "
                f"| {row['gait_prob_margin_mean']:.3f} |"
            )
    Path(path).write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--history-length", type=int, default=10)
    parser.add_argument("--previous-action-dim", type=int, default=9)
    parser.add_argument(
        "--collection-num-envs",
        type=int,
        default=None,
        help="Parallel environment count used during collection; enables temporal stability analysis.",
    )
    args = parser.parse_args()

    data, metadata = load_data(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else Path(args.input_dir) / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    history = torch.tensor(data["history"], dtype=torch.float32)
    history_sequence = split_history(history, args.history_length)
    if args.previous_action_dim < 0 or args.previous_action_dim >= history_sequence.shape[-1]:
        raise ValueError(
            f"previous-action dim must be in [0, {history_sequence.shape[-1] - 1}], "
            f"got {args.previous_action_dim}"
        )
    sensor_sequence = (
        history_sequence
        if args.previous_action_dim == 0
        else history_sequence[:, :, :-args.previous_action_dim]
    )
    sensor_history = sensor_sequence.reshape(sensor_sequence.shape[0], -1)
    features = {
        "history": history,
        "history_temporal_summary": make_temporal_summary(history, args.history_length),
        "history_without_previous_action": sensor_history,
        "history_temporal_summary_without_previous_action": make_temporal_summary(
            sensor_history,
            args.history_length,
        ),
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
    stability_rows = summarize_temporal_stability(metadata, args.collection_num_envs)

    write_csv(output_dir / "probe_results.csv", probe_rows)
    write_csv(output_dir / "latent_sensitivity.csv", sensitivity_rows)
    write_csv(output_dir / "task_speed_gait_probabilities.csv", task_rows)
    write_csv(output_dir / "temporal_stability.csv", stability_rows)
    (output_dir / "speed_label_values.json").write_text(json.dumps(speed_values, indent=2) + "\n")
    write_summary(
        output_dir / "summary.md",
        probe_rows,
        sensitivity_rows,
        task_rows,
        stability_rows,
    )

    print(f"Wrote: {output_dir / 'probe_results.csv'}")
    print(f"Wrote: {output_dir / 'latent_sensitivity.csv'}")
    print(f"Wrote: {output_dir / 'task_speed_gait_probabilities.csv'}")
    print(f"Wrote: {output_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
