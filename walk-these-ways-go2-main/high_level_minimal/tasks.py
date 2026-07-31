from dataclasses import dataclass
from types import SimpleNamespace
import csv

import torch

from go2_gym.envs.wrappers.high_level_reward_metrics import (
    CANONICAL_REWARD_NAMES,
    UNIFIED_REWARD_PROFILES,
)

from .config import GAIT_NAMES, REWARD_PROFILE


@dataclass
class TaskSpec:
    task_id: str
    condition: str
    vx_low: float
    vx_high: float
    sampling_weight: float = 1.0


def read_task_specs(path, selected_task_ids):
    selected = set(selected_task_ids)
    grouped = {}
    with open(path, newline="") as file:
        for row in csv.DictReader(file):
            task_id = row["task_id"]
            if task_id not in selected or row.get("use_for_training", "yes") != "yes":
                continue
            entry = grouped.setdefault(
                task_id,
                {"condition": row["condition"], "speeds": []},
            )
            entry["speeds"].append(float(row["vx"]))

    missing = selected.difference(grouped)
    if missing:
        raise ValueError(f"Task map is missing selected tasks: {sorted(missing)}")

    return [
        TaskSpec(
            task_id=task_id,
            condition=grouped[task_id]["condition"],
            vx_low=min(grouped[task_id]["speeds"]),
            vx_high=max(grouped[task_id]["speeds"]),
        )
        for task_id in selected_task_ids
    ]


def parse_eval_specs(text, path):
    """Parse task:speed pairs into fixed-speed evaluation specifications."""
    task_ids = [item.split(":", 1)[0].strip() for item in text.split(",")]
    base_specs = {spec.task_id: spec for spec in read_task_specs(path, task_ids)}
    specs = []
    for item in text.split(","):
        task_id, speed_text = item.strip().split(":", 1)
        base = base_specs[task_id]
        speed = float(speed_text)
        specs.append(
            TaskSpec(
                task_id=f"{task_id}@{speed:g}",
                condition=base.condition,
                vx_low=speed,
                vx_high=speed,
            )
        )
    return specs


def reward_weight_vector():
    profile = UNIFIED_REWARD_PROFILES[REWARD_PROFILE]
    return [float(profile.get(name, 0.0)) for name in CANONICAL_REWARD_NAMES]


def _allocate_counts(specs, num_envs):
    if num_envs < len(specs):
        raise ValueError("num_envs must be at least the number of task specifications")
    weights = torch.tensor([spec.sampling_weight for spec in specs], dtype=torch.float64)
    quotas = weights / weights.sum() * num_envs
    counts = torch.floor(quotas).to(torch.long)
    counts.clamp_(min=1)
    while int(counts.sum()) < num_envs:
        counts[torch.argmax(quotas - counts)] += 1
    while int(counts.sum()) > num_envs:
        candidates = torch.where(counts > 1)[0]
        counts[candidates[torch.argmin(quotas[candidates] - counts[candidates])]] -= 1
    return counts.tolist()


def build_assignment(specs, num_envs, device):
    task_ids = []
    conditions = []
    vx_lows = []
    vx_highs = []
    push_axes = []
    reward_weights = []
    weights = reward_weight_vector()

    for task_index, (spec, count) in enumerate(zip(specs, _allocate_counts(specs, num_envs))):
        task_ids.extend([task_index] * count)
        conditions.extend([spec.condition] * count)
        vx_lows.extend([spec.vx_low] * count)
        vx_highs.extend([spec.vx_high] * count)
        push_axes.extend([1 if spec.condition == "push_lateral" else -1] * count)
        reward_weights.extend([weights] * count)

    return SimpleNamespace(
        task_ids=torch.tensor(task_ids, device=device, dtype=torch.long),
        conditions=conditions,
        vx_lows=torch.tensor(vx_lows, device=device),
        vx_highs=torch.tensor(vx_highs, device=device),
        push_axes=push_axes,
        reward_weights=torch.tensor(reward_weights, device=device),
    )


def fixed_gait_action(num_envs, gait_name, device):
    gait_id = GAIT_NAMES.index(gait_name)
    action = torch.zeros(num_envs, len(GAIT_NAMES) + 5, device=device)
    action[:, gait_id] = 1.0
    return action
