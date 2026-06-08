import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

from gait_project_config import (
    MAINLINE_TASK_MAP,
    MAINLINE_TEMPLATE_LIBRARY,
    VIS_EDGE_RESET_MARGIN,
    VIS_TELEPORT_THRESH,
    VIS_TERRAIN_LENGTH,
    VIS_TERRAIN_WIDTH,
)


def parse_strings(value):
    return [item.strip() for item in value.split(",") if item.strip()]


def format_vx_list(values):
    return ",".join(f"{float(value):g}" for value in values)


def build_scene_rows(task_map, include_label_types, task_ids):
    rows = task_map[task_map["use_for_training"] == "yes"].copy()
    if include_label_types:
        rows = rows[rows["speed_label_type"].isin(include_label_types)]
    if task_ids:
        rows = rows[rows["task_id"].isin(task_ids)]
    if rows.empty:
        raise ValueError("No training scenes matched the requested filters.")

    scenes = []
    group_cols = ["task_id", "condition", "target_gait", "speed_label_type"]
    for keys, group in rows.groupby(group_cols, sort=False):
        task_id, condition, target_gait, label_type = keys
        vx_values = sorted(group["vx"].astype(float).unique())
        scenes.append(
            {
                "task_id": task_id,
                "condition": condition,
                "target_gait": target_gait,
                "label_type": label_type,
                "vx_values": vx_values,
                "style_reward_strength": ",".join(sorted(group["style_reward_strength"].astype(str).unique())),
            }
        )
    return scenes


def scene_command(scene, args):
    vx_values = scene["vx_values"]
    num_envs = min(args.max_envs, len(vx_values))
    cmd = [
        sys.executable,
        "scripts/play_task_gait_oracle.py",
        "--library",
        args.library,
        "--task-id",
        scene["task_id"],
        "--condition",
        scene["condition"],
        "--num-envs",
        str(num_envs),
        "--vx-list",
        format_vx_list(vx_values[:num_envs]),
        "--spread-vx",
        "--steps",
        str(args.steps),
        "--print-interval",
        str(args.print_interval),
        "--terrain-length",
        str(args.terrain_length),
        "--terrain-width",
        str(args.terrain_width),
        "--teleport-thresh",
        str(args.teleport_thresh),
    ]
    if args.edge_reset_margin is not None:
        cmd.extend(["--edge-reset-margin", str(args.edge_reset_margin)])
    if args.no_render:
        cmd.append("--no-render")
    return cmd


def print_scene(index, total, scene):
    print("\n" + "=" * 80)
    print(f"Scene {index}/{total}")
    print(f"task_id: {scene['task_id']}")
    print(f"condition: {scene['condition']}")
    print(f"target_gait: {scene['target_gait']}")
    print(f"label_type: {scene['label_type']}")
    print(f"vx: {format_vx_list(scene['vx_values'])}")
    print(f"style_reward_strength: {scene['style_reward_strength']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-map", default=str(MAINLINE_TASK_MAP))
    parser.add_argument("--library", default=str(MAINLINE_TEMPLATE_LIBRARY))
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--max-envs", type=int, default=4)
    parser.add_argument("--print-interval", type=int, default=200)
    parser.add_argument("--terrain-length", type=float, default=VIS_TERRAIN_LENGTH)
    parser.add_argument("--terrain-width", type=float, default=VIS_TERRAIN_WIDTH)
    parser.add_argument("--teleport-thresh", type=float, default=VIS_TELEPORT_THRESH)
    parser.add_argument("--edge-reset-margin", type=float, default=VIS_EDGE_RESET_MARGIN)
    parser.add_argument(
        "--include-label-types",
        type=parse_strings,
        default=None,
        help="Comma-separated label types to show, e.g. hard,conditional_hard. Defaults to all training rows.",
    )
    parser.add_argument(
        "--task-ids",
        type=parse_strings,
        default=None,
        help="Comma-separated task ids to show. Defaults to all training scenes.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-pause", action="store_true")
    parser.add_argument("--no-render", action="store_true")
    args = parser.parse_args()

    task_map_path = Path(args.task_map)
    if not task_map_path.exists():
        raise FileNotFoundError(f"Task map not found: {task_map_path}")
    if not Path(args.library).exists():
        raise FileNotFoundError(f"Template library not found: {args.library}")

    task_map = pd.read_csv(task_map_path)
    scenes = build_scene_rows(task_map, args.include_label_types, args.task_ids)
    print(f"Found {len(scenes)} training scenes from {task_map_path}")

    for index, scene in enumerate(scenes, start=1):
        print_scene(index, len(scenes), scene)
        cmd = scene_command(scene, args)
        print("Command:")
        print(" ".join(cmd))

        if args.dry_run:
            continue
        if not args.no_pause:
            input("Press Enter to start this scene, or Ctrl+C to stop...")
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
