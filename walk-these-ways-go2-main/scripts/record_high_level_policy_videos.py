"""Record short, close-following videos of a trained high-level gait policy."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import isaacgym

assert isaacgym
from isaacgym import gymapi
import cv2
import numpy as np
import torch

from gait_project_config import MAINLINE_TASK_MAP, TRAIN_MESH_TYPE
from play_oracle_policy_training_map import (
    append_command_vx_obs,
    augment_for_checkpoint,
    latest_checkpoint,
    load_model,
    load_run_args,
    select_eval_specs,
    set_deterministic_vx,
)
from train_high_level_oracle_ppo import (
    GAIT_SHORT_NAMES,
    OracleConditionHighLevelEnv,
    read_task_specs,
)
from train_high_level_ppo import find_logdir, load_low_level_policy


def video_stem(task_id, vx):
    return f"{task_id}_vx{vx:.2f}".replace(".", "p")


def parse_single_eval(eval_text):
    if "," in eval_text or ":" not in eval_text:
        raise ValueError("A recording child requires exactly one task_id:vx item")
    task_id, vx_text = eval_text.split(":", 1)
    return task_id, float(vx_text)


def low_level_env(oracle_env):
    # Oracle -> high-level gait wrapper -> history wrapper -> Isaac Gym task.
    return oracle_env.env.env.env


def encode_frame(writer, rgba):
    image = np.asarray(rgba)
    if image.ndim == 2:
        image = image.reshape(image.shape[0], image.shape[1] // 4, 4)
    if image.ndim != 3 or image.shape[2] < 3:
        raise RuntimeError(f"Unexpected Isaac Gym camera image shape: {image.shape}")
    writer.write(cv2.cvtColor(image[:, :, :3], cv2.COLOR_RGB2BGR))


def set_static_camera(raw_env, forward_offset, lateral_offset, camera_height):
    bx, by, bz = (float(value) for value in raw_env.root_states[0, :3])
    target_x = bx + forward_offset
    raw_env.gym.set_camera_location(
        raw_env.rendering_camera,
        raw_env.envs[0],
        gymapi.Vec3(target_x, by - lateral_offset, bz + camera_height),
        gymapi.Vec3(target_x, by, bz - 0.15),
    )


def capture_static_camera(raw_env):
    raw_env.gym.step_graphics(raw_env.sim)
    raw_env.gym.render_all_camera_sensors(raw_env.sim)
    return raw_env.gym.get_camera_image(
        raw_env.sim,
        raw_env.envs[0],
        raw_env.rendering_camera,
        gymapi.IMAGE_COLOR,
    )


def run_child(args):
    task_id, vx = parse_single_eval(args.eval)
    checkpoint = Path(args.checkpoint) if args.checkpoint else latest_checkpoint(args.run_dir)
    run_dir = checkpoint.parent.parent
    run_args = load_run_args(run_dir)
    oracle_condition_obs = not bool(run_args.get("no_oracle_condition_obs", False))
    selector_only = bool(run_args.get("selector_only", False))
    selector_latent_cmd_only = bool(run_args.get("selector_latent_cmd_only", False))
    decision_interval = int(run_args.get("decision_interval", 1))

    specs = read_task_specs(
        args.task_map,
        style_reward_scale=0.0,
        reward_profile=run_args.get("reward_profile", "canonical_efficiency_v4_physical"),
    )
    specs = select_eval_specs(specs, args.eval)
    logdir = find_logdir(args.label, args.run_index)
    policy = load_low_level_policy(logdir)
    env = OracleConditionHighLevelEnv(
        specs,
        logdir,
        policy,
        num_envs=1,
        render=False,
        oracle_condition_obs=False,
        terrain_size=args.terrain_length,
        terrain_length=args.terrain_length,
        terrain_width=args.terrain_width,
        edge_reset_margin=args.edge_reset_margin,
        teleport_thresh=args.teleport_thresh,
        mesh_type=args.mesh_type,
        selector_hold_steps=int(run_args.get("selector_hold_steps", 3)),
        recording_width_px=args.width,
        recording_height_px=args.height,
    )
    model, iteration = load_model(checkpoint, env, run_args)

    base_obs = env.reset()
    set_deterministic_vx(env)
    obs = augment_for_checkpoint(base_obs, env.assignment.task_ids, len(specs), oracle_condition_obs)
    obs = append_command_vx_obs(obs, env.command_vx(), selector_latent_cmd_only)

    raw_env = low_level_env(env)
    if not hasattr(raw_env, "rendering_camera"):
        raise RuntimeError("The low-level environment did not create its recording camera")
    set_static_camera(
        raw_env,
        args.camera_forward_offset,
        args.camera_lateral_offset,
        args.camera_height,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{video_stem(task_id, vx)}.mp4"
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        args.fps,
        (args.width, args.height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer: {output_path}")

    warmup_steps = int(round(args.warmup_seconds / env.env.high_level_dt))
    frame_count = int(round(args.duration_seconds * args.fps))
    steps_per_frame = max(1, int(round(1.0 / (args.fps * env.env.high_level_dt))))
    action = None
    selected_gaits = []
    reward_sum = 0.0
    vx_error_sum = 0.0
    recorded = 0
    total_steps = warmup_steps + frame_count * steps_per_frame

    try:
        with torch.inference_mode():
            for step in range(total_steps):
                if action is None or step % decision_interval == 0:
                    action = (
                        model.act_student_selector_only(obs)
                        if selector_only
                        else model.act_student(obs)
                    )
                next_obs, reward, _, _ = env.step(action)
                set_deterministic_vx(env)
                obs = augment_for_checkpoint(
                    next_obs,
                    env.assignment.task_ids,
                    len(specs),
                    oracle_condition_obs,
                )
                obs = append_command_vx_obs(obs, env.command_vx(), selector_latent_cmd_only)

                if step < warmup_steps or (step - warmup_steps) % steps_per_frame != 0:
                    continue
                frame = capture_static_camera(raw_env)
                encode_frame(writer, frame)
                gait_id = int(torch.argmax(action[0, : env.num_gaits]).item())
                selected_gaits.append(gait_id)
                reward_sum += float(reward[0].item())
                vx_error_sum += float(torch.abs(env.measured_vx()[0] - env.command_vx()[0]).item())
                recorded += 1
    finally:
        writer.release()

    counts = np.bincount(selected_gaits, minlength=env.num_gaits)
    dominant_id = int(np.argmax(counts))
    summary = {
        "task_id": task_id,
        "vx": vx,
        "checkpoint": str(checkpoint),
        "checkpoint_iteration": iteration,
        "duration_seconds": args.duration_seconds,
        "fps": args.fps,
        "frames": recorded,
        "resolution": [args.width, args.height],
        "terrain_length": args.terrain_length,
        "terrain_width": args.terrain_width,
        "dominant_gait": GAIT_SHORT_NAMES[env.env.gait_names[dominant_id]],
        "gait_frame_counts": {
            GAIT_SHORT_NAMES[name]: int(counts[index])
            for index, name in enumerate(env.env.gait_names)
        },
        "mean_reward": reward_sum / max(recorded, 1),
        "mean_vx_abs_error": vx_error_sum / max(recorded, 1),
        "video": str(output_path),
    }
    with (output_dir / f"{video_stem(task_id, vx)}.json").open("w") as file:
        json.dump(summary, file, indent=2)
    print(json.dumps(summary, indent=2))


def run_parent(args):
    items = [item.strip() for item in args.eval.split(",") if item.strip()]
    if not items:
        raise ValueError("--eval must contain at least one task_id:vx item")
    checkpoint = Path(args.checkpoint) if args.checkpoint else latest_checkpoint(args.run_dir)
    for item in items:
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--checkpoint",
            str(checkpoint),
            "--label",
            args.label,
            "--run-index",
            str(args.run_index),
            "--task-map",
            args.task_map,
            "--eval",
            item,
            "--duration-seconds",
            str(args.duration_seconds),
            "--warmup-seconds",
            str(args.warmup_seconds),
            "--fps",
            str(args.fps),
            "--width",
            str(args.width),
            "--height",
            str(args.height),
            "--terrain-length",
            str(args.terrain_length),
            "--terrain-width",
            str(args.terrain_width),
            "--camera-forward-offset",
            str(args.camera_forward_offset),
            "--camera-lateral-offset",
            str(args.camera_lateral_offset),
            "--camera-height",
            str(args.camera_height),
            "--edge-reset-margin",
            str(args.edge_reset_margin),
            "--teleport-thresh",
            str(args.teleport_thresh),
            "--mesh-type",
            args.mesh_type,
            "--output-dir",
            args.output_dir,
            "--no-spawn",
        ]
        print(f"[record] {item}", flush=True)
        subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--label", default="gait-conditioned-agility/pretrain-go2/train")
    parser.add_argument("--run-index", type=int, default=0)
    parser.add_argument("--task-map", default=str(MAINLINE_TASK_MAP))
    parser.add_argument("--eval", required=True)
    parser.add_argument("--duration-seconds", type=float, default=5.0)
    parser.add_argument("--warmup-seconds", type=float, default=1.0)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--terrain-length", type=float, default=20.0)
    parser.add_argument("--terrain-width", type=float, default=8.0)
    parser.add_argument("--camera-forward-offset", type=float, default=4.0)
    parser.add_argument("--camera-lateral-offset", type=float, default=7.0)
    parser.add_argument("--camera-height", type=float, default=3.5)
    parser.add_argument("--edge-reset-margin", type=float, default=1.0)
    parser.add_argument("--teleport-thresh", type=float, default=1.5)
    parser.add_argument("--mesh-type", default=TRAIN_MESH_TYPE, choices=["heightfield", "trimesh"])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--no-spawn", action="store_true")
    args = parser.parse_args()

    if args.checkpoint is None and args.run_dir is None:
        raise ValueError("Use either --checkpoint or --run-dir")
    if args.no_spawn:
        run_child(args)
    else:
        run_parent(args)


if __name__ == "__main__":
    main()
