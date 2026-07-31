import argparse
import json
from pathlib import Path

import isaacgym

assert isaacgym
from isaacgym import gymapi
import cv2
import numpy as np
import torch

from .config import LOW_LEVEL_LABEL, TASK_MAP, validate_decision_interval
from .environment import HighLevelEnvironment
from .evaluate import create_model, latest_checkpoint, load_run_config
from .low_level import find_run
from .tasks import parse_eval_specs


def parse_args():
    parser = argparse.ArgumentParser(description="Record one minimal-policy scene.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--eval", required=True, help="Exactly one task:speed pair.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--decision-interval", type=int, default=None)
    return parser.parse_args()


def raw_environment(env):
    return env.env.env.env


def set_camera(raw_env):
    x, y, z = (float(value) for value in raw_env.root_states[0, :3])
    raw_env.gym.set_camera_location(
        raw_env.rendering_camera,
        raw_env.envs[0],
        gymapi.Vec3(x + 4.0, y - 7.0, z + 3.5),
        gymapi.Vec3(x + 4.0, y, z - 0.15),
    )


def capture_frame(raw_env):
    raw_env.gym.step_graphics(raw_env.sim)
    raw_env.gym.render_all_camera_sensors(raw_env.sim)
    image = raw_env.gym.get_camera_image(
        raw_env.sim,
        raw_env.envs[0],
        raw_env.rendering_camera,
        gymapi.IMAGE_COLOR,
    )
    image = np.asarray(image)
    if image.ndim == 2:
        image = image.reshape(image.shape[0], image.shape[1] // 4, 4)
    return cv2.cvtColor(image[:, :, :3], cv2.COLOR_RGB2BGR)


def main():
    args = parse_args()
    if "," in args.eval:
        raise ValueError("Record one task:speed pair at a time.")

    run_dir = Path(args.run_dir)
    config, minimal_run = load_run_config(run_dir)
    decision_interval = args.decision_interval or int(config.get("decision_interval", 1))
    validate_decision_interval(decision_interval)

    specs = parse_eval_specs(args.eval, TASK_MAP)
    low_level_run = Path(
        config["low_level_run"]
        if "low_level_run" in config
        else find_run(LOW_LEVEL_LABEL)
    )
    env = HighLevelEnvironment(
        specs,
        low_level_run,
        num_envs=1,
        terrain_length=12.0,
        terrain_width=8.0,
        recording_width=args.width,
        recording_height=args.height,
    )
    observation = env.reset()
    model, selector_only = create_model(env, config, minimal_run)
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else latest_checkpoint(run_dir)
    checkpoint = torch.load(checkpoint_path, map_location=env.device)
    model.load_state_dict(checkpoint["model"], strict=False)
    model.eval()

    raw_env = raw_environment(env)
    if not hasattr(raw_env, "rendering_camera"):
        raise RuntimeError("The Isaac Gym recording camera was not created.")
    set_camera(raw_env)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{args.eval.replace(':', '_vx').replace('.', 'p')}.mp4"
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        args.fps,
        (args.width, args.height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer: {output_path}")

    high_level_dt = env.env.high_level_dt
    simulation_fps = 1.0 / high_level_dt
    total_steps = round(args.duration / high_level_dt)
    total_frames = round(args.duration * args.fps)
    action = None
    gait_counts = torch.zeros(env.num_gaits, device=env.device)

    try:
        with torch.inference_mode():
            for step in range(total_steps):
                if action is None or step % decision_interval == 0:
                    action = model.act_student(observation, selector_only)
                observation, _, _, _ = env.step(action)
                frame = capture_frame(raw_env)
                frame_start = round(step * args.fps / simulation_fps)
                frame_end = round((step + 1) * args.fps / simulation_fps)
                for _ in range(max(1, frame_end - frame_start)):
                    writer.write(frame)
                    gait_counts[torch.argmax(action[0, : env.num_gaits])] += 1
    finally:
        writer.release()

    summary = {
        "eval": args.eval,
        "video": str(output_path),
        "frames": total_frames,
        "decision_interval": decision_interval,
        "gait_frame_counts": gait_counts.cpu().tolist(),
    }
    with open(output_path.with_suffix(".json"), "w") as file:
        json.dump(summary, file, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
