"""录制单个任务速度点的学生策略视频。

录像使用与独立评测相同的模型恢复路径，只把并行环境数缩减为 1，
并通过 Isaac Gym 相机传感器输出 MP4。
"""

import argparse
import json
from pathlib import Path

import isaacgym

# Isaac Gym 必须先于 PyTorch 导入。
from isaacgym import gymapi
import cv2
import numpy as np
import torch

from .config import validate_decision_interval
from .environment import HighLevelEnvironment
from .evaluate import create_model, latest_checkpoint, load_run_config
from .tasks import parse_eval_specs


def parse_args():
    """解析录像参数。

    输入：进程命令行。
    输出：包含运行目录、检查点、唯一任务速度点、输出目录、时长、帧率、分辨率和周期。
    内部逻辑：限制一次只录一个场景，默认输出 1080p、30 FPS、5 秒视频。
    作用：让录像路径和独立评测使用同一模型恢复配置。
    """
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
    """拆开高层、步态和历史包装器，取得原始 Isaac Gym 环境。

    输入：``HighLevelEnvironment``。
    输出：最底层 ``VelocityTrackingEasyEnv``。
    内部逻辑：按照当前固定包装层级连续访问三次 ``env``。
    作用：访问相机传感器、Gym API、仿真实例和机器人根状态。
    """
    return env.env.env.env


def set_camera(raw_env):
    """把相机放在机器人前侧上方。

    输入：带 ``root_states`` 和相机传感器的原始环境。
    输出：无显式返回；更新 Isaac Gym 相机位置和观察目标。
    内部逻辑：读取第一个机器人位置，增加固定前向、侧向和高度偏移。
    作用：在单场景录像中同时展示机器人和前方地形。
    """
    x, y, z = (float(value) for value in raw_env.root_states[0, :3])
    raw_env.gym.set_camera_location(
        raw_env.rendering_camera,
        raw_env.envs[0],
        gymapi.Vec3(x + 4.0, y - 7.0, z + 3.5),
        gymapi.Vec3(x + 4.0, y, z - 0.15),
    )


def capture_frame(raw_env):
    """从 Isaac Gym 相机捕获一帧 OpenCV 图像。

    输入：原始环境。
    输出：``[height,width,3]`` 的 uint8 BGR 数组。
    内部逻辑：推进图形管线、渲染相机传感器、读取 RGBA；兼容扁平返回格式并转换颜色顺序。
    作用：连接 Isaac Gym 图像接口与 OpenCV 视频编码器。
    """
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
    """恢复学生策略，执行单环境仿真并写入视频。

    输入：命令行参数、运行配置、模型检查点和底层 WTW 资源。
    输出：无 Python 返回值；生成 MP4 和记录帧数、周期、步态计数的同名 JSON。
    内部逻辑：
        创建单环境和相机，加载学生模型；按高层周期更新步态指令并推进仿真；每个 0.1 秒
        仿真帧按目标视频帧率重复写入，同时统计各步态出现帧数。
    作用：生成与定量评测模型一致的可视化证据，并保留机器可读元数据。
    """
    args = parse_args()
    if "," in args.eval:
        raise ValueError("Record one task:speed pair at a time.")

    run_dir = Path(args.run_dir)
    config = load_run_config(run_dir)
    decision_interval = args.decision_interval or int(config.get("decision_interval", 1))
    validate_decision_interval(decision_interval)

    specs = parse_eval_specs(args.eval)
    low_level_run = Path(config["low_level_run"])
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
    model, selector_only = create_model(env, config)
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else latest_checkpoint(run_dir)
    checkpoint = torch.load(checkpoint_path, map_location=env.device)
    model.load_state_dict(checkpoint["model"])
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

    # 仿真只在每个高层步渲染一次；写视频时按目标帧率重复该帧。
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
