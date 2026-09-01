#!/usr/bin/env python3
"""Record one robot crossing a continuous four-section demonstration course."""

from __future__ import annotations

# Isaac Gym must be imported before torch. The recorder establishes that order.
import record_high_level_policy_videos as recorder

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from go2_gym.utils.terrain import Terrain
import train_high_level_oracle_ppo as trainer
from train_high_level_oracle_ppo import GAIT_NAMES


SEGMENT_LENGTHS = (3.0, 4.0, 4.0, 5.0)
SEGMENTS = (
    ("平地", "小跑", "trotting"),
    ("连续上坡", "双脚跳", "pronking"),
    ("粗糙坡面", "小跑", "trotting"),
    ("踏石", "跳跃跑", "bounding"),
)


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = (
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc") if bold else
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
        Path("/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf"),
    )
    path = next((item for item in candidates if item.exists()), None)
    if path is None:
        raise FileNotFoundError("A Chinese font is required for video labels")
    return ImageFont.truetype(str(path), size=size)


def segment_index(distance: float) -> int:
    boundary = 0.0
    for index, length in enumerate(SEGMENT_LENGTHS):
        boundary += length
        if distance < boundary:
            return index
    return len(SEGMENTS) - 1


def build_route(terrain, start_m: float) -> None:
    """Write a connected flat/ramp/rough/stones route into one height field."""
    heights = terrain.height_field_raw
    horizontal = float(terrain.horizontal_scale)
    vertical = float(terrain.vertical_scale)
    start = int(round(start_m / horizontal))
    flat_end = start + int(round(SEGMENT_LENGTHS[0] / horizontal))
    ramp_end = flat_end + int(round(SEGMENT_LENGTHS[1] / horizontal))
    rough_end = ramp_end + int(round(SEGMENT_LENGTHS[2] / horizontal))
    stones_end = rough_end + int(round(SEGMENT_LENGTHS[3] / horizontal))
    if stones_end >= heights.shape[0] - 2:
        raise ValueError("The route does not fit; increase --terrain-length")

    heights[:] = 0
    ramp_height_m = 0.10 * SEGMENT_LENGTHS[1]
    ramp_values = np.linspace(0.0, ramp_height_m, ramp_end - flat_end, endpoint=False)
    heights[flat_end:ramp_end, :] = np.round(ramp_values[:, None] / vertical).astype(np.int16)
    base_units = int(round(ramp_height_m / vertical))

    rng = np.random.default_rng(20260807)
    rough_rows = rough_end - ramp_end
    coarse_rows = max(2, int(np.ceil(rough_rows / 5)))
    coarse_cols = max(2, int(np.ceil(heights.shape[1] / 5)))
    coarse = rng.uniform(-0.035, 0.035, size=(coarse_rows, coarse_cols))
    rough = np.repeat(np.repeat(coarse, 5, axis=0), 5, axis=1)
    rough = rough[:rough_rows, :heights.shape[1]]
    heights[ramp_end:rough_end, :] = base_units + np.round(rough / vertical).astype(np.int16)

    # Mild, regular stepping stones: 0.78 m platforms separated by 0.10 m gaps.
    stone_rows = stones_end - rough_end
    x_m = np.arange(stone_rows)[:, None] * horizontal
    y_m = np.arange(heights.shape[1])[None, :] * horizontal
    period = 0.88
    stone_size = 0.78
    row_id = np.floor(x_m / period).astype(np.int32)
    stagger = (row_id % 2) * (period * 0.5)
    on_x = np.mod(x_m, period) < stone_size
    on_y = np.mod(y_m + stagger, period) < stone_size
    on_stone = on_x & on_y
    stone_area = np.full((stone_rows, heights.shape[1]), base_units - int(round(0.05 / vertical)), dtype=np.int16)
    stone_area[on_stone] = base_units
    heights[rough_end:stones_end, :] = stone_area

    # Short level strips make section boundaries traversable and visually clean.
    transition = max(1, int(round(0.20 / horizontal)))
    heights[ramp_end:ramp_end + transition, :] = base_units
    heights[rough_end - transition:rough_end, :] = base_units
    heights[stones_end:stones_end + transition, :] = base_units


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--terrain-length", type=float, default=36.0)
    known, recorder_args = parser.parse_known_args()
    if known.terrain_length < 34.0:
        raise ValueError("Use --terrain-length >= 34 so the entire route fits after the spawn point")

    original_make = Terrain.make_condition_terrain

    def make_route(self, cfg, condition, difficulty):
        terrain = original_make(self, cfg, "flat", difficulty)
        build_route(terrain, start_m=known.terrain_length * 0.5)
        return terrain

    Terrain.make_condition_terrain = make_route

    # The low-level checkpoint was trained with randomized spawn yaw. A long,
    # narrow presentation route needs deterministic alignment after every reset.
    original_apply_condition = trainer.apply_condition_cfg

    def apply_route_condition(cfg, condition):
        original_apply_condition(cfg, condition)
        cfg.terrain.yaw_init_range = 0.0
        cfg.terrain.x_init_range = 0.0
        cfg.terrain.y_init_range = 0.0

    trainer.apply_condition_cfg = apply_route_condition

    # This is a position-scheduled visualization. Check the current section at
    # every high-level step instead of inheriting the trained 100-step hold.
    original_load_run_args = recorder.load_run_args

    def load_route_run_args(run_dir):
        run_args = dict(original_load_run_args(run_dir))
        run_args["decision_interval"] = 1
        run_args["selector_hold_steps"] = 1
        return run_args

    recorder.load_run_args = load_route_run_args

    original_load = recorder.load_model
    state = {"raw_env": None, "origin_x": 0.0, "origin_y": 0.0, "last_segment": 0}

    def load_route_model(checkpoint, env, run_args):
        model, iteration = original_load(checkpoint, env, run_args)
        raw_env = recorder.low_level_env(env)
        state["raw_env"] = raw_env
        state["origin_x"] = float(raw_env.env_origins[0, 0].item())
        state["origin_y"] = float(raw_env.env_origins[0, 1].item())

        def route_action(obs):
            distance = float(raw_env.root_states[0, 0].item()) - state["origin_x"]
            index = segment_index(max(distance, 0.0))
            state["last_segment"] = index
            gait_id = GAIT_NAMES.index(SEGMENTS[index][2])
            action = torch.zeros(
                obs.shape[0], env.num_high_level_actions,
                device=obs.device, dtype=obs.dtype,
            )
            action[:, gait_id] = 1.0
            return action

        model.act_student_selector_only = route_action
        model.act_student = route_action
        return model, iteration

    recorder.load_model = load_route_model
    recorder.video_stem = lambda task_id, vx: "p2_continuous_four_terrain"

    original_set_vx = recorder.set_deterministic_vx

    def set_straight_route_command(env):
        original_set_vx(env)
        raw_env = state["raw_env"]
        if raw_env is None:
            return
        quat = raw_env.root_states[:, 3:7]
        qx, qy, qz, qw = quat.unbind(dim=1)
        yaw = torch.atan2(
            2.0 * (qw * qz + qx * qy),
            1.0 - 2.0 * (qy.square() + qz.square()),
        )
        lateral_error = raw_env.root_states[:, 1] - state["origin_y"]
        yaw_rate = torch.clamp(-1.2 * yaw - 0.35 * lateral_error, -0.8, 0.8)
        env.env.set_velocity_command(env.vx_cmd, 0.0, yaw_rate)

    recorder.set_deterministic_vx = set_straight_route_command

    original_capture = recorder.capture_static_camera

    def capture_following(raw_env):
        bx, by, bz = (float(value) for value in raw_env.root_states[0, :3])
        raw_env.gym.set_camera_location(
            raw_env.rendering_camera,
            raw_env.envs[0],
            recorder.gymapi.Vec3(bx + 2.2, by - 4.8, bz + 2.5),
            recorder.gymapi.Vec3(bx + 1.0, by, bz + 0.05),
        )
        return original_capture(raw_env)

    recorder.capture_static_camera = capture_following
    original_encode = recorder.encode_frame

    def encode_labeled(writer, rgba):
        image = np.asarray(rgba)
        if image.ndim == 2:
            image = image.reshape(image.shape[0], image.shape[1] // 4, 4)
        pil_image = Image.fromarray(image[:, :, :3]).convert("RGB")
        draw = ImageDraw.Draw(pil_image, "RGBA")
        terrain_name, gait_cn, gait_name = SEGMENTS[state["last_segment"]]
        draw.rounded_rectangle((38, 32, 530, 142), radius=14, fill=(10, 35, 65, 220))
        draw.text((64, 48), f"{terrain_name}  |  {gait_cn}", font=font(34, bold=True), fill="white")
        draw.text((66, 98), f"当前步态：{gait_name}", font=font(21), fill=(205, 228, 246, 255))
        draw.rectangle((0, pil_image.height - 48, pil_image.width, pil_image.height), fill=(8, 27, 48, 215))
        draw.text(
            (38, pil_image.height - 39),
            "连续组合地形上的显式步态切换演示｜不代表高层策略已自主识别四种地形",
            font=font(19),
            fill="white",
        )
        original_encode(writer, np.asarray(pil_image.convert("RGBA")))

    recorder.encode_frame = encode_labeled
    sys.argv = [sys.argv[0], "--terrain-length", str(known.terrain_length), *recorder_args]
    recorder.main()

    output_dir = Path(recorder_args[recorder_args.index("--output-dir") + 1])
    metadata_path = output_dir / "p2_continuous_four_terrain.json"
    metadata = json.loads(metadata_path.read_text())
    metadata.update({
        "control_mode": "position_scheduled_gait_visualization",
        "terrain_sequence": [item[0] for item in SEGMENTS],
        "gait_sequence": [item[2] for item in SEGMENTS],
        "heading_control": "Spawn yaw fixed to zero with proportional yaw/lateral route correction.",
        "claim_limit": "Continuous execution demo only; not evidence of autonomous terrain recognition or navigation.",
    })
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
