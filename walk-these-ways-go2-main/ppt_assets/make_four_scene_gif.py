#!/usr/bin/env python3
"""Combine four recorded terrain clips into one labeled PPT GIF."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent
RECORDINGS = ROOT / "recordings"
OUTPUT = ROOT / "p2_four_terrain_gaits.gif"
FONT = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc")
FONT_BOLD = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc")

CLIPS = (
    ("flat/flat_trot_efficiency_vx1p00.mp4", "平地", "小跑  trot", "1.0 m/s"),
    ("ramp/ramp_up_trot_robustness_vx1p00.mp4", "上坡", "双脚跳  pronk", "1.0 m/s"),
    ("rough/rough_slope_trot_robustness_vx0p50.mp4", "粗糙地面", "双脚跳  pronk", "0.5 m/s"),
    ("stones/stepping_stones_easy_bound_highspeed_vx2p00.mp4", "踏石", "小跑  trot", "2.0 m/s"),
)


def font(size: int, bold: bool = False):
    return ImageFont.truetype(str(FONT_BOLD if bold else FONT), size=size)


def read_frames(path: Path):
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open {path}")
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (1280, 720), interpolation=cv2.INTER_LANCZOS4)
        yield frame
    capture.release()


def label_frame(frame: np.ndarray, terrain: str, gait: str, speed: str) -> Image.Image:
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image, "RGBA")
    draw.rounded_rectangle((34, 30, 490, 150), radius=14, fill=(10, 35, 65, 225))
    draw.text((58, 45), f"{terrain} · {gait}", font=font(31, bold=True), fill="white")
    draw.text((60, 102), f"目标速度 {speed}", font=font(21), fill=(205, 228, 246))
    draw.rectangle((0, 672, 1280, 720), fill=(8, 27, 48, 220))
    draw.text(
        (34, 684),
        "固定步态可视化｜用于展示动作形态，不作为策略自主选择的证据",
        font=font(18), fill=(240, 246, 250),
    )
    return image


def main() -> None:
    frames = []
    for relative_path, terrain, gait, speed in CLIPS:
        path = RECORDINGS / relative_path
        if not path.exists():
            raise FileNotFoundError(path)
        frames.extend(label_frame(frame, terrain, gait, speed) for frame in read_frames(path))
    if not frames:
        raise RuntimeError("No frames were generated")
    frames[0].save(
        OUTPUT,
        save_all=True,
        append_images=frames[1:],
        duration=100,
        loop=0,
        optimize=True,
        disposal=2,
    )
    print(f"Wrote {OUTPUT} with {len(frames)} frames")


if __name__ == "__main__":
    main()
