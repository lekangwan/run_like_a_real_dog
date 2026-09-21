#!/usr/bin/env python3
"""Generate presentation assets from validated July 2026 experiment outputs."""

from __future__ import annotations

import csv
import json
import shutil
import statistics
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from PIL import Image, ImageDraw, ImageFont


REPO = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
RUN = REPO / "runs/high_level_oracle_gait/20260720_v4_flat_ramp_per_env_baseline_seed22550_stage2_iter050_direct"
EVAL = RUN / "independent_eval"
PAIRED = EVAL / "paired_three_seed_summary.csv"
OOD_EVAL = EVAL / "ood_rough_stones"
ROUGH_PAIRED = OOD_EVAL / "rough_three_seed_summary.csv"
STONES_PAIRED = OOD_EVAL / "stones_three_seed_summary.csv"
VIDEO_DIR = REPO / "reports/20260721_closure/videos_all_terrains_1080p"
TERRAIN_PREVIEWS = (
    ("平地", "训练场景", REPO / "logs/gait_condition_previews/v6_ramp_up/flat.png"),
    ("连续上坡", "训练场景", REPO / "logs/gait_condition_previews/v6_ramp_up/ramp_up.png"),
    ("粗糙坡面", "未见地形测试", REPO / "logs/gait_condition_previews/v6_ramp_up/rough_slope.png"),
    ("踏石", "未见地形测试", REPO / "logs/gait_condition_previews/v4_moderate/stepping_stones_easy.png"),
)
SEEDS = (22650, 22750, 22850)
SPEEDS = (0.5, 1.0, 1.5, 2.0)
SPEED_TOKEN = {0.5: "0p50", 1.0: "1p00", 1.5: "1p50", 2.0: "2p00"}

NAVY = "#12355B"
BLUE = "#1F6FB2"
CYAN = "#37A7C4"
LIGHT_BLUE = "#EAF3FA"
INK = "#172436"
MUTED = "#5B6B7D"
GREEN = "#2A8C73"
RED = "#C94C4C"
AMBER = "#E6A23C"
GRAY = "#D8E1EA"


def configure_fonts() -> Path:
    candidates = (
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
        Path("/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf"),
    )
    font_path = next((p for p in candidates if p.exists()), None)
    if font_path is None:
        raise FileNotFoundError("No Chinese font found; install Noto Sans CJK or Droid Sans Fallback")
    font_manager.fontManager.addfont(str(font_path))
    family = font_manager.FontProperties(fname=str(font_path)).get_name()
    plt.rcParams.update({"font.family": family, "axes.unicode_minus": False})
    return font_path


FONT_PATH = configure_fonts()


def require_sources() -> None:
    sources = [
        PAIRED,
        VIDEO_DIR / "flat_trot_efficiency_vx1p00.mp4",
        VIDEO_DIR / "ramp_up_trot_robustness_vx1p50.mp4",
        VIDEO_DIR / "rough_slope_trot_robustness_vx1p00.mp4",
        ROUGH_PAIRED,
        STONES_PAIRED,
        *(path for _, _, path in TERRAIN_PREVIEWS),
    ]
    missing = [str(p) for p in sources if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing source files:\n" + "\n".join(missing))


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def save_figure(fig: plt.Figure, filename: str) -> None:
    fig.savefig(OUT / filename, dpi=100, facecolor="white", bbox_inches=None)
    plt.close(fig)


def draw_box(ax, xy, width, height, title, detail="", *, face=LIGHT_BLUE, edge=BLUE,
             dashed=False, title_size=19, detail_size=12):
    x, y = xy
    patch = FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        linewidth=2.2,
        edgecolor=edge,
        facecolor=face,
        linestyle="--" if dashed else "-",
    )
    ax.add_patch(patch)
    ax.text(x + width / 2, y + height * 0.61, title, ha="center", va="center",
            fontsize=title_size, color=INK, weight="bold")
    if detail:
        ax.text(x + width / 2, y + height * 0.30, detail, ha="center", va="center",
                fontsize=detail_size, color=MUTED, linespacing=1.25)
    return patch


def arrow(ax, start, end, *, color=BLUE, dashed=False, width=2.2):
    ax.add_patch(FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=18, linewidth=width,
        color=color, linestyle="--" if dashed else "-", shrinkA=4, shrinkB=4,
    ))


def generate_framework() -> None:
    fig, ax = plt.subplots(figsize=(19.2, 10.8), dpi=100)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    ax.text(0.055, 0.92, "基于本体感知历史的 Go2 高层步态自适应控制", fontsize=34,
            color=NAVY, weight="bold", va="center")
    ax.text(0.057, 0.865, "部署时不使用地形编号、视觉或雷达；底层 WTW 控制器保持冻结",
            fontsize=17, color=MUTED, va="center")
    ax.add_patch(Rectangle((0.055, 0.832), 0.89, 0.006, facecolor=CYAN, edgecolor="none"))

    draw_box(ax, (0.055, 0.57), 0.15, 0.17, "目标速度", "$v_x^{cmd}$", face="#F5F8FB")
    draw_box(ax, (0.055, 0.34), 0.15, 0.17, "本体感知历史", "关节 / 机身 / 接触\n历史序列", face="#F5F8FB")
    draw_box(ax, (0.265, 0.39), 0.18, 0.29, "学生历史编码器", "从历史中估计\n环境与动力学隐变量  z", face="#E7F3FA")
    draw_box(ax, (0.505, 0.50), 0.17, 0.18, "高层步态选择器", "双脚跳 / 小跑 /\n跳跃 / 踱步（含相位模板）", face="#DCEEFF")
    draw_box(ax, (0.505, 0.27), 0.17, 0.15, "连续参数调节", "频率 / 持续时间 / 摆动高度\n站宽 / 机身俯仰修正", face="#FFF7E8", edge=AMBER, dashed=True)
    ax.text(0.59, 0.235, "实验分支：汇报主模型中固定为默认值", ha="center", va="center",
            fontsize=12.5, color="#9A6500", weight="bold")
    draw_box(ax, (0.73, 0.39), 0.16, 0.27, "冻结的 WTW\n底层策略", "步态命令 + 本体状态\n→ 12 维关节动作", face="#E8F5F1", edge=GREEN)
    ax.add_patch(FancyBboxPatch(
        (0.92, 0.43), 0.065, 0.19,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        linewidth=2.2, edgecolor=NAVY, facecolor=NAVY,
    ))
    ax.text(0.9525, 0.535, "Go2", ha="center", va="center", fontsize=18,
            color="white", weight="bold")
    ax.text(0.9525, 0.475, "执行", ha="center", va="center", fontsize=11, color="white")

    arrow(ax, (0.205, 0.655), (0.265, 0.59))
    arrow(ax, (0.205, 0.425), (0.265, 0.48))
    arrow(ax, (0.445, 0.56), (0.505, 0.59))
    arrow(ax, (0.445, 0.48), (0.505, 0.345), color=AMBER, dashed=True)
    arrow(ax, (0.675, 0.59), (0.73, 0.56))
    arrow(ax, (0.675, 0.345), (0.73, 0.47), color=AMBER, dashed=True)
    arrow(ax, (0.89, 0.525), (0.92, 0.525), color=GREEN)

    draw_box(ax, (0.265, 0.105), 0.18, 0.13, "训练期教师编码器", "14 维仿真特权物理量",
             face="#F2EEF9", edge="#7256A4", dashed=True, title_size=16, detail_size=11)
    arrow(ax, (0.355, 0.235), (0.355, 0.39), color="#7256A4", dashed=True)
    ax.text(0.37, 0.29, "隐变量蒸馏\n部署时移除", fontsize=11.5, color="#7256A4", va="center")

    ax.text(0.055, 0.075,
            "当前已验证：条件相关的离散步态选择。  尚未完成：连续参数稳定增益、全地形泛化与实机部署。",
            fontsize=15, color=INK, weight="bold")
    save_figure(fig, "p2_system_framework.png")


def adaptive_gait_rows() -> list[dict[str, float | str]]:
    specs = (
        ("平地", "flat_trot_efficiency", SPEEDS, SEEDS, EVAL, "20260720_seed{seed}_adaptive"),
        ("上坡", "ramp_up_trot_robustness", SPEEDS, SEEDS, EVAL, "20260720_seed{seed}_adaptive"),
        ("粗糙", "rough_slope_trot_robustness", SPEEDS, (23250, 23350, 23450), OOD_EVAL,
         "20260806_seed{seed}_rough_adaptive"),
        ("踏石", "stepping_stones_easy_bound_highspeed", (1.7, 2.0), (23550, 23650, 23750), OOD_EVAL,
         "20260806_seed{seed}_stones_adaptive"),
    )
    result = []
    for condition, task, speeds, seeds, root, template in specs:
        for speed in speeds:
            seed_rows = []
            for seed in seeds:
                token = SPEED_TOKEN.get(speed, str(speed).replace(".", "p") + "0")
                path = root / template.format(seed=seed) / f"{task}_vx{token}" / "independent_eval_summary.csv"
                seed_rows.append(load_csv(path)[0])
            row: dict[str, float | str] = {"condition": condition, "speed": speed}
            for gait in ("pronk", "trot", "bound", "pace"):
                row[gait] = statistics.mean(float(item[f"{gait}_ratio"]) for item in seed_rows)
            result.append(row)
    return result


def generate_gait_distribution() -> None:
    rows = adaptive_gait_rows()
    fig, ax = plt.subplots(figsize=(19.2, 10.8), dpi=100)
    x = np.arange(len(rows))
    labels = [f"{r['condition']}\n{r['speed']:.1f}" for r in rows]
    colors = {"trot": "#1F6FB2", "pronk": "#E6A23C", "bound": "#2A8C73", "pace": "#8B6BB1"}
    names = {"trot": "小跑", "pronk": "双脚跳", "bound": "跳跃", "pace": "踱步"}
    bottom = np.zeros(len(rows))
    for gait in ("trot", "pronk", "bound", "pace"):
        values = np.array([float(r[gait]) * 100 for r in rows])
        ax.bar(x, values, bottom=bottom, width=0.68, color=colors[gait], label=names[gait],
               edgecolor="white", linewidth=0.8)
        for idx, value in enumerate(values):
            if value >= 8:
                ax.text(idx, bottom[idx] + value / 2, f"{value:.1f}%", ha="center", va="center",
                        color="white", fontsize=14, weight="bold")
        bottom += values

    fig.text(0.08, 0.92, "已训练与未见地形的步态选择分布", fontsize=31, color=NAVY, weight="bold")
    fig.text(0.08, 0.865, "平地、上坡为训练场景；粗糙地面、踏石为未见地形评测",
             fontsize=17, color=MUTED)
    ax.set_ylabel("选择比例（%）", fontsize=16, color=INK)
    ax.set_xticks(x, labels, fontsize=11.5)
    ax.set_ylim(0, 100)
    ax.grid(axis="y", color=GRAY, linewidth=0.8, alpha=0.7)
    ax.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.axvline(7.5, color=NAVY, linewidth=1.5, linestyle="--", alpha=0.7)
    ax.text(3.5, 103.5, "训练场景", ha="center", va="bottom", fontsize=14, color=BLUE, weight="bold")
    ax.text(10.5, 103.5, "未见地形", ha="center", va="bottom", fontsize=14, color=AMBER, weight="bold")
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.12), frameon=False, fontsize=15)
    ax.text(0, -0.245,
            "数据来源：汇报主模型独立评测；每个点 32 个并行环境 × 1000 步 × 3 个随机种子。",
            transform=ax.transAxes, fontsize=12.5, color=MUTED)
    fig.subplots_adjust(left=0.08, right=0.97, top=0.80, bottom=0.25)
    save_figure(fig, "p2_gait_distribution.png")


def paired_rows() -> dict[tuple[str, float], dict[str, str]]:
    return {(row["condition"], float(row["cmd_vx"])): row for row in load_csv(PAIRED)}


def generate_eval_comparison() -> None:
    rows = paired_rows()
    speeds = (1.0, 1.5)
    metrics = (
        ("速度误差", "adaptive_vx_err_mean_mean", "forced_trot_vx_err_mean_mean", "m/s", 3),
        ("接触滑移", "adaptive_contact_slip_penalty_mean", "forced_trot_contact_slip_penalty_mean", "归一化值", 3),
        ("落地冲击", "adaptive_impact_velocity_rms_mean", "forced_trot_impact_velocity_rms_mean", "m/s", 3),
        ("机械功率", "adaptive_mechanical_power_abs_mean", "forced_trot_mechanical_power_abs_mean", "W", 1),
    )
    fig, axes = plt.subplots(1, 4, figsize=(19.2, 10.8), dpi=100)
    fig.suptitle("中速上坡：自适应策略与固定小跑对比", fontsize=31, color=NAVY,
                 weight="bold", x=0.055, ha="left", y=0.93)
    fig.text(0.055, 0.872, "柱高为三个随机种子的实测均值；四项指标均为越低越好",
             fontsize=17, color=MUTED)
    x = np.arange(len(speeds))
    width = 0.34
    for ax, (title, adaptive_key, baseline_key, unit, decimals) in zip(axes, metrics):
        adaptive = [float(rows[("ramp_up", speed)][adaptive_key]) for speed in speeds]
        baseline = [float(rows[("ramp_up", speed)][baseline_key]) for speed in speeds]
        adaptive_bars = ax.bar(x - width / 2, adaptive, width, color=GREEN, label="自适应策略")
        baseline_bars = ax.bar(x + width / 2, baseline, width, color=BLUE, label="固定小跑")
        ax.set_title(title, fontsize=21, color=INK, weight="bold", pad=18)
        ax.text(0.5, 1.01, f"越低越好 · {unit}", transform=ax.transAxes, ha="center",
                fontsize=12, color=MUTED)
        ymax = max(adaptive + baseline) * 1.24
        ax.set_ylim(0, ymax)
        ax.set_xticks(x, [f"{speed:.1f} m/s" for speed in speeds])
        for pair_index, (adaptive_bar, baseline_bar, adaptive_value, baseline_value) in enumerate(
            zip(adaptive_bars, baseline_bars, adaptive, baseline)
        ):
            for bar, value in ((adaptive_bar, adaptive_value), (baseline_bar, baseline_value)):
                is_better = value == min(adaptive_value, baseline_value)
                label = f"{value:.{decimals}f}"
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    value + ymax * 0.025,
                    label,
                    ha="center",
                    va="bottom",
                    fontsize=13,
                    weight="bold" if is_better else "normal",
                    color=INK,
                )
        ax.grid(axis="y", color=GRAY, linewidth=0.8, alpha=0.65)
        ax.set_axisbelow(True)
        for spine in ("top", "right", "left"):
            ax.spines[spine].set_visible(False)
        ax.tick_params(axis="x", labelsize=13)
        ax.tick_params(axis="y", labelsize=11)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.145), ncol=2,
               frameon=False, fontsize=15)
    fig.text(0.055, 0.105, "自适应策略降低了速度误差、接触滑移和落地冲击，但机械功率高于固定小跑。",
             fontsize=14, color=INK, weight="bold")
    fig.text(0.055, 0.072,
             "数据来源：paired_three_seed_summary.csv；同检查点、同地形、同随机种子，自适应选择器与强制小跑严格成对评测。",
             fontsize=12.5, color=MUTED)
    fig.subplots_adjust(left=0.055, right=0.975, top=0.78, bottom=0.23, wspace=0.34)
    save_figure(fig, "p2_eval_comparison.png")


def generate_ood_generalization() -> None:
    rows = load_csv(ROUGH_PAIRED) + load_csv(STONES_PAIRED)
    gait_lookup = {
        (str(row["condition"]), float(row["speed"])): row
        for row in adaptive_gait_rows()
    }
    task_meta = {
        "rough_slope_trot_robustness": ("粗糙", "粗糙地面"),
        "stepping_stones_easy_bound_highspeed": ("踏石", "踏石"),
    }
    gait_names = {"trot": "小跑", "pronk": "双脚跳", "bound": "跳跃", "pace": "踱步"}
    columns = (
        ("速度误差", "adaptive_vx_err_mean_mean", "forced_trot_vx_err_mean_mean"),
        ("机械功率", "adaptive_mechanical_power_abs_mean", "forced_trot_mechanical_power_abs_mean"),
        ("接触滑移", "adaptive_contact_slip_penalty_mean", "forced_trot_contact_slip_penalty_mean"),
        ("落地冲击", "adaptive_impact_velocity_rms_mean", "forced_trot_impact_velocity_rms_mean"),
        ("足端擦碰", "adaptive_scuffing_ratio_mean", "forced_trot_scuffing_ratio_mean"),
    )
    ranked_rows = []
    for row in rows:
        short, display = task_meta[row["task_id"]]
        speed = float(row["cmd_vx"])
        gait_row = gait_lookup[(short, speed)]
        gait = max(("trot", "pronk", "bound", "pace"), key=lambda key: float(gait_row[key]))
        improvements = []
        for _, adaptive_key, baseline_key in columns:
            adaptive_value = float(row[adaptive_key])
            baseline_value = float(row[baseline_key])
            improvement = 100.0 * (baseline_value - adaptive_value) / baseline_value
            improvements.append(improvement)
        ranked_rows.append({
            "scene": f"{display} {speed:.1f}",
            "gait": f"{gait_names[gait]}\n{float(gait_row[gait]) * 100:.1f}%",
            "improvements": improvements,
            "summary": statistics.mean(improvements),
        })

    ranked_rows.sort(key=lambda item: float(item["summary"]), reverse=True)
    column_best = [max(float(item["improvements"][idx]) for item in ranked_rows)
                   for idx in range(len(columns))]
    table_rows = []
    cell_colors = []
    for rank, item in enumerate(ranked_rows, start=1):
        improvements = [float(value) for value in item["improvements"]]
        table_rows.append([
            str(rank), str(item["scene"]), str(item["gait"]),
            *(f"{value:+.1f}%" for value in improvements),
        ])
        cell_colors.append([
            "#EAF3FA" if rank == 1 else "#F5F7FA",
            "#EAF3FA" if rank == 1 else "#F5F7FA",
            "#EAF3FA",
            *("#DFF2EA" if value >= 0 else "#F8DFDF" for value in improvements),
        ])

    headers = ["排名", "未见场景", "主要步态"] + [item[0] for item in columns]
    fig, ax = plt.subplots(figsize=(19.2, 10.8), dpi=100)
    ax.axis("off")
    fig.text(0.055, 0.92, "未见地形综合性能评测", fontsize=31, color=NAVY, weight="bold")
    fig.text(0.055, 0.865, "表中为相对固定小跑的改善比例：正值表示改善，负值表示退化",
             fontsize=17, color=MUTED)
    table = ax.table(
        cellText=table_rows, colLabels=headers, cellColours=cell_colors,
        colColours=[NAVY] * len(headers), cellLoc="center", colLoc="center",
        colWidths=[0.065, 0.15, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12],
        bbox=[0.04, 0.25, 0.92, 0.50],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(13)
    table.scale(1, 1.65)
    for (row_index, col_index), cell in table.get_celld().items():
        cell.set_edgecolor("white")
        cell.set_linewidth(1.5)
        if row_index == 0:
            cell.get_text().set_color("white")
            cell.get_text().set_weight("bold")
        elif col_index < 3:
            cell.get_text().set_weight("bold")
            cell.get_text().set_color(INK)
        elif col_index >= 3:
            value = float(ranked_rows[row_index - 1]["improvements"][col_index - 3])
            if np.isclose(value, column_best[col_index - 3]):
                cell.get_text().set_weight("bold")

    fig.text(0.055, 0.175,
             "踏石两档速度的综合取舍相对较好；粗糙地面仅冲击等局部指标改善，多数指标仍退化。",
             fontsize=15, color=INK, weight="bold")
    fig.text(0.055, 0.125,
             "排序按五项相对改善比例等权平均，仅用于汇总展示；不代表新的训练奖励或统计显著性。",
             fontsize=15, color=INK, weight="bold")
    fig.text(0.055, 0.075,
             "比例 =（固定小跑 − 自适应策略）/ 固定小跑；每点 32 个环境 × 1000 步 × 3 个新随机种子。",
             fontsize=12.5, color=MUTED)
    save_figure(fig, "p2_ood_generalization.png")


def read_video_frame(path: Path, frame_index: int) -> np.ndarray:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise RuntimeError(f"Cannot read frame {frame_index} from {path}")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def crop_simulation_view(frame: np.ndarray) -> np.ndarray:
    """Enlarge the robot while retaining enough terrain to identify the scene."""
    height, width = frame.shape[:2]
    x0, x1 = int(width * 0.18), int(width * 0.84)
    y0, y1 = int(height * 0.12), int(height * 0.79)
    cropped = frame[y0:y1, x0:x1]
    return cv2.resize(cropped, (1920, 1080), interpolation=cv2.INTER_LANCZOS4)


def pil_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    path = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc") if bold else FONT_PATH
    return ImageFont.truetype(str(path), size=size)


def overlay_banner(frame: np.ndarray, title: str, subtitle: str, *, footer: str = "") -> np.ndarray:
    image = Image.fromarray(frame).convert("RGB")
    draw = ImageDraw.Draw(image, "RGBA")
    draw.rounded_rectangle((54, 48, 770, 180), radius=14, fill=(10, 35, 65, 220))
    draw.text((82, 67), title, font=pil_font(40, bold=True), fill=(255, 255, 255, 255))
    draw.text((84, 125), subtitle, font=pil_font(24), fill=(205, 228, 246, 255))
    if footer:
        draw.rectangle((0, 1008, 1920, 1080), fill=(8, 27, 48, 218))
        draw.text((62, 1027), footer, font=pil_font(22), fill=(238, 245, 250, 255))
    return np.asarray(image)


def generate_scene() -> None:
    path = VIDEO_DIR / "ramp_up_trot_robustness_vx1p50.mp4"
    frame = crop_simulation_view(read_video_frame(path, 24))
    frame = overlay_banner(frame, "Isaac Gym 上坡场景", "Unitree Go2 · 目标速度 1.5 m/s",
                           footer="仿真评估画面｜该单环境录像用于展示场景，不作为步态比例的统计证据")
    Image.fromarray(frame).save(OUT / "p2_isaac_gym_scene.png", quality=96)


def generate_terrain_gallery() -> None:
    """Arrange the repository's real height-field previews into one PPT slide."""
    canvas = Image.new("RGB", (1920, 1080), "white")
    draw = ImageDraw.Draw(canvas, "RGBA")
    draw.text((68, 30), "四足机器人仿真地形设置", font=pil_font(46, bold=True), fill=NAVY)
    draw.text(
        (70, 94),
        "训练场景与未见地形测试场景｜俯视高度图 + 三维高度场",
        font=pil_font(25),
        fill=MUTED,
    )

    positions = ((58, 160), (982, 160), (58, 610), (982, 610))
    for (title, role, path), (x, y) in zip(TERRAIN_PREVIEWS, positions):
        panel_w, panel_h = 880, 405
        draw.rounded_rectangle(
            (x, y, x + panel_w, y + panel_h),
            radius=10,
            fill=(247, 250, 253, 255),
            outline=(205, 218, 231, 255),
            width=2,
        )

        preview = Image.open(path).convert("RGB")
        preview = preview.crop((0, 55, preview.width, preview.height))
        preview.thumbnail((844, 326), Image.Resampling.LANCZOS)
        px = x + (panel_w - preview.width) // 2
        py = y + 64 + (326 - preview.height) // 2
        canvas.paste(preview, (px, py))

        tag_fill = (31, 111, 178, 255) if role == "训练场景" else (42, 140, 115, 255)
        draw.text((x + 20, y + 14), title, font=pil_font(28, bold=True), fill=INK)
        tag_box = draw.textbbox((0, 0), role, font=pil_font(20))
        tag_w = tag_box[2] - tag_box[0] + 28
        draw.rounded_rectangle(
            (x + panel_w - tag_w - 18, y + 14, x + panel_w - 18, y + 51),
            radius=8,
            fill=tag_fill,
        )
        draw.text((x + panel_w - tag_w - 4, y + 18), role, font=pil_font(20), fill="white")

    canvas.save(OUT / "p2_terrain_gallery.png", quality=96)


def write_data_snapshot() -> None:
    gait_rows = adaptive_gait_rows()
    with (OUT / "gait_distribution_source.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("condition", "speed", "pronk", "trot", "bound", "pace"))
        writer.writeheader()
        writer.writerows(gait_rows)
    shutil.copy2(PAIRED, OUT / "eval_comparison_source.csv")
    shutil.copy2(ROUGH_PAIRED, OUT / "rough_ood_source.csv")
    shutil.copy2(STONES_PAIRED, OUT / "stones_ood_source.csv")


def write_manifest() -> None:
    manifest = {
        "main_checkpoint": str(RUN.relative_to(REPO) / "checkpoints/high_level_000049.pt"),
        "evaluation_seeds": list(SEEDS),
        "parallel_envs": 32,
        "steps_per_point": 1000,
        "paired_summary": str(PAIRED.relative_to(REPO)),
        "rough_ood_summary": str(ROUGH_PAIRED.relative_to(REPO)),
        "stones_ood_summary": str(STONES_PAIRED.relative_to(REPO)),
        "video_sources": [str(p.relative_to(REPO)) for p in sorted(VIDEO_DIR.glob("*.mp4"))],
        "claims": {
            "validated": "Flat mostly selects trot; medium-speed ramp often selects pronk and improves tracking/contact metrics with a power cost.",
            "not_validated": "Broad unseen-terrain superiority, stable continuous-parameter gains, sim-to-real deployment.",
        },
    }
    (OUT / "source_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    require_sources()
    generate_framework()
    generate_gait_distribution()
    generate_eval_comparison()
    generate_ood_generalization()
    generate_scene()
    generate_terrain_gallery()
    write_data_snapshot()
    write_manifest()
    print(f"Generated PPT assets in {OUT}")


if __name__ == "__main__":
    main()
