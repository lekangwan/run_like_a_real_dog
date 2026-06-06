import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_CONDITIONS = [
    "flat",
    "ramp_up",
    "very_low_friction",
    "rough_mid",
    "rough_slope",
    "stairs_up_low",
    "stairs_down_low",
    "discrete_obstacles_low",
    "push_hard",
]


CONDITION_NOTES = {
    "flat": {
        "geometry": "zero-height flat trimesh",
        "physics": "friction=1.0, no pushes",
        "purpose": "nominal baseline; trot should be a strong reference gait",
    },
    "low_friction": {
        "geometry": "flat",
        "physics": "friction randomized around 0.25",
        "purpose": "moderate slippery ground; tests slip/lateral/yaw stability",
    },
    "very_low_friction": {
        "geometry": "flat",
        "physics": "friction randomized around 0.12",
        "purpose": "strong slippery ground; should expose friction-limited gait behavior",
    },
    "rough": {
        "geometry": "random height noise around +/-0.10 m",
        "physics": "normal friction",
        "purpose": "moderate uneven terrain; tests foot clearance and scuffing",
    },
    "rough_mid": {
        "geometry": "random height noise around +/-0.12 m",
        "physics": "normal friction",
        "purpose": "v4 moderate-hard uneven terrain; should still be traversable",
    },
    "rough_hard": {
        "geometry": "random height noise around +/-0.16 m",
        "physics": "normal friction",
        "purpose": "strong uneven terrain; should make clearance/impact matter more",
    },
    "rough_slope": {
        "geometry": "sloped terrain with random roughness",
        "physics": "normal friction",
        "purpose": "combined slope and roughness; tests posture plus clearance",
    },
    "ramp_up": {
        "geometry": "continuous uphill ramp, about 0.8 m rise over 4 m, no steps or center platform",
        "physics": "normal friction",
        "purpose": "to-real robustness condition; tests pitch control, traction, and uphill traversal",
    },
    "slope": {
        "geometry": "legacy pyramid slope with a center platform",
        "physics": "normal friction",
        "purpose": "legacy preview only; prefer ramp_up for a true continuous ramp",
    },
    "stairs": {
        "geometry": "stairs-down alias kept for old experiments",
        "physics": "normal friction",
        "purpose": "legacy stair condition; prefer stairs_up/stairs_down now",
    },
    "stairs_up": {
        "geometry": "ascending stairs",
        "physics": "normal friction",
        "purpose": "tests foot clearance, progress, and impact on climbing",
    },
    "stairs_up_low": {
        "geometry": "ascending low stairs, step height 0.08 m",
        "physics": "normal friction",
        "purpose": "v4 traversable stair climb; tests clearance without total failure",
    },
    "stairs_down": {
        "geometry": "descending stairs",
        "physics": "normal friction",
        "purpose": "tests impact and body stability during descent",
    },
    "stairs_down_low": {
        "geometry": "descending low stairs, step height 0.08 m",
        "physics": "normal friction",
        "purpose": "v4 traversable stair descent; tests impact without total failure",
    },
    "discrete_obstacles": {
        "geometry": "random block obstacles",
        "physics": "normal friction",
        "purpose": "tests clearance and obstacle robustness",
    },
    "discrete_obstacles_low": {
        "geometry": "random low block obstacles, height around 0.08 m",
        "physics": "normal friction",
        "purpose": "v4 lower obstacles; should avoid the no-progress failure seen in v3",
    },
    "stepping_stones": {
        "geometry": "stone-like support patches separated by lower gaps",
        "physics": "normal friction",
        "purpose": "tests precise support and swing clearance",
    },
    "stepping_stones_easy": {
        "geometry": "larger support patches with shallow gaps",
        "physics": "normal friction",
        "purpose": "v4 easier stepping stones; tests support precision while staying traversable",
    },
    "push": {
        "geometry": "flat",
        "physics": "random push every 0.5 s, max xy velocity kick 1.0",
        "purpose": "moderate external disturbance recovery",
    },
    "push_hard": {
        "geometry": "flat",
        "physics": "random push every 0.4 s, max xy velocity kick 1.5",
        "purpose": "strong external disturbance recovery",
    },
}


def parse_list(text):
    return [item.strip() for item in text.split(",") if item.strip()]


def make_grid(size, resolution):
    xs = np.linspace(-size / 2, size / 2, resolution)
    ys = np.linspace(-size / 2, size / 2, resolution)
    return np.meshgrid(xs, ys)


def smooth_noise(rng, shape, scale, coarse=18):
    low = rng.uniform(-scale, scale, size=(coarse, coarse))
    repeat_x = int(np.ceil(shape[0] / coarse))
    repeat_y = int(np.ceil(shape[1] / coarse))
    noise = np.kron(low, np.ones((repeat_x, repeat_y)))[: shape[0], : shape[1]]
    return noise


def terrain_height(condition, x, y, seed):
    rng = np.random.default_rng(seed)
    z = np.zeros_like(x)
    arrows = []
    overlay_text = []

    if condition in ("low_friction", "very_low_friction"):
        mu = 0.25 if condition == "low_friction" else 0.12
        overlay_text.append(f"flat geometry, friction ~= {mu:.2f}")
    elif condition == "rough":
        z = smooth_noise(rng, x.shape, 0.10)
    elif condition == "rough_mid":
        z = smooth_noise(rng, x.shape, 0.12)
    elif condition == "rough_hard":
        z = smooth_noise(rng, x.shape, 0.16)
    elif condition == "ramp_up":
        z = 0.20 * (x - x.min())
    elif condition == "slope":
        z = 0.28 * x
    elif condition == "rough_slope":
        z = 0.24 * x + smooth_noise(rng, x.shape, 0.05)
    elif condition in ("stairs", "stairs_down", "stairs_up", "stairs_up_low", "stairs_down_low"):
        step_width = 0.31
        step_height = 0.08 if condition in ("stairs_up_low", "stairs_down_low") else 0.16
        steps = np.floor((x - x.min()) / step_width)
        if condition in ("stairs", "stairs_down", "stairs_down_low"):
            z = -step_height * steps
        else:
            z = step_height * steps
        z -= z.mean()
    elif condition in ("discrete_obstacles", "discrete_obstacles_low"):
        z = np.zeros_like(x)
        for _ in range(16):
            cx, cy = rng.uniform(-1.7, 1.7, size=2)
            sx, sy = rng.uniform(0.18, 0.45, size=2)
            h = 0.08 if condition == "discrete_obstacles_low" else rng.uniform(0.08, 0.20)
            mask = (np.abs(x - cx) < sx) & (np.abs(y - cy) < sy)
            z[mask] = np.maximum(z[mask], h)
    elif condition in ("stepping_stones", "stepping_stones_easy"):
        z = -0.05 * np.ones_like(x) if condition == "stepping_stones_easy" else -0.08 * np.ones_like(x)
        spacing = 0.75
        radius = 0.34 if condition == "stepping_stones_easy" else 0.25
        for cx in np.arange(-1.5, 1.51, spacing):
            for cy in np.arange(-1.5, 1.51, spacing):
                mask = (x - cx) ** 2 + (y - cy) ** 2 < radius**2
                z[mask] = 0.03 if condition == "stepping_stones_easy" else 0.04
    elif condition == "push":
        overlay_text.append("flat geometry, random push: 0.5 s, max 1.0")
        arrows = [(-1.2, 0.0, 1.0, 0.0), (0.9, 0.8, -0.7, -0.5)]
    elif condition == "push_hard":
        overlay_text.append("flat geometry, random push: 0.4 s, max 1.5")
        arrows = [(-1.4, 0.0, 1.4, 0.0), (1.0, 0.9, -1.0, -0.8)]

    return z, arrows, overlay_text


def plot_condition(condition, output_dir, size, resolution, seed):
    x, y = make_grid(size, resolution)
    z, arrows, overlay_text = terrain_height(condition, x, y, seed)
    note = CONDITION_NOTES.get(condition, {})

    fig = plt.figure(figsize=(10, 4.5), constrained_layout=True)
    ax_map = fig.add_subplot(1, 2, 1)
    image = ax_map.imshow(
        z,
        origin="lower",
        extent=[-size / 2, size / 2, -size / 2, size / 2],
        cmap="terrain",
        vmin=-0.15,
        vmax=0.85,
    )
    ax_map.set_title(f"{condition}: top view")
    ax_map.set_xlabel("x [m]")
    ax_map.set_ylabel("y [m]")
    for x0, y0, dx, dy in arrows:
        ax_map.arrow(x0, y0, dx, dy, width=0.035, color="crimson", length_includes_head=True)
    for i, text in enumerate(overlay_text):
        ax_map.text(
            0.02,
            0.96 - 0.08 * i,
            text,
            transform=ax_map.transAxes,
            ha="left",
            va="top",
            fontsize=8,
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
        )
    fig.colorbar(image, ax=ax_map, shrink=0.8, label="height [m]")

    ax_3d = fig.add_subplot(1, 2, 2, projection="3d")
    stride = max(1, resolution // 55)
    ax_3d.plot_surface(
        x[::stride, ::stride],
        y[::stride, ::stride],
        z[::stride, ::stride],
        cmap="terrain",
        linewidth=0,
        antialiased=True,
        vmin=-0.15,
        vmax=0.85,
    )
    ax_3d.set_title("height surface")
    ax_3d.set_xlabel("x [m]")
    ax_3d.set_ylabel("y [m]")
    ax_3d.set_zlabel("z [m]")
    ax_3d.set_zlim(-0.2, 0.9)
    ax_3d.view_init(elev=28, azim=-135)

    fig.suptitle(
        f"{note.get('geometry', '')} | {note.get('physics', '')}\n{note.get('purpose', '')}",
        fontsize=10,
    )
    path = output_dir / f"{condition}.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def write_descriptions(conditions, output_dir):
    path = output_dir / "condition_descriptions.csv"
    with open(path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["condition", "geometry", "physics", "purpose"])
        writer.writeheader()
        for condition in conditions:
            note = CONDITION_NOTES.get(condition, {})
            writer.writerow(
                {
                    "condition": condition,
                    "geometry": note.get("geometry", ""),
                    "physics": note.get("physics", ""),
                    "purpose": note.get("purpose", ""),
                }
            )
    return path


def make_overview(image_paths, output_dir):
    images = [(path.stem, plt.imread(path)) for path in image_paths]
    cols = 3
    rows = int(np.ceil(len(images) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5.2, rows * 2.8), constrained_layout=True)
    axes = np.array(axes).reshape(-1)
    for ax, (name, image) in zip(axes, images):
        ax.imshow(image)
        ax.set_title(name)
        ax.axis("off")
    for ax in axes[len(images) :]:
        ax.axis("off")
    path = output_dir / "overview.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--conditions", default=",".join(DEFAULT_CONDITIONS))
    parser.add_argument("--output-dir", default="logs/gait_condition_previews/v6_ramp_up")
    parser.add_argument("--size", type=float, default=4.0)
    parser.add_argument("--resolution", type=int, default=140)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    conditions = parse_list(args.conditions)

    image_paths = [
        plot_condition(condition, output_dir, args.size, args.resolution, args.seed + i)
        for i, condition in enumerate(conditions)
    ]
    descriptions = write_descriptions(conditions, output_dir)
    overview = make_overview(image_paths, output_dir)
    print(f"Saved {len(image_paths)} condition previews to: {output_dir}")
    print(f"Overview: {overview}")
    print(f"Descriptions: {descriptions}")


if __name__ == "__main__":
    main()
