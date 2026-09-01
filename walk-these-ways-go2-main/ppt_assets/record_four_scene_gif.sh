#!/usr/bin/env bash
set -euo pipefail

# GPU recording job. Run manually from the repository root.

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${GO2_WTW_PYTHON:-/home/lekangwan/miniconda3/envs/go2_wtw/bin/python3}"
GPU_INDEX="${GO2_WTW_GPU:-0}"
RUN_DIR="runs/high_level_oracle_gait/20260720_v4_flat_ramp_per_env_baseline_seed22550_stage2_iter050_direct"
CHECKPOINT="${RUN_DIR}/checkpoints/high_level_000049.pt"
OUT="ppt_assets/recordings"

cd "${PROJECT_ROOT}"
export CUDA_VISIBLE_DEVICES="${GPU_INDEX}"
export PYTHONPATH="${PROJECT_ROOT}/scripts:${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

record_clip() {
  local folder="$1"
  local item="$2"
  local gait="$3"
  mkdir -p "${OUT}/${folder}"
  "${PYTHON_BIN}" -B ppt_assets/record_forced_gait_video.py \
    --force-gait "${gait}" \
    --checkpoint "${CHECKPOINT}" \
    --eval "${item}" \
    --duration-seconds 3 \
    --warmup-seconds 0.5 \
    --fps 10 \
    --width 1280 \
    --height 720 \
    --terrain-length 12 \
    --terrain-width 6 \
    --camera-forward-offset 3 \
    --camera-lateral-offset 5 \
    --camera-height 2.7 \
    --edge-reset-margin 1 \
    --mesh-type trimesh \
    --output-dir "${OUT}/${folder}" \
    --no-spawn
}

record_clip flat "flat_trot_efficiency:1.0" trotting
record_clip ramp "ramp_up_trot_robustness:1.0" pronking
record_clip rough "rough_slope_trot_robustness:0.5" pronking
record_clip stones "stepping_stones_easy_bound_highspeed:2.0" trotting

"${PYTHON_BIN}" -B ppt_assets/make_four_scene_gif.py

echo "Finished: ppt_assets/p2_four_terrain_gaits.gif"
