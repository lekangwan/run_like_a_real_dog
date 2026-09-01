#!/usr/bin/env bash
set -euo pipefail

# GPU recording job. Run manually from the repository root.

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${GO2_WTW_PYTHON:-/home/lekangwan/miniconda3/envs/go2_wtw/bin/python3}"
GPU_INDEX="${GO2_WTW_GPU:-0}"
RUN_DIR="runs/high_level_oracle_gait/20260720_v4_flat_ramp_per_env_baseline_seed22550_stage2_iter050_direct"
CHECKPOINT="${RUN_DIR}/checkpoints/high_level_000049.pt"
OUT="ppt_assets/continuous_route"

cd "${PROJECT_ROOT}"
mkdir -p "${OUT}"
export CUDA_VISIBLE_DEVICES="${GPU_INDEX}"
export PYTHONPATH="${PROJECT_ROOT}/scripts:${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

"${PYTHON_BIN}" -B ppt_assets/record_continuous_four_terrain.py \
  --checkpoint "${CHECKPOINT}" \
  --eval flat_trot_efficiency:1.0 \
  --duration-seconds 17 \
  --warmup-seconds 0 \
  --fps 15 \
  --width 1920 \
  --height 1080 \
  --terrain-length 36 \
  --terrain-width 10 \
  --camera-forward-offset 2.2 \
  --camera-lateral-offset 4.8 \
  --camera-height 2.5 \
  --edge-reset-margin 0.5 \
  --mesh-type trimesh \
  --output-dir "${OUT}" \
  --no-spawn

echo "Finished: ${OUT}/p2_continuous_four_terrain.mp4"
