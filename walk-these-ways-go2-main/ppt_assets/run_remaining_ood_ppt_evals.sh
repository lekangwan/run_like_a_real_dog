#!/usr/bin/env bash
set -euo pipefail

# Finish the remaining stepping-stones evaluations needed by the PPT.
# Completed combined result files are skipped, so this script is safe to rerun.

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${GO2_WTW_PYTHON:-/home/lekangwan/miniconda3/envs/go2_wtw/bin/python3}"
GPU_INDEX="${GO2_WTW_GPU:-0}"

cd "${PROJECT_ROOT}"
export CUDA_VISIBLE_DEVICES="${GPU_INDEX}"
export PYTHONPATH="${PROJECT_ROOT}/scripts:${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

RUN_DIR="runs/high_level_oracle_gait/20260720_v4_flat_ramp_per_env_baseline_seed22550_stage2_iter050_direct"
CHECKPOINT="${RUN_DIR}/checkpoints/high_level_000049.pt"
OUTPUT_ROOT="${RUN_DIR}/independent_eval/ood_rough_stones"
EVAL_ITEMS="stepping_stones_easy_bound_highspeed:1.7,stepping_stones_easy_bound_highspeed:2.0"

run_eval() {
  local seed="$1"
  local mode="$2"
  local output_dir="${OUTPUT_ROOT}/20260806_seed${seed}_stones_${mode}"

  if [[ -f "${output_dir}/independent_eval_summary.csv" ]]; then
    echo "[skip] completed: ${output_dir}"
    return
  fi

  mkdir -p "${output_dir}"
  command=(
    "${PYTHON_BIN}" -B scripts/evaluate_high_level_policy_by_task.py
    --seed "${seed}"
    --run-dir "${RUN_DIR}"
    --checkpoint "${CHECKPOINT}"
    --eval "${EVAL_ITEMS}"
    --num-envs 32
    --steps 1000
    --warmup-steps 50
    --force-zero-residuals
    --output-dir "${output_dir}"
  )
  if [[ "${mode}" == "forced_trot" ]]; then
    command+=(--force-gait trotting)
  fi

  echo "[run] seed=${seed} mode=${mode}"
  "${command[@]}" 2>&1 | tee "${output_dir}/runner.log"
}

# Seed 23550 is already complete. Seed 23650 adaptive was run immediately
# before this script was created; the completion check above makes reruns safe.
run_eval 23650 adaptive
run_eval 23650 forced_trot
run_eval 23750 adaptive
run_eval 23750 forced_trot

"${PYTHON_BIN}" -B scripts/analyze_adaptive_vs_forced_trot.py \
  --root "${OUTPUT_ROOT}" \
  --seeds 23550,23650,23750 \
  --adaptive-template '20260806_seed{seed}_stones_adaptive' \
  --baseline-template '20260806_seed{seed}_stones_forced_trot' \
  --output-csv "${OUTPUT_ROOT}/stones_three_seed_summary.csv" \
  --output-markdown "${OUTPUT_ROOT}/stones_three_seed_summary.md"

echo "Completed stepping-stones paired evaluation."
echo "Summary: ${OUTPUT_ROOT}/stones_three_seed_summary.md"
