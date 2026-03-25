#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ISAACLAB_DIR="${ISAACLAB:-$ROOT/IsaacLab}"
RUN_DIR_BASE="$ISAACLAB_DIR/logs/rsl_rl/forklift_pallet_insert_lift"
OUTPUT_DIR="$ROOT/outputs/exp83_eval_bonus_weight_sweep"
SEEDS=(42 43 44)
WEIGHTS=(0.5 1.0 1.5)

mkdir -p "$OUTPUT_DIR"
export TERM=xterm

if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook 2>/dev/null)" || true
  while [[ "${CONDA_SHLVL:-0}" =~ ^[0-9]+$ ]] && [[ "${CONDA_SHLVL:-0}" -gt 0 ]]; do
    conda deactivate 2>/dev/null || break
  done
fi
unset CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_SHLVL CONDA_PROMPT_MODIFIER CONDA_PYTHON_EXE _CE_CONDA _CE_M 2>/dev/null || true

bash "$ROOT/forklift_pallet_insert_lift_project/scripts/install_into_isaaclab.sh" "$ISAACLAB_DIR"

weight_tag() {
  local weight="$1"
  echo "${weight//./p}"
}

find_checkpoint() {
  local weight="$1"
  local seed="$2"
  local tag run_name run_dir
  tag="$(weight_tag "$weight")"
  run_name="exp83_bonusw${tag}_seed${seed}_iter50_256cam"
  run_dir="$(find "$RUN_DIR_BASE" -maxdepth 1 -type d -name "*_${run_name}" | sort | tail -n 1)"
  if [[ -z "$run_dir" ]]; then
    echo "[ERROR] run dir not found for weight=${weight} seed=${seed}" >&2
    return 1
  fi
  echo "$run_dir/model_49.pt"
}

run_eval() {
  local weight="$1"
  local seed="$2"
  local tag label checkpoint
  tag="$(weight_tag "$weight")"
  label="exp83_eval_bonusw${tag}_seed${seed}"
  checkpoint="$(find_checkpoint "$weight" "$seed")"

  echo
  echo "============================================================"
  echo "[EVAL] $label"
  echo "[CKPT] $checkpoint"
  echo "============================================================"
  (
    cd "$ISAACLAB_DIR"
    exec env -u CONDA_PREFIX -u CONDA_DEFAULT_ENV -u CONDA_SHLVL \
      -u CONDA_PROMPT_MODIFIER -u CONDA_PYTHON_EXE -u _CE_CONDA -u _CE_M \
      TERM="$TERM" PYTHONUNBUFFERED=1 \
      ./isaaclab.sh -p ../scripts/eval_exp83_checkpoint.py \
      --task Isaac-Forklift-PalletInsertLift-Direct-v0 \
      --headless \
      --enable_cameras \
      --checkpoint "$checkpoint" \
      --label "$label" \
      --num_envs 32 \
      --rollouts 2 \
      --seed 20260325 \
      --output_dir "$OUTPUT_DIR"
  )
}

for weight in "${WEIGHTS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    run_eval "$weight" "$seed"
  done
done

echo
echo "[DONE] Bonus-weight unified eval suite finished."
