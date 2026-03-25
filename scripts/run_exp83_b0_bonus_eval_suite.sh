#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ISAACLAB_DIR="${ISAACLAB:-$ROOT/IsaacLab}"
OUTPUT_DIR="$ROOT/outputs/exp83_eval"

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

run_eval() {
  local label="$1"
  local checkpoint="$2"
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

run_eval exp83_eval_b0_seed42 \
  /home/uniubi/projects/forklift_sim/IsaacLab/logs/rsl_rl/forklift_pallet_insert_lift/2026-03-25_14-55-31_exp83_b0_seed42_iter50_256cam/model_49.pt

run_eval exp83_eval_b0_seed43 \
  /home/uniubi/projects/forklift_sim/IsaacLab/logs/rsl_rl/forklift_pallet_insert_lift/2026-03-25_16-14-40_exp83_b0_seed43_iter50_256cam/model_49.pt

run_eval exp83_eval_b0_seed44 \
  /home/uniubi/projects/forklift_sim/IsaacLab/logs/rsl_rl/forklift_pallet_insert_lift/2026-03-25_16-41-30_exp83_b0_seed44_iter50_256cam/model_49.pt

run_eval exp83_eval_bonus_seed42 \
  /home/uniubi/projects/forklift_sim/IsaacLab/logs/rsl_rl/forklift_pallet_insert_lift/2026-03-25_12-15-30_exp83_clean_bonus_seed42_iter50_256cam/model_49.pt

run_eval exp83_eval_bonus_seed43 \
  /home/uniubi/projects/forklift_sim/IsaacLab/logs/rsl_rl/forklift_pallet_insert_lift/2026-03-25_12-42-30_exp83_clean_bonus_seed43_iter50_256cam/model_49.pt

run_eval exp83_eval_bonus_seed44 \
  /home/uniubi/projects/forklift_sim/IsaacLab/logs/rsl_rl/forklift_pallet_insert_lift/2026-03-25_13-09-13_exp83_clean_bonus_seed44_iter50_256cam/model_49.pt

echo
echo "[DONE] Unified eval suite finished."
