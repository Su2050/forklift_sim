#!/usr/bin/env bash
set -u

ROOT="/home/uniubi/projects/forklift_sim"
ISAACLAB="$ROOT/IsaacLab"
OUT_DIR="$ROOT/outputs/exp83_steer_usage_diagnostics/misalignment_grid"
LOG_DIR="$ROOT/logs"
TASK="Isaac-Forklift-PalletInsertLift-Direct-v0"
GRID_X_ROOT="-3.40"
Y_VALUES="-0.10,-0.05,0.0,0.05,0.10"
YAW_VALUES="-4,-2,0,2,4"
SEED="20260327"

mkdir -p "$OUT_DIR" "$LOG_DIR"

run_grid() {
  local checkpoint="$1"
  local label="$2"
  local mode="$3"
  local summary="$OUT_DIR/${label}_${mode}_summary.json"
  local force_flag=()
  local ts
  ts="$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)"
  local log="$LOG_DIR/${ts}_${label}_${mode}.log"

  if [[ -f "$summary" ]]; then
    echo "[SKIP] $label $mode -> $summary"
    return 0
  fi

  echo "[RUN] $label $mode"
  if [[ "$mode" == "zero_steer" ]]; then
    force_flag=(--force_zero_steer)
  fi
  (
    cd "$ISAACLAB" || exit 1
    exec env -u CONDA_PREFIX -u CONDA_DEFAULT_ENV -u CONDA_SHLVL -u CONDA_PROMPT_MODIFIER -u CONDA_PYTHON_EXE -u _CE_CONDA -u _CE_M \
      TERM=xterm PYTHONUNBUFFERED=1 \
      ./isaaclab.sh -p ../scripts/eval_exp83_misalignment_grid.py \
      --task "$TASK" \
      --headless \
      --enable_cameras \
      --checkpoint "$checkpoint" \
      --label "$label" \
      --num_envs 1 \
      --seed "$SEED" \
      --x_root "$GRID_X_ROOT" \
      --y_values="$Y_VALUES" \
      --yaw_deg_values="$YAW_VALUES" \
      --episodes_per_point 1 \
      "${force_flag[@]}" \
      --output_dir "$OUT_DIR"
  ) 2>&1 | tee "$log"
}

run_grid \
  "$ISAACLAB/logs/rsl_rl/forklift_pallet_insert_lift/2026-03-27_01-16-53_exp83_bonusw1p0_repro_r1_seed42_iter100_256cam/model_99.pt" \
  "exp83_grid_r1_seed42_iter100" \
  "zero_steer"

run_grid \
  "$ISAACLAB/logs/rsl_rl/forklift_pallet_insert_lift/2026-03-27_03-04-32_exp83_bonusw1p0_repro_r1_seed44_iter100_256cam/model_99.pt" \
  "exp83_grid_r1_seed44_iter100" \
  "normal"

run_grid \
  "$ISAACLAB/logs/rsl_rl/forklift_pallet_insert_lift/2026-03-27_03-04-32_exp83_bonusw1p0_repro_r1_seed44_iter100_256cam/model_99.pt" \
  "exp83_grid_r1_seed44_iter100" \
  "zero_steer"
