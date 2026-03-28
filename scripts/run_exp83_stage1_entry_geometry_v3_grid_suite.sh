#!/usr/bin/env bash
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ISAACLAB="$ROOT/IsaacLab"
OUT_DIR="$ROOT/outputs/exp83_stage1_entry_geometry_v3_grid"
LOG_DIR="$ROOT/logs"
TASK="Isaac-Forklift-PalletInsertLift-Direct-v0"
GRID_X_ROOT="-3.50"
Y_VALUES="-0.15,-0.10,-0.05,0.0,0.05,0.10,0.15"
YAW_VALUES="-6,-4,-2,0,2,4,6"
SEED="20260328"
GRID_TIMEOUT_S="${GRID_TIMEOUT_S:-2400}"

mkdir -p "$OUT_DIR" "$LOG_DIR"
FAILED_LIST="$OUT_DIR/failed_labels.txt"
COMPLETED_LIST="$OUT_DIR/completed_labels.txt"

: > "$FAILED_LIST"
: > "$COMPLETED_LIST"

find_checkpoint() {
  local run_name="$1"
  ls -1dt "$ISAACLAB"/logs/rsl_rl/forklift_pallet_insert_lift/*_"$run_name"/model_49.pt 2>/dev/null | head -n 1
}

run_grid() {
  local checkpoint="$1"
  local label="$2"
  local mode="$3"
  local summary="$OUT_DIR/${label}_${mode}_summary.json"
  local force_flag=()
  local ts log
  ts="$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)"
  log="$LOG_DIR/${ts}_${label}_${mode}.log"

  if [[ -z "$checkpoint" ]]; then
    echo "[MISS] $label -> checkpoint not found"
    return 1
  fi

  if [[ -f "$summary" ]]; then
    echo "[SKIP] $label $mode -> $summary"
    echo "${label}_${mode}" >> "$COMPLETED_LIST"
    return 0
  fi

  if [[ "$mode" == "zero_steer" ]]; then
    force_flag=(--force_zero_steer)
  fi

  echo "[RUN] $label $mode"
  (
    cd "$ISAACLAB"
    exec env -u CONDA_PREFIX -u CONDA_DEFAULT_ENV -u CONDA_SHLVL -u CONDA_PROMPT_MODIFIER -u CONDA_PYTHON_EXE -u _CE_CONDA -u _CE_M \
      TERM=xterm PYTHONUNBUFFERED=1 \
      timeout --signal=TERM --kill-after=30 "${GRID_TIMEOUT_S}" \
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
  ) >"$log" 2>&1
  local status=$?
  if [[ $status -eq 0 && -f "$summary" ]]; then
    echo "[DONE] $label $mode"
    echo "${label}_${mode}" >> "$COMPLETED_LIST"
    return 0
  fi

  if [[ $status -eq 124 || $status -eq 137 ]]; then
    echo "[TIMEOUT] $label $mode timeout=${GRID_TIMEOUT_S}s log=$log"
  else
    echo "[FAIL] $label $mode status=$status log=$log"
  fi
  echo "${label}_${mode}" >> "$FAILED_LIST"
  return 0
}

for seed in 42 43 44; do
  run_name="exp83_stage1_entry_geometry_v3_seed${seed}_iter50_256cam"
  checkpoint="$(find_checkpoint "$run_name")"
  label="exp83_stage1_entry_geometry_v3_seed${seed}_iter50"
  run_grid "$checkpoint" "$label" "normal"
  run_grid "$checkpoint" "$label" "zero_steer"
done

echo "[DONE] stage1 entry geometry v3 grid suite completed"
