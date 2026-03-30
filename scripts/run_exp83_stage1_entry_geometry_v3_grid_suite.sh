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
LOCK_FILE="$OUT_DIR/suite.lock"

exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  echo "[EXIT] suite already active: $LOCK_FILE"
  exit 0
fi

touch "$FAILED_LIST" "$COMPLETED_LIST"

append_unique() {
  local item="$1"
  local file="$2"
  touch "$file"
  if ! rg -Fxq "$item" "$file" 2>/dev/null; then
    printf '%s\n' "$item" >> "$file"
  fi
}

remove_item() {
  local item="$1"
  local file="$2"
  local tmp
  touch "$file"
  tmp="$(mktemp)"
  rg -Fvx "$item" "$file" > "$tmp" || true
  mv "$tmp" "$file"
}

record_done() {
  local item="$1"
  append_unique "$item" "$COMPLETED_LIST"
  remove_item "$item" "$FAILED_LIST"
}

record_failed() {
  local item="$1"
  append_unique "$item" "$FAILED_LIST"
}

label_recorded() {
  local item="$1"
  rg -Fxq "$item" "$COMPLETED_LIST" 2>/dev/null || rg -Fxq "$item" "$FAILED_LIST" 2>/dev/null
}

find_checkpoint() {
  local run_name="$1"
  ls -1dt "$ISAACLAB"/logs/rsl_rl/forklift_pallet_insert_lift/*_"$run_name"/model_49.pt 2>/dev/null | head -n 1
}

run_grid() {
  local checkpoint="$1"
  local label="$2"
  local mode="$3"
  local item="${label}_${mode}"
  local summary="$OUT_DIR/${label}_${mode}_summary.json"
  local partial_summary="$OUT_DIR/${label}_${mode}_partial_summary.json"
  local force_flag=()
  local ts log
  ts="$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)"
  log="$LOG_DIR/${ts}_${label}_${mode}.log"

  if [[ -z "$checkpoint" ]]; then
    echo "[MISS] $label -> checkpoint not found"
    record_failed "$item"
    return 0
  fi

  if [[ -f "$summary" ]]; then
    echo "[SKIP] $label $mode -> $summary"
    record_done "$item"
    return 0
  fi

  if label_recorded "$item"; then
    echo "[SKIP] $label $mode -> already recorded"
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
    record_done "$item"
    return 0
  fi

  if [[ $status -eq 124 || $status -eq 137 ]]; then
    echo "[TIMEOUT] $label $mode timeout=${GRID_TIMEOUT_S}s log=$log"
    if [[ -f "$partial_summary" ]]; then
      echo "[PARTIAL] $label $mode partial_summary=$partial_summary"
    fi
  else
    echo "[FAIL] $label $mode status=$status log=$log"
    if [[ -f "$partial_summary" ]]; then
      echo "[PARTIAL] $label $mode partial_summary=$partial_summary"
    fi
  fi
  record_failed "$item"
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
