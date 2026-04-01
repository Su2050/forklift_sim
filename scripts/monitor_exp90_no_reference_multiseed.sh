#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_SCRIPT="$ROOT/scripts/run_exp90_no_reference_baseline.sh"
ISAAC_LOG_ROOT="${ISAAC_LOG_ROOT:-$ROOT/IsaacLab/logs/rsl_rl/forklift_pallet_insert_lift}"
LOG_DIR="${LOG_DIR:-$ROOT/logs}"
STATE_DIR="${STATE_DIR:-$ROOT/outputs/exp90_no_reference_monitor}"
SEEDS_STR="${EXP90_SEEDS:-42 43 44}"
NUM_ENVS="${NUM_ENVS:-64}"
MAX_ITERATIONS="${MAX_ITERATIONS:-400}"
RUN_NAME_PREFIX="${RUN_NAME_PREFIX:-exp9_0_no_reference_master_init}"
BLOCK_IF_ANY_FORKLIFT_TRAINING="${BLOCK_IF_ANY_FORKLIFT_TRAINING:-1}"
LOOP="${LOOP:-0}"
SLEEP_S="${SLEEP_S:-300}"
DRY_RUN="${DRY_RUN:-0}"

mkdir -p "$LOG_DIR" "$STATE_DIR"

LOCK_FILE="$STATE_DIR/monitor.lock"
STATUS_FILE="$STATE_DIR/status.txt"
FINAL_CKPT="model_$((MAX_ITERATIONS - 1)).pt"

exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  echo "[EXIT] monitor already active: $LOCK_FILE"
  exit 0
fi

if [[ ! -f "$RUN_SCRIPT" ]]; then
  echo "[FATAL] missing run script: $RUN_SCRIPT" >&2
  exit 2
fi

if [[ ! -d "$ISAAC_LOG_ROOT" ]]; then
  echo "[FATAL] missing IsaacLab log root: $ISAAC_LOG_ROOT" >&2
  exit 2
fi

read -r -a SEEDS <<< "$SEEDS_STR"
if [[ "${#SEEDS[@]}" -eq 0 ]]; then
  echo "[FATAL] no seeds configured via EXP90_SEEDS" >&2
  exit 2
fi

ts() {
  TZ=Asia/Shanghai date "+%F %T %Z"
}

run_name_for_seed() {
  local seed="$1"
  echo "${RUN_NAME_PREFIX}_seed${seed}_iter${MAX_ITERATIONS}"
}

latest_run_dir_for_seed() {
  local seed="$1"
  local run_name
  run_name="$(run_name_for_seed "$seed")"
  find "$ISAAC_LOG_ROOT" -maxdepth 1 -mindepth 1 -type d -name "*_${run_name}" | sort | tail -n 1
}

seed_is_running() {
  local seed="$1"
  local run_name
  run_name="$(run_name_for_seed "$seed")"
  pgrep -af "scripts/reinforcement_learning/rsl_rl/train.py.*agent.run_name=${run_name}" >/dev/null 2>&1
}

any_forklift_training_running() {
  pgrep -af "scripts/reinforcement_learning/rsl_rl/train.py.*Isaac-Forklift-PalletInsertLift-Direct-v0" >/dev/null 2>&1
}

seed_is_completed() {
  local seed="$1"
  local run_dir
  run_dir="$(latest_run_dir_for_seed "$seed")"
  [[ -n "$run_dir" && -f "$run_dir/$FINAL_CKPT" ]]
}

write_status() {
  local message="$1"
  printf "[%s] %s\n" "$(ts)" "$message" | tee "$STATUS_FILE"
}

launch_seed() {
  local seed="$1"
  local run_name launch_log ts_compact
  run_name="$(run_name_for_seed "$seed")"
  ts_compact="$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)"
  launch_log="$LOG_DIR/${ts_compact}_monitor_${run_name}.log"

  if [[ "$DRY_RUN" == "1" ]]; then
    write_status "[DRY-RUN] would launch seed=${seed} run_name=${run_name} log=${launch_log}"
    return 0
  fi

  write_status "[START] launching seed=${seed} run_name=${run_name} log=${launch_log}"
  (
    cd "$ROOT"
    SEED="$seed" NUM_ENVS="$NUM_ENVS" MAX_ITERATIONS="$MAX_ITERATIONS" bash "$RUN_SCRIPT"
  ) >"$launch_log" 2>&1

  write_status "[STARTED] seed=${seed} via ${RUN_SCRIPT}; wrapper_log=${launch_log}"
}

check_once() {
  local seed completed_count run_dir
  completed_count=0

  for seed in "${SEEDS[@]}"; do
    if seed_is_running "$seed"; then
      run_dir="$(latest_run_dir_for_seed "$seed")"
      write_status "[RUNNING] seed=${seed} run_dir=${run_dir:-N/A}"
      return 0
    fi

    if seed_is_completed "$seed"; then
      completed_count=$((completed_count + 1))
      continue
    fi

    if [[ "$BLOCK_IF_ANY_FORKLIFT_TRAINING" == "1" ]] && any_forklift_training_running; then
      write_status "[WAIT] seed=${seed} pending, but another forklift training process is active"
      return 0
    fi

    launch_seed "$seed"
    return 0
  done

  write_status "[DONE] all seeds completed (${completed_count}/${#SEEDS[@]})"
  return 10
}

if [[ "$LOOP" == "1" ]]; then
  while true; do
    if check_once; then
      sleep "$SLEEP_S"
      continue
    fi
    exit 0
  done
fi

set +e
check_once
rc=$?
set -e

if [[ "$rc" -eq 10 ]]; then
  exit 0
fi
exit "$rc"
