#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SUITE="$ROOT/scripts/run_exp83_stage1_entry_geometry_v3_grid_suite.sh"
OUT_DIR="$ROOT/outputs/exp83_stage1_entry_geometry_v3_grid"
LOG_DIR="$ROOT/logs"
EXPECTED=6
SLEEP_S="${SLEEP_S:-120}"
GRID_TIMEOUT_S="${GRID_TIMEOUT_S:-2400}"

mkdir -p "$LOG_DIR" "$OUT_DIR"
touch "$OUT_DIR/completed_labels.txt" "$OUT_DIR/failed_labels.txt"

count_done() {
  local c f
  c="$(sort -u "$OUT_DIR/completed_labels.txt" 2>/dev/null | sed '/^$/d' | wc -l)"
  f="$(sort -u "$OUT_DIR/failed_labels.txt" 2>/dev/null | sed '/^$/d' | wc -l)"
  echo $((c + f))
}

suite_running() {
  pgrep -af "run_exp83_stage1_entry_geometry_v3_grid_suite.sh" >/dev/null 2>&1
}

kill_stale_children() {
  pkill -f "eval_exp83_misalignment_grid.py --task Isaac-Forklift-PalletInsertLift-Direct-v0" 2>/dev/null || true
  pkill -f "run_exp83_stage1_entry_geometry_v3_grid_suite.sh" 2>/dev/null || true
}

while true; do
  done_n="$(count_done)"
  if [[ "$done_n" -ge "$EXPECTED" ]]; then
    echo "[DONE] grid monitoring complete: ${done_n}/${EXPECTED}"
    exit 0
  fi

  if ! suite_running; then
    ts="$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)"
    log="$LOG_DIR/${ts}_monitor_exp83_stage1_entry_geometry_v3_grid.log"
    echo "[RESTART] launching grid suite, progress=${done_n}/${EXPECTED}, log=$log"
    (
      export GRID_TIMEOUT_S
      exec bash "$SUITE"
    ) >"$log" 2>&1 &
  fi

  sleep "$SLEEP_S"
done
