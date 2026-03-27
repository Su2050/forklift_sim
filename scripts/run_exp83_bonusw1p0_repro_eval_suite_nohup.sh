#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$ROOT/logs"
WORKER_SCRIPT="$ROOT/scripts/run_exp83_bonusw1p0_repro_eval_suite.sh"
KILL_EXISTING="${EXP83_REPRO_EVAL_KILL_EXISTING:-0}"

mkdir -p "$LOG_DIR"

if [[ ! -x "$WORKER_SCRIPT" ]]; then
  echo "[FATAL] worker script is missing or not executable: $WORKER_SCRIPT" >&2
  exit 2
fi

existing="$(pgrep -af "eval_exp83_checkpoint.py" || true)"
if [[ -n "$existing" ]]; then
  echo "[WARN] detected existing eval_exp83_checkpoint.py processes:"
  echo "$existing"
  if [[ "$KILL_EXISTING" == "1" ]]; then
    echo "[INFO] stopping existing eval_exp83_checkpoint.py processes before launch..."
    pkill -f "eval_exp83_checkpoint.py" || true
    sleep 2
    pkill -9 -f "eval_exp83_checkpoint.py" || true
    sleep 1
  else
    echo "[FATAL] existing eval_exp83_checkpoint.py processes detected. Re-run with EXP83_REPRO_EVAL_KILL_EXISTING=1 to stop them first." >&2
    exit 3
  fi
fi

echo "[INFO] Memory snapshot before launch:"
free -h
echo
echo "[INFO] GPU snapshot before launch:"
nvidia-smi
echo

ts="$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)"
nohup_log="$LOG_DIR/${ts}_eval_exp83_bonusw1p0_repro_iter100_suite.log"

nohup bash "$WORKER_SCRIPT" >"$nohup_log" 2>&1 < /dev/null &
pid="$!"

echo "[STARTED] pid=$pid"
echo "[LOG] $nohup_log"
echo "[WORKER] $WORKER_SCRIPT"
