#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${LOG_DIR:-$ROOT/logs}"
SEEDS="${SEEDS:-42,43}"
NUM_ENVS="${NUM_ENVS:-64}"
MAX_ITERATIONS="${MAX_ITERATIONS:-200}"
RUN_PREFIX="${RUN_PREFIX:-exp9_0_tipgate_ab_multiseed_o3}"
ISAACLAB_DIR="${ISAACLAB_DIR:-/home/uniubi/projects/forklift_sim/IsaacLab}"
CONDA_BASE="${CONDA_BASE:-/home/uniubi/miniconda3}"
CONDA_ENV_PATH="${CONDA_ENV_PATH:-${CONDA_BASE}/envs/env_isaaclab}"

mkdir -p "$LOG_DIR"
if [[ -z "${TERM:-}" || "${TERM}" == "dumb" ]]; then
  export TERM="xterm"
fi

TS="$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)"
MASTER_LOG="$LOG_DIR/${TS}_${RUN_PREFIX}_worker.log"

CMD=(
  bash "$ROOT/scripts/run_exp90_tip_gate_ab_multiseed_worker.sh"
  --timestamp "$TS"
  --seeds "$SEEDS"
  --num-envs "$NUM_ENVS"
  --max-iterations "$MAX_ITERATIONS"
  --run-prefix "$RUN_PREFIX"
  --root "$ROOT"
  --isaaclab-dir "$ISAACLAB_DIR"
  --conda-base "$CONDA_BASE"
  --conda-env-path "$CONDA_ENV_PATH"
)

nohup env TERM="$TERM" PYTHONUNBUFFERED=1 "${CMD[@]}" >"$MASTER_LOG" 2>&1 &
PID=$!

echo "Started Exp9.0 tip-gate multiseed A/B worker with nohup."
echo "master_log: $MASTER_LOG"
echo "pid: $PID"
echo "seeds: $SEEDS"
echo "num_envs: $NUM_ENVS"
echo "max_iterations: $MAX_ITERATIONS"
echo "run_prefix: $RUN_PREFIX"
echo ""
echo "Monitor:"
echo "  tail -f $MASTER_LOG"
