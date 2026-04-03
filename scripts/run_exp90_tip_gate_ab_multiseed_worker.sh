#!/usr/bin/env bash
set -euo pipefail

TIMESTAMP=""
SEEDS="42,43"
NUM_ENVS="64"
MAX_ITERATIONS="200"
RUN_PREFIX="exp9_0_tipgate_ab_multiseed_o3"
ROOT=""
ISAACLAB_DIR=""
CONDA_BASE=""
CONDA_ENV_PATH=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --timestamp)
      TIMESTAMP="$2"
      shift 2
      ;;
    --seeds)
      SEEDS="$2"
      shift 2
      ;;
    --num-envs)
      NUM_ENVS="$2"
      shift 2
      ;;
    --max-iterations)
      MAX_ITERATIONS="$2"
      shift 2
      ;;
    --run-prefix)
      RUN_PREFIX="$2"
      shift 2
      ;;
    --root)
      ROOT="$2"
      shift 2
      ;;
    --isaaclab-dir)
      ISAACLAB_DIR="$2"
      shift 2
      ;;
    --conda-base)
      CONDA_BASE="$2"
      shift 2
      ;;
    --conda-env-path)
      CONDA_ENV_PATH="$2"
      shift 2
      ;;
    *)
      echo "[FATAL] unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

if [[ -z "$TIMESTAMP" || -z "$ROOT" || -z "$ISAACLAB_DIR" || -z "$CONDA_BASE" || -z "$CONDA_ENV_PATH" ]]; then
  echo "[FATAL] missing required arguments" >&2
  exit 2
fi

DOC_DIR="$ROOT/docs/exp9_0"
MULTISEED_REPORT="$DOC_DIR/exp9_0_tipgate_ab_multiseed_compare_${TIMESTAMP}.md"
SUMMARY_SCRIPT="$ROOT/scripts/summarize_exp90_tip_gate_ab_multiseed.py"

IFS=',' read -r -a seed_list <<< "$SEEDS"
strict_logs=()
relaxed_logs=()

for seed in "${seed_list[@]}"; do
  seed="$(echo "$seed" | xargs)"
  [[ -n "$seed" ]] || continue
  seed_ts="${TIMESTAMP}_seed${seed}"
  echo "[BATCH] starting seed=$seed timestamp=$seed_ts"
  bash "$ROOT/scripts/run_exp90_tip_gate_ab_short_worker.sh" \
    --timestamp "$seed_ts" \
    --seed "$seed" \
    --num-envs "$NUM_ENVS" \
    --max-iterations "$MAX_ITERATIONS" \
    --run-prefix "$RUN_PREFIX" \
    --root "$ROOT" \
    --isaaclab-dir "$ISAACLAB_DIR" \
    --conda-base "$CONDA_BASE" \
    --conda-env-path "$CONDA_ENV_PATH"
  strict_logs+=("$ROOT/logs/${seed_ts}_train_${RUN_PREFIX}_strict_seed${seed}_iter${MAX_ITERATIONS}.log")
  relaxed_logs+=("$ROOT/logs/${seed_ts}_train_${RUN_PREFIX}_relaxed0175_seed${seed}_iter${MAX_ITERATIONS}.log")
done

python "$SUMMARY_SCRIPT" \
  --strict-logs "${strict_logs[@]}" \
  --relaxed-logs "${relaxed_logs[@]}" \
  --output "$MULTISEED_REPORT"

echo "[DONE] multiseed report: $MULTISEED_REPORT"
