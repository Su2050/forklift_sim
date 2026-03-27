#!/usr/bin/env bash
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ISAACLAB_DIR="${ISAACLAB:-$ROOT/IsaacLab}"
RUN_DIR_BASE="$ISAACLAB_DIR/logs/rsl_rl/forklift_pallet_insert_lift"
OUTPUT_DIR="${EXP83_REPRO_EVAL_OUTPUT_DIR:-$ROOT/outputs/exp83_eval_bonusw1p0_repro_iter100}"
LOG_DIR="$ROOT/logs"

SEEDS_STR="${EXP83_REPRO_EVAL_SEEDS:-42 43 44}"
REPEATS_STR="${EXP83_REPRO_EVAL_REPEATS:-r1 r2}"
# shellcheck disable=SC2206
SEEDS=($SEEDS_STR)
# shellcheck disable=SC2206
REPEATS=($REPEATS_STR)

EVAL_TIMEOUT_SEC="${EXP83_REPRO_EVAL_TIMEOUT_SEC:-1200}"
NUM_ENVS="${EXP83_REPRO_EVAL_NUM_ENVS:-16}"
ROLLOUTS="${EXP83_REPRO_EVAL_ROLLOUTS:-4}"
SKIP_COMPLETED="${EXP83_REPRO_EVAL_SKIP_COMPLETED:-1}"
RUN_TAG="${EXP83_REPRO_EVAL_RUN_TAG:-exp83_bonusw1p0_repro}"
CHECKPOINT_NAME="${EXP83_REPRO_EVAL_CHECKPOINT_NAME:-model_99.pt}"
MAX_ITERATIONS="${EXP83_REPRO_EVAL_MAX_ITERATIONS:-100}"

mkdir -p "$OUTPUT_DIR"
mkdir -p "$LOG_DIR"
export TERM=xterm

FAILED_MANIFEST="$OUTPUT_DIR/repro_eval_failed_labels.txt"
SKIPPED_MANIFEST="$OUTPUT_DIR/repro_eval_skipped_labels.txt"
COMPLETED_MANIFEST="$OUTPUT_DIR/repro_eval_completed_labels.txt"
rm -f "$FAILED_MANIFEST" "$SKIPPED_MANIFEST" "$COMPLETED_MANIFEST"

if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook 2>/dev/null)" || true
  while [[ "${CONDA_SHLVL:-0}" =~ ^[0-9]+$ ]] && [[ "${CONDA_SHLVL:-0}" -gt 0 ]]; do
    conda deactivate 2>/dev/null || break
  done
fi
unset CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_SHLVL CONDA_PROMPT_MODIFIER CONDA_PYTHON_EXE _CE_CONDA _CE_M 2>/dev/null || true

if ! bash "$ROOT/forklift_pallet_insert_lift_project/scripts/install_into_isaaclab.sh" "$ISAACLAB_DIR"; then
  echo "[FATAL] failed to sync task into IsaacLab" >&2
  exit 2
fi

find_checkpoint() {
  local repeat_id="$1"
  local seed="$2"
  local run_name run_dir
  run_name="${RUN_TAG}_${repeat_id}_seed${seed}_iter${MAX_ITERATIONS}_256cam"
  run_dir="$(find "$RUN_DIR_BASE" -maxdepth 1 -type d -name "*_${run_name}" | sort | tail -n 1)"
  if [[ -z "$run_dir" ]]; then
    echo "[ERROR] run dir not found for repeat=${repeat_id} seed=${seed}" >&2
    return 1
  fi
  if [[ ! -f "$run_dir/$CHECKPOINT_NAME" ]]; then
    echo "[ERROR] checkpoint not found: $run_dir/$CHECKPOINT_NAME" >&2
    return 1
  fi
  echo "$run_dir/$CHECKPOINT_NAME"
}

run_eval() {
  local repeat_id="$1"
  local seed="$2"
  local label checkpoint summary_path episode_path ts eval_log

  label="exp83_eval_bonusw1p0_repro_${repeat_id}_seed${seed}_iter${MAX_ITERATIONS}"
  checkpoint="$(find_checkpoint "$repeat_id" "$seed")" || return 1
  summary_path="$OUTPUT_DIR/${label}_summary.json"
  episode_path="$OUTPUT_DIR/${label}_episodes.csv"

  if [[ "$SKIP_COMPLETED" == "1" ]] && [[ -f "$summary_path" ]] && [[ -f "$episode_path" ]]; then
    echo "[SKIP] existing summary found for ${label}"
    return 10
  fi

  ts="$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)"
  eval_log="$LOG_DIR/${ts}_eval_${label}.log"

  echo
  echo "============================================================"
  echo "[EVAL] $label"
  echo "[CKPT] $checkpoint"
  echo "============================================================"
  echo "[TRY] label=${label} num_envs=${NUM_ENVS} rollouts=${ROLLOUTS} timeout=${EVAL_TIMEOUT_SEC}s log=${eval_log}"
  rm -f "$summary_path" "$episode_path"

  if timeout --signal=TERM --kill-after=20 "${EVAL_TIMEOUT_SEC}" bash -lc "
    cd '$ISAACLAB_DIR' &&
    exec env -u CONDA_PREFIX -u CONDA_DEFAULT_ENV -u CONDA_SHLVL \
      -u CONDA_PROMPT_MODIFIER -u CONDA_PYTHON_EXE -u _CE_CONDA -u _CE_M \
      TERM='$TERM' PYTHONUNBUFFERED=1 \
      ./isaaclab.sh -p ../scripts/eval_exp83_checkpoint.py \
      --task Isaac-Forklift-PalletInsertLift-Direct-v0 \
      --headless \
      --enable_cameras \
      --checkpoint '$checkpoint' \
      --label '$label' \
      --num_envs '$NUM_ENVS' \
      --rollouts '$ROLLOUTS' \
      --seed 20260325 \
      --output_dir '$OUTPUT_DIR'
  " >"$eval_log" 2>&1; then
    if [[ -f "$summary_path" ]]; then
      echo "[OK] label=${label}"
      return 0
    fi
    echo "[WARN] eval exited without summary file: label=${label} log=${eval_log}" >&2
    return 1
  fi

  echo "[WARN] eval failed: label=${label} log=${eval_log}" >&2
  return 1
}

failed=0
declare -a FAILED_LABELS=()
declare -a SKIPPED_LABELS=()
declare -a COMPLETED_LABELS=()

for repeat_id in "${REPEATS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    label="exp83_eval_bonusw1p0_repro_${repeat_id}_seed${seed}_iter${MAX_ITERATIONS}"
    run_eval "$repeat_id" "$seed"
    rc=$?
    case "$rc" in
      0)
        COMPLETED_LABELS+=("$label")
        ;;
      10)
        SKIPPED_LABELS+=("$label")
        ;;
      *)
        failed=1
        FAILED_LABELS+=("$label")
        ;;
    esac
  done
done

printf "%s\n" "${COMPLETED_LABELS[@]:-}" | sed '/^$/d' > "$COMPLETED_MANIFEST"
printf "%s\n" "${SKIPPED_LABELS[@]:-}" | sed '/^$/d' > "$SKIPPED_MANIFEST"
printf "%s\n" "${FAILED_LABELS[@]:-}" | sed '/^$/d' > "$FAILED_MANIFEST"

echo
echo "[SUMMARY] completed=${#COMPLETED_LABELS[@]} skipped=${#SKIPPED_LABELS[@]} failed=${#FAILED_LABELS[@]}"
echo "[SUMMARY] completed manifest: $COMPLETED_MANIFEST"
echo "[SUMMARY] skipped manifest:   $SKIPPED_MANIFEST"
echo "[SUMMARY] failed manifest:    $FAILED_MANIFEST"
if [[ "$failed" -ne 0 ]]; then
  echo "[DONE] bonusw=1.0 repro unified eval finished with failures."
  if [[ -s "$FAILED_MANIFEST" ]]; then
    echo "[FAILED LABELS]"
    cat "$FAILED_MANIFEST"
  fi
  exit 1
fi
echo "[DONE] bonusw=1.0 repro unified eval finished."
