#!/usr/bin/env bash
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ISAACLAB_DIR="${ISAACLAB:-$ROOT/IsaacLab}"
RUN_DIR_BASE="$ISAACLAB_DIR/logs/rsl_rl/forklift_pallet_insert_lift"
OUTPUT_DIR="$ROOT/outputs/exp83_eval_bonus_weight_sweep"
LOG_DIR="$ROOT/logs"
SEEDS=(42 43 44)
WEIGHTS=(0.5 1.0 1.5)
EVAL_TIMEOUT_SEC="${EXP83_EVAL_TIMEOUT_SEC:-900}"
PRIMARY_NUM_ENVS="${EXP83_EVAL_NUM_ENVS:-16}"
PRIMARY_ROLLOUTS="${EXP83_EVAL_ROLLOUTS:-4}"
FALLBACK_NUM_ENVS="${EXP83_EVAL_FALLBACK_NUM_ENVS:-8}"
FALLBACK_ROLLOUTS="${EXP83_EVAL_FALLBACK_ROLLOUTS:-8}"
LAST_RESORT_NUM_ENVS="${EXP83_EVAL_LAST_NUM_ENVS:-4}"
LAST_RESORT_ROLLOUTS="${EXP83_EVAL_LAST_ROLLOUTS:-16}"
SKIP_COMPLETED="${EXP83_EVAL_SKIP_COMPLETED:-1}"

mkdir -p "$OUTPUT_DIR"
mkdir -p "$LOG_DIR"
export TERM=xterm

FAILED_MANIFEST="$OUTPUT_DIR/bonus_weight_eval_failed_labels.txt"
SKIPPED_MANIFEST="$OUTPUT_DIR/bonus_weight_eval_skipped_labels.txt"
COMPLETED_MANIFEST="$OUTPUT_DIR/bonus_weight_eval_completed_labels.txt"
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

weight_tag() {
  local weight="$1"
  echo "${weight//./p}"
}

find_checkpoint() {
  local weight="$1"
  local seed="$2"
  local tag run_name run_dir
  tag="$(weight_tag "$weight")"
  run_name="exp83_bonusw${tag}_seed${seed}_iter50_256cam"
  run_dir="$(find "$RUN_DIR_BASE" -maxdepth 1 -type d -name "*_${run_name}" | sort | tail -n 1)"
  if [[ -z "$run_dir" ]]; then
    echo "[ERROR] run dir not found for weight=${weight} seed=${seed}" >&2
    return 1
  fi
  echo "$run_dir/model_49.pt"
}

run_eval() {
  local weight="$1"
  local seed="$2"
  local tag label checkpoint
  tag="$(weight_tag "$weight")"
  label="exp83_eval_bonusw${tag}_seed${seed}"
  checkpoint="$(find_checkpoint "$weight" "$seed")"
  local summary_path="$OUTPUT_DIR/${label}_summary.json"
  local episode_path="$OUTPUT_DIR/${label}_episodes.csv"
  local ts eval_log

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
  rm -f "$summary_path" "$episode_path"

  local combos=(
    "${PRIMARY_NUM_ENVS}:${PRIMARY_ROLLOUTS}"
    "${FALLBACK_NUM_ENVS}:${FALLBACK_ROLLOUTS}"
    "${LAST_RESORT_NUM_ENVS}:${LAST_RESORT_ROLLOUTS}"
  )

  local combo num_envs rollouts
  for combo in "${combos[@]}"; do
    IFS=":" read -r num_envs rollouts <<<"$combo"
    echo "[TRY] label=${label} num_envs=${num_envs} rollouts=${rollouts} timeout=${EVAL_TIMEOUT_SEC}s log=${eval_log}"
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
        --num_envs '$num_envs' \
        --rollouts '$rollouts' \
        --seed 20260325 \
        --output_dir '$OUTPUT_DIR'
    " >"$eval_log" 2>&1; then
      if [[ -f "$summary_path" ]]; then
        echo "[OK] label=${label} num_envs=${num_envs} rollouts=${rollouts}"
        return 0
      fi
      echo "[WARN] eval exited without summary file: label=${label} log=${eval_log}"
    else
      echo "[WARN] eval failed: label=${label} num_envs=${num_envs} rollouts=${rollouts} log=${eval_log}"
    fi
    sleep 2
  done

  echo "[ERROR] all eval attempts failed for ${label}" >&2
  return 1
}

failed=0
declare -a FAILED_LABELS=()
declare -a SKIPPED_LABELS=()
declare -a COMPLETED_LABELS=()
for weight in "${WEIGHTS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    tag="$(weight_tag "$weight")"
    label="exp83_eval_bonusw${tag}_seed${seed}"
    run_eval "$weight" "$seed"
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
  echo "[DONE] Bonus-weight unified eval suite finished with failures."
  if [[ -s "$FAILED_MANIFEST" ]]; then
    echo "[FAILED LABELS]"
    cat "$FAILED_MANIFEST"
  fi
  exit 1
fi
echo "[DONE] Bonus-weight unified eval suite finished."
