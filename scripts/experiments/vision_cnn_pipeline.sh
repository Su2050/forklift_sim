#!/usr/bin/env bash
set -euo pipefail

# Vision CNN experiment pipeline skeleton.
#
# This script is intentionally a controller/orchestrator:
# - manages branch setup
# - runs stages in sequence
# - standardizes log naming and output locations
# - keeps scratch RL and pretrained fine-tune experiments comparable
#
# Notes:
# 1) Git cannot have both "exp/vision_cnn" and "exp/vision_cnn/*" as branches.
#    To preserve the requested naming namespace, this script uses:
#      exp/vision_cnn/base
#      exp/vision_cnn/actor_backbone_upgrade
#      exp/vision_cnn/scratch_rl_baseline
#      exp/vision_cnn/cnn_pretrain
#      exp/vision_cnn/pretrained_rl_finetune
# 2) RL stages have runnable default commands.
# 3) Data collection / pretrain stages are wired as hooks and will stop with
#    a clear message until their Python entrypoints are implemented.
#
# Examples:
#   bash scripts/experiments/vision_cnn_pipeline.sh full --version s1.0zd --mode smoke
#   bash scripts/experiments/vision_cnn_pipeline.sh full --version s1.0zd --mode formal --seed 7
#   bash scripts/experiments/vision_cnn_pipeline.sh resume --from-phase pretrain-cnn --version s1.0zd
#   COLLECT_DATA_CMD="python scripts/collect_detection_data.py --frames 5000" \
#   PRETRAIN_CMD="python scripts/train_vision_cnn_pretrain.py --config configs/vision.yaml" \
#   bash scripts/experiments/vision_cnn_pipeline.sh full --version s1.0zd --mode formal

PROJECT_DIR="${PROJECT_DIR:-/home/uniubi/projects/forklift_sim}"
ISAACLAB_DIR="${ISAACLAB_DIR:-${PROJECT_DIR}/IsaacLab}"
INSTALL_SCRIPT="${INSTALL_SCRIPT:-${PROJECT_DIR}/forklift_pallet_insert_lift_project/scripts/install_into_isaaclab.sh}"

LOG_DIR="${PROJECT_DIR}/logs"
DATA_DIR="${PROJECT_DIR}/data/vision_cnn"
OUTPUT_DIR="${PROJECT_DIR}/outputs/vision_cnn"
REPORT_DIR="${PROJECT_DIR}/docs/diagnostic_reports/vision_cnn"

TASK_NAME="${TASK_NAME:-Isaac-Forklift-PalletInsertLift-Direct-v0}"
NUM_ENVS="${NUM_ENVS:-128}"
COLLECT_NUM_ENVS="${COLLECT_NUM_ENVS:-32}"
SEED="${SEED:-42}"
MODE="${MODE:-smoke}"
VERSION="${VERSION:-}"
MAX_ITERATIONS="${MAX_ITERATIONS:-}"
SANITY_ITERATIONS="${SANITY_ITERATIONS:-}"
FROM_PHASE="${FROM_PHASE:-}"
DRY_RUN="${DRY_RUN:-0}"
NOHUP_MODE="${NOHUP_MODE:-0}"
ALLOW_DIRTY="${ALLOW_DIRTY:-0}"
PRETRAINED_CKPT="${PRETRAINED_CKPT:-}"
FREEZE_BACKBONE_ITERS="${FREEZE_BACKBONE_ITERS:-0}"
BASE_BRANCH_SOURCE="${BASE_BRANCH_SOURCE:-}"
PIPELINE_LOG_FILE="${PIPELINE_LOG_FILE:-}"

COLLECT_DATA_CMD="${COLLECT_DATA_CMD:-}"
PRETRAIN_CMD="${PRETRAIN_CMD:-}"
FINETUNE_CMD="${FINETUNE_CMD:-}"
COMPARE_CMD="${COMPARE_CMD:-}"

BRANCH_NAMESPACE="exp/vision_cnn"
BRANCH_BASE="${BRANCH_NAMESPACE}/base"
BRANCH_BACKBONE="${BRANCH_NAMESPACE}/actor_backbone_upgrade"
BRANCH_SCRATCH="${BRANCH_NAMESPACE}/scratch_rl_baseline"
BRANCH_PRETRAIN="${BRANCH_NAMESPACE}/cnn_pretrain"
BRANCH_FINETUNE="${BRANCH_NAMESPACE}/pretrained_rl_finetune"

STAGES=(
  "branch-setup"
  "backbone-upgrade-sanity"
  "scratch-rl"
  "collect-data"
  "pretrain-cnn"
  "finetune-rl"
  "compare-report"
)

usage() {
  cat <<'EOF'
Usage:
  bash scripts/experiments/vision_cnn_pipeline.sh <full|resume> [options]

Options:
  --version <name>              Required. Experiment version, e.g. s1.0zd
  --mode <smoke|formal>         Default: smoke
  --from-phase <phase>          Required for resume
  --seed <int>                  Default: 42
  --num-envs <int>              Default: 128
  --collect-num-envs <int>      Default: 32 (collect-data only)
  --max-iterations <int>        Override RL train iterations
  --sanity-iterations <int>     Override sanity iterations
  --pretrained-ckpt <path>      Backbone checkpoint used by finetune stage
  --freeze-backbone-iters <n>   Metadata knob for later finetune wiring
  --base-branch-source <ref>    Base branch/commit for exp/vision_cnn/base
  --nohup                       Relaunch the whole pipeline under nohup
  --pipeline-log-file <path>    Log file used with --nohup
  --dry-run                     Print commands without executing them
  --allow-dirty                 Allow running with a dirty git worktree
  --help                        Show this message

Hook environment variables:
  COLLECT_DATA_CMD              Full command string for collect-data stage
  PRETRAIN_CMD                  Full command string for pretrain-cnn stage
  FINETUNE_CMD                  Full command string overriding default finetune
  COMPARE_CMD                   Full command string overriding compare-report

Phases:
  branch-setup
  backbone-upgrade-sanity
  scratch-rl
  collect-data
  pretrain-cnn
  finetune-rl
  compare-report

Examples:
  bash scripts/experiments/vision_cnn_pipeline.sh full --version s1.0zd --mode formal
  bash scripts/experiments/vision_cnn_pipeline.sh full --version s1.0zd --mode formal --nohup
  bash scripts/experiments/vision_cnn_pipeline.sh resume --from-phase finetune-rl --version s1.0zd --nohup
EOF
}

log() {
  printf '[%s] %s\n' "$(TZ=Asia/Shanghai date '+%Y-%m-%d %H:%M:%S')" "$*"
}

die() {
  log "ERROR: $*"
  exit 1
}

timestamp_bj() {
  TZ=Asia/Shanghai date '+%Y%m%d_%H%M%S'
}

require_file() {
  local path="$1"
  [[ -e "${path}" ]] || die "Required file not found: ${path}"
}

ensure_directories() {
  mkdir -p "${LOG_DIR}" "${DATA_DIR}" "${OUTPUT_DIR}" "${REPORT_DIR}" "${RUN_DIR}"
}

registry_file() {
  echo "${RUN_DIR}/pipeline_state.env"
}

append_registry() {
  local key="$1"
  local value="$2"
  printf '%s=%q\n' "${key}" "${value}" >> "$(registry_file)"
}

load_registry() {
  local file
  file="$(registry_file)"
  if [[ -f "${file}" ]]; then
    # shellcheck disable=SC1090
    source "${file}"
  fi
}

pipeline_log_file_default() {
  echo "${RUN_DIR}/pipeline_nohup_$(timestamp_bj).log"
}

relaunch_args_without_nohup_flags() {
  local filtered=()
  local skip_next=0
  local arg

  for arg in "${ORIGINAL_ARGS[@]}"; do
    if [[ "${skip_next}" == "1" ]]; then
      skip_next=0
      continue
    fi

    case "${arg}" in
      --nohup)
        continue
        ;;
      --pipeline-log-file)
        skip_next=1
        continue
        ;;
      *)
        filtered+=("${arg}")
        ;;
    esac
  done

  printf '%s\n' "${filtered[@]}"
}

maybe_relaunch_with_nohup() {
  local pipeline_log
  local script_path
  local filtered_args=()
  local line
  local pid

  [[ "${NOHUP_MODE}" == "1" ]] || return 0
  [[ -z "${VISION_CNN_NOHUP_CHILD:-}" ]] || return 0

  pipeline_log="${PIPELINE_LOG_FILE:-$(pipeline_log_file_default)}"
  mkdir -p "$(dirname "${pipeline_log}")"
  script_path="${PROJECT_DIR}/scripts/experiments/vision_cnn_pipeline.sh"
  require_file "${script_path}"

  while IFS= read -r line; do
    [[ -n "${line}" ]] && filtered_args+=("${line}")
  done < <(relaunch_args_without_nohup_flags)

  if [[ "${DRY_RUN}" == "1" ]]; then
    log "dry-run: would relaunch pipeline with nohup"
    log "pipeline log: ${pipeline_log}"
    return 0
  fi

  (
    cd "${PROJECT_DIR}"
    nohup env VISION_CNN_NOHUP_CHILD=1 bash "${script_path}" "${filtered_args[@]}" > "${pipeline_log}" 2>&1 &
    pid=$!
    printf '%s\n' "${pid}" > "${RUN_DIR}/pipeline_nohup.pid"
    log "pipeline relaunched with nohup"
    log "pid: ${pid}"
    log "pipeline log: ${pipeline_log}"
  )
  exit 0
}

normalize_stage_key() {
  echo "$1" | tr '[:lower:]-' '[:upper:]_'
}

log_file_for_type() {
  local log_type="$1"
  echo "${LOG_DIR}/$(timestamp_bj)_${log_type}_${VERSION}.log"
}

stage_note_file() {
  local stage="$1"
  echo "${RUN_DIR}/$(normalize_stage_key "${stage}")_note.txt"
}

write_stage_note() {
  local stage="$1"
  shift
  {
    echo "stage=${stage}"
    echo "version=${VERSION}"
    echo "mode=${MODE}"
    echo "branch=$(git -C "${PROJECT_DIR}" branch --show-current 2>/dev/null || true)"
    printf '%s\n' "$@"
  } > "$(stage_note_file "${stage}")"
}

run_or_print() {
  local workdir="$1"
  local cmd="$2"

  log "cwd: ${workdir}"
  log "cmd: ${cmd}"
  if [[ "${DRY_RUN}" == "1" ]]; then
    log "dry-run enabled, command not executed"
    return 0
  fi

  (
    cd "${workdir}"
    bash -lc "${cmd}"
  )
}

run_logged_stage() {
  local stage="$1"
  local log_type="$2"
  local workdir="$3"
  local cmd="$4"
  local logfile

  logfile="$(log_file_for_type "${log_type}")"
  log "stage ${stage} -> ${logfile}"
  write_stage_note "${stage}" "log_file=${logfile}" "command=${cmd}"
  append_registry "$(normalize_stage_key "${stage}")_LOG" "${logfile}"

  if [[ "${DRY_RUN}" == "1" ]]; then
    log "dry-run enabled, log file reserved: ${logfile}"
    log "cmd: ${cmd}"
    return 0
  fi

  (
    cd "${workdir}"
    bash -lc "${cmd}" > "${logfile}" 2>&1
  )

  log "stage ${stage} completed"
}

require_git_repo() {
  git -C "${PROJECT_DIR}" rev-parse --show-toplevel >/dev/null 2>&1 || die "Not a git repo: ${PROJECT_DIR}"
}

require_clean_worktree() {
  if [[ "${ALLOW_DIRTY}" == "1" ]]; then
    return 0
  fi

  git -C "${PROJECT_DIR}" diff --quiet || die "Working tree has unstaged changes. Re-run with --allow-dirty if intentional."
  git -C "${PROJECT_DIR}" diff --cached --quiet || die "Working tree has staged changes. Re-run with --allow-dirty if intentional."
  if [[ -n "$(git -C "${PROJECT_DIR}" ls-files --others --exclude-standard)" ]]; then
    die "Working tree has untracked files. Re-run with --allow-dirty if intentional."
  fi
}

branch_exists() {
  local branch="$1"
  git -C "${PROJECT_DIR}" show-ref --verify --quiet "refs/heads/${branch}"
}

ref_exists() {
  local ref="$1"
  git -C "${PROJECT_DIR}" rev-parse --verify "${ref}" >/dev/null 2>&1
}

ensure_branch_exists() {
  local branch="$1"
  local base_ref="$2"

  if branch_exists "${branch}"; then
    log "branch exists: ${branch}"
    return 0
  fi

  [[ -n "${base_ref}" ]] || die "Missing base ref for branch creation: ${branch}"
  ref_exists "${base_ref}" || die "Base ref does not exist: ${base_ref}"

  if [[ "${DRY_RUN}" == "1" ]]; then
    log "dry-run: git branch ${branch} ${base_ref}"
    return 0
  fi

  git -C "${PROJECT_DIR}" branch "${branch}" "${base_ref}"
  log "created branch ${branch} from ${base_ref}"
}

checkout_branch() {
  local branch="$1"

  if [[ "${DRY_RUN}" == "1" ]]; then
    log "dry-run: git checkout ${branch}"
    return 0
  fi

  git -C "${PROJECT_DIR}" checkout "${branch}" >/dev/null
  log "checked out ${branch}"
}

install_patch_into_isaaclab() {
  require_file "${INSTALL_SCRIPT}"
  require_file "${ISAACLAB_DIR}/isaaclab.sh"

  run_or_print "${PROJECT_DIR}" "bash \"${INSTALL_SCRIPT}\" \"${ISAACLAB_DIR}\""
}

default_sanity_iterations() {
  if [[ -n "${SANITY_ITERATIONS}" ]]; then
    echo "${SANITY_ITERATIONS}"
  elif [[ "${MODE}" == "smoke" ]]; then
    echo "30"
  else
    echo "100"
  fi
}

default_train_iterations() {
  if [[ -n "${MAX_ITERATIONS}" ]]; then
    echo "${MAX_ITERATIONS}"
  elif [[ "${MODE}" == "smoke" ]]; then
    echo "30"
  else
    echo "2000"
  fi
}

default_sanity_command() {
  local iters
  iters="$(default_sanity_iterations)"
  cat <<EOF
export PYTHONUNBUFFERED=1
export TERM=xterm
CONDA_PREFIX="" bash isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task "${TASK_NAME}" \
  --headless --enable_cameras --num_envs "${NUM_ENVS}" --max_iterations "${iters}" \
  --seed "${SEED}"
EOF
}

default_scratch_command() {
  local iters
  iters="$(default_train_iterations)"
  cat <<EOF
export PYTHONUNBUFFERED=1
export TERM=xterm
CONDA_PREFIX="" bash isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task "${TASK_NAME}" \
  --headless --enable_cameras --num_envs "${NUM_ENVS}" --max_iterations "${iters}" \
  --seed "${SEED}"
EOF
}

default_finetune_command() {
  local iters
  local pretrained_arg=""
  iters="$(default_train_iterations)"

  load_registry
  local ckpt="${PRETRAINED_CKPT:-}"
  [[ -n "${ckpt}" ]] || die "finetune-rl requires PRETRAINED_CKPT to be set (either via arg or previous stage registry)"
  require_file "${ckpt}"

  # Placeholder flags below should be replaced once RL entrypoint supports them.
  pretrained_arg="--vision_backbone_ckpt \"${ckpt}\" --freeze_backbone_iters \"${FREEZE_BACKBONE_ITERS}\""

  cat <<EOF
export PYTHONUNBUFFERED=1
export TERM=xterm
echo "[TODO] Replace placeholder RL finetune flags if your training entrypoint uses different option names."
CONDA_PREFIX="" bash isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task "${TASK_NAME}" \
  --headless --enable_cameras --num_envs "${NUM_ENVS}" --max_iterations "${iters}" \
  --seed "${SEED}" \
  ${pretrained_arg}
EOF
}

default_collect_command() {
  local default_script="${PROJECT_DIR}/scripts/collect_detection_data.py"

  if [[ -n "${COLLECT_DATA_CMD}" ]]; then
    echo "${COLLECT_DATA_CMD}"
    return 0
  fi

  if [[ -f "${default_script}" ]]; then
    cat <<EOF
export PYTHONUNBUFFERED=1
export TERM=xterm
CONDA_PREFIX="" bash "${ISAACLAB_DIR}/isaaclab.sh" -p "${default_script}" \
  --output-dir "${DATA_DIR}" --version "${VERSION}" \
  --headless --enable_cameras --num-envs "${COLLECT_NUM_ENVS}" --seed "${SEED}"
EOF
    return 0
  fi

  die "collect-data stage is not wired yet. Set COLLECT_DATA_CMD or add scripts/collect_detection_data.py."
}

default_pretrain_command() {
  local default_script="${PROJECT_DIR}/scripts/train_vision_cnn_pretrain.py"

  if [[ -n "${PRETRAIN_CMD}" ]]; then
    echo "${PRETRAIN_CMD}"
    return 0
  fi

  if [[ -f "${default_script}" ]]; then
    cat <<EOF
python "${default_script}" --data-dir "${DATA_DIR}" --output-dir "${RUN_DIR}/pretrain" --version "${VERSION}"
# TODO: The pretrain script should output the final ckpt path to a known location or stdout
# For now, we mock the registry update so the pipeline doesn't break if the user manually sets it
echo "PRETRAINED_CKPT=${RUN_DIR}/pretrain/model_best.pt" >> "\$(registry_file)"
EOF
    return 0
  fi

  die "pretrain-cnn stage is not wired yet. Set PRETRAIN_CMD or add scripts/train_vision_cnn_pretrain.py."
}

run_branch_setup() {
  local base_source

  require_git_repo
  require_clean_worktree

  if [[ -n "${BASE_BRANCH_SOURCE}" ]]; then
    base_source="${BASE_BRANCH_SOURCE}"
  else
    base_source="$(git -C "${PROJECT_DIR}" branch --show-current)"
  fi

  [[ -n "${base_source}" ]] || die "Unable to infer base branch source. Use --base-branch-source."

  log "using base source: ${base_source}"
  write_stage_note "branch-setup" \
    "namespace=${BRANCH_NAMESPACE}" \
    "git_note=exp/vision_cnn is treated as namespace; shared ancestor branch is exp/vision_cnn/base"

  ensure_branch_exists "${BRANCH_BASE}" "${base_source}"
  ensure_branch_exists "${BRANCH_BACKBONE}" "${BRANCH_BASE}"
  ensure_branch_exists "${BRANCH_SCRATCH}" "${BRANCH_BACKBONE}"
  ensure_branch_exists "${BRANCH_PRETRAIN}" "${BRANCH_BACKBONE}"
  ensure_branch_exists "${BRANCH_FINETUNE}" "${BRANCH_BACKBONE}"

  append_registry "BRANCH_BASE" "${BRANCH_BASE}"
  append_registry "BRANCH_BACKBONE" "${BRANCH_BACKBONE}"
  append_registry "BRANCH_SCRATCH" "${BRANCH_SCRATCH}"
  append_registry "BRANCH_PRETRAIN" "${BRANCH_PRETRAIN}"
  append_registry "BRANCH_FINETUNE" "${BRANCH_FINETUNE}"
}

run_backbone_upgrade_sanity() {
  checkout_branch "${BRANCH_BACKBONE}"
  install_patch_into_isaaclab
  run_logged_stage "backbone-upgrade-sanity" "sanity_check" "${ISAACLAB_DIR}" "$(default_sanity_command)"
}

run_scratch_rl() {
  checkout_branch "${BRANCH_SCRATCH}"
  install_patch_into_isaaclab
  run_logged_stage "scratch-rl" "$( [[ "${MODE}" == "smoke" ]] && echo "smoke_train" || echo "train" )" "${ISAACLAB_DIR}" "$(default_scratch_command)"
}

run_collect_data() {
  checkout_branch "${BRANCH_PRETRAIN}"
  install_patch_into_isaaclab
  run_logged_stage "collect-data" "sanity_check" "${PROJECT_DIR}" "$(default_collect_command)"
}

run_pretrain_cnn() {
  checkout_branch "${BRANCH_PRETRAIN}"
  run_logged_stage "pretrain-cnn" "$( [[ "${MODE}" == "smoke" ]] && echo "smoke_train" || echo "train" )" "${PROJECT_DIR}" "$(default_pretrain_command)"
}

run_finetune_rl() {
  local cmd

  checkout_branch "${BRANCH_FINETUNE}"
  install_patch_into_isaaclab

  if [[ -n "${FINETUNE_CMD}" ]]; then
    cmd="${FINETUNE_CMD}"
  else
    cmd="$(default_finetune_command)"
  fi

  run_logged_stage "finetune-rl" "$( [[ "${MODE}" == "smoke" ]] && echo "smoke_train" || echo "train" )" "${ISAACLAB_DIR}" "${cmd}"
}

run_compare_report() {
  local report_file

  load_registry
  report_file="${REPORT_DIR}/$(timestamp_bj)_vision_cnn_compare_${VERSION}.md"

  if [[ -n "${COMPARE_CMD}" ]]; then
    run_logged_stage "compare-report" "sanity_check" "${PROJECT_DIR}" "${COMPARE_CMD}"
    return 0
  fi

  if [[ "${DRY_RUN}" == "1" ]]; then
    log "dry-run: would write compare report to ${report_file}"
    return 0
  fi

  cat > "${report_file}" <<EOF
# Vision CNN Compare Report

- version: ${VERSION}
- mode: ${MODE}
- seed: ${SEED}
- num_envs: ${NUM_ENVS}
- branch_backbone: ${BRANCH_BACKBONE}
- branch_scratch: ${BRANCH_SCRATCH}
- branch_pretrain: ${BRANCH_PRETRAIN}
- branch_finetune: ${BRANCH_FINETUNE}

## Artifacts

- scratch log: ${SCRATCH_RL_LOG:-TBD}
- collect-data log: ${COLLECT_DATA_LOG:-TBD}
- pretrain log: ${PRETRAIN_CNN_LOG:-TBD}
- finetune log: ${FINETUNE_RL_LOG:-TBD}
- pretrained ckpt: ${PRETRAINED_CKPT:-TBD} (freeze iters: ${FREEZE_BACKBONE_ITERS})

## Manual Comparison Checklist

- success rate
- mean reward
- convergence speed
- stability
- best checkpoint performance

## Conclusion

- TBD
EOF

  append_registry "COMPARE_REPORT_FILE" "${report_file}"
  log "compare report created: ${report_file}"
}

run_stage() {
  case "$1" in
    branch-setup) run_branch_setup ;;
    backbone-upgrade-sanity) run_backbone_upgrade_sanity ;;
    scratch-rl) run_scratch_rl ;;
    collect-data) run_collect_data ;;
    pretrain-cnn) run_pretrain_cnn ;;
    finetune-rl) run_finetune_rl ;;
    compare-report) run_compare_report ;;
    *) die "Unknown stage: $1" ;;
  esac
}

phase_index() {
  local target="$1"
  local idx
  for idx in "${!STAGES[@]}"; do
    if [[ "${STAGES[$idx]}" == "${target}" ]]; then
      echo "${idx}"
      return 0
    fi
  done
  return 1
}

parse_args() {
  ACTION="${1:-}"
  [[ -n "${ACTION}" ]] || {
    usage
    exit 1
  }
  shift || true

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --version)
        VERSION="${2:?missing value for --version}"
        shift 2
        ;;
      --mode)
        MODE="${2:?missing value for --mode}"
        shift 2
        ;;
      --from-phase)
        FROM_PHASE="${2:?missing value for --from-phase}"
        shift 2
        ;;
      --seed)
        SEED="${2:?missing value for --seed}"
        shift 2
        ;;
      --num-envs)
        NUM_ENVS="${2:?missing value for --num-envs}"
        shift 2
        ;;
      --collect-num-envs)
        COLLECT_NUM_ENVS="${2:?missing value for --collect-num-envs}"
        shift 2
        ;;
      --max-iterations)
        MAX_ITERATIONS="${2:?missing value for --max-iterations}"
        shift 2
        ;;
      --sanity-iterations)
        SANITY_ITERATIONS="${2:?missing value for --sanity-iterations}"
        shift 2
        ;;
      --pretrained-ckpt)
        PRETRAINED_CKPT="${2:?missing value for --pretrained-ckpt}"
        shift 2
        ;;
      --freeze-backbone-iters)
        FREEZE_BACKBONE_ITERS="${2:?missing value for --freeze-backbone-iters}"
        shift 2
        ;;
      --base-branch-source)
        BASE_BRANCH_SOURCE="${2:?missing value for --base-branch-source}"
        shift 2
        ;;
      --nohup)
        NOHUP_MODE="1"
        shift
        ;;
      --pipeline-log-file)
        PIPELINE_LOG_FILE="${2:?missing value for --pipeline-log-file}"
        shift 2
        ;;
      --dry-run)
        DRY_RUN="1"
        shift
        ;;
      --allow-dirty)
        ALLOW_DIRTY="1"
        shift
        ;;
      --help|-h)
        usage
        exit 0
        ;;
      *)
        die "Unknown argument: $1"
        ;;
    esac
  done
}

main() {
  local start_index=0
  local idx

  ORIGINAL_ARGS=("$@")
  parse_args "$@"
  [[ -n "${VERSION}" ]] || die "--version is required"
  [[ "${MODE}" == "smoke" || "${MODE}" == "formal" ]] || die "--mode must be smoke or formal"
  [[ "${ACTION}" == "full" || "${ACTION}" == "resume" ]] || die "Action must be full or resume"

  RUN_DIR="${OUTPUT_DIR}/${VERSION}"
  ensure_directories
  maybe_relaunch_with_nohup

  if [[ "${ACTION}" == "resume" ]]; then
    [[ -n "${FROM_PHASE}" ]] || die "--from-phase is required for resume"
    phase_index "${FROM_PHASE}" >/dev/null || die "Unknown --from-phase: ${FROM_PHASE}"
    start_index="$(phase_index "${FROM_PHASE}")"
  else
    : > "$(registry_file)"
  fi

  append_registry "VERSION" "${VERSION}"
  append_registry "MODE" "${MODE}"
  append_registry "SEED" "${SEED}"
  append_registry "NUM_ENVS" "${NUM_ENVS}"
  append_registry "COLLECT_NUM_ENVS" "${COLLECT_NUM_ENVS}"
  append_registry "MAX_ITERATIONS" "${MAX_ITERATIONS}"
  append_registry "SANITY_ITERATIONS" "${SANITY_ITERATIONS}"
  append_registry "PRETRAINED_CKPT" "${PRETRAINED_CKPT}"

  for idx in "${!STAGES[@]}"; do
    if (( idx < start_index )); then
      continue
    fi

    log "============================================================"
    log "running stage: ${STAGES[$idx]}"
    log "============================================================"
    run_stage "${STAGES[$idx]}"
  done

  log "pipeline finished"
  log "run directory: ${RUN_DIR}"
}

main "$@"
