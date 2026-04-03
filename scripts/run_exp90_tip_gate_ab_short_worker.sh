#!/usr/bin/env bash
set -euo pipefail

TIMESTAMP=""
SEED="42"
NUM_ENVS="64"
MAX_ITERATIONS="100"
RUN_PREFIX="exp9_0_tipgate_ab_short_o3"
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
    --seed)
      SEED="$2"
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

LOG_DIR="$ROOT/logs"
DOC_DIR="$ROOT/docs/exp9_0"
ASSET_DST_DIR="$(cd "${ISAACLAB_DIR}/.." && pwd)/assets"
SUMMARY_SCRIPT="$ROOT/scripts/summarize_exp90_tip_gate_ab.py"

mkdir -p "$LOG_DIR" "$DOC_DIR" "$ASSET_DST_DIR"
if [[ -z "${TERM:-}" || "${TERM}" == "dumb" ]]; then
  export TERM="xterm"
fi

STRICT_RUN_NAME="${RUN_PREFIX}_strict_seed${SEED}_iter${MAX_ITERATIONS}"
RELAXED_RUN_NAME="${RUN_PREFIX}_relaxed0175_seed${SEED}_iter${MAX_ITERATIONS}"
STRICT_LOG="$LOG_DIR/${TIMESTAMP}_train_${STRICT_RUN_NAME}.log"
RELAXED_LOG="$LOG_DIR/${TIMESTAMP}_train_${RELAXED_RUN_NAME}.log"
REPORT_PATH="$DOC_DIR/exp9_0_tipgate_ab_short_compare_${TIMESTAMP}.md"

echo "[INFO] timestamp: $TIMESTAMP"
echo "[INFO] root: $ROOT"
echo "[INFO] IsaacLab: $ISAACLAB_DIR"
echo "[INFO] conda env: $CONDA_ENV_PATH"
echo "[INFO] strict log: $STRICT_LOG"
echo "[INFO] relaxed log: $RELAXED_LOG"
echo "[INFO] report path: $REPORT_PATH"

if [[ ! -f "${CONDA_BASE}/bin/activate" ]]; then
  echo "[FATAL] missing conda activate script: ${CONDA_BASE}/bin/activate" >&2
  exit 2
fi
if [[ ! -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]]; then
  echo "[FATAL] missing conda.sh: ${CONDA_BASE}/etc/profile.d/conda.sh" >&2
  exit 2
fi
if [[ ! -d "${CONDA_ENV_PATH}" ]]; then
  echo "[FATAL] missing conda env: ${CONDA_ENV_PATH}" >&2
  exit 2
fi
if [[ ! -f "${ISAACLAB_DIR}/isaaclab.sh" ]]; then
  echo "[FATAL] missing isaaclab.sh: ${ISAACLAB_DIR}/isaaclab.sh" >&2
  exit 2
fi
if [[ ! -d "${ROOT}/assets" ]]; then
  echo "[FATAL] missing source assets directory: ${ROOT}/assets" >&2
  exit 2
fi

# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_PATH}"

python - <<'PY'
import importlib.metadata as metadata
import isaaclab
import isaacsim

print("[INFO] Python runtime precheck passed.")
print("[INFO] rsl-rl-lib=" + metadata.version("rsl-rl-lib"))
PY

rsync -a "${ROOT}/assets/" "${ASSET_DST_DIR}/"
echo "[INFO] Synced assets -> ${ASSET_DST_DIR}"

bash "$ROOT/forklift_pallet_insert_lift_project/scripts/install_into_isaaclab.sh" "$ISAACLAB_DIR"

build_base_cmd() {
  local run_name="$1"
  local -a cmd=(
    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py
    --task Isaac-Forklift-PalletInsertLift-Direct-v0
    --headless
    --enable_cameras
    --seed "$SEED"
    --num_envs "$NUM_ENVS"
    --max_iterations "$MAX_ITERATIONS"
    agent.run_name="$run_name"
    env.use_camera=true
    env.use_asymmetric_critic=true
    env.stage_1_mode=true
    env.use_reference_trajectory=false
    env.alpha_2=0.0
    env.alpha_3=0.0
    env.camera_width=256
    env.camera_height=256
    env.clean_insert_reward_gate_enable=true
    env.clean_insert_use_push_gate=true
    env.clean_insert_center_sigma_m=0.10
    env.clean_insert_yaw_sigma_deg=6.0
    env.clean_insert_tip_sigma_m=0.10
    env.clean_insert_push_sigma_m=0.10
    env.clean_insert_gate_r_cd=false
    env.clean_insert_gate_r_cpsi=false
    env.clean_insert_dirty_penalty_enable=false
    env.clean_insert_gate_floor=0.05
    env.clean_insert_push_free_bonus_enable=false
    env.preinsert_align_reward_enable=true
    env.preinsert_insert_frac_max=0.40
    env.preinsert_y_err_delta_weight=2.0
    env.preinsert_yaw_err_delta_weight=1.5
    env.preinsert_dist_front_delta_weight=0.15
    env.postinsert_align_enable=true
    env.postinsert_align_weight=2.0
    env.postinsert_center_sigma_m=0.40
    env.postinsert_tip_sigma_m=0.40
    env.postinsert_center_weight=1.0
    env.postinsert_tip_weight=1.0
    env.postinsert_yaw_sigma_deg=10.0
    env.postinsert_yaw_weight=0.5
    env.prehold_reachable_tip_band_m=0.17
    agent.policy.class_name=rsl_rl.modules.VisionActorCritic
    agent.policy.backbone_type=resnet34
    agent.policy.imagenet_backbone_init=true
    agent.policy.freeze_backbone=true
    'agent.obs_groups.policy=[image,proprio]'
    'agent.obs_groups.critic=[critic]'
  )
  printf '%s\n' "${cmd[@]}"
}

run_variant() {
  local variant="$1"
  local run_name="$2"
  local log_path="$3"
  local -a cmd=()
  mapfile -t cmd < <(build_base_cmd "$run_name")
  if [[ "$variant" == "strict" ]]; then
    cmd+=(
      env.tip_align_entry_m=0.12
      env.tip_align_exit_m=0.16
    )
  elif [[ "$variant" == "relaxed0175" ]]; then
    cmd+=(
      env.tip_align_entry_m=0.175
      env.tip_align_exit_m=0.21
    )
  else
    echo "[FATAL] unknown variant: $variant" >&2
    exit 2
  fi

  echo ""
  echo "[RUN] variant=$variant run_name=$run_name"
  echo "[RUN] log=$log_path"
  printf '[RUN] cmd:'
  printf ' %q' "${cmd[@]}"
  printf '\n'

  local train_pid
  local old_pwd="$PWD"
  cd "$ISAACLAB_DIR"
  nohup env TERM="$TERM" PYTHONUNBUFFERED=1 "${cmd[@]}" >"$log_path" 2>&1 &
  train_pid=$!
  echo "[RUN] pid=$train_pid"
  wait "$train_pid"
  cd "$old_pwd"

  echo "[DONE] variant=$variant finished"
}

run_variant "strict" "$STRICT_RUN_NAME" "$STRICT_LOG"
run_variant "relaxed0175" "$RELAXED_RUN_NAME" "$RELAXED_LOG"

python "$SUMMARY_SCRIPT" \
  --strict-log "$STRICT_LOG" \
  --relaxed-log "$RELAXED_LOG" \
  --output "$REPORT_PATH"

echo ""
echo "[DONE] tip-gate A/B finished."
echo "[DONE] report: $REPORT_PATH"
