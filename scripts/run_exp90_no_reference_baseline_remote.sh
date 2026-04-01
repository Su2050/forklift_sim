#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ISAACLAB_DIR="${ISAACLAB_DIR:-/home/uniubi/projects/forklift_sim/IsaacLab}"
ISAACLAB_PARENT="$(cd "${ISAACLAB_DIR}/.." && pwd)"
ASSET_DST_DIR="${ASSET_DST_DIR:-${ISAACLAB_PARENT}/assets}"
CONDA_BASE="${CONDA_BASE:-/home/uniubi/miniconda3}"
CONDA_ENV_PATH="${CONDA_ENV_PATH:-${CONDA_BASE}/envs/env_isaaclab}"
LOG_DIR="${LOG_DIR:-$ROOT/logs}"
SEED="${SEED:-42}"
NUM_ENVS="${NUM_ENVS:-64}"
MAX_ITERATIONS="${MAX_ITERATIONS:-400}"
RUN_NAME="${RUN_NAME:-exp9_0_no_reference_master_init_seed${SEED}_iter${MAX_ITERATIONS}}"
DRY_RUN="${DRY_RUN:-0}"

if [[ ! -f "${CONDA_BASE}/bin/activate" ]]; then
  echo "[FATAL] missing conda activate script: ${CONDA_BASE}/bin/activate" >&2
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
if [[ ! -d "${ISAACLAB_DIR}/source/isaaclab_tasks/isaaclab_tasks" ]]; then
  echo "[FATAL] invalid IsaacLab checkout: ${ISAACLAB_DIR}" >&2
  exit 2
fi
if [[ ! -d "${ROOT}/assets" ]]; then
  echo "[FATAL] missing source assets directory: ${ROOT}/assets" >&2
  exit 2
fi

# Activate the known-good remote runtime instead of relying on the login shell.
# This host currently runs IsaacLab via the conda env, not via _isaac_sim/python.sh.
# shellcheck disable=SC1091
source "${CONDA_BASE}/bin/activate" "${CONDA_ENV_PATH}"

python - <<'PY'
import importlib.metadata as metadata
import isaaclab
import isaacsim

print("[INFO] Python runtime precheck passed.")
print(f"[INFO] rsl-rl-lib={metadata.version('rsl-rl-lib')}")
PY

mkdir -p "$LOG_DIR"
if [[ -z "${TERM:-}" || "${TERM}" == "dumb" ]]; then
  export TERM="xterm"
fi

mkdir -p "${ASSET_DST_DIR}"
rsync -a "${ROOT}/assets/" "${ASSET_DST_DIR}/"
echo "[INFO] Synced assets -> ${ASSET_DST_DIR}"

bash "$ROOT/forklift_pallet_insert_lift_project/scripts/install_into_isaaclab.sh" "$ISAACLAB_DIR"

TS="$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)"
LOG="$LOG_DIR/${TS}_train_${RUN_NAME}.log"

CMD=(
  ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py
  --task Isaac-Forklift-PalletInsertLift-Direct-v0
  --headless
  --enable_cameras
  --seed "$SEED"
  --num_envs "$NUM_ENVS"
  --max_iterations "$MAX_ITERATIONS"
  agent.run_name="$RUN_NAME"
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
  env.clean_insert_gate_floor=0.15
  env.clean_insert_center_sigma_m=0.10
  env.clean_insert_yaw_sigma_deg=6.0
  env.clean_insert_tip_sigma_m=0.10
  env.clean_insert_push_sigma_m=0.10
  env.clean_insert_gate_r_cd=false
  env.clean_insert_gate_r_cpsi=false
  env.clean_insert_dirty_penalty_enable=false
  env.clean_insert_push_free_bonus_enable=true
  env.clean_insert_push_free_bonus_weight=1.0
  env.preinsert_align_reward_enable=true
  agent.policy.class_name=rsl_rl.modules.VisionActorCritic
  agent.policy.backbone_type=resnet34
  agent.policy.imagenet_backbone_init=true
  agent.policy.freeze_backbone=true
  'agent.obs_groups.policy=[image,proprio]'
  'agent.obs_groups.critic=[critic]'
)

echo "[INFO] Repo root: $ROOT"
echo "[INFO] IsaacLab: $ISAACLAB_DIR"
echo "[INFO] Asset dir: $ASSET_DST_DIR"
echo "[INFO] Conda env: $CONDA_ENV_PATH"
echo "[INFO] Log: $LOG"
echo "[INFO] Seed: $SEED"
echo "[INFO] Num envs: $NUM_ENVS"
echo "[INFO] Max iterations: $MAX_ITERATIONS"
echo "[INFO] Run name: $RUN_NAME"

if [[ "$DRY_RUN" == "1" ]]; then
  echo "[DRY-RUN] Working directory: $ISAACLAB_DIR"
  printf '[DRY-RUN] Command:'
  printf ' %q' "${CMD[@]}"
  printf '\n'
  exit 0
fi

cd "$ISAACLAB_DIR"
nohup env TERM="$TERM" PYTHONUNBUFFERED=1 \
  "${CMD[@]}" >"$LOG" 2>&1 &

echo "Started Exp9.0 remote no-reference baseline."
echo "log: $LOG"
echo "run_name: $RUN_NAME"
echo "pid: $!"
