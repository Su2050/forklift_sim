#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/uniubi/projects/forklift_sim"
ISAACLAB_DIR="${1:-${PROJECT_ROOT}/IsaacLab}"
VERSION="${2:-s1.0vision}"
LOG_TYPE="${3:-train}"
NUM_ENVS="${NUM_ENVS:-1024}"
MAX_ITERATIONS="${MAX_ITERATIONS:-2000}"
RUN_NAME="${RUN_NAME:-vision_stage1_${VERSION}}"

if [[ ! -d "${ISAACLAB_DIR}/source/isaaclab_tasks/isaaclab_tasks" ]]; then
  echo "Error: ${ISAACLAB_DIR} is not a valid IsaacLab checkout"
  exit 1
fi

mkdir -p "${PROJECT_ROOT}/logs"
BEIJING_TS="$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)"
LOG_FILE="${PROJECT_ROOT}/logs/${BEIJING_TS}_${LOG_TYPE}_${VERSION}.log"

bash "${PROJECT_ROOT}/forklift_pallet_insert_lift_project/scripts/install_into_isaaclab.sh" "${ISAACLAB_DIR}"

cd "${ISAACLAB_DIR}"
nohup env TERM=xterm PYTHONUNBUFFERED=1 CONDA_PREFIX="" CONDA_DEFAULT_ENV="" \
  ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Forklift-PalletInsertLift-Direct-v0 \
  --headless --enable_cameras --num_envs "${NUM_ENVS}" --max_iterations "${MAX_ITERATIONS}" \
  agent.run_name="${RUN_NAME}" \
  env.use_camera=true \
  env.use_asymmetric_critic=true \
  env.stage_1_mode=true \
  env.camera_width=64 \
  env.camera_height=64 \
  agent.policy.class_name=rsl_rl.modules.VisionActorCritic \
  agent.obs_groups.policy='[image, proprio]' \
  agent.obs_groups.critic='[critic]' \
  > "${LOG_FILE}" 2>&1 &

echo "Started vision Stage-1 training."
echo "log: ${LOG_FILE}"
echo "run_name: ${RUN_NAME}"
