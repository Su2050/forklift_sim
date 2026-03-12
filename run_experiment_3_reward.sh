#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/uniubi/projects/forklift_sim"
ISAACLAB_DIR="${PROJECT_ROOT}/IsaacLab"
LOG_TYPE="train"
# 集中算力，环境数从 64 增加到 128
NUM_ENVS="128"
MAX_ITERATIONS="2000"
RUN_NAME="exp3_reward_shaping_rrl"

mkdir -p "${PROJECT_ROOT}/logs"
BEIJING_TS="$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)"
LOG_FILE="${PROJECT_ROOT}/logs/${BEIJING_TS}_${LOG_TYPE}_${RUN_NAME}.log"

cd "${ISAACLAB_DIR}"
nohup env TERM=xterm PYTHONUNBUFFERED=1 CONDA_PREFIX="" CONDA_DEFAULT_ENV="" \
  ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Forklift-PalletInsertLift-Direct-v0 \
  --headless --enable_cameras --num_envs "${NUM_ENVS}" --max_iterations "${MAX_ITERATIONS}" \
  agent.run_name="${RUN_NAME}" \
  env.use_camera=true \
  env.use_asymmetric_critic=true \
  env.stage_1_mode=true \
  env.camera_width=256 \
  env.camera_height=256 \
  agent.policy.class_name=rsl_rl.modules.VisionActorCritic \
  agent.obs_groups.policy='[image, proprio]' \
  agent.obs_groups.critic='[critic]' \
  agent.policy.imagenet_backbone_init=true \
  agent.policy.freeze_backbone=true \
  > "${LOG_FILE}" 2>&1 &

echo "Started Experiment 3 (Reward Shaping + RRL) training."
echo "log: ${LOG_FILE}"
echo "run_name: ${RUN_NAME}"
