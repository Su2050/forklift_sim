#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/uniubi/projects/forklift_sim"
ISAACLAB_DIR="${PROJECT_ROOT}/IsaacLab"
RUN_NAME="branch_b_paper_native_inv_reward_fix_oob"
LOG_TYPE="train"

mkdir -p "${PROJECT_ROOT}/logs"
BEIJING_TS="$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)"
LOG_FILE="${PROJECT_ROOT}/logs/${BEIJING_TS}_${LOG_TYPE}_${RUN_NAME}.log"

cd "${ISAACLAB_DIR}"

# 运行训练脚本 (注意：不加 --resume，从头开始训练)
# 设置 max_iterations=2000，因为是从头训练，需要较长的时间
nohup env TERM=xterm PYTHONUNBUFFERED=1 CONDA_PREFIX="" CONDA_DEFAULT_ENV="" \
  ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Forklift-PalletInsertLift-Direct-v0 \
  --headless \
  --enable_cameras \
  --num_envs 64 \
  --max_iterations 2000 \
  agent.run_name="${RUN_NAME}" \
  env.use_camera=true \
  env.use_asymmetric_critic=true \
  env.stage_1_mode=false \
  env.camera_width=256 \
  env.camera_height=256 \
  agent.policy.class_name=rsl_rl.modules.VisionActorCritic \
  agent.policy.backbone_type="resnet34" \
  agent.obs_groups.policy='[image, proprio]' \
  agent.obs_groups.critic='[critic]' \
  agent.policy.imagenet_backbone_init=true \
  agent.policy.freeze_backbone=true \
  > "${LOG_FILE}" 2>&1 &

echo "Started Branch B (Paper Native) training."
echo "log: ${LOG_FILE}"
echo "run_name: ${RUN_NAME}"
