#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/uniubi/projects/forklift_sim"
ISAACLAB_DIR="${PROJECT_ROOT}/IsaacLab"
RUN_NAME="exp5_9_nuclear_reward_early_stop_0.28"

# 找到最新的 5.9 目录
RUN_DIR=$(ls -td ${ISAACLAB_DIR}/logs/rsl_rl/forklift_pallet_insert_lift/*${RUN_NAME}* | head -n 1)
RUN_DIR_NAME=$(basename ${RUN_DIR})

echo "Waiting for training to finish and save model_836.pt in ${RUN_DIR}..."

# 等待 train.py 进程结束
while pgrep -f "train.py.*${RUN_NAME}" > /dev/null; do
    sleep 5
done

echo "Training finished. Checking for saved models..."
ls -l ${RUN_DIR}/*.pt

echo "Generating video..."
cd "${ISAACLAB_DIR}"
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
  --task Isaac-Forklift-PalletInsertLift-Direct-v0 \
  --num_envs 16 \
  --video --video_length 400 --video_dir "${PROJECT_ROOT}/outputs/exp5_9_video" \
  --load_run "${RUN_DIR_NAME}" \
  agent.run_name="play_${RUN_NAME}" \
  env.use_camera=true \
  env.use_asymmetric_critic=true \
  env.stage_1_mode=true \
  env.camera_width=256 \
  env.camera_height=256 \
  agent.policy.class_name=rsl_rl.modules.VisionActorCritic \
  agent.obs_groups.policy='[image, proprio]' \
  agent.policy.imagenet_backbone_init=true \
  agent.policy.freeze_backbone=true

echo "Video generation complete. Output in ${PROJECT_ROOT}/outputs/exp5_9_video"
