#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/uniubi/projects/forklift_sim"
ISAACLAB_DIR="${PROJECT_ROOT}/IsaacLab"
CHECKPOINT="/home/uniubi/projects/forklift_sim/IsaacLab/logs/rsl_rl/forklift_pallet_insert_lift/2026-03-14_21-46-38_exp5_9_nuclear_reward_early_stop_0.28/model_836.pt"

# 如果 836 不存在，尝试找 835
if [ ! -f "$CHECKPOINT" ]; then
    CHECKPOINT="/home/uniubi/projects/forklift_sim/IsaacLab/logs/rsl_rl/forklift_pallet_insert_lift/2026-03-14_21-46-38_exp5_9_nuclear_reward_early_stop_0.28/model_835.pt"
fi

echo "Using checkpoint: $CHECKPOINT"

cd "${ISAACLAB_DIR}"

# 使用统一的录制脚本，生成 global 和 camera 两个视角的视频
env TERM=xterm PYTHONUNBUFFERED=1 CONDA_PREFIX="" CONDA_DEFAULT_ENV="" \
  ./isaaclab.sh -p ../scripts/experiments/play_and_record.py \
  --task Isaac-Forklift-PalletInsertLift-Direct-v0 \
  --num_envs 1 \
  --checkpoint "${CHECKPOINT}" \
  --headless \
  --video_length 800 \
  --video_folder exp5_9_perfect_insertion \
  --view_mode both \
  agent.run_name="play_exp5_9" \
  env.use_camera=true \
  env.use_asymmetric_critic=true \
  env.stage_1_mode=true \
  env.camera_width=256 \
  env.camera_height=256 \
  agent.policy.class_name=rsl_rl.modules.VisionActorCritic \
  agent.obs_groups.policy='[image, proprio]' \
  agent.policy.imagenet_backbone_init=true \
  agent.policy.freeze_backbone=true

echo "Video generation complete."
