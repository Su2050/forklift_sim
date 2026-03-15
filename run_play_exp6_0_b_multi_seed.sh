#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/uniubi/projects/forklift_sim"
ISAACLAB_DIR="${PROJECT_ROOT}/IsaacLab"
CHECKPOINT="/home/uniubi/projects/forklift_sim/IsaacLab/logs/rsl_rl/forklift_pallet_insert_lift/2026-03-15_07-52-04_exp6_0_b_micro_generalization_stage1/model_1450.pt"

echo "Using checkpoint: $CHECKPOINT"

cd "${ISAACLAB_DIR}"

SEEDS=(100 200 300)

for SEED in "${SEEDS[@]}"; do
    echo "=================================================="
    echo "Running with seed: $SEED"
    echo "=================================================="
    
    # 使用统一的录制脚本，生成 global 和 camera 两个视角的视频
    env TERM=xterm PYTHONUNBUFFERED=1 CONDA_PREFIX="" CONDA_DEFAULT_ENV="" \
      ./isaaclab.sh -p ../scripts/experiments/play_and_record.py \
      --task Isaac-Forklift-PalletInsertLift-Direct-v0 \
      --num_envs 1 \
      --checkpoint "${CHECKPOINT}" \
      --headless \
      --video_length 800 \
      --video_folder "exp6_0_b_best_1450_seed_${SEED}" \
      --view_mode both \
      --seed "${SEED}" \
      agent.run_name="play_exp6_0_b_seed_${SEED}" \
      env.use_camera=true \
      env.use_asymmetric_critic=true \
      env.stage_1_mode=true \
      env.camera_width=256 \
      env.camera_height=256 \
      agent.policy.class_name=rsl_rl.modules.VisionActorCritic \
      agent.obs_groups.policy='[image, proprio]' \
      agent.policy.imagenet_backbone_init=true \
      agent.policy.freeze_backbone=true
done

echo "Multi-seed video generation complete."
