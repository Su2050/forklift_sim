#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/uniubi/projects/forklift_sim"
ISAACLAB_DIR="${PROJECT_ROOT}/IsaacLab"

cd "${ISAACLAB_DIR}"

# 我们随便传一个存在的 checkpoint，但是因为环境改变了，它会表现得像随机策略
# 我们的目的只是看相机视角，所以策略乱动也没关系
CHECKPOINT="/home/uniubi/projects/forklift_sim/IsaacLab/logs/rsl_rl/forklift_pallet_insert_lift/2026-03-15_07-52-04_exp6_0_b_micro_generalization_stage1/model_1450.pt"

env TERM=xterm PYTHONUNBUFFERED=1 CONDA_PREFIX="" CONDA_DEFAULT_ENV="" \
  ./isaaclab.sh -p ../scripts/experiments/play_and_record.py \
  --task Isaac-Forklift-PalletInsertLift-Direct-v0 \
  --num_envs 1 \
  --checkpoint "${CHECKPOINT}" \
  --headless \
  --video_length 200 \
  --video_folder test_75deg_dynamic \
  --view_mode camera \
  agent.run_name="test_75deg_dynamic" \
  env.use_camera=true \
  env.camera_width=256 \
  env.camera_height=256 \
  env.stage1_init_y_min_m=-0.0 \
  env.stage1_init_y_max_m=0.0 \
  env.stage1_init_yaw_deg_min=-0.0 \
  env.stage1_init_yaw_deg_max=0.0 \
  env.stage1_init_x_min_m=2.0 \
  env.stage1_init_x_max_m=2.0
