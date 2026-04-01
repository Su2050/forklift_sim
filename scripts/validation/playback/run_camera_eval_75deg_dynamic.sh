#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
ISAACLAB_DIR="${PROJECT_ROOT}/IsaacLab"

cd "${ISAACLAB_DIR}"

CHECKPOINT="${1:-${CHECKPOINT:-}}"

if [[ -z "${CHECKPOINT}" ]]; then
  echo "Usage: $0 /abs/path/to/checkpoint.pt"
  echo "Or set CHECKPOINT=/abs/path/to/checkpoint.pt"
  exit 1
fi

env TERM=xterm PYTHONUNBUFFERED=1 CONDA_PREFIX="" CONDA_DEFAULT_ENV="" \
  ./isaaclab.sh -p ../scripts/validation/playback/play_and_record.py \
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
