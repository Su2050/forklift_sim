#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
ISAACLAB_DIR="${PROJECT_ROOT}/IsaacLab"

cd "${ISAACLAB_DIR}"

# 运行 camera_eval 脚本，使用 75度 俯仰角，将托盘生成在货叉根部 (x=0.9)
env TERM=xterm PYTHONUNBUFFERED=1 CONDA_PREFIX="" CONDA_DEFAULT_ENV="" \
  ./isaaclab.sh -p ../scripts/validation/observations/camera_eval.py \
  --enable_cameras \
  --cam-name "test_75deg" \
  --cam-x 130.0 \
  --cam-y 0.0 \
  --cam-z 250.0 \
  --pitch-deg 75.0 \
  --steps 100
