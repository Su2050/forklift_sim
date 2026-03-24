#!/usr/bin/env bash
# Exp8.3 clean_insert_hold near-field smoke:
# - 从 Phase 1 ready 基线出发
# - 使用 stage_1_mode=true 的近场 reset
# - 重点观察 clean insert / hold / anti-push 指标

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ISAACLAB="${ISAACLAB:-$ROOT/IsaacLab}"
MAX_ITERATIONS="${1:-50}"
NUM_ENVS="${2:-64}"
CAMERA_WIDTH="${CAMERA_WIDTH:-256}"
CAMERA_HEIGHT="${CAMERA_HEIGHT:-256}"
RUN_NAME="${RUN_NAME:-exp8_3_clean_insert_hold_nearfield_iter${MAX_ITERATIONS}}"
LOG_TYPE="train"
BEIJING_TS="$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)"
LOG="$ROOT/logs/${BEIJING_TS}_${LOG_TYPE}_${RUN_NAME}.log"
mkdir -p "$ROOT/logs"

export TERM=xterm

if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook 2>/dev/null)" || true
  while [[ "${CONDA_SHLVL:-0}" =~ ^[0-9]+$ ]] && [[ "${CONDA_SHLVL:-0}" -gt 0 ]]; do
    conda deactivate 2>/dev/null || break
  done
fi
unset CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_SHLVL CONDA_PROMPT_MODIFIER CONDA_PYTHON_EXE _CE_CONDA _CE_M 2>/dev/null || true

echo "[INFO] Log: $LOG"
echo "[INFO] IsaacLab: $ISAACLAB"
echo "[INFO] TERM=$TERM"
echo "[INFO] MAX_ITERATIONS=$MAX_ITERATIONS"
echo "[INFO] NUM_ENVS=$NUM_ENVS"
echo "[INFO] CAMERA_WIDTH=$CAMERA_WIDTH"
echo "[INFO] CAMERA_HEIGHT=$CAMERA_HEIGHT"

bash "$ROOT/forklift_pallet_insert_lift_project/scripts/install_into_isaaclab.sh" "$ISAACLAB"

cd "$ISAACLAB"
nohup env TERM="$TERM" PYTHONUNBUFFERED=1 CONDA_PREFIX="" CONDA_DEFAULT_ENV="" \
  ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Forklift-PalletInsertLift-Direct-v0 \
  --headless \
  --enable_cameras \
  --num_envs "$NUM_ENVS" \
  --max_iterations "$MAX_ITERATIONS" \
  agent.run_name="$RUN_NAME" \
  env.use_camera=true \
  env.use_asymmetric_critic=true \
  env.stage_1_mode=true \
  env.camera_width="$CAMERA_WIDTH" \
  env.camera_height="$CAMERA_HEIGHT" \
  env.clean_insert_reward_gate_enable=true \
  env.clean_insert_use_push_gate=true \
  agent.policy.class_name=rsl_rl.modules.VisionActorCritic \
  agent.policy.backbone_type=resnet34 \
  agent.policy.imagenet_backbone_init=true \
  agent.policy.freeze_backbone=true \
  'agent.obs_groups.policy=[image,proprio]' \
  'agent.obs_groups.critic=[critic]' \
  > "$LOG" 2>&1 &

echo "Started Exp8.3 clean_insert_hold near-field smoke."
echo "log: $LOG"
echo "run_name: $RUN_NAME"
echo "pid: $!"
