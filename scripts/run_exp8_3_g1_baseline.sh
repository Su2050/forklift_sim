#!/usr/bin/env bash
# Exp8.3 G1 正式基线：只改 trajectory terminal geometry package 到 s_success_center，max_iterations=400。
# 用法：
#   bash scripts/run_exp8_3_g1_baseline.sh
# 行为：
#   - 先退出 conda，避免误用 base/python
#   - 采用与 G1 smoke / B0′ baseline 一致的 nohup + 后台启动方式
#   - 日志写入 logs/YYYYMMDD_HHMMSS_train_exp8_3_g1_traj_terminal_s_success_center_baseline.log

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ISAACLAB="${ISAACLAB:-$ROOT/IsaacLab}"
RUN_NAME="exp8_3_g1_traj_terminal_s_success_center_baseline"
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
echo "[INFO] CONDA_PREFIX=${CONDA_PREFIX:-<empty>}"

cd "$ISAACLAB"
nohup env TERM="$TERM" PYTHONUNBUFFERED=1 CONDA_PREFIX="" CONDA_DEFAULT_ENV="" \
  ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Forklift-PalletInsertLift-Direct-v0 \
  --headless \
  --enable_cameras \
  --num_envs 64 \
  --max_iterations 400 \
  agent.run_name="$RUN_NAME" \
  env.use_camera=true \
  env.use_asymmetric_critic=true \
  env.stage_1_mode=false \
  env.camera_width=256 \
  env.camera_height=256 \
  env.exp83_traj_goal_mode=success_center \
  agent.policy.class_name=rsl_rl.modules.VisionActorCritic \
  agent.policy.backbone_type=resnet34 \
  'agent.obs_groups.policy=[image, proprio]' \
  'agent.obs_groups.critic=[critic]' \
  agent.policy.imagenet_backbone_init=true \
  agent.policy.freeze_backbone=true \
  > "$LOG" 2>&1 &

echo "Started Exp8.3 G1 baseline training."
echo "log: $LOG"
echo "run_name: $RUN_NAME"
echo "pid: $!"
