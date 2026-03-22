#!/usr/bin/env bash
# Exp8.3 runtime U0 真实 env 路径 sanity（一次性 headless 检查）
# 用法：
#   bash scripts/run_exp8_3_runtime_u0_sanity.sh [b0prime|g2b|g3]
# 默认：b0prime

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ISAACLAB="${ISAACLAB:-/home/uniubi/projects/forklift_sim/IsaacLab}"
PROFILE="${1:-b0prime}"
LOG_TYPE="sanity_check"
RUN_NAME="exp8_3_runtime_u0_${PROFILE}"
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

extra_overrides=()
case "$PROFILE" in
  b0prime)
    extra_overrides+=(env.exp83_traj_goal_mode=front)
    ;;
  g2b)
    extra_overrides+=(env.exp83_traj_goal_mode=front)
    extra_overrides+=(env.exp83_target_center_family_mode=success_center)
    ;;
  g3)
    extra_overrides+=(env.exp83_traj_goal_mode=success_center)
    extra_overrides+=(env.exp83_target_center_family_mode=success_center)
    ;;
  *)
    echo "[ERROR] Unsupported profile: $PROFILE" >&2
    exit 1
    ;;
esac

echo "[INFO] runtime_u0 profile: $PROFILE"
echo "[INFO] log: $LOG"
echo "[INFO] ISAACLAB: $ISAACLAB"

bash "$ROOT/forklift_pallet_insert_lift_project/scripts/install_into_isaaclab.sh" "$ISAACLAB"

cd "$ISAACLAB"
nohup env TERM="$TERM" PYTHONUNBUFFERED=1 CONDA_PREFIX="" CONDA_DEFAULT_ENV="" \
  ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Forklift-PalletInsertLift-Direct-v0 \
  --headless \
  --enable_cameras \
  --num_envs 32 \
  --max_iterations 2 \
  agent.run_name="$RUN_NAME" \
  env.use_camera=true \
  env.use_asymmetric_critic=true \
  env.stage_1_mode=false \
  env.camera_width=256 \
  env.camera_height=256 \
  env.exp83_runtime_u0_enable=true \
  env.exp83_runtime_u0_fail_fast=true \
  env.exp83_runtime_u0_eps_pos_m=0.001 \
  env.exp83_runtime_u0_eps_yaw_deg=15.0 \
  agent.policy.class_name=rsl_rl.modules.VisionActorCritic \
  agent.policy.backbone_type=resnet34 \
  'agent.obs_groups.policy=[image, proprio]' \
  'agent.obs_groups.critic=[critic]' \
  agent.policy.imagenet_backbone_init=true \
  agent.policy.freeze_backbone=true \
  "${extra_overrides[@]}" \
  > "$LOG" 2>&1 &

echo "Started Exp8.3 runtime U0 sanity."
echo "profile: $PROFILE"
echo "log: $LOG"
echo "pid: $!"
