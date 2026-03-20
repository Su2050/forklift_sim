#!/usr/bin/env bash
# Exp8.3 B0′ smoke：与 B0（20260319_215340）CLI 对齐，仅缩短 max_iterations。
# 用法：在已配置 IsaacLab 的终端执行（需能 import isaaclab）：
#   bash scripts/run_exp8_3_b0prime_smoke.sh
# 日志：logs/YYYYMMDD_HHMMSS_train_exp8_3_b0prime_smoke.log

set -euo pipefail

# nohup / Cursor / CI 常见 TERM=dumb，会触发 `tabs: terminal type 'dumb' cannot reset tabs`
# 并干扰 isaaclab.sh 内部分工具；非 dumb 时保留用户终端类型。
if [[ -z "${TERM:-}" || "${TERM}" == "dumb" ]]; then
  export TERM=xterm-256color
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ISAACLAB="${ISAACLAB:-$ROOT/IsaacLab}"
LOG="$ROOT/logs/$(date +%Y%m%d_%H%M%S)_train_exp8_3_b0prime_smoke.log"
mkdir -p "$ROOT/logs"

echo "[INFO] Log: $LOG"
echo "[INFO] IsaacLab: $ISAACLAB"

# 必须退出 conda：`isaaclab.sh` 在存在 CONDA_PREFIX 时会优先用 conda 里的 python；
# 常见 base 环境没有 isaaclab → ModuleNotFoundError。本 smoke 依赖 Isaac Sim Kit 的 `_isaac_sim/python.sh`。
if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook 2>/dev/null)" || true
  while [[ "${CONDA_SHLVL:-0}" =~ ^[0-9]+$ ]] && [[ "${CONDA_SHLVL:-0}" -gt 0 ]]; do
    conda deactivate 2>/dev/null || break
  done
fi
unset CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_SHLVL CONDA_PROMPT_MODIFIER CONDA_PYTHON_EXE _CE_CONDA _CE_M 2>/dev/null || true
echo "[INFO] 已清理 conda 环境变量（CONDA_PREFIX=${CONDA_PREFIX:-<空>}）"

cd "$ISAACLAB"
bash isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Forklift-PalletInsertLift-Direct-v0 \
  --headless \
  --enable_cameras \
  --num_envs 64 \
  --max_iterations 80 \
  agent.run_name=exp8_3_b0prime_smoke \
  env.use_camera=true \
  env.use_asymmetric_critic=true \
  env.stage_1_mode=false \
  env.camera_width=256 \
  env.camera_height=256 \
  agent.policy.class_name=rsl_rl.modules.VisionActorCritic \
  agent.policy.backbone_type=resnet34 \
  'agent.obs_groups.policy=[image, proprio]' \
  'agent.obs_groups.critic=[critic]' \
  agent.policy.imagenet_backbone_init=true \
  agent.policy.freeze_backbone=true \
  2>&1 | tee "$LOG"
