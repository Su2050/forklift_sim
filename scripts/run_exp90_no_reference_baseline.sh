#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ISAACLAB_DIR="${ISAACLAB:-$ROOT/IsaacLab}"
LOG_DIR="$ROOT/logs"
SEED="${SEED:-42}"
NUM_ENVS="${NUM_ENVS:-64}"
MAX_ITERATIONS="${MAX_ITERATIONS:-400}"
RUN_NAME="${RUN_NAME:-exp9_0_no_reference_master_init_seed${SEED}_iter${MAX_ITERATIONS}}"

mkdir -p "$LOG_DIR"
export TERM=xterm

if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook 2>/dev/null)" || true
  while [[ "${CONDA_SHLVL:-0}" =~ ^[0-9]+$ ]] && [[ "${CONDA_SHLVL:-0}" -gt 0 ]]; do
    conda deactivate 2>/dev/null || break
  done
fi
unset CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_SHLVL CONDA_PROMPT_MODIFIER CONDA_PYTHON_EXE _CE_CONDA _CE_M 2>/dev/null || true

bash "$ROOT/forklift_pallet_insert_lift_project/scripts/install_into_isaaclab.sh" "$ISAACLAB_DIR"

TS="$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)"
LOG="$LOG_DIR/${TS}_train_${RUN_NAME}.log"

echo "[INFO] IsaacLab: $ISAACLAB_DIR"
echo "[INFO] Log: $LOG"
echo "[INFO] Seed: $SEED"
echo "[INFO] Num envs: $NUM_ENVS"
echo "[INFO] Max iterations: $MAX_ITERATIONS"
echo "[INFO] Run name: $RUN_NAME"

cd "$ISAACLAB_DIR"
nohup env -u CONDA_PREFIX -u CONDA_DEFAULT_ENV -u CONDA_SHLVL \
  -u CONDA_PROMPT_MODIFIER -u CONDA_PYTHON_EXE -u _CE_CONDA -u _CE_M \
  TERM="$TERM" PYTHONUNBUFFERED=1 \
  ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Forklift-PalletInsertLift-Direct-v0 \
  --headless \
  --enable_cameras \
  --seed "$SEED" \
  --num_envs "$NUM_ENVS" \
  --max_iterations "$MAX_ITERATIONS" \
  agent.run_name="$RUN_NAME" \
  env.use_camera=true \
  env.use_asymmetric_critic=true \
  env.stage_1_mode=true \
  env.use_reference_trajectory=false \
  env.alpha_2=0.0 \
  env.alpha_3=0.0 \
  env.camera_width=256 \
  env.camera_height=256 \
  env.clean_insert_reward_gate_enable=true \
  env.clean_insert_use_push_gate=true \
  env.clean_insert_gate_floor=0.15 \
  env.clean_insert_center_sigma_m=0.10 \
  env.clean_insert_yaw_sigma_deg=6.0 \
  env.clean_insert_tip_sigma_m=0.10 \
  env.clean_insert_push_sigma_m=0.10 \
  env.clean_insert_gate_r_cd=false \
  env.clean_insert_gate_r_cpsi=false \
  env.clean_insert_dirty_penalty_enable=false \
  env.clean_insert_push_free_bonus_enable=true \
  env.clean_insert_push_free_bonus_weight=1.0 \
  env.preinsert_align_reward_enable=true \
  agent.policy.class_name=rsl_rl.modules.VisionActorCritic \
  agent.policy.backbone_type=resnet34 \
  agent.policy.imagenet_backbone_init=true \
  agent.policy.freeze_backbone=true \
  'agent.obs_groups.policy=[image,proprio]' \
  'agent.obs_groups.critic=[critic]' \
  >"$LOG" 2>&1 &

echo "Started Exp9.0 no-reference baseline."
echo "log: $LOG"
echo "run_name: $RUN_NAME"
echo "pid: $!"
