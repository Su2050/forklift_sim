#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ISAACLAB_DIR="${ISAACLAB:-$ROOT/IsaacLab}"
LOG_DIR="$ROOT/logs"
SEEDS=(42 43 44)
WEIGHTS=(0.5 1.0 1.5)

mkdir -p "$LOG_DIR"
export TERM=xterm

if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook 2>/dev/null)" || true
  while [[ "${CONDA_SHLVL:-0}" =~ ^[0-9]+$ ]] && [[ "${CONDA_SHLVL:-0}" -gt 0 ]]; do
    conda deactivate 2>/dev/null || break
  done
fi
unset CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_SHLVL CONDA_PROMPT_MODIFIER CONDA_PYTHON_EXE _CE_CONDA _CE_M 2>/dev/null || true

echo "[INFO] IsaacLab: $ISAACLAB_DIR"
echo "[INFO] Weights: ${WEIGHTS[*]}"
echo "[INFO] Seeds: ${SEEDS[*]}"

bash "$ROOT/forklift_pallet_insert_lift_project/scripts/install_into_isaaclab.sh" "$ISAACLAB_DIR"

weight_tag() {
  local weight="$1"
  echo "${weight//./p}"
}

run_case() {
  local weight="$1"
  local seed="$2"
  local tag ts log run_name
  tag="$(weight_tag "$weight")"
  ts="$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)"
  run_name="exp83_bonusw${tag}_seed${seed}_iter50_256cam"
  log="$LOG_DIR/${ts}_train_${run_name}.log"

  echo "[START] weight=${weight} seed=${seed} run_name=${run_name} log=${log}"
  (
    cd "$ISAACLAB_DIR"
    exec env -u CONDA_PREFIX -u CONDA_DEFAULT_ENV -u CONDA_SHLVL \
      -u CONDA_PROMPT_MODIFIER -u CONDA_PYTHON_EXE -u _CE_CONDA -u _CE_M \
      TERM="$TERM" PYTHONUNBUFFERED=1 \
      ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
      --task Isaac-Forklift-PalletInsertLift-Direct-v0 \
      --headless \
      --enable_cameras \
      --seed "$seed" \
      --num_envs 64 \
      --max_iterations 50 \
      agent.run_name="$run_name" \
      env.use_camera=true \
      env.use_asymmetric_critic=true \
      env.stage_1_mode=true \
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
      env.clean_insert_push_free_bonus_weight="$weight" \
      agent.policy.class_name=rsl_rl.modules.VisionActorCritic \
      agent.policy.backbone_type=resnet34 \
      agent.policy.imagenet_backbone_init=true \
      agent.policy.freeze_backbone=true \
      'agent.obs_groups.policy=[image,proprio]' \
      'agent.obs_groups.critic=[critic]'
  ) >"$log" 2>&1
  echo "[DONE] weight=${weight} seed=${seed} log=${log}"
}

for weight in "${WEIGHTS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    run_case "$weight" "$seed"
  done
done

echo "[DONE] all bonus-weight sweep runs completed"
