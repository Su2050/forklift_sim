#!/usr/bin/env bash
# Exp8.3 clean_insert_hold single-factor ablation batch
# Runs 5 near-field ablations sequentially to avoid GPU contention.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ISAACLAB="${ISAACLAB:-$ROOT/IsaacLab}"
MAX_ITERATIONS="${1:-50}"
NUM_ENVS="${2:-64}"
SEED="${SEED:-42}"
CAMERA_WIDTH="${CAMERA_WIDTH:-256}"
CAMERA_HEIGHT="${CAMERA_HEIGHT:-256}"

mkdir -p "$ROOT/logs"
export TERM=xterm

if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook 2>/dev/null)" || true
  while [[ "${CONDA_SHLVL:-0}" =~ ^[0-9]+$ ]] && [[ "${CONDA_SHLVL:-0}" -gt 0 ]]; do
    conda deactivate 2>/dev/null || break
  done
fi
unset CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_SHLVL CONDA_PROMPT_MODIFIER CONDA_PYTHON_EXE _CE_CONDA _CE_M 2>/dev/null || true

echo "[INFO] IsaacLab: $ISAACLAB"
echo "[INFO] MAX_ITERATIONS=$MAX_ITERATIONS"
echo "[INFO] NUM_ENVS=$NUM_ENVS"
echo "[INFO] SEED=$SEED"
echo "[INFO] CAMERA_WIDTH=$CAMERA_WIDTH"
echo "[INFO] CAMERA_HEIGHT=$CAMERA_HEIGHT"

bash "$ROOT/forklift_pallet_insert_lift_project/scripts/install_into_isaaclab.sh" "$ISAACLAB"

run_one() {
  local name="$1"
  shift

  local ts log
  ts="$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)"
  log="$ROOT/logs/${ts}_train_${name}.log"

  echo
  echo "============================================================"
  echo "[INFO] Starting $name"
  echo "[INFO] Log: $log"
  echo "============================================================"

  (
    cd "$ISAACLAB"
    exec env -u CONDA_PREFIX -u CONDA_DEFAULT_ENV -u CONDA_SHLVL -u CONDA_PROMPT_MODIFIER -u CONDA_PYTHON_EXE -u _CE_CONDA -u _CE_M \
      TERM="$TERM" PYTHONUNBUFFERED=1 \
      ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
      --task Isaac-Forklift-PalletInsertLift-Direct-v0 \
      --headless \
      --enable_cameras \
      --seed "$SEED" \
      --num_envs "$NUM_ENVS" \
      --max_iterations "$MAX_ITERATIONS" \
      agent.run_name="$name" \
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
      "$@"
  ) >"$log" 2>&1

  echo "[INFO] Finished $name"
  echo "[INFO] Log saved to $log"
}

run_one exp83_ablate_B0_baseline \
  env.clean_insert_gate_r_cd=false \
  env.clean_insert_gate_r_cpsi=false \
  env.clean_insert_dirty_penalty_enable=false \
  env.clean_insert_gate_floor=0.15 \
  env.clean_insert_center_sigma_m=0.10 \
  env.clean_insert_yaw_sigma_deg=6.0 \
  env.clean_insert_tip_sigma_m=0.10 \
  env.clean_insert_push_sigma_m=0.10

run_one exp83_ablate_F1_gate_r_cd \
  env.clean_insert_gate_r_cd=true \
  env.clean_insert_gate_r_cpsi=false \
  env.clean_insert_dirty_penalty_enable=false \
  env.clean_insert_gate_floor=0.15 \
  env.clean_insert_center_sigma_m=0.10 \
  env.clean_insert_yaw_sigma_deg=6.0 \
  env.clean_insert_tip_sigma_m=0.10 \
  env.clean_insert_push_sigma_m=0.10

run_one exp83_ablate_F2_gate_r_cpsi \
  env.clean_insert_gate_r_cd=false \
  env.clean_insert_gate_r_cpsi=true \
  env.clean_insert_dirty_penalty_enable=false \
  env.clean_insert_gate_floor=0.15 \
  env.clean_insert_center_sigma_m=0.10 \
  env.clean_insert_yaw_sigma_deg=6.0 \
  env.clean_insert_tip_sigma_m=0.10 \
  env.clean_insert_push_sigma_m=0.10

run_one exp83_ablate_F3_dirty_penalty \
  env.clean_insert_gate_r_cd=false \
  env.clean_insert_gate_r_cpsi=false \
  env.clean_insert_dirty_penalty_enable=true \
  env.clean_insert_dirty_penalty_weight=8.0 \
  env.clean_insert_gate_floor=0.15 \
  env.clean_insert_center_sigma_m=0.10 \
  env.clean_insert_yaw_sigma_deg=6.0 \
  env.clean_insert_tip_sigma_m=0.10 \
  env.clean_insert_push_sigma_m=0.10

run_one exp83_ablate_F4_tight_gate_package \
  env.clean_insert_gate_r_cd=false \
  env.clean_insert_gate_r_cpsi=false \
  env.clean_insert_dirty_penalty_enable=false \
  env.clean_insert_gate_floor=0.05 \
  env.clean_insert_center_sigma_m=0.08 \
  env.clean_insert_yaw_sigma_deg=5.0 \
  env.clean_insert_tip_sigma_m=0.08 \
  env.clean_insert_push_sigma_m=0.06

echo
echo "[DONE] All 5 ablations finished."
