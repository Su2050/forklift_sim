#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ISAACLAB_DIR="${ISAACLAB:-$ROOT/IsaacLab}"
LOG_DIR="$ROOT/logs"

SEEDS_STR="${EXP83_REPRO_SEEDS:-42 43 44}"
REPEAT_IDS_STR="${EXP83_REPRO_REPEAT_IDS:-r1 r2}"
# shellcheck disable=SC2206
SEEDS=($SEEDS_STR)
# shellcheck disable=SC2206
REPEAT_IDS=($REPEAT_IDS_STR)

MAX_ITERATIONS="${EXP83_REPRO_MAX_ITERATIONS:-100}"
NUM_ENVS="${EXP83_REPRO_NUM_ENVS:-64}"
CAMERA_WIDTH="${EXP83_REPRO_CAMERA_WIDTH:-256}"
CAMERA_HEIGHT="${EXP83_REPRO_CAMERA_HEIGHT:-256}"
BONUS_WEIGHT="${EXP83_REPRO_BONUS_WEIGHT:-1.0}"
RUN_TAG="${EXP83_REPRO_RUN_TAG:-exp83_bonusw1p0_repro}"

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
echo "[INFO] Run tag: $RUN_TAG"
echo "[INFO] Seeds: ${SEEDS[*]}"
echo "[INFO] Repeats: ${REPEAT_IDS[*]}"
echo "[INFO] Max iterations: $MAX_ITERATIONS"
echo "[INFO] Num envs: $NUM_ENVS"
echo "[INFO] Camera: ${CAMERA_WIDTH}x${CAMERA_HEIGHT}"
echo "[INFO] Bonus weight: $BONUS_WEIGHT"

bash "$ROOT/forklift_pallet_insert_lift_project/scripts/install_into_isaaclab.sh" "$ISAACLAB_DIR"

run_case() {
  local seed="$1"
  local repeat_id="$2"
  local ts log run_name

  ts="$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)"
  run_name="${RUN_TAG}_${repeat_id}_seed${seed}_iter${MAX_ITERATIONS}_256cam"
  log="$LOG_DIR/${ts}_train_${RUN_TAG}_${repeat_id}_seed${seed}.log"

  echo "[START] repeat=${repeat_id} seed=${seed} run_name=${run_name} log=${log}"
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
      --num_envs "$NUM_ENVS" \
      --max_iterations "$MAX_ITERATIONS" \
      agent.run_name="$run_name" \
      env.use_camera=true \
      env.use_asymmetric_critic=true \
      env.stage_1_mode=true \
      env.camera_width="$CAMERA_WIDTH" \
      env.camera_height="$CAMERA_HEIGHT" \
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
      env.clean_insert_push_free_bonus_weight="$BONUS_WEIGHT" \
      agent.policy.class_name=rsl_rl.modules.VisionActorCritic \
      agent.policy.backbone_type=resnet34 \
      agent.policy.imagenet_backbone_init=true \
      agent.policy.freeze_backbone=true \
      'agent.obs_groups.policy=[image,proprio]' \
      'agent.obs_groups.critic=[critic]'
  ) >"$log" 2>&1
  echo "[DONE] repeat=${repeat_id} seed=${seed} run_name=${run_name} log=${log}"
}

for repeat_id in "${REPEAT_IDS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    run_case "$seed" "$repeat_id"
  done
done

echo "[DONE] bonusw=1.0 repro batch completed"
