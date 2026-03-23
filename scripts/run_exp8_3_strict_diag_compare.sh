#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ISAACLAB="${ISAACLAB:-$ROOT/IsaacLab}"
PROFILE="${1:-all}"
MAX_ITERATIONS="${2:-200}"
NUM_ENVS="${3:-64}"
RUN_MODE="${4:-background}"
LOG_TYPE="train"

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
echo "[INFO] PROFILE=$PROFILE"
echo "[INFO] MAX_ITERATIONS=$MAX_ITERATIONS"
echo "[INFO] NUM_ENVS=$NUM_ENVS"
echo "[INFO] RUN_MODE=$RUN_MODE"
echo "[INFO] TERM=$TERM"

bash "$ROOT/forklift_pallet_insert_lift_project/scripts/install_into_isaaclab.sh" "$ISAACLAB"

cd "$ISAACLAB"

run_profile() {
  local profile="$1"
  local traj_mode=""
  local target_center_family_mode=""
  local run_name=""

  case "$profile" in
    b0prime)
      traj_mode="front"
      target_center_family_mode="front_center"
      run_name="exp8_3_strict_diag_compare_b0prime_iter${MAX_ITERATIONS}"
      ;;
    g2b)
      traj_mode="front"
      target_center_family_mode="success_center"
      run_name="exp8_3_strict_diag_compare_g2b_iter${MAX_ITERATIONS}"
      ;;
    g3)
      traj_mode="success_center"
      target_center_family_mode="success_center"
      run_name="exp8_3_strict_diag_compare_g3_iter${MAX_ITERATIONS}"
      ;;
    *)
      echo "[ERROR] Unsupported profile: $profile" >&2
      exit 1
      ;;
  esac

  local beijing_ts
  local log
  beijing_ts="$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)"
  log="$ROOT/logs/${beijing_ts}_${LOG_TYPE}_${run_name}.log"

  if [[ "$RUN_MODE" == "foreground" ]]; then
    echo "[INFO] Running profile in foreground: $profile"
    echo "[INFO] log: $log"
    env TERM="$TERM" PYTHONUNBUFFERED=1 CONDA_PREFIX="" CONDA_DEFAULT_ENV="" \
      ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
      --task Isaac-Forklift-PalletInsertLift-Direct-v0 \
      --headless \
      --enable_cameras \
      --num_envs "$NUM_ENVS" \
      --max_iterations "$MAX_ITERATIONS" \
      agent.run_name="$run_name" \
      env.use_camera=true \
      env.use_asymmetric_critic=true \
      env.stage_1_mode=false \
      env.camera_width=256 \
      env.camera_height=256 \
      env.exp83_traj_goal_mode="$traj_mode" \
      env.exp83_target_center_family_mode="$target_center_family_mode" \
      agent.policy.class_name=rsl_rl.modules.VisionActorCritic \
      agent.policy.backbone_type=resnet34 \
      'agent.obs_groups.policy=[image, proprio]' \
      'agent.obs_groups.critic=[critic]' \
      agent.policy.imagenet_backbone_init=true \
      agent.policy.freeze_backbone=true \
      > "$log" 2>&1
    echo "[INFO] Finished profile: $profile"
    echo "[INFO] log: $log"
    return 0
  fi

  nohup env TERM="$TERM" PYTHONUNBUFFERED=1 CONDA_PREFIX="" CONDA_DEFAULT_ENV="" \
    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Forklift-PalletInsertLift-Direct-v0 \
    --headless \
    --enable_cameras \
    --num_envs "$NUM_ENVS" \
    --max_iterations "$MAX_ITERATIONS" \
    agent.run_name="$run_name" \
    env.use_camera=true \
    env.use_asymmetric_critic=true \
    env.stage_1_mode=false \
    env.camera_width=256 \
    env.camera_height=256 \
    env.exp83_traj_goal_mode="$traj_mode" \
    env.exp83_target_center_family_mode="$target_center_family_mode" \
    agent.policy.class_name=rsl_rl.modules.VisionActorCritic \
    agent.policy.backbone_type=resnet34 \
    'agent.obs_groups.policy=[image, proprio]' \
    'agent.obs_groups.critic=[critic]' \
    agent.policy.imagenet_backbone_init=true \
    agent.policy.freeze_backbone=true \
    > "$log" 2>&1 &

  echo "Started Exp8.3 strict diagnostic compare run."
  echo "profile: $profile"
  echo "log: $log"
  echo "run_name: $run_name"
  echo "pid: $!"
}

case "$PROFILE" in
  all)
    if [[ "$RUN_MODE" == "foreground" ]]; then
      run_profile g2b
      run_profile g3
    else
      queue_ts="$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)"
      queue_log="$ROOT/logs/${queue_ts}_${LOG_TYPE}_exp8_3_strict_diag_compare_queue_iter${MAX_ITERATIONS}.log"
      nohup bash -lc "cd '$ROOT' && bash scripts/run_exp8_3_strict_diag_compare.sh g2b '$MAX_ITERATIONS' '$NUM_ENVS' foreground && bash scripts/run_exp8_3_strict_diag_compare.sh g3 '$MAX_ITERATIONS' '$NUM_ENVS' foreground" > "$queue_log" 2>&1 &
      echo "Started Exp8.3 strict diagnostic compare queue."
      echo "profiles: g2b -> g3"
      echo "queue_log: $queue_log"
      echo "pid: $!"
    fi
    ;;
  b0prime|g2b|g3)
    run_profile "$PROFILE"
    ;;
  *)
    echo "[ERROR] Unsupported profile selection: $PROFILE" >&2
    echo "Usage: bash scripts/run_exp8_3_strict_diag_compare.sh [b0prime|g2b|g3|all] [max_iterations] [num_envs] [background|foreground]" >&2
    exit 1
    ;;
esac
