#!/usr/bin/env bash
# S1.0O 消融实验统一启动脚本
# 用法: ./run_experiment.sh <variant> [max_iter] [num_envs] [seed]
# 示例:
#   ./run_experiment.sh N 300           # Round 0: 基线对齐
#   ./run_experiment.sh A1 600          # Round 1: A1 消融
#   ./run_experiment.sh A1B1C1 2000     # Round 2: 赢家全组合
set -euo pipefail

VARIANT="${1:?用法: $0 <variant> [max_iter] [num_envs] [seed]}"
MAX_ITER="${2:-600}"
NUM_ENVS="${3:-1024}"
SEED="${4:-42}"

ISAACLAB_DIR="/home/uniubi/projects/forklift_sim/IsaacLab"
PROJECT_DIR="/home/uniubi/projects/forklift_sim"
LOG_DIR="${PROJECT_DIR}/logs"

# 确保日志目录存在
mkdir -p "${LOG_DIR}"

# 1. 切到对应实验分支
BRANCH="exp/DO-O-${VARIANT}"
echo "[1/4] 切换到分支 ${BRANCH} ..."
cd "${PROJECT_DIR}"
git checkout "${BRANCH}"

# 记录当前 commit hash（便于回溯）
GIT_HASH=$(git rev-parse --short HEAD)
echo "       commit: ${GIT_HASH}"

# 2. 安装到 IsaacLab
echo "[2/4] 安装到 IsaacLab ..."
bash forklift_pallet_insert_lift_project/scripts/install_into_isaaclab.sh "${ISAACLAB_DIR}"

# 3. 生成日志文件名
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/${TIMESTAMP}_train_s1.0o_${VARIANT}_s${SEED}.log"

# 4. 启动训练
echo "[3/4] 启动训练 -> ${LOG_FILE}"
cd "${ISAACLAB_DIR}"
nohup ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Forklift-PalletInsertLift-Direct-v0 \
  --headless --num_envs "${NUM_ENVS}" --max_iterations "${MAX_ITER}" \
  > "${LOG_FILE}" 2>&1 &

TRAIN_PID=$!
echo "[4/4] 训练已后台启动 (PID: ${TRAIN_PID})"
echo "  日志: ${LOG_FILE}"
echo "  分支: ${BRANCH} (${GIT_HASH})"
echo "  配置: iter=${MAX_ITER}, envs=${NUM_ENVS}, seed=${SEED}"
echo ""
echo "监控: tail -f ${LOG_FILE}"
echo "停止: kill ${TRAIN_PID}"
