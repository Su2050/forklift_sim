#!/usr/bin/env bash
# S1.0O 消融实验统一启动脚本
# 用法: ./run_experiment.sh <variant> [max_iter] [num_envs] [seed] [resume_run] [resume_ckpt]
# 示例:
#   ./run_experiment.sh N 300           # Round 0: 基线对齐
#   ./run_experiment.sh A1 600          # Round 1: A1 消融
#   ./run_experiment.sh A1B1C1 2000     # Round 2: 赢家全组合
#   ./run_experiment.sh A3B1C2 1000 1024 42 "2026-02-11_22-50-21" "model_999.pt"  # resume 续训
set -euo pipefail

# 退出 conda 环境，避免干扰 IsaacLab 自带的 Python
conda deactivate 2>/dev/null || true

# 设置 TERM 防止 nohup 下 tabs 命令报错
export TERM="${TERM:-xterm}"

VARIANT="${1:?用法: $0 <variant> [max_iter] [num_envs] [seed] [resume_run] [resume_ckpt]}"
MAX_ITER="${2:-600}"
NUM_ENVS="${3:-1024}"
SEED="${4:-42}"
RESUME_RUN="${5:-}"
RESUME_CKPT="${6:-model_.*.pt}"

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

# 4. 构建训练命令
RESUME_ARGS=""
if [[ -n "${RESUME_RUN}" ]]; then
  RESUME_ARGS="--resume --load_run ${RESUME_RUN} --checkpoint ${RESUME_CKPT}"
  echo "[INFO] Resume 模式: run=${RESUME_RUN}, ckpt=${RESUME_CKPT}"
fi

# 5. 启动训练
echo "[3/4] 启动训练 -> ${LOG_FILE}"
cd "${ISAACLAB_DIR}"
nohup ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Forklift-PalletInsertLift-Direct-v0 \
  --headless --num_envs "${NUM_ENVS}" --max_iterations "${MAX_ITER}" \
  ${RESUME_ARGS} \
  > "${LOG_FILE}" 2>&1 &

TRAIN_PID=$!
echo "[4/4] 训练已后台启动 (PID: ${TRAIN_PID})"
echo "  日志: ${LOG_FILE}"
echo "  分支: ${BRANCH} (${GIT_HASH})"
echo "  配置: iter=${MAX_ITER}, envs=${NUM_ENVS}, seed=${SEED}"
if [[ -n "${RESUME_RUN}" ]]; then
  echo "  续训: run=${RESUME_RUN}, ckpt=${RESUME_CKPT}"
fi
echo ""
echo "监控: tail -f ${LOG_FILE}"
echo "停止: kill ${TRAIN_PID}"
