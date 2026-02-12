#!/usr/bin/env bash
# =============================================================================
# 后台压测脚本 — 20 seed x 20 episode (400 total)
#
# 使用方式:
#   cd /home/uniubi/projects/forklift_sim
#   nohup bash forklift_expert_policy_project/scripts/run_stress_test.sh \
#       > logs/stress_test/large_v4/runner.log 2>&1 &
#
# 脚本会自动:
#   1. 检查分支为 master 且工作区干净
#   2. 清除 __pycache__
#   3. 运行 20 seed x 20 episode 压测
#   4. 生成汇总分析报告
# =============================================================================
set -euo pipefail

# ---- 路径 ----
PROJECT_ROOT="/home/uniubi/projects/forklift_sim"
ISAACLAB_DIR="${PROJECT_ROOT}/IsaacLab"
EXPERT_PROJECT="${PROJECT_ROOT}/forklift_expert_policy_project"
PLAY_SCRIPT="${EXPERT_PROJECT}/scripts/play_expert.py"
OUTPUT_DIR="${PROJECT_ROOT}/logs/stress_test/large_v4"
ANALYZE_SCRIPT="${OUTPUT_DIR}/analyze_all.py"

# ---- 测试参数 ----
SEEDS=(0 42 123 256 512 777 1024 1337 2026 3000 4096 5555 6789 7777 8192 9000 9876 11111 22222 99999)
EPISODES=20
TASK="Isaac-Forklift-PalletInsertLift-Direct-v0"

# ---- 环境设置 ----
export TERM=xterm
export PYTHONPATH="${EXPERT_PROJECT}:${PYTHONPATH:-}"

# 退出 conda — nohup 下 conda deactivate 不起作用,
# 必须 unset CONDA_PREFIX 否则 isaaclab.sh 会用 conda python
unset CONDA_PREFIX 2>/dev/null || true
unset CONDA_DEFAULT_ENV 2>/dev/null || true
conda deactivate 2>/dev/null || true

echo "============================================================"
echo "  Expert Policy Stress Test — large_v4"
echo "  Seeds: ${#SEEDS[@]}  Episodes/seed: ${EPISODES}"
echo "  Total: $(( ${#SEEDS[@]} * EPISODES )) episodes"
echo "  Output: ${OUTPUT_DIR}"
echo "  Started: $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"

# ---- 前置检查 ----
cd "${PROJECT_ROOT}"

BRANCH=$(git branch --show-current)
if [[ "${BRANCH}" != "master" ]]; then
    echo "[ERROR] Not on master branch (current: ${BRANCH}). Aborting."
    exit 1
fi
echo "[OK] Branch: master"

# 检查工作区是否干净（忽略 untracked 文件）
DIRTY=$(git diff --name-only HEAD 2>/dev/null || true)
if [[ -n "${DIRTY}" ]]; then
    echo "[ERROR] Working directory has uncommitted changes. Aborting."
    echo "  Modified files:"
    echo "${DIRTY}" | sed 's/^/    /'
    exit 1
fi
echo "[OK] Working directory clean"

# 清除 __pycache__
find "${EXPERT_PROJECT}" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
echo "[OK] Cleared __pycache__"

# 创建输出目录
mkdir -p "${OUTPUT_DIR}"

# ---- 运行压测 ----
cd "${ISAACLAB_DIR}"

TOTAL_SEEDS=${#SEEDS[@]}
COMPLETED=0

for seed in "${SEEDS[@]}"; do
    COMPLETED=$((COMPLETED + 1))
    LOG_FILE="${OUTPUT_DIR}/seed_${seed}.log"

    echo ""
    echo "[${COMPLETED}/${TOTAL_SEEDS}] $(date '+%H:%M:%S') Starting seed=${seed} ..."

    ./isaaclab.sh -p "${PLAY_SCRIPT}" \
        --task "${TASK}" \
        --num_envs 1 \
        --headless \
        --episodes "${EPISODES}" \
        --seed "${seed}" \
        --quiet \
        --log_file "${LOG_FILE}" \
        2>/dev/null || {
            echo "[WARN] seed=${seed} exited with non-zero status"
        }

    echo "[${COMPLETED}/${TOTAL_SEEDS}] $(date '+%H:%M:%S') seed=${seed} done -> ${LOG_FILE}"
done

echo ""
echo "============================================================"
echo "  All seeds completed at $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"

# ---- 运行分析 ----
if [[ -f "${ANALYZE_SCRIPT}" ]]; then
    echo ""
    echo "[ANALYSIS] Running analyze_all.py ..."
    cd "${PROJECT_ROOT}"
    python3 "${ANALYZE_SCRIPT}" | tee "${OUTPUT_DIR}/report.txt"
    echo ""
    echo "[ANALYSIS] Report saved to ${OUTPUT_DIR}/report.txt"
else
    echo "[WARN] analyze_all.py not found at ${ANALYZE_SCRIPT}, skipping analysis"
fi

echo ""
echo "============================================================"
echo "  STRESS TEST COMPLETE"
echo "  Finished: $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"
