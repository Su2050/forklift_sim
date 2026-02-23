#!/usr/bin/env bash
# 两组对比实验（顺序执行，共享 GPU）
#
# 实验 A: 基于 s1.0x 从零训练的 model_1999.pt 继续训练 2000 iters
# 实验 B: 基于旧 anti_hack 的 model_1999.pt 继续训练 2000 iters
#
# 注意：请使用 nohup 运行本脚本，防止 SSH 断开：
#   nohup bash scripts/train_resume_from_1999.sh > logs/resume_runner.log 2>&1 &
#
#   # 或自动等待当前训练结束：
#   nohup bash scripts/train_resume_from_1999.sh --wait-pid <PID> > logs/resume_runner.log 2>&1 &

PROJ_DIR="/home/uniubi/projects/forklift_sim"
ISAACLAB_DIR="${PROJ_DIR}/IsaacLab"
LOG_DIR="${PROJ_DIR}/logs"

# ---- 可选：等待指定 PID 结束 ----
if [[ "${1:-}" == "--wait-pid" ]] && [[ -n "${2:-}" ]]; then
    TARGET_PID="$2"
    echo "[INFO] 等待 PID ${TARGET_PID} (s1.0x 从零训练) 结束..."
    while kill -0 "${TARGET_PID}" 2>/dev/null; do
        sleep 30
    done
    echo "[INFO] PID ${TARGET_PID} 已结束"
fi

cd "${ISAACLAB_DIR}"

# 禁用 python 缓冲，保证日志实时写入
export PYTHONUNBUFFERED=1
export TERM=xterm

# ============================================================
# 实验 A: s1.0x 从零训练 → 继续 2000 iters
# ============================================================
echo ""
echo "============================================"
echo " 实验 A: s1.0x 继续训练 2000 iters"
echo "============================================"

LOG_A="${LOG_DIR}/$(date +%Y%m%d_%H%M%S)_train_s1.0x_continue.log"

bash isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Forklift-PalletInsertLift-Direct-v0 \
    --headless --num_envs 1024 --max_iterations 2000 \
    --resume \
    --load_run "2026-02-23_08-12-59" \
    --checkpoint "model_1999.pt" \
    > "${LOG_A}" 2>&1 || echo "[警告] 实验 A 异常退出，继续执行实验 B..."

echo "[实验 A 完成] 日志: ${LOG_A}"

# ============================================================
# 实验 B: 旧 anti_hack model_1999.pt → 继续 2000 iters
# ============================================================
echo ""
echo "============================================"
echo " 实验 B: anti_hack model_1999 继续训练 2000 iters"
echo "============================================"

LOG_B="${LOG_DIR}/$(date +%Y%m%d_%H%M%S)_train_s1.0x_resume.log"

bash isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Forklift-PalletInsertLift-Direct-v0 \
    --headless --num_envs 1024 --max_iterations 2000 \
    --resume \
    --load_run "2026-02-22_21-34-12" \
    --checkpoint "model_1999.pt" \
    > "${LOG_B}" 2>&1 || echo "[警告] 实验 B 异常退出..."

echo "[实验 B 完成] 日志: ${LOG_B}"

# ============================================================
echo ""
echo "============================================"
echo " 全部完成！对比日志："
echo "   A (s1.0x 继续):       ${LOG_A}"
echo "   B (anti_hack resume): ${LOG_B}"
echo "============================================"
