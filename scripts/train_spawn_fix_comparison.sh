#!/usr/bin/env bash
#
# 实验：修复叉齿穿透托盘初始化 Bug (x_max: -2.5 → -3.0)
# 对比：Resume 训练 vs 从零训练
#
# 用法（整个脚本用 nohup 运行，避免 GPU 资源竞争）：
#   nohup bash scripts/train_spawn_fix_comparison.sh \
#     > logs/$(date +%Y%m%d_%H%M%S)_spawn_fix_runner.log 2>&1 &
#

ISAACLAB_DIR="/home/uniubi/projects/forklift_sim/IsaacLab"
LOG_DIR="/home/uniubi/projects/forklift_sim/logs"

export PYTHONUNBUFFERED=1
export TERM=xterm
unset CONDA_PREFIX

cd "${ISAACLAB_DIR}"

# ============================================================
# Experiment A: Resume 训练
#   基于旧 early_stop_fly 下最佳模型 model_3600.pt，
#   在新 spawn 范围下续训 1000 轮
# ============================================================
TIMESTAMP_A=$(date +%Y%m%d_%H%M%S)
LOG_A="${LOG_DIR}/${TIMESTAMP_A}_train_s1.0y_resume.log"
echo "[$(date)] === Experiment A: resume from 2026-02-22_23-25-38/model_3600.pt, 1000 more iters ===" | tee "${LOG_A}"

bash isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Forklift-PalletInsertLift-Direct-v0 \
    --headless --num_envs 1024 --max_iterations 1000 \
    --resume \
    --load_run "2026-02-22_23-25-38" \
    --checkpoint "model_3600.pt" \
    >> "${LOG_A}" 2>&1 || true

echo "[$(date)] === Experiment A finished ===" | tee -a "${LOG_A}"

# ============================================================
# Experiment B: 从零训练 3600 轮
#   全新策略，完全在安全 spawn 范围下学习
# ============================================================
TIMESTAMP_B=$(date +%Y%m%d_%H%M%S)
LOG_B="${LOG_DIR}/${TIMESTAMP_B}_train_s1.0y_scratch.log"
echo "[$(date)] === Experiment B: from scratch, 3600 iters ===" | tee "${LOG_B}"

bash isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Forklift-PalletInsertLift-Direct-v0 \
    --headless --num_envs 1024 --max_iterations 3600 \
    >> "${LOG_B}" 2>&1 || true

echo "[$(date)] === Experiment B finished ===" | tee -a "${LOG_B}"

echo ""
echo "[$(date)] ============================================"
echo "[$(date)] All experiments complete."
echo "[$(date)] Log A (resume):  ${LOG_A}"
echo "[$(date)] Log B (scratch): ${LOG_B}"
echo "[$(date)] ============================================"
