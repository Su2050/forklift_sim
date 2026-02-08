#!/bin/bash
# S1.0h 训练启动脚本
# 注意：
#   1. 必须先退出 conda: conda deactivate（Isaac Sim 使用自带 Python 3.11）
#   2. 如果 rsl_rl 未安装，请先运行: cd IsaacLab && ./isaaclab.sh -i rsl_rl

cd /home/uniubi/projects/forklift_sim/IsaacLab

LOG_FILE="/home/uniubi/projects/forklift_sim/logs/$(date +%Y%m%d_%H%M%S)_train_s1.0h.log"

# TERM=xterm: 防止 Isaac Sim 终端兼容性问题
# PYTHONUNBUFFERED=1: 确保 nohup 模式下日志实时写入（不被 Python 缓冲）
nohup env TERM=xterm PYTHONUNBUFFERED=1 ./isaaclab.sh -p \
  scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Forklift-PalletInsertLift-Direct-v0 \
  --headless \
  --num_envs 1024 \
  --max_iterations 2000 \
  agent.run_name=exp_s1.0h \
  > "$LOG_FILE" 2>&1 &

TRAIN_PID=$!
echo "=========================================="
echo "S1.0h 训练已启动"
echo "进程 PID: $TRAIN_PID"
echo "日志文件: $LOG_FILE"
echo "=========================================="
echo ""
echo "查看实时日志: tail -f $LOG_FILE"
echo "停止训练: kill $TRAIN_PID"
