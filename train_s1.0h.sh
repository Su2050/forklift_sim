#!/bin/bash
# S1.0h 训练启动脚本
# 注意：如果 rsl_rl 未安装，请先运行: cd /home/uniubi/projects/forklift_sim/IsaacLab && ./isaaclab.sh -i rsl_rl

cd /home/uniubi/projects/forklift_sim/IsaacLab

# 使用 nohup 启动训练，日志输出到项目根目录的 logs 文件夹
nohup ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Forklift-PalletInsertLift-Direct-v0 \
  --headless \
  --num_envs 1024 \
  --max_iterations 2000 \
  > /home/uniubi/projects/forklift_sim/logs/20260208_train_s1.0h.log 2>&1 &

TRAIN_PID=$!
echo "=========================================="
echo "S1.0h 训练已启动"
echo "进程 PID: $TRAIN_PID"
echo "日志文件: /home/uniubi/projects/forklift_sim/logs/20260208_train_s1.0h.log"
echo "=========================================="
echo ""
echo "查看实时日志: tail -f /home/uniubi/projects/forklift_sim/logs/20260208_train_s1.0h.log"
echo "停止训练: kill $TRAIN_PID"
