#!/bin/bash
LOG_FILE="/home/uniubi/projects/forklift_sim/logs/20260314_081732_train_exp5_5_f_micro_step_0.28.log"
OUT_FILE="/home/uniubi/projects/forklift_sim/docs/0310-0314experiments/exp5_5f_auto_analysis.md"

echo "# Exp 5.5f (0.28m) 自动分析报告" > $OUT_FILE
echo "每 10 分钟自动生成一次分析结论。开始时间：$(date)" >> $OUT_FILE
echo "---" >> $OUT_FILE

while true; do
    sleep 600 # 10 分钟
    
    TIME=$(date "+%H:%M:%S")
    ITER=$(grep "Learning iteration" $LOG_FILE | tail -n 1 | sed -n 's/.*iteration \([0-9]*\/[0-9]*\).*/\1/p')
    
    # 提取最近 10 次的平均 rg
    AVG_RG=$(grep "paper_reward/rg:" $LOG_FILE | tail -n 10 | awk '{sum+=$2} END {print sum/NR}')
    AVG_YAW=$(grep "err/yaw_deg_mean:" $LOG_FILE | tail -n 10 | awk '{sum+=$2} END {print sum/NR}')
    AVG_DISP=$(grep "diag/pallet_disp_xy_mean:" $LOG_FILE | tail -n 10 | awk '{sum+=$2} END {print sum/NR}')
    
    echo "## 时间: $TIME (Iteration: $ITER)" >> $OUT_FILE
    echo "- **近 10 代平均成功率 (rg)**: $AVG_RG" >> $OUT_FILE
    echo "- **近 10 代平均偏航角**: $AVG_YAW 度" >> $OUT_FILE
    echo "- **近 10 代平均推盘位移**: $AVG_DISP 米" >> $OUT_FILE
    
    # 简单逻辑判断
    if (( $(echo "$AVG_RG > 0.45" | bc -l) )); then
        echo "- **🤖 AI 结论**: 成功率已稳定在 45% 以上！Agent 已经完全适应了 0.28m 的阈值。可以考虑进行下一步退火（0.26m）。" >> $OUT_FILE
    elif (( $(echo "$AVG_RG > 0.1" | bc -l) )); then
        echo "- **🤖 AI 结论**: 成功率在 $AVG_RG 左右波动。Agent 正在努力适应，姿态和推盘位移可能存在波动，请继续保持耐心等待收敛。" >> $OUT_FILE
    else
        echo "- **🤖 AI 结论**: 警告！成功率跌回了 10% 以下。请检查是否发生了灾难性遗忘，或者推盘惩罚导致了探索瘫痪。" >> $OUT_FILE
    fi
    echo "---" >> $OUT_FILE
done
