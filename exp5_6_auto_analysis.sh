#!/bin/bash
LOG_FILE="/home/uniubi/projects/forklift_sim/logs/20260314_091510_train_exp5_6_expert_reward_refactor_0.25.log"
MD_FILE="/home/uniubi/projects/forklift_sim/docs/0310-0314experiments/exp5_6_auto_analysis.md"

echo "# Exp 5.6 自动分析报告 (每10分钟更新)" > "$MD_FILE"

while true; do
    sleep 600
    
    if [ -f "$LOG_FILE" ]; then
        TIME=$(date "+%H:%M:%S")
        ITER=$(grep -a "Iteration:" "$LOG_FILE" | tail -n 1 | awk '{print $4}')
        
        # 提取最近 10 次的 rg 成功率并计算平均值
        RG_AVG=$(grep -a "Episode Reward/rg:" "$LOG_FILE" | tail -n 10 | awk '{sum+=$3} END {if (NR>0) print sum/NR; else print 0}')
        YAW_AVG=$(grep -a "Metrics/yaw_err_deg:" "$LOG_FILE" | tail -n 10 | awk '{sum+=$2} END {if (NR>0) print sum/NR; else print 0}')
        DISP_AVG=$(grep -a "Metrics/pallet_disp_m:" "$LOG_FILE" | tail -n 10 | awk '{sum+=$2} END {if (NR>0) print sum/NR; else print 0}')
        
        echo "## 更新时间: $TIME (Iteration: $ITER)" >> "$MD_FILE"
        echo "- **近10次平均 rg 成功率**: $RG_AVG" >> "$MD_FILE"
        echo "- **近10次平均 Yaw 误差**: $YAW_AVG" >> "$MD_FILE"
        echo "- **近10次平均推盘位移**: $DISP_AVG" >> "$MD_FILE"
        
        if (( $(echo "$RG_AVG > 0.3" | bc -l) )); then
            echo "- **结论**: 🚀 破冰成功！成功率超过 30%，新策略有效！" >> "$MD_FILE"
        elif (( $(echo "$RG_AVG > 0" | bc -l) )); then
            echo "- **结论**: ⏳ 正在艰难破冰，有少量成功，继续观察。" >> "$MD_FILE"
        else
            echo "- **结论**: ❌ 依然是 0，可能需要检查是否又陷入了探索瘫痪。" >> "$MD_FILE"
        fi
        echo "---" >> "$MD_FILE"
    fi
done
