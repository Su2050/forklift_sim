#!/bin/bash
LOG_FILE="/home/uniubi/projects/forklift_sim/logs/20260314_135430_train_exp5_7_break_local_optima_rg_5000.log"
MD_FILE="/home/uniubi/projects/forklift_sim/docs/0310-0314experiments/exp5_7_live_monitor.md"

while true; do
    if [ -f "$LOG_FILE" ]; then
        # 提取最新的 Iteration
        ITER=$(grep -a "Iteration:" "$LOG_FILE" | tail -n 1 | awk '{print $4}')
        
        if [ ! -z "$ITER" ]; then
            # 提取各项指标
            RG=$(grep -a "Episode Reward/rg:" "$LOG_FILE" | tail -n 1 | awk '{print $3}')
            YAW=$(grep -a "Metrics/yaw_err_deg:" "$LOG_FILE" | tail -n 1 | awk '{print $2}')
            LAT=$(grep -a "Metrics/lateral_err_m:" "$LOG_FILE" | tail -n 1 | awk '{print $2}')
            DISP=$(grep -a "Metrics/pallet_disp_m:" "$LOG_FILE" | tail -n 1 | awk '{print $2}')
            DIST=$(grep -a "Metrics/dist_front_m:" "$LOG_FILE" | tail -n 1 | awk '{print $2}')
            
            # 获取当前时间
            TIME=$(date "+%H:%M:%S")
            
            # 写入 Markdown 表格
            echo "| $TIME | $ITER | $RG | $YAW | $LAT | $DISP | $DIST |" >> "$MD_FILE"
        fi
    fi
    sleep 60
done
