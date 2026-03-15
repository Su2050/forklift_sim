#!/usr/bin/env bash
set -euo pipefail

LOG_FILE=$(ls -t /home/uniubi/projects/forklift_sim/logs/*_train_exp7_1_camera_fov_75deg.log | head -n 1)

echo "Auto-Stopping Monitor for Exp 7.1"
echo "Target: rg > 0.7 AND yaw_deg_mean < 3.0"
echo "Log file: $LOG_FILE"
echo "--------------------------------------------------------"

while true; do
    tail -n 100 "$LOG_FILE" > /tmp/temp_log_stop_7_1.txt
    
    ITER=$(grep -oP "iteration \d+" /tmp/temp_log_stop_7_1.txt | tail -n 1 | awk '{print $2}')
    RG=$(grep "paper_reward/rg:" /tmp/temp_log_stop_7_1.txt | tail -n 1 | awk '{print $2}')
    YAW=$(grep "err/yaw_deg_mean:" /tmp/temp_log_stop_7_1.txt | tail -n 1 | awk '{print $2}')
    
    if [ -n "$ITER" ] && [ -n "$RG" ] && [ -n "$YAW" ]; then
        # Exp 7.1 是极窄范围，我们要求更高的成功率和更好的姿态
        if (( $(echo "$RG > 0.7" | bc -l) )) && (( $(echo "$YAW < 3.0" | bc -l) )); then
            echo "[$(date)] SUCCESS! Target reached at Iteration $ITER."
            echo "rg: $RG, yaw: $YAW"
            echo "Stopping training process..."
            
            # 杀掉训练进程
            pkill -f "train.py.*exp7_1"
            
            echo "Training stopped. Checkpoint saved."
            exit 0
        fi
    fi
    
    sleep 20
done
