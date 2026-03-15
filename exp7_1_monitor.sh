#!/usr/bin/env bash
set -euo pipefail

# 找到最新的 exp7_1 日志文件
LOG_FILE=$(ls -t /home/uniubi/projects/forklift_sim/logs/*_train_exp7_1_camera_fov_75deg.log | head -n 1)
OUTPUT_MD="/home/uniubi/projects/forklift_sim/docs/0310-0314experiments/exp7_1_live_monitor.md"

echo "# Exp 7.1 Live Monitor (Camera FOV 75°)" > "$OUTPUT_MD"
echo "Log file: $LOG_FILE" >> "$OUTPUT_MD"
echo '```' >> "$OUTPUT_MD"

echo "Monitoring $LOG_FILE..."

while true; do
    # 提取最新的指标
    tail -n 100 "$LOG_FILE" > /tmp/temp_log_7_1.txt
    
    ITER=$(grep -oP "iteration \d+" /tmp/temp_log_7_1.txt | tail -n 1 | awk '{print $2}')
    RG=$(grep "paper_reward/rg:" /tmp/temp_log_7_1.txt | tail -n 1 | awk '{print $2}')
    YAW=$(grep "err/yaw_deg_mean:" /tmp/temp_log_7_1.txt | tail -n 1 | awk '{print $2}')
    LAT=$(grep "err/lateral_mean:" /tmp/temp_log_7_1.txt | tail -n 1 | awk '{print $2}')
    DISP=$(grep "diag/pallet_disp_xy_mean:" /tmp/temp_log_7_1.txt | tail -n 1 | awk '{print $2}')
    
    if [ -n "$ITER" ] && [ -n "$RG" ]; then
        TIMESTAMP=$(date +"%H:%M:%S")
        printf "[%s] Iter: %s | rg: %.4f | yaw: %.4f | lat: %.4f | disp: %.4f\n" "$TIMESTAMP" "$ITER" "$RG" "$YAW" "$LAT" "$DISP" >> "$OUTPUT_MD"
    fi
    
    sleep 20
done
