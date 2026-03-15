#!/usr/bin/env bash
LOG_FILE="/home/uniubi/projects/forklift_sim/logs/$(ls -t /home/uniubi/projects/forklift_sim/logs/ | grep exp6_0_b | head -n 1)"

echo "Monitoring Exp 6.0b (Micro Generalization)"
echo "Target: rg > 0.5 AND yaw_deg_mean < 5.0"
echo "Log file: $LOG_FILE"
echo "--------------------------------------------------------"

while true; do
  if [ -f "$LOG_FILE" ]; then
    # 强制不使用缓存读取
    tail -n 100 "$LOG_FILE" > /tmp/temp_log.txt
    
    ITER=$(grep -oP "iteration \d+" /tmp/temp_log.txt | tail -n 1 | awk '{print $2}')
    if [ -n "$ITER" ]; then
      RG=$(grep -a "paper_reward/rg:" /tmp/temp_log.txt | tail -n 1 | awk '{print $2}')
      YAW=$(grep -a "err/yaw_deg_mean:" /tmp/temp_log.txt | tail -n 1 | awk '{print $2}')
      LAT=$(grep -a "err/lateral_mean:" /tmp/temp_log.txt | tail -n 1 | awk '{print $2}')
      DISP=$(grep -a "diag/pallet_disp_xy_mean:" /tmp/temp_log.txt | tail -n 1 | awk '{print $2}')
      
      echo "[$(date +'%H:%M:%S')] Iter: $ITER | rg: $RG | yaw: $YAW | lat: $LAT | disp: $DISP"
      
      # 简易早停提醒
      if (( $(echo "$RG > 0.5" | bc -l) )) && (( $(echo "$YAW < 5.0" | bc -l) )); then
          echo "!!! EARLY STOPPING CONDITION MET !!!"
          echo "Consider stopping the training now!"
      fi
    fi
  else
    echo "Waiting for log file..."
  fi
  sleep 20
done
