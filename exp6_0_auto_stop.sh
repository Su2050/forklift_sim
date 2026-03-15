#!/usr/bin/env bash
LOG_FILE="/home/uniubi/projects/forklift_sim/logs/$(ls -t /home/uniubi/projects/forklift_sim/logs/ | grep exp6_0_b | head -n 1)"

echo "Auto-Stopping Monitor for Exp 6.0b"
echo "Target: rg > 0.5 AND yaw_deg_mean < 5.0"
echo "Log file: $LOG_FILE"
echo "--------------------------------------------------------"

while true; do
  if [ -f "$LOG_FILE" ]; then
    ITER=$(grep -oP "iteration \d+" "$LOG_FILE" | tail -n 1 | awk '{print $2}')
    if [ -n "$ITER" ]; then
      RG=$(grep -a "paper_reward/rg:" "$LOG_FILE" | tail -n 1 | awk '{print $2}')
      YAW=$(grep -a "err/yaw_deg_mean:" "$LOG_FILE" | tail -n 1 | awk '{print $2}')
      
      # 使用 awk 进行浮点数比较
      RG_MET=$(awk -v rg="$RG" 'BEGIN { print (rg > 0.5) ? 1 : 0 }')
      YAW_MET=$(awk -v yaw="$YAW" 'BEGIN { print (yaw < 5.0) ? 1 : 0 }')

      if [ "$RG_MET" -eq 1 ] && [ "$YAW_MET" -eq 1 ]; then
          echo "[$(date +'%H:%M:%S')] !!! EARLY STOPPING CONDITION MET at Iter $ITER !!!"
          echo "rg: $RG > 0.5, yaw: $YAW < 5.0"
          echo "Killing training process..."
          pkill -f "train.py.*exp6_0_b"
          echo "Training stopped automatically."
          exit 0
      fi
    fi
  fi
  sleep 30
done
