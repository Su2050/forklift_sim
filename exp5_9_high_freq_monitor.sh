#!/usr/bin/env bash
LOG_FILE="/home/uniubi/projects/forklift_sim/logs/20260314_212136_train_exp5_9_nuclear_reward_early_stop_0.28.log"

echo "Monitoring Exp 5.9 for Early Stopping (High Frequency)"
echo "Target: rg > 0.5 (50%) AND yaw_deg_mean < 5.0"
echo "--------------------------------------------------------"

while true; do
  if [ -f "$LOG_FILE" ]; then
    ITER=$(grep -oP "Iteration \d+" "$LOG_FILE" | tail -n 1 | awk '{print $2}')
    if [ -n "$ITER" ]; then
      RG=$(grep -a "paper_reward/rg:" "$LOG_FILE" | tail -n 1 | awk '{print $2}')
      YAW=$(grep -a "err/yaw_deg_mean:" "$LOG_FILE" | tail -n 1 | awk '{print $2}')
      LAT=$(grep -a "err/tip_y_err_mean:" "$LOG_FILE" | tail -n 1 | awk '{print $2}')
      DISP=$(grep -a "diag/pallet_disp_xy_mean:" "$LOG_FILE" | tail -n 1 | awk '{print $2}')
      
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
  sleep 10
done
