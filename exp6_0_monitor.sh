#!/usr/bin/env bash
LOG_FILE="/home/uniubi/projects/forklift_sim/logs/$(ls -t /home/uniubi/projects/forklift_sim/logs/ | grep exp6_0 | head -n 1)"

echo "Monitoring Exp 6.0 (Generalization)"
echo "Target: rg > 0.5 with widened spawn ranges"
echo "--------------------------------------------------------"

while true; do
  if [ -f "$LOG_FILE" ]; then
    ITER=$(grep -oP "Iteration \d+" "$LOG_FILE" | tail -n 1 | awk '{print $2}')
    if [ -n "$ITER" ]; then
      RG=$(grep -a "paper_reward/rg:" "$LOG_FILE" | tail -n 1 | awk '{print $2}')
      YAW=$(grep -a "err/yaw_deg_mean:" "$LOG_FILE" | tail -n 1 | awk '{print $2}')
      LAT=$(grep -a "err/lateral_mean:" "$LOG_FILE" | tail -n 1 | awk '{print $2}')
      DISP=$(grep -a "diag/pallet_disp_xy_mean:" "$LOG_FILE" | tail -n 1 | awk '{print $2}')
      
      echo "[$(date +'%H:%M:%S')] Iter: $ITER | rg: $RG | yaw: $YAW | lat: $LAT | disp: $DISP"
    fi
  else
    echo "Waiting for log file..."
  fi
  sleep 60
done
