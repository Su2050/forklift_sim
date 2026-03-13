#!/bin/bash
LOG_FILE="/home/uniubi/projects/forklift_sim/logs/20260314_073134_train_exp5_5_d_smooth_curriculum_0.25.log"
OUT_FILE="/home/uniubi/projects/forklift_sim/docs/0310-0313experiments/exp5_5d_live_monitor.md"

echo "# Exp 5.5d Live Monitor (Threshold: 0.25m)" > $OUT_FILE
echo "Updated every 60 seconds. (Last updated: $(date))" >> $OUT_FILE
echo "" >> $OUT_FILE
echo "| Time | Iteration | rg (Success) | Yaw Err (deg) | Pallet Disp (m) |" >> $OUT_FILE
echo "|---|---|---|---|---|" >> $OUT_FILE

while true; do
    if [ -f "$LOG_FILE" ]; then
        TIME=$(date "+%H:%M:%S")
        ITER=$(grep "Learning iteration" $LOG_FILE | tail -n 1 | sed -n 's/.*iteration \([0-9]*\/[0-9]*\).*/\1/p')
        RG=$(grep "paper_reward/rg:" $LOG_FILE | tail -n 1 | awk '{print $2}')
        YAW=$(grep "err/yaw_deg_mean:" $LOG_FILE | tail -n 1 | awk '{print $2}')
        DISP=$(grep "diag/pallet_disp_xy_mean:" $LOG_FILE | tail -n 1 | awk '{print $2}')
        
        if [ ! -z "$ITER" ]; then
            echo "| $TIME | $ITER | $RG | $YAW | $DISP |" >> $OUT_FILE
        fi
    fi
    sleep 60
done
