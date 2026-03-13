#!/bin/bash
LOG_FILE="/home/uniubi/projects/forklift_sim/logs/20260312_223434_train_exp3_3_conditional_push_penalty_rrl.log"

echo "Waiting for training to reach iteration 150..."
while true; do
    if [ -f "$LOG_FILE" ]; then
        ITER=$(grep "Learning iteration" "$LOG_FILE" | tail -n 1 | awk '{print $4}' | cut -d'/' -f1)
        if [ ! -z "$ITER" ] && [ "$ITER" -ge 150 ]; then
            echo "Reached iteration $ITER. Dumping key metrics..."
            echo "--- Metrics at Iteration $ITER ---"
            tail -n 200 "$LOG_FILE" | grep -E "Iteration|push_free_success_rate_total|push_free_insert_rate_total|phase/frac_inserted|traj/commit_gate_mean|reward/r_commit_front|reward/r_commit_insert|err/dist_front_mean|diag/pallet_disp_xy_mean|traj/corridor_frac|traj/yaw_traj_deg_mean|reward/pen_push_cond"
            break
        fi
    fi
    sleep 60
done
