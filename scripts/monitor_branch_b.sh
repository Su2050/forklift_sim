#!/bin/bash
LOG_FILE="/home/uniubi/projects/forklift_sim/logs/20260317_193256_train_branch_b_paper_native.log"

echo "================================================="
echo "Branch B (Paper Native) Training Monitor"
echo "Time: $(date)"
echo "================================================="

# 提取最新的 iteration 信息
ITER=$(grep "Learning iteration" $LOG_FILE | tail -n 1 | awk '{print $3}')
echo "Current Iteration: $ITER"

# 提取关键指标
echo "--- Key Metrics ---"
grep "Mean value_function loss:" $LOG_FILE | tail -n 1
grep "Mean reward:" $LOG_FILE | tail -n 1
grep "paper_reward/rg:" $LOG_FILE | tail -n 1
grep "phase/frac_inserted:" $LOG_FILE | tail -n 1
grep "phase/frac_aligned:" $LOG_FILE | tail -n 1
grep "err/dist_front_mean:" $LOG_FILE | tail -n 1
grep "err/yaw_deg_mean:" $LOG_FILE | tail -n 1
grep "diag/pallet_disp_xy_mean:" $LOG_FILE | tail -n 1

echo "--- Trajectory Rewards ---"
grep "paper_reward/r_d:" $LOG_FILE | tail -n 1
grep "paper_reward/r_cd:" $LOG_FILE | tail -n 1
grep "paper_reward/r_cpsi:" $LOG_FILE | tail -n 1

echo "================================================="
