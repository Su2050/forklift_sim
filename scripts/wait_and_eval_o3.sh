#!/usr/bin/env bash
set -euo pipefail

echo "Waiting for O3 training to finish..."
while pgrep -f "train.py.*exp9_0_no_reference_rewardfix_o3" > /dev/null; do
    sleep 60
done
echo "Training finished."

echo "Generating summary..."
python3 scripts/generate_o3_summary.py

echo "Running validation..."
export TERM=xterm
source /home/uniubi/miniconda3/bin/activate /home/uniubi/miniconda3/envs/env_isaaclab
bash scripts/validation/run_smoke_validation.sh > outputs/validation/manual_runs/post_o3_validation_summary.log 2>&1

echo "All tasks completed."