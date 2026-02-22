#!/bin/bash
set -e
export ISAACLAB_PATH=/home/uniubi/projects/forklift_sim/IsaacLab
export PYTHONPATH=/home/uniubi/projects/forklift_sim/forklift_expert_policy_project:$PYTHONPATH

LOGDIR=/home/uniubi/projects/forklift_sim/logs/smoke_s1.0v_phase4/100ep
mkdir -p $LOGDIR
PY=$ISAACLAB_PATH/_isaac_sim/python.sh
SC=/home/uniubi/projects/forklift_sim/forklift_expert_policy_project/scripts/play_expert.py

: > $LOGDIR/progress.log

for S in 0 7 13 21 33 42 55 66 77 88 99 111 123 144 155 177 188 200 222 250; do
  echo "=== seed=$S start $(date +%H:%M:%S) ===" >> $LOGDIR/progress.log
  $PY $SC --task Isaac-Forklift-PalletInsertLift-Direct-v0 \
    --num_envs 1 --headless --episodes 5 --seed $S \
    --log_file $LOGDIR/s${S}.log > $LOGDIR/f${S}.log 2>&1
  echo "=== seed=$S done exit=$? $(date +%H:%M:%S) ===" >> $LOGDIR/progress.log
done
echo "ALL_DONE" >> $LOGDIR/progress.log
