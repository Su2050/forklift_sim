---
name: isaaclab-training-workflow
description: Standard operating procedure for modifying code, committing changes, cleaning up zombie processes, checking memory, starting IsaacLab RL training, and analyzing logs. Use when the user asks to start a new experiment, modify training code, or run the training loop.
---

# IsaacLab Training Workflow

## 1. Context and Purpose
This skill defines the standard operating procedure (SOP) for running Reinforcement Learning experiments in the IsaacLab environment. It ensures that code changes are properly tracked, system resources are clean before starting heavy GPU training, and logs are correctly monitored.

## 2. The Standard Workflow (Step-by-Step)

When asked to start a new training run or modify existing training logic, ALWAYS follow these exact steps in order:

### Step 1: Commit Current Changes
Before making any new modifications or starting a new run, ensure the working directory is clean.
```bash
git add .
git commit -m "chore: save state before starting new experiment"
```
*(If the user already asked you to modify code, do the modification first, then commit).*

### Step 2: Clean Up Zombie Processes
IsaacLab `train.py` processes often leave zombie child processes that consume GPU memory. You MUST kill them before starting a new run.
```bash
# 1. Try graceful kill first
pkill -f "train.py"
# 2. Wait a moment
sleep 2
# 3. Force kill any stubborn processes
pkill -9 -f "train.py"
```

### Step 3: Check System Resources
Verify that RAM and VRAM have been successfully freed.
```bash
free -h
nvidia-smi
```
*Note: Ensure there is enough VRAM (usually >10GB free) before proceeding.*

### Step 4: Sync Patch to IsaacLab (Crucial)
If any files in `forklift_pallet_insert_lift_project/isaaclab_patch/` were modified, they MUST be synced to the actual IsaacLab directory before running.
```bash
bash forklift_pallet_insert_lift_project/scripts/install_into_isaaclab.sh /home/uniubi/projects/forklift_sim/IsaacLab
```

### Step 5: Start Training
Run the standard training script. It is usually wrapped in a shell script (e.g., `run_experiment_4_paper_native.sh`).
```bash
bash run_experiment_4_paper_native.sh
```
*Note: The script should output the path to the newly created log file.*

### Step 6: Setup Monitoring
Once training starts, locate the new log file and set up a background monitoring script (or use `tail -f`) to track key metrics like `rg` (success rate), `yaw_deg_mean`, and `pallet_disp_xy_mean`.
```bash
# Example: Check the first few iterations
head -n 50 <path_to_new_log_file>
```

## 3. Best Practices
*   **Never skip the cleanup step**: Starting a new training run without killing old `train.py` processes will almost certainly lead to CUDA Out Of Memory (OOM) errors.
*   **Always sync patches**: Modifying the patch directory without running `install_into_isaaclab.sh` means the training will run on old code.
*   **Document the experiment**: After starting the run, update the relevant markdown files in `docs/0310-0314experiments/` with the goal and configuration of the new experiment.