# Forklift Pallet Insert+Lift (Isaac Lab task patch for DGX Spark)

This repository contains a **drop-in task** you can copy into an existing **Isaac Lab** checkout, then run PPO training/evaluation on **DGX Spark**.

Why it’s a “patch”:
- Isaac Lab’s training scripts import `isaaclab_tasks` to register tasks.
- The simplest zero-magic integration is to place this task under `IsaacLab/source/isaaclab_tasks/isaaclab_tasks/` and add one import line.

---

## 0) Recommended versions for DGX Spark

- Isaac Sim on DGX Spark is supported as **aarch64**; Isaac Sim 5.1 docs list DGX OS 7.2.3 + driver 580.95.05 for Spark.
- Isaac Lab: use the **release/2.3.0** branch for Spark support.

(See NVIDIA’s Spark playbook “Install and Use Isaac Sim and Isaac Lab | DGX Spark”.)

---

## 1) Install Isaac Sim + Isaac Lab on DGX Spark

Follow the official Spark playbook (recommended):
- https://build.nvidia.com/spark/isaac

After install, you should have an `IsaacLab/` repo with `isaaclab.sh` working.

---

## 2) Apply this patch into IsaacLab

From **this repo**:

```bash
# assuming:
#   - this repo is: ~/forklift_pallet_insert_lift_project
#   - IsaacLab is: ~/IsaacLab
bash scripts/install_into_isaaclab.sh ~/IsaacLab
```

What it does:
- Copies the folder `isaaclab_patch/source/isaaclab_tasks/.../forklift_pallet_insert_lift` into your IsaacLab checkout.
- Appends an import line into `IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/__init__.py` if missing.

---

## 3) Train (teacher policy, privileged state)

Inside `~/IsaacLab`:

```bash
./isaaclab.sh -i rsl_rl
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Forklift-PalletInsertLift-Direct-v0 \
  --headless \
  --num_envs 128
```

Notes:
- Start headless for speed.
- If you want to record videos during evaluation, use `play.py` with `--video`.

---

## 4) Evaluate + record video

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
  --task Isaac-Forklift-PalletInsertLift-Direct-v0 \
  --num_envs 32 \
  --load_run <RUN_FOLDER_NAME> \
  --checkpoint <PATH/TO/model_XXXX.pt> \
  --headless --video --video_length 400
```

---

## 5) Export policy (ONNX/JIT placeholder)

Export is usually done from the trained runner + normalizer.
This patch includes a **starter export script** under `scripts/export_policy_stub.py`.
In practice you’ll point it at the runner checkpoint directory used by Isaac Lab’s rsl_rl workflow.

---

## Task design summary (what this environment trains)

**Task boundary**: align → insert → lift (no navigation / carry / place).

**Success KPI (default)**:
- Forks inserted ≥ 2/3 into the pallet (measured along the pallet X axis),
- Lifted by ≥ 0.12 m, held for 1.0 s,
- Lateral error ≤ 3 cm and yaw error ≤ 3° at time of lifting.

Implementation detail:
- Fork tip is estimated each step by taking the robot body whose position is maximal along the robot forward axis.
  This avoids hard-coding a “fork tip” link name and makes the patch portable across forklift variants.

---

## Files

- `isaaclab_patch/.../forklift_pallet_insert_lift/` : the task (env + config + PPO runner cfg)
- `scripts/install_into_isaaclab.sh` : copies patch into IsaacLab
- `scripts/export_policy_stub.py` : placeholder exporter skeleton

---

## Troubleshooting

1) If the forklift doesn’t move:
- Check joint names printed by:
  ```bash
  ./isaaclab.sh -p scripts/environments/random_agent.py --task Isaac-Forklift-PalletInsertLift-Direct-v0 --num_envs 1
  ```
  Then open Isaac Sim UI once (non-headless) to confirm joint names.
- The default joint names are based on Isaac Sim forklift_c examples: front/back wheel joints + rotator joints + lift_joint.

2) If you see “Environment doesn’t exist”:
- Ensure the import line was added to `isaaclab_tasks/direct/__init__.py`.
- Ensure you are running from the IsaacLab repo (so `isaaclab_tasks` is importable).

