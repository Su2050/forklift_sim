# 快速开始：10 分钟跑通训练与回放

本页适合：**第一次接触 Isaac Lab + 强化学习**，只想先跑起来的人。

你将得到：训练日志、checkpoint、（可选）回放视频。

---

## 0. 你需要准备什么（最少前置）

- **一台有 NVIDIA GPU 的 Linux 机器**（推荐），或 DGX Spark
- 已安装并能运行的 **Isaac Lab**（本仓库里已包含 `IsaacLab/`）
- 已安装 `rsl_rl` 组件（如果没装，下面会给命令）

如果你在 DGX Spark 上安装 Isaac Sim/Isaac Lab 的过程不熟，请先看：
- `docs/dgx_spark：isaac_sim_isaac_lab_安装与叉车任务推进记录.md`
以及：
- `docs/00_prereqs_and_versions.md`

---

## 1. 任务是什么（一句话）

训练叉车完成：**对准托盘 → 把货叉插入托盘 → 抬升托盘**（不包含导航/搬运/放置）。

任务 ID（训练与回放都用它）：

- `Isaac-Forklift-PalletInsertLift-Direct-v0`

---

## 2. 确认任务已经“注册”进 IsaacLab

这个任务是在 `isaaclab_tasks` 里注册的。正常情况下，你会通过 patch 脚本把任务拷贝进 IsaacLab 并自动加一行 import。

如果你是在本仓库自带的 `IsaacLab/` 目录里训练（不是外部 Spark 环境），可以先检查任务目录是否存在：

- `IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/forklift_pallet_insert_lift/`

并检查这个 import 是否存在：

- `IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/__init__.py` 里包含 `forklift_pallet_insert_lift`

如果你是“把 patch 安装进另一份 IsaacLab”，请在 patch 仓库里执行：

```bash
bash scripts/install_into_isaaclab.sh /path/to/IsaacLab
```

---

## 3. 安装 PPO 训练依赖（只需一次）

在 IsaacLab 根目录执行：

```bash
cd <你的IsaacLab目录>
./isaaclab.sh -i rsl_rl
```

---

## 4. 开始训练（headless 推荐）

在 IsaacLab 根目录执行：

```bash
cd <你的IsaacLab目录>

./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Forklift-PalletInsertLift-Direct-v0 \
  --headless \
  --num_envs 128 \
  2>&1 | tee forklift_train.log
```

### 你应该看到什么（验证检查点）

- 终端输出类似：
  - `[INFO] Logging experiment in directory: .../logs/rsl_rl/forklift_pallet_insert_lift`
- 训练过程中会创建目录（位置在 IsaacLab 根目录下）：
  - `logs/rsl_rl/forklift_pallet_insert_lift/<时间戳>_<可选run_name>/`
- 该目录下会出现：
  - `params/env.yaml`、`params/agent.yaml`（本次运行配置快照）
  - 若干 checkpoint 文件（文件名由 rsl-rl 版本决定，常见是 `model_XXXX.pt` 一类）

如果没有产生 `logs/` 或者报 “Environment doesn’t exist”，直接跳到：
- `docs/06_troubleshooting.md`

---

## 5. 回放/评估 + 录视频（可选）

当你已经有 checkpoint 后，在 IsaacLab 根目录执行：

```bash
cd <你的IsaacLab目录>

./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
  --task Isaac-Forklift-PalletInsertLift-Direct-v0 \
  --num_envs 16 \
  --headless \
  --video --video_length 300 \
  --load_run ".*" \
  --checkpoint ".*"
```

说明：

- `--load_run` 和 `--checkpoint` **支持正则**，默认用 `.*` 会自动选择“最新”的一次 run 与其中“最新”的 checkpoint。
- 输出视频通常在（相对 IsaacLab 根目录）：
  - `logs/rsl_rl/forklift_pallet_insert_lift/<run>/videos/play/`

---

## 6. 下一步读什么

- 想看懂“到底在训练什么”：`docs/02_task_overview.md`
- 想理解动作/观测/奖励：`docs/03_task_design_rl.md`
- 想掌握日志/产物目录与参数：`docs/04_training_and_artifacts.md`

