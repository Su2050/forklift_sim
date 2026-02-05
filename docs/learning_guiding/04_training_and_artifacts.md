# 训练与产物：命令怎么写、参数是什么意思、日志和 checkpoint 在哪里

本页适合：已经能跑训练，但想系统了解“训练产物、目录结构、参数含义、怎么续训”的读者。

---

## 1. 训练脚本入口在哪里？

训练脚本（本仓库自带 IsaacLab）：

- `IsaacLab/scripts/reinforcement_learning/rsl_rl/train.py`

它会根据 `--task` 找到任务注册的配置入口，然后启动 PPO 训练。

任务 ID：

- `Isaac-Forklift-PalletInsertLift-Direct-v0`

---

## 2. 一条可直接复制的训练命令（推荐）

```bash
cd <你的IsaacLab目录>

./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Forklift-PalletInsertLift-Direct-v0 \
  --headless \
  --num_envs 128 \
  2>&1 | tee forklift_train.log
```

---

## 3. 关键参数解释（新手最常改的）

### 3.1 `--headless`

- 开启：不弹 UI，速度快，适合训练
- 关闭：会起 Isaac Sim UI，适合调试（但慢很多）

### 3.2 `--num_envs 128`

含义：一次并行跑多少个独立的小环境（相当于 128 辆叉车同时“刷经验”）。\n越大通常采样越快，但更吃 GPU 显存。

如果显存不够，先降到 32 或 64：

```bash
--num_envs 64
```

### 3.3 `--device`

训练通常用 GPU：

- `--device cuda:0`

（不写一般也会默认使用 CUDA；具体取决于你的 IsaacLab 配置与启动参数）

### 3.4 `--experiment_name` / `--run_name`

这两个参数影响日志目录命名：\n训练脚本会把日志写到：

- `logs/rsl_rl/<experiment_name>/<时间戳>_<run_name>/`

其中 `<experiment_name>` 默认来自任务的 PPO 配置：\n叉车任务里是：

- `forklift_pallet_insert_lift`

你也可以在命令行覆盖：

```bash
--experiment_name forklift_pallet_insert_lift
--run_name debug_steerfix
```

### 3.5 `--max_iterations`

训练迭代数（不是物理步数）。想快速冒烟可以先跑小一点：

```bash
--max_iterations 50
```

---

## 4. 日志目录到底在哪？里面有什么？

训练脚本会打印类似：

- `[INFO] Logging experiment in directory: /abs/path/to/IsaacLab/logs/rsl_rl/forklift_pallet_insert_lift`

并在其下创建具体 run 目录：

- `logs/rsl_rl/forklift_pallet_insert_lift/2026-02-02_12-34-56_<可选run_name>/`

run 目录常见内容：

- `params/env.yaml`：环境配置快照（包含 reward 系数、阈值、仿真 dt 等）
- `params/agent.yaml`：算法配置快照（PPO 超参数等）
- `videos/train/`：如果你训练时加了 `--video`，会在这里录训练过程视频
- `model_*.pt`（或类似命名）：checkpoint（模型权重）

> checkpoint 文件名由 rsl-rl 版本决定，但 `play.py` 支持用正则自动匹配最新文件。

---

## 5. checkpoint 怎么选？怎么续训？

### 5.1 “自动选最新 checkpoint”（推荐）

配合 `--load_run` 和 `--checkpoint` 使用正则：\n默认 `.*` 会挑“最新”：

```bash
--load_run ".*" --checkpoint ".*"
```

### 5.2 手动指定某次 run 与某个 checkpoint

你可以指定 `--load_run` 为具体目录名（时间戳那段），`--checkpoint` 为具体文件名。\n例如：

```bash
--resume \
--load_run "2026-02-02_12-34-56_debug_steerfix" \
--checkpoint "model_1000.pt"
```

---

## 6. 训练是否“在进步”？看哪些信号

新手最容易误判：\n强化学习早期 reward 可能是负的、episode 很短都正常。\n更稳妥的观察方式：

- **episode length** 是否随训练变长（能更久不翻车/不超时）
- **成功率** 是否出现（哪怕很低）
- 策略回放时是否出现“对准→插入→抬升”的雏形

如果长期"完全不动/一直翻车"，优先看：`docs/06_troubleshooting.md`

---

## 7. 下一步读什么

- 想回放/录视频/导出模型：`docs/05_evaluation_and_export.md`
- 想给叉车加 RGB 相机、把输入改成图像：`docs/07_vision_input.md`

