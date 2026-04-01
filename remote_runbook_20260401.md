# 远程主机运行手册 2026-04-01

这份文件记录了远程主机 `uniubi@192.168.60.229` 上，当前 `exp/exp9_0` 分支已经核对过、并且可用的运行方式。

## 关键路径

- 当前代码仓库：`/home/uniubi/projects/forklift_sim_exp9`
- 复用的 IsaacLab 运行目录：`/home/uniubi/projects/forklift_sim/IsaacLab`
- 当前主机可用 conda 环境：`/home/uniubi/miniconda3/envs/env_isaaclab`
- 训练脚本会同步资产到：`/home/uniubi/projects/forklift_sim/assets`

## 先看这几条

- 不要直接在登录后的 `base` 环境里跑命令。
- 这台机器当前不是走 `IsaacLab/_isaac_sim/python.sh`，而是走 `env_isaaclab`。
- `/home/uniubi/projects/forklift_sim_exp9` 里的大 USD 资产不受 git 管理；只做 `git clone` 或 `git pull` 不会自动获得这些文件。当前远端之所以有 `assets/`，是因为训练启动脚本会先把 `/home/uniubi/projects/forklift_sim_exp9/assets/` 同步到 `/home/uniubi/projects/forklift_sim/assets/`。
- 根目录目前没有 `README.md`，这份文件就是这台机器的入口说明。

## 1. 激活环境

```bash
source /home/uniubi/miniconda3/bin/activate /home/uniubi/miniconda3/envs/env_isaaclab
```

可选自检：

```bash
cd /home/uniubi/projects/forklift_sim/IsaacLab
TERM=xterm bash ./isaaclab.sh -p -c "import isaaclab, isaacsim, importlib.metadata as m; print('ok'); print(m.version('rsl-rl-lib'))"
```

## 2. 启动 exp9.0 训练

默认启动方式：

```bash
source /home/uniubi/miniconda3/bin/activate /home/uniubi/miniconda3/envs/env_isaaclab
cd /home/uniubi/projects/forklift_sim_exp9
bash scripts/run_exp90_no_reference_baseline_remote.sh
```

常用变体：

```bash
source /home/uniubi/miniconda3/bin/activate /home/uniubi/miniconda3/envs/env_isaaclab
cd /home/uniubi/projects/forklift_sim_exp9
SEED=43 bash scripts/run_exp90_no_reference_baseline_remote.sh
```

```bash
source /home/uniubi/miniconda3/bin/activate /home/uniubi/miniconda3/envs/env_isaaclab
cd /home/uniubi/projects/forklift_sim_exp9
SEED=44 NUM_ENVS=32 MAX_ITERATIONS=100 bash scripts/run_exp90_no_reference_baseline_remote.sh
```

这个远程专用脚本会自动做四件事：

- 激活 `env_isaaclab`
- 把当前仓库的 `assets/` 同步到 `/home/uniubi/projects/forklift_sim/assets`
- 把当前 `forklift_sim_exp9` 的 patch 安装进旧的 `/home/uniubi/projects/forklift_sim/IsaacLab`
- 在复用的 IsaacLab 目录里启动训练

## 3. 查看训练进度

当前已经实际启动并检查过的一次运行：

- 运行名：`exp9_0_no_reference_master_init_seed42_iter400`
- 包装日志：`/home/uniubi/projects/forklift_sim_exp9/logs/20260401_151243_train_exp9_0_no_reference_master_init_seed42_iter400.log`
- IsaacLab 运行目录：`/home/uniubi/projects/forklift_sim/IsaacLab/logs/rsl_rl/forklift_pallet_insert_lift/2026-04-01_15-12-49_exp9_0_no_reference_master_init_seed42_iter400`

持续看日志：

```bash
tail -f /home/uniubi/projects/forklift_sim_exp9/logs/20260401_151243_train_exp9_0_no_reference_master_init_seed42_iter400.log
```

查看训练进程：

```bash
pgrep -af "scripts/reinforcement_learning/rsl_rl/train.py.*exp9_0_no_reference_master_init_seed42_iter400"
```

查看当前 run 已经写出的 checkpoint：

```bash
find /home/uniubi/projects/forklift_sim/IsaacLab/logs/rsl_rl/forklift_pallet_insert_lift/2026-04-01_15-12-49_exp9_0_no_reference_master_init_seed42_iter400 -maxdepth 1 -type f -name 'model_*.pt' | sort
```

## 4. play 回放 / 录视频

下面这条命令的语法已经在远程机上对过，当前这个 run 也已经确认有 `model_250.pt` 等 checkpoint。

```bash
source /home/uniubi/miniconda3/bin/activate /home/uniubi/miniconda3/envs/env_isaaclab
cd /home/uniubi/projects/forklift_sim/IsaacLab
TERM=xterm bash ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
  --task Isaac-Forklift-PalletInsertLift-Direct-v0 \
  --num_envs 1 \
  --headless \
  --video --video_length 300 \
  --load_run "2026-04-01_15-12-49_exp9_0_no_reference_master_init_seed42_iter400" \
  --checkpoint "model_250.pt"
```

视频输出目录通常在：

```bash
/home/uniubi/projects/forklift_sim/IsaacLab/logs/rsl_rl/forklift_pallet_insert_lift/2026-04-01_15-12-49_exp9_0_no_reference_master_init_seed42_iter400/videos/play
```

## 5. 参考轨迹可视化

这条命令已经在远程主机上实际运行成功，能生成 PNG 和 manifest JSON：

```bash
source /home/uniubi/miniconda3/bin/activate /home/uniubi/miniconda3/envs/env_isaaclab
cd /home/uniubi/projects/forklift_sim_exp9
python forklift_pallet_insert_lift_project/scripts/visualize_reference_trajectory_cases.py \
  --output-dir outputs/reftraj_$(date +%Y%m%d_%H%M%S)
```

## 6. 验证 / sanity check

这条验证命令路径已经在远程机上核对通过：

```bash
source /home/uniubi/miniconda3/bin/activate /home/uniubi/miniconda3/envs/env_isaaclab
cd /home/uniubi/projects/forklift_sim/IsaacLab
TERM=xterm bash ./isaaclab.sh -p /home/uniubi/projects/forklift_sim_exp9/scripts/validation/success/verify_forklift_insert_lift.py \
  --sanity-check \
  --headless \
  --enable_cameras
```

手动控制模式：

```bash
source /home/uniubi/miniconda3/bin/activate /home/uniubi/miniconda3/envs/env_isaaclab
cd /home/uniubi/projects/forklift_sim/IsaacLab
TERM=xterm bash ./isaaclab.sh -p /home/uniubi/projects/forklift_sim_exp9/scripts/validation/success/verify_forklift_insert_lift.py \
  --manual --auto-align
```

## 7. 常见问题

如果训练一启动就失败，优先检查：

- 当前是不是在 `env_isaaclab`
- `/home/uniubi/projects/forklift_sim/assets/pallet_com_shifted.usd` 是否存在
- 直接重新执行：

```bash
cd /home/uniubi/projects/forklift_sim_exp9
bash scripts/run_exp90_no_reference_baseline_remote.sh
```

如果 `isaaclab.sh` 报：

```bash
tabs: terminal type 'dumb' cannot reset tabs
```

先执行：

```bash
export TERM=xterm
```

远程训练启动脚本已经自动处理了这一点，但手动运行 `play` / `verify` 时依然建议保留。
