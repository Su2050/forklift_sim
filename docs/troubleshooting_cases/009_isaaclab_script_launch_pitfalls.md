# IsaacLab 脚本常见踩坑记录

## 1. 必须先退出 conda 环境

**现象**：`ModuleNotFoundError: No module named 'gymnasium'`，日志显示 `Using python from: /home/uniubi/miniconda3/bin/python`

**原因**：`isaaclab.sh` 会检测系统 Python 路径。如果 conda 环境处于激活状态，`isaaclab.sh` 会错误地使用 conda 的 Python 而不是 Isaac Sim 自带的 Python 3.11。conda 中没有安装 `gymnasium` 等 Isaac Sim 依赖。

**解决**：运行任何 `isaaclab.sh` 命令前，先执行 `conda deactivate`。

```bash
conda deactivate
cd /home/uniubi/projects/forklift_sim/IsaacLab
./isaaclab.sh -p your_script.py ...
```

---

## 2. `import isaaclab_tasks` 必须在 AppLauncher 之后

**现象**：`ModuleNotFoundError: No module named 'omni.physics'`

**原因**：`isaaclab_tasks` 的导入链最终会触及 `omni.physics.tensors`，该模块属于 Isaac Sim 运行时，只有在 `AppLauncher` 初始化 Isaac Sim 之后才可用。如果在脚本顶部直接 `import isaaclab_tasks`，此时 Isaac Sim 尚未启动，导致报错。

**解决**：遵循 IsaacLab 推荐的两阶段 import 模式：

```python
# ===== 阶段 1：仅 import 标准库 + argparse + AppLauncher =====
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
# 添加自己的参数 ...
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# ===== 阶段 2：Isaac Sim 已启动，安全 import 一切 =====
import gymnasium as gym
import isaaclab_tasks  # 触发 gym.register()
from isaaclab_tasks.utils import parse_env_cfg
# 其余 import ...
```

**关键规则**：`gymnasium`、`isaaclab_tasks`、`torch`（在 Isaac Sim 环境中）等都应放在 `AppLauncher` 之后 import。

---

## 3. 自定义环境未注册到 gymnasium

**现象**：`gymnasium.error.NameNotFound: Environment 'Isaac-Forklift-PalletInsertLift-Direct' doesn't exist.`

**原因**：IsaacLab 的自定义环境通过 `isaaclab_tasks/__init__.py` 中的 `gym.register()` 注册。如果脚本中没有 `import isaaclab_tasks`，注册不会发生。

此外，自定义环境代码必须先安装到 IsaacLab 源码树中（通过 `install_into_isaaclab.sh`），否则即使 import 了也找不到。

**解决**：两步缺一不可：

```bash
# 步骤 1：安装自定义环境（只需运行一次，除非 IsaacLab 源码被清理）
cd /home/uniubi/projects/forklift_sim/forklift_pallet_insert_lift_project/scripts
bash install_into_isaaclab.sh

# 步骤 2：脚本中确保 import（在 AppLauncher 之后）
import isaaclab_tasks  # noqa: F401
```

---

## 4. 环境创建需要传入 cfg 对象

**现象**：`TypeError: ForkliftPalletInsertLiftEnv.__init__() missing 1 required positional argument: 'cfg'`

**原因**：IsaacLab 环境不支持 `gym.make(task_id)` 的纯字符串方式创建，必须显式传入配置对象 `cfg`。

**解决**：使用 `parse_env_cfg` 生成配置后传入 `gym.make`：

```python
from isaaclab_tasks.utils import parse_env_cfg

env_cfg = parse_env_cfg(args.task, num_envs=args.num_envs, use_fabric=True)
env = gym.make(args.task, cfg=env_cfg, render_mode="rgb_array")
```

---

## 5. TERM=dumb 导致 isaaclab.sh 静默退出

**现象**：`isaaclab.sh` 执行后立即退出，无有用输出，终端显示 `tabs: terminal type 'dumb' cannot reset tabs`

**原因**：`isaaclab.sh` 内部执行 `tabs 4` 设置制表符宽度，当 `TERM=dumb`（常见于 nohup、自动化脚本、某些 IDE 终端）时该命令失败，加上 `set -e`，脚本直接退出。

**解决**：显式设置 `TERM=xterm`：

```bash
TERM=xterm ./isaaclab.sh -p your_script.py --headless ...
```

---

## 6. forklift_expert 模块找不到

**现象**：`ModuleNotFoundError: No module named 'forklift_expert'`

**原因**：`forklift_expert_policy_project` 不在 IsaacLab 的 Python 路径中。

**解决**：运行时通过 `PYTHONPATH` 注入：

```bash
PYTHONPATH=/home/uniubi/projects/forklift_sim/forklift_expert_policy_project:$PYTHONPATH \
./isaaclab.sh -p /path/to/play_expert.py ...
```

---

## 完整运行命令模板

```bash
# 0. 退出 conda
conda deactivate

# 1. 安装自定义环境（首次或 IsaacLab 更新后）
cd /home/uniubi/projects/forklift_sim/forklift_pallet_insert_lift_project/scripts
bash install_into_isaaclab.sh

# 2. 运行 play_expert
cd /home/uniubi/projects/forklift_sim/IsaacLab

PYTHONPATH=/home/uniubi/projects/forklift_sim/forklift_expert_policy_project:$PYTHONPATH \
./isaaclab.sh -p \
  /home/uniubi/projects/forklift_sim/forklift_expert_policy_project/scripts/play_expert.py \
  --task Isaac-Forklift-PalletInsertLift-Direct-v0 \
  --num_envs 1 \
  --headless \
  --video \
  --video_length 600 \
  --video_dir /home/uniubi/projects/forklift_sim/forklift_expert_policy_project/data/videos/expert_play
```
