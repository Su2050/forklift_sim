# 前置条件与版本建议（避免卡在兼容性/安装）

本页适合：还没完全跑通 Isaac Sim / Isaac Lab / 训练链路的人；或在不同机器（本地工作站 vs DGX Spark）之间切换的人。

---

## 1. 你需要哪些组件？各自是干什么的？

- **Isaac Sim**：负责仿真（物理、场景、渲染、USD 资产加载）
- **Isaac Lab**：负责把仿真包装成“可训练的 RL 环境”，并提供训练脚本、任务注册、并行环境等
- **RSL-RL（PPO）**：负责算法训练（这里用 PPO）

本仓库已经包含一个 `IsaacLab/` 目录（以及自定义叉车任务），你主要需要确保：
你的 Isaac Sim 能被 Isaac Lab 正确找到并启动。

---

## 2. 推荐的运行方式（两条路线）

### 路线 A：直接使用本仓库自带的 `IsaacLab/`

适合：你就在当前仓库里训练，不需要把任务 patch 到另一份 IsaacLab。

- 任务代码路径：`IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/forklift_pallet_insert_lift/`
- 训练脚本：`IsaacLab/scripts/reinforcement_learning/rsl_rl/train.py`

### 路线 B：把任务 patch 安装进你自己的 IsaacLab（例如 DGX Spark 上另一份 IsaacLab）

适合：你有一台独立训练机/集群，想把任务作为“补丁”拷进去。

- patch 脚本：`forklift_pallet_insert_lift_project/scripts/install_into_isaaclab.sh`
- 该脚本会：
  - 拷贝任务目录到 `IsaacLab/source/isaaclab_tasks/.../direct/`
  - 并向 `isaaclab_tasks/direct/__init__.py` 追加 import（用于注册任务）

---

## 3. 版本建议（经验向）

不同机器的“最稳组合”会不同，这里给一个保守建议：

- **DGX Spark（aarch64）**：推荐使用 Spark 官方建议的 Isaac Sim / 驱动组合，并选 Isaac Lab 的 Spark 支持分支
- **普通 x86_64 工作站**：按 Isaac Lab 对应版本的 Isaac Sim 要求来

你可以参考仓库已有的安装推进记录（包含驱动/CUDA/OS 信息与踩坑点）：

- `docs/dgx_spark：isaac_sim_isaac_lab_安装与叉车任务推进记录.md`

---

## 4. 训练前的一次性检查（建议你做）

### 4.1 确认 RSL-RL 已安装

在 IsaacLab 根目录执行（只需一次）：

```bash
./isaaclab.sh -i rsl_rl
```

### 4.2 冒烟测试（可选但强烈推荐）

目的：先确认“Sim + Lab + RL”链路能跑通，再跑叉车任务。
如果你是在 DGX Spark 上，记录文档里已经给了冒烟命令。

---

## 5. 常见环境变量/参数（你可能会用到）

- **`--headless`**：训练推荐开（更快）
- **`--num_envs`**：显存不够就降
- **`LD_PRELOAD=/lib/.../libgomp.so.1`**：某些平台（例如 Spark）可能需要（详见记录文档）

