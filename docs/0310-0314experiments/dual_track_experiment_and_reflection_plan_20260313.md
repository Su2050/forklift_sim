---
name: 论文原生方法复现计划 (去过度设计版)
overview: 砍掉所有过度设计的补丁（复杂门控、退火课程、空间衰减），回归论文最本真的方法：基于参考轨迹的全局一致奖励 + 宽广随机初始分布。
todos:
  - id: cleanup-overdesign
    content: 砍掉代码中所有复杂的阶段划分、软门控、退火课程和空间衰减等过度设计补丁。
    status: pending
  - id: implement-reference-trajectory
    content: 严格复现论文中的参考轨迹（如 Clothoid 曲线近似），并正确计算 r_cd（距离轨迹误差）和 r_cpsi（轨迹切线角度误差）。
    status: pending
  - id: restore-random-initialization
    content: 废弃“喂到嘴边”的极窄出生点，恢复论文中类似图 4a 的宽广随机初始分布（如 1.8m x 1.8m 区域，合理的偏航角随机）。
    status: pending
  - id: verify-rrl-baseline
    content: 确认基座严格锁定在 ResNet18 + ImageNet 预训练权重 + 全程冻结，不进行任何微调。
    status: pending
  - id: run-end-to-end-training
    content: 在上述干净、原生的配置下，启动端到端 PPO 训练，并监控参考轨迹奖励的引导效果。
    status: pending
isProject: false
---

# 论文原生方法复现计划 (去过度设计版)

## 1. 目标

- 彻底摒弃之前的“面多加水、水多加面”的过度设计（如复杂的阶段划分、软门控、退火课程、空间衰减、核弹早停等）。
- 严格按照论文《Visual-Based Forklift Learning System Enabling Zero-Shot Sim2Real Without Real-World Data》的方法，用最简单的全局一致奖励和宽广随机初始分布，实现端到端的泛化插入。

## 2. 核心行动点 (Action Items)

### 2.1 砍掉所有过度设计的“补丁”
- **移除阶段化门控**：删除代码中所有类似于 `w_align`、`w_band` 等人为设计的阶段切换和软门控逻辑。
- **废弃退火课程**：不再玩“收紧 $r_g$ 阈值”的游戏，直接将 $r_g$ 阈值设定为物理意义上真正的“插入成功”标准（如 0.1m 或 0.15m）。
- **废弃核弹奖励**：将 $r_g$ 的权重恢复到合理水平（如论文中隐含的与 Shaping 奖励同量级的权重，而不是 5000 分）。
- **废弃空间衰减**：不再人为干预姿态奖励在远场的权重。

### 2.2 严格复现“参考轨迹 (Reference Trajectory)”
- 论文成功的核心在于使用参考轨迹（如 Clothoid 曲线近似）来引导 Agent，而不是直接惩罚它与托盘中心线的绝对误差。
- **计算 $r_{cd}$**：叉臂中心到参考轨迹的最短距离。
- **计算 $r_{c\psi}$**：叉臂朝向与参考轨迹在最近点处切线的角度差。
- **奖励形式**：严格使用 $R^+ = \alpha_1 \frac{1}{r_d} + \alpha_2 \frac{1}{r_{cd}} + \alpha_3 \frac{1}{r_{c\psi}} + \alpha_4 r_g$（如果 $1/x$ 导致数值爆炸，可以退而求其次使用 $exp(-x/\sigma)$，但必须保持轨迹引导的本质）。

### 2.3 恢复宽广的随机初始分布
- 废弃 Exp 5.1 中“喂到嘴边”的极窄出生点（横向 ±5cm，偏航 ±2°）。
- 根据论文图 4a，恢复一个合理的、宽广的初始随机区域（例如：距离托盘 1.5m~3.0m，横向偏差 ±0.5m~1.0m，偏航角 ±15°~30°）。
- 目标是让 Agent 真正学会在接近过程中自主寻的并顺着轨迹调整姿态。

### 2.4 锁定 RRL 视觉基座
- 确保代码严格使用 **ResNet18 + ImageNet 预训练权重 + 全程冻结 (`freeze_backbone=True`)**。
- 坚决不进行任何形式的 Backbone 微调，将物理理解的任务完全交给下游的 Actor/Critic MLP。

## 3. 实验执行顺序

1. **代码清理与重构**：在当前分支（或新建一个纯净分支）中，删掉所有旧的势函数、门控和课程代码，只保留最基础的物理计算和论文原生的 Reward 公式。
2. **参考轨迹可视化与验证**：在 Isaac Sim 中画出生成的参考轨迹，确保 $r_{cd}$ 和 $r_{c\psi}$ 的计算在几何上是绝对正确的。
3. **基线端到端训练**：在宽广初始分布下，使用标准 PPO 启动训练。
4. **观察与复盘**：观察 Agent 是否能被参考轨迹有效引导。如果失败，复盘的重点应放在“轨迹生成是否合理”、“奖励权重（$\alpha_1$ 到 $\alpha_8$）是否平衡”，而不是去加新的门控补丁。