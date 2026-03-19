# Exp 8.3：轨迹跟踪对象修正（fork_center 替换 root_pos）

**日期**：2026-03-19
**实验分支**：`exp/vision_cnn/exp8_paper_native_refine` (Run: `exp8_3_fork_center_traj`)

## 1. 问题发现

在对 Exp 8.2 进行深度诊断时，通过 ResNet34 特征相似度分析排除了"视觉盲区"假设后，我们用 `check_traj_tangent.py` 脚本发现了一个隐藏极深的**轨迹几何 Bug**。

### Bug 描述
在 `env.py` 的两个关键函数中，轨迹的生成和查询都使用了 **车体中心（`root_pos`）** 作为参考点：

```python
# _build_reference_trajectory: 起点用的是车体中心
p0 = self.robot.data.root_pos_w[env_ids, :2]

# _query_reference_trajectory: 查询点用的也是车体中心
root_pos = self.robot.data.root_pos_w[:, :2]
```

而论文中明确要求的是 **叉臂中心（center of the forks）**：
> "$r_d$ and $r_{cd}$ are the distances from the **center of the forks** to the pallet and clothoid curve"

### Bug 后果

叉尖在车体中心前方 **1.87m**，叉臂中心在车体中心前方 **1.27m**。

我们的轨迹设计中，终点前 1.2m 是直线段（`traj_pre_dist_m = 1.2`），1.2m 以外是 Hermite 曲线段。

当叉尖刚好到达托盘前沿（完美插入位置）时：
- **叉臂中心**在托盘前沿后方 ~0.6m → 处于**直线段**上 → 切线角度 **= 0°** (正确！)
- **车体中心**在托盘前沿后方 ~1.87m → 处于**曲线段**上 → 切线角度 **= -17.35°** (致命错误！)

这就是为什么 Agent 在 Exp 8.2 中偏航角死活卡在 20° 左右：**它在完美地执行一个错误的指令。** 轨迹告诉它"你现在应该偏 17 度"，它照做了，拿满了 $r_{c\psi}$ 奖励，但因为车头是歪的，永远也插不进托盘。

## 2. 修复内容

仅修改了 `env.py` 中的两行代码：

### 修改 1：`_build_reference_trajectory` 的起点
```python
# 修改前
p0 = self.robot.data.root_pos_w[env_ids, :2]  # 车体中心

# 修改后
fork_center = self._compute_fork_center()
p0 = fork_center[env_ids, :2]  # 叉臂中心
```

### 修改 2：`_query_reference_trajectory` 的查询点
```python
# 修改前
root_pos = self.robot.data.root_pos_w[:, :2]  # 车体中心

# 修改后
fork_center = self._compute_fork_center()
query_pos = fork_center[:, :2]  # 叉臂中心
```

## 3. 初步训练结果（前 11 代）

修复后的 Exp 8.3 在刚启动的第 11 代就展现出了惊人的改善：

| 指标 | Exp 8.2 @ 100代 (Bug版) | Exp 8.3 @ 11代 (修复版) | 改善 |
| :--- | :--- | :--- | :--- |
| **偏航角 (`yaw_deg_mean`)** | 15.5° | **7.3°** | 初始偏航角直接减半！ |
| **推盘位移 (`pallet_disp_xy`)** | 0.10 m | **0.03 m** | 几乎不碰托盘 |
| **$r_{c\psi}$ (对齐奖励)** | 5.0 | **8.2** | 对齐奖励翻倍 |
| **$R_{plus}$ (总正奖励)** | 59.9 | **110.6** | 总正奖励翻倍 |

**这说明修复后的轨迹在近端给出的切线方向终于是正确的了。** Agent 自然而然地就把车头摆正了，不需要额外的学习。

## 4. 其他配置（与 Exp 8.2 完全一致）

- 奖励形式：`1/x` + `clip(20)`
- 惩罚权重：$r_p$=0.5, $r_a$=0.1, $r_{ini}$=5.0, $r_{bound}$=0.5
- 参考轨迹：三次 Hermite 样条（Clothoid 近似），终点切线强制对齐托盘
- 动态权重：`dist_front < 0.5m` 时 $\alpha_3$ 放大 3 倍
- 视觉基座：ResNet34（ImageNet 预训练，全程冻结）

## 5. 反思

这个 Bug 完美诠释了"传感器先于算法"的铁律——在奖励函数中，参考轨迹就是 Agent 的"传感器"。如果参考信号给出了错误的方向，再精确的视觉特征、再完美的奖励公式都无法弥补。

我们在 Exp 8.2 中把失败归因于"冻结 ResNet34 的几何认知障碍"，差一点就走上了"引入相对位姿辅助"的弯路。幸亏通过定量的 ResNet34 特征相似度分析排除了视觉假设，才最终找到了这个隐藏极深的几何错误。
