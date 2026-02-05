# lift_joint 真正根因分析（2026-02-05）

> ✅ **本文档记录最终确认的正确根因**。与姊妹文档 `lift_joint_debugging_biography_2026-02-05.md`（错误分析版）形成对比。

---

## 1. 问题回顾

**现象**：
- 按 `R` 发送举升命令
- `set_joint_position_target` 正常调用
- `dof_pos_target` 正确设置
- 但 `lift_joint` 位置始终为 0

---

## 2. 错误的分析路径（之前的结论）

| 阶段 | 假设 | 验证结果 |
|------|------|----------|
| A | 输入/控制链路问题 | ❌ 排除 |
| B | DriveAPI 缺失 | ❌ 排除 |
| C | PhysX API 调用方式错误 | ❌ 排除 |
| D | MassAPI 缺失导致质量为 0 | ❌ **这是错误结论** |

**错误结论**：看到 USD 层 `MassAPI mass=0.0` 就认为 PhysX 质量为 0。

---

## 3. 真正的根因发现过程

### 3.1 添加 PhysX 运行时状态查询

在 `_ensure_robot_masses()` 中直接查询 PhysX：

```python
masses = self.robot.root_physx_view.get_masses()
print(f"[DEBUG] PhysX masses: {masses}")
```

### 3.2 关键日志输出

```
[DEBUG] PhysX masses shape: torch.Size([1, 8]), values: 
  tensor([[6287, 269, 269, 2.98, 2.98, 112, 135, 135]])
[DEBUG] PhysX masses 全部大于 0，无需修复
```

**发现**：
- PhysX 的 masses **不是 0**！
- body = 6287 kg，lift = 112 kg
- PhysX 根据 `density=3000` 和碰撞体体积**自动计算**了正确的 mass

### 3.3 那为什么不动？做力学分析

**lift 重力**：
```
F_gravity = m × g = 112 × 9.8 = 1098 N
```

**drive 产生的力**（spring model）：
```
F_drive = stiffness × (target - current)
        = 5000 × (0.01667 - 0)
        = 83.35 N
```

**力平衡**：
```
83 N << 1098 N
```

**结论**：**drive 产生的力远小于重力，lift 根本抬不起来！**

---

## 4. 真正的根因

### 🎯 根因：`stiffness` 参数太小

- lift_joint 的 `stiffness=5000` 
- 当 `target - current = 0.017m` 时，只产生 83N 的力
- 但 lift 的重力是 1098N
- **力不够，无法克服重力**

### 为什么 `dof_force` 显示为 0？

这是因为 PhysX drive 的力计算方式：
```
force = stiffness × position_error + damping × velocity_error
```

当 position 卡在 lower_limit (0) 时：
- position_error 很小（0.017m）
- velocity = 0（因为卡住了）
- 计算出的力 ≈ 83N，但这个力无法让 position 增加
- PhysX 可能在某些情况下不报告这个"无效力"

---

## 5. 解决方案

### 修改 `env_cfg.py` 中的 lift actuator 配置：

**Before**：
```python
"lift": ImplicitActuatorCfg(
    joint_names_expr=["lift_joint"],
    velocity_limit_sim=1.0,
    effort_limit_sim=10000.0,
    stiffness=5000.0,      # 太小！
    damping=1000.0,
),
```

**After**：
```python
"lift": ImplicitActuatorCfg(
    joint_names_expr=["lift_joint"],
    velocity_limit_sim=1.0,
    effort_limit_sim=50000.0,   # 提高力矩限制
    stiffness=200000.0,         # 提高 40 倍！
    damping=10000.0,            # 提高阻尼减少振荡
),
```

### 计算验证：

需要的最小 stiffness：
```
stiffness × Δx > F_gravity
stiffness > 1098 / 0.017 = 64,600
```

设置 `stiffness=200000` 确保有足够余量。

---

## 6. 核心教训

### 6.1 USD 配置 ≠ PhysX 运行时状态

| 层级 | 显示值 | 实际含义 |
|------|--------|----------|
| USD MassAPI | mass=0.0, density=3000.0 | 让 PhysX 自动计算 |
| PhysX 运行时 | mass=112kg | 根据 density×volume 计算出的实际值 |

**教训**：查 PhysX API（`get_masses()`），不要只看 USD 配置。

### 6.2 遇到"力已施加但不动"→ 先做力学分析

**正确的排查顺序**：
1. 查 PhysX 实际 mass（`get_masses()`）
2. 计算重力 / 摩擦力 / 外力
3. 计算 drive 能产生的力
4. 比较是否足够

**如果一开始就这样做**：
```
lift 重力 = 112 × 9.8 = 1098 N
drive 力 = 5000 × 0.017 = 85 N
85 << 1098 → 力不够！→ 增加 stiffness
```
**一步到位，不需要 30+ 次调试**。

### 6.3 简单原因优先

在深挖 MassAPI、DriveAPI、PhysX API 之前，应该先检查：
- 参数是否合理？
- 力是否足够？
- 单位是否正确？

---

## 7. 诊断清单（下次遇到类似问题）

```
[ ] 1. 查 PhysX 实际 masses: root_physx_view.get_masses()
[ ] 2. 计算外力（重力 = m × g）
[ ] 3. 计算 drive 力（stiffness × position_error）
[ ] 4. 比较：drive 力 > 外力？
[ ] 5. 如果不够，调整 stiffness / effort_limit
```

---

## 8. 总结

| 项目 | 错误分析 | 正确分析 |
|------|----------|----------|
| 根因 | MassAPI 缺失 | stiffness 太小 |
| 依据 | USD 配置 mass=0 | PhysX 实际 mass=112kg |
| 解决 | 补 MassAPI | 增加 stiffness |
| 耗时 | 30+ 次调试 | 理论上 1-2 步 |

**最终修复**：`stiffness: 5000 → 200000`
