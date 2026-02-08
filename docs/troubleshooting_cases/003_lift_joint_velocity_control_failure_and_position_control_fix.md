# 003: lift_joint 速度控制完全失效 — 从诊断到位置控制修复全过程

- **日期**: 2026-02-08
- **严重程度**: Critical（阻塞举升功能，RL 训练无法完成最终目标）
- **涉及文件**: `env.py`, `env_cfg.py`
- **关联**: [002_lift_joint_locked_by_usd_driveapi_stiffness.md](002_lift_joint_locked_by_usd_driveapi_stiffness.md)

---

## 1. 问题描述

lift_joint（货叉升降棱柱关节）在使用 `set_joint_velocity_target()` + `stiffness=0`（速度控制模式）时 **完全无响应**：
- `joint_pos` 恒为 0.0000m
- `joint_vel` 恒为 0.0000 rad/s
- 托盘未被举升

该问题导致 RL 训练无法完成"插入 → 举升 → 保持"的最终目标。

---

## 2. 排查过程

### 2.1 第一轮：DriveAPI 时机修复（未解决）

**假设**: USD 文件中 `lift_joint` 的 DriveAPI 原始 `stiffness=100000` 锁死了关节。

**操作**: 
- 在 `_setup_scene()` 中 `clone_environments()` 之前调用 `_fix_lift_joint_drive()`，将 USD DriveAPI 参数覆盖为 `stiffness=0, damping=10000, maxForce=50000`
- 确保 PhysX 在 `sim.reset()` 时 bake 到正确值

**验证结果**:
```
[lift_drive] 已覆盖 ... stiffness: 100000.0 → 0.0
```
日志确认时机正确，但 **lift_pos 仍为 0.0000m**。

### 2.2 第二轮：提高 ImplicitActuatorCfg damping（未解决）

**发现**: Isaac Lab 的 `ImplicitActuator` 在 `sim.reset()` 时会用 `ImplicitActuatorCfg` 的 stiffness/damping **覆盖** USD DriveAPI 值（源码确认：`articulation.py` 第 1727-1731 行）。

```python
if isinstance(actuator, ImplicitActuator):
    self.write_joint_stiffness_to_sim(actuator.stiffness, ...)
    self.write_joint_damping_to_sim(actuator.damping, ...)
```

原配置 `damping=1000` 产生的力仅 `1000 × 0.125 = 125N`，可能不足以克服重力。

**操作**: 将 `env_cfg.py` 中 lift damping 从 1000 提高到 10000，effort_limit 从 5000 提高到 50000。

**验证结果**: lift_pos 仍然恒为 0。

### 2.3 第三轮：发现 joint 数据全零异常

**关键观察**: 验证脚本中 **轮子关节速度也报告为 0**，但叉车物理上明显在前进（位置从 x=-3.5 移动到 x=-2.64）。

```
实际速度: 前轮=0.0000 rad/s, 后轮=0.0000 rad/s
位置变化: (0.0091, 0.0000, -0.0000)  ← 叉车在动！
```

这说明 `self.robot.data.joint_pos` 和 `self.robot.data.joint_vel` 的数据缓冲区 **根本没有被刷新**，而非 lift 驱动问题。但此时仍无法确定 lift 在 PhysX 层面是否真的在动。

### 2.4 第四轮：回溯 32 份实验日志（关键突破）

**方法**: 回溯 `docs/logs/` 目录下 32 份实验日志和对话历史，寻找 2 月 5 日下午成功举升的证据。

**关键发现 — logs32（2/5 13:35，唯一通过 drive 系统成功举升的实验）**:

```
[TRACE] lab.joint_pos_target(lift)=0.07174    ← 位置目标递增
[TRACE] lab.joint_vel_target(lift)=0.00000    ← 速度目标为 0！
[TRACE] lab.joint_pos(lift)=0.05652           ← 位置正确读取（非零）
[TRACE] lab.joint_vel(lift)=0.20758           ← 速度正确读取（非零）
[TRACE] physx.stiffness=200000.00, damping=10000.00, max_force=50000.00
```

logs32 使用的是 **位置控制**（`stiffness=200000` + `set_joint_position_target()` + 累积位置目标），而非速度控制。lift 稳定升至 0.23m，速度 ~0.207 m/s。

**对照**: 所有使用速度控制（`stiffness=0` + `set_joint_velocity_target()`）的 31 次实验均失败。

**结论**: `set_joint_velocity_target()` 对该棱柱关节完全不起作用；只有位置控制被证明可行。

另外发现 2/5 晚间的 R 键成功举升用的是 `write_joint_state_to_sim()`——直接写 PhysX 状态绕过 drive 系统，不适用于 RL 训练。

### 2.5 第五轮：实施位置控制 + PhysX 直读诊断

**操作**:
1. `env_cfg.py`: `stiffness=200000.0`（位置控制）
2. `env.py` `_apply_action()`: 从 `set_joint_velocity_target` 切换为累积 `set_joint_position_target`
3. 添加 PhysX view 直读诊断

**验证结果（第一次位置控制测试）**:

```
[DIAG step=140] physx.dof_pos(lift)=0.09154, lab.joint_pos=0.00000, lift_target=0.11250
[DIAG step=200] physx.dof_pos(lift)=0.34136, lab.joint_pos=0.00000, lift_target=0.36250
[DIAG step=280] physx.dof_pos(lift)=0.67405, lab.joint_pos=0.00000, lift_target=0.69583
```

- `physx.dof_pos(lift)` 正常递增（**lift 物理上已在工作！**）
- `lab.joint_pos` 仍为 0（Isaac Lab 数据缓冲区不更新）
- `pallet_z` 从 0.005m 升到 0.48m（托盘被物理举起）

### 2.6 第六轮：修复数据缓冲区（最终解决）

**根因确认**: `self.robot.data.joint_pos`（Isaac Lab `ArticulationData`）在 Fabric clone 失败时不会被 `scene.update()` 刷新，恒为初始值 0。但 `root_physx_view.get_dof_positions()` 能正确返回实际值。

**操作**:
1. `__init__()`: 将 `_joint_pos`/`_joint_vel` 从 `robot.data` 引用改为独立张量
2. `_get_rewards()` 和 `_get_observations()` 开头：从 PhysX view 手动刷新

```python
self._joint_pos[:] = self.robot.root_physx_view.get_dof_positions()
self._joint_vel[:] = self.robot.root_physx_view.get_dof_velocities()
```

注意 `_get_rewards()` 在 DirectRLEnv 的 `step()` 中先于 `_get_observations()` 被调用，两处都需要刷新。

**验证结果（最终）**:

```
[DIAG step=140] physx.dof_pos(lift)=0.09154, lab.joint_pos=0.09154  ← 一致！
步数 0:   lift_pos=0.0003m
步数 40:  lift_pos=0.1498m
步数 80:  lift_pos=0.3164m
步数 120: lift_pos=0.4828m
步数 160: lift_pos=0.6491m
最终:     lift_pos=0.8110m ✅
```

举升测试通过，测试结果从 5/8 提升到 6/8。

---

## 3. 根因总结

实际上存在 **两个独立的 bug 叠加**:

| # | Bug | 表现 | 影响范围 |
|---|-----|------|----------|
| 1 | `set_joint_velocity_target()` 对 lift_joint 棱柱关节无效 | 速度控制模式下关节完全不动 | 仅 lift_joint |
| 2 | Fabric clone 失败导致 `robot.data.joint_pos/vel` 不更新 | 所有关节数据恒为 0 | 所有关节 |

Bug 1 导致 lift 物理上不动；修复后 Bug 2 掩盖了物理上已经工作的事实（读回的数据仍为 0），误导排查方向。

---

## 4. 最终修复方案

### 4.1 env_cfg.py — 位置控制

```python
"lift": ImplicitActuatorCfg(
    joint_names_expr=["lift_joint"],
    velocity_limit=1.0,
    effort_limit=50000.0,
    stiffness=200000.0,    # 位置控制（logs32 验证值）
    damping=10000.0,       # 阻尼（logs32 验证值）
),
```

### 4.2 env.py — _apply_action() 累积位置目标

```python
# 替代原来的 set_joint_velocity_target
self._lift_pos_target += lift_v * self.cfg.sim.dt  # 每物理子步累积
self._lift_pos_target = torch.clamp(self._lift_pos_target, 0.0, 2.0)
self.robot.set_joint_position_target(
    self._lift_pos_target.unsqueeze(-1), joint_ids=[self._lift_id]
)
```

关键细节：`_apply_action()` 每 env step 被调用 `decimation`（=4）次，必须用 `sim.dt`（单子步 dt）而非 `step_dt`。

### 4.3 env.py — PhysX view 直读刷新关节数据

```python
# 在 _get_rewards() 和 _get_observations() 开头
self._joint_pos[:] = self.robot.root_physx_view.get_dof_positions()
self._joint_vel[:] = self.robot.root_physx_view.get_dof_velocities()
```

### 4.4 env.py — _reset_idx() 重置 lift target

```python
self._lift_pos_target[env_ids] = 0.0
```

### 4.5 env.py — _fix_lift_joint_drive() USD DriveAPI

```python
drive_api.GetStiffnessAttr().Set(200000.0)  # 位置控制
drive_api.GetDampingAttr().Set(10000.0)
drive_api.GetMaxForceAttr().Set(50000.0)
```

---

## 5. 验证结果

| 测试项 | 修复前 | 修复后 |
|--------|--------|--------|
| 环境初始化 | ✅ | ✅ |
| 接近测试 | ✅ | ✅ |
| 对齐测试 | ✅ | ✅ |
| 插入测试 | ✅ | ✅ |
| **举升测试** | **❌ lift_pos=0.0000m** | **✅ lift_pos=0.8110m** |
| 总通过数 | 5/8 | 6/8 |

举升关键指标:
- lift_pos: 0 → 0.811m
- fork_tip_z: -0.0004m → 0.811m  
- pallet_z: 0.005m → 0.480m（托盘被物理举起）
- 举升速度: ~0.15 m/s（与 logs32 的 0.207 m/s 同量级）

---

## 6. 经验教训

1. **速度控制 vs 位置控制**: 对于 PhysX 棱柱关节，`set_joint_velocity_target()` 可能完全不起作用。应优先验证 `set_joint_position_target()` + 高刚度（stiffness=200000）模式
2. **数据缓冲区可能失效**: Fabric clone 失败时 `robot.data.joint_pos` 不会被更新，但 `root_physx_view.get_dof_positions()` 仍然可靠。遇到"关节数据全零但物理在动"时应直接查 PhysX view
3. **两个 bug 叠加会严重误导排查**: Bug 2（数据全零）掩盖了 Bug 1 的修复进展。加入 PhysX 直读诊断是打破僵局的关键
4. **历史日志是宝藏**: 回溯 32 份实验日志找到 logs32（唯一成功的 drive 举升），其 TRACE 数据直接指出了位置控制 + stiffness=200000 的正确配置
5. **注意 `_apply_action()` 的调用频率**: 它每 env step 被调用 `decimation` 次（每物理子步一次），累积量必须用 `sim.dt` 而非 `step_dt`
