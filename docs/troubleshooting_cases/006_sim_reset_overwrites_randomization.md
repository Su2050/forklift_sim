# 006: sim.reset() 全局重置覆盖环境随机化 — 策略完全无法学习

> **日期**：2026-02-09
> **影响版本**：S1.0h（该 bug 在 S0.7 postmortem 中已记录，但修复未迁移到 S1.0h）
> **影响文件**：`env.py` `_reset_idx()` 方法，`env_cfg.py` `observation_space`
> **严重程度**：致命 — 策略完全无法学习

---

## 现象

S1.0h 训练 163 iterations（约 10 小时），策略无任何学习迹象：

| 指标 | 期望值 | 实际值 | 异常程度 |
|------|--------|--------|----------|
| lateral_mean | ~0.3m（±0.6m 随机化） | **0.0000** | 致命 |
| yaw_deg_mean | ~7.2°（±14.3° 随机化） | **~0.0003** | 致命 |
| dist_front_mean | 波动 | **0.5533（恒定）** | 致命 |
| action_noise_std | ~3.0（初始） | **0.07（崩溃）** | 致命 |
| Mean episode length | 逐步增长 | **359.00（锁死在 max）** | 致命 |
| steps/s | >10,000 | **~300-1000** | 严重 |

## 根因：数学证明

`_reset_idx()` 中第 1161-1163 行：

```python
self.scene.write_data_to_sim()   # 写 ALL 1024 环境的 actuator 数据
self.sim.reset()                  # 全局 PhysX 引擎重置！
self.scene.update(self.cfg.sim.dt)
```

`self.sim.reset()` 不是逐环境操作，而是**完整的 PhysX 引擎重初始化**。它将所有 1024 个环境的位姿覆盖回 `env_cfg.py` 的默认 `init_state`：

- 叉车默认位置：`(-3.5, 0.0, 0.03)`
- 托盘默认位置：`(0.0, 0.0, 0.15)`

由此计算：

```
tip_x = -3.5 + 1.8667(fork_forward_offset) = -1.6333
pallet_front_x = 0.0 - 2.16 * 0.5 = -1.08
dist_front = -1.08 - (-1.6333) = 0.5533   ← 精确匹配日志！
lateral = |0.0 - 0.0| = 0.0000             ← 精确匹配日志！
yaw_deg = 0.0                               ← 精确匹配日志！
```

**三个指标全部精确匹配默认 init_state 的计算值**，证明 `sim.reset()` 覆盖了 `_reset_idx()` 中写入的所有随机化位姿。

## 因果链

```
_reset_idx() 写入随机位姿（x ∈ [-4.0, -2.5], y ∈ [-0.6, 0.6], yaw ∈ [-0.25, 0.25]）
    ↓
sim.reset() 全局 PhysX 引擎重初始化
    ↓
所有 1024 个环境的位姿被覆盖回 config 默认值 (-3.5, 0.0, 0.03)
    ↓
lateral_mean = 0, yaw_deg_mean = 0, dist_front_mean = 0.5533（恒定）
    ↓
策略看到的所有环境状态完全相同（零多样性）
    ↓
action_noise_std 快速崩溃（所有环境返回相同梯度信号）
    ↓
策略学到"什么都不做"→ episode 全部超时 → 训练无效
```

## 附带问题

### sim.reset() 的性能影响

每次 `sim.reset()` 约 **75ms**（触发完整 PhysX 引擎重初始化 + 碰撞网格重解析）。
在 1024 环境、64 步/iteration 的配置下，`_reset_idx()` 每步至少调用一次，导致每 iteration 额外 ~5 秒开销。

### observation_space 配置不匹配

`env_cfg.py` 声明 `observation_space = 14`，但 `_get_observations()` 实际返回 13 维：
- d_xy_r(2) + cos/sin_dyaw(2) + v_xy_r(2) + yaw_rate(1) + lift_pos(1) + lift_vel(1) + insert_norm(1) + actions(3) = **13**

RSL-RL 从实际 tensor shape 推断网络输入维度，所以训练不会崩溃，但 Gymnasium spec 不一致。

### 冗余的 robot.reset()

`_reset_idx()` 末尾的 `self.robot.reset(env_ids)` 是冗余的 — `super()._reset_idx()` 已通过 `scene.reset(env_ids)` 调用过一次。

## 修复

### env.py `_reset_idx()`

```python
# 删除三行：
# self.scene.write_data_to_sim()
# self.sim.reset()
# self.scene.update(self.cfg.sim.dt)

# 替换 fork_tip_z0 初始化（_fork_z_base=0.0，误差为零）：
# 之前：tip = self._compute_fork_tip(); self._fork_tip_z0[env_ids] = tip[:, 2][env_ids]
# 之后：
self._fork_tip_z0[env_ids] = z.squeeze(-1)

# 简化 lift_height_reset（恒等于 0）：
# 之前：lift_height_reset = tip[:, 2][env_ids] - self._fork_tip_z0[env_ids]
# 之后：
self._last_lift_pos[env_ids] = 0.0

# 删除冗余 robot.reset()：
# self.robot.reset(env_ids)  ← super()._reset_idx() 已调用
```

### env_cfg.py

```python
observation_space = 13  # 之前错误地写为 14
```

## 修复后验证

### Headless verify：8/8 测试通过

插入深度 0.80m，举升 0.82m，托盘被成功抬起 0.61m。

### 训练指标恢复正常（iteration 0 即可确认）

| 指标 | 修复前 | 修复后 | 改善 |
|------|--------|--------|------|
| lateral_mean | 0.0000 | **0.2987** | 随机化生效 |
| yaw_deg_mean | ~0.0003 | **7.2080** | 随机化生效 |
| dist_front_mean | 0.5533（恒定） | **0.4106（波动）** | 不再锁死 |
| action_noise_std | 0.07 | **3.01** | 正常探索 |
| steps/s | ~300-1000 | **18,812→24,233** | 24x 提速 |
| ETA (2000 iter) | 10+ 小时 | **~1.5 小时** | 7x 缩短 |

## Verify 测试与训练环境的 gap 分析

此 bug 之所以在 verify 测试中无法发现，是因为 verify 脚本与训练环境之间存在结构性差异：

| 方面 | verify 脚本 | 训练 |
|------|------------|------|
| num_envs | **1** | **1024** |
| sim.reset() 影响 | 只影响 1 个环境（无害） | 影响全部 1024 个环境（致命） |
| 动作来源 | 硬编码/键盘（持续且稳定） | 策略网络（初期随机） |
| 状态读取 | 直接读 env 内部属性 | 通过 `_get_observations()` |
| 指标计算 | 脚本自己独立计算 | env 的 `_get_rewards()` |
| 重置方式 | 手动 teleport + 缓存同步 | `_reset_idx()` |

**关键盲区：verify 在 num_envs=1 下通过不代表训练正常。** `sim.reset()` 的跨环境干扰只在 num_envs > 1 时暴露。

## 经验教训

1. **永远不要在 `_reset_idx()` 中调用 `sim.reset()`** — 这是全局 PhysX 引擎操作，不是逐环境操作。Isaac Lab 的正确模式是：`super()._reset_idx()` + 直接写入 PhysX（`write_root_link_pose_to_sim()`）。

2. **修复必须跨版本迁移** — S0.7 postmortem 已记录此 bug 并修复，但 S1.0h 重写代码时未迁移修复。建议在 postmortem 文档中标记"此修复是否已应用到最新版本"。

3. **日志中的"魔数"是强有力的诊断工具** — `0.5533` 和 `0.0000` 精确匹配默认 init_state 的计算值，直接定位到 `sim.reset()` 覆盖随机化。

4. **verify 测试需要多环境冒烟测试** — 单环境 verify 无法捕获跨环境干扰 bug。建议增加 `--num-envs 4 --smoke-test` 模式。

5. **删除 `scene.update()` 后需注意缓存初始化** — `_compute_fork_tip()` 依赖 `robot.data.root_pos_w`（由 `scene.update()` 刷新）。删除后需改为从随机化参数直接计算。本例中 `_fork_z_base = 0.0`，用 `z.squeeze(-1)` 替代误差为精确零。
