# env.py 代码重建后三次关键 Bug 修复 — Postmortem

> **日期**：2026-02-06  
> **影响文件**：`IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/forklift_pallet_insert_lift/env.py`  
> **背景**：S0.7 训练前，原始 `env.py`、`env_cfg.py`、`rsl_rl_ppo_cfg.py` 三个文件因 `git clean -fd` 丢失（未被 git 追踪）。通过备份骨架 + S0.6 训练 yaml + transcript 中的 S0.4 `_get_rewards()` + `.pyc` 字节码分析完成重建。重建后经历了三次 bug 修复，以下逐一记录。

---

## Bug 1：`sim.reset()` 性能灾难 — 25x 减速

### 时间线

- **发现时间**：S0.7 训练启动后立即发现
- **用户反馈**：「速度 ~824 steps/s 这个数量怎么这么小啊，之前都是 4 万多啊」

### 现象

| 指标 | S0.6 基线 | S0.7 重建后 |
|------|----------|------------|
| steps/s | 16,600 | **830** |
| collection time | 2.8s | **78s** |
| learning time | 0.11s | 0.23s |
| PhysX 碰撞警告 | 少量 | **数百条/iter** |

### 根因分析

重建的 `_reset_idx()` 保留了从旧备份文件带来的三行代码：

```python
# _reset_idx 中，在写入 robot/pallet 状态后：
self.scene.write_data_to_sim()   # 刷新缓冲 → 正常操作
self.sim.reset()                 # ← 性能杀手！触发完整的 PhysX 引擎重置
self.scene.update(self.cfg.sim.dt)
```

**`self.sim.reset()` 与 `self.scene.reset(env_ids)` 的本质区别**：

| | `sim.reset()` | `scene.reset(env_ids)` |
|---|---|---|
| 作用 | 完整的 PhysX 物理引擎重初始化 | 仅重置指定环境的内部缓冲区 |
| 开销 | 极高（~75ms，1024 env） | 极低（<1ms） |
| 调用场景 | 仅在 `__init__` 时调用一次 | 每次 episode reset 调用 |
| 副作用 | 重新解析所有 USD 碰撞网格，触发大量 `convexHull` 警告 | 无 |

在 1024 环境、短 episode（~30 步就超时）的场景下，reset 每 iter 可能触发上百次，每次 `sim.reset()` 约 75ms，累计远超物理仿真本身。

### 修复

```python
# 删除 sim.reset() 及其前后的 write/update
# 增量奖励基线改为从 reset 随机化参数直接推算：

# fork_tip_z0: reset 时 lift=0，直接用 robot z 高度
self._fork_tip_z0[env_ids] = z.squeeze(-1)

# dist_front baseline: 从随机化的 x 位置推算
self._last_dist_front[env_ids] = torch.clamp(
    self._pallet_front_x - x.squeeze(-1), min=0.0
)

# E_align baseline: 从随机化的 y, yaw 推算
y_err_reset = torch.abs(y.squeeze(-1))
yaw_err_deg_reset = torch.abs(yaw.squeeze(-1)) * (180.0 / math.pi)
E_align_reset = y_err_reset / self.cfg.lat_ready_m + yaw_err_deg_reset / self.cfg.yaw_ready_deg
self._last_E_align[env_ids] = E_align_reset
```

### 验证

| | 修复前 | 修复后 |
|---|---|---|
| steps/s (4 env dry-run) | — | 56 (GPU 未满载，正常) |
| 碰撞警告数 | 几百条 | **1 条**（初始化时一次性） |

---

## Bug 2：fork tip 重复计算 + PhysX 张量浪费

### 时间线

- **发现时间**：Bug 1 修复后，速度恢复到 8,000-10,000 steps/s，但仍比 S0.6 的 16,600 低约 40%
- **排查方式**：代码审查，分析 `_compute_fork_tip()` 调用次数和 PhysX fallback 开销

### 现象

| 指标 | S0.6 基线 | Bug 1 修复后 |
|------|----------|-------------|
| steps/s | 16,600 | 8,000-10,000 |
| collection time | 2.8s | 6-8s |

### 根因分析

三个性能问题叠加：

**问题 A**：`_compute_fork_tip()` 每步调用 3 次

```
_apply_action()    → _compute_fork_tip()  # 计算 lock_drive_steer
_get_observations() → _compute_fork_tip()  # 计算 insert_norm 观测
_get_rewards()     → _compute_fork_tip()  # 计算全部奖励的基础量
```

该函数访问 `body_pos_w` (shape `[1024, num_bodies, 3]`) 并做 `argmax`，每次调用开销不小。

**问题 B**：PhysX fallback 每步创建临时张量

```python
# 每步执行：
full_targets = self.robot.data.joint_pos_target.clone()  # clone [1024, 7]
env_indices = torch.arange(...)  # 重新创建 [1024] int32
self.robot.root_physx_view.set_dof_position_targets(full_targets, env_indices)
```

**问题 C**：`_apply_action` 的 lock_drive_steer 实时计算

在 action 应用阶段计算 fork tip + 对齐检查，仅用于判断是否锁定驱动/转向，精度要求不高但开销与 reward 计算相当。

### 修复

**A. fork tip 缓存**：

```python
# __init__ 中新增
self._cached_fork_tip: torch.Tensor | None = None

# _pre_physics_step 中失效缓存
self._cached_fork_tip = None

# _compute_fork_tip 中缓存
def _compute_fork_tip(self) -> torch.Tensor:
    if self._cached_fork_tip is not None:
        return self._cached_fork_tip
    # ... 计算逻辑 ...
    self._cached_fork_tip = tip
    return tip
```

`_get_observations` 和 `_get_rewards` 在同一物理步之后调用，共享缓存。从 3 次计算降到 2 次。

**B. 预分配 env_indices + 去掉 clone**：

```python
# __init__ 中预分配
self._all_env_indices = torch.arange(self.num_envs, device=self.device, dtype=torch.int32)

# _apply_action 中直接 in-place 修改 + 复用索引
targets = self.robot.data.joint_pos_target  # 不 clone
targets[:, self._lift_id] = self._lift_target_pos
self.robot.root_physx_view.set_dof_position_targets(targets, self._all_env_indices)
```

**C. lock_drive_steer 用上一步近似**：

```python
# 替代实时计算 fork tip，用上一步缓存的值：
inserted_enough = self._last_insert_depth >= self._insert_thresh
E_thresh = 1.0
aligned_enough = self._last_E_align < E_thresh
```

一步延迟对 lock 判定几乎无影响，但省去了 `_apply_action` 中的整个 fork tip 计算。

### 验证

| | Bug 1 修复后 | Bug 2 修复后 |
|---|---|---|
| steps/s (1024 env) | 8,000-10,000 | **11,000-15,000** |
| collection time | 6-8s | **4.3-5.6s** |

---

## Bug 3：`super()._reset_idx()` 缺失 — Episode 永远在第 1 步超时

### 时间线

- **发现时间**：Bug 1+2 修复后启动 1024 env 训练，运行正常但分析日志发现致命问题
- **发现方式**：用户询问「训练正常吗，从日志来看」，分析后发现 `Mean episode length: 1.00`

### 现象

| 指标 | 期望值 | 实际值 |
|------|--------|--------|
| Mean episode length | ~750-1350 | **1.00** |
| term/frac_timeout | ~0 | **1.0000** |
| r_align | 非零 | **0.0000** |
| r_approach | 非零 | **0.0000** |
| r_insert | 非零 | **0.0000** |
| r_lift | 非零 | **0.0000** |
| noise_std | 应变化 | 57.73（完全不变） |

所有增量奖励精确为 0，策略完全没在学习。

### 根因分析

重写的 `_reset_idx` 没有调用基类方法：

```python
def _reset_idx(self, env_ids):
    if env_ids is None:
        env_ids = torch.arange(self.num_envs, device=self.device)
    # ← 缺少 super()._reset_idx(env_ids)
    self._last_insert_depth[env_ids] = 0.0
    # ... 其余重置逻辑 ...
```

Isaac Lab `DirectRLEnv._reset_idx` 基类做了两件关键事情：

```python
# isaaclab/envs/direct_rl_env.py line 586-607
def _reset_idx(self, env_ids):
    self.scene.reset(env_ids)              # 1. 刷新状态写入到仿真
    # ... event manager, noise model ...
    self.episode_length_buf[env_ids] = 0   # 2. 重置 episode 计数器
```

缺少后的连锁效应：

```
初始化时 episode_length_buf = 0
→ 第 1 步：buf += 1，buf=1，_get_dones 检查 buf >= max_episode_length-1 (1349)，不超时
→ ... 正常运行到 episode 结束
→ _reset_idx 被调用，但 episode_length_buf 没归零！buf 仍为 1350
→ 第 2 个 episode 第 1 步：buf += 1，buf=1351，1351 >= 1349 → 立即超时
→ 此后所有 episode 都在第 1 步超时
→ 每步都是 _is_first_step=True → 所有增量奖励被 where(is_first_step, 0, ...) 过滤
→ 策略只收到 time_penalty + pen_dist_far 两个常数惩罚，完全无法学习
```

**隐蔽性分析**：这个 bug 极其难以从表面发现——

- 训练进程正常运行，不崩溃
- 日志格式完全正确，所有 key 都有输出
- 速度正常（11,000+ steps/s）
- 只有仔细分析 `episode_length: 1.00` 和 `所有奖励=0` 的异常组合才能诊断

### 修复

一行修复：

```python
def _reset_idx(self, env_ids):
    if env_ids is None:
        env_ids = torch.arange(self.num_envs, device=self.device)

    # call base class: scene.reset(env_ids) + episode_length_buf reset
    super()._reset_idx(env_ids)

    # ---- reset counters / buffers ----
    # ... 其余不变 ...
```

### 验证

**干跑（4 env, 3 iter）**：

| 指标 | 修复前 | 修复后 |
|------|--------|--------|
| episode_length | 1.00 | **168.00** |
| frac_timeout | 1.0 | **0.0** |
| r_align | 0.0 | **0.0101** |
| r_approach | 0.0 | **0.0228** |

**正式训练（1024 env）**：

| 指标 | 修复前 | 修复后 (iter 2668) |
|------|--------|-------------------|
| episode_length | 1.00 | **1322** |
| insert_norm | 0.0 | **0.213** |
| frac_inserted | 0% | **10.7%** |
| frac_lifted | 0.1% | **62.0%** |
| hold_counter_max | 0 | **13** |

---

## 总结与防范措施

### 三个 bug 的共性

| | Bug 1 | Bug 2 | Bug 3 |
|---|---|---|---|
| 类型 | 性能 | 性能 | 功能 |
| 来源 | 备份代码遗留 | 重建时未优化 | 重建时遗漏 |
| 影响 | 25x 减速 | 1.5x 减速 | 训练完全失效 |
| 发现方式 | 用户对比速度 | 代码审查 | 日志数值分析 |
| 表面可见性 | 高（明显慢） | 中（偏慢但能跑） | **极低（一切看起来正常）** |

### 防范建议

1. **重写任何 Isaac Lab 的 `_reset_idx` 时，首行必须 `super()._reset_idx(env_ids)`**
2. **正式训练前做 short run（~5 iter），核对**：
   - `Mean episode length` 是否合理（应 >> 1）
   - 增量奖励是否非零
   - `frac_timeout` 是否 < 1.0
3. **`sim.reset()` 只在 `__init__`/`_setup_scene` 中调用一次，绝不在 `_reset_idx` 中调用**
4. **性能基线**：记录正常训练的 steps/s，新版本偏差 >2x 时立即排查
5. **git add 所有源文件**，防止 `git clean` 误删

---

### 最终修复后的代码状态

```
env.py 关键方法调用链：

_pre_physics_step(actions)
  └─ invalidate fork_tip cache

_apply_action()
  ├─ lock_drive_steer: 用 _last_insert_depth / _last_E_align 近似（零开销）
  ├─ wheel/steer/lift targets
  ├─ PhysX fallback: in-place + pre-allocated indices
  └─ write_data_to_sim()

[physics steps × decimation]

_get_observations()
  └─ _compute_fork_tip()  ←─┐ 缓存共享（同一物理步只计算一次）
                             │
_get_rewards()               │
  └─ _compute_fork_tip()  ←─┘

_get_dones()

_reset_idx(env_ids)
  ├─ super()._reset_idx(env_ids)  ← scene.reset + episode_length_buf=0
  ├─ reset custom buffers
  ├─ write robot/pallet poses
  └─ compute baselines from reset params (no sim step needed)
```

---

*文档日期：2026-02-06*  
*关联文件*：
- `env.py`：`IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/forklift_pallet_insert_lift/env.py`
- 奖励文档：`docs/rewards/reward_function.md`
- S0.7 训练日志：`20260206_train_s0.7.log`
