# Forklift Pallet Insert-Lift Reward Function v4.4

本文档详细描述叉车托盘插入-抬升任务的奖励函数设计。

---

## 1. 整体架构：两阶段课程学习（Phase Curriculum）

奖励函数采用**两阶段课程学习**设计，根据叉车状态自动切换阶段：

```
┌─────────────────────────────────────────────────────────────────────┐
│                           Phase 0: Docking                          │
│                         （靠近 + 粗对齐阶段）                          │
│                                                                     │
│  目标：引导叉车接近托盘并粗略对齐                                      │
│  启用：距离惩罚、接近奖励、对齐惩罚（gate内）、防摸鱼、平滑             │
│  禁用：插入进度、抬升奖励                                             │
├─────────────────────────────────────────────────────────────────────┤
│                        ↓ dock_condition 满足 ↓                       │
│        (dist_x < 0.8m && lateral < 0.1m && yaw < 10°) × 20步        │
├─────────────────────────────────────────────────────────────────────┤
│                      Phase 1: Insert + Lift                         │
│                       （精插入 + 抬升阶段）                            │
│                                                                     │
│  目标：精确对齐、插入托盘、抬升货物                                    │
│  启用：插入进度（gated）、抬升奖励（gated）、插入里程碑bonus           │
│  降权：距离惩罚（×0.2）                                               │
│  禁用：接近奖励                                                       │
└─────────────────────────────────────────────────────────────────────┘
```

### Phase 切换条件

```python
dock_condition = (
    (dist_x < dock_dist_m) &           # 距离阈值 (0.8m)
    (lateral_err < dock_lat_m) &       # 横向误差阈值 (0.1m)
    (yaw_err_deg < dock_yaw_deg)       # 偏航误差阈值 (10°)
)
# 连续满足 dock_hold_steps (20) 步后切换到 phase1
```

---

## 2. 奖励组件详解

### 2.1 距离惩罚 `distance_penalty`

**目的**：提供持续的"接近托盘"压力

```python
distance_penalty = rew_distance * dist_x
# dist_x: fork tip 到 pallet front 的距离
# rew_distance: -1.0 (v4.4)

# Phase 调整：
# - Phase 0: 全权重 (distance_penalty)
# - Phase 1: 降权 ×0.2 (distance_penalty * phase1_distance_scale)
```

| 参数 | 值 | 说明 |
|------|-----|------|
| `rew_distance` | -1.0 | 距离惩罚系数 |
| `phase1_distance_scale` | 0.2 | Phase1 降权系数 |

**设计理由**：
- Phase 0 需要强距离梯度引导接近
- Phase 1 近场操作时降低距离压力，避免干扰精对齐

---

### 2.2 增量接近奖励 `approach_reward`

**目的**：奖励"距离减小"的增量行为，提供方向性引导

```python
delta_dist = last_dist_x - dist_x  # 距离减小为正
approach_reward = rew_approach * clamp(delta_dist, -approach_clip, approach_clip)

# Phase 调整：
# - Phase 0: 启用
# - Phase 1: 禁用 (返回 0)
```

| 参数 | 值 | 说明 |
|------|-----|------|
| `rew_approach` | 5.0 | 接近奖励系数 (v4.4) |
| `approach_clip` | 0.2 | 增量裁剪阈值 (m) |

**设计理由**：
- 增量奖励比状态奖励更能提供"方向"信息
- clip 防止偶发大步导致奖励爆炸

---

### 2.3 对齐惩罚 `align_penalty`（Gated）

**目的**：惩罚横向偏移和偏航角度，引导精确对齐

```python
# 只有在 align gate 内才生效
in_align_zone = (dist_x < align_gate_dist_m)  # 1.0m

align_penalty = where(
    in_align_zone,
    rew_align * lateral_err + rew_yaw * yaw_err,
    0.0  # 远处不惩罚对齐
)
```

| 参数 | 值 | 说明 |
|------|-----|------|
| `rew_align` | -2.0 | 横向对齐惩罚系数 |
| `rew_yaw` | -0.5 | 偏航对齐惩罚系数 |
| `align_gate_dist_m` | 1.0 | 对齐惩罚激活距离 |

**设计理由**：
- 远处对齐惩罚会分散注意力，应先专注接近
- 只有靠近后（<1m）才启用对齐惩罚

---

### 2.4 Gated Progress 插入进度奖励

**目的**：奖励"对齐后"的插入推进，惩罚"未对齐硬怼"

```python
gate_aligned = (lateral_err <= gate_lateral_err_m) & (yaw_err <= gate_yaw_err_deg)

progress_reward = where(
    gate_aligned,
    rew_progress * progress,           # 对齐时奖励推进
    -rew_wrong_progress * clamp(progress, min=0)  # 未对齐时惩罚正向推进
)

# Phase 调整：
# - Phase 0: 禁用 (返回 0)
# - Phase 1: 启用
```

| 参数 | 值 | 说明 |
|------|-----|------|
| `rew_progress` | 4.0 | 插入进度奖励系数 |
| `rew_wrong_progress` | 2.0 | 未对齐推进惩罚系数 |
| `gate_lateral_err_m` | 0.05 | 对齐 Gate 横向阈值 |
| `gate_yaw_err_deg` | 5.0 | 对齐 Gate 偏航阈值 |

**设计理由**：
- 防止"斜怼"行为：未对齐时推进会被惩罚
- Gate 阈值略宽于成功阈值，给策略一定容错空间

---

### 2.5 增量型 Lift 奖励（Gated）

**目的**：奖励"插入到位后"的抬升动作

```python
gate_inserted = (insert_norm >= insert_gate_norm)  # 60% 插入深度

lift_reward = where(
    gate_inserted,
    rew_lift * clamp(delta_lift, min=0),  # 只奖励正向抬升
    0.0  # 未插入到位不给抬升奖励
)

# Phase 调整：
# - Phase 0: 禁用 (返回 0)
# - Phase 1: 启用
```

| 参数 | 值 | 说明 |
|------|-----|------|
| `rew_lift` | 1.0 | 抬升奖励系数 |
| `insert_gate_norm` | 0.6 | 插入 Gate 阈值 (60%) |

**设计理由**：
- 使用增量（delta_lift）而非状态（lift_pos），防止"抬起来站着刷分"
- Gate 机制确保插入到位后才能抬升领奖

---

### 2.6 插入里程碑 Bonus

**目的**：首次插入到位时发放一次性奖励，鼓励突破

```python
newly_reached_insert = gate_inserted & (~reached_insert_gate_flag)
insert_bonus = where(newly_reached_insert, rew_insert_bonus, 0.0)
# 更新标记，确保只发放一次
reached_insert_gate_flag |= gate_inserted

# Phase 调整：
# - Phase 0: 禁用 (返回 0)
# - Phase 1: 启用
```

| 参数 | 值 | 说明 |
|------|-----|------|
| `rew_insert_bonus` | 20.0 | 插入里程碑 bonus |

---

### 2.7 Stall 惩罚（防摸鱼）

**目的**：惩罚"远处低速站岗"行为

```python
is_stall = (dist_x > stall_dist_thresh) & (v_norm < stall_v_thresh)
stall_penalty = where(is_stall, rew_stall, 0.0)
```

| 参数 | 值 | 说明 |
|------|-----|------|
| `rew_stall` | -0.15 | Stall 惩罚值 |
| `stall_dist_thresh` | 0.8 | 距离阈值 (m) |
| `stall_v_thresh` | 0.1 | 速度阈值 (m/s) |

**设计理由**：
- 比固定 time penalty 更聪明
- 只惩罚"无意义拖延"，不惩罚"近场慢速精调"

---

### 2.8 Action Rate 惩罚（动作平滑）

**目的**：惩罚动作剧烈变化，鼓励平滑控制

```python
action_diff = actions - last_actions
action_rate = (action_diff ** 2).sum(dim=1)
action_rate_penalty = rew_action_rate * action_rate
# 首步不惩罚
```

| 参数 | 值 | 说明 |
|------|-----|------|
| `rew_action_rate` | -0.02 | 动作变化率惩罚系数 |

---

### 2.9 Phase 0 提前抬升惩罚

**目的**：防止 Phase 0 阶段的无意义抬升

```python
pen_pre_lift = where(
    (phase == 0) & (lift_pos > 0.01),
    -0.1 * lift_pos,
    0.0
)
```

---

### 2.10 成功奖励 + 时间奖金

**目的**：奖励任务成功，越早完成奖金越高

```python
# 成功条件
success = (insert_depth >= thresh) & (lateral <= 0.03m) & (yaw <= 3°) & (lift >= 0.15m)
# 连续满足 hold_steps 步确认成功

time_ratio = episode_length / max_episode_length
time_bonus = rew_success_time * (1.0 - time_ratio)
success_reward = rew_success + time_bonus
```

| 参数 | 值 | 说明 |
|------|-----|------|
| `rew_success` | 100.0 | 成功基础奖励 |
| `rew_success_time` | 30.0 | 时间奖金系数 |

---

## 3. 奖励汇总公式

```python
reward = (
    distance_penalty          # 距离惩罚（phase0 全权重，phase1 ×0.2）
    + align_penalty           # 对齐惩罚（只在 <1m 时生效）
    + approach_reward         # 接近奖励（phase0 启用，phase1 禁用）
    + progress_reward         # 插入进度（phase0 禁用，phase1 启用）
    + lift_reward             # 抬升奖励（phase0 禁用，phase1 启用）
    + insert_bonus            # 插入里程碑（phase0 禁用，phase1 启用）
    + pen_pre_lift            # 提前抬升惩罚（phase0 only）
    + stall_penalty           # 防摸鱼惩罚
    + action_rate_penalty     # 动作平滑惩罚
    + success_reward          # 成功奖励 + 时间奖金
)
```

---

## 4. 参数配置总览

### 核心奖励参数

| 参数 | 值 | 类型 | Phase 0 | Phase 1 |
|------|-----|------|---------|---------|
| `rew_distance` | -1.0 | 惩罚 | ✓ (×1.0) | ✓ (×0.2) |
| `rew_approach` | 5.0 | 奖励 | ✓ | ✗ |
| `rew_align` | -2.0 | 惩罚 | ✓ (gated) | ✓ (gated) |
| `rew_yaw` | -0.5 | 惩罚 | ✓ (gated) | ✓ (gated) |
| `rew_progress` | 4.0 | 奖励 | ✗ | ✓ (gated) |
| `rew_lift` | 1.0 | 奖励 | ✗ | ✓ (gated) |
| `rew_insert_bonus` | 20.0 | 奖励 | ✗ | ✓ |
| `rew_stall` | -0.15 | 惩罚 | ✓ | ✓ |
| `rew_action_rate` | -0.02 | 惩罚 | ✓ | ✓ |

### Gate 阈值参数

| 参数 | 值 | 说明 |
|------|-----|------|
| `align_gate_dist_m` | 1.0 | 对齐惩罚激活距离 |
| `gate_lateral_err_m` | 0.05 | Progress gate 横向阈值 |
| `gate_yaw_err_deg` | 5.0 | Progress gate 偏航阈值 |
| `insert_gate_norm` | 0.6 | Lift gate 插入阈值 |

### Phase Curriculum 参数

| 参数 | 值 | 说明 |
|------|-----|------|
| `dock_dist_m` | 0.8 | Dock 距离阈值 |
| `dock_lat_m` | 0.10 | Dock 横向阈值 |
| `dock_yaw_deg` | 10.0 | Dock 偏航阈值 |
| `dock_hold_steps` | 20 | 连续满足步数 |
| `phase1_distance_scale` | 0.2 | Phase1 距离惩罚降权 |

### 成功条件参数

| 参数 | 值 | 说明 |
|------|-----|------|
| `max_lateral_err_m` | 0.03 | 成功条件：横向误差 |
| `max_yaw_err_deg` | 3.0 | 成功条件：偏航误差 |
| `lift_delta_m` | 0.15 | 成功条件：抬升高度 |
| `hold_steps` | 10 | 成功确认步数 |

---

## 5. 设计原则总结

1. **增量优于状态**：奖励"做对动作"而非"占着位置"
2. **Gate 机制防刷分**：确保在正确条件下才给奖励
3. **课程学习分阶段**：先学接近，再学精插抬升
4. **惩罚/奖励分离**：惩罚用状态型，奖励用增量型
5. **防摸鱼条款**：stall 惩罚比 time penalty 更智能

---

*文档版本：v4.4*
*更新日期：2026-02-03*
