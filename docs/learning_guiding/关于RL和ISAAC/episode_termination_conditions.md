# Episode 终止条件

> 对应源码：`env.py` → `_get_dones()` 方法
>
> 配置参数：`env_cfg.py` → `ForkliftPalletInsertLiftEnvCfg`

## 概览

Episode 终止分为两类：

| 类型 | 含义 |
|------|------|
| **terminated** | 环境判定任务结束（成功或不可恢复的失败），会被 RL 算法视为真正的终端状态 |
| **time_out** | 超时截断，RL 算法会做 value bootstrap（不视为真正终端） |

```python
terminated = tipped | success | _early_stop_fly | _early_stop_stall | _early_stop_dz_stuck
return terminated, time_out
```

## 终止条件详情

### 1. 成功（success）

完成 **插入 + 举升 + 保持** 的全流程。

- 插入深度 `insert_norm` 达标（deep insert）
- 举升高度 `lift_delta` ≥ `lift_delta_m`（0.3m）
- 横向误差 `|y_err|` < `max_lateral_err_m`（0.15m），航向误差 `|yaw_err|` < `max_yaw_err_deg`（5°）
- 以上条件在 Schmitt trigger 下连续保持 `hold_time_s`（0.33s ≈ 10 步）
- **奖励**：`rew_success` + 时间奖励（越快完成越高）

### 2. 翻车（tipped）

叉车 roll 或 pitch 超过 `max_roll_pitch_rad`。

| 参数 | 值 |
|------|------|
| `max_roll_pitch_rad` | 0.45 rad ≈ 25.8° |

### 3. 飞离过远（early_stop_fly）

叉车到托盘的 XY 平面距离 `d_xy` 超过阈值，且**连续**超过指定步数。

| 参数 | 值 |
|------|------|
| `early_stop_d_xy_max` | 3.0 m |
| `early_stop_d_xy_steps` | 30 步 ≈ 1.0 s |
| `rew_early_stop_fly` | -2.0 |

**已知问题**：叉车初始位置 `(-3.5, 0, 0.03)`，托盘位置 `(0, 0, 0.15)`，初始 d_xy = **3.5m**，已超过 3.0m 阈值。模型必须在 30 步内将距离压到 3.0m 以下，否则直接判死。详见下方 [阈值过严分析](#阈值过严分析early_stop_fly)。

### 4. 完全停滞（early_stop_stall）

势能变化和动作幅度都极小，表明叉车完全卡住不动。

| 参数 | 值 |
|------|------|
| `early_stop_stall_phi_eps` | 0.001（势能变化阈值） |
| `early_stop_stall_action_eps` | 0.05（动作幅度阈值） |
| `early_stop_stall_steps` | 60 步 ≈ 2.0 s |
| `rew_early_stop_stall` | -1.0 |

触发条件：势能变化 < `phi_eps` **且** 动作最大绝对值 < `action_eps`，连续超过 60 步。

### 5. 死区卡住（early_stop_dz_stuck）

在死区内 `insert_norm` 和 `y_err` 长时间无变化。

| 参数 | 值 |
|------|------|
| `dz_stuck_steps` | 99999（默认不激活） |
| `rew_early_stop_dz_stuck` | -2.0 |

> 当前阈值 99999 步意味着 episode 会先超时（36s ≈ 1080 步），此条件实际上不会触发。

### 6. 超时（time_out）

Episode 时长达到上限。

| 参数 | 值 |
|------|------|
| `episode_length_s` | 36.0 s |
| `max_episode_length` | 36.0 / dt ≈ 1080 步 |
| `rew_timeout` | -10.0 |

超时不算 `terminated`，RL 算法会对其做 value bootstrap。

---

## 阈值过严分析：early_stop_fly

### 问题

| 项目 | 值 |
|------|------|
| 叉车初始位置 | `(-3.5, 0.0, 0.03)` |
| 托盘位置 | `(0.0, 0.0, 0.15)` |
| 初始 d_xy | **3.5 m** |
| early_stop_d_xy_max | 3.0 m |
| early_stop_d_xy_steps | 30 步 ≈ 1.0 s |

初始距离 3.5m **已经超过** 3.0m 阈值。episode 一开始 fly_counter 就在累加，模型只有 30 步（约 1 秒）的窗口来把距离压到 3.0m 以下。

### 影响

- **训练期间**（1024 envs）：统计上绝大部分环境会在前几步内开始接近托盘，问题不明显
- **play/评估期间**（少量 envs）：特定 seed 下策略可能在初始几步犹豫或微调方向，导致 30 步内未过线直接 terminated
- **探索惩罚**：学习初期策略随机行走，初始就在禁区内意味着探索空间被严重压缩

### 建议调整方案

| 方案 | 修改 | 优缺点 |
|------|------|--------|
| A. 放宽距离阈值 | `early_stop_d_xy_max`: 3.0 → **5.0** | 简单直接；给足初始距离余量（3.5m + 1.5m 探索空间） |
| B. 增加容忍步数 | `early_stop_d_xy_steps`: 30 → **90** | 保留距离约束但给更多反应时间（≈ 3s） |
| C. 延迟激活 | 在前 N 步内不累加 fly_counter | 允许初始阶段自由探索，之后恢复约束 |
| D. A+B 组合 | `d_xy_max`=5.0, `steps`=60 | 最稳妥 |

**推荐方案 D**：将 `early_stop_d_xy_max` 提高到 5.0m，`early_stop_d_xy_steps` 增加到 60 步。既防止叉车真正跑飞，又不因初始位置就触发惩罚。
