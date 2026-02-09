# S1.0i 奖励函数详解

> **版本**：S1.0i（基于 S1.0h，修复后退对齐漏洞）
> **源文件**：`env.py` `_get_rewards()` + `env_cfg.py`
> **最后更新**：2026-02-09

---

## 1. 名词定义

| 名词 | 符号 | 含义 | 单位 |
|------|------|------|------|
| 货叉尖端 | `tip` | 叉车货叉最前端的世界坐标位置，由 `_compute_fork_tip()` 运动学计算 | m |
| 托盘前部 | `pallet_front_x` | 托盘 pocket 开口面的 x 坐标 = `pallet_pos_x - pallet_depth / 2` | m |
| 插入深度 | `insert_depth` | 货叉尖端超过托盘前部的距离 = `clamp(tip_x - pallet_front_x, min=0)` | m |
| 归一化插入深度 | `insert_norm` | `insert_depth / pallet_depth`，范围 [0, 1] | - |
| 前部距离 | `dist_front` | 货叉尖端到托盘前部的剩余距离 = `clamp(pallet_front_x - tip_x, min=0)`，正值=尚未到达 | m |
| 横向误差 | `y_err` | 叉车与托盘在 y 轴的绝对距离 = `abs(pallet_y - robot_y)` | m |
| 偏航误差 | `yaw_err_deg` | 叉车与目标方向的偏航角度差（远处对准托盘中心方向，近处对准托盘朝向） | deg |
| 归一化对齐误差 | `E_align` | `y_err / lat_ready_m + yaw_err_deg / yaw_ready_deg`，E > 1 表示未满足对齐要求 | - |
| 举升高度 | `lift_height` | 货叉当前 z 坐标与 reset 时基准 z 的差值 | m |
| 举升增量 | `delta_lift` | 当前步与上一步 `lift_height` 的差值 | m |
| 距离增量 | `delta_dist` | 上一步 `dist_front` 减当前步（正值=接近） | m |
| 对齐增量 | `delta_E_align` | 上一步 `E_align` 减当前步（正值=对齐改善） | - |
| 前进速度 | `v_forward` | 叉车在自身坐标系下的纵向（前进方向）线速度 | m/s |

### 距离自适应权重

| 名词 | 符号 | 含义 |
|------|------|------|
| 近处权重 | `w_close` | `smoothstep((d_far - dist_front) / (d_far - d_close))`，dist_front <= d_close 时为 1，>= d_far 时为 0 |
| 远处权重 | `w_far` | `1 - w_close` |
| 对齐就绪门控 | `w_ready` | `w_lat * w_yaw`，两个方向都满足阈值时为 1，任一超标则趋近 0 |
| 接近权重 | `w_approach` | `0.2 + 0.8 * w_ready`，即使未对齐也保留 20% 接近奖励 |
| 举升门控 | `w_lift` | `w_lift_base * w_ready`，插入深度 >= 60% 且对齐才允许举升奖励 |

### 距离自适应阈值

对齐阈值随 `dist_front` 在远/近之间插值，**远处宽松（允许粗对齐就接近），近处严格（精确对齐才能插入）**：

| 参数 | 远处值 | 近处值 | 含义 |
|------|--------|--------|------|
| `lat_ready_m` | 0.6 m | 0.10 m | 横向对齐阈值 |
| `yaw_ready_deg` | 30° | 10° | 偏航对齐阈值 |

---

## 2. 奖励项一览

### 阶段 1：接近 + 对齐

| 奖励项 | 公式 | 系数 | 正/负 | 说明 |
|--------|------|------|-------|------|
| 接近奖励 `r_approach` | `k_approach * w_approach * delta_dist` | 远 10.0 / 近 8.0 | 接近=正 | 增量式，每步靠近一点就给正奖励 |
| 对齐改善奖励 `r_align` | `k_align * delta_E_align` | 远 2.0 / 近 10.0 | 改善=正 | 增量式，对齐质量提高就给正奖励 |
| 前进速度奖励 `r_forward` | `k_forward * clamp(v_forward, min=0)` | 0.5 | 正 | 全距离、无门控，直接奖励前进 |
| 绝对对齐惩罚 `pen_align_abs` | `-k_align_abs * clamp(E_align, 0, 2)` | 0.05 | 负 | 距离无关，E_align 自身含距离自适应阈值 |

### 阶段 2：插入

| 奖励项 | 公式 | 系数 | 正/负 | 说明 |
|--------|------|------|-------|------|
| 插入奖励 `r_insert` | `gamma * Phi_insert_t - Phi_insert_{t-1}` | 远 8.0 / 近 15.0 | 插入=正 | 势函数 shaping，`Phi = k * w_ready * insert_norm` |

### 阶段 3：举升

| 奖励项 | 公式 | 系数 | 正/负 | 说明 |
|--------|------|------|-------|------|
| 举升奖励 `r_lift` | `gamma * Phi_lift_t - Phi_lift_{t-1}` | 20.0 | 举升=正 | 势函数 shaping，`Phi = k * w_lift * lift_height` |
| 空举惩罚 `pen_premature` | `-k_pre * (1 - w_lift_base) * clamp(delta_lift, min=0)` | 远 12.0 / 近 5.0 | 负 | 插入不够深就举升 → 惩罚 |

### 常驻惩罚

| 奖励项 | 公式 | 值 | 说明 |
|--------|------|---|------|
| 动作 L2 | `rew_action_l2 * sum(actions^2)` | -0.01 | 抑制大动作 |
| 时间惩罚 | `rew_time_penalty` | -0.003/步 | 鼓励尽快完成 |
| 远距惩罚 `pen_dist_far` | `-k_dist_far * clamp(dist_front - 2.0, min=0)` | k=0.3 | dist > 2m 额外惩罚 |

### 成功奖励

| 奖励项 | 公式 | 值 | 说明 |
|--------|------|---|------|
| 成功奖励 | `rew_success + rew_success_time * (1 - time_ratio)` | 100 + 最多 30 | 插入 >= 2/3 深度 + 对齐(3cm, 3°) + 举升 >= 0.12m，持续 1s |

---

## 3. 设计原则

### 3.1 距离自适应

所有阈值和系数随 `dist_front` 在 `[d_close=1.1m, d_far=2.6m]` 之间 smoothstep 插值：

- **远处（dist > 2.6m）**：高接近系数(10)、低对齐系数(2)、宽松对齐阈值(0.6m/30°)
  → 鼓励先接近，粗对齐即可
- **近处（dist < 1.1m）**：低接近系数(8)、高对齐系数(10)、严格对齐阈值(0.1m/10°)
  → 要求精确对齐后再插入

### 3.2 势函数 shaping（防奖励泵）

插入和举升使用势函数 shaping 而非简单增量奖励：

```
r = gamma * Phi_t - Phi_{t-1}
```

其中 `gamma = 0.99`（PPO 折扣因子）。这确保：
- 前进（Phi 增大）→ 正奖励
- 后退（Phi 减小）→ 负奖励（自动扣分）
- 循环前进-后退刷分 → 因 `gamma < 1` 每次净亏损

### 3.3 偏航目标分段

- **远处**：yaw 目标 = 叉车→托盘中心方向（`atan2(dy, dx)`），允许策略先打角消除横向偏差
- **近处**：yaw 目标 = 托盘朝向（平行对接），确保精确插入
- 两段通过 `w_far/w_close` 平滑混合

### 3.4 首步保护

所有增量奖励在 episode reset 后的**第一步被清零**（`_is_first_step` 标记），避免因缓存初始化不精确导致的首步异常奖励脉冲。

---

## 4. S1.0i 相对 S1.0h 的修改

| 项目 | S1.0h | S1.0i | 原因 |
|------|-------|-------|------|
| `pen_align_abs` 中的 `w_close` | 有 | **删除** | w_close 使后退→惩罚降低，形成"后退对齐"漏洞 |
| `k_align_abs` | 0.10 | **0.05** | 去掉 w_close 后适用范围更广，降低强度避免过度惩罚 |
| `r_forward` 中的 `w_far * w_yaw` | 有 | **删除** | 让前进奖励全距离、无条件生效 |
| `k_forward` | 0.02 | **0.5** | 原值过小（每步 ~0.0001），提高到有实际意义的水平 |

### 修改效果（400 iteration 对比）

| 指标 | S1.0h | S1.0i |
|------|-------|-------|
| dist_front_mean | 0.99（持续恶化） | **0.59（稳定）** |
| Mean reward | -26（始终为负） | **+3.2（转正）** |
| r_forward | 0.0001 | **0.059** |
| r_approach | -0.005（后退） | **-0.001（几乎不退）** |
| yaw_deg_mean | 6.8° | **3.9°** |
| frac_aligned | 2.3% | **6.9%** |

---

## 5. 成功判定条件

同时满足以下三个条件，并**持续保持 1 秒**（`hold_time_s`）：

| 条件 | 阈值 |
|------|------|
| 插入深度 >= 2/3 托盘深度 | `insert_depth >= 1.44m` |
| 横向误差 <= 3cm | `y_err <= 0.03m` |
| 偏航误差 <= 3° | `yaw_err <= 3.0°` |
| 举升高度 >= 12cm | `lift_height >= 0.12m` |

成功后获得 `100 + 30 * (1 - time_ratio)` 的一次性奖励（越快完成奖励越高）。

---

## 6. 终止条件

| 条件 | 阈值 | 说明 |
|------|------|------|
| 超时 | 12s（359 步） | episode_length_s = 12.0 |
| 倾翻 | roll/pitch > 25° | max_roll_pitch_rad = 0.45 |
| 成功 | 持续 hold 1s | 触发成功奖励后终止 |

---

## 7. 所有参数速查表

| 参数 | 值 | 所在文件 |
|------|---|---------|
| `d_far` | 2.6 m | env_cfg.py |
| `d_close` | 1.1 m | env_cfg.py |
| `lat_ready_far` | 0.6 m | env_cfg.py |
| `lat_ready_close` | 0.10 m | env_cfg.py |
| `yaw_ready_far_deg` | 30° | env_cfg.py |
| `yaw_ready_close_deg` | 10° | env_cfg.py |
| `insert_gate_norm` | 0.60 | env_cfg.py |
| `insert_ramp_norm` | 0.10 | env_cfg.py |
| `k_app_far / k_app_close` | 10.0 / 8.0 | env_cfg.py |
| `k_align_far / k_align_close` | 2.0 / 10.0 | env_cfg.py |
| `k_ins_far / k_ins_close` | 8.0 / 15.0 | env_cfg.py |
| `k_pre_far / k_pre_close` | 12.0 / 5.0 | env_cfg.py |
| `k_align_abs` | 0.05 | env_cfg.py |
| `k_lift` | 20.0 | env_cfg.py |
| `k_forward` | 0.5 | env_cfg.py |
| `k_dist_far` | 0.3 | env_cfg.py |
| `rew_action_l2` | -0.01 | env_cfg.py |
| `rew_time_penalty` | -0.003 | env_cfg.py |
| `rew_success` | 100.0 | env_cfg.py |
| `rew_success_time` | 30.0 | env_cfg.py |
| `insert_fraction` | 2/3 | env_cfg.py |
| `lift_delta_m` | 0.12 m | env_cfg.py |
| `hold_time_s` | 1.0 s | env_cfg.py |
| `max_lateral_err_m` | 0.03 m | env_cfg.py |
| `max_yaw_err_deg` | 3.0° | env_cfg.py |
| `pallet_depth_m` | 2.16 m | env_cfg.py |
