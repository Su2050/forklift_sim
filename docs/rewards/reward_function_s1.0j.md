# S1.0j 奖励函数详解

> **版本**：S1.0j（基于 S1.0i，消灭"速度刷分"局部最优）
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
| 前部距离 | `dist_front` | 货叉尖端到托盘前部的剩余距离（x 轴）= `clamp(pallet_front_x - tip_x, min=0)` | m |
| 横向误差 | `y_err` | 叉车与托盘在 y 轴的绝对距离 = `abs(pallet_y - robot_y)` | m |
| **2D 接近距离** | `d_xy` | `sqrt(dist_front_clamped^2 + y_err^2)`，同时衡量纵向和横向偏差 | m |
| 2D 接近增量 | `delta_d_xy` | `last_d_xy - d_xy`，正值=接近（含横向修正） | m |
| 偏航误差 | `yaw_err_deg` | 叉车与目标方向的偏航角度差（远处对准托盘中心方向，近处对准托盘朝向） | deg |
| 归一化对齐误差 | `E_align` | `y_err / lat_ready_m + yaw_err_deg / yaw_ready_deg`，E > 1 表示未满足对齐要求 | - |
| 举升高度 | `lift_height` | 货叉当前 z 坐标与 reset 时基准 z 的差值 | m |
| 举升增量 | `delta_lift` | 当前步与上一步 `lift_height` 的差值 | m |

### 距离自适应权重

| 名词 | 符号 | 含义 |
|------|------|------|
| 近处权重 | `w_close` | `smoothstep((d_far - dist_front) / (d_far - d_close))`，dist_front <= d_close 时为 1，>= d_far 时为 0 |
| 远处权重 | `w_far` | `1 - w_close` |
| 对齐就绪门控 | `w_ready` | `w_lat * w_yaw`，两个方向都满足阈值时为 1，任一超标则趋近 0 |
| 举升门控 | `w_lift` | `w_lift_base * w_ready`，插入深度 >= 60% 且对齐才允许举升奖励 |

### 距离自适应阈值

| 参数 | 远处值 | 近处值 | 含义 |
|------|--------|--------|------|
| `lat_ready_m` | 0.6 m | 0.10 m | 横向对齐阈值 |
| `yaw_ready_deg` | 30 deg | 10 deg | 偏航对齐阈值 |

---

## 2. 奖励项一览

### 阶段 1：接近 + 对齐

| 奖励项 | 公式 | 系数 | 正/负 | 说明 |
|--------|------|------|-------|------|
| 接近奖励 `r_approach` | `k_approach * delta_d_xy` | 远 10.0 / 近 8.0 | 接近=正 | **S1.0j: 改用 2D 距离，去掉 w_approach 门控** |
| 对齐改善奖励 `r_align` | `k_align * delta_E_align` | 远 2.0 / 近 10.0 | 改善=正 | 增量式，对齐质量提高就给正奖励 |
| 绝对对齐惩罚 `pen_align_abs` | `-k_align_abs * clamp(E_align, 0, 2)` | 0.05 | 负 | 距离无关，E_align 自身含距离自适应阈值 |

> **S1.0j 删除了 `r_forward`（前进速度奖励）**。详见第 4 节。

### 阶段 2：插入

| 奖励项 | 公式 | 系数 | 正/负 | 说明 |
|--------|------|------|-------|------|
| 插入奖励 `r_insert` | `gamma * Phi_insert_t - Phi_insert_{t-1}` | 远 8.0 / 近 15.0 | 插入=正 | 势函数 shaping，`Phi = k * w_ready * insert_norm` |

### 阶段 3：举升

| 奖励项 | 公式 | 系数 | 正/负 | 说明 |
|--------|------|------|-------|------|
| 举升奖励 `r_lift` | `gamma * Phi_lift_t - Phi_lift_{t-1}` | 20.0 | 举升=正 | 势函数 shaping，`Phi = k * w_lift * lift_height` |
| 空举惩罚 `pen_premature` | `-k_pre * (1 - w_lift_base) * clamp(delta_lift, min=0)` | 远 12.0 / 近 5.0 | 负 | 插入不够深就举升则惩罚 |

### 常驻惩罚

| 奖励项 | 公式 | 值 | 说明 |
|--------|------|---|------|
| 动作 L2 | `rew_action_l2 * sum(actions^2)` | -0.01 | 抑制大动作 |
| 时间惩罚 | `rew_time_penalty` | -0.003/步 | 鼓励尽快完成 |
| 远距惩罚 `pen_dist_far` | `-k_dist_far * clamp(dist_front - 2.0, min=0)` | k=0.3 | dist > 2m 额外惩罚 |
| **连续距离惩罚** `pen_dist_cont` | `-k_dist_cont * d_xy` | k=0.03 | **S1.0j 新增：所有距离上持续施压** |

### 终局奖励/惩罚

| 奖励项 | 公式 | 值 | 说明 |
|--------|------|---|------|
| 成功奖励 | `rew_success + rew_success_time * (1 - time_ratio)` | 100 + 最多 30 | 成功条件持续 1s |
| **超时惩罚** `pen_timeout` | `rew_timeout if (timeout and not success)` | -10.0 | **S1.0j 新增：跑满时间未成功的显著负信号** |

---

## 3. 设计原则

### 3.1 距离自适应

所有阈值和系数随 `dist_front` 在 `[d_close=1.1m, d_far=2.6m]` 之间 smoothstep 插值：

- **远处**：高接近系数(10)、低对齐系数(2)、宽松对齐阈值(0.6m/30 deg)
- **近处**：低接近系数(8)、高对齐系数(10)、严格对齐阈值(0.1m/10 deg)

### 3.2 势函数 shaping（防奖励泵）

插入和举升使用势函数 shaping：`r = gamma * Phi_t - Phi_{t-1}`（gamma=0.99）

### 3.3 偏航目标分段

- **远处**：yaw 目标 = 叉车到托盘中心方向
- **近处**：yaw 目标 = 托盘朝向（平行对接）
- 通过 `w_far/w_close` 平滑混合

### 3.4 首步保护

所有增量奖励在 episode reset 后的第一步被清零（`_is_first_step` 标记）。

### 3.5 2D 接近距离（S1.0j 新增）

使用 `d_xy = sqrt(dist_front^2 + y_err^2)` 替代 1D `dist_front` 作为接近度量，使横向修正也产生正向接近奖励，避免"横向修正是纯成本"的问题。

---

## 4. S1.0j 相对 S1.0i 的修改

### 4.1 问题诊断

S1.0i 训练 2000 iter 后：Mean reward=+15.43，但 frac_inserted=0%，dist_front 卡在 0.57m。

**根因**：`r_forward`（前进速度奖励）每步 +0.073，是最大正项。策略通过"制造前进速度但不产生净位移"即可获得正收益，形成"速度刷分"局部最优。

三股力叠加：
1. `r_forward` 奖励车体前向速度，不是朝托盘入口的有效推进
2. `r_approach` 被 `w_approach` 门控压低至 ~0.33x，无法与 r_forward 竞争
3. 无全距离距离惩罚，策略在 0.55m 处无"应更近"的压力

### 4.2 改动清单

| 项目 | S1.0i | S1.0j | 原因 |
|------|-------|-------|------|
| `r_forward` | `k_forward * clamp(v_forward, 0)` | **删除** | 速度!=位移，策略可原地晃动刷分 |
| `k_forward` | 0.5 | **删除** | 随 r_forward 删除 |
| `r_approach` 门控 | `w_approach = 0.2+0.8*w_ready` (~0.33x) | **无门控** | 有效强度从 ~2.7 恢复到 ~8-10 |
| `r_approach` 距离 | 1D `dist_front`（仅 x 轴） | **2D `d_xy`** | 横向修正也算接近 |
| `pen_dist_cont` | 无 | **新增** `-0.03*d_xy` | 所有距离上持续施压 |
| `pen_timeout` | 无 | **新增** `-10.0` | 超时未成功终局惩罚 |

### 4.3 预期效果

| 场景 | S1.0i 每 episode | S1.0j 每 episode |
|------|------------------|------------------|
| 原地不动 359 步 | +15（靠 r_forward 撑正） | **-18**（不再有利可图） |
| 0.1m/s 主动接近 | ~0 | **-11**（比原地好 +7 分） |

---

## 5. 成功判定条件

同时满足以下条件并持续 1 秒：

| 条件 | 阈值 |
|------|------|
| 插入深度 >= 2/3 托盘深度 | insert_depth >= 1.44m |
| 横向误差 <= 3cm | y_err <= 0.03m |
| 偏航误差 <= 3 deg | yaw_err <= 3.0 deg |
| 举升高度 >= 12cm | lift_height >= 0.12m |

---

## 6. 终止条件

| 条件 | 阈值 |
|------|------|
| 超时 | 12s（359 步）+ pen_timeout=-10 |
| 倾翻 | roll/pitch > 25 deg |
| 成功 | 持续 hold 1s |

---

## 7. 所有参数速查表

| 参数 | 值 | 所在文件 |
|------|---|---------|
| `d_far` | 2.6 m | env_cfg.py |
| `d_close` | 1.1 m | env_cfg.py |
| `lat_ready_far` | 0.6 m | env_cfg.py |
| `lat_ready_close` | 0.10 m | env_cfg.py |
| `yaw_ready_far_deg` | 30 deg | env_cfg.py |
| `yaw_ready_close_deg` | 10 deg | env_cfg.py |
| `insert_gate_norm` | 0.60 | env_cfg.py |
| `insert_ramp_norm` | 0.10 | env_cfg.py |
| `k_app_far / k_app_close` | 10.0 / 8.0 | env_cfg.py |
| `k_align_far / k_align_close` | 2.0 / 10.0 | env_cfg.py |
| `k_ins_far / k_ins_close` | 8.0 / 15.0 | env_cfg.py |
| `k_pre_far / k_pre_close` | 12.0 / 5.0 | env_cfg.py |
| `k_align_abs` | 0.05 | env_cfg.py |
| `k_lift` | 20.0 | env_cfg.py |
| `k_dist_far` | 0.3 | env_cfg.py |
| `k_dist_cont` | 0.03 | env_cfg.py |
| `rew_action_l2` | -0.01 | env_cfg.py |
| `rew_time_penalty` | -0.003 | env_cfg.py |
| `rew_success` | 100.0 | env_cfg.py |
| `rew_success_time` | 30.0 | env_cfg.py |
| `rew_timeout` | -10.0 | env_cfg.py |
| `insert_fraction` | 2/3 | env_cfg.py |
| `lift_delta_m` | 0.12 m | env_cfg.py |
| `hold_time_s` | 1.0 s | env_cfg.py |
| `max_lateral_err_m` | 0.03 m | env_cfg.py |
| `max_yaw_err_deg` | 3.0 deg | env_cfg.py |
| `pallet_depth_m` | 2.16 m | env_cfg.py |
