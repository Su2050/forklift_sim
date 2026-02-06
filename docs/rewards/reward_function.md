# 奖励函数 S0.4 → S0.5 → S0.6 → S0.7 多阶段训练方案

> **当前活跃版本：S1.0c**，自 2026-02-07 起生效。  
> 代码位置：`IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/forklift_pallet_insert_lift/env.py` + `env_cfg.py`

---

## 1. 设计背景与演进

### 1.1 问题历程

| 版本 | 核心问题 | 根因 |
|------|----------|------|
| v4 (Gate 机制) | 策略在托盘前反复蹭，永远进不了 Phase1 | 接近+对齐绑太紧，phase 切换条件过于苛刻 |
| S0 (极简两阶段) | 对齐好但不敢前进 | `w_ready` 门控太严（lat=0.10m, yaw=10°），接近奖励被完全抑制 |
| S0.2 | 远处对齐躺平 | `k_approach` 过高（25），`w_ready` 门控阻止接近 |
| S0.3 | 接近但停在 1.5m 处空举 | `w_ready` 仍太严，策略陷入"对齐不够→不给奖→不知道接下去做什么" |
| **S0.4** | **学会前进+插入+举升，但对齐恶化** | 对齐信号太弱 `|r_align|=0.006` vs `r_approach=0.019` |
| **S0.5** | **对齐恢复到 4.3°，但不敢前进插入** | `k_align=8` 过强，与 `k_approach=5` 在 0.85m 处互相抵消 |
| **S0.6** | **打破平衡点，恢复插入动力（进行中）** | 调整 approach:align 比值从 0.625:1 到 1.33:1 |

### 1.2 核心设计思想

**无条件接近 + 对齐 → 软门控插入 → 严格门控举升**

抛弃了硬性的 phase 状态机，改用两个软门控权重实现自然过渡：

```
接近（无条件） ──┐
                ├──→ 插入（w_ready 门控） ──→ 举升（w_lift 门控）
对齐（无条件） ──┘
```

关键突破：
- **接近奖励不受对齐门控**，避免"远处对齐躺平"
- **对齐信号双向**（改善得分，恶化扣分），持续施加对齐压力
- **分层门控**：对齐质量影响插入收益，插入深度影响举升收益

---

## 2. 二阶段训练策略

### 2.1 为什么分多阶段？

单次训练难以同时学好所有技能。`k_approach`、`k_align`、`k_insert` 三者之间存在竞争关系，单一比值无法覆盖所有学习阶段。多阶段训练的核心思路是：

1. **S0.4（基础能力）**：高 approach，低 align → 先学会接近和操作
2. **S0.5（对齐修正）**：高 align，低 approach → 修正对齐，代价是暂时抑制前进
3. **S0.6（均衡调优）**：适中 approach，适中 align，高 insert → 打破平衡，实现对齐+插入并进

每阶段从上一阶段的 checkpoint 恢复，保留已学到的能力，同时微调目标方向。

### 2.2 阶段一：S0.4 基础能力训练（从零开始）

**目标**：让策略学会完整的 接近→插入→举升 流程

**参数配置**：

```python
# 阶段阈值（放宽的门控阈值）
lat_ready_m    = 0.50    # 放宽 5 倍（S0 的 0.10→0.50），让 w_ready 更早 > 0
yaw_ready_deg  = 30.0    # 放宽 3 倍（S0 的 10→30），降低对齐门槛

# 举升门控
insert_gate_norm = 0.60  # 插入深度达到 60% 才开始给举升奖励
insert_ramp_norm = 0.10  # 60%→70% 线性过渡

# 奖励系数
k_align          = 2.0   # 对齐改善奖励
k_approach       = 8.0   # 接近奖励（无条件，双向）
k_insert         = 10.0  # 插入进度奖励
k_lift           = 20.0  # 举升奖励
k_wrongins       = 4.0   # 斜怼惩罚
k_dist_far       = 0.3   # 距离过远惩罚
k_premature_lift = 5.0   # 空举惩罚（未插入时举升扣分）

# 时间压力
rew_time_penalty = -0.003

# 终点奖励
rew_success      = 100.0
rew_success_time = 30.0

# 其他
rew_action_l2    = -0.01
```

**PPO 算法参数**：

```python
entropy_coef = 0.02    # 提高（默认 0.01→0.02），防止过早收敛
# 其余保持默认：lr=3e-4, gamma=0.99, lam=0.95, clip=0.2
```

**训练命令**：

```bash
cd /home/uniubi/projects/forklift_sim/IsaacLab
nohup ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Forklift-PalletInsertLift-Direct-v0 \
  --num_envs 1024 --max_iterations 2000 --headless \
  > ../20260206_train_s0.4.log 2>&1 &
```

**训练结果（iter 1050，约 70 分钟）**：

| 指标 | 起始值 | 终态 | 评价 |
|------|--------|------|------|
| dist_front | 2.88m | **0.53m** | 学会接近 |
| insert_norm | 0.00 | **0.38** | 学会插入 |
| frac_inserted | 0% | **22.5%** | 有效插入 |
| frac_lifted | 0% | **62%** | 学会举升 |
| frac_success_now | 0% | **0.1%** | 首次成功 |
| yaw_deg | 26.6° | **37.4°** | 恶化（问题） |
| lateral | 0.31m | **0.66m** | 恶化（问题） |
| noise_std | 1.0 | **24.17** | 探索充足 |

**关键发现**：对齐信号 `|r_align|≈0.006` 被 `r_approach≈0.019` 和 `r_lift≈0.022` 淹没，导致策略忽视对齐。

### 2.3 阶段二：S0.5 对齐微调（从 checkpoint 恢复）

**目标**：在保留接近/插入能力的基础上，强化对齐精度

**参数修改（仅 3 个）**：

```python
k_align    = 8.0   # 2.0 → 8.0（4 倍，让对齐信号不被淹没）
k_wrongins = 8.0   # 4.0 → 8.0（2 倍，加强斜怼惩罚）
k_approach = 5.0   # 8.0 → 5.0（降低，策略已会前进，减少对齐竞争）
```

**其他参数全部不变**，保持训练稳定性。

**恢复训练命令**：

```bash
cd /home/uniubi/projects/forklift_sim/IsaacLab
TERM=xterm CONDA_PREFIX= CONDA_DEFAULT_ENV= \
nohup ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Forklift-PalletInsertLift-Direct-v0 \
  --num_envs 1024 --max_iterations 2000 --headless \
  --resume --load_run 2026-02-06_09-39-12 --checkpoint model_1050.pt \
  > ../20260206_train_s0.5.log 2>&1 &
```

**训练结果（iter 1050→1850，共 800 轮，约 55 分钟）**：

| 指标 | iter 1050 (起点) | iter 1066 (谷底) | iter 1376 (稳定) | iter 1850 (终态) |
|------|------------------|------------------|------------------|------------------|
| yaw_deg | 7.1° | **46.3°** | **5.5°** | **4.3°** |
| lateral | 0.31m | **0.57m** | **0.34m** | **0.30m** |
| dist_front | 2.48m | 0.52m | 0.86m | **0.85m** |
| insert_norm | 0.00 | 0.18 | 0.10 | **0.10** |
| frac_inserted | 0% | 10% | 6.5% | **5.9%** |
| frac_aligned | 1.2% | 0.5% | 3.2% | **4.3%** |
| frac_success_now | 0% | 0% | 0.2% | **0.1%** (散发) |
| reward | -7.7 | **-83** | -50 | **-50** |
| noise_std | 25.4 | 25.6 | 32.2 | **41.3** |
| w_ready | 0.24 | 0.17 | 0.31 | **0.35** |

**训练过程四阶段**：

1. **奖励震荡期**（iter 1050→1066，约 16 轮）：value function 适应新奖励尺度，yaw 从 7° 飙到 46°，reward 从 -7 跌到 -83。这是恢复训练的正常现象。
2. **对齐恢复期**（iter 1066→1149，约 83 轮）：k_align=8.0 发挥作用，yaw 从 46° 恢复到 10°，lateral 从 0.57m 恢复到 0.34m。`r_align` 在 iter 1149 首次变为正值。
3. **对齐精细化期**（iter 1149→1500，约 350 轮）：yaw 从 10° 进一步降到 4.5°，frac_aligned 达到 4.3%（历史最高）。
4. **瓶颈停滞期**（iter 1500→1850，约 350 轮）：对齐稳定在 yaw≈4.5° 但 dist_front 卡在 0.85m 不再减小，insert_norm 在 0.05~0.10 徘徊。noise_std 异常飙升到 41（PPO 在无效探索）。

**S0.5 关键成就**：对齐从 S0.4 的 37° 降到 4.3°（改善 8.6 倍）。

**S0.5 暴露的问题**：`k_align=8` 与 `k_approach=5` 在 dist_front≈0.85m 处形成奖励平衡点。前进时 `r_approach≈+0.008` 恰好被对齐恶化的 `r_align≈-0.008` 抵消，策略选择"远处站桩对齐"而不敢前进插入。

### 2.4 阶段三：S0.6 插入动力微调（从 S0.5 checkpoint 恢复）

**目标**：打破 0.85m 平衡点，在保持对齐的同时恢复前进和插入能力

**瓶颈分析**：

```
S0.5 在 dist_front=0.85m 的奖励信号：
  前进: r_approach = k_approach(5) × delta_dist ≈ +0.008
  对齐恶化: r_align = k_align(8) × delta_E_align ≈ -0.008
  净收益 ≈ 0  →  策略选择原地不动
```

**参数修改（仅 3 个）**：

```python
k_align    = 6.0   # 8.0 → 6.0（降 25%，仍为 S0.4 的 3 倍，保留对齐能力）
k_approach = 8.0   # 5.0 → 8.0（提高 60%，恢复前进动力）
k_insert   = 15.0  # 10.0 → 15.0（提高 50%，增强插入正反馈）
```

**其他参数不变**（k_wrongins=8, k_lift=20, k_premature_lift=5, entropy_coef=0.02）。

**修改后平衡分析**（以 w_ready=0.35 为例）：

```
S0.6 在 dist_front=0.85m 的奖励信号：
  前进: r_approach = k_approach(8) × delta_dist ≈ +0.013
  对齐恶化: r_align = k_align(6) × delta_E_align ≈ -0.006
  净收益 ≈ +0.007  →  策略有正向激励继续前进

  有效插入信号 = (15 - 8×0.65) × progress = 9.8 × progress（比 S0.5 翻倍）
```

**恢复训练命令**：

```bash
cd /home/uniubi/projects/forklift_sim/IsaacLab
TERM=xterm CONDA_PREFIX= CONDA_DEFAULT_ENV= \
nohup ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Forklift-PalletInsertLift-Direct-v0 \
  --num_envs 1024 --max_iterations 2000 --headless \
  --resume --load_run 2026-02-06_10-59-36 --checkpoint model_1850.pt \
  > ../20260206_train_s0.6.log 2>&1 &
```

**预期效果**：

- 对齐可能从 4.3° 暂时恶化到 8-12°（可接受，仍远优于 S0.4 的 37°）
- dist_front 应从 0.85m 下降到 0.4-0.6m
- insert_norm 应从 0.05 上升到 0.2+
- noise_std 应随着找到明确方向而开始下降

---

## 3. 奖励函数实现细节

### 3.1 软门控权重

```python
# E_align: 归一化对齐误差综合指标
E_align = y_err / lat_ready_m + yaw_err_deg / yaw_ready_deg

# w_ready: 对齐就绪软权重 [0, 1]
# E_align < 1 时 w_ready > 0（横向误差 < 0.5m 且偏航 < 30° 时开始生效）
w_ready = clamp(1.0 - E_align, min=0.0, max=1.0)

# w_lift: 举升门控软权重 [0, 1]
# 插入深度达到 60% 后开始线性增长，70% 达到满权重
w_lift = clamp((insert_norm - 0.60) / 0.10, min=0.0, max=1.0)
```

### 3.2 奖励项定义（7 项奖励 + 3 项常驻惩罚 + 1 项终点奖励）

#### 阶段 1：接近 + 对齐（无条件，始终生效）

```python
# 1a. 接近奖励（双向：靠近得分，远离扣分）
r_approach = k_approach * delta_dist
# delta_dist = last_dist - current_dist，接近为正

# 1b. 对齐改善奖励（双向：改善得分，恶化扣分）
r_align = k_align * delta_E_align
# delta_E_align = last_E_align - E_align，改善为正
```

#### 阶段 2：插入（软门控于对齐质量）

```python
# 2a. 插入进度奖励（只奖励正向插入）
r_insert = k_insert * clamp(progress, min=0.0)

# 2b. 斜怼惩罚（未对齐时的正向插入会被惩罚）
pen_wrongins = -k_wrongins * (1.0 - w_ready) * clamp(progress, min=0.0)
# 当 w_ready=0 时惩罚最大，w_ready=1 时惩罚为零
```

#### 阶段 3：举升（严格门控于插入深度）

```python
# 3a. 举升奖励（只在插入够深时给予）
r_lift = k_lift * w_lift * clamp(delta_lift, min=0.0)

# 3b. 空举惩罚（未插入时的举升会被惩罚）
pen_premature = -k_premature_lift * (1.0 - w_lift) * clamp(delta_lift, min=0.0)
# 当 w_lift=0 时惩罚最大，w_lift=1 时惩罚为零
```

#### 常驻项

```python
# 动作平滑惩罚
rew_action = rew_action_l2 * sum(actions^2)   # -0.01 * ||a||²

# 时间压力（每步固定惩罚，鼓励高效完成）
rew_time = rew_time_penalty                     # -0.003

# 距离过远惩罚（防止远处躺平）
pen_dist_far = -k_dist_far * clamp(dist_front - 2.0, min=0.0)
```

#### 终点奖励

```python
# 成功条件：插入 >= 2/3 深度 且 对齐误差 < 阈值 且 举升 >= 0.12m 且 保持 >= 1s
success = inserted_enough & aligned_enough & lifted_enough & hold_time_met

# 奖励 = 基础奖励 + 时间奖金（越早完成奖金越高）
success_reward = 100.0 + 30.0 * (1.0 - time_ratio)
```

### 3.3 成功判定标准

| 条件 | 阈值 | 说明 |
|------|------|------|
| 插入深度 | ≥ 2/3 × 2.16m = 1.44m | 货叉插入托盘的归一化深度 |
| 横向误差 | ≤ 0.03m | 严格对齐 |
| 偏航误差 | ≤ 3.0° | 严格对齐 |
| 举升高度 | ≥ 0.12m | 相对初始叉尖高度 |
| 保持时间 | ≥ 1.0s | 连续保持以上所有条件 |

### 3.4 首步保护

所有增量型奖励在 episode 首步跳过（输出 0），防止缓存初始化导致的大跳变：

```python
r_approach = where(is_first_step, 0, r_approach)
r_align    = where(is_first_step, 0, r_align)
r_lift     = where(is_first_step, 0, r_lift)
pen_premature = where(is_first_step, 0, pen_premature)
```

---

## 4. 完整参数表（S0.6 当前生效值）

### 4.1 奖励系数演变总表

| 参数 | S0 (旧) | S0.4 | S0.5 | **S0.6 (当前)** | 说明 |
|------|---------|------|------|-----------------|------|
| **`k_align`** | 1.0 | 2.0 | 8.0 | **6.0** | 对齐改善奖励 |
| **`k_approach`** | 12.0 | 8.0 | 5.0 | **8.0** | 接近奖励 |
| **`k_insert`** | 10.0 | 10.0 | 10.0 | **15.0** | 插入进度奖励 |
| `k_lift` | 20.0 | 20.0 | 20.0 | 20.0 | 举升奖励 |
| `k_wrongins` | 6.0 | 4.0 | 8.0 | 8.0 | 斜怼惩罚 |
| `k_dist_far` | - | 0.3 | 0.3 | 0.3 | 距离过远惩罚 |
| `k_premature_lift` | - | 5.0 | 5.0 | 5.0 | 空举惩罚 |
| **approach:align 比值** | **12:1** | **4:1** | **0.625:1** | **1.33:1** | **核心调控比值** |

> **approach:align 比值**是最关键的调控杠杆。过高（>4:1）策略冲但不对齐，过低（<1:1）策略对齐但不前进。1~2:1 是理想区间。

### 4.2 其他奖励参数 (`env_cfg.py`)

| 参数 | 值 | 说明 |
|------|-----|------|
| `lat_ready_m` | 0.50 | 对齐就绪横向阈值 |
| `yaw_ready_deg` | 30.0 | 对齐就绪偏航阈值 |
| `insert_gate_norm` | 0.60 | 举升门控插入深度阈值 |
| `insert_ramp_norm` | 0.10 | 举升权重线性过渡区间 |
| `rew_time_penalty` | -0.003 | 每步时间惩罚 |
| `rew_success` | 100.0 | 成功基础奖励 |
| `rew_success_time` | 30.0 | 时间奖金上限 |
| `rew_action_l2` | -0.01 | 动作 L2 惩罚 |

### 4.2 PPO 算法参数 (`rsl_rl_ppo_cfg.py`)

| 参数 | 值 | 说明 |
|------|-----|------|
| `num_steps_per_env` | 64 | 每环境收集步数 |
| `actor_hidden_dims` | [256, 256, 128] | Actor 网络结构 |
| `critic_hidden_dims` | [256, 256, 128] | Critic 网络结构 |
| `activation` | elu | 激活函数 |
| `learning_rate` | 3e-4 | 学习率（自适应调度） |
| `gamma` | 0.99 | 折扣因子 |
| `lam` | 0.95 | GAE lambda |
| `entropy_coef` | 0.02 | 熵系数（提高以防过早收敛） |
| `clip_param` | 0.2 | PPO clip 范围 |
| `num_learning_epochs` | 5 | 每次更新的 epoch 数 |
| `num_mini_batches` | 4 | mini-batch 数量 |

### 4.3 环境参数

| 参数 | 值 | 说明 |
|------|-----|------|
| `num_envs` | 1024 | 并行环境数 |
| `episode_length_s` | 45.0s | 最大 episode 时长 |
| `decimation` | 4 | 控制频率 = 120/4 = 30Hz |
| `wheel_speed_rad_s` | 15.0 | 驱动轮最大角速度 |
| `steer_angle_rad` | 0.5 | 最大转向角 |
| `lift_speed_m_s` | 1.0 | 举升速度 |

---

## 5. 观测空间与动作空间

### 5.1 观测空间（13 维）

| 索引 | 名称 | 说明 |
|------|------|------|
| 0 | `d_xy_r_x` | 机器人坐标系下托盘相对位置 X |
| 1 | `d_xy_r_y` | 机器人坐标系下托盘相对位置 Y |
| 2 | `cos_dyaw` | 偏航角差的余弦 |
| 3 | `sin_dyaw` | 偏航角差的正弦 |
| 4 | `v_xy_r_x` | 机器人坐标系下速度 X |
| 5 | `v_xy_r_y` | 机器人坐标系下速度 Y |
| 6 | `yaw_rate` | 偏航角速度 |
| 7 | `lift_pos` | 举升关节位置 |
| 8 | `lift_vel` | 举升关节速度 |
| 9 | `insert_norm` | 归一化插入深度 |
| 10-12 | 其他 | 预留 |

### 5.2 动作空间（3 维，[-1, 1]）

| 索引 | 名称 | 缩放 | 说明 |
|------|------|------|------|
| 0 | drive | ×15.0 rad/s | 前后驱动 |
| 1 | steer | ×0.5 rad | 左右转向 |
| 2 | lift | ×1.0 m/s | 举升速度 |

---

## 6. 日志监控指标

训练日志中输出以下关键指标用于实时监控：

### 6.1 核心诊断

| 指标 | 含义 | 健康范围 |
|------|------|----------|
| `s0/E_align` | 归一化对齐误差 | 越小越好，< 1.0 表示在门控阈值内 |
| `s0/w_ready` | 对齐就绪权重 | 0.2+ 表示有基本对齐 |
| `s0/w_lift` | 举升门控权重 | > 0 表示插入足够深 |

### 6.2 奖励分量

| 指标 | 含义 | 期望 |
|------|------|------|
| `s0/r_align` | 对齐改善奖励 | 正值=在改善 |
| `s0/r_approach` | 接近奖励 | 正值=在接近 |
| `s0/r_insert` | 插入进度奖励 | 正值=在插入 |
| `s0/r_lift` | 举升奖励 | 正值=有效举升 |
| `s0/pen_wrongins` | 斜怼惩罚 | 接近 0 最好 |
| `s0/pen_premature` | 空举惩罚 | 接近 0 最好 |
| `s0/pen_dist_far` | 远距惩罚 | 接近 0 最好 |

### 6.3 误差与阶段

| 指标 | 含义 |
|------|------|
| `err/yaw_deg_mean` | 平均偏航误差（度） |
| `err/lateral_mean` | 平均横向误差（米） |
| `err/dist_front_mean` | 平均前距（米） |
| `err/insert_norm_mean` | 平均归一化插入深度 |
| `phase/frac_inserted` | 插入达标比例 |
| `phase/frac_aligned` | 对齐达标比例 |
| `phase/frac_lifted` | 举升达标比例 |
| `phase/frac_success_now` | 当前步成功比例 |

---

## 7. 经验教训

### 7.1 奖励系数设计

1. **approach:align 比值是核心杠杆**：过高（>4:1）策略冲但不对齐（S0.4），过低（<1:1）策略对齐但不前进（S0.5）。1~2:1 是理想区间（S0.6 使用 1.33:1）。
2. **竞争性奖励会形成平衡点**：当两个奖励信号大小相当但方向相反时，策略会找到一个"既不前进也不后退"的安全平衡点。S0.5 中 `k_approach=5` 和 `k_align=8` 在 0.85m 处精确抵消。
3. **增量奖励优于绝对奖励**：使用 `delta_E_align` 而非 `E_align` 避免了"站桩刷分"，策略必须持续改善才能得到正奖励。
4. **空举惩罚很重要**：没有 `k_premature_lift` 时策略会在插入前就开始举升，浪费行为能力。

### 7.2 门控与阈值

5. **门控阈值要放宽**：严格阈值（lat=0.1m, yaw=10°）会导致 `w_ready≈0`，所有门控奖励失效。放宽到 lat=0.5m, yaw=30° 后学习显著加速。

### 7.3 训练策略

6. **多阶段微调优于单次训练**：复杂任务中多个技能目标（接近、对齐、插入、举升）相互竞争，单一参数配置很难同时优化所有目标。分阶段逐步微调更稳定有效。
7. **恢复训练可行但有 spike**：从 checkpoint 恢复后 value function loss 会短暂飙升（奖励尺度变化导致），通常 50-100 轮后稳定。
8. **entropy_coef 不能太低**：默认 0.01 导致 noise_std 快速收敛到 0.27，策略陷入局部最优。提高到 0.02 后 noise_std 保持在 25+ 确保充分探索。

### 7.4 诊断信号

9. **noise_std 持续飙升是危险信号**：S0.5 中 noise_std 从 25 飙升到 41 且不收敛，说明 PPO 在大量随机探索但找不到有效策略改进方向，即策略已陷入瓶颈。
10. **reward 平台期 + noise_std 上升 = 需要干预**：如果 reward 连续 200+ 轮无改善且 noise_std 持续上升，应考虑调整参数而非继续等待。

---

## 8. S0.7：收敛微调 + 对齐信号强化

### 8.1 S0.6 终态分析

S0.6 跑完 2000 轮（iter 1850→3849），核心发现：

| 指标 | iter 1900 (起始) | iter 3849 (终态) | 趋势 |
|------|------------------|------------------|------|
| reward | -47.6 | -35.1 | 缓慢改善 |
| dist_front | 0.95m | 0.57m | 显著改善 |
| yaw_deg | 4.5° | 8.4° | 恶化 |
| insert_norm | 0.06 | 0.14 | 改善 |
| frac_lifted | 20% | 63% | 大幅改善 |
| noise_std | 42.7 | **79.3** | 持续飙升 |
| frac_success 累计 | 0 | 245 次 | 偶发，无增长趋势 |

**核心问题**：`noise_std` 从 42 爆炸到 79，策略越来越随机。`entropy_coef=0.02` 过高，PPO 的 entropy bonus 阻止策略收敛。

### 8.2 S0.7 策略

**目标**：抑制 noise_std 爆炸，让策略在 S0.6 学到的接近/插入基础上收敛。

**参数变更**（基于 S0.6）：

| 参数 | S0.6 | S0.7 | 原因 |
|------|------|------|------|
| `entropy_coef` | 0.02 | **0.005** | 4x 降低，抑制 noise_std 爆炸 |
| `k_align` | 6.0 | **10.0** | 强化对齐信号，对齐在 S0.6 恶化 |

**恢复策略**：从 `model_2600.pt`（S0.6 成功密度最高区间）恢复，训练 2000 轮。

**代码重建**：原始文件因 `git clean` 丢失（untracked 文件被清理）。基于以下四重交叉验证重写：
1. 备份文件骨架
2. S0.6 `env.yaml` / `agent.yaml` 精确参数
3. 对话 transcript 中完整的 S0.4 `_get_rewards()` 代码
4. `.pyc` 字节码结构分析

### 8.3 代码重建后的三次关键 Bug 修复

从备份代码重建 `env.py` 后，干跑和初始训练暴露了三个严重 bug，依次修复：

#### Bug 1：`sim.reset()` 性能灾难（25x 减速）

| | |
|---|---|
| **现象** | 训练速度 830 steps/s，S0.6 基线 16,600 steps/s，collection 78s vs 2.8s |
| **根因** | `_reset_idx` 中保留了备份代码的 `self.scene.write_data_to_sim()` + `self.sim.reset()` + `self.scene.update()`。`sim.reset()` 会触发**完整的 PhysX 仿真重置**，而非仅重置指定环境，在每次 episode reset 时反复执行极其昂贵。同时引发大量 `PhysX collision - triangle mesh falling back to convexHull` 警告 |
| **修复** | 删除 `sim.reset()` 和 `scene.write_data_to_sim()`（由 `super()._reset_idx()` 负责）。增量奖励基线从 reset 的随机化参数直接推算，不需要额外的仿真步来读回 |
| **效果** | 速度恢复到 11,000-14,000 steps/s，碰撞警告从几百条降到 1 条 |

#### Bug 2：fork tip 重复计算 + PhysX 张量浪费

| | |
|---|---|
| **现象** | 优化 Bug 1 后速度仍比 S0.6 低约 30%（12,000 vs 16,600） |
| **根因** | `_compute_fork_tip()` 每步调用 3 次（action/obs/reward），每次遍历所有 body 做 argmax。PhysX fallback 每步 `.clone()` 全关节目标张量并重新创建 `env_indices` |
| **修复** | (1) fork tip 缓存：`_pre_physics_step` 失效缓存，`_get_observations` 和 `_get_rewards` 共享同一步的缓存值。(2) `_apply_action` 中 lock_drive_steer 改用上一步的 `_last_insert_depth` 和 `_last_E_align` 近似，避免额外计算。(3) 预分配 `_all_env_indices`，去掉 `targets.clone()` |
| **效果** | 进一步提升到 12,000-15,000 steps/s |

#### Bug 3：`super()._reset_idx()` 缺失（episode 长度卡在 1）

| | |
|---|---|
| **现象** | `Mean episode length: 1.00`，`frac_timeout: 1.0`，所有增量奖励为 0，策略完全无法学习 |
| **根因** | 重写的 `_reset_idx` 没有调用 `super()._reset_idx(env_ids)`。基类方法负责 `self.scene.reset(env_ids)`（刷新状态到仿真）和 `self.episode_length_buf[env_ids] = 0`（重置 episode 计数器）。缺失后 episode 计数器永远不归零，每步都满足 `>= max_episode_length - 1`，立即超时 |
| **修复** | 在 `_reset_idx` 开头添加 `super()._reset_idx(env_ids)` |
| **效果** | Episode 长度恢复到 1000-1350 步，所有奖励分量正常工作 |

> 详细的修复过程和代码变更见 [`docs/diagnostic_reports/env_py_bugfix_postmortem_2026-02-06.md`](../diagnostic_reports/env_py_bugfix_postmortem_2026-02-06.md)

### 8.4 S0.7 训练命令

```bash
cd /home/uniubi/projects/forklift_sim/IsaacLab && \
TERM=xterm CONDA_PREFIX= nohup ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Forklift-PalletInsertLift-Direct-v0 \
  --num_envs 1024 \
  --max_iterations 2000 \
  --headless \
  --resume --load_run 2026-02-06_11-58-37 --checkpoint model_2600.pt \
  > /home/uniubi/projects/forklift_sim/20260206_train_s0.7.log 2>&1 &
```

### 8.5 S0.7 早期训练数据（修复后，iter 2600-2668）

| 指标 | iter 2600 (起始) | iter 2668 (当前) | 趋势 |
|------|-----------------|-----------------|------|
| reward | -14.8 | -181.5 | 下降中（正常，episode 变长奖励变大） |
| episode_length | 31.9 | **1322** | 从异常恢复到正常 |
| noise_std | 57.71 | 57.74 | 暂未变化（从 S0.6 继承，需更多轮数） |
| dist_front | 2.81m | **1.81m** | 在改善，开始接近 |
| insert_norm | 0.000 | **0.213** | 开始插入 |
| frac_inserted | 0% | **10.7%** | 有效插入比例上升 |
| frac_lifted | 0% | **62.0%** | 在积极举升 |
| frac_aligned | 0% | 1.4% | 低，对齐待改善 |
| yaw_deg | 10.2° | 12.7° | 略恶化（适应期正常） |
| E_align | 1.16 | 1.32 | 略恶化 |
| r_approach | 0 → | +0.007 | 正向接近奖励 |
| r_insert | 0 → | +0.015 | 正向插入奖励 |
| r_lift | 0 → | +0.040 | 正向举升奖励 |
| hold_counter_max | 0 | 13 | 已出现接近成功的尝试 |
| 速度 | — | 11,000-15,000 steps/s | 正常 |
| ETA | — | ~2.5h | — |

**初步评估**：训练正常运行。从 S0.6 的 model_2600.pt 恢复后，策略在 68 轮内开始重新学会接近、插入和举升。noise_std 暂未变化，需要更多轮数观察 `entropy_coef=0.005` 的收敛效果。

### 8.6 预期效果

- **noise_std**：应在 200-400 轮内从 57 开始下降，目标 < 20
- **对齐**：`k_align=10` 配合收敛应使 yaw_deg 稳定在 < 5°
- **成功率**：如果 noise_std 有效下降，frac_success 应呈现持续增长而非偶发

### 8.7 参数总表（S0/S0.4/S0.5/S0.6/S0.7）

| 参数 | S0 | S0.4 | S0.5 | S0.6 | **S0.7** |
|------|-----|------|------|------|----------|
| k_align | 5.0 | 5.0 | 8.0 | 6.0 | **10.0** |
| k_approach | 5.0 | 5.0 | 5.0 | 8.0 | 8.0 |
| k_insert | 10.0 | 10.0 | 10.0 | 15.0 | 15.0 |
| k_lift | 20.0 | 20.0 | 20.0 | 20.0 | 20.0 |
| entropy_coef | 0.01 | 0.02 | 0.02 | 0.02 | **0.005** |
| approach:align | 1:1 | 1:1 | 0.625:1 | 1.33:1 | **0.8:1** |

---

## 9. 模型路径与日志

| 阶段 | 训练目录 | Checkpoint | 日志文件 |
|------|----------|------------|----------|
| S0.4 | `logs/.../2026-02-06_09-39-12/` | model_1050.pt | `20260206_093906_train_s0.4.log` |
| S0.5 | `logs/.../2026-02-06_10-59-36/` | model_1850.pt | `20260206_train_s0.5.log` |
| S0.6 | `logs/.../2026-02-06_11-58-37/` | model_2600.pt (最佳) | `20260206_train_s0.6.log` |
| S0.7 | 从 S0.6 同目录恢复 | 进行中 | `20260206_train_s0.7.log` |

> 日志路径前缀：`/home/uniubi/projects/forklift_sim/`  
> 模型路径前缀：`IsaacLab/logs/rsl_rl/forklift_pallet_insert_lift/`

---

## 10. 经验教训（S0.7 新增）

11. **entropy_coef 是 noise_std 的直接控制杠杆**：`entropy_coef=0.02` 在 S0.4-S0.6 中持续导致 noise_std 爆炸（24→41→79），降低到 0.005 是让策略收敛的关键操作。
12. **untracked 文件必须 git add**：`git clean -fd` 会删除所有未追踪文件，包括关键源代码。始终将项目代码纳入版本控制。
13. **`.pyc` 文件是宝贵的恢复线索**：虽然 Python 3.11 反编译工具不成熟，但通过 `marshal`/`dis` 模块分析字节码，可以提取函数签名、局部变量、字符串常量等关键信息，配合 training yaml 和对话记录实现精确重建。
14. **重写 `_reset_idx` 必须调用 `super()`**：Isaac Lab 的 `DirectRLEnv._reset_idx` 负责 `scene.reset()`（刷新状态到仿真）和 `episode_length_buf` 归零。遗漏导致 episode 永远在第 1 步超时，所有增量奖励归零，训练完全失效。这类 bug 极其隐蔽——训练正常运行、不报错、日志格式正确，只有分析具体数值才能发现。
15. **`sim.reset()` ≠ `scene.reset()`**：`sim.reset()` 触发完整的 PhysX 物理引擎重置（极其昂贵），`scene.reset(env_ids)` 仅重置指定环境的缓冲区（轻量）。在 `_reset_idx` 中绝不能调用 `sim.reset()`。
16. **从备份恢复的代码需要全面验证**：即使静态检查和导入测试通过，运行时行为（性能、episode 长度、奖励数值）仍可能有严重偏差。应在正式训练前做 short run 并核对所有关键指标。

---

## 11. S1.0：距离自适应融合奖励函数（One-shot 多阶段训练）

### 11.1 设计动机

S0.4→S0.5→S0.6→S0.7 的"分阶段切权重"本质上在解决同一个矛盾：

- **远处的主要矛盾是"敢往前走"**
- **近处的主要矛盾是"别歪、别斜怼、把叉插进去并举起来"**
- **最后的主要矛盾是"别再乱抖了快收敛"**

S0.7c 的终态暴露了分阶段训练的局限性：

| 指标 | S0.7c 终态值 | 问题 |
|------|-------------|------|
| `dist_front_mean` | 2.25m | 卡在远处不接近 |
| `frac_lifted` | 59% | 远处空举严重 |
| `pen_premature` | -0.03 | 空举惩罚太弱（k=5.0） |
| `r_approach` | 0.005 | 前进信号微弱 |

**根因**：策略把 lift 当成廉价动作在远处刷，drive 的均值策略没学出来。

### 11.2 核心思路

不再分阶段手动切参数，而是让**系数随状态自动变形** —— 用 `dist_front` 做"课程进度条"：

- **远处（w_close≈0）**：高 approach + 低 align + 强制不许乱举升（≈S0.4）
- **中近（w_close↑）**：align 权重抬升、wrongins 抬升（≈S0.5）
- **近处且对齐好**：insert 权重抬升、靠近依然有正收益（≈S0.6）
- **进入插入/举升区间**：强化稳定/收敛（≈S0.7）

### 11.3 距离自适应权重：smoothstep

```python
def smoothstep(x):
    x = torch.clamp(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)

d_far   = 2.6   # 远端（略大于 S0.7c 卡住的 2.2m）
d_close = 1.1   # 近端（进入精对齐区间）
w_close = smoothstep((d_far - dist_front) / (d_far - d_close))  # 远=0 近=1
w_far   = 1.0 - w_close
```

选择 `smoothstep` 而非硬分段的原因：两端导数为零，过渡区平滑，不会在阈值处产生奖励跳变。

### 11.4 动态系数插值

5 个关键系数按距离自动从"远处值"过渡到"近处值"：

```python
k_approach = k_app_far * w_far + k_app_close * w_close
k_align    = k_align_far * w_far + k_align_close * w_close
k_insert   = k_ins_far * w_far + k_ins_close * w_close
k_wrongins = k_wrong_far * w_far + k_wrong_close * w_close
k_premature = k_pre_far * w_far + k_pre_close * w_close
```

### 11.5 参数总表

| 参数 | 远处值 (w_far=1) | 近处值 (w_close=1) | 设计意图 |
|------|-----------------|-------------------|----------|
| `k_approach` | **10.0** | 8.0 | 远处强拉前进，近处略降但保持正向动力 |
| `k_align` | 2.0 | **10.0** | 远处弱（允许探索），近处强（精对齐） |
| `k_insert` | 8.0 | **15.0** | 近处强化插入收益 |
| `k_wrongins` | 2.0 | **8.0** | 远处弱（允许探索），近处强（防斜怼） |
| `k_premature` | **12.0** | 5.0 | 远处极强（按死乱举升），近处放松（允许合理举升） |

其他不随距离变化的参数：`k_lift=20.0`、`k_dist_far=0.3`。

### 11.6 新增奖励项

**远距离前进速度奖励 `r_forward`**：

```python
r_forward = k_forward * w_far * clamp(v_xy_r_x, min=0.0)   # k_forward = 0.02
```

作用：让"向前开"在远处变成明确可学习的技能，而不是靠 delta_dist 的偶然正反馈。只在远处生效（`w_far`），近处自动消失。

### 11.7 wrongins 改为二次项

```python
# S0.7:  pen_wrongins = -k * (1.0 - w_ready)   * clamp(progress)
# S1.0:  pen_wrongins = -k * (1.0 - w_ready)**2 * clamp(progress)
```

目的：减少"抵消平衡点"。当 `w_ready=0.7`（差一点就对齐了），旧版惩罚系数为 `0.3 × k`，新版为 `0.09 × k`，让策略"差一点对齐也敢推进"。

### 11.8 PPO 参数

| 参数 | S0.7 值 | S1.0 值 | 原因 |
|------|---------|---------|------|
| `init_noise_std` | 1.0 | **3.0** | S0.7c 验证过，从零开始需要更大初始探索 |
| `entropy_coef` | 0.005 | 0.005 | 不变 |
| `max_iterations` | 2000 | 2000 | 不变 |

### 11.9 日志新增监控指标

| 指标 | 含义 | 用途 |
|------|------|------|
| `s0/w_close` | 距离自适应近处权重 | 确认 smoothstep 在工作 |
| `s0/w_far` | 距离自适应远处权重 | 与 w_close 互补 |
| `s0/r_forward` | 远距离前进速度奖励 | 远处应有正值 |
| `s0/k_approach_eff` | 实际生效的 approach 系数 | 远处≈10, 近处≈8 |
| `s0/k_align_eff` | 实际生效的 align 系数 | 远处≈2, 近处≈10 |
| `s0/k_premature_eff` | 实际生效的空举惩罚系数 | 远处≈12, 近处≈5 |

### 11.10 训练命令

```bash
cd /home/uniubi/projects/forklift_sim/IsaacLab && \
TERM=xterm CONDA_PREFIX= nohup ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Forklift-PalletInsertLift-Direct-v0 \
  --num_envs 1024 \
  --max_iterations 2000 \
  --headless \
  agent.run_name=exp_s1.0 \
  > /home/uniubi/projects/forklift_sim/20260207_train_s1.0.log 2>&1 &
```

### 11.11 验收指标

| 时间线 | 预期 |
|--------|------|
| 200-400 iters | `dist_front_mean` 从 2.7 降到 < 1.8 |
| 400-800 iters | `yaw_deg_mean` 开始明显下降 |
| 800+ iters | `w_ready_mean` > 0.2 后 `insert_norm_mean` 持续上升 |
| 1500+ iters | `frac_success_now` 出现趋势增长（非偶发） |
| 全程 | `frac_lifted` 在 `dist_front > 2.0` 时应显著低于 S0.7c 的 60% |

### 11.12 修改策略时的文件与模型兼容性

**每次改策略需要改哪些文件？**

| 改动类型 | 需改的文件 | 举例 |
|----------|-----------|------|
| 只调奖励系数数值 | `env_cfg.py` | 把 `k_app_far` 从 10 改到 12 |
| 改奖励函数逻辑/结构 | `env.py` + `env_cfg.py` | 加 smoothstep、r_forward |
| 改 PPO 超参 | `rsl_rl_ppo_cfg.py` | 调 entropy_coef、init_noise_std |
| 大版本升级 | 三个都动 | S0.7 → S1.0 |

**改完后旧模型能否 play？**

Play 时模型做的事是 `obs → 网络前向推理 → action`。奖励函数在 play 时不参与 action 计算，它只是旁路算个数字打到日志里。

因此：只要 obs 维度、action 维度、网络结构没变，旧模型加载后的行为（轨迹、动作）和以前完全一样。唯一变化的是日志里显示的 reward 数值会按新公式计算。

**会导致旧模型无法加载的情况：**

- 改了 `observation_space`（比如加了新观测维度）→ 网络输入维度不匹配
- 改了 `action_space` → 网络输出维度不匹配
- 改了网络结构（`hidden_dims`）→ 权重 shape 不匹配

S1.0 的 obs=13、action=3、网络 `[256, 256, 128]` 均未改动，所有 S0.4-S0.7 的模型都可以正常加载 play。

---

## 12. S1.0b：斜怼修复（中间迭代）

S1.0 训练暴露了"斜怼"局部最优：机器人以 22°+ 的歪角插入，`dist_front` 卡在 1.6m。

**S1.0b 参数调整**：

| 参数 | S1.0 | S1.0b | 原因 |
|------|------|-------|------|
| `d_close` | 1.1 | **1.8** | 让 k_align 更早到满值 |
| `k_ins_far` | 8.0 | **0.0** | 远处不给插入奖励 |
| `k_wrong_far` | 2.0 | **4.0** | 远处也加强斜怼惩罚 |

**S1.0b 训练结果**（iter 0→462）：斜怼问题未根本解决。`yaw_deg` 从 14° 恶化到 27°，`E_align` 从 1.34 恶化到 1.92，`w_ready` 停滞在 0.15 左右。

---

## 13. S1.0c：信号结构重构 — 两刀切断斜怼局部最优

### 13.1 S1.0b 失败的根因分析

从 S1.0b 训练日志中提取关键指标的演变趋势（iter 50→460）：

- **yaw_deg**: 14.2 → 17.2 → 21.0 → 25.2 → **27.1** (单调恶化)
- **E_align**: 1.34 → 1.49 → 1.64 → 1.79 → **1.92** (单调恶化)
- **insert_norm**: 0.16 → 0.38 → 0.54 → 0.57 → **0.57** (持续上升!)
- **r_align**: -0.005 → -0.000 → -0.010 → -0.001 → **+0.007** (在零附近震荡)

结论：机器人学会了"接近+插入"，但对齐在持续恶化。典型的"斜怼"局部最优。

### 13.2 三个结构性缺陷

#### 缺陷 1：`r_align` 是纯增量信号 → 稳态下梯度为零

`r_align = k_align * delta_E_align`：当机器人以稳定的歪角前进时，`E_align` 不变，`delta_E_align ≈ 0`，`r_align ≈ 0`。增量奖励只奖励"改善"，不惩罚"维持一个坏状态"。

#### 缺陷 2：net_insert 在任何对齐质量下恒为正 → 斜怼有利可图

```
net_insert = [k_insert - k_wrongins * (1 - w_ready)^2] * progress
```

在 iter 460（k_insert=10.35, k_wrongins=6.76）：
- w_ready=0（最差对齐）：net = 10.35 - 6.76 = **+3.59**（仍然为正！）
- 要让 net=0 需要 k_wrongins >= 10.35，当前只有 6.76

#### 缺陷 3：奖励量级不对等

- 斜怼收益：r_insert(0.037) + r_lift(0.131) = **+0.168/step**
- 停下对齐：r_align(~0.007, 不稳定) = **+0.007/step**
- 比值 ≈ 24:1，策略理性地选择"继续斜怼"

### 13.3 S1.0c 改动方案

#### 改动 1：新增绝对对齐惩罚 `pen_align_abs`

```python
pen_align_abs = -k_align_abs * w_close * clamp(E_align, 0.0, 2.0)
# k_align_abs = 0.05
```

- 提供**持续的**、与对齐误差成正比的惩罚（只在近处生效）
- 保留原有的增量 `r_align = k_align * delta_E_align`（仍提供"改善方向"）

数值验证：
- 歪角 27°（E_align=1.92, w_close=0.69）：pen = -0.066/step
- 良好对齐（E_align=0.47, w_close=0.69）：pen = -0.016/step
- **差值 = 0.050/step**，1349 步 episode 内总差 = **67.5**（占总 reward 的 42%）

#### 改动 2：用 `w_ready` 直接门控 `r_insert`，移除 `pen_wrongins`

```python
# 旧（S1.0/S1.0b）：
r_insert     = k_insert * clamp(progress, min=0)
pen_wrongins = -k_wrongins * (1 - w_ready)**2 * clamp(progress, min=0)

# 新（S1.0c）：
r_insert = k_insert * w_ready * clamp(progress, min=0.0)
# pen_wrongins 完全移除
```

数值验证：
- w_ready=0（最差对齐）：r_insert = **0**（vs 旧方案 +3.59*progress）
- w_ready=0.15（S1.0b 水平）：r_insert = 1.55*progress（vs 旧 5.47*progress，降 72%）
- w_ready=0.7（良好对齐）：r_insert = 7.25*progress

关键：插入奖励从"恒为正"变为"严格正比于对齐质量"。

### 13.4 参数变更汇总

| 参数 | S1.0b | S1.0c | 改动 |
|------|-------|-------|------|
| `k_align_abs` | (无) | **0.05** | 新增绝对对齐惩罚 |
| `k_wrong_far` | 4.0 | **(移除)** | w_ready 门控取代 wrongins |
| `k_wrong_close` | 8.0 | **(移除)** | w_ready 门控取代 wrongins |
| r_insert 公式 | `k * progress` | `k * w_ready * progress` | 直接门控 |
| pen_wrongins | `-(1-w)^2 * progress` | **(移除)** | 被门控取代 |

其他参数（d_far, d_close, k_ins_far, k_pre_far/close, k_forward 等）保持 S1.0b 不变。

### 13.5 预期训练动态

改动后在 dist_front ~1.5m（w_close ≈ 0.69）时的每步收益对比：

**"斜怼"策略**（yaw=27, w_ready=0.08）：
- pen_align_abs = -0.066/step（新，持续惩罚）
- r_insert ≈ +0.002/step（大幅降低）
- r_approach ≈ +0.010/step
- 小计 ≈ **-0.054/step**（净亏损）

**"先对齐再插入"策略**（yaw=8, w_ready=0.7）：
- pen_align_abs = -0.016/step（轻微）
- r_insert ≈ +0.022/step（解锁完整插入奖励）
- r_approach ≈ +0.010/step
- 小计 ≈ **+0.016/step**（净盈利）

**收益差 = 0.070/step**，梯度清晰地指向"必须先对齐"。

### 13.6 日志变更

| 移除 | 新增 |
|------|------|
| `s0/pen_wrongins` | `s0/pen_align_abs` |

### 13.7 S1.0c 训练完成分析（iter 0→1999）

#### 13.7.1 训练完成状态

- **训练迭代**：iter 0 → 1999/2000（完成 99.95%）
- **总时长**：约 2 小时 28 分钟
- **总步数**：131,072,000 timesteps
- **训练速度**：平均 12,000-16,000 steps/s

#### 13.7.2 关键指标演变趋势

**对齐质量指标演变**：

| 迭代 | yaw_deg | E_align | w_ready | lateral | 趋势 |
|------|---------|---------|---------|---------|------|
| 50   | 15.1°   | 1.38    | 0.18    | 0.44m   | 初始 |
| 200  | 21.3°   | 1.65    | 0.15    | 0.47m   | 恶化 |
| 500  | 23.4°   | 1.77    | 0.16    | 0.50m   | 恶化 |
| 700  | **24.8°** | **1.91** | 0.14    | 0.54m   | **最差** |
| 900  | 22.0°   | 1.72    | 0.15    | 0.50m   | 改善 |
| 1200 | 21.0°   | 1.69    | 0.15    | 0.50m   | 改善 |
| 1600 | 24.3°   | 1.79    | 0.15    | 0.49m   | 波动 |
| 1999 | **21.2°** | **1.70** | **0.15** | **0.50m** | **终态** |

**关键发现**：
- yaw_deg：15° → 24.8°（iter 700 最差）→ 21.2°（终态）
- E_align：1.38 → 1.91（iter 700 最差）→ 1.70（终态）
- **对齐质量未改善**：终态 yaw 21.2° 仍远高于目标 3°，且比初始 15° 更差

**距离和插入指标演变**：

| 迭代 | dist_front | insert_norm | frac_inserted | frac_lifted |
|------|-----------|-------------|---------------|-------------|
| 50   | 1.79m     | 0.12        | 0.09          | 0.64        |
| 200  | 1.54m     | 0.50        | 0.20          | 0.64        |
| 500  | 1.52m     | 0.53        | 0.21          | 0.63        |
| 800  | 1.48m     | 0.69        | 0.24          | 0.67        |
| 1200 | 1.61m     | 0.57        | 0.21          | 0.64        |
| 1999 | **1.47m** | **0.69**    | **0.25**      | **0.66**    |

**关键发现**：
- dist_front：从 1.79m 改善到 1.47m（改善 18%）
- insert_norm：从 0.12 提升到 0.69（提升 475%）
- **插入能力显著提升**，但距离目标（插入深度 ≥ 0.67）仍有差距

**奖励信号演变**：

| 迭代 | r_insert | r_lift | pen_align_abs | r_approach | noise_std | reward |
|------|----------|--------|---------------|------------|-----------|--------|
| 50   | 0.004    | 0.006  | -0.035        | 0.010      | 3.21      | -238   |
| 200  | 0.004    | 0.139  | -0.043        | 0.014      | 3.20      | -8     |
| 500  | 0.005    | 0.076  | -0.045        | 0.006      | 2.97      | -38    |
| 1000 | 0.007    | 0.184  | -0.043        | 0.008      | 2.62      | 97     |
| 1999 | **0.006** | **0.297** | **-0.044** | **0.009** | **2.84** | **3** |

**关键发现**：
- r_insert：稳定在 0.004-0.007/step（w_ready 门控生效，vs S1.0b 的 0.037，降 84%）
- r_lift：0.006 → 0.297/step（**大幅上升**，仍是最大正向信号）
- pen_align_abs：稳定在 -0.041 到 -0.046/step（持续惩罚生效）
- noise_std：3.21 → 2.84（下降 12%，策略在收敛）
- **r_lift 仍占主导**：0.297 vs pen_align_abs(-0.044) = 6.75:1

#### 13.7.3 S1.0c vs S1.0b 终态对比

| 指标 | S1.0b iter 460 | S1.0c iter 1999 | 变化 | 评价 |
|------|---------------|----------------|------|------|
| **yaw_deg** | **27.1°** (单调恶化) | **21.2°** (波动) | **改善 5.9°** | ✓ 改善但仍在高位 |
| **E_align** | 1.92 (单调恶化) | 1.70 (波动) | 改善 0.22 | ✓ 改善 |
| **w_ready** | 0.147 (停滞) | 0.153 (波动) | 略改善 | ≈ 无变化 |
| **r_insert** | 0.037 | **0.006** | **降 84%** | ✓ w_ready 门控生效 |
| **r_lift** | 0.131 | **0.297** | **升 127%** | ✗ 问题加剧 |
| **pen_align_abs** | (无) | **-0.044** | 新增 | ✓ 持续惩罚生效 |
| **noise_std** | 3.49 (上升) | **2.84** (下降) | **改善** | ✓ 策略收敛 |
| **dist_front** | 1.47m | 1.47m | 相同 | ≈ 无变化 |
| **insert_norm** | 0.57 | 0.69 | 改善 | ✓ 插入更深 |
| **frac_success** | 0.0000 | 0.0000 | 相同 | ✗ 仍无稳定成功 |

#### 13.7.4 核心问题诊断

**S1.0c 的改进**：
1. ✓ **w_ready 门控成功**：`r_insert` 从 0.037 降到 0.006（-84%），斜怼不再从插入中获得奖励
2. ✓ **pen_align_abs 生效**：持续提供 -0.044/step 的对齐惩罚
3. ✓ **策略收敛**：`noise_std` 从 3.21 降到 2.84，策略在收敛而非发散
4. ✓ **yaw 不再单调恶化**：从 S1.0b 的 27.1° 改善到 21.2°，且不再单调上升

**残留问题**：
1. ✗ **r_lift 绕过对齐门控**：
   - `r_lift` 从 0.131 上升到 0.297/step（+127%）
   - 仍是最大正向信号，是 `pen_align_abs`(-0.044) 的 **6.75 倍**
   - 机器人通过 `r_approach` 物理撞入 → `insert_norm` 被动升高 → `w_lift > 0` → 获得 `r_lift`

2. ✗ **对齐质量未改善**：
   - yaw 终态 21.2° 仍远高于目标 3°
   - E_align 终态 1.70 仍远高于理想 <1.0
   - w_ready 终态 0.15 仍很低（理想 >0.5）

3. ✗ **成功率极低**：
   - 整个训练过程中 `frac_success` 最高仅 0.1%（1024 个环境中约 1 个成功）
   - 成功案例都是偶发，无稳定策略

#### 13.7.5 根本原因分析

**奖励信号量级不对等**：

在 iter 1999 的终态：
- **"斜怼+举升"策略**：
  - r_lift = +0.297/step（最大正向信号）
  - r_insert = +0.006/step（w_ready 门控后很小）
  - r_approach = +0.009/step
  - pen_align_abs = -0.044/step
  - 小计 ≈ **+0.268/step**（净盈利）

- **"先对齐再插入"策略**：
  - r_lift = 0（w_lift=0，因为 insert_norm 低）
  - r_insert = 0（w_ready=0，因为对齐差）
  - r_approach = +0.009/step
  - pen_align_abs = -0.044/step
  - 小计 ≈ **-0.035/step**（净亏损）

**收益差 = 0.303/step**，策略理性地选择了"斜怼+举升"。

**r_lift 绕过机制**：

即使 `r_insert` 被 w_ready 门控，机器人仍可以通过以下路径获得 `r_lift`：

```
r_approach (无条件) → 物理撞入托盘 → insert_norm 被动升高 → w_lift > 0 → r_lift 正向
```

这个路径完全不经过 `r_insert` 的 w_ready 门控，导致对齐门控失效。

#### 13.7.6 结论与下一步建议

**S1.0c 的成就**：
- ✓ 成功抑制了 `r_insert` 的斜怼收益（w_ready 门控生效）
- ✓ 提供了持续的对齐惩罚（pen_align_abs 生效）
- ✓ 策略在收敛（noise_std 下降）
- ✓ yaw 不再单调恶化（从 27.1° 改善到 21.2°）

**S1.0c 的局限**：
- ✗ 对齐质量未改善（yaw 21.2° 仍远高于目标 3°）
- ✗ r_lift 绕过对齐门控，仍是最大正向信号
- ✗ 成功率极低（0.1%），无稳定策略

**下一步建议（S1.0d）**：

**必须对 `r_lift` 也做 w_ready 门控**：

```python
# 当前（S1.0c）：
r_lift = k_lift * w_lift * clamp(delta_lift, min=0.0)

# 建议（S1.0d）：
r_lift = k_lift * w_lift * w_ready * clamp(delta_lift, min=0.0)
```

这样，只有在对齐质量足够好（w_ready > 0）时，举升才能获得奖励，彻底切断"斜怼+举升"的收益路径。

**预期效果**：
- 在 w_ready=0.15（当前水平）时：`r_lift` 从 0.297 降到 0.045/step（降 85%）
- 在 w_ready=0.7（良好对齐）时：`r_lift` 恢复到 0.208/step
- **收益差反转**："先对齐再插入"策略将获得更高收益

---

*文档版本：S0.4 → S0.5 → S0.6 → S0.7 → S1.0 → S1.0b → S1.0c 多阶段训练*  
*更新日期：2026-02-07*
