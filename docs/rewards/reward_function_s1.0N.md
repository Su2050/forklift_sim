# S1.0N 奖励函数（对齐突破 + hold 抗抖 + 探索保温）

> **版本**：S1.0N v2.3（已实现并通过 30 iter smoke test）
>
> **日期**：2026-02-10
>
> **前置版本**：S1.0M（拆除三颗雷 + 对齐信号增强 + 噪声控制）
>
> **目标**：解决 S1.0M 训练中"success_now 到 success 转化率低、对齐精度平台化（0.26m/6.8°）、策略过早确定化（std≈0.03）"的三重瓶颈，让 `frac_success` 从 ~0.2% 突破到 ≥1%。
>
> **重要**：obs 维度 13→15，S1.0M checkpoint 不兼容，**必须从头训练**。

---

## 1. S1.0M 遗留问题诊断（为什么要做 S1.0N）

### 1.1 S1.0M 2000 iter 训练的成就与瓶颈

S1.0M 成功打通了主干链路（success 从零到非零），但三个新瓶颈阻止 success 进一步提升：

| 指标 | S1.0M (2000 iter) | 说明 |
|------|-------------------|------|
| `frac_success_now` | 0.29%~1.46% | **已能"到达"成功位姿（里程碑！）** |
| `frac_success` | 0.10%~0.29% | 比 success_now 低 3~5x，hold 一抖就掉线 |
| `hold_counter_max` | 10（满格） | 偶尔能站住，但 hold_counter_mean 极低 |
| `lateral_mean` | 0.26m | 从 iter ~150 后完全平台化，不再下降 |
| `yaw_deg_mean` | 6.7°~7.0° | 同样平台化 |
| `frac_aligned` | 18%~20% | 约 1/5 环境能到对齐区，但精度不够 |
| `action noise std` | 0.03 | **几乎确定性策略，探索完全枯死** |
| `phi1` | 1.66 | 稳定但无法再驱动对齐下探 |
| `hit_fine_align` | 0.1%~0.3% | 极稀疏，fine/precise 里程碑几乎不触发 |

### 1.2 根因分析（按影响排序）

#### 瓶颈一：Hold 计数器"一抖归零"，success_now 无法转化为 success

S1.0M 的 hold 逻辑是硬二值的：

```python
# env.py 第 981-983 行
self._hold_counter = torch.where(
    success_now, self._hold_counter + 1, torch.zeros_like(self._hold_counter)
)
```

- 只要有一步不满足 `success_now`（inserted + aligned + lifted），计数器直接归零
- 物理仿真中碰撞面微小抖动是常态，`insert_depth`、`y_err`、`lift_height` 在阈值边缘频繁穿越
- 结果：`hold_counter_max=10`（满格）说明偶尔能连续站住，但大多数"到达"都因一帧抖动归零
- 数据验证：`frac_success_now ≈ 0.3~1.5%`，`frac_success ≈ 0.1~0.3%`，转化率仅 ~20%
- **注意**：抖动不只发生在 y/yaw 维度，insert_depth 微弹和 lift_height 微降同样会触发归零

#### 瓶颈二：对齐精度平台化，缺少"从 0.26m 往 0.15m 下探"的驱动力

- S1.0M 已有 fine_align（y<0.10, yaw<5°）和 precise_align（y<0.05, yaw<3°）里程碑
- 但对当前策略水平，这两个里程碑太稀疏（`hit_fine_align` 仅 0.1~0.3%）
- 在 0.15m~0.30m 区间内没有持续性的"往中心吸"的信号
- 策略学到"大概对齐就够了"，停在 0.26m 的盆地不再下探

#### 瓶颈三：策略探索塌缩（action std 0.50→0.03）

- S1.0M 用 `init_noise_std=0.5` + `entropy_coef=0.0005` 达成了"能精控"的目标
- 但 std 从 0.50 一路单调下降到 0.03（iter ~1500 后稳定），等价于确定性策略
- steer action 有效探索范围 ≈ ±0.03 rad（约 ±1.7°），不足以发现"向中线转向"的更优策略
- 对齐这种需要微调搜索的任务很容易陷入局部最优

#### 补充瓶颈：观测空间缺少直接对齐信息

- 当前 13 维观测中 `d_xy_r` 是 **robot frame** 下的相对位置
- reward 中的 `y_err` 是 **pallet center line frame** 下的横向误差
- Agent 要推算 `y_err` 需做坐标变换：`y_err = d_xy_r * rotate(dyaw)`
- 对 3 层 MLP 是非平凡的几何运算，agent 学会了"往前开"但没学会"往中线修正"

---

## 2. S1.0N 改动清单

### 2.1 P0-A [核心改动]：Hold Counter 全维度 Hysteresis（Schmitt trigger 抗抖）

**问题**：hold 逻辑硬二值，一步不满足即归零，物理抖动下转化率极低。**且原 v2 方案只保护了 y/yaw，insert/lift 微弹仍会归零。**

**方案**：全维度 Schmitt trigger（对齐、插入、举升都做 entry/exit）+ grace zone hold constant

**新增参数**（env_cfg.py）：

```python
hysteresis_ratio: float = 1.2       # 对齐 exit = entry × 1.2
insert_exit_epsilon: float = 0.02   # 插入深度 exit 容差 (norm)
lift_exit_epsilon: float = 0.01     # 举升高度 exit 容差 (m)
```

计算得到：
- **对齐** Entry: `y_err ≤ 0.15m` 且 `yaw_err ≤ 8°` — Exit: `y_err > 0.18m` 或 `yaw_err > 9.6°`
- **插入** Entry: `insert_depth ≥ insert_thresh` — Exit: `insert_depth < insert_thresh - 0.02`
- **举升** Entry: `lift_height ≥ lift_delta_m` — Exit: `lift_height < lift_delta_m - 0.01`

**修改逻辑**（env.py `_get_rewards`）：

```python
# --- 全维度 Schmitt trigger hold counter ---

# 对齐 entry/exit
align_entry = (y_err <= self.cfg.max_lateral_err_m) & (yaw_err_deg <= self.cfg.max_yaw_err_deg)
exit_y = self.cfg.max_lateral_err_m * self.cfg.hysteresis_ratio      # 0.18m
exit_yaw = self.cfg.max_yaw_err_deg * self.cfg.hysteresis_ratio      # 9.6°
align_exit_exceeded = (y_err > exit_y) | (yaw_err_deg > exit_yaw)

# 插入 entry/exit（带容差防微弹）
insert_entry = insert_depth >= self._insert_thresh
insert_exit_exceeded = insert_depth < (self._insert_thresh - self.cfg.insert_exit_epsilon)

# 举升 entry/exit（带容差防微降）
lift_entry = lift_height >= self.cfg.lift_delta_m
lift_exit_exceeded = lift_height < (self.cfg.lift_delta_m - self.cfg.lift_exit_epsilon)

# 三段式更新
still_ok = insert_entry & align_entry & lift_entry
any_exit_exceeded = align_exit_exceeded | insert_exit_exceeded | lift_exit_exceeded
grace_zone = (~still_ok) & (~any_exit_exceeded)

self._hold_counter = torch.where(
    still_ok,
    self._hold_counter + 1,
    torch.where(
        grace_zone,
        self._hold_counter,              # hold constant: 不加不减，保持记忆
        torch.zeros_like(self._hold_counter),  # 真正跑飞：归零
    ),
)
success = self._hold_counter >= self._hold_steps
```

**grace zone 为什么选 hold constant 而不是 -1**：

- `-1/step` 的隐含假设是 agent 大部分时间在 entry 内（+1），偶尔掉到 grace（-1），净值能上升
- 但如果阈值边缘频繁抖动（比如每两步出一次 entry），`+1/-1` 的净增长接近 0，仍难攒满 hold_steps
- hold constant 更宽容：只要没有"真的跑飞"，已积累的计数不会丢失

**新增日志**：

```python
"phase/hold_counter_mean": self._hold_counter.float().mean().item(),
"phase/grace_zone_frac": grace_zone.float().mean().item(),
```

> `grace_zone_frac` 应 < 0.3。如果过高说明大部分时间在阈值边缘徘徊，可能需要进一步放宽 exit 阈值。

**预期效果**：`frac_success` 明显追上 `frac_success_now`，转化率从 ~20% 提升到 ≥50%。

---

### 2.2 P0-B1：新增 gate_align 里程碑（0.15m/8°，绑定 approach）

**问题**：fine_align（0.10m/5°）对当前策略水平太稀疏，"到达对齐区"没有一次性奖励信号。

**方案**：在 entry 条件本身加一个里程碑，让 18% 的 `frac_aligned` 都能在首次触达时获得明确奖励。

**新增参数**（env_cfg.py）：

```python
rew_milestone_gate_align: float = 2.5   # y < 0.15m & yaw < 8°
```

**修改**（env.py）：

- `_milestone_flags` 从 `(N, 6)` 扩展到 `(N, 7)`
- 新增条件（**绑定 approach flag 防早触发**）：

```python
milestone_gate_align = (
    (y_err <= 0.15) & (yaw_err_deg <= 8.0) &
    self._milestone_flags[:, 0]    # 必须先触发过 approach（dist_front ≤ 0.25m）
)
```

**为什么要绑定 approach**：
- 部分初始位姿可能恰好满足 y/yaw 条件（初始 y 范围 [-0.6, 0.6]，有些样本 |y| < 0.15）
- 如果不绑定，agent 可能学到"先吃 gate_align 再说"的懒惰策略，不推进到插入/举升
- 绑定 approach（dist_front ≤ 0.25m）确保 gate_align 只在"已接近托盘"时才触发

**里程碑完整链（S1.0N）**：

| 序号 | 名称 | 条件 | 奖励 | 状态 |
|------|------|------|------|------|
| 0 | approach | dist_front ≤ 0.25m | 1.0 | 沿用 |
| 1 | coarse_align | y≤0.20 & yaw≤10° | 2.0 | 沿用 |
| 2 | **gate_align** | **y≤0.15 & yaw≤8° & approach已触发** | **2.5** | **新增** |
| 3 | insert10 | insert_norm ≥ 0.10 | 5.0 | 沿用 |
| 4 | insert30 | insert_norm ≥ 0.30 | 10.0 | 沿用 |
| 5 | fine_align | y≤0.10 & yaw≤5° | 5.0 | 沿用 |
| 6 | precise_align | y≤0.05 & yaw≤3° | 8.0 | 沿用 |

---

### 2.3 P0-B2：Hold-Align 奖励（Delta/Potential-Based Shaping，防刷分）

**问题**：在对齐区（0.15m~0.30m）内没有持续性的"往中心吸"的信号，策略停在 0.26m 盆地。

**v2 方案的风险**（per-step 绝对正奖）：

1. **挂机刷分**：agent 可能学到"插到 0.35，挪到 exit 区内，然后不冒险做最后的 lift/hold 成功，把 r_hold_align 当工资领"。per-step 正奖在长 episode 中可以堆积。
2. **开启太晚**：原方案门控在 `insert_norm >= 0.35` 才开，但对齐平台化问题发生在更早阶段（还没深插时），触发频率不够打不动平台。

**v2.1 方案：Delta/Potential-Based Shaping**

定义对齐势函数（零到一之间，越对齐越大）：

```python
phi_align = torch.exp(-(y_err / self.cfg.hold_align_sigma_y) ** 2) \
          * torch.exp(-(yaw_err_deg / self.cfg.hold_align_sigma_yaw) ** 2)
```

每 step 给"进步奖励"：

```python
r_hold_align = self.cfg.k_hold_align * (phi_align - self._prev_phi_align)
self._prev_phi_align = phi_align.detach()   # detach 防计算图/显存累积
```

**新增状态量**（env.py `__init__`）：

```python
self._prev_phi_align = torch.zeros(self.num_envs, device=self.device)
```

**重置**（env.py `_reset_idx`）：

```python
# 初始化为当前 state 的 phi_align，让 reset 后第一步 delta=0（防"开局白嫖"）
phi_align_init = self._compute_phi_align(env_ids)
self._prev_phi_align[env_ids] = phi_align_init.detach()
```

> **为什么不初始化为 0**：如果 reset 后初始位姿恰好 phi_align 较大（y/yaw 接近 0），第一步会白拿 `k * phi_align` 的正奖。初始化为当前值让第一步 delta=0，shaping 严格只奖"进步"。需要在 env.py 中抽取一个 `_compute_phi_align(env_ids)` 辅助方法（用同源 y_err/yaw_err 计算）。

**新增参数**（env_cfg.py）：

```python
k_hold_align: float = 0.1          # delta shaping 权重
hold_align_sigma_y: float = 0.15   # 横向尺度 (m)
hold_align_sigma_yaw: float = 8.0  # 偏航尺度 (deg)
```

**加入总奖励**：

```python
rew = r_pot + pen_premature + pen_dense + r_terminal + milestone_reward + r_hold_align + early_stop_penalty
```

**为什么 delta/potential-based 更优**：

- **防刷分**：原地不动 → phi_align 不变 → delta=0 → 无奖励。只奖"变好"，不奖"原地不动"
- **可更早开启**：不依赖 insert_gate_norm，approach 后对齐改善即有信号
- **近似 PBRS**：使用 γ=1 简化（严格 PBRS 为 γΦ(s')-Φ(s)，γ=0.99 vs 1.0 在 k=0.1 下差异 ~0.001/step，可忽略）
- **天然防"挂机领年金"**：势函数有上限（≤1.0），delta 收益有限

**已知简化与观察项**：
- delta 可产生负值（对齐变差时扣分）。如果训练中出现策略"畏缩"（接近对齐区反而不敢微调），可加安全阀：`r_hold_align = k * torch.clamp(phi - prev, min=0.0)`（只奖进步不罚退步）。当前先保留正负，观察 `s0/r_hold_align` 的分布再决定。

**新增日志**：

```python
"s0/r_hold_align": r_hold_align.mean().item(),
```

---

### 2.4 P0-C：Action Std 保护（log_std 下限 + entropy 保温）

**问题**：action std 从 0.50 一路降到 0.03，策略冻死在局部最优。

**方案**：双保险——硬下限 + 软保温。

#### 2.4.1 硬下限：自定义 ActorCritic 子类 clamp log_std

创建新文件 `forklift_pallet_insert_lift/clamped_actor_critic.py`（直接在包目录下，非子目录）：

```python
"""带 log_std 下限保护的 ActorCritic 子类。"""
import math
import torch
from rsl_rl.modules import ActorCritic

LOG_STD_MIN = math.log(0.05)   # std_min = 0.05, 保守起步
LOG_STD_MAX = math.log(1.5)    # std_max = 1.5, 防爆

class ClampedActorCritic(ActorCritic):
    """ActorCritic with clamped log_std to prevent std collapse."""

    def _update_distribution(self, obs: torch.Tensor) -> None:
        mean = self.actor(obs)
        # 关键：clamp log_std 防止塌缩
        clamped_log_std = torch.clamp(self.log_std, LOG_STD_MIN, LOG_STD_MAX)
        self.distribution = torch.distributions.Normal(mean, clamped_log_std.exp())
```

**注册方式**（`__init__.py` 中 monkey-patch 到 `rsl_rl.modules`）：

```python
from .clamped_actor_critic import ClampedActorCritic as _ClampedActorCritic
import rsl_rl.modules as _rsl_modules
_rsl_modules.ClampedActorCritic = _ClampedActorCritic
```

然后在 `rsl_rl_ppo_cfg.py` 中使用 `class_name="rsl_rl.modules.ClampedActorCritic"`。这样 rsl_rl runner 的 `eval(class_name)` 能正确解析，因为 `import rsl_rl` 已在 runner scope 中。

**实现注意事项**：

1. **log_prob / entropy 一致性**：`self.distribution` 被后续 `log_prob()` 和 `entropy()` 调用共享，因此在 `_update_distribution` 中 clamp 即可保证所有下游使用一致的 clamped std，无需额外处理。
2. **梯度行为**：`torch.clamp` 在越界区间梯度为 0 — log_std 参数可能永远停在 min 以下，但分布实际用的是 min 值。这对防塌缩已足够（目的是兜底，不是推升）。
3. **方法签名**：rsl_rl `ActorCritic._update_distribution(self, obs)` 参数名为 `obs`（非 `observations`），已验证兼容。
4. **smoke test 已验证**：30 iter 训练中 `ClampedActorCritic` 正确加载，`Mean action noise std` 从 0.50 开始，无报错。

#### 2.4.2 软保温：entropy_coef 温和上调

```python
# rsl_rl_ppo_cfg.py
entropy_coef = 0.0015   # S1.0M: 0.0005 → S1.0N: 0.0015（3x，温和保温）
```

**为什么不是 10x（0.005）**：S1.0M 文档设计时已考虑了 entropy/std 的平衡，10x 跳跃可能破坏已有精控能力。3x 配合硬下限已足够防止塌缩。

#### 2.4.3 init_noise_std 保持不变

```python
init_noise_std = 0.5   # 保持 S1.0M 值，有 std_min 兜底后不需要降起点
```

#### 2.4.4 升级条件（训练中观察触发）

若 200 iter 后 `lateral_mean` 仍无下降趋势且 action std 稳定在 0.05 附近（下限处）：

- 将 `std_min` 从 0.05 提升到 0.08（`LOG_STD_MIN = ln(0.08) ≈ -2.53`）
- 或将 `entropy_coef` 从 0.0015 上调到 0.002

**预期效果**：action std 稳定在 0.05~0.15 区间，既不冻住也不乱晃。

---

### 2.5 P0-D：观测空间扩充（13 → 15 维）

**问题**：当前 obs 中 `d_xy_r` 是 robot frame，agent 需要做坐标变换才能推算 pallet center line frame 的 y_err。

**方案**：直接暴露带符号的 `y_err_obs` 和 `yaw_err_obs`（与 `_get_rewards` 同源计算）。

**修改**（env_cfg.py）：

```python
observation_space = 15   # 原 13
```

**修改**（env.py `_get_observations`）：

```python
# 在 pallet center line frame 下计算（与 _get_rewards 同源）
cp = torch.cos(pallet_yaw); sp = torch.sin(pallet_yaw)
v_lat = torch.stack([-sp, cp], dim=-1)
y_signed = torch.sum((root_pos[:, :2] - pallet_pos[:, :2]) * v_lat, dim=-1)
y_err_obs = torch.clamp(y_signed / 0.5, -1.0, 1.0)   # 硬 clip 防尾部大值

dyaw_signed = torch.atan2(torch.sin(yaw - pallet_yaw), torch.cos(yaw - pallet_yaw))
yaw_err_obs = torch.clamp(dyaw_signed / (15.0 * math.pi / 180.0), -1.0, 1.0)  # 硬 clip

obs = torch.cat([
    ...,
    y_err_obs.unsqueeze(-1),      # +1 维
    yaw_err_obs.unsqueeze(-1),    # +1 维
], dim=-1)
```

**关键点**：

1. **带符号**：`y_err_obs` 正=右偏，负=左偏，直接告诉 agent "你偏了多少、往哪偏"
2. **硬 clip [-1, 1]**：防止初始化分布较宽时尾部大值导致网络前期梯度异常
3. **符号方向验证**（smoke test 必检项）：确认 `y_err_obs > 0` 对应"车在中线右侧"、`yaw_err_obs > 0` 对应"车头偏右"。坐标系定义中 `v_lat = [-sp, cp]` 方向需与物理直觉一致。如果反了，训练不会崩但会更慢更绕，需调整符号。

**S1.0M Checkpoint 不兼容**：obs 维度 13→15，actor 第一层权重尺寸不匹配。**S1.0N 必须从头训练**，不能续训 S1.0M checkpoint。如果未来需要热启动，需要写兼容层（旧网络补零或重训第一层），但 S1.0N 首次训练不需要。

---

## 3. 参数速查表（S1.0M → S1.0N 差异）

| 参数 | S1.0M | S1.0N | 说明 |
|------|------:|------:|------|
| `hysteresis_ratio` | — | **1.2** | 新增：对齐 exit = entry × 1.2 |
| `insert_exit_epsilon` | — | **0.02** | 新增：插入深度 exit 容差 (norm) |
| `lift_exit_epsilon` | — | **0.01** | 新增：举升高度 exit 容差 (m) |
| `rew_milestone_gate_align` | — | **2.5** | 新增：gate_align 里程碑（绑定 approach） |
| `k_hold_align` | — | **0.1** | 新增：delta hold-align shaping 权重 |
| `hold_align_sigma_y` | — | **0.15** | 新增：hold-align 横向尺度 (m) |
| `hold_align_sigma_yaw` | — | **8.0** | 新增：hold-align 偏航尺度 (deg) |
| `observation_space` | 13 | **15** | 新增 y_err_obs / yaw_err_obs |
| `entropy_coef` | 0.0005 | **0.0015** | 3x 保温 |
| `init_noise_std` | 0.5 | 0.5 | 不变 |
| `log_std_min`（代码层） | — | **-3.0** | ln(0.05)，自定义 ActorCritic |
| `log_std_max`（代码层） | — | **0.405** | ln(1.5)，自定义 ActorCritic |
| `_milestone_flags` | (N,6) | **(N,7)** | 新增 gate_align |
| `_prev_phi_align` | — | **(N,)** | 新增：delta shaping 状态量 |
| `hold_counter` 逻辑 | 二值归零 | **全维度 Schmitt + grace hold** | 对齐/插入/举升 |

其余参数沿用 S1.0M 不变：

- 成功阈值：`max_lateral_err_m=0.15`、`max_yaw_err_deg=8.0`
- hold 时长：`hold_time_s=0.33`（≈10 steps）
- 对齐尺度：`y_scale1=0.15`、`yaw_scale1=10`、`y_scale2=0.08`、`yaw_scale2=5`
- 门控：`y_gate2=0.40`、`yaw_gate2=20`
- fine_align / precise_align 里程碑（5.0 / 8.0）
- 举升门控：`insert_gate_norm=0.35`、`insert_ramp_norm=0.08`
- PPO：`gamma=0.99`、`learning_rate=3e-4`、`num_steps_per_env=64`
- 网络：`[256, 256, 128]`、`activation=elu`

---

## 4. 新增日志项

| 日志 key | 含义 |
|----------|------|
| `phase/hold_counter_mean` | hold 计数器的均值（追踪 hysteresis 效果） |
| `phase/grace_zone_frac` | 在 grace zone 中的环境比例（应 < 0.3，过高说明阈值边缘拥挤） |
| `milestone/hit_gate_align` | 首次触发 gate_align（y<0.15m & yaw<8° & approach 已触发）的比例 |
| `s0/r_hold_align` | delta hold-align 奖励的均值 |
| `Mean action noise std`（rsl_rl 自带） | 模型分布 std 的均值。由于 ClampedActorCritic 设置 `self.distribution` 使用 clamped std，此值即反映 clamp 后的真实分布 std。若 ≥0.05 说明 std_min 正在生效。 |

---

## 5. 涉及文件

| 文件 | 改动 |
|------|------|
| `env_cfg.py` | `hysteresis_ratio`、`insert_exit_epsilon`、`lift_exit_epsilon`、`rew_milestone_gate_align`、`k_hold_align`、`hold_align_sigma_y`、`hold_align_sigma_yaw`、`observation_space` 13→15 |
| `env.py` | 全维度 hold hysteresis 逻辑、gate_align 里程碑（`_milestone_flags` 6→7，绑定 approach）、delta hold-align shaping（`_prev_phi_align` 状态量 + detach + reset 初始化为当前 phi）、`_compute_phi_align` 辅助方法、`_get_observations` 扩充（带 clip）、新增日志 |
| `rsl_rl_ppo_cfg.py` | `entropy_coef` 0.0005→0.0015、`class_name="rsl_rl.modules.ClampedActorCritic"` |
| **新建** `clamped_actor_critic.py` | 自定义 ActorCritic 子类，clamp log_std ≥ -3.0 |
| `__init__.py` | 导入 ClampedActorCritic 并 monkey-patch 到 `rsl_rl.modules` |

---

## 6. 验收标准（500 iter）

| 指标 | S1.0M (2000 iter) | S1.0N 目标 (500 iter) | 一票否决 |
|------|:------------------:|:---------------------:|----------|
| `frac_success` | 0.2% | **≥1%** | — |
| `frac_success / frac_success_now` | ~20% | **≥50%**（hysteresis 生效标志） | — |
| `lateral_mean` | 0.26m | **<0.20m**，冲 <0.15m | — |
| `yaw_deg_mean` | 6.8° | **<6°**，冲 <5° | — |
| `action std` | 0.03 | **稳定在 0.05~0.15** | 若 <0.04 则 std_min 失效 |
| `frac_aligned` | 18% | **≥25%** | — |
| `hold_counter_mean` | — | **>0 且上升趋势** | — |
| `grace_zone_frac` | — | **<0.3**（过高需放宽 exit） | — |
| `hit_gate_align` | — | **>10%** | — |

---

## 7. 降级/未做项（及理由）

### 7.1 k_phi1/k_phi2 补偿 → P2 可选

S1.0M 的 scale 收紧让 phi1 从 2.0 降到 1.66（-17%）。原 S1.0N 计划建议 k_phi1 6→8 补偿。

**降级理由**：S1.0N 新增了 gate_align 里程碑（+2.5）和 delta hold-align shaping 两个额外信号源。在这些新信号的加持下，phi 补偿可能导致对齐方向梯度叠加过强。

**机械化触发条件**：连续 20 个 iter 的 `phi1` 滑动均值 < 1.2 → 启用 k_phi1 6→8、k_phi2 10→14。不要"看着曲线发呆"才做决定。

### 7.2 y 随机化收紧 → 移除

原计划建议将初始 y 从 [-0.6, 0.6] 收紧到 [-0.4, 0.4]。

**移除理由**：牺牲泛化性，且设计稿未提及。S1.0N 的其他改动（obs 扩充 + delta hold-align）应能在不收紧起始分布的情况下提升对齐精度。

---

## 8. 课程收紧路线（后续版本，不在 S1.0N 首次训练中实施）

采用**指标门槛触发**而非固定 iter。**空间和时间分步收紧**（不同时加难度）：

| 阶段 | 触发条件 | max_lateral_err_m | max_yaw_err_deg | hold_time_s |
|------|----------|:-:|:-:|:-:|
| S1.0N（当前） | — | 0.15 | 8.0 | 0.33 |
| 收紧-1a（空间） | `frac_success` 滑动均值 > 1% 稳定 | **0.08** | **5.0** | 0.33 |
| 收紧-1b（时间） | 收紧-1a 后 `frac_success` 恢复到 > 1% | 0.08 | 5.0 | **0.67** |
| 收紧-2a（空间） | `frac_success` > 5% | **0.03** | **3.0** | 0.67 |
| 收紧-2b（时间） | 收紧-2a 后 `frac_success` 恢复到 > 5% | 0.03 | 3.0 | **1.0** |

**为什么分步**：一次收紧两个维度（空间更苛刻 + 时间更苛刻）很可能把刚站稳的策略又踹回去。先收紧空间、等策略适应后再收紧时间，让每一步难度增量可控。

---

## 9. 预期观测到的训练行为

按设计稿的"物理直觉"，S1.0N 训练中最可能出现的变化顺序：

1. **最先**：`frac_success` 明显抬升（追上 `frac_success_now`）—— 全维度 hysteresis 生效
2. **其次**：`action std` 回到 0.05~0.15 区间 —— std_min + entropy 生效
3. **然后**：`lateral_mean` 开始从 0.26m 往下掉 —— delta hold-align + obs 扩充联合生效
4. **最后**：`yaw_deg_mean` 跟随下降 —— 对齐精度整体提升

> 如果顺序反了（例如 std 先暴涨到 >0.3），说明 entropy_coef 过高或 std_min 设置不当，需要回调。

---

## 10. Smoke Test 检查清单（30 iter）

| 检查项 | 预期 | 失败处理 |
|--------|------|----------|
| obs shape | 15 维，无 runtime error | 检查 observation_space 和 cat 维度 |
| 自定义模型加载 | `ClampedActorCritic` 通过 `rsl_rl.modules.ClampedActorCritic` 正确 resolve | 检查 `__init__.py` monkey-patch 和 `class_name` 路径 |
| action std | ≥ 0.05（不会 <0.05 因为有 clamp） | 检查 _update_distribution 是否被调用 |
| Mean action noise std | ≥ 0.05（rsl_rl 自带日志，反映 clamped 后的分布 std） | 若持续 <0.05 说明 clamp 未生效 |
| hold_counter_mean | >0（非全零） | 检查 hysteresis 逻辑是否生效 |
| grace_zone_frac | 0~1 之间合理值 | 检查 grace_zone 计算 |
| hit_gate_align | >0（有触发） | 检查 approach flag 绑定逻辑 |
| r_hold_align | 有正有负（delta 特征），第一步约 0 | 检查 _prev_phi_align 初始化（应为当前 phi，非 0）|
| y_err_obs 符号 | 车在右侧时 > 0 | 验证 v_lat 方向是否正确 |
| insert_depth 单位 | 确认 insert_depth 与 insert_exit_epsilon 单位一致 | 打印典型值范围（0~1 norm 还是 0~0.6m），确认 epsilon=0.02 合理 |
| 无 NaN/Inf | reward 和 obs 均无异常值 | 检查除零保护 |

---

**文件结束。**
