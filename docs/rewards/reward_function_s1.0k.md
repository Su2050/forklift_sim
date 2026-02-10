# S1.0k 奖励函数（严格中心线定义 + 三阶段势函数）

> **版本**：S1.0k（基于 S1.0j 变量体系，重构为“严格定义 + 三阶段势函数”）
>
> **S1.0L 更新**（2026-02-10）：
> - Stage1/2 距离参考点改为 `base`（`dist_front_base`），插入深度仍使用 `tip`
> - `gamma=1.0`（纯差分 shaping）
> - `ins_start=0.10, ins_ramp=0.15`（推迟 Stage3 接管）
> - 默认不再使用 `(1-w3)` 压制 `Phi1/Phi2`（保留回滚开关）
> - `Phi_ins` baseline 调整为 `(0.4 + 0.6*w_align3)`
> - 新增里程碑一次性奖励 + 失败早停（跑飞/摆烂）
>
> **目标**：把“对齐 → 微调插入 → 居深/举升”明确成 **可计算的几何量**，并用 **潜势函数差分 shaping**（防止原地抖动/刷分）。
>
> **说明**：本文件写成“可直接落地到 `env.py::_get_rewards()` 的实现草案”，变量命名尽量沿用 S1.0j 的习惯（`tip / dist_front / insert_norm / lift_height` 等）。

---

## 0. 这版 S1.0k 解决的核心问题（对照 S1.0j）

S1.0j 的阶段 1 对齐误差 `E_align = y_err/lat_ready_m + yaw_err/yaw_ready_deg` 里，`lat_ready_m/yaw_ready_deg` 会随距离放宽/收紧（距离自适应）。这会引入一个“站远点更容易对齐”的漏洞（你训练里很像卡在这）。本版本做两件事：

1. **把“对齐误差”从“距离自适应阈值分母”里解耦**：  
   - 对齐误差以“托盘插入中心线”为基准，使用 *固定尺度/固定容差*（或只用于 gating，不用于归一化分母）。
2. **Stage1 显式引入“距离带 [2,3]m”目标**（你的想法），用“到区间的距离”定义：  
   - 远了 → 前进  
   - 太近且侧偏大 → 倒车“退到可操作区间”再修正  
   这两种行为不需要写 if/else，会自然从几何目标里涌现。

---

## 1) 严格几何定义（核心）

> 只用地面平面（x-y）就够了。z 留给举升。

### 1.1 托盘插入方向与中心线

设托盘在世界系的 yaw 为 `pallet_yaw`（弧度），定义：

- 插入方向（指向托盘内部）单位向量  
  \[
  \mathbf{u}_{in} = (\cos\psi_p,\ \sin\psi_p)
  \]
- 托盘外侧方向（从托盘口朝外）  
  \[
  \mathbf{u}_{out} = -\mathbf{u}_{in}
  \]
- 横向（中心线的法向）单位向量  
  \[
  \mathbf{v}_{lat} = (-\sin\psi_p,\ \cos\psi_p)
  \]

托盘“中心线”就是：过托盘参考点 `pallet_pos_xy`，方向为 `u_in` 的直线。

> 注：参考点取托盘中心/口中心都可以；因为横向偏移是点到直线的垂向距离，沿 u_in 平移不会改变横向投影。

### 1.2 叉车中心线参考点

建议仍沿用你现有的 `robot_pos_xy`（底盘参考点）来算横向偏移；插入深度仍沿用 `tip` 来算（因为几何上叉尖最关键）。

### 1.3 三个误差（必须 log 的三兄弟）

令 `r = robot_pos_xy - pallet_pos_xy`：

- 横向偏移（中心线重叠度）  
  \[
  y = r \cdot \mathbf{v}_{lat},\quad y_{err}=|y|
  \]
- 航向误差（中心线平行度）  
  \[
  yaw\_err = wrap(\psi_{robot} - \psi_p)
  \]
  用 `abs(yaw_err)`（转成 deg 方便调参）。
- 前向距离（到托盘口的沿插入轴距离）  
  采用“在托盘插入轴上的标量坐标”定义 `dist_front/insert_depth`：

  令 `tip_rel = tip_xy - pallet_pos_xy`，  
  \[
  s_{tip} = tip\_rel \cdot \mathbf{u}_{in}
  \]
  托盘口平面在托盘轴上的坐标（以托盘中心为 0）：
  \[
  s_{front} = -\frac{pallet\_depth}{2}
  \]
  那么：
  \[
  dist\_front = \max(s_{front}-s_{tip},\ 0)
  \]
  \[
  insert\_depth = \max(s_{tip}-s_{front},\ 0)
  \]
  \[
  insert\_norm = clamp(insert\_depth/pallet\_depth,\ 0,\ 1)
  \]

---

## 2) 三阶段定义（严格、可验收）

### Stage 1：对齐（粗对齐 + 距离带）
目标：**叉车中心线与托盘中心线尽量重叠且平行**，并把 `dist_front` 维持在可操作距离带 `[2,3]m`：

- 距离带：`[d1_min, d1_max] = [2.0, 3.0]`
- 到区间的距离（0 表示已在带内）：
  \[
  e_{band} = dist\_to\_interval(dist\_front, [d1_{min}, d1_{max}])
  \]

为什么用区间：  
- `dist_front > 3m`：前进会减少 `e_band`  
- `dist_front < 2m`：倒车会减少 `e_band`  
- 在带内 `e_band=0`：Stage1 不再推动“继续靠近”，把主导权交给 Stage2

### Stage 2：微调插入（从 2~3m 推到口前 0m）
目标：在 Stage1 基础上，**把 `dist_front → 0`**，同时收紧对齐（更小 y、更小 yaw）。

### Stage 3：居深（插入深度 → 1）+ 举升
目标：`insert_norm → 1`；满足深度门槛后允许举升 `lift_height`。

---

## 3) 势函数 shaping 设计（核心公式）

> 关键原则：**潜势 Φ 必须非负**（≥0）。  
> 否则“原地不动”会得到 `gamma*Φ - Φ` 的反号漏洞。

统一形态：

- 势函数差分 shaping  
  \[
  r_{pot} = \gamma \Phi(s_t) - \Phi(s_{t-1})
  \]
  建议 `gamma=0.99`（沿用你 S1.0j）。

### 3.1 Stage 1 势函数 Φ1

先定义 Stage1 误差（无距离自适应分母）：

\[
E_1 = \frac{e_{band}}{e_{band\_scale}}
     + \frac{y_{err}}{y_{scale1}}
     + \frac{|yaw\_err_{deg}|}{yaw_{scale1}}
\]

推荐初始尺度（可调）：
- `e_band_scale = 0.5 m`
- `y_scale1 = 0.25 m`
- `yaw_scale1 = 15 deg`

把误差变成非负势函数（越大越好）：
\[
\Phi_1 = k_{phi1} \cdot \frac{1}{1+E_1}
\]

推荐：`k_phi1 = 6.0`

### 3.2 Stage 2 势函数 Φ2（带软门控）

误差：
\[
E_2 = \frac{dist_{front}}{d2_{scale}}
     + \frac{y_{err}}{y_{scale2}}
     + \frac{|yaw\_err_{deg}|}{yaw_{scale2}}
\]

推荐初始尺度：
- `d2_scale = 1.0 m`
- `y_scale2 = 0.12 m`
- `yaw_scale2 = 8 deg`
- `k_phi2 = 10.0`

为了严格阶段化：Stage2 只有在“已进入距离带、且大致对齐”时才显著生效。用两个软门控：

1) 距离带权重（从 3m 到 2m 平滑打开）
\[
w_{band} = smoothstep\left(\frac{d1_{max}-dist_{front}}{d1_{max}-d1_{min}}\right)
\]
它在 `dist_front>=3` 时为 0，在 `dist_front<=2` 时为 1，在 2.5m 时约 0.5。

2) 粗对齐权重（不硬阈值，用指数门控避免梯度断崖）  
\[
w_{align2} = \exp\left(-\left(\frac{y_{err}}{y_{gate2}}\right)^2 - \left(\frac{|yaw\_err_{deg}|}{yaw_{gate2}}\right)^2\right)
\]
推荐：`y_gate2=0.25m, yaw_gate2=15deg`

Stage2 势函数：
\[
\Phi_2 = k_{phi2} \cdot \frac{1}{1+E_2} \cdot w_{band} \cdot w_{align2}
\]

### 3.3 Stage 3 插入势函数 Φins

插入势函数最好直接用 `insert_norm`（你 S1.0j 就这么做，对的）：

- 严对齐门控（更严格）
\[
w_{align3} = \exp\left(-\left(\frac{y_{err}}{y_{gate3}}\right)^2 - \left(\frac{|yaw\_err_{deg}|}{yaw_{gate3}}\right)^2\right)
\]
推荐：`y_gate3=0.10m, yaw_gate3=8deg`

- 插入势函数（保留一个 baseline，避免门控过早把信号压死）：
\[
\Phi_{ins} = k_{ins} \cdot (0.2 + 0.8\cdot w_{align3}) \cdot insert_{norm}
\]
推荐：`k_ins = 18.0`

- Stage3 权重（插入开始后平滑接管）
\[
w_3 = smoothstep\left(\frac{insert_{norm}-ins_{start}}{ins_{ramp}}\right)
\]
推荐：`ins_start=0.02, ins_ramp=0.05`（2% 开始，7% 左右就接近满权重）

最终：
\[
\Phi_{ins} \leftarrow w_3 \cdot \Phi_{ins}
\]

### 3.4 举升势函数 Φlift 与空举惩罚

沿用 S1.0j 的思路（正确）：**只有插入足够深才鼓励举升**，否则惩罚“空举”。

- 举升门控（插入 >= 60% 开始开启）
\[
w_{lift\_base} = smoothstep\left(\frac{insert_{norm}-0.60}{0.10}\right)
\]
\[
w_{lift} = w_{lift\_base} \cdot w_{align3}
\]

- 举升势函数（非负）：
\[
\Phi_{lift} = k_{lift} \cdot w_{lift} \cdot lift_{height}
\]
推荐：`k_lift=20.0`

- 空举惩罚（当插入不够深时举升增量为正就罚）：
\[
pen_{premature} = -k_{pre} \cdot (1-w_{lift\_base}) \cdot clamp(\Delta lift, min=0)
\]
推荐：`k_pre=10.0`

---

## 4) 总奖励结构（建议）

每步总奖励：

\[
r = r_{pot} + pen_{premature} + pen_{dense} + r_{terminal}
\]

- `r_pot = gamma*Phi_total(t) - Phi_total(t-1)`
- `Phi_total = Phi1*(1-w3) + Phi2*(1-w3) + Phi_ins + Phi_lift`
  - 注意：Phi2 自己已经含 `w_band*w_align2`，Phi1 仍在远处提供对齐/回到距离带的驱动力
- `pen_dense`：常驻惩罚（时间、动作 L2、连续距离惩罚）
- `r_terminal`：成功奖励、超时惩罚等

---

## 5) 参考实现（Torch 伪代码，可直接改成你的 env.py）

> 这段尽量写成“你一粘贴就能改”的形式。实际变量名请按你工程里的 tensor 替换。
> 关键：你需要在 env 里缓存 `last_phi_total`（每个 env 一个标量），以及 `_is_first_step` 逻辑。

```python
import torch
import math

def smoothstep01(x: torch.Tensor) -> torch.Tensor:
    x = torch.clamp(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)

def wrap_to_pi(x: torch.Tensor) -> torch.Tensor:
    # returns in [-pi, pi]
    return torch.atan2(torch.sin(x), torch.cos(x))

def dist_to_interval(x: torch.Tensor, a: float, b: float) -> torch.Tensor:
    # 0 if in [a,b], else distance to nearest endpoint
    return torch.where(
        x < a, a - x,
        torch.where(x > b, x - b, torch.zeros_like(x))
    )

def compute_reward_s1k(
    *,
    # state tensors (N,)
    robot_pos_xy: torch.Tensor,   # (N,2)
    robot_yaw: torch.Tensor,      # (N,)
    tip_xy: torch.Tensor,         # (N,2)
    pallet_pos_xy: torch.Tensor,  # (N,2)
    pallet_yaw: torch.Tensor,     # (N,)
    pallet_depth: float,          # scalar
    lift_height: torch.Tensor,    # (N,)
    last_lift_height: torch.Tensor,# (N,)
    actions: torch.Tensor,        # (N,act_dim)
    last_phi_total: torch.Tensor, # (N,)
    is_first_step: torch.Tensor,  # (N,) bool
    # terminal flags (N,)
    success: torch.Tensor,
    timeout: torch.Tensor,
    time_ratio: torch.Tensor,     # (N,) 0..1
    # cfg scalars
    gamma: float = 0.99,
):
    # ---- geometry ----
    cp = torch.cos(pallet_yaw)
    sp = torch.sin(pallet_yaw)
    u_in = torch.stack([cp, sp], dim=-1)               # (N,2)
    v_lat = torch.stack([-sp, cp], dim=-1)             # (N,2)

    # lateral error (centerline overlap)
    rel_robot = robot_pos_xy - pallet_pos_xy
    y_signed = torch.sum(rel_robot * v_lat, dim=-1)
    y_err = torch.abs(y_signed)

    # yaw error (parallel)
    yaw_err = wrap_to_pi(robot_yaw - pallet_yaw)
    yaw_err_deg = torch.abs(yaw_err) * (180.0 / math.pi)

    # dist_front / insert_norm in pallet axis
    rel_tip = tip_xy - pallet_pos_xy
    s_tip = torch.sum(rel_tip * u_in, dim=-1)  # scalar coordinate along u_in
    s_front = -0.5 * pallet_depth

    dist_front = torch.clamp(s_front - s_tip, min=0.0)
    insert_depth = torch.clamp(s_tip - s_front, min=0.0)
    insert_norm = torch.clamp(insert_depth / pallet_depth, 0.0, 1.0)

    # ---- Stage 1: band + coarse align (Phi1 >=0) ----
    d1_min, d1_max = 2.0, 3.0
    e_band = dist_to_interval(dist_front, d1_min, d1_max)

    e_band_scale = 0.5
    y_scale1 = 0.25
    yaw_scale1 = 15.0
    k_phi1 = 6.0

    E1 = (e_band / e_band_scale) + (y_err / y_scale1) + (yaw_err_deg / yaw_scale1)
    phi1 = k_phi1 / (1.0 + E1)

    # ---- Stage 2: micro approach to mouth, gated by band + coarse align ----
    d2_scale = 1.0
    y_scale2 = 0.12
    yaw_scale2 = 8.0
    k_phi2 = 10.0

    E2 = (dist_front / d2_scale) + (y_err / y_scale2) + (yaw_err_deg / yaw_scale2)
    phi2_base = k_phi2 / (1.0 + E2)

    # band weight opens from 3m -> 2m
    w_band = smoothstep01((d1_max - dist_front) / (d1_max - d1_min))

    # coarse align gate (soft)
    y_gate2, yaw_gate2 = 0.25, 15.0
    w_align2 = torch.exp(- (y_err / y_gate2) ** 2 - (yaw_err_deg / yaw_gate2) ** 2)

    phi2 = phi2_base * w_band * w_align2

    # ---- Stage 3: insertion deepening ----
    ins_start, ins_ramp = 0.02, 0.05
    w3 = smoothstep01((insert_norm - ins_start) / ins_ramp)

    y_gate3, yaw_gate3 = 0.10, 8.0
    w_align3 = torch.exp(- (y_err / y_gate3) ** 2 - (yaw_err_deg / yaw_gate3) ** 2)

    k_ins = 18.0
    phi_ins = k_ins * (0.2 + 0.8 * w_align3) * insert_norm
    phi_ins = phi_ins * w3

    # ---- lift potential + premature penalty ----
    insert_gate_norm, insert_ramp_norm = 0.60, 0.10
    w_lift_base = smoothstep01((insert_norm - insert_gate_norm) / insert_ramp_norm)
    w_lift = w_lift_base * w_align3

    k_lift = 20.0
    phi_lift = k_lift * w_lift * lift_height

    delta_lift = lift_height - last_lift_height
    k_pre = 10.0
    pen_premature = -k_pre * (1.0 - w_lift_base) * torch.clamp(delta_lift, min=0.0)

    # ---- total potential & shaping ----
    # before insertion, phi1 & phi2 dominate; after insertion, phi_ins/phi_lift dominate
    phi_total = (phi1 + phi2) * (1.0 - w3) + phi_ins + phi_lift
    r_pot = gamma * phi_total - last_phi_total

    # first-step protection
    r_pot = torch.where(is_first_step, torch.zeros_like(r_pot), r_pot)

    # ---- dense penalties (keep negative pressure so "原地摆烂"不会赚钱) ----
    rew_time_penalty = -0.003
    rew_action_l2 = -0.01
    k_dist_cont = 0.02  # 比 S1.0j 的 0.03 略轻，避免压死探索

    d_xy = torch.sqrt(dist_front * dist_front + y_err * y_err)
    pen_dense = (
        rew_time_penalty
        + rew_action_l2 * torch.sum(actions * actions, dim=-1)
        - k_dist_cont * d_xy
    )

    # ---- terminal ----
    rew_success = 100.0
    rew_success_time = 30.0
    rew_timeout = -10.0

    r_terminal = torch.zeros_like(r_pot)
    r_terminal = torch.where(
        success,
        rew_success + rew_success_time * (1.0 - time_ratio),
        r_terminal
    )
    r_terminal = torch.where(timeout & (~success), r_terminal + rew_timeout, r_terminal)

    reward = r_pot + pen_premature + pen_dense + r_terminal

    # return reward + aux for logging
    aux = dict(
        y_err=y_err,
        yaw_err_deg=yaw_err_deg,
        dist_front=dist_front,
        insert_norm=insert_norm,
        e_band=e_band,
        w_band=w_band,
        w3=w3,
        w_align2=w_align2,
        w_align3=w_align3,
        w_lift_base=w_lift_base,
        phi1=phi1,
        phi2=phi2,
        phi_ins=phi_ins,
        phi_lift=phi_lift,
        phi_total=phi_total,
        r_pot=r_pot,
        pen_premature=pen_premature,
        pen_dense=pen_dense,
    )
    return reward, phi_total, aux
```

---

## 6) 参数速查表（建议初值）

| 参数 | 建议值 | 含义 |
|---|---:|---|
| `d1_min, d1_max` | 2.0, 3.0 | Stage1 距离带（到口距离） |
| `e_band_scale` | 0.5 | Stage1 距离带误差归一化尺度 |
| `y_scale1, yaw_scale1` | 0.25m, 15deg | Stage1 对齐误差尺度 |
| `k_phi1` | 6.0 | Stage1 势函数强度 |
| `d2_scale` | 1.0 | Stage2 前向距离尺度 |
| `y_scale2, yaw_scale2` | 0.12m, 8deg | Stage2 更严格的对齐尺度 |
| `k_phi2` | 10.0 | Stage2 势函数强度 |
| `ins_start, ins_ramp` | 0.02, 0.05 | 插入接管阈值与缓坡 |
| `y_gate2, yaw_gate2` | 0.25m, 15deg | Stage2 软门控（粗对齐） |
| `y_gate3, yaw_gate3` | 0.10m, 8deg | Stage3 软门控（严对齐） |
| `k_ins` | 18.0 | 插入势函数强度 |
| `insert_gate_norm` | 0.60 | 允许举升的插入深度门槛 |
| `insert_ramp_norm` | 0.10 | 举升门控缓坡 |
| `k_lift` | 20.0 | 举升势函数强度 |
| `k_pre` | 10.0 | 空举惩罚系数 |
| `rew_time_penalty` | -0.003/step | 时间惩罚 |
| `rew_action_l2` | -0.01 | 动作 L2 惩罚 |
| `k_dist_cont` | 0.02 | 连续距离惩罚（比 0.03 略轻） |
| `rew_timeout` | -10.0 | 超时惩罚 |
| `rew_success, rew_success_time` | 100, 30 | 成功奖励 |

---

## 7) 训练侧一个提醒（不写进 reward，但你最好同步改）

如果 PPO 的 `action std` 在训练中一路膨胀到几十（你之前日志就出现过），再精致的 reward 也会被随机噪声淹没。建议至少做其一：
- entropy_coef 退火到 0
- clamp log_std / std 上限（例如 std<=2~3）
- 或改用 tanh-squashed distribution 并正确计算 logprob/entropy

---

## 8) S1.0L 实装差异（相对 S1.0k）

### 8.1 Stage 距离项

- Stage1/2 的 `e_band / w_band / E2` 距离项改为 `dist_front_base`
- `insert_depth / insert_norm` 仍保持 `tip` 几何定义，不影响成功判定

```python
rel_base = root_pos[:, :2] - pallet_pos[:, :2]
s_base = torch.sum(rel_base * u_in, dim=-1)
dist_front_base = torch.clamp(s_front - s_base, min=0.0)
stage_dist_front = dist_front_base
```

### 8.2 势函数与门控参数

- `gamma: 0.99 -> 1.0`
- `ins_start: 0.02 -> 0.10`
- `ins_ramp: 0.05 -> 0.15`
- `y_gate3: 0.10 -> 0.18`
- `yaw_gate3: 8 -> 12`
- `phi_ins = k_ins * (0.4 + 0.6*w_align3) * insert_norm * w3`

### 8.3 总势函数组合

```python
# 默认（S1.0L）
phi_total = phi1 + phi2 + phi_ins + phi_lift

# 回滚开关（对照实验）
if suppress_preinsert_phi_with_w3:
    phi_total = (phi1 + phi2) * (1.0 - w3) + phi_ins + phi_lift
```

### 8.4 新增奖励/终止机制

- 里程碑一次性奖励：`approach / coarse_align / insert10 / insert30`
- 失败早停：`early_stop_fly`（跑飞）+ `early_stop_stall`（低动作且势函数长期不变）
- 早停惩罚在 `_get_rewards()` 结算，done 在 `_get_dones()` 读取，避免时序错位

---

**文件结束。**
