# S1.0L 奖励函数（基于 S1.0k 整改）

> **版本**：S1.0L（基于 S1.0k 训练停滞诊断，修复三大根因）
>
> **日期**：2026-02-10
>
> **前置版本**：S1.0k（严格中心线定义 + 三阶段势函数）
>
> **目标**：解决 S1.0k 训练 500+ iter 零成功率的根因——距离带尺度失配、Stage3 过早接管、PBRS 负底噪——并补充里程碑奖励与失败早停机制。

---

## 1. S1.0k 训练停滞诊断（为什么要做 S1.0L）

### 1.1 S1.0k 的问题表现

S1.0k 训练跑了 553 iter（~6500 万步），核心指标：

- `frac_success = 0.0000`：从未有任何环境触发成功
- `frac_inserted = 0.0000`：从未达到插入阈值
- `insert_norm_mean ≈ 0.02~0.06`：插入深度极低
- `w_band ≈ 1.0`：距离带门控长期焊死
- `e_band ≈ 1.6~1.8`：距离带误差永远不可能为 0
- `r_pot ≈ -0.016`：shaping 变成固定负常数

### 1.2 根因分析

| 根因 | 说明 | 日志证据 |
|------|------|----------|
| **A: 距离带尺度失配** | `d1_min=2.0, d1_max=3.0` 用在 fork tip 的 `dist_front` 上，但 tip 到托盘口只有 0.2~0.5m，`e_band` 恒为 ~1.7，Stage1 课程直接塌方 | `e_band=1.7599`，`w_band=0.999979` |
| **B: w3 过早接管** | `ins_start=0.02` 导致 2% 插入就开始压制 phi1/phi2，对齐 shaping 被过早削弱 | `w3=0.16~0.30`，但 `frac_inserted=0` |
| **C: PBRS 负底噪** | `gamma=0.99` 在慢变化 Phi 上，`r_pot ≈ (0.99-1)*1.68 = -0.0168`，变成每步固定扣税 | `r_pot=-0.0161` 与理论值完全吻合 |
| **D: 策略学会"少动"** | 负项太稳、正项太弱，最稳提升方式是减少动作 → `pen_dense` 减小但任务不推进 | `‖a‖²` 从 1.25 降到 0.29 |

### 1.3 根因之间的因果链

```
距离带 [2,3]m 用于 tip 的 dist_front（实际 0.2~0.5m）
    → e_band 恒为 ~1.7（数学上锁死，不是"学不会"）
    → w_band 恒为 ~1（Stage2 永远"全开"，失去阶段意义）
    → Stage1 的"太近要倒车"和 Stage2 的"dist_front→0"天然对打
    → r_pot = (γ-1)*Φ ≈ -0.017（每步固定扣税，无方向信息）
    → 策略发现"少动少罚"是最稳提升路径
    → insert_norm 退化，frac_success 永远为 0
```

---

## 2. S1.0L 改动清单

### 2.1 Stage1/2 距离参考点改为 base（A1）

**问题**：`dist_front` 用 fork tip 计算，tip 已经在托盘口附近，距离只有 0.2~0.5m，与 [2,3]m 的 band 完全失配。

**修复**：Stage1/2 的 `e_band`、`w_band`、`E2` 的距离项改用 `dist_front_base`（robot root 到托盘口的距离），插入深度（`insert_depth/insert_norm`）仍基于 tip。

```python
# env.py: 新增 dist_front_base
rel_base = root_pos[:, :2] - pallet_pos[:, :2]
s_base = torch.sum(rel_base * u_in, dim=-1)
dist_front_base = torch.clamp(s_front - s_base, min=0.0)
stage_dist_front = dist_front_base  # stage_distance_ref == "base"
```

```python
# env_cfg.py
stage_distance_ref: str = "base"
```

**效果**：`dist_front_base` 初始分布 ~1.4~2.9m，与 [2,3] band 产生有效交集，`e_band` 和 `w_band` 不再焊死。

### 2.2 推迟 Stage3 接管 + 取消 phi1/phi2 压制（A2）

**问题**：`ins_start=0.02` 导致 2% 插入就开始用 `(1-w3)` 压制 phi1/phi2，对齐 shaping 在最需要的时候被削弱。

**修复**：

```python
# env_cfg.py
ins_start: float = 0.10   # 原 0.02 → 10% 深度再开始接管
ins_ramp: float = 0.15    # 原 0.05 → 25% 左右才满接管
suppress_preinsert_phi_with_w3: bool = False  # 默认不压制
```

```python
# env.py: phi_total 组合
if self.cfg.suppress_preinsert_phi_with_w3:
    phi_total = (phi1 + phi2) * (1.0 - w3) + phi_ins + phi_lift  # 旧逻辑（回滚用）
else:
    phi_total = phi1 + phi2 + phi_ins + phi_lift  # S1.0L 默认
```

### 2.3 PBRS 改纯差分（A3）

**问题**：`gamma=0.99` 在 Phi 变化缓慢时，`r_pot ≈ -0.01 * Phi` 变成每步固定负税。

**修复**：

```python
# env_cfg.py
gamma: float = 1.0  # 纯差分，进步立刻见效
```

### 2.4 Stage3 门控放宽 + phi_ins baseline 提升

**问题**：`y_gate3=0.10, yaw_gate3=8` 过紧，`w_align3 ≈ 0.07`，导致插入信号过弱。

**修复**：

```python
# env_cfg.py
y_gate3: float = 0.18    # 原 0.10
yaw_gate3: float = 12.0  # 原 8.0
```

```python
# env.py: baseline 提升
phi_ins = self.cfg.k_ins * (0.4 + 0.6 * w_align3) * insert_norm * w3
# 原：(0.2 + 0.8 * w_align3)
```

### 2.5 里程碑奖励（一次性触发）

**目的**：为探索到正确轨迹的回合提供强正信号，打破"暗黑森林"。

```python
# env_cfg.py
rew_milestone_approach: float = 1.0       # dist_front_tip < 0.25m
rew_milestone_coarse_align: float = 2.0   # y_err < 0.20m & yaw < 10°
rew_milestone_insert_10: float = 5.0      # insert_norm > 0.10
rew_milestone_insert_30: float = 10.0     # insert_norm > 0.30
```

实现要点：
- 每个环境维护 `_milestone_flags (N, 4)` bool 缓冲区
- 每步检查条件，只在**首次达成**时给奖励
- `_reset_idx` 中按 `env_ids` 清零，避免跨 episode 泄漏

### 2.6 失败早停

**目的**：避免跑满 359 步全是失败流水账，提高有效样本密度。

```python
# env_cfg.py
early_stop_d_xy_max: float = 3.0           # 跑飞阈值
early_stop_d_xy_steps: int = 30            # 连续 N 步超限则终止
early_stop_stall_phi_eps: float = 0.001    # phi 变化阈值
early_stop_stall_steps: int = 60           # 连续 N 步无进展则终止
early_stop_stall_action_eps: float = 0.05  # 动作幅度阈值
rew_early_stop_fly: float = -2.0           # 跑飞惩罚
rew_early_stop_stall: float = -1.0         # 摆烂惩罚
```

实现要点：
- 计数器更新和惩罚在 `_get_rewards()` 中完成（确保奖励路径稳定）
- 终止判定在 `_get_dones()` 中只读取 flag，不重复计算
- `terminated = tipped | success | _early_stop_fly | _early_stop_stall`

---

## 3. 参数速查表（S1.0L 实际值）

| 参数 | S1.0k | S1.0L | 说明 |
|------|------:|------:|------|
| `gamma` | 0.99 | **1.0** | 纯差分 shaping |
| `stage_distance_ref` | — | **"base"** | Stage1/2 用 root 距离 |
| `d1_min, d1_max` | 2.0, 3.0 | 2.0, 3.0 | 不变（base 距离下有效） |
| `ins_start` | 0.02 | **0.10** | 推迟 Stage3 接管 |
| `ins_ramp` | 0.05 | **0.15** | 接管缓坡加宽 |
| `y_gate3` | 0.10 | **0.18** | 放宽 Stage3 门控 |
| `yaw_gate3` | 8.0 | **12.0** | 放宽 Stage3 门控 |
| `phi_ins baseline` | 0.2 | **0.4** | 提升插入信号下限 |
| `suppress_preinsert_phi_with_w3` | — | **False** | 不压制 phi1/phi2 |
| `rew_milestone_approach` | — | 1.0 | 新增 |
| `rew_milestone_coarse_align` | — | 2.0 | 新增 |
| `rew_milestone_insert_10` | — | 5.0 | 新增 |
| `rew_milestone_insert_30` | — | 10.0 | 新增 |
| `early_stop_d_xy_max` | — | 3.0 | 新增 |
| `early_stop_stall_steps` | — | 60 | 新增 |

其余参数（`k_phi1/k_phi2/k_ins/k_lift` 等）沿用 S1.0k 不变。

---

## 4. 新增日志项

| 日志 key | 含义 |
|----------|------|
| `err/dist_front_base_mean` | robot root 到托盘口的距离（base 参考点） |
| `err/stage_dist_front_mean` | Stage1/2 实际使用的距离值 |
| `s0/r_milestone` | 里程碑奖励（per step 均值） |
| `s0/pen_early_stop` | 早停惩罚（per step 均值） |
| `milestone/hit_approach` | 当前 batch 中首次触发 approach 的比例 |
| `milestone/hit_align` | 当前 batch 中首次触发 coarse_align 的比例 |
| `milestone/hit_insert10` | 当前 batch 中首次触发 insert 10% 的比例 |
| `milestone/hit_insert30` | 当前 batch 中首次触发 insert 30% 的比例 |
| `term/frac_early_fly` | 因跑飞被早停的环境比例 |
| `term/frac_early_stall` | 因摆烂被早停的环境比例 |

---

## 5. 训练结果（500 iter，~410 万步）

### 5.1 运行环境

```
GPU: NVIDIA Tegra NVIDIA GB10 (92GB)
num_envs: 128
max_iterations: 500
总步数: 4,096,000
总耗时: 20 分 15 秒
```

### 5.2 关键指标演化

| 指标 | iter 0 | iter ~100 | iter ~500 | 趋势 |
|------|--------|-----------|-----------|------|
| `insert_norm_mean` | 0.020 | 0.11~0.14 | 0.12~0.15 | 上升后平台 |
| `frac_inserted` | 0.000 | 0.11~0.23 | 0.16~0.26 | 稳定非零 |
| `frac_aligned` | 0.010 | 0.01~0.04 | 0.02~0.05 | 微升但极低 |
| `lateral_mean` | — | 0.23m | 0.23~0.28m | 不收敛 |
| `yaw_deg_mean` | — | 7~9° | 6.2~8.5° | 不收敛 |
| `frac_success` | 0.000 | 0.000 | 0.000 | 未突破 |
| `hold_counter_max` | 0 | 0 | 0 | 未突破 |
| `frac_lifted` | 0.000 | 0.000 | 0.000 | 未突破 |
| `w_band` | 0.617 | 0.93~0.98 | 0.93~0.97 | 不再焊死 1.0 |
| `e_band` | 0.083 | 0.33~0.43 | 0.36~0.44 | 可学习量级 |
| `r_pot` | 0.000 | -0.002~+0.012 | +0.002~+0.012 | 接近 0，方向正确 |
| `action noise std` | 0.98 | 0.76 | 0.78~0.80 | 下降后回弹 |
| Mean reward | -10.9 | -19.1 | -17.2~-19.1 | 波动不收敛 |
| `phi_ins` | — | 0.8~1.0 | 1.0~1.4 | 持续上升 |
| `w3` | — | 0.24~0.29 | 0.29~0.39 | 稳步上升 |

### 5.3 里程碑触发情况

| 里程碑 | 是否触发 | 频率 |
|--------|----------|------|
| `hit_approach`（dist_front < 0.25m） | 是 | 零星（~1%） |
| `hit_align`（y<0.2m & yaw<10°） | 是 | 极少（<1%） |
| `hit_insert10`（insert_norm > 10%） | 是 | 定期（1~2%） |
| `hit_insert30`（insert_norm > 30%） | 是 | 偶发（<1%） |

### 5.4 早停情况

| 早停类型 | 触发率 |
|----------|--------|
| `frac_early_fly` | 0.000（无跑飞） |
| `frac_early_stall` | 0.000（无摆烂检测触发） |

早停未触发说明策略一直在"活跃但低效地探索"——没有摆烂也没有跑飞，但也跑满了每个 episode 的 359 步。

### 5.5 典型 iteration 日志（iter 492）

```
Learning iteration 492/500

Mean action noise std: 0.78
Mean reward: -18.35

phi/phi1: 2.0304
phi/phi2: 1.0085
phi/phi_ins: 1.4121
phi/phi_lift: 0.0000
phi/phi_total: 4.4511

s0/r_pot: 0.0074
s0/pen_premature: -0.0030
s0/pen_dense: -0.0539
s0/r_milestone: 0.0078

s0/w_band: 0.9668
s0/w3: 0.3455
s0/w_align2: 0.3783
s0/w_align3: 0.2745

err/lateral_mean: 0.2467
err/yaw_deg_mean: 6.9788
err/insert_norm_mean: 0.1491
err/dist_front_base_mean: 1.6010

phase/frac_inserted: 0.2500
phase/frac_aligned: 0.0391
phase/frac_success: 0.0000
```

---

## 6. 效果评估

### 6.1 S1.0L 相对 S1.0k 的改善

| 对比项 | S1.0k (553 iter) | S1.0L (500 iter) | 改善 |
|--------|:---------:|:---------:|------|
| `e_band` | ~1.7（焊死） | 0.36~0.44 | 从失配区间进入可学习量级 |
| `w_band` | ~1.0（焊死） | 0.93~0.97 | 不再恒等于 1 |
| `r_pot` | -0.016（固定负税） | ~0.005（接近 0） | 消除负底噪 |
| `insert_norm_mean` | 0.02~0.06（退化） | 0.12~0.15（上升） | 从零插入到稳定插入 |
| `frac_inserted` | 0.000 | 0.16~0.26 | 从零突破到 20%+ |
| `w3` | 0.16~0.30（过早开） | 0.29~0.39 | 接管时机合理化 |
| `phi_ins` | — | 1.0~1.4 | 插入势函数有效激活 |
| `milestone/hit_insert30` | — | 偶发触发 | 有环境突破 30% 深度 |

**结论：S1.0L 的三大修复全部验证有效。**距离带课程结构恢复正常，shaping 信号方向正确，策略从"完全不插入"进化到"稳定插入 10~15%"。

### 6.2 S1.0L 仍存在的瓶颈

**核心瓶颈：策略学会了"往前推"但没学会"对齐"。**

| 问题 | 数据 | 成功阈值 | 差距 |
|------|------|----------|------|
| 横向偏移 | `lateral_mean ≈ 0.25m` | `≤ 0.03m` | **8 倍** |
| 偏航误差 | `yaw_deg_mean ≈ 7.5°` | `≤ 3.0°` | **2.5 倍** |
| 对齐率 | `frac_aligned ≈ 3%` | 需要高频触发 | 极低 |

原因分析：

1. **对齐奖励信号被插入信号冲淡** — phi1 ≈ 2.0 中对齐分量（y/0.25 + yaw/15）在 E1 总和里占比不够大
2. **没有直接的"对齐→强奖励"通路** — 只通过势函数间接影响，缺乏里程碑级信号
3. **action std ≈ 0.78 仍然偏高** — 精细转向控制被探索噪声淹没，从 0.76 回弹到 0.78~0.80

---

## 7. 下一步建议（S1.0M 方向）

### 7.1 放宽成功判定阈值（课程学习起点）

```python
# 先让 success 出现，后续再收紧
max_lateral_err_m: float = 0.15    # 原 0.03 → 暂时放宽 5 倍
max_yaw_err_deg: float = 8.0       # 原 3.0 → 暂时放宽 2.7 倍
```

### 7.2 增加对齐相关里程碑

```python
# 强化对齐信号
rew_milestone_fine_align: float = 5.0   # lateral < 0.10m & yaw < 5°
rew_milestone_precise_align: float = 8.0  # lateral < 0.05m & yaw < 3°
```

### 7.3 限制 action std 上限

在 PPO 配置中 clamp `log_std` 上限（例如 `std_max=1.5`），或做退火：前 200 iter 保持高探索，之后逐步降低。

### 7.4 验收标准

改完后应在 200 iter 内看到：
- `frac_aligned` 从 3% 抬到 10%+
- `hold_counter_max` 开始非零
- `frac_success` 从 0 突破（哪怕 0.01 也是里程碑）

---

## 8. 涉及文件

| 文件 | 说明 |
|------|------|
| `forklift_pallet_insert_lift_project/isaaclab_patch/source/isaaclab_tasks/isaaclab_tasks/direct/forklift_pallet_insert_lift/env_cfg.py` | 参数配置 |
| `forklift_pallet_insert_lift_project/isaaclab_patch/source/isaaclab_tasks/isaaclab_tasks/direct/forklift_pallet_insert_lift/env.py` | 奖励/终止/重置逻辑 |
| `docs/diagnostic_reports/success_sanity_check_2026-02-10.md` | sanity check 报告 + S1.0L 落地记录 |
| `docs/rewards/reward_function_s1.0k.md` | S1.0k 原始设计 + S1.0L 差异说明 |

---

**文件结束。**
