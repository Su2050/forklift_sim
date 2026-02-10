# S1.0M 奖励函数（基于 S1.0L 评审整改）

> **版本**：S1.0M（拆除三颗雷 + 对齐信号增强 + 噪声控制）
>
> **日期**：2026-02-10
>
> **前置版本**：S1.0L（距离带修复 + gamma=1.0 + w3 推迟）
>
> **目标**：解决 S1.0L 训练中"插入有进展但对齐停滞、举升永不可达、hold 过严"的三重阻塞，让 `frac_success` 从零突破。

---

## 1. S1.0L 遗留问题诊断（为什么要做 S1.0M）

### 1.1 S1.0L 500 iter 训练的成就与瓶颈

S1.0L 成功修复了 S1.0k 的三大根因（距离带尺度失配、PBRS 负底噪、w3 过早接管），但 `frac_success` 仍为 0：

| 指标 | S1.0L (500 iter) | 说明 |
|------|-------------------|------|
| `insert_norm_mean` | 0.12~0.15 | 插入已从零突破到稳定 10~15% |
| `frac_inserted` | 0.16~0.26 | 约 20% 环境达到插入阈值 |
| `frac_aligned` | 0.02~0.05 | **极低，策略没学会对齐** |
| `lateral_mean` | 0.23~0.28m | 成功需 ≤0.03m，差 8 倍 |
| `yaw_deg_mean` | 6.2~8.5° | 成功需 ≤3.0°，差 2.5 倍 |
| `w_lift_base` | **0.0000** | **永远为零——举升 shaping 完全锁死** |
| `phi_lift` | **0.0000** | **零举升信号** |
| `hold_counter_max` | 0 | 从未触发过 hold |
| `frac_success` | 0.0000 | 零成功 |
| `action noise std` | 0.78~0.80 | 偏高，精细控制被淹没 |

### 1.2 根因分析（按杀伤力排序）

#### 致命雷：举升门控 `insert_gate_norm=0.60` 物理不可达

- sanity check 实测物理最大 `insert_depth ≈ 1.03m`（convexDecomposition 碰撞限制）
- `pallet_depth_m = 2.16m` → 物理最大 `insert_norm ≈ 1.03/2.16 ≈ 0.477`
- `insert_gate_norm = 0.60` 要求 `insert_norm ≥ 0.60`（即 1.296m）→ **永远不可达**
- 后果：
  - `w_lift_base = smoothstep((0.477-0.60)/0.10) = 0`（永远为零）
  - `phi_lift = k_lift * w_lift * lift_height = 0`（零举升 shaping）
  - `pen_premature = -10 * delta_lift`（一切举升都被重罚）
  - 但 success 要求 `lift_height ≥ 0.12m` → **永远不可能成功**

#### 严重雷：`hold_steps=30` 在物理抖动下极难维持

- `hold_time_s=1.0`，`ctrl_dt=1/30` → `hold_steps=30`
- sanity check A2 实测：从理论成功位姿开始，hold_counter 最高仅 4/30
- 插入深度微小回落即跌破阈值，hold 计数器秒归零
- success 变成"连续 1 秒毫无抖动卡在阈值上方"的极端事件

#### 对齐相关问题

- 成功对齐阈值过紧（0.03m / 3°），策略几乎不可能触及
- phi2 被 `w_align2` 门控丢弃 63%（y_gate2=0.25 太紧）
- 对齐信号被插入信号冲淡，无直接"对齐→强奖励"通路
- action std ≈ 0.78 偏高，精细转向被噪声淹没

---

## 2. S1.0M 改动清单

### 2.1 P0-0a [致命修复]：举升门控下调到物理可达区间

**问题**：`insert_gate_norm=0.60` 在物理上不可达，举升 shaping 永远锁死。

**修复**：

```python
# env_cfg.py
insert_gate_norm: float = 0.35   # 原 0.60 → 0.35（物理最大 ~0.477，留余量）
insert_ramp_norm: float = 0.08   # 原 0.10 → 0.08（0.35~0.43 打开）
k_pre: float = 5.0               # 原 10.0 → 5.0（降低空举惩罚，允许探索）
```

**修复后效果验证**（smoke test iter 29）：
- `w_lift_base: 0.1023`（从永恒 0 → 非零！）
- `phi_lift: -0.0047`（已激活，不再锁死）
- `pen_premature: -0.0026`（从 ~-0.005 大幅降低）

### 2.2 P0-0b [严重修复]：降低 hold 严苛度

**问题**：`hold_steps=30` 在物理抖动下 sanity check 最高只能维持 4 步。

**修复**：

```python
# env_cfg.py
hold_time_s: float = 0.33   # 原 1.0 → 0.33（hold_steps: 30 → ~10）
```

课程阶段先出 success，后续逐步收紧回 1.0s。

### 2.3 P0-1：放宽成功对齐阈值

**问题**：`max_lateral_err_m=0.03` 对当前策略是 8 倍差距。

**修复**：

```python
# env_cfg.py
max_lateral_err_m: float = 0.15   # 原 0.03 → 0.15（5 倍放宽）
max_yaw_err_deg: float = 8.0      # 原 3.0 → 8.0（2.7 倍放宽）
```

### 2.4 P0-2：放宽 w_align2 门控

**问题**：`y_gate2=0.25` 在 `y≈0.25m` 时 `w_align2≈0.37`，丢弃 63% 的 phi2 信号。

**修复**：

```python
# env_cfg.py
y_gate2: float = 0.40    # 原 0.25 → 0.40（w_align2: 0.37 → 0.67）
yaw_gate2: float = 20.0  # 原 15.0 → 20.0
```

**效果验证**（smoke test）：`w_align2: 0.4244`（从 S1.0L 的 ~0.30 提升 40%）

### 2.5 P0-3：收紧对齐尺度

**目的**：增大对齐误差在 E1/E2 中的权重，让 phi1/phi2 对对齐更敏感。

```python
# env_cfg.py
y_scale1: float = 0.15    # 原 0.25（lateral 权重 1.0 → 1.67）
yaw_scale1: float = 10.0  # 原 15.0（yaw 权重 0.5 → 0.75）
y_scale2: float = 0.08    # 原 0.12（lateral 权重 2.08 → 3.13）
yaw_scale2: float = 5.0   # 原 8.0（yaw 权重 0.94 → 1.50）
```

> **风险提示**：会同时降低 phi1/phi2 绝对值。smoke test 中 `phi1=1.41`（S1.0L 为 ~2.0），下降约 30%，仍在可接受范围。如长期训练中 phi1/phi2 持续走低，考虑同步上调 `k_phi1/k_phi2`。

### 2.6 P1-1：新增对齐里程碑

**目的**：给对齐一条"强奖励通路"，弥补势函数间接信号不足。

```python
# env_cfg.py
rew_milestone_fine_align: float = 5.0      # y < 0.10m & yaw < 5°
rew_milestone_precise_align: float = 8.0   # y < 0.05m & yaw < 3°
```

实现：`_milestone_flags` 从 `(N,4)` 扩展到 `(N,6)`，一次性触发，与现有 4 个里程碑使用相同的 flag/reset 机制。条件变量与 `aligned_enough` 判定使用同源的 `y_err` / `yaw_err_deg`，避免"里程碑判过了、success 判不过"的不一致。

### 2.7 P1-2：降低动作噪声

```python
# rsl_rl_ppo_cfg.py
init_noise_std = 0.5    # 原 1.0（初始 std 减半）
entropy_coef = 0.0005   # 原 0.001（降低推高 std 的梯度压力）
```

**效果验证**（smoke test 30 iter）：`action noise std: 0.50`（S1.0L 同期为 ~0.98），精确控制在目标区间。

---

## 3. 参数速查表（S1.0L → S1.0M 差异）

| 参数 | S1.0L | S1.0M | 说明 |
|------|------:|------:|------|
| `insert_gate_norm` | 0.60 | **0.35** | 致命修复：物理可达区间 |
| `insert_ramp_norm` | 0.10 | **0.08** | 缓坡配套调整 |
| `k_pre` | 10.0 | **5.0** | 降低空举惩罚 |
| `hold_time_s` | 1.0 | **0.33** | hold_steps: 30→~10 |
| `max_lateral_err_m` | 0.03 | **0.15** | 放宽 5 倍（课程起点） |
| `max_yaw_err_deg` | 3.0 | **8.0** | 放宽 2.7 倍 |
| `y_gate2` | 0.25 | **0.40** | phi2 信号保留翻倍 |
| `yaw_gate2` | 15.0 | **20.0** | 放宽偏航门控 |
| `y_scale1` | 0.25 | **0.15** | 对齐敏感度提升 |
| `yaw_scale1` | 15.0 | **10.0** | 对齐敏感度提升 |
| `y_scale2` | 0.12 | **0.08** | 对齐敏感度提升 |
| `yaw_scale2` | 8.0 | **5.0** | 对齐敏感度提升 |
| `rew_milestone_fine_align` | — | **5.0** | 新增 |
| `rew_milestone_precise_align` | — | **8.0** | 新增 |
| `init_noise_std` | 1.0 | **0.5** | PPO 配置 |
| `entropy_coef` | 0.001 | **0.0005** | PPO 配置 |

其余参数（`gamma`、`stage_distance_ref`、`k_phi1/k_phi2/k_ins/k_lift`、`suppress_preinsert_phi_with_w3`、早停参数等）沿用 S1.0L 不变。

---

## 4. 新增日志项

| 日志 key | 含义 |
|----------|------|
| `milestone/hit_fine_align` | 首次触发 fine_align（y<0.10m & yaw<5°）的比例 |
| `milestone/hit_precise_align` | 首次触发 precise_align（y<0.05m & yaw<3°）的比例 |

---

## 5. Smoke Test 结果（30 iter，64 envs）

### 5.1 关键指标对比（iter 29）

| 指标 | S1.0L (30 iter) | S1.0M (30 iter) | 变化 |
|------|:---------------:|:---------------:|------|
| `w_lift_base` | **0.0000** | **0.1023** | 从锁死到激活 |
| `phi_lift` | **0.0000** | **-0.0047** | 从零到非零 |
| `pen_premature` | -0.0054 | -0.0026 | 惩罚减半 |
| `w_align2` | 0.2383 | 0.4244 | 信号保留 +78% |
| `frac_aligned` | 0.0312 | 0.0938 | 对齐率 +200% |
| `frac_lifted` | 0.0000 | 0.0312 | 有环境在举升 |
| `action noise std` | 0.98 | 0.50 | 精确控制在目标区间 |
| `s0/r_milestone` | 0.0156 | 0.1562 | 里程碑奖励 +10x |
| `milestone/hit_insert30` | 0.0000 | 0.0156 | 有触发 |
| `phi1` | 1.88 | 1.41 | 下降 ~25%（P0-3 scale 收紧的代价） |

### 5.2 结论

- **P0-0a 致命修复验证通过**：`w_lift_base` 和 `phi_lift` 已从永恒 0 变为非零，举升 shaping 已解锁
- **P0-0b + P0-1 联合效果**：`frac_aligned` 在 30 iter 即达 9.4%，远超 S1.0L 的 3%
- **P0-2 效果显著**：`w_align2` 从 0.24 提升到 0.42，phi2 有效信号增加近一倍
- **P1-2 噪声控制精确**：action std 从 0.98 控制到 0.50
- **P0-3 风险可控**：phi1 下降约 25%（1.88→1.41），但对齐敏感度提升的收益预计超过幅值下降的代价

---

## 6. 验收标准（正式训练 500 iter）

| 指标 | 期望 | 一票否决 |
|------|------|----------|
| `s0/w_lift_base` | > 0（非零） | 是 |
| `phi/phi_lift` | > 0（非零） | 是 |
| `phase/hold_counter_max` | > 0（非零） | — |
| `phase/frac_aligned` | 从 3% → **15%+** | — |
| `err/lateral_mean` | 从 0.25m → **< 0.15m** | — |
| `err/yaw_deg_mean` | 从 7.5° → **< 5°** | — |
| `phase/frac_success` | 从 0 → **> 0**（哪怕 0.01） | — |
| `action noise std` | 从 0.78 → **< 0.5** | — |
| `phi/phi1` | **不低于 S1.0L 的 50%**（≥1.0） | 若过低则回滚 P0-3 |

---

## 7. 涉及文件

| 文件 | 改动 |
|------|------|
| `env_cfg.py` | 举升门控、hold、成功阈值、对齐尺度、门控、里程碑参数（共 16 个参数） |
| `env.py` | `_milestone_flags` 4→6，里程碑逻辑扩展，日志新增 2 项 |
| `rsl_rl_ppo_cfg.py` | `init_noise_std`、`entropy_coef` |

---

## 8. 下一步方向（若 S1.0M 仍不够）

1. **若 phi1/phi2 被压太低**：上调 `k_phi1`（6→8）和 `k_phi2`（10→14）补回势能幅值
2. **若 action std 回弹**：自定义 ActorCritic 子类，在 `_update_distribution` 中 clamp `log_std`
3. **若 success 出现但极稀少**：继续放宽 `hold_time_s`（0.33→0.17），或加 hysteresis 机制
4. **课程收紧路线**：success 稳定后逐步收紧 `max_lateral_err_m`（0.15→0.08→0.03）和 `hold_time_s`（0.33→0.67→1.0）

---

**文件结束。**
