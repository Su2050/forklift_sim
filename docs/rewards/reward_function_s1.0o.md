# S1.0O 奖励函数设计文档

> 版本：S1.0O | 基于 S1.0N 的三大卡点优化 | 2026-02-11

---

## 一、版本目标

S1.0N（2000 iter）暴露了三大卡点：

| 卡点 | 症状 | 根因 |
|------|------|------|
| A：策略不举升 | `frac_lifted` ~1.7% | premature 惩罚过狠 + lift 缺少正向梯度 |
| B：横向误差平台化 | `lateral_mean` ~0.28m | hold-align sigma 太窄，0.2~0.4m 区间梯度极弱 |
| C：hold 转化不稳定 | `success_now → success` 转化率低 | 单帧越界即清零 hold_counter |

S1.0O 的目标是**同时解决 A/B/C 三个卡点**，将 `frac_success` 从 <1% 提升至 5%+ 并稳定。

---

## 二、Round 1 消融实验结果（7 个单因素，600 iter）

### S1.0N 基线（2000 iter）

| `frac_lifted` | `frac_success` | `lateral_mean` | `yaw_deg_mean` |
|:---:|:---:|:---:|:---:|
| 1.66% | 0.49% | 0.281 m | 4.99° |

### A 类（Lift 提升）

| 变体 | 改动 | `frac_lifted` | `frac_success` | 结论 |
|------|------|:---:|:---:|------|
| A1 | `insert_gate_norm` 0.35→0.20 | 0% | 0% | gate 生效但 lift 未发生 |
| A2 | `insert_ramp_norm` 0.08→0.20 | 0% | 0% | 淘汰（信号被稀释） |
| **A3** | premature 分段温和化 + lift delta 势函数 | **1.27%** | **0.10%** | **A 类赢家** |

### B 类（Lateral 改善）

| 变体 | 改动 | `frac_lifted` | `lateral_mean` | 结论 |
|------|------|:---:|:---:|------|
| **B1** | sigma_y 0.15→0.25, sigma_yaw 8→12, k 0.1→0.3 | **2.73%** | **0.215** | **B 类赢家** |
| B2 | 粗+细双势函数 | 0% | 0.250 | 淘汰（reward 为负） |

### C 类（Hold 稳定）

| 变体 | 改动 | `frac_lifted` | `frac_aligned` | 结论 |
|------|------|:---:|:---:|------|
| C1 | exit debounce（`exit_debounce_steps=3`） | ~1.5% | 0.223 | 中性偏弱 |
| **C2** | hold counter 越界衰减（`decay=0.8`） | **3.0%** | **0.297** | **C 类赢家** |

---

## 三、各赢家具体改动

### A3：premature 分段温和化 + lift 进度 delta 势函数

**问题**：S1.0N 的 premature lift 惩罚基于 `w_lift_base`（二元门控），在浅插入时惩罚过重，导致策略学会"永远不 lift"。

**改动 1 — premature 惩罚分段温和化**

`env_cfg.py` 新增：
```python
premature_hard_thresh: float = 0.05    # insert_norm < 此值时全额惩罚
premature_soft_thresh: float = 0.20    # insert_norm >= 此值时惩罚 → 0
```

`env.py` 逻辑：
```python
# 原: pen_premature = -k_pre * (1.0 - w_lift_base) * clamp(delta_lift, min=0)
# 新: 用 smoothstep 在 [hard_thresh, soft_thresh] 区间平滑过渡
premature_fade = smoothstep(
    (insert_norm - premature_hard_thresh)
    / (premature_soft_thresh - premature_hard_thresh + 1e-6)
)
pen_premature = -k_pre * (1.0 - premature_fade) * clamp(delta_lift, min=0)
```

效果：`insert_norm >= 0.20` 时惩罚接近 0，策略可以安全探索 lift。

**改动 2 — lift 进度 delta 势函数**

`env_cfg.py` 新增：
```python
k_lift_progress: float = 0.4    # lift delta shaping 权重
sigma_lift: float = 0.08        # lift 误差尺度 (m)
```

`env.py` 逻辑：
```python
lift_err = abs(lift_height - lift_delta_m)  # lift_delta_m = 0.12m
phi_lift_progress = exp(-(lift_err / sigma_lift)^2)
r_lift_progress = k_lift_progress * (phi_lift_progress - prev_phi_lift_progress)
# 加入总奖励
rew += r_lift_progress
```

效果：为 lift 提供独立的正向梯度（不依赖 yaw/y 对齐），引导策略"靠近目标举升高度就有奖励"。

---

### B1：增大 hold-align 的 sigma 和权重

**问题**：S1.0N 的 `hold_align_sigma_y=0.15` 在 lateral 0.25m 常态区间梯度极弱（"够用就停"盆地）。

**改动**（仅 `env_cfg.py` 参数调整，无代码逻辑修改）：

| 参数 | S1.0N | S1.0O-B1 |
|------|-------|----------|
| `k_hold_align` | 0.1 | **0.3** |
| `hold_align_sigma_y` | 0.15 m | **0.25 m** |
| `hold_align_sigma_yaw` | 8.0° | **12.0°** |

效果：在 0.2~0.4m 的 lateral 区间提供有效梯度，让策略持续减小横向误差而非停在盆地。

---

### C2：hold counter 越界衰减

**问题**：S1.0N 的 hold counter 在任意一维越界时立即清零，单帧物理抖动就"砍头"。

**改动 1 — `env_cfg.py`**：
```python
hold_counter_decay: float = 0.8   # 越界时 hold_counter *= 0.8（而非清零）
```

**改动 2 — `env.py`**：
```python
# _hold_counter 类型: int32 → float32（支持乘法衰减）
self._hold_counter = torch.zeros((num_envs,), dtype=torch.float32, device=device)

# 越界更新逻辑:
decayed = self._hold_counter * hold_counter_decay
self._hold_counter = torch.where(
    still_ok, self._hold_counter + 1,
    torch.where(grace_zone, self._hold_counter, decayed)  # 越界衰减而非归零
)
```

效果：短暂越界不会完全抹掉积累的 hold 进度，让 `success_now → success` 的转化更鲁棒。

---

## 四、Round 2 组合实验计划

### 组合分支

| 序号 | 分支名 | 组合 | 创建方式 |
|------|--------|------|----------|
| R2-1 | `exp/DO-O-A3B1` | A3 + B1 | A3 merge B1 |
| R2-2 | `exp/DO-O-A3C2` | A3 + C2 | A3 merge C2 |
| R2-3 | `exp/DO-O-A3B1C2` | A3 + B1 + C2 | A3B1 merge C2 |

### 训练配置

- **迭代数**：1000 iter（Round 1 用 600 只为快筛，Round 2 需更充分收敛信号）
- **环境数**：1024
- **种子**：42
- **预计耗时**：每个 ~50 分钟

### 验收 KPI 阈值

| KPI | Round 2 期望 | 说明 |
|-----|-------------|------|
| `phase/frac_lifted` | >= 5% | A 类改进核心指标 |
| `err/lateral_mean` | < 0.22 m | B 类改进核心指标 |
| `phase/frac_success` | >= 1% | C 类改进核心指标 |
| `frac_success / frac_success_now` | >= 30% | hold 转化率 |
| `phase/grace_zone_frac` | < 0.30 | 不能过松 |

### 启动命令

```bash
# R2-1: A3 + B1
cd /home/uniubi/projects/forklift_sim && git checkout master 2>/dev/null; TERM=xterm bash run_experiment.sh A3B1 1000

# R2-2: A3 + C2
cd /home/uniubi/projects/forklift_sim && git checkout master 2>/dev/null; TERM=xterm bash run_experiment.sh A3C2 1000

# R2-3: A3 + B1 + C2（全组合）
cd /home/uniubi/projects/forklift_sim && git checkout master 2>/dev/null; TERM=xterm bash run_experiment.sh A3B1C2 1000
```

---

## 五、Round 2.5 最终验证方案

从 Round 2 中选出表现最佳的组合（预期为 A3B1C2），执行：

1. **长周期训练**：2000 iter，确认指标不出现晚期崩塌
2. **双 seed 复现**：seed=42 + seed=123，确认结果可复现
3. **验收标准**：
   - `frac_success` 稳定 >= 3%
   - `frac_lifted` 稳定 >= 10%
   - `lateral_mean` 稳定 < 0.20m

通过后即为 **S1.0O 正式版本**，合入 master。

---

## 六、参数汇总（S1.0O 全组合 = A3 + B1 + C2）

以下为相对 S1.0N 的全部参数变化：

```python
# === A3: premature 分段温和化 + lift 势函数 ===
premature_hard_thresh: float = 0.05    # [新增] insert_norm < 此值时全额惩罚
premature_soft_thresh: float = 0.20    # [新增] insert_norm >= 此值时惩罚 → 0
k_lift_progress: float = 0.4           # [新增] lift delta shaping 权重
sigma_lift: float = 0.08               # [新增] lift 误差尺度

# === B1: 增大 hold-align sigma + k ===
k_hold_align: float = 0.3              # [改] 0.1 → 0.3
hold_align_sigma_y: float = 0.25       # [改] 0.15 → 0.25
hold_align_sigma_yaw: float = 12.0     # [改] 8.0 → 12.0

# === C2: hold counter 越界衰减 ===
hold_counter_decay: float = 0.8        # [新增] 越界时 counter *= 0.8
# _hold_counter dtype: int32 → float32  # [改] 支持浮点衰减
```

---

## 七、A3B1C2_v2 变体（已淘汰）

在 A3B1C2 全组合基础上，将 **`hold_align_sigma_yaw`** 从 12° 改回 8°，其余参数不变。

| 参数 | A3B1C2 | A3B1C2_v2 |
|------|--------|-----------|
| `hold_align_sigma_yaw` | 12.0° | **8.0°** |

**结论**：从头用 sigma_yaw=8 训练不如二阶段策略（先 12 站稳再微调到 8），已改为二阶段方案。

---

## 八、二阶段训练策略

### 设计思路

直接将 sigma_yaw 从 12 改为 8 从头训练（A3B1C2_v2），等于要求策略同时学会"大致对齐"和"精确 yaw 控制"，增加了早期学习难度。更好的做法是：

1. **Stage A**：sigma_yaw=12（宽松），让策略先学会"进入 hold、出 success"
2. **Stage B**：从 Stage A 最佳 checkpoint 恢复，收紧 sigma_yaw 到 8，同时降低探索噪声，做"磨角度"

这类似于课程学习（curriculum learning）：先学大动作，再精修细节。

### Stage A：sigma_yaw=12 主线续训

- **分支**：`exp/DO-O-A3B1C2`（不改参数，从原始 1000 iter 结果继续）
- **Resume 来源**：`2026-02-11_22-50-21/model_999.pt`
- **续训**：1000 iter（迭代 999→1999）
- **验收**：success_now >= 0.02 且持续 100+ iter

### Stage B：sigma_yaw=8 精修微调

- **分支**：`exp/DO-O-A3B1C2_stageB`
- **Resume 来源**：Stage A 产出的最佳 checkpoint

**参数变更**（相对 A3B1C2）：

| 参数 | Stage A (A3B1C2) | Stage B |
|------|-------------------|---------|
| `hold_align_sigma_yaw` | 12.0° | **8.0°** |
| `entropy_coef` | 0.0015 | **0.0005** |
| `desired_kl` | 0.01 | **0.008** |

**设计依据**：
- `sigma_yaw` 12→8：收紧 yaw 梯度，让策略在已掌握大致对齐的基础上精修角度
- `entropy_coef` 0.0015→0.0005：策略已学会正确行为，降低探索压力
- `desired_kl` 0.01→0.008：更保守的策略更新步幅，防止微调时跳出好区域
- `ClampedActorCritic` 的 `LOG_STD_MIN = log(0.05)` 保持不变（仍需最低探索）

**验收**：
- yaw_deg_mean 从 ~6° 降到 <4°
- success/success_now 转化率 >= 40%
- lateral_mean 不退化（保持 <0.22m）

### 可选 Stage C：sigma_yaw=6

如果 Stage B 效果好，可从 Stage B 最佳 checkpoint 继续微调 sigma_yaw 到 6，用同样的 resume + 分支策略。
