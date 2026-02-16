# Expert Policy v6 完整总结报告

> 分支: `exp/expert-v6`（基于 `master` @ `c59e4ba`）
> 日期: 2026-02-14
> 状态: **Smoke Test 阶段，未进入 Full Stress Test**

---

## 目录

1. [背景与动机](#1-背景与动机)
2. [Phase A 诊断（零代码变更）](#2-phase-a-诊断零代码变更)
3. [v6 修复方案设计](#3-v6-修复方案设计)
4. [实施与迭代修复](#4-实施与迭代修复)
5. [测试体系](#5-测试体系)
6. [性能对比](#6-性能对比)
7. [结论与反思](#7-结论与反思)
8. [文件清单](#8-文件清单)

---

## 1. 背景与动机

### 1.1 问题发现

在对 v5-BC（含 v5-A lat_true 替代、v5-B retreat 公式优化、v5-C lat-dependent steer bonus）运行
400 episode 压力测试后，发现两个核心瓶颈:

| 瓶颈 | 数据证据 |
|------|---------|
| **横向漂移 (lat drift)** | 61.5% episode 结束时 \|lat\| 饱和 (>=0.499m)；35.8% episode 曾对齐后漂走 |
| **近距未对齐接触** | 视频观察到叉车货叉与托盘未对齐就碰撞并推动托盘 |

### 1.2 v5-BC 基线性能 (400 ep)

```
avg max ins:          0.300
reached ins>=0.10:    80.0%
reached ins>=0.50:    13.5%
reached ins>=0.75:    4.0%
avg vf0%:             15.4%
lat_sat (end):        61.5%
lat drift pattern:    35.8%
```

### 1.3 根本矛盾

环境中 `dist_front` 是机器人车身到托盘中心的距离减去托盘半深度（1.08m），而叉车货叉长度约
1.87m。这意味着当 `dist_front=1.87` 时，货叉尖端已经接触托盘前沿，但策略将其视为"还有 1.87m
距离"，继续全速前进。

---

## 2. Phase A 诊断（零代码变更）

### 2.1 方法

在 `master` 分支上（不修改任何代码），对 3 个 seed (0, 42, 99999) 各运行 20 episode，共 60 episode，
收集逐步轨迹数据。使用 `scripts/analyze_drift.py` 分析漂移事件频率。

### 2.2 核心发现

来源: `logs/diag_v6/drift_report.txt`

**事件频率:**

| 事件 | EP 出现率 | 含义 |
|------|----------|------|
| steer_sat (转向饱和) | 90% | 直接原因 — 转向无法纠偏 |
| lat_runaway (lat 失控) | 82% | 饱和导致 lat 单调增长 |
| contact_misaligned (接触时未对齐) | 80% | Layer 1 问题确认 |
| yaw_diverge (偏航发散) | 77% | 链条起点 |
| lat_zero_cross (lat 跨零/过纠偏) | 52% | Ackermann 副作用 |
| stuck (推托盘) | 40% | 物理接触后果 |

**决定性对比:**

| 对比组 | Episodes | Drift 率 | avg max_ins |
|--------|----------|----------|-------------|
| steer_sat >= 20 步 | 50 | **80%** | 0.296 |
| steer_sat < 5 步 | 7 | **0%** | 0.398 |

> **结论:** 转向饱和是漂移的唯一直接原因。低饱和 episode 零漂移、更高插入。

### 2.3 漂移链条模型（5 步因果链）

```
1. k_lat 纠正横向误差
      ↓ Ackermann 耦合效应
2. yaw 恶化 (yaw_diverge, 77% ep)
      ↓ 大 yaw 反向驱动 lat
3. lat 跨零过纠偏 (51% 为过度纠偏)
      ↓ 需要更大转向来回拉
4. steer 饱和 (90% ep, 平均最长连续 69 步 = 690 真实步)
      ↓ 无法纠正
5. lat 单调增长 → 物理推托盘 (40% ep)
```

### 2.4 成功 episode 特征

成功组 (ins >= 0.75, 3 ep) 平均初始距离 1.87m（最近），短接近路径 = 更少漂移机会。

---

## 3. v6 修复方案设计

基于诊断结果，设计双层修复 + MAE (最小可归因实验) 实施顺序。

### 3.1 Layer 1: Fork-Aware Distance + Alignment Gating

**目标:** 解决"货叉已接触但策略不知道"的感知盲区。

**L1-A: Fork-Aware Distance**
```python
fork_length = 1.87  # metres
dist_to_contact = max(dist_front - fork_length, 0.0)
```
- 引入 `dist_to_contact` (dtc) 作为货叉尖端到托盘前沿的距离
- fork-proximity slowdown: `dtc < 1.0m` 时开始减速，`dtc=0` 时最低 20% 速度

**L1-B: Alignment Gate**
- 当 `dtc < gate_margin (0.30m)` 且 `|lat| > gate_lat_ok` 或 `|yaw| > gate_yaw_ok` 时
- 限制速度为 `gate_creep_speed`，防止高速推动未对齐的托盘

### 3.2 Layer 2: Anti-Saturation + Yaw Priority

**目标:** 打断漂移链条，减少转向饱和时间。

**L2-A: Anti-Saturation Speed Control**
- 当 `|raw_steer| > eff_max_steer * sat_speed_thresh (0.85)` 时
- 按饱和程度主动降速，最低保留 `sat_speed_min_factor (40%)` 速度
- 原理: 低速 = Ackermann 转弯更紧 = 更快纠正 = 减少饱和时间

**L2-B: Yaw-Priority Steering**
- `yaw_priority = min(|yaw| / 20deg, 1.0)`
- `eff_k_lat = k_lat * (1 - 0.50 * yaw_priority)` — 降低横向增益
- `eff_k_yaw = k_yaw * (1 + 0.30 * yaw_priority)` — 提升偏航增益
- 打断"修 lat → yaw 恶化"的恶性循环

**L2-C: Raise Max Steer Limits**
- `max_steer_far`: 0.65 → 0.80
- `max_steer_near`: 0.40 → 0.45
- 减少转向饱和概率

### 3.3 MAE 实施顺序

| 阶段 | 内容 | 风险 |
|------|------|------|
| v6-A | Layer 1 (L1-A + L1-B) + L2-C | 低 — 保护性措施 |
| v6-B | L2-A + L2-B | 中 — 控制器逻辑变更 |

---

## 4. 实施与迭代修复

### 4.1 Commit 历史 (exp/expert-v6 分支)

共 6 个功能 commit（不含从 master 继承的 docs/script commit）:

| # | Commit | 描述 |
|---|--------|------|
| 1 | `a38ed49` | **v6-A 核心:** fork-aware distance + alignment gate + steer limit uplift |
| 2 | `e2708d0` | **v6-B 核心:** anti-saturation speed control + yaw-priority steering |
| 3 | `711e31b` | **修复:** dist_front vs dist_to_contact 用途分离 |
| 4 | `839cc0c` | **修复:** 消除 gate-retreat 死区 + 增大 retreat target dist |
| 5 | `c45528a` | **修复:** 消除 soft dead zone |
| 6 | `f06c2eb` | **修复:** balanced gate/retreat 阈值 |

### 4.2 各 Commit 详细说明

#### Commit 1: v6-A 核心实现 (`a38ed49`)

**变更:**
- `ExpertConfig` 新增: `fork_length=1.87`, `fork_slow_dist=1.0`, `fork_slow_min=0.20`,
  `gate_margin=0.30`, `gate_lat_ok=0.15`, `gate_yaw_ok=8deg`, `gate_creep_speed=0.15`
- `max_steer_far`: 0.65 → 0.80, `max_steer_near`: 0.40 → 0.45
- `retreat_dist_thresh`: 1.0 → 0.10（改用 dtc）
- `_decode_obs()` 新增 `dist_to_contact` 计算
- `act()` 中所有距离阈值从 `dist_front` 改为 `dist_to_contact`
- 新增 fork-proximity slowdown 和 alignment gate 逻辑
- 单元测试新增 test_case 10-14

**问题:** 将所有控制逻辑都改用 `dtc` 导致转向削弱（dtc 远大于 dist_front 时"近区"范围扩大）。

#### Commit 2: v6-B 核心实现 (`e2708d0`)

**变更:**
- `ExpertConfig` 新增: `sat_speed_thresh=0.85`, `sat_speed_min_factor=0.40`,
  `yaw_priority_angle=20deg`, `yaw_priority_lat_reduce=0.50`, `yaw_priority_yaw_boost=0.30`
- 实现 yaw-priority steering（动态 k_lat/k_yaw 调整）
- 实现 anti-saturation speed control（饱和时降速）
- 单元测试新增 test_case 15-17

#### Commit 3: dist_front vs dist_to_contact 用途分离 (`711e31b`)

**动机:** Smoke test 暴露 v6-A 回归 — 将所有阈值改用 `dtc` 后，近区范围从 ~1m 扩展到 ~2.87m
（dist_front=1.0 → dtc=0），导致远距就开始衰减转向增益。

**修复原则:**
- `dist_to_contact` **仅用于保护性特性**: fork-proximity slowdown, alignment gate, retreat trigger
- **控制逻辑**（steer limits, gain decay, lat bonus, misalignment speed reduction）**回归 dist_front**

这是一个关键的架构决策: **感知用 dtc，控制用 dist_front**。

#### Commit 4: 消除 gate-retreat 死区 (`839cc0c`)

**动机:** Smoke test 出现 `vf0=85%+` 的 episode — 叉车在 gate 区域内因 `|lat|` 介于
`gate_lat_ok(0.15)` 和 `retreat_lat_thresh(0.10)` 之间，gate 限速但不够触发 retreat，卡住不动。
另外 `retreat_target_dist=1.8` < `fork_length=1.87`，retreat 后 dtc 仍为 0。

**修复:**
- `gate_creep_speed`: 0.15 → 0.30（超过静摩擦，Ackermann 能工作）
- `retreat_dist_thresh`: 0.10 → 0.30（与 gate_margin 对齐，消除死区）
- `retreat_target_dist`: 1.8 → 2.5（确保 retreat 后 dtc > 0）

#### Commit 5: 消除 soft dead zone (`c45528a`)

**动机:** 上一轮修复消除了 `vf0>80%`，但 retreat 活动飙升至 45-55%，avg max ins 下降。
原因是 `retreat_lat_thresh=0.20` 过于严格 — Ackermann 转向需要前进才能纠正横向，但 retreat
在任何中等 lat 时就触发。

**修复:** 调整阈值使 retreat 更少触发:
- `retreat_lat_thresh`: 0.20 → 0.35（允许中等 lat 继续 docking 纠正）
- 但这个版本的修改幅度不够，后续被 commit 6 覆盖。

#### Commit 6: balanced gate/retreat 阈值 (`f06c2eb`)

**动机:** commit 5 后 retreat 仍占 ~50% 步数 → "retreat-dock 死循环"。根本原因是
gate 阈值（严）和 retreat 阈值（也严）之间的间隙太小。

**最终 balanced 参数:**

| 参数 | 之前 | 最终 | 理由 |
|------|------|------|------|
| `gate_lat_ok` | 0.15 | **0.25** | 允许中等 lat 继续接近 |
| `gate_yaw_ok` | 8 deg | **12 deg** | 配合放宽 |
| `retreat_lat_thresh` | 0.20 | **0.35** | 平衡: 0.20 太紧致循环, 0.48 太松致卡住 |
| `retreat_yaw_thresh` | 12 deg | **20 deg** | 平衡 |
| `retreat_cooldown` | 50 | **80** | 适中，避免快速重触发 |

---

## 5. 测试体系

### 5.1 单元测试

共 **18 个 test case** (`tests/test_retreat_logic.py`)，覆盖:

| 类别 | Test Cases | 覆盖内容 |
|------|-----------|---------|
| 基础控制 | 0, 1, 2, 6 | lat_true 计算、retreat 转向方向/比例、docking 转向 |
| Retreat 逻辑 | 3, 3b, 4, 5, 5b, 7, 8, 14 | 退出条件、yaw 约束、误触发、cooldown、target_dist、rate limit skip、lat_sat 参数化、dtc 触发 |
| v5-C 特性 | 9 | lat-dependent steer bonus |
| v6-A 特性 | 10, 11, 12, 13 | dist_to_contact 计算、fork-proximity slowdown、alignment gate、info 字段 |
| v6-B 特性 | 15, 16, 17 | yaw-priority steering、anti-saturation speed reduction、info 字段 |

运行命令:
```bash
cd IsaacLab && ./isaaclab.sh -p -m pytest \
  ../forklift_expert_policy_project/tests/test_retreat_logic.py -v
```

最终版本: **18/18 PASSED**

### 5.2 Smoke Test

**配置:** 3 seed (0, 42, 99999) x 5 episode = 15 episode total

**最终版 (balanced 阈值, commit f06c2eb) 结果:**

| seed | avg vf0% | avg max ins | ins>=0.1 | ins>=0.75 | stuck retreat |
|------|----------|-------------|----------|-----------|---------------|
| 0 | 1.3% | 0.096 | 2/5 | 0/5 | 0/5 |
| 42 | 9.0% | 0.007 | 0/5 | 0/5 | 0/5 |
| 99999 | 16.5% | 0.060 | 2/5 | 0/5 | 0/5 |
| **总计** | **~8.9%** | **~0.054** | **4/15 (27%)** | **0/15** | **0/15** |

### 5.3 未完成的测试

- **Full Stress Test (400 ep):** 因 smoke test 已暴露明显退化，未进入大规模压力测试阶段。

---

## 6. 性能对比

### 6.1 核心指标跨版本对比

| 指标 | v4 (400ep) | v5-A (400ep) | v5-BC (400ep) | **v6 smoke (15ep)** |
|------|-----------|-------------|--------------|---------------------|
| avg max ins | 0.301 | 0.307 | 0.300 | **0.054** |
| ins>=0.10 | 79.2% | 80.2% | 80.0% | **27%** |
| ins>=0.50 | 12.8% | 14.5% | 13.5% | N/A |
| ins>=0.75 | 3.0% | 5.0% | 4.0% | **0%** |
| avg vf0% | 18.9% | 18.5% | 15.4% | **8.9%** |
| retreat stuck (>80%) | 0 | 0 | 0 | **0** |
| terminated (early) | 0% | 0% | 6.0% | 0% |

> **注意:** v6 仅 15ep smoke test 数据，与 v4/v5 的 400ep 不完全可比。但退化幅度极大，趋势明确。

### 6.2 v6 改善的方面

- **vf0 降低:** 18.9% (v4) → 8.9% (v6) — 叉车更少出现完全停滞
- **retreat 死循环消除:** 所有版本均为 0，但 v6 是唯一经历过此问题并显式修复的版本
- **安全机制完整:** alignment gate 防止未对齐高速推动托盘

### 6.3 v6 退化的方面

- **avg max ins 暴跌:** 0.300 (v5-BC) → 0.054 (v6)，下降 82%
- **insertion 率暴跌:** 80% → 27%
- **lift 率归零:** 4% → 0%
- **retreat 占比偏高:** 大多数 episode 35-50% 步数花在 retreat

---

## 7. 结论与反思

### 7.1 v6 做对了什么

1. **诊断方法论正确:** Phase A 零代码诊断精准定位了漂移链条和 fork-awareness 盲区
2. **fork-aware distance 概念正确:** `dist_to_contact` 准确反映了货叉到托盘的真实距离
3. **alignment gate 理念正确:** 近距时检查对齐再允许接近，防止了最恶劣的"推歪托盘"情况
4. **MAE 分步实施正确:** v6-A/v6-B 分离实施，便于归因
5. **测试体系完善:** 18 个单元测试 + 3 轮 smoke test 快速迭代

### 7.2 v6 的根本问题

**核心结论: 基础 PD 控制器的横向收敛能力不足，v6 的安全机制无法弥补这一根本缺陷。**

具体表现:
- **Gate 限速 → 更慢接近 → 更多时间漂移:** alignment gate 降低了速度，但 Ackermann 转向
  在低速下纠偏效率并未等比提高，反而给了更多步数让 lat 漂走
- **Retreat 是双刃剑:** 每次 retreat 虽然拉开了距离，但也浪费了 ~80 步，且 retreat 后重新
  接近时可能再次偏离
- **Anti-saturation 降速副作用:** 降速减少了饱和，但也延长了接近时间
- **Yaw-priority 对抗性:** 降低 k_lat 来保护 yaw 的同时，lat 纠正被削弱

这些机制叠加后的净效果: **更安全但更慢、更犹豫、更低效。**

### 7.3 v5-BC 为何反而更好？

v5-BC 没有 gate/retreat 保护，"鲁莽"地全速前进:
- **短接近时间:** 漂移机会窗口小
- **高速 Ackermann:** 更大前进速度 = 更快的纠偏角速度
- **成功靠运气分布:** 初始条件好的 episode 能直接插入；差的直接推歪 — 但 4% lift 率高于 v6 的 0%

### 7.4 教训总结

1. **"做减法"往往比"做加法"更难:** 每增加一个安全机制都带来副作用，层层叠加后系统变得过于
   保守
2. **Ackermann 运动学的根本限制:** 后轮驱动叉车只能通过前进/后退来改变横向位置，纯 PD 控制
   在大 lat + 大 yaw 场景下收敛性差
3. **保护机制要"精准打击"而非"地毯覆盖":** gate + retreat + anti-sat + yaw-priority 四重
   保护相互干扰，不如选择一个最关键的机制做精
4. **15-D 观测空间的根本限制:** 环境提供的 `y_err_obs` 被 clip 在 +-0.5m，虽然 expert 内部
   计算了 `lat_true`，但这反映了 obs 设计对控制器的制约
5. **规则策略的天花板:** 对于需要长时间精细控制的任务，规则策略的参数空间可能无法覆盖所有
   初始条件的解

### 7.5 可能的后续方向

| 方向 | 思路 | 复杂度 |
|------|------|--------|
| **A. 精简 v6** | 只保留 fork-aware distance + alignment gate（去掉 anti-sat 和 yaw-priority），减少副作用叠加 | 低 |
| **B. 改进控制器架构** | 引入 MPC 或轨迹优化替代 PD，处理 Ackermann 耦合 | 高 |
| **C. 回退 v5-BC + 局部改进** | 接受"不够安全但更有效"的策略，仅在极端场景加保护 | 低 |
| **D. 修改环境** | 扩大 obs 范围（取消 y_err clip）、增加 episode 长度、提供更多信息 | 中 |
| **E. 端到端 RL** | 放弃规则策略，直接用 RL 学习，BC 仅作 warmstart | 高 |

---

## 8. 文件清单

### 8.1 代码

| 路径 | 说明 |
|------|------|
| `forklift_expert_policy_project/forklift_expert/expert_policy.py` | 核心策略（ExpertConfig + act()） |
| `forklift_expert_policy_project/forklift_expert/obs_spec.json` | 15-D 观测空间规格 |
| `forklift_expert_policy_project/forklift_expert/action_spec.json` | 3-D 动作空间规格 |
| `forklift_expert_policy_project/scripts/play_expert.py` | 采样/录制/smoke test 脚本 |
| `forklift_expert_policy_project/scripts/analyze_drift.py` | 漂移诊断分析脚本 |
| `forklift_expert_policy_project/scripts/run_stress_test.sh` | 压力测试批量运行脚本 |
| `forklift_expert_policy_project/scripts/collect_demos.py` | BC 数据采集脚本 |
| `forklift_expert_policy_project/scripts/bc_train.py` | BC 训练脚本 |
| `forklift_expert_policy_project/tests/test_retreat_logic.py` | 18 个单元测试 |

### 8.2 文档

| 路径 | 说明 |
|------|------|
| `docs/expert_v6_fix_plan.md` | v6 修复方案原始计划 |
| `docs/expert_v6_summary_report.md` | 本报告 |
| `forklift_expert_policy_project/README.md` | 项目说明 |
| `forklift_expert_policy_project/instructions.md` | 设计指导 |
| `forklift_expert_policy_project/bc_train_design.md` | BC 训练设计 |

### 8.3 日志与数据

| 路径 | 说明 |
|------|------|
| `logs/diag_v6/drift_report.txt` | Phase A 诊断报告 (60 ep) |
| `logs/diag_v6/seed_{0,42,99999}.log` | 诊断逐步轨迹数据 |
| `logs/diag_v6/seed_{0,42,99999}_stdout.log` | 诊断运行输出 |
| `logs/smoke_v6/seed_{0,42,99999}.log` | v6 最终版 smoke test 日志 |
| `logs/stress_test/large_v4/report.txt` | v4 基线 400ep 压力测试报告 |
| `logs/stress_test/large_v5a/report.txt` | v5-A 基线 400ep 压力测试报告 |
| `logs/stress_test/large_v5bc/report.txt` | v5-BC 基线 400ep 压力测试报告 |

### 8.4 视频

| 路径 | 说明 |
|------|------|
| `data/videos/expert_v5bc/rl-video-step-0.mp4` | 专家策略可视化录制 (v5-BC master 版本) |

### 8.5 Git 分支

| 分支 | HEAD | 说明 |
|------|------|------|
| `exp/expert-v6` | `f06c2eb` | v6 开发分支 (6 commit from master) |
| `exp/expert-v5bc` | `ab339c1` | v5-BC 基线分支 |
| `master` | `c59e4ba` | 主分支 (v5-BC 合入后) |
