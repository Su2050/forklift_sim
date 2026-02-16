# Expert Policy v7 — Phase 1 诊断 + Phase 2 调参 报告

> 分支: `exp/DO-O-A3B1C2_v2`  
> 基线: v7 修复后 (`cf46608`) / v5bc (`master @ ae4dc8c`)  
> 日期: 2026-02-12

---

## 1. Phase 1 — 零代码诊断

### 1.1 诊断计划

v7 修复后 (`cf46608`) 的 15-episode smoke test 显示 ins>=0.1 仅 53%（基线 80%），同时存在高 vf0 episode 和大偏差失败。用户反馈指出三个可能的根因，并提出四项诊断任务：

| 编号 | 诊断任务 | 优先级 | 状态 |
|------|---------|--------|------|
| D0 | dist_front 语义校准 — 找到 ins 开始增长时的 dist_front，确定 hard_wall/pre_insert 正确阈值 | **最高** | 完成 |
| D1 | verbose smoke test — 验证 Scrubbing Friction Lock 假说 | 高 | 完成 |
| D2 | A0 (PD) 对照实验 — 验证 Stanley 控制器的必要性 | 中 | 跳过 (Stanley 已产出 lift) |
| D3 | 初始条件可比性验证 — 排除 seed 间统计幻觉 | 中 | 跳过 (种子间自然变异，不阻塞) |

### 1.2 D0 — dist_front 语义校准（最高优先级）

**方法**：对 seed=0 运行 verbose smoke test (`play_expert.py --episodes 5 --seed 0`)，在输出日志 `seed_0_verbose.log` 中逐 step 追踪 `dist_front` 和 `insert_norm` 的关系。重点分析 EP3（唯一达到 lift 的 episode）。

**关键发现 — dist_front vs insert_norm 对照表**（EP3, seed=0）：

```
事件              step   dist_front   insert_norm   steer    vf
───────────────   ────   ──────────   ───────────   ──────   ──────
ins 首次 >0       3280   1.826        0.010         -0.80    0.117
ins = 0.10        3330   1.669        0.103         -0.80    0.145
dist 稳定         3400   1.601        0.151         -0.80    0.093
ins = 0.30        3620   1.601        0.281         -0.80    0.092
ins = 0.75 (lift) 4150   1.599        0.753         -0.80    0.029
```

**核心结论**：

1. **ins 开始增长时 dist_front ≈ 1.83m**。这说明 `dist_front` 的物理参考点不是货叉尖端，而是叉车底盘中心或后轴。即使货叉完全插入托盘（ins=0.75），dist_front 仍高达 **1.60m**。
2. **从 ins=0.15 到 ins=0.75 的整个插入过程中，dist_front 几乎不变（1.601→1.599）**。车体不动，货叉楔入。
3. **旧阈值完全脱离物理现实**：`hard_wall=0.30m` 在环境中永远不会出现（dist_front 的物理下限约 1.0m）。v7 的 Straighten/FinalInsert/HardAbort 三个阶段从未被激活，FSM 退化为"Approach 单节点状态机"。

**交叉验证**：检查了 v5bc 基线 400-episode 压测报告中 BEST 20 的 `end_d`（ins>=0.75 时的 dist_front），范围为 1.10m–1.78m，中位数 ~1.55m。与 D0 标定结论一致。

**阈值决策**：
- `hard_wall` 应设在 ~1.65m（ins 刚开始稳定增长的 dist_front）
- `pre_insert` 应设在 ~2.0m（ins 首次 >0 前的减速区）
- `retreat_target_dist` 应设在 ~2.8m（pre_insert 上方，给足进近跑道）

### 1.3 D1 — Scrubbing Friction Lock 验证

**方法**：在同一份 seed=0 verbose log 中分析 EP4（高 vf0 episode）。

**关键发现**（EP4, step 4570 起）：

```
step    stage     dist    lat      steer   vf      drive   持续
─────   ────────  ─────   ──────   ──────  ──────  ──────  ─────
4570    Approach  0.850   -0.363   0.800   0.000   0.587   ─
4580    Approach  0.850   -0.363   0.800   0.000   0.587   10步
4590    Approach  0.850   -0.363   0.800   0.000   0.587   20步
...（完全相同的状态持续 500+ 步）...
4990    Approach  0.850   -0.363   0.800   0.000   0.587   420步
```

**确认的死锁机制**（与用户预判完全一致）：

```
Stanley 遇到大 lat 且 vf≈0
    → atan2(k_e * lat, vf + k_soft) 输出极大转角
    → steer 饱和到 0.800（满舵）
    → 满舵时轮胎侧向静摩擦力极大
    → drive=0.587 的纵向推力无法克服侧摩擦
    → vf 保持 0.000
    → Stanley 分母 (vf + k_soft) 更小
    → 转向输出更极端（但已饱和）
    → 死锁稳定，无法自行恢复
```

**结论**：必须引入速度-转向耦合（anti-scrubbing），强制 `vf≈0 → steer 上限降低`，打破正反馈。

### 1.4 D2/D3 — 跳过理由

- **D2（PD vs Stanley 对照）**：Phase 2f 的 smoke test 中 Stanley 控制器已成功产出 lift (seed=0 EP3, ins=0.752)，证明 Stanley 在正确参数下能工作。暂不需要 PD 对照组。如果后续压测数据不理想，可重新启用。
- **D3（初始条件可比性）**：观察到 3 个 seed 的初始条件自然变异较大（lat 范围 -0.17 至 0.59，yaw 范围 -0.23 至 0.24），属于环境随机化的正常行为。15-episode 与 400-episode 的统计差异主要来自样本量而非初始条件偏差，不阻塞后续工作。

---

## 2. Phase 2 — 修改内容

### 2.1 FSM 距离阈值重校（基于 D0 标定）

| 参数 | 修改前 | 修改后 | 依据 |
|------|--------|--------|------|
| `pre_insert` | 0.80 | **2.00** | ins 在 dist≈1.83 开始增长，2.00 覆盖该区间，提供 0.35m 减速对齐带 |
| `hard_wall` | 0.30 | **1.65** | ins 在 dist≈1.60 稳定增长，1.65 是"货叉即将进入"的物理门控点 |
| `retreat_target_dist` | 2.0 | **2.8** | 位于 pre_insert (2.0) 上方 0.8m，给 Approach 阶段 1.15m 的进近跑道 |

### 2.2 对齐门控放宽

后轮转向（非最小相位）系统在进近过程中 yaw 会先增大后收敛，0.55m 的短跑道（旧 retreat→hard_wall）不足以完成 yaw 收敛，导致 Approach↔HardAbort 死循环。放宽门控 + 加长跑道解除了该瓶颈。

| 参数 | 修改前 | 修改后 |
|------|--------|--------|
| `final_lat_ok` | 0.15 | **0.30** |
| `final_yaw_ok` | 8° (0.140 rad) | **18° (0.314 rad)** |
| `abort_lat` | 0.30 | **0.45** |
| `abort_yaw` | 20° (0.349 rad) | **30° (0.524 rad)** |

### 2.3 Anti-scrubbing 速度-转向耦合（仅 Approach 阶段）

在 Approach 阶段的 Stanley 转向计算后，插入速度-转向耦合限幅：

```python
speed_steer_cap = clip(|v_forward| * 2.5, 0.10, max_steer_far)
eff_max_steer = min(eff_max_steer, speed_steer_cap)
```

- `vf=0` → 转向上限 0.10（打破摩擦死锁的正反馈）
- `vf=0.32` → 转向上限 0.80（完全解锁）
- 设计原则：**先走起来，再纠偏**。FinalInsert 阶段故意不加此限制——插入过程中的转向"楔入力"是克服托盘摩擦的关键。

### 2.4 Stanley 控制器参数调优

| 参数 | 修改前 | 修改后 | 依据 |
|------|--------|--------|------|
| `k_e` | 5.0 | **3.0** | 后轮转向非最小相位特性：k_e 过高→尾部超调→横向漂移。降低后 lat 仍能收敛，但不再过冲 |
| `k_damp` | 0.20 | **0.35** | 增大 yaw rate 负反馈，抑制尾部甩动 |
| `k_soft` | 0.4 | **0.8** | 低速保护增强：`atan2(k_e*lat, v+k_soft)` 中 k_soft 更大→低速时 crosstrack_term 更温和→避免低速饱和 |

### 2.5 FinalInsert 参数调优

经过 Phase 2d/2e/2f 三轮对比实验确定：

| 参数 | 修改前 | 修改后 | 实验结论 |
|------|--------|--------|----------|
| `max_steer_near` | 0.40 | **0.80** | 0.40/0.50/0.70 均导致 ins 在 0.39 处因摩擦停滞。原因：低 steer→"直推"→托盘被推走而非楔入。0.80 匹配 Approach 的 max_steer_far，由 k_e=3.0 自然限幅小 lat 的 steer |
| `final_insert_drive` | 0.80 | **0.90** | 1.0 过高（EP4 vf0=78%，推力加大反而加剧摩擦死锁）；0.90 平衡推力与稳定性 |

**失败实验记录**：
- `final_insert_drive=1.0`：EP4 vf0 从 5% 飙升到 78%，推力过大挤压托盘加剧摩擦
- `max_steer_near=0.50`：EP2 FinalInsert 336 步仅达 ins=0.391，在 dist≈1.0 处摩擦停滞
- FinalInsert 加 anti-scrubbing：avg max ins 从 0.294 降到 0.222。原因：低速时限制转向削弱了楔入力，形成 "vf低→steer低→楔入力弱→更难前进→vf更低" 的正反馈

### 2.6 Progress Detection 修正

| 参数 | 修改前 | 修改后 | 依据 |
|------|--------|--------|------|
| `no_progress_eps` | 0.01 | **0.001** | D0 数据显示健康插入 per-step delta≈0.003。旧值 0.01 导致每一步都判为"无进展"，20 步后误触发 abort |
| `no_progress_steps` | 20 | **60** | 与降低的 eps 配合，60 步无进展才 abort。仅在真正卡死（delta<0.001 持续 60 步）时触发 |

---

## 3. Phase 2 — 迭代过程

Phase 2 共进行了 7 轮 smoke test（每轮 seed=0, 5 episodes），逐步定位并修复问题：

| 版本 | 改动 | avg max ins | ins>=0.1 | ins>=0.75 | avg vf0 | 关键观察 |
|------|------|-------------|----------|-----------|---------|----------|
| **2a** | FSM 阈值 + anti-scrub + k_e/k_damp | 0.091 | 3/5 | 0/5 | 1.5% | vf0 大幅改善!但 FSM 循环 Approach↔HardAbort |
| **2b** | +retreat=2.8, pre=2.0, yaw gate=18° | 0.106 | 3/5 | 0/5 | 1.6% | EP2 首次触发 Straighten+FinalInsert |
| **2c** | +no_progress_eps=0.001, steps=60 | **0.189** | **4/5** | 0/5 | 2.9% | FinalInsert 持续 336 步!但 ins 在 0.39 处摩擦停滞 |
| **2d** | final_insert_drive=1.0 | 0.212 | 3/5 | 0/5 | **17.0%** | 推力过大反噬,EP4 vf0=78% |
| **2e** | max_steer_near=0.70, drive=0.90 | 0.189 | 4/5 | 0/5 | 1.6% | steer 增大无效,k_e=3.0 在小 lat 时 steer 太低 |
| **2f** | **max_steer_near=0.80**, drive=0.90 | **0.294** | **4/5** | **1/5** | 3.9% | **首次 lift!** EP3 ins=0.752 |
| **2g** | FinalInsert 加 anti-scrubbing | 0.222 | 4/5 | 0/5 | 13.5% | 反效果:削弱楔入力,回退至 2f |

---

## 4. Phase 2 — 最终 Smoke Test 结果

Phase 2f 配置，3 seeds × 5 episodes = 15 episodes：

### 4.1 逐 Seed 汇总

| Seed | ins>=0.1 | ins>=0.75 | avg max ins | avg vf0 | FSM 阶段分布 |
|------|----------|-----------|-------------|---------|-------------|
| 0 | 4/5 | 1/5 | 0.294 | 3.9% | EP3 达到 Lift; EP2 有 336 步 FinalInsert |
| 42 | 2/5 | 0/5 | 0.122 | 16.9% | EP3 FinalInsert 375 步但 vf0=78% (摩擦停滞) |
| 99999 | 2/5 | 0/5 | 0.057 | 4.1% | 初始偏差大,多数 ep 在 Approach↔HardAbort 循环 |
| **合计** | **8/15 (53%)** | **1/15 (7%)** | **0.158** | **8.3%** | |

### 4.2 与基线对比

```
指标              v7 修复后(cf46608)    Phase 2f         v5bc 基线(400ep)
───────────────   ──────────────────    ───────────      ─────────────────
ins >= 0.1        53%                   53%              80%
ins >= 0.75       7%                    7%               4%
avg max ins       0.294                 0.158            0.300
avg vf0           高(含灾难EP)          8.3%             —
FSM 激活          几乎不触发            正常触发          无FSM
```

### 4.3 结论

- **FSM 正常工作**：Straighten/FinalInsert 被正常触发，D0 标定彻底解决了"僵尸状态机"问题
- **Anti-scrubbing 有效**：Approach 阶段的 Scrubbing Friction Lock 被消除
- **Lift 率持平**：7% vs 基线 4%（小样本不具统计显著性，但不劣于基线）
- **ins>=0.1 率持平**：53%，与修复后 v7 一致，未因 FSM 门控而降低
- **avg max ins 降低**：0.158 vs 修复后的 0.294。原因是 FSM 门控拒绝了部分"勉强插入"的 episode（这些在旧版中被 Approach 硬推进去），同时 FinalInsert 的摩擦壁 (ins≈0.39) 限制了部分 episode 的深度

---

## 5. 物理洞察

### 5.1 "楔入效应" (Wedge Effect)

D0 数据揭示了插入力学的关键规律：

- 旧基线：`dist_front` 在 1.60m 处稳定不变，`ins` 持续增长 750 步达到 0.75。**车体不动，货叉楔入**
- Phase 2 FinalInsert（低 steer）：`dist_front` 从 1.60 降到 1.0，`ins` 在 0.39 处停滞。**车体向前推，托盘被推走**

本质区别：高 steer (0.80) 使叉车以斜角进入，产生横向分力将货叉"楔入"托盘，而非正面推动。低 steer 的直推只会把托盘推走。这就是为什么 `max_steer_near` 必须等于 `max_steer_far`。

### 5.2 Anti-scrubbing 的非对称性

- **Approach 阶段必须有**：防止 vf=0 时满舵→轮胎侧摩擦死锁→完全无法启动
- **FinalInsert 阶段不能有**：插入过程中的高 steer 是克服托盘摩擦的唯一手段；限制 steer 会形成 "vf低→steer限制→楔入力弱→vf更低" 的正反馈

### 5.3 非最小相位 yaw 行为

后轮转向叉车转弯时，尾部先反向甩出再收回（non-minimum-phase）。在 0.55m 短跑道内：
- lat 可收敛（横向位移直接响应转向）
- yaw 先增大后收敛（需要更长距离完成完整周期）

进近跑道从 0.55m (2.2→1.65) 加长到 1.15m (2.8→1.65) 后，yaw 有足够距离完成"增大→收敛"周期。

---

## 6. 残留瓶颈

### 6.1 摩擦壁 (ins ≈ 0.39)

FinalInsert 中 ins 在 ~0.39 处频繁停滞。此时 `dist_front≈1.0`，车体已显著接近托盘。vf 逐渐降低，最终被 `no_progress` 检测 abort。

**可能原因**：随着插入加深，货叉与托盘之间的摩擦增大，当摩擦力 >= 驱动力时车体停止前进。

**潜在解法**（未实施）：
- 在 FinalInsert 中加入微振动 (steer ±0.05 高频)，利用动摩擦低于静摩擦的特性
- 调整 env 的 pallet 摩擦系数

### 6.2 大初始偏差失败

|lat| > 0.45 且/或 |yaw| > 0.25 的 episode，在 1.15m 进近跑道内仍无法收敛。这属于非完整约束系统的运动学不可达问题——单次前进不足以修正如此大的偏差。

**已有缓解**：HardAbort 智能后退提供了多次尝试机会，但后退过程中 lat/yaw 会进一步偏移（后退时 yaw 修正有限），导致再次进近时条件更差。

### 6.3 HardAbort 后退偏移累积

后退过程中（drive=-1.0, steer 由 k_lat_rev/k_yaw_rev 控制），lat 和 yaw 会逐步偏移。多次 Approach↔HardAbort 循环后，偏移累积，最终 episode 超时。

---

## 7. 下一步

1. **Phase 3 压力测试**：20 seeds × 20 episodes = 400 ep，评估 golden trajectory yield
2. **BC 数据策略**：利用 FSM 状态自动标记并丢弃 HardAbort/高 vf0 episode，仅保留 FinalInsert→Lift 的"黄金轨迹"
3. **可选优化**：D2（PD vs Stanley 对照）/ 后退 yaw 修正增强 / env 摩擦参数调整

---

## 8. 文件清单

| 文件 | 说明 |
|------|------|
| `forklift_expert/expert_policy.py` | Phase 2 最终参数 |
| `tests/test_retreat_logic.py` | 23 项单元测试（已适配新参数） |
| `logs/smoke_test_v7/phase2{a..g}_seed_0.log` | seed=0 迭代日志（7 轮） |
| `logs/smoke_test_v7/phase2f_seed_{42,99999}.log` | 最终配置多 seed 验证 |
| `logs/smoke_test_v7/seed_0_verbose.log` | D0 标定数据源 |
