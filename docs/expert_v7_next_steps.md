# Expert Policy v7 — 后续计划（v2, 合并审查反馈）

> 分支: `exp/expert-v7` @ `f3e487e`  
> 前置文档: `docs/expert_v7_summary.md`  
> 日期: 2026-02-16  
> 变更: 合并两份专业审查反馈，新增 D0 语义校准、重塑 Phase 3 决策框架

---

## 当前状态

v7 修复后 3-seed smoke test (15 episodes):
- ins>=0.1: **53%** (基线 80%)
- avg max ins: **0.294** (基线 0.300)
- ins>=0.75: **7%** (基线 4%)
- 残留问题: 高 vf0 (3/15), 大偏差失败 (4/15), 横向漂移 (2/15)

**关键发现**: 基线成功 episode (ins>=0.75) 的 `dist_front` 在 1.10–1.78m (中位数 1.55m)。
当前 `hard_wall=0.30m` 永远不会被触发 → FSM 退化为 Approach 单阶段（"僵尸状态机"）。

---

## Phase 1: 诊断（零代码改动，按优先级排序）

### D0: dist_front 语义校准（最高优先级）

**为什么最优先**: fork_length=0 是止血补丁，但 `hard_wall=0.30` 使 FSM 四阶段设计完全失效。
不校准这个坐标系，后续所有调参都在错误空间里进行。

**物理背景**: `dist_front = d_x - pallet_half_depth`，其中 `d_x` 是底盘中心/后轴到托盘中心的距离。
即便货叉完全插入（ins=0.75），底盘仍距托盘前沿 1.1–1.5m。

**具体行动**:
1. 跑 1 个成功 episode（verbose 模式），逐步记录 `dist_front` 和 `insert_norm`
2. 找到 ins 首次开始增长（>0.01）时的 dist_front（预估 ~1.5–1.7m）
3. 找到 ins>=0.75 时的 dist_front（预估 ~1.1–1.2m）
4. 据此推导正确阈值：
   - `pre_insert ≈ ins开始增长时的dist_front + 0.10`（预估 ~1.60–1.80m）
   - `hard_wall ≈ ins>=0.75时的dist_front + 0.10`（预估 ~1.20–1.30m）
   - `retreat_target_dist > pre_insert`（确保 HardAbort 退到 FSM 可重新接入的距离）

### D1: 高 vf0 episode 诊断

**预判根因 — Scrubbing Friction Lock**:
Stanley 在低速+大 lat 时，`atan2` 输出极大转角（满舵）。PhysX 中满舵轮胎侧向静摩擦极大，
纵向推力不足以克服 → 速度降为零 → Stanley 分母更小 → 转角更大 → 正反馈死锁。

**诊断方法**:
- 对 seed=42 跑 verbose 模式，重点看 EP2/EP4 的逐步数据
- 检查 `steer` 是否持续饱和在 ±0.80、`v_forward` 是否为 0、`drive` 是否正常
- 如确认 Scrubbing Lock → Phase 2 引入速度-转向耦合

```bash
PYTHONPATH=forklift_expert_policy_project:$PYTHONPATH \
./isaaclab.sh -p forklift_expert_policy_project/scripts/play_expert.py \
  --task Isaac-Forklift-PalletInsertLift-Direct-v0 \
  --num_envs 1 --headless --episodes 5 --seed 42 \
  --log_file logs/smoke_test_v7/seed_42_verbose.log
```

### D2: Stanley vs PD 对照实验

- 切到 v7-A0 commit (`33f5850`, FSM+PD)，跑同样 3 seed x 5 ep
- 对比维度（不只看成功率）：
  - ins>=0.1 率、avg max ins
  - vf0 发生率、持续时间
  - 最长连续 steer_sat 步数
  - 漂移反弹次数
- **判据**: 对 BC 而言"轨迹平滑、少犹豫、少卡死"比"多 5% 浅插入率"更重要

### D3: 初始条件可比性

- 对比 v7 smoke test 的 `init(d, lat, yaw)` 与基线 v5bc 报告中 seed=0/42/99999 的前 5 ep
- 确认初始条件一致；如不一致，用初始条件分桶对比

---

## Phase 2: 定向修复（按 D0–D3 结果执行）

### 2A: FSM 阈值重校准（D0 驱动，最高优先级）

基于 D0 数据重新设定：
- `hard_wall`: 0.30 → **~1.20m**
- `pre_insert`: 0.80 → **~1.60m**
- `retreat_target_dist`: 2.0 → **> pre_insert**
- 这将激活 FSM 四阶段，让 Straighten/FinalInsert/HardAbort 真正参与控制

### 2B: vf0 修复 — 速度-转向耦合（D1 确认后）

引入物理闭环：无前进速度 → 剥夺大转向权限
```python
eff_max_steer = _clip(abs(v_forward) * 2.0, 0.10, cfg.max_steer_far)
```

Stanley 分母改用 `max(abs(v_forward), commanded_v)` 避免零速正反馈。
优先调 `k_soft`（0.4→0.8）而非直接砍 `k_e`。

### 2C: k_e / k_damp 调参（横向漂移修复）

叉车是后轮转向（非最小相位），k_e=5.0 比例增益过高 → 超调发散。
- `k_e`: 5.0 → **3.0**
- `k_damp`: 0.20 → **0.35–0.40**（加大偏航角速度阻尼，"按住车尾"防超调）

### 2D: 死代码清理

- 移除 v5 legacy 未使用参数：`k_lat`, `k_yaw`, `retreat_steer_gain`, `retreat_k_yaw`, `ins_v_max` 等
- `fork_length=0` 已成为永久设计 → 删除 dtc 相关命名，统一用 dist_front
- 在 D2 明确 Stanley/PD 决策之后执行，避免清理后又回退

---

## Phase 3: 大规模验证 + BC 决策（重塑决策框架）

### 3.1 Stress Test

20 seed x 20 ep = 400 total（与基线 v5bc 同规模）

### 3.2 指标体系（不再"唯成功率论"）

| 指标 | 达标线 | 说明 |
|------|--------|------|
| ins>=0.1 率 | >= 65% | 放宽（v7 过滤了低质量成功） |
| avg max ins | >= 0.28 | 同基线 |
| ins>=0.75 率 | >= 3% | 同基线 |
| **golden trajectory yield** | **>= 40%** | 新指标：无 HardAbort + 无高 vf0 + ins>=0.3 的纯净轨迹比例 |
| stuck-in-retreat 率 | 0% | 安全底线 |

### 3.3 BC 数据纯度视角

v7 的 53% 成功率 vs v5bc 的 80% 看似劣势，但：
- v7 ins>=0.75 率 (7%) 高于基线 (4%) — 成功轨迹质量更高
- v5bc 80% 中可能包含靠碰撞运气挤入的"劣质成功"，喂给网络会学到噪声

**BC 黄金定律: 数据纯度 >>> 专家胜率**

行动：
1. Phase 1 中必须**抽查可视化 v7 成功局的视频**，确认轨迹是否更直、更平滑
2. 如果 v7 轨迹质量明显优于 v5bc，即使成功率低也可直接锁定
3. 未来 `collect_demos.py` 中，利用 FSM 状态无情丢弃 HardAbort/高 vf0 残次局

### 3.4 决策逻辑

```
if golden_yield >= 40% AND avg_max_ins >= 0.28:
    → 锁定 v7，进入 BC training pipeline
    → collect_demos.py 跑 10000 局，丢弃残次局
elif ins>=0.1 在 55-65% AND golden_yield >= 30%:
    → Phase 2 微调 + 重测
else:
    → 回退 v5bc 作为 BC 数据源
    → v7 FSM/Stanley 经验留档供 v8 参考
```

---

## 优先级清单（按 ROI 排序）

1. **D0: dist_front 语义校准** — 激活 FSM 四阶段的前提，否则一直在错误坐标系调参
2. **D1: 高 vf0 根因** — 最可能把 53% 拉回 70%+ 的单点修复
3. **D2: Stanley vs PD 对照** — 避免为复杂而复杂，如果 PD 不差就回退
4. **2A: FSM 阈值重校准** — D0 数据驱动的直接修改
5. **2B: 速度-转向耦合** — D1 确认后的针对性修复
6. **2C: k_e/k_damp 调参** — 横向漂移修复
7. **Phase 3 stress test** — 最终验证

预计总耗时: 6–9 小时
