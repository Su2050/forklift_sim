# Expert Policy v7 — 工作总结

> 分支: `exp/expert-v7`  
> 基线: `master @ ae4dc8c` (v5bc expert, 400-episode stress test)  
> 日期: 2026-02-16

---

## 1. 目标

在 v5bc 基线上引入三项架构改进：

1. **FSM 状态机** — 替换原有的硬编码阶段切换，引入 Approach / Straighten / FinalInsert / HardAbort / Lift 五阶段，带迟滞防抖
2. **Stanley 控制器** — 替换 PD 控制器，用几何追踪算法（crosstrack + heading）提供更对称的横向纠偏
3. **安全门控** — 插入前检查对齐度，未对齐则 HardAbort 后退修正，避免货叉歪斜撞击托盘

---

## 2. 实现过程（7 commits）

| Commit   | 内容 |
|----------|------|
| `33f5850` | **v7-A0**: FSM 五阶段架构 + 旧 PD 控制器（对照组） |
| `4eb72b2` | **v7-A1**: Stanley 控制器替换 Approach 阶段的 PD |
| `b6b6e0f` | **v7-A2**: Straighten 子阶段 + `insert_norm` 进度检测 |
| `0694e5f` | 23 项纯 Python 单元测试（Stanley / FSM / 逆转极性 / 进度检测） |
| `6f6f398` | Smoke test 调参：k_e 3→5, max_steer_far 0.65→0.80, FSM 阈值放宽 |
| `cf46608` | **关键修复**：fork_length=0, FinalInsert 使用 Stanley 转向 |

### 关键修复说明（cf46608）

初始版本使用 `fork_length=1.87m` 计算 `dtc = dist_front - fork_length` 作为 FSM 距离阈值。但 env 初始 `dist_front` 范围仅 1.75–2.58m，导致 `dtc` 在大量 episode 开局即为 0，FSM 直接跳入 HardAbort，Approach 阶段完全无法工作。

同时，FinalInsert 原设计为 `steer=0`（盲推），与基线行为（持续转向插入）根本矛盾。

修复方案：
- `fork_length` 设为 0.0（`dtc = dist_front`），使 FSM 阈值回到 dist_front 空间
- FinalInsert 改用 Stanley 转向 + `max_steer_near` 限幅
- `hard_wall` 调为 0.30m，`pre_insert` 调为 0.80m，与基线 `stop_dist/slow_dist` 对齐

---

## 3. 性能对比

### 3.1 修复前 vs 修复后 vs 基线

```
指标              修复前v7(10ep)    修复后v7(15ep)    基线v5bc(400ep)
──────────────    ──────────────    ──────────────    ───────────────
ins >= 0.1           0%               53%               80%
ins >= 0.75          0%                7%                4%
avg max ins          0.008            0.294             0.300
avg min |lat|        0.257            0.182             0.131
```

### 3.2 逐 seed 结果（修复后，每 seed 5 episodes）

| Seed  | ins>=0.1 | ins>=0.75 | avg max ins | 备注 |
|-------|----------|-----------|-------------|------|
| 0     | 4/5      | 1/5       | 0.438       | EP3 达到 lift (ins=0.756) |
| 42    | 1/5      | 0/5       | 0.178       | 3 个 episode 高 vf0 |
| 99999 | 3/5      | 0/5       | 0.267       | EP2 高 vf0, EP4 大偏差失败 |
| **合计** | **8/15** | **1/15** | **0.294** | |

---

## 4. 核心结论

### 有效的改进

- **Stanley 控制器**：提供了对称、可解释的横向纠偏能力；`k_e * lat / (v + k_soft)` 公式在低速时自动增强修正力度
- **FSM 状态机框架**：代码结构清晰，状态转换可追踪，单元测试可独立验证每个阶段逻辑
- **HardAbort 智能后退**：`k_lat_rev * lat + k_yaw_rev * yaw` 在后退时主动修正对齐，优于旧版固定转向

### 失败的设计

- **fork_length 概念**：env 初始 dist_front 与 fork_length 量级相近，dtc 偏移导致 FSM 工作范围崩溃。**根本原因**是 `d_xy_r_x` 的参考点离货叉尖端很远（约 2.4–2.8m），fork_length=1.87 严重低估了实际距离
- **FinalInsert 盲推**（steer=0）：基线数据显示 80% 的成功 episode 全程在 docking 阶段完成插入（持续转向），单独的"盲推"阶段与物理动力学不兼容
- **FSM 阶段未被触发**：修复后的 15 个 episode 中，绝大多数全程在 Approach 完成插入，FinalInsert/Straighten 基本未触发（dist_front 在 episode 结束前未降到 hard_wall 以下）

---

## 5. 三类残留问题

### 5.1 高 vf0（3/15 episodes）

87–91% 步数前进速度为零。受影响 episode：
- seed=42 EP2: init(lat=-0.170, yaw=-0.024, d=2.58), vf0=87%
- seed=42 EP4: init(lat=-0.336, yaw=0.120, d=2.28), vf0=88%
- seed=99999 EP2: init(lat=-0.234, yaw=0.220, d=2.43), vf0=91%

可能原因：Stanley 转向饱和导致叉车原地打转，或物理碰撞卡住。需 verbose 诊断。

### 5.2 大初始偏差失败（4/15 episodes）

|lat| > 0.35 且 |yaw| > 0.15 的 episode 无法在有限 Approach 距离内收敛：
- seed=0 EP0: init(lat=0.567, yaw=0.240)
- seed=42 EP0: init(lat=0.585, yaw=-0.163)
- seed=42 EP4: init(lat=-0.336, yaw=0.120)
- seed=99999 EP4: init(lat=0.378, yaw=-0.226)

基线 v5bc 对同类条件的成功率待确认（初始条件可比性尚未验证）。

### 5.3 横向漂移（2/15 episodes）

lat 先收敛到接近零后反弹：
- seed=42 EP1: min_lat=0.001 → end_lat=2.040
- seed=99999 EP0: min_lat=0.000 → end_lat=-1.603

基线也有 35.8% 的 drift pattern，可能是 env 动力学固有特性。

---

## 6. 文件清单

- `forklift_expert_policy_project/forklift_expert/expert_policy.py` — v7 expert 主逻辑
- `forklift_expert_policy_project/tests/test_retreat_logic.py` — 23 项单元测试
- `logs/smoke_test_v7/seed_{0,42,99999}_v7fix.log` — 修复后 smoke test 日志
- `logs/smoke_test_v7/seed_0_v{1..5}.log` — 修复前调参迭代日志
