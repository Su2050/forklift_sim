# Expert Policy v7 — 后续计划

> 分支: `exp/expert-v7` @ `cf46608`  
> 前置文档: `docs/expert_v7_summary.md`  
> 日期: 2026-02-16

---

## 当前状态

v7 修复后 3-seed smoke test (15 episodes):
- ins>=0.1: **53%** (基线 80%)
- avg max ins: **0.294** (基线 0.300)
- 残留问题: 高 vf0 (3/15), 大偏差失败 (4/15), 横向漂移 (2/15)

---

## Phase 1: 诊断（零代码改动）

目标：搞清楚 v7 与基线之间 53% vs 80% 差距的根本原因。

### D1: 高 vf0 episode 逐步诊断

- 对 3 个高 vf0 episode 跑 verbose 模式（不加 `--quiet`）
- 逐步打印 `steer / drive / v_forward / lat / yaw`
- 判断是否为：
  - (a) Stanley 转向饱和导致原地打转
  - (b) 物理碰撞卡住（与托盘/地面）
  - (c) 转向方向错误（极性问题）

具体命令示例：
```bash
PYTHONPATH=forklift_expert_policy_project:$PYTHONPATH \
./isaaclab.sh -p scripts/play_expert.py \
  --task Isaac-Forklift-PalletInsertLift-Direct-v0 \
  --num_envs 1 --headless --episodes 5 --seed 42 \
  --log_file logs/smoke_test_v7/seed_42_verbose.log
```

### D2: Stanley vs PD 控制器对比

- 切到 v7-A0 commit (`33f5850`, FSM + PD) 跑同样 3 seed x 5 ep
- 对比 A0 (PD) vs A2+fix (Stanley) 的 insertion rate 和 avg max ins
- 如果 PD 在同样 FSM 框架下表现相当或更好，则 Stanley 的额外复杂度不值得

### D3: 初始条件可比性验证

- 对比 v7 smoke test 的 `init(d, lat, yaw)` 与基线 v5bc stress test 报告中 seed=0/42/99999 的前 5 个 episode
- 确认初始条件是否一致（排除环境随机化差异）
- 如果不一致，用基线的相同 seed 数据按初始条件分桶对比

---

## Phase 2: 快速改进（根据诊断结果选择性执行）

### 2A: 如果 vf0 由转向饱和引起

- 降低 `k_e`（5.0 → 3.0）或 `max_steer_far`（0.80 → 0.65），减少极端转角
- 或引入 **速度-转向耦合**：`eff_max_steer = f(v_forward)`，低速时自动降低最大转向

### 2B: 如果大偏差失败不可避免

- 计算基线 v5bc 在相同初始条件桶（|lat|>0.35 + |yaw|>0.15）的成功率
- 如果基线也失败，则此为 env 固有极限，不需额外优化
- 如果基线成功，对比基线在这些 episode 上的控制策略（docking 阶段的转向/速度模式）

### 2C: 如果 Stanley 无明显优势

- 考虑回退到 PD 控制器 + FSM 架构
- PD 更简单、参数少、与基线一致性更好，BC 训练可能更稳定

### 2D: 清理死代码

- 移除 v5 legacy 参数：`k_lat`, `k_yaw`, `retreat_steer_gain`, `retreat_k_yaw`（v7 FSM 已不使用）
- 移除 `ins_v_max`, `ins_v_min`, `ins_lat_ok`, `ins_yaw_ok` 等旧 insertion 阶段参数
- 简化 `ExpertConfig`，只保留 v7 FSM 实际使用的字段

---

## Phase 3: 大规模验证 + 决策

### 3.1 Stress Test

- 20 seed x 20 episodes = 400 total（与基线 v5bc 完全同规模）
- 使用 `run_stress_test.sh`（需修改分支检查逻辑，或直接用 `play_expert.py` 循环）
- 生成与基线报告格式一致的 `report.txt`

### 3.2 达标门限

| 指标 | 达标线 | 基线参考 |
|------|--------|---------|
| ins>=0.1 率 | >= 75% | 80% |
| avg max ins | >= 0.28 | 0.300 |
| ins>=0.75 率 | >= 3% | 4% |
| stuck-in-retreat 率 | 0% | 0% |

### 3.3 决策逻辑

```
if v7 stress test 达标:
    → 锁定 v7 为 BC 训练数据源
    → 进入 BC training pipeline 开发
elif v7 仅略低于达标线 (ins>=0.1 在 65-75%):
    → Phase 2 微调 + 重测
else:
    → 回退到 v5bc 作为 BC 数据源
    → v7 FSM/Stanley 经验留档供 v8 参考
```

---

## 时间估算

| Phase | 预计耗时 | 依赖 |
|-------|---------|------|
| Phase 1 (D1+D2+D3) | 2-3 小时 | 无（只跑测试 + 分析） |
| Phase 2 | 1-2 小时 | Phase 1 结论 |
| Phase 3 stress test | 3-4 小时（运行时间） | Phase 2 完成 |
| 总计 | 6-9 小时 | |
