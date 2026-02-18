# s1.0v Phase-2b Bugfix Batch — 验证报告

> 日期: 2026-02-18  
> 分支: feat/s1.0u  
> 策略版本: s1.0v (Phase-2b bugfix)

---

## 1. 修复内容概述

本批次一次性修复了 3 个确定性 bug + 1 个回归 bug：

| # | Bug | 修复方案 | 涉及文件 |
|---|-----|---------|---------|
| Fix-1 | `lift_on_insert_norm=0.75` 物理不可达 (实际上限 ~0.477) | 阈值降至 `0.40`；lift 阶段保持 `drive=0.15` 前推、`steer=0` | `expert_policy.py` |
| Fix-2 | Planner kappa 量纲错误：`tan(steer)` 未乘 `steer_angle_rad` | `PlannerParams` 新增 `steer_angle_rad=0.6`；kappa 公式改为 `tan(steer * steer_angle_rad) / wheelbase` | `align_lattice_planner_s10u.py` + `expert_policy.py` |
| Fix-3 | Retreat 阶段无 OOB 防护，倒车可超出 3.0m | 三重防护：target_dist 2.7→2.5、oob_guard 强制终止、边界减速带 (dist>2.3 渐减 70%) | `expert_policy.py` |
| Fix-4 | Handover guard 回归 bug：`align_plan→docking` 交接时强制 retreat 导致全速倒车越界 | 删除 handover guard 代码块，恢复 docking 正常 PD 控制流 | `expert_policy.py` |

---

## 2. 验证方案

### Step-1: 快速回归 (15ep)

- **Seeds**: 42, 88, 123 × 5 episodes
- **目标**: terminated ≤ 37% (基线)，lift 出现，无新异常
- **命令**: `play_expert.py --num_envs 1 --headless --episodes 5`

### Step-2: 广泛验证 (50ep)

- **Seeds**: 0, 7, 13, 42, 55, 88, 99, 123, 177, 200 × 5 episodes
- **目标**: 确认修复在多样化初始条件下稳定

---

## 3. 结果

### 3.1 核心指标

| 指标 | Phase-2 基线 (18ep) | Phase-2b 15ep | Phase-2b 50ep |
|------|---------------------|---------------|---------------|
| **Terminated (dist>3.0)** | **37%** | **0%** | **0%** |
| Inserted (ins≥0.1) | 0% | 0% | 0% |
| Lift triggered | 0% | 0% | 0% |
| Truncated (超时) | 63% | 100% | 100% |

### 3.2 距离 / 横向偏移统计 (50ep)

| 指标 | min | mean | max | stdev |
|------|-----|------|-----|-------|
| dist (m) | 1.905 | 2.260 | 2.537 | 0.200 |
| \|lat\| (m) | 0.022 | 0.437 | 2.163 | — |

### 3.3 Retreat 安全性 (50ep)

| 指标 | 值 |
|------|-----|
| Retreat 结束的 episodes | 16/50 (32%) |
| Retreat 最大 dist | 2.498 m |
| 超出 OOB 限制 (3.0m) | 0 |

### 3.4 结束阶段分布 (50ep)

| 阶段 | 数量 | 占比 |
|------|------|------|
| docking | 34 | 68% |
| retreat | 16 | 32% |
| lift | 0 | 0% |

### 3.5 各 Seed 汇总

| seed | term | ins | lift | trunc | avg_dist | avg\|lat\| |
|------|------|-----|------|-------|----------|-----------|
| 0 | 0 | 0 | 0 | 5 | 2.246 | 0.354 |
| 7 | 0 | 0 | 0 | 5 | 2.181 | 0.446 |
| 13 | 0 | 0 | 0 | 5 | 2.161 | 0.726 |
| 42 | 0 | 0 | 0 | 5 | 2.290 | 0.204 |
| 55 | 0 | 0 | 0 | 5 | 2.311 | 0.420 |
| 88 | 0 | 0 | 0 | 5 | 2.360 | 0.491 |
| 99 | 0 | 0 | 0 | 5 | 2.393 | 0.373 |
| 123 | 0 | 0 | 0 | 5 | 2.247 | 0.678 |
| 177 | 0 | 0 | 0 | 5 | 2.223 | 0.357 |
| 200 | 0 | 0 | 0 | 5 | 2.192 | 0.327 |

---

## 4. 结论

### 已解决

- **OOB 终止完全消除**：50 个 episode 中 terminated=0（从 37% 降至 0%）
- **Retreat 安全可控**：三重防护生效，最大 retreat dist=2.498，远低于 3.0m OOB 限制
- **Handover guard 回归 bug 修复**：删除有害代码，docking PD 控制器正常接管
- **Planner 曲率物理一致**：kappa 现在正确反映 `steer_angle_rad=0.6` 的缩放

### 未解决（非 bug，属调优范畴）

- **Docking 控制器无法将 dist 推至 fork_reach (≈1.87m) 以内**：
  - 平均 dist=2.26m，最小 1.905m，始终未达到插入距离
  - 这导致 insert_norm 恒为 0，lift 无法触发
  - **根因**：docking 阶段的 PD 控制器 gain 和切换逻辑需要调优，不是 bug

### 下一步建议（Tuning Batch）

1. **调低 docking 触发距离或增大 drive gain**，使 forklift 在 lat<0.3 时更积极前推
2. **增加 docking 阶段最大步数**或改用更激进的前推策略
3. **评估 planner 修复后 align_plan 是否产生更好的对齐**，以减少 docking 阶段的横向偏差

---

## 5. 文件变更清单

| 文件 | 变更 |
|------|------|
| `forklift_expert/expert_policy.py` | Fix-1 (lift threshold+drive)、Fix-3 (retreat OOB 三重防护)、Fix-4 (删除 handover guard)、Fix-2 (透传 steer_angle_rad) |
| `forklift_expert/align_lattice_planner_s10u.py` | Fix-2 (PlannerParams 新增 steer_angle_rad、kappa 公式修正) |
