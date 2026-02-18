# s1.0v Phase-2b Bugfix Delivery — 综合总结

> 日期: 2026-02-18  
> 分支: `feat/s1.0u`  
> 策略版本: `s1.0v` (Phase-1 → Phase-2 → **Phase-2b bugfix**)

---

## 0. 背景

Phase-1 修复了时间尺度不匹配（`dt`）和 planner 基本触发问题。  
Phase-2 的 6-seed 冒烟测试暴露了两大核心问题：
- **37% 的 episode 因 `dist>3.0` 被环境 OOB 终止**
- **lift 从未触发**，insert_norm 恒为 0

深入排查 `forklift_pallet_insert_lift_project` 环境源码后，定位出 3 个确定性 bug。在修复过程中又发现 1 个因 Phase-2 改动引入的回归 bug。Phase-2b 将这 4 个 bug 一次性修复。

---

## 1. Bug 清单与修复

### Fix-1: Lift 阈值物理不可达

| 项目 | 内容 |
|------|------|
| **症状** | `lift` 阶段从未触发 |
| **根因** | `lift_on_insert_norm=0.75`，但环境中由于 pallet 凸分解，`insert_norm` 物理上限约 0.477 |
| **修复** | 阈值降至 `0.40`；lift 阶段增加 `drive=0.15`（前推防物理抖动回退）、`raw_steer=0.0`（抑制横向力） |
| **文件** | `expert_policy.py` L211, L639-640 |

### Fix-2: Planner 曲率量纲不匹配

| 项目 | 内容 |
|------|------|
| **症状** | Planner 路径预测偏差大，drift 频繁触发重规划 |
| **根因** | Planner 用 `tan(steer)/wheelbase` 计算曲率，但环境中 `steer` 被乘以 `steer_angle_rad=0.6` 后才是物理转角。导致 planner 预估曲率比实际大 ~1.85x |
| **修复** | `PlannerParams` 新增 `steer_angle_rad=0.6`；kappa 改为 `tan(steer × steer_angle_rad)/wheelbase`；policy 构造 planner 参数时透传 |
| **文件** | `align_lattice_planner_s10u.py` L82, L213；`expert_policy.py` L819 |

### Fix-3: Retreat 阶段无 OOB 防护

| 项目 | 内容 |
|------|------|
| **症状** | Retreat 倒车时 `dist` 轻易超过 3.0m，触发环境 OOB 终止 |
| **根因** | retreat 逻辑只检查 `alignment_improved` 和 `target_dist`，缺少对 OOB 边界的硬保护 |
| **修复** | 三重防护：(1) `bbox_retreat_target_dist` 2.7→2.5；(2) 新增 `oob_guard`：`dist≥2.85` 强制结束；(3) 边界减速带：`dist>2.3` 时倒车速度渐减 70% |
| **文件** | `expert_policy.py` L91, L680-685, L701-704 |

### Fix-4: Handover Guard 回归 bug（Phase-2 引入）

| 项目 | 内容 |
|------|------|
| **症状** | 15ep v1 回归测试中仍有 OOB 终止，分析发现 docking 阶段也出现高速倒车 |
| **根因** | Phase-2 新增的 handover guard（`align_plan→docking` 交接时 `if dist > align_x_max_abs - 0.10` 强制切 retreat），绕过了所有减速/距离保护，导致全速倒车直接越界 |
| **修复** | 删除整个 handover guard 代码块，恢复 docking PD 控制器正常接管。PD 控制器的比例增益自然会在 dist 较大时输出正向 drive，将 forklift 拉回 |
| **文件** | `expert_policy.py`（删除原 L902-905） |

---

## 2. 其他改进（Phase-2 遗留，本次保留）

| 改进 | 内容 |
|------|------|
| **Planner 启发式收紧** | 后退惩罚阈值 0.5→0.3，权重 0.3→0.8，减少不必要的后退 |
| **Planner reverse_penalty** | 1.5→2.5，增大逆行搜索代价 |
| **align_rev_drive** | -0.35→-0.25，限制逆行位移 |
| **align_x_max_abs** | 3.0→2.85，预留 0.15m 安全余量 |
| **align_x_headroom** | 0.6→0.3，收紧 planner 逆行搜索边界 |
| **Event-driven 重规划** | 替代固定频率重规划，基于计划完成、bbox_abort、drift 三类事件触发 |
| **Drift 监测** | 定期比较实际位姿与规划路径，偏差超阈值立即重规划 |
| **Planner 快照日志** | 每次重规划输出起终点、expansion 数、primitive 列表，方便调试 |

---

## 3. 验证结果

### 3.1 快速回归 (15ep: seed 42/88/123 × 5)

| 指标 | Phase-2 基线 | Phase-2b v2 |
|------|-------------|-------------|
| Terminated | 37% | **0%** |
| Retreat max dist | >3.0 | 2.48 |
| 无新异常 | — | ✓ |

### 3.2 广泛验证 (50ep: 10 seed × 5)

| 指标 | 值 |
|------|-----|
| **Terminated (dist>3.0)** | **0/50 (0%)** |
| Truncated (超时) | 50/50 (100%) |
| Inserted (ins≥0.1) | 0/50 |
| Lift triggered | 0/50 |

| 统计量 | dist (m) | \|lat\| (m) |
|--------|----------|------------|
| min | 1.905 | 0.022 |
| mean | 2.260 | 0.437 |
| max | 2.537 | 2.163 |

| 结束阶段 | 数量 | 占比 |
|----------|------|------|
| docking | 34 | 68% |
| retreat | 16 | 32% |
| lift | 0 | 0% |

Retreat 最大 dist = 2.498m，远低于 3.0m OOB 限制。

---

## 4. 已解决 vs 未解决

### ✓ 已解决

1. **OOB 终止完全消除**：50ep 中 terminated = 0（37% → 0%）
2. **Retreat 安全可控**：三重防护生效，max_dist = 2.498
3. **Planner 曲率物理一致**：kappa 正确反映 `steer_angle_rad=0.6`
4. **Handover guard 回归修复**：删除有害代码

### ✗ 未解决（非 bug，属调优范畴）

- **Docking 控制器未能将 dist 推至 fork_reach (~1.87m) 以内**
  - 平均 dist=2.26m，min=1.905m，始终不够近
  - 导致 insert_norm 恒为 0，lift 无法触发
  - 根因：docking PD 控制器的 gain/切换逻辑需要调优

---

## 5. 下一步建议（Tuning Batch）

1. **增强 docking 前推**：在 `|lat|<0.3` 时提高 drive gain，使 forklift 更积极地向 pallet 推进
2. **评估 planner 修复效果**：Fix-2 修正 kappa 后，planner 的对齐质量应有改善，可能减少 docking 阶段的横向偏差
3. **微调 retreat 策略**：当前 32% 的 episode 以 retreat 结束，表明 retreat 触发条件可能偏敏感
4. **增加 episode 步数上限**：当前 1078 步可能不足以完成完整的 align→dock→insert→lift 流程

---

## 6. 变更文件清单

| 文件 | 行数 | 变更摘要 |
|------|------|---------|
| `forklift_expert/expert_policy.py` | 984 | +96 -21：4 个 fix + drift 监测 + event-driven replan + 快照日志 |
| `forklift_expert/align_lattice_planner_s10u.py` | 381 | +8 -3：steer_angle_rad 参数 + kappa 修正 + 启发式收紧 |

---

## 7. 交付物清单

压缩包 `s1.0v_phase2b_delivery.tar.gz` 包含：

```
s1.0v_phase2b_delivery/
├── summary.md                           # 本文件
├── results.md                           # 详细验证数据报告
├── code/
│   ├── expert_policy.py                 # 修复后的完整源码
│   ├── align_lattice_planner_s10u.py    # 修复后的完整源码
│   └── play_expert.py                   # 调试脚本（含快照日志）
├── patch/
│   └── s1.0v_phase2b.patch              # git diff (方便 review)
├── logs/
│   ├── 15ep_v2/                         # 快速回归日志
│   │   ├── progress.log
│   │   ├── v2_s42.log / v2_s88.log / v2_s123.log
│   │   └── v2_f42.log / v2_f88.log / v2_f123.log
│   └── 50ep/                            # 广泛验证日志
│       ├── progress.log
│       ├── s{0,7,13,42,55,88,99,123,177,200}.log
│       └── f{0,7,13,42,55,88,99,123,177,200}.log
```
