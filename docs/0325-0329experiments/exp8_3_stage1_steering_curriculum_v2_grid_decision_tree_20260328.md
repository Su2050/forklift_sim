# Exp8.3 Stage1 Steering Curriculum v2 Grid 结果与“结果 -> 下一步动作”决策树

日期：2026-03-28

相关上游文档：

- [steer usage diagnostics 总结](/home/uniubi/projects/forklift_sim/docs/0325-0329experiments/exp8_3_steer_usage_diagnostics_summary_20260327.md)
- [stage1 steering curriculum v2 计划](/home/uniubi/projects/forklift_sim/docs/0325-0329experiments/exp8_3_stage1_steering_curriculum_v2_plan_20260327.md)
- [参考轨迹作用与可视化分析](/home/uniubi/projects/forklift_sim/docs/0325-0329experiments/exp8_3_trajectory_role_and_visualization_analysis_20260328.md)

本文件的目的不是再复述“训练有没有进展”，而是把这轮 `v2` 的完整 grid 结果整理成一份 **后续实验分流手册**。以后只要跑出类似的 `normal / zero-steer grid`，就可以直接按本文件里的决策树往下走。

---

## 1. 这轮 grid 为什么要跑

这轮 grid 的核心问题只有一个：

**`v2` 到底有没有让策略真正开始依赖 steering，而不是继续靠“往前叉”的捷径成功。**

因此我们对 `v2` 采用了两层闸门：

1. 训练闸门：`3 seeds x 50 iter` 后，至少 `2/3` seed 要出现稳定的 `push_free / hold / 非零 success`
2. steering 闸门：对同一批 checkpoint 做 `normal misalignment grid` 和 `zero-steer grid`  
   真正合格的配置应该满足：**`normal` 明显强于 `zero-steer`**

只有这两个条件都满足，才值得把该配置往 `200 iter` 甚至更长训练推进。

---

## 2. 这轮 v2 的训练先验结果

对应训练日志：

- [seed42](/home/uniubi/projects/forklift_sim/logs/20260327_183103_train_exp83_stage1_steering_curriculum_v2_seed42_iter50_256cam.log)
- [seed43](/home/uniubi/projects/forklift_sim/logs/20260327_185805_train_exp83_stage1_steering_curriculum_v2_seed43_iter50_256cam.log)
- [seed44](/home/uniubi/projects/forklift_sim/logs/20260327_192458_train_exp83_stage1_steering_curriculum_v2_seed44_iter50_256cam.log)

尾窗结果是：

| seed | frac_inserted | frac_inserted_push_free | frac_hold_entry | frac_success |
|---|---:|---:|---:|---:|
| 42 | 0.2188 | 0.0156 | 0.0156 | 0.0000 |
| 43 | 0.1562 | 0.0156 | 0.0156 | 0.0000 |
| 44 | 0.2500 | 0.0000 | 0.0000 | 0.0000 |

训练阶段已经说明：

- `v2` 比 `v1` 更平衡，没有明显塌成“纯 no-insert”
- 但它也没有在 tail 上真正打开 success
- 所以它本来就只处在“可以进 grid 验证 steering gap”的状态，还不够支持长训

---

## 3. 这轮 v2 grid 的完整结果

输出目录：

- [exp83_stage1_steering_curriculum_v2_grid](/home/uniubi/projects/forklift_sim/outputs/exp83_stage1_steering_curriculum_v2_grid)

6 条 summary 分别是：

- [seed42 normal](/home/uniubi/projects/forklift_sim/outputs/exp83_stage1_steering_curriculum_v2_grid/exp83_stage1_v2_seed42_iter50_normal_summary.json)
- [seed42 zero-steer](/home/uniubi/projects/forklift_sim/outputs/exp83_stage1_steering_curriculum_v2_grid/exp83_stage1_v2_seed42_iter50_zero_steer_summary.json)
- [seed43 normal](/home/uniubi/projects/forklift_sim/outputs/exp83_stage1_steering_curriculum_v2_grid/exp83_stage1_v2_seed43_iter50_normal_summary.json)
- [seed43 zero-steer](/home/uniubi/projects/forklift_sim/outputs/exp83_stage1_steering_curriculum_v2_grid/exp83_stage1_v2_seed43_iter50_zero_steer_summary.json)
- [seed44 normal](/home/uniubi/projects/forklift_sim/outputs/exp83_stage1_steering_curriculum_v2_grid/exp83_stage1_v2_seed44_iter50_normal_summary.json)
- [seed44 zero-steer](/home/uniubi/projects/forklift_sim/outputs/exp83_stage1_steering_curriculum_v2_grid/exp83_stage1_v2_seed44_iter50_zero_steer_summary.json)

### 3.1 汇总表

| case | success | inserted | inserted_push_free | hold | clean_ready | dirty | timeout | mean_max_pallet_disp_xy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| s42 normal | 0.4082 | 0.4898 | 0.3265 | 0.4082 | 0.3265 | 0.1837 | 0.5918 | 3.7895 |
| s42 zero | 0.3673 | 0.4694 | 0.3061 | 0.3673 | 0.3061 | 0.1633 | 0.6327 | 3.9498 |
| s43 normal | 0.3673 | 0.4694 | 0.2653 | 0.3673 | 0.2653 | 0.2245 | 0.6327 | 3.8930 |
| s43 zero | 0.3878 | 0.5102 | 0.3061 | 0.3878 | 0.3061 | 0.2449 | 0.6122 | 3.7421 |
| s44 normal | 0.3878 | 0.5510 | 0.2653 | 0.3878 | 0.2653 | 0.2857 | 0.6122 | 3.5647 |
| s44 zero | 0.3878 | 0.4694 | 0.3265 | 0.3878 | 0.3265 | 0.1633 | 0.6122 | 4.0032 |

### 3.2 直接结论

这轮最关键的结论非常明确：

**`normal` 没有明显强于 `zero-steer`。**

甚至在 `seed43` 和 `seed44` 上，`zero-steer` 局部指标还略强。

如果只看最关键的 `success_rate_ep`：

- `seed42`: `0.4082 -> 0.3673`
- `seed43`: `0.3673 -> 0.3878`
- `seed44`: `0.3878 -> 0.3878`

所以：

- `v2` 确实比旧 aligned 配置更“难”
- 但它还**没有**把 steering 变成一个真正必要的技能
- 当前 success basin 仍然没有被“normal-only steering”明显支撑起来

---

## 4. 本轮的正式判定

按计划里的硬标准，这轮 `v2` **不通过**。

### 条件 1：`normal` 明显强于 `zero-steer`

结果：**不满足**

解释：

- steering gap 很小
- 且 gap 不稳定、跨 seed 不一致
- 不能据此声称“策略已经真正学会用 steering”

### 条件 2：至少 `2/3` seed 出现稳定的 `push_free / hold / 非零 success`

结果：**不满足**

解释：

- `seed42/43` 只有弱的 `push_free / hold`
- 三条 run 的训练尾窗 `success` 都是 `0`
- 所以训练端也不足以支持大算力放大

### 最终判定

**当前 `stage1 steering curriculum v2` 还不值得往 `200 iter / 400 iter` 推。**

---

## 5. 现在能从这轮结果学到什么

### 5.1 学到的不是“v2 完全没用”

`v2` 仍然有两个正面信号：

- 相比 `v1`，它更平衡，没有明显塌成“纯 no-insert”
- 它在 grid 上能打出一定的 success，不是完全坏掉

所以它不是彻底失败，而是：

**“方向对了一半，但力度和切入方式还不够让 steering 成为主导技能。”**

### 5.2 真正暴露出来的问题

这轮结果最说明问题的一点是：

**现在的课程虽然已经比 aligned reset 更偏，但仍没有有效堵住“主要靠前推也能过一部分 grid”的捷径。**

换句话说：

- 我们已经把任务稍微变难了
- 但还没有把 “steering 必要性” 提升到足够高

这和之前的判断是一致的：

- 当前系统不是没有 trajectory / alignment 信息
- 而是这些信息还没有在 actor 的学习里变成“不可替代”的 steering skill

---

## 6. 结果 -> 下一步动作：决策树

下面是后续实验可以直接套用的分流规则。

### 分支 A：`normal` 明显强于 `zero-steer`，且 `2/3` seed 出现稳定 success

含义：

- steering gap 已经打开
- 训练端也已经能稳定接上 success

动作：

1. 直接保留当前配置
2. 推到 `200 iter`
3. `200 iter` 后做统一 eval + 新一轮 grid
4. 若仍稳定，再考虑 `300~400 iter`

### 分支 B：`normal` 明显强于 `zero-steer`，但 success 还弱

含义：

- steering 已经开始成为必要技能
- 但后半段 `clean insert / hold / success` 还没完全接上

动作：

1. 不大改 reset
2. 只做小幅后段 shaping 调整
3. 先推 `100~200 iter`
4. 再评估 success 是否被接通

### 分支 C：`normal` 和 `zero-steer` 差不多，但训练还有一些 success

含义：

- 课程变难了
- 但 steering 仍然不是决定性因素
- 当前依然存在“前推近似可行”的捷径

动作：

1. **不要**推长训
2. 进入下一版课程 `v3`
3. 重点继续扩大 steering 必要性，而不是继续扫 reward weight

这轮 `v2` 就属于 **分支 C**。

### 分支 D：`normal` 和 `zero-steer` 差不多，而且训练 success 也弱

含义：

- 当前课程既没逼出 steering
- 也没保住足够的任务可学性

动作：

1. 不继续这条配置
2. 回到更浅一层重新设计入口
3. 优先检查：
   - reset 是否让 steering 真正必要
   - pre-insert shaping 是否在奖励“先纠偏”而不是“继续前冲”
   - reference trajectory 入口几何本身是否足够清晰

---

## 7. 基于本轮结果，我建议的下一步

这轮 `v2` 已经告诉我们：

- 直接推长训，不值得
- 继续微调 `bonus weight`，也不是当前主问题

### 我建议的主线动作

1. **先做 runtime top-down 参考轨迹可视化**

原因：

- 现在最需要确认的是：轨迹本身在更大 `y / yaw` 偏差时，到底有没有提供清晰的 steering 几何
- 如果轨迹入口本身就不够强，那继续改 reward/curriculum 只会低效试错

相关脚本已经落地：

- [visualize_exp83_runtime_trajectory_topdown.py](/home/uniubi/projects/forklift_sim/scripts/visualize_exp83_runtime_trajectory_topdown.py)

2. 在可视化后，再决定 `v3` 应该往哪边改

如果 top-down 看起来：

- **轨迹本身很合理，但 agent 没用上**
  - 下一步优先改 actor 可用的 steering 信号
  - 例如更强的 pre-insert signed shaping，或更能逼 steering 的 reset

- **轨迹本身在起始段就几乎直插**
  - 下一步优先改 trajectory entry geometry
  - 不是先改 PPO 或 bonus

### 暂时不建议做的

- 直接把 `v2` 推到 `200/400 iter`
- 回 wide reset
- 再回去加重 dirty penalty / gate
- 再做一轮 bonus weight 扫参

---

## 8. 一句话收口

这轮 `v2 grid` 的正式结论是：

**`v2` 比 `v1` 更平衡，但还没有真正打开 steering gap；因此它还不能作为值得放大到几百 iter 的候选配置。**

按这轮结果，最合理的下一步不是长训，而是：

**先用 runtime top-down 轨迹可视化，确认参考轨迹本身在偏差更大时到底有没有提供足够强的 steering 几何。**
