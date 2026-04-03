# Exp9.0 No-Reference Reward Override O1

日期：`2026-04-01`

## 1. 背景

本页记录对 `logs/20260401_151243_train_exp9_0_no_reference_master_init_seed42_iter400.log` 的复核结论，以及基于该结论启动的一轮最小 reward/config override 实验。

基线运行的核心现象不是“完全插不进去”，而是“经常 inserted，但很难稳定进入 hold/success”。

## 2. 复核结论

### 2.1 gate 链路

- `aligned` 判定是 `center_y_err <= 0.15m && yaw_err_deg <= 8deg`
- 当前 stage 1 的 `hold_entry` 不要求 lift，而是 `inserted && aligned && tip_entry`
- 因此真实瓶颈链路是：
  `inserted -> aligned(center + yaw) -> tip_entry -> hold_entry -> success`

### 2.2 日志统计

对 `iter 315-322` 复算得到：

- `phase/frac_inserted = 0.5469 ~ 0.6094`
- `phase/frac_aligned = 0.0000 ~ 0.0156`
- `phase/frac_tip_constraint_ok = 0.1406 ~ 0.2188`
- `err/center_lateral_inserted_mean = 0.3151 ~ 0.3514m`
- `err/tip_lateral_inserted_mean = 0.3100 ~ 0.3556m`
- `err/yaw_deg_inserted_mean = 6.9583 ~ 8.2942deg`

对最后 `50` 轮复算得到：

- `phase/frac_inserted = 0.5391`
- `phase/frac_aligned = 0.0425`
- `phase/frac_tip_constraint_ok = 0.1334`
- `phase/frac_hold_entry = 0.0025`
- `phase/frac_success = 0.0009`
- `diag/hold_exit_exceeded_frac = 0.9675`
- `err/center_lateral_inserted_mean = 0.4241m`
- `err/tip_lateral_inserted_mean = 0.4301m`
- `err/yaw_deg_inserted_mean = 6.3491deg`

补充统计：

- 最后 `50` 轮里，`yaw_deg_inserted_mean <= 8deg` 的轮数是 `45/50`
- 最后 `50` 轮里，`center_lateral_inserted_mean <= 0.15m` 的轮数是 `0/50`
- 最后 `50` 轮里，`tip_lateral_inserted_mean <= 0.12m` 的轮数是 `0/50`
- 全程 `phase/frac_success > 0` 的 iteration 有 `16` 次，说明策略已经能偶发成功，但远不稳定

### 2.3 结论

- 主瓶颈已经从“是否能插入”转移为“插入后能否把 center/tip 横向误差继续压进 hold 区”
- `yaw` 不是当前主阻塞，最多算次要约束
- reward 在当前配置下足以支撑一种“高 inserted、低 hold/success”的局部最优

## 3. Reward 侧判断

当前 no-reference 基线中：

- `use_reference_trajectory=false`
- `alpha_2=0`
- `alpha_3=0`
- `clean_insert_gate_floor=0.15`
- `clean_insert_push_free_bonus_enable=true`
- `clean_insert_dirty_penalty_enable=false`
- `preinsert_insert_frac_max=0.20`

最后 `50` 轮平均值：

- `paper_reward/R_plus = 5.2494`
- `paper_reward/R_minus = -0.6135`
- `paper_reward/r_d_raw = 2.2749`
- `paper_reward/r_d_clean_gate = 0.4227`
- `paper_reward/r_d = 1.0078`
- `paper_reward/r_clean_insert_bonus = 0.0844`
- `paper_reward/rg = 0.0025`
- `diag/clean_insert_gate_inserted_mean = 0.1507`
- `diag/clean_align_gate_inserted_mean = 0.0020`

这说明：

- post-insert 的 `r_d` 虽然被 clean gate 衰减，但仍主要靠 floor 在持续给正奖励
- `push_free bonus` 会继续鼓励“已经插入但还不够干净”的样本
- pre-insert 的连续纠偏 shaping 在 `insert_norm >= 0.20` 后基本退出，导致“已经插深但还不够准”的样本缺少继续横移纠偏的密集信号

## 4. O1 最小 override 方案

本轮只改 override，不改任务代码：

```bash
env.clean_insert_gate_floor=0.05
env.clean_insert_gate_start_frac=0.15
env.clean_insert_gate_ramp_frac=0.25
env.clean_insert_push_free_bonus_enable=false
env.preinsert_insert_frac_max=0.45
env.preinsert_y_err_delta_weight=2.0
env.preinsert_yaw_err_delta_weight=0.6
env.preinsert_dist_front_delta_weight=0.10
```

对应意图：

1. 更早、更强地收紧 post-insert `r_d`
2. 暂时去掉 `inserted && push_free` 这条过早正反馈
3. 把连续纠偏 shaping 延长到接近 success 深度，并把重点放在横向误差而不是继续前冲

## 5. 本轮启动

- run name：`exp9_0_no_reference_rewardfix_o1_seed42`
- seed：`42`
- num envs：`64`
- max iterations：`2000`
- 包装日志：`logs/20260401_163351_train_exp9_0_no_reference_rewardfix_o1_seed42.log`
- 类型：`train`

启动状态：

- 已进入 `Learning iteration 0/2000`
- 启动后首 `3` 个 iteration 已正常刷出汇总

早期 sanity snapshot：

- `paper_reward/R_plus = 3.3107 -> 2.9027 -> 3.1414`
- `phase/frac_inserted = 0.5625 -> 0.5938 -> 0.5938`
- `phase/frac_aligned` 已出现 `0.0625`
- `phase/frac_tip_constraint_ok` 已出现 `0.2188`
- `paper_reward/r_clean_insert_bonus = 0.0000`，符合本轮关闭 bonus 的预期
- `diag/clean_insert_gate_inserted_mean = 0.0547`，明显低于基线尾段约 `0.1507`

## 6. 预期观察点

优先观察以下指标是否比基线改善：

- `phase/frac_aligned`
- `phase/frac_tip_constraint_ok`
- `phase/frac_hold_entry`
- `phase/frac_success`
- `err/center_lateral_inserted_mean`
- `err/tip_lateral_inserted_mean`
- `paper_reward/R_plus`
- `diag/clean_insert_gate_inserted_mean`

如果这轮 override 有效，最先应该看到的不是 `success` 立刻大涨，而是：

1. `aligned` 和 `tip_constraint_ok` 先上升
2. `center/tip lateral inserted mean` 先下降
3. `hold_entry` 占比逐步从接近 `0` 抬起来
