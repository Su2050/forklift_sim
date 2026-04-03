# Exp9.0 O3 训练结果总结

**日志**: `20260402_093917_train_exp9_0_no_reference_rewardfix_o3_seed42.log`
**最终迭代**: 1999/2000

## 1. 终态指标 (Last 50 均值)

| Metric | Last 50 Mean |
|---|---|
| `phase/frac_inserted` | 0.5809 |
| `phase/frac_aligned` | 0.0500 |
| `phase/frac_tip_constraint_ok` | 0.1369 |
| `phase/frac_hold_entry` | 0.0006 |
| `phase/frac_success` | 0.0003 |
| `err/center_lateral_inserted_mean` | 0.3791 |
| `err/tip_lateral_inserted_mean` | 0.3841 |
| `err/yaw_deg_inserted_mean` | 6.1275 |
| `paper_reward/r_postinsert_align` | 1.4605 |
| `paper_reward/R_plus` | 5.4161 |
| `paper_reward/r_d` | 0.7871 |
| `diag/max_hold_counter` | 0.4740 |

## 2. 阶段趋势

| Metric | 0-500 | 500-1000 | 1000-1500 | 1500-2000 |
|---|---|---|---|---|
| `phase/frac_inserted` | 0.5691 | 0.5627 | 0.5852 | 0.5839 |
| `phase/frac_aligned` | 0.0441 | 0.0456 | 0.0547 | 0.0514 |
| `err/center_lateral_inserted_mean` | 0.3832 | 0.3822 | 0.3763 | 0.3754 |
| `err/tip_lateral_inserted_mean` | 0.3880 | 0.3942 | 0.3911 | 0.3902 |
| `err/yaw_deg_inserted_mean` | 6.8006 | 7.3365 | 7.1544 | 7.3684 |
| `diag/max_hold_counter` | 1.1064 | 1.0242 | 1.2317 | 0.9550 |
| `paper_reward/r_postinsert_align` | 1.4184 | 1.3934 | 1.4685 | 1.4677 |

**Success > 0 迭代数**: 41/2000
