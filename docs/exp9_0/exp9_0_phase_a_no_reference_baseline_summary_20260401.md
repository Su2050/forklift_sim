# Exp9.0 Phase A No-Reference Baseline Summary

日期：`2026-04-01`

本页汇总 `Phase A` 的 `A` 组结果：

- 训练设置：连续初始分布 + no-reference
- seed：`42 / 43 / 44`
- 训练步数：每个 seed `400` iterations

## 1. 结果总表

| Seed | Final Iter | Time Elapsed | Mean Reward | frac_inserted | frac_inserted_push_free | frac_aligned | frac_hold_entry | frac_success | frac_success_strict | frac_push_free_success | center_lateral_mean | yaw_deg_mean | pallet_disp_xy_mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | 399 | 03:37:44 | 4299.37 | 0.5781 | 0.1406 | 0.0156 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.4442 | 18.0551 | 1.0533 |
| 43 | 399 | 03:34:12 | 2997.74 | 0.7344 | 0.1094 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.3881 | 16.8089 | 1.1195 |
| 44 | 399 | 03:35:16 | 3766.69 | 0.7188 | 0.1406 | 0.0469 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.4141 | 13.1965 | 1.2188 |
| Mean | - | 03:35:44 | 3687.93 | 0.6771 | 0.1302 | 0.0208 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.4155 | 16.0202 | 1.1305 |

## 2. 结论

1. 三个 seed 在 `400` iterations 结束时，`phase/frac_success`、`phase/frac_success_strict`、`phase/frac_push_free_success` 全部为 `0.0000`。
2. `phase/frac_inserted` 已经达到 `0.5781 ~ 0.7344`，说明模型并不是“完全插不进去”；主要瓶颈更像是对齐与保持 success gate，而不是插入本身。
3. `phase/frac_aligned` 仍然很低，三 seed 末轮只有 `0.0000 / 0.0156 / 0.0469`；`phase/frac_hold_entry` 也全部为 `0.0000`，说明还没有形成稳定 success 闭环。
4. 从均值看，末轮 `err/center_lateral_mean = 0.4155`、`err/yaw_deg_mean = 16.0202`、`diag/pallet_disp_xy_mean = 1.1305`，当前对齐质量距离成功门槛仍有明显差距。

## 3. 判读

- Phase A 已完成，可以作为后续 `B / C / D` 组对照基线。
- 当前 `A` 组的核心结论是：`master` 初始分布 + no-reference 在这套设置下能学到一定插入，但不能稳定转化为 success。
- 因此后续如果 `B` 组（离散 case + no-reference）或 `C / D` 组（离线 reference）能把 success 从 `0` 抬起来，就会非常有辨识度。

## 4. 对应运行目录

- `seed42`: `/home/uniubi/projects/forklift_sim/IsaacLab/logs/rsl_rl/forklift_pallet_insert_lift/2026-03-31_18-51-50_exp9_0_no_reference_master_init_seed42_iter400`
- `seed43`: `/home/uniubi/projects/forklift_sim/IsaacLab/logs/rsl_rl/forklift_pallet_insert_lift/2026-03-31_22-30-09_exp9_0_no_reference_master_init_seed43_iter400`
- `seed44`: `/home/uniubi/projects/forklift_sim/IsaacLab/logs/rsl_rl/forklift_pallet_insert_lift/2026-04-01_02-05-09_exp9_0_no_reference_master_init_seed44_iter400`
