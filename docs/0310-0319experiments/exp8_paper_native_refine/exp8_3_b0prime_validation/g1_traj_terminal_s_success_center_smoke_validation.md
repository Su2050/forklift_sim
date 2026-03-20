# Exp8.3 G1 smoke 验证（trajectory terminal geometry package -> `s_success_center`）

**日期**：2026-03-20  
**分支**：`exp/exp8_3_geom_validation_b0prime`  
**git rev**：`d741c915257b47fda9527802bd06df3bbabe7337`  
**日志**：`logs/20260320_154622_train_exp8_3_g1_traj_terminal_s_success_center_smoke.log`  
**对照日志**：`logs/20260320_101115_train_exp8_3_b0prime_smoke.log`  
**IsaacLab run 目录**：`IsaacLab/logs/rsl_rl/forklift_pallet_insert_lift/2026-03-20_15-46-31_exp8_3_g1_traj_terminal_s_success_center_smoke`

## 改动定义

- **实验名**：`G1`
- **干预类型**：`trajectory-only`
- **唯一改动**：把 trajectory terminal geometry package 从 `front` 改到 `s_success_center`
- **实现方式**：
  - 新增 `env.exp83_traj_goal_mode`
  - `B0′` 基线使用 `front`
  - `G1` smoke 使用 `success_center`
- **本次 smoke CLI 关键 override**：
  - `env.exp83_traj_goal_mode=success_center`
  - 其余 `num_envs / camera / backbone / run style` 与 `B0′ smoke` 保持一致

## smoke 通过性结论

按“代码 smoke 是否通过”的标准，本次 **通过**：

- 无 `Traceback` / `RuntimeError` / `Killed`
- 训练正常进入 `iter 79/80`
- 新日志字段持续输出
- 关键常量已按预期切换：
  - `geom/s_traj_end = -0.8160`
  - `geom/s_success_center = -0.8160`
  - `geom/s_rd_target = -0.4800`

补充说明：

- 日志末尾未看到 `Closing sim app` 收尾行，但训练进程已退出、且无异常报错。
- 因此按本轮约定，仍记为 **smoke 已通过**。

## 与 `B0′ smoke` 的同 horizon 对照

| iter | run | `geom/s_traj_end` | `err/dist_front_mean` | `traj/d_traj_mean` | `traj/yaw_traj_deg_mean` | `err/yaw_deg_mean` | `diag/pallet_disp_xy_mean` | `s_center_mean` | `phase/frac_inserted` | `phase/frac_rg` | `phase/frac_success` |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | `B0′ smoke` | -1.0800 | 1.9670 | 0.0406 | 5.9204 | 8.3020 | 0.0000 | -3.6393 | 0.0000 | 0.0000 | 0.0000 |
| 0 | `G1 smoke` | -0.8160 | 1.9665 | 0.0476 | 4.9925 | 8.2953 | 0.0000 | -3.6388 | 0.0000 | 0.0000 | 0.0000 |
| 20 | `B0′ smoke` | -1.0800 | 1.3606 | 0.3130 | 16.4033 | 11.7217 | 0.0556 | -3.0121 | 0.0000 | 0.0000 | 0.0000 |
| 20 | `G1 smoke` | -0.8160 | 0.8856 | 0.3123 | 17.9689 | 8.0678 | 0.0099 | -2.5566 | 0.0000 | 0.0000 | 0.0000 |
| 40 | `B0′ smoke` | -1.0800 | 2.9438 | 0.9247 | 3.3841 | 8.0379 | 0.0000 | -4.6156 | 0.0000 | 0.0000 | 0.0000 |
| 40 | `G1 smoke` | -0.8160 | 0.5886 | 0.4642 | 15.1091 | 14.8735 | 0.1956 | -2.1306 | 0.0469 | 0.0000 | 0.0000 |
| 79 | `B0′ smoke` | -1.0800 | 2.7846 | 0.7813 | 2.8648 | 7.9078 | 0.0000 | -4.4567 | 0.0000 | 0.0000 | 0.0000 |
| 79 | `G1 smoke` | -0.8160 | 0.6539 | 0.4182 | 15.3546 | 10.8557 | 0.1736 | -2.1952 | 0.0156 | 0.0000 | 0.0000 |

## 直接观察

1. **`G1` 明显改变了策略的“是否往入口推进”。**  
   相比 `B0′ smoke`，`G1` 在 `iter 20/40/79` 的 `err/dist_front_mean` 都显著更低，`s_center_mean` 也显著更靠近托盘前沿。这说明“轨迹终点过浅”不是边角因素，而是能直接影响 early behavior 的主变量之一。

2. **`G1` 已经出现了非零插入信号。**  
   `B0′ smoke` 的 `phase/frac_inserted` 全程为 `0.0000`；而 `G1 smoke` 从中段开始多次出现非零，峰值达到 `0.0625`。这说明只改 trajectory terminal geometry package，就足以把策略从“远场不进门”推到“开始触达插入态”。

3. **但 `G1` 不是“干净解决”，而是“更能进去，但质量还不够好”。**  
   到中后段，`G1` 的：
   - `traj/yaw_traj_deg_mean` 升到约 `14-16°`
   - `err/yaw_deg_mean` 升到约 `11-15°`
   - `diag/pallet_disp_xy_mean` 升到约 `0.15-0.20m`
   - `err/center_lateral_mean` / `err/tip_lateral_mean` 明显偏大

   这说明它虽然更愿意往里走，但同时也更容易带着偏航和推盘进入近场。

4. **`G1` 还没有触发更高层的成功语义。**  
   全程仍未出现非零的：
   - `phase/frac_rg`
   - `phase/frac_success`

   因此 `G1` 当前回答的是“轨迹终点浅是不是重要矛盾”，而不是“只改轨迹终点就已经足够达成任务”。

## 机制判断

本次 `G1 smoke` 支持如下判断：

- **支持**：trajectory terminal geometry package 过浅，确实是 `B0′` 长期停留在远场 approach 的重要原因之一。
- **不支持**：只改 trajectory terminal geometry package 就能独立解决 8.3 的全部问题。
- **新增现象**：当轨迹终点推进到 `s_success_center` 后，策略会更积极地接近并发生部分插入，但也更容易伴随偏航增大和托盘位移增大。

## 一句话结论

**`G1 smoke` 通过了代码/接线层 smoke 检查，并提供了第一条强机制证据：把 trajectory terminal geometry package 推到 `s_success_center` 后，策略不再像 `B0′` 那样长期卡在远场，而是开始出现接近入口与非零插入；但当前仍伴随明显偏航与推盘，因此 `G1` 是“有效推进但未闭环”的正向结果。**

## 对下一步的含义

- 可以按“smoke 已通过”标准提交当前 `G1` 改动
- 后续应继续按计划推进：
  - `G1-400 iter`：确认这条趋势在 `400 iter` 是否持续
  - `G2`：只改 `r_d` 目标点
  - `G3`：统一 `p_goal + target_center family`

当前不建议跳过 `G2 / G3`，直接把 `G1` 当最终方案，因为 smoke 已显示其副作用（偏航/推盘）并不小
