# Exp8.3 B0 -> B0′ build-order 早窗验证（iter 0-50）

**日期**：2026-03-20  
**分支**：`exp/exp8_3_geom_validation_b0prime`  
**对照日志**：
- `B0`：`logs/20260319_215340_train_exp8_3_fork_center_traj.log`
- `B0′ smoke`：`logs/20260320_101115_train_exp8_3_b0prime_smoke.log`

## 对比口径

- 按验证计划，`B0 -> B0′` 只使用 **共享旧指标 + 早期 traj 指标** 做硬比较。
- `B0` 历史日志没有 `err/center_lateral_mean`、`s_center_mean` 等新字段，因此这些字段只用于确认 `B0′` 日志接线成功，不参与 `B0 -> B0′` 主结论。
- 当前仅为 **single-seed early-window** 结论，不代表几何问题已经解决。

## 核心对比

| iter | `B0 traj/d_traj_mean` | `B0′ traj/d_traj_mean` | `B0 yaw_deg_mean` | `B0′ yaw_deg_mean` | `B0 pallet_disp` | `B0′ pallet_disp` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 17.3824 | 0.0406 | 8.2993 | 8.3020 | 0.0000 | 0.0000 |
| 1 | 16.7440 | 0.0552 | 7.8427 | 7.8961 | 0.0000 | 0.0000 |
| 10 | 6.8264 | 0.4233 | 7.1023 | 9.9129 | 0.0242 | 0.1047 |
| 20 | 0.8322 | 0.3130 | 10.1468 | 11.7217 | 0.0589 | 0.0556 |
| 50 | 0.9279 | 1.0202 | 7.7794 | 8.5502 | 0.0369 | 0.0000 |

## 结论

1. **B0′ 明确修掉了 build-order 造成的早期轨迹污染。**  
   最直接的证据是 `traj/d_traj_mean` 起点量级从 `17.3824` 降到 `0.0406`，且 `iter 1` 仍只有 `0.0552`。这与“参考轨迹已基于当前 episode reset 张量重建”一致。

2. **因此可以确认：`_build_reference_trajectory()` 的调用时机 bug 确实会污染 `r_cd / r_cpsi` 主路径。**  
   这一步已经足够支持计划里的 `B0 -> B0′` 专用判断，后续几何实验应以 `B0′` 作为新基线。

3. **但 B0′ 并不等于“整体几何质量自动更优”。**  
   在 `iter 10-20` 区间，`B0′` 的 `yaw_deg_mean` 与 `pallet_disp_xy_mean` 一度高于 `B0`，说明修掉错轨迹之后，策略不再被旧轨迹误导，但也没有自然变成更好的 approach / insert 策略。

4. **到 `iter 50` 时，B0′ 仍未出现成功相关信号。**  
   `phase/frac_success = 0.0000`、`diag/success_term_frac = 0.0000`。这说明 `B0′` 的成立结论是“前置 bug 修复有效”，而不是“任务已经学会”。

## B0′ 日志接线检查

`B0′ smoke` 已确认以下计划内字段进入 stdout：

- `geom/s_traj_end`
- `geom/s_rd_target`
- `geom/s_success_center`
- `err/root_lateral_mean`
- `err/center_lateral_mean`
- `err/tip_lateral_mean`
- `s_center_mean`
- `s_tip_mean`
- `phase/frac_rg`
- `phase/frac_success`
- `diag/out_of_bounds_frac`
- `diag/success_term_frac`

## 下一步

- 已启动 `B0′-400 iter` 正式基线：`logs/20260320_111306_train_exp8_3_b0prime_baseline.log`
- 待 `B0′-400` 跑完后，按计划进入第一轮主矩阵：`G1 / G2 / G3`
