# Exp8.3 Runtime U1 Target-Center Family 运行态验证记录

**日期**：2026-03-23  
**目的**：在真实 env reset 路径上验证 `r_d / rg / done侧 out_of_bounds` 的 `target_center family` 接线是否与配置一致，并确认 `G2b / G3` 的几何关系是否就是我们怀疑的那样。

## 代码落点

- 主工作树 patch 源：`forklift_pallet_insert_lift_project/isaaclab_patch/source/isaaclab_tasks/isaaclab_tasks/direct/forklift_pallet_insert_lift/env.py`
- 配置项：`forklift_pallet_insert_lift_project/isaaclab_patch/source/isaaclab_tasks/isaaclab_tasks/direct/forklift_pallet_insert_lift/env_cfg.py`
- sanity 入口脚本：`scripts/run_exp8_3_target_center_family_u1_sanity.sh`

## 检查口径

runtime U1 在 `_reset_idx()` 中、`runtime U0` 之后执行，固定对每个 env 构造 4 类 probe：

1. `family_center`
   - 当前 `target_center family` 对应的 `fork_center` 目标点
   - 期望：`dist_center_family = 0`，`r_d = reward_max`，`rg = true`，`out_of_bounds = false`
2. `alternate_center`
   - 与当前 family 相反的另一组中心
   - 期望：到当前 family 的距离等于 `|s_alt - s_family|`
3. `traj_goal_center`
   - 当前 `traj_goal_mode` 对应的轨迹终点
   - 期望：到当前 family 的距离等于 `|s_traj - s_family|`
   - 同时记录 `traj_goal_inside_rg`
4. `oob_center`
   - 沿托盘轴在当前 family 中心外推 `paper_out_of_bounds_dist + margin`
   - 期望：`out_of_bounds = true`

本轮固定容差：

- `eps_m = 0.001 m`
- `probe_margin_m = 0.02 m`
- `fail_fast = true`

## 执行记录

> 注：这轮 U1 为前台运行，关键结论来自控制台摘要；IsaacLab run 目录如下。

| profile | 分支 | git rev | IsaacLab run 目录 | `runtime_u1` 结果 |
| --- | --- | --- | --- | --- |
| `b0prime` | `exp/exp8_3_geom_validation_b0prime` | `c6484b49` | `IsaacLab/logs/rsl_rl/forklift_pallet_insert_lift/2026-03-23_12-05-43_exp8_3_target_center_family_u1_b0prime_fg` | `fail=0/32`, `mode=front_center`, `delta_traj=0.6000`, `traj_goal_inside_rg=0` |
| `g2b` | `exp/exp8_3_geom_validation_b0prime` | `c6484b49` | `IsaacLab/logs/rsl_rl/forklift_pallet_insert_lift/2026-03-23_12-11-47_exp8_3_target_center_family_u1_g2b_fg` | `fail=0/32`, `mode=success_center`, `delta_traj=0.2640`, `traj_goal_inside_rg=1` |
| `g3` | `exp/exp8_3_geom_validation_b0prime` | `c6484b49` | `IsaacLab/logs/rsl_rl/forklift_pallet_insert_lift/2026-03-23_12-16-19_exp8_3_target_center_family_u1_g3_fg` | `fail=0/32`, `mode=success_center`, `delta_traj=0.0000`, `traj_goal_inside_rg=1` |

## 关键几何摘要

| profile | `geom/s_rd_target` | `geom/s_traj_end` | `geom/s_success_center` | `delta_alt` | `delta_traj` | `traj_goal_inside_rg` | 直接读法 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `b0prime` | -0.4800 | -1.0800 | -0.8160 | 0.3360 | 0.6000 | 0 | `front_center` family 与 `front` traj goal 分离明显 |
| `g2b` | -0.8160 | -1.0800 | -0.8160 | 0.3360 | 0.2640 | 1 | `success_center` family 下，`front` traj goal 已落入 `rg` 阈值内 |
| `g3` | -0.8160 | -0.8160 | -0.8160 | 0.3360 | 0.0000 | 1 | traj goal 与 family target 完全重合 |

其中：

- `delta_alt = |s_front_center - s_success_center| = 0.336 m`
- `paper_rg_dist_thresh = 0.28 m`

因此：

- `G2b`: `delta_traj = 0.264 m < 0.28 m`
- `G3`: `delta_traj = 0.000 m < 0.28 m`
- `b0prime`: `delta_traj = 0.600 m > 0.28 m`

## 结论

1. **`r_d / rg / out_of_bounds` 的 `target_center family` 接线已在真实 env 路径上通过 runtime U1。**  
   三个 profile 全部 `fail=0/32`，说明当前主工作树中，reward 侧和 done 侧已经共用同一套 `target_center family` 几何。

2. **`G2b` 的 low-level 几何怀疑被正式坐实。**  
   在 `target_center family = success_center`、`traj_goal_mode = front` 时，`traj_goal` 到 family center 的距离只有 `0.264 m`，已经小于 `rg` 阈值 `0.28 m`。也就是说，`G2b` 的轨迹终点天然落在当前 `rg` 盆地里。

3. **`G3` 则不是“接近”，而是“完全统一”。**  
   `G3` 的 `delta_traj = 0`，说明 trajectory terminal geometry package 与 `target_center family` 完全重合；它不只是比 `G2b` 更深，而是已经取消了这层几何错位。

4. **`b0prime` 仍然是 clean control。**  
   `b0prime` 的 `traj_goal` 与 `front_center family` 相距 `0.600 m`，明显不在 `rg` 阈值内，因此它不带 `G2b/G3` 这类“traj goal 本身靠近 family reward basin”的结构性耦合。

## 对 confirm800 的含义

这轮 U1 不能单独解释 `G2b` 为什么在 `800 iter` 后段塌成零推进，但它已经证明：

- `G2b` 不是一个“trajectory goal 与 reward/done center 彼此独立”的方案。
- `G2b` 的 `front` traj goal 实际上已经进入 `success_center family` 的近场奖励区。
- 因此 `G2b` 的行为塌缩不能再简单归因成“目标还不够深”；更合理的读法是：**当前这组几何已经把策略推向一个容易停在近场盆地里的局部最优。**

## 下一步建议

1. **不再把 `G2b` 作为首选主线继续叠机制。**  
   U1 已经说明它的 low-level 几何本身带有近场盆地重叠；继续在其上叠 `G4` 会让问题来源更混。

2. **下一轮主线应切到 `G3` 基座。**  
   原因不是 `G3` 已经成功，而是它至少消除了 `traj_goal` 与 `target_center family` 的这层几何错位。

3. **在 `G3` 基座上补一轮 stricter success/near-field 诊断。**  
   当前更值得补的是：
   - `frac_inserted_z_valid`
   - `frac_success_strict`
   - `frac_push_free_success`
   - `max_hold_counter`
   - tip / fork_center 口径的近场对齐指标

## 一句话结论

**runtime U1 已经把 low-level 几何关系钉实：`G2b` 的 `front` trajectory goal 确实落在 `success_center family` 的 `rg` 盆地内，而 `G3` 则把 trajectory goal 与 family target 完全统一；因此下一步不应继续把 `G2b` 作为主候选，而应转向 `G3` 基座并补 stricter success 诊断。**
