# Exp8.3 Runtime U0 运行态验证记录

**日期**：2026-03-22  
**目的**：在真实 env reset 路径上验证 `fresh reset tensors -> _build_reference_trajectory -> _query_reference_trajectory` 接线已闭环，避免仅依赖 pure-torch 预检。

## 代码落点

- 主工作树 patch 源：`forklift_pallet_insert_lift_project/isaaclab_patch/source/isaaclab_tasks/isaaclab_tasks/direct/forklift_pallet_insert_lift/env.py`
- 配置项：`forklift_pallet_insert_lift_project/isaaclab_patch/source/isaaclab_tasks/isaaclab_tasks/direct/forklift_pallet_insert_lift/env_cfg.py`
- sanity 入口脚本：`scripts/run_exp8_3_runtime_u0_sanity.sh`

## 检查口径

runtime U0 在 `_reset_idx()` 中、`_build_reference_trajectory()` 之后执行，检查 4 项：

1. `traj_pts[0]` 与当前 reset 的 `fork_center` 对齐
2. `traj_pts[-1]` 与当前 `_exp83_traj_goal_s()` 对应 `p_goal` 对齐
3. reset 后对当前 pose 立刻查询轨迹，`d_traj` 近 0
4. 轨迹起点切线 yaw 与 `robot_yaw` 在离散容差内一致

本轮固定容差：

- `eps_pos = 0.001 m`
- `eps_yaw_deg = 15.0 deg`
- `fail_fast = true`

## 运行记录

| profile | 分支 | git rev | 日志 | runtime_u0 结果 | 训练收尾 | Traceback |
| --- | --- | --- | --- | --- | --- | --- |
| `b0prime` | `exp/exp8_3_geom_validation_b0prime` | `12bd9374` | `logs/20260322_093124_sanity_check_exp8_3_runtime_u0_b0prime.log` | `fail=0/32`, `max_d_start=0`, `max_d_end=0`, `max_d_traj=0`, `max_yaw_deg=5.6017` | `iter 1/2` | 无 |
| `g2b` | `exp/exp8_3_g2b_target_family_success_center` | `c4a7b279` | `forklift_sim_wt_g2b/logs/20260322_093317_sanity_check_exp8_3_runtime_u0_g2b.log` | `fail=0/32`, `max_d_start=0`, `max_d_end=0`, `max_d_traj=0`, `max_yaw_deg=5.6017` | `iter 1/2` | 无 |
| `g3` | `exp/exp8_3_g3_traj_and_target_success_center` | `cb6c34e0` | `forklift_sim_wt_g3/logs/20260322_093455_sanity_check_exp8_3_runtime_u0_g3.log` | `fail=0/32`, `max_d_start=0`, `max_d_end=0`, `max_d_traj=0`, `max_yaw_deg=4.9668` | `iter 1/2` | 无 |

> 注：`max_iterations=2` 的 runner 采用 0-based 记法，日志出现 `Learning iteration 1/2` 代表本次 sanity run 已跑到最终轮次。

## 结论

1. 三个执行分支（主工作树、`G2b`、`G3`）均在真实 env 路径上通过 runtime U0。
2. `fail=0/32` 且 `max_d_start / max_d_end / max_d_traj` 全为 `0`，说明轨迹起终点与当前 reset 张量严格一致，未观察到 episode 错位。
3. `max_yaw_deg` 在 `4.97~5.60 deg`，满足既定 `15 deg` 离散切线容差，和 pure-torch U0 口径一致。
4. 因此可放行进入 `confirm800` 长跑阶段（`G2b` 主候选 + `G3` 保守对照）。
