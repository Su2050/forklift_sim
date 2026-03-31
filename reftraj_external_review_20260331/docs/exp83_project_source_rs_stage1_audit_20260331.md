# Exp8.3 项目源 RS 参考轨迹接入与 Stage1 审计

日期：2026-03-31

## 这次改了什么

本次改动不再直接改 IsaacLab 工作副本，而是改项目源：

- patch 源环境：[env.py](/home/uniubi/projects/forklift_sim/forklift_pallet_insert_lift_project/isaaclab_patch/source/isaaclab_tasks/isaaclab_tasks/direct/forklift_pallet_insert_lift/env.py)
- patch 源配置：[env_cfg.py](/home/uniubi/projects/forklift_sim/forklift_pallet_insert_lift_project/isaaclab_patch/source/isaaclab_tasks/isaaclab_tasks/direct/forklift_pallet_insert_lift/env_cfg.py)
- patch 源内 vendored RS 实现：[rs.py](/home/uniubi/projects/forklift_sim/forklift_pallet_insert_lift_project/isaaclab_patch/source/isaaclab_tasks/isaaclab_tasks/direct/forklift_pallet_insert_lift/rs/rs.py)
- 轨迹批量可视化脚本：[visualize_reference_trajectory_cases.py](/home/uniubi/projects/forklift_sim/forklift_pallet_insert_lift_project/scripts/visualize_reference_trajectory_cases.py)

关键点：

- 任务代码已经支持 `traj_model = "rs_exact"`
- 参考轨迹是在 `vehicle/root pose` 上做 RS，再映射成 `fork_center path`
- `visualize_reference_trajectory_cases.py` 现在支持 `--traj-model rs_exact`
- 审计采样已经改成按当前 stage1 初始化范围跑 `5 x 5 x 5 = 125` 个 case

## 初始化范围

这次先把项目源里的 stage1 初始化范围同步成当前训练口径：

- `x ∈ [-3.60, -3.45]`
- `y ∈ [-0.15, +0.15]`
- `yaw ∈ [-6°, +6°]`

## RS 参数

当前 RS 审计使用：

- `traj_rs_min_turn_radius_m = 2.34`
- `traj_rs_sample_step_m = 0.05`

这里的 `2.34m` 是按项目里已有的 Ackermann 口径取的近似物理值：

- `wheelbase ≈ 1.6m`
- `steer_angle_rad ≈ 0.6`
- `R_min ≈ wheelbase / tan(0.6) ≈ 2.34m`

## 批量审计命令

```bash
python forklift_pallet_insert_lift_project/scripts/visualize_reference_trajectory_cases.py \
  --traj-model rs_exact \
  --grid-count-x 5 \
  --grid-count-y 5 \
  --grid-count-yaw 5
```

## 输出

- overlay: [overlay_all_cases.png](/home/uniubi/projects/forklift_sim/forklift_pallet_insert_lift_project/outputs/reference_trajectory_stage1_viz_rs_exact/overlay_all_cases.png)
- manifest: [reference_trajectory_stage1_manifest.json](/home/uniubi/projects/forklift_sim/forklift_pallet_insert_lift_project/outputs/reference_trajectory_stage1_viz_rs_exact/reference_trajectory_stage1_manifest.json)

代表 case：

- 极端 heading change: [c103_xm3p450_ym0p150_yawp0p000.png](/home/uniubi/projects/forklift_sim/forklift_pallet_insert_lift_project/outputs/reference_trajectory_stage1_viz_rs_exact/c103_xm3p450_ym0p150_yawp0p000.png)
- 极端 curvature: [c51_xm3p525_ym0p150_yawm6p000.png](/home/uniubi/projects/forklift_sim/forklift_pallet_insert_lift_project/outputs/reference_trajectory_stage1_viz_rs_exact/c51_xm3p525_ym0p150_yawm6p000.png)

## 结果

入口几何层面：

- `entry_ok = 125 / 125`
- `delta_s_min = -0.3570`
- `delta_s_max = -0.3500`

也就是说，RS 版本至少没有再出现 “`s_start >= s_pre`” 这种入口站位错误。

但更深一层的几何质量并不好：

- `root_y_abs_max_max = 0.1507m`
- `root_heading_change_deg_max = 178.42deg`
- `root_curvature_max_max = 229.75 1/m`

这说明：

- shortest-RS 在当前 near-field reset 上，会给出明显折返/打结的局部最短路
- 这些轨迹虽然“入口关系合法”，但并不适合直接作为训练参考轨迹

## 结论

结论很明确：

1. `RS support` 已经成功接入项目源代码。
2. 但在当前 `±0.15m / ±6deg` 的 stage1 near-field 课程上，**shortest exact RS 不能直接作为默认训练参考轨迹**。
3. 所以当前配置里不应把 `traj_model` 直接常驻切成 `rs_exact`。

## 当前处理策略

为了避免把正式训练直接带进坏轨迹：

- 代码层已经支持 `traj_model = "rs_exact"`
- 但 patch 源默认值先保留 `traj_model = "root_path_first"`
- RS 只作为显式 audit / ablation / override 使用

## 下一步建议

下一步不建议直接 install 并开训 shortest-RS。

更合理的是二选一：

1. 做 `forward-preferred RS`
   - 不是取 exact shortest 一条
   - 而是在前几条最短 RS 中按 `reverse length / direction switches` 重排序
2. 调整 near-field goal / curriculum
   - 当前 goal 离 start 太近
   - 对于物理 `R_min ≈ 2.34m` 的车辆，天然容易落到折返解

在这两条里，我更建议先做第 1 条，因为它仍然保留“RS family”这个核心要求，同时比直接缩小物理半径更自洽。
