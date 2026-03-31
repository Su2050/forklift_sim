# Exp8.3 Reference Trajectory Scope Audit

日期：2026-03-31

## 1. 这次在审什么

用户指出 [c19_xm3p450_ym0p080_yawm3p000.png](/home/uniubi/projects/forklift_sim/forklift_pallet_insert_lift_project/outputs/reference_trajectory_stage1_viz/c19_xm3p450_ym0p080_yawm3p000.png) 里的参考轨迹“明显不对”，因为它看起来没有考虑叉车的驱动轮、整车轮距/轴距、以及整车真实转弯方式。

这个质疑是对的，而且是核心问题。当前系统里：

- 参考轨迹决定了 `d_traj / yaw_traj_err / signed traj obs`
- reward 会用这些量把 agent 往“轨迹”上拉
- done / out_of_bounds 也部分依赖同一套 family/trajectory 几何

所以如果参考轨迹本身只是在“叉臂中心点层面”成立，而不是在“整车可行性层面”成立，那么后面很多 reward / steering 实验都可能建立在不完整前提上。

## 2. 结论先说

### 2.1 图本身不是画错了，问题是它画出来的对象太弱了

当前参考轨迹本质上是：

- 起点：`fork_center`
- 中间：`fork_center -> p_pre` 的三次 Hermite 样条
- 末段：`p_pre -> p_goal` 的直线插入

它验证的是：

- `s_start < s_pre < s_goal`

它**没有验证**：

- 整车 wheelbase / axle path
- 最小转弯半径
- 车体 sweep
- 驱动轮/后轮路径是否合理

所以这条轨迹更准确的名字不是“叉车参考轨迹”，而是：

**fork-center entry guide**

### 2.2 之前看的那套图默认还读了旧配置，不是当前训练配置

原始可视化脚本默认读的是：

- [forklift_pallet_insert_lift_project/isaaclab_patch/.../env_cfg.py](/home/uniubi/projects/forklift_sim/forklift_pallet_insert_lift_project/isaaclab_patch/source/isaaclab_tasks/isaaclab_tasks/direct/forklift_pallet_insert_lift/env_cfg.py)

不是当前训练正在用的：

- [IsaacLab/.../env_cfg.py](/home/uniubi/projects/forklift_sim/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/forklift_pallet_insert_lift/env_cfg.py)

这两者当前差异至少包括：

- legacy: `y = ±0.08`, `yaw = ±3°`
- current: `y = ±0.15`, `yaw = ±6°`
- current 还有 `stage1_steer_action_scale = 0.65`

也就是说，用户最开始看到的那套图，本来就不能完全代表当前训练状态。

## 3. 代码证据

### 3.1 参考轨迹构造只围绕 fork_center

训练环境里，参考轨迹构造在：

- [env.py:1009](/home/uniubi/projects/forklift_sim/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/forklift_pallet_insert_lift/env.py#L1009)

关键逻辑是：

- `p0 = fork_center`
- `p_pre = pallet + (s_goal - traj_pre_dist_m) * u_in`
- `p_goal = pallet + s_goal * u_in`
- 用起点朝向 `t0` 和托盘轴向 `u_in` 构造 Hermite 曲线

这里没有 wheelbase、没有最小转角、没有整车运动学约束。

### 3.2 reward 追的是 fork_center 到轨迹的距离

轨迹查询在：

- [env.py:1272](/home/uniubi/projects/forklift_sim/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/forklift_pallet_insert_lift/env.py#L1272)

查询点直接就是：

- `fork_center[:, :2]`

reward 里也是：

- [env.py:1693](/home/uniubi/projects/forklift_sim/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/forklift_pallet_insert_lift/env.py#L1693)
- [env.py:1753](/home/uniubi/projects/forklift_sim/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/forklift_pallet_insert_lift/env.py#L1753)

会把：

- `d_traj`
- `yaw_traj_err`
- `dist_center_family`

这些量转成正向奖励或门控。

### 3.3 out_of_bounds 也不是整车边界，而是 fork_center family distance

done 口径里：

- [env.py:2335](/home/uniubi/projects/forklift_sim/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/forklift_pallet_insert_lift/env.py#L2335)

`out_of_bounds` 仍然来自：

- `fork_center -> target_center_family` 的距离

不是整车 footprint 或驱动轮路径是否越界。

## 4. 这次做了什么修正和验证

### 4.1 修了可视化脚本的配置来源

脚本：

- [visualize_reference_trajectory_cases.py](/home/uniubi/projects/forklift_sim/forklift_pallet_insert_lift_project/scripts/visualize_reference_trajectory_cases.py)

现在默认优先读取当前 active 的：

- [IsaacLab/.../env_cfg.py](/home/uniubi/projects/forklift_sim/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/forklift_pallet_insert_lift/env_cfg.py)

只有找不到时才 fallback 到 legacy patch。

### 4.2 给图里补了 implied root trajectory

现在图里除了蓝色 `fork-center reference trajectory`，还会画一条灰色虚线：

- `implied root trajectory`

它不是完整动力学解，只是一个很保守的 proxy：

- 假设 `root = fork_center - fixed_offset * tangent`

这个 proxy 的意义不是“证明真实车辆一定这么走”，而是：

**提醒我们：即使 fork_center 的几何入口成立，整车可能也会被隐含要求做一个非常夸张的绕行。**

### 4.3 给 manifest 增加了 root proxy 指标

每个 case 现在多了：

- `root_y_abs_max`
- `root_heading_change_deg`
- `root_curvature_max`

并在单图里直接显示。

## 5. 重新生成后的结果

### 5.1 current 配置审计结果

输出目录：

- [reference_trajectory_stage1_viz_scope_audit_current](/home/uniubi/projects/forklift_sim/forklift_pallet_insert_lift_project/outputs/reference_trajectory_stage1_viz_scope_audit_current)

代表图：

- [current c19](/home/uniubi/projects/forklift_sim/forklift_pallet_insert_lift_project/outputs/reference_trajectory_stage1_viz_scope_audit_current/c19_xm3p450_ym0p150_yawm6p000.png)
- [current overlay](/home/uniubi/projects/forklift_sim/forklift_pallet_insert_lift_project/outputs/reference_trajectory_stage1_viz_scope_audit_current/overlay_all_cases.png)
- [current manifest](/home/uniubi/projects/forklift_sim/forklift_pallet_insert_lift_project/outputs/reference_trajectory_stage1_viz_scope_audit_current/reference_trajectory_stage1_manifest.json)

结果非常关键：

- `27/27` case 都是 `entry_ok = true`
- 但 worst case 的 `root_y_abs_max = 1.5201 m`
- worst case 的 `root_heading_change_deg = 432.27 deg`
- worst case 的 `root_curvature_max = 290.42 1/m`

这说明：

- 仅靠 `s_start < s_pre < s_goal`，已经完全不足以证明“这条参考轨迹对整车是合理的”
- 当前检查能通过，不代表轨迹真的像“叉车只需要轻微修正就能进托盘”

### 5.2 legacy 配置审计结果

输出目录：

- [reference_trajectory_stage1_viz_scope_audit_legacy](/home/uniubi/projects/forklift_sim/forklift_pallet_insert_lift_project/outputs/reference_trajectory_stage1_viz_scope_audit_legacy)

代表图：

- [legacy c19](/home/uniubi/projects/forklift_sim/forklift_pallet_insert_lift_project/outputs/reference_trajectory_stage1_viz_scope_audit_legacy/c19_xm3p450_ym0p080_yawm3p000.png)
- [legacy overlay](/home/uniubi/projects/forklift_sim/forklift_pallet_insert_lift_project/outputs/reference_trajectory_stage1_viz_scope_audit_legacy/overlay_all_cases.png)
- [legacy manifest](/home/uniubi/projects/forklift_sim/forklift_pallet_insert_lift_project/outputs/reference_trajectory_stage1_viz_scope_audit_legacy/reference_trajectory_stage1_manifest.json)

legacy 也一样：

- `27/27` case `entry_ok = true`
- worst case 的 `root_y_abs_max = 1.3901 m`
- worst case 的 `root_heading_change_deg = 434.38 deg`
- worst case 的 `root_curvature_max = 177.11 1/m`

也就是说，问题不是最近几次 steering 调参才引入的；这套 fork-center-only 轨迹检查从更早的时候开始就偏弱。

## 6. 为什么这会直接阻碍当前实验

当前我们一直在做的，是：

- 调 steering scale
- 调 sign-safe clip
- 调 preinsert guidance

但这些实验默认有个前提：

- 参考轨迹至少在“想教 agent 往哪儿转”这件事上是合理的

现在看，这个前提并不够成立。

因为当 reward 在拉：

- `d_traj`
- `yaw_traj_err`

它拉的是一个 fork-center corridor，而不是一个整车可执行 corridor。

这会带来两个风险：

1. agent 学到“沿错误代理轨迹转向”
2. 我们把训练失败误判成 PPO / reward / seed 问题，而不是参考轨迹代理本身太弱

## 7. 当前最合理的控制变量计划

### 7.1 先暂停继续扫 PPO 训练超参

在 trajectory preflight 没补强之前，不建议继续：

- 扫更多 steer scale
- 直接拉长到几百 iter
- 扫更多 reward weight

因为这些实验都默认参考轨迹的教学方向基本正确。

### 7.2 先做 Trajectory Preflight V1

目标：不改训练，只改审计。

内容：

- 固定 current cfg
- 跑 [visualize_reference_trajectory_cases.py](/home/uniubi/projects/forklift_sim/forklift_pallet_insert_lift_project/scripts/visualize_reference_trajectory_cases.py)
- 对 `3 x 3 x 3 = 27` 个代表 case 输出：
  - `entry_ok`
  - `root_y_abs_max`
  - `root_heading_change_deg`
  - `root_curvature_max`

验收：

- 审计结果必须先稳定复现
- 明确 worst case 和代表性 case
- 先证明“问题确实存在且不是看图错觉”

这个阶段已经完成。

### 7.3 再做 Trajectory Preflight V2

目标：仍然不改 PPO，只补“更接近整车”的前置判据。

建议只改一个变量层面：

- 从“fork-center-only”升级到“fork_center + implied root proxy gate”

具体可做：

- 给代表 case 增加 root proxy 阈值统计
- 先筛掉明显过激的 `p_pre / tangent` 组合
- 只对轨迹构造参数做单因素修改，例如：
  - `traj_pre_dist_m`
  - Hermite 切线长度系数
  - reset x/y/yaw 分布

每次只改一个，重新看：

- `entry_ok`
- `root_y_abs_max`
- `root_heading_change_deg`
- `root_curvature_max`

不要在这一阶段同时改 reward。

### 7.4 只有 preflight 过关，再恢复 PPO

恢复训练前的闸门建议改成：

1. 参考轨迹审计先过
2. 再跑 normal vs zero-steer grid
3. 最后再做多 seed 训练

顺序不能反过来。

## 8. 当前判断

这次用户指出的是一个真正的基础问题，不是“图看着别扭”这么简单。

最准确的表述应该是：

**当前参考轨迹并不是整车运动学轨迹，而只是 fork_center 的几何引导轨迹。**

它可以拿来做一个弱代理，但不能再被当作“已经验证过的整车正确轨迹”。

如果不先把这层 scope 说清楚，后面的 steering / reward / PPO 实验很容易被带偏。

