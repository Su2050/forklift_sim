# Exp8.3 Trajectory Preflight V2 修复计划

日期：2026-03-31

相关背景文档：

- [exp83_reference_trajectory_scope_audit_20260331.md](/home/uniubi/projects/forklift_sim/docs/0330-0331experiments/exp83_reference_trajectory_scope_audit_20260331.md)
- [exp83_next_optimization_vs_visual_forklift_paper_20260331.md](/home/uniubi/projects/forklift_sim/docs/0330-0331experiments/exp83_next_optimization_vs_visual_forklift_paper_20260331.md)

## 1. 当前问题定义

当前系统中的 reference trajectory 存在明确基础问题：

- 轨迹起点和跟踪对象是 `fork_center`
- 只检查 `s_start < s_pre < s_goal`
- reward / `d_traj` / `yaw_traj_err` / `out_of_bounds` 都和这条 fork-center-only 轨迹深度绑定

但它没有覆盖：

- 整车 wheelbase / rear-drive path
- 最小转弯半径
- 车体 sweep
- 整车是否能自然地走出这条路

所以现在最需要修的，不是 reward 权重，也不是 PPO 超参，而是：

**先把 trajectory 从“弱几何代理”修成“至少经过整车 proxy 审计的 approach guide”。**

## 2. 这次计划的总目标

这轮计划的目标不是立刻把成功率拉高，而是分 3 步把基础链条理顺：

1. 先把 trajectory preflight 做对
2. 再把 PPO 任务缩回到 `approach/alignment`
3. 最后才恢复 `insert / hold / lift` 的后半段

一句话说：

**先修“老师指的路”对不对，再修“学生会不会学”。**

## 3. 分支策略

### 3.1 基线冻结

先把当前“trajectory 问题已经被确认”的状态视作冻结基线：

- 用于保留审计结论
- 不再从这个点继续扫 PPO 超参

建议打一个只读标签：

- `exp8_3_pre_trajfix_baseline_20260331`

### 3.2 修复分支

建议新开 3 条主分支，严格按阶段推进。

1. `exp/exp8_3_traj_preflight_v2`
   - 只做 trajectory generator / trajectory audit / vehicle proxy 检查
   - 不跑正式 PPO 训练

2. `exp/exp8_3_approach_only_trajv2`
   - 从 `traj_preflight_v2` 通过点切出
   - 只做 `approach/alignment` 任务

3. `exp/exp8_3_insert_ready_gate`
   - 从 `approach_only_trajv2` 的稳定点切出
   - 只做 endpoint decision / lift gate

## 4. 总体约束

整个修复过程中必须遵守下面 4 条：

1. 每一阶段只改一层
   - 先改 trajectory
   - 再改 task definition
   - 最后改 decision policy

2. 在 `Trajectory Preflight V2` 通过前
   - 不继续扫 steer scale
   - 不继续扫 clean bonus weight
   - 不继续跑几百 iter 长训

3. 每次改动都必须有配套审计产物
   - png
   - manifest/json
   - md 结果总结

4. 每个阶段都要有明确“通过门槛”和“失败回退点”

## 5. Phase A：Trajectory Preflight V2

### 5.1 目标

把当前 trajectory 检查从：

- fork-center-only entry check

升级成：

- 至少带 `implied root / rear-axle proxy` 的 vehicle-aware preflight

### 5.2 这一阶段允许修改的内容

只允许改这些：

- trajectory generator
- trajectory visualization / audit scripts
- trajectory manifest 指标

不允许改这些：

- PPO 超参
- actor 输入
- reward 权重
- success / hold 判定逻辑

### 5.3 第一批提交要做什么

第一批提交建议只做 3 件事：

1. 增加 vehicle-aware proxy 指标
   - `root_y_abs_max`
   - `root_heading_change_deg`
   - `root_curvature_max`
   - 如能拿到更真实几何，再补 `rear_axle_path` 指标

2. 新建 `Trajectory Preflight V2` 脚本
   - 输入：current stage1 case grid
   - 输出：
     - top-down png
     - overlay
     - manifest
     - worst-case case list

3. 定义 trajectory 通过阈值
   - 哪些 case 属于“明显离谱”
   - 哪些 case 可以接受

### 5.4 第二批提交要做什么

只改 trajectory generator，不碰别的：

候选方案按顺序做，且每次只做一个：

1. `fork_center Hermite -> root_path first`
2. `root_path first -> rear-axle/bicycle proxy`
3. 调 `traj_pre_dist_m`
4. 调 Hermite 切线长度系数

每改一次都重新跑：

- `current 27 cases`
- `legacy 27 cases`

### 5.5 Phase A 通过标准

必须同时满足：

1. `entry_ok` 不能再成为唯一通过标准
2. worst-case `root proxy` 指标明显下降
3. 不再出现“27/27 全绿，但 implied root 一看就离谱”的情况
4. 至少能指出一套“current stage1 下可接受”的 trajectory family

### 5.6 Phase A 失败处理

如果发现：

- 任何基于 `fork_center -> p_pre` 的生成器都天然会逼出离谱 root path

那就不要再补 patch 了，直接转为：

- 彻底改成 `rear-axle / root-centric` 的 trajectory 定义

## 6. Phase B：Approach-Only on Trajectory V2

### 6.1 目标

在 trajectory 过关后，把 PPO 任务从：

- `approach + insert + hold + success`

缩回到：

- `approach + alignment + stop-ready`

### 6.2 允许修改的内容

只允许改：

- success 定义
- reward 的后半段口径
- eval 口径

不允许改：

- trajectory generator
- camera placement
- backbone / PPO 主超参

### 6.3 第一批提交要做什么

1. 新建 `approach_only` success 定义
   - 到达可插入位姿
   - lateral / yaw 已进入窗口
   - 速度较低或停稳

2. reward 只保留 approach 相关项
   - 接近托盘前沿
   - 减小横向偏差
   - 减小朝向偏差
   - wrong-sign steering 护栏
   - push/碰撞惩罚

3. 暂时去掉：
   - clean insert 必须成立
   - hold 必须成立
   - 最终 success 必须已经插深

### 6.4 验证实验

先做最小闭环：

- `3 seeds x 50 iter`
- 每个 seed 立刻跑：
  - `3x3 normal`
  - `3x3 zero-steer`

### 6.5 Phase B 通过标准

必须同时满足：

1. 至少 `2/3 seeds`
2. `normal > zero-steer`
3. 不再是单边固定 steering bias
4. 单点诊断里 steering sign 能随误差方向变化

### 6.6 Phase B 失败处理

如果 trajectory 已经过关，但 `normal` 仍不强于 `zero-steer`，再按顺序排查：

1. camera 视角是否真的能看到 fork/pallet 对位
2. actor 输入里 signed alignment 是否足够
3. 近场 task 是否仍然定义得过难

## 7. Phase C：Insert-Ready Gate

### 7.1 目标

把“现在是否应该执行 insert / lift”从主 PPO 中拆出去。

### 7.2 允许修改的内容

只允许改：

- endpoint 数据采集
- classifier / gate
- 后处理执行流程

不允许改：

- approach PPO
- trajectory generator

### 7.3 第一批提交要做什么

1. 采集 endpoint 数据
   - success / fail 两类
   - 对齐当前相机视角

2. 训练二分类 gate
   - `insert_ready` 或 `lift_ready`

3. 先接规则执行器
   - 小速度前进
   - lift
   - reverse

### 7.4 Phase C 通过标准

1. endpoint classifier 在验证集上稳定区分 success/fail
2. 和 `approach_only` 组合后，整体成功率高于纯 PPO 直推后半段

## 8. 每个分支的第一批提交建议

### `exp/exp8_3_traj_preflight_v2`

第一批提交：

- 新增 `vehicle-aware trajectory audit` 脚本或扩展现有脚本
- 新增 manifest 指标
- 新增一份结果 md

### `exp/exp8_3_approach_only_trajv2`

第一批提交：

- 新增 `approach_only` success 定义
- 精简 reward 到 `approach/alignment`
- 新增 `normal vs zero-steer` 回归脚本配置

### `exp/exp8_3_insert_ready_gate`

第一批提交：

- 新增 endpoint 采样脚本
- 新增 gate 训练脚本或最小 baseline
- 新增执行流程说明文档

## 9. 建议的提交节奏

建议强制按这个节奏：

1. 改脚本/配置前先 commit 当前状态
2. 每完成一个单因素改动就 commit
3. 每完成一轮验证就写 md 并 commit

这样才能保证后面回看时，知道到底是哪一刀起作用。

## 10. 立即执行顺序

按优先级，下一步只做下面这 4 步：

1. 冻结当前 baseline
2. 切 `exp/exp8_3_traj_preflight_v2`
3. 做 `Trajectory Preflight V2` 第一批提交
4. 在 `Trajectory Preflight V2` 通过前，暂停新的 PPO 扫参/长训

## 11. 最终判断标准

这次修复是否成功，不看“有没有又多跑几条训练”，而看下面这条链是否打通：

1. trajectory 先过 vehicle-aware preflight
2. `approach_only` 学会真正 steering-based alignment
3. `normal > zero-steer`
4. endpoint gate 能稳定接上后半段

只有这条链打通，后面再往“自动插托盘”推进，才是在正确方向上加速。

