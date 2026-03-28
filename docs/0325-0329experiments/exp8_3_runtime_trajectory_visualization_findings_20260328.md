# Exp8.3 Runtime 参考轨迹可视化发现总结

日期：2026-03-28

## 1. 这次可视化想回答什么

这次可视化不是在回答“策略最终成功率高不高”，而是在回答更底层的 3 个问题：

1. 当前 runtime 里真实生成的参考轨迹到底长什么样。
2. 参考轨迹的入口几何，是否真的把 steering 变成了必要动作。
3. 当前 near-field 训练里，策略到底是在“沿轨迹转向插入”，还是主要靠“往前叉”。

## 2. 先给结论

### 2.1 `fork_center` 起点相对 `p_pre` 的位置关系不合理，是不是已经可以确定？

可以说：**这是当前 `stage1` 课程里的一个已确认几何问题。**

但需要把结论说严谨：

- 可以确认的是：
  - 在当前 `stage1` reset 几何下，runtime 真实生成参考轨迹时，`fork_center start` 在托盘插入轴 `s` 上已经始终位于 `p_pre` 的前方。
  - 这会显著削弱“先纠偏，再接入直插段”的入口几何约束。
- 还不能直接说：
  - 这就是当前所有训练失败的唯一根因。

更准确的表述应该是：

**它是一个已经被可视化和数值共同坐实的几何缺陷，而且这个缺陷非常可能是当前 steering gap 打不开的重要原因之一。**

## 3. 这次可视化到底看到了什么

### 3.1 runtime 真实生成的参考轨迹，不是固定模板，而是 reset 时现生成的

环境在 reset 时调用 `_build_reference_trajectory()` 生成并缓存轨迹：

- `env.py:996`

关键几何定义是：

- 起点 `p0` 用的是 `fork_center`
- 起始切线 `t0` 用的是当前 robot yaw
- 终点前先定义 `p_pre`
- 再用 `Hermite 曲线 + 末段直线` 组成参考轨迹

这说明当前系统不是用一条预先画死的“标准路径”，而是每次根据当前起点位姿现场生成一条路径。

### 3.2 当前 `stage1` reset 下，`fork_center start` 始终已经越过了 `p_pre`

当前 `stage1 v2` 的 reset 范围是：

- `x_root ∈ [-3.55, -3.25]`
- `y ∈ [-0.08, 0.08]`
- `yaw ∈ [-3°, 3°]`

对应配置位置：

- `env_cfg.py:99-107`

这次我对 `25` 个代表性 case 做了 runtime 轨迹可视化，manifest 在：

- `outputs/exp83_runtime_traj_topdown_stage1v2/exp83_runtime_traj_viz_stage1v2_manifest.json`

从 manifest 反算托盘坐标系里的 `s_start - s_pre` 后，结果是：

- 共 `25` 个 case
- `delta_s = s_start - s_pre`
- 最小值：`+0.1398 m`
- 最大值：`+0.1467 m`
- 平均值：`+0.1432 m`

也就是说：

**所有 case 里，`fork_center start` 都已经在 `p_pre` 前面大约 `14 cm`。**

这不是偶然个例，而是当前这套课程几何的系统性现象。

### 3.3 代表性 case 直接把这个问题看得很清楚

代表性 overlay case：

- `x=-3.40, y=+0.08, yaw=+3°`

新版双视图图像在：

- `outputs/exp83_runtime_traj_topdown_stage1v2_overlay_v2view/exp83_runtime_traj_viz_stage1v2_seed42_normal_overlay_v2view_rollout_xm3p40_yp0p080_yawp3p0.png`
- `outputs/exp83_runtime_traj_topdown_stage1v2_overlay_v2view/exp83_runtime_traj_viz_stage1v2_seed42_zero_overlay_v2view_rollout_zero_steer_xm3p40_yp0p080_yawp3p0.png`

对应 manifest：

- `outputs/exp83_runtime_traj_topdown_stage1v2_overlay_v2view/exp83_runtime_traj_viz_stage1v2_seed42_normal_overlay_v2view_manifest.json`
- `outputs/exp83_runtime_traj_topdown_stage1v2_overlay_v2view/exp83_runtime_traj_viz_stage1v2_seed42_zero_overlay_v2view_manifest.json`

这个 case 的关键数值是：

- `s_start = -2.1350`
- `s_pre = -2.2800`
- `s_goal = -1.0800`
- `delta_s = s_start - s_pre = +0.1450`
- `y_start = +0.1463`

这意味着：

- 起点不是“还在 `p_pre` 后面，准备去接轨迹入口”
- 而是“已经跑到 `p_pre` 前面去了”

因此当前轨迹入口并不是在表达一个很清晰的：

“先通过 steering 把车摆正，再向前接入直插段”

而更像是在一个已经很靠前的局部区域里，允许 agent 继续向前推进。

### 3.4 当前 actor 并没有直接拿到“该往左还是往右打”的轨迹信号

这次复核代码后，还确认了另一件关键事：

- env 确实计算了带符号的 `y_err_obs / yaw_err_obs`
- 但 camera 模式下 actor 的 `policy` 输入是 `image + proprio`
- 其中 `proprio` 实际是 `easy8`
- actor 不直接吃 `y_err_obs / yaw_err_obs`

对应位置：

- `env.py:1483-1514`
- `rsl_rl_ppo_cfg.py:39-41`

这意味着当前参考轨迹主要是通过 reward shaping 生效，不是显式给 actor 一个 signed steering supervision。

所以系统现状更像：

- 轨迹在训练里负责“给分”
- actor 靠视觉和探索自己摸索怎么做动作

这会进一步放大“前推捷径”。

### 3.5 `normal` 和 `zero-steer` 的 gap 仍然很小

这次 `v2` grid 已经完整跑完，结果在：

- `outputs/exp83_stage1_steering_curriculum_v2_grid`

三组 seed 的 `success_rate_ep` 对比：

- `seed42`: `normal 0.4082` vs `zero-steer 0.3673`
- `seed43`: `normal 0.3673` vs `zero-steer 0.3878`
- `seed44`: `normal 0.3878` vs `zero-steer 0.3878`

这组结果说明：

**即使把 `y/yaw` 放宽到 `±0.08m / ±3°`，当前策略依然没有表现出“明显依赖 steering 才能成功”的特征。**

### 3.6 这次可视化还发现了什么额外问题

除了 `s_start > s_pre` 这件最重要的事，这次还发现了 3 个附加问题：

1. 当前 near-field 课程里的 `y/yaw` 偏差虽然比旧版更大，但仍然不足以强迫策略稳定学会 steering。
2. 参考轨迹虽然在世界坐标里看上去“形状没坏”，但放到托盘坐标系看，入口段的几何约束不够强。
3. rollout overlay 里，`normal` 和 `zero-steer` 的轨迹差异也不大，这与 grid 结果是一致的，不是单次偶然现象。

## 4. 这意味着什么

### 4.1 现在最不该做的事

- 不该直接把当前 `stage1 v2` 往 `200/400 iter` 推
- 不该继续只扫 `bonus weight`
- 不该把“轨迹存在”误解成“agent 已经学会了轨迹跟踪”

### 4.2 现在最该改的，不一定是曲线形式本身

从当前证据看，优先级应该是：

1. **先改几何关系**
   - 让 `fork_center start` 位于 `p_pre` 后面，而不是前面
   - 也就是让 trajectory entry 真正存在
2. **再看曲线形式**
   - 如果几何关系改正后，入口 steering 还是不明显
   - 再讨论是否需要换 Hermite、改切线长度、改 `p_pre` 定义

所以现在更像是在回答：

**“entry geometry 先错位了”**

而不是：

**“曲线一定选错了”。**

## 5. 当前最靠谱的下一步

基于这次可视化，我认为后续优先级应该是：

1. 设计 `v3`，优先修正 `fork_center start` 与 `p_pre` 的轴向关系
2. 继续保留双视图 runtime 轨迹可视化作为几何回归检查
3. 只有在 `s_start < s_pre` 后，才值得继续看 steering gap 是否真正被打开

## 6. 用最通俗的话讲，这次发现的重大 bug 是什么

可以把它理解成：

**系统本来想教叉车“先摆正，再插进去”；但现在给它画的参考路线，起点已经站到“预对位点的前面”去了。**

这就像：

- 你本来想让人先走到起跑线，再按赛道跑
- 但实际上他一开始就已经站到起跑线前面了
- 那这条赛道的“入口引导”就被削弱了

结果就是：

- 叉车没有被强迫学会“先转向摆正”
- 它更容易学成“差不多对着就往前叉”

这就是这次 runtime 可视化最重要的收获。
