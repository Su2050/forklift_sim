# Exp8.3 RS Near-Field Feasibility Takeaway 2026-03-31

## 背景

本轮已经完成了几组围绕 Stage1 当前初始化范围的 RS 审计：

- `rs_exact` 到 final front-goal
- `rs_forward_preferred` 到 final front-goal
- `forward-preferred max_extra_length_m` 单因素放宽
- 离线额外分析：把 RS 目标改成 `root_pre` 再接直线

当前 Stage1 范围是：

- `x in [-3.60, -3.45]`
- `y in [-0.15, 0.15]`
- `yaw in [-6 deg, 6 deg]`

最小转弯半径保持物理设定：

- `traj_rs_min_turn_radius_m = 2.34`

## 核心发现

### 1. direct shortest RS 到 final front-goal 不适合作为 near-field 主参考

`rs_exact` 的问题不是“实现错了”，而是它会优先给出局部最短、但对 RL 很不友好的解：

- heading change 可到 `178.42 deg`
- curvature 可到 `229.75 1/m`

这类解虽然在数学上合法，但在当前 near-field 课程里会把参考轨迹变成折返/打结样式，不适合直接喂给 reward。

### 2. forward-preferred RS 当前不是“选不优”，而是“根本选不出来”

在当前 forward-preferred 筛选门槛下：

- `125/125` case 全 fallback 到 `root_path_first`

所以当前看到的温和轨迹，并不是“forward-preferred RS 生效了”，而是“RS 被全部挡掉后，系统退回旧模型”。

### 3. 单纯放宽 `max_extra_length_m` 没有实际救回来

单因素试探：

- `1.50 -> 3.00`

结果没有任何变化，仍然是：

- `125/125` fallback

更关键的是离线统计显示：

- 想让当前 near-field case 开始接受真实 RS，`extra_length` 不是 `2m`、`3m` 级别
- 而是要到 **约 `13~14m`** 才开始放出

这已经说明问题不是“小调一下门槛”，而是**几何层面结构性不匹配**。

### 4. 把 RS 目标改成 `root_pre` 也没有从根本上解决问题

离线分析里，我把 RS 目标从 final front-goal 改成了 `root_pre`，再假想后半段接直线。

结果虽然比 direct-to-final 稍好：

- 最小所需 `extra_length` 从 `13.26m` 降到 `6.62m`

但仍然远超当前 forward-preferred 想要的“接近 shortest”范畴：

- `12.0m` 阈值下也只有 `10/125` case 可接受
- `20.0m` 时才到 `110/125`

所以即便改成 “RS 前半段 + 直线后半段”，在当前 near-field + 真实最小转弯半径下，也仍然不够自然。

## 结论

当前这个结论已经足够明确：

**在 Stage1 当前 near-field 课程上，不应该继续把 exact / forward-preferred RS 当成训练主参考轨迹的首选方向。**

原因不是代码没写完，而是：

- near-field 距离太短
- terminal pose 约束太硬
- 最小转弯半径太大

三者叠加后，RS family 天然更容易给出“对几何规划成立、但对 RL reward 不友好”的路径。

## 对训练的直接建议

### 当前阶段

- 训练主参考轨迹继续保留 `root_path_first`
- RS 保持为：
  - 独立工具链
  - 诊断/对比工具
  - 后续 wide-reset 或更长距离阶段的候选 planner

### 不建议继续做的事

- 不建议继续扫 `max_extra_length_m`
- 不建议继续扫 `max_reverse_frac`
- 不建议为了“让 RS 通过”而把筛选放宽到十几米 extra-length

### 更值得的下一步

如果还想沿 RS 这条线继续探索，更合理的方向应该是：

1. 把 RS 应用到更长距离的阶段，而不是当前 near-field Stage1
2. 或者重新定义更上游的中间目标，而不是直接 exact pose 到托盘前方
3. 但在当前这个 near-field 训练问题上，优先级应回到：
   - `root_path_first` 参考轨迹是否足够 vehicle-aware
   - reward / steering / insert gate 的训练闭环

## 一句话总结

**RS 在当前 near-field 课程上不是“还差一点调参”，而是“从几何规模上就不太适合做主参考轨迹”。**
