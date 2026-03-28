# Exp8.3 Stage1 Entry Geometry v3 计划

日期：2026-03-28
分支：`exp/exp8_3_stage1_entry_geometry_v3`

## 1. 背景

`stage1 steering curriculum v2` 已经证明两件事：

1. 当前系统里确实存在“只会往前叉也能成功”的捷径。
2. 仅仅把 `y / yaw` 放宽，并不足以稳定拉开 `normal > zero-steer` 的 steering gap。

最新 runtime 双视图可视化又补上了一个更底层的发现：

- 当前 `stage1` 典型 case 中，`fork_center start` 在托盘坐标系下已经始终位于 `p_pre` 前方。
- 也就是说，入口几何本身就在削弱“先转向纠偏，再接入直插段”这件事。

因此，`v3` 的目标不再是继续微调 reward，而是：

**优先修正 `s_start / s_pre` 的几何关系，让 trajectory entry 真正存在。**

## 2. 核心假设

当前 steering gap 打不开，更像是因为：

1. `fork_center start` 相对 `p_pre` 过于靠前
2. 参考轨迹入口段没有形成足够明确的 steering 必要性
3. 因此 actor 仍能主要依赖“前推近似解”

所以 `v3` 的第一主因假设是：

**先把 `s_start < s_pre` 做对，比继续调曲线形式或 bonus 权重更重要。**

## 3. 本轮不优先做什么

这轮 `v3` 明确不优先做：

- 不先回 `wide reset`
- 不先继续扫 `bonus weight`
- 不先加重 `gate / dirty penalty`
- 不先大改 Hermite 曲线形式
- 不先把起点从 `fork_center` 换成 `root`

原因是：这些改动都会把问题重新混杂，而当前最明确、最可验证的缺陷，是 entry geometry。

## 4. v3 设计思路

`v3` 只优先动两类参数：

### 4.1 第一类：让起点后移

通过把 `stage1_init_x_*` 往后移，直接让 `fork_center start` 在轴向上退回 `p_pre` 后方。

候选方向：

- `x` 整体后移 `0.10m`
- `x` 整体后移 `0.15m`

### 4.2 第二类：让 `p_pre` 前移

通过减小 `traj_pre_dist_m`，把 `p_pre` 向 `p_goal` 方向推近。

候选方向：

- `traj_pre_dist_m: 1.20 -> 1.10`
- `traj_pre_dist_m: 1.20 -> 1.05`
- 如有必要，再试 `1.00`

### 4.3 优先尝试“小组合”，而不是一次大改

本轮更推荐组合式轻改，而不是单边极端改动。

原因：

- 只改 `x` 太多，可能把 approach 难度一下子抬高
- 只改 `p_pre` 太多，可能把末段轨迹入口压得过短

因此优先候选是：

- `V3-A`: `x` 后移 `0.10m` + `traj_pre_dist_m = 1.10`
- `V3-B`: `x` 后移 `0.15m` + `traj_pre_dist_m = 1.05`

## 5. 先验推导

基于当前可视化统计：

- 当前 `delta_s = s_start - s_pre ≈ +0.14m`

如果想让入口真正存在，至少要做到：

- `delta_s < 0`

如果想留一点 margin，更合理的目标是：

- `delta_s <= -0.05m`

这意味着：

- 单靠 `x` 后移 `0.15m`，大致能把 `delta_s` 拉回到接近 `0` 或略负
- 单靠 `traj_pre_dist_m` 从 `1.20` 降到 `1.05`，也大致能把 `delta_s` 拉回到接近 `0`
- 更稳妥的是“小幅 x 后移 + 小幅 pre_dist 前移”的组合

## 6. 执行步骤

### Phase 0：冻结基线

基线提交：

- 当前分支上游：`1fd0d6fa` `Add trajectory visualization findings and dual-view plots`

这意味着 `v3` 所有改动都建立在：

- `v2 grid` 已完成
- runtime 双视图可视化已具备
- entry geometry 问题已被观察到

### Phase 1：几何-only 参数筛选

这一阶段**不先训练**，只做 runtime geometry 验证。

对每个候选配置，至少验证：

1. `stage1` 代表性 case：
   - `x` 取区间中点和边界
   - `y ∈ {-0.08, 0, +0.08}`
   - `yaw ∈ {-3°, 0°, +3°}`
2. 双视图 runtime 轨迹：
   - 世界坐标
   - 托盘坐标
3. 直接读取：
   - `s_start`
   - `s_pre`
   - `s_goal`
   - `delta_s`

几何验收标准：

- 所有代表性 case 都满足 `s_start < s_pre`
- 更理想地，`delta_s <= -0.05m`

只有通过 Phase 1，才允许进入训练。

### Phase 2：代表性 rollout overlay

对通过 Phase 1 的候选配置，至少选 1 个代表性 case 做：

- runtime 参考轨迹 + rollout overlay
- `normal`
- `zero-steer`

这一阶段主要不是看成功率，而是看：

- 是否出现更明确的 steering entry
- rollout 是否仍然只是沿着“前推捷径”走

### Phase 3：短训筛选

只对 Phase 1/2 最好的 `1~2` 个候选配置跑：

- `3 seeds x 50 iter`
- `256x256`
- near-field
- `bonusw=1.0`

主要看：

- `phase/frac_inserted_push_free`
- `phase/frac_hold_entry`
- `phase/frac_success`
- `phase/frac_dirty_insert`
- `diag/pallet_disp_xy_mean`

### Phase 4：steering gap 回归

对每个 seed 的末尾 checkpoint 继续跑：

- `normal misalignment grid`
- `zero-steer grid`

硬标准：

1. `normal` 明显强于 `zero-steer`
2. 至少 `2/3` seed 出现稳定的 `push_free / hold / 非零 success`

只有两条都满足，才值得往 `200 iter` 推。

## 7. 本轮验证必须包含哪些内容

这轮验证不允许只看训练日志，必须同时包含：

1. runtime 双视图可视化
2. `delta_s` 数值表
3. representative rollout overlay
4. `normal / zero-steer grid`

也就是说：

**v3 的“通过”必须是几何、行为、steering gap 三层同时过关。**

## 8. 成功 / 失败分支

### 成功情形

如果某个候选配置满足：

- `s_start < s_pre`
- `normal > zero-steer`
- `2/3` seed 出现稳定 `push_free / hold / success`

则：

- 选它作为 `v3 winner`
- 推到 `200 iter`
- 再做统一 eval + grid

### 失败情形 A：几何没修正过来

如果 `delta_s` 仍然为正：

- 不开训练
- 继续改 entry geometry

### 失败情形 B：几何修正了，但 steering gap 仍然不开

说明：

- 问题不只在 `s_start / s_pre`
- 下一步才值得讨论曲线形式、signed actor 输入、或更强 steering curriculum

但这一步必须建立在“entry geometry 已经先修正”的前提下。

## 9. 当前优先候选

当前最值得先试的两个候选是：

### `V3-A`

- `stage1_init_x_min_m = -3.65`
- `stage1_init_x_max_m = -3.35`
- `traj_pre_dist_m = 1.10`

理由：

- 改动比较温和
- 更像是在修 entry geometry，而不是重做任务

### `V3-B`

- `stage1_init_x_min_m = -3.70`
- `stage1_init_x_max_m = -3.40`
- `traj_pre_dist_m = 1.05`

理由：

- 更强地保证 `s_start < s_pre`
- 适合作为更激进的对照线

## 10. 一句话目标

`v3` 的目标不是“再训出一个高 success checkpoint”，而是：

**先把 entry geometry 摆正，让参考轨迹入口真正要求 steering，再决定要不要把训练往几百 iter 推。**
