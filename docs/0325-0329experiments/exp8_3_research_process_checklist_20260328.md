# Exp8.3 研究流程固定 Checklist

日期：2026-03-28

## 1. 这份 checklist 是干什么的

这份清单不是为了解释某一轮具体实验结果，而是为了避免后续再犯“机制本身没先验证清楚，就先去跑大批训练”的同类错误。

当前最需要防的，是下面 4 类“同质错误”：

1. 以为 actor 在用某个信息，实际上只有 env 或 critic 在用
2. 以为 success 变好，就等于 steering 学出来了
3. 以为 reward / 单 seed / 训练尾窗可以代表真实方向
4. 以为一个代表点的几何没问题，就等于整段几何都没问题

这 4 类问题都已经在 `exp8.3` 最近几轮实验里真实出现过，所以后面每一轮新实验都应该过这份清单。

---

## 2. Checklist A：actor 到底看到了什么

### 目标

防止我们误以为某个信号在指导 policy，实际上它只是：

- reward 里在用
- critic 里在用
- env 里算出来了，但 actor 根本没拿到

### 必查项

1. 列出本轮你指望 policy 利用的关键信号：
   - 例如 `y_err`
   - `yaw_err`
   - `d_traj`
   - `yaw_traj_err`
   - `s_traj`

2. 对每个信号都回答 3 个问题：
   - actor 是否直接看到
   - critic 是否看到
   - reward 是否使用

3. 如果某个信号只是 reward/critic 在用，而 actor 不直接看到：
   - 不能再口头假设“policy 应该会自然学会利用它”
   - 必须降级为“间接 shaping 信号”

### 本项目中的典型例子

- 当前 actor 看到的是 `image + easy8`
- signed 的 `y_err_obs / yaw_err_obs` 并不直接喂给 actor
- 所以“轨迹会自动教会 steering”这件事不能直接假设成立

### 通过标准

只有当“你依赖的 steering 关键信号”明确在 actor 观测里，或者有证据表明 actor 可以稳定从视觉中恢复它，才允许继续把该机制当成 steering 学习主因。

---

## 3. Checklist B：success 变好，到底是不是 steering 学出来了

### 目标

防止把“前推捷径成功”误判成“策略学会了转向纠偏”。

### 必查项

每个候选配置、每个代表 checkpoint，都固定做：

1. `normal misalignment grid`
2. `zero-steer grid`

并至少比较：

- `success_rate_ep`
- `ever_inserted_push_free_rate`
- `ever_hold_entry_rate`
- `ever_dirty_insert_rate`

### 解释规则

- 如果 `normal` 没明显强于 `zero-steer`
  - 就不能说 steering 学出来了
- 如果 `normal > zero-steer`
  - 才说明 steering 正在变成必要技能

### 建议附加项

再补一层动作诊断：

- `mean_abs_steer`
- `max_abs_steer`
- `steer` 与 `y_err / yaw_err` 的相关性

### 通过标准

只有当 `normal` 明显强于 `zero-steer`，才允许说“这条配置在 steering 上是真进步，不只是 success 漂亮”。

---

## 4. Checklist C：不要被 reward / 单 seed / 尾窗骗

### 目标

防止把：

- `mean_reward`
- 单 seed 的好结果
- 最后几轮的漂亮尾窗

当成方向已经正确的证据。

### 必查项

1. 至少看 `3 seeds`
2. 训练阶段不要只看：
   - `mean_reward`
   - `frac_success`
3. 必须同时看：
   - `phase/frac_inserted_push_free`
   - `phase/frac_hold_entry`
   - `phase/frac_dirty_insert`
   - `diag/pallet_disp_xy_mean`
   - `err/dist_front_mean`

4. 对关键配置至少补一轮统一 eval

### 解释规则

- reward 高，不代表 basin 对
- 单 seed 成功，不代表 family 稳
- 尾窗有 success，不代表 steering 学出来

### 通过标准

只有当：

- 多 seed 不再剧烈分叉
- 统一 eval 支持训练结论
- 关键诊断指标与想要的机制方向一致

才能把该配置列为主线候选。

---

## 5. Checklist D：几何不要只看一个点

### 目标

防止“中间一个代表点没问题，但边界点依然错”的情况。

### 必查项

凡是改这些参数：

- `stage1_init_x_*`
- `stage1_init_y_*`
- `stage1_init_yaw_*`
- `traj_pre_dist_m`
- `traj_goal_mode`

都必须先做 geometry-only sweep。

至少扫：

1. `x` 的前边界 / 中点 / 后边界
2. `y ∈ {-y_max, 0, +y_max}`
3. `yaw ∈ {-yaw_max, 0, +yaw_max}`

并直接读取：

- `s_start`
- `s_pre`
- `s_goal`
- `delta_s = s_start - s_pre`
- `y_start`

### 解释规则

- 只要某些边界 case 仍然 `s_start >= s_pre`
  - 就不能说 entry geometry 已修好
- 只看一个“好看”的代表点是不够的

### 通过标准

至少要满足：

- 所有代表性 case 都有 `s_start < s_pre < s_goal`

更理想的是：

- `delta_s <= -0.05m`

这样才说明 trajectory entry 真正存在。

---

## 6. 每轮新实验的推荐顺序

以后任何涉及 trajectory / steering / stage1 reset 的实验，推荐固定按这个顺序：

1. **Actor 信号检查**
   - actor 到底看到了什么
2. **Geometry-only sweep**
   - 双视图可视化
   - `s_start / s_pre / s_goal`
3. **Smoke**
   - 2 iter
4. **短训**
   - `3 seeds x 50 iter`
5. **Grid**
   - `normal`
   - `zero-steer`
6. **统一 eval**
   - 只对真正有希望的候选做

如果前面某一层没过，就不要急着进入下一层。

---

## 7. 当前最需要记住的一句话

**以后不要再只问“这个配置训出来成不成功”，要先问“这个配置想依赖的那个机制，actor 真的看得到、几何真的成立、评估真的能区分出来吗”。**

这就是这份 checklist 的核心作用。
