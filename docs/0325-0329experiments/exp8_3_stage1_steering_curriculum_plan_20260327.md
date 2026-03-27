## exp8.3 stage1 steering curriculum plan (2026-03-27)

### 背景

最新 steering 诊断已经说明，当前 near-field 强 checkpoint 的成功几乎不依赖 steer：

- normal vs zero-steer 的 near-field eval 几乎不掉点
- normal vs zero-steer 的 misalignment grid 成功格点完全一致
- 当前 `stage1` 初始化过于对齐：
  - `x ∈ [-3.55, -3.25]`
  - `y ∈ [-0.05, 0.05]`
  - `yaw ∈ [-2°, 2°]`

这意味着当前训练主线存在一条明显捷径：

**策略只要学会“往前叉”就能在一部分 run 上进入 success，而不需要真正学会 steering-based correction。**

因此，下一阶段的目标不再是继续微调 post-insert bonus，而是：

**先堵住“前推捷径”，把 stage1 变成“不会转向就过不了”的课程。**

---

### 分支策略

本计划在一个新的实现分支上推进：

- 基线来源：`exp/exp8_3_steer_usage_diagnostics`
- 新分支：`exp/exp8_3_stage1_steering_curriculum`

这样做的原因：

1. `exp/exp8_3_steer_usage_diagnostics` 已经完成诊断闭环，应该保留为只读诊断基线。
2. 接下来要同时改 reset 和 reward 入口，属于新阶段实现，不应继续堆在诊断分支上。
3. 后续如果需要把 PPO 方差控制单独拿出来做，可以再从同一基线切出独立分支。

---

### 总体策略

最有效的推进顺序是这 4 步：

1. 先把 `stage1` 变成“必须转向才能成功”的课程
2. 把 reward 的重点前移到“插入前纠偏”
3. 把 `zero-steer gap` 变成新的验收标准
4. 在 steering 学出来之前，不再把主精力放在长训练和 bonus 扫参上

---

### 第 1 步：放宽 stage1 reset，但先不拉远 x

第一版只改 `y / yaw`，不动 `x`。

当前默认：

- `stage1_init_x_min_m = -3.55`
- `stage1_init_x_max_m = -3.25`
- `stage1_init_y_min_m = -0.05`
- `stage1_init_y_max_m = 0.05`
- `stage1_init_yaw_deg_min = -2.0`
- `stage1_init_yaw_deg_max = 2.0`

第一版课程改成：

- `y: ±0.05 -> ±0.10`
- `yaw: ±2° -> ±4°`
- `x` 保持不变

如果这版仍然能靠直着往前叉成功，再推进到第二版：

- `y: ±0.15`
- `yaw: ±6°`

设计原则：

- 不先把问题混成“远距离 approach + steering + insert”三件事一起学
- 先只让 near-field 中的 steering 变成必要技能

---

### 第 2 步：把 reward 重点前移到 pre-insert correction

当前 reward 主线的问题是：

- `clean_insert_push_free_bonus` 生效太晚
- clean signal 主要在“已经插进去且 push-free”后才明显

这会导致策略在早期更容易学到：

- 直接前推
- dirty insert
- 或 retreat / no-insert

第一版 pre-insert shaping 要补的是：

1. 奖励 `|y_err|` 下降
2. 奖励 `|yaw_err|` 下降
3. 奖励 `dist_front` 下降，但惩罚 near-field retreat

具体实现原则：

- 只在 `pre-insert` 阶段启用，不直接影响 post-insert clean gate 主体
- 用 delta shaping，奖励“相对上一步变好”
- reset 时精确初始化 prev cache，避免首步白嫖

建议第一版新增配置项：

- `preinsert_align_reward_enable`
- `preinsert_active_dist_max_m`
- `preinsert_insert_frac_max`
- `preinsert_y_err_delta_weight`
- `preinsert_yaw_err_delta_weight`
- `preinsert_dist_front_delta_weight`
- `preinsert_retreat_penalty_weight`
- `preinsert_delta_clip_y_m`
- `preinsert_delta_clip_yaw_deg`
- `preinsert_delta_clip_dist_m`

建议第一版默认逻辑：

- 仅在 `dist_front < 某阈值` 且 `insert_norm < 某阈值` 时启用
- `delta_y = prev_y_err - y_err`
- `delta_yaw = prev_yaw_err_deg - yaw_err_deg`
- `delta_front = prev_dist_front - dist_front`
- 正向只奖励改善项
- 负向显式惩罚 retreat

---

### 第 3 步：把 zero-steer gap 变成硬验收标准

从这一阶段开始，一个配置是否“有效”，不能只看训练里的 `success`。

必须同时看：

1. normal policy 的 `misalignment grid` 成功率
2. zero-steer policy 的 `misalignment grid` 成功率

真正想要的不是“normal 也能成功”，而是：

**normal 明显强于 zero-steer。**

建议定义新的关键验收口径：

- `normal_success_rate_grid`
- `zero_steer_success_rate_grid`
- `steer_gap = normal_success_rate_grid - zero_steer_success_rate_grid`

第一阶段目标不是追求绝对高 success，而是先把：

- `steer_gap` 从接近 `0`
- 拉到明显大于 `0`

---

### 第 4 步：在 steering 学出来之前，不再做什么

在 steering-based correction 没有被学出来之前，不建议优先做：

1. 回 wide reset
2. 加重 `gate / dirty penalty`
3. 继续扫更多 `bonus weight`
4. 直接上 `200 / 400 iter` 长训练

原因很简单：

- 这些动作大概率只会把“前推捷径”学得更稳
- 不会真正扩大 steering-based good basin

---

### 具体执行顺序

#### Phase A：代码实现

1. 从 `exp/exp8_3_steer_usage_diagnostics` 切 `exp/exp8_3_stage1_steering_curriculum`
2. 修改 `stage1` 默认 reset：
   - `y -> ±0.10`
   - `yaw -> ±4°`
3. 在 reward 中加入第一版 pre-insert delta shaping
4. 在 reset 中正确初始化 `prev_y_err / prev_yaw_err / prev_dist_front`
5. 补日志：
   - `paper_reward/r_preinsert_align`
   - `paper_reward/r_preinsert_retreat`
   - `diag/preinsert_active_frac`
   - `diag/preinsert_y_delta_mean`
   - `diag/preinsert_yaw_delta_mean`
   - `diag/preinsert_dist_delta_mean`

#### Phase B：最小验证

1. 语法检查
2. 2-iter smoke
3. 确认新日志项能正常打出

#### Phase C：短训验证

1. `50 iter / 256x256 / near-field`
2. 先跑 `3 seeds`
3. 不看 `mean_reward`
4. 主要看：
   - `phase/frac_success`
   - `phase/frac_inserted_push_free`
   - `phase/frac_dirty_insert`
   - `diag/pallet_disp_xy_mean`
   - `paper_reward/r_preinsert_align`
   - `paper_reward/r_preinsert_retreat`

#### Phase D：诊断回归

训练完后必须做：

1. normal `misalignment grid`
2. zero-steer `misalignment grid`

只有当 `steer_gap` 明显大于当前诊断基线时，才说明这版真正逼出了 steering。

---

### 本阶段成功标准

本阶段的成功标准不是“最终成功率立刻最高”，而是：

1. normal vs zero-steer 出现明显 gap
2. steering 不再只是接近 0 的装饰动作
3. success basin 不再完全贴着当前超对齐 reset

只要做到这 3 点，就说明我们已经真正开始让策略学“纠偏后插入”，而不是“正着往前推”。

---

### 预期风险

1. 第一版 reset 放宽后，短期成功率可能下降
2. pre-insert shaping 权重过大时，可能把策略推向 retreat
3. 如果 zero-steer gap 仍接近 0，说明 reset 还不够宽或者 shaping 入口仍太晚

这三种都不意味着方向错了，只意味着还需要继续调“必须 steering”的门槛。

---

### 一句话总结

接下来这条线的核心不是“再调出一个更高 success 的 bonus 配方”，而是：

**先把 near-field 课程改成不会 steering 就过不了，再用 zero-steer gap 验证策略是否真的学会了转向纠偏。**
