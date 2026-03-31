# Exp8.3 Stage1 `steer_scale=0.75` 单因素训练结果

日期：2026-03-31

## 1. 实验目的

这轮实验只回答一个问题：

> 在已经确认 `1.0` 过大、`0.5` train-time 过于保守之后，把 `stage1_steer_action_scale` 固定到 `0.75`，能不能让训练出来的 `normal policy` 真正强于 `zero-steer`？

这轮必须和前面的 scale sweep 区分开：

- frozen-checkpoint sweep 回答的是“推理时可用区间”
- 这轮 train-time 实验回答的是“把这个 scale 变成真实训练默认值之后，是否会把 steering 学成有效能力”

对应代码提交：

- IsaacLab `e7de931` `Raise stage1 steering scale to 0.75`

## 2. 控制变量

固定不动：

- 分支：`exp/exp8_3_force_steering_curriculum`
- seed：`42`
- camera：`256x256`
- 训练长度：`50 iter`
- reward / hold / success 逻辑：不变
- stage1 reset：不变
- observation：不变

唯一变化的变量：

- `stage1_steer_action_scale: 0.5 -> 0.75`

## 3. 训练运行信息

- run dir：
  - `IsaacLab/logs/rsl_rl/forklift_pallet_insert_lift/2026-03-31_11-15-24_exp83_stage1_steer075_seed42_iter50_256cam`
- checkpoint：
  - `IsaacLab/logs/rsl_rl/forklift_pallet_insert_lift/2026-03-31_11-15-24_exp83_stage1_steer075_seed42_iter50_256cam/model_49.pt`

## 4. 训练阶段结果

### 4.1 中途确实长出了 clean / hold / success

和 `stage1_steer_action_scale=0.5` 的 train-time 版本相比，`0.75` 明显不是“完全闷死”的。

例如在中后段，训练里反复出现过：

- `phase/frac_inserted_push_free = 0.0156 ~ 0.0312`
- `phase/frac_clean_insert_ready = 0.0156 ~ 0.0312`
- `phase/frac_hold_entry = 0.0156 ~ 0.0312`
- `phase/frac_success = 0.0156`

这说明：

- `0.75` 至少把策略推到了 good basin 边界附近
- 它比 `0.5` 更像“能活下去的 train-time 默认值”

### 4.2 但到最后仍然没有稳住

最终 `iter 49` 的关键标量是：

- `phase/frac_inserted = 0.0625`
- `phase/frac_inserted_push_free = 0.0156`
- `phase/frac_clean_insert_ready = 0.0156`
- `phase/frac_hold_entry = 0.0156`
- `phase/frac_success = 0.0156`

同时：

- `phase/frac_dirty_insert = 0.0469`
- `diag/preinsert_wrong_sign_abort_frac = 0.0156`

这说明最后的状态不是“完全 clean”，而是：

- 仍然会反复摸到 clean / hold
- 但也仍然会掉回 dirty insert

所以单看训练标量，只能说 `0.75` 比 `0.5` 明显更合理，还不能说它已经把 steering 学成了稳定能力。

## 5. 固定 `3x3` 验收

为了避免只看训练标量猜结论，验收仍然固定为同一套 `3x3`：

- `x_root = -3.40`
- `y ∈ {-0.10, 0.0, +0.10}`
- `yaw ∈ {-4°, 0°, +4°}`
- `episodes_per_point = 1`

### 5.1 `3x3 normal`

文件：

- `outputs/exp83_stage1_steer075_seed42_3x3/exp83_stage1_steer075_seed42_iter50_recheck3x3_normal_normal_summary.json`
- `outputs/exp83_stage1_steer075_seed42_3x3/exp83_stage1_steer075_seed42_iter50_recheck3x3_normal_normal_rows.csv`

结果：

- `success = 4/9 = 0.4444`
- `inserted = 6/9`
- `push_free = 3/9`
- `hold = 4/9`
- `clean = 3/9`
- `dirty = 3/9`
- `timeout = 5/9`
- `mean_abs_steer_applied = 0.0403`

### 5.2 `3x3 zero-steer`

文件：

- `outputs/exp83_stage1_steer075_seed42_3x3/exp83_stage1_steer075_seed42_iter50_recheck3x3_zero_zero_steer_summary.json`
- `outputs/exp83_stage1_steer075_seed42_3x3/exp83_stage1_steer075_seed42_iter50_recheck3x3_zero_zero_steer_rows.csv`

结果：

- `success = 5/9 = 0.5556`
- `inserted = 5/9`
- `push_free = 4/9`
- `hold = 5/9`
- `clean = 4/9`
- `dirty = 2/9`
- `timeout = 4/9`
- `mean_abs_steer_applied = 0.0`

## 6. 这轮结果说明了什么

### 6.1 `0.75` train-time 确实比旧版强

和 `wrong-sign abort + scale=1.0` 那版相比，这次最重要的正向变化是：

- `normal success` 从 `1/9` 提高到 `4/9`
- `normal clean` 从 `0/9` 提高到 `3/9`
- `normal push-free` 从 `0/9` 提高到 `3/9`

也就是说：

- 把 full-steer 改成 `0.75`，不只是“训练标量好看一点”
- 它确实把最终 checkpoint 的 `normal` 可用性明显抬起来了

### 6.2 但它仍然没有通过真正的门槛

真正的验收门槛一直没变：

> `normal` 必须明显强于 `zero-steer`

这轮并没有过：

- `normal = 4/9`
- `zero-steer = 5/9`

所以当前最准确的判断不是“scale 问题已经解决”，而是：

- `0.75` 解决了“steer 幅度过大导致纯粹伤害”的一部分问题
- 但它还没有把 steering 变成真正必要、真正有益的能力

### 6.3 更关键的是：policy 仍然在输出单边正 steering bias

看逐点 `rows.csv`，有一个非常关键的现象：

- `normal` 9 个格点上的 `mean_steer_raw` 全都是正数，约 `0.036 ~ 0.044`
- `zero-steer` 9 个格点上的 `mean_steer_raw` 也全都是正数，约 `0.040 ~ 0.044`

也就是说：

- 训练时把 scale 改成 `0.75`，主要是在减轻这个固定正偏 steering 的伤害
- 但 policy 本身并没有学会“根据 `y/yaw` 正负切换左右转向”

所以当前主瓶颈已经进一步缩小为：

> `0.75` 解决的是 amplitude 过大，不是 steering sign 语义本身。

## 7. 从控制变量角度，下一步不该做什么

这轮之后，有三件事不建议继续做：

- 不建议继续扫更多 `steer_scale`
- 不建议直接把这版往几百 iter 拉长
- 不建议把 scale 结论误读成“steering 已经学出来了”

原因很简单：

- amplitude 这条线已经基本回答完了
- train-time `0.75` 虽然明显优于 `1.0`
- 但仍然是 `zero-steer > normal`

继续在 amplitude 上多扫几个点，信息增益会很低。

## 8. 下一步最合理的单因素实验

下一步建议固定不动：

- `stage1_steer_action_scale = 0.75`
- reward
- reset
- observation

只改一个变量：

- 在 stage1 preinsert，把 applied steer 做成“sign-safe”

更具体地说，下一刀应当瞄准：

- 不再继续改 amplitude
- 而是防止 policy 在 preinsert 阶段把 steering 长期打到与 `steer_target` 语义不一致的方向

实验目的不是直接做最终方案，而是先回答一个更具体的问题：

> 如果只把 stage1 steering 的 sign 语义保护好，`normal` 能不能第一次稳定反超 `zero-steer`？

## 9. 一句话总结

> `stage1_steer_action_scale = 0.75` 这轮已经证明：它能明显减轻 full-steer 的伤害，并把 `normal` 从 `1/9` 拉到 `4/9`；但由于 policy 仍然在 9 个格点上输出单边正 steering bias，最终仍是 `zero-steer (5/9) > normal (4/9)`。因此 amplitude 这条线可以基本收住，下一步最该改的是 steering sign 语义，而不是继续扫 scale 或直接长训。
