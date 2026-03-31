# Exp8.3 Wrong-Sign Abort Frozen Checkpoint Steer-Scale Sweep

日期：2026-03-31

## 1. 目的

在 `stage1_steer_action_scale=0.5` 的 train-time 版本提前失败之后，需要先回答一个更具体的问题：

> 对同一个 frozen checkpoint，推理时 steering scale 的可用区间到底在哪里？

如果这件事不先钉清楚，后面继续开 train-time gain 实验就容易盲扫。

## 2. 固定条件

- checkpoint 固定：
  - `IsaacLab/logs/rsl_rl/forklift_pallet_insert_lift/2026-03-31_09-48-47_exp83_wrong_sign_abort_seed42_iter50_256cam/model_49.pt`
- grid 固定：
  - `x_root = -3.40`
  - `y ∈ {-0.10, 0.0, +0.10}`
  - `yaw ∈ {-4°, 0°, +4°}`
  - `episodes_per_point = 1`

唯一变化的变量：

- `steer_scale`

## 3. 扫描结果

### 3.1 `steer_scale = 1.0`

文件：

- `outputs/exp83_wrong_sign_abort_seed42_3x3/exp83_wrong_sign_abort_seed42_iter50_recheck3x3_normal_normal_summary.json`

结果：

- `success = 1/9 = 0.1111`
- `inserted = 3/9`
- `push_free = 0/9`
- `clean = 0/9`
- `dirty = 3/9`
- `timeout = 2/9`
- `mean_abs_steer_applied = 0.3080`

### 3.2 `steer_scale = 0.75`

文件：

- `outputs/exp83_wrong_sign_abort_seed42_3x3/exp83_wrong_sign_abort_seed42_iter50_recheck3x3_scale075_steer_scale_p0p75_summary.json`

结果：

- `success = 4/9 = 0.4444`
- `inserted = 4/9`
- `push_free = 3/9`
- `clean = 3/9`
- `dirty = 1/9`
- `timeout = 0/9`
- `mean_abs_steer_applied = 0.2374`

### 3.3 `steer_scale = 0.5`

文件：

- `outputs/exp83_wrong_sign_abort_seed42_3x3/exp83_wrong_sign_abort_seed42_iter50_recheck3x3_half_half_steer_summary.json`

结果：

- `success = 4/9 = 0.4444`
- `inserted = 4/9`
- `push_free = 3/9`
- `clean = 3/9`
- `dirty = 1/9`
- `timeout = 0/9`
- `mean_abs_steer_applied = 0.1575`

### 3.4 `steer_scale = 0.25`

文件：

- `outputs/exp83_wrong_sign_abort_seed42_3x3/exp83_wrong_sign_abort_seed42_iter50_recheck3x3_scale025_steer_scale_p0p25_summary.json`

结果：

- `success = 4/9 = 0.4444`
- `inserted = 5/9`
- `push_free = 3/9`
- `clean = 3/9`
- `dirty = 2/9`
- `timeout = 5/9`
- `mean_abs_steer_applied = 0.0820`

### 3.5 `steer_scale = 0.0`

文件：

- `outputs/exp83_wrong_sign_abort_seed42_3x3/exp83_wrong_sign_abort_seed42_iter50_recheck3x3_zero_zero_steer_summary.json`

结果：

- `success = 4/9 = 0.4444`
- `inserted = 5/9`
- `push_free = 3/9`
- `clean = 3/9`
- `dirty = 2/9`
- `timeout = 5/9`
- `mean_abs_steer_applied = 0.0`

## 4. 这条曲线说明了什么

### 4.1 `1.0` 明显过大

这点已经非常明确：

- success 最低
- clean / push-free 为 0
- dirty insert 明显偏多

所以 `1.0` 不是合理的 stage1 applied steer 默认值。

### 4.2 `0.25` 太接近 `zero-steer`

`0.25` 和 `0.0` 的行为非常像：

- success 都是 `4/9`
- clean 都是 `3/9`
- timeout 都很高
- dirty 也都偏高

这说明：

- scale 太小时，系统又会退回到“近似 straight-in” 的行为
- steering 虽然没完全消失，但已经不够主导近场纠偏

### 4.3 `0.5` 和 `0.75` 落在更好的 plateau 上

`0.5` 和 `0.75` 有两个共同特征：

- 都达到 `4/9 success`
- 都有 `3/9 clean`
- 都把 `timeout` 压到 `0`
- 都把 `dirty_insert` 降到 `1/9`

这说明 frozen-checkpoint 的可用区间不是一个单点，而是一个大致的 plateau：

- 约在 `0.5 ~ 0.75`

### 4.4 在 train-time 候选里，`0.75` 比 `0.5` 更合理

原因不是因为 `0.75` 的 success 更高，而是因为：

- `0.5` 已经被真实 train-time 实验验证为“过于保守”，在 `iter 22` 仍然 `inserted = 0`
- `0.75` 保留了和 `0.5` 一样的 frozen-checkpoint 成功率和 clean 水平
- 但 applied steer 更大，更不容易在训练早期直接塌成“完全不插”

因此下一步最合理的单因素训练实验不是：

- 再试 `0.5`
- 或直接跳去更复杂的 gate/schedule

而是：

- 先试 `stage1_steer_action_scale = 0.75`

## 5. 控制变量下的下一步

下一轮训练建议固定不动：

- reward
- reset
- observation
- hold/success 逻辑

只改一个变量：

- `stage1_steer_action_scale: 0.5 -> 0.75`

然后跑：

- `seed42`
- `50 iter`
- `256x256`

验收仍然固定：

- `3x3 normal`
- `3x3 zero-steer`

## 6. 一句话总结

> frozen-checkpoint 的 steer-scale 曲线已经说明：`1.0` 明显过大，`0.25` 又太接近 zero-steer，而 `0.5 ~ 0.75` 是更合理的可用区间；结合 train-time `0.5` 已经早停失败，下一轮最合理的单因素训练实验应是 `stage1_steer_action_scale = 0.75`。
