# Exp8.3 Wrong-Sign Abort 与 Half-Steer Ablation 结果

日期：2026-03-31

## 1. 背景

这一轮的目标不是继续同时改很多 reward，而是回答一个更具体的问题：

> 当前 `normal steering` 为什么仍然不如 `zero-steer`？

为了把问题进一步压缩，我先完成了两层控制变量：

1. 固定 `wrong-sign abort` 版本，比较 `normal` 和 `zero-steer`
2. 在同一个 checkpoint 上，再补 `half-steer`

这样能把问题拆成：

- 是不是“根本不该 steering”
- 还是“steering 幅度太大”

## 2. 这次实验固定了什么

### 2.1 固定条件

- 分支：`exp/exp8_3_force_steering_curriculum`
- checkpoint：
  - `IsaacLab/logs/rsl_rl/forklift_pallet_insert_lift/2026-03-31_09-48-47_exp83_wrong_sign_abort_seed42_iter50_256cam/model_49.pt`
- 网格评估：
  - `x_root = -3.40`
  - `y ∈ {-0.10, 0.0, +0.10}`
  - `yaw ∈ {-4°, 0°, +4°}`
  - `episodes_per_point = 1`

### 2.2 唯一变化的变量

- `normal`：按训练环境默认 steering 执行
- `zero-steer`：把 applied steer 强制设为 `0`
- `half-steer`：把 applied steer 缩放到 `0.5`

## 3. 结果

### 3.1 `normal`

文件：

- `outputs/exp83_wrong_sign_abort_seed42_3x3/exp83_wrong_sign_abort_seed42_iter50_recheck3x3_normal_normal_summary.json`

关键指标：

- `success_rate = 1/9 = 0.1111`
- `ever_inserted_rate = 3/9 = 0.3333`
- `ever_inserted_push_free_rate = 0/9 = 0.0`
- `ever_hold_entry_rate = 2/9 = 0.2222`
- `ever_clean_insert_ready_rate = 0/9 = 0.0`
- `ever_dirty_insert_rate = 3/9 = 0.3333`
- `timeout_frac = 2/9 = 0.2222`
- `mean_abs_steer_applied = 0.3080`

### 3.2 `zero-steer`

文件：

- `outputs/exp83_wrong_sign_abort_seed42_3x3/exp83_wrong_sign_abort_seed42_iter50_recheck3x3_zero_zero_steer_summary.json`

关键指标：

- `success_rate = 4/9 = 0.4444`
- `ever_inserted_rate = 5/9 = 0.5556`
- `ever_inserted_push_free_rate = 3/9 = 0.3333`
- `ever_hold_entry_rate = 4/9 = 0.4444`
- `ever_clean_insert_ready_rate = 3/9 = 0.3333`
- `ever_dirty_insert_rate = 2/9 = 0.2222`
- `timeout_frac = 5/9 = 0.5556`
- `mean_abs_steer_applied = 0.0`

### 3.3 `half-steer`

文件：

- `outputs/exp83_wrong_sign_abort_seed42_3x3/exp83_wrong_sign_abort_seed42_iter50_recheck3x3_half_half_steer_summary.json`

关键指标：

- `success_rate = 4/9 = 0.4444`
- `ever_inserted_rate = 4/9 = 0.4444`
- `ever_inserted_push_free_rate = 3/9 = 0.3333`
- `ever_hold_entry_rate = 4/9 = 0.4444`
- `ever_clean_insert_ready_rate = 3/9 = 0.3333`
- `ever_dirty_insert_rate = 1/9 = 0.1111`
- `timeout_frac = 0/9 = 0.0`
- `mean_abs_steer_applied = 0.1575`

## 4. 这组结果真正说明了什么

### 4.1 `wrong-sign abort` 不是白做，但它不是主解

它解决的是：

- 极端错误 steering 的 safety 问题

它没有解决的是：

- `normal steering` 如何稳定优于 `zero-steer`

也就是说，它更像是护栏，不是主导 steering 学习的主因。

### 4.2 当前最像的主问题是：`stage1 applied steer` 幅度太大

最关键的对照是：

- `normal = 1/9 success`
- `zero-steer = 4/9 success`
- `half-steer = 4/9 success`

如果问题是“完全不该 steering”，那 `half-steer` 应该仍然差。

但实际不是：

- `half-steer` 直接追平了 `zero-steer` 的 success
- 而且比 `zero-steer` 更少 dirty insert
- 也没有出现 `zero-steer` 那种大量 timeout

所以当前更合理的解释是：

> 不是 policy 一 steering 就必错，而是当前 `stage1` 的 applied steer 太猛，很多本来可以 clean 的近场轨迹，被过大的 steering 幅度自己打坏了。

### 4.3 这也解释了为什么 `normal` 比 `zero-steer` 更差

当前 `normal` 的 mean applied steer 约 `0.308`，而 `half-steer` 降到约 `0.157` 后，成功率明显改善。

这说明：

- steering 并不是完全没用
- 但当前幅度让它在近场更容易把车打出 clean corridor

### 4.4 但 `half-steer` 还没有解决“steering 符号偏置”

逐点看 9 个格点，`half-steer` 的表现有一个更细的特征：

- 对于 `yaw=+4°` 这一侧，`half-steer` 明显优于 `normal`
- 对于中心附近点，`half-steer` 也能拿到 clean success
- 但对于 `yaw=-4°, y=+0.10` 这个点，`zero-steer` 能 clean success，而 `half-steer` 仍然失败

更关键的是：

- `normal` 的 `mean_steer_applied` 在 9 个点上全部是正值
- `half-steer` 的 `mean_steer_applied` 在 9 个点上也全部是正值，只是幅度减半

这说明：

> `half-steer` 并没有真正解决“steering 会不会按错位方向翻转”这个问题；它更像是把原本过大的单边 steering 偏置，压到了一个不那么伤的范围。

所以当前最准确的理解不是：

- “问题已经解决，只差训练时间”

而是：

- “幅度问题被部分识别出来了，但符号/时机问题仍在”

## 5. 由此产生的单因素决策

### 5.1 已执行的改动

已经把这个结论接成真实环境配置：

- `stage1_steer_action_scale = 0.5`

对应提交：

- IsaacLab `4eee480` `Dampen stage1 steering amplitude`

### 5.2 当前正在验证的实验

正在运行：

- `IsaacLab/logs/rsl_rl/forklift_pallet_insert_lift/2026-03-31_10-51-47_exp83_stage1_halfsteer_seed42_iter50_256cam`

它的目的不是“再看一次能不能插”。

它要验证的是：

> 把 `half-steer` 从 eval 里的人工缩放，变成训练环境的真实默认配置后，能不能把这个优势稳定转成训练结果。

### 5.3 这轮训练结束后的唯一验收方式

下一步只做这套统一验收：

1. `3x3 normal`
2. `3x3 zero-steer`

通过标准：

- `normal` 不再明显弱于 `zero-steer`
- `normal` 的 `success` 至少接近 `zero-steer`
- `push_free / clean_insert_ready / hold_entry` 必须是非零

## 6. 接下来仍然保持控制变量

### 6.1 如果 `stage1_steer_action_scale=0.5` 通过

下一步：

- 不改 reward
- 不改 reset
- 直接做 `3 seeds x 50 iter`

### 6.2 如果 `0.5` 有改善，但仍未过线

下一步只扫一个变量：

- `stage1_steer_action_scale`

候选：

- `0.35`
- `0.65`

但这一步的目的要非常明确：

- 不是盲目继续扫
- 而是验证：当前主要是不是仍然以 gain 为主因

如果 `0.35 / 0.65` 都不能把 `normal` 拉到明显不弱于 `zero-steer`，就要停止继续扫 gain，转去新的单因素问题：

- signed steering bias
- 或 steering 生效时机

### 6.3 当前明确不做

- 不继续扫 reward weight
- 不再同时改 gain 和 gate
- 不再同时改 reset 和 steering
- 不把问题重新扩大成 perception/reward/reset 三线混合排查

## 7. 当前一句话总结

到目前为止，最扎实的单因素结论是：

> `wrong-sign abort` 解决了部分错误 steering 的安全问题，但真正限制 `normal` 表现的更可能是 `stage1` 下 steering 幅度偏大；`half-steer` 已经在固定 checkpoint 的 3x3 对照里证明这是当前最值得优先验证的变量。
