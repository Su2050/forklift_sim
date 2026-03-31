# Exp8.3 Stage1 Sign-Safe Clip 单因素结果

日期：2026-03-31

## 1. 实验目的

上一轮 `stage1_steer_action_scale=0.75` 已经证明了一件事：

- `normal` 从 `1/9` 提高到了 `4/9`
- 但仍然是 `zero-steer = 5/9` 更强

因此下一步不应该继续扫 scale，而应该先回答一个更具体的问题：

> 如果只修正 stage1 preinsert 的 wrong-sign steering，让 applied steer 变成 sign-safe，`normal` 能不能第一次追平甚至反超 `zero-steer`？

## 2. 控制变量

固定不动：

- 分支：`exp/exp8_3_force_steering_curriculum`
- seed：`42`
- camera：`256x256`
- 训练长度：`50 iter`
- `stage1_steer_action_scale = 0.75`
- reward / reset / observation：不变

唯一新增的变量：

- `stage1_clip_wrong_sign_steer_enable = True`

对应代码提交：

- IsaacLab `270600f` `Clip wrong-sign steering in stage1 preinsert`

## 3. 改动内容

这次没有再改 reward，也没有再动 reset。

只做了一件事：

- 在 stage1 preinsert，如果当前 applied steer 和 `steer_target` 的方向相反，就把这一拍的 applied steer 直接裁成 `0`

也就是说，这次不是“给 agent 更多 steering 奖励”，而是：

- 不允许它在最关键的 preinsert 阶段持续把 steering 打到明显错误的方向

同时补了一个训练诊断口：

- `diag/preinsert_wrong_sign_clipped_frac`

## 4. 训练结果

运行目录：

- `IsaacLab/logs/rsl_rl/forklift_pallet_insert_lift/2026-03-31_12-04-55_exp83_stage1_signsafe_clip_seed42_iter50_256cam`

checkpoint：

- `IsaacLab/logs/rsl_rl/forklift_pallet_insert_lift/2026-03-31_12-04-55_exp83_stage1_signsafe_clip_seed42_iter50_256cam/model_49.pt`

### 4.1 训练中段的信号明显更干净

几个最关键的现象：

- `diag/preinsert_wrong_sign_clipped_frac` 持续非零，说明 sign-safe clip 真的在介入
- 中段多次出现 `inserted_push_free / clean_insert_ready / hold_entry`
- 多个 clean/hold 高点对应的 `dirty_insert` 被压到了 `0`

例如在 `iter 24`：

- `phase/frac_inserted = 0.0625`
- `phase/frac_inserted_push_free = 0.0625`
- `phase/frac_clean_insert_ready = 0.0625`
- `phase/frac_hold_entry = 0.0625`
- `phase/frac_dirty_insert = 0.0000`

这和上一轮 `0.75` 的重要差别是：

- 上一轮会反复掉回 dirty insert
- 这轮在出现 clean/hold 的时刻，dirty 被明显压低了

### 4.2 但训练末尾没有把中段高点完全稳住

最终 `iter 49` 的标量是：

- `phase/frac_inserted = 0.0000`
- `phase/frac_inserted_push_free = 0.0000`
- `phase/frac_clean_insert_ready = 0.0000`
- `phase/frac_hold_entry = 0.0000`
- `phase/frac_success = 0.0000`

所以这轮不能说“训练已经稳定收敛”，更准确地说是：

- 它显著改善了中段行为
- 但最终是否真的更好，必须靠固定 grid eval 来看

## 5. 固定 `3x3` 验收

验收条件保持完全不变：

- `x_root = -3.40`
- `y ∈ {-0.10, 0.0, +0.10}`
- `yaw ∈ {-4°, 0°, +4°}`
- `episodes_per_point = 1`

### 5.1 `3x3 normal`

文件：

- `outputs/exp83_stage1_signsafe_clip_seed42_3x3/exp83_stage1_signsafe_clip_seed42_iter50_recheck3x3_normal_normal_summary.json`
- `outputs/exp83_stage1_signsafe_clip_seed42_3x3/exp83_stage1_signsafe_clip_seed42_iter50_recheck3x3_normal_normal_rows.csv`

结果：

- `success = 5/9 = 0.5556`
- `inserted = 6/9`
- `push_free = 4/9`
- `hold = 5/9`
- `clean = 4/9`
- `dirty = 2/9`
- `timeout = 4/9`
- `mean_abs_steer_applied = 0.0159`

### 5.2 `3x3 zero-steer`

文件：

- `outputs/exp83_stage1_signsafe_clip_seed42_3x3/exp83_stage1_signsafe_clip_seed42_iter50_recheck3x3_zero_zero_steer_summary.json`
- `outputs/exp83_stage1_signsafe_clip_seed42_3x3/exp83_stage1_signsafe_clip_seed42_iter50_recheck3x3_zero_zero_steer_rows.csv`

结果：

- `success = 5/9 = 0.5556`
- `inserted = 5/9`
- `push_free = 4/9`
- `hold = 5/9`
- `clean = 4/9`
- `dirty = 1/9`
- `timeout = 4/9`
- `mean_abs_steer_applied = 0.0`

## 6. 这轮最关键的发现

### 6.1 `normal` 第一次追平了 `zero-steer`

这轮最重要的结论非常明确：

- 上一轮：`normal = 4/9`, `zero-steer = 5/9`
- 这轮：`normal = 5/9`, `zero-steer = 5/9`

也就是说：

- sign-safe clip 没有把系统推成更差
- 相反，它把 `normal` 从“落后”推到了“追平”

这说明当前主因确实包含：

- wrong-sign steering 在伤害 stage1 normal policy

### 6.2 它修复的不是全部格点，而是一个非常具体的分叉点

和上一轮 `0.75 normal` 相比，这次最关键的逐点变化是：

- `y=-0.10, yaw=0°` 这个点，从 `timeout/dirty` 变成了 `clean success`

其余主要成功点：

- `y=+0.10, yaw=-4°`
- `y=-0.10, yaw=0°`
- `y=0.0, yaw=0°`
- `y=+0.10, yaw=0°`
- `y=-0.10, yaw=+4°`

失败点仍然集中在：

- `y=-0.10, yaw=-4°`
- `y=0.0, yaw=+4°`
- `y=+0.10, yaw=+4°`

所以这轮的本质不是“整体完全变了”，而是：

- 它把一个最典型的 wrong-sign 受害点拉回来了

### 6.3 更重要的是：policy 的 steer sign 不再是“9 个点全正”

上一轮 `0.75` 的一个核心问题是：

- 9 个格点上的 `mean_steer_raw` 几乎全是正数

这轮已经不是这样了。

在 `normal rows` 里：

- `y=0.0, yaw=+4°` 上 `mean_steer_raw ≈ -0.0006`
- `y=+0.10, yaw=+4°` 上 `mean_steer_raw ≈ -0.0024`

虽然幅度还很小，但这说明：

- policy 的 steer sign 已经不再是“所有格点统一单边正偏”
- sign-safe clip 至少把策略从“只会一边打方向”往“开始出现符号分化”推了一步

## 7. 当前还没有解决的事

### 7.1 `normal` 只是追平，不是反超

这轮不能宣告 steering 已经学出来，因为：

- `normal = 5/9`
- `zero-steer = 5/9`

这仍然不满足我们更严格的目标：

> `normal` 必须明显强于 `zero-steer`

### 7.2 当前 steering 幅度已经很小

这轮一个很重要的新现象是：

- `normal mean_abs_steer_applied` 只有 `0.0159`
- 上一轮 `0.75 normal` 是 `0.0403`

这说明 sign-safe clip 之后，系统已经明显更保守了。

换句话说，这轮的成功更像是：

- 先把错误 steering 削掉
- 还没有把“有益 steering”真正放大出来

## 8. 控制变量下的下一步

现在最合理的下一步，不是继续扫别的奖励权重，也不是立刻多 seed。

更干净的下一刀是：

- 保留 `stage1_clip_wrong_sign_steer_enable = True`
- 其他 reward / reset / observation 全不变
- 只把 `stage1_steer_action_scale` 从 `0.75` 回提到 `1.0`

原因很直接：

- 过去的 `1.0` 失败，很大一部分是因为 wrong-sign steering 会直接伤害 normal policy
- 现在 wrong-sign 已经有 sign-safe 保护
- 而当前 `normal` 的 applied steer 幅度又已经变得很小

所以新的关键问题变成了：

> 在 sign-safe 保护还在的前提下，适当把 steering 幅度放回去，能不能让 `normal` 从“追平 zero-steer”进一步走到“反超 zero-steer”？

## 9. 一句话总结

> sign-safe clip 这轮已经把 `normal` 从 `4/9` 提高到了 `5/9`，并且第一次追平了 `zero-steer`；同时 policy 的 steer sign 也开始出现真正的正负分化。当前最合理的下一刀，不是继续扫奖励，而是在保留 sign-safe 的前提下，把 `stage1_steer_action_scale` 从 `0.75` 回提到 `1.0`，测试“更安全的 full-scale steering”能否把 `normal` 推到真正强于 `zero-steer`。
