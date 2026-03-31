# Exp8.3 Stage1 Sign-Safe + Full-Scale 早停记录

日期：2026-03-31

## 1. 实验目的

上一轮 `sign-safe clip + steer_scale=0.75` 已经把 `normal` 从 `4/9` 提高到 `5/9`，并第一次追平了 `zero-steer`。

因此下一步最合理的单因素验证是：

> 保留 sign-safe clip，只把 `stage1_steer_action_scale` 从 `0.75` 回提到 `1.0`，看看“更安全的 full-scale steering”能不能把 `normal` 从追平推进到反超。

对应代码提交：

- IsaacLab `8facb15` `Restore stage1 full steer under sign-safe clip`

## 2. 控制变量

固定不动：

- `stage1_clip_wrong_sign_steer_enable = True`
- reward / reset / observation：不变
- seed：`42`
- camera：`256x256`

唯一变化的变量：

- `stage1_steer_action_scale: 0.75 -> 1.0`

## 3. 运行信息

- run dir：
  - `IsaacLab/logs/rsl_rl/forklift_pallet_insert_lift/2026-03-31_12-46-35_exp83_stage1_signsafe_fullscale_seed42_iter50_256cam`

这轮原计划跑满 `50 iter`，但最终在 `iter 23` 提前停掉。

## 4. 为什么提前停掉

这轮不是一开始就坏掉。

相反，在早期它一度给出过“full-scale 也许真的有机会”的信号：

- `iter 13` 曾出现：
  - `phase/frac_inserted = 0.0781`
  - `phase/frac_inserted_push_free = 0.0156`
  - `phase/frac_clean_insert_ready = 0.0156`
  - `phase/frac_hold_entry = 0.0781`
  - `phase/frac_success = 0.0312`

但继续往后跑，趋势没有往 clean basin 稳定，反而越来越明显地滑向 dirty basin。

到 `iter 23`，最新有效标量已经变成：

- `phase/frac_inserted = 0.5312`
- `phase/frac_inserted_push_free = 0.0000`
- `phase/frac_clean_insert_ready = 0.0000`
- `phase/frac_hold_entry = 0.0000`
- `phase/frac_success = 0.0000`
- `phase/frac_dirty_insert = 0.5312`
- `phase/frac_push_free = 0.3125`
- `traj/d_traj_mean = 0.3221`
- `traj/yaw_traj_deg_mean = 10.6862`

这已经非常清楚地说明：

- insertion 的确更多了
- 但几乎全是 dirty insert
- clean / hold / success 没有跟上
- 继续把它从 `23` 拉到 `50`，大概率只是继续验证“full-scale 在 sign-safe 下仍然太猛”

所以这次 early-stop 的目的不是中断一条上升曲线，而是节省算力。

## 5. 这轮失败说明了什么

### 5.1 sign-safe clip 不能单独拯救 full-scale

这是这轮最重要的结论。

上一轮已经证明：

- sign-safe clip 本身是有效的
- 它能把 `normal` 推到和 `zero-steer` 打平

但这轮又证明了另一半：

- 仅仅有 sign-safe clip，还不足以让 `steer_scale=1.0` 重新变成好配置

也就是说：

- wrong-sign steering 的确是一个主因
- 但 full-scale 的问题不只是 wrong-sign
- 它还会把系统推向“更激进但更脏”的插入方式

### 5.2 当前 full-scale 的主要收益是“更容易插”，不是“更容易 clean”

这轮最典型的现象就是：

- `inserted` 快速上升
- `dirty_insert` 几乎同步上升
- `push_free / clean / hold / success` 并没有同步上来

这说明 full-scale 现在提供的是：

- 更强的承诺和推进

但它没有同时提供：

- 更好的近场姿态稳定
- 更干净的插入几何

所以它不是把系统推到 good basin，而是把系统更快推到了 dirty basin。

## 6. 对当前实验线的影响

这轮 early-stop 之后，有两件事基本可以确定：

### 6.1 `sign-safe + 0.75` 目前仍然是更合理的基线

因为相对比起来：

- `sign-safe + 0.75` 至少能做到 `normal = 5/9`，并且与 `zero-steer` 打平
- `sign-safe + 1.0` 在训练中后段则明显表现为 dirty collapse

所以如果要继续往“能自动插托盘”的方向推进，当前更好的落脚点仍然是：

- 保留 sign-safe
- 保留 `stage1_steer_action_scale = 0.75`

### 6.2 下一步不应该继续在 amplitude 上扫

这条线到现在已经回答得很清楚了：

- 无保护的 `1.0` 太大
- train-time `0.5` 太保守
- `0.75` 是更好的折中
- sign-safe 之后，`1.0` 仍然会滑向 dirty basin

所以 amplitude 这条线的主要信息增益已经接近榨干。

## 7. 下一步最合理的单因素方向

当前最合理的下一步不是继续扫更多 scale，而是：

- 以 `sign-safe + 0.75` 为当前最好基线
- 继续只改一个变量
- 把主目标从“让它更敢插”切换到“让它在已经敢插的前提下更干净”

更具体地说，下一刀更值得瞄准：

- 继续压 dirty insert
- 或者进一步强化 near-field clean hold 的稳定性

而不是再让 steering 更激进。

## 8. 一句话总结

> `sign-safe clip` 已经证明自己是有效的，但 `sign-safe + full-scale` 这轮又证明：wrong-sign 不是 full-scale 唯一的问题。把 `stage1_steer_action_scale` 直接拉回 `1.0`，会让系统更快滑向 dirty insert，而不是更快进入 clean hold。因此当前更合理的基线仍然是 `sign-safe + 0.75`，下一步应继续围绕“如何把插入变干净”做单因素推进，而不是继续扫 amplitude。
