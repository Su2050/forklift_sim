# Exp9.0 最近实验结果方法论分析

日期：`2026-04-03`

## 1. 分析范围

本次分析按
[`exp9_0_training_result_analysis_methodology_20260403.md`](./exp9_0_training_result_analysis_methodology_20260403.md)
的方法执行，重点覆盖最近几组最关键结果：

- `O3` 主线训练结果
- post-insert freeze ablation
- `Case A` pre-hold controller ablation
- `Case A` tip gate sweep + replay
- `strict vs relaxed 0.175` 的双 seed、`200 iter` A/B

主要数据来源：

- `exp9_0_rewardfix_o3_result_20260402.md`
- `exp9_0_postinsert_freeze_ablation_and_correction_20260402.md`
- `exp9_0_case_a_tip_gate_sweep_and_replay_20260402.md`
- `exp9_0_tipgate_ab_short_compare_20260402_181030_seed42.md`
- `exp9_0_tipgate_ab_short_compare_20260402_181030_seed43.md`
- `exp9_0_tipgate_ab_multiseed_compare_20260402_181030.md`

## 2. 第一步：先确认这是不是学习问题

按方法论，第一步不是看 success，而是先确认：

- 物理是否可达
- 动作是否透传
- 是否存在错误冻结
- 成功判定是否明显失真

### 2.1 已经可以排除的解释

最近诊断已经可以排除两个强假设：

1. **不是 `insert_entry` 之后过早冻结了微调控制。**
   - `postinsert freeze ablation` 里，当前真实逻辑和 `no_freeze` 都能明显移动。
   - 只有人为做成 `freeze_on_insert` 才会几乎完全不动。

2. **不是“车根本没动，所以当然过不了线”。**
   - `Case A replay` 里，`root_y` 明显在变，`yaw` 也还能继续改善。
   - 但 `fork center/tip lateral` 改善极小。

### 2.2 当前更像什么问题

这一步的结论是：

- 当前主矛盾不是“环境坏了”或“动作没进去”
- 更像是 **深插入后的局部几何 / controllability 问题 + strict gate 过严**

也就是说，这仍然是“学习与任务定义交界处”的问题，而不是一个明显的底层 bug。

## 3. 第二步：做任务漏斗定位

按任务漏斗看，最近结果非常一致地指向：

- `inserted` 不低
- `hold/success` 极低

### 3.1 O3 主线训练的漏斗形态

`O3` 的 `last 50` 均值：

- `phase/frac_inserted = 0.5809`
- `phase/frac_aligned = 0.0500`
- `phase/frac_tip_constraint_ok = 0.1369`
- `phase/frac_hold_entry = 0.0006`
- `phase/frac_success = 0.0003`

这说明：

- 策略不是完全不会插入
- 它能到“插入后、接近对齐”的区域
- 但从 pre-hold / hold 到 success 的转化极差

### 3.2 最新 multiseed A/B 的漏斗形态

双 seed、最后 `20` 个 iteration 的跨 seed 均值：

- strict:
  - `phase/frac_inserted = 0.5953`
  - `phase/frac_prehold_reachable_band = 0.0090`
  - `phase/frac_prehold_reachable_band_companion = 0.0098`
  - `phase/frac_hold_entry = 0.0008`
  - `phase/frac_success = 0.0000`
- relaxed 0.175:
  - `phase/frac_inserted = 0.5652`
  - `phase/frac_prehold_reachable_band = 0.0004`
  - `phase/frac_prehold_reachable_band_companion = 0.0008`
  - `phase/frac_hold_entry = 0.0031`
  - `phase/frac_success = 0.0012`

漏斗定位很明确：

- **主 bottleneck 不在“能不能插入”**
- **主 bottleneck 在 pre-hold -> hold -> success 这一段**

## 4. 第三步：区分“没到过”还是“到过但转化不了”

这是最近结果里最关键的一步。

### 4.1 strict 组明显存在 near-pass 堆积

strict 组的跨 seed 结果显示：

- `phase/frac_prehold_reachable_band = 0.0090`
- `phase/frac_hold_entry = 0.0008`

也就是：

- 到达 `0.17` 带的频率明显高于真正进入 hold 的频率

更细一点看每个 seed：

- `seed42`
  - `band0175 > 0` 的 iteration：`90`
  - `hold_entry > 0` 的 iteration：`31`
  - `band0175 > hold` 的 iteration：`76`
- `seed43`
  - `band0175 > 0` 的 iteration：`86`
  - `hold_entry > 0` 的 iteration：`9`
  - `band0175 > hold` 的 iteration：`83`

这个模式非常像：

- 策略**经常到达 near-pass 区**
- 但在 strict gate 下，很多状态无法转成真正的 hold

### 4.2 relaxed 组把一部分 near-pass 转成了 hold / success

relaxed `0.175` 后：

- `band0175` 基本消失
- `hold_entry` 增加到 `0.0031`
- `success` 增加到 `0.0012`

这说明：

- relaxed gate 不是“没变化”
- 它确实把一部分原来堆积在 near-pass 区的样本，转成了 hold 甚至 success

### 4.3 但 relaxed 并没有带来 strict success

同一个结果里还有一个非常重要的反证：

- `phase/frac_success_strict` 在 strict 和 relaxed 中都还是 `0.0000`

也就是说：

- gate mismatch 是真实存在的
- 但“只把 gate 放宽”还不足以解决最终 strict 任务

这是本轮分析最需要避免误读的地方。

## 5. 第四步：看“偶发命中”还是“稳定闭环”

最近结果里，确实已经看到：

- `strict` 和 `relaxed` 都偶尔能有非零 `hold` / `success`
- `O3` 主线里也有 `success > 0` 的 iteration

但按方法论，这只能说明：

- 策略碰到过 success manifold

不能说明：

- 策略已经学会稳定闭环

证据是：

- `O3` 主线 `last 50`：`success = 0.0003`
- multiseed strict `last 20`：`success = 0.0000`
- multiseed relaxed `last 20`：`success = 0.0012`
- `success_strict` 始终没有起来

所以当前仍然属于：

- **偶发命中，不是稳定成功**

## 6. 第五步：做跨 seed 一致性判断

这一步的结论很关键，因为它决定我们能不能把前面的现象当成机制现象。

### 6.1 两个 seed 的方向是一致的

`seed42` 和 `seed43` 都表现出同一方向：

- strict 下有明显 `band017 / band0175` 堆积
- relaxed 下 `hold` 更高
- relaxed 下 `success` 也略高
- relaxed 下 `band > hold` 的堆积现象基本消失

说明：

- “strict gate 卡住 near-pass 样本”不是单 seed 偶然现象
- 是至少在 `seed42/43` 上重复出现的模式

### 6.2 但 relaxed 的几何误差并没有更好

这也是跨 seed 里最重要的限制项。

跨 seed `last 20` 均值：

- strict:
  - `err/center_lateral_inserted_mean = 0.3593`
  - `err/tip_lateral_inserted_mean = 0.3611`
  - `err/yaw_deg_inserted_mean = 6.3513`
- relaxed:
  - `err/center_lateral_inserted_mean = 0.4118`
  - `err/tip_lateral_inserted_mean = 0.4179`
  - `err/yaw_deg_inserted_mean = 7.1823`

也就是说：

- relaxed 的 `hold/success` 更高
- 但 inserted 几何质量平均并没有更好，反而更差

这说明 relaxed 的主要作用更像：

- **改变后段分类与闭环门槛**

而不是：

- **真正让策略学会更严格的几何对齐**

## 7. 结合 fixed-case 诊断后的最可信解释

把最近训练结果和 fixed-case 诊断放在一起，当前最可信的解释是：

### 7.1 解释一：strict tip gate 与当前可达带存在明显 gap

`Case A` 固定诊断里：

- `tip gate = 0.12 / 0.14 / 0.16` 全灭
- `0.170` 开始部分解锁
- `0.175` 开始全部 controller 解锁

这说明：

- 深插入偏心态的稳定可达带更接近 `tip ≈ 0.17~0.175m`
- 原始 `0.12m` gate 对当前状态分布来说太严格

而 multiseed 训练中：

- strict 下恰好出现 `band017 / band0175` 堆积
- relaxed 下恰好这部分堆积被转成 hold / success

这两者是对得上的。

### 7.2 解释二：这不只是 gate mismatch，还叠加了局部 controllability 问题

如果问题只是“门太窄”，那么 relaxed 后应该看到：

- inserted 几何误差更好，或者
- 后续 strict success 逐步抬起来

但目前没有看到。

再结合 `Case A replay`：

- `root_y` 会动
- `yaw` 能改一点
- 但 `center/tip lateral` 几乎不再改善

更合理的解释是：

- **策略经常能走到 near-pass 区**
- **但深插入后已经进入一个 lateral correction 很弱的坏盆地**
- strict gate 把这类样本全部判成失败
- relaxed gate 只是在判定层面“接住了”一部分 near-pass 样本

因此，当前问题更像是：

- `gate mismatch` 是真的
- 但它不是唯一问题
- 背后还叠加了深插入后的局部可纠偏性不足

## 8. 当前最合理的结论

按方法论，这一轮最近结果可以归纳成下面 4 句话：

1. `exp9_0` 的主 bottleneck 不是“插不进去”，而是 **pre-hold -> hold -> success** 这段转化率过低。
2. strict 条件下，策略已经**频繁到达 0.17 / 0.175 near-pass 区**，并不是“从未接近成功”。
3. relaxed `0.175` 能把一部分 near-pass 样本转成 `hold/success`，说明 **strict gate 过严** 这个解释成立。
4. 但 relaxed 并没有提升 strict success，且 inserted 几何误差并不更优，说明 **只放宽 gate 不是最终解法**。

## 9. 下一步探索方向

按方法论，下一步不该回到“继续盲调 reward”，而应优先做能缩小不确定性的实验。

### 9.1 第一优先级：做 training gate curriculum，而不是永久放宽 gate

建议方向：

- 训练时用 `0.175 -> 0.12` 的 anneal / curriculum
- 评估时保留 strict `0.12`
- 全程保留 `band017 / band0175 / hold / success_strict` 诊断

要回答的问题是：

- relaxed gate 只是“重新分类”了样本
- 还是能作为训练桥梁，最后把策略带进 strict success 区

这是当前信息增益最高的一步。

### 9.2 第二优先级：把“0.175 到 0.12 的最后收紧段”做成显式训练目标

如果 relaxed 只是把 near-pass 样本接住，但不能继续压进 strict 区，那么下一步要针对这段 gap 做单独塑形。

推荐方向：

- 保持 strict success 不变
- 对 `0.175 -> 0.12` 这段 gap 增加显式 progress shaping
- 或在 `pre-hold reachable band` 内再加细粒度 reward / diagnostic band

重点不是奖励越多越好，而是让策略感受到：

- 从 near-pass 到 strict-pass 的增量收益

### 9.3 第三优先级：继续针对 `Case A` 做分布与 controllability 联动实验

当前 fixed-case 已经说明：

- 这是一个真实存在的坏盆地

下一步最值得做的不是更多 controller 微调，而是：

- 让训练更频繁遇到 `Case A` 近邻状态
- 观察在更高访问频率下，strict gap 是否能被逐步压缩

可以考虑：

- reset / curriculum 向 near-pass 深插入偏心态倾斜
- 但 success 口径仍保持 strict 不变

### 9.4 暂时不建议优先做的方向

暂时不建议优先：

- 再开一轮只改 reward 权重的 O4
- 继续争论 inserted 后是不是 freeze 太早
- 只用更长训练去“碰碰运气”

因为这些方向对当前最大不确定性的缩小都不够直接。

## 10. 一句话收束

最近 `exp9_0` 的结果已经把问题收缩得很明确：

- **策略不是完全不会，而是频繁到达 near-pass 区；**
- **strict gate 的确过严；**
- **但只放宽 gate 还不能让 strict 任务成立，因为深插入后的最后一段纠偏能力仍然不足。**

所以，下一步最值得做的，不是继续盲调 reward，而是：

- **把 relaxed gate 当成训练桥，而不是最终目标；**
- **用 curriculum 或 gap-specific shaping，把 `0.175 -> 0.12` 这段真正学出来。**
