# `exp8.3` 多轮实验复盘与下一步建议

> 日期: `2026-03-25`
> 分支: `exp/exp8_3_clean_insert_hold`
> 范围: `Phase 1 hold/success 修复` 之后，到 `clean_insert_hold`、单因素/双因素 ablation、`B0 + 轻 bonus` 多 seed 短训为止
> 适用边界: 当前结论主要针对 `near-field + 256x256 + 50 iter` 下的 `clean insert + stable hold`，**不是**对 `wide reset` 或完整自动插托盘任务的最终结论

---

## 1. 一页结论

- 当前主问题已经不是 `hold/success` 逻辑没接好。`Phase 1` 修复之后，`phase/frac_hold_entry` 和 `phase/frac_success` 已经可以稳定离开 `0`，说明 runtime 判据本身已经打开。
- 当前主问题也不是“完全不会插入”。近场 `256x256` 下，多轮实验都证明策略已经能反复把 `phase/frac_inserted` 打开。
- 当前真正的瓶颈是: **怎样在不压死 insertion 的前提下，把 dirty insert 压下去，并把 clean insert / hold / success 稳定住。**
- 这轮实验最重要的证伪是: **在我们当前这套 gate 形式和当前测试过的强度范围内，继续堆强 gate / 强 penalty 不是正确方向。** 单因素 `F3`、双因素 `D1`、`D3` 都证明，约束一旦过强，策略会直接学成“不插”，而不是“更干净地插”。
- 当前最值得继续的主线不是 `tight gate`，而是回到 `B0` 附近做**非常轻、非常局部**的 shaping。`B0 + 轻 push-free bonus` 这条线至少证明了: 它不会像强 gate 那样系统性压死 insertion；但它是否优于 `B0`，当前证据还不够。
- 但这条轻量主线现在还**不能算稳定**。最新 `3 seeds x 50 iter` 中，`seed 42` 和 `seed 44` 尾窗都有非零 `success`，但 `seed 43` 完全塌成“不插”，说明方差仍然很大。

一句话总结:

**我们已经知道“什么太强会坏掉”，但还没有得到一个足够稳的 clean-insert 策略。下一步应该先把“B0 vs B0+轻 bonus”的同口径多 seed 对照做干净，再决定要不要继续沿这条线走。**

---

## 2. 最近实验链条到底告诉了我们什么

| 阶段 | 代表实验 | 结果 | 得到的结论 |
|------|----------|------|------------|
| `Phase 1` | `hold_success_fix` | `frac_hold_entry` / `frac_success` 能离开 `0` | success/hold 逻辑接线问题已基本解决 |
| `Phase 2 v1` | `clean_insert_hold_iter200_256cam` | 尾窗首次再次出现非零 `phase/frac_success`，但推盘很大 | 方向对，但 dirty insert 是主瓶颈 |
| `Phase 2 v2` | 第二批 `tighten clean insert reward gating` | 推盘下降，但 insertion / success 也被一起压掉 | 强 gate / 强 penalty 过猛 |
| 单因素 | `B0 / F1 / F2 / F3 / F4` | `B0` 仍是唯一尾窗明确非零 success；`F2/F4` 更干净但无 tail success；`F3` 直接压死 insertion | 最优方向不在“继续变硬”，而在“轻度、局部 shaping” |
| 双因素 | `D1 = F2 + F4`、`D3 = gate_r_cpsi + soft-F4` | 都学成了“不插” | 在当前实现与强度范围内，`gate_r_cpsi + tighter gate` 这条路不工作 |
| 当前多 seed | `B0 + 轻 push-free bonus` | `2/3` seed 尾窗有非零 success，`1/3` seed 塌成不插 | 这条线有希望，但证据只够支持“继续验证”，不够支持“已经优于 B0” |

---

## 3. 这几轮实验里，哪些认识现在已经比较稳了

### 3.1 已经基本证实的事

1. `hold/success` 逻辑修复是必要且有效的。
   在修复之前，很多“reward 看起来还行但 success 永远为 0”的现象，部分来自 runtime 逻辑与 cfg 漂移。修完之后，`hold_entry`、`success_geom_strict`、`success` 都真实打开了。

2. `clean_insert_hold` 方向本身不是错的。
   `iter200_256cam` 那轮已经证明，clean-insert 相关信号并不是虚假的单点噪声，而是可以在整轮里反复出现。

3. 当前策略已经具备 near-field insertion 能力。
   所以现在的优化重点不该再放在“怎么更快接近托盘”，而应放在“插入后如何不推盘、如何稳定 hold”。

4. 强约束确实能压低 dirty insert。
   但在当前实现和当前测试过的参数区间里，它的副作用不是小幅降脏插，而是直接把 insertion 本身压没。这意味着当前 reward landscape 对“推进”很敏感，不适合继续用强 gating 粗暴裁剪。

### 3.2 已经基本证伪的事

1. “继续收紧 `gate_r_cpsi` / `tight gate package` 就能自然转成 clean insert”
   - 在我们当前实现和当前测试过的参数范围内，不成立。
   - `D1` 和 `D3` 都明确失败，后段几乎是“规规矩矩停在外面，不往里插”。

2. “dirty penalty 单独开就能解决问题”
   - 不成立。
   - `F3` 基本直接把策略打成不插。

3. “只要看到单 seed 有改善，就可以继续往这个方向做长跑”
   - 不成立。
   - `F2` 的单 seed 看起来曾经有希望，但后续 `D2` 没有复现，说明当前方差足以误导方向判断。

---

## 4. 这轮实验里我们自己的方法上有哪些教训

### 4.1 归因有时做得太粗

第二批 `clean_insert_hold` 我一次性同时改了:

- `r_cd` gate
- `r_cpsi` gate
- gate 强度
- dirty penalty

结果虽然很快证明“整体太强会坏”，但代价是不能立即知道到底是谁起了主要负作用。  
后面单因素和双因素 ablation 其实就是在补这个归因债。

### 4.2 太早根据单 seed 做方向判断

`F2`、`F4` 都出现过“单 seed 看着更干净”的时刻，但一上多 seed 或双因素，就出现完全不同的结局。  
这说明当前训练方差已经大到不能只看一条 run 决策。

### 4.3 没有先做多 seed 的 `B0` 对照

这是当前最重要的缺口。  
现在虽然 `B0 + 轻 bonus` 看起来比强 gate 更健康，但我们手里只有:

- 单 seed 的 `B0`
- 多 seed 的 `B0 + bonus`

缺少**同口径、多 seed 的纯 `B0` 基线**。  
在这个对照补齐之前，不能下结论说“轻 bonus 已经优于 `B0`”。

### 4.4 工程问题还在污染长跑体验

多次 `200/800 iter` 文本日志都停在 `199/200`、`799/800`，虽然模型文件通常已经保存到最后一轮附近，但这仍然会干扰实验判断和自动化整理。  
这不是当前 reward 方向的主矛盾，但确实应该作为并行工程问题记录下来。

---

## 5. 对当下状态的判断

当前状态可以概括成三句话:

1. **主线没有死。**  
   `B0` 及其附近的轻量 shaping 仍然能保住 insertion，甚至在部分 seed 上保住尾窗 nonzero success。

2. **主线还不稳。**  
   最新 `3 seeds x 50 iter` 的 `B0 + 轻 bonus` 中:
   - `seed 42`: 尾窗 `phase/frac_success = 0.0156`，`diag/pallet_disp_xy_mean = 0.1298`
   - `seed 43`: 尾窗完全塌成 `phase/frac_inserted = 0`
   - `seed 44`: 尾窗 `phase/frac_success = 0.0156`，但 `diag/pallet_disp_xy_mean = 0.2849`，dirty insert 仍重

3. **当前最缺的不是更强的约束，而是更稳的因果判断。**  
   在没有多 seed `B0` 对照之前，再继续发明新的 reward 项，风险很大。

4. **当前阶段的结论边界必须写清楚。**  
   现在这些实验主要回答的是 near-field 下 `clean insert + stable hold` 是否在变好；它们还不能直接推出 `wide reset` 或完整自动插托盘任务已经有答案。

---

## 6. 接下来最应该做什么

### 6.1 第一优先级: 补齐多 seed `B0` 对照

推荐直接跑:

- `B0 baseline`
- `near-field + 256x256 + 50 iter`
- `seed = 42, 43, 44`

目的不是再看单条最好成绩，而是回答一个更关键的问题:

**`B0 + 轻 bonus` 的当前表现，到底是真的更好，还是只是方差导致的“看起来更好”。**

只有这个问题先回答清楚，后面的任何 bonus weight sweep 才有意义。

### 6.2 第二优先级: 给短训补一个固定 eval 口径

当前我们主要靠训练日志窗口判断方向，这足够做第一轮筛选，但不够做最终定论。  
建议在 `50 iter` 短训之后，固定取末尾 checkpoint 做一次统一 eval，至少对齐:

- `phase/frac_success`
- `phase/frac_inserted`
- `phase/frac_inserted_push_free`
- `diag/pallet_disp_xy_mean`

这样可以减少“训练中窗口波动”对方向判断的干扰。

### 6.3 第三优先级: 如果 `B0 + bonus` 确实优于 `B0`，只扫 bonus 权重

如果多 seed 对照表明 `B0 + 轻 bonus` 在均值或稳定性上优于 `B0`，下一步不要再碰 gate 结构，建议只扫一个变量:

- `clean_insert_push_free_bonus_weight`

推荐候选:

- `0.5`
- `1.0`
- `2.0`

保持其余设置完全不变。  
这样才能知道“轻 bonus 本身是否是正确方向”，而不是又把问题混进其他因素里。

### 6.4 第四优先级: 只有多 seed 短训稳定后，才重新上 `100/200 iter`

`50 iter` 当前更适合作为**筛选实验**，而不是最终方向定论。  
它能帮助我们快速识别“是否会很快塌成不插”或“是否明显变脏”，但不能单独证明某个设置就是最终最优。

进入下一阶段前，至少要满足:

- 不再出现 `1/3` seed 直接塌成“不插”
- `phase/frac_success` 在多个 seed 的尾窗都不是恒 `0`
- `diag/pallet_disp_xy_mean` 不显著劣于当前 `B0`

如果这些做不到，就不应该再上 `100/200 iter`，更不应该直接回到 `800 iter`。

### 6.5 工程并行项: 单独查 `199/200` / `799/800` 收尾问题

这件事不需要阻塞 reward 主线，但建议单独开一个小任务查:

- 为什么文本日志经常停在最后一轮前后
- 是否是 logger flush / shutdown hook / Isaac 退出时序问题

因为只要这个问题存在，长跑结果的自动整理就一直会受影响。

---

## 7. 建议的具体执行顺序

### Step 1

先跑 `B0 baseline` 的 `3 seeds x 50 iter`，口径与当前 bonus 实验完全一致:

- `near-field`
- `256x256`
- `64 envs`
- `50 iter`
- `seed 42/43/44`

### Step 2

把 `B0` 与当前 `B0 + 轻 bonus` 做同口径对照，主看:

- `phase/frac_success`
- `phase/frac_inserted_push_free`
- `phase/frac_clean_insert_ready`
- `phase/frac_inserted`
- `diag/pallet_disp_xy_mean`

### Step 3

对 `B0` 与 `B0 + bonus` 的末尾 checkpoint 补统一 eval，确认训练日志趋势与固定评估口径一致。

### Step 4

如果 `B0 + bonus` 的跨 seed 均值或稳定性更好，再做 `bonus weight sweep`。

### Step 5

只有在短训多 seed 结果稳定后，再选择 `1` 个最优设置进入 `100/200 iter / 256x256`。

---

## 8. 当前最不该做的事

1. 不要再继续加硬 gate。
2. 不要再把 `r_cpsi` 或 `r_cd` 往更强的 gate 方向叠。
3. 不要在没有多 seed `B0` 对照的情况下，直接宣布当前 `bonus` 已经更好。
4. 不要把 `50 iter` 的单次结果当成最终方向定论。
5. 不要现在就再上 `800 iter` 长跑。

---

## 9. 一句话判断

**到目前为止，最有价值的新认识不是“我们已经找到答案”，而是“在当前实现和当前强度范围内，强 gate 这条路不工作；真正值得继续的是 `B0` 附近的轻量 shaping，但必须先用同口径多 seed 对照和固定 eval 把因果判断做干净”。**
