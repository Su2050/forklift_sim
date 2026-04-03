# Exp9.0 Post-Insert Freeze Ablation And Correction Diagnostic

日期：`2026-04-02`

## 1. 目标

这轮诊断只回答两个问题：

1. 当前代码是否会在 `insert_entry` 后过早冻结 `drive/steer`，导致“已插入但偏心”根本无法继续微调？
2. 对于固定的 post-insert 偏心/偏航 case，当前动力学 + 简单闭环控制器能不能把误差继续压下去？

对应脚本：

- `scripts/validation/physics/eval_postinsert_correction.py`

脚本设计了三种冻结策略对照：

- `current_hold_freeze`：当前真实逻辑，只在上一拍 `hold_entry` 后冻结
- `freeze_on_insert`：人为 ablation，`insert_entry` 后立即冻结
- `no_freeze`：人为 ablation，完全禁用冻结

## 2. 动作透传探针（Phase A）

固定 case：

- `insert_depth = 1.00m`
- `lateral = +0.18m`
- `yaw = +4.0deg`
- 这是一个 `inserted-but-not-hold` 的 case

结果：

| Regime | root disp | abs(yaw delta) | hold hit | ext mask | int mask |
| --- | ---: | ---: | :---: | ---: | ---: |
| current_hold_freeze | `0.1289` | `0.68deg` | `False` | `0` | `0` |
| freeze_on_insert | `0.0015` | `0.00deg` | `False` | `24` | `0` |
| no_freeze | `0.1282` | `0.74deg` | `False` | `0` | `0` |

结论：

1. **当前代码不是 `insert_entry` 就冻结。**
   - `current` 和 `no_freeze` 都有明显位移
   - 只有 `freeze_on_insert` 几乎完全不动
2. **在 inserted-but-not-hold 阶段，当前冻结逻辑没有提前介入。**
   - `current` 与 `no_freeze` 基本一致

因此，“现在 success=0 的主因是插入后立刻被冻结”这个假设，**不成立**。

## 3. 固定 Case 纠偏诊断（Phase B）

### 3.1 Case A：更深插入 + lateral bias

固定 case：

- `insert_depth = 1.00m`
- `lateral = +0.18m`
- `yaw = +4.0deg`

结果：

| Regime | center init->best->final | tip init->best->final | yaw init->best->final | hold max | hold hit |
| --- | --- | --- | --- | ---: | :---: |
| current | `0.139 -> 0.139 -> 0.157` | `0.178 -> 0.172 -> 0.172` | `3.76 -> 1.50 -> 1.50` | `0` | `False` |
| freeze_on_insert | `0.138 -> 0.138 -> 0.146` | `0.178 -> 0.174 -> 0.174` | `3.78 -> 2.70 -> 2.70` | `0` | `False` |
| no_freeze | `0.138 -> 0.138 -> 0.152` | `0.178 -> 0.172 -> 0.173` | `3.78 -> 1.90 -> 1.94` | `0` | `False` |

判读：

- 三种策略都**进不了 hold**
- `current` 与 `no_freeze` 很接近
- 说明这个更偏“孔内 tip 约束/控制可纠偏性”问题，而不是冻结问题

### 3.2 Case B：更深插入 + yaw bias

固定 case：

- `insert_depth = 1.00m`
- `lateral = +0.05m`
- `yaw = +9.0deg`

结果：

| Regime | center init->best->final | tip init->best->final | yaw init->best->final | hold max | hold hit | ext/int mask |
| --- | --- | --- | --- | ---: | :---: | --- |
| current | `0.042 -> 0.042 -> 0.061` | `0.048 -> 0.001 -> 0.003` | `8.64 -> 5.50 -> 5.50` | `180` | `True` | `0 / 179` |
| freeze_on_insert | `0.042 -> 0.042 -> 0.061` | `0.048 -> 0.003 -> 0.006` | `8.62 -> 5.30 -> 5.31` | `180` | `True` | `180 / 179` |
| no_freeze | `0.039 -> 0.036 -> 0.062` | `0.051 -> 0.001 -> 0.062` | `8.61 -> 0.00 -> 0.00` | `180` | `True` | `0 / 0` |

判读：

- 这个 case 本身比较接近 hold 区，三种策略都能长期保持 hold
- `current` 在进入 hold 后会被冻结，因此 yaw 最终停在约 `5.5deg`
- `no_freeze` 可以把 yaw 继续压到 `0deg`

这说明：

- **hold 后冻结会阻止进一步“优化姿态”**
- 但这里它**没有阻止进入 hold / success-valid 区**

### 3.3 Case C：near-hold but not stable

固定 case：

- `insert_depth = 0.94m`
- `lateral = +0.11m`
- `yaw = +7.0deg`

结果：

| Regime | center init->best->final | tip init->best->final | yaw init->best->final | hold max | hold hit | ext/int mask |
| --- | --- | --- | --- | ---: | :---: | --- |
| current | `0.029 -> 0.001 -> 0.018` | `0.102 -> 0.063 -> 0.063` | `7.00 -> 7.00 -> 7.78` | `177` | `True` | `0 / 176` |
| freeze_on_insert | `0.029 -> 0.001 -> 0.002` | `0.101 -> 0.065 -> 0.065` | `6.95 -> 6.41 -> 6.43` | `180` | `True` | `180 / 179` |
| no_freeze | `0.029 -> 0.001 -> 0.043` | `0.102 -> 0.000 -> 0.043` | `7.00 -> 0.00 -> 0.00` | `180` | `True` | `0 / 0` |

判读：

- 这是最接近“最后微调”的 case
- `current` 已经能进入并长期保持 hold
- `no_freeze` 能继续把 yaw/tip 压得更低，但也会继续移动，不是纯收益单调

说明：

- `hold` 冻结更多是在“锁定一个已经过线的姿态”
- 它可能损失最终最优姿态，但**不是当前 success=0 的主矛盾**

## 4. 总结

### 4.1 已经可以排除的假设

可以排除：

- **“当前代码在刚插入时就把微调控制冻结，所以 success 起不来”**

证据：

- Phase A 中 `current` 与 `no_freeze` 几乎一样能动
- 只有人为制造的 `freeze_on_insert` 才会让 inserted case 基本失去动作透传

### 4.2 现在更像真的问题是什么

当前更像两类问题：

1. **pre-hold / tip gate 之前的纠偏能力仍不够**
   - Case A 在三种策略下都进不了 hold
   - 这更像控制/奖励/动力学可纠偏性问题
2. **一旦已经进入 hold，当前逻辑会锁定姿态，不再继续做“更优”对齐**
   - Case B / C 中 `no_freeze` 可以继续把 yaw 压到更低
   - 但这更像“post-hold refinement”问题，不是主 success blocker

### 4.3 对下一步的建议

下一步更值得做的是：

1. 继续针对 **Case A 类的 pre-hold 孔内纠偏** 做 controller / reward 诊断
2. 如果要研究“最终姿态是否还能更稳”，可以做一个更小的 ablation：
   - 不改成 `insert_entry` 冻结
   - 而是把当前 `hold_entry` 冻结改成：
     - `hold_counter >= K` 后再冻结，或
     - `success` 后再冻结

不建议回到旧假设去改“按插入就冻结/不冻结”，因为这轮结果已经表明：

- **`insert_entry` 冻结是坏的**
- **当前主线代码已经避免了这个问题**
