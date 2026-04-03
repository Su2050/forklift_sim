# Exp9.0 Case A Tip Gate Sweep And Replay Diagnostic

日期：`2026-04-02`

## 1. 目标

上一轮 `Case A` controller ablation 已经说明：

- 更强的 pre-hold controller 也救不回 `tip_align_entry_m=0.12m`
- `center_y_err` 基本卡在 `0.138m`
- `tip_y_err` 最多只能从 `0.178m` 压到 `0.167m` 左右

因此这轮继续追两个更直接的问题：

1. `Case A` 到底是不是一个 **tip gate-vs-physics mismatch**？
2. 轨迹上到底是“车根本没动”，还是“车在动但 fork tip 横向几乎不再跟着改善”？

对应脚本：

- `scripts/validation/physics/eval_case_a_tip_gate_sweep.py`
- `scripts/validation/physics/replay_case_a_prehold_diagnostic.py`

## 2. 固定 Case

两份脚本都使用同一个固定诊断 case：

- `insert_depth = 1.00m`
- `lateral = +0.18m`
- `yaw = +4.0deg`

这就是前面反复暴露出来的 `Case A`：

- 已经深插入
- `center` 不算离谱，但没有继续明显改善
- `tip` 始终过不了默认 `0.12m` gate

## 3. Tip Gate Sweep

### 3.1 诊断设置

本轮选了 4 个代表性 controller 变体：

1. `Phase-B ref`
2. `Tip priority`
3. `Deep pull-out`
4. `Balanced slow`

然后只改一个东西：

- 把诊断用 `tip gate` 依次设为 `0.12 / 0.14 / 0.16 / 0.18m`

其余 `center/yaw/insert` 条件保持不变。

### 3.2 结果

| tip gate | tip_ok / 4 | hold / 4 | 结论 |
| --- | ---: | ---: | --- |
| `0.12` | `0` | `0` | 全部失败 |
| `0.14` | `0` | `0` | 全部失败 |
| `0.16` | `0` | `0` | 仍然全部失败 |
| `0.18` | `4` | `4` | 全部在 `step 0` 立即命中 |

按 variant 看，严格阈值下的最佳 `tip` 也只到：

- `Phase-B ref`: `0.1658`
- `Deep pull-out`: `0.1682`
- `Balanced slow`: `0.1698`
- `Tip priority`: `0.1707`

也就是说：

- `0.16m` 仍然过不了
- 一旦放宽到 `0.18m`，`Case A` 从初始帧就已经满足 `tip/hold`

这不是“再多跑一会儿就可能过线”的形态，而是一个非常像 **硬阈值卡边** 的现象。

### 3.3 判读

这轮 gate sweep 给出的信号非常强：

1. `Case A` 并不是离 `hold` 很远
2. 它更像是稳定停在 `tip ≈ 0.166~0.171m` 这段可达带里
3. 默认 `tip_align_entry_m=0.12m` 对这个深插入偏心态来说，明显比当前物理可达带更严格

因此，“Case A 主要是 controller 不够强”这个解释，优先级已经下降了。

### 3.4 Fine Boundary Sweep (`0.17 / 0.175 / 0.18`)

为了把边界卡得更精确，我又补跑了一轮更细的 sweep：

- `0.170m`
- `0.175m`
- `0.180m`

新输出文件：

- `outputs/validation/manual_runs/20260402_160844_case_a_tip_gate_sweep.csv`

结果：

| tip gate | tip_ok / 4 | hold / 4 | 结论 |
| --- | ---: | ---: | --- |
| `0.170` | `3` | `3` | 只差 `Tip priority` 仍未过线 |
| `0.175` | `4` | `4` | 全部 controller 解锁 |
| `0.180` | `4` | `4` | 全部 controller 解锁，且 `step 0` 命中 |

这一轮把可达边界进一步缩到了：

- **至少一个 controller 被解锁**：`0.170m`
- **所有测试 controller 都被解锁**：`0.175m`

所以如果只看 `Case A` 这类深插入偏心态：

- `0.170m` 已经非常接近物理可达边缘
- `0.175m` 更像是“稳定可达带”的起点
- 原始 `0.120m` 与当前可达带之间存在明显 gap

## 4. Replay Diagnostic

### 4.1 诊断设置

为了直接看轨迹上卡在哪里，回放脚本记录了下面这些时间序列：

- `drive_cmd`
- `steer_cmd`
- `root_y`
- `fork_center_y`
- `fork_tip_y`
- `yaw_err_deg`
- `insert_norm`

回放了 3 个代表性 variant：

1. `Phase-B ref`
2. `Deep pull-out`
3. `Balanced slow`

输出文件：

- CSV: `outputs/validation/manual_runs/20260402_153948_case_a_prehold_replay.csv`
- PNG: `outputs/validation/manual_runs/20260402_153948_case_a_prehold_replay.png`

### 4.2 结果总表

| Variant | root_y span | center init->final | tip init->final | yaw init->final | hold hit |
| --- | ---: | --- | --- | --- | :---: |
| Phase-B ref | `0.268m` | `0.138 -> 0.141` | `0.178 -> 0.170` | `3.77 -> 2.70deg` | `False` |
| Deep pull-out | `0.377m` | `0.138 -> 0.138` | `0.178 -> 0.169` | `3.79 -> 2.91deg` | `False` |
| Balanced slow | `0.289m` | `0.138 -> 0.141` | `0.178 -> 0.171` | `3.79 -> 2.84deg` | `False` |

几个非常关键的现象：

1. `root_y` 明显在持续变化
   - 不是“车根本没动”
   - 位移量大约有 `0.27~0.38m`
2. `yaw` 也确实在变好
   - 大约改善了 `0.88~1.11deg`
3. 但 `center_y` 基本不改善
   - 最终仍停在 `0.138~0.141m`
4. `tip_y` 只改善了约 `0.008~0.010m`
   - 远远不够把 `0.178m` 压进 `0.12m`

### 4.3 判读

这说明 `Case A` 的问题不是简单的：

- 动作没透传
- 车不动
- yaw 完全收不回来

真正更像的是：

- 车体在继续运动
- yaw 也在一点点修
- 但在深插入状态下，`root lateral -> fork center/tip lateral` 的有效映射已经非常弱

换句话说：

- 系统还有“运动”
- 但已经几乎没有“有用的横向纠偏自由度”

这和前一轮 controller ablation 的结论是对得上的。

## 5. 综合结论

这轮可以把问题进一步收缩到一个更明确的判断：

### 5.1 `Case A` 很像 gate-vs-physics mismatch

证据链是：

1. 更强 controller 也过不了 `0.12m`
2. `0.16m` 仍然全灭
3. `0.18m` 立即全通，而且是 `step 0` 命中

这说明当前 `Case A` 并不是“再推一推就会自然过线”，而是：

- 当前几何/接触/动力学下
- 它稳定落在 `tip ≈ 0.17m` 一带
- 默认 gate 把这类状态全部算成 pre-hold failure

### 5.2 主瓶颈不是“车不动”，而是“动了也难把 tip 再横移进去”

Replay 直接说明：

- `root_y` 可以继续变化
- `yaw` 还能继续改善
- 但 `center/tip` 几乎没有对应幅度的横向收益

这更像一个 **深插入后的 lateral correction bad basin**。

### 5.3 继续只做 reward/controller 微调，边际收益很可能有限

因为当前更像卡在：

- 诊断 gate 设定
- 深插入接触几何
- pre-hold 可达集本身

而不是单纯卡在“controller 还不够 aggressive”。

### 5.4 训练日志里已经补上 `pre-hold reachable band`

为了把这个判断直接带回训练主线，我在环境日志里新增了 3 个指标：

1. `diag/prehold_reachable_tip_band_m`
   - 当前默认值是 `0.17`
2. `phase/frac_prehold_reachable_band`
   - 所有并行 env 中，有多少比例已经满足：
     - `inserted`
     - `align_entry`
     - `lift_entry`
     - `valid_insert_z`
     - `tip_gate_active`
     - `tip_y_err <= 0.17`
     - 但 **还没有进入 strict hold**
3. `diag/prehold_reachable_band_frac_of_inserted`
   - 上面这个 band，在 `inserted` 子集内的条件频率

这个定义刻意做成“除了 strict tip gate 之外，其余 hold 条件都已经满足”，这样训练日志能直接回答：

- 策略是不是经常能摸到 `0.17` 带
- 但因为 `0.12` 的 strict gate，始终拿不到 `hold/success`

这比只看 `phase/frac_hold_entry` 更能分辨：

- 到底是策略根本进不去 pre-hold 带
- 还是已经频繁到达 `0.17` 带，只是被 strict gate 卡死

## 6. 对 Exp9.0 的下一步建议

如果下一步要继续打透这个问题，我建议优先做下面几件，而不是直接开 O4：

1. 做一个 **diagnostic-only 的 gate curriculum**
   - 目前 `0.17 / 0.175 / 0.18` 已经把边界缩到很窄
   - 如果还要更细，可以继续看 `0.172 / 0.173 / 0.174`
2. 做一个 **训练口径 vs 诊断口径分离**
   - 保持 strict success 不变
   - 现在已经有 `pre-hold reachable band` 指标
   - 下一步就是在真实训练日志里观察它和 `phase/frac_hold_entry` 的分叉
3. 做一个 **坏盆地避免型策略**
   - 不是指望深插入后再把 `tip` 从 `0.17` 救到 `0.12`
   - 而是在更早阶段避免进入这种深插入偏心态

当前最重要的一句话总结是：

> `Case A` 现在更像一个 `tip gate` 与深插入物理可达带不匹配的问题，  
> 而不是单纯的 controller 太弱或插入后过早冻结。
