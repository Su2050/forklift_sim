# Exp9.0 Case A Pre-Hold Controller Ablation

日期：`2026-04-02`

## 1. 目标

在上一轮 `postinsert` 诊断里，`Case A` 暴露出这样一个现象：

- 已经是深插入状态
- `center_y_err` 和 `yaw_err` 不算离谱
- 但 `tip_y_err` 过不了 `tip_align_entry_m=0.12m`
- 无法进入 `hold_entry`

本轮不再讨论“是否过早冻结”，而是直接做一个更强的 **pre-hold correction controller ablation**：

- 固定同一个 `Case A`
- 禁用 `hold-freeze`
- 用一组更激进的 controller 变体做 sweep
- 看这个 case 在当前物理里到底能不能被 controller 救回 `tip/hold gate`

对应脚本：

- `scripts/validation/physics/eval_prehold_case_a_ablation.py`

## 2. 固定 Case

`Case A`：

- `insert_depth = 1.00m`
- `lateral = +0.18m`
- `yaw = +4.0deg`

判定阈值：

- `center <= 0.15m`
- `tip <= 0.12m`
- `yaw <= 8.0deg`

也就是说，这个 case 的主矛盾本来就更偏向：

- `tip_y_err` 不够小
- 不是 `yaw` 先卡死

## 3. Controller Sweep

本轮 sweep 了 8 个 controller variant，全部运行在 **禁用 hold-freeze** 的条件下：

1. `Phase-B ref`
2. `Tip priority`
3. `Aggressive tip`
4. `Tip + yaw`
5. `Center priority`
6. `Deep pull-out`
7. `Balanced slow`
8. `Yaw first`

这些变体覆盖了几类方向：

- 更强 `tip_y` 权重
- 更深的 reverse pull-out corridor
- 更强的 `yaw` 纠偏
- 更慢、更保守的平衡控制

## 4. 结果总表

| Variant | center init->best->final | tip init->best->final | yaw init->best->final | max insert | tip_ok | hold |
| --- | --- | --- | --- | ---: | :---: | :---: |
| Phase-B ref | `0.138 -> 0.138 -> 0.141` | `0.178 -> 0.167 -> 0.170` | `3.77 -> 2.65 -> 2.70` | `0.462` | `False` | `False` |
| Tip priority | `0.138 -> 0.138 -> 0.144` | `0.178 -> 0.171 -> 0.173` | `3.79 -> 2.72 -> 2.77` | `0.461` | `False` | `False` |
| Aggressive tip | `0.138 -> 0.138 -> 0.141` | `0.178 -> 0.172 -> 0.175` | `3.79 -> 3.03 -> 3.22` | `0.461` | `False` | `False` |
| Tip + yaw | `0.138 -> 0.138 -> 0.145` | `0.178 -> 0.173 -> 0.177` | `3.79 -> 3.02 -> 3.04` | `0.461` | `False` | `False` |
| Center priority | `0.138 -> 0.138 -> 0.144` | `0.178 -> 0.171 -> 0.173` | `3.79 -> 2.72 -> 2.77` | `0.461` | `False` | `False` |
| Deep pull-out | `0.138 -> 0.138 -> 0.138` | `0.178 -> 0.168 -> 0.169` | `3.79 -> 2.91 -> 2.91` | `0.461` | `False` | `False` |
| Balanced slow | `0.138 -> 0.138 -> 0.140` | `0.178 -> 0.167 -> 0.167` | `3.79 -> 2.62 -> 2.62` | `0.461` | `False` | `False` |
| Yaw first | `0.138 -> 0.138 -> 0.144` | `0.178 -> 0.171 -> 0.173` | `3.79 -> 2.72 -> 2.77` | `0.461` | `False` | `False` |

## 5. 关键结论

### 5.1 没有一个 variant 能把 `tip_y_err` 压进 gate

最佳 variant 是：

- `Balanced slow`
- `tip_y_err: 0.178 -> 0.167`

但离 `0.12m` 仍差明显一截。

也就是说：

- 这轮 sweep 已经不只是“现有 controller 太弱”
- 而是 **更强的 pull-out / tip-priority / yaw-priority controller 也救不回这个 case**

### 5.2 `center_y_err` 基本不动

所有 variant：

- `center_y_err` 的 best 值都仍停在 `0.138m`

这非常关键，因为 `tip_y_err` 大致可以分解成：

- `center lateral offset`
- 加上 `yaw` 带来的 fork tip 额外横摆

本轮 sweep 里：

- `yaw` 确实能被压低一些
- 但 `center` 几乎没被继续压下去

于是 `tip` 最多只从 `0.178` 降到 `0.167`

### 5.3 这更像 pre-hold controllability / physics bottleneck

综合看，这轮更支持下面这个判断：

- 当前 `Case A` 的主瓶颈不是 reward 稀疏本身
- 也不是 current controller 没做足够多 reverse / steer
- 而是 **在当前深插入 + 当前碰撞/动力学条件下，叉车几乎没有足够的横向继续修正自由度**

也就是说，现在更像是：

- 一旦进入这类深插入偏心状态
- 物理系统允许的“纠偏可达集”非常有限
- controller 只能把 `yaw` 稍微修一点
- 但无法把 `center/tip` 真正推进到 gate 内

## 6. 对 Exp9.0 的意义

这轮结果把问题进一步缩小了。

### 已经基本可以排除

可以继续排除：

1. `insert_entry` 过早冻结是主因
2. 只是现有 controller 太保守
3. 只要把 `tip_y` / `yaw` 权重再往上调，Case A 就会自然被救回来

### 当前更值得怀疑

现在更值得怀疑的是：

1. **深插入后的碰撞/接触约束本身让 lateral correction 几乎不可达**
2. **`tip_align_entry_m=0.12m` 对当前几何与动力学来说，在这类深插入偏心状态下过紧**
3. **训练需要避免策略进入这种“深插入但 tip 已经锁死”的坏盆地**

## 7. 下一步建议

下一步更有价值的方向是：

1. 做一个 **Case A 的几何/动力学轨迹回放诊断**
   - 直接记录 root / fork center / fork tip / steer / insert depth 的时间序列
   - 看看到底是“车在动但 tip 不横移”，还是“车根本无法产生横移”
2. 做一个 **tip gate 课程 / 可达性 ablation**
   - 例如只在诊断里把 `tip_align_entry_m` 放宽到 `0.14 / 0.16 / 0.18`
   - 看 Case A 是否能开始进入 hold
   - 如果一放宽就通，说明现在更像 gate 与物理可达集不匹配
3. 做一个 **坏盆地避免策略**
   - 不是指望深插入后再救回来
   - 而是在 pre-hold 阶段提前避免进入这种深插入偏心状态

当前我的判断是：

> `Case A` 已经很接近一个“深插入后 lateral correction 难以继续发生”的物理坏盆地。  
> 继续只做 reward shaping，大概率收益有限。  
> 下一步更值得打的是“这是不是 gate-vs-physics mismatch”。
