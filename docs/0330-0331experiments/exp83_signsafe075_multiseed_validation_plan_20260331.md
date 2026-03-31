# Exp8.3 Sign-Safe + 0.75 多 Seed 验证计划

日期：2026-03-31

## 1. 当前阶段结论

到目前为止，围绕 stage1 steering 已经有三条连续的控制变量结果：

1. `stage1_steer_action_scale = 0.75`
   - `normal = 4/9`
   - `zero-steer = 5/9`
   - 结论：比原始 full-scale 明显更好，但还不够

2. `sign-safe clip + 0.75`
   - `normal = 5/9`
   - `zero-steer = 5/9`
   - 结论：第一次把 `normal` 从落后推到追平，是当前最好基线

3. `sign-safe clip + 1.0`
   - 训练在 `iter 23` 提前停掉
   - `inserted` 很高，但几乎全部滑向 dirty insert
   - 结论：当前 full-scale 仍然过猛，不值得继续在 amplitude 上扫

因此当前最合理的落脚点已经很明确：

- 保留 `sign-safe clip`
- 保留 `stage1_steer_action_scale = 0.75`

## 2. 下一步要回答的问题

现在最关键的问题已经不是：

- 这条线在 `seed42` 上有没有机会

而是：

> `sign-safe + 0.75` 这个改动，是否只对 `seed42` 有效，还是它真的扩大了 good basin？

也就是说，现在最值得做的是多 seed 验证，而不是立刻再改 reward。

## 3. 为什么现在优先做多 seed，而不是继续改单变量

原因有三个：

### 3.1 当前最好的配置已经出现了

`sign-safe + 0.75` 是目前唯一一条满足下面两点的线：

- `normal` 不再落后于 `zero-steer`
- steering sign 开始出现真正的正负分化

这说明它值得先做稳定性验证。

### 3.2 当前最重要的未知量是 basin 宽度，不是局部指标

我们现在最想知道的，不是：

- `seed42` 上是不是还能再多出一个 success 点

而是：

- 这条改动能不能把更多 run 稳定推到 good basin

这件事最直接的证据就是多 seed。

### 3.3 继续改单变量容易打断当前最好的基线

如果现在又去改 reward / hold / reset，就会让判断重新混起来：

- 到底是 sign-safe 起作用
- 还是新 reward 起作用

所以当前更合理的顺序是：

1. 先把 `sign-safe + 0.75` 的稳定性看清楚
2. 再决定下一刀该砍在 basin 扩大还是 dirty insert 抑制

## 4. 控制变量设计

### 4.1 固定不动

- `stage1_clip_wrong_sign_steer_enable = True`
- `stage1_steer_action_scale = 0.75`
- reward：不变
- reset：不变
- observation：不变
- 相机：`256x256`
- 训练长度：`50 iter`

### 4.2 唯一变化的变量

- `seed`

## 5. 实验执行方案

### 5.1 为什么不重跑 `seed42`

当前的 `seed42` 结果已经来自与目标配置完全一致的版本：

- sign-safe 开启
- scale = `0.75`

并且已经完成了固定 `3x3 normal / zero-steer` 验收。

因此从控制变量角度，当前 suite 可以直接把已有 `seed42` 作为第一条样本。

这样做的好处是：

- 不重复烧算力
- 能更快得到 `3 seeds` 的整体图景

### 5.2 需要补跑的内容

补两条训练：

- `seed43`
- `seed44`

每条训练结束后都必须立刻补同一套 `3x3` 验收：

- `normal`
- `zero-steer`

### 5.3 固定验收口径

所有 seed 统一使用：

- `x_root = -3.40`
- `y ∈ {-0.10, 0.0, +0.10}`
- `yaw ∈ {-4°, 0°, +4°}`
- `episodes_per_point = 1`

关注的不是单一 `success`，而是这四类指标一起看：

- `success`
- `push_free`
- `clean_insert_ready`
- `dirty_insert`

## 6. 决策门槛

### 6.1 值得继续往前推进的条件

如果 `seed43/44` 里至少有一个也满足下面趋势：

- `normal >= zero-steer`
- 有非零 `push_free / clean / hold`

那么可以判断：

- `sign-safe + 0.75` 不只是 `seed42` 偶然
- 这条线值得被当作当前 Phase baseline

下一步就应该继续围绕这个基线，去做“压 dirty / 稳 clean hold”的单因素实验。

### 6.2 不值得继续把它当主线的条件

如果 `seed43/44` 都表现为：

- `normal` 明显弱于 `zero-steer`
- 或重新退回“高 dirty / 低 clean / 低 hold”

那就说明：

- `sign-safe + 0.75` 仍然主要是 `seed42` 现象
- 它还不足以算真正扩大了 good basin

这种情况下，下一步更应该回到：

- steering guidance 语义本身
- 或 near-field clean hold 稳定性

而不是直接把这版当 baseline 扩到更多训练。

## 7. 计划中的产出

这轮多 seed 验证结束后，需要补一份新的结果 md，统一整理：

- `seed42` 既有结果
- `seed43` 新结果
- `seed44` 新结果
- 以及三者的共同点 / 分叉点

最终要回答的是一句话：

> `sign-safe + 0.75` 到底是当前最好基线，还是只是 `seed42` 的局部幸运解。

## 8. 一句话总结

> 当前最合理的下一步不是继续改单变量，而是先把 `sign-safe + 0.75` 做完多 seed 验证。因为当前最重要的未知量已经不是“这个点子在 `seed42` 上行不行”，而是“它是否真的扩大了 good basin”。为节省算力，现有 `seed42` 结果直接作为 suite 第一条，接下来只补跑 `seed43/44` 并统一做固定 `3x3 normal / zero-steer` 验收。
