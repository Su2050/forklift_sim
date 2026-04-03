# 强化学习训练结果分析与下一步探索方法论

日期：`2026-04-03`

## 1. 目的

这份方法论的目标，不是单纯“看训练好不好”，而是把训练结果转化成下一步探索方向。

在类似当前 forklift RL 项目里，真正重要的问题通常不是：

- reward 要不要再加一项
- 某个 sigma 要不要再调小一点

而是：

- 失败主要卡在任务链条的哪一段
- 这个失败是“根本到不了”，还是“到了但过不了线”
- 问题更像 gate mismatch、局部 controllability、训练分布不足，还是环境/物理本身有 bug

只有把这几个问题分清楚，下一步探索才会有方向感，而不是落入盲目调参。

## 2. 总体原则

### 2.1 先定位阶段，再讨论参数

不要先看最终 `success`，要先看任务漏斗。

在当前类任务里，可以把任务拆成：

1. 接近目标
2. 插入
3. 插入后对齐
4. pre-hold
5. hold
6. success

如果只盯最终 `success=0`，你只知道“没成”，不知道“卡在哪”。

### 2.2 先排除结构性问题，再做数值微调

优先排查：

- 成功判定是否合理
- 物理上是否可达
- 观测和动作是否真的透传
- 某一阶段是否被错误冻结
- 某个 gate 是否过严

只有在这些结构性问题基本明确后，reward 权重和 sigma 微调才值得做。

### 2.3 优先做可证伪的实验

好的下一步实验，应该能明确区分两种解释。

例如：

- `strict vs relaxed gate` 是在检验“是不是门太窄”
- `freeze ablation` 是在检验“是不是插入后过早冻结”
- `fixed-case replay` 是在检验“这个状态物理上还能不能被纠回来”

如果一个实验做完以后，仍然不能缩小不确定性，那它的信息增益就不高。

### 2.4 短跑用来辨方向，多 seed 用来下结论

建议把实验分成两个层次：

- `短跑 / 小预算`：用来判断方向对不对
- `多 seed / 稍长训练`：用来验证这个结论是不是稳

单 seed 很适合发现现象，但不适合最终定性。

## 3. 指标体系

建议把指标分成三层。

### 3.1 结果指标

这是最终任务层面的指标：

- `phase/frac_success`
- `phase/frac_success_strict`
- 最终 episodic success

这些指标告诉我们“有没有成”，但通常不足以告诉我们“为什么没成”。

### 3.2 阶段指标

这是最关键的一层，负责回答“卡在哪一段”：

- `phase/frac_inserted`
- `phase/frac_aligned`
- `phase/frac_prehold_reachable_band`
- `phase/frac_prehold_reachable_band_companion`
- `phase/frac_hold_entry`
- `diag/max_hold_counter`

这层指标通常决定下一步探索方向。

### 3.3 几何 / 物理指标

这是解释机制的一层，负责回答“为什么卡住”：

- `err/center_lateral_inserted_mean`
- `err/tip_lateral_inserted_mean`
- `err/yaw_deg_inserted_mean`
- `diag/pallet_disp_xy_inserted_mean`
- `phase/frac_inserted_push_free`
- 接触、漂移、z 合法性、hold exit 等诊断

如果阶段指标告诉你“卡在 pre-hold”，几何/物理指标负责解释：是 lateral 压不下来，还是 yaw 不行，还是深插入后已经缺少有效横向纠偏自由度。

## 4. 标准分析流程

### 4.1 第一步：先确认这是不是学习问题

优先用 validation、固定 case、脚本回放去排除：

- success 判定坏了
- 物理不可达
- 观测异常
- 动作没有真实作用到系统
- 环境逻辑在某一阶段错误冻结

如果这一层没排清，后面所有训练分析都可能是伪问题。

### 4.2 第二步：做任务漏斗定位

把训练结果按阶段看成漏斗，而不是看单个最终指标。

典型判读方式：

- `inserted` 低：前段接近或插入本身没学到
- `inserted` 高但 `hold` 低：说明不是前段问题
- `prehold band` 高但 `hold` 低：说明像是“快到线但转化不了”
- `hold` 有但 `success` 低：说明卡在最后保持或成功闭环

### 4.3 第三步：区分“没到过”还是“到过但转化不了”

这是决定下一步方向最重要的问题。

两种情况完全不同：

- `没到过`
  说明训练分布、探索、早段几何收敛、观测或 controller 更值得优先看

- `到过但转化不了`
  说明更像 gate mismatch、local controllability、post-hold refinement 或成功判定问题

当前项目里，`prehold_reachable_band` 就是典型的“到过但还没过线”的诊断指标。

### 4.4 第四步：看“偶发成功”还是“稳定成功”

要区分下面两种现象：

- 偶尔 `success > 0`
- 持续稳定地 `hold/success` 抬升

偶发命中只说明策略碰到过 success manifold，不代表已经真正学会。

### 4.5 第五步：做跨 seed 对比

同一个现象，如果多个 seed 都出现，说明更像机制性现象；
如果只有单个 seed 出现，说明可能只是训练噪声或初始化差异。

建议至少看：

- last-20 或 last-50 mean
- 首次命中 iteration
- 事件计数
- 每个 seed 的方向是否一致

## 5. 常见失败类型与对应探索方向

| 现象 | 更可能的解释 | 下一步优先方向 |
| --- | --- | --- |
| `inserted` 长期很低 | 早段接近、插入策略、观测或探索不足 | reset / obs / preinsert shaping / curriculum |
| `inserted` 高，但 `hold` 基本没有 | 问题集中在插入后纠偏或 hold gate | pre-hold diagnostics / gate ablation / controllability 检查 |
| `prehold band` 高，但 `hold` 低 | 到达 near-pass 区，但 strict gate 卡住 | gate sweep / strict-vs-relaxed A/B |
| `hold` 有，但 `success` 低 | hold 后 refinement 不足，或 success 判定太严 | post-hold replay / success gate 诊断 |
| 偶尔 `success > 0`，但不稳定 | 策略能碰到 success manifold，但不可持续 | 稳定性分析 / 多 seed / hold persistence |
| 单个 seed 好，其他 seed 不行 | 高方差或训练不稳 | 多 seed 验证 / 降低偶然性 / 更稳的训练配置 |
| controller 改很多仍救不回固定 case | 问题可能不是 controller 太弱，而是物理坏盆地或 gate mismatch | fixed-case replay / geometry reachability / gate-threshold sweep |

## 6. 如何设计下一步实验

下一步实验应该优先满足四个条件。

### 6.1 一次只问一个明确问题

例如：

- “是不是 `tip gate` 太严？”
- “是不是 inserted 后过早 freeze？”
- “Case A 这种状态物理上还能不能纠回来？”

不要在同一轮实验里同时改 reward、obs、reset、controller、success gate。

### 6.2 尽量只改一个变量

这样才能把结果解释清楚。

好实验：

- 只放宽 `tip_align_entry_m`
- 只新增一个 companion diagnostic band
- 只在固定 case 下改 controller

坏实验：

- 同时放宽 gate、改 reward、改 reset、再换 backbone

### 6.3 优先选信息增益最高的实验

判断标准不是“哪个更可能提高分数”，而是“哪个更能缩小不确定性”。

通常优先级可以是：

1. fixed-case diagnostics
2. gate / freeze / success criterion ablation
3. 短跑 A/B
4. 多 seed 复验
5. 长训练大预算
6. reward 细调

### 6.4 先用短跑看方向，再决定要不要上大预算

推荐节奏：

1. 固定 case 诊断
2. `1 seed + 小预算` 做方向验证
3. `2+ seed + 稍长预算` 验证结论
4. 只有方向被证实后，再开正式长训练

## 7. 如何写出“下一步建议”

一份好的训练分析，最后不应该只是“建议继续调 reward”，而应该明确回答：

1. 当前主 bottleneck 在哪一段
2. 当前最可信的解释是什么
3. 还有哪个关键不确定性没被排除
4. 下一步做哪个实验最能提高信息增益

建议采用下面这个结论模板。

### 7.1 结论模板

**当前瓶颈**

- 任务主要卡在 `X -> Y` 阶段

**支持证据**

- 指标 A 表明……
- 指标 B 表明……
- 固定 case / replay 表明……

**当前最可信解释**

- 更像是 `gate mismatch / local controllability / training distribution / success criterion`

**仍未排除的不确定性**

- 例如：是否只在单个 seed 成立
- 例如：是否只在短跑中成立

**下一步实验**

- 做一个只改 `X` 的 A/B
- 预算是 `1~2 seed + 100~200 iter`
- 目标是验证“解释 A”还是“解释 B”

## 8. 在当前 forklift 项目里的具体用法

结合当前项目，推荐优先使用下面这套判读逻辑：

### 8.1 如果 `frac_inserted` 高，但 `frac_hold_entry` 很低

优先判断：

- 是否存在 `prehold_reachable_band > hold`
- 是否存在固定 case 下“能动但纠不回来”

如果是，下一步优先做：

- gate sweep
- strict vs relaxed 短跑 A/B
- fixed-case replay

而不是先继续加 shaping。

### 8.2 如果 strict 组里 `band` 高，relaxed 组里 `hold/success` 更高

这通常说明：

- strict 条件下存在 near-pass 堆积
- relaxed gate 把其中一部分转化成了 hold 或 success

这时下一步就不该再说“模型完全不会”，而应该转向：

- 如何定义更合理的 success / hold gate
- 如何把 near-pass 区进一步稳定地变成 strict success

### 8.3 如果 fixed-case controller ablation 也救不回来

这通常说明：

- 问题不只是 policy 训练强度不够
- 可能是局部物理几何坏盆地
- 也可能是当前 gate 与可达带不匹配

这时应优先继续做：

- replay + 几何时间序列分析
- gate threshold sweep
- reachability diagnostics

而不是盲调 controller。

## 9. 一个简明版的工作流

如果要把这套方法浓缩成日常执行流程，可以记成下面 7 步：

1. 先排除环境/物理/判定 bug。
2. 用阶段指标搭出任务漏斗。
3. 找出主 bottleneck 所在阶段。
4. 区分“没到过”还是“到过但没转化”。
5. 用几何/物理指标解释为什么卡住。
6. 设计只改一个变量的对照实验。
7. 用多 seed 和稍长预算验证方向。

## 10. 最终标准

一个好的下一步探索方向，不是“看起来可能有效”，而是同时满足：

- 能明确检验一个假设
- 能明显缩小当前不确定性
- 改动范围小，解释清楚
- 即使结果是否定的，也能为后续决策提供信息

如果一轮实验做完后，你更清楚“问题不是什么”，那它依然是有价值的好实验。
