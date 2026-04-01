# Exp9.0 Baseline No-Success Reward Strategy Review

日期：`2026-04-01`

## 1. 背景

`Phase A` 的 `no-reference baseline` 已经完成，结果表现出一个非常明确的现象：

- `phase/frac_inserted` 已经不低
- 但 `phase/frac_hold_entry` 仍然是 `0`
- `phase/frac_success` 仍然是 `0`

这说明当前主问题不像是“完全插不进去”，而更像是：

- 策略能学到“接近并插入”
- 但奖励与控制闭环没有把策略稳定推到最终 success 区域

本页记录对
`/home/uniubi/projects/forklift_sim/forklift_pallet_insert_lift_project`
当前实现的代码排查结论，并附上通俗解释，便于后续讨论和改动落地。

## 2. 代码级排查结论

### 2.1 主引导目标点与真正 success 目标没有对齐

当前最主要的正向引导之一，是让 `fork_center` 靠近 `target_center_family`。

默认配置：

- `exp83_target_center_family_mode = "front_center"`

对应代码：

- `env_cfg.py` 中默认仍为 `front_center`
- `env.py` 中 `front_center` 和 `success_center` 使用了两套不同公式

影响：

- 训练时主要奖励在鼓励策略去追一个“和真正 success 几何不一致”的点
- 策略更容易学成“朝奖励高点冲过去”，而不是“朝最终 success 条件靠过去”

从几何公式看，当前这套 `front_center` 默认目标并不是严格贴合现有 success 判定的中心位置；在当前 `insert_fraction=0.40` 的设定下，这种不一致会把训练注意力分散到“和 success 不完全同源”的区域。

简化判断：

- success 看的是一套几何
- 主引导奖励追的是另一套几何
- 两套目标不重合，是一个高优先级问题

相关代码：

- [env_cfg.py](/home/uniubi/projects/forklift_sim/forklift_pallet_insert_lift_project/isaaclab_patch/source/isaaclab_tasks/isaaclab_tasks/direct/forklift_pallet_insert_lift/env_cfg.py#L264)
- [env.py](/home/uniubi/projects/forklift_sim/forklift_pallet_insert_lift_project/isaaclab_patch/source/isaaclab_tasks/isaaclab_tasks/direct/forklift_pallet_insert_lift/env.py#L1135)
- [env.py](/home/uniubi/projects/forklift_sim/forklift_pallet_insert_lift_project/isaaclab_patch/source/isaaclab_tasks/isaaclab_tasks/direct/forklift_pallet_insert_lift/env.py#L1146)
- [env.py](/home/uniubi/projects/forklift_sim/forklift_pallet_insert_lift_project/isaaclab_patch/source/isaaclab_tasks/isaaclab_tasks/direct/forklift_pallet_insert_lift/env.py#L1165)

### 2.2 中途奖励的“表扬标准”比最终过关标准宽松

当前 `rg` 奖励触发条件是：

- `dist_center_family < 0.28`
- `tip_y_err < 0.20`
- `yaw_err_deg < 15`

但真正进入 `hold_entry` / `success` 所依赖的关键条件更严格：

- `center_y_err <= 0.15`
- `yaw_err_deg <= 8`
- 近场时 `tip_y_err <= 0.12`

这会带来一个很典型的问题：

- 策略可以在“已经能拿到中途奖励，但还进不了 success”的区域长期停留
- 也就是 reward 在鼓励“差不多”，而 success 要求的是“真的到位”

结果上就容易出现：

- `frac_inserted` 不低
- `frac_rg` 不是 0
- 但 `frac_hold_entry` 和 `frac_success` 仍然是 0

相关代码：

- [env.py](/home/uniubi/projects/forklift_sim/forklift_pallet_insert_lift_project/isaaclab_patch/source/isaaclab_tasks/isaaclab_tasks/direct/forklift_pallet_insert_lift/env.py#L1178)
- [hold_logic.py](/home/uniubi/projects/forklift_sim/forklift_pallet_insert_lift_project/isaaclab_patch/source/isaaclab_tasks/isaaclab_tasks/direct/forklift_pallet_insert_lift/hold_logic.py#L59)
- [env_cfg.py](/home/uniubi/projects/forklift_sim/forklift_pallet_insert_lift_project/isaaclab_patch/source/isaaclab_tasks/isaaclab_tasks/direct/forklift_pallet_insert_lift/env_cfg.py#L174)
- [env_cfg.py](/home/uniubi/projects/forklift_sim/forklift_pallet_insert_lift_project/isaaclab_patch/source/isaaclab_tasks/isaaclab_tasks/direct/forklift_pallet_insert_lift/env_cfg.py#L175)
- [env_cfg.py](/home/uniubi/projects/forklift_sim/forklift_pallet_insert_lift_project/isaaclab_patch/source/isaaclab_tasks/isaaclab_tasks/direct/forklift_pallet_insert_lift/env_cfg.py#L461)

### 2.3 真正的 success / timeout 奖励没有接进总 reward

配置里虽然定义了：

- `rew_success`
- `rew_success_time`
- `rew_timeout`

但在实际 `_get_rewards()` 中，总奖励最终仍然是：

- `rew = R_plus + R_minus`

其中主要由各种 shaping 项组成，并没有看到把：

- success 终局奖励
- timeout 终局惩罚

显式接进最终 reward 和。

这意味着当前训练更像是在优化：

- “怎样拿过程分”

而不是明确优化：

- “怎样真正通关”

success 目前主要作为终止条件存在，而不是强烈的最终目标信号。

相关代码：

- [env_cfg.py](/home/uniubi/projects/forklift_sim/forklift_pallet_insert_lift_project/isaaclab_patch/source/isaaclab_tasks/isaaclab_tasks/direct/forklift_pallet_insert_lift/env_cfg.py#L352)
- [env_cfg.py](/home/uniubi/projects/forklift_sim/forklift_pallet_insert_lift_project/isaaclab_patch/source/isaaclab_tasks/isaaclab_tasks/direct/forklift_pallet_insert_lift/env_cfg.py#L356)
- [env.py](/home/uniubi/projects/forklift_sim/forklift_pallet_insert_lift_project/isaaclab_patch/source/isaaclab_tasks/isaaclab_tasks/direct/forklift_pallet_insert_lift/env.py#L2036)
- [env.py](/home/uniubi/projects/forklift_sim/forklift_pallet_insert_lift_project/isaaclab_patch/source/isaaclab_tasks/isaaclab_tasks/direct/forklift_pallet_insert_lift/env.py#L2082)
- [env.py](/home/uniubi/projects/forklift_sim/forklift_pallet_insert_lift_project/isaaclab_patch/source/isaaclab_tasks/isaaclab_tasks/direct/forklift_pallet_insert_lift/env.py#L2093)
- [env.py](/home/uniubi/projects/forklift_sim/forklift_pallet_insert_lift_project/isaaclab_patch/source/isaaclab_tasks/isaaclab_tasks/direct/forklift_pallet_insert_lift/env.py#L2261)

### 2.4 插入后过早锁死驱动与转向，堵住了最后微调空间

当前动作逻辑里：

- 一旦 `insert_depth >= _insert_thresh`
- 就直接把 `drive` 和 `steer` 置零

这会带来一个非常现实的问题：

- 策略虽然已经“插进去”
- 但它常常还没有“对正并稳住”
- 这时本来最需要的是最后一点点微调
- 但控制已经被锁死

于是策略会大量停在：

- “插入成立”
- 但“hold_entry / success 还没成立”

这和 `Phase A` 结果非常一致。

相关代码：

- [env.py](/home/uniubi/projects/forklift_sim/forklift_pallet_insert_lift_project/isaaclab_patch/source/isaaclab_tasks/isaaclab_tasks/direct/forklift_pallet_insert_lift/env.py#L1036)
- [env.py](/home/uniubi/projects/forklift_sim/forklift_pallet_insert_lift_project/isaaclab_patch/source/isaaclab_tasks/isaaclab_tasks/direct/forklift_pallet_insert_lift/env.py#L2103)
- [env.py](/home/uniubi/projects/forklift_sim/forklift_pallet_insert_lift_project/isaaclab_patch/source/isaaclab_tasks/isaaclab_tasks/direct/forklift_pallet_insert_lift/env.py#L2234)

## 3. 一条重要的排除项

这次排查里，没有发现旧的那类“Stage 1 因为 lift 被锁死，所以 success 理论上永远不可能”的硬错误仍然存在。

原因：

- `stage1_success_without_lift = true`
- `require_lift` 在 Stage 1 下已经关闭

也就是说，现在的 `success = 0` 更像是：

- 奖励目标不一致
- 过渡奖励过松
- 终局奖励不足
- 插入后控制过早冻结

这些因素叠加造成的结果。

相关代码：

- [env_cfg.py](/home/uniubi/projects/forklift_sim/forklift_pallet_insert_lift_project/isaaclab_patch/source/isaaclab_tasks/isaaclab_tasks/direct/forklift_pallet_insert_lift/env_cfg.py#L84)
- [env.py](/home/uniubi/projects/forklift_sim/forklift_pallet_insert_lift_project/isaaclab_patch/source/isaaclab_tasks/isaaclab_tasks/direct/forklift_pallet_insert_lift/env.py#L521)

## 4. 修复建议

建议优先按下面顺序做最小修复：

1. 把主引导目标从当前 `front_center` 改为和最终 success 同源的 `success_center`
2. 把 `rg` 的判定改成尽量复用 hold/success 同源几何，不再用一套更松的口径单独发奖励
3. 把 `rew_success` / `rew_timeout` 真正接回总 reward
4. 不要在“刚达到插入阈值”时就立即锁死 `drive/steer`，至少保留近场微调空间

## 5. 通俗版讲解：现在为什么会卡住

把这套训练想成在教一个新司机完成一件事：

- 把叉子稳稳插进托盘
- 姿势对
- 角度对
- 最后还能稳住

现在的问题，不像是车坏了，而像是旁边有四个教练在同时指挥，但他们说的不是同一件事。

### 5.1 第一个问题：教练把终点线画错了

真正考试要的是：

- 插得够
- 位置正
- 角度正
- 能稳住

但平时最主要的奖励却在说：

- 你继续往另一个点冲

于是学员最容易学会的，不是“稳稳对正后进去”，而是“先冲到那个奖励点再说”。

这就像驾校考试要求“把车停进车位”，但教练平时一直奖励“谁停得最靠墙谁最棒”。  
久而久之，学员会拼命往里怼，却不一定停得正。

### 5.2 第二个问题：教练发小红花的标准太松

现在有一种中途奖励，只要看起来“差不多靠近了”“姿势也还行”，就会给表扬。

可真正过关时要求更严格。

于是策略会学成这样：

- 在“已经能拿表扬，但还没真正合格”的地方待着

这像练投篮时，只要球擦到篮筐就鼓掌，但真正比赛要的是进球。  
最后球员会很会“擦边”，但不一定真能投进。

### 5.3 第三个问题：真正过关时没人特别高兴

现在训练里的大多数分数，来自一路上的各种过程分。

但“真正成功”这件事，没有被特别强烈地接进总分里。  
“最后没成功”这件事，也没有足够明确地变成代价。

所以策略容易学成：

- 很会拿过程分
- 但不一定拼命去冲最后的通关

这就像学生平时总因为“字写得工整”“步骤好看”拿表扬，但“答案对没对”反而没有那么重。  
那学生就可能越来越会写过程，却不一定真正高分。

### 5.4 第四个问题：刚插进去一点，教练就把方向盘收走了

当前逻辑里，一旦叉子插到某个深度：

- 前进不让动了
- 转向也不让动了

可很多时候，恰恰是在“已经插进去一些”之后，才最需要最后那一点点微调。

这就像倒车入库时，车尾刚进线，教练突然说：

- 好了，不许再打方向了

那车当然可能已经进去了，但大概率是歪着进去的，最后也停不正。

## 6. 通俗版讲解：如果修掉之后，训练行为会怎么变

### 6.1 先会变得没那么莽

如果把主引导目标改成和 success 一致，策略一开始可能看起来反而没有那么猛冲。

因为它不再追求：

- 先往里顶进去

而会开始学：

- 怎么用更正确的姿势进去

这时候你看到的，可能不是插入次数立刻暴涨，反而更像是动作开始变稳。

### 6.2 然后会开始学“有用的小动作”

如果把中途奖励也改成和最终过关同方向，策略在接近托盘口时，会开始更认真地做这些动作：

- 轻微摆正车头
- 收小横向偏差
- 在快接触时不再乱扭

这时最先改善的，通常不是 success 本身，而是这些中间过程：

- 对正比例提高
- 干净插入比例提高
- 托盘被顶跑的情况减少

### 6.3 再之后，真正成功会开始冒出来

如果把 success 奖励和 timeout 惩罚真正接回总 reward，策略会第一次清楚地感受到：

- 什么叫真正值钱
- 什么叫最后没做成是吃亏的

这时训练通常会经历一个过程：

- 先冒出零星成功
- 再从偶然成功变成可重复成功
- 再慢慢学会一整套稳定闭环

### 6.4 最后，插进去之后不再定格，而是会补最后半把方向

如果插入后不再立刻锁死控制，策略就终于有能力从：

- “已经插进去了”

走到：

- “插得正”
- “站得稳”
- “真正成功”

训练行为会更像一个会补动作的司机，而不是一个只会猛冲的司机。

## 7. 对 `Phase A` 结果的直观解释

当前 `Phase A` 最像下面这种状态：

- 学员已经敢冲了
- 也经常能插进去
- 但老师奖励的是另一件事
- 而且刚插进去就不让他再修正

所以最后出现的就是：

- `frac_inserted` 不低
- 但 `frac_hold_entry = 0`
- `frac_success = 0`

这并不说明训练完全没学到东西。  
更准确地说，它说明当前策略大概率学到了：

- “怎么靠近并插进去”

但还没有被奖励结构和控制逻辑正确地引导到：

- “怎么在插进去以后继续微调，直到真正过关”

## 8. 当前建议

如果只做最小闭环修复，建议直接做下面四件事：

1. 统一主引导目标到 success 几何
2. 统一中途奖励和 hold/success 几何口径
3. 把真正 success / timeout 接回总 reward
4. 放开插入后的最后微调能力

如果这四件事都做对，最期待看到的变化顺序通常是：

1. 姿势更正
2. 托盘位移更小
3. `hold_entry` 开始从 `0` 抬起来
4. `success` 开始从 `0` 抬起来
5. 成功从偶发变成稳定
