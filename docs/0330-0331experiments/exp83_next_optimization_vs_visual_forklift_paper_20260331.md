# Exp8.3 下一步优化方向：对照 Visual-Based Forklift Learning System

日期：2026-03-31

参考论文：

- [Visual-Based Forklift Learning System Enabling Zero-Shot Sim2Real Without Real-World Data.pdf](/home/uniubi/projects/forklift_sim/docs/Visual-Based%20Forklift%20Learning%20System%20Enabling%20Zero-Shot%20Sim2Real%20Without%20Real-World%20Data.pdf)

## 1. 论文里真正做了什么

这篇论文的核心不是“先写一条非常强的手工轨迹，再让策略死命跟轨迹”，而是下面这套结构：

1. 把任务拆成两段
   - approach policy：视觉 PPO，只负责开到托盘前面
   - decision policy：停稳后判断“现在能不能抬”

2. approach policy 的输入很朴素
   - 左右两侧相机
   - 速度信息
   - 上一步动作
   - 图像先过 ImageNet 预训练 ResNet

3. reward 确实用了 reference trajectory
   - 但它的作用更像“靠近/对齐的弱引导”
   - 不是把整个任务定义成“严格轨迹跟踪”

4. 他们的成功目标比我们当前主线更窄
   - 重点是 approach 到合适位置
   - 然后由一个单独 classifier 决定 lift
   - 不是一个 PPO 同时解决“接近 + 干净插入 + hold + success”

5. sim2real 主要靠 photorealism + domain randomization
   - 地面颜色
   - 托盘/支架颜色
   - 光照强度和色温
   - 观测速度噪声
   - 动作扰动

## 2. 这篇论文和我们现在最大的差异

### 2.1 他们没有把 RL 主目标压成“插入后稳定 hold”

他们把“能不能抬”单独拿出去做 decision policy 了。

这很重要，因为这意味着：

- RL 不需要在同一个优化地形里同时解决
  - 怎么靠近
  - 怎么摆正
  - 怎么 clean insert
  - 怎么保持 hold

而我们现在的主线，实际上是把这些都塞进一个 reward 里了。

### 2.2 他们的 reference trajectory 是辅助项，不是系统唯一几何真相

论文里 reference trajectory 用来给正向引导，这没有问题。

但我们现在的问题是：

- 当前系统里 `d_traj / yaw_traj_err / signed traj obs / out_of_bounds` 都和它深度绑定
- 而我们刚审计过，这条轨迹当前只是 `fork_center-only` 的几何代理
- 它还没有通过整车可行性检查

所以在我们这里，trajectory 已经不是“辅助引导”，而是“核心真相”，这比论文里的依赖程度更重，也更危险。

### 2.3 他们的 approach task 更像“开到位”，我们现在更像“直接学插”

论文实物结果里，成功标准更接近：

- 开到合适位置
- 停下
- 再判定 lift

而我们当前阶段一直在逼 PPO 直接学：

- 近场纠偏
- 插进去
- 不推盘
- hold
- success

所以两者的任务难度并不在一个层级。

## 3. 从论文里该借鉴什么

### 3.1 先把任务拆开

这是当前最该借鉴的一点。

对我们来说，下一步不应该继续默认“一个视觉 PPO 直接把 clean insert + hold 全学出来”，而应该先拆成：

1. `approach/alignment policy`
   - 只负责从可见范围开到托盘前的可插入位姿

2. `insert-or-lift gate`
   - 先判断当前是否已经满足插入/抬升前置条件

3. `final insert / lift execution`
   - 可以先做成规则控制或更简单的子策略

### 3.2 视觉输入应该尽量贴近操作者真实可见信息

论文选择左右侧相机，并且就是为了看 forks 和 pallet 的对位关系。

这对我们当前也很重要：

- 如果我们的视觉里 fork/pallet 相对关系不直接、不可见或太弱
- 再好的 trajectory reward 也很难逼出稳定 steering

所以要重新检查：

- 当前相机视角里，叉尖/叉臂/托盘口是否真的同时清楚可见
- 在 misalignment case 下，图像里有没有足够强的左右差异

### 3.3 参考轨迹只能当辅助，不能当唯一几何代理

论文里它可以工作，是因为：

- 任务是 approach 为主
- 轨迹更多是 reward shaping
- 最终 lift 决策被拆了出去

对我们来说，trajectory 下一步最合理的角色应该是：

- 仅用于远场 approach/alignment shaping
- 不再主导最终 insert / hold 的全部判据

## 4. 从论文里不该照搬什么

### 4.1 不能继续默认 fork-center-only trajectory 就够了

论文中也在看 fork 的路径，但他们没有把这个代理发展成我们现在这么重的系统依赖。

我们已经确认：

- 当前 trajectory preflight 只检查 `s_start < s_pre < s_goal`
- 但 implied root proxy 已经能出现非常夸张的转向/侧摆

所以我们不能因为论文也用了 fork path，就得出“那我们这套也没问题”。

### 4.2 不能在 trajectory 本身可疑时继续大规模扫 PPO

论文的成功前提之一是：

- 任务定义相对干净
- policy 主要学 approach

而我们现在最基础的问题是 trajectory 代理还没站稳。

所以当前继续做：

- 更多 seed
- 更长 iter
- 更多 reward weight

都不是最优先。

## 5. 针对当前系统，下一步该怎么优化

我建议按下面的顺序推进。

### Phase A：先修 trajectory，而不是先训更多 PPO

目标：

- 把当前参考轨迹从“fork-center-only 的弱代理”升级成“至少经过整车 proxy 审计的引导”

具体做法：

1. 做 `Trajectory Preflight V2`
   - 输入还是当前 reset case grid
   - 输出不只看 `delta_s`
   - 还看：
     - `root_y_abs_max`
     - `root_heading_change_deg`
     - `root_curvature_max`

2. 把 trajectory 生成从“fork_center Hermite”升级成“root/rear-axle path first，再映射到 fork center”
   - 最简单版本先用 bicycle-model proxy
   - 不需要一上来做高保真动力学
   - 但必须把最小转弯半径和整车朝向连续性接进去

3. 控制变量
   - 只改 trajectory generator
   - reward、actor、camera 全都先不动

通过标准：

- current stage1 case grid 下，不再出现“entry 全绿，但 implied root 明显离谱”的现象

### Phase B：把 PPO 目标缩回到 approach/alignment

目标：

- 不再让 PPO 一次性负责 clean insert + hold

建议做法：

1. 新建 `approach_only` 任务口径
   - success = 到达可插入位姿并停稳
   - 不要求已经 lift
   - 甚至不要求已经 deep insert

2. reward 只保留下面这些
   - 接近托盘前沿
   - 减小 lateral / yaw error
   - 不推盘
   - 不发生 wrong-sign steering
   - 停稳 bonus

3. 暂时去掉“必须通过插入后 hold 才算成功”这层苛刻定义

控制变量：

- 只改 task definition
- 先不改 domain randomization
- 先不扫更多 reward weight

通过标准：

- `normal > zero-steer`
- 至少 2/3 seeds 在 50 iter 内稳定达到非零 approach success

### Phase C：单独做 insert/lift decision

目标：

- 模仿论文，把“现在能不能执行下一步”从 PPO 主任务里拆出去

可以先做两个简单版本：

1. `insert_ready` classifier
   - 输入：当前左右相机 + 低维状态
   - 输出：当前是否已经到达可执行插入/抬升的位姿

2. `rule-based execution`
   - 一旦 classifier 通过
   - 就进入一个保守规则控制：
     - 低速前进
     - lift
     - reverse

这一步的价值是：

- 先验证“approach 学会后，系统能否完成后半段”
- 而不是继续让 PPO 背整个任务

### Phase D：最后再回到 clean insert / hold 强化

只有 A/B/C 都过了，才值得继续强化：

- push-free clean insert
- hold stabilization
- 更大范围 reset
- 长训练

## 6. 最值得先做的 3 个实验

### 实验 1：Vehicle-aware trajectory preflight

目的：

- 证明新 trajectory generator 比旧版更像整车可行路径

只改：

- trajectory generator

不改：

- reward
- actor inputs
- PPO 超参

看：

- trajectory audit manifest
- worst-case root proxy 指标

### 实验 2：Approach-only PPO on new trajectory

目的：

- 验证去掉“插入后 hold”耦合之后，PPO 能否先学会真正 steering-based approach

只改：

- success/task definition

不改：

- camera
- backbone
- PPO 基本超参

看：

- `3x3 normal vs zero-steer`
- early steer sign usage
- 3 seeds x 50 iter

### 实验 3：Insert-ready classifier

目的：

- 验证 approach 学会之后，后半段是否可以靠更简单的 gate 接上

只改：

- stationary endpoint dataset + binary classifier

不改：

- approach PPO

看：

- endpoint image 上的分类准确率
- 实际回放中 false positive / false negative

## 7. 当前最不值得先做的事

现在不建议优先做：

- 继续扫更多 steer scale
- 继续扫更多 clean bonus weight
- 直接把当前配置拉到几百 iter
- 在 trajectory 未修前继续做多 seed 大批量训练

这些动作的共同问题是：

- 它们默认 trajectory 的教学方向大体正确

而当前这个前提已经不成立。

## 8. 一句话决策

如果只用一句话概括下一步：

**先把“参考轨迹”从弱几何代理修成至少经过整车 proxy 审计的 approach guide，再把 PPO 目标收缩成 approach/alignment，最后用单独的 decision policy 去接 insert/lift。**

这条路线和论文最一致，也最适合我们当前暴露出来的问题。

