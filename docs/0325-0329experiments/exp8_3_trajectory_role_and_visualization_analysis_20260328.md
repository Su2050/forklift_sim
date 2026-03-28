# Exp8.3 参考轨迹在训练中的作用，以及缺失的轨迹可视化验证

日期：2026-03-28

## 1. 问题背景

在最近的 steering 诊断里，我们已经确认：当前 near-field 配置下，强 checkpoint 的 `normal` 和 `zero-steer` 表现几乎一样，说明策略主要学到的是“往前叉”，而不是“依靠转向纠偏后再插入”。

这会自然引出两个问题：

1. 系统明明有参考轨迹，理论上在 `y / yaw` 偏差更大时，轨迹不应该帮助 agent 学会转向吗？
2. 现在训练到底是不是“根据轨迹明确知道该往哪边打方向”，还是只是“把轨迹拿来做奖励 shaping，靠 PPO 自己摸索”？

这两个问题都很关键。而且从工程角度看，当前还有一项非常重要但尚未完成的验证：**把训练时真实生成的参考轨迹可视化出来**，确认轨迹本身是不是合理，尤其是它在起始几步到底有没有提供“清晰的转向几何”。

## 2. 当前系统里，参考轨迹到底是怎么用的

### 2.1 参考轨迹是在 reset 时现生成的

当前环境会在 reset 时，为每个 env 生成并缓存一条参考轨迹，函数是：

- [env.py:996](/home/uniubi/projects/forklift_sim/forklift_pallet_insert_lift_project/isaaclab_patch/source/isaaclab_tasks/isaaclab_tasks/direct/forklift_pallet_insert_lift/env.py#L996)

它做的事情是：

- 起点 `p0` 用的是 **当前叉臂中心**，不是车体中心
- 起始切线 `t0` 用的是 **当前机器人 yaw**
- 终点附近先定义一个 `p_pre`（pre-align 点）
- 再用 **三次 Hermite 样条 + 末段直线** 组成参考轨迹

这有一个非常重要的含义：

**这条轨迹不是一个固定的“标准模板”，而是每次 reset 后，根据当前起点位姿现生成的。**

也就是说，轨迹的起始方向本来就是顺着当前叉车朝向去接的。它不是一个“从外部强行告诉你左打还是右打”的绝对参考，而更像是一条“从当前状态到目标插入点的几何走廊”。

### 2.2 训练时会查询“当前叉臂中心相对轨迹”的几何量

运行时环境会查询：

- 到轨迹最近点的距离 `d_traj`
- 当前朝向和最近点切线方向的偏航误差 `yaw_traj_err_deg`
- 最近点对应的轨迹进度 `s_traj_norm`

对应函数是：

- [env.py:1259](/home/uniubi/projects/forklift_sim/forklift_pallet_insert_lift_project/isaaclab_patch/source/isaaclab_tasks/isaaclab_tasks/direct/forklift_pallet_insert_lift/env.py#L1259)

注意这里也有一个很关键的细节：

- `d_traj` 是**到轨迹的最短距离**
- `yaw_traj_err_deg` 是**当前朝向和轨迹切线的绝对偏差**

也就是说，这里给出的本质上是“偏了多少”，不是“该往左还是该往右”的直接监督。

### 2.3 轨迹主要通过 reward shaping 进入训练

当前 reward 里和轨迹直接相关的主项是：

- `r_d_raw = 1 / dist_center_family`
- `r_cd_raw = 1 / d_traj`
- `r_cpsi_raw = 1 / yaw_traj_err_rad`

位置在：

- [env.py:1604](/home/uniubi/projects/forklift_sim/forklift_pallet_insert_lift_project/isaaclab_patch/source/isaaclab_tasks/isaaclab_tasks/direct/forklift_pallet_insert_lift/env.py#L1604)

其中：

- `r_d` 更偏向“接近目标中心族”
- `r_cd` 更偏向“贴着参考轨迹走”
- `r_cpsi` 更偏向“朝向和轨迹切线对齐”

但它们依然都是**标量奖励**，不是一个显式控制器。

所以，当前系统的本质不是：

“给定轨迹，系统明确告诉 agent 该往左打还是往右打”

而是：

“给定轨迹，系统根据你离轨迹有多远、朝向和切线差多少，给你高低不同的 reward；agent 通过 PPO 自己去学一套动作策略”

换句话说，**现在并不是轨迹跟踪控制，而是轨迹 shaping RL。**

## 3. 为什么“理论上应该会转向”，但实际没明显学出来

### 3.1 轨迹并没有直接把“转向符号”喂给 actor

从观测构造看，环境确实计算了带符号的：

- `y_err_obs`
- `yaw_err_obs`

位置在：

- [env.py:1487](/home/uniubi/projects/forklift_sim/forklift_pallet_insert_lift_project/isaaclab_patch/source/isaaclab_tasks/isaaclab_tasks/direct/forklift_pallet_insert_lift/env.py#L1487)

但关键在于：**actor policy 实际并没有直接吃这两个量。**

当前 PPO 的 `obs_groups` 是：

- [rsl_rl_ppo_cfg.py:39](/home/uniubi/projects/forklift_sim/forklift_pallet_insert_lift_project/isaaclab_patch/source/isaaclab_tasks/isaaclab_tasks/direct/forklift_pallet_insert_lift/agents/rsl_rl_ppo_cfg.py#L39)

也就是：

- `policy = [image, proprio]`
- `critic = [critic]`

而 camera 模式下，policy 的 `proprio` 实际是 `easy8`：

- [env.py:1368](/home/uniubi/projects/forklift_sim/forklift_pallet_insert_lift_project/isaaclab_patch/source/isaaclab_tasks/isaaclab_tasks/direct/forklift_pallet_insert_lift/env.py#L1368)

内容只有：

- `v_x_r, v_y_r, yaw_rate, lift_pos, lift_vel, prev_drive, prev_steer, prev_lift`

也就是说，**actor 不直接看到带符号的 `y_err_obs / yaw_err_obs`，这些更完整的几何量主要在 critic 的低维 privileged state 里。**

这意味着：

- 对 actor 来说，“往左打还是往右打”并不是低维输入里直接告诉它的
- 它要么从图像里自己推断，要么靠探索慢慢学出来

这就大大削弱了“轨迹会自动教会 steering”这件事。

### 3.2 当前 stage1 初始化太对齐，前推捷径太容易

当前 `stage1` 的 reset 范围已经比最初稍微放宽，但依然不大：

- [env_cfg.py:104](/home/uniubi/projects/forklift_sim/forklift_pallet_insert_lift_project/isaaclab_patch/source/isaaclab_tasks/isaaclab_tasks/direct/forklift_pallet_insert_lift/env_cfg.py#L104)
- [env_cfg.py:106](/home/uniubi/projects/forklift_sim/forklift_pallet_insert_lift_project/isaaclab_patch/source/isaaclab_tasks/isaaclab_tasks/direct/forklift_pallet_insert_lift/env_cfg.py#L106)

也就是：

- `y = ±0.08m`
- `yaw = ±3°`

在这个范围下，即使轨迹本身是合理的，agent 也仍然很可能通过“主要往前推”来获得不低的早期回报。

这就导致一个现象：

- 在理论上，轨迹对大偏差情况当然应该有帮助
- 但在当前训练分布里，很多样本根本没把 steering 变成“必要技能”

### 3.3 当前 reward 更像在奖励“离轨迹近”，不是在给 steering 指令

这点很重要：

- `r_cd` 奖励你离轨迹近
- `r_cpsi` 奖励你朝向贴近切线

但它们都没有直接构造成“当前该左打/右打多少”的监督信号。

所以如果：

- reset 本来就很对齐
- actor 又没有直接吃带符号几何误差
- reward 只是标量偏好而不是控制标签

那么训练出来的最容易的局部策略，就很可能是：

**“先学会往前叉；只在少数情况下顺带蹭到一点 steering，而不是系统性学会纠偏。”**

## 4. 所以你刚才提的判断是对的

你的怀疑是成立的：

> 如果有了轨迹，叉车大致应该往哪儿打方向不应该也是很明确的吗？

从“人类几何直觉”来说，这句话是对的。  
但从“当前 RL 实现”来说，问题在于：

1. 轨迹确实定义了一个几何方向场
2. 但这个方向场没有被显式翻译成 actor 的 signed steering supervision
3. actor 主要靠 `image + easy8` 在学
4. reward 只是沿轨迹的 shaping，不是轨迹跟踪控制器

所以当前系统并不是“参考轨迹明确教 steering”，而更像是：

**“参考轨迹提供了一个几何偏好，但这个偏好目前还不足以保证 actor 真正学会 steering。”**

## 5. 缺失的关键验证：把训练时真实生成的参考轨迹画出来

这是我非常同意你的一点：  
**现在最缺的可视化，不是再看 rollout 视频，而是直接看 runtime 里真实生成的参考轨迹。**

因为在当前阶段，我们必须把两个问题拆开：

1. 轨迹本身是不是合理的？
2. 轨迹是合理的，但 agent 没学会用它？

如果不把真实轨迹画出来，我们很容易把这两件事混在一起。

## 6. 最应该怎么可视化

### 6.1 第一优先级：俯视图直接把轨迹画上去

这是最有价值、信息密度最高的第一步。

建议在 top-down 图里同时画：

- 托盘位置与朝向
- 托盘前沿 / 插入轴线
- 叉车初始 `fork_center`
- 叉车初始朝向箭头
- 生成的参考轨迹（Hermite + 直线）
- `p_pre` 和 `p_goal`
- 轨迹上的切线箭头（每隔若干采样点画一次）

最好一张图里直接叠这几类 case：

- 当前 `stage1` 默认 reset
- `misalignment grid` 里的代表性点
  - 成功点
  - dirty 点
  - timeout 点

这样一眼就能看出：

- 轨迹是不是光滑且接线正确
- 在偏差更大的格点上，轨迹一开始到底有没有给出明确“向左/向右修正”的几何趋势
- 某些失败点是不是其实轨迹入口就很差

### 6.2 第二优先级：把 rollout 实际走过的轨迹叠到参考轨迹上

这一步是用来回答：

- 参考轨迹本身没问题，但 agent 没跟
- 还是参考轨迹本身就不提供足够强的 steering 几何

俯视图里建议同时画：

- 参考轨迹
- 实际 `fork_center` 轨迹
- 关键时刻姿态箭头

最好直接对比：

- `normal policy`
- `zero-steer`

如果两条 rollout 都和参考轨迹差不多，那说明当前轨迹入口可能本来就接近直插。  
如果 `normal` 和 `zero-steer` 都明显偏离轨迹，那说明 agent 并没有真正利用这条参考路径。

### 6.3 第三优先级：相机画面 + 俯视 inset 联动

这个不是第一步必须做，但后面很有价值。

原因是现在 actor 主要靠图像决策，所以要回答：

- 从视觉上，当前帧是否真的足够让 actor 判断“该往哪边打方向”

最直观的方法是：

- 左边放第一视角图像
- 右上角放 top-down 轨迹和当前位姿
- 底部打出当前 `y_err / yaw_err / d_traj / yaw_traj_err_deg`

这一步更像“策略可解释性验证”，优先级低于前两个。

## 7. 现有脚本里，哪些能用，哪些不能直接用

### 7.1 现有 `verify_trajectory_and_fov.py` 不能直接当当前验证

- [verify_trajectory_and_fov.py](/home/uniubi/projects/forklift_sim/scripts/verify_trajectory_and_fov.py)

这个脚本还是旧版逻辑：

- 用的是旧 Bézier 轨迹
- 用的是固定 toy setup
- `s_front=-0.6`
- 保存路径也还是旧目录

所以它不能直接代表当前环境里的 runtime 参考轨迹。

### 7.2 现有 `test_exp83_geometry_preflight.py` 很有用，但它不是 runtime 可视化

- [test_exp83_geometry_preflight.py](/home/uniubi/projects/forklift_sim/forklift_pallet_insert_lift_project/tests/test_exp83_geometry_preflight.py)

它能验证：

- Hermite + 直线段生成是否和 env 一致
- 端点、弧长、查询几何是否正确

这对“轨迹数学是否自洽”很重要。  
但它还不能回答：

- 训练时真实 reset 出来的轨迹长什么样
- 对那些实际失败/成功 case，轨迹在俯视图上是否合理

所以它是必要但不充分。

## 8. 我建议的落地顺序

### Step 1

先做一个 **runtime top-down 轨迹可视化脚本**，输入：

- 一个 checkpoint 或纯 reset 模式
- 若干指定的 `y / yaw` case

输出：

- 参考轨迹俯视图 PNG
- 轨迹采样点 CSV

### Step 2

在同一套 case 上叠加 rollout 实际轨迹：

- `normal`
- `zero-steer`

输出：

- 俯视对比图
- 每条 trajectory 的关键指标表

### Step 3

只有在确认参考轨迹本身没问题之后，才继续把重点放回 curriculum / reward。

否则会有风险：

- 其实轨迹入口几何就没有很好地表达 steering
- 我们却把问题全怪在 reward 或 PPO 方差上

## 9. 当前结论

当前最准确的结论是：

1. 系统里确实有“参考轨迹”
2. 但现在的训练方式本质上是 **trajectory-shaped PPO**，不是显式轨迹跟踪控制
3. actor 并没有直接拿到 signed 的 `y_err / yaw_err` 作为低维 steering 监督
4. 在当前较窄的 `stage1` reset 下，agent 完全可能主要学会“往前叉”
5. 因此，你提出的“把真实 runtime 参考轨迹可视化出来”非常关键，而且这一步目前确实还没做完整

如果只能先做一个可视化，我会优先做：

**俯视图直接画 runtime 参考轨迹 + 起点位姿 + 托盘轴线。**

因为这是最快、最直接区分“轨迹本身有问题”还是“agent 没学会用轨迹”的方法。
