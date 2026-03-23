# PPO 中的 `step`、`rollout`、`episode` 与 `36s` 超时：以叉车任务为例

本页适合：刚接触本项目 PPO 训练流程，容易被 `physics step`、`env step`、`rollout 64 steps`、`episode_length_s=36.0` 这些概念绕晕的读者。

本文基于当前任务实现与训练配置：

- 任务环境：`forklift_pallet_insert_lift_project/isaaclab_patch/source/isaaclab_tasks/isaaclab_tasks/direct/forklift_pallet_insert_lift/env.py`
- 环境配置：`forklift_pallet_insert_lift_project/isaaclab_patch/source/isaaclab_tasks/isaaclab_tasks/direct/forklift_pallet_insert_lift/env_cfg.py`
- PPO 配置：`forklift_pallet_insert_lift_project/isaaclab_patch/source/isaaclab_tasks/isaaclab_tasks/direct/forklift_pallet_insert_lift/agents/rsl_rl_ppo_cfg.py`
- 策略网络：`forklift_pallet_insert_lift_project/isaaclab_patch/source/isaaclab_tasks/isaaclab_tasks/direct/forklift_pallet_insert_lift/vision_actor_critic.py`
- IsaacLab 基类：`IsaacLab/source/isaaclab/isaaclab/envs/direct_rl_env.py`

---

## 1. 一张表先把概念对齐

| 概念 | 在本项目里的意思 | 当前值 |
|------|------------------|--------|
| `physics step` | 物理引擎内部最小仿真步长 | `dt = 1/120 s` |
| `env step` | Agent 做 1 次决策、环境推进 1 次 | `decimation * dt = 4 * 1/120 = 1/30 s` |
| `rollout chunk` | PPO 一次参数更新前，每个 env 先采的一小段轨迹 | `64 env steps` |
| `iteration` | 1 次完整 PPO 外循环：采样 -> GAE -> 更新 | 每个 env 采 `64` 步，再做 `5 * 4 = 20` 次 mini-batch 更新 |
| `episode` | 1 个 env 从 `reset` 开始到成功/失败/超时结束的“一次完整尝试” | 最长 `36 s` |
| `max_episode_length` | `episode_length_s` 换算成 env step 后的上限 | `1080 env steps` |

最重要的直觉是：

- **`64 steps` 不是一整局任务的长度，只是 PPO 一次暂时拿来学习的一小段轨迹。**
- **一整局真正的上限是 `36 s`，也就是大约 `1080 env steps`。**

---

## 2. 本项目当前 PPO 到底在训练什么

### 2.1 当前用的是非对称 Actor-Critic

当前任务注册会把 PPO 训练器指到自定义配置和自定义策略网络：

- `__init__.py` 里注册任务 ID：`Isaac-Forklift-PalletInsertLift-Direct-v0`
- `rsl_rl_ppo_cfg.py` 里定义 PPO 超参数
- `vision_actor_critic.py` 里定义网络结构

当前 observation group 的设计是：

- `policy`：`image + proprio`
- `critic`：`critic`

也就是说：

- **Actor 看图像 + easy8 proprio**
- **Critic 不看图像，只看 15 维低维 privileged state**

这是典型的 **asymmetric critic** 设计：让 critic 学得更稳，但 policy 仍然按真实部署条件决策。

### 2.2 当前策略网络结构

当前配置对应的是 `VisionActorCritic`：

- 视觉 backbone：`resnet34`
- `imagenet_backbone_init=True`
- `freeze_backbone=True`
- actor 隐藏层：`[256, 256, 128]`
- critic 隐藏层：`[256, 256, 128]`
- activation：`ELU`
- 动作分布：高斯策略，`noise_std_type="log"`，`init_noise_std=0.4`

更细一点：

- 图像先过 `ResNet34` backbone
- 再过一个 `image_proj` 两层 MLP 投到 `256` 维
- `proprio` 过一个小 MLP 编成 `128` 维
- 两者拼接后喂给 actor head
- critic 则单独吃 15 维低维状态，输出标量 `V(s)`

### 2.3 当前任务默认是 Stage 1，动作维度实际重点是 `drive + steer`

虽然环境内部统一维护 3 维动作缓存，但当前默认配置里：

- `action_space = 2`
- 第 3 维 `lift` 会在环境里自动补成 0
- `alpha_lift = 0.0`

所以当前这套默认训练，重点还是：

- 接近托盘
- 沿参考轨迹前进
- 对齐后尝试插入前段

而不是完整的“成熟举升策略”。

---

## 3. 一个 `iteration` 内，PPO 实际做了什么

可以把 1 个 `iteration` 看成下面 3 步：

1. **采样 rollout**
2. **算 GAE / return**
3. **做 PPO 更新**

### 3.1 先采样：128 个 env 并行，每个先采 64 步

当前 runner 配置：

- `num_envs = 128`
- `num_steps_per_env = 64`

所以 1 个 iteration 会收集：

```text
128 * 64 = 8192 条 transition
```

这里的 “1 step” 指的是 1 条完整交互：

1. 先拿当前观测 `obs_t`
2. actor 根据 `obs_t` 采样动作 `a_t`
3. critic 同时给出 `V(s_t)`
4. env 执行动作，返回 `reward_t`、`done_t`、`obs_{t+1}`
5. 把这条 transition 写进 rollout buffer

这一阶段通常会存下：

- `obs_t`
- `a_t`
- `reward_t`
- `done_t`
- `log_prob_old`
- `V_old(s_t)`
- `action mean / std`

这些信息后面都会参与 PPO 更新。

### 3.2 采满 64 步后，倒着算 GAE

RSL-RL 会先用最后一个状态的 critic 值做 bootstrap，然后从后往前算：

```text
delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
A_t = delta_t + gamma * lambda * A_{t+1}
R_t = A_t + V(s_t)
```

当前 PPO 配置是：

- `gamma = 0.99`
- `lam = 0.95`

这一步的直觉是：

- critic 先估一个“这个状态本来值多少”
- rollout 结束后，再根据真实 reward 序列修正这个判断
- 修正量就是 advantage

所以 actor 不是“直接跟着 reward 走”，而是：

- **看这次动作相对 critic 原本预期，到底更好还是更差**

### 3.3 然后用这 8192 条样本做 PPO 更新

当前配置：

- `num_learning_epochs = 5`
- `num_mini_batches = 4`

含义是：

- 同一批 `8192` 样本会被完整扫 `5` 遍
- 每一遍切成 `4` 个 mini-batch
- 所以总共做 `5 * 4 = 20` 次梯度更新

每个 mini-batch 大约是：

```text
8192 / 4 = 2048 条样本
```

更新时用的是 PPO clipped objective：

- `clip_param = 0.2`
- `learning_rate = 3e-4`
- `schedule = "adaptive"`
- `desired_kl = 0.008`
- `value_loss_coef = 1.0`
- `entropy_coef = 0.0005`
- `max_grad_norm = 1.0`

### 3.4 为什么更新完这批数据就丢掉

因为这批数据是用旧策略 `pi_old` 采出来的。

PPO 允许你在同一批 on-policy 数据上多学几遍，但不能无限复用。学完这 `5` 个 epoch 后，这批数据就会被清空，下一轮必须重新采样。这就是 **on-policy**。

---

## 4. `episode_length_s = 36.0` 到底是什么意思

它的含义不是：

- 训练脚本总共只跑 36 秒
- 一次 PPO 更新只有 36 秒

它的真正含义是：

- **单个 env 里，一局任务从 reset 开始，最多允许持续 36 秒仿真时间**

这里的“一局”，就是 1 个 **episode**。

### 4.1 什么叫“一局”

对单个 env 来说：

1. 执行 `reset`
2. 叉车和托盘被放到新的初始状态
3. agent 开始持续做 `观测 -> 动作 -> 环境推进`
4. 直到出现任意一种情况：
   - `success`
   - `tipped`
   - `out_of_bounds`
   - `time_out`
5. 这一次尝试结束，env 再次 `reset`

所以：

- **1 个 episode = 1 次完整尝试**
- **1 个 env 在训练过程中会反复经历很多个 episode**
- **128 个并行 env 会各自独立地滚动自己的 episode**

### 4.2 36 秒为什么会变成 1080 步

IsaacLab 基类里，episode 步数上限的换算公式是：

```text
max_episode_length = ceil(episode_length_s / (sim.dt * decimation))
```

本项目参数是：

- `sim.dt = 1/120 s`
- `decimation = 4`

所以：

```text
1 env step = 4 * (1/120) = 1/30 s
max_episode_length = 36 / (1/30) = 1080
```

也就是说：

- **1 个 env step 约等于 0.0333 秒**
- **1 个 episode 最长 1080 个 env step**

---

## 5. 四层关系图：`physics step`、`env step`、`rollout`、`episode`

```text
第 1 层：physics step
- 物理引擎内部最小步长
- dt = 1/120 s

4 个 physics step
↓ 由 decimation = 4 决定

第 2 层：env step
- 1 次 agent 决策 + 1 次 env.step()
- 持续时间 = 4 * (1/120) = 1/30 s

64 个 env step
↓ 由 num_steps_per_env = 64 决定

第 3 层：1 个 rollout chunk
- PPO 一次采样块
- 持续时间 = 64 * (1/30) ≈ 2.13 s

1080 个 env step
↓ 由 episode_length_s = 36.0 决定

第 4 层：1 个 episode 最大长度
- 最长 36 s
- 最长 1080 个 env step
- 约等于 1080 / 64 ≈ 16.9 个 rollout chunk
```

一句话总结：

- **rollout 是训练时临时切出来的一小段**
- **episode 是一整局任务**

---

## 6. 单个 env 从 `reset` 到 `timeout` 的时间轴

下面固定只看 **1 个 env**。

```text
reset
│
├─ episode counter = 0
├─ 当前这一局开始
└─ rollout buffer 里还没有这个 env 的样本

时间往右走
────────────────────────────────────────────────────────────────────────────>

physics step:
  p1 p2 p3 p4 | p5 p6 p7 p8 | p9 p10 p11 p12 | ... | 每 4 个 physics step
              ↓             ↓                ↓
env step:
  e1          e2            e3               ...                     e64
  (1/30s)     (2/30s)       (3/30s)                                  (64/30s≈2.13s)

episode counter:
  1           2             3                ...                     64

rollout buffer slot
(当前 iteration 内):
  slot[0]     slot[1]       slot[2]          ...                     slot[63]
```

这表示：

- `e1` 时，这个 env 贡献了本轮 rollout 的第 1 条 transition
- `e64` 时，这个 env 贡献了本轮 rollout 的第 64 条 transition
- 但这通常**不代表 episode 结束**

如果这个 env 在前 64 步里没有成功/失败/超时，它会继续跑同一局：

```text
iteration #1 结束后：
  这个 env 已经跑到 episode step 64
  但当前 episode 还没结束

iteration #2 开始：
  继续同一局

episode counter:
  65  66  67  ... 128

rollout buffer slot
(新一轮 iteration，buffer 编号重新从 0 开始):
  slot[0] slot[1] slot[2] ... slot[63]
```

所以：

- `episode counter` 记录的是“这局已经走了多少步”
- `rollout buffer slot` 记录的是“当前这轮 PPO 采样里排第几个”

二者不是同一个计数器。

---

## 7. 为什么 `64 steps ≈ 2.13s` 看起来这么短，但训练仍然成立

这个疑问非常合理。

如果我们误以为：

- “agent 必须在 64 步、也就是 2.13 秒里完成整局任务”

那对叉车的“接近 + 对齐 + 插入”来说，确实几乎不可能。

但实际上 PPO 并没有这么要求。

### 7.1 `64 steps` 只是一次更新前的采样片段

它真正表示的是：

- **每次更新前，先从每个 env 身上截取 64 步经验**

而不是：

- **必须在这 64 步内完成任务**

对于单个 env 而言，如果它一直没 done，那么一个完整 episode 最长会横跨：

```text
1080 / 64 ≈ 17 个 rollout chunk
```

### 7.2 rollout 在 64 步被截断，不代表 episode 真结束

如果这个 env 在第 64 步时还没 done：

- PPO 不会把它当成真正终点
- 会用最后一个状态的 `V(s)` 继续做 bootstrap
- 然后在下一轮 rollout 里，env 继续沿着同一个 episode 往后跑

所以 “64 步 rollout” 更像是：

- **训练器分批取数的窗口长度**

而不是：

- **任务本身的时间预算**

### 7.3 为什么这里用 64 步也还能学长任务

有两个主要原因：

1. **critic 可以在 rollout 边界做 bootstrap**  
   这让 PPO 不必等到一整局跑完，才能给前面的动作估值。

2. **本任务不是纯稀疏奖励**  
   当前 reward 里有很多 dense shaping：
   - 距离托盘更近
   - 更贴近参考轨迹
   - 偏航更对齐
   - 到达特定几何区域

这意味着即使只看 64 步，agent 也通常能收到“更接近目标/更偏离目标”的学习信号。

如果任务是那种“只有最终成功才给 1 分，中间全是 0”的超长时序问题，那么 64 步 rollout 往往会更难学；但本项目不是这种纯终点稀疏结构。

---

## 8. 一个经常被混淆的点：`time_out` 和真正失败终止不是一回事

当前环境里：

- `terminated`：真正的任务结束，比如成功、翻车、出界
- `time_out`：只是这局用完了时间预算

这两类在 RL 里处理方式不同。

本项目的超时会做 value bootstrap，因此：

- **超时更像“截断”**
- **不是“不可恢复失败”那种真正 terminal**

这也是为什么文档里常会把：

- `terminated`
- `time_out`

分开写。

如果你想继续看终止条件细节，可以再读：

- `docs/learning_guiding/关于RL和ISAAC/episode_termination_conditions.md`

---

## 9. 当前 PPO 关键超参数一览

### 9.1 采样与更新

| 参数 | 当前值 | 含义 |
|------|--------|------|
| `num_envs` | `128` | 并行环境数 |
| `num_steps_per_env` | `64` | 每个 env 每次 rollout 采样步数 |
| `max_iterations` | `2000` | PPO 外循环最大次数 |
| `num_learning_epochs` | `5` | 每轮 rollout 的学习 epoch 数 |
| `num_mini_batches` | `4` | 每个 epoch 切成的 mini-batch 数 |

### 9.2 PPO 本体

| 参数 | 当前值 | 含义 |
|------|--------|------|
| `learning_rate` | `3e-4` | Adam 学习率 |
| `schedule` | `adaptive` | 按 KL 动态调学习率 |
| `desired_kl` | `0.008` | 目标 KL，上下超界时自动调 lr |
| `clip_param` | `0.2` | PPO clip 范围 |
| `gamma` | `0.99` | 回报折扣因子 |
| `lam` | `0.95` | GAE 参数 |
| `value_loss_coef` | `1.0` | value loss 权重 |
| `entropy_coef` | `0.0005` | 熵正则权重 |
| `max_grad_norm` | `1.0` | 梯度裁剪阈值 |

### 9.3 一个特别容易混的点

环境配置 `env_cfg.py` 里也有一个 `gamma`，但那个是 **reward shaping 用的环境参数**，不是 PPO 算法的 discount。PPO 真正用来算 GAE / return 的 `gamma` 来自 `rsl_rl_ppo_cfg.py`，当前值是 `0.99`。

---

## 10. 这几个概念最终应该怎么记

如果只记一句话，记这个：

> **`physics step` 是仿真底层时钟，`env step` 是 agent 决策频率，`rollout 64 steps` 是 PPO 一次更新前临时取的一小段数据，`episode 36s/1080 steps` 才是一整局任务真正允许的最长时间。**

再压缩成最短版：

- **一局 = 1 个 episode = 从 reset 到 done/timeout**
- **64 步不是一局，只是一小段 rollout**
- **36 秒才是一局的最长仿真时间预算**
- **单个 env 的一整局，最长会跨大约 17 个 rollout chunk**

---

## 11. 相关阅读

- 训练命令与日志目录：`docs/learning_guiding/04_training_and_artifacts.md`
- 终止条件与 timeout：`docs/learning_guiding/关于RL和ISAAC/episode_termination_conditions.md`
- RL 基本组成：`docs/learning_guiding/关于RL和ISAAC/08_rl_components_and_overviews.md`
