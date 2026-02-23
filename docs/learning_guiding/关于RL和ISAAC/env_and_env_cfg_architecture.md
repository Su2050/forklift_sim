# env.py 与 env_cfg.py 架构讲解

本文档讲解本项目 `env.py` 和 `env_cfg.py` 的类结构、核心方法、外部依赖类，以及完整的调用链路。

源码位置：
- `forklift_pallet_insert_lift_project/isaaclab_patch/source/isaaclab_tasks/isaaclab_tasks/direct/forklift_pallet_insert_lift/env.py`
- `forklift_pallet_insert_lift_project/isaaclab_patch/source/isaaclab_tasks/isaaclab_tasks/direct/forklift_pallet_insert_lift/env_cfg.py`

---

## 目录

1. [全局类关系图](#1-全局类关系图)
2. [env.py 的 7 个核心方法](#2-envpy-的-7-个核心方法)
3. [env_cfg.py 的结构](#3-env_cfgpy-的结构)
4. [外部核心类](#4-外部核心类)
5. [完整调用链](#5-完整调用链)
6. [Python 语法补充](#6-python-语法补充)

---

## 1. 全局类关系图

```
Isaac Lab 框架层                         你的项目层
══════════════                          ══════════

DirectRLEnvCfg (配置基类)               ForkliftPalletInsertLiftEnvCfg (env_cfg.py)
  ├── sim: SimulationCfg                  ├── 继承 DirectRLEnvCfg 的所有字段
  ├── scene: InteractiveSceneCfg          ├── robot_cfg: ArticulationCfg    ← 叉车配置
  ├── episode_length_s                    ├── pallet_cfg: RigidObjectCfg    ← 托盘配置
  ├── observation_space                   ├── ground_cfg: GroundPlaneCfg
  └── action_space                        ├── pallet_depth_m, insert_fraction...
                                          └── 奖励/KPI 参数（几十个）

DirectRLEnv (环境基类)                  ForkliftPalletInsertLiftEnv (env.py)
  ├── __init__(cfg)                       ├── 继承 DirectRLEnv
  │   └── self.cfg = cfg                  ├── __init__: 查找关节、初始化缓存
  ├── _setup_scene()   ← 需重写          ├── _setup_scene: 创建资产 + 物理补丁
  ├── _pre_physics_step()  ← 需重写      ├── _pre_physics_step: 缓存动作
  ├── _apply_action()  ← 需重写          ├── _apply_action: 动作→关节控制
  ├── _get_observations() ← 需重写       ├── _get_observations: 15维观测
  ├── _get_rewards()   ← 需重写          ├── _get_rewards: 三阶段势函数奖励
  ├── _get_dones()     ← 需重写          ├── _get_dones: 终止条件
  └── _reset_idx()     ← 需重写          └── _reset_idx: 重置环境
```

**核心思想**：Isaac Lab 定义了"接口"（哪些方法必须实现），你的项目填写"实现"（具体怎么做）。

---

## 2. env.py 的 7 个核心方法

框架会自动按固定顺序调用这些方法，你只需要实现它们。

### 2.1 `_setup_scene()` — 启动时调用一次

**做什么**：创建所有仿真物体并克隆环境。

```python
def _setup_scene(self):
    self.robot = Articulation(self.cfg.robot_cfg)     # 用配置创建叉车
    self.pallet = RigidObject(self.cfg.pallet_cfg)    # 用配置创建托盘
    self._setup_pallet_physics()                       # 补丁：创建缺失的 RigidBodyAPI
    self._fix_lift_joint_drive()                       # 补丁：修复 lift 关节驱动
    spawn_ground_plane(...)                             # 创建地面
    self.scene.clone_environments(...)                  # 克隆 env_0 → env_1~N
```

**执行顺序很重要**：
1. 先创建资产（spawn，此时 `modify_*` 被调用）
2. 再运行补丁（`define_*` 补上缺失的 API）
3. 最后克隆（env_0 的所有属性被复制到所有环境）

### 2.2 `_reset_idx(env_ids)` — 每个 episode 开始时调用

**做什么**：把指定环境重置到初始状态。

```python
def _reset_idx(self, env_ids):
    # 1) 清零所有缓存（计数器、里程碑、势函数等）
    self._hold_counter[env_ids] = 0
    self._milestone_flags[env_ids] = False
    ...

    # 2) 托盘回到固定位置 (0, 0, 0.15)
    self._write_root_pose(self.pallet, pallet_pos, pallet_quat, env_ids)

    # 3) 叉车随机化初始位姿
    #    x ∈ [-4.0, -2.5], y ∈ [-0.6, 0.6], yaw ∈ [-0.25, 0.25]
    self._write_root_pose(self.robot, pos, quat, env_ids)

    # 4) 关节归零（升降收回、轮子停转、转向回正）
    self._write_joint_state(self.robot, joint_pos, joint_vel, env_ids)
```

**注意**：`env_ids` 是一个 tensor，只包含需要重置的那些环境的索引。1024 个环境中可能只有几十个需要重置，其余继续运行。

### 2.3 `_pre_physics_step(actions)` — 每步开头

**做什么**：缓存策略网络输出的动作（限幅到 [-1, 1]）。

```python
def _pre_physics_step(self, actions):
    self.actions = torch.clamp(actions, -1.0, 1.0)
```

### 2.4 `_apply_action()` — 每个物理子步

每步被调用 `decimation=4` 次（1 个 env step = 4 个物理步）。

**做什么**：把归一化动作解码为物理控制量，写入仿真。

```python
def _apply_action(self):
    # 1) 解码动作
    drive = actions[:, 0] * wheel_speed_rad_s    # 驱动速度 (rad/s)
    steer = actions[:, 1] * steer_angle_rad      # 转向角度 (rad)
    lift_v = actions[:, 2] * lift_speed_m_s      # 举升速度 (m/s)

    # 2) 安全制动：插入够深后禁止继续前进
    drive = torch.where(inserted, 0, drive)
    steer = torch.where(inserted, 0, steer)

    # 3) 写入关节控制
    self.robot.set_joint_velocity_target(drive, joint_ids=wheel_ids)   # 轮子速度
    self.robot.set_joint_position_target(steer, joint_ids=rotator_ids) # 转向角度
    self.robot.set_joint_position_target(lift_target, joint_ids=[lift_id]) # 升降位置

    # 4) 提交到 PhysX
    self.robot.write_data_to_sim()
```

### 2.5 `_get_observations()` — 每步物理仿真后

**做什么**：从仿真状态构造 15 维观测向量给策略网络。

```python
def _get_observations(self):
    return {"policy": torch.cat([
        d_xy_r,          # (2) 到托盘的相对距离（叉车坐标系）
        cos_dyaw,        # (1) 偏航差的 cos
        sin_dyaw,        # (1) 偏航差的 sin
        v_xy_r,          # (2) 叉车速度（叉车坐标系）
        yaw_rate,        # (1) 偏航角速度
        lift_pos,        # (1) 升降关节位置
        lift_vel,        # (1) 升降关节速度
        insert_norm,     # (1) 归一化插入深度
        self.actions,    # (3) 上一步动作
        y_err_obs,       # (1) 横向误差
        yaw_err_obs,     # (1) 偏航误差
    ], dim=-1)}          # 总计 15 维
```

**数据来源**：全部来自 `self.robot.data.*` 和 `self.pallet.data.*`，即 PhysX 仿真引擎的实时状态。

### 2.6 `_get_rewards()` — 每步

**做什么**：计算奖励（三阶段势函数 + 里程碑 + 惩罚），约 600 行代码。是整个项目中最复杂的方法。

奖励组成概要：
- **Stage 1**：远场接近 + 粗对齐（距离带 [d1_min, d1_max]）
- **Stage 2**：近场微调接近（从距离带推到口前）
- **Stage 3**：插入深化 + 举升
- **里程碑奖励**：首次达到特定阈值时的一次性奖励
- **惩罚**：动作 L2、时间惩罚、死区卡住、空举等
- **成功奖励**：100 分 + 时间奖励

### 2.7 `_get_dones()` — 每步

**做什么**：判断是否终止。

```python
def _get_dones(self):
    terminated = (翻车 | 任务成功 | 飞远 | 卡死 | 死区卡住)
    time_out = (步数 >= 最大 episode 长度)
    return terminated, time_out
```

终止条件：
- `tipped`：roll 或 pitch 超过 25°（翻车）
- `success`：hold_counter 达到要求的步数
- `_early_stop_fly`：叉车飞太远
- `_early_stop_stall`：势函数长时间无进展
- `_early_stop_dz_stuck`：在死区卡住

---

## 3. env_cfg.py 的结构

```python
@configclass
class ForkliftPalletInsertLiftEnvCfg(DirectRLEnvCfg):  # 继承框架配置基类

    # ── 环境基础 ──
    decimation = 4                    # 每 4 个物理步调用一次策略
    episode_length_s = 36.0           # 一个 episode 最长 36 秒
    action_space = 3                  # 动作维度：[驱动, 转向, 举升]
    observation_space = 15            # 观测维度

    # ── 仿真 ──
    sim = SimulationCfg(dt=1/120)     # 物理时间步 1/120 秒

    # ── 场景 ──
    scene = InteractiveSceneCfg(num_envs=128, env_spacing=6.0)

    # ── 资产配置（嵌套对象树） ──
    robot_cfg = ArticulationCfg(...)  # 叉车（第 283 行）
    pallet_cfg = RigidObjectCfg(...)  # 托盘（第 368 行）
    ground_cfg = GroundPlaneCfg()     # 地面

    # ── 托盘几何参数 ──
    pallet_depth_m = 2.16             # 托盘深度 (1.2m × 1.8)

    # ── KPI ──
    insert_fraction = 0.40            # 插入成功比例
    lift_delta_m = 0.3                # 举升高度
    hold_time_s = 0.33                # 保持时间
    max_lateral_err_m = 0.15          # 最大横向误差
    max_yaw_err_deg = 5.0             # 最大偏航误差

    # ── 奖励参数 ──（约 150 行）
    gamma = 1.0                       # 势函数折扣因子
    k_phi1 = 6.0                      # Stage1 势函数强度
    k_phi2 = 10.0                     # Stage2 势函数强度
    k_ins = 18.0                      # 插入势函数强度
    k_lift = 20.0                     # 举升势函数强度
    rew_success = 100.0               # 成功奖励
    ...                               # 几十个奖励/检测器参数
```

它本质上是一个**巨大的配置数据结构**，所有可调参数都声明在这里。训练时修改参数只需改这个文件，不需要动 `env.py` 的逻辑。

---

## 4. 外部核心类

### 4.1 `DirectRLEnv`（Isaac Lab 框架）

| 角色 | 说明 |
|---|---|
| 是什么 | RL 环境基类，定义了完整的训练循环接口 |
| 你需要做什么 | 继承它，重写 7 个方法 |
| 它帮你做什么 | 自动管理仿真步进、环境重置调度、episode 计数 |

框架自动执行的训练循环：

```python
while not done:
    obs = env._get_observations()       # 获取观测
    actions = policy(obs)                # 策略网络推理
    env._pre_physics_step(actions)       # 缓存动作
    for _ in range(decimation):          # 4 次物理子步
        env._apply_action()              # 写入控制
        env.sim.step()                   # PhysX 前进一步
    rewards = env._get_rewards()         # 计算奖励
    dones = env._get_dones()             # 判断终止
    if any(dones):
        env._reset_idx(done_env_ids)     # 重置结束的环境
```

### 4.2 `Articulation`（叉车）

铰接体封装类，管理由多个刚体 + 关节组成的机器人。

| 方法/属性 | 作用 | env.py 中的使用 |
|---|---|---|
| `find_joints(names)` | 按名字找关节索引 | `__init__` 中查找轮子/转向/升降关节 |
| `set_joint_velocity_target(vel, ids)` | 设速度目标 | `_apply_action` 控制轮子 |
| `set_joint_position_target(pos, ids)` | 设位置目标 | `_apply_action` 控制转向和升降 |
| `write_data_to_sim()` | 将控制写入 PhysX | `_apply_action` 末尾 |
| `.data.root_pos_w` | 读取根节点世界位置 (N, 3) | `_get_observations` |
| `.data.root_quat_w` | 读取根节点姿态 (N, 4) | `_get_observations` / `_get_dones` |
| `.data.root_lin_vel_w` | 读取线速度 (N, 3) | `_get_observations` |
| `.data.root_ang_vel_w` | 读取角速度 (N, 3) | `_get_observations` |
| `.root_physx_view.get_dof_positions()` | 直读 PhysX 关节位置 | `_get_observations` |
| `.root_physx_view.get_dof_velocities()` | 直读 PhysX 关节速度 | `_get_observations` |
| `.joint_names` | 所有关节名称列表 | 调试 |
| `.data.default_joint_pos` | 默认关节位置 | `_reset_idx` |

### 4.3 `RigidObject`（托盘）

单刚体封装类。比 Articulation 简单得多——没有关节，只需要读取位置和姿态。

| 方法/属性 | 作用 | env.py 中的使用 |
|---|---|---|
| `.data.root_pos_w` | 读取托盘世界位置 (N, 3) | 计算距离、插入深度 |
| `.data.root_quat_w` | 读取托盘姿态 (N, 4) | 计算对齐误差 |

### 4.4 其他辅助

| 类/函数 | 来源 | 作用 |
|---|---|---|
| `SimulationCfg` | `isaaclab.sim` | 仿真参数（dt、渲染间隔） |
| `InteractiveSceneCfg` | `isaaclab.scene` | 场景参数（环境数量、间距） |
| `ArticulationCfg` | `isaaclab.assets` | 叉车资产配置 |
| `RigidObjectCfg` | `isaaclab.assets` | 托盘资产配置 |
| `ImplicitActuatorCfg` | `isaaclab.actuators` | 关节执行器配置 |
| `spawn_ground_plane` | `isaaclab.sim.spawners` | 创建地面 |
| `sample_uniform` | `isaaclab.utils.math` | 均匀随机采样（初始化位姿用） |

---

## 5. 完整调用链

### 5.1 从训练脚本到环境实例化

```
训练脚本: train.py
    │
    ▼ gym.make("Isaac-Forklift-PalletInsertLift-Direct-v0")
    │
    ├─► __init__.py 的 gym.register 查找到：
    │     环境类 = ForkliftPalletInsertLiftEnv (env.py)
    │     配置类 = ForkliftPalletInsertLiftEnvCfg (env_cfg.py)
    │
    ▼ 框架实例化
    cfg = ForkliftPalletInsertLiftEnvCfg()  ← env_cfg.py 里的配置
    env = ForkliftPalletInsertLiftEnv(cfg)  ← env.py 里的环境类
    │
    ├─► env.__init__(cfg)
    │     ├─► super().__init__(cfg)         ← DirectRLEnv 保存 self.cfg
    │     │   └─► self._setup_scene()       ← 创建叉车/托盘/地面 + 克隆
    │     ├─► find_joints(...)              ← 查找关节索引
    │     └─► 初始化所有 tensor 缓存        ← 计数器、势函数、里程碑等
    │
    ▼ 训练循环开始
```

### 5.2 每个训练步的执行流程

```
┌──────────────────────────────────────────────────┐
│  obs = env._get_observations()                    │ → 15 维向量
│  actions = PPO_policy(obs)                         │ → 3 维动作
│  env._pre_physics_step(actions)                    │ → 缓存动作
│  for i in range(4):  # decimation=4               │
│      env._apply_action()                           │ → 写入关节控制
│      env.sim.step()                                │ → PhysX 物理前进
│  rewards = env._get_rewards()                      │ → 奖励计算
│  terminated, time_out = env._get_dones()           │ → 终止判断
│  if any terminated/time_out:                       │
│      env._reset_idx(done_ids)                      │ → 重置那些环境
└──────────────────────────────────────────────────┘
       │ 重复以上循环 max_iterations 次
       ▼
    训练结束，保存模型
```

### 5.3 `self.cfg.pallet_cfg` 的数据流

```python
# env_cfg.py 中声明（配置数据）：
pallet_cfg: RigidObjectCfg = RigidObjectCfg(
    prim_path="/World/envs/env_.*/Pallet",
    spawn=sim_utils.UsdFileCfg(
        usd_path="...pallet_com_shifted.usd",
        rigid_props=..., mass_props=..., collision_props=...,
    ),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(0,0,0.15)),
)

# env.py 中使用（运行逻辑）：
self.pallet = RigidObject(self.cfg.pallet_cfg)
#             ^^^^^^^^^^^  ^^^^^^^^^^^^^^^^^
#             框架类        传入上面声明的配置
#
# RigidObject 内部自动执行：
#   1. 读 pallet_cfg.spawn.usd_path    → 加载 USD 文件
#   2. 读 pallet_cfg.spawn.scale       → 缩放 1.8 倍
#   3. 读 pallet_cfg.spawn.rigid_props → 调用 modify_rigid_body（可能失败）
#   4. 读 pallet_cfg.spawn.mass_props  → 调用 modify_mass（成功）
#   5. 读 pallet_cfg.spawn.collision_props → 调用 modify_collision（成功）
#   6. 读 pallet_cfg.init_state        → 设初始位置 (0, 0, 0.15)
```

---

## 6. Python 语法补充

### 6.1 类继承 vs 类型注解

```python
class ForkliftPalletInsertLiftEnv(DirectRLEnv):    # 括号 = 继承
    cfg: ForkliftPalletInsertLiftEnvCfg             # 冒号 = 类型注解（提示 IDE）
```

- **括号 `(DirectRLEnv)`**：继承，表示本类是 DirectRLEnv 的子类
- **冒号 `cfg: EnvCfg`**：类型注解，只是告诉 IDE "self.cfg 是 EnvCfg 类型"，运行时不执行

### 6.2 `@configclass` 装饰器

```python
@configclass
class ForkliftPalletInsertLiftEnvCfg(DirectRLEnvCfg):
    decimation = 4
    episode_length_s = 36.0
    ...
```

`@configclass` 类似标准库的 `@dataclass`，自动生成 `__init__` 方法。所以虽然类定义里没写 `__init__`，但可以直接用 `EnvCfg(decimation=8)` 来创建对象并覆盖默认值。

### 6.3 嵌套对象创建

```python
pallet_cfg = RigidObjectCfg(
    spawn=sim_utils.UsdFileCfg(
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
        ),
    ),
)
```

等价于拆开写：

```python
rigid_props = sim_utils.RigidBodyPropertiesCfg(rigid_body_enabled=True)
spawn = sim_utils.UsdFileCfg(rigid_props=rigid_props)
pallet_cfg = RigidObjectCfg(spawn=spawn)
```

嵌套写法只是把中间变量省略了，减少代码行数。

### 6.4 `self` 关键字

```python
class MyEnv:
    def __init__(self, cfg):
        self.cfg = cfg          # 把 cfg 存到实例上
        self.counter = 0        # 创建实例属性

    def step(self):
        self.counter += 1       # 通过 self 访问实例属性
        print(self.cfg.dt)      # 通过 self 访问之前存的配置
```

`self` 是 Python 类方法的第一个参数，指向类的实例本身。所有实例属性都通过 `self.xxx` 访问。

### 6.5 `super().__init__(cfg)`

```python
class ForkliftPalletInsertLiftEnv(DirectRLEnv):
    def __init__(self, cfg, ...):
        super().__init__(cfg, ...)   # 调用父类 DirectRLEnv 的 __init__
```

`super()` 返回父类对象。`super().__init__(cfg)` 意思是"先执行父类的初始化"，父类会做 `self.cfg = cfg`、调用 `_setup_scene()` 等基础工作。
