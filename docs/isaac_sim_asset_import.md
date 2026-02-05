# Isaac Sim 资产导入指南

本文档说明如何将同事在 Isaac Sim 中设置好的 USD 资产和配置导入到本项目，以及如何确保 IsaacLab 代码与新资产正确匹配。

---

## 目录

1. [USD 资产文件导入](#1-usd-资产文件叉车托盘模型)
2. [完整场景导入](#2-完整场景包含所有配置)
3. [配置参数更新](#3-配置参数位置缩放物理属性)
4. [推荐工作流](#4-推荐工作流)
5. [常见问题](#5-常见问题)
6. [**IsaacLab 代码同步指南**](#6-isaaclab-代码同步指南)（重要）

---

## 1. USD 资产文件（叉车/托盘模型）

如果同事调整了叉车或托盘的 USD 资产（如尺寸、物理属性、碰撞体等）：

### 方式 A：本地文件

1. 将同事的 `.usd`/`.usda`/`.usdc` 文件复制到项目目录，例如：
   ```
   /home/uniubi/projects/forklift_sim/assets/forklift.usd
   /home/uniubi/projects/forklift_sim/assets/pallet.usd
   ```

2. 修改 `env_cfg.py` 中的路径引用：
   ```python
   # 绝对路径
   usd_path="/home/uniubi/projects/forklift_sim/assets/forklift.usd"
   
   # 或相对路径（相对于 IsaacLab 根目录）
   usd_path="../assets/forklift.usd"
   ```

### 方式 B：Nucleus 服务器

1. 同事将资产上传到 Nucleus 服务器
2. 使用 Nucleus 路径引用：
   ```python
   usd_path="omniverse://localhost/Projects/forklift/pallet.usd"
   # 或使用 Isaac Sim 内置资产目录
   usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Pallet/pallet.usd"
   ```

## 2. 完整场景（包含所有配置）

如果同事在 Isaac Sim GUI 中设置好了整个场景（包括物理属性、位置、材质等）：

1. **同事导出场景**：
   - Isaac Sim 中：`File -> Save As`
   - 保存为 `.usd` 文件

2. **导入到项目**：
   - 将场景文件放到项目目录
   - 在代码中加载整个场景，而不是单独加载各资产

3. **注意事项**：
   - 确保场景中的 prim 路径与代码中的引用一致
   - 检查是否有绝对路径需要修改为相对路径

## 3. 配置参数（位置、缩放、物理属性）

如果同事只是调整了参数（如托盘高度、缩放比例），需要手动更新配置文件。

### 主要配置文件

```
IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/forklift_pallet_insert_lift/env_cfg.py
```

### 关键参数

| 参数 | 说明 | 示例 |
|------|------|------|
| `scale` | 缩放比例 | `scale=(1.8, 1.8, 1.8)` |
| `init_state.pos` | 初始位置 (x, y, z) | `pos=(0.0, 0.0, 0.15)` |
| `init_state.rot` | 初始旋转 (w, x, y, z) | `rot=(1.0, 0.0, 0.0, 0.0)` |
| `rigid_props.kinematic_enabled` | 是否为运动学物体 | `False` = 动态刚体 |
| `rigid_props.disable_gravity` | 是否禁用重力 | `False` = 受重力影响 |
| `mass_props.mass` | 质量 (kg) | `mass=45.0` |

### 示例：更新托盘配置

```python
pallet_cfg: RigidObjectCfg = RigidObjectCfg(
    prim_path="/World/envs/env_.*/Pallet",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Pallet/pallet.usd",
        scale=(1.8, 1.8, 1.8),  # 放大 1.8 倍
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            kinematic_enabled=False,  # 动态刚体
            disable_gravity=False,    # 受重力影响
        ),
        mass_props=sim_utils.MassPropertiesCfg(mass=45.0),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.15),  # 调整高度使 pocket 与货叉对齐
        rot=(1.0, 0.0, 0.0, 0.0),
    ),
)
```

## 4. 推荐工作流

### 场景调试流程

1. **同事在 Isaac Sim GUI 中**：
   - 手动调整托盘高度，使 pocket 与货叉对齐
   - 测试叉车能否物理插入托盘
   - 记录最终的位置、缩放参数

2. **参数传递**：
   - 如果只是参数调整：让同事告诉你具体数值，更新 `env_cfg.py`
   - 如果修改了 USD 资产本身：将 USD 文件复制到项目并更新路径引用

### 验证导入结果

导入后运行验证脚本：

```bash
cd /home/uniubi/projects/forklift_sim/IsaacLab
./isaaclab.sh -p ../scripts/verify_forklift_insert_lift.py --manual
```

## 5. 常见问题

### Q: 导入后资产位置不对？
A: 检查 `init_state.pos` 配置，确保与 USD 资产的原点位置匹配。

### Q: 物理属性没有生效？
A: 确保 `rigid_body_enabled=True`，且 `kinematic_enabled=False`（如果需要动态物理）。

### Q: 缩放后碰撞体不正确？
A: Isaac Sim 中的缩放会同时影响视觉和碰撞体。如果需要单独调整，需要修改 USD 资产本身。

### Q: 如何检查 USD 资产结构？
A: 运行几何兼容性检查脚本：
```bash
cd /home/uniubi/projects/forklift_sim/IsaacLab
./isaaclab.sh -p ../scripts/verify_geometry_compatibility.py
```

---

## 6. IsaacLab 代码同步指南（重要）

当导入新的 USD 资产时，必须确保 IsaacLab 代码与资产正确匹配。以下是 USD 资产与代码之间的关联关系。

### 6.1 USD 资产与代码的关联架构

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           USD 资产文件                                   │
│  forklift_c.usd / pallet.usd                                            │
│  ├── Prim 结构（/Robot, /Pallet）                                       │
│  ├── 关节定义（joint names）                                            │
│  ├── 碰撞体（collision shapes）                                         │
│  └── 原点位置                                                            │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                     env_cfg.py（环境配置）                               │
│  ├── usd_path: USD 文件路径                                             │
│  ├── prim_path: 场景中的 Prim 路径模式                                  │
│  ├── init_state: 初始位置/姿态                                          │
│  ├── rigid_props: 物理属性（覆盖 USD 原始设置）                         │
│  ├── actuators: 执行器配置（关节名称必须与 USD 匹配）                   │
│  └── 几何参数: pallet_depth_m 等                                        │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        env.py（环境逻辑）                                │
│  ├── find_joints(): 按名称查找关节 ID                                   │
│  ├── _apply_action(): 应用控制到关节                                    │
│  ├── _compute_fork_tip(): 计算货叉尖端位置                              │
│  └── _force_pallet_rigid_body(): 为纯视觉资产添加物理 API               │
└─────────────────────────────────────────────────────────────────────────┘
```

### 6.2 关键匹配点

#### 6.2.1 关节名称匹配（最重要）

**USD 资产中的关节名称必须与代码中引用的名称完全一致！**

当前叉车 (`forklift_c.usd`) 的关节名称：

| 关节 | USD 中的名称 | 代码中的引用位置 |
|------|-------------|-----------------|
| 左前轮 | `left_front_wheel_joint` | `env_cfg.py` actuators, `env.py` find_joints |
| 右前轮 | `right_front_wheel_joint` | `env_cfg.py` actuators, `env.py` find_joints |
| 左后轮 | `left_back_wheel_joint` | `env_cfg.py` actuators, `env.py` find_joints |
| 右后轮 | `right_back_wheel_joint` | `env_cfg.py` actuators, `env.py` find_joints |
| 左转向 | `left_rotator_joint` | `env_cfg.py` actuators, `env.py` find_joints |
| 右转向 | `right_rotator_joint` | `env_cfg.py` actuators, `env.py` find_joints |
| 升降 | `lift_joint` | `env_cfg.py` actuators, `env.py` find_joints |

**如果更换叉车 USD 资产，必须检查并更新以下代码：**

**env_cfg.py 中的 actuators 配置：**
```python
actuators={
    "front_wheels": ImplicitActuatorCfg(
        joint_names_expr=["left_front_wheel_joint", "right_front_wheel_joint"],  # ← 必须与 USD 匹配
        ...
    ),
    "back_wheels": ImplicitActuatorCfg(
        joint_names_expr=["left_back_wheel_joint", "right_back_wheel_joint"],  # ← 必须与 USD 匹配
        ...
    ),
    "rotators": ImplicitActuatorCfg(
        joint_names_expr=["left_rotator_joint", "right_rotator_joint"],  # ← 必须与 USD 匹配
        ...
    ),
    "lift": ImplicitActuatorCfg(
        joint_names_expr=["lift_joint"],  # ← 必须与 USD 匹配
        ...
    ),
}
```

**env.py 中的关节查找：**
```python
self._front_wheel_ids, _ = self.robot.find_joints(
    ["left_front_wheel_joint", "right_front_wheel_joint"],  # ← 必须与 USD 匹配
    preserve_order=True
)
self._lift_id, _ = self.robot.find_joints(["lift_joint"], preserve_order=True)  # ← 必须与 USD 匹配
```

#### 6.2.2 Prim 路径匹配

**env_cfg.py 中的 prim_path 必须与场景中的实际路径匹配：**

```python
robot_cfg: ArticulationCfg = ArticulationCfg(
    prim_path="/World/envs/env_.*/Robot",  # ← 通配符模式，匹配所有环境
    spawn=sim_utils.UsdFileCfg(
        usd_path="...",  # USD 会被加载到这个 prim_path
    ),
)

pallet_cfg: RigidObjectCfg = RigidObjectCfg(
    prim_path="/World/envs/env_.*/Pallet",  # ← 通配符模式
    spawn=sim_utils.UsdFileCfg(...),
)
```

**注意**：`env_.*/Robot` 中的 `Robot` 和 `Pallet` 是在场景中创建的 prim 名称，不是 USD 文件内部的路径。

#### 6.2.3 几何参数匹配

如果更换托盘资产或调整缩放比例，必须更新 `env_cfg.py` 中的几何参数：

```python
# pallet geometry assumptions
pallet_depth_m: float = 2.16  # ← 原始深度 1.2m × 缩放 1.8 = 2.16m

# KPI thresholds
insert_fraction: float = 2.0 / 3.0  # 插入深度阈值
lift_delta_m: float = 0.12          # 举升高度阈值
max_lateral_err_m: float = 0.03     # 横向对齐阈值
max_yaw_err_deg: float = 3.0        # 偏航对齐阈值
```

### 6.3 更换 USD 资产的完整检查清单

#### 更换叉车资产

1. **获取新 USD 的关节名称**
   - 在 Isaac Sim 中打开 USD 文件
   - 查看 Stage 面板中的关节列表
   - 记录所有关节的准确名称

2. **更新 env_cfg.py**
   - [ ] `robot_cfg.spawn.usd_path` - USD 文件路径
   - [ ] `actuators` 中的 `joint_names_expr` - 所有关节名称
   - [ ] `robot_cfg.init_state.joint_pos` - 关节初始位置字典的键名

3. **更新 env.py**
   - [ ] `find_joints()` 中的关节名称列表
   - [ ] 如果关节数量/类型变化，检查 `_apply_action()` 逻辑

4. **验证**
   ```bash
   ./isaaclab.sh -p ../scripts/verify_forklift_insert_lift.py --manual
   ```

#### 更换托盘资产

1. **获取新托盘的尺寸**
   - 原点位置
   - 长宽高
   - pocket 位置和大小

2. **更新 env_cfg.py**
   - [ ] `pallet_cfg.spawn.usd_path` - USD 文件路径
   - [ ] `pallet_cfg.spawn.scale` - 缩放比例
   - [ ] `pallet_cfg.init_state.pos` - 初始位置（Z 坐标影响 pocket 高度）
   - [ ] `pallet_depth_m` - 托盘深度（用于插入深度计算）

3. **更新 env.py**
   - [ ] `_pallet_front_x` 计算逻辑（如果托盘原点不在中心）

4. **验证**
   ```bash
   ./isaaclab.sh -p ../scripts/verify_geometry_compatibility.py
   ```

### 6.4 调试技巧

#### 查看 USD 关节名称

方法 1：在 Isaac Sim GUI 中打开 USD，查看 Stage 面板

方法 2：使用 Python 脚本：
```python
from pxr import Usd, UsdPhysics

stage = Usd.Stage.Open("path/to/forklift.usd")
for prim in stage.Traverse():
    if prim.IsA(UsdPhysics.Joint):
        print(f"Joint: {prim.GetPath()}")
```

#### 验证关节匹配

如果启动时报错 `Joint not found`，检查：
1. USD 中的关节名称拼写
2. `env_cfg.py` 和 `env.py` 中的名称是否完全一致（区分大小写）

#### 查看运行时关节信息

```python
# 在环境初始化后
print("关节名称:", self.robot.joint_names)
print("关节 ID:", self._front_wheel_ids, self._lift_id)
```

### 6.5 物理 API 自动修补

Isaac Sim 中的一些 USD 资产（如 `pallet.usd`）是纯视觉资产，没有物理 API。IsaacLab 需要物理 API 才能正常工作。

`env.py` 中的 `_force_pallet_rigid_body()` 函数会自动为托盘添加必要的物理 API：

```python
# 在 _setup_scene() 中自动调用
_force_pallet_rigid_body(
    self.sim.stage,
    rigid_body_enabled=True,
    kinematic_enabled=False,  # False = 动态刚体，可被推动
    disable_gravity=False,    # False = 受重力影响
)
```

如果导入的新 USD 资产已经有物理 API，这个函数不会覆盖原有设置。

### 6.6 配置文件与代码文件对照表

| 文件 | 作用 | 需要与 USD 匹配的内容 |
|------|------|----------------------|
| `env_cfg.py` | 环境配置 | USD 路径、关节名称、几何参数 |
| `env.py` | 环境逻辑 | 关节名称、物理 API 修补 |
| `verify_forklift_insert_lift.py` | 验证脚本 | 无（使用 env 提供的接口） |
| `verify_geometry_compatibility.py` | 几何检查 | USD 路径 |

---

## 7. 强化学习 State/Observation/Action 与 USD 资产的关系

本节详细解释强化学习中的 **Observation（观测）** 和 **Action（动作）** 是如何从 USD 资产中获取/设置的。

### 7.1 整体架构

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         USD 资产 (forklift_c.usd)                        │
│  ├── Articulation Root (机器人根节点)                                   │
│  │   ├── Links/Bodies (刚体链接：body, wheels, forks...)               │
│  │   └── Joints (关节：wheel_joint, lift_joint, rotator_joint...)      │
│  └── 物理属性 (质量、惯量、碰撞体、约束...)                             │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                      PhysX 仿真引擎（每帧更新）
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    IsaacLab 封装层 (Articulation 类)                     │
│                                                                         │
│  读取状态（Observation 来源）:                                          │
│  ├── robot.data.root_pos_w      → 根部世界位置 (N, 3)                  │
│  ├── robot.data.root_quat_w     → 根部世界姿态 (N, 4)                  │
│  ├── robot.data.root_lin_vel_w  → 根部线速度 (N, 3)                    │
│  ├── robot.data.root_ang_vel_w  → 根部角速度 (N, 3)                    │
│  ├── robot.data.joint_pos       → 所有关节位置 (N, J)                  │
│  ├── robot.data.joint_vel       → 所有关节速度 (N, J)                  │
│  └── robot.data.body_pos_w      → 所有 body 世界位置 (N, B, 3)         │
│                                                                         │
│  写入控制（Action 应用）:                                               │
│  ├── robot.set_joint_velocity_target()  → 设置关节速度目标             │
│  ├── robot.set_joint_position_target()  → 设置关节位置目标             │
│  ├── robot.set_joint_effort_target()    → 设置关节力矩目标             │
│  └── robot.write_data_to_sim()          → 将控制命令写入仿真           │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        env.py (RL 环境)                                  │
│                                                                         │
│  _get_observations():  从 robot.data.* 读取 → 构造 obs tensor          │
│  _apply_action():      解码 action → 调用 robot.set_joint_*_target()   │
│  _get_rewards():       从状态计算奖励                                   │
│  _get_dones():         判断终止条件                                     │
└─────────────────────────────────────────────────────────────────────────┘
```

### 7.2 Observation（观测）的数据来源

RL 策略的观测（输入）完全来自 **IsaacLab 封装的 API**，这些 API 底层从 PhysX 仿真引擎读取 USD 资产的实时状态。

#### 7.2.1 可用的状态数据

| IsaacLab API | 数据类型 | 说明 | 对应 USD 概念 |
|--------------|----------|------|--------------|
| `robot.data.root_pos_w` | (N, 3) | 根部世界位置 | Articulation Root 的 xformOp:translate |
| `robot.data.root_quat_w` | (N, 4) | 根部世界姿态 (w,x,y,z) | Articulation Root 的 xformOp:orient |
| `robot.data.root_lin_vel_w` | (N, 3) | 根部线速度 | PhysX 刚体速度 |
| `robot.data.root_ang_vel_w` | (N, 3) | 根部角速度 | PhysX 刚体角速度 |
| `robot.data.joint_pos` | (N, J) | 所有关节位置 | USD Joint 的当前位置 |
| `robot.data.joint_vel` | (N, J) | 所有关节速度 | USD Joint 的当前速度 |
| `robot.data.body_pos_w` | (N, B, 3) | 所有 body 世界位置 | 各 Link/Body 的世界位置 |
| `robot.data.body_quat_w` | (N, B, 4) | 所有 body 世界姿态 | 各 Link/Body 的世界姿态 |

其中 N = 环境数量，J = 关节数量，B = body 数量。

#### 7.2.2 本项目的 Observation 构成

在 `env.py` 的 `_get_observations()` 中，观测向量（13 维）由以下部分组成：

```python
obs = torch.cat([
    d_xy_r,           # 2: 托盘相对于叉车的 XY 距离（叉车坐标系）
    cos_dyaw, sin_dyaw,  # 2: 叉车与托盘的朝向差
    v_xy_r,           # 2: 叉车 XY 速度（叉车坐标系）
    yaw_rate,         # 1: 叉车偏航角速度
    lift_pos, lift_vel,  # 2: 升降关节位置和速度
    insert_norm,      # 1: 归一化插入深度
    self.actions,     # 3: 上一步的动作（用于策略平滑）
], dim=-1)  # 总计: 13 维
```

**数据流**：
```
USD Joint "lift_joint" 
    → PhysX 仿真计算当前位置 
    → robot.data.joint_pos[:, lift_id] 
    → lift_pos (observation 第 8 维)
```

#### 7.2.3 RigidObject（非铰接体）的状态

对于托盘这类 RigidObject：

```python
pallet.data.root_pos_w   # 托盘世界位置 (N, 3)
pallet.data.root_quat_w  # 托盘世界姿态 (N, 4)
pallet.data.root_lin_vel_w  # 托盘线速度
pallet.data.root_ang_vel_w  # 托盘角速度
```

### 7.3 Action（动作）的应用方式

RL 策略输出的动作通过 **IsaacLab 封装的 API** 写入 USD 关节，PhysX 引擎根据执行器配置（Actuator）将其转换为实际的物理控制。

#### 7.3.1 可用的控制方式

| IsaacLab API | 控制类型 | 说明 |
|--------------|----------|------|
| `robot.set_joint_position_target(pos, joint_ids)` | 位置控制 | 关节移动到目标位置 |
| `robot.set_joint_velocity_target(vel, joint_ids)` | 速度控制 | 关节以目标速度转动 |
| `robot.set_joint_effort_target(effort, joint_ids)` | 力矩控制 | 关节施加目标力矩 |

#### 7.3.2 本项目的 Action 应用

在 `env.py` 的 `_apply_action()` 中，3 维动作被解码并应用：

```python
# 动作解码
drive = actions[:, 0] * wheel_speed_rad_s   # 驱动：轮子速度目标 (rad/s)
steer = actions[:, 1] * steer_angle_rad     # 转向：转向关节角度目标 (rad)
lift_v = actions[:, 2] * lift_speed_m_s     # 升降：升降关节速度目标 (m/s)

# 应用到关节
robot.set_joint_velocity_target(drive, joint_ids=front_wheel_ids)  # 前轮速度
robot.set_joint_velocity_target(drive, joint_ids=back_wheel_ids)   # 后轮速度
robot.set_joint_position_target(steer, joint_ids=rotator_ids)      # 转向位置
robot.set_joint_velocity_target(lift_v, joint_ids=[lift_id])       # 升降速度

# 写入仿真
robot.write_data_to_sim()
```

**数据流**：
```
action[0] = 0.5 (归一化)
    → drive = 0.5 * 15.0 = 7.5 rad/s
    → robot.set_joint_velocity_target(7.5, joint_ids=[front_wheel_ids])
    → PhysX 执行器根据 stiffness/damping 计算力矩
    → USD Joint "left_front_wheel_joint" 转动
```

#### 7.3.3 执行器配置（Actuator）的作用

动作如何转换为物理力矩，由 `env_cfg.py` 中的执行器配置决定：

```python
actuators={
    "front_wheels": ImplicitActuatorCfg(
        joint_names_expr=["left_front_wheel_joint", "right_front_wheel_joint"],
        velocity_limit=40.0,     # 最大速度限制
        effort_limit=500.0,      # 最大力矩限制
        stiffness=0.0,           # 位置刚度（速度控制时为 0）
        damping=100.0,           # 速度阻尼（决定速度跟踪的"软硬"）
    ),
    "lift": ImplicitActuatorCfg(
        joint_names_expr=["lift_joint"],
        velocity_limit=1.0,
        effort_limit=500.0,
        stiffness=2000.0,        # 位置刚度（用于精确定位）
        damping=200.0,
    ),
}
```

**执行器类型**：
- **ImplicitActuator**：使用 PhysX 内置的 PD 控制器，根据 stiffness/damping 计算力矩
- **ExplicitActuator**（高级）：可自定义力矩计算函数

### 7.4 USD 资产需要满足的条件

为了让 IsaacLab 正确读取状态和应用动作，USD 资产必须满足：

#### 7.4.1 对于 Articulation（铰接体，如叉车）

1. **有 Articulation Root**：USD 中必须有一个 prim 应用了 `UsdPhysics.ArticulationRootAPI`
2. **有物理关节**：关节必须应用了 `UsdPhysics.RevoluteJoint` 或 `UsdPhysics.PrismaticJoint`
3. **关节有唯一名称**：用于在代码中通过 `find_joints()` 查找
4. **有刚体属性**：各 link 必须有 `UsdPhysics.RigidBodyAPI`

#### 7.4.2 对于 RigidObject（刚体，如托盘）

1. **有 RigidBody API**：必须应用了 `UsdPhysics.RigidBodyAPI`
2. **有碰撞体**：必须有 collision shape（mesh 或 primitive）

### 7.5 如何查看 USD 资产中可用的关节/body

#### 方法 1：Isaac Sim GUI

1. 打开 USD 文件
2. 在 Stage 面板中展开层级
3. 选择关节 prim，查看 Properties 面板

#### 方法 2：Python 脚本

```python
from pxr import Usd, UsdPhysics

stage = Usd.Stage.Open("path/to/forklift.usd")

print("=== Joints ===")
for prim in stage.Traverse():
    if prim.IsA(UsdPhysics.RevoluteJoint) or prim.IsA(UsdPhysics.PrismaticJoint):
        print(f"Joint: {prim.GetPath()}")

print("\n=== RigidBodies ===")
for prim in stage.Traverse():
    if prim.HasAPI(UsdPhysics.RigidBodyAPI):
        print(f"RigidBody: {prim.GetPath()}")
```

#### 方法 3：运行时查看

```python
# 在 env.py 或验证脚本中
print("关节名称:", self.robot.joint_names)
print("Body 名称:", self.robot.body_names)
print("关节数量:", self.robot.num_joints)
print("Body 数量:", self.robot.num_bodies)
```

### 7.6 常见问题

#### Q: 更换 USD 后观测维度不对？

A: 检查新 USD 的关节数量是否与代码中的 `observation_space` 一致。如果关节数量变化，需要调整 `_get_observations()` 中的观测构造。

#### Q: 动作没有效果？

A: 检查以下几点：
1. `env_cfg.py` 中的 `joint_names_expr` 是否与 USD 关节名称匹配
2. 执行器的 `effort_limit` 是否足够大
3. 是否调用了 `robot.write_data_to_sim()`

#### Q: 某些状态读不到（返回 0 或 NaN）？

A: 检查 USD 资产是否有对应的物理 API：
- 关节速度需要关节有 `UsdPhysics.DriveAPI`
- 刚体速度需要 `UsdPhysics.RigidBodyAPI`

### 7.7 总结：USD 与 RL 的接口关系

```
┌─────────────────────────────────────────────────────────────────┐
│                        USD 资产                                  │
│  ├── Joint 名称          →  代码中 find_joints() 查找           │
│  ├── Joint 类型          →  决定可用的控制方式                  │
│  ├── Body/Link 结构      →  决定 body_pos_w 的维度              │
│  └── 物理属性            →  影响仿真行为                        │
├─────────────────────────────────────────────────────────────────┤
│                        env_cfg.py                                │
│  ├── actuators           →  定义动作如何转换为物理控制          │
│  ├── observation_space   →  观测向量维度                        │
│  └── action_space        →  动作向量维度                        │
├─────────────────────────────────────────────────────────────────┤
│                        env.py                                    │
│  ├── _get_observations() →  从 robot.data.* 读取状态            │
│  └── _apply_action()     →  通过 robot.set_joint_*() 应用动作   │
└─────────────────────────────────────────────────────────────────┘
```

**关键点**：
- USD 资产定义了"有什么"（关节、body、物理属性）
- IsaacLab 封装了"怎么读/写"（robot.data.*, robot.set_joint_*）
- env.py 决定了"读什么给策略、策略输出怎么用"
