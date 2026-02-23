# RigidBody / Collision / Mass 属性完整参考手册

本文档详细列出 Isaac Lab 中 `modify_rigid_body_properties`、`modify_collision_properties`、`modify_mass_properties` 所修改的每个属性的含义、PhysX 默认值，以及本项目的实际配置。

源码位置：`IsaacLab/source/isaaclab/isaaclab/sim/schemas/schemas_cfg.py`

---

## 目录

1. [核心机制](#1-核心机制)
2. [RigidBodyPropertiesCfg — 刚体动力学属性](#2-rigidbodypropertiescfg--刚体动力学属性)
3. [CollisionPropertiesCfg — 碰撞检测属性](#3-collisionpropertiescfg--碰撞检测属性)
4. [MassPropertiesCfg — 质量属性](#4-masspropertiescfg--质量属性)
5. [ArticulationRootPropertiesCfg — 铰接体根属性](#5-articulationrootpropertiescfg--铰接体根属性)
6. [ConvexDecompositionPropertiesCfg — 凸分解碰撞参数](#6-convexdecompositionpropertiescfg--凸分解碰撞参数)
7. [JointDrivePropertiesCfg — 关节驱动属性](#7-jointdrivepropertiescfg--关节驱动属性)

---

## 1. 核心机制

### 1.1 所有字段默认 None = 不修改

```python
rigid_body_enabled: bool | None = None  # None 表示保留 USD/PhysX 原始值
```

只需要设你想覆盖的字段，其余自动沿用 USD 文件中的值或 PhysX 引擎默认值。

### 1.2 这些属性修改的是什么？

这三组 Cfg 修改的是 **PhysX 引擎的运行时行为参数**，不涉及几何形状（Mesh）。它们控制的是"物体如何参与物理仿真"，而非"物体长什么样"。

### 1.3 本项目中的实际使用

```python
# env_cfg.py 托盘配置
pallet_cfg = RigidObjectCfg(
    spawn=sim_utils.UsdFileCfg(
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            kinematic_enabled=False,
            disable_gravity=False,
            max_depenetration_velocity=1.0,
        ),
        mass_props=sim_utils.MassPropertiesCfg(mass=45.0),
        collision_props=sim_utils.CollisionPropertiesCfg(
            collision_enabled=True,
            contact_offset=0.02,
            rest_offset=0.005,
        ),
    ),
)
```

---

## 2. RigidBodyPropertiesCfg — 刚体动力学属性

控制"这个物体如何参与物理仿真"。对应 USD 中的 `RigidBodyAPI` + `PhysxRigidBodyAPI`。

| 属性 | 含义 | PhysX 默认值 | 本项目托盘 | 本项目叉车 |
|---|---|---|---|---|
| `rigid_body_enabled` | 是否启用刚体仿真 | `True` | `True` | `True` |
| `kinematic_enabled` | 是否为运动学物体（只能被代码移动，不受力） | `False` | `False` | — |
| `disable_gravity` | 是否关闭重力 | `False` | `False` | — |
| `linear_damping` | 线性阻尼（模拟空气阻力，减缓平移） | `0.0` | 未设置 | — |
| `angular_damping` | 角阻尼（减缓旋转） | `0.05` | 未设置 | — |
| `max_linear_velocity` | 最大线速度上限 (m/s) | `100.0` | 未设置 | `20.0` |
| `max_angular_velocity` | 最大角速度上限 (deg/s) | `5729.58` (≈100 rad/s) | 未设置 | `20.0` |
| `max_depenetration_velocity` | 穿透时最大分离速度 (m/s) | `100.0` | `1.0` | `5.0` |
| `max_contact_impulse` | 接触点最大冲量 | `inf` | 未设置 | — |
| `enable_gyroscopic_forces` | 是否启用陀螺力（高速旋转进动） | `False` | 未设置 | `True` |
| `retain_accelerations` | 跨子步保持加速度 | `False` | 未设置 | — |
| `solver_position_iteration_count` | 位置求解迭代次数（越多越精确，越慢） | `4` | 未设置 | — |
| `solver_velocity_iteration_count` | 速度求解迭代次数 | `1` | 未设置 | — |
| `sleep_threshold` | 动能低于此值时进入休眠（节省性能） | `0.005` | 未设置 | — |
| `stabilization_threshold` | 稳定化的动能阈值 | `0.001` | 未设置 | — |

### 重要属性直觉理解

#### `kinematic_enabled` — 最关键的开关

```
kinematic_enabled = True  → 物体只能被代码 set_position 移动
                            不受力、不受重力、不会被碰撞推动
                            像是"被上帝的手固定住"

kinematic_enabled = False → 物体是完全动态的
                            受重力、受碰撞力、可以被推/举
                            像是"放在桌子上的真实物体"
```

#### `max_depenetration_velocity` — 防弹飞

当两个物体意外穿透时，PhysX 要把它们分开。默认分离速度可达 100 m/s（极其暴力，物体会弹飞）。设成 1.0 后，分离过程变得温和，避免托盘被弹飞。

```
默认 100.0 m/s:  穿透 → 💥 弹飞到天上
设为   1.0 m/s:  穿透 → 温和推开，物体不会失控
```

#### `linear_damping / angular_damping` — 空气阻力

```
damping = 0.0:  没有阻力，物体一旦动起来就不停（太空中的行为）
damping > 0.0:  模拟空气/介质阻力，物体会逐渐减速
```

#### `sleep_threshold` — 性能优化

```
物体动能 > sleep_threshold:  PhysX 正常计算物理（消耗 CPU）
物体动能 < sleep_threshold:  进入休眠，不再参与物理计算（节省性能）
                              直到被外力唤醒
```

---

## 3. CollisionPropertiesCfg — 碰撞检测属性

控制"碰撞检测的灵敏度和精度"。对应 USD 中的 `CollisionAPI` + `PhysxCollisionAPI`。

| 属性 | 含义 | PhysX 默认值 | 本项目托盘 |
|---|---|---|---|
| `collision_enabled` | 是否开启碰撞 | `True` | `True` |
| `contact_offset` | 碰撞检测提前量 (m) | `0.02` | `0.02` |
| `rest_offset` | 静止时最小间隙 (m) | `0.0` | `0.005` |
| `torsional_patch_radius` | 扭转摩擦接触片半径 (m) | `0.0`（不启用） | 未设置 |
| `min_torsional_patch_radius` | 扭转摩擦最小半径 (m) | `0.0` | 未设置 |

### 重要属性直觉理解

#### `contact_offset` — 碰撞提前量

两个物体还差 `contact_offset` 距离就开始计算碰撞力。好处是防止高速穿透；坏处是在狭窄通道中让有效空间变小。

```
Mesh 表面：        ┃    ┃    ← 实际几何边界
                   ┃    ┃
contact_offset:  ┃│    │┃   ← 碰撞检测边界（向外扩了 0.02m）
                 ┃│    │┃
                   ↕
             有效通道变窄了 0.04m（两侧各 0.02m）
```

在本项目中，这是"凸分解膨胀卡死 (Convex Decomposition Wedging)"的帮凶之一——碰撞壳比 Mesh 表面向外膨胀了 2cm，导致货叉在托盘内部狭窄通道里更容易被卡住。

#### `rest_offset` — 静止间隙

```
rest_offset = 0.0:    两个物体静止时完全贴合
rest_offset = 0.005:  静止时保持 5mm 间隙（看起来略微悬浮）
rest_offset < 0:      允许轻微穿透（用于特殊效果）
```

约束条件：`rest_offset` 必须小于 `contact_offset`。

#### `torsional_patch_radius` — 扭转摩擦

模拟物体接触面积产生的旋转阻力。例如一个重箱子放在地板上，试图绕竖直轴旋转它时的摩擦阻力。设为 0 时不启用。

---

## 4. MassPropertiesCfg — 质量属性

最简单的一组。对应 USD 中的 `MassAPI`。

| 属性 | 含义 | PhysX 默认值 | 本项目托盘 |
|---|---|---|---|
| `mass` | 质量 (kg) | `0.0`（由 density 计算） | `45.0` |
| `density` | 密度 (kg/m³) | `1000.0` | 未设置 |

### 质量计算优先级

PhysX 有一套优先级规则：

```python
if mass > 0:
    使用指定的 mass（忽略 density）
elif density > 0:
    mass = density × 碰撞体积（由 CollisionAPI 的形状计算）
else:
    使用 PhysX 默认密度 1000 kg/m³ 计算
```

本项目显式设了 `mass=45.0`，所以密度被忽略，托盘恒定 45kg。

### 注意：质心 (Center of Mass) 不在这里设

`MassPropertiesCfg` 只能设 mass 和 density。质心的偏移需要直接操作 USD 的 `MassAPI`：

```python
from pxr import UsdPhysics, Gf
mass_api = UsdPhysics.MassAPI(prim)
mass_api.GetCenterOfMassAttr().Set(Gf.Vec3f(-0.2, 0.0, 0.0))  # 前移 20cm
```

这正是本项目 `scripts/shift_pallet_com.py` 做的事情。

---

## 5. ArticulationRootPropertiesCfg — 铰接体根属性

控制铰接系统（如叉车）的全局物理行为。对应 USD 中的 `ArticulationRootAPI`。

| 属性 | 含义 | PhysX 默认值 | 本项目叉车 |
|---|---|---|---|
| `articulation_enabled` | 是否启用铰接体 | `True` | 未设置 |
| `enabled_self_collisions` | 内部各 link 之间是否互相碰撞 | `True` | `False` |
| `solver_position_iteration_count` | 位置求解迭代次数 | `4` | `4` |
| `solver_velocity_iteration_count` | 速度求解迭代次数 | `1` | `0` |
| `sleep_threshold` | 休眠动能阈值 | `0.005` | `0.005` |
| `stabilization_threshold` | 稳定化阈值 | `0.001` | `0.001` |
| `fix_root_link` | 是否将根节点固定到世界 | `None` | 未设置 |

### 重要属性说明

#### `enabled_self_collisions = False`

关闭后，叉车内部的各 link（底盘、轮子、货叉）不会互相碰撞。好处是避免不必要的碰撞计算，也避免模型设计不完美导致的内部穿透抖动。

#### `solver_velocity_iteration_count = 0`

速度求解器关闭。位置求解器 = 4 已经足够，关闭速度求解可以提升性能。在高精度需求场景（如柔性物体接触）才需要开启。

#### `fix_root_link`

如果设为 `True`，会在根节点和世界之间创建一个 FixedJoint，机器人就像被钉在桌子上。本项目不设置（叉车需要自由移动）。

---

## 6. ConvexDecompositionPropertiesCfg — 凸分解碰撞参数

控制 VHACD 算法如何将凹形 Mesh 分解为多个凸体。本项目在 `env.py` 中通过低级 API 设置。

| 属性 | 含义 | PhysX 默认值 | 本项目托盘 |
|---|---|---|---|
| `max_convex_hulls` | 最多拆成几个凸体 | `32` | `8` |
| `hull_vertex_limit` | 每个凸体最大顶点数 | `64` | `64` |
| `min_thickness` | 凸体最小厚度 (m) | `0.001` | 未设置 |
| `voxel_resolution` | 体素化分辨率 | `500000` | 未设置 |
| `error_percentage` | 分解误差容忍度 (%) | `10` | 未设置 |
| `shrink_wrap` | 是否将凸包顶点投影回原始 Mesh 表面 | `False` | 未设置 |

### 参数调优影响

```
max_convex_hulls 越大 → 碰撞越精确（凹陷保留更好）→ 性能越差
max_convex_hulls 越小 → 碰撞越粗糙（凹陷可能被填平）→ 性能越好

本项目实测：
  maxConvexHulls=1  (convexHull):  ~17000 steps/s，但无法插入
  maxConvexHulls=8:                足以保留 pocket，允许插入 ✅
  maxConvexHulls=32:               ~890 steps/s，精度更高但太慢
```

---

## 7. JointDrivePropertiesCfg — 关节驱动属性

控制关节的驱动方式。对应 USD 中的 `DriveAPI`。

| 属性 | 含义 | 本项目叉车举升关节 |
|---|---|---|
| `drive_type` | 驱动类型：`"force"` 或 `"acceleration"` | — |
| `max_effort` | 最大驱动力/力矩 | — |
| `max_velocity` | 最大速度 (线性 m/s / 角 rad/s) | — |
| `stiffness` | 位置控制刚度（PD 控制器的 P） | — |
| `damping` | 速度控制阻尼（PD 控制器的 D） | — |

本项目中关节驱动通过 `ImplicitActuatorCfg` 配置（更高层的封装），而非直接使用 `JointDrivePropertiesCfg`：

```python
# env_cfg.py 中的配置
"lift": ImplicitActuatorCfg(
    joint_names_expr=["lift_joint"],
    velocity_limit=1.0,
    effort_limit=50000.0,   # 50kN
    stiffness=200000.0,     # 位置控制 P 增益
    damping=10000.0,        # 速度控制 D 增益
)
```

`ImplicitActuatorCfg` 内部会将 stiffness/damping 写入 PhysX 的 DriveAPI，覆盖 USD 文件中的原始值。

### 驱动力计算公式

PhysX 关节驱动力 = `stiffness × (target_pos - current_pos) + damping × (target_vel - current_vel)`

```
纯位置控制:  stiffness > 0, damping > 0    → 设 target_pos，关节移动到目标位置
纯速度控制:  stiffness = 0, damping > 0    → 设 target_vel，关节以目标速度转动
纯力矩控制:  stiffness = 0, damping = 0    → 直接设 effort
```

---

## 附录：属性层级总览

```
env_cfg.py 中的配置
│
├── RigidObjectCfg（托盘）
│   └── UsdFileCfg
│       ├── rigid_props: RigidBodyPropertiesCfg
│       │   → 修改 RigidBodyAPI + PhysxRigidBodyAPI
│       │   → 控制：能不能动、受不受重力、速度上限、穿透行为
│       │
│       ├── mass_props: MassPropertiesCfg
│       │   → 修改 MassAPI
│       │   → 控制：质量、密度
│       │
│       └── collision_props: CollisionPropertiesCfg
│           → 修改 CollisionAPI + PhysxCollisionAPI
│           → 控制：碰撞开关、检测提前量、静止间隙
│
├── ArticulationCfg（叉车）
│   └── UsdFileCfg
│       ├── rigid_props: RigidBodyPropertiesCfg（同上）
│       ├── mass_props: MassPropertiesCfg（同上）
│       ├── articulation_props: ArticulationRootPropertiesCfg
│       │   → 修改 ArticulationRootAPI
│       │   → 控制：自碰撞、求解器迭代、休眠
│       │
│       └── actuators: dict[str, ImplicitActuatorCfg]
│           → 修改 DriveAPI（关节驱动）
│           → 控制：stiffness、damping、速度/力矩上限
│
└── GroundPlaneCfg（地面）
    → 静态碰撞体，无需物理属性配置
```
