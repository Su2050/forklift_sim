# USD 物理属性 Pipeline 与 Isaac Sim/IsaacLab 实体概念全解

本文档回答两个核心问题：
1. USD 文件中的物理属性是如何被 Isaac Lab 加载和应用的？
2. Isaac Sim / IsaacLab 中除了刚体、关节外还有哪些实体概念？

---

## 目录

1. [USD 文件里到底有什么](#1-usd-文件里到底有什么)
2. [每个物理 API 标签的作用](#2-每个物理-api-标签的作用)
3. [Isaac Lab 的 Spawn Pipeline](#3-isaac-lab-的-spawn-pipeline)
4. [`modify_` vs `define_` 致命区别](#4-modify_-vs-define_-致命区别)
5. [完整 Pipeline 流程图（以托盘为例）](#5-完整-pipeline-流程图以托盘为例)
6. [Isaac Sim / IsaacLab 完整实体概念体系](#6-isaac-sim--isaaclab-完整实体概念体系)
7. [本项目用了哪些实体](#7-本项目用了哪些实体)

---

## 1. USD 文件里到底有什么

USD 文件只**必须**有几何 Mesh（视觉形状）。其余都是**可选**的物理 API 标签（Schema），可以有也可以没有。把 USD 想象成一个"贴标签"的系统：

```
pallet.usd 文件结构（Prim 树）
│
├── /Pallet              ← 根 Xform（变换节点）
│   ├── /Pallet/Mesh_0   ← Mesh 几何体（视觉形状）—— 必须有
│   ├── /Pallet/Mesh_1   ← 可能有多个 Mesh
│   └── ...
│
│  可选"标签"（API Schema）：
│   [RigidBodyAPI]        ← 刚体动力学（有没有这张标签决定了物体能不能动）
│   [MassAPI]             ← 质量属性
│   [CollisionAPI]        ← 碰撞开关
│   [MeshCollisionAPI]    ← 碰撞近似方式
│   [PhysxRigidBodyAPI]   ← PhysX 特有的高级刚体参数
│   [PhysxCollisionAPI]   ← PhysX 特有的碰撞参数
```

### Nucleus 的 `pallet.usd` 实际有什么？

根据代码中的诊断日志和注释（`env.py` 第 432-434 行）：

```
Nucleus pallet.usd 实际内容：
  ✅ Mesh 几何体（多个子 Mesh）
  ✅ CollisionAPI（在子 Mesh 上）
  ✅ MeshCollisionAPI（approximation = "convexDecomposition"）
  ❌ RigidBodyAPI      ← 没有！
  ❌ MassAPI           ← 没有！
```

结论：有形状、有碰撞壳，但**不能动、没有质量**——它本质上是一个静态障碍物。

---

## 2. 每个物理 API 标签的作用

| API 标签 | 解决什么问题 | 没有它会怎样 |
|----------|------------|------------|
| **Mesh（几何体）** | 定义"长什么样" | 什么都看不见 |
| **CollisionAPI** | 开启碰撞检测 | 物体互相穿过，像幽灵一样 |
| **MeshCollisionAPI** | 定义碰撞形状的近似方式 | 默认用 triangleMesh，只能用于 kinematic/static |
| **RigidBodyAPI** | 赋予物理动力学（受力、受重力、能被推动） | 物体是静态的，永远纹丝不动（像墙壁） |
| **MassAPI** | 定义质量、惯量、质心 | PhysX 按密度和碰撞体积自动估算（可能不准） |
| **PhysxRigidBodyAPI** | PhysX 高级参数（max_depenetration_velocity 等） | 使用 PhysX 默认值 |
| **PhysxCollisionAPI** | PhysX 高级碰撞参数（contact_offset, rest_offset） | 使用 PhysX 默认值 |

### 类比理解

```
Mesh           = 一个塑料模型（能看，但什么都不会发生）
+ CollisionAPI = 给模型套上一层"碰撞壳"（别的东西碰到会被挡住）
+ RigidBodyAPI = 给模型装上"引擎"（它现在能动了，受重力，能被推）
+ MassAPI      = 给引擎标注"45公斤"（决定推它要多大的力）
```

### 碰撞近似方式对比

`MeshCollisionAPI.approximation` 的取值：

| 取值 | 说明 | 优缺点 |
|------|------|--------|
| `"none"` (triangleMesh) | 完全精确 | 只能用于 kinematic/static |
| `"convexHull"` | 包成一个"气球" | 丢失所有凹陷 |
| `"convexDecomposition"` | 多个凸体拼出凹陷 | **托盘用的**，平衡精度和性能 |
| `"boundingCube"` | 一个包围盒 | 最粗糙 |
| `"boundingSphere"` | 一个包围球 | 最粗糙 |
| `"sdf"` | 符号距离场 | 最精确，支持动态，但最慢 |

用托盘举例：

```
原始形状（有 pocket 凹槽）：
  ┌──────────┐
  │  ┌────┐  │
  │  │    │  │  ← pocket（叉齿要从这里插入）
  │  └────┘  │
  └──────────┘

convexHull 近似：
  ┌──────────┐
  │██████████│  ← pocket 被填满了！叉齿无法插入
  │██████████│
  └──────────┘

convexDecomposition 近似：
  ┌──────────┐
  │▓▓┌────┐▓▓│  ← 用多个凸体拼出，保留了 pocket 空间
  │▓▓│    │▓▓│
  │▓▓└────┘▓▓│
  └──────────┘
```

---

## 3. Isaac Lab 的 Spawn Pipeline

当你在 `env_cfg.py` 中写：

```python
pallet_cfg = RigidObjectCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="...pallet.usd",
        rigid_props=RigidBodyPropertiesCfg(...),     # 想要设置刚体
        mass_props=MassPropertiesCfg(mass=45.0),     # 想要设置质量
        collision_props=CollisionPropertiesCfg(...),  # 想要设置碰撞参数
    ),
)
```

Isaac Lab 在 spawn 时的调用链（`IsaacLab/sim/spawners/from_files/from_files.py` 第 38 行）：

```
spawn_from_usd()
│
├── 1. 加载 USD 文件到场景（创建 prim 引用）
│
├── 2. if rigid_props is not None:
│      schemas.modify_rigid_body_properties(prim_path, cfg.rigid_props)
│      ^^^^^^^^
│      注意是 modify，不是 define！
│
├── 3. if collision_props is not None:
│      schemas.modify_collision_properties(prim_path, cfg.collision_props)
│
└── 4. if mass_props is not None:
       schemas.modify_mass_properties(prim_path, cfg.mass_props)
```

**关键点**：Isaac Lab 的标准 spawn 流程只调用 `modify_`，不调用 `define_`。

---

## 4. `modify_` vs `define_` 致命区别

### `modify_rigid_body_properties`（spawn 默认调用的）

源码位置：`IsaacLab/source/isaaclab/isaaclab/sim/schemas/schemas.py` 第 242 行

```python
def modify_rigid_body_properties(prim_path, cfg, stage) -> bool:
    rigid_body_prim = stage.GetPrimAtPath(prim_path)
    if not UsdPhysics.RigidBodyAPI(rigid_body_prim):
        return False   # ← USD 没有 RigidBodyAPI？直接返回 False，什么都不做！
    # ... 修改已有属性 ...
    return True
```

### `define_rigid_body_properties`（需要手动调用的）

源码位置：`IsaacLab/source/isaaclab/isaaclab/sim/schemas/schemas.py` 第 208 行

```python
def define_rigid_body_properties(prim_path, cfg, stage):
    prim = stage.GetPrimAtPath(prim_path)
    if not UsdPhysics.RigidBodyAPI(prim):
        UsdPhysics.RigidBodyAPI.Apply(prim)  # ← 没有？我给你创建一个！
    modify_rigid_body_properties(prim_path, cfg, stage)  # 然后再修改属性
```

### 三组函数完全同理

| 函数 | 行为 | USD 已有 API | USD 没有 API |
|------|------|------------|------------|
| `modify_rigid_body_properties` | 只改不建 | 修改属性值 | **return False，静默失败** |
| `define_rigid_body_properties` | 先建后改 | 直接修改 | 先 Apply 创建，再修改 |
| `modify_collision_properties` | 只改不建 | 修改属性值 | **return False，静默失败** |
| `define_collision_properties` | 先建后改 | 直接修改 | 先 Apply 创建，再修改 |
| `modify_mass_properties` | 只改不建 | 修改属性值 | **return False，静默失败** |
| `define_mass_properties` | 先建后改 | 直接修改 | 先 Apply 创建，再修改 |

### 这意味着什么？

- 如果 USD 文件**已有**所有 API → `env_cfg.py` 中声明的 `rigid_props` / `mass_props` 通过 `modify_` 就能正常生效
- 如果 USD 文件**缺少** API → `modify_` 静默失败，你在 `env_cfg.py` 里写了也白写，必须在代码中手动调用 `define_` 补上

---

## 5. 完整 Pipeline 流程图（以托盘为例）

```
                    USD 文件（Nucleus HTTP 下载）
                    ┌─────────────────────────────┐
                    │  ✅ Mesh（视觉几何）          │
                    │  ✅ CollisionAPI（碰撞壳）     │
                    │  ❌ RigidBodyAPI              │
                    │  ❌ MassAPI                   │
                    └─────────────────────────────┘
                                │
                    ① Isaac Lab spawn_from_usd()
                                │
                    ┌───────────▼───────────────────┐
                    │  加载 USD → 创建 prim 引用       │
                    │                                │
                    │  modify_rigid_body(rigid_props) │
                    │  → USD 没有 RigidBodyAPI        │
                    │  → return False ❌ 静默失败！    │
                    │                                │
                    │  modify_collision(collision_props)│
                    │  → USD 有 CollisionAPI          │
                    │  → 修改 contact_offset 等 ✅     │
                    │                                │
                    │  modify_mass(mass_props)        │
                    │  → USD 没有 MassAPI             │
                    │  → return False ❌ 静默失败！    │
                    └───────────┬───────────────────┘
                                │
                    此时托盘状态：有形状、有碰撞壳，
                    但没有刚体、没有质量 → 静态障碍物
                                │
                    ② env.py _setup_pallet_physics() 补丁
                                │
                    ┌───────────▼───────────────────┐
                    │  define_rigid_body_properties() │
                    │  → 没有 RigidBodyAPI？先创建！   │
                    │  → 然后设置 enabled=True,       │
                    │    kinematic=False ✅            │
                    │                                │
                    │  遍历子 Mesh:                    │
                    │  → CollisionAPI.Apply()         │
                    │  → MeshCollisionAPI             │
                    │    approximation=convexDecomp ✅ │
                    └───────────┬───────────────────┘
                                │
                    ③ clone_environments() 克隆
                                │
                    ┌───────────▼───────────────────┐
                    │  模板 env_0 的所有物理属性       │
                    │  被复制到 env_1 ~ env_N ✅      │
                    └───────────┬───────────────────┘
                                │
                    ④ PhysX 仿真启动
                                │
                    ┌───────────▼───────────────────┐
                    │  托盘现在是：                    │
                    │  ✅ 有几何形状                   │
                    │  ✅ 有凸分解碰撞壳（能被插入）   │
                    │  ✅ 有刚体动力学（能被推动/举起） │
                    │  ✅ 有质量（45kg，受重力）       │
                    └─────────────────────────────┘
```

### 对比：叉车（forklift_c.usd）为什么不需要补丁？

```
forklift_c.usd 是 NVIDIA 官方的完整 Physics-ready 资产：
  ✅ ArticulationRootAPI
  ✅ RigidBodyAPI（每个 link 上都有）
  ✅ MassAPI
  ✅ 所有 Joint + DriveAPI
  ✅ CollisionAPI

→ spawn_from_usd() 调用 modify_* 全部成功
→ env_cfg.py 中声明的 rigid_props 直接生效
→ 不需要任何补丁
```

---

## 6. Isaac Sim / IsaacLab 完整实体概念体系

整个体系分为三层：

```
┌─────────────────────────────────────────────────────────────┐
│  第三层：IsaacLab 封装（Python 高层 API，写 RL 用的）          │
├─────────────────────────────────────────────────────────────┤
│  第二层：USD Physics Schema（贴在 USD 文件上的"标签"）         │
├─────────────────────────────────────────────────────────────┤
│  第一层：PhysX 引擎（NVIDIA 的物理仿真引擎，底层 C++）        │
└─────────────────────────────────────────────────────────────┘
```

### 6.1 第一层：PhysX 引擎的基本概念

| 概念 | 是什么 | 现实世界类比 |
|------|--------|------------|
| **Rigid Body（刚体）** | 不变形的固体 | 木块、钢板、托盘 |
| **Articulation（铰接体）** | 多个刚体通过关节连接组成的系统 | 机器人、叉车、机械臂 |
| **Joint（关节）** | 连接两个刚体的约束 | 铰链、滑轨、万向节 |
| **Deformable Body（可变形体）** | 可以弯曲/拉伸的物体 | 橡胶、布料、海绵 |
| **Particle System（粒子系统）** | 大量小粒子的集合 | 水、沙子、烟雾 |
| **Collision Shape（碰撞形状）** | 物体用于碰撞检测的几何外壳 | 物体的"轮廓" |
| **Physics Material（物理材质）** | 摩擦力、弹性等表面属性 | 橡胶地面 vs 冰面 |
| **Physics Scene（物理场景）** | 管理整个仿真世界的容器 | 整个"物理引擎房间" |

### 6.2 第二层：USD Physics Schema（"标签"体系）

贴在 USD Prim 上的标签，每贴一个就获得一种能力。

#### A. 刚体相关

| Schema / API 标签 | 作用 | 关键属性 |
|---|---|---|
| `RigidBodyAPI` | 让物体有动力学（能动、受力） | `rigid_body_enabled`, `kinematic_enabled` |
| `MassAPI` | 定义质量属性 | `mass`, `density`, `centerOfMass`, `diagonalInertia` |
| `PhysxRigidBodyAPI` | PhysX 高级刚体参数 | `max_depenetration_velocity`, `sleep_threshold` |

#### B. 碰撞相关

| Schema / API 标签 | 作用 | 关键属性 |
|---|---|---|
| `CollisionAPI` | 开启碰撞检测 | `collision_enabled` |
| `MeshCollisionAPI` | 碰撞形状近似方式 | `approximation` |
| `PhysxCollisionAPI` | PhysX 高级碰撞参数 | `contact_offset`, `rest_offset` |
| `PhysxConvexDecompositionCollisionAPI` | 凸分解碰撞的详细参数 | `maxConvexHulls`, `hullVertexLimit` |

#### C. 铰接体 / 关节相关

| Schema / API 标签 | 作用 | 本项目中的对应 |
|---|---|---|
| `ArticulationRootAPI` | 标记铰接体的根节点 | 叉车 body |
| `RevoluteJoint` | 旋转关节（绕一个轴转） | wheel_joint, rotator_joint |
| `PrismaticJoint` | 滑移关节（沿一个轴平移） | lift_joint |
| `FixedJoint` | 固定关节（锁死两个刚体） | 把机器人固定到世界 |
| `SphericalJoint` | 球关节（三自由度旋转） | 万向节 |
| `D6Joint` | 六自由度通用关节 | 任意约束组合 |
| `DriveAPI` | 关节驱动（位置/速度/力矩控制） | 每个 joint 上都有 |
| `JointLimitAPI` | 关节运动范围限制 | lift_joint 的上下限 |

#### D. 可变形体

| Schema / API 标签 | 作用 | 用途 |
|---|---|---|
| `DeformableBodyAPI` | 让 Mesh 可变形 | 软体物理（橡胶、肉） |
| `PhysxDeformableBodyAPI` | PhysX 可变形体参数 | 弹性系数、阻尼 |
| `DeformableSurfaceAPI` | 可变形表面（布料） | 衣服、帐篷 |

#### E. 物理材质

| Schema / API 标签 | 作用 |
|---|---|
| `MaterialAPI` | 绑定物理材质到物体 |
| `PhysxMaterialAPI` | 摩擦系数、恢复系数（弹性碰撞程度） |

#### F. 其他

| Schema / API 标签 | 作用 |
|---|---|
| `FilteredPairsAPI` | 碰撞过滤（指定哪些物体之间不碰撞） |
| `PhysxSceneAPI` | 整个物理场景的全局设置（重力方向、求解器迭代次数） |
| `FixedTendonAPI` | 固定肌腱（模拟肌肉或弹簧连接） |
| `SpatialTendonAPI` | 空间肌腱 |

### 6.3 第三层：IsaacLab 封装层

IsaacLab 把底层概念包装成更易用的 Python 类。

#### 资产类（Assets）

| IsaacLab 类 | 底层对应 | 说明 |
|---|---|---|
| `Articulation` | ArticulationRoot + 多个 RigidBody + Joint | 铰接机器人（叉车、机械臂） |
| `RigidObject` | 单个 RigidBody | 单个刚体（托盘、箱子） |
| `RigidObjectCollection` | 多个独立 RigidBody 的集合 | 多个箱子的集合 |
| `DeformableObject` | DeformableBody | 可变形物体 |
| `SurfaceGripper` | 吸附式抓手 | 真空吸盘 |

#### 传感器类（Sensors）

| IsaacLab 类 | 作用 | RL 中的用途 |
|---|---|---|
| `Camera` / `TiledCamera` | RGB/深度/分割图像 | 视觉观测 |
| `ContactSensor` | 接触力检测 | 检测叉齿是否碰到托盘 |
| `RayCaster` | 射线检测（激光雷达） | 障碍物距离感知 |
| `RayCasterCamera` | 基于射线的深度相机 | 点云 |
| `Imu` | 惯性测量单元（加速度/角速度） | 机器人姿态感知 |
| `FrameTransformer` | 坐标系变换跟踪 | 跟踪特定 link 的世界位姿 |

#### 地形类（Terrains）

| 类型 | 说明 |
|---|---|
| `GroundPlane` | 平地 |
| `HfTerrain` | 高度场地形（凹凸不平的地面） |
| `MeshTerrain` | 自定义 Mesh 地形 |

### 6.4 概念层级关系总图

```
Isaac Sim / IsaacLab 完整概念图
═══════════════════════════════

物理实体 (能参与仿真的东西)
├── 刚性物体
│   ├── RigidBody（单个刚体）─── IsaacLab: RigidObject
│   │   ├── + CollisionAPI      → 能碰撞
│   │   ├── + MassAPI           → 有质量
│   │   └── + MaterialAPI       → 有摩擦/弹性
│   │
│   └── Articulation（铰接系统）─── IsaacLab: Articulation
│       ├── ArticulationRootAPI → 标记根节点
│       ├── 多个 RigidBody      → 各个 link（底盘、轮子、货叉...）
│       ├── Joint（关节）        → 连接 link
│       │   ├── RevoluteJoint   → 旋转（轮子、转向）
│       │   ├── PrismaticJoint  → 滑移（升降）
│       │   ├── FixedJoint      → 固定
│       │   └── DriveAPI        → 驱动控制
│       └── Tendon（肌腱）       → 弹簧/线缆连接
│
├── 柔性物体
│   ├── DeformableBody ─── IsaacLab: DeformableObject
│   └── DeformableSurface（布料）
│
├── 粒子
│   └── ParticleSystem（流体/沙子）
│
├── 静态环境
│   ├── GroundPlane / Terrain（地面）
│   └── Static Collider（静态障碍物，只有 CollisionAPI 没有 RigidBodyAPI）
│
└── 感知设备 ─── IsaacLab: Sensors
    ├── Camera（相机）
    ├── ContactSensor（接触传感器）
    ├── RayCaster（激光雷达/射线）
    ├── Imu（惯性传感器）
    └── FrameTransformer（位姿跟踪）
```

---

## 7. 本项目用了哪些实体

```
叉车仿真项目 (forklift_sim)：
├── Articulation     → 叉车 (forklift_c.usd)
│   ├── RevoluteJoint × 6  → 4 个轮子 + 2 个转向
│   ├── PrismaticJoint × 1 → 升降 lift_joint
│   └── DriveAPI × 7       → 每个关节的驱动
├── RigidObject      → 托盘 (pallet.usd / pallet_com_shifted.usd)
│   ├── RigidBodyAPI       → 运行时由 _setup_pallet_physics() 注入
│   └── CollisionAPI       → USD 自带 + 运行时强制 convexDecomposition
├── GroundPlane      → 地面
└── （未使用传感器 —— 纯向量观测，不需要 Camera/Lidar 等）
```

---

## 附录：常见问题

### Q: 为什么 `env_cfg.py` 里写了 `rigid_props` 但托盘还是没有物理？

A: 因为 Isaac Lab 的 `spawn_from_usd()` 只调用 `modify_`（只改不建），而 Nucleus 的 `pallet.usd` 没有 `RigidBodyAPI`，`modify_` 静默返回 False。需要在代码中手动调用 `define_rigid_body_properties()` 来先创建再修改。

### Q: 叉车为什么不需要补丁？

A: `forklift_c.usd` 是 NVIDIA 官方的完整 Physics-ready 资产，自带了所有需要的 API（ArticulationRootAPI、RigidBodyAPI、MassAPI、DriveAPI 等）。`modify_` 能直接修改已有属性。

### Q: 如何判断一个 USD 文件有哪些物理 API？

A: 方法一：Isaac Sim GUI 中打开，查看 Properties 面板。方法二：Python 脚本遍历检查：

```python
from pxr import Usd, UsdPhysics

stage = Usd.Stage.Open("path/to/asset.usd")
for prim in stage.Traverse():
    apis = []
    if prim.HasAPI(UsdPhysics.RigidBodyAPI): apis.append("RigidBody")
    if prim.HasAPI(UsdPhysics.CollisionAPI): apis.append("Collision")
    if prim.HasAPI(UsdPhysics.MassAPI): apis.append("Mass")
    if prim.HasAPI(UsdPhysics.MeshCollisionAPI): apis.append("MeshCollision")
    if apis:
        print(f"{prim.GetPath()}: {', '.join(apis)}")
```

### Q: 如果未来要加视觉观测怎么办？

A: 在 `env_cfg.py` 中添加 `Camera` 或 `TiledCamera` 传感器配置，在 `env.py` 的 `_get_observations()` 中将图像数据拼接到观测向量中（或使用 CNN 编码器处理）。
