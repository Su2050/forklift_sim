# 托盘物理仿真优化报告

**日期**: 2026-02-05  
**状态**: 部分完成

---

## 1. 问题背景

在叉车仿真环境中，托盘存在以下物理问题：

1. **托盘无法被推动**：托盘被设置为 `kinematic=True`（运动学物体），物理引擎不会计算其动力学
2. **货叉举升时穿透托盘**：碰撞检测未能正确阻止货叉穿过托盘

---

## 2. 已完成的修复

### 2.1 托盘动态化（已完成 ✅）

**问题**：托盘原本是 `kinematic=True`，无法被推动

**修复内容**：

修改 `env_cfg.py` 中的 `pallet_cfg`：

```python
rigid_props=sim_utils.RigidBodyPropertiesCfg(
    rigid_body_enabled=True,
    kinematic_enabled=False,  # 改为 False，使托盘成为动态物体
    disable_gravity=False,    # 改为 False，受重力影响
    max_depenetration_velocity=1.0,
),
mass_props=sim_utils.MassPropertiesCfg(mass=30.0),  # 减轻质量便于推动
```

**修复位置**：
- `IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/forklift_pallet_insert_lift/env_cfg.py`

**效果**：
- ✅ 托盘现在可以被叉车推动
- ✅ 托盘受重力影响，会落到地面

### 2.2 举升执行器参数调整（已完成 ✅）

**问题**：举升力不足，无法抬起托盘

**修复内容**：

修改 `env_cfg.py` 中的 `lift` 执行器配置：

```python
"lift": ImplicitActuatorCfg(
    joint_names_expr=["lift_joint"],
    velocity_limit=1.0,
    effort_limit=5000.0,   # 从 500.0 增加到 5000.0
    stiffness=0.0,         # 从 2000.0 改为 0.0（速度控制模式）
    damping=1000.0,        # 从 200.0 增加到 1000.0
),
```

**关键参数说明**：
- `stiffness=0.0`：禁用位置控制，改为纯速度控制模式
- `effort_limit=5000.0`：提供足够的力矩来抬升托盘
- `damping=1000.0`：增加阻尼以获得更稳定的速度响应

### 2.3 运行时 RigidBody 属性强制设置（已完成 ✅）

**问题**：Isaac Sim 的 `pallet.usd` 是纯视觉资产，没有 `RigidBodyAPI`

**修复内容**：

在 `env.py` 中添加 `_force_pallet_rigid_body()` 函数，在环境克隆前强制应用物理属性：

```python
def _force_pallet_rigid_body(stage, *, rigid_body_enabled, kinematic_enabled, ...):
    """为托盘 prim 添加 RigidBodyAPI"""
    for prim in stage.Traverse():
        if _PALLET_ROOT_RE.match(path):
            UsdPhysics.RigidBodyAPI.Apply(prim)
            PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
            # 设置属性...
```

**修复位置**：
- `IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/forklift_pallet_insert_lift/env.py`

### 2.4 凸分解碰撞体设置（已完成 ✅）

**问题**：托盘默认使用 `boundingCube` 碰撞，货叉无法插入 pocket

**修复内容**：

在 `env.py` 中添加 `_force_pallet_convex_decomposition()` 函数：

```python
def _force_pallet_convex_decomposition(stage, pallet_root_path):
    """设置凸分解碰撞体"""
    for prim in Usd.PrimRange(root):
        if prim.IsA(UsdGeom.Mesh):
            mesh_collision_api = UsdPhysics.MeshCollisionAPI.Apply(prim)
            mesh_collision_api.GetApproximationAttr().Set("convexDecomposition")
            
            convex_api = PhysxSchema.PhysxConvexDecompositionCollisionAPI.Apply(prim)
            convex_api.GetMaxConvexHullsAttr().Set(32)
            convex_api.GetHullVertexLimitAttr().Set(64)
```

**效果**：
- ✅ 货叉可以插入托盘的 pocket

### 2.5 诊断日志功能（已完成 ✅）

**新增功能**：

添加诊断日志函数，将托盘物理属性输出到文件：

```python
_DIAG_LOG_PATH = "/home/uniubi/projects/forklift_sim/docs/logs/pallet_diag.log"

def _log_pallet_usd(stage, pallet_path, label):
    """记录托盘 USD 物理属性"""
    # 输出 RigidBody 和 Collision 状态
    
def _log_pallet_physx(env, label):
    """记录 PhysX 运行时状态"""
```

**日志位置**：`docs/logs/pallet_diag.log`

---

## 3. 当前状态

| 功能 | 状态 | 说明 |
|------|------|------|
| 托盘可被推动 | ✅ 已修复 | `kinematic=False` 生效 |
| 货叉可插入 pocket | ✅ 已修复 | 凸分解碰撞设置正确 |
| 举升力足够 | ✅ 已修复 | `effort_limit=5000` |
| 举升时不穿透 | ❌ 待修复 | 举升时货叉穿过托盘 |

---

## 4. 遗留问题

### 4.1 举升穿透问题（待修复 ⚠️）

**现象**：
- 货叉可以正常插入托盘 pocket
- 托盘可以被推动
- 但是按 Q/E 举升货叉时，货叉会穿过托盘

**可能原因**：
1. **碰撞体 bake 时机**：USD 属性在运行时修改，但 PhysX 可能已经 bake 了碰撞体
2. **碰撞过滤**：叉车和托盘可能在同一个碰撞组中被排除
3. **Contact offset 不足**：碰撞检测的提前量可能太小

**诊断日志显示**：
```
修改后 托盘 USD 状态:
  RigidBody: kinematic=False ✅
  Collision: approx=boundingCube ❌ (期望 convexDecomposition)
```

**根本原因**：运行时修改 USD 属性**无法触发 PhysX 重新烹饪碰撞体**

---

## 5. 相关文件

| 文件 | 说明 |
|------|------|
| `env.py` | 环境主逻辑，包含物理属性设置函数 |
| `env_cfg.py` | 环境配置，包含托盘和执行器参数 |
| `collision_mesh_guide.md` | 碰撞体概念指南 |
| `pallet_diag.log` | 诊断日志输出 |

---

## 6. 参考资料

- [collision_mesh_guide.md](./collision_mesh_guide.md) - 碰撞体类型详解
- [Isaac Sim Physics 文档](https://docs.omniverse.nvidia.com/isaacsim/latest/features/physics.html)

---

**最后更新**: 2026-02-05
