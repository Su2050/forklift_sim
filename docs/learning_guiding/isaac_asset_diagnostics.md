# Isaac Sim/Lab 资产诊断与调试指南

本文档介绍如何在 Isaac Sim 和 Isaac Lab 中诊断和调试各类资产（Assets）的配置问题，帮助快速定位仿真中的异常行为。

---

## 目录

1. [核心概念](#1-核心概念)
2. [USD Prim 诊断](#2-usd-prim-诊断)
3. [Articulation 关节体诊断](#3-articulation-关节体诊断)
4. [Rigid Body 刚体诊断](#4-rigid-body-刚体诊断)
5. [Collision 碰撞体诊断](#5-collision-碰撞体诊断)
6. [Joint Drive 关节驱动诊断](#6-joint-drive-关节驱动诊断)
7. [Isaac Lab Actuator 诊断](#7-isaac-lab-actuator-诊断)
8. [常见问题排查清单](#8-常见问题排查清单)

---

## 1. 核心概念

### 1.1 术语对照表

| 术语 | 英文 | 说明 |
|------|------|------|
| 资产 | Asset | 场景中可加载的对象（USD 文件、机器人、物体等） |
| 原语 | Prim | USD 中的基本单元，场景树的节点 |
| 关节体 | Articulation | 由多个刚体通过关节连接的结构（如机器人） |
| 刚体 | Rigid Body | 具有物理属性的不可变形物体 |
| 关节 | Joint | 连接两个刚体的约束（旋转关节、直线关节等） |
| 驱动器 | Drive | 关节的动力系统，提供力/力矩控制 |
| 执行器 | Actuator | Isaac Lab 对驱动器的封装 |

### 1.2 层级关系

```
USD Stage (场景)
└── Prim (原语)
    ├── Articulation (关节体)
    │   ├── RigidBody (刚体 - link)
    │   │   └── CollisionAPI (碰撞体)
    │   └── Joint (关节)
    │       └── DriveAPI (驱动器)
    └── RigidObject (独立刚体)
        └── CollisionAPI (碰撞体)
```

---

## 2. USD Prim 诊断

### 2.1 遍历场景中的所有 Prim

```python
from pxr import Usd, UsdPhysics, UsdGeom

def diagnose_prims(stage, root_path="/World"):
    """遍历指定路径下的所有 Prim 并打印基本信息"""
    root_prim = stage.GetPrimAtPath(root_path)
    if not root_prim.IsValid():
        print(f"[ERROR] 路径不存在: {root_path}")
        return
    
    for prim in Usd.PrimRange(root_prim):
        path = prim.GetPath().pathString
        prim_type = prim.GetTypeName()
        
        # 收集 API 信息
        apis = []
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            apis.append("RigidBodyAPI")
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            apis.append("CollisionAPI")
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            apis.append("ArticulationRootAPI")
        if prim.HasAPI(UsdPhysics.MassAPI):
            apis.append("MassAPI")
        
        print(f"{path}")
        print(f"  Type: {prim_type}")
        if apis:
            print(f"  APIs: {', '.join(apis)}")
        print()

# 使用示例（在 Isaac Lab 环境中）
# diagnose_prims(self.sim.stage, "/World/envs/env_0/Robot")
```

### 2.2 检查特定 Prim 的属性

```python
def get_prim_attributes(stage, prim_path):
    """获取指定 Prim 的所有属性"""
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        print(f"[ERROR] Prim 不存在: {prim_path}")
        return
    
    print(f"Prim: {prim_path}")
    print(f"Type: {prim.GetTypeName()}")
    print(f"APIs: {prim.GetAppliedSchemas()}")
    print("\n属性列表:")
    
    for attr in prim.GetAttributes():
        name = attr.GetName()
        value = attr.Get()
        print(f"  {name}: {value}")
```

---

## 3. Articulation 关节体诊断

### 3.1 检查 Articulation 基本信息

```python
def diagnose_articulation(robot):
    """诊断 Isaac Lab Articulation 对象"""
    print("=" * 60)
    print("[Articulation 诊断]")
    print("=" * 60)
    
    # 基本信息
    print(f"Prim path: {robot.cfg.prim_path}")
    print(f"Num bodies: {robot.num_bodies}")
    print(f"Num joints: {robot.num_joints}")
    print(f"Device: {robot.device}")
    
    # 关节名称
    print(f"\n关节名称:")
    for i, name in enumerate(robot.joint_names):
        print(f"  [{i}] {name}")
    
    # 刚体名称
    print(f"\n刚体名称:")
    for i, name in enumerate(robot.body_names):
        print(f"  [{i}] {name}")
    
    # 关节限制
    if hasattr(robot.data, 'joint_limits'):
        print(f"\n关节限制:")
        limits = robot.data.joint_limits[0]  # 取第一个环境
        for i, name in enumerate(robot.joint_names):
            low = limits[i, 0].item()
            high = limits[i, 1].item()
            print(f"  {name}: [{low:.4f}, {high:.4f}]")
    
    # 默认位置
    if hasattr(robot.data, 'default_joint_pos'):
        print(f"\n默认关节位置:")
        default_pos = robot.data.default_joint_pos[0]
        for i, name in enumerate(robot.joint_names):
            print(f"  {name}: {default_pos[i].item():.4f}")
    
    print("=" * 60)

# 使用示例
# diagnose_articulation(self.robot)
```

### 3.2 查找特定关节

```python
def find_joint_info(robot, joint_name):
    """查找并打印特定关节的详细信息"""
    joint_ids, _ = robot.find_joints([joint_name], preserve_order=True)
    
    if len(joint_ids) == 0:
        print(f"[ERROR] 未找到关节: {joint_name}")
        return None
    
    joint_id = int(joint_ids[0].item()) if hasattr(joint_ids[0], 'item') else int(joint_ids[0])
    
    print(f"关节名称: {joint_name}")
    print(f"关节索引: {joint_id}")
    
    # 当前状态
    pos = robot.data.joint_pos[0, joint_id].item()
    vel = robot.data.joint_vel[0, joint_id].item()
    print(f"当前位置: {pos:.6f}")
    print(f"当前速度: {vel:.6f}")
    
    # 限制
    if hasattr(robot.data, 'joint_limits'):
        limits = robot.data.joint_limits[0, joint_id]
        print(f"位置限制: [{limits[0].item():.4f}, {limits[1].item():.4f}]")
    
    return joint_id
```

---

## 4. Rigid Body 刚体诊断

### 4.1 检查 RigidBodyAPI 配置

```python
from pxr import UsdPhysics, PhysxSchema

def diagnose_rigid_body(stage, prim_path):
    """诊断刚体的物理属性"""
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        print(f"[ERROR] Prim 不存在: {prim_path}")
        return
    
    print(f"刚体诊断: {prim_path}")
    print("-" * 40)
    
    # RigidBodyAPI
    if prim.HasAPI(UsdPhysics.RigidBodyAPI):
        rb_api = UsdPhysics.RigidBodyAPI(prim)
        enabled = rb_api.GetRigidBodyEnabledAttr().Get()
        kinematic = rb_api.GetKinematicEnabledAttr().Get()
        print(f"RigidBodyAPI:")
        print(f"  enabled: {enabled}")
        print(f"  kinematic: {kinematic}")
    else:
        print("[WARN] 无 RigidBodyAPI")
    
    # MassAPI
    if prim.HasAPI(UsdPhysics.MassAPI):
        mass_api = UsdPhysics.MassAPI(prim)
        mass = mass_api.GetMassAttr().Get()
        density = mass_api.GetDensityAttr().Get()
        com = mass_api.GetCenterOfMassAttr().Get()
        print(f"MassAPI:")
        print(f"  mass: {mass}")
        print(f"  density: {density}")
        print(f"  centerOfMass: {com}")
    
    # PhysxRigidBodyAPI (PhysX 特定属性)
    if prim.HasAPI(PhysxSchema.PhysxRigidBodyAPI):
        physx_rb = PhysxSchema.PhysxRigidBodyAPI(prim)
        disable_gravity = physx_rb.GetDisableGravityAttr().Get()
        max_depenetration = physx_rb.GetMaxDepenetrationVelocityAttr().Get()
        print(f"PhysxRigidBodyAPI:")
        print(f"  disableGravity: {disable_gravity}")
        print(f"  maxDepenetrationVelocity: {max_depenetration}")
```

### 4.2 Isaac Lab RigidObject 诊断

```python
def diagnose_rigid_object(rigid_obj):
    """诊断 Isaac Lab RigidObject"""
    print("=" * 60)
    print("[RigidObject 诊断]")
    print("=" * 60)
    
    print(f"Prim path: {rigid_obj.cfg.prim_path}")
    print(f"Num instances: {rigid_obj.num_instances}")
    print(f"Device: {rigid_obj.device}")
    
    # 位置和姿态
    pos = rigid_obj.data.root_pos_w[0].cpu().numpy()
    quat = rigid_obj.data.root_quat_w[0].cpu().numpy()
    print(f"\n世界坐标位置: [{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}]")
    print(f"世界坐标姿态 (quat): [{quat[0]:.4f}, {quat[1]:.4f}, {quat[2]:.4f}, {quat[3]:.4f}]")
    
    # 速度
    lin_vel = rigid_obj.data.root_lin_vel_w[0].cpu().numpy()
    ang_vel = rigid_obj.data.root_ang_vel_w[0].cpu().numpy()
    print(f"线速度: [{lin_vel[0]:.4f}, {lin_vel[1]:.4f}, {lin_vel[2]:.4f}]")
    print(f"角速度: [{ang_vel[0]:.4f}, {ang_vel[1]:.4f}, {ang_vel[2]:.4f}]")
    
    print("=" * 60)
```

---

## 5. Collision 碰撞体诊断

### 5.1 检查碰撞体配置

```python
def diagnose_collision(stage, prim_path):
    """诊断碰撞体配置"""
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        print(f"[ERROR] Prim 不存在: {prim_path}")
        return
    
    print(f"碰撞体诊断: {prim_path}")
    print("-" * 40)
    
    # CollisionAPI
    if prim.HasAPI(UsdPhysics.CollisionAPI):
        collision_api = UsdPhysics.CollisionAPI(prim)
        enabled = collision_api.GetCollisionEnabledAttr().Get()
        print(f"CollisionAPI:")
        print(f"  enabled: {enabled}")
    else:
        print("[WARN] 无 CollisionAPI")
        return
    
    # MeshCollisionAPI (网格碰撞)
    if prim.HasAPI(UsdPhysics.MeshCollisionAPI):
        mesh_api = UsdPhysics.MeshCollisionAPI(prim)
        approx = mesh_api.GetApproximationAttr().Get()
        print(f"MeshCollisionAPI:")
        print(f"  approximation: {approx}")
        # 常见值:
        # - "none": 使用原始网格（高精度，低性能）
        # - "convexHull": 单个凸包（中等精度）
        # - "convexDecomposition": 凸分解（高精度，推荐用于复杂形状）
        # - "boundingCube": 包围盒
        # - "boundingSphere": 包围球
```

### 5.2 遍历所有碰撞体

```python
def find_all_collisions(stage, root_path):
    """查找指定路径下的所有碰撞体"""
    root_prim = stage.GetPrimAtPath(root_path)
    if not root_prim.IsValid():
        return []
    
    collisions = []
    for prim in Usd.PrimRange(root_prim):
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            path = prim.GetPath().pathString
            prim_type = prim.GetTypeName()
            
            approx = None
            if prim.HasAPI(UsdPhysics.MeshCollisionAPI):
                mesh_api = UsdPhysics.MeshCollisionAPI(prim)
                approx = mesh_api.GetApproximationAttr().Get()
            
            collisions.append({
                "path": path,
                "type": prim_type,
                "approximation": approx
            })
            print(f"[Collision] {path} (type={prim_type}, approx={approx})")
    
    return collisions
```

---

## 6. Joint Drive 关节驱动诊断

### 6.1 检查 DriveAPI 配置

这是诊断关节控制问题的**关键步骤**。

```python
from pxr import UsdPhysics, PhysxSchema

def diagnose_joint_drive(stage, joint_path):
    """诊断关节驱动器配置"""
    prim = stage.GetPrimAtPath(joint_path)
    if not prim.IsValid():
        print(f"[ERROR] 关节不存在: {joint_path}")
        return
    
    print(f"关节驱动诊断: {joint_path}")
    print("-" * 40)
    
    # 确定关节类型
    joint_type = None
    drive_name = None
    
    if prim.IsA(UsdPhysics.RevoluteJoint):
        joint_type = "Revolute (旋转)"
        drive_name = "angular"
        joint = UsdPhysics.RevoluteJoint(prim)
        axis = joint.GetAxisAttr().Get()
        lower = joint.GetLowerLimitAttr().Get()
        upper = joint.GetUpperLimitAttr().Get()
        print(f"关节类型: {joint_type}")
        print(f"旋转轴: {axis}")
        print(f"角度限制: [{lower}°, {upper}°]")
        
    elif prim.IsA(UsdPhysics.PrismaticJoint):
        joint_type = "Prismatic (直线)"
        drive_name = "linear"
        joint = UsdPhysics.PrismaticJoint(prim)
        axis = joint.GetAxisAttr().Get()
        lower = joint.GetLowerLimitAttr().Get()
        upper = joint.GetUpperLimitAttr().Get()
        print(f"关节类型: {joint_type}")
        print(f"移动轴: {axis}")
        print(f"位置限制: [{lower}m, {upper}m]")
    else:
        print(f"[WARN] 未知关节类型: {prim.GetTypeName()}")
        return
    
    # 检查 DriveAPI
    drive_api = UsdPhysics.DriveAPI.Get(prim, drive_name)
    if drive_api:
        drive_type = drive_api.GetTypeAttr().Get() if drive_api.GetTypeAttr() else "未设置"
        stiffness = drive_api.GetStiffnessAttr().Get() if drive_api.GetStiffnessAttr() else "未设置"
        damping = drive_api.GetDampingAttr().Get() if drive_api.GetDampingAttr() else "未设置"
        max_force = drive_api.GetMaxForceAttr().Get() if drive_api.GetMaxForceAttr() else "未设置"
        target_pos = drive_api.GetTargetPositionAttr().Get() if drive_api.GetTargetPositionAttr() else "未设置"
        target_vel = drive_api.GetTargetVelocityAttr().Get() if drive_api.GetTargetVelocityAttr() else "未设置"
        
        print(f"\nDriveAPI ({drive_name}):")
        print(f"  type: {drive_type}")
        print(f"  stiffness: {stiffness}")
        print(f"  damping: {damping}")
        print(f"  maxForce: {max_force}")
        print(f"  targetPosition: {target_pos}")
        print(f"  targetVelocity: {target_vel}")
        
        # 诊断建议
        print(f"\n诊断建议:")
        if stiffness == 0 and damping == 0:
            print("  [WARN] stiffness 和 damping 都为 0，驱动器无法工作！")
        elif stiffness == 0:
            print("  [INFO] stiffness=0，关节使用速度控制模式")
        else:
            print("  [INFO] stiffness>0，关节使用位置控制模式")
        
        if max_force == 0:
            print("  [WARN] maxForce=0，驱动器无法输出力！")
    else:
        print(f"\n[WARN] 无 DriveAPI ({drive_name})，关节无法被控制！")
        print("  建议: 使用 UsdPhysics.DriveAPI.Apply(prim, drive_name) 添加驱动器")
    
    # PhysxJointAPI
    if prim.HasAPI(PhysxSchema.PhysxJointAPI):
        physx_joint = PhysxSchema.PhysxJointAPI(prim)
        print(f"\nPhysxJointAPI: 存在")
    else:
        print(f"\nPhysxJointAPI: 不存在")
```

### 6.2 动态添加/修改 DriveAPI

```python
def setup_joint_drive(stage, joint_path, joint_type="linear", 
                      stiffness=5000.0, damping=1000.0, max_force=10000.0):
    """为关节添加或修改驱动器配置
    
    Args:
        joint_type: "linear" (直线关节) 或 "angular" (旋转关节)
    """
    prim = stage.GetPrimAtPath(joint_path)
    if not prim.IsValid():
        print(f"[ERROR] 关节不存在: {joint_path}")
        return False
    
    # 获取或创建 DriveAPI
    drive_api = UsdPhysics.DriveAPI.Get(prim, joint_type)
    if not drive_api:
        print(f"[INFO] 正在添加 DriveAPI ({joint_type})...")
        drive_api = UsdPhysics.DriveAPI.Apply(prim, joint_type)
    
    # 设置参数
    drive_api.CreateTypeAttr().Set("force")
    drive_api.CreateStiffnessAttr().Set(stiffness)
    drive_api.CreateDampingAttr().Set(damping)
    drive_api.CreateMaxForceAttr().Set(max_force)
    
    print(f"[INFO] DriveAPI 已配置:")
    print(f"  type: force")
    print(f"  stiffness: {stiffness}")
    print(f"  damping: {damping}")
    print(f"  maxForce: {max_force}")
    
    return True
```

---

## 7. Isaac Lab Actuator 诊断

### 7.1 检查 Actuator 配置

```python
def diagnose_actuators(robot):
    """诊断 Isaac Lab Articulation 的所有执行器"""
    print("=" * 60)
    print("[Actuator 诊断]")
    print("=" * 60)
    
    if not hasattr(robot, 'actuators'):
        print("[ERROR] robot.actuators 不存在")
        return
    
    for name, actuator in robot.actuators.items():
        print(f"\n执行器: {name}")
        print(f"  类型: {type(actuator).__name__}")
        print(f"  关节索引: {actuator.joint_indices}")
        print(f"  关节名称: {actuator.joint_names}")
        
        # 控制参数
        if hasattr(actuator, 'stiffness'):
            print(f"  stiffness: {actuator.stiffness}")
        if hasattr(actuator, 'damping'):
            print(f"  damping: {actuator.damping}")
        if hasattr(actuator, 'effort_limit'):
            print(f"  effort_limit: {actuator.effort_limit}")
        if hasattr(actuator, 'velocity_limit'):
            print(f"  velocity_limit: {actuator.velocity_limit}")
        
        # 诊断建议
        if hasattr(actuator, 'stiffness'):
            stiffness_val = actuator.stiffness[0, 0].item() if actuator.stiffness.numel() > 0 else 0
            if stiffness_val == 0:
                print(f"  [WARN] stiffness=0，执行器使用速度控制模式")
            else:
                print(f"  [INFO] stiffness>0，执行器使用位置控制模式")
    
    print("=" * 60)

# 使用示例
# diagnose_actuators(self.robot)
```

### 7.2 检查 Actuator 与 USD DriveAPI 的一致性

```python
def compare_actuator_and_drive(robot, actuator_name, stage):
    """比较 Isaac Lab Actuator 配置与 USD DriveAPI 配置"""
    print(f"\n比较 Actuator '{actuator_name}' 与 USD DriveAPI")
    print("-" * 40)
    
    if actuator_name not in robot.actuators:
        print(f"[ERROR] 执行器不存在: {actuator_name}")
        return
    
    actuator = robot.actuators[actuator_name]
    
    # Isaac Lab 配置
    lab_stiffness = actuator.stiffness[0, 0].item() if actuator.stiffness.numel() > 0 else 0
    lab_damping = actuator.damping[0, 0].item() if actuator.damping.numel() > 0 else 0
    print(f"Isaac Lab Actuator:")
    print(f"  stiffness: {lab_stiffness}")
    print(f"  damping: {lab_damping}")
    
    # 获取对应的 USD 关节路径
    joint_name = actuator.joint_names[0]
    # 需要找到完整路径（这里假设路径格式）
    joint_path = f"{robot.cfg.prim_path.replace('env_.*', 'env_0')}/{joint_name}"
    
    prim = stage.GetPrimAtPath(joint_path)
    if prim.IsValid():
        # 确定 drive 类型
        drive_name = "linear" if prim.IsA(UsdPhysics.PrismaticJoint) else "angular"
        drive_api = UsdPhysics.DriveAPI.Get(prim, drive_name)
        
        if drive_api:
            usd_stiffness = drive_api.GetStiffnessAttr().Get() if drive_api.GetStiffnessAttr() else "N/A"
            usd_damping = drive_api.GetDampingAttr().Get() if drive_api.GetDampingAttr() else "N/A"
            print(f"\nUSD DriveAPI:")
            print(f"  stiffness: {usd_stiffness}")
            print(f"  damping: {usd_damping}")
            
            # 一致性检查
            if lab_stiffness != usd_stiffness or lab_damping != usd_damping:
                print(f"\n[WARN] 配置不一致！Isaac Lab 可能会覆盖 USD 设置。")
        else:
            print(f"\n[WARN] USD 中没有 DriveAPI")
    else:
        print(f"\n[WARN] 未找到 USD 关节: {joint_path}")
```

---

## 8. 常见问题排查清单

### 8.1 关节不移动

检查步骤：

1. **确认关节存在**
   ```python
   joint_ids, _ = robot.find_joints(["joint_name"])
   print(f"关节索引: {joint_ids}")  # 应该不为空
   ```

2. **确认 DriveAPI 存在且配置正确**
   ```python
   diagnose_joint_drive(stage, joint_path)
   # stiffness > 0 或 damping > 0
   # maxForce > 0
   ```

3. **确认 Actuator 配置**
   ```python
   diagnose_actuators(robot)
   # 检查对应 actuator 的 stiffness、damping
   ```

4. **确认控制命令格式**
   ```python
   # 检查 joint_ids 类型（应为整数列表）
   print(f"joint_ids type: {type(joint_ids[0])}")
   # 检查目标张量形状
   print(f"target shape: {target.shape}")  # 应为 (N, num_joints)
   ```

### 8.2 物体穿透

检查步骤：

1. **确认 CollisionAPI 存在**
   ```python
   find_all_collisions(stage, prim_path)
   ```

2. **确认碰撞近似类型**
   - 静态物体可用 `none`（原始网格）
   - 动态物体必须用 `convexHull` 或 `convexDecomposition`

3. **确认 RigidBodyAPI 配置**
   ```python
   diagnose_rigid_body(stage, prim_path)
   # kinematic 应为 False（动态物体）
   ```

### 8.3 物体不受重力影响

检查步骤：

1. **确认 RigidBodyAPI enabled**
   ```python
   # enabled 应为 True
   # kinematic 应为 False
   ```

2. **确认 disableGravity**
   ```python
   # PhysxRigidBodyAPI.disableGravity 应为 False
   ```

### 8.4 调试日志模板

```python
def full_scene_diagnostic(env):
    """完整场景诊断"""
    print("\n" + "=" * 80)
    print("完整场景诊断报告")
    print("=" * 80)
    
    # 1. Articulation 诊断
    if hasattr(env, 'robot'):
        diagnose_articulation(env.robot)
        diagnose_actuators(env.robot)
    
    # 2. Rigid Object 诊断
    if hasattr(env, 'pallet'):
        diagnose_rigid_object(env.pallet)
    
    # 3. 关键关节驱动诊断
    if hasattr(env, '_lift_id'):
        joint_path = f"{env.robot.cfg.prim_path.replace('env_.*', 'env_0')}/lift_joint"
        diagnose_joint_drive(env.sim.stage, joint_path)
    
    # 4. 碰撞体诊断
    find_all_collisions(env.sim.stage, "/World/envs/env_0")
    
    print("\n" + "=" * 80)
    print("诊断完成")
    print("=" * 80)
```

---

## 附录：快速诊断代码片段

### A.1 在环境初始化时添加诊断

```python
# 在 env.py 的 __init__ 方法末尾添加
def __init__(self, cfg, render_mode=None, **kwargs):
    super().__init__(cfg, render_mode, **kwargs)
    
    # ... 现有代码 ...
    
    # 添加诊断
    if os.environ.get("ISAAC_DIAGNOSTIC", "0") == "1":
        full_scene_diagnostic(self)
```

运行时启用诊断：
```bash
ISAAC_DIAGNOSTIC=1 ./isaaclab.sh -p script.py
```

### A.2 保存诊断结果到文件

```python
import sys
from contextlib import redirect_stdout

def save_diagnostic_to_file(env, filepath):
    """将诊断结果保存到文件"""
    with open(filepath, 'w') as f:
        with redirect_stdout(f):
            full_scene_diagnostic(env)
    print(f"诊断结果已保存到: {filepath}")
```

---

## 9. 常见错误与经验教训

### 9.1 PhysX ArticulationView API 方法名混淆

**错误信息**：
```
AttributeError: 'ArticulationView' object has no attribute 'set_joint_position_targets'. 
Did you mean: 'set_dof_position_targets'?
```

**原因**：
- Isaac Lab 的高层 API 使用 `joint` 术语（如 `set_joint_position_target`）
- PhysX 的底层 `ArticulationView` 使用 `dof`（Degree of Freedom）术语
- 两者命名不一致，容易混淆

**正确用法**：

| Isaac Lab 高层 API | PhysX ArticulationView 底层 API |
|-------------------|--------------------------------|
| `robot.set_joint_position_target()` | `robot.root_physx_view.set_dof_position_targets()` |
| `robot.set_joint_velocity_target()` | `robot.root_physx_view.set_dof_velocity_targets()` |
| `robot.set_joint_effort_target()` | `robot.root_physx_view.set_dof_actuation_forces()` |
| `robot.data.joint_pos` | `robot.root_physx_view.get_dof_positions()` |
| `robot.data.joint_vel` | `robot.root_physx_view.get_dof_velocities()` |

**经验教训**：
- 使用底层 PhysX API 时，注意术语是 `dof` 不是 `joint`
- 遇到 `AttributeError` 时，注意 Python 的提示 "Did you mean: ..."
- 优先使用 Isaac Lab 的高层 API，除非需要绕过其封装

---

### 9.2 Isaac Lab Actuator 与 USD DriveAPI 配置不一致

**现象**：
- `set_joint_position_target` 调用成功，但关节不移动
- USD 中的 DriveAPI 配置与 Isaac Lab Actuator 配置不同

**诊断方法**：
```python
# 检查 USD 中的 DriveAPI 配置
drive_api = UsdPhysics.DriveAPI.Get(prim, "linear")  # 或 "angular"
print(f"USD stiffness: {drive_api.GetStiffnessAttr().Get()}")
print(f"USD damping: {drive_api.GetDampingAttr().Get()}")

# 检查 Isaac Lab Actuator 配置
print(f"Actuator stiffness: {robot.actuators['lift'].stiffness}")
print(f"Actuator damping: {robot.actuators['lift'].damping}")
```

**可能原因**：
1. Isaac Lab 可能用 Actuator 配置覆盖 USD 配置，但覆盖可能不完整
2. PhysX drive 没有被正确激活
3. 关节类型与 drive 类型不匹配（`linear` vs `angular`）

**解决方向**：
- 确保 USD 中的 DriveAPI 和 Isaac Lab Actuator 配置一致
- 尝试直接使用 PhysX API 绕过 Isaac Lab 封装
- 检查关节类型（Prismatic 用 `linear`，Revolute 用 `angular`）

---

### 9.3 PhysX API `indices` 参数不能传 `None`

**错误信息**：
```
File "...omni/physics/tensors/impl/frontend_torch.py", line 88, in as_contiguous_uint32
    return tensor.to(torch.int32).contiguous()
           ^^^^^^^^^
AttributeError: 'NoneType' object has no attribute 'to'
```

**原因**：
- 本版本的 `set_dof_position_targets` 必须显式传入 `indices`
- 传 `None` 会触发 `NoneType` 错误；省略参数会触发“缺少必需参数”错误

**错误用法**：
```python
# ❌ 错误：显式传递 None
self.robot.root_physx_view.set_dof_position_targets(full_targets, indices=None)
```

**正确用法**：
```python
# ✅ 正确：传递有效的索引 tensor（通常是环境索引）
env_indices = torch.arange(full_targets.shape[0], device=full_targets.device, dtype=torch.int32)
self.robot.root_physx_view.set_dof_position_targets(full_targets, env_indices)
```

**经验教训**：
- 该 API 在不同版本里参数要求可能不同，必须以运行时错误为准
- 直接构造 `env_indices` 最稳妥

---

### 9.4 Prismatic Joint（直线关节）控制注意事项

**特殊性**：
- Prismatic joint 的 drive 类型是 `"linear"`，不是 `"angular"`
- 位置单位是米（m），不是弧度
- 速度单位是米/秒（m/s），不是弧度/秒

**正确的 DriveAPI 设置**：
```python
# 对于 Prismatic joint
drive_api = UsdPhysics.DriveAPI.Apply(prim, "linear")  # 注意是 "linear"
drive_api.CreateTypeAttr().Set("force")
drive_api.CreateStiffnessAttr().Set(5000.0)  # N/m
drive_api.CreateDampingAttr().Set(1000.0)    # N·s/m
drive_api.CreateMaxForceAttr().Set(10000.0)  # N
```

**常见错误**：
- 对 Prismatic joint 使用 `"angular"` drive → 不会工作
- 混淆力（N）和力矩（N·m）单位

---

### 9.5 🔴 【重大教训】USD 配置 ≠ PhysX 运行时状态

> 这是一次耗费 30+ 次调试才发现的问题，值得重点记录。

**误导性的 USD 日志**：
```
MassAPI mass=0.0, density=3000.0
```
看到 `mass=0.0`，直觉认为"质量为 0，所以推不动"。

**实际的 PhysX 运行时状态**：
```python
masses = self.robot.root_physx_view.get_masses()
# tensor([[6287, 269, 269, 2.98, 2.98, 112, 135, 135]])
# lift = 112 kg，根本不是 0！
```

**为什么会这样？**
- USD 中 `mass=0.0` 配合 `density>0` 表示："让 PhysX 自动计算质量"
- PhysX 会根据 `density × 碰撞体体积` 计算出实际 mass
- USD 配置是"输入"，PhysX 运行时是"输出"，两者不同！

**正确的诊断方法**：
```python
# ❌ 错误：只看 USD 配置
mass_api = UsdPhysics.MassAPI(prim)
mass = mass_api.GetMassAttr().Get()  # 可能是 0.0

# ✅ 正确：查询 PhysX 运行时状态
masses = robot.root_physx_view.get_masses()
print(f"PhysX 实际质量: {masses}")  # 这才是真正的值
```

**经验教训**：
- **永远用 API 查运行时状态，不要只看配置文件**
- USD 配置只是"意图"，PhysX 运行时才是"事实"
- 这类"配置 vs 运行时"的差异在物理引擎中很常见

---

### 9.6 🔴 【重大教训】力不够大也会导致"不动"

**经典误诊路径**：
```
关节不动 → 检查 DriveAPI ✓ → 检查 MassAPI → mass=0? → 结论：质量为 0
```

**正确诊断路径**：
```
关节不动 → 查 PhysX 实际 mass → mass=112kg ✓ → 做力学分析 → 发现力不够！
```

**力学分析示例（lift_joint）**：
```python
# 已知
lift_mass = 112  # kg（PhysX 查询得到）
gravity = 9.8    # m/s²
stiffness = 5000 # N/m（Actuator 配置）
target_pos = 0.01667  # m（目标位置）
current_pos = 0.0     # m（当前位置）

# 计算
F_gravity = lift_mass * gravity  # = 1098 N（重力）
F_drive = stiffness * (target_pos - current_pos)  # = 83 N（drive 力）

# 判断
print(f"重力: {F_gravity}N, Drive 力: {F_drive}N")
# 83 N << 1098 N → 力不够，抬不起来！
```

**解决方案**：
```python
# 计算需要的最小 stiffness
# stiffness × Δx > F_gravity
# stiffness > 1098 / 0.017 ≈ 64600

# 设置足够大的 stiffness（留余量）
"lift": ImplicitActuatorCfg(
    joint_names_expr=["lift_joint"],
    stiffness=200000.0,  # 从 5000 增加到 200000
    damping=10000.0,
    effort_limit_sim=50000.0,
)
```

**诊断清单（关节不动时）**：

```markdown
[ ] 1. 查 PhysX 实际 masses: root_physx_view.get_masses()
[ ] 2. 计算外力（重力 = m × g，摩擦力等）
[ ] 3. 计算 drive 力（stiffness × position_error）
[ ] 4. 比较：drive 力 > 外力？
[ ] 5. 如果不够，调整 stiffness / effort_limit
```

**经验教训**：
- **"有力不代表力够大"**
- 遇到"不动"时，先做简单的力平衡计算
- Prismatic joint（直线关节）抬升时必须克服重力
- stiffness 参数的物理意义：每米位移产生的力（N/m）

---

### 9.7 stiffness/damping 参数的物理意义与计算

**Position-based Drive（PD 控制器）模型**：
```
F = stiffness × (target_pos - current_pos) + damping × (target_vel - current_vel)
```

**参数物理意义**：

| 参数 | 单位 | 物理意义 |
|------|------|----------|
| stiffness | N/m（直线）或 N·m/rad（旋转） | 弹簧刚度，每单位位移产生的力 |
| damping | N·s/m（直线）或 N·m·s/rad（旋转） | 阻尼系数，抑制振荡 |
| effort_limit | N（直线）或 N·m（旋转） | 最大输出力/力矩 |

**计算 stiffness 的经验公式**：

对于需要克服重力的关节：
```python
# 最小 stiffness（理论值）
min_stiffness = (mass * gravity) / max_displacement

# 推荐 stiffness（留 3-5 倍余量）
recommended_stiffness = min_stiffness * 4
```

对于水平运动的关节（不需要克服重力）：
```python
# 可以使用较小的 stiffness
stiffness = 5000  # 足够跟踪目标即可
```

**damping 的经验值**：
```python
# 临界阻尼（无振荡）
damping = 2 * sqrt(stiffness * effective_mass)

# 实际应用中，通常取 stiffness 的 5-10%
damping = stiffness * 0.05
```

---

### 9.8 常见"力不够"场景汇总

| 场景 | 需要克服的力 | 计算公式 |
|------|-------------|----------|
| 垂直举升 | 重力 | F > m × g |
| 水平推动 | 摩擦力 | F > μ × m × g |
| 加速运动 | 惯性力 | F > m × a |
| 旋转启动 | 转动惯量 | τ > I × α |

**示例：托盘举升**：
```python
# 托盘 + 货物质量
total_mass = 500  # kg
gravity = 9.8     # m/s²
safety_factor = 2  # 安全系数

# 需要的力
required_force = total_mass * gravity * safety_factor  # = 9800 N

# 设置 effort_limit
effort_limit_sim = 50000.0  # 远大于 required_force
```

---

## 10. 完整诊断流程图（关节不动问题）

```
┌─────────────────────────────────────┐
│  关节不动                           │
└─────────────────┬───────────────────┘
                  │
                  ▼
┌─────────────────────────────────────┐
│ 1. 检查控制命令是否到达             │
│    - joint_ids 正确？               │
│    - target 形状/数值正确？         │
└─────────────────┬───────────────────┘
                  │ ✓
                  ▼
┌─────────────────────────────────────┐
│ 2. 检查 DriveAPI 是否存在           │
│    - 直线关节用 "linear"            │
│    - 旋转关节用 "angular"           │
└─────────────────┬───────────────────┘
                  │ ✓
                  ▼
┌─────────────────────────────────────┐
│ 3. 查询 PhysX 运行时 mass           │
│    root_physx_view.get_masses()     │
│    ⚠️ 不要只看 USD 配置！           │
└─────────────────┬───────────────────┘
                  │
                  ▼
┌─────────────────────────────────────┐
│ 4. 做力学分析                       │
│    - 计算重力/摩擦力等外力          │
│    - 计算 drive 产生的力            │
│    - 比较：drive 力 > 外力？        │
└─────────────────┬───────────────────┘
                  │
        ┌─────────┴─────────┐
        │                   │
        ▼ 是                ▼ 否
┌───────────────┐   ┌───────────────┐
│ 检查其他原因  │   │ 增加 stiffness│
│ - kinematic?  │   │ 增加 effort   │
│ - 碰撞卡住?   │   │ limit         │
│ - limits?     │   └───────────────┘
└───────────────┘
```

---

## 参考资料

- [Isaac Sim Physics Documentation](https://docs.omniverse.nvidia.com/isaacsim/latest/features/physics/physics_simulation.html)
- [Isaac Lab Articulation API](https://isaac-sim.github.io/IsaacLab/main/api/lab/isaaclab.assets.html)
- [USD Physics Schema](https://openusd.org/docs/api/usd_physics_page_front.html)
- [PhysX Documentation](https://nvidia-omniverse.github.io/PhysX/physx/5.3.0/docs/)
