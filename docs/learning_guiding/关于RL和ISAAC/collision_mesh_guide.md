# Isaac Sim 碰撞体（Collision Mesh）完全指南

本文档详细解释物理仿真中的碰撞体概念，以及如何在 Isaac Sim 中正确配置碰撞体以实现叉车插入托盘等复杂交互。

---

## 1. 核心概念：Visual Mesh vs Collision Mesh

在物理仿真中，每个物体实际上有**两套几何体**：

### 1.1 Visual Mesh（视觉网格）

- **用途**：渲染显示，让人眼看到物体的外观
- **精度**：可以非常精细，包含复杂的凹凸、空洞、细节
- **性能**：主要消耗 GPU 渲染资源
- **USD 属性**：`UsdGeom.Mesh`

### 1.2 Collision Mesh（碰撞体/碰撞网格）

- **用途**：物理碰撞检测，决定物体之间如何碰撞
- **精度**：通常简化，以提高物理计算效率
- **性能**：消耗 CPU/PhysX 计算资源
- **USD 属性**：`UsdPhysics.CollisionAPI` + 碰撞几何体

### 1.3 两者的关系

```
┌────────────────────────────────────────────────────────────────┐
│                     同一个物体（如托盘）                        │
│                                                                │
│   Visual Mesh (渲染用)              Collision Mesh (物理用)    │
│   ┌─────┬───┬─────┐                ┌─────────────────┐         │
│   │     │   │     │                │█████████████████│         │
│   │     │ ← 空洞  │                │█████████████████│         │
│   │     │   │     │                │█████████████████│         │
│   │     │   │     │                │█████████████████│         │
│   └─────┴───┴─────┘                └─────────────────┘         │
│         ↑                                  ↑                   │
│   人眼看到的：有 pocket              PhysX 看到的：实心         │
│   可以插入货叉                       货叉无法进入               │
└────────────────────────────────────────────────────────────────┘
```

**关键理解**：你在 Isaac Sim 视口中"看到"的是 Visual Mesh，但物理引擎"感知"的是 Collision Mesh。两者可能完全不同！

---

## 2. 凸包（Convex Hull）与凹形碰撞

### 2.1 什么是凸包？

**凸包（Convex Hull）** 是能够完全包围一组点的最小凸多面体。

直观理解：想象用一根橡皮筋套住物体的所有顶点，橡皮筋收紧后的形状就是凸包。

```
原始形状（凹形，有空洞）：           凸包（凸形，空洞被填满）：

    ████████                            ████████████
    █      █                            ████████████
    █  ██  █          ───────>          ████████████
    █  ██  █          凸包化             ████████████
    █      █                            ████████████
    ████████                            ████████████
```

### 2.2 为什么 PhysX 默认使用凸包？

| 特性 | 凸包碰撞 | 凹形碰撞 |
|------|----------|----------|
| 碰撞检测速度 | O(1) - 极快 | O(n) - 较慢 |
| 数值稳定性 | 非常稳定 | 可能有穿透 |
| 内存占用 | 小 | 大 |
| 支持的物理效果 | 全部 | 部分限制 |

**结论**：凸包是性能和稳定性的最佳折中，所以 PhysX 默认使用凸包。

### 2.3 凸包的问题：填充凹形区域

对于叉车托盘这类有"pocket"（插入孔）的物体，凸包会导致：

```
真实托盘截面（俯视图）：              凸包碰撞体：

  ┌─────┬───┬─────┐                 ┌─────────────────┐
  │板    │空 │板    │                 │█████████████████│
  │     │洞 │     │     ────>       │█████████████████│
  │     │   │     │                 │█████████████████│
  └─────┴───┴─────┘                 └─────████████████┘
        ↑                                  ↑
   货叉可以插入这里                    货叉被"实心墙"挡住
```

**这就是你遇到的问题**：视觉上看到 pocket，但货叉无法插入，因为碰撞体是实心的。

---

## 3. 碰撞体类型详解

PhysX/Isaac Sim 支持多种碰撞体类型：

### 3.1 Primitive Shapes（基本形状）

| 类型 | USD Schema | 特点 | 适用场景 |
|------|------------|------|----------|
| Box | `UsdGeom.Cube` | 最快，轴对齐 | 箱子、地板 |
| Sphere | `UsdGeom.Sphere` | 极快，各向同性 | 球、轮子 |
| Capsule | `UsdGeom.Capsule` | 快，常用于角色 | 机器人关节 |
| Cylinder | `UsdGeom.Cylinder` | 较快 | 圆柱物体 |

### 3.2 Convex Mesh（凸网格）

```python
# USD 中的凸网格碰撞
physics:approximation = "convexHull"
```

- 自动从 visual mesh 生成凸包
- 快速但会填充凹形区域
- 默认选项

### 3.3 Convex Decomposition（凸分解）

```python
# USD 中的凸分解
physics:approximation = "convexDecomposition"
```

- 将凹形物体分解为多个凸体
- 兼顾精度和性能
- **推荐用于托盘等有 pocket 的物体**

```
原始凹形：                  凸分解结果：

  ┌─────┬───┬─────┐        ┌─────┐ ┌─────┐
  │     │   │     │        │凸体1│ │凸体2│
  │     │   │     │   ──>  │     │ │     │
  │     │   │     │        │     │ │     │
  └─────┴───┴─────┘        └─────┘ └─────┘
        ↑                    ↑   ↑   ↑
    一个凹形              分解为多个凸体
                          中间的空隙保留
```

### 3.4 Triangle Mesh（三角网格）

```python
# USD 中的三角网格碰撞
physics:approximation = "meshSimplification"
# 或完全匹配
physics:approximation = "none"
```

- 与 visual mesh 完全一致
- 最精确，但计算开销大
- **仅支持静态物体（kinematic）或作为碰撞目标**
- 动态物体之间的三角网格碰撞可能不稳定

### 3.5 SDF Collision（符号距离场碰撞）

```python
# PhysX 5.x 新特性
physics:approximation = "sdf"
```

- 使用 Signed Distance Field 表示碰撞体
- 支持任意凹形，性能较好
- Isaac Sim 2023.1+ 支持

---

## 4. 在 Isaac Sim 中查看和修改碰撞体

### 4.1 查看当前碰撞体

**方法 1：Isaac Sim GUI**

1. 打开 USD 文件
2. 菜单栏：`Window` → `Physics` → `Debug` → `Show Collision Shapes`
3. 碰撞体会以线框或半透明形式显示

**方法 2：Property 面板**

1. 在 Stage 面板中选择物体
2. 在 Property 面板中查找 `Physics` 部分
3. 查看 `Collision` 相关属性

### 4.2 修改碰撞体类型

**方法 1：Isaac Sim GUI**

1. 选择物体
2. 右键 → `Add` → `Physics` → `Collision`
3. 在 Property 面板中修改 `approximation` 属性

**方法 2：Python 脚本**

```python
from pxr import Usd, UsdPhysics, PhysxSchema

stage = Usd.Stage.Open("path/to/pallet.usd")
prim = stage.GetPrimAtPath("/World/Pallet")

# 添加碰撞 API
collision_api = UsdPhysics.CollisionAPI.Apply(prim)

# 设置碰撞近似类型
mesh_collision_api = UsdPhysics.MeshCollisionAPI.Apply(prim)
mesh_collision_api.GetApproximationAttr().Set("convexDecomposition")

stage.Save()
```

### 4.3 使用 Convex Decomposition

在 Isaac Sim 中启用凸分解：

**GUI 方式**：
1. 选择物体
2. Property 面板 → Physics → Collision
3. 将 `approximation` 设置为 `convexDecomposition`
4. （可选）调整分解参数：
   - `maxConvexHulls`：最大凸体数量
   - `resolution`：分解精度

**Python 方式**：

```python
from pxr import Usd, PhysxSchema

stage = Usd.Stage.Open("path/to/pallet.usd")
prim = stage.GetPrimAtPath("/World/Pallet")

# 应用 PhysX 碰撞 API
physx_collision = PhysxSchema.PhysxCollisionAPI.Apply(prim)

# 设置凸分解参数
physx_collision.GetContactOffsetAttr().Set(0.02)
physx_collision.GetRestOffsetAttr().Set(0.01)

# 设置碰撞近似为凸分解
mesh_collision_api = UsdPhysics.MeshCollisionAPI.Apply(prim)
mesh_collision_api.GetApproximationAttr().Set("convexDecomposition")

# 凸分解参数（可选）
convex_api = PhysxSchema.PhysxConvexDecompositionCollisionAPI.Apply(prim)
convex_api.GetMaxConvexHullsAttr().Set(32)  # 最大凸体数
convex_api.GetHullVertexLimitAttr().Set(64)  # 每个凸体最大顶点数

stage.Save()
```

---

## 5. 托盘碰撞体的解决方案

针对叉车托盘无法插入的问题，有以下解决方案：

### 5.1 方案 A：使用凸分解（推荐）

**优点**：自动化，精度好，性能可接受

**步骤**：

```python
# 修改托盘 USD 的碰撞体
from pxr import Usd, UsdPhysics, PhysxSchema

stage = Usd.Stage.Open("/path/to/pallet.usd")
pallet_prim = stage.GetPrimAtPath("/World/Pallet")

# 设置凸分解
mesh_collision = UsdPhysics.MeshCollisionAPI.Apply(pallet_prim)
mesh_collision.GetApproximationAttr().Set("convexDecomposition")

# 配置凸分解参数
convex_api = PhysxSchema.PhysxConvexDecompositionCollisionAPI.Apply(pallet_prim)
convex_api.GetMaxConvexHullsAttr().Set(16)   # 足够表示 pocket
convex_api.GetHullVertexLimitAttr().Set(32)

stage.Save()
```

### 5.2 方案 B：手动创建多个碰撞体

**优点**：最精确控制，性能最优

**步骤**：

1. 在 Isaac Sim 中打开托盘 USD
2. 删除现有的碰撞体
3. 手动添加多个 Box 碰撞体，避开 pocket 区域：

```
托盘俯视图：

  ┌─────────────────────────┐
  │  ┌───┐       ┌───┐      │
  │  │Box│       │Box│      │  ← 顶板碰撞体
  │  │ 1 │       │ 2 │      │
  │  └───┘       └───┘      │
  │                         │
  │  ┌───────────────────┐  │
  │  │      Box 3        │  │  ← 底板碰撞体
  │  └───────────────────┘  │
  └─────────────────────────┘
```

**Python 方式**：

```python
from pxr import Usd, UsdGeom, UsdPhysics, Gf

stage = Usd.Stage.Open("/path/to/pallet.usd")

# 创建碰撞体容器
collision_root = stage.DefinePrim("/World/Pallet/Collisions", "Xform")

# 左侧板碰撞体
left_box = UsdGeom.Cube.Define(stage, "/World/Pallet/Collisions/LeftBoard")
left_box.GetSizeAttr().Set(0.15)
left_box.AddTranslateOp().Set(Gf.Vec3d(-0.3, 0, 0.05))
UsdPhysics.CollisionAPI.Apply(left_box.GetPrim())

# 右侧板碰撞体
right_box = UsdGeom.Cube.Define(stage, "/World/Pallet/Collisions/RightBoard")
right_box.GetSizeAttr().Set(0.15)
right_box.AddTranslateOp().Set(Gf.Vec3d(0.3, 0, 0.05))
UsdPhysics.CollisionAPI.Apply(right_box.GetPrim())

# 底板碰撞体
bottom_box = UsdGeom.Cube.Define(stage, "/World/Pallet/Collisions/Bottom")
bottom_box.GetSizeAttr().Set(2.0)
bottom_box.AddScaleOp().Set(Gf.Vec3d(1.0, 0.6, 0.02))
bottom_box.AddTranslateOp().Set(Gf.Vec3d(0, 0, 0))
UsdPhysics.CollisionAPI.Apply(bottom_box.GetPrim())

stage.Save()
```

### 5.3 方案 C：使用三角网格碰撞（仅限 kinematic）

**优点**：最精确

**限制**：托盘必须保持 kinematic（固定不动）

```python
from pxr import Usd, UsdPhysics

stage = Usd.Stage.Open("/path/to/pallet.usd")
pallet_prim = stage.GetPrimAtPath("/World/Pallet")

# 设置为三角网格碰撞
mesh_collision = UsdPhysics.MeshCollisionAPI.Apply(pallet_prim)
mesh_collision.GetApproximationAttr().Set("none")  # 完全匹配 visual mesh

# 必须设置为 kinematic
rigid_body = UsdPhysics.RigidBodyAPI.Apply(pallet_prim)
rigid_body.GetKinematicEnabledAttr().Set(True)

stage.Save()
```

---

## 6. IsaacLab 中的碰撞体配置

在 IsaacLab 的 `env_cfg.py` 中，可以通过配置影响碰撞行为：

### 6.1 RigidObjectCfg 碰撞参数

```python
from isaaclab.sim import RigidObjectCfg, CollisionPropertiesCfg

pallet_cfg = RigidObjectCfg(
    prim_path="/World/envs/env_.*/Pallet",
    spawn=sim_utils.UsdFileCfg(
        usd_path="path/to/pallet.usd",
        # 碰撞属性
        collision_props=CollisionPropertiesCfg(
            collision_enabled=True,
            contact_offset=0.02,   # 碰撞检测提前量
            rest_offset=0.01,      # 静止时的偏移
        ),
    ),
)
```

### 6.2 关键参数说明

| 参数 | 说明 | 典型值 |
|------|------|--------|
| `contact_offset` | 碰撞检测的提前距离 | 0.01 - 0.05 |
| `rest_offset` | 物体静止时的最小间隙 | 0.005 - 0.02 |
| `collision_enabled` | 是否启用碰撞 | True/False |

**注意**：`contact_offset` 过大会导致物体看起来"悬浮"，过小可能导致穿透。

---

## 7. 调试碰撞问题的流程

当遇到"物体无法插入/穿透/碰撞异常"时，按以下流程排查：

### 7.1 调试流程图

```
物体无法正确碰撞？
        │
        ▼
┌───────────────────┐
│ 1. 查看碰撞体可视化 │  ← Show Collision Shapes
└───────────────────┘
        │
        ▼
┌───────────────────┐
│ 碰撞体形状正确吗？  │
└───────────────────┘
        │
   ┌────┴────┐
   │         │
   ▼         ▼
 正确      不正确
   │         │
   ▼         ▼
检查物理    修改碰撞体类型
参数       (凸分解/手动)
   │
   ▼
┌───────────────────┐
│ 检查 contact_offset │
│ 检查 rest_offset    │
│ 检查 friction       │
└───────────────────┘
```

### 7.2 常见问题速查表

| 现象 | 可能原因 | 解决方案 |
|------|----------|----------|
| 无法插入 pocket | 凸包填充了空洞 | 使用凸分解或手动碰撞体 |
| 物体穿透 | contact_offset 太小 | 增大 contact_offset |
| 物体悬浮 | rest_offset 太大 | 减小 rest_offset |
| 物体滑动 | 摩擦系数低 | 增大 friction |
| 碰撞抖动 | 物理步长太大 | 减小 dt 或增加 substeps |

---

## 8. 完整示例：修复托盘碰撞体

以下是修复托盘碰撞体的完整脚本：

```python
#!/usr/bin/env python3
"""
修复托盘碰撞体，使货叉能够插入 pocket
"""

from pxr import Usd, UsdGeom, UsdPhysics, PhysxSchema, Gf

def fix_pallet_collision(usd_path: str, output_path: str = None):
    """
    修复托盘碰撞体
    
    Args:
        usd_path: 原始托盘 USD 路径
        output_path: 输出路径（默认覆盖原文件）
    """
    stage = Usd.Stage.Open(usd_path)
    
    # 查找托盘 prim（可能需要根据实际 USD 结构调整路径）
    pallet_prim = None
    for prim in stage.Traverse():
        if "pallet" in prim.GetName().lower():
            pallet_prim = prim
            break
    
    if pallet_prim is None:
        print("未找到托盘 prim")
        return
    
    print(f"找到托盘: {pallet_prim.GetPath()}")
    
    # 方法 1：设置凸分解（推荐）
    mesh_collision = UsdPhysics.MeshCollisionAPI.Apply(pallet_prim)
    mesh_collision.GetApproximationAttr().Set("convexDecomposition")
    
    # 凸分解参数
    if pallet_prim.HasAPI(PhysxSchema.PhysxConvexDecompositionCollisionAPI):
        convex_api = PhysxSchema.PhysxConvexDecompositionCollisionAPI(pallet_prim)
    else:
        convex_api = PhysxSchema.PhysxConvexDecompositionCollisionAPI.Apply(pallet_prim)
    
    convex_api.GetMaxConvexHullsAttr().Set(24)      # 最大 24 个凸体
    convex_api.GetHullVertexLimitAttr().Set(48)    # 每个凸体最多 48 个顶点
    convex_api.GetMinThicknessAttr().Set(0.005)    # 最小厚度
    
    print("已设置凸分解碰撞")
    
    # 保存
    save_path = output_path or usd_path
    stage.Export(save_path)
    print(f"已保存到: {save_path}")


def verify_collision_settings(usd_path: str):
    """
    验证 USD 文件的碰撞体设置
    """
    stage = Usd.Stage.Open(usd_path)
    
    print("=== 碰撞体设置验证 ===\n")
    
    for prim in stage.Traverse():
        # 检查是否有碰撞 API
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            print(f"Prim: {prim.GetPath()}")
            
            # 检查碰撞近似类型
            if prim.HasAPI(UsdPhysics.MeshCollisionAPI):
                mesh_api = UsdPhysics.MeshCollisionAPI(prim)
                approx = mesh_api.GetApproximationAttr().Get()
                print(f"  碰撞近似: {approx}")
            
            # 检查凸分解设置
            if prim.HasAPI(PhysxSchema.PhysxConvexDecompositionCollisionAPI):
                convex_api = PhysxSchema.PhysxConvexDecompositionCollisionAPI(prim)
                print(f"  凸分解 - maxConvexHulls: {convex_api.GetMaxConvexHullsAttr().Get()}")
                print(f"  凸分解 - hullVertexLimit: {convex_api.GetHullVertexLimitAttr().Get()}")
            
            print()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("用法: python fix_pallet_collision.py <pallet.usd> [output.usd]")
        sys.exit(1)
    
    usd_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    # 先验证当前设置
    print("=== 修复前 ===")
    verify_collision_settings(usd_path)
    
    # 修复碰撞体
    fix_pallet_collision(usd_path, output_path)
    
    # 验证修复后
    print("\n=== 修复后 ===")
    verify_collision_settings(output_path or usd_path)
```

---

## 9. 总结：碰撞体配置最佳实践

### 9.1 选择碰撞体类型的决策树

```
物体形状？
    │
    ├─ 简单凸形（箱子、球）──────────> 使用 Primitive Shape 或 Convex Hull
    │
    ├─ 复杂凸形 ────────────────────> 使用 Convex Hull
    │
    └─ 凹形（有空洞/pocket）
            │
            ├─ 动态物体 ────────────> 使用 Convex Decomposition
            │
            └─ 静态物体 ────────────> 使用 Triangle Mesh 或 Convex Decomposition
```

### 9.2 性能 vs 精度权衡

| 碰撞体类型 | 精度 | 性能 | 适用场景 |
|------------|------|------|----------|
| Primitive | ★★☆ | ★★★ | 简单物体 |
| Convex Hull | ★★☆ | ★★★ | 凸形物体 |
| Convex Decomposition | ★★★ | ★★☆ | 凹形动态物体 |
| Triangle Mesh | ★★★ | ★☆☆ | 静态环境 |
| SDF | ★★★ | ★★☆ | 复杂凹形（PhysX 5.x） |

### 9.3 记住这些关键点

1. **Visual ≠ Collision**：你看到的不一定是物理引擎感知的
2. **凸包会填充空洞**：默认的凸包碰撞会填满所有凹形区域
3. **凸分解是折中方案**：对于需要 pocket 的物体，凸分解是最佳选择
4. **三角网格有限制**：仅用于静态物体或作为碰撞目标
5. **调试先看碰撞体**：遇到碰撞问题，第一步是可视化碰撞体

---

## 10. PhysX 碰撞检测流水线与 Cooking 机制

> 本节内容合并自 `physx_collision_deep_dive.md`。

### 10.1 碰撞检测三阶段流水线

物理引擎每一帧的碰撞检测分为三个阶段：

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  Broad Phase │ -> │ Narrow Phase│ -> │  Response   │
│  (粗筛)      │    │  (精确检测)  │    │  (响应计算)  │
└─────────────┘    └─────────────┘    └─────────────┘
     AABB包围盒        几何相交测试       力/位置修正
```

- **Broad Phase**: 使用轴对齐包围盒 (AABB) 快速排除不可能碰撞的物体对
- **Narrow Phase**: 对可能碰撞的物体对进行精确的几何相交测试
- **Response**: 计算碰撞响应（反弹力、摩擦力、位置修正）

Narrow Phase 的计算复杂度直接取决于碰撞体形状：

| 碰撞体类型 | 计算复杂度 | 精度 | 适用场景 |
|-----------|-----------|------|---------|
| 球体 (Sphere) | O(1) | 低 | 球形物体 |
| 胶囊体 (Capsule) | O(1) | 中 | 人物角色 |
| 盒子 (Box) | O(1) | 低 | 箱子、建筑 |
| 凸包 (Convex Hull) | O(n) | 中 | 简单实体 |
| 三角网格 (TriMesh) | O(n²) | 高 | 静态地形 |
| 凸分解 (Convex Decomposition) | O(k×n) | 高 | 复杂动态物体 |

### 10.2 PhysX Cooking 机制

**Cooking** 是 PhysX 将 USD 几何数据转换为高效碰撞数据结构的过程。

```
原始 USD 几何数据          Cooking 过程              PhysX 运行时数据
┌─────────────┐    ┌─────────────────────┐    ┌─────────────────┐
│  顶点列表    │    │  构建 BVH 树        │    │  优化的碰撞结构  │
│  面索引      │ -> │  计算凸包           │ -> │  快速查询索引    │
│  法线        │    │  生成空间哈希       │    │  预计算数据      │
└─────────────┘    └─────────────────────┘    └─────────────────┘
```

**关键：Cooking 只在场景加载时执行一次！**

```python
# 以下代码不会生效！
def _setup_scene(self):
    # 场景已经加载完成，Cooking 已经完成
    mesh_collision_api.GetApproximationAttr().Set("convexDecomposition")
    # 此时修改 USD 属性，PhysX 不会重新 Cook
```

加载时序：
1. Isaac Lab 加载 USD 文件
2. PhysX 读取 USD 中的碰撞属性
3. PhysX 执行 Cooking，生成碰撞数据结构
4. 仿真开始运行
5. **此时再修改 USD 属性，PhysX 已经使用 Cooking 后的数据，不会重新处理**

> **注意**：本项目的 `env.py` 中 `_setup_pallet_physics()` 在 `clone_environments()` **之前**调用，
> 即在 PhysX Cooking 之前修改 USD 属性，因此能够生效。这也是为什么补丁代码的调用顺序至关重要。

### 10.3 最佳实践：预处理 USD 文件

```
预处理阶段（只执行一次）:
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  原始 USD    │ ->  │  修改碰撞属性 │ ->  │  保存新 USD  │
│  (Nucleus)   │     │  添加 API     │     │  (本地)      │
└──────────────┘     └──────────────┘     └──────────────┘

运行阶段（每次训练）:
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  加载新 USD  │ ->  │  PhysX Cook  │ ->  │  正确碰撞    │
│  (本地)      │     │  (凸分解)     │     │  (无穿透)    │
└──────────────┘     └──────────────┘     └──────────────┘
```

### 10.4 USD 碰撞属性层级结构

```
Prim: /Root/Xform/Mesh_015
├── physics:collisionEnabled = true                            (CollisionAPI)
├── physics:approximation = "convexDecomposition"              (MeshCollisionAPI)
├── physxConvexDecompositionCollision:maxConvexHulls = 32      (PhysxConvexDecompositionCollisionAPI)
├── physxConvexDecompositionCollision:hullVertexLimit = 64
├── physxCollision:contactOffset = 0.02                        (PhysxCollisionAPI)
└── physxCollision:restOffset = 0.005
```

---

## 11. 参考资料

- [NVIDIA PhysX 文档 - Collision](https://nvidia-omniverse.github.io/PhysX/physx/5.3.0/docs/Geometry.html)
- [Isaac Sim 文档 - Physics](https://docs.omniverse.nvidia.com/isaacsim/latest/features/physics.html)
- [USD Physics Schema](https://openusd.org/release/api/usd_physics_page_front.html)
