# PhysX 碰撞体原理深度解析：从穿透问题到凸分解

## 问题背景

在 Isaac Sim 叉车仿真中，我们遇到了一个典型问题：**叉车的货叉（fork）在举升托盘时会穿透托盘**。这个问题的根本原因在于物理引擎的碰撞检测机制。

## 1. 物理引擎碰撞检测基础

### 1.1 碰撞检测的核心概念

物理引擎（如 PhysX）在每一帧都需要回答一个问题：**哪些物体正在接触或相交？**

```
碰撞检测流程:
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  Broad Phase │ -> │ Narrow Phase│ -> │  Response   │
│  (粗筛)      │    │  (精确检测)  │    │  (响应计算)  │
└─────────────┘    └─────────────┘    └─────────────┘
     AABB包围盒        几何相交测试       力/位置修正
```

- **Broad Phase**: 使用轴对齐包围盒 (AABB) 快速排除不可能碰撞的物体对
- **Narrow Phase**: 对可能碰撞的物体对进行精确的几何相交测试
- **Response**: 计算碰撞响应（反弹力、摩擦力、位置修正）

### 1.2 为什么碰撞体形状很重要

Narrow Phase 的计算复杂度直接取决于碰撞体的几何形状：

| 碰撞体类型 | 计算复杂度 | 精度 | 适用场景 |
|-----------|-----------|------|---------|
| 球体 (Sphere) | O(1) | 低 | 球形物体 |
| 胶囊体 (Capsule) | O(1) | 中 | 人物角色 |
| 盒子 (Box) | O(1) | 低 | 箱子、建筑 |
| 凸包 (Convex Hull) | O(n) | 中 | 简单实体 |
| 三角网格 (TriMesh) | O(n²) | 高 | 静态地形 |
| 凸分解 (Convex Decomposition) | O(k×n) | 高 | 复杂动态物体 |

## 2. 三角网格 vs 凸包 vs 凸分解

### 2.1 三角网格 (Triangle Mesh)

三角网格直接使用模型的渲染网格作为碰撞体。

```
优点：
- 精度最高，完全匹配视觉外观
- 无需预处理

缺点：
- 计算开销极大
- **关键限制：PhysX 中三角网格只能用于静态物体**
- 动态物体使用三角网格会导致穿透或不稳定
```

**为什么三角网格不能用于动态物体？**

PhysX 的碰撞检测算法对三角网格做了很多优化假设（如空间哈希、BVH 树），这些假设基于物体不会移动。当物体移动时，这些数据结构需要重建，开销巨大且容易产生数值误差。

### 2.2 凸包 (Convex Hull)

凸包是能包围所有顶点的最小凸多面体。

```
        原始形状              凸包
       ___________         ___________
      /     _     \       /           \
     |    _| |_    |  ->  |           |
     |   |_____|   |      |           |
      \___________/        \___________/
      
      （凹形被填平）
```

**凸包的数学特性：**
- 任意两点之间的线段完全在形状内部
- 这个特性使得碰撞检测可以使用高效的 GJK/EPA 算法

```
优点：
- 计算效率高
- 支持动态物体
- 数值稳定

缺点：
- 对凹形物体精度很低
- 托盘的凹槽会被"填平"
```

### 2.3 凸分解 (Convex Decomposition) ⭐

凸分解将一个凹形物体分解为多个凸包的组合。

```
        托盘原始形状                    凸分解后
    ┌───────────────────┐         ┌───┐ ┌───┐ ┌───┐
    │ ┌───┐     ┌───┐   │         │   │ │   │ │   │
    │ │   │     │   │   │   ->    └───┘ └───┘ └───┘
    │ └───┘     └───┘   │         ┌─────────────────┐
    │                   │         │     底板         │
    └───────────────────┘         └─────────────────┘
    
    （4个凹槽被识别并保留）       （多个凸包组合）
```

**工作原理：**

1. **VHACD 算法** (Volumetric Hierarchical Approximate Convex Decomposition)
   - 将体积划分为小的体素 (voxel)
   - 迭代合并体素形成凸区域
   - 直到达到目标凸包数量

2. **关键参数：**
   ```python
   maxConvexHulls = 32      # 最多分解成多少个凸包
   hullVertexLimit = 64     # 每个凸包最多多少顶点
   ```

**为什么凸分解能解决穿透问题？**

```
托盘使用单一凸包时:
                        ┌─────────────────────────┐
    叉车货叉 =====>     │        (实心区域)        │
                        └─────────────────────────┘
    货叉无法进入凹槽，但物理引擎认为整个区域都是实心的
    当货叉强行进入时，产生巨大的穿透力，导致不稳定行为

托盘使用凸分解时:
                        ┌───┐         ┌───┐
    叉车货叉 =====>     │   │  空隙    │   │
                        └───┘         └───┘
                        ┌─────────────────────────┐
                        │         底板             │
                        └─────────────────────────┘
    货叉可以正确进入凹槽，与各个凸包单独进行碰撞检测
```

## 3. PhysX Cooking 机制

### 3.1 什么是 Cooking？

Cooking 是 PhysX 将几何数据转换为高效碰撞数据结构的过程。

```
原始 USD 几何数据          Cooking 过程              PhysX 运行时数据
┌─────────────┐    ┌─────────────────────┐    ┌─────────────────┐
│  顶点列表    │    │  构建 BVH 树        │    │  优化的碰撞结构  │
│  面索引      │ -> │  计算凸包           │ -> │  快速查询索引    │
│  法线        │    │  生成空间哈希       │    │  预计算数据      │
└─────────────┘    └─────────────────────┘    └─────────────────┘
```

### 3.2 Cooking 的时机

**关键问题：Cooking 只在场景加载时执行一次！**

```python
# 这段代码不会生效！
def _setup_scene(self):
    # 场景已经加载完成，Cooking 已经完成
    mesh_collision_api.GetApproximationAttr().Set("convexDecomposition")
    # 此时修改 USD 属性，PhysX 不会重新 Cook
```

这就是为什么运行时修改碰撞属性不生效的原因：
1. Isaac Lab 加载 USD 文件
2. PhysX 读取 USD 中的碰撞属性
3. PhysX 执行 Cooking，生成碰撞数据结构
4. 仿真开始运行
5. **此时再修改 USD 属性，PhysX 已经使用 Cooking 后的数据，不会重新处理**

### 3.3 强制重新 Cooking 的方法

理论上可以调用底层 API 强制重新 Cooking，但这会带来：
- 仿真暂停
- 性能开销
- 状态重置风险

**最佳实践：在加载前预处理 USD 文件**

## 4. 解决方案：预处理 USD 文件

### 4.1 方案设计

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

### 4.2 关键代码解析

```python
from pxr import Usd, UsdGeom, UsdPhysics, PhysxSchema

# 1. 启用碰撞
UsdPhysics.CollisionAPI.Apply(prim)
collision_api = UsdPhysics.CollisionAPI(prim)
collision_api.CreateCollisionEnabledAttr().Set(True)

# 2. 设置碰撞近似方式为凸分解
UsdPhysics.MeshCollisionAPI.Apply(prim)
mesh_collision_api = UsdPhysics.MeshCollisionAPI(prim)
mesh_collision_api.GetApproximationAttr().Set("convexDecomposition")

# 3. 配置凸分解参数
PhysxSchema.PhysxConvexDecompositionCollisionAPI.Apply(prim)
convex_api = PhysxSchema.PhysxConvexDecompositionCollisionAPI(prim)
convex_api.GetMaxConvexHullsAttr().Set(32)       # 最多32个凸包
convex_api.GetHullVertexLimitAttr().Set(64)      # 每个凸包最多64顶点

# 4. 设置接触参数（防止微小穿透）
PhysxSchema.PhysxCollisionAPI.Apply(prim)
physx_collision_api = PhysxSchema.PhysxCollisionAPI(prim)
physx_collision_api.GetContactOffsetAttr().Set(0.02)  # 接触检测扩展
physx_collision_api.GetRestOffsetAttr().Set(0.005)    # 静止时的间隙
```

### 4.3 USD 属性层级结构

```
Prim: /Root/Xform/Mesh_015
├── physics:collisionEnabled = true         (CollisionAPI)
├── physics:approximation = "convexDecomposition"  (MeshCollisionAPI)
├── physxConvexDecompositionCollision:maxConvexHulls = 32
├── physxConvexDecompositionCollision:hullVertexLimit = 64
├── physxCollision:contactOffset = 0.02
└── physxCollision:restOffset = 0.005
```

## 5. 接触参数详解

### 5.1 Contact Offset 和 Rest Offset

```
                    Contact Offset (0.02m)
                    |<---------------->|
    ┌───────────────┐                  ┌───────────────┐
    │    物体 A     │      间隙        │    物体 B     │
    └───────────────┘                  └───────────────┘
                    |<-->|
                    Rest Offset (0.005m)

Contact Offset: 开始计算接触力的距离
Rest Offset:    平衡状态下的最小间隙
```

**作用：**
- 防止数值误差导致的微小穿透
- 提前检测接近的物体，平滑接触响应
- 避免物体"陷入"彼此

### 5.2 参数调优建议

| 场景 | Contact Offset | Rest Offset |
|------|----------------|-------------|
| 精密组装 | 0.001~0.005 | 0.0005~0.002 |
| 一般物体 | 0.01~0.02 | 0.002~0.01 |
| 粗糙堆叠 | 0.02~0.05 | 0.01~0.02 |

## 6. 举一反三：其他应用场景

### 6.1 场景一：机械臂抓取

```python
# 抓取器（Gripper）使用凸分解
# 确保手指能正确接触不规则形状的物体
mesh_collision_api.GetApproximationAttr().Set("convexDecomposition")
convex_api.GetMaxConvexHullsAttr().Set(16)  # 手指形状相对简单
```

### 6.2 场景二：复杂地形

```python
# 地形是静态的，可以使用三角网格获得最高精度
mesh_collision_api.GetApproximationAttr().Set("meshSimplification")
# 或者对于性能敏感的场景
mesh_collision_api.GetApproximationAttr().Set("convexHull")  # 简单凸包
```

### 6.3 场景三：可变形物体

```python
# 软体物理需要特殊处理
# PhysX 4.x+ 支持 Soft Body 和 FEM
# 需要使用不同的碰撞体系
```

### 6.4 常见碰撞近似类型

| 近似类型 | USD 值 | 适用场景 |
|---------|--------|---------|
| 无近似（原始网格） | `"none"` | 静态高精度物体 |
| 凸包 | `"convexHull"` | 简单凸形动态物体 |
| 凸分解 | `"convexDecomposition"` | 复杂凹形动态物体 |
| 包围盒 | `"boundingCube"` | 快速近似 |
| 球体 | `"boundingSphere"` | 滚动物体 |
| 网格简化 | `"meshSimplification"` | 静态地形 |

## 7. 调试技巧

### 7.1 可视化碰撞体

在 Isaac Sim 中：
1. `Window` -> `Physics` -> `Debug`
2. 启用 `Show Collision Shapes`

### 7.2 验证凸分解结果

```python
# 检查 USD 中的碰撞属性
from pxr import Usd, UsdPhysics, PhysxSchema

stage = Usd.Stage.Open("your_file.usd")
for prim in stage.Traverse():
    if prim.HasAPI(UsdPhysics.MeshCollisionAPI):
        api = UsdPhysics.MeshCollisionAPI(prim)
        approx = api.GetApproximationAttr().Get()
        print(f"{prim.GetPath()}: approximation = {approx}")
```

### 7.3 性能监控

```python
# 在 Isaac Sim 中监控物理性能
import carb
settings = carb.settings.get_settings()
settings.set("/physics/debugDraw/simulationStatistics", True)
```

## 8. 总结

### 8.1 关键要点

1. **碰撞体类型选择**：动态凹形物体必须使用凸分解
2. **Cooking 机制**：碰撞数据结构在加载时生成，运行时无法更改
3. **预处理方案**：修改源 USD 文件是最可靠的解决方案
4. **参数调优**：根据场景精度需求调整凸包数量和接触参数

### 8.2 最佳实践清单

- [ ] 动态物体避免使用三角网格碰撞
- [ ] 凹形物体使用凸分解
- [ ] 在仿真开始前完成所有碰撞属性设置
- [ ] 适当设置 Contact Offset 和 Rest Offset
- [ ] 使用可视化工具验证碰撞体形状
- [ ] 监控物理仿真性能

### 8.3 问题排查流程

```
穿透问题排查:
    ↓
1. 检查碰撞是否启用 (CollisionAPI)
    ↓
2. 检查碰撞近似类型 (MeshCollisionAPI.approximation)
    ↓
3. 如果是凹形物体，是否使用凸分解？
    ↓
4. 凸分解参数是否合适？(maxHulls, vertexLimit)
    ↓
5. 接触参数是否合适？(contactOffset, restOffset)
    ↓
6. 修改是否在 Cooking 之前生效？
```

---

## 附录：本次修复的具体改动

**原始文件**: `omniverse://...Isaac/Props/Pallet/pallet.usd`

**修复后文件**: `/home/uniubi/projects/forklift_sim/assets/pallet_convex.usd`

**修改内容**:
```
Mesh: /Root/Xform/Mesh_015
- CollisionAPI: enabled = true
- MeshCollisionAPI: approximation = "convexDecomposition"
- PhysxConvexDecompositionCollisionAPI: maxHulls=32, vertexLimit=64
- PhysxCollisionAPI: contactOffset=0.02, restOffset=0.005
```

**下一步**: 更新 `env_cfg.py` 中的 `pallet_usd_path` 指向新文件。
