# 托盘碰撞近似方法演变与性能影响分析

> 日期：2026-02-08
> 发现背景：s1.0h 训练速度从 ~17000 steps/s 降至 ~890 steps/s，排查后发现根因为碰撞近似方法变更

## 核心发现

s1.0h 引入 `convexDecomposition` 是本项目首次实现叉车对托盘的**真正物理插入**。此前版本要么托盘无物理属性（s0/s0.6），要么碰撞体被 PhysX 自动回退为单凸包导致插入孔被填平（s1.0e/f/g）。

## 各版本碰撞近似对比

| 版本 | `_setup_pallet_physics` | 碰撞近似方法 | 物理插入能力 | `frac_inserted` | `insert_norm_mean` 峰值 | 训练速度 (steps/s) |
|------|------------------------|-------------|------------|----------------|----------------------|-------------------|
| s0 | 无 | 无（托盘无物理属性） | 叉子穿透 mesh（无碰撞） | 0.0000 | ~0.015 | ~20000 |
| s0.6 | 无 | 无（托盘无物理属性） | 叉子穿透 mesh（无碰撞） | 0.0000 | ~0.06 | ~19000-24000 |
| s1.0e | 有 | PhysX 回退到 `convexHull`（单凸包） | 被实心凸包阻挡 | 0.0000 | ~0.0001 | ~11000-17000 |
| s1.0f | 有 | PhysX 回退到 `convexHull`（单凸包） | 被实心凸包阻挡 | 0.0000 | 0.0000 | ~10000-17000 |
| s1.0g | 有 | PhysX 回退到 `convexHull`（单凸包） | 被实心凸包阻挡 | 0.0000 | ~0.0001 | ~17000 |
| s1.0h (初始) | 有 | `convexDecomposition`(maxConvexHulls=32) | **真正物理插入** | 待训练确认 | 待训练确认 | ~890 |
| s1.0h (优化) | 有 | `convexDecomposition`(maxConvexHulls=8) | **真正物理插入** ✅ | 待训练确认 | 待训练确认 | 待训练确认 |

## 各版本训练指标详细对比

### `frac_lifted`（货叉抬升达标率）与 `lift_delta_mean`

| 版本 | `frac_lifted` 峰值 | `lift_delta_mean` 峰值 | 说明 |
|------|-------------------|----------------------|------|
| s0 | 0.4453 (44%) | 0.1643m | 叉车学会了抬叉，但托盘无物理属性不会跟着动 |
| s0.6 | 0.9170 (92%) | 0.5505m | 同上，抬叉效果很好但托盘不动 |
| s1.0e | 0.1416 (14%) | -0.0394m | 托盘有碰撞体，叉子被挡住无法入位后举升 |
| s1.0f | 0.1191 (12%) | -0.0621m | 同上 |

> **注意**：`frac_lifted` 和 `lift_delta_mean` 测量的是**货叉尖端 z 坐标变化**（`tip[:, 2] - fork_tip_z0`），不是托盘位置。所以 s0/s0.6 中高 `frac_lifted` 仅说明叉车学会了抬叉动作，不代表托盘实际被举起。

### 各版本 `insert_norm_mean` 非零的原因

- **s0/s0.6**（`insert_norm_mean` ~0.01-0.06）：托盘无 `RigidBodyAPI`，无碰撞体，货叉可以在几何上穿过 mesh 模型，产生小量正值。但这不是真正的物理插入。
- **s1.0e/f/g**（`insert_norm_mean` ≈ 0）：托盘有 `convexHull` 碰撞体，货叉被实心凸包物理阻挡，无法接近托盘内部。
- **s1.0h**：使用 `convexDecomposition` 保留了托盘的凹形结构（插入孔），货叉可以真正从孔中插入。

## 技术细节

### 为什么 s1.0e/f/g 的叉车无法插入？

s1.0e/f/g 的 `_setup_pallet_physics` 代码为托盘成功应用了 `RigidBodyAPI` 和 `CollisionAPI`，使其成为动态刚体。但**没有显式设置碰撞近似方法**。

PhysX 在运行时检测到三角网格（triangle mesh）不能用于动态刚体碰撞，自动回退到 `convexHull`：

```
PhysicsUSD: Parse collision - triangle mesh collision (approximation None/MeshSimplification)
cannot be a part of a dynamic body, falling back to convexHull approximation:
/World/envs/env_0/Pallet
```

`convexHull`（单个凸包）会将托盘的凹形结构（插入孔/pocket）填平为实心凸块，导致叉车货叉在物理上无法进入托盘。

### 为什么 s0/s0.6 更快？

s0 和 s0.6 根本没有 `_setup_pallet_physics` 函数，托盘没有 `RigidBodyAPI`，不参与物理仿真，自然没有碰撞计算开销。但也意味着托盘不会对叉车产生任何物理反馈。

### convexDecomposition 参数说明

```python
convex_api.GetMaxConvexHullsAttr().Set(N)    # 最大凸体数
convex_api.GetHullVertexLimitAttr().Set(64)   # 每个凸体最大顶点数
```

- `maxConvexHulls`：PhysX VHACD 算法将凹形网格分解为多个凸包的上限。值越大，近似越精确，但碰撞检测开销越高。
- 1024 环境 × N 个凸包/托盘 = N×1024 个碰撞体参与 PhysX 碰撞检测

### 性能影响估算

| maxConvexHulls | 碰撞体总数 (1024 envs) | 实测 steps/s |
|---------------|----------------------|-------------|
| 1 (convexHull) | 1024 | ~17000（但无法插入） |
| 8 | ≤ 8192 | 待训练确认 |
| 32 | ≤ 32768 | ~890 |

## maxConvexHulls=8 验证结果（2026-02-08）

使用 `verify_forklift_insert_lift.py --headless` 进行单环境自动化验证，**全部关键测试通过**：

### 插入测试 ✅

- `insert_depth: 0.8009m`（`insert_norm: 37.08%`）
- `tip_inside_pallet: True`
- 货叉从理想对齐位置出发，164 步后达到目标插入深度
- 插入过程中横向误差 < 0.12cm，偏航误差 < 0.33°

### 举升测试 ✅

- 升降关节位置变化：`0.0000m → 0.8239m`（+0.8239m）
- 货叉尖端高度变化：`-0.0004m → 0.8235m`（+0.8239m）
- **托盘位置 z 变化：`0.0050m → 0.6120m`（+0.6070m）** -- 托盘确实跟随叉子升起
- 插入深度在举升过程中保持稳定（0.80m → 0.83m）

### 碰撞体诊断

```
[诊断] 修改后 托盘 USD 状态: /World/envs/env_0/Pallet
[诊断] RigidBody: /World/envs/env_0/Pallet type=Xform enabled=True kinematic=False
[诊断] Collision: /World/envs/env_0/Pallet/Xform/Mesh_015 type=Mesh approx=convexDecomposition
[诊断] RigidBody 数量: 1
[诊断] Collision/Mesh 数量: 1
```

**结论：`maxConvexHulls=8` 足以保留托盘插入孔的物理形状，支持真正的货叉插入和托盘举升。**

## 修改记录

- **2026-02-08**：将 `maxConvexHulls` 从 32 降低到 8，在保留托盘插入孔物理形状的同时降低碰撞计算开销。修改位置：
  - `env.py` 函数 `_force_pallet_convex_decomposition()`（第 118 行）
  - `env.py` 方法 `_setup_pallet_physics()`（第 480 行）

## 后续验证要点

1. 1024 环境训练速度是否有显著提升（对比 maxConvexHulls=32 时的 ~890 steps/s）
2. 训练中 `insert_norm_mean` 是否 > 0，`frac_inserted` 是否 > 0
3. 可在 Isaac Sim GUI 中开启 Physics Debug Visualization 检查凸分解后的碰撞线框
