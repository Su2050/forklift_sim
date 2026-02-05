#!/usr/bin/env python3
"""
修复托盘碰撞体，使货叉能够插入 pocket 且举升时不穿透。

问题背景：
- Nucleus 上的 pallet.usd 默认使用 boundingCube 碰撞体
- 运行时修改 USD 属性无法触发 PhysX 重新烹饪碰撞体
- 解决方案：预先修改 USD 文件，设置凸分解碰撞

使用方法：
    cd /home/uniubi/projects/forklift_sim/IsaacLab
    ./isaaclab.sh -p ../scripts/fix_pallet_collision.py
"""

import os
import sys
import argparse


def main():
    parser = argparse.ArgumentParser(description="修复托盘 USD 碰撞体")
    parser.add_argument(
        "--output",
        type=str,
        default="/home/uniubi/projects/forklift_sim/assets/pallet_convex.usd",
        help="输出文件路径",
    )
    parser.add_argument(
        "--max-hulls",
        type=int,
        default=32,
        help="凸分解最大凸体数量",
    )
    parser.add_argument(
        "--hull-vertex-limit",
        type=int,
        default=64,
        help="每个凸体最大顶点数",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="仅验证现有文件的碰撞设置，不修改",
    )
    args = parser.parse_args()

    # 延迟导入 pxr（需要 Isaac Sim 环境）
    try:
        from pxr import Usd, UsdGeom, UsdPhysics, PhysxSchema
    except ImportError:
        print("[错误] 无法导入 pxr 模块，请通过 isaaclab.sh -p 运行此脚本")
        sys.exit(1)

    # 尝试导入 omni.client 用于从 Nucleus 下载
    try:
        import omni.client
        HAS_OMNI_CLIENT = True
    except ImportError:
        HAS_OMNI_CLIENT = False
        print("[警告] omni.client 不可用，将尝试直接打开 Nucleus 路径")

    # Nucleus 上的原始托盘路径
    NUCLEUS_PALLET_PATH = "omniverse://localhost/NVIDIA/Assets/Isaac/4.2/Isaac/Props/Pallet/pallet.usd"
    
    # 备选路径（不同 Isaac Sim 版本可能不同）
    ALTERNATIVE_PATHS = [
        "omniverse://localhost/NVIDIA/Assets/Isaac/4.1/Isaac/Props/Pallet/pallet.usd",
        "omniverse://localhost/NVIDIA/Assets/Isaac/4.0/Isaac/Props/Pallet/pallet.usd",
        "omniverse://localhost/NVIDIA/Assets/Isaac/2023.1.1/Isaac/Props/Pallet/pallet.usd",
    ]

    output_path = args.output
    output_dir = os.path.dirname(output_path)

    # 确保输出目录存在
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"[信息] 创建输出目录: {output_dir}")

    if args.verify_only:
        # 仅验证模式
        if not os.path.exists(output_path):
            print(f"[错误] 文件不存在: {output_path}")
            sys.exit(1)
        verify_collision_settings(output_path)
        return

    # 尝试打开 Nucleus 文件
    stage = None
    source_path = None
    
    for path in [NUCLEUS_PALLET_PATH] + ALTERNATIVE_PATHS:
        try:
            print(f"[信息] 尝试打开: {path}")
            stage = Usd.Stage.Open(path)
            if stage:
                source_path = path
                print(f"[成功] 已打开: {path}")
                break
        except Exception as e:
            print(f"[警告] 无法打开 {path}: {e}")
            continue

    if stage is None:
        print("[错误] 无法从 Nucleus 打开任何托盘 USD 文件")
        print("[提示] 请确保：")
        print("  1. Nucleus 服务器正在运行")
        print("  2. Isaac Sim 资产已下载")
        print("  3. 或手动指定本地 pallet.usd 路径")
        sys.exit(1)

    # 统计修改
    collision_count = 0
    mesh_count = 0

    print("\n" + "=" * 60)
    print("[信息] 开始修改碰撞体属性...")
    print("=" * 60)

    # 遍历所有 prim
    for prim in stage.Traverse():
        prim_path = prim.GetPath().pathString
        
        # 只处理 Mesh 类型的 prim
        if not prim.IsA(UsdGeom.Mesh):
            continue
            
        mesh_count += 1
        print(f"\n[处理] Mesh: {prim_path}")

        # 1. 应用 CollisionAPI（显式启用碰撞）
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            UsdPhysics.CollisionAPI.Apply(prim)
            print(f"  [+] 应用 CollisionAPI")
        
        collision_api = UsdPhysics.CollisionAPI(prim)
        # 确保碰撞启用
        collision_api.CreateCollisionEnabledAttr().Set(True)
        print(f"  [+] 启用碰撞")

        # 2. 设置 MeshCollisionAPI，使用凸分解
        if not prim.HasAPI(UsdPhysics.MeshCollisionAPI):
            UsdPhysics.MeshCollisionAPI.Apply(prim)
            print(f"  [+] 应用 MeshCollisionAPI")
        
        mesh_collision_api = UsdPhysics.MeshCollisionAPI(prim)
        mesh_collision_api.GetApproximationAttr().Set("convexDecomposition")
        print(f"  [+] 设置碰撞近似: convexDecomposition")

        # 3. 设置凸分解参数
        if not prim.HasAPI(PhysxSchema.PhysxConvexDecompositionCollisionAPI):
            PhysxSchema.PhysxConvexDecompositionCollisionAPI.Apply(prim)
            print(f"  [+] 应用 PhysxConvexDecompositionCollisionAPI")
        
        convex_api = PhysxSchema.PhysxConvexDecompositionCollisionAPI(prim)
        convex_api.GetMaxConvexHullsAttr().Set(args.max_hulls)
        convex_api.GetHullVertexLimitAttr().Set(args.hull_vertex_limit)
        print(f"  [+] 凸分解参数: maxHulls={args.max_hulls}, hullVertexLimit={args.hull_vertex_limit}")

        # 4. 设置 PhysxCollisionAPI 参数（接触偏移，防止薄片穿模）
        if not prim.HasAPI(PhysxSchema.PhysxCollisionAPI):
            PhysxSchema.PhysxCollisionAPI.Apply(prim)
            print(f"  [+] 应用 PhysxCollisionAPI")
        
        physx_collision_api = PhysxSchema.PhysxCollisionAPI(prim)
        physx_collision_api.GetContactOffsetAttr().Set(0.02)  # 2cm 接触偏移
        physx_collision_api.GetRestOffsetAttr().Set(0.005)    # 0.5cm 静止偏移
        print(f"  [+] 接触参数: contactOffset=0.02, restOffset=0.005")

        collision_count += 1

    print("\n" + "=" * 60)
    print(f"[统计] 处理了 {mesh_count} 个 Mesh，修改了 {collision_count} 个碰撞体")
    print("=" * 60)

    # 保存到本地
    print(f"\n[信息] 保存到: {output_path}")
    stage.Export(output_path)
    print(f"[成功] USD 文件已保存")

    # 验证保存的文件
    print("\n" + "=" * 60)
    print("[验证] 检查保存后的碰撞设置...")
    print("=" * 60)
    verify_collision_settings(output_path)

    print("\n[完成] 托盘碰撞体修复完成！")
    print(f"[下一步] 请更新 env_cfg.py 中的 pallet_usd_path 为: {output_path}")


def verify_collision_settings(usd_path: str):
    """验证 USD 文件的碰撞体设置"""
    from pxr import Usd, UsdGeom, UsdPhysics, PhysxSchema

    stage = Usd.Stage.Open(usd_path)
    if stage is None:
        print(f"[错误] 无法打开: {usd_path}")
        return

    print(f"\n[验证] 文件: {usd_path}")
    
    mesh_count = 0
    correct_count = 0
    
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        
        mesh_count += 1
        prim_path = prim.GetPath().pathString
        
        # 检查 CollisionAPI
        has_collision = prim.HasAPI(UsdPhysics.CollisionAPI)
        collision_enabled = False
        if has_collision:
            collision_api = UsdPhysics.CollisionAPI(prim)
            collision_enabled = collision_api.GetCollisionEnabledAttr().Get()
        
        # 检查 MeshCollisionAPI
        approx = "N/A"
        if prim.HasAPI(UsdPhysics.MeshCollisionAPI):
            mesh_api = UsdPhysics.MeshCollisionAPI(prim)
            approx = mesh_api.GetApproximationAttr().Get() or "N/A"
        
        # 检查凸分解参数
        max_hulls = "N/A"
        vertex_limit = "N/A"
        if prim.HasAPI(PhysxSchema.PhysxConvexDecompositionCollisionAPI):
            convex_api = PhysxSchema.PhysxConvexDecompositionCollisionAPI(prim)
            max_hulls = convex_api.GetMaxConvexHullsAttr().Get()
            vertex_limit = convex_api.GetHullVertexLimitAttr().Get()
        
        # 检查接触参数
        contact_offset = "N/A"
        rest_offset = "N/A"
        if prim.HasAPI(PhysxSchema.PhysxCollisionAPI):
            physx_api = PhysxSchema.PhysxCollisionAPI(prim)
            contact_offset = physx_api.GetContactOffsetAttr().Get()
            rest_offset = physx_api.GetRestOffsetAttr().Get()
        
        is_correct = (
            has_collision and 
            collision_enabled and 
            approx == "convexDecomposition"
        )
        
        status = "✅" if is_correct else "❌"
        if is_correct:
            correct_count += 1
        
        print(f"\n{status} Mesh: {prim_path}")
        print(f"   CollisionAPI: {has_collision}, enabled={collision_enabled}")
        print(f"   Approximation: {approx}")
        print(f"   ConvexDecomposition: maxHulls={max_hulls}, vertexLimit={vertex_limit}")
        print(f"   Contact: offset={contact_offset}, rest={rest_offset}")

    print(f"\n[总结] {correct_count}/{mesh_count} 个 Mesh 碰撞设置正确")
    
    if correct_count == mesh_count and mesh_count > 0:
        print("[结果] ✅ 所有碰撞体设置正确！")
    else:
        print("[结果] ❌ 部分碰撞体设置不正确，请检查")


if __name__ == "__main__":
    main()
