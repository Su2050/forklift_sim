#!/usr/bin/env python3
"""
修复托盘碰撞体 - 使用 SimulationApp 初始化

问题背景：
- Nucleus 上的 pallet.usd 默认使用 boundingCube 碰撞体
- 运行时修改 USD 属性无法触发 PhysX 重新烹饪碰撞体
- 解决方案：预先修改 USD 文件，设置凸分解碰撞

使用方法：
    cd /home/uniubi/projects/forklift_sim/IsaacLab
    ./isaaclab.sh -p /home/uniubi/projects/forklift_sim/scripts/fix_pallet_collision_simple.py
"""

import os
import sys

# ============== 配置参数 ==============
OUTPUT_PATH = "/home/uniubi/projects/forklift_sim/assets/pallet_convex.usd"
MAX_HULLS = 32
HULL_VERTEX_LIMIT = 64
CONTACT_OFFSET = 0.02
REST_OFFSET = 0.005
# =====================================


def main():
    print("=" * 60)
    print("[开始] 修复托盘碰撞体")
    print("=" * 60)
    
    # 首先初始化 SimulationApp（这会加载 Omniverse 运行时）
    print("[信息] 初始化 Isaac Sim 运行时...")
    
    try:
        from isaacsim import SimulationApp
    except ImportError:
        try:
            from omni.isaac.kit import SimulationApp
        except ImportError:
            print("[错误] 无法导入 SimulationApp")
            print("[提示] 请确保使用 isaaclab.sh -p 运行此脚本")
            sys.exit(1)
    
    # 创建 headless 模式的 SimulationApp
    simulation_app = SimulationApp({"headless": True})
    print("[成功] SimulationApp 已初始化")
    
    # 现在可以导入 pxr 和 Isaac Sim 工具了
    from pxr import Usd, UsdGeom, UsdPhysics, PhysxSchema
    print("[成功] 已导入 pxr 模块")
    
    # 导入 Isaac Sim 资产工具（使用 isaacsim.storage.native 获取正确的资产路径）
    try:
        import isaacsim.storage.native as nucleus_utils
    except ImportError:
        import isaacsim.core.utils.nucleus as nucleus_utils
    
    assets_root = nucleus_utils.get_assets_root_path()
    print(f"[信息] Assets Root Path = {assets_root}")
    
    if assets_root is None:
        print("[错误] 无法获取资产根路径，Nucleus 服务器可能未运行")
        simulation_app.close()
        sys.exit(1)
    
    ISAAC_NUCLEUS_DIR = f"{assets_root}/Isaac"
    print(f"[信息] ISAAC_NUCLEUS_DIR = {ISAAC_NUCLEUS_DIR}")

    # 确保输出目录存在
    output_dir = os.path.dirname(OUTPUT_PATH)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"[信息] 创建输出目录: {output_dir}")

    # 构建托盘 USD 路径
    pallet_nucleus_path = f"{ISAAC_NUCLEUS_DIR}/Props/Pallet/pallet.usd"
    print(f"[信息] 托盘 Nucleus 路径: {pallet_nucleus_path}")
    
    # 检查文件是否存在（使用 omni.client）
    import omni.client
    result, _ = omni.client.stat(pallet_nucleus_path)
    file_exists = (result == omni.client.Result.OK)
    print(f"[信息] 文件存在: {file_exists} (omni.client.stat result: {result})")
    
    if not file_exists:
        print("[错误] 托盘 USD 文件不存在")
        print("[提示] 请确保 Nucleus 服务器正在运行，且 Isaac Sim 资产已下载")
        simulation_app.close()
        sys.exit(1)
    
    # 直接打开 Nucleus 路径
    stage = None
    source_path = pallet_nucleus_path
    
    # 打开 USD 文件
    try:
        print(f"[信息] 打开: {source_path}")
        stage = Usd.Stage.Open(source_path)
        if stage:
            print(f"[成功] 已打开: {source_path}")
        else:
            raise RuntimeError("Stage is None")
    except Exception as e:
        print(f"[错误] 无法打开 USD 文件: {e}")
        simulation_app.close()
        sys.exit(1)

    # 统计
    mesh_count = 0
    collision_count = 0

    print("\n" + "=" * 60)
    print("[信息] 开始修改碰撞体属性...")
    print(f"[参数] maxHulls={MAX_HULLS}, hullVertexLimit={HULL_VERTEX_LIMIT}")
    print(f"[参数] contactOffset={CONTACT_OFFSET}, restOffset={REST_OFFSET}")
    print("=" * 60)

    # 遍历所有 prim
    for prim in stage.Traverse():
        prim_path = prim.GetPath().pathString
        
        # 只处理 Mesh 类型
        if not prim.IsA(UsdGeom.Mesh):
            continue
            
        mesh_count += 1
        print(f"\n[处理] Mesh: {prim_path}")

        # 1. 应用 CollisionAPI
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            UsdPhysics.CollisionAPI.Apply(prim)
            print("  [+] 应用 CollisionAPI")
        
        collision_api = UsdPhysics.CollisionAPI(prim)
        collision_api.CreateCollisionEnabledAttr().Set(True)
        print("  [+] 启用碰撞")

        # 2. 设置 MeshCollisionAPI，使用凸分解
        if not prim.HasAPI(UsdPhysics.MeshCollisionAPI):
            UsdPhysics.MeshCollisionAPI.Apply(prim)
            print("  [+] 应用 MeshCollisionAPI")
        
        mesh_collision_api = UsdPhysics.MeshCollisionAPI(prim)
        mesh_collision_api.GetApproximationAttr().Set("convexDecomposition")
        print("  [+] 设置碰撞近似: convexDecomposition")

        # 3. 设置凸分解参数
        if not prim.HasAPI(PhysxSchema.PhysxConvexDecompositionCollisionAPI):
            PhysxSchema.PhysxConvexDecompositionCollisionAPI.Apply(prim)
            print("  [+] 应用 PhysxConvexDecompositionCollisionAPI")
        
        convex_api = PhysxSchema.PhysxConvexDecompositionCollisionAPI(prim)
        convex_api.GetMaxConvexHullsAttr().Set(MAX_HULLS)
        convex_api.GetHullVertexLimitAttr().Set(HULL_VERTEX_LIMIT)
        print(f"  [+] 凸分解: maxHulls={MAX_HULLS}, vertexLimit={HULL_VERTEX_LIMIT}")

        # 4. 设置 PhysxCollisionAPI 参数
        if not prim.HasAPI(PhysxSchema.PhysxCollisionAPI):
            PhysxSchema.PhysxCollisionAPI.Apply(prim)
            print("  [+] 应用 PhysxCollisionAPI")
        
        physx_collision_api = PhysxSchema.PhysxCollisionAPI(prim)
        physx_collision_api.GetContactOffsetAttr().Set(CONTACT_OFFSET)
        physx_collision_api.GetRestOffsetAttr().Set(REST_OFFSET)
        print(f"  [+] 接触参数: contactOffset={CONTACT_OFFSET}, restOffset={REST_OFFSET}")

        collision_count += 1

    print("\n" + "=" * 60)
    print(f"[统计] 处理了 {mesh_count} 个 Mesh，修改了 {collision_count} 个碰撞体")
    print("=" * 60)

    # 保存文件
    print(f"\n[信息] 保存到: {OUTPUT_PATH}")
    stage.Export(OUTPUT_PATH)
    print("[成功] USD 文件已保存")

    # 验证
    print("\n" + "=" * 60)
    print("[验证] 检查保存后的碰撞设置...")
    print("=" * 60)
    
    verify_stage = Usd.Stage.Open(OUTPUT_PATH)
    if verify_stage:
        verified_count = 0
        for prim in verify_stage.Traverse():
            if prim.IsA(UsdGeom.Mesh) and prim.HasAPI(UsdPhysics.MeshCollisionAPI):
                mesh_api = UsdPhysics.MeshCollisionAPI(prim)
                approx = mesh_api.GetApproximationAttr().Get()
                if approx == "convexDecomposition":
                    print(f"  ✅ {prim.GetPath()} - Approximation: {approx}")
                    verified_count += 1
                else:
                    print(f"  ❌ {prim.GetPath()} - Approximation: {approx} (预期 convexDecomposition)")
        
        print(f"\n[总结] {verified_count}/{mesh_count} 个 Mesh 碰撞设置验证通过")
    else:
        print("[错误] 无法打开保存的文件进行验证")

    print("\n" + "=" * 60)
    print("[完成] 托盘碰撞体修复完成！")
    print(f"[下一步] 请更新 env_cfg.py 中的 pallet_usd_path 为:")
    print(f"         {OUTPUT_PATH}")
    print("=" * 60)
    
    # 关闭 SimulationApp
    simulation_app.close()


if __name__ == "__main__":
    main()
