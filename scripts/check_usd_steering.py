#!/usr/bin/env python3
"""
检查 USD 文件中的转向控制逻辑

分析：
1. 关节层级关系
2. rotator_joint 控制的是前轮还是后轮
3. 转向控制逻辑是否正确
"""

import sys
from pathlib import Path

# 添加 IsaacLab 路径
isaaclab_path = Path(__file__).resolve().parent.parent / "IsaacLab"
sys.path.insert(0, str(isaaclab_path / "source"))

# 首先初始化 Isaac Sim
from isaaclab.app import AppLauncher
import argparse

parser = argparse.ArgumentParser(description="检查 USD 文件中的转向控制逻辑")
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()

# 启动 Isaac Sim
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# 在 Isaac Sim 初始化后导入
from pxr import Usd, UsdGeom, UsdPhysics
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR


def print_section(title):
    """打印分节标题"""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def analyze_joint_hierarchy(stage, root_path=None):
    """分析关节层级关系"""
    print_section("关节层级关系分析")
    
    # 如果没有指定根路径，尝试找到 Robot 节点
    if root_path is None:
        # 尝试常见的路径
        possible_paths = [
            "/ForkliftC",
            "/Robot",
            "/World/Robot",
            "/World/envs/env_0/Robot"
        ]
        root_prim = None
        for path in possible_paths:
            prim = stage.GetPrimAtPath(path)
            if prim.IsValid():
                root_path = path
                root_prim = prim
                break
        
        if root_prim is None:
            # 使用根节点
            root_prim = stage.GetPseudoRoot()
            root_path = "/"
    else:
        root_prim = stage.GetPrimAtPath(root_path)
        if not root_prim.IsValid():
            print(f"[警告] 无法找到路径: {root_path}，尝试查找 Robot 节点")
            root_prim = None
    
    if root_prim is None or not root_prim.IsValid():
        print(f"[错误] 无法找到有效的根节点")
        return {}
    
    print(f"根节点: {root_path}")
    
    # 查找所有关节
    joints = {}
    
    def traverse_prim(prim, depth=0):
        indent = "  " * depth
        prim_path = prim.GetPath().pathString
        
        # 检查是否是关节
        if prim.IsA(UsdPhysics.Joint):
            joint_name = prim.GetName()
            joints[joint_name] = {
                "path": prim_path,
                "parent": str(prim.GetParent().GetPath()) if prim.GetParent().IsValid() else None,
                "type": prim.GetTypeName(),
            }
            print(f"{indent}[JOINT] {joint_name}")
            print(f"{indent}      路径: {prim_path}")
            print(f"{indent}      父节点: {joints[joint_name]['parent']}")
            print(f"{indent}      类型: {prim.GetTypeName()}")
            
            # 检查关节属性
            joint_api = UsdPhysics.Joint(prim)
            if joint_api:
                # 获取 body0 和 body1
                body0_rel = joint_api.GetBody0Rel()
                body1_rel = joint_api.GetBody1Rel()
                if body0_rel:
                    targets = body0_rel.GetTargets()
                    if targets:
                        print(f"{indent}      Body0: {targets[0]}")
                if body1_rel:
                    targets = body1_rel.GetTargets()
                    if targets:
                        print(f"{indent}      Body1: {targets[0]}")
        else:
            # 检查是否是轮子相关的节点
            prim_name = prim.GetName().lower()
            if "wheel" in prim_name or "rotator" in prim_name:
                print(f"{indent}[NODE] {prim.GetName()} ({prim.GetTypeName()})")
        
        # 递归遍历子节点
        for child in prim.GetChildren():
            traverse_prim(child, depth + 1)
    
    traverse_prim(root_prim)
    
    return joints


def check_steering_joints(stage, joints):
    """检查转向关节控制的是哪个轮子"""
    print_section("转向关节分析")
    
    rotator_joints = {}
    wheel_joints = {}
    
    for joint_name, joint_info in joints.items():
        if "rotator" in joint_name.lower():
            rotator_joints[joint_name] = joint_info
        elif "wheel" in joint_name.lower():
            wheel_joints[joint_name] = joint_info
    
    print("转向关节 (rotator_joint):")
    for name, info in rotator_joints.items():
        print(f"  {name}:")
        print(f"    路径: {info['path']}")
        print(f"    父节点: {info['parent']}")
        
        # 检查父节点是否是轮子
        if info['parent']:
            parent_prim = stage.GetPrimAtPath(info['parent'])
            if parent_prim.IsValid():
                parent_name = parent_prim.GetName().lower()
                if "front" in parent_name:
                    print(f"    [结论] 控制前轮")
                elif "back" in parent_name or "rear" in parent_name:
                    print(f"    [结论] 控制后轮")
                else:
                    print(f"    [需要检查] 父节点名称: {parent_name}")
    
    print("\n轮子关节:")
    for name, info in wheel_joints.items():
        print(f"  {name}:")
        print(f"    路径: {info['path']}")
        print(f"    父节点: {info['parent']}")


def check_steering_mechanics(stage, root_path=None):
    """检查转向机构"""
    print_section("转向机构检查")
    
    # 如果没有指定根路径，尝试找到 Robot 节点
    if root_path is None:
        possible_paths = [
            "/ForkliftC",
            "/Robot",
            "/World/Robot",
            "/World/envs/env_0/Robot"
        ]
        root_prim = None
        for path in possible_paths:
            prim = stage.GetPrimAtPath(path)
            if prim.IsValid():
                root_path = path
                root_prim = prim
                break
        
        if root_prim is None:
            root_prim = stage.GetPseudoRoot()
    
    if root_prim is None or not root_prim.IsValid():
        print("[警告] 无法找到有效的根节点")
        return
    
    # 查找 rotator 相关的节点
    rotator_nodes = []
    wheel_nodes = []
    
    def find_steering_nodes(prim):
        name = prim.GetName().lower()
        path = prim.GetPath().pathString
        
        if "rotator" in name:
            rotator_nodes.append({
                "name": prim.GetName(),
                "path": path,
                "parent": str(prim.GetParent().GetPath()) if prim.GetParent().IsValid() else None,
            })
        elif "wheel" in name:
            wheel_nodes.append({
                "name": prim.GetName(),
                "path": path,
                "parent": str(prim.GetParent().GetPath()) if prim.GetParent().IsValid() else None,
            })
        
        for child in prim.GetChildren():
            find_steering_nodes(child)
    
    find_steering_nodes(root_prim)
    
    print("Rotator 节点:")
    for node in rotator_nodes:
        print(f"  {node['name']}")
        print(f"    路径: {node['path']}")
        print(f"    父节点: {node['parent']}")
        if node['parent']:
            parent_name = node['parent'].lower()
            if "front" in parent_name:
                print(f"    [结论] 控制前轮")
            elif "back" in parent_name or "rear" in parent_name:
                print(f"    [结论] 控制后轮")
    
    print("\n轮子节点:")
    for node in wheel_nodes:
        print(f"  {node['name']}")
        print(f"    路径: {node['path']}")
        print(f"    父节点: {node['parent']}")


def main():
    """主函数"""
    print("=" * 80)
    print("USD 文件转向控制逻辑检查")
    print("=" * 80)
    
    # 打开 USD 文件
    usd_path = f"{ISAAC_NUCLEUS_DIR}/Robots/IsaacSim/ForkliftC/forklift_c.usd"
    print(f"\nUSD 文件路径: {usd_path}")
    
    stage = Usd.Stage.Open(usd_path)
    if not stage:
        print(f"[错误] 无法打开 USD 文件: {usd_path}")
        return
    
    print("[INFO] USD 文件打开成功")
    
    # 分析关节层级
    joints = analyze_joint_hierarchy(stage)
    
    # 检查转向关节
    check_steering_joints(stage, joints)
    
    # 检查转向机构
    check_steering_mechanics(stage)
    
    print_section("检查完成")
    print("\n建议:")
    print("1. 如果 rotator_joint 控制的是后轮，这是不正常的")
    print("2. 标准叉车应该是前轮转向，后轮驱动")
    print("3. 如果确实是后轮转向，需要修改控制逻辑或 USD 文件")
    
    # 关闭 Isaac Sim
    simulation_app.close()


if __name__ == "__main__":
    main()
