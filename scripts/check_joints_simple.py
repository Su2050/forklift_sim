#!/usr/bin/env python3
"""
简化版关节检查脚本：通过环境直接检查关节信息
"""

import sys
from pathlib import Path
import torch

# 添加 IsaacLab 路径
isaaclab_path = Path(__file__).resolve().parent.parent / "IsaacLab"
sys.path.insert(0, str(isaaclab_path / "source"))

# 首先初始化 Isaac Sim
from isaaclab.app import AppLauncher
import argparse

parser = argparse.ArgumentParser(description="检查关节信息")
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()

# 启动 Isaac Sim
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# 在 Isaac Sim 初始化后导入
from isaaclab_tasks.direct.forklift_pallet_insert_lift.env_cfg import ForkliftPalletInsertLiftEnvCfg
from isaaclab_tasks.direct.forklift_pallet_insert_lift.env import ForkliftPalletInsertLiftEnv


def print_section(title):
    """打印分节标题"""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def main():
    """主函数"""
    print("=" * 80)
    print("关节信息检查（通过环境）")
    print("=" * 80)
    
    # 创建环境配置
    cfg = ForkliftPalletInsertLiftEnvCfg()
    cfg.scene.num_envs = 1
    
    print("\n[INFO] 正在创建环境...")
    env = ForkliftPalletInsertLiftEnv(cfg)
    
    print("[INFO] 环境创建成功")
    
    # 获取所有关节名称
    print_section("所有关节信息")
    joint_names = env.robot.joint_names
    print(f"总关节数: {len(joint_names)}")
    print("\n关节列表:")
    for i, name in enumerate(joint_names):
        marker = ""
        if i in env._rotator_ids:
            marker = " [转向关节]"
        elif i in env._front_wheel_ids:
            marker = " [前轮]"
        elif i in env._back_wheel_ids:
            marker = " [后轮]"
        elif i == env._lift_id:
            marker = " [升降]"
        print(f"  [{i:2d}] {name}{marker}")
    
    # 检查转向关节
    print_section("转向关节详细分析")
    print("转向关节 IDs:", env._rotator_ids)
    print("转向关节名称:")
    for rot_id in env._rotator_ids:
        joint_name = joint_names[rot_id]
        print(f"  [{rot_id}] {joint_name}")
        
        # 检查名称中是否包含 front/back
        name_lower = joint_name.lower()
        if "front" in name_lower:
            print(f"    [结论] 名称包含 'front'，可能是前轮转向")
        elif "back" in name_lower or "rear" in name_lower:
            print(f"    [结论] 名称包含 'back/rear'，可能是后轮转向")
        else:
            print(f"    [需要检查] 名称不明确，需要查看 USD 文件层级关系")
    
    # 检查轮子关节
    print_section("轮子关节分析")
    print("前轮关节 IDs:", env._front_wheel_ids)
    print("前轮关节名称:")
    for wheel_id in env._front_wheel_ids:
        print(f"  [{wheel_id}] {joint_names[wheel_id]}")
    
    print("\n后轮关节 IDs:", env._back_wheel_ids)
    print("后轮关节名称:")
    for wheel_id in env._back_wheel_ids:
        print(f"  [{wheel_id}] {joint_names[wheel_id]}")
    
    # 检查关节层级关系（通过 USD stage）
    print_section("USD 文件层级关系检查")
    stage = env.sim.stage
    
    # 查找 rotator_joint 的父节点
    for rot_id in env._rotator_ids:
        joint_name = joint_names[rot_id]
        # 尝试找到对应的 prim
        # 关节通常在 Robot 下
        robot_prim_paths = [
            "/World/envs/env_0/Robot",
            "/World/envs/env_0/Robot/" + joint_name,
        ]
        
        for path in robot_prim_paths:
            prim = stage.GetPrimAtPath(path)
            if prim.IsValid():
                print(f"\n关节 {joint_name} 的 USD 路径: {path}")
                parent = prim.GetParent()
                if parent.IsValid():
                    parent_path = str(parent.GetPath())
                    parent_name = parent.GetName().lower()
                    print(f"  父节点路径: {parent_path}")
                    print(f"  父节点名称: {parent.GetName()}")
                    
                    # 检查父节点名称
                    if "front" in parent_name:
                        print(f"  [结论] 父节点包含 'front'，rotator_joint 控制前轮")
                    elif "back" in parent_name or "rear" in parent_name:
                        print(f"  [结论] 父节点包含 'back/rear'，rotator_joint 控制后轮")
                    else:
                        print(f"  [需要进一步检查] 父节点名称: {parent.GetName()}")
                
                # 检查是否有子节点
                children = prim.GetChildren()
                if children:
                    print(f"  子节点:")
                    for child in children:
                        print(f"    - {child.GetName()}")
                break
    
    # 测试转向控制
    print_section("转向控制测试")
    env.reset()
    
    # 获取初始状态
    initial_pos = env.robot.data.root_pos_w[0].clone()
    initial_yaw = env.robot.data.root_quat_w[0].clone()
    
    # 设置转向角度
    test_steer = 0.3  # 右转
    actions = torch.tensor([[0.0, test_steer, 0.0]], device=env.device)
    
    # 执行一步
    env.step(actions)
    
    # 检查转向关节角度
    rotator_angles = env._joint_pos[0, env._rotator_ids]
    print(f"测试转向动作: steer={test_steer}")
    print(f"转向关节角度 (rad): {rotator_angles}")
    print(f"转向关节角度 (deg): {torch.rad2deg(rotator_angles)}")
    
    # 检查位置变化
    final_pos = env.robot.data.root_pos_w[0].clone()
    displacement = final_pos - initial_pos
    print(f"位置变化: {displacement}")
    
    print_section("检查完成")
    
    # 关闭环境
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
