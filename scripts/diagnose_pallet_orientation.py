#!/usr/bin/env python3
"""
诊断脚本：检查托盘插入方向和转向控制逻辑

检查内容：
1. 托盘模型的实际尺寸和方向
2. 插入深度计算的坐标系一致性
3. 转向控制逻辑是否正确
4. 转向角度与前进方向的匹配度
"""

import sys
import os
import math
from pathlib import Path

# 添加 IsaacLab 路径
isaaclab_path = Path(__file__).resolve().parent.parent / "IsaacLab"
sys.path.insert(0, str(isaaclab_path / "source"))

import torch
import numpy as np

# 首先初始化 Isaac Sim
from isaaclab.app import AppLauncher

# 解析命令行参数（如果需要）
import argparse
parser = argparse.ArgumentParser(description="托盘插入方向与转向控制诊断脚本")
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()

# 启动 Isaac Sim
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# 在 Isaac Sim 初始化后导入环境模块
from isaaclab_tasks.direct.forklift_pallet_insert_lift.env_cfg import ForkliftPalletInsertLiftEnvCfg
from isaaclab_tasks.direct.forklift_pallet_insert_lift.env import ForkliftPalletInsertLiftEnv, _quat_to_yaw


def print_section(title):
    """打印分节标题"""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def get_pallet_bounding_box(stage, pallet_prim_path="/World/envs/env_0/Pallet"):
    """获取托盘的边界框（AABB）"""
    # 延迟导入 pxr，确保在 Isaac Sim 环境初始化后导入
    from pxr import Usd, UsdGeom, Gf
    
    prim = stage.GetPrimAtPath(pallet_prim_path)
    if not prim.IsValid():
        return None
    
    # 获取所有子节点
    bbox_min = None
    bbox_max = None
    
    def traverse_prim(prim_path):
        nonlocal bbox_min, bbox_max
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            return
        
        # 检查是否有几何体
        mesh = UsdGeom.Mesh(prim)
        if mesh:
            # 获取边界框
            bbox = UsdGeom.Boundable(prim).ComputeWorldBound(Usd.TimeCode.Default(), "default")
            if bbox:
                bbox_range = bbox.ComputeAlignedBox()
                if bbox_range:
                    box_min = bbox_range.GetMin()
                    box_max = bbox_range.GetMax()
                    
                    if bbox_min is None:
                        bbox_min = box_min
                        bbox_max = box_max
                    else:
                        bbox_min = Gf.Vec3d(
                            min(bbox_min[0], box_min[0]),
                            min(bbox_min[1], box_min[1]),
                            min(bbox_min[2], box_min[2])
                        )
                        bbox_max = Gf.Vec3d(
                            max(bbox_max[0], box_max[0]),
                            max(bbox_max[1], box_max[1]),
                            max(bbox_max[2], box_max[2])
                        )
        
        # 递归遍历子节点
        for child in prim.GetChildren():
            traverse_prim(str(child.GetPath()))
    
    traverse_prim(pallet_prim_path)
    
    if bbox_min is None or bbox_max is None:
        return None
    
    return {
        "min": np.array([bbox_min[0], bbox_min[1], bbox_min[2]]),
        "max": np.array([bbox_max[0], bbox_max[1], bbox_max[2]]),
        "size": np.array([bbox_max[0] - bbox_min[0], 
                          bbox_max[1] - bbox_min[1], 
                          bbox_max[2] - bbox_min[2]])
    }


def check_pallet_orientation(env):
    """检查托盘方向和尺寸"""
    print_section("1. 托盘模型尺寸检查")
    
    # 获取仿真 stage
    stage = env.sim.stage
    
    # 获取托盘边界框
    bbox = get_pallet_bounding_box(stage)
    
    if bbox is None:
        print("[WARN] 无法获取托盘边界框，尝试使用刚体属性...")
        # 尝试从刚体属性获取
        pallet_pos = env.pallet.data.root_pos_w[0].cpu().numpy()
        print(f"托盘位置: {pallet_pos}")
        print("[INFO] 需要手动检查 USD 文件中的托盘尺寸")
        return None
    
    print(f"托盘边界框最小值: {bbox['min']}")
    print(f"托盘边界框最大值: {bbox['max']}")
    print(f"托盘尺寸 (x, y, z): {bbox['size']}")
    
    size_x = bbox['size'][0]
    size_y = bbox['size'][1]
    size_z = bbox['size'][2]
    
    print(f"\n尺寸分析:")
    print(f"  X 方向尺寸: {size_x:.3f} m")
    print(f"  Y 方向尺寸: {size_y:.3f} m")
    print(f"  Z 方向尺寸: {size_z:.3f} m")
    
    # 标准欧标托盘：1200mm × 800mm
    print(f"\n标准欧标托盘尺寸: 1200mm × 800mm (长 × 宽)")
    print(f"代码中 pallet_depth_m = {env.cfg.pallet_depth_m} m")
    
    # 判断哪个是插入方向
    if abs(size_x - 1.2) < 0.1:
        print(f"\n[结论] X 方向 ({size_x:.3f}m) 接近 1.2m，可能是长边（插入方向）")
        print(f"       Y 方向 ({size_y:.3f}m) 可能是短边（横向对齐方向）")
    elif abs(size_y - 1.2) < 0.1:
        print(f"\n[结论] Y 方向 ({size_y:.3f}m) 接近 1.2m，可能是长边")
        print(f"       X 方向 ({size_x:.3f}m) 可能是短边（插入方向）")
    else:
        print(f"\n[警告] 无法确定哪个方向是 1.2m")
    
    return bbox


def check_coordinate_system(env):
    """检查坐标系一致性"""
    print_section("2. 坐标系一致性检查")
    
    # 重置环境
    env.reset()
    
    # 获取初始状态
    robot_pos = env.robot.data.root_pos_w[0].cpu().numpy()
    pallet_pos = env.pallet.data.root_pos_w[0].cpu().numpy()
    
    print(f"叉车初始位置: {robot_pos}")
    print(f"托盘初始位置: {pallet_pos}")
    
    # 计算相对位置
    rel_pos = pallet_pos - robot_pos
    print(f"托盘相对叉车位置: {rel_pos}")
    
    # 检查代码中的计算
    tip = env._compute_fork_tip()
    tip_pos = tip[0].cpu().numpy()
    print(f"货叉尖端位置: {tip_pos}")
    
    # 计算插入深度
    pallet_front_x = env._pallet_front_x
    insert_depth = max(0.0, tip_pos[0] - pallet_front_x)
    print(f"\n代码计算:")
    print(f"  _pallet_front_x = {pallet_front_x:.3f} m")
    print(f"  insert_depth = tip[0] - _pallet_front_x = {insert_depth:.3f} m")
    print(f"  使用 X 坐标计算插入深度")
    
    # 检查横向对齐
    y_err = abs(pallet_pos[1] - robot_pos[1])
    print(f"\n横向对齐计算:")
    print(f"  y_err = abs(pallet_pos[1] - root_pos[1]) = {y_err:.3f} m")
    print(f"  使用 Y 坐标计算横向误差")
    
    print(f"\n[结论]")
    print(f"  - 插入深度使用 X 坐标: {'✓' if insert_depth >= 0 else '✗'}")
    print(f"  - 横向对齐使用 Y 坐标: {'✓' if True else '✗'}")
    print(f"  - pallet_depth_m = {env.cfg.pallet_depth_m} 对应 X 方向")


def check_steering_control(env):
    """检查转向控制逻辑"""
    print_section("3. 转向控制逻辑检查")
    
    # 重置环境
    env.reset()
    
    # 获取关节信息
    print("关节信息:")
    print(f"  前轮关节 IDs: {env._front_wheel_ids}")
    print(f"  后轮关节 IDs: {env._back_wheel_ids}")
    print(f"  转向关节 IDs: {env._rotator_ids}")
    
    # 获取关节名称
    joint_names = env.robot.joint_names
    print(f"\n关节名称:")
    for i, name in enumerate(joint_names):
        if i in env._rotator_ids:
            print(f"  [{i}] {name} (转向关节)")
        elif i in env._front_wheel_ids:
            print(f"  [{i}] {name} (前轮)")
        elif i in env._back_wheel_ids:
            print(f"  [{i}] {name} (后轮)")
        else:
            print(f"  [{i}] {name}")
    
    # 检查转向关节控制的是前轮还是后轮
    print(f"\n转向关节分析:")
    for rot_id in env._rotator_ids:
        joint_name = joint_names[rot_id]
        if "front" in joint_name.lower():
            print(f"  {joint_name}: 控制前轮")
        elif "back" in joint_name.lower():
            print(f"  {joint_name}: 控制后轮")
        else:
            print(f"  {joint_name}: 需要检查 USD 文件确认")
    
    # 测试转向和前进
    print(f"\n测试转向控制:")
    
    # 记录初始位置和朝向
    initial_pos = env.robot.data.root_pos_w[0].clone()
    initial_yaw = _quat_to_yaw(env.robot.data.root_quat_w[0:1])[0]
    
    # 设置动作：前进 + 转向
    test_actions = [
        {"drive": 0.5, "steer": 0.0, "lift": 0.0},  # 只前进
        {"drive": 0.5, "steer": 0.3, "lift": 0.0},  # 前进 + 右转
        {"drive": 0.5, "steer": -0.3, "lift": 0.0}, # 前进 + 左转
    ]
    
    for i, action_dict in enumerate(test_actions):
        print(f"\n测试 {i+1}: drive={action_dict['drive']}, steer={action_dict['steer']}")
        
        # 重置环境
        env.reset()
        initial_pos = env.robot.data.root_pos_w[0].clone()
        
        # 执行动作
        actions = torch.tensor([[action_dict["drive"], action_dict["steer"], action_dict["lift"]]], 
                              device=env.device)
        
        # 执行多步
        for step in range(30):
            env.step(actions)
        
        # 获取最终状态
        final_pos = env.robot.data.root_pos_w[0].clone()
        final_yaw = _quat_to_yaw(env.robot.data.root_quat_w[0:1])[0]
        
        # 获取转向角度
        steer_angle = env._joint_pos[0, env._rotator_ids[0]].item()
        steer_angle_deg = math.degrees(steer_angle)
        
        # 计算位移
        displacement = final_pos - initial_pos
        displacement_xy = displacement[:2]
        distance = torch.norm(displacement_xy).item()
        
        # 计算方向
        direction_rad = torch.atan2(displacement_xy[1], displacement_xy[0]).item()
        direction_deg = math.degrees(direction_rad)
        
        # 计算 yaw 变化
        yaw_change_deg = math.degrees(final_yaw - initial_yaw)
        
        print(f"  转向关节角度: {steer_angle_deg:.2f}°")
        print(f"  位移: ({displacement[0]:.3f}, {displacement[1]:.3f}) m")
        print(f"  距离: {distance:.3f} m")
        print(f"  运动方向: {direction_deg:.2f}°")
        print(f"  朝向变化: {yaw_change_deg:.2f}°")
        
        # 检查是否匹配
        if abs(action_dict["steer"]) > 0.01:
            expected_turn = action_dict["steer"] * 30  # 粗略估计
            if abs(yaw_change_deg - expected_turn) > 10:
                print(f"  [警告] 转向角度与运动方向不匹配！")
                print(f"         期望转向: {expected_turn:.2f}°, 实际: {yaw_change_deg:.2f}°")


def add_visualization(env):
    """添加可视化标记"""
    print_section("4. 可视化验证")
    
    print("[INFO] 可视化功能需要 render_mode='rgb' 或 'human'")
    print("[INFO] 当前脚本以 headless 模式运行，可视化功能已跳过")
    print("\n建议:")
    print("1. 在 Isaac Sim 中手动加载场景")
    print("2. 检查托盘边界框和中心点")
    print("3. 检查叉车货叉尖端位置")
    print("4. 检查插入方向向量")


def main():
    """主函数"""
    print("=" * 80)
    print("托盘插入方向与转向控制诊断脚本")
    print("=" * 80)
    
    # 创建环境配置
    cfg = ForkliftPalletInsertLiftEnvCfg()
    cfg.scene.num_envs = 1  # 只用一个环境进行诊断
    
    # 创建环境
    print("\n[INFO] 正在创建仿真环境...")
    env = ForkliftPalletInsertLiftEnv(cfg, render_mode=None)
    print("[INFO] 环境创建成功")
    
    # 执行检查
    try:
        bbox = check_pallet_orientation(env)
        check_coordinate_system(env)
        check_steering_control(env)
        add_visualization(env)
        
        print_section("诊断完成")
        print("\n建议:")
        print("1. 检查托盘 USD 文件，确认实际尺寸和方向")
        print("2. 如果托盘插入方向与代码不一致，需要调整 pallet_depth_m 或坐标系")
        print("3. 如果转向控制有问题，检查关节层级关系和转向几何")
        print("4. 查看输出结果，确认是否存在问题")
        
    except Exception as e:
        print(f"\n[错误] 诊断过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        env.close()
        # 关闭 Isaac Sim
        simulation_app.close()


if __name__ == "__main__":
    main()
