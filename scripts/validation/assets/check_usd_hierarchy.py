
import sys
from pathlib import Path
import isaaclab.sim as sim_utils
from isaaclab.app import AppLauncher
import argparse

# 初始化 Isaac Sim
parser = argparse.ArgumentParser(description="检查 USD 关节层级结构")
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from isaaclab_tasks.direct.forklift_pallet_insert_lift.env_cfg import ForkliftPalletInsertLiftEnvCfg
from isaaclab_tasks.direct.forklift_pallet_insert_lift.env import ForkliftPalletInsertLiftEnv
from pxr import Usd, UsdGeom, UsdPhysics

def print_prim_hierarchy(prim, depth=0):
    indent = "  " * depth
    print(f"{indent}- {prim.GetName()} ({prim.GetTypeName()})")
    
    # 检查是否有关节 API
    if prim.HasAPI(UsdPhysics.DriveAPI):
        print(f"{indent}  [DriveAPI]")
    
    for child in prim.GetChildren():
        print_prim_hierarchy(child, depth + 1)

def check_hierarchy():
    cfg = ForkliftPalletInsertLiftEnvCfg()
    cfg.scene.num_envs = 1
    env = ForkliftPalletInsertLiftEnv(cfg)
    
    print("\n" + "="*80)
    print("USD 关节层级结构检查")
    print("="*80)
    
    stage = env.sim.stage
    # 找到 Robot prim
    robot_prim_path = "/World/envs/env_0/Robot"
    robot_prim = stage.GetPrimAtPath(robot_prim_path)
    
    if not robot_prim.IsValid():
        print(f"Error: Robot prim not found at {robot_prim_path}")
        return

    print(f"Robot Root: {robot_prim_path}")
    
    # 重点检查后轮和转向关节的关系
    # 遍历 Robot 下的所有 Joint
    print("\n关键关节关系分析:")
    
    rotator_joints = ["left_rotator_joint", "right_rotator_joint"]
    back_wheel_joints = ["left_back_wheel_joint", "right_back_wheel_joint"]
    
    for rot_name in rotator_joints:
        rot_path = f"{robot_prim_path}/{rot_name}"
        rot_prim = stage.GetPrimAtPath(rot_path)
        if rot_prim.IsValid():
            joint = UsdPhysics.Joint(rot_prim)
            body0 = joint.GetBody0Rel().GetTargets()
            body1 = joint.GetBody1Rel().GetTargets()
            print(f"\n关节 {rot_name}:")
            print(f"  Body0 (Parent): {body0}")
            print(f"  Body1 (Child):  {body1}")
            
            # 检查 Body1 (转向架) 下面是否有轮子关节
            if body1:
                child_link_path = body1[0]
                print(f"  转向架 Link: {child_link_path}")
                # 检查这个 Link 是否是后轮关节的 Parent
                
    for wheel_name in back_wheel_joints:
        wheel_path = f"{robot_prim_path}/{wheel_name}"
        wheel_prim = stage.GetPrimAtPath(wheel_path)
        if wheel_prim.IsValid():
            joint = UsdPhysics.Joint(wheel_prim)
            body0 = joint.GetBody0Rel().GetTargets()
            body1 = joint.GetBody1Rel().GetTargets()
            print(f"\n关节 {wheel_name}:")
            print(f"  Body0 (Parent): {body0}")
            print(f"  Body1 (Child):  {body1}")

    env.close()
    simulation_app.close()

if __name__ == "__main__":
    check_hierarchy()
