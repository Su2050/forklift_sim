import argparse
from omni.isaac.lab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import cv2
import os
from isaaclab_tasks.direct.forklift_pallet_insert_lift.env import ForkliftPalletInsertLiftEnv
from isaaclab_tasks.direct.forklift_pallet_insert_lift.env_cfg import ForkliftPalletInsertLiftEnvCfg

def main():
    cfg = ForkliftPalletInsertLiftEnvCfg()
    cfg.scene.num_envs = 1
    cfg.use_camera = True
    cfg.camera_width = 256
    cfg.camera_height = 256
    
    env = ForkliftPalletInsertLiftEnv(cfg)
    env.reset()
    
    out_dir = "/home/uniubi/projects/forklift_sim/images/insertion_seq"
    os.makedirs(out_dir, exist_ok=True)
    
    # 从 0.5m 到 -0.8m (完全插入)
    distances = [0.5, 0.4, 0.3, 0.2, 0.1, 0.0, -0.1, -0.2, -0.3, -0.4, -0.5, -0.6, -0.7, -0.8]
    
    # 获取托盘和叉车的默认 Z 高度，防止穿模
    pallet_z = env.pallet.data.root_pos_w[0, 2].item()
    robot_z = env.robot.data.root_pos_w[0, 2].item()
    
    pallet_pos = torch.tensor([[0.0, 0.0, pallet_z]], device=env.device)
    pallet_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=env.device)
    
    for d in distances:
        # 托盘深度是 2.16m，前沿在 -1.08m
        # 叉尖需要距离托盘前沿 d 米，所以叉尖 x = -1.08 - d
        # 叉车中心 x = 叉尖 x - fork_forward_offset
        robot_x = -1.08 - d - env._fork_forward_offset
        robot_pos = torch.tensor([[robot_x, 0.0, robot_z]], device=env.device)
        robot_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=env.device)
        
        env.robot.write_root_pose_to_sim(torch.cat([robot_pos, robot_quat], dim=-1))
        env.pallet.write_root_pose_to_sim(torch.cat([pallet_pos, pallet_quat], dim=-1))
        
        # 重置关节
        joint_pos = torch.zeros((1, env.robot.num_joints), device=env.device)
        joint_vel = torch.zeros((1, env.robot.num_joints), device=env.device)
        env.robot.write_joint_state_to_sim(joint_pos, joint_vel)
        
        # 步进仿真以更新物理和渲染相机
        for _ in range(5):
            env.step(torch.zeros((1, 2), device=env.device))
            
        # 捕获图像
        if env._camera is not None:
            img = env._camera.data.output["rgb"][0].cpu().numpy()
            if img.shape[-1] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                
            filename = os.path.join(out_dir, f"dist_{d:+.1f}m.png")
            cv2.imwrite(filename, img)
            print(f"Saved {filename}")

    simulation_app.close()

if __name__ == "__main__":
    main()
