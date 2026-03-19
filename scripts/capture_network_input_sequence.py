import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import torchvision.transforms as T
import cv2
import os
import numpy as np
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
    
    out_dir = "/home/uniubi/projects/forklift_sim/images/insertion_seq_network_input"
    os.makedirs(out_dir, exist_ok=True)
    
    # 模拟网络输入预处理 (参考 rsl_rl/modules/vision_backbone.py)
    # 论文中提到将 352x288 缩放为 224x224，我们环境是 256x256，也缩放为 224x224
    transform = T.Compose([
        T.Resize((224, 224), antialias=True),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 反归一化用于可视化
    def denormalize(tensor):
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(tensor.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(tensor.device)
        return tensor * std + mean

    distances = [0.5, 0.4, 0.3, 0.2, 0.1, 0.0, -0.1, -0.2, -0.3, -0.4, -0.5, -0.6, -0.7, -0.8]
    
    pallet_z = env.pallet.data.root_pos_w[0, 2].item()
    robot_z = env.robot.data.root_pos_w[0, 2].item()
    
    pallet_pos = torch.tensor([[0.0, 0.0, pallet_z]], device=env.device)
    pallet_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=env.device)
    
    for d in distances:
        robot_x = -1.08 - d - env._fork_forward_offset
        robot_pos = torch.tensor([[robot_x, 0.0, robot_z]], device=env.device)
        robot_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=env.device)
        
        env.robot.write_root_pose_to_sim(torch.cat([robot_pos, robot_quat], dim=-1))
        env.pallet.write_root_pose_to_sim(torch.cat([pallet_pos, pallet_quat], dim=-1))
        
        joint_pos = torch.zeros((1, env.robot.num_joints), device=env.device)
        joint_vel = torch.zeros((1, env.robot.num_joints), device=env.device)
        env.robot.write_joint_state_to_sim(joint_pos, joint_vel)
        
        for _ in range(5):
            env.step(torch.zeros((1, 2), device=env.device))
            
        if env._camera is not None:
            # 获取原始图像 [1, H, W, C]
            raw_img = env._camera.data.output["rgb"].clone()
            
            # 转换为网络输入格式 [1, C, H, W]，并归一化到 [0, 1]
            if raw_img.shape[-1] == 4:
                raw_img = raw_img[..., :3] # 去掉 alpha 通道
            img_chw = raw_img.permute(0, 3, 1, 2).float() / 255.0
            
            # 应用预处理 (Resize + Normalize)
            net_input = transform(img_chw)
            
            # 反归一化并转回 numpy 用于保存
            vis_img = denormalize(net_input[0])
            vis_img = torch.clamp(vis_img, 0.0, 1.0)
            vis_img = (vis_img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            
            # 转为 BGR 保存
            vis_img_bgr = cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR)
            
            filename = os.path.join(out_dir, f"net_input_{d:+.1f}m.png")
            cv2.imwrite(filename, vis_img_bgr)
            print(f"Saved {filename} (Shape: {vis_img_bgr.shape})")

    simulation_app.close()

if __name__ == "__main__":
    main()
