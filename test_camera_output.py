# 必须先导入 simulation app
import argparse
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True # 强制开启相机
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import numpy as np
from PIL import Image
import gymnasium as gym
import isaaclab_tasks
from isaaclab_tasks.direct.forklift_pallet_insert_lift.env_cfg import ForkliftPalletInsertLiftEnvCfg

def main():
    cfg = ForkliftPalletInsertLiftEnvCfg()
    cfg.scene.num_envs = 1
    cfg.use_camera = True
    # 强制让材质加载失败，模拟headless训练时的状态
    cfg.wait_for_textures = False 
    
    env = gym.make("Isaac-Forklift-PalletInsertLift-Direct-v0", cfg=cfg)
    obs, _ = env.reset()
    
    # 提取相机图像
    # obs["policy"]["image"] shape: (1, 3, 256, 256), range: [0, 1]
    img_tensor = obs["policy"]["image"][0].cpu().numpy()
    
    # 转换为 HWC 格式并保存
    img_hwc = np.transpose(img_tensor, (1, 2, 0))
    img_uint8 = (img_hwc * 255).astype(np.uint8)
    
    Image.fromarray(img_uint8).save("actual_network_input.png")
    print("Saved actual network input to actual_network_input.png")
    
    simulation_app.close()

if __name__ == "__main__":
    main()
