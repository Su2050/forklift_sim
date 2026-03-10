import argparse
import sys
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
args_cli.enable_cameras = True
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
import isaaclab_tasks
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

def run_env(cam_path):
    env_cfg = parse_env_cfg("Isaac-Forklift-PalletInsertLift-Direct-v0", device="cuda:0", num_envs=1)
    env_cfg.viewer.cam_prim_path = cam_path
    env = gym.make("Isaac-Forklift-PalletInsertLift-Direct-v0", cfg=env_cfg, render_mode="rgb_array")
    env.reset()
    for _ in range(10):
        action = torch.zeros((1, env.action_space.shape[0]), device="cuda:0")
        env.step(action)
    env.close()
    print(f"Finished {cam_path}")

run_env("/OmniverseKit_Persp")
run_env("/World/envs/env_0/Robot/body/Camera")

simulation_app.close()
