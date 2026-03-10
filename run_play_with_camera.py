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
import os
import torch
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner
from isaaclab.envs import ViewerCfg
import isaaclab_tasks

env_cfg = isaaclab_tasks.utils.parse_env_cfg("Isaac-Forklift-PalletInsertLift-Direct-v0", device="cuda:0", num_envs=1)
# Set the viewer camera to the robot's camera
env_cfg.viewer.cam_prim_path = "/World/envs/env_0/Robot/body/Camera"
env_cfg.viewer.resolution = (640, 480)

env = gym.make("Isaac-Forklift-PalletInsertLift-Direct-v0", cfg=env_cfg, render_mode="rgb_array")

video_kwargs = {
    "video_folder": "outputs/play_videos",
    "step_trigger": lambda step: step == 0,
    "video_length": 600,
    "disable_logger": True,
}
env = gym.wrappers.RecordVideo(env, **video_kwargs)
env = RslRlVecEnvWrapper(env)

agent_cfg = isaaclab_tasks.utils.parse_env_cfg("Isaac-Forklift-PalletInsertLift-Direct-v0", device="cuda:0", num_envs=1, use_default_kwargs=False)
agent_cfg = isaaclab_tasks.utils.parse_env_cfg("Isaac-Forklift-PalletInsertLift-Direct-v0", device="cuda:0", num_envs=1) # Need to load agent cfg, actually let's use the standard play.py but modify the env_cfg
