import argparse
import sys
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Isaac-Forklift-PalletInsertLift-Direct-v0")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--video_length", type=int, default=1200)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--video_folder", type=str, default="play_global")
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
import isaaclab_tasks
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

# Load configs
env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")

# Create environment
env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")

# Wrap for video recording
video_kwargs = {
    "video_folder": os.path.join(os.path.dirname(args_cli.checkpoint), "videos", args_cli.video_folder),
    "step_trigger": lambda step: step == 0,
    "video_length": args_cli.video_length,
    "disable_logger": True,
}
env = gym.wrappers.RecordVideo(env, **video_kwargs)

# Wrap for RSL-RL
env = RslRlVecEnvWrapper(env)

# Create runner
runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
runner.load(args_cli.checkpoint)
policy = runner.get_inference_policy(device=env.unwrapped.device)

# Play
obs = env.get_observations()
if isinstance(obs, tuple):
    obs = obs[0]
for i in range(args_cli.video_length):
    actions = policy(obs)
    step_res = env.step(actions)
    if len(step_res) == 4:
        obs, _, _, _ = step_res
    else:
        obs, _, _, _, _ = step_res

# Cleanup
env.close()
simulation_app.close()
