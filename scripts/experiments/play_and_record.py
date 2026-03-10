import argparse
import sys
import subprocess
import os

# First pass to check if we are in "both" mode before initializing Isaac Sim
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--view_mode", type=str, choices=["global", "camera", "both"], default="both")
args_early, _ = parser.parse_known_args()

if args_early.view_mode == "both":
    print(f"\n{'='*50}\n[INFO] Running in 'both' mode. Will spawn two separate processes.\n{'='*50}\n")
    
    # Reconstruct command line arguments, replacing '--view_mode both' with specific modes
    # We need to find the original python executable and script
    cmd_base = [sys.executable] + sys.argv
    
    # Remove --view_mode both if it exists
    cmd_base = [arg for arg in cmd_base if arg not in ["--view_mode", "both"]]
    
    # Run global
    cmd_global = cmd_base + ["--view_mode", "global"]
    print(f"[INFO] Spawning global view process: {' '.join(cmd_global)}")
    subprocess.run(cmd_global, check=True)
    
    # Run camera
    cmd_camera = cmd_base + ["--view_mode", "camera"]
    print(f"\n[INFO] Spawning camera view process: {' '.join(cmd_camera)}")
    subprocess.run(cmd_camera, check=True)
    
    print(f"\n{'='*50}\n[INFO] Finished both recordings.\n{'='*50}\n")
    sys.exit(0)

# If we are here, view_mode is either 'global' or 'camera'
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Isaac-Forklift-PalletInsertLift-Direct-v0")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--video_length", type=int, default=1200)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--view_mode", type=str, choices=["global", "camera", "both"], default="both", help="Which camera view to record")
parser.add_argument("--video_folder", type=str, default="play_videos", help="Base folder for videos")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
args_cli.enable_cameras = True
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner
import isaaclab_tasks
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

def run_simulation(view_type):
    print(f"\n{'='*50}\n[INFO] Starting recording for view: {view_type}\n{'='*50}\n")
    
    # Load configs
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")

    # Set camera path based on view type
    if view_type == "camera":
        env_cfg.viewer.cam_prim_path = "/World/envs/env_0/Robot/body/Camera"
    else:
        env_cfg.viewer.cam_prim_path = "/OmniverseKit_Persp" # Global view
        
    env_cfg.viewer.resolution = (640, 480)

    # Create environment
    env_cfg.seed = 42 # 固定随机种子以确保两个视角的环境初始化完全一致
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")

    # Wrap for video recording
    folder_suffix = "camera" if view_type == "camera" else "global"
    video_kwargs = {
        "video_folder": os.path.join(os.path.dirname(args_cli.checkpoint), "videos", f"{args_cli.video_folder}_{folder_suffix}"),
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
    print(f"[INFO] Finished recording for view: {view_type}\n")

run_simulation(args_cli.view_mode)

simulation_app.close()
