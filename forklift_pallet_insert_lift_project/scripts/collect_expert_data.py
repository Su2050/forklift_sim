import argparse
import os
import h5py
import torch
import numpy as np
from tqdm import tqdm

from isaaclab.app import AppLauncher

# Add argparse
parser = argparse.ArgumentParser(description="Collect expert data for visual pretraining.")
parser.add_argument("--num_envs", type=int, default=128, help="Number of environments to simulate.")
parser.add_argument("--num_steps", type=int, default=500, help="Number of steps to collect.")
parser.add_argument("--output_path", type=str, default="expert_dataset.h5", help="Output HDF5 file path.")
parser.add_argument("--expert_ckpt", type=str, default="/home/uniubi/projects/forklift_sim/IsaacLab/logs/rsl_rl/forklift_pallet_insert_lift/2026-02-27_17-43-22/model_1999.pt", help="Path to expert checkpoint.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Launch app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
from rsl_rl.modules import ActorCritic

# Import tasks
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

def main():
    # 1. Setup Environment
    env_cfg = parse_env_cfg(
        "Isaac-Forklift-PalletInsertLift-Direct-v0",
        device=args_cli.device,
        num_envs=args_cli.num_envs,
    )
    # Force enable cameras and asymmetric critic (to get 15-dim state in obs_dict["critic"])
    env_cfg.use_camera = True
    env_cfg.use_asymmetric_critic = True
    
    env = gym.make("Isaac-Forklift-PalletInsertLift-Direct-v0", cfg=env_cfg)
    
    # 2. Setup Expert Policy
    dummy_obs = {
        "policy": torch.zeros(1, 15, device=env.unwrapped.device),
        "critic": torch.zeros(1, 15, device=env.unwrapped.device)
    }
    dummy_obs_groups = {
        "policy": ["policy"],
        "critic": ["critic"]
    }
    
    expert_policy = ActorCritic(
        obs=dummy_obs,
        obs_groups=dummy_obs_groups,
        num_actions=3,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[256, 256, 128],
        critic_hidden_dims=[256, 256, 128],
        init_noise_std=1.0, # will be overwritten by load_state_dict
        noise_std_type="log",
    )
    expert_policy.to(env.unwrapped.device)
    
    print(f"Loading expert model from: {args_cli.expert_ckpt}")
    ckpt = torch.load(args_cli.expert_ckpt, map_location=env.unwrapped.device)
    expert_policy.load_state_dict(ckpt['model_state_dict'])
    expert_policy.eval()
    
    # 3. Data Collection Loop
    obs, _ = env.reset()
    
    # We will write to HDF5 incrementally
    f = h5py.File(args_cli.output_path, "w")
    
    max_capacity = args_cli.num_steps * args_cli.num_envs
    img_shape = obs["policy"]["image"].shape[1:] # [3, 64, 64]
    
    chunk_size = min(128, max_capacity)
    dset_img = f.create_dataset("image", shape=(max_capacity, *img_shape), dtype=np.float32, chunks=(chunk_size, *img_shape))
    dset_proprio = f.create_dataset("proprio", shape=(max_capacity, 8), dtype=np.float32)
    dset_state = f.create_dataset("state", shape=(max_capacity, 15), dtype=np.float32)
    dset_action = f.create_dataset("action", shape=(max_capacity, 3), dtype=np.float32)
    
    ptr = 0
    
    print(f"Collecting {max_capacity} samples...")
    for step in tqdm(range(args_cli.num_steps)):
        # Expert acts based on the 15-dim state
        state_tensor = obs["critic"]
        
        with torch.no_grad():
            expert_obs_dict = {"policy": state_tensor}
            actions = expert_policy.act_inference(expert_obs_dict)
            
        next_obs, rewards, dones, truncated, infos = env.step(actions)
        
        # Save to HDF5
        # Move tensors to CPU numpy
        img_np = obs["policy"]["image"].cpu().numpy()
        proprio_np = obs["policy"]["proprio"].cpu().numpy()
        state_np = state_tensor.cpu().numpy()
        action_np = actions.cpu().numpy()
        
        batch_size = img_np.shape[0]
        
        dset_img[ptr:ptr+batch_size] = img_np
        dset_proprio[ptr:ptr+batch_size] = proprio_np
        dset_state[ptr:ptr+batch_size] = state_np
        dset_action[ptr:ptr+batch_size] = action_np
        
        ptr += batch_size
        obs = next_obs
        
    f.close()
    print(f"Data saved to {args_cli.output_path}")
    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()
