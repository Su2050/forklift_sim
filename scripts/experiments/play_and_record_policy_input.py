import argparse
import os
import subprocess
import sys


parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--view_mode", type=str, choices=["global", "policy_camera", "both"], default="both")
args_early, _ = parser.parse_known_args()

if args_early.view_mode == "both":
    print(f"\n{'=' * 50}\n[INFO] Running in 'both' mode.\n{'=' * 50}\n")
    cmd_base = [sys.executable] + sys.argv

    filtered_cmd = []
    skip_next = False
    for arg in cmd_base:
        if skip_next:
            skip_next = False
            continue
        if arg == "--view_mode":
            skip_next = True
            continue
        filtered_cmd.append(arg)
    cmd_base = filtered_cmd

    cmd_global = cmd_base + ["--view_mode", "global"]
    print(f"[INFO] Spawning global view process: {' '.join(cmd_global)}")
    subprocess.run(cmd_global, check=True)

    cmd_policy = cmd_base + ["--view_mode", "policy_camera"]
    print(f"\n[INFO] Spawning policy-camera process: {' '.join(cmd_policy)}")
    subprocess.run(cmd_policy, check=True)

    print(f"\n{'=' * 50}\n[INFO] Finished both recordings.\n{'=' * 50}\n")
    sys.exit(0)


from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Isaac-Forklift-PalletInsertLift-Direct-v0")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--video_length", type=int, default=1200)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--view_mode", type=str, choices=["global", "policy_camera", "both"], default="both")
parser.add_argument("--video_folder", type=str, default="play_videos")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--fps", type=float, default=30.0)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
args_cli.enable_cameras = True
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import cv2
import gymnasium as gym
import torch
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner
import isaaclab_tasks
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg


def _unwrap_obs(obs):
    if isinstance(obs, tuple):
        return obs[0]
    return obs


def _extract_policy_image_frame(obs) -> "torch.Tensor":
    obs = _unwrap_obs(obs)

    if not hasattr(obs, "keys"):
        raise TypeError(f"Expected dict-like observations, got {type(obs)}")

    image = None
    if "image" in obs:
        image = obs["image"]
    else:
        policy_obs = obs.get("policy")
        if hasattr(policy_obs, "keys") and "image" in policy_obs:
            image = policy_obs["image"]

    if image is None:
        raise KeyError("Cannot resolve policy image from observations")
    if image.ndim != 4:
        raise ValueError(f"Expected image tensor with 4 dims, got shape={tuple(image.shape)}")

    if image.shape[1] == 3:
        frame = image[0].detach().float()
        if frame.max() > 1.0:
            frame = frame / 255.0
        frame = torch.clamp(frame, 0.0, 1.0).permute(1, 2, 0)
    elif image.shape[-1] == 3:
        frame = image[0].detach().float()
        if frame.max() > 1.0:
            frame = frame / 255.0
        frame = torch.clamp(frame, 0.0, 1.0)
    else:
        raise ValueError(f"Unexpected image shape={tuple(image.shape)}")

    return torch.round(frame * 255.0).to(dtype=torch.uint8).cpu()


def _make_env_and_policy(view_mode: str):
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")
    env_cfg.seed = args_cli.seed

    render_mode = None
    if view_mode == "global":
        env_cfg.viewer.cam_prim_path = "/OmniverseKit_Persp"
        env_cfg.viewer.resolution = (640, 480)
        render_mode = "rgb_array"

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=render_mode)

    if view_mode == "global":
        video_kwargs = {
            "video_folder": os.path.join(
                os.path.dirname(args_cli.checkpoint),
                "videos",
                f"{args_cli.video_folder}_global",
            ),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    env = RslRlVecEnvWrapper(env)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(args_cli.checkpoint)
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    return env, policy


def run_global():
    print(f"\n{'=' * 50}\n[INFO] Starting recording for view: global\n{'=' * 50}\n")
    env, policy = _make_env_and_policy("global")
    try:
        obs = _unwrap_obs(env.get_observations())
        for _ in range(args_cli.video_length):
            actions = policy(obs)
            step_res = env.step(actions)
            obs = _unwrap_obs(step_res[0])
    finally:
        env.close()
    print("[INFO] Finished recording for view: global")


def run_policy_camera():
    print(f"\n{'=' * 50}\n[INFO] Starting recording for view: policy_camera\n{'=' * 50}\n")
    if args_cli.num_envs != 1:
        raise ValueError("policy_camera recording requires --num_envs 1")

    env, policy = _make_env_and_policy("policy_camera")
    writer = None
    video_path = os.path.join(
        os.path.dirname(args_cli.checkpoint),
        "videos",
        f"{args_cli.video_folder}_policy_camera",
        "video.mp4",
    )
    os.makedirs(os.path.dirname(video_path), exist_ok=True)

    try:
        obs = _unwrap_obs(env.get_observations())
        first_frame = _extract_policy_image_frame(obs).numpy()
        height, width = first_frame.shape[:2]
        writer = cv2.VideoWriter(
            video_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            float(args_cli.fps),
            (width, height),
        )
        if not writer.isOpened():
            raise RuntimeError(f"Failed to open video writer for {video_path}")

        for _ in range(args_cli.video_length):
            frame = _extract_policy_image_frame(obs).numpy()
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            actions = policy(obs)
            step_res = env.step(actions)
            obs = _unwrap_obs(step_res[0])
    finally:
        if writer is not None:
            writer.release()
        env.close()

    print(f"[INFO] Finished recording for view: policy_camera")
    print(f"[INFO] Policy-camera video saved to: {video_path}")


if args_cli.view_mode == "global":
    run_global()
elif args_cli.view_mode == "policy_camera":
    run_policy_camera()
else:
    raise ValueError(f"Unsupported view_mode: {args_cli.view_mode}")

simulation_app.close()
