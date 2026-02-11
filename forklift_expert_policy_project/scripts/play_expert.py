"""
Play the rule-based expert policy in the IsaacLab forklift environment and
record a video (headless-compatible).

This script is intended for **visual verification** of the expert behaviour
before committing to a full demo-collection run.

Usage
-----
.. code-block:: bash

    ./isaaclab.sh -p forklift_expert_policy_project/scripts/play_expert.py \\
        --task Isaac-Forklift-PalletInsertLift-Direct-v0 \\
        --num_envs 1 --headless \\
        --video --video_length 600

Output
------
* Console: per-step debug info (stage, dist, lat, yaw, insert_norm, ...)
* Video (if ``--video``): saved to ``data/videos/expert_play/``
"""
from __future__ import annotations

import argparse
import os
import time
from typing import Any, Dict, List

import numpy as np
import torch
import gymnasium as gym

from forklift_expert import ForkliftExpertPolicy, ExpertConfig


# ---------------------------------------------------------------------------
# Helpers  (shared with collect_demos.py)
# ---------------------------------------------------------------------------
def _to_numpy(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _unwrap_obs(obs) -> np.ndarray:
    if isinstance(obs, dict):
        obs = obs.get("policy", obs.get("obs", next(iter(obs.values()))))
    return _to_numpy(obs)


def _to_action_tensor(act_np: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(act_np).float().to(device)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Play expert policy with optional video recording")
    ap.add_argument("--task", type=str, default="Isaac-Forklift-PalletInsertLift-Direct-v0")
    ap.add_argument("--num_envs", type=int, default=1,
                    help="Number of parallel envs (1 recommended for video)")
    ap.add_argument("--episodes", type=int, default=3,
                    help="Number of episodes to run")
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--seed", type=int, default=0)

    # Video
    ap.add_argument("--video", action="store_true",
                    help="Record video (works in headless mode)")
    ap.add_argument("--video_length", type=int, default=600,
                    help="Maximum steps to record per video")
    ap.add_argument("--video_dir", type=str, default="data/videos/expert_play",
                    help="Directory to save video files")

    # Obs / action spec paths
    ap.add_argument("--obs_spec", type=str,
                    default=os.path.join(
                        os.path.dirname(__file__), "..",
                        "forklift_expert", "obs_spec.json"))
    ap.add_argument("--action_spec", type=str,
                    default=os.path.join(
                        os.path.dirname(__file__), "..",
                        "forklift_expert", "action_spec.json"))

    # Enable cameras flag (for IsaacLab AppLauncher)
    ap.add_argument("--enable_cameras", action="store_true")
    args = ap.parse_args()

    # If video is requested, force enable_cameras (needed for offscreen render)
    if args.video:
        args.enable_cameras = True

    # ---- Load expert ----
    obs_spec = ForkliftExpertPolicy.load_json(args.obs_spec)
    action_spec = ForkliftExpertPolicy.load_json(args.action_spec)
    act_dim = int(action_spec.get("action_dim", 3))

    expert = ForkliftExpertPolicy(
        obs_spec=obs_spec, action_spec=action_spec, cfg=ExpertConfig()
    )

    # ---- Create env ----
    env_kwargs: Dict[str, Any] = {
        "headless": bool(args.headless),
        "num_envs": int(args.num_envs),
    }
    render_mode = "rgb_array" if args.video else None

    try:
        env = gym.make(args.task, render_mode=render_mode, **env_kwargs)
    except TypeError:
        # Fallback if env doesn't accept render_mode / kwargs
        try:
            env = gym.make(args.task, **env_kwargs)
        except TypeError:
            env = gym.make(args.task)

    # ---- Wrap for video recording ----
    if args.video:
        os.makedirs(args.video_dir, exist_ok=True)
        video_kwargs = {
            "video_folder": args.video_dir,
            "step_trigger": lambda step: step == 0,
            "video_length": args.video_length,
            "disable_logger": True,
        }
        print(f"[play_expert] recording video to {args.video_dir}")
        print(f"[play_expert] video_length={args.video_length}")
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # ---- Detect device ----
    obs_raw, info = env.reset(seed=args.seed)
    if isinstance(obs_raw, dict):
        _first = next(iter(obs_raw.values()))
    else:
        _first = obs_raw
    device = _first.device if isinstance(_first, torch.Tensor) else torch.device("cpu")

    obs_np = _unwrap_obs(obs_raw)
    n_envs = obs_np.shape[0] if obs_np.ndim > 1 else 1
    vec = obs_np.ndim > 1

    print(f"[play_expert] n_envs={n_envs}  vec={vec}  device={device}")
    print(f"[play_expert] running {args.episodes} episodes ...")

    # ---- Run episodes ----
    ep_done = 0
    step = 0
    t0 = time.time()

    while ep_done < args.episodes:
        # Compute action
        if vec:
            act_np = np.zeros((n_envs, act_dim), dtype=np.float32)
            dbg_list: List[Dict[str, Any]] = []
            for i in range(n_envs):
                a_i, dbg_i = expert.act(obs_np[i].astype(np.float32))
                act_np[i] = a_i
                dbg_list.append(dbg_i)
        else:
            a, dbg = expert.act(obs_np.astype(np.float32))
            act_np = a[None, :]
            dbg_list = [dbg]

        # Step env
        act_tensor = _to_action_tensor(act_np if vec else act_np[0], device)
        next_obs_raw, reward, terminated, truncated, step_info = env.step(act_tensor)
        next_obs_np = _unwrap_obs(next_obs_raw)

        term_np = _to_numpy(terminated).astype(np.bool_).reshape(-1)
        trunc_np = _to_numpy(truncated).astype(np.bool_).reshape(-1)
        done_np = np.logical_or(term_np, trunc_np)

        # Print debug info (env 0 only)
        d = dbg_list[0]
        if step % 10 == 0 or done_np[0]:
            print(
                f"  step={step:4d}  stage={d['stage']:10s}  "
                f"dist={d['dist_front']:.3f}  lat={d['lat']:.4f}  "
                f"yaw={d['yaw']:.4f}  ins={d['insert_norm']:.3f}  "
                f"drv={d['drive']:.3f}  str={d['steer']:.3f}  "
                f"lft={d['lift']:.3f}"
                + ("  *** DONE ***" if done_np[0] else "")
            )

        # Handle done
        for i in range(n_envs if vec else 1):
            if done_np[i]:
                ep_done += 1
                expert.reset()

        obs_np = next_obs_np
        step += 1

        # Stop after video_length if recording
        if args.video and step >= args.video_length:
            print(f"[play_expert] reached video_length={args.video_length}, stopping.")
            break

    elapsed = time.time() - t0
    print(f"[play_expert] done. episodes={ep_done}  steps={step}  elapsed={elapsed:.1f}s")
    if args.video:
        print(f"[play_expert] video saved to: {args.video_dir}/")

    env.close()


if __name__ == "__main__":
    main()
