"""
Play the rule-based expert policy in the IsaacLab forklift environment and
record a video (headless-compatible).

Usage
-----
.. code-block:: bash

    ./isaaclab.sh -p forklift_expert_policy_project/scripts/play_expert.py \\
        --task Isaac-Forklift-PalletInsertLift-Direct-v0 \\
        --num_envs 1 --headless \\
        --video --video_length 600

    # Batch stress-test (no video, quiet mode):
    ./isaaclab.sh -p .../play_expert.py \\
        --task Isaac-Forklift-PalletInsertLift-Direct-v0 \\
        --num_envs 1 --headless \\
        --episodes 10 --seed 42 --quiet \\
        --log_file /path/to/log.log

Output
------
* Console (stderr): per-step debug info or episode summaries
* Video (if ``--video``): saved to ``data/videos/expert_play/``
* Log file (if ``--log_file``): episode-level data for batch analysis
"""
from __future__ import annotations

# ===========================================================================
# Step 1: Parse arguments and launch AppLauncher BEFORE any IsaacLab imports
# ===========================================================================
import argparse
import os
import sys

parser = argparse.ArgumentParser(description="Play expert policy with optional video recording")
parser.add_argument("--task", type=str, default="Isaac-Forklift-PalletInsertLift-Direct-v0")
parser.add_argument("--num_envs", type=int, default=1,
                    help="Number of parallel envs (1 recommended for video)")
parser.add_argument("--episodes", type=int, default=3,
                    help="Number of episodes to run")
parser.add_argument("--seed", type=int, default=0)

# Video
parser.add_argument("--video", action="store_true",
                    help="Record video (works in headless mode)")
parser.add_argument("--video_length", type=int, default=600,
                    help="Maximum steps to record per video")
parser.add_argument("--video_dir", type=str, default="data/videos/expert_play",
                    help="Directory to save video files")

# Stress-test / batch mode
parser.add_argument("--quiet", action="store_true",
                    help="Suppress per-step logs, only print episode summaries")
parser.add_argument("--log_file", type=str, default=None,
                    help="Path to write episode logs (also writes to stderr)")

# Obs / action spec paths
parser.add_argument("--obs_spec", type=str,
                    default=os.path.join(
                        os.path.dirname(__file__), "..",
                        "forklift_expert", "obs_spec.json"))
parser.add_argument("--action_spec", type=str,
                    default=os.path.join(
                        os.path.dirname(__file__), "..",
                        "forklift_expert", "action_spec.json"))

# AppLauncher adds --headless, --enable_cameras, etc.
from isaaclab.app import AppLauncher
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# If video is requested, force enable_cameras (needed for offscreen render)
if args.video:
    args.enable_cameras = True

# Launch the simulation application
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# ===========================================================================
# Step 2: Now safe to import IsaacLab / gymnasium / custom modules
# ===========================================================================
import time
from typing import Any, Dict, List

import numpy as np
import torch
import gymnasium as gym

import isaaclab_tasks  # noqa: F401 -- triggers gym.register()
from isaaclab_tasks.utils import parse_env_cfg

from forklift_expert import ForkliftExpertPolicy, ExpertConfig


# ---------------------------------------------------------------------------
# Logging helper -- Kit/Omniverse may redirect stdout, so we write to stderr
# ---------------------------------------------------------------------------
_log_file_handle = None


def _init_log() -> None:
    global _log_file_handle
    if args.log_file:
        os.makedirs(os.path.dirname(os.path.abspath(args.log_file)), exist_ok=True)
        _log_file_handle = open(args.log_file, "w")


def log(msg: str) -> None:
    """Write message to stderr (always visible) and optional log file."""
    sys.stderr.write(msg + "\n")
    sys.stderr.flush()
    if _log_file_handle:
        _log_file_handle.write(msg + "\n")
        _log_file_handle.flush()


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
    _init_log()

    # ---- Load expert ----
    obs_spec = ForkliftExpertPolicy.load_json(args.obs_spec)
    action_spec = ForkliftExpertPolicy.load_json(args.action_spec)
    act_dim = int(action_spec.get("action_dim", 3))

    expert = ForkliftExpertPolicy(
        obs_spec=obs_spec, action_spec=action_spec, cfg=ExpertConfig()
    )

    # ---- Create env via IsaacLab cfg ----
    env_cfg = parse_env_cfg(args.task, num_envs=args.num_envs, use_fabric=True)
    render_mode = "rgb_array" if args.video else None

    env = gym.make(args.task, cfg=env_cfg, render_mode=render_mode)

    # ---- Wrap for video recording ----
    if args.video:
        os.makedirs(args.video_dir, exist_ok=True)
        video_kwargs = {
            "video_folder": args.video_dir,
            "step_trigger": lambda step: step == 0,
            "video_length": args.video_length,
            "disable_logger": True,
        }
        log(f"[play_expert] recording video to {args.video_dir}")
        log(f"[play_expert] video_length={args.video_length}")
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

    log(f"[play_expert] n_envs={n_envs}  vec={vec}  device={device}")
    log(f"[play_expert] running {args.episodes} episodes ...")

    # ---- Episode tracking ----
    ep_done = 0
    step = 0
    ep_step = 0
    t0 = time.time()

    # Per-episode accumulators
    stage_counts: Dict[str, int] = {}
    ep_min_lat = float("inf")
    ep_max_ins = 0.0
    ep_init_dist = 0.0
    ep_init_lat = 0.0
    ep_init_yaw = 0.0
    ep_init_dy = 0.0
    ep_vf_zero_count = 0

    # Summary across all episodes
    ep_summaries: List[Dict[str, Any]] = []

    def _reset_ep_accumulators(dbg: Dict[str, Any]) -> None:
        nonlocal stage_counts, ep_min_lat, ep_max_ins, ep_step
        nonlocal ep_init_dist, ep_init_lat, ep_init_yaw, ep_init_dy, ep_vf_zero_count
        stage_counts = {}
        ep_min_lat = abs(dbg.get("lat", 0.5))
        ep_max_ins = dbg.get("insert_norm", 0.0)
        ep_init_dist = dbg.get("dist_front", 0.0)
        ep_init_lat = dbg.get("lat", 0.0)
        ep_init_yaw = dbg.get("yaw", 0.0)
        ep_init_dy = dbg.get("d_y", 0.0)
        ep_step = 0
        ep_vf_zero_count = 0

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

        # Track per-step data (env 0)
        d = dbg_list[0]
        stage_name = d["stage"]
        stage_counts[stage_name] = stage_counts.get(stage_name, 0) + 1
        ep_min_lat = min(ep_min_lat, abs(d["lat"]))
        ep_max_ins = max(ep_max_ins, d["insert_norm"])
        if abs(d["v_forward"]) < 0.01:
            ep_vf_zero_count += 1

        # Initialize accumulators on first step of episode
        if ep_step == 0:
            _reset_ep_accumulators(d)

        ep_step += 1

        # Print per-step debug info (env 0 only)
        if not args.quiet and (step % 10 == 0 or done_np[0]):
            log(
                f"  step={step:4d}  stage={d['stage']:10s}  "
                f"dist={d['dist_front']:.3f}  lat={d['lat']:.4f}(clip={d['lat_clipped']:.3f})  "
                f"yaw={d['yaw']:.4f}  ins={d['insert_norm']:.3f}  "
                f"drv={d['drive']:.3f}  str={d['steer']:.3f}  "
                f"lft={d['lift']:.3f}  vf={d['v_forward']:.3f}  "
                f"d_y={d['d_y']:.3f}"
                + ("  *** DONE ***" if done_np[0] else "")
            )

        # Handle done
        for i in range(n_envs if vec else 1):
            if done_np[i]:
                reason = "terminated" if term_np[i] else "truncated"
                final_stage = d["stage"]
                summary = {
                    "ep": ep_done,
                    "steps": ep_step,
                    "reason": reason,
                    "final_stage": final_stage,
                    "init_dist": ep_init_dist,
                    "init_lat": ep_init_lat,
                    "init_yaw": ep_init_yaw,
                    "init_dy": ep_init_dy,
                    "min_lat": ep_min_lat,
                    "max_ins": ep_max_ins,
                    "vf_zero_pct": ep_vf_zero_count / max(ep_step, 1) * 100,
                    "stages": dict(stage_counts),
                    "final_dist": d["dist_front"],
                    "final_lat": d["lat"],
                    "final_yaw": d["yaw"],
                    "final_ins": d["insert_norm"],
                }
                ep_summaries.append(summary)

                log(
                    f"  [EP {ep_done:3d}] {ep_step:4d} steps  "
                    f"reason={reason:10s}  final_stage={final_stage:10s}  "
                    f"init(d={ep_init_dist:.2f} lat={ep_init_lat:.3f} yaw={ep_init_yaw:.3f} dy={ep_init_dy:.3f})  "
                    f"end(d={d['dist_front']:.2f} lat={d['lat']:.3f} ins={d['insert_norm']:.3f})  "
                    f"min_lat={ep_min_lat:.3f}  max_ins={ep_max_ins:.3f}  "
                    f"vf0={ep_vf_zero_count}/{ep_step}({summary['vf_zero_pct']:.0f}%)  "
                    f"stages={stage_counts}"
                )

                ep_done += 1
                expert.reset()
                stage_counts = {}
                ep_step = 0
                ep_vf_zero_count = 0

        obs_np = next_obs_np
        step += 1

        # Stop after video_length if recording
        if args.video and step >= args.video_length:
            log(f"[play_expert] reached video_length={args.video_length}, stopping.")
            break

    elapsed = time.time() - t0
    log(f"\n[play_expert] done. episodes={ep_done}  steps={step}  elapsed={elapsed:.1f}s")
    if args.video:
        log(f"[play_expert] video saved to: {args.video_dir}/")

    # ---- Print aggregate summary ----
    if ep_summaries:
        n = len(ep_summaries)
        n_term = sum(1 for s in ep_summaries if s["reason"] == "terminated")
        n_trunc = n - n_term
        avg_steps = sum(s["steps"] for s in ep_summaries) / n
        avg_vf0 = sum(s["vf_zero_pct"] for s in ep_summaries) / n
        avg_max_ins = sum(s["max_ins"] for s in ep_summaries) / n
        avg_min_lat = sum(s["min_lat"] for s in ep_summaries) / n
        n_reached_ins = sum(1 for s in ep_summaries if s["max_ins"] >= 0.1)
        n_reached_lift = sum(1 for s in ep_summaries if s["max_ins"] >= 0.75)

        n_retreat_stuck = 0
        for s in ep_summaries:
            retreat_pct = s["stages"].get("retreat", 0) / max(s["steps"], 1) * 100
            if retreat_pct > 80:
                n_retreat_stuck += 1

        log(f"\n{'='*70}")
        log(f"  AGGREGATE SUMMARY  ({n} episodes, seed={args.seed})")
        log(f"{'='*70}")
        log(f"  terminated: {n_term}/{n}  truncated(timeout): {n_trunc}/{n}")
        log(f"  avg steps/ep: {avg_steps:.0f}")
        log(f"  avg vf=0 pct: {avg_vf0:.1f}%")
        log(f"  avg min |lat|: {avg_min_lat:.3f}")
        log(f"  avg max ins:  {avg_max_ins:.3f}")
        log(f"  reached insertion (ins>=0.1): {n_reached_ins}/{n}")
        log(f"  reached lift (ins>=0.75):     {n_reached_lift}/{n}")
        log(f"  stuck in retreat (>80%):      {n_retreat_stuck}/{n}")
        log(f"{'='*70}")

    if _log_file_handle:
        _log_file_handle.close()

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
