"""
Collect demonstration data by running the rule-based expert policy in the
IsaacLab forklift environment.

Output: ``data/demos_YYYYMMDD_HHMMSS.npz``

Adapted for IsaacLab's vectorised ``DirectRLEnv`` which:
  - returns obs as ``{"policy": torch.Tensor}`` (dict format)
  - expects actions as ``torch.Tensor``
  - auto-resets individual envs on ``done`` (no manual reset needed)
"""
import argparse
import os
import time
import json
from typing import Any, Dict, List

import numpy as np

import torch

# gymnasium is available in the IsaacLab python env
import gymnasium as gym

from forklift_expert import ForkliftExpertPolicy, ExpertConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _to_numpy(x) -> np.ndarray:
    """Convert torch.Tensor / np.ndarray / scalar to numpy."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _unwrap_obs(obs) -> np.ndarray:
    """Handle both dict-style obs (IsaacLab) and raw tensor/array obs.

    IsaacLab's ``DirectRLEnv._get_observations()`` returns
    ``{"policy": tensor}``.  Standard gym envs return a flat array/tensor.
    """
    if isinstance(obs, dict):
        # Isaac Lab convention: "policy" key holds the actor observation
        obs = obs.get("policy", obs.get("obs", next(iter(obs.values()))))
    return _to_numpy(obs)


def _to_action_tensor(act_np: np.ndarray, device: torch.device) -> torch.Tensor:
    """Convert numpy action array to a torch tensor on the correct device."""
    return torch.from_numpy(act_np).float().to(device)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Collect expert demonstrations")
    ap.add_argument("--task", type=str,
                    default="Isaac-Forklift-PalletInsertLift-Direct-v0")
    ap.add_argument("--num_envs", type=int, default=64)
    ap.add_argument("--episodes", type=int, default=3000)
    ap.add_argument("--max_steps", type=int, default=400,
                    help="Max steps per episode (safety limit)")
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="")
    ap.add_argument("--obs_spec", type=str,
                    default=os.path.join(
                        os.path.dirname(__file__), "..",
                        "forklift_expert", "obs_spec.json"))
    ap.add_argument("--action_spec", type=str,
                    default=os.path.join(
                        os.path.dirname(__file__), "..",
                        "forklift_expert", "action_spec.json"))
    args = ap.parse_args()

    # ---- Load specs ----
    obs_spec = ForkliftExpertPolicy.load_json(args.obs_spec)
    action_spec = ForkliftExpertPolicy.load_json(args.action_spec)
    act_dim = int(action_spec.get("action_dim", 0))
    if act_dim <= 0:
        raise ValueError("action_dim must be > 0 in action_spec.json")

    np.random.seed(args.seed)

    # ---- Create env ----
    env_kwargs: Dict[str, Any] = {}
    env_kwargs["headless"] = bool(args.headless)
    env_kwargs["num_envs"] = int(args.num_envs)

    try:
        env = gym.make(args.task, **env_kwargs)
    except TypeError:
        env = gym.make(args.task)

    # ---- Initial reset ----
    obs_raw, info = env.reset(seed=args.seed)
    obs_np = _unwrap_obs(obs_raw)

    # Detect device (for sending actions back as torch tensors)
    if isinstance(obs_raw, dict):
        _first = next(iter(obs_raw.values()))
    else:
        _first = obs_raw
    if isinstance(_first, torch.Tensor):
        device = _first.device
    else:
        device = torch.device("cpu")

    # IsaacLab envs are always vectorised: (num_envs, obs_dim)
    if obs_np.ndim == 1:
        vec = False
        n_envs = 1
        obs_dim = obs_np.shape[0]
    else:
        vec = True
        n_envs = obs_np.shape[0]
        obs_dim = obs_np.shape[1]

    print(f"[collect] n_envs={n_envs}  obs_dim={obs_dim}  act_dim={act_dim}  "
          f"device={device}  vec={vec}")

    # ---- Per-env expert instances (each keeps its own rate-limit state) ----
    experts: List[ForkliftExpertPolicy] = []
    for _ in range(n_envs):
        experts.append(
            ForkliftExpertPolicy(
                obs_spec=obs_spec,
                action_spec=action_spec,
                cfg=ExpertConfig(),
            )
        )

    # ---- Storage (transition-level) ----
    obs_buf: List[np.ndarray] = []
    act_buf: List[np.ndarray] = []
    done_buf: List[np.ndarray] = []
    ep_id_buf: List[np.ndarray] = []
    env_id_buf: List[np.ndarray] = []
    info_buf: List[Dict[str, Any]] = []

    ep_ids = np.zeros((n_envs,), dtype=np.int64)
    ep_done_counts = 0

    t0 = time.time()
    steps = 0

    # ---- Main collection loop ----
    while ep_done_counts < args.episodes:
        # ---- Compute actions (per-env expert) ----
        if not vec:
            a, dbg = experts[0].act(obs_np.astype(np.float32))
            act_np = a[None, :]  # (1, act_dim)
            dbg_list = [dbg]
        else:
            act_np = np.zeros((n_envs, act_dim), dtype=np.float32)
            dbg_list = []
            for i in range(n_envs):
                a_i, dbg_i = experts[i].act(obs_np[i].astype(np.float32))
                act_np[i] = a_i
                dbg_list.append(dbg_i)

        # ---- Step ----
        # Isaac Lab expects torch tensors for actions
        act_tensor = _to_action_tensor(
            act_np if vec else act_np[0], device
        )
        next_obs_raw, reward, terminated, truncated, step_info = env.step(
            act_tensor
        )
        next_obs_np = _unwrap_obs(next_obs_raw)

        term_np = _to_numpy(terminated).astype(np.bool_).reshape(-1)
        trunc_np = _to_numpy(truncated).astype(np.bool_).reshape(-1)
        done_np = np.logical_or(term_np, trunc_np)

        # ---- Record transitions ----
        if not vec:
            obs_buf.append(obs_np.astype(np.float32)[None, :])
            act_buf.append(act_np.astype(np.float32))
            done_buf.append(done_np[:1])
            ep_id_buf.append(ep_ids.copy()[:1])
            env_id_buf.append(np.array([0], dtype=np.int64))
            info_buf.append(dbg_list[0])
        else:
            obs_buf.append(obs_np.astype(np.float32))
            act_buf.append(act_np.astype(np.float32))
            done_buf.append(done_np.astype(np.bool_))
            ep_id_buf.append(ep_ids.copy())
            env_id_buf.append(np.arange(n_envs, dtype=np.int64))
            # Summarise per-step debug info (averaged across envs)
            info_buf.append({
                "stage_counts": {
                    s: sum(1 for d in dbg_list if d["stage"] == s)
                    for s in ("docking", "insertion", "lift")
                },
                "dist_front_mean": float(
                    np.mean([d["dist_front"] for d in dbg_list])),
                "lat_mean": float(
                    np.mean([d["lat"] for d in dbg_list])),
                "yaw_mean": float(
                    np.mean([d["yaw"] for d in dbg_list])),
                "insert_norm_mean": float(
                    np.mean([d["insert_norm"] for d in dbg_list])),
            })

        # ---- Update episode counters; reset expert state on done ----
        for i in range(n_envs if vec else 1):
            if done_np[i]:
                ep_ids[i] += 1
                ep_done_counts += 1
                # Reset the per-env expert state (rate-limit, backoff)
                experts[i].reset()

        # IsaacLab auto-resets done envs; next_obs already contains the
        # new-episode observations for those envs.  No manual reset needed.
        obs_np = next_obs_np if vec else (
            next_obs_np if not bool(done_np[0])
            else _unwrap_obs(env.reset()[0])
        )

        steps += 1
        if steps % 50 == 0:
            elapsed = time.time() - t0
            eps = ep_done_counts / max(elapsed, 1e-6)
            print(f"[collect] episodes={ep_done_counts}/{args.episodes}  "
                  f"steps={steps}  elapsed={elapsed:.1f}s  "
                  f"eps/s={eps:.1f}")

        # Safety stop
        safety_limit = args.max_steps * max(
            1, args.episodes // max(1, n_envs)
        ) * 5
        if steps >= safety_limit:
            print("[collect] WARNING: reached safety step limit; stopping.")
            break

    # ---- Pack & save ----
    obs_arr = np.concatenate(obs_buf, axis=0)
    act_arr = np.concatenate(act_buf, axis=0)
    done_arr = np.concatenate(done_buf, axis=0)
    ep_id_arr = np.concatenate(ep_id_buf, axis=0)
    env_id_arr = np.concatenate(env_id_buf, axis=0)

    meta = {
        "task": args.task,
        "num_envs": args.num_envs,
        "episodes_target": args.episodes,
        "episodes_done": int(ep_done_counts),
        "obs_dim": int(obs_arr.shape[-1]),
        "action_dim": int(act_arr.shape[-1]),
        "seed": args.seed,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "obs_spec_path": os.path.abspath(args.obs_spec),
        "action_spec_path": os.path.abspath(args.action_spec),
    }

    out = args.out
    if not out:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        out = os.path.join("data", f"demos_{stamp}.npz")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)

    np.savez_compressed(
        out,
        obs=obs_arr.astype(np.float32),
        act=act_arr.astype(np.float32),
        done=done_arr.astype(np.bool_),
        episode_id=ep_id_arr.astype(np.int64),
        env_id=env_id_arr.astype(np.int64),
        meta=json.dumps(meta, ensure_ascii=False),
    )

    print(f"[collect] saved: {out}")
    print(f"[collect] transitions: {obs_arr.shape[0]}  "
          f"obs_dim={obs_arr.shape[1]}  act_dim={act_arr.shape[1]}")
    print(f"[collect] episodes_done: {ep_done_counts}")


if __name__ == "__main__":
    main()
