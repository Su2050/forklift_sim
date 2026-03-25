"""Unified deterministic evaluation for exp8.3 near-field checkpoints.

This script runs a fixed number of rollout batches for one checkpoint and saves:
  - summary JSON
  - per-episode CSV

The evaluation is intentionally aligned with recent exp8.3 comparisons:
  - near-field
  - 256x256 camera
  - deterministic policy
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path

parser = argparse.ArgumentParser(description="Unified deterministic eval for exp8.3 checkpoints")
parser.add_argument("--task", type=str, default="Isaac-Forklift-PalletInsertLift-Direct-v0")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--label", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument(
    "--rollouts",
    type=int,
    default=2,
    help="Number of parallel rollout batches. Total evaluated episodes = num_envs * rollouts.",
)
parser.add_argument("--seed", type=int, default=20260325)
parser.add_argument("--output_dir", type=str, default="/home/uniubi/projects/forklift_sim/outputs/exp83_eval")

from isaaclab.app import AppLauncher

AppLauncher.add_app_launcher_args(parser)
args, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

from rsl_rl.runners import OnPolicyRunner

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

from isaaclab_tasks.direct.forklift_pallet_insert_lift.hold_logic import compute_hold_logic


def _quat_to_yaw(q: torch.Tensor) -> torch.Tensor:
    w, x, y, z = q.unbind(-1)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return torch.atan2(siny_cosp, cosy_cosp)


def _compute_step_state(raw_env):
    pallet_pos = raw_env.pallet.data.root_pos_w
    tip = raw_env._compute_fork_tip()
    fork_center = raw_env._compute_fork_center()

    robot_yaw = _quat_to_yaw(raw_env.robot.data.root_quat_w)
    pallet_yaw = _quat_to_yaw(raw_env.pallet.data.root_quat_w)
    yaw_err = torch.atan2(torch.sin(robot_yaw - pallet_yaw), torch.cos(robot_yaw - pallet_yaw))
    yaw_err_deg = torch.abs(yaw_err) * (180.0 / math.pi)

    cp = torch.cos(pallet_yaw)
    sp = torch.sin(pallet_yaw)
    u_in = torch.stack([cp, sp], dim=-1)
    v_lat = torch.stack([-sp, cp], dim=-1)

    rel_tip = tip[:, :2] - pallet_pos[:, :2]
    s_tip = torch.sum(rel_tip * u_in, dim=-1)
    s_front = -0.5 * raw_env.cfg.pallet_depth_m
    dist_front = torch.clamp(s_front - s_tip, min=0.0)
    insert_depth = torch.clamp(s_tip - s_front, min=0.0)

    rel_fc = fork_center[:, :2] - pallet_pos[:, :2]
    center_y_err = torch.abs(torch.sum(rel_fc * v_lat, dim=-1))
    tip_y_err = torch.abs(torch.sum(rel_tip * v_lat, dim=-1))
    lift_height = tip[:, 2] - raw_env._fork_tip_z0

    hold_state = compute_hold_logic(
        center_y_err=center_y_err,
        yaw_err_deg=yaw_err_deg,
        insert_depth=insert_depth,
        lift_height=lift_height,
        tip_y_err=tip_y_err,
        dist_front=dist_front,
        hold_counter=raw_env._hold_counter,
        cfg=raw_env._hold_logic_cfg,
    )

    pallet_init_pos_xy = torch.tensor(raw_env.cfg.pallet_cfg.init_state.pos[:2], device=raw_env.device)
    pallet_disp_xy = torch.norm(pallet_pos[:, :2] - pallet_init_pos_xy, dim=-1)
    push_free = pallet_disp_xy < raw_env.cfg.push_free_disp_thresh_m

    inserted = hold_state.insert_entry
    hold_entry = hold_state.hold_entry
    clean_insert_ready = hold_entry & push_free
    dirty_insert = inserted & ~push_free
    success = raw_env._hold_counter >= raw_env._hold_steps

    return {
        "inserted": inserted,
        "push_free": push_free,
        "inserted_push_free": inserted & push_free,
        "hold_entry": hold_entry,
        "clean_insert_ready": clean_insert_ready,
        "dirty_insert": dirty_insert,
        "success": success,
        "pallet_disp_xy": pallet_disp_xy,
    }


@hydra_task_config(args.task, args.agent)
def main(env_cfg, agent_cfg):
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.seed = args.seed
    env_cfg.sim.device = "cuda:0"
    env_cfg.use_camera = True
    env_cfg.use_asymmetric_critic = True
    env_cfg.stage_1_mode = True
    env_cfg.camera_width = 256
    env_cfg.camera_height = 256

    env = gym.make(args.task, cfg=env_cfg)
    env_wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    raw_env = env.unwrapped

    runner = OnPolicyRunner(env_wrapped, agent_cfg.to_dict(), log_dir=None, device="cuda:0")
    runner.load(args.checkpoint)
    try:
        policy_nn = runner.alg.policy
    except AttributeError:
        policy_nn = runner.alg.actor_critic

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / f"{args.label}_summary.json"
    episode_path = output_dir / f"{args.label}_episodes.csv"

    global_counts = {
        "active_env_steps": 0.0,
        "step_inserted_sum": 0.0,
        "step_inserted_push_free_sum": 0.0,
        "step_hold_entry_sum": 0.0,
        "step_clean_insert_ready_sum": 0.0,
        "step_dirty_insert_sum": 0.0,
        "step_success_sum": 0.0,
        "step_pallet_disp_xy_sum": 0.0,
    }
    episode_rows: list[dict[str, float | int | str]] = []

    obs = env_wrapped.get_observations()
    max_steps = int(raw_env.max_episode_length) + 5

    for rollout in range(args.rollouts):
        done_mask = torch.zeros(args.num_envs, dtype=torch.bool, device=raw_env.device)
        ep_inserted = torch.zeros(args.num_envs, dtype=torch.bool, device=raw_env.device)
        ep_inserted_push_free = torch.zeros(args.num_envs, dtype=torch.bool, device=raw_env.device)
        ep_hold_entry = torch.zeros(args.num_envs, dtype=torch.bool, device=raw_env.device)
        ep_clean_insert_ready = torch.zeros(args.num_envs, dtype=torch.bool, device=raw_env.device)
        ep_dirty_insert = torch.zeros(args.num_envs, dtype=torch.bool, device=raw_env.device)
        ep_success = torch.zeros(args.num_envs, dtype=torch.bool, device=raw_env.device)
        ep_max_pallet_disp = torch.zeros(args.num_envs, dtype=torch.float32, device=raw_env.device)
        ep_max_hold_counter = torch.zeros(args.num_envs, dtype=torch.float32, device=raw_env.device)
        ep_len = torch.zeros(args.num_envs, dtype=torch.int32, device=raw_env.device)
        ep_done_len = torch.full((args.num_envs,), -1, dtype=torch.int32, device=raw_env.device)

        step_count = 0
        while not done_mask.all() and step_count < max_steps:
            with torch.inference_mode():
                actions = policy_nn.act_inference(obs)
                obs, _, dones, _ = env_wrapped.step(actions)

            step_count += 1
            state = _compute_step_state(raw_env)
            active = ~done_mask
            active_count = int(active.sum().item())

            if active_count > 0:
                global_counts["active_env_steps"] += active_count
                global_counts["step_inserted_sum"] += float(state["inserted"][active].sum().item())
                global_counts["step_inserted_push_free_sum"] += float(state["inserted_push_free"][active].sum().item())
                global_counts["step_hold_entry_sum"] += float(state["hold_entry"][active].sum().item())
                global_counts["step_clean_insert_ready_sum"] += float(state["clean_insert_ready"][active].sum().item())
                global_counts["step_dirty_insert_sum"] += float(state["dirty_insert"][active].sum().item())
                global_counts["step_success_sum"] += float(state["success"][active].sum().item())
                global_counts["step_pallet_disp_xy_sum"] += float(state["pallet_disp_xy"][active].sum().item())

                ep_inserted[active] |= state["inserted"][active]
                ep_inserted_push_free[active] |= state["inserted_push_free"][active]
                ep_hold_entry[active] |= state["hold_entry"][active]
                ep_clean_insert_ready[active] |= state["clean_insert_ready"][active]
                ep_dirty_insert[active] |= state["dirty_insert"][active]
                ep_success[active] |= state["success"][active]
                ep_max_pallet_disp[active] = torch.maximum(ep_max_pallet_disp[active], state["pallet_disp_xy"][active])
                ep_max_hold_counter[active] = torch.maximum(ep_max_hold_counter[active], raw_env._hold_counter[active])
                ep_len[active] += 1

            if isinstance(dones, torch.Tensor):
                newly_done = dones.bool() & ~done_mask
            else:
                newly_done = torch.tensor(dones, dtype=torch.bool, device=raw_env.device) & ~done_mask
            ep_done_len[newly_done] = ep_len[newly_done]
            done_mask |= newly_done

        for env_id in range(args.num_envs):
            ep_length = int(ep_done_len[env_id].item())
            if ep_length < 0:
                ep_length = int(ep_len[env_id].item())
            timeout = int(ep_length >= int(raw_env.max_episode_length) - 1 and not bool(ep_success[env_id].item()))
            episode_rows.append(
                {
                    "label": args.label,
                    "rollout": rollout,
                    "env_id": env_id,
                    "success": int(ep_success[env_id].item()),
                    "ever_inserted": int(ep_inserted[env_id].item()),
                    "ever_inserted_push_free": int(ep_inserted_push_free[env_id].item()),
                    "ever_hold_entry": int(ep_hold_entry[env_id].item()),
                    "ever_clean_insert_ready": int(ep_clean_insert_ready[env_id].item()),
                    "ever_dirty_insert": int(ep_dirty_insert[env_id].item()),
                    "max_pallet_disp_xy": float(ep_max_pallet_disp[env_id].item()),
                    "max_hold_counter": float(ep_max_hold_counter[env_id].item()),
                    "episode_length": ep_length,
                    "timeout": timeout,
                }
            )

        obs = env_wrapped.get_observations()

    total_episodes = len(episode_rows)
    active_env_steps = max(global_counts["active_env_steps"], 1.0)

    def mean_bool(key: str) -> float:
        return sum(int(row[key]) for row in episode_rows) / max(total_episodes, 1)

    def mean_float(key: str) -> float:
        return sum(float(row[key]) for row in episode_rows) / max(total_episodes, 1)

    def p90_float(key: str) -> float:
        vals = sorted(float(row[key]) for row in episode_rows)
        if not vals:
            return 0.0
        idx = min(len(vals) - 1, max(0, math.ceil(0.9 * len(vals)) - 1))
        return vals[idx]

    summary = {
        "label": args.label,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "num_envs": args.num_envs,
        "rollouts": args.rollouts,
        "total_episodes": total_episodes,
        "eval_seed": args.seed,
        "success_rate_ep": mean_bool("success"),
        "ever_inserted_rate": mean_bool("ever_inserted"),
        "ever_inserted_push_free_rate": mean_bool("ever_inserted_push_free"),
        "ever_hold_entry_rate": mean_bool("ever_hold_entry"),
        "ever_clean_insert_ready_rate": mean_bool("ever_clean_insert_ready"),
        "ever_dirty_insert_rate": mean_bool("ever_dirty_insert"),
        "timeout_frac": mean_bool("timeout"),
        "mean_episode_length": mean_float("episode_length"),
        "mean_max_pallet_disp_xy": mean_float("max_pallet_disp_xy"),
        "p90_max_pallet_disp_xy": p90_float("max_pallet_disp_xy"),
        "mean_max_hold_counter": mean_float("max_hold_counter"),
        "p90_max_hold_counter": p90_float("max_hold_counter"),
        "step_mean_frac_inserted": global_counts["step_inserted_sum"] / active_env_steps,
        "step_mean_frac_inserted_push_free": global_counts["step_inserted_push_free_sum"] / active_env_steps,
        "step_mean_frac_hold_entry": global_counts["step_hold_entry_sum"] / active_env_steps,
        "step_mean_frac_clean_insert_ready": global_counts["step_clean_insert_ready_sum"] / active_env_steps,
        "step_mean_frac_dirty_insert": global_counts["step_dirty_insert_sum"] / active_env_steps,
        "step_mean_frac_success": global_counts["step_success_sum"] / active_env_steps,
        "step_mean_pallet_disp_xy": global_counts["step_pallet_disp_xy_sum"] / active_env_steps,
    }

    with episode_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(episode_rows[0].keys()))
        writer.writeheader()
        writer.writerows(episode_rows)

    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"[INFO] Wrote summary to: {summary_path}")
    print(f"[INFO] Wrote episodes to: {episode_path}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
