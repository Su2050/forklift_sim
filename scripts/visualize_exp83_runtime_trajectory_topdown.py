"""Visualize runtime-generated Exp8.3 reference trajectories in top-down view.

This script answers a missing validation question:
  what trajectory is the *current env* actually generating at reset time?

By default it:
  - fixes stage1 reset to one or more (x, y, yaw) cases
  - calls the real env reset path
  - reads raw_env._traj_pts / _traj_tangents / pallet pose / fork center
  - renders a top-down PNG for each case

Optionally, if a checkpoint is provided, it also rolls out the deterministic
policy and overlays the executed fork-center path on top of the reference path.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser(description="Top-down visualization for runtime Exp8.3 reference trajectory")
parser.add_argument("--task", type=str, default="Isaac-Forklift-PalletInsertLift-Direct-v0")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--checkpoint", type=str, default=None)
parser.add_argument("--label", type=str, default="exp83_runtime_traj")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--seed", type=int, default=20260328)
parser.add_argument("--x_root", type=float, default=-3.40)
parser.add_argument("--y_values", type=str, default="-0.08,0.0,0.08")
parser.add_argument("--yaw_deg_values", type=str, default="-3,0,3")
parser.add_argument("--rollout_steps", type=int, default=0)
parser.add_argument("--force_zero_steer", action="store_true")
parser.add_argument("--tangent_stride", type=int, default=4)
parser.add_argument("--axis_pad_m", type=float, default=0.40)
parser.add_argument(
    "--output_dir",
    type=str,
    default="/home/uniubi/projects/forklift_sim/outputs/exp83_runtime_traj_topdown",
)

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


def _quat_to_yaw(q: torch.Tensor) -> torch.Tensor:
    w, x, y, z = q.unbind(-1)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return torch.atan2(siny_cosp, cosy_cosp)


def _parse_list(spec: str) -> list[float]:
    return [float(v.strip()) for v in spec.split(",") if v.strip()]


def _set_fixed_stage1_reset(raw_env, seed_val: int, x_root: float, y_m: float, yaw_deg: float) -> None:
    torch.manual_seed(seed_val)
    np.random.seed(seed_val)
    raw_env.cfg.stage1_init_x_min_m = x_root
    raw_env.cfg.stage1_init_x_max_m = x_root
    raw_env.cfg.stage1_init_y_min_m = y_m
    raw_env.cfg.stage1_init_y_max_m = y_m
    raw_env.cfg.stage1_init_yaw_deg_min = yaw_deg
    raw_env.cfg.stage1_init_yaw_deg_max = yaw_deg
    with torch.inference_mode():
        all_ids = torch.arange(raw_env.num_envs, device=raw_env.device)
        raw_env._reset_idx(all_ids)


def _compute_case_geometry(raw_env, env_id: int = 0) -> dict[str, object]:
    pallet_pos = raw_env.pallet.data.root_pos_w[env_id, :2].detach().cpu().numpy()
    pallet_yaw = float(_quat_to_yaw(raw_env.pallet.data.root_quat_w[env_id : env_id + 1])[0].item())
    root_pos = raw_env.robot.data.root_pos_w[env_id, :2].detach().cpu().numpy()
    root_yaw = float(_quat_to_yaw(raw_env.robot.data.root_quat_w[env_id : env_id + 1])[0].item())
    fork_center = raw_env._compute_fork_center()[env_id, :2].detach().cpu().numpy()
    traj_pts = raw_env._traj_pts[env_id].detach().cpu().numpy()
    traj_tangents = raw_env._traj_tangents[env_id].detach().cpu().numpy()
    traj_s = raw_env._traj_s_norm[env_id].detach().cpu().numpy()

    u_in = np.array([math.cos(pallet_yaw), math.sin(pallet_yaw)], dtype=np.float32)
    v_lat = np.array([-math.sin(pallet_yaw), math.cos(pallet_yaw)], dtype=np.float32)
    s_goal = float(raw_env._exp83_traj_goal_s())
    p_goal = pallet_pos + s_goal * u_in
    p_pre = pallet_pos + (s_goal - raw_env.cfg.traj_pre_dist_m) * u_in

    return {
        "pallet_pos": pallet_pos,
        "pallet_yaw": pallet_yaw,
        "root_pos": root_pos,
        "root_yaw": root_yaw,
        "fork_center": fork_center,
        "traj_pts": traj_pts,
        "traj_tangents": traj_tangents,
        "traj_s_norm": traj_s,
        "p_goal": p_goal,
        "p_pre": p_pre,
        "u_in": u_in,
        "v_lat": v_lat,
        "pallet_depth_m": float(raw_env.cfg.pallet_depth_m),
    }


def _run_rollout_overlay(env_wrapped, raw_env, policy_nn, rollout_steps: int, force_zero_steer: bool) -> dict[str, object]:
    obs = env_wrapped.get_observations()
    fork_centers: list[np.ndarray] = []
    root_positions: list[np.ndarray] = []
    actions: list[tuple[float, float]] = []
    dones_seen = False

    for _ in range(rollout_steps):
        with torch.inference_mode():
            raw_actions = policy_nn.act_inference(obs)
            applied_actions = raw_actions.clone()
            if force_zero_steer:
                applied_actions[:, 1] = 0.0
            obs, _, dones, _ = env_wrapped.step(applied_actions)

        fork_centers.append(raw_env._compute_fork_center()[0, :2].detach().cpu().numpy())
        root_positions.append(raw_env.robot.data.root_pos_w[0, :2].detach().cpu().numpy())
        actions.append(
            (
                float(raw_actions[0, 0].item()),
                float(raw_actions[0, 1].item()),
            )
        )
        if isinstance(dones, torch.Tensor):
            if bool(dones[0].item()):
                dones_seen = True
                break
        else:
            if bool(dones[0]):
                dones_seen = True
                break

    return {
        "fork_center_path": np.asarray(fork_centers, dtype=np.float32),
        "root_path": np.asarray(root_positions, dtype=np.float32),
        "raw_actions": actions,
        "done_early": dones_seen,
    }


def _write_traj_csv(path: Path, geom: dict[str, object]) -> None:
    traj_pts = np.asarray(geom["traj_pts"])
    traj_tangents = np.asarray(geom["traj_tangents"])
    traj_s = np.asarray(geom["traj_s_norm"])
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["idx", "x", "y", "tx", "ty", "s_norm"],
        )
        writer.writeheader()
        for idx in range(traj_pts.shape[0]):
            writer.writerow(
                {
                    "idx": idx,
                    "x": float(traj_pts[idx, 0]),
                    "y": float(traj_pts[idx, 1]),
                    "tx": float(traj_tangents[idx, 0]),
                    "ty": float(traj_tangents[idx, 1]),
                    "s_norm": float(traj_s[idx]),
                }
            )


def _to_pallet_frame(points_xy: np.ndarray, pallet_pos: np.ndarray, u_in: np.ndarray, v_lat: np.ndarray) -> np.ndarray:
    rel = points_xy - pallet_pos.reshape(1, 2)
    s = rel @ u_in.reshape(2, 1)
    lat = rel @ v_lat.reshape(2, 1)
    return np.concatenate([s, lat], axis=1)


def _scalar_geometry(geom: dict[str, object]) -> dict[str, float]:
    pallet_pos = np.asarray(geom["pallet_pos"])
    fork_center = np.asarray(geom["fork_center"])
    p_pre = np.asarray(geom["p_pre"])
    p_goal = np.asarray(geom["p_goal"])
    u_in = np.asarray(geom["u_in"])
    v_lat = np.asarray(geom["v_lat"])
    s_goal = float(np.dot(p_goal - pallet_pos, u_in))
    s_pre = float(np.dot(p_pre - pallet_pos, u_in))
    s_start = float(np.dot(fork_center - pallet_pos, u_in))
    y_start = float(np.dot(fork_center - pallet_pos, v_lat))
    return {
        "s_goal": s_goal,
        "s_pre": s_pre,
        "s_start": s_start,
        "delta_s": s_start - s_pre,
        "y_start": y_start,
    }


def _plot_case(
    png_path: Path,
    geom: dict[str, object],
    *,
    label: str,
    y_m: float,
    yaw_deg: float,
    tangent_stride: int,
    axis_pad_m: float,
    rollout: dict[str, object] | None,
    force_zero_steer: bool,
) -> None:
    pallet_pos = np.asarray(geom["pallet_pos"])
    root_pos = np.asarray(geom["root_pos"])
    fork_center = np.asarray(geom["fork_center"])
    traj_pts = np.asarray(geom["traj_pts"])
    traj_tangents = np.asarray(geom["traj_tangents"])
    p_goal = np.asarray(geom["p_goal"])
    p_pre = np.asarray(geom["p_pre"])
    u_in = np.asarray(geom["u_in"])
    v_lat = np.asarray(geom["v_lat"])
    pallet_depth_m = float(geom["pallet_depth_m"])
    root_yaw = float(geom["root_yaw"])
    scalar = _scalar_geometry(geom)
    s_goal = scalar["s_goal"]
    s_pre = scalar["s_pre"]
    s_start = scalar["s_start"]
    y_start = scalar["y_start"]
    delta_s = scalar["delta_s"]

    fig, (ax_world, ax_pf) = plt.subplots(1, 2, figsize=(14, 6.8))

    # Reference trajectory.
    ax_world.plot(traj_pts[:, 0], traj_pts[:, 1], color="tab:blue", linewidth=2.5, label="reference traj")
    tangent_ix = np.arange(0, traj_pts.shape[0], max(int(tangent_stride), 1))
    ax_world.quiver(
        traj_pts[tangent_ix, 0],
        traj_pts[tangent_ix, 1],
        traj_tangents[tangent_ix, 0],
        traj_tangents[tangent_ix, 1],
        angles="xy",
        scale_units="xy",
        scale=6.0,
        width=0.003,
        color="tab:blue",
        alpha=0.55,
    )

    # Pallet axes / landmarks.
    axis_len = 0.9
    lat_len = 0.6
    ax_world.arrow(
        pallet_pos[0],
        pallet_pos[1],
        u_in[0] * axis_len,
        u_in[1] * axis_len,
        head_width=0.04,
        length_includes_head=True,
        color="tab:red",
        linewidth=2.0,
        label="pallet insert axis",
    )
    ax_world.arrow(
        pallet_pos[0],
        pallet_pos[1],
        v_lat[0] * lat_len,
        v_lat[1] * lat_len,
        head_width=0.04,
        length_includes_head=True,
        color="tab:orange",
        linewidth=1.5,
        label="pallet lateral axis",
    )
    front_center = pallet_pos + (-0.5 * pallet_depth_m) * u_in
    front_a = front_center + 0.35 * v_lat
    front_b = front_center - 0.35 * v_lat
    ax_world.plot([front_a[0], front_b[0]], [front_a[1], front_b[1]], color="tab:red", linewidth=3.0, alpha=0.8)

    # Key points.
    ax_world.scatter([traj_pts[0, 0]], [traj_pts[0, 1]], color="tab:green", s=70, label="traj start")
    ax_world.scatter([p_pre[0]], [p_pre[1]], color="tab:purple", s=60, label="p_pre")
    ax_world.scatter([p_goal[0]], [p_goal[1]], color="tab:brown", s=60, label="p_goal")
    ax_world.scatter([pallet_pos[0]], [pallet_pos[1]], color="tab:red", s=60, label="pallet center")

    # Robot initial pose.
    ax_world.scatter([root_pos[0]], [root_pos[1]], color="black", s=50, label="root start")
    ax_world.scatter([fork_center[0]], [fork_center[1]], color="tab:green", s=45, marker="x", label="fork center start")
    ax_world.arrow(
        root_pos[0],
        root_pos[1],
        math.cos(root_yaw) * 0.5,
        math.sin(root_yaw) * 0.5,
        head_width=0.05,
        length_includes_head=True,
        color="black",
        linewidth=2.0,
    )

    if rollout is not None:
        fc_path = np.asarray(rollout["fork_center_path"])
        root_path = np.asarray(rollout["root_path"])
        if fc_path.size > 0:
            ax_world.plot(fc_path[:, 0], fc_path[:, 1], color="tab:green", linewidth=2.0, alpha=0.85, label="rollout fork_center")
            ax_world.scatter([fc_path[-1, 0]], [fc_path[-1, 1]], color="tab:green", s=35)
        if root_path.size > 0:
            ax_world.plot(root_path[:, 0], root_path[:, 1], color="gray", linewidth=1.2, alpha=0.75, label="rollout root")

    all_pts = [traj_pts, np.asarray([p_pre, p_goal, pallet_pos, root_pos, fork_center])]
    if rollout is not None:
        fc_path = np.asarray(rollout["fork_center_path"])
        root_path = np.asarray(rollout["root_path"])
        if fc_path.size > 0:
            all_pts.append(fc_path)
        if root_path.size > 0:
            all_pts.append(root_path)
    stacked = np.concatenate(all_pts, axis=0)
    x_min, y_min = stacked.min(axis=0) - axis_pad_m
    x_max, y_max = stacked.max(axis=0) + axis_pad_m
    ax_world.set_xlim(float(x_min), float(x_max))
    ax_world.set_ylim(float(y_min), float(y_max))
    ax_world.set_aspect("equal", adjustable="box")
    ax_world.grid(True, alpha=0.25)

    # Pallet-frame view.
    traj_pf = _to_pallet_frame(traj_pts, pallet_pos, u_in, v_lat)
    traj_tang_pf = np.stack(
        [
            traj_tangents[:, 0] * u_in[0] + traj_tangents[:, 1] * u_in[1],
            traj_tangents[:, 0] * v_lat[0] + traj_tangents[:, 1] * v_lat[1],
        ],
        axis=1,
    )
    root_pf = _to_pallet_frame(root_pos.reshape(1, 2), pallet_pos, u_in, v_lat)[0]
    fc_pf = _to_pallet_frame(fork_center.reshape(1, 2), pallet_pos, u_in, v_lat)[0]
    p_pre_pf = _to_pallet_frame(p_pre.reshape(1, 2), pallet_pos, u_in, v_lat)[0]
    p_goal_pf = _to_pallet_frame(p_goal.reshape(1, 2), pallet_pos, u_in, v_lat)[0]
    pallet_pf = np.array([0.0, 0.0], dtype=np.float32)

    ax_pf.plot(traj_pf[:, 0], traj_pf[:, 1], color="tab:blue", linewidth=2.5, label="reference traj")
    ax_pf.quiver(
        traj_pf[tangent_ix, 0],
        traj_pf[tangent_ix, 1],
        traj_tang_pf[tangent_ix, 0],
        traj_tang_pf[tangent_ix, 1],
        angles="xy",
        scale_units="xy",
        scale=6.0,
        width=0.003,
        color="tab:blue",
        alpha=0.55,
    )
    ax_pf.axhline(0.0, color="tab:orange", linewidth=1.5, alpha=0.75, label="lateral = 0")
    ax_pf.axvline(-0.5 * pallet_depth_m, color="tab:red", linewidth=2.5, alpha=0.75, label="pallet front")
    ax_pf.scatter([fc_pf[0]], [fc_pf[1]], color="tab:green", s=45, marker="x", label="fork center start")
    ax_pf.scatter([root_pf[0]], [root_pf[1]], color="black", s=50, label="root start")
    ax_pf.scatter([p_pre_pf[0]], [p_pre_pf[1]], color="tab:purple", s=60, label="p_pre")
    ax_pf.scatter([p_goal_pf[0]], [p_goal_pf[1]], color="tab:brown", s=60, label="p_goal")
    ax_pf.scatter([pallet_pf[0]], [pallet_pf[1]], color="tab:red", s=60, label="pallet center")

    if rollout is not None:
        fc_path = np.asarray(rollout["fork_center_path"])
        root_path = np.asarray(rollout["root_path"])
        if fc_path.size > 0:
            fc_path_pf = _to_pallet_frame(fc_path, pallet_pos, u_in, v_lat)
            ax_pf.plot(fc_path_pf[:, 0], fc_path_pf[:, 1], color="tab:green", linewidth=2.0, alpha=0.85, label="rollout fork_center")
            ax_pf.scatter([fc_path_pf[-1, 0]], [fc_path_pf[-1, 1]], color="tab:green", s=35)
        if root_path.size > 0:
            root_path_pf = _to_pallet_frame(root_path, pallet_pos, u_in, v_lat)
            ax_pf.plot(root_path_pf[:, 0], root_path_pf[:, 1], color="gray", linewidth=1.2, alpha=0.75, label="rollout root")

    pf_all = [traj_pf, np.asarray([p_pre_pf, p_goal_pf, root_pf, fc_pf])]
    if rollout is not None:
        if fc_path.size > 0:
            pf_all.append(fc_path_pf)
        if root_path.size > 0:
            pf_all.append(root_path_pf)
    pf_stacked = np.concatenate(pf_all, axis=0)
    s_min, lat_min = pf_stacked.min(axis=0) - axis_pad_m
    s_max, lat_max = pf_stacked.max(axis=0) + axis_pad_m
    ax_pf.set_xlim(float(s_min), float(s_max))
    ax_pf.set_ylim(float(lat_min), float(lat_max))
    ax_pf.set_aspect("equal", adjustable="box")
    ax_pf.grid(True, alpha=0.25)
    ax_pf.set_xlabel("s along pallet axis (m)")
    ax_pf.set_ylabel("lateral y (m)")
    ax_pf.set_title("Pallet-frame view")

    text = (
        f"s_start = {s_start:+.3f} m\n"
        f"s_pre   = {s_pre:+.3f} m\n"
        f"s_goal  = {s_goal:+.3f} m\n"
        f"delta_s = s_start - s_pre = {delta_s:+.3f} m\n"
        f"y_start = {y_start:+.3f} m"
    )
    ax_pf.text(
        0.02,
        0.98,
        text,
        transform=ax_pf.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "0.7"},
    )

    mode_suffix = " + rollout" if rollout is not None else ""
    if force_zero_steer:
        mode_suffix += " [zero-steer]"
    ax_world.set_title(
        f"{label}{mode_suffix}\n"
        f"reset: x={args.x_root:.2f}, y={y_m:+.3f}, yaw={yaw_deg:+.1f} deg"
    )
    ax_world.legend(loc="best", fontsize=8)
    ax_pf.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(png_path, dpi=180)
    plt.close(fig)


@hydra_task_config(args.task, args.agent)
def main(env_cfg, agent_cfg):
    if args.rollout_steps > 0 and not args.checkpoint:
        raise ValueError("--rollout_steps > 0 requires --checkpoint")

    env_cfg.scene.num_envs = args.num_envs
    env_cfg.seed = args.seed
    env_cfg.sim.device = "cuda:0"
    env_cfg.stage_1_mode = True
    env_cfg.use_asymmetric_critic = True
    env_cfg.use_camera = bool(args.checkpoint)
    if args.checkpoint:
        env_cfg.camera_width = 256
        env_cfg.camera_height = 256

    env = gym.make(args.task, cfg=env_cfg)
    env_wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    raw_env = env.unwrapped

    policy_nn = None
    if args.checkpoint:
        runner = OnPolicyRunner(env_wrapped, agent_cfg.to_dict(), log_dir=None, device="cuda:0")
        runner.load(args.checkpoint)
        try:
            policy_nn = runner.alg.policy
        except AttributeError:
            policy_nn = runner.alg.actor_critic

    y_values = _parse_list(args.y_values)
    yaw_values = _parse_list(args.yaw_deg_values)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest: list[dict[str, object]] = []

    for yaw_deg in yaw_values:
        for y_m in y_values:
            mode_tag = "rollout" if args.rollout_steps > 0 else "traj_only"
            if args.force_zero_steer:
                mode_tag += "_zero_steer"
            case_tag = f"{args.label}_{mode_tag}_x{args.x_root:+.2f}_y{y_m:+.3f}_yaw{yaw_deg:+.1f}"
            case_tag = case_tag.replace(".", "p").replace("+", "p").replace("-", "m")

            _set_fixed_stage1_reset(raw_env, args.seed, args.x_root, y_m, yaw_deg)
            geom = _compute_case_geometry(raw_env)
            scalar = _scalar_geometry(geom)

            rollout = None
            if args.rollout_steps > 0 and policy_nn is not None:
                rollout = _run_rollout_overlay(
                    env_wrapped,
                    raw_env,
                    policy_nn,
                    rollout_steps=args.rollout_steps,
                    force_zero_steer=args.force_zero_steer,
                )

            png_path = output_dir / f"{case_tag}.png"
            csv_path = output_dir / f"{case_tag}_traj.csv"
            json_path = output_dir / f"{case_tag}.json"

            _write_traj_csv(csv_path, geom)
            _plot_case(
                png_path,
                geom,
                label=args.label,
                y_m=y_m,
                yaw_deg=yaw_deg,
                tangent_stride=args.tangent_stride,
                axis_pad_m=args.axis_pad_m,
                rollout=rollout,
                force_zero_steer=args.force_zero_steer,
            )

            meta = {
                "label": args.label,
                "x_root": args.x_root,
                "y_m": y_m,
                "yaw_deg": yaw_deg,
                "checkpoint": args.checkpoint,
                "rollout_steps": args.rollout_steps,
                "force_zero_steer": bool(args.force_zero_steer),
                "pallet_pos": np.asarray(geom["pallet_pos"]).tolist(),
                "pallet_yaw_deg": math.degrees(float(geom["pallet_yaw"])),
                "root_pos": np.asarray(geom["root_pos"]).tolist(),
                "root_yaw_deg": math.degrees(float(geom["root_yaw"])),
                "fork_center": np.asarray(geom["fork_center"]).tolist(),
                "p_pre": np.asarray(geom["p_pre"]).tolist(),
                "p_goal": np.asarray(geom["p_goal"]).tolist(),
                "s_start_m": scalar["s_start"],
                "s_pre_m": scalar["s_pre"],
                "s_goal_m": scalar["s_goal"],
                "delta_s_start_minus_pre_m": scalar["delta_s"],
                "y_start_m": scalar["y_start"],
                "traj_num_points": int(np.asarray(geom["traj_pts"]).shape[0]),
                "traj_csv": str(csv_path),
                "png": str(png_path),
            }
            if rollout is not None:
                meta["rollout_steps_recorded"] = int(np.asarray(rollout["fork_center_path"]).shape[0])
                meta["rollout_done_early"] = bool(rollout["done_early"])
            json_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
            manifest.append(meta)
            print(f"[OK] wrote {png_path}")

    manifest_path = output_dir / f"{args.label}_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[DONE] wrote manifest to {manifest_path}")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
