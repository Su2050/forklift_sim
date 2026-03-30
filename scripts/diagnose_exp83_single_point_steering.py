"""Single-point steering diagnosis for Exp8.3 visual checkpoints.

Runs the exact same fixed stage1 reset twice:
  1) normal policy actions
  2) zero-steer override

For each run it records a per-step CSV plus a compact summary. It also renders
one combined PNG so we can inspect whether steering goes bad at:
  - raw policy output
  - action scaling / target application
  - actual steering joint tracking
  - downstream geometry evolution (y/yaw/traj errors)
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

parser = argparse.ArgumentParser(description="Single-point normal vs zero-steer diagnosis for exp8.3")
parser.add_argument("--task", type=str, default="Isaac-Forklift-PalletInsertLift-Direct-v0")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--label", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--seed", type=int, default=20260330)
parser.add_argument("--x_root", type=float, default=-3.50)
parser.add_argument("--y_m", type=float, default=0.0)
parser.add_argument("--yaw_deg", type=float, default=0.0)
parser.add_argument("--max_steps", type=int, default=0)
parser.add_argument(
    "--output_dir",
    type=str,
    default="/home/uniubi/projects/forklift_sim/outputs/exp83_single_point_steer_diag",
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
from isaaclab_tasks.direct.forklift_pallet_insert_lift.hold_logic import compute_hold_logic
from isaaclab_tasks.utils.hydra import hydra_task_config


def _quat_to_yaw(q: torch.Tensor) -> torch.Tensor:
    w, x, y, z = q.unbind(-1)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return torch.atan2(siny_cosp, cosy_cosp)


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


def _compute_case_geometry(raw_env, env_id: int = 0) -> dict[str, np.ndarray | float]:
    pallet_pos = raw_env.pallet.data.root_pos_w[env_id, :2].detach().cpu().numpy()
    pallet_yaw = float(_quat_to_yaw(raw_env.pallet.data.root_quat_w[env_id : env_id + 1])[0].item())
    root_pos = raw_env.robot.data.root_pos_w[env_id, :2].detach().cpu().numpy()
    root_yaw = float(_quat_to_yaw(raw_env.robot.data.root_quat_w[env_id : env_id + 1])[0].item())
    fork_center = raw_env._compute_fork_center()[env_id, :2].detach().cpu().numpy()
    traj_pts = raw_env._traj_pts[env_id].detach().cpu().numpy()
    traj_tangents = raw_env._traj_tangents[env_id].detach().cpu().numpy()

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
        "u_in": u_in,
        "v_lat": v_lat,
        "p_goal": p_goal,
        "p_pre": p_pre,
        "pallet_depth_m": float(raw_env.cfg.pallet_depth_m),
    }


def _write_rows(path: Path, rows: list[dict[str, float | int | str | bool]]) -> None:
    if not rows:
        path.write_text("")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _collect_step_metrics(
    raw_env,
    *,
    step_idx: int,
    mode: str,
    raw_action: np.ndarray | None,
    applied_action: np.ndarray | None,
    done_flag: bool,
) -> dict[str, float | int | str | bool]:
    env_id = 0
    pallet_pos = raw_env.pallet.data.root_pos_w
    pallet_quat = raw_env.pallet.data.root_quat_w
    robot_pos = raw_env.robot.data.root_pos_w
    robot_quat = raw_env.robot.data.root_quat_w
    tip = raw_env._compute_fork_tip()
    fork_center = raw_env._compute_fork_center()

    pallet_yaw = _quat_to_yaw(pallet_quat)
    robot_yaw = _quat_to_yaw(robot_quat)
    cp = torch.cos(pallet_yaw)
    sp = torch.sin(pallet_yaw)
    u_in = torch.stack([cp, sp], dim=-1)
    v_lat = torch.stack([-sp, cp], dim=-1)

    rel_root = robot_pos[:, :2] - pallet_pos[:, :2]
    rel_tip = tip[:, :2] - pallet_pos[:, :2]
    rel_fc = fork_center[:, :2] - pallet_pos[:, :2]

    root_y_signed = torch.sum(rel_root * v_lat, dim=-1)
    center_y_signed = torch.sum(rel_fc * v_lat, dim=-1)
    tip_y_signed = torch.sum(rel_tip * v_lat, dim=-1)

    yaw_err_signed = torch.atan2(torch.sin(robot_yaw - pallet_yaw), torch.cos(robot_yaw - pallet_yaw))
    yaw_err_deg = torch.abs(yaw_err_signed) * (180.0 / math.pi)

    s_tip = torch.sum(rel_tip * u_in, dim=-1)
    s_front = -0.5 * raw_env.cfg.pallet_depth_m
    dist_front = torch.clamp(s_front - s_tip, min=0.0)
    insert_depth = torch.clamp(s_tip - s_front, min=0.0)

    dists = torch.norm(raw_env._traj_pts - fork_center[:, :2].unsqueeze(1), dim=-1)
    min_dists, min_indices = torch.min(dists, dim=1)
    closest_traj_pt = raw_env._traj_pts[:, :, :][env_id, min_indices[env_id]]
    closest_tangent = raw_env._traj_tangents[:, :, :][env_id, min_indices[env_id]]
    traj_s_norm = raw_env._traj_s_norm[env_id, min_indices[env_id]]
    rel_to_traj = fork_center[env_id, :2] - closest_traj_pt
    d_traj_signed = closest_tangent[0] * rel_to_traj[1] - closest_tangent[1] * rel_to_traj[0]
    traj_yaw = torch.atan2(closest_tangent[1], closest_tangent[0])
    yaw_traj_signed = torch.atan2(
        torch.sin(robot_yaw[env_id] - traj_yaw),
        torch.cos(robot_yaw[env_id] - traj_yaw),
    )

    center_y_err = torch.abs(center_y_signed)
    tip_y_err = torch.abs(tip_y_signed)
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

    dof_pos = raw_env.robot.root_physx_view.get_dof_positions()
    left_steer_pos = dof_pos[env_id, raw_env._left_rotator_id[0]]
    right_steer_pos = dof_pos[env_id, raw_env._right_rotator_id[0]]
    mean_steer_pos = 0.5 * (left_steer_pos + right_steer_pos)

    actor_proprio = raw_env._get_easy8()[env_id].detach().cpu().numpy()
    raw_drive = float("nan") if raw_action is None else float(raw_action[0])
    raw_steer = float("nan") if raw_action is None else float(raw_action[1])
    applied_drive = float("nan") if applied_action is None else float(applied_action[0])
    applied_steer = float("nan") if applied_action is None else float(applied_action[1])

    steer_target_rad = float("nan") if applied_action is None else float(applied_action[1] * raw_env.cfg.steer_angle_rad)
    drive_target_rad_s = float("nan") if applied_action is None else float(applied_action[0] * raw_env.cfg.wheel_speed_rad_s)

    inserted = bool(hold_state.insert_entry[env_id].item())
    hold_entry = bool(hold_state.hold_entry[env_id].item())
    success = bool((raw_env._hold_counter[env_id] >= raw_env._hold_steps).item())
    dirty_insert = inserted and not bool(push_free[env_id].item())

    return {
        "mode": mode,
        "step": step_idx,
        "done": done_flag,
        "root_x": float(robot_pos[env_id, 0].item()),
        "root_y": float(robot_pos[env_id, 1].item()),
        "root_yaw_deg": math.degrees(float(robot_yaw[env_id].item())),
        "fork_center_x": float(fork_center[env_id, 0].item()),
        "fork_center_y": float(fork_center[env_id, 1].item()),
        "tip_x": float(tip[env_id, 0].item()),
        "tip_y": float(tip[env_id, 1].item()),
        "raw_drive": raw_drive,
        "raw_steer": raw_steer,
        "applied_drive": applied_drive,
        "applied_steer": applied_steer,
        "drive_target_rad_s": drive_target_rad_s,
        "steer_target_rad": steer_target_rad,
        "left_steer_joint_pos_rad": float(left_steer_pos.item()),
        "right_steer_joint_pos_rad": float(right_steer_pos.item()),
        "mean_steer_joint_pos_rad": float(mean_steer_pos.item()),
        "root_y_signed_m": float(root_y_signed[env_id].item()),
        "center_y_signed_m": float(center_y_signed[env_id].item()),
        "tip_y_signed_m": float(tip_y_signed[env_id].item()),
        "yaw_err_signed_deg": math.degrees(float(yaw_err_signed[env_id].item())),
        "yaw_err_abs_deg": float(yaw_err_deg[env_id].item()),
        "d_traj_m": float(min_dists[env_id].item()),
        "d_traj_signed_m": float(d_traj_signed.item()),
        "yaw_traj_signed_deg": math.degrees(float(yaw_traj_signed.item())),
        "traj_s_norm": float(traj_s_norm.item()),
        "traj_closest_idx": int(min_indices[env_id].item()),
        "dist_front_m": float(dist_front[env_id].item()),
        "insert_depth_m": float(insert_depth[env_id].item()),
        "pallet_disp_xy_m": float(pallet_disp_xy[env_id].item()),
        "hold_counter": float(raw_env._hold_counter[env_id].item()),
        "inserted": inserted,
        "push_free": bool(push_free[env_id].item()),
        "hold_entry": hold_entry,
        "clean_insert_ready": hold_entry and bool(push_free[env_id].item()),
        "dirty_insert": dirty_insert,
        "success": success,
        "policy_vx_r": float(actor_proprio[0]),
        "policy_vy_r": float(actor_proprio[1]),
        "policy_yaw_rate": float(actor_proprio[2]),
        "policy_lift_pos": float(actor_proprio[3]),
        "policy_lift_vel": float(actor_proprio[4]),
        "policy_prev_drive": float(actor_proprio[5]),
        "policy_prev_steer": float(actor_proprio[6]),
        "policy_prev_lift": float(actor_proprio[7]),
        "policy_y_err_obs": float(actor_proprio[8]) if actor_proprio.shape[0] > 8 else float("nan"),
        "policy_yaw_err_obs": float(actor_proprio[9]) if actor_proprio.shape[0] > 9 else float("nan"),
        "policy_d_traj_signed_obs": float(actor_proprio[10]) if actor_proprio.shape[0] > 10 else float("nan"),
        "policy_yaw_traj_err_obs": float(actor_proprio[11]) if actor_proprio.shape[0] > 11 else float("nan"),
    }


def _build_mode_summary(
    rows: list[dict[str, float | int | str | bool]],
    *,
    mode: str,
    csv_path: Path,
    max_steps: int,
) -> dict[str, float | int | str | bool]:
    step_rows = [r for r in rows if int(r["step"]) > 0]
    initial_row = rows[0]
    final_row = rows[-1]
    terminal_row = step_rows[-2] if len(step_rows) >= 2 and bool(step_rows[-1]["done"]) else step_rows[-1]

    def _mean_abs(key: str) -> float:
        vals = [abs(float(r[key])) for r in step_rows if not math.isnan(float(r[key]))]
        return float(sum(vals) / len(vals)) if vals else 0.0

    def _max_abs(key: str) -> float:
        vals = [abs(float(r[key])) for r in step_rows if not math.isnan(float(r[key]))]
        return max(vals) if vals else 0.0

    def _first_step(key: str, predicate) -> int:
        for row in step_rows:
            if predicate(row[key]):
                return int(row["step"])
        return -1

    tracking_err = [
        abs(float(r["mean_steer_joint_pos_rad"]) - float(r["steer_target_rad"]))
        for r in step_rows
        if not math.isnan(float(r["steer_target_rad"]))
    ]

    return {
        "mode": mode,
        "csv_path": str(csv_path),
        "episode_steps": len(step_rows),
        "done": bool(final_row["done"]),
        "success": any(bool(r["success"]) for r in step_rows),
        "ever_inserted": any(bool(r["inserted"]) for r in step_rows),
        "ever_hold_entry": any(bool(r["hold_entry"]) for r in step_rows),
        "ever_clean_insert_ready": any(bool(r["clean_insert_ready"]) for r in step_rows),
        "ever_dirty_insert": any(bool(r["dirty_insert"]) for r in step_rows),
        "timeout_like": bool(len(step_rows) >= max_steps and not any(bool(r["success"]) for r in step_rows)),
        "initial_center_y_signed_m": float(initial_row["center_y_signed_m"]),
        "final_center_y_signed_m": float(terminal_row["center_y_signed_m"]),
        "initial_yaw_err_signed_deg": float(initial_row["yaw_err_signed_deg"]),
        "final_yaw_err_signed_deg": float(terminal_row["yaw_err_signed_deg"]),
        "initial_d_traj_m": float(initial_row["d_traj_m"]),
        "final_d_traj_m": float(terminal_row["d_traj_m"]),
        "min_dist_front_m": min(float(r["dist_front_m"]) for r in step_rows) if step_rows else float(initial_row["dist_front_m"]),
        "min_d_traj_m": min(float(r["d_traj_m"]) for r in step_rows) if step_rows else float(initial_row["d_traj_m"]),
        "max_pallet_disp_xy_m": max(float(r["pallet_disp_xy_m"]) for r in step_rows) if step_rows else float(initial_row["pallet_disp_xy_m"]),
        "max_hold_counter": max(float(r["hold_counter"]) for r in step_rows) if step_rows else float(initial_row["hold_counter"]),
        "mean_abs_raw_steer": _mean_abs("raw_steer"),
        "mean_abs_applied_steer": _mean_abs("applied_steer"),
        "mean_abs_joint_steer_rad": _mean_abs("mean_steer_joint_pos_rad"),
        "mean_abs_steer_tracking_err_rad": float(sum(tracking_err) / len(tracking_err)) if tracking_err else 0.0,
        "max_abs_raw_steer": _max_abs("raw_steer"),
        "max_abs_applied_steer": _max_abs("applied_steer"),
        "max_abs_joint_steer_rad": _max_abs("mean_steer_joint_pos_rad"),
        "first_abs_raw_steer_gt_0p1_step": _first_step("raw_steer", lambda v: abs(float(v)) > 0.1),
        "first_insert_step": _first_step("inserted", lambda v: bool(v)),
        "first_hold_step": _first_step("hold_entry", lambda v: bool(v)),
        "first_success_step": _first_step("success", lambda v: bool(v)),
    }


def _plot_compare(
    png_path: Path,
    geom: dict[str, np.ndarray | float],
    mode_rows: dict[str, list[dict[str, float | int | str | bool]]],
    mode_summaries: dict[str, dict[str, float | int | str | bool]],
) -> None:
    colors = {"normal": "tab:red", "zero_steer": "tab:green"}
    traj_pts = np.asarray(geom["traj_pts"])
    traj_tangents = np.asarray(geom["traj_tangents"])
    pallet_pos = np.asarray(geom["pallet_pos"])
    p_pre = np.asarray(geom["p_pre"])
    p_goal = np.asarray(geom["p_goal"])
    u_in = np.asarray(geom["u_in"])
    v_lat = np.asarray(geom["v_lat"])

    fig, axes = plt.subplots(3, 2, figsize=(16, 14))
    ax_top, ax_steer = axes[0]
    ax_joint, ax_y = axes[1]
    ax_yaw, ax_front = axes[2]

    ax_top.plot(traj_pts[:, 0], traj_pts[:, 1], color="tab:blue", linewidth=2.5, label="reference traj")
    stride = max(int(len(traj_pts) / 6), 1)
    idx = np.arange(0, len(traj_pts), stride)
    ax_top.quiver(
        traj_pts[idx, 0],
        traj_pts[idx, 1],
        traj_tangents[idx, 0],
        traj_tangents[idx, 1],
        angles="xy",
        scale_units="xy",
        scale=6.0,
        color="tab:blue",
        alpha=0.45,
    )
    ax_top.scatter([pallet_pos[0]], [pallet_pos[1]], color="tab:red", s=80, label="pallet center")
    ax_top.scatter([p_pre[0]], [p_pre[1]], color="tab:purple", s=70, label="p_pre")
    ax_top.scatter([p_goal[0]], [p_goal[1]], color="tab:brown", s=70, label="p_goal")
    ax_top.plot(
        [pallet_pos[0] - 0.35 * u_in[0], pallet_pos[0] + 0.35 * u_in[0]],
        [pallet_pos[1] - 0.35 * u_in[1], pallet_pos[1] + 0.35 * u_in[1]],
        color="tab:red",
        linewidth=3.0,
        alpha=0.7,
        label="pallet insert axis",
    )
    ax_top.plot(
        [pallet_pos[0] - 0.35 * v_lat[0], pallet_pos[0] + 0.35 * v_lat[0]],
        [pallet_pos[1] - 0.35 * v_lat[1], pallet_pos[1] + 0.35 * v_lat[1]],
        color="tab:orange",
        linewidth=3.0,
        alpha=0.7,
        label="pallet lateral axis",
    )

    for mode, rows in mode_rows.items():
        c = colors[mode]
        step_rows = [r for r in rows if int(r["step"]) > 0]
        if not step_rows:
            continue
        fc_xy = np.array([[float(r["fork_center_x"]), float(r["fork_center_y"])] for r in step_rows], dtype=np.float32)
        root_xy = np.array([[float(r["root_x"]), float(r["root_y"])] for r in step_rows], dtype=np.float32)
        ax_top.plot(fc_xy[:, 0], fc_xy[:, 1], color=c, linewidth=2.2, label=f"{mode} fork_center")
        ax_top.plot(root_xy[:, 0], root_xy[:, 1], color=c, linewidth=1.1, alpha=0.45, linestyle="--", label=f"{mode} root")
        ax_top.scatter([fc_xy[0, 0]], [fc_xy[0, 1]], color=c, s=45)

    ax_top.set_aspect("equal", adjustable="box")
    ax_top.grid(True, alpha=0.25)
    ax_top.legend(loc="best", fontsize=8)
    ax_top.set_title("World Top-Down")

    for mode, rows in mode_rows.items():
        c = colors[mode]
        step_rows = [r for r in rows if int(r["step"]) > 0]
        steps = [int(r["step"]) for r in step_rows]
        ax_steer.plot(steps, [float(r["raw_steer"]) for r in step_rows], color=c, linewidth=2.0, label=f"{mode} raw")
        ax_steer.plot(steps, [float(r["applied_steer"]) for r in step_rows], color=c, linewidth=1.6, linestyle="--", label=f"{mode} applied")
    ax_steer.axhline(0.0, color="black", linewidth=0.8, alpha=0.4)
    ax_steer.grid(True, alpha=0.25)
    ax_steer.set_title("Normalized Steering Command")
    ax_steer.set_xlabel("step")
    ax_steer.legend(loc="best", fontsize=8)

    for mode, rows in mode_rows.items():
        c = colors[mode]
        step_rows = [r for r in rows if int(r["step"]) > 0]
        steps = [int(r["step"]) for r in step_rows]
        ax_joint.plot(steps, [float(r["steer_target_rad"]) for r in step_rows], color=c, linewidth=2.0, label=f"{mode} target")
        ax_joint.plot(steps, [float(r["mean_steer_joint_pos_rad"]) for r in step_rows], color=c, linewidth=1.4, linestyle="--", label=f"{mode} joint")
    ax_joint.axhline(0.0, color="black", linewidth=0.8, alpha=0.4)
    ax_joint.grid(True, alpha=0.25)
    ax_joint.set_title("Steering Target vs Joint Position")
    ax_joint.set_xlabel("step")
    ax_joint.legend(loc="best", fontsize=8)

    for mode, rows in mode_rows.items():
        c = colors[mode]
        step_rows = [r for r in rows if int(r["step"]) >= 0]
        steps = [int(r["step"]) for r in step_rows]
        ax_y.plot(steps, [float(r["center_y_signed_m"]) for r in step_rows], color=c, linewidth=2.0, label=f"{mode} center_y")
        ax_y.plot(steps, [float(r["d_traj_signed_m"]) for r in step_rows], color=c, linewidth=1.2, linestyle="--", label=f"{mode} d_traj_signed")
    ax_y.axhline(0.0, color="black", linewidth=0.8, alpha=0.4)
    ax_y.grid(True, alpha=0.25)
    ax_y.set_title("Signed Lateral Error")
    ax_y.set_xlabel("step")
    ax_y.legend(loc="best", fontsize=8)

    for mode, rows in mode_rows.items():
        c = colors[mode]
        step_rows = [r for r in rows if int(r["step"]) >= 0]
        steps = [int(r["step"]) for r in step_rows]
        ax_yaw.plot(steps, [float(r["yaw_err_signed_deg"]) for r in step_rows], color=c, linewidth=2.0, label=f"{mode} yaw_pallet")
        ax_yaw.plot(steps, [float(r["yaw_traj_signed_deg"]) for r in step_rows], color=c, linewidth=1.2, linestyle="--", label=f"{mode} yaw_traj")
    ax_yaw.axhline(0.0, color="black", linewidth=0.8, alpha=0.4)
    ax_yaw.grid(True, alpha=0.25)
    ax_yaw.set_title("Signed Yaw Error")
    ax_yaw.set_xlabel("step")
    ax_yaw.legend(loc="best", fontsize=8)

    for mode, rows in mode_rows.items():
        c = colors[mode]
        step_rows = [r for r in rows if int(r["step"]) >= 0]
        steps = [int(r["step"]) for r in step_rows]
        ax_front.plot(steps, [float(r["dist_front_m"]) for r in step_rows], color=c, linewidth=2.0, label=f"{mode} dist_front")
        ax_front.plot(steps, [float(r["hold_counter"]) for r in step_rows], color=c, linewidth=1.2, linestyle="--", label=f"{mode} hold_counter")
    ax_front.grid(True, alpha=0.25)
    ax_front.set_title("Approach / Hold")
    ax_front.set_xlabel("step")
    ax_front.legend(loc="best", fontsize=8)

    summary_lines = []
    for mode in ("normal", "zero_steer"):
        s = mode_summaries[mode]
        summary_lines.append(
            f"{mode}: success={int(bool(s['success']))}, insert={int(bool(s['ever_inserted']))}, "
            f"hold={int(bool(s['ever_hold_entry']))}, clean={int(bool(s['ever_clean_insert_ready']))}, "
            f"mean|raw steer|={float(s['mean_abs_raw_steer']):.3f}, "
            f"mean track err={float(s['mean_abs_steer_tracking_err_rad']):.3f} rad"
        )
    fig.suptitle(
        f"{args.label} | x={args.x_root:.2f}, y={args.y_m:+.3f}, yaw={args.yaw_deg:+.1f} deg\n"
        + "\n".join(summary_lines),
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    fig.savefig(png_path, dpi=180)
    plt.close(fig)


def _run_mode(env_wrapped, raw_env, policy_nn, *, mode: str, force_zero_steer: bool) -> list[dict[str, float | int | str | bool]]:
    _set_fixed_stage1_reset(raw_env, args.seed, args.x_root, args.y_m, args.yaw_deg)
    obs = env_wrapped.get_observations()
    rows: list[dict[str, float | int | str | bool]] = []
    rows.append(_collect_step_metrics(raw_env, step_idx=0, mode=mode, raw_action=None, applied_action=None, done_flag=False))

    max_steps = args.max_steps if args.max_steps > 0 else int(raw_env.max_episode_length) + 5
    for step_idx in range(1, max_steps + 1):
        with torch.inference_mode():
            raw_actions = policy_nn.act_inference(obs)
            applied_actions = raw_actions.clone()
            if force_zero_steer:
                applied_actions[:, 1] = 0.0
            obs, _, dones, _ = env_wrapped.step(applied_actions)

        raw_np = raw_actions[0].detach().cpu().numpy()
        applied_np = applied_actions[0].detach().cpu().numpy()
        if isinstance(dones, torch.Tensor):
            done_flag = bool(dones[0].item())
        else:
            done_flag = bool(dones[0])
        rows.append(
            _collect_step_metrics(
                raw_env,
                step_idx=step_idx,
                mode=mode,
                raw_action=raw_np,
                applied_action=applied_np,
                done_flag=done_flag,
            )
        )
        if done_flag:
            break
    return rows


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

    # One fixed reset geometry snapshot for plotting.
    _set_fixed_stage1_reset(raw_env, args.seed, args.x_root, args.y_m, args.yaw_deg)
    geom = _compute_case_geometry(raw_env)

    obs_probe = env_wrapped.get_observations()
    if isinstance(obs_probe, dict):
        policy_keys = sorted(str(k) for k in obs_probe.keys())
        policy_type = type(obs_probe.get("policy")).__name__
    else:
        policy_keys = [type(obs_probe).__name__]
        policy_type = type(obs_probe).__name__

    max_steps = args.max_steps if args.max_steps > 0 else int(raw_env.max_episode_length) + 5
    mode_rows: dict[str, list[dict[str, float | int | str | bool]]] = {}
    mode_summaries: dict[str, dict[str, float | int | str | bool]] = {}

    for mode, force_zero in (("normal", False), ("zero_steer", True)):
        rows = _run_mode(env_wrapped, raw_env, policy_nn, mode=mode, force_zero_steer=force_zero)
        csv_path = output_dir / f"{args.label}_{mode}_steps.csv"
        _write_rows(csv_path, rows)
        mode_rows[mode] = rows
        mode_summaries[mode] = _build_mode_summary(rows, mode=mode, csv_path=csv_path, max_steps=max_steps)
        print(
            f"[DONE] mode={mode} steps={mode_summaries[mode]['episode_steps']} "
            f"success={int(bool(mode_summaries[mode]['success']))} "
            f"insert={int(bool(mode_summaries[mode]['ever_inserted']))} "
            f"hold={int(bool(mode_summaries[mode]['ever_hold_entry']))}",
            flush=True,
        )

    png_path = output_dir / f"{args.label}_normal_vs_zero_steer.png"
    _plot_compare(png_path, geom, mode_rows, mode_summaries)

    summary = {
        "label": args.label,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "x_root": args.x_root,
        "y_m": args.y_m,
        "yaw_deg": args.yaw_deg,
        "seed": args.seed,
        "png": str(png_path),
        "actor_obs_probe_keys": policy_keys,
        "actor_policy_value_type": policy_type,
        "camera_enabled": bool(raw_env._camera_enabled),
        "actor_proprio_dim": int(raw_env.cfg.easy8_dim),
        "actor_proprio_fields": [
            "v_x_r",
            "v_y_r",
            "yaw_rate",
            "lift_pos",
            "lift_vel",
            "prev_drive",
            "prev_steer",
            "prev_lift",
            "y_err_obs",
            "yaw_err_obs",
            "d_traj_signed_obs",
            "yaw_traj_err_obs",
        ],
        "critic_privileged_dim": int(raw_env.cfg.privileged_dim),
        "note": "With camera-enabled policy obs, actor uses image + proprio; inspect actor_proprio_fields for whether signed alignment is present.",
        "modes": mode_summaries,
    }
    summary_path = output_dir / f"{args.label}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[DONE] wrote summary to {summary_path}", flush=True)

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
