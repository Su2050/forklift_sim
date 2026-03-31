#!/usr/bin/env python3
"""Visualize standalone reference-trajectory models for Exp8.3."""

from __future__ import annotations

import argparse
import ast
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from exp83_reference_trajectory_lib import (
    Pose2D,
    compute_aligned_front_goal_pose,
    plan_root_path_first_to_front_goal,
    plan_rs_to_front_goal,
    vehicle_to_fork_center,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
CURRENT_CFG_PATH = REPO_ROOT / "IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/forklift_pallet_insert_lift/env_cfg.py"
OUT_DIR = REPO_ROOT / "outputs" / "exp83_reference_path_models"

CFG_KEYS = {
    "pallet_depth_m",
    "fork_reach_m",
    "traj_pre_dist_m",
    "traj_vehicle_curve_min_span_m",
    "traj_vehicle_final_straight_min_m",
}


def _literal_value(node: ast.AST):
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -_literal_value(node.operand)
    raise ValueError(type(node).__name__)


def load_cfg_defaults(cfg_path: Path) -> dict[str, object]:
    tree = ast.parse(cfg_path.read_text(encoding="utf-8"), filename=str(cfg_path))
    values: dict[str, object] = {}
    for node in tree.body:
        if not isinstance(node, ast.ClassDef) or node.name != "ForkliftPalletInsertLiftEnvCfg":
            continue
        for stmt in node.body:
            if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name) and stmt.value is not None:
                name = stmt.target.id
                if name in CFG_KEYS:
                    values[name] = _literal_value(stmt.value)
        break
    missing = sorted(CFG_KEYS - set(values))
    if missing:
        raise RuntimeError(f"missing cfg keys: {missing}")
    return values


def draw_pose(ax, pose: Pose2D, *, color: str, label: str, scale: float = 0.18) -> None:
    ax.scatter([pose.x], [pose.y], color=color, s=36, label=label)
    ax.arrow(
        pose.x,
        pose.y,
        scale * math.cos(pose.yaw),
        scale * math.sin(pose.yaw),
        width=0.008,
        head_width=0.05,
        head_length=0.07,
        color=color,
        length_includes_head=True,
    )


def draw_rigid_links(
    ax,
    vehicle_xy: np.ndarray,
    fork_xy: np.ndarray,
    *,
    color: str = "#888888",
    max_links: int = 7,
    label: str = "root->fork rigid offset",
) -> None:
    if vehicle_xy.shape[0] == 0:
        return
    count = min(max_links, vehicle_xy.shape[0])
    indices = np.unique(np.linspace(0, vehicle_xy.shape[0] - 1, count, dtype=int))
    first = True
    for idx in indices:
        ax.plot(
            [vehicle_xy[idx, 0], fork_xy[idx, 0]],
            [vehicle_xy[idx, 1], fork_xy[idx, 1]],
            color=color,
            lw=1.0,
            alpha=0.55,
            ls=":",
            label=label if first else None,
        )
        first = False


def draw_start_end_markers(ax, plan, *, include_labels: bool) -> None:
    ax.scatter(
        [plan.vehicle_xy[0, 0]],
        [plan.vehicle_xy[0, 1]],
        color="#111111",
        s=34,
        marker="o",
        label="root start" if include_labels else None,
    )
    ax.scatter(
        [plan.fork_center_xy[0, 0]],
        [plan.fork_center_xy[0, 1]],
        color="#d62728",
        s=34,
        marker="x",
        label="fork start" if include_labels else None,
    )
    ax.scatter(
        [plan.vehicle_xy[-1, 0]],
        [plan.vehicle_xy[-1, 1]],
        color="#ff7f0e",
        s=34,
        marker="s",
        label="root end" if include_labels else None,
    )
    ax.scatter(
        [plan.fork_center_xy[-1, 0]],
        [plan.fork_center_xy[-1, 1]],
        color="#2ca02c",
        s=34,
        marker="+",
        label="fork end" if include_labels else None,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize root-path-first vs exact RS reference paths.")
    parser.add_argument("--start-x", type=float, default=-3.45)
    parser.add_argument("--start-y", type=float, default=-0.15)
    parser.add_argument("--start-yaw-deg", type=float, default=-6.0)
    parser.add_argument("--pallet-x", type=float, default=0.0)
    parser.add_argument("--pallet-y", type=float, default=0.0)
    parser.add_argument("--pallet-yaw-deg", type=float, default=0.0)
    parser.add_argument("--min-turn-radius-m", type=float, default=0.55)
    parser.add_argument("--goal-stop-buffer-m", type=float, default=None)
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--tag", type=str, default="sample")
    args = parser.parse_args()

    cfg = load_cfg_defaults(CURRENT_CFG_PATH)
    pallet_xy = np.array([args.pallet_x, args.pallet_y], dtype=np.float64)
    pallet_yaw = math.radians(args.pallet_yaw_deg)
    start = Pose2D(args.start_x, args.start_y, math.radians(args.start_yaw_deg))
    fork_reach_m = float(cfg["fork_reach_m"])
    pallet_depth_m = float(cfg["pallet_depth_m"])
    stop_buffer_m = float(args.goal_stop_buffer_m if args.goal_stop_buffer_m is not None else cfg["traj_pre_dist_m"])
    vehicle_to_fc = vehicle_to_fork_center(fork_reach_m)

    goal_pose, fork_goal_xy = compute_aligned_front_goal_pose(
        pallet_xy=pallet_xy,
        pallet_yaw=pallet_yaw,
        pallet_depth_m=pallet_depth_m,
        vehicle_to_fork_center_m=vehicle_to_fc,
        fork_front_stop_buffer_m=stop_buffer_m,
    )

    root_plan = plan_root_path_first_to_front_goal(
        start=start,
        pallet_xy=pallet_xy,
        pallet_yaw=pallet_yaw,
        pallet_depth_m=pallet_depth_m,
        fork_reach_m=fork_reach_m,
        fork_front_stop_buffer_m=stop_buffer_m,
        curve_min_span_m=float(cfg["traj_vehicle_curve_min_span_m"]),
        final_straight_min_m=float(cfg["traj_vehicle_final_straight_min_m"]),
        num_samples=64,
    )
    rs_plan = plan_rs_to_front_goal(
        start=start,
        pallet_xy=pallet_xy,
        pallet_yaw=pallet_yaw,
        pallet_depth_m=pallet_depth_m,
        fork_reach_m=fork_reach_m,
        fork_front_stop_buffer_m=stop_buffer_m,
        min_turn_radius_m=float(args.min_turn_radius_m),
    )

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"exp83_rs_front_goal_{args.tag}"

    fig, axes = plt.subplots(1, 2, figsize=(13, 6), constrained_layout=True)
    for ax, plan, title in [
        (axes[0], root_plan, "root-path-first"),
        (axes[1], rs_plan, "RS exact"),
    ]:
        ax.plot(plan.vehicle_xy[:, 0], plan.vehicle_xy[:, 1], color="#444444", lw=2.0, ls="--", label="vehicle root path")
        ax.plot(plan.fork_center_xy[:, 0], plan.fork_center_xy[:, 1], color="#1f77b4", lw=2.0, label="mapped fork-center path")
        draw_rigid_links(ax, plan.vehicle_xy, plan.fork_center_xy)
        draw_pose(ax, start, color="#111111", label="root start pose")
        draw_pose(ax, goal_pose, color="#ff7f0e", label="root goal pose")
        draw_start_end_markers(ax, plan, include_labels=True)
        ax.scatter([fork_goal_xy[0]], [fork_goal_xy[1]], color="#2ca02c", s=42, label="fork goal")
        ax.scatter([pallet_xy[0]], [pallet_xy[1]], color="#9467bd", s=42, label="pallet center")
        u_in = np.array([math.cos(pallet_yaw), math.sin(pallet_yaw)])
        axis_s = np.linspace(-2.5, 0.8, 200)
        axis_pts = pallet_xy + axis_s[:, None] * u_in
        ax.plot(axis_pts[:, 0], axis_pts[:, 1], "--", color="#999999", lw=1.0, label="pallet axis")
        ax.set_title(title)
        ax.axis("equal")
        ax.grid(True, alpha=0.25)
        ax.set_xlabel("world x (m)")
        ax.set_ylabel("world y (m)")
        ax.text(
            0.02,
            0.16,
            f"fork-center = root + {vehicle_to_fc:.2f}m * heading",
            transform=ax.transAxes,
            fontsize=9,
            va="bottom",
            ha="left",
            bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "#cccccc"},
        )
        if plan.model == "rs_exact":
            segs = plan.metadata.get("segments", [])
            seg_text = " | ".join(f"{seg['type']}{seg['length_m']:+.2f}m" for seg in segs)
            ax.text(
                0.02,
                0.02,
                f"segments: {seg_text}\nlen={plan.metadata.get('total_length_m', 0.0):.2f}m | reverse={plan.metadata.get('reverse_length_m', 0.0):.2f}m",
                transform=ax.transAxes,
                fontsize=9,
                va="bottom",
                ha="left",
                bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "#cccccc"},
            )
        else:
            ax.text(
                0.02,
                0.02,
                f"mode={plan.metadata.get('mode', 'curve+straight')}\ncurve_span_s={plan.metadata.get('curve_span_s', float('nan')):.3f}",
                transform=ax.transAxes,
                fontsize=9,
                va="bottom",
                ha="left",
                bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "#cccccc"},
            )

        ax.legend(loc="upper right", fontsize=8)
    fig.suptitle(
        f"Exp8.3 front-goal trajectory compare | start=({args.start_x:+.2f}, {args.start_y:+.2f}, {args.start_yaw_deg:+.1f}deg) | "
        f"goal=({goal_pose.x:+.2f}, {goal_pose.y:+.2f}, {math.degrees(goal_pose.yaw):+.1f}deg)"
    )
    fig_path = out_dir / f"{stem}.png"
    fig.savefig(fig_path, dpi=180)
    plt.close(fig)

    summary = {
        "start": {"x": start.x, "y": start.y, "yaw_deg": math.degrees(start.yaw)},
        "goal_pose": {"x": goal_pose.x, "y": goal_pose.y, "yaw_deg": math.degrees(goal_pose.yaw)},
        "fork_goal_xy": fork_goal_xy.tolist(),
        "min_turn_radius_m": float(args.min_turn_radius_m),
        "root_path_first": root_plan.metadata,
        "rs_exact": rs_plan.metadata,
    }
    json_path = out_dir / f"{stem}.json"
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[viz_compare] figure: {fig_path}")
    print(f"[viz_compare] summary: {json_path}")


if __name__ == "__main__":
    main()
