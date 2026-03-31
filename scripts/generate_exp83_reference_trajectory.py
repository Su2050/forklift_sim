#!/usr/bin/env python3
"""Generate standalone Exp8.3 reference trajectories.

This keeps reference-path generation outside the environment so we can:
- feed explicit start pose / goal pose / min-turn-radius
- optionally derive a goal pose aligned in front of the pallet
- export points and metadata for inspection, tests, and downstream tools
"""

from __future__ import annotations

import argparse
import ast
import csv
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
    plan_rs_exact,
    vehicle_to_fork_center,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
CURRENT_CFG_PATH = REPO_ROOT / "IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/forklift_pallet_insert_lift/env_cfg.py"
OUT_DIR = REPO_ROOT / "outputs" / "exp83_reference_path_generator"
CFG_KEYS = {"pallet_depth_m", "fork_reach_m", "traj_pre_dist_m", "traj_vehicle_curve_min_span_m", "traj_vehicle_final_straight_min_m"}


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


def write_rows_csv(path: Path, *, vehicle_xy: np.ndarray, vehicle_yaw: np.ndarray, fork_xy: np.ndarray, fork_yaw: np.ndarray) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["idx", "vehicle_x", "vehicle_y", "vehicle_yaw_deg", "fork_x", "fork_y", "fork_yaw_deg"])
        for i in range(vehicle_xy.shape[0]):
            writer.writerow(
                [
                    i,
                    float(vehicle_xy[i, 0]),
                    float(vehicle_xy[i, 1]),
                    math.degrees(float(vehicle_yaw[i])),
                    float(fork_xy[i, 0]),
                    float(fork_xy[i, 1]),
                    math.degrees(float(fork_yaw[i])),
                ]
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate standalone Exp8.3 reference trajectory.")
    parser.add_argument("--model", choices=["root_path_first", "rs_exact"], default="rs_exact")
    parser.add_argument("--start-x", type=float, required=True)
    parser.add_argument("--start-y", type=float, required=True)
    parser.add_argument("--start-yaw-deg", type=float, required=True)
    parser.add_argument("--goal-x", type=float, default=None)
    parser.add_argument("--goal-y", type=float, default=None)
    parser.add_argument("--goal-yaw-deg", type=float, default=None)
    parser.add_argument("--pallet-x", type=float, default=0.0)
    parser.add_argument("--pallet-y", type=float, default=0.0)
    parser.add_argument("--pallet-yaw-deg", type=float, default=0.0)
    parser.add_argument("--goal-mode", choices=["explicit", "front_of_pallet"], default="front_of_pallet")
    parser.add_argument("--goal-stop-buffer-m", type=float, default=None)
    parser.add_argument("--pallet-depth-m", type=float, default=None)
    parser.add_argument("--fork-reach-m", type=float, default=None)
    parser.add_argument("--min-turn-radius-m", type=float, default=0.55)
    parser.add_argument("--curve-min-span-m", type=float, default=None)
    parser.add_argument("--final-straight-min-m", type=float, default=None)
    parser.add_argument("--sample-step-m", type=float, default=0.03)
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--tag", type=str, default="sample")
    args = parser.parse_args()

    cfg = load_cfg_defaults(CURRENT_CFG_PATH)
    pallet_depth_m = float(args.pallet_depth_m if args.pallet_depth_m is not None else cfg["pallet_depth_m"])
    fork_reach_m = float(args.fork_reach_m if args.fork_reach_m is not None else cfg["fork_reach_m"])
    stop_buffer_m = float(args.goal_stop_buffer_m if args.goal_stop_buffer_m is not None else cfg["traj_pre_dist_m"])
    curve_min_span_m = float(args.curve_min_span_m if args.curve_min_span_m is not None else cfg["traj_vehicle_curve_min_span_m"])
    final_straight_min_m = float(args.final_straight_min_m if args.final_straight_min_m is not None else cfg["traj_vehicle_final_straight_min_m"])

    start = Pose2D(args.start_x, args.start_y, math.radians(args.start_yaw_deg))
    pallet_xy = np.array([args.pallet_x, args.pallet_y], dtype=np.float64)
    pallet_yaw = math.radians(args.pallet_yaw_deg)
    vehicle_to_fc = vehicle_to_fork_center(fork_reach_m)

    if args.goal_mode == "explicit":
        if args.goal_x is None or args.goal_y is None or args.goal_yaw_deg is None:
            raise SystemExit("--goal-mode explicit requires --goal-x --goal-y --goal-yaw-deg")
        goal_pose = Pose2D(args.goal_x, args.goal_y, math.radians(args.goal_yaw_deg))
        fork_goal_xy = None
    else:
        goal_pose, fork_goal_xy = compute_aligned_front_goal_pose(
            pallet_xy=pallet_xy,
            pallet_yaw=pallet_yaw,
            pallet_depth_m=pallet_depth_m,
            vehicle_to_fork_center_m=vehicle_to_fc,
            fork_front_stop_buffer_m=stop_buffer_m,
        )

    if args.model == "root_path_first":
        plan = plan_root_path_first_to_front_goal(
            start=start,
            pallet_xy=pallet_xy,
            pallet_yaw=pallet_yaw,
            pallet_depth_m=pallet_depth_m,
            fork_reach_m=fork_reach_m,
            fork_front_stop_buffer_m=stop_buffer_m,
            curve_min_span_m=curve_min_span_m,
            final_straight_min_m=final_straight_min_m,
            num_samples=max(64, int(math.ceil(max(1.0, math.hypot(goal_pose.x - start.x, goal_pose.y - start.y)) / args.sample_step_m))),
        )
    else:
        plan = plan_rs_exact(
            start=start,
            goal=goal_pose,
            min_turn_radius_m=float(args.min_turn_radius_m),
            vehicle_to_fork_center_m=vehicle_to_fc,
            sample_step_m=float(args.sample_step_m),
        )
        if fork_goal_xy is not None:
            plan.metadata["fork_goal_xy"] = fork_goal_xy.tolist()
            plan.metadata["vehicle_to_fork_center_m"] = vehicle_to_fc

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"exp83_{args.model}_{args.tag}"

    rows_path = out_dir / f"{stem}_rows.csv"
    write_rows_csv(
        rows_path,
        vehicle_xy=plan.vehicle_xy,
        vehicle_yaw=plan.vehicle_yaw,
        fork_xy=plan.fork_center_xy,
        fork_yaw=plan.fork_center_yaw,
    )

    summary = {
        "model": args.model,
        "goal_mode": args.goal_mode,
        "start": {"x": start.x, "y": start.y, "yaw_deg": math.degrees(start.yaw)},
        "goal_pose": {"x": goal_pose.x, "y": goal_pose.y, "yaw_deg": math.degrees(goal_pose.yaw)},
        "fork_goal_xy": None if fork_goal_xy is None else fork_goal_xy.tolist(),
        "min_turn_radius_m": float(args.min_turn_radius_m),
        "pallet_depth_m": pallet_depth_m,
        "fork_reach_m": fork_reach_m,
        "goal_stop_buffer_m": stop_buffer_m,
        "metadata": plan.metadata,
    }
    summary_path = out_dir / f"{stem}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(7.8, 6.6), constrained_layout=True)
    ax.plot(plan.vehicle_xy[:, 0], plan.vehicle_xy[:, 1], color="#444444", lw=2.0, ls="--", label="vehicle path")
    ax.plot(plan.fork_center_xy[:, 0], plan.fork_center_xy[:, 1], color="#1f77b4", lw=2.0, label="fork-center path")
    draw_pose(ax, start, color="#111111", label="start")
    draw_pose(ax, goal_pose, color="#ff7f0e", label="goal")
    if fork_goal_xy is not None:
        ax.scatter([fork_goal_xy[0]], [fork_goal_xy[1]], color="#2ca02c", s=42, label="fork goal")
        u_in = np.array([math.cos(pallet_yaw), math.sin(pallet_yaw)])
        axis_s = np.linspace(-4.5, 0.8, 200)
        axis_pts = pallet_xy + axis_s[:, None] * u_in
        ax.plot(axis_pts[:, 0], axis_pts[:, 1], "--", color="#999999", lw=1.0, label="pallet axis")
        ax.scatter([pallet_xy[0]], [pallet_xy[1]], color="#9467bd", s=42, label="pallet center")
    if args.model == "rs_exact":
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
    ax.set_title(
        f"{args.model} | start=({args.start_x:+.2f}, {args.start_y:+.2f}, {args.start_yaw_deg:+.1f}deg)\n"
        f"goal=({goal_pose.x:+.2f}, {goal_pose.y:+.2f}, {math.degrees(goal_pose.yaw):+.1f}deg)"
    )
    ax.axis("equal")
    ax.grid(True, alpha=0.25)
    ax.set_xlabel("world x (m)")
    ax.set_ylabel("world y (m)")
    ax.legend(loc="best")
    fig_path = out_dir / f"{stem}.png"
    fig.savefig(fig_path, dpi=180)
    plt.close(fig)

    print(f"[generator] figure: {fig_path}")
    print(f"[generator] rows: {rows_path}")
    print(f"[generator] summary: {summary_path}")


if __name__ == "__main__":
    main()
