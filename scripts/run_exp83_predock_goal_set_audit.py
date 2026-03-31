#!/usr/bin/env python3
"""Audit a pre-dock goal set for the current Stage1 near-field reset band.

The goal set is defined in root coordinates as:

    s = root_goal_s - d_pre
    y in [-y_tol, +y_tol]
    yaw in [-yaw_tol, +yaw_tol]

For each Stage1 start case, this tool searches pose-to-goal-set exact RS candidates,
but scores them using world-frame motion direction so that reverse motion is measured
from the sampled path itself rather than from library-internal segment signs.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime
import importlib.util
import itertools
import json
import math
from pathlib import Path
import statistics
import sys
from types import ModuleType

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = REPO_ROOT / "forklift_pallet_insert_lift_project"
CFG_PATH = (
    PROJECT_ROOT
    / "isaaclab_patch/source/isaaclab_tasks/isaaclab_tasks/direct/forklift_pallet_insert_lift/env_cfg.py"
)
VIZ_SCRIPT = PROJECT_ROOT / "scripts" / "visualize_reference_trajectory_cases.py"
OUT_ROOT = REPO_ROOT / "outputs" / "exp83_predock_goal_set_audit"


@dataclass
class GoalSetCaseResult:
    case_id: str
    root_x: float
    root_y: float
    yaw_deg: float
    goal_root_s: float
    goal_root_y: float
    goal_yaw_deg: float
    goal_offset_y: float
    goal_offset_yaw_deg: float
    root_total_length_m: float
    reverse_length_m: float
    reverse_frac: float
    direction_switches: int
    final_forward: bool
    root_heading_change_deg: float
    root_curvature_max: float
    score: float
    path_family: str


def load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def stamped_name(base_name: str, run_timestamp: str, suffix: str) -> str:
    return f"{base_name}_{run_timestamp}{suffix}"


def pretty_repo_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def compute_world_motion_stats(xy: np.ndarray, yaw: np.ndarray) -> dict[str, float | int | bool]:
    if xy.shape[0] < 2:
        return {
            "total_length_m": 0.0,
            "reverse_length_m": 0.0,
            "reverse_frac": 0.0,
            "direction_switches": 0,
            "final_forward": True,
        }

    motion = np.diff(xy, axis=0)
    ds = np.linalg.norm(motion, axis=1)
    tangents = np.stack([np.cos(yaw[:-1]), np.sin(yaw[:-1])], axis=1)
    signed_progress = np.sum(motion * tangents, axis=1)
    eps = 1e-6
    reverse_mask = signed_progress < -eps
    forward_mask = signed_progress > eps
    direction = np.zeros_like(signed_progress, dtype=np.int8)
    direction[forward_mask] = 1
    direction[reverse_mask] = -1

    last_nonzero = 1
    for i in range(direction.shape[0]):
        if direction[i] == 0:
            direction[i] = last_nonzero
        else:
            last_nonzero = int(direction[i])

    switches = int(sum(direction[i] != direction[i - 1] for i in range(1, direction.shape[0])))
    total_length_m = float(np.sum(ds))
    reverse_length_m = float(np.sum(ds[reverse_mask]))
    final_forward = bool(direction[-1] >= 0)
    return {
        "total_length_m": total_length_m,
        "reverse_length_m": reverse_length_m,
        "reverse_frac": reverse_length_m / max(total_length_m, 1e-9),
        "direction_switches": switches,
        "final_forward": final_forward,
    }


def sample_rs_candidates_world_scored(
    *,
    viz: ModuleType,
    root_start_xy: np.ndarray,
    root_start_yaw: float,
    root_goal_xy: np.ndarray,
    root_goal_yaw: float,
    min_turn_radius_m: float,
    sample_step_m: float,
    max_candidates: int,
) -> list[dict[str, object]]:
    x_rs_goal, y_rs_goal, th_rs_goal = viz.rs_local_goal(root_start_xy, root_start_yaw, root_goal_xy, root_goal_yaw)
    all_segs = viz.exact_rs.rs_all_paths(x_rs_goal, y_rs_goal, th_rs_goal, float(min_turn_radius_m))
    if not all_segs:
        return []

    sampled_paths = viz.exact_rs.rs_sample_path_multi(
        float(root_start_xy[0]),
        float(root_start_xy[1]),
        float(root_start_yaw),
        float(root_goal_xy[0]),
        float(root_goal_xy[1]),
        float(root_goal_yaw),
        float(min_turn_radius_m),
        step=float(sample_step_m),
        max_paths=min(int(max_candidates), len(all_segs)),
    )

    fallback_dir = np.array([math.cos(root_start_yaw), math.sin(root_start_yaw)], dtype=np.float64)
    candidates: list[dict[str, object]] = []
    for segs, world_pts in zip(all_segs[: len(sampled_paths)], sampled_paths, strict=True):
        xy = np.asarray([[pt[0], pt[1]] for pt in world_pts], dtype=np.float64)
        yaw = np.asarray([pt[2] for pt in world_pts], dtype=np.float64)
        tangents = np.stack([np.cos(yaw), np.sin(yaw)], axis=1)
        heading_change_deg, curvature_max = viz.compute_path_heading_curvature(
            xy,
            fallback_dir=fallback_dir,
            tangents=tangents,
        )
        motion_stats = compute_world_motion_stats(xy, yaw)
        seg_types = [seg_type for seg_type, seg_len in segs if abs(seg_len) > 1e-6]
        family = "".join(seg_types) if seg_types else "S"
        candidates.append(
            {
                "xy": xy,
                "yaw": yaw,
                "tangents": tangents,
                "family": family,
                "heading_change_deg": heading_change_deg,
                "curvature_max": curvature_max,
                **motion_stats,
            }
        )
    return candidates


def select_goal_set_candidate(
    *,
    viz: ModuleType,
    root_xy: np.ndarray,
    yaw_deg: float,
    pallet_xy: np.ndarray,
    pallet_yaw_deg: float,
    root_goal_s: float,
    root_to_fc: float,
    d_pre: float,
    y_tol: float,
    yaw_tol_deg: float,
    lateral_samples: int,
    yaw_samples: int,
    min_turn_radius_m: float,
    sample_step_m: float,
    max_candidates: int,
    reverse_weight: float,
    switch_weight: float,
    terminal_reverse_penalty: float,
    center_y_weight: float,
    center_yaw_weight: float,
) -> tuple[GoalSetCaseResult, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] | None:
    yaw = math.radians(yaw_deg)
    pallet_yaw = math.radians(pallet_yaw_deg)
    u_in = np.array([math.cos(pallet_yaw), math.sin(pallet_yaw)], dtype=np.float64)
    v_lat = np.array([-math.sin(pallet_yaw), math.cos(pallet_yaw)], dtype=np.float64)
    fork_center_xy = root_xy + root_to_fc * np.array([math.cos(yaw), math.sin(yaw)], dtype=np.float64)

    goal_root_s = root_goal_s - float(d_pre)
    y_offsets = np.linspace(-float(y_tol), float(y_tol), int(lateral_samples), dtype=np.float64)
    yaw_offsets_deg = np.linspace(-float(yaw_tol_deg), float(yaw_tol_deg), int(yaw_samples), dtype=np.float64)

    candidate_pool: list[
        tuple[
            float,
            bool,
            GoalSetCaseResult,
            tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        ]
    ] = []

    for y_offset, yaw_offset_deg in itertools.product(y_offsets.tolist(), yaw_offsets_deg.tolist()):
        goal_root_xy = pallet_xy + goal_root_s * u_in + y_offset * v_lat
        goal_yaw = pallet_yaw + math.radians(yaw_offset_deg)
        path_candidates = sample_rs_candidates_world_scored(
            viz=viz,
            root_start_xy=root_xy,
            root_start_yaw=yaw,
            root_goal_xy=goal_root_xy,
            root_goal_yaw=goal_yaw,
            min_turn_radius_m=min_turn_radius_m,
            sample_step_m=sample_step_m,
            max_candidates=max_candidates,
        )
        for candidate in path_candidates:
            score = (
                float(candidate["total_length_m"])
                + float(reverse_weight) * float(candidate["reverse_length_m"])
                + float(switch_weight) * int(candidate["direction_switches"])
                + (0.0 if bool(candidate["final_forward"]) else float(terminal_reverse_penalty))
                + float(center_y_weight) * abs(float(y_offset)) / max(float(y_tol), 1e-9)
                + float(center_yaw_weight) * abs(float(yaw_offset_deg)) / max(float(yaw_tol_deg), 1e-9)
            )
            p_goal_set = goal_root_xy + root_to_fc * np.array([math.cos(goal_yaw), math.sin(goal_yaw)], dtype=np.float64)
            row = GoalSetCaseResult(
                case_id=viz.format_case_id(0, float(root_xy[0]), float(root_xy[1]), float(yaw_deg)),
                root_x=float(root_xy[0]),
                root_y=float(root_xy[1]),
                yaw_deg=float(yaw_deg),
                goal_root_s=float(goal_root_s),
                goal_root_y=float(y_offset),
                goal_yaw_deg=float(math.degrees(goal_yaw)),
                goal_offset_y=float(y_offset),
                goal_offset_yaw_deg=float(yaw_offset_deg),
                root_total_length_m=float(candidate["total_length_m"]),
                reverse_length_m=float(candidate["reverse_length_m"]),
                reverse_frac=float(candidate["reverse_frac"]),
                direction_switches=int(candidate["direction_switches"]),
                final_forward=bool(candidate["final_forward"]),
                root_heading_change_deg=float(candidate["heading_change_deg"]),
                root_curvature_max=float(candidate["curvature_max"]),
                score=float(score),
                path_family=f"goal_set_{candidate['family']}",
            )
            payload = (
                root_xy,
                fork_center_xy,
                p_goal_set,
                goal_root_xy,
                candidate["xy"] + root_to_fc * candidate["tangents"],
                candidate["tangents"],
                candidate["xy"],
            )
            candidate_pool.append((score, row.reverse_length_m <= 1e-6, row, payload))

    if not candidate_pool:
        return None

    reverse_free_pool = [item for item in candidate_pool if item[1]]
    if reverse_free_pool:
        _, _, best_row, best_payload = min(reverse_free_pool, key=lambda item: item[0])
    else:
        _, _, best_row, best_payload = min(candidate_pool, key=lambda item: item[0])
    return best_row, best_payload


def build_goal_set_grid(
    *,
    viz: ModuleType,
    cfg: dict[str, object],
    pallet_xy: np.ndarray,
    pallet_yaw_deg: float,
    d_pre: float,
    y_tol: float,
    yaw_tol_deg: float,
    grid_count_x: int,
    grid_count_y: int,
    grid_count_yaw: int,
    lateral_samples: int,
    yaw_samples: int,
) -> tuple[list[GoalSetCaseResult], list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]], float]:
    x_vals = np.linspace(
        float(cfg["stage1_init_x_min_m"]),
        float(cfg["stage1_init_x_max_m"]),
        grid_count_x,
        dtype=np.float64,
    ).tolist()
    y_vals = np.linspace(
        float(cfg["stage1_init_y_min_m"]),
        float(cfg["stage1_init_y_max_m"]),
        grid_count_y,
        dtype=np.float64,
    ).tolist()
    yaw_vals = np.linspace(
        float(cfg["stage1_init_yaw_deg_min"]),
        float(cfg["stage1_init_yaw_deg_max"]),
        grid_count_yaw,
        dtype=np.float64,
    ).tolist()

    pallet_yaw = math.radians(pallet_yaw_deg)
    u_in = np.array([math.cos(pallet_yaw), math.sin(pallet_yaw)], dtype=np.float64)
    root_to_fc = float(cfg["fork_reach_m"]) - viz.FORK_CENTER_BACKOFF_M
    s_goal_fc = viz.exp83_traj_goal_s(
        pallet_depth_m=float(cfg["pallet_depth_m"]),
        insert_fraction=float(cfg["insert_fraction"]),
        mode=str(cfg["exp83_traj_goal_mode"]),
    )
    p_goal_fc = pallet_xy + s_goal_fc * u_in
    root_goal_xy = p_goal_fc - root_to_fc * u_in
    root_goal_s, _ = viz.project_axis(root_goal_xy, pallet_xy, pallet_yaw_deg)

    cases: list[GoalSetCaseResult] = []
    payloads: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []

    idx = 1
    for root_x in x_vals:
        for root_y in y_vals:
            for yaw_deg in yaw_vals:
                root_xy = np.array([root_x, root_y], dtype=np.float64)
                selected = select_goal_set_candidate(
                    viz=viz,
                    root_xy=root_xy,
                    yaw_deg=float(yaw_deg),
                    pallet_xy=pallet_xy,
                    pallet_yaw_deg=pallet_yaw_deg,
                    root_goal_s=float(root_goal_s),
                    root_to_fc=root_to_fc,
                    d_pre=float(d_pre),
                    y_tol=float(y_tol),
                    yaw_tol_deg=float(yaw_tol_deg),
                    lateral_samples=int(lateral_samples),
                    yaw_samples=int(yaw_samples),
                    min_turn_radius_m=float(cfg["traj_rs_min_turn_radius_m"]),
                    sample_step_m=float(cfg["traj_rs_sample_step_m"]),
                    max_candidates=int(cfg["traj_rs_forward_preferred_max_candidates"]),
                    reverse_weight=float(cfg["traj_rs_forward_preferred_reverse_weight"]),
                    switch_weight=float(cfg["traj_rs_forward_preferred_switch_weight"]),
                    terminal_reverse_penalty=float(cfg["traj_rs_forward_preferred_terminal_reverse_penalty"]),
                    center_y_weight=0.2,
                    center_yaw_weight=0.2,
                )
                if selected is None:
                    continue
                case, payload = selected
                case.case_id = viz.format_case_id(idx, root_x, root_y, yaw_deg)
                cases.append(case)
                payloads.append(payload)
                idx += 1
    return cases, payloads, float(root_goal_s)


def draw_goal_set_overlay(
    *,
    out_path: Path,
    cases: list[GoalSetCaseResult],
    payloads: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    pallet_xy: np.ndarray,
    pallet_yaw_deg: float,
    root_goal_s: float,
    d_pre: float,
    y_tol: float,
) -> None:
    pallet_yaw = math.radians(pallet_yaw_deg)
    u_in = np.array([math.cos(pallet_yaw), math.sin(pallet_yaw)], dtype=np.float64)
    v_lat = np.array([-math.sin(pallet_yaw), math.cos(pallet_yaw)], dtype=np.float64)
    goal_set_s = root_goal_s - d_pre
    root_goal_line = pallet_xy + goal_set_s * u_in + np.array([-y_tol, y_tol], dtype=np.float64).reshape(-1, 1) * v_lat
    final_goal_pt = pallet_xy + root_goal_s * u_in

    fig, ax = plt.subplots(figsize=(9.2, 8.4))
    axis_s = np.linspace(goal_set_s - 0.8, root_goal_s + 0.8, 200)
    axis_pts = pallet_xy + axis_s[:, None] * u_in
    ax.plot(axis_pts[:, 0], axis_pts[:, 1], "--", color="#8e8e8e", lw=1.1, label="pallet s-axis")

    bbox_points = [axis_pts[[0, -1]], root_goal_line, final_goal_pt.reshape(1, 2)]
    for case, payload in zip(cases, payloads, strict=True):
        root_xy, fork_center_xy, p_goal_set, goal_root_xy, pts, _, root_path = payload
        ax.plot(root_path[:, 0], root_path[:, 1], color="#404040", alpha=0.16, lw=1.3)
        ax.plot(pts[:, 0], pts[:, 1], color="#1f77b4", alpha=0.22, lw=1.4)
        ax.scatter(fork_center_xy[0], fork_center_xy[1], color="#d62728", s=16, alpha=0.55)
        ax.scatter(p_goal_set[0], p_goal_set[1], color="#ff7f0e", s=11, alpha=0.35)
        bbox_points.extend([root_xy.reshape(1, 2), fork_center_xy.reshape(1, 2), p_goal_set.reshape(1, 2), pts, root_path])

    ax.plot(root_goal_line[:, 0], root_goal_line[:, 1], color="#ff7f0e", lw=4.0, alpha=0.85, label="pre-dock goal-set line")
    ax.scatter(final_goal_pt[0], final_goal_pt[1], color="#2ca02c", s=72, label="final root goal")

    num_reverse_free = sum(case.reverse_length_m <= 1e-6 for case in cases)
    num_heading_gt_180 = sum(case.root_heading_change_deg > 180.0 for case in cases)
    title = (
        f"pre-dock goal-set audit | d_pre={d_pre:.2f} m | y_tol={y_tol:.2f} m | "
        f"reverse_free={num_reverse_free}/{len(cases)} | >180deg={num_heading_gt_180}/{len(cases)}"
    )
    ax.set_title(title, fontsize=11.5)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    ax.set_xlabel("world x (m)")
    ax.set_ylabel("world y (m)")
    all_pts = np.vstack(bbox_points)
    mins = all_pts.min(axis=0)
    maxs = all_pts.max(axis=0)
    center = 0.5 * (mins + maxs)
    span = max(float(np.max(maxs - mins)) + 0.30, 1.2)
    half_span = 0.5 * span
    ax.set_xlim(center[0] - half_span, center[0] + half_span)
    ax.set_ylim(center[1] - half_span, center[1] + half_span)
    ax.legend(loc="lower right", fontsize=9, frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def draw_goal_set_geometry(
    *,
    out_path: Path,
    cfg: dict[str, object],
    root_goal_s: float,
    d_pre: float,
    y_tol: float,
    yaw_tol_deg: float,
) -> None:
    goal_set_s = root_goal_s - d_pre
    fig, ax = plt.subplots(figsize=(11.2, 2.8))
    ax.hlines(0.0, goal_set_s - 0.6, root_goal_s + 0.6, color="#7f7f7f", lw=1.2)
    ax.fill_between(
        [float(cfg["stage1_init_x_min_m"]), float(cfg["stage1_init_x_max_m"])],
        -0.16,
        0.16,
        color="#d62728",
        alpha=0.22,
        label="current Stage1 start band",
    )
    ax.axvline(goal_set_s, color="#ff7f0e", lw=2.2, label="pre-dock goal-set center line")
    ax.axvline(root_goal_s, color="#2ca02c", lw=2.2, label="final root goal")
    ax.scatter([goal_set_s, root_goal_s], [0.0, 0.0], s=44, color=["#ff7f0e", "#2ca02c"])
    ax.annotate(
        f"goal-set center={goal_set_s:.2f}",
        xy=(goal_set_s, 0.0),
        xytext=(0, -18),
        textcoords="offset points",
        ha="center",
        fontsize=9,
        color="#ff7f0e",
    )
    ax.annotate(
        f"root_goal={root_goal_s:.2f}",
        xy=(root_goal_s, 0.0),
        xytext=(0, 12),
        textcoords="offset points",
        ha="center",
        fontsize=9,
        color="#2ca02c",
    )
    ax.set_title(
        f"pre-dock goal-set geometry | d_pre={d_pre:.2f} m | y_tol={y_tol:.2f} m | yaw_tol={yaw_tol_deg:.1f} deg",
        fontsize=12,
    )
    ax.set_xlabel("pallet-axis s (m)")
    ax.set_yticks([])
    ax.grid(True, axis="x", alpha=0.25)
    ax.legend(loc="upper left", ncol=3, frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def select_representative_case_ids(cases: list[GoalSetCaseResult]) -> list[str]:
    if not cases:
        return []
    worst_reverse = max(cases, key=lambda case: case.reverse_length_m)
    worst_heading = max(cases, key=lambda case: case.root_heading_change_deg)
    shortest = min(cases, key=lambda case: case.root_total_length_m)
    positive_corner = max(cases, key=lambda case: (case.root_y, case.yaw_deg, -case.root_x))
    negative_corner = min(cases, key=lambda case: (case.root_y, case.yaw_deg, case.root_x))
    ordered = [worst_reverse.case_id, worst_heading.case_id, shortest.case_id, positive_corner.case_id, negative_corner.case_id]
    unique_ids: list[str] = []
    for case_id in ordered:
        if case_id not in unique_ids:
            unique_ids.append(case_id)
    return unique_ids


def draw_goal_set_case(
    *,
    out_path: Path,
    case: GoalSetCaseResult,
    payload: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    pallet_xy: np.ndarray,
    pallet_yaw_deg: float,
    root_goal_s: float,
    d_pre: float,
    y_tol: float,
    yaw_tol_deg: float,
    cfg_path: Path,
) -> None:
    root_xy, fork_center_xy, p_goal_set, goal_root_xy, pts, tangents, root_path = payload
    pallet_yaw = math.radians(pallet_yaw_deg)
    u_in = np.array([math.cos(pallet_yaw), math.sin(pallet_yaw)], dtype=np.float64)
    v_lat = np.array([-math.sin(pallet_yaw), math.cos(pallet_yaw)], dtype=np.float64)

    fig = plt.figure(figsize=(12.6, 7.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[4.2, 1.8], wspace=0.12)
    ax = fig.add_subplot(gs[0, 0])
    ax_info = fig.add_subplot(gs[0, 1])
    ax_info.axis("off")

    axis_s = np.linspace(case.goal_root_s - 0.8, root_goal_s + 0.8, 200)
    axis_pts = pallet_xy + axis_s[:, None] * u_in
    goal_set_line = goal_root_xy + np.array([-y_tol, y_tol], dtype=np.float64).reshape(-1, 1) * v_lat - case.goal_offset_y * v_lat
    goal_yaw = math.radians(case.goal_yaw_deg)
    goal_dir = np.array([math.cos(goal_yaw), math.sin(goal_yaw)], dtype=np.float64)
    final_goal_root = pallet_xy + root_goal_s * u_in

    ax.plot(axis_pts[:, 0], axis_pts[:, 1], "--", color="#888888", lw=1.2, label="pallet s-axis")
    ax.plot(goal_set_line[:, 0], goal_set_line[:, 1], color="#ff7f0e", lw=4.0, alpha=0.75, label="goal-set band")
    ax.plot(root_path[:, 0], root_path[:, 1], color="#444444", lw=2.0, ls="--", alpha=0.95, label="vehicle/root reference")
    ax.plot(pts[:, 0], pts[:, 1], color="#1f77b4", lw=1.8, label="fork-center mapping")
    ax.scatter(root_xy[0], root_xy[1], color="#444444", s=36, label="root start")
    ax.scatter(fork_center_xy[0], fork_center_xy[1], color="#d62728", s=42, label="fork_center start")
    ax.scatter(p_goal_set[0], p_goal_set[1], color="#ff7f0e", s=42, label="chosen pre-dock goal")
    ax.scatter(final_goal_root[0], final_goal_root[1], color="#2ca02c", s=42, label="final root goal")
    ax.arrow(
        goal_root_xy[0],
        goal_root_xy[1],
        0.22 * goal_dir[0],
        0.22 * goal_dir[1],
        width=0.008,
        head_width=0.05,
        head_length=0.06,
        color="#ff7f0e",
        length_includes_head=True,
    )

    title = (
        f"{case.case_id} | len={case.root_total_length_m:.3f} m | rev={case.reverse_length_m:.3f} m | "
        f"dpsi={case.root_heading_change_deg:.1f} deg"
    )
    fig.suptitle(title, fontsize=13)

    info_lines = [
        "Pre-dock Goal Set",
        f"root = ({case.root_x:+.3f}, {case.root_y:+.3f})",
        f"yaw = {case.yaw_deg:+.1f} deg",
        f"d_pre = {d_pre:.2f} m",
        f"y_tol = {y_tol:.2f} m",
        f"yaw_tol = {yaw_tol_deg:.1f} deg",
        f"goal_y = {case.goal_offset_y:+.3f} m",
        f"goal_yaw = {case.goal_offset_yaw_deg:+.1f} deg",
        f"path_family = {case.path_family}",
        f"len = {case.root_total_length_m:.3f} m",
        f"reverse = {case.reverse_length_m:.3f} m",
        f"reverse_frac = {case.reverse_frac:.3f}",
        f"switches = {case.direction_switches}",
        f"final_forward = {case.final_forward}",
        f"dpsi = {case.root_heading_change_deg:.2f} deg",
        f"kappa_max = {case.root_curvature_max:.3f} 1/m",
        "",
        "Scoring",
        "world-motion-aware RS selection",
        "goal-set center penalties are weak",
        "path naturalness dominates center preference",
        "",
        "Cfg source",
        pretty_repo_path(cfg_path),
    ]
    ax_info.text(
        0.0,
        0.98,
        "\n".join(info_lines),
        ha="left",
        va="top",
        fontsize=9.1,
        family="monospace",
        linespacing=1.22,
        bbox={"boxstyle": "round", "facecolor": "#f7f7f7", "edgecolor": "#d9d9d9", "alpha": 0.98},
    )

    bbox_points = np.vstack(
        [
            axis_pts[[0, -1]],
            goal_set_line,
            root_xy[None, :],
            fork_center_xy[None, :],
            p_goal_set[None, :],
            final_goal_root[None, :],
            pts,
            root_path,
        ]
    )
    mins = bbox_points.min(axis=0)
    maxs = bbox_points.max(axis=0)
    center = 0.5 * (mins + maxs)
    span = max(float(np.max(maxs - mins)) + 0.25, 1.2)
    half_span = 0.5 * span
    ax.set_xlim(center[0] - half_span, center[0] + half_span)
    ax.set_ylim(center[1] - half_span, center[1] + half_span)
    ax.set_aspect("equal", adjustable="box")
    ax.set_box_aspect(1.0)
    ax.grid(True, alpha=0.25)
    ax.set_xlabel("world x (m)")
    ax.set_ylabel("world y (m)")
    handles, labels = ax.get_legend_handles_labels()
    ax_info.legend(handles, labels, loc="lower left", fontsize=9, frameon=False)
    fig.subplots_adjust(left=0.05, right=0.985, top=0.90, bottom=0.08, wspace=0.10)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def write_markdown(path: Path, summary: dict) -> None:
    lines = [
        "# Exp8.3 Pre-Dock Goal-Set Audit",
        "",
        f"- cfg_path: `{summary['cfg_path']}`",
        f"- scan_mode: `{summary['scan_mode']}`",
        f"- note: `{summary['note']}`",
        "",
        "## Best Combo",
        "",
        f"- d_pre_m: `{summary['best_combo']['d_pre_m']:.2f}`",
        f"- y_tol_m: `{summary['best_combo']['y_tol_m']:.2f}`",
        f"- yaw_tol_deg: `{summary['best_combo']['yaw_tol_deg']:.1f}`",
        f"- overlay: `{summary['best_combo']['overlay_path']}`",
        f"- geometry: `{summary['best_combo']['geometry_path']}`",
        f"- reverse_free: `{summary['best_combo']['num_reverse_free']}/{summary['best_combo']['num_cases']}`",
        f"- mean_reverse_frac: `{summary['best_combo']['reverse_frac_mean']:.3f}`",
        f"- mean_length: `{summary['best_combo']['root_total_length_mean']:.3f} m`",
        f"- mean_heading_change: `{summary['best_combo']['root_heading_change_mean']:.3f} deg`",
        "",
        "## Top Rows",
        "",
        "| rank | d_pre | y_tol | yaw_tol | reverse_free | zero_len | >180deg | >10m | mean_len_nonzero |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for i, row in enumerate(summary["top_rows"], start=1):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(i),
                    f"{row['d_pre_m']:.2f}",
                    f"{row['y_tol_m']:.2f}",
                    f"{row['yaw_tol_deg']:.1f}",
                    f"{row['num_reverse_free']}/{row['num_cases']}",
                    str(row["num_zero_len"]),
                    str(row["num_heading_gt_180"]),
                    str(row["num_length_gt_10m"]),
                    f"{row['root_total_length_mean_nonzero']:.3f}",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Representative Cases",
            "",
        ]
    )
    for item in summary["best_combo"]["representative_cases"]:
        lines.append(f"- `{item['case_id']}`: `{item['image_path']}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit a pre-dock goal set for the current Stage1 near-field reset band.")
    parser.add_argument("--cfg-path", type=Path, default=CFG_PATH)
    parser.add_argument("--d-pre-values", type=float, nargs="+", default=[1.05, 1.25, 1.50, 1.75, 2.00])
    parser.add_argument("--y-tol-values", type=float, nargs="+", default=[0.05, 0.10, 0.15])
    parser.add_argument("--yaw-tol-values", type=float, nargs="+", default=[3.0, 6.0, 8.0])
    parser.add_argument("--lateral-samples", type=int, default=5)
    parser.add_argument("--yaw-samples", type=int, default=5)
    parser.add_argument("--grid-count-x", type=int, default=5)
    parser.add_argument("--grid-count-y", type=int, default=5)
    parser.add_argument("--grid-count-yaw", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, default=OUT_ROOT)
    parser.add_argument("--timestamp-tag", type=str, default=None)
    args = parser.parse_args()

    viz = load_module("exp83_viz_cases_predock", VIZ_SCRIPT)
    cfg = viz.load_cfg_defaults(args.cfg_path)
    pallet_xy = np.array([0.0, 0.0], dtype=np.float64)
    pallet_yaw_deg = 0.0

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    run_timestamp = args.timestamp_tag or datetime.now().strftime("%Y%m%d_%H%M%S")

    rows: list[dict[str, object]] = []
    best_cases: list[GoalSetCaseResult] | None = None
    best_payloads: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] | None = None
    best_root_goal_s = 0.0

    for d_pre, y_tol, yaw_tol_deg in itertools.product(args.d_pre_values, args.y_tol_values, args.yaw_tol_values):
        cases, payloads, root_goal_s = build_goal_set_grid(
            viz=viz,
            cfg=cfg,
            pallet_xy=pallet_xy,
            pallet_yaw_deg=pallet_yaw_deg,
            d_pre=float(d_pre),
            y_tol=float(y_tol),
            yaw_tol_deg=float(yaw_tol_deg),
            grid_count_x=int(args.grid_count_x),
            grid_count_y=int(args.grid_count_y),
            grid_count_yaw=int(args.grid_count_yaw),
            lateral_samples=int(args.lateral_samples),
            yaw_samples=int(args.yaw_samples),
        )
        if not cases:
            continue

        reverse_fracs = [case.reverse_frac for case in cases]
        lengths = [case.root_total_length_m for case in cases]
        headings = [case.root_heading_change_deg for case in cases]
        reverse_lengths = [case.reverse_length_m for case in cases]
        nonzero_lengths = [value for value in lengths if value > 1e-9]
        row = {
            "d_pre_m": float(d_pre),
            "y_tol_m": float(y_tol),
            "yaw_tol_deg": float(yaw_tol_deg),
            "num_cases": len(cases),
            "num_zero_len": sum(value <= 1e-9 for value in lengths),
            "num_reverse_free": sum(case.reverse_length_m <= 1e-6 for case in cases),
            "num_reverse_frac_le_005": sum(case.reverse_frac <= 0.05 for case in cases),
            "reverse_frac_mean": statistics.mean(reverse_fracs),
            "reverse_frac_max": max(reverse_fracs),
            "reverse_length_mean": statistics.mean(reverse_lengths),
            "root_total_length_mean": statistics.mean(lengths),
            "root_total_length_mean_nonzero": (statistics.mean(nonzero_lengths) if nonzero_lengths else 0.0),
            "root_total_length_max": max(lengths),
            "root_heading_change_mean": statistics.mean(headings),
            "root_heading_change_max": max(headings),
            "num_heading_gt_180": sum(value > 180.0 for value in headings),
            "num_heading_gt_270": sum(value > 270.0 for value in headings),
            "num_length_gt_10m": sum(value > 10.0 for value in lengths),
        }
        rows.append(row)

        if best_cases is None:
            best_cases = cases
            best_payloads = payloads
            best_root_goal_s = root_goal_s
        else:
            current_best = rows[0]
            # best selection is applied after sorting below

    if not rows:
        raise RuntimeError("no goal-set audit rows were produced")

    rows.sort(
        key=lambda row: (
            int(row["num_heading_gt_180"]),
            int(row["num_heading_gt_270"]),
            int(row["num_length_gt_10m"]),
            -int(row["num_reverse_free"]),
            int(row["num_zero_len"]),
            float(row["root_total_length_mean_nonzero"]),
            float(row["reverse_frac_mean"]),
            float(row["d_pre_m"]),
            float(row["y_tol_m"]),
            float(row["yaw_tol_deg"]),
        )
    )
    best_row = rows[0]

    best_cases, best_payloads, best_root_goal_s = build_goal_set_grid(
        viz=viz,
        cfg=cfg,
        pallet_xy=pallet_xy,
        pallet_yaw_deg=pallet_yaw_deg,
        d_pre=float(best_row["d_pre_m"]),
        y_tol=float(best_row["y_tol_m"]),
        yaw_tol_deg=float(best_row["yaw_tol_deg"]),
        grid_count_x=int(args.grid_count_x),
        grid_count_y=int(args.grid_count_y),
        grid_count_yaw=int(args.grid_count_yaw),
        lateral_samples=int(args.lateral_samples),
        yaw_samples=int(args.yaw_samples),
    )

    overlay_path = out_dir / stamped_name("overlay_predock_goal_set_best", run_timestamp, ".png")
    geometry_path = out_dir / stamped_name("geometry_predock_goal_set_best", run_timestamp, ".png")
    manifest_path = out_dir / stamped_name("predock_goal_set_best_manifest", run_timestamp, ".json")
    summary_json_path = out_dir / stamped_name("predock_goal_set_audit_summary", run_timestamp, ".json")
    summary_md_path = out_dir / stamped_name("predock_goal_set_audit_summary", run_timestamp, ".md")

    draw_goal_set_overlay(
        out_path=overlay_path,
        cases=best_cases,
        payloads=best_payloads,
        pallet_xy=pallet_xy,
        pallet_yaw_deg=pallet_yaw_deg,
        root_goal_s=best_root_goal_s,
        d_pre=float(best_row["d_pre_m"]),
        y_tol=float(best_row["y_tol_m"]),
    )
    draw_goal_set_geometry(
        out_path=geometry_path,
        cfg=cfg,
        root_goal_s=best_root_goal_s,
        d_pre=float(best_row["d_pre_m"]),
        y_tol=float(best_row["y_tol_m"]),
        yaw_tol_deg=float(best_row["yaw_tol_deg"]),
    )

    payload_by_id = {case.case_id: payload for case, payload in zip(best_cases, best_payloads, strict=True)}
    representative_cases: list[dict[str, str]] = []
    for case_id in select_representative_case_ids(best_cases):
        case = next(case for case in best_cases if case.case_id == case_id)
        image_path = out_dir / stamped_name(case.case_id, run_timestamp, ".png")
        draw_goal_set_case(
            out_path=image_path,
            case=case,
            payload=payload_by_id[case_id],
            pallet_xy=pallet_xy,
            pallet_yaw_deg=pallet_yaw_deg,
            root_goal_s=best_root_goal_s,
            d_pre=float(best_row["d_pre_m"]),
            y_tol=float(best_row["y_tol_m"]),
            yaw_tol_deg=float(best_row["yaw_tol_deg"]),
            cfg_path=args.cfg_path,
        )
        representative_cases.append({"case_id": case_id, "image_path": str(image_path)})

    best_combo = dict(best_row)
    best_combo["overlay_path"] = str(overlay_path)
    best_combo["geometry_path"] = str(geometry_path)
    best_combo["representative_cases"] = representative_cases

    summary = {
        "tool": "run_exp83_predock_goal_set_audit.py",
        "cfg_path": str(args.cfg_path),
        "scan_mode": "current_stage1_near_field_to_predock_goal_set",
        "note": "Goal-set search uses world-motion-aware reverse stats from sampled RS paths, not only library segment signs.",
        "top_rows": rows[:10],
        "best_combo": best_combo,
    }
    manifest = {
        "summary": summary,
        "cases": [asdict(case) for case in best_cases],
    }

    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    summary_json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(summary_md_path, summary)

    print(f"[predock_goal_set_audit] best d_pre={best_row['d_pre_m']:.2f} y_tol={best_row['y_tol_m']:.2f} yaw_tol={best_row['yaw_tol_deg']:.1f}")
    print(f"[predock_goal_set_audit] overlay: {overlay_path}")
    print(f"[predock_goal_set_audit] geometry: {geometry_path}")
    print(f"[predock_goal_set_audit] manifest: {manifest_path}")
    print(f"[predock_goal_set_audit] summary_json: {summary_json_path}")
    print(f"[predock_goal_set_audit] summary_md: {summary_md_path}")


if __name__ == "__main__":
    main()
