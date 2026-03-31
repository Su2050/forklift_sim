#!/usr/bin/env python3
"""Visualize current Stage-1 reference trajectory entry geometry.

This script is intentionally pure Python:
- no Isaac Lab runtime required
- reads the current task defaults from env_cfg.py via AST
- mirrors the trajectory construction logic in env.py

Outputs:
- per-case top-down PNGs
- one overlay summary PNG
- a manifest JSON with s_start / s_pre / s_goal / delta_s
"""

from __future__ import annotations

import argparse
import ast
from datetime import datetime
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[2]
PATCH_CFG_PATH = (
    PROJECT_ROOT
    / "isaaclab_patch/source/isaaclab_tasks/isaaclab_tasks/direct/forklift_pallet_insert_lift/env_cfg.py"
)
INSTALLED_CFG_PATH = (
    REPO_ROOT
    / "IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/forklift_pallet_insert_lift/env_cfg.py"
)
CFG_PATH = PATCH_CFG_PATH

RS_DIR = (
    PROJECT_ROOT
    / "isaaclab_patch/source/isaaclab_tasks/isaaclab_tasks/direct/forklift_pallet_insert_lift/rs"
)
if str(RS_DIR) not in sys.path:
    sys.path.append(str(RS_DIR))
import rs as exact_rs

FORK_CENTER_BACKOFF_M = 0.6

CFG_KEYS = {
    "stage1_init_x_min_m",
    "stage1_init_x_max_m",
    "stage1_init_y_min_m",
    "stage1_init_y_max_m",
    "stage1_init_yaw_deg_min",
    "stage1_init_yaw_deg_max",
    "pallet_depth_m",
    "insert_fraction",
    "traj_pre_dist_m",
    "traj_vehicle_curve_min_span_m",
    "traj_vehicle_final_straight_min_m",
    "traj_num_samples",
    "fork_reach_m",
    "exp83_traj_goal_mode",
    "traj_model",
    "traj_rs_min_turn_radius_m",
    "traj_rs_sample_step_m",
    "traj_rs_forward_preferred_max_candidates",
    "traj_rs_forward_preferred_max_extra_length_m",
    "traj_rs_forward_preferred_max_reverse_frac",
    "traj_rs_forward_preferred_max_direction_switches",
    "traj_rs_forward_preferred_require_final_forward",
    "traj_rs_forward_preferred_reverse_weight",
    "traj_rs_forward_preferred_switch_weight",
    "traj_rs_forward_preferred_terminal_reverse_penalty",
}


@dataclass
class CaseMetrics:
    case_id: str
    root_x: float
    root_y: float
    yaw_deg: float
    s_start: float
    s_pre: float
    s_goal: float
    delta_s: float
    y_start: float
    root_y_abs_max: float
    root_heading_change_deg: float
    root_curvature_max: float
    entry_ok: bool
    path_mode: str


def _literal_value(node: ast.AST):
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -_literal_value(node.operand)
    if isinstance(node, ast.Tuple):
        return tuple(_literal_value(elt) for elt in node.elts)
    if isinstance(node, ast.List):
        return [_literal_value(elt) for elt in node.elts]
    raise ValueError(f"unsupported literal node: {type(node).__name__}")


def load_cfg_defaults(cfg_path: Path) -> dict[str, object]:
    tree = ast.parse(cfg_path.read_text(encoding="utf-8"), filename=str(cfg_path))
    values: dict[str, object] = {}

    for node in tree.body:
        if not isinstance(node, ast.ClassDef) or node.name != "ForkliftPalletInsertLiftEnvCfg":
            continue
        for stmt in node.body:
            name = None
            value_node = None
            if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                name = stmt.target.id
                value_node = stmt.value
            elif isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
                name = stmt.targets[0].id
                value_node = stmt.value
            if name in CFG_KEYS and value_node is not None:
                try:
                    values[name] = _literal_value(value_node)
                except ValueError:
                    pass
        break

    missing = sorted(CFG_KEYS - set(values))
    if missing:
        raise RuntimeError(f"failed to parse cfg defaults for keys: {missing}")
    return values


def exp83_traj_goal_s(*, pallet_depth_m: float, insert_fraction: float, mode: str) -> float:
    s_front = -0.5 * pallet_depth_m
    if mode == "front":
        return s_front
    if mode == "success_center":
        return s_front + (insert_fraction * pallet_depth_m - FORK_CENTER_BACKOFF_M)
    raise ValueError(f"unsupported exp83_traj_goal_mode: {mode}")


def _compute_path_tangents(pts: np.ndarray, fallback_dir: np.ndarray) -> np.ndarray:
    diffs = np.diff(pts, axis=0)
    norms = np.linalg.norm(diffs, axis=1, keepdims=True)
    safe_dirs = np.where(norms > 1e-9, diffs / np.maximum(norms, 1e-9), fallback_dir.reshape(1, 2))
    return np.concatenate([safe_dirs, safe_dirs[-1:, :]], axis=0)


def compute_path_heading_curvature(
    pts: np.ndarray,
    *,
    fallback_dir: np.ndarray,
) -> tuple[float, float]:
    tangents = _compute_path_tangents(pts, fallback_dir)
    headings = np.unwrap(np.arctan2(tangents[:, 1], tangents[:, 0]))
    heading_change_deg = float(abs(headings[-1] - headings[0]) * (180.0 / math.pi))
    if len(pts) < 2:
        return heading_change_deg, 0.0
    ds = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    dtheta = np.diff(headings)
    curvature = np.abs(dtheta) / np.maximum(ds, 1e-9)
    curvature_max = float(np.max(curvature)) if curvature.size else 0.0
    return heading_change_deg, curvature_max


def wrap_angle_np(theta: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(theta), np.cos(theta))


def resample_pose_sequence_np(
    xy: np.ndarray,
    yaw: np.ndarray,
    *,
    num_samples: int,
) -> tuple[np.ndarray, np.ndarray]:
    if xy.shape[0] == 1:
        return np.repeat(xy, num_samples, axis=0), np.repeat(yaw, num_samples, axis=0)

    diffs = np.diff(xy, axis=0)
    seg_lens = np.linalg.norm(diffs, axis=1)
    s = np.concatenate([np.zeros((1,), dtype=np.float64), np.cumsum(seg_lens, dtype=np.float64)], axis=0)
    if float(s[-1]) < 1e-9:
        return np.repeat(xy[:1], num_samples, axis=0), np.repeat(yaw[:1], num_samples, axis=0)

    s_new = np.linspace(0.0, float(s[-1]), num_samples, dtype=np.float64)
    x_new = np.interp(s_new, s, xy[:, 0])
    y_new = np.interp(s_new, s, xy[:, 1])
    yaw_unwrapped = np.unwrap(yaw)
    yaw_new = np.interp(s_new, s, yaw_unwrapped)
    return np.stack([x_new, y_new], axis=1), wrap_angle_np(yaw_new)


def sample_rs_root_path_np(
    *,
    root_start_xy: np.ndarray,
    root_start_yaw: float,
    root_goal_xy: np.ndarray,
    root_goal_yaw: float,
    min_turn_radius_m: float,
    sample_step_m: float,
    num_samples: int,
) -> tuple[np.ndarray, np.ndarray]:
    world_pts = exact_rs.rs_sample_path(
        float(root_start_xy[0]),
        float(root_start_xy[1]),
        float(root_start_yaw),
        float(root_goal_xy[0]),
        float(root_goal_xy[1]),
        float(root_goal_yaw),
        float(min_turn_radius_m),
        step=float(sample_step_m),
    )
    if not world_pts:
        raise RuntimeError("RS sampler failed to find a path")
    xy = np.asarray([[pt[0], pt[1]] for pt in world_pts], dtype=np.float64)
    yaw = np.asarray([pt[2] for pt in world_pts], dtype=np.float64)
    root_pts, root_yaw = resample_pose_sequence_np(xy, yaw, num_samples=num_samples)
    root_tangents = np.stack([np.cos(root_yaw), np.sin(root_yaw)], axis=1)
    return root_pts, root_tangents


def rs_local_goal(start_xy: np.ndarray, start_yaw: float, goal_xy: np.ndarray, goal_yaw: float) -> tuple[float, float, float]:
    dx = float(goal_xy[0] - start_xy[0])
    dy = float(goal_xy[1] - start_xy[1])
    cos_t = math.cos(float(start_yaw))
    sin_t = math.sin(float(start_yaw))
    lx = dx * cos_t + dy * sin_t
    ly = -dx * sin_t + dy * cos_t
    lphi = math.atan2(math.sin(float(goal_yaw - start_yaw)), math.cos(float(goal_yaw - start_yaw)))
    return -lx, -ly, lphi


def rs_candidate_stats(segs: list[tuple[str, float]]) -> dict[str, float | int | bool]:
    total = float(sum(abs(seg_len) for _, seg_len in segs))
    reverse = float(sum(abs(seg_len) for _, seg_len in segs if seg_len < 0.0))
    switches = sum(
        (segs[i][1] >= 0.0) != (segs[i - 1][1] >= 0.0)
        for i in range(1, len(segs))
    )
    final_forward = bool(segs and segs[-1][1] > 0.0)
    return {
        "total_length_m": total,
        "reverse_length_m": reverse,
        "reverse_frac": reverse / max(total, 1e-9),
        "direction_switches": int(switches),
        "final_forward": final_forward,
    }


def sample_forward_preferred_rs_root_path_np(
    *,
    root_start_xy: np.ndarray,
    root_start_yaw: float,
    root_goal_xy: np.ndarray,
    root_goal_yaw: float,
    min_turn_radius_m: float,
    sample_step_m: float,
    num_samples: int,
    max_candidates: int,
    max_extra_length_m: float,
    max_reverse_frac: float,
    max_direction_switches: int,
    require_final_forward: bool,
    reverse_weight: float,
    switch_weight: float,
    terminal_reverse_penalty: float,
) -> tuple[np.ndarray, np.ndarray] | None:
    x_rs_goal, y_rs_goal, th_rs_goal = rs_local_goal(root_start_xy, root_start_yaw, root_goal_xy, root_goal_yaw)
    all_segs = exact_rs.rs_all_paths(x_rs_goal, y_rs_goal, th_rs_goal, float(min_turn_radius_m))
    if not all_segs:
        return None

    shortest_total = float(sum(abs(seg_len) for _, seg_len in all_segs[0]))
    sampled_paths = exact_rs.rs_sample_path_multi(
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
    candidates: list[tuple[float, np.ndarray, np.ndarray]] = []
    for segs, world_pts in zip(all_segs[: len(sampled_paths)], sampled_paths, strict=True):
        stats = rs_candidate_stats(segs)
        if float(stats["total_length_m"]) > shortest_total + float(max_extra_length_m):
            continue
        if float(stats["reverse_frac"]) > float(max_reverse_frac):
            continue
        if int(stats["direction_switches"]) > int(max_direction_switches):
            continue
        if bool(require_final_forward) and not bool(stats["final_forward"]):
            continue
        score = (
            float(stats["total_length_m"])
            + float(reverse_weight) * float(stats["reverse_length_m"])
            + float(switch_weight) * int(stats["direction_switches"])
            + (0.0 if bool(stats["final_forward"]) else float(terminal_reverse_penalty))
        )
        xy = np.asarray([[pt[0], pt[1]] for pt in world_pts], dtype=np.float64)
        yaw = np.asarray([pt[2] for pt in world_pts], dtype=np.float64)
        root_pts, root_yaw = resample_pose_sequence_np(xy, yaw, num_samples=num_samples)
        root_tangents = np.stack([np.cos(root_yaw), np.sin(root_yaw)], axis=1)
        candidates.append((score, root_pts, root_tangents))

    if not candidates:
        return None
    best = min(candidates, key=lambda item: item[0])
    return best[1], best[2]


def build_reference_trajectory(
    *,
    root_xy: np.ndarray,
    yaw_deg: float,
    pallet_xy: np.ndarray,
    pallet_yaw_deg: float,
    fork_reach_m: float,
    pallet_depth_m: float,
    traj_pre_dist_m: float,
    traj_vehicle_curve_min_span_m: float,
    traj_vehicle_final_straight_min_m: float,
    traj_num_samples: int,
    insert_fraction: float,
    traj_goal_mode: str,
    traj_model: str,
    traj_rs_min_turn_radius_m: float,
    traj_rs_sample_step_m: float,
    traj_rs_forward_preferred_max_candidates: int,
    traj_rs_forward_preferred_max_extra_length_m: float,
    traj_rs_forward_preferred_max_reverse_frac: float,
    traj_rs_forward_preferred_max_direction_switches: int,
    traj_rs_forward_preferred_require_final_forward: bool,
    traj_rs_forward_preferred_reverse_weight: float,
    traj_rs_forward_preferred_switch_weight: float,
    traj_rs_forward_preferred_terminal_reverse_penalty: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    yaw = math.radians(yaw_deg)
    pallet_yaw = math.radians(pallet_yaw_deg)

    u_robot = np.array([math.cos(yaw), math.sin(yaw)], dtype=np.float64)
    u_in = np.array([math.cos(pallet_yaw), math.sin(pallet_yaw)], dtype=np.float64)
    v_lat = np.array([-math.sin(pallet_yaw), math.cos(pallet_yaw)], dtype=np.float64)

    root_to_fc = fork_reach_m - FORK_CENTER_BACKOFF_M
    fork_center_xy = root_xy + root_to_fc * u_robot

    s_goal = exp83_traj_goal_s(
        pallet_depth_m=pallet_depth_m,
        insert_fraction=insert_fraction,
        mode=traj_goal_mode,
    )
    p_goal = pallet_xy + s_goal * u_in
    root_goal = p_goal - root_to_fc * u_in
    root_start_s, root_start_y = project_axis(root_xy, pallet_xy, pallet_yaw_deg)
    root_goal_s, _ = project_axis(root_goal, pallet_xy, pallet_yaw_deg)
    root_pre_nominal_s = root_goal_s - traj_pre_dist_m
    root_pre_s = max(root_pre_nominal_s, root_start_s + traj_vehicle_curve_min_span_m)
    root_pre_s = min(root_pre_s, root_goal_s - traj_vehicle_final_straight_min_m)
    root_pre = pallet_xy + root_pre_s * u_in
    p_pre = root_pre + root_to_fc * u_in

    path_mode = traj_model
    if traj_model == "rs_exact":
        root_path, root_tangents = sample_rs_root_path_np(
            root_start_xy=root_xy,
            root_start_yaw=yaw,
            root_goal_xy=root_goal,
            root_goal_yaw=pallet_yaw,
            min_turn_radius_m=traj_rs_min_turn_radius_m,
            sample_step_m=traj_rs_sample_step_m,
            num_samples=traj_num_samples,
        )
    elif traj_model == "rs_forward_preferred":
        rs_payload = sample_forward_preferred_rs_root_path_np(
            root_start_xy=root_xy,
            root_start_yaw=yaw,
            root_goal_xy=root_goal,
            root_goal_yaw=pallet_yaw,
            min_turn_radius_m=traj_rs_min_turn_radius_m,
            sample_step_m=traj_rs_sample_step_m,
            num_samples=traj_num_samples,
            max_candidates=traj_rs_forward_preferred_max_candidates,
            max_extra_length_m=traj_rs_forward_preferred_max_extra_length_m,
            max_reverse_frac=traj_rs_forward_preferred_max_reverse_frac,
            max_direction_switches=traj_rs_forward_preferred_max_direction_switches,
            require_final_forward=traj_rs_forward_preferred_require_final_forward,
            reverse_weight=traj_rs_forward_preferred_reverse_weight,
            switch_weight=traj_rs_forward_preferred_switch_weight,
            terminal_reverse_penalty=traj_rs_forward_preferred_terminal_reverse_penalty,
        )
        if rs_payload is None:
            path_mode = "rs_forward_preferred_fallback_root_path_first"
        else:
            root_path, root_tangents = rs_payload
            path_mode = "rs_forward_preferred"
    if path_mode == "rs_forward_preferred_fallback_root_path_first":
        traj_model = "root_path_first"
    if traj_model == "root_path_first":
        num_curve = int(traj_num_samples * 0.7)
        num_line = traj_num_samples - num_curve

        span_s = max(root_pre_s - root_start_s, 1e-6)
        yaw_rel0 = math.atan2(math.sin(yaw - pallet_yaw), math.cos(yaw - pallet_yaw))
        slope0 = math.tan(yaw_rel0)
        a = (slope0 * span_s + 2.0 * root_start_y) / (span_s**3)
        b = (-2.0 * slope0 * span_s - 3.0 * root_start_y) / (span_s**2)

        ds_curve = np.linspace(0.0, span_s, num_curve, dtype=np.float64).reshape(-1, 1)
        y_curve = a * ds_curve**3 + b * ds_curve**2 + slope0 * ds_curve + root_start_y
        dy_ds = 3.0 * a * ds_curve**2 + 2.0 * b * ds_curve + slope0
        s_curve = root_start_s + ds_curve
        root_pts_curve = pallet_xy + s_curve * u_in + y_curve * v_lat
        root_curve_dirs = u_in.reshape(1, 2) + dy_ds * v_lat.reshape(1, 2)
        root_curve_tangents = root_curve_dirs / np.maximum(
            np.linalg.norm(root_curve_dirs, axis=1, keepdims=True),
            1e-9,
        )
        root_curve_tangents[0] = u_robot
        root_curve_tangents[-1] = u_in

        if num_line > 0:
            t_line = np.linspace(0.0, 1.0, num_line + 1, dtype=np.float64).reshape(-1, 1)[1:]
            root_pts_line = (1 - t_line) * root_pre + t_line * root_goal
            root_line_tangents = np.repeat(u_in.reshape(1, 2), num_line, axis=0)
            root_path = np.concatenate([root_pts_curve, root_pts_line], axis=0)
            root_tangents = np.concatenate([root_curve_tangents, root_line_tangents], axis=0)
        else:
            root_path = root_pts_curve
            root_tangents = root_curve_tangents
    elif traj_model != "root_path_first":
        raise ValueError(f"unsupported traj_model: {traj_model}")

    pts = root_path + root_to_fc * root_tangents
    tangents = np.copy(root_tangents)
    return fork_center_xy, p_pre, p_goal, pts, tangents, root_path, path_mode


def project_axis(point_xy: np.ndarray, pallet_xy: np.ndarray, pallet_yaw_deg: float) -> tuple[float, float]:
    pallet_yaw = math.radians(pallet_yaw_deg)
    u_in = np.array([math.cos(pallet_yaw), math.sin(pallet_yaw)], dtype=np.float64)
    v_lat = np.array([-math.sin(pallet_yaw), math.cos(pallet_yaw)], dtype=np.float64)
    rel = point_xy - pallet_xy
    s = float(np.dot(rel, u_in))
    y = float(np.dot(rel, v_lat))
    return s, y


def _pretty_cfg_path(cfg_path: Path) -> str:
    try:
        return str(cfg_path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return cfg_path.name


def draw_case(
    *,
    out_path: Path,
    case: CaseMetrics,
    root_xy: np.ndarray,
    fork_center_xy: np.ndarray,
    p_pre: np.ndarray,
    p_goal: np.ndarray,
    pts: np.ndarray,
    root_path: np.ndarray,
    pallet_xy: np.ndarray,
    pallet_yaw_deg: float,
    cfg_path: Path,
    traj_model: str,
):
    pallet_yaw = math.radians(pallet_yaw_deg)
    u_in = np.array([math.cos(pallet_yaw), math.sin(pallet_yaw)], dtype=np.float64)
    v_lat = np.array([-math.sin(pallet_yaw), math.cos(pallet_yaw)], dtype=np.float64)

    fig = plt.figure(figsize=(11.5, 7.2), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[3.8, 1.6], wspace=0.08)
    ax = fig.add_subplot(gs[0, 0])
    ax_info = fig.add_subplot(gs[0, 1])
    ax_info.axis("off")

    axis_s = np.linspace(case.s_pre - 0.6, case.s_goal + 0.6, 200)
    axis_pts = pallet_xy + axis_s[:, None] * u_in
    ax.plot(axis_pts[:, 0], axis_pts[:, 1], "--", color="#888888", lw=1.2, label="pallet s-axis")

    lat_line = np.stack([pallet_xy - 0.25 * v_lat, pallet_xy + 0.25 * v_lat], axis=0)
    ax.plot(lat_line[:, 0], lat_line[:, 1], "-", color="#bbbbbb", lw=1.0)

    ax.plot(root_path[:, 0], root_path[:, 1], color="#444444", lw=2.0, ls="--", alpha=0.95, label="vehicle/root reference trajectory")
    ax.plot(pts[:, 0], pts[:, 1], color="#1f77b4", lw=1.8, label="mapped fork-center trajectory")
    ax.scatter(pts[0, 0], pts[0, 1], color="#1f77b4", s=24)
    ax.scatter(root_xy[0], root_xy[1], color="#444444", s=36, label="robot root")
    ax.scatter(fork_center_xy[0], fork_center_xy[1], color="#d62728", s=42, label="fork_center start")
    ax.scatter(p_pre[0], p_pre[1], color="#ff7f0e", s=42, label="p_pre")
    ax.scatter(p_goal[0], p_goal[1], color="#2ca02c", s=42, label="p_goal")
    ax.scatter(pallet_xy[0], pallet_xy[1], color="#9467bd", s=40, label="pallet center")

    heading_vec = fork_center_xy - root_xy
    if np.linalg.norm(heading_vec) > 1e-9:
        hv = heading_vec / np.linalg.norm(heading_vec)
        ax.arrow(
            fork_center_xy[0],
            fork_center_xy[1],
            0.25 * hv[0],
            0.25 * hv[1],
            width=0.01,
            head_width=0.06,
            head_length=0.08,
            color="#d62728",
            length_includes_head=True,
        )

    entry_text = "OK" if case.entry_ok else "AHEAD"
    entry_color = "#2ca02c" if case.entry_ok else "#d62728"
    fig.suptitle(
        f"{case.case_id} | delta_s={case.delta_s:+.3f} m | entry={entry_text}",
        color=entry_color,
        fontsize=13,
    )
    info_lines = [
        "Metrics",
        f"root = ({case.root_x:+.3f}, {case.root_y:+.3f})",
        f"yaw = {case.yaw_deg:+.1f} deg",
        f"s_start = {case.s_start:+.4f}",
        f"s_pre = {case.s_pre:+.4f}",
        f"s_goal = {case.s_goal:+.4f}",
        f"delta_s = {case.delta_s:+.4f}",
        f"y_start = {case.y_start:+.4f}",
        f"root|y|_max = {case.root_y_abs_max:+.4f}",
        f"root_dpsi = {case.root_heading_change_deg:+.2f} deg",
        f"root_kappa_max = {case.root_curvature_max:+.3f} 1/m",
        "",
        "Scope",
        f"traj_model = {traj_model}",
        f"path_mode = {case.path_mode}",
        "vehicle-reference-first",
        "fork_center path is mapped from vehicle path",
        "still proxy-level, not full dynamics",
        "",
        "Expectation",
        "vehicle path first, then fork path",
        "",
        "Cfg source",
        _pretty_cfg_path(cfg_path),
    ]
    ax_info.text(
        0.0,
        0.98,
        "\n".join(info_lines),
        ha="left",
        va="top",
        fontsize=10,
        family="monospace",
        linespacing=1.35,
        bbox={"boxstyle": "round", "facecolor": "#f7f7f7", "edgecolor": "#d9d9d9", "alpha": 0.98},
    )

    ax.set_aspect("equal", adjustable="box")
    ax.set_box_aspect(1.0)
    ax.grid(True, alpha=0.25)
    ax.set_xlabel("world x (m)")
    ax.set_ylabel("world y (m)")

    bbox_points = np.vstack(
        [
            axis_pts[[0, -1]],
            lat_line,
            pts,
            root_xy[None, :],
            fork_center_xy[None, :],
            p_pre[None, :],
            p_goal[None, :],
            pallet_xy[None, :],
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

    handles, labels = ax.get_legend_handles_labels()
    ax_info.legend(handles, labels, loc="lower left", fontsize=9, frameon=False)

    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def draw_overlay(
    *,
    out_path: Path,
    cases: list[CaseMetrics],
    overlay_payloads: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    pallet_xy: np.ndarray,
    pallet_yaw_deg: float,
    traj_model: str,
):
    pallet_yaw = math.radians(pallet_yaw_deg)
    u_in = np.array([math.cos(pallet_yaw), math.sin(pallet_yaw)], dtype=np.float64)

    fig, ax = plt.subplots(figsize=(9, 9))

    s_vals = [case.s_pre for case in cases] + [case.s_goal for case in cases]
    axis_s = np.linspace(min(s_vals) - 0.8, max(s_vals) + 0.8, 200)
    axis_pts = pallet_xy + axis_s[:, None] * u_in
    ax.plot(axis_pts[:, 0], axis_pts[:, 1], "--", color="#999999", lw=1.2, label="pallet s-axis")

    any_bad = False
    label_points = len(cases) <= 40
    for case, payload in zip(cases, overlay_payloads, strict=True):
        root_xy, fork_center_xy, p_pre, p_goal, pts, _, root_path = payload
        ax.plot(root_path[:, 0], root_path[:, 1], color="#444444", alpha=0.18, lw=1.4)
        ax.plot(pts[:, 0], pts[:, 1], color="#1f77b4", alpha=0.22, lw=1.5)
        marker_color = "#2ca02c" if case.entry_ok else "#d62728"
        if not case.entry_ok:
            any_bad = True
        ax.scatter(fork_center_xy[0], fork_center_xy[1], color=marker_color, s=26)
        if label_points:
            ax.text(fork_center_xy[0], fork_center_xy[1], case.case_id, fontsize=7, color=marker_color)
        ax.scatter(p_pre[0], p_pre[1], color="#ff7f0e", s=12, alpha=0.35)
        ax.scatter(p_goal[0], p_goal[1], color="#2ca02c", s=12, alpha=0.35)

    n_bad = sum(1 for case in cases if not case.entry_ok)
    status = f"{traj_model}: {n_bad}/{len(cases)} cases with s_start >= s_pre"
    ax.set_title(status, color=("#d62728" if any_bad else "#2ca02c"), fontsize=13)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    ax.set_xlabel("world x (m)")
    ax.set_ylabel("world y (m)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def format_case_id(index: int, root_x: float, root_y: float, yaw_deg: float) -> str:
    def token(v: float, scale: int = 1000) -> str:
        sign = "p" if v >= 0 else "m"
        return f"{sign}{abs(v):.3f}".replace(".", "p")

    return f"c{index:02d}_x{token(root_x)}_y{token(root_y)}_yaw{token(yaw_deg, 10)}"


def stamped_name(base_name: str, run_timestamp: str, suffix: str) -> str:
    return f"{base_name}_{run_timestamp}{suffix}"


def build_case_grid(
    cfg: dict[str, object],
    pallet_xy: np.ndarray,
    pallet_yaw_deg: float,
    *,
    grid_count_x: int,
    grid_count_y: int,
    grid_count_yaw: int,
) -> tuple[list[CaseMetrics], list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]]:
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

    cases: list[CaseMetrics] = []
    payloads: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []

    idx = 1
    for root_x in x_vals:
        for root_y in y_vals:
            for yaw_deg in yaw_vals:
                root_xy = np.array([root_x, root_y], dtype=np.float64)
                fork_center_xy, p_pre, p_goal, pts, tangents, root_path, path_mode = build_reference_trajectory(
                    root_xy=root_xy,
                    yaw_deg=yaw_deg,
                    pallet_xy=pallet_xy,
                    pallet_yaw_deg=pallet_yaw_deg,
                    fork_reach_m=float(cfg["fork_reach_m"]),
                    pallet_depth_m=float(cfg["pallet_depth_m"]),
                    traj_pre_dist_m=float(cfg["traj_pre_dist_m"]),
                    traj_vehicle_curve_min_span_m=float(cfg["traj_vehicle_curve_min_span_m"]),
                    traj_vehicle_final_straight_min_m=float(cfg["traj_vehicle_final_straight_min_m"]),
                    traj_num_samples=int(cfg["traj_num_samples"]),
                    insert_fraction=float(cfg["insert_fraction"]),
                    traj_goal_mode=str(cfg["exp83_traj_goal_mode"]),
                    traj_model=str(cfg.get("traj_model", "root_path_first")),
                    traj_rs_min_turn_radius_m=float(cfg.get("traj_rs_min_turn_radius_m", 2.34)),
                    traj_rs_sample_step_m=float(cfg.get("traj_rs_sample_step_m", 0.05)),
                    traj_rs_forward_preferred_max_candidates=int(cfg.get("traj_rs_forward_preferred_max_candidates", 8)),
                    traj_rs_forward_preferred_max_extra_length_m=float(cfg.get("traj_rs_forward_preferred_max_extra_length_m", 1.5)),
                    traj_rs_forward_preferred_max_reverse_frac=float(cfg.get("traj_rs_forward_preferred_max_reverse_frac", 0.35)),
                    traj_rs_forward_preferred_max_direction_switches=int(cfg.get("traj_rs_forward_preferred_max_direction_switches", 1)),
                    traj_rs_forward_preferred_require_final_forward=bool(cfg.get("traj_rs_forward_preferred_require_final_forward", True)),
                    traj_rs_forward_preferred_reverse_weight=float(cfg.get("traj_rs_forward_preferred_reverse_weight", 3.0)),
                    traj_rs_forward_preferred_switch_weight=float(cfg.get("traj_rs_forward_preferred_switch_weight", 0.8)),
                    traj_rs_forward_preferred_terminal_reverse_penalty=float(cfg.get("traj_rs_forward_preferred_terminal_reverse_penalty", 2.0)),
                )
                s_start, y_start = project_axis(fork_center_xy, pallet_xy, pallet_yaw_deg)
                s_pre, _ = project_axis(p_pre, pallet_xy, pallet_yaw_deg)
                s_goal, _ = project_axis(p_goal, pallet_xy, pallet_yaw_deg)
                root_y_abs_max = max(
                    abs(project_axis(point_xy, pallet_xy, pallet_yaw_deg)[1]) for point_xy in root_path
                )
                yaw_rad = math.radians(yaw_deg)
                root_heading_change_deg, root_curvature_max = compute_path_heading_curvature(
                    root_path,
                    fallback_dir=np.array([math.cos(yaw_rad), math.sin(yaw_rad)], dtype=np.float64),
                )
                case = CaseMetrics(
                    case_id=format_case_id(idx, root_x, root_y, yaw_deg),
                    root_x=root_x,
                    root_y=root_y,
                    yaw_deg=yaw_deg,
                    s_start=s_start,
                    s_pre=s_pre,
                    s_goal=s_goal,
                    delta_s=s_start - s_pre,
                    y_start=y_start,
                    root_y_abs_max=root_y_abs_max,
                    root_heading_change_deg=root_heading_change_deg,
                    root_curvature_max=root_curvature_max,
                    entry_ok=(s_start < s_pre < s_goal),
                    path_mode=path_mode,
                )
                cases.append(case)
                payloads.append((root_xy, fork_center_xy, p_pre, p_goal, pts, tangents, root_path))
                idx += 1

    return cases, payloads


def main():
    parser = argparse.ArgumentParser(description="Visualize current Stage-1 reference trajectory entry geometry.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to write PNGs and manifest JSON.",
    )
    parser.add_argument("--pallet-x", type=float, default=0.0)
    parser.add_argument("--pallet-y", type=float, default=0.0)
    parser.add_argument("--pallet-yaw-deg", type=float, default=0.0)
    parser.add_argument("--grid-count-x", type=int, default=5)
    parser.add_argument("--grid-count-y", type=int, default=5)
    parser.add_argument("--grid-count-yaw", type=int, default=5)
    parser.add_argument(
        "--traj-model",
        type=str,
        choices=["root_path_first", "rs_exact", "rs_forward_preferred"],
        default=None,
        help="Override traj_model from cfg for auditing.",
    )
    parser.add_argument(
        "--cfg-path",
        type=Path,
        default=CFG_PATH,
        help="Env cfg path to read. Defaults to the project patch env_cfg.py.",
    )
    parser.add_argument(
        "--timestamp-tag",
        type=str,
        default=None,
        help="Optional run timestamp tag. Defaults to current local time YYYYMMDD_HHMMSS.",
    )
    args = parser.parse_args()

    cfg = load_cfg_defaults(args.cfg_path)
    pallet_xy = np.array([args.pallet_x, args.pallet_y], dtype=np.float64)
    pallet_yaw_deg = float(args.pallet_yaw_deg)
    if args.traj_model is not None:
        cfg["traj_model"] = str(args.traj_model)
    traj_model = str(cfg.get("traj_model", "root_path_first"))

    if args.output_dir is None:
        out_dir = PROJECT_ROOT / "outputs" / f"reference_trajectory_stage1_viz_{traj_model}"
    else:
        out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    run_timestamp = args.timestamp_tag or datetime.now().strftime("%Y%m%d_%H%M%S")

    overlay_path = out_dir / stamped_name("overlay_all_cases", run_timestamp, ".png")
    manifest_path = out_dir / stamped_name("reference_trajectory_stage1_manifest", run_timestamp, ".json")

    cases, payloads = build_case_grid(
        cfg,
        pallet_xy,
        pallet_yaw_deg,
        grid_count_x=int(args.grid_count_x),
        grid_count_y=int(args.grid_count_y),
        grid_count_yaw=int(args.grid_count_yaw),
    )
    for case, payload in zip(cases, payloads, strict=True):
        root_xy, fork_center_xy, p_pre, p_goal, pts, _, root_path = payload
        draw_case(
            out_path=out_dir / stamped_name(case.case_id, run_timestamp, ".png"),
            case=case,
            root_xy=root_xy,
            fork_center_xy=fork_center_xy,
            p_pre=p_pre,
            p_goal=p_goal,
            pts=pts,
            root_path=root_path,
            pallet_xy=pallet_xy,
            pallet_yaw_deg=pallet_yaw_deg,
            cfg_path=args.cfg_path,
            traj_model=traj_model,
        )

    draw_overlay(
        out_path=overlay_path,
        cases=cases,
        overlay_payloads=payloads,
        pallet_xy=pallet_xy,
        pallet_yaw_deg=pallet_yaw_deg,
        traj_model=traj_model,
    )

    summary = {
        "cfg_path": str(args.cfg_path),
        "run_timestamp": run_timestamp,
        "validation_scope": [
            f"traj_model={traj_model}",
            "vehicle_reference_path_first",
            "fork_center_path_mapped_from_vehicle_reference",
            "proxy_level_check_not_full_dynamics",
            "does_not_check_full_wheel_contact_or_chassis_sweep",
        ],
        "cfg_path_used": str(args.cfg_path),
        "parsed_cfg": cfg,
        "pallet_xy": pallet_xy.tolist(),
        "pallet_yaw_deg": pallet_yaw_deg,
        "grid_count_x": int(args.grid_count_x),
        "grid_count_y": int(args.grid_count_y),
        "grid_count_yaw": int(args.grid_count_yaw),
        "overlay_path": str(overlay_path),
        "num_cases": len(cases),
        "num_entry_ok": sum(1 for case in cases if case.entry_ok),
        "num_entry_bad": sum(1 for case in cases if not case.entry_ok),
        "path_mode_counts": {
            mode: sum(1 for case in cases if case.path_mode == mode)
            for mode in sorted({case.path_mode for case in cases})
        },
        "delta_s_min": min(case.delta_s for case in cases),
        "delta_s_max": max(case.delta_s for case in cases),
        "delta_s_mean": sum(case.delta_s for case in cases) / len(cases),
        "root_y_abs_max_max": max(case.root_y_abs_max for case in cases),
        "root_heading_change_deg_max": max(case.root_heading_change_deg for case in cases),
        "root_curvature_max_max": max(case.root_curvature_max for case in cases),
        "cases": [asdict(case) for case in cases],
    }
    manifest_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[traj_viz] run_timestamp: {run_timestamp}")
    print(f"[traj_viz] wrote {len(cases)} case PNGs to: {out_dir}")
    print(f"[traj_viz] overlay: {overlay_path}")
    print(f"[traj_viz] manifest: {manifest_path}")
    print(
        "[traj_viz] delta_s stats: "
        f"min={summary['delta_s_min']:+.4f}, "
        f"max={summary['delta_s_max']:+.4f}, "
        f"mean={summary['delta_s_mean']:+.4f}"
    )


if __name__ == "__main__":
    main()
