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
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[2]
LEGACY_CFG_PATH = (
    PROJECT_ROOT
    / "isaaclab_patch/source/isaaclab_tasks/isaaclab_tasks/direct/forklift_pallet_insert_lift/env_cfg.py"
)
CURRENT_CFG_PATH = (
    REPO_ROOT
    / "IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/forklift_pallet_insert_lift/env_cfg.py"
)
CFG_PATH = CURRENT_CFG_PATH if CURRENT_CFG_PATH.exists() else LEGACY_CFG_PATH

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
    "traj_num_samples",
    "fork_reach_m",
    "exp83_traj_goal_mode",
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


def build_reference_trajectory(
    *,
    root_xy: np.ndarray,
    yaw_deg: float,
    pallet_xy: np.ndarray,
    pallet_yaw_deg: float,
    fork_reach_m: float,
    pallet_depth_m: float,
    traj_pre_dist_m: float,
    traj_num_samples: int,
    insert_fraction: float,
    traj_goal_mode: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    yaw = math.radians(yaw_deg)
    pallet_yaw = math.radians(pallet_yaw_deg)

    u_robot = np.array([math.cos(yaw), math.sin(yaw)], dtype=np.float64)
    u_in = np.array([math.cos(pallet_yaw), math.sin(pallet_yaw)], dtype=np.float64)

    fork_center_xy = root_xy + (fork_reach_m - FORK_CENTER_BACKOFF_M) * u_robot
    p0 = fork_center_xy

    s_goal = exp83_traj_goal_s(
        pallet_depth_m=pallet_depth_m,
        insert_fraction=insert_fraction,
        mode=traj_goal_mode,
    )
    p_goal = pallet_xy + s_goal * u_in
    p_pre = pallet_xy + (s_goal - traj_pre_dist_m) * u_in

    dist = np.linalg.norm(p_pre - p0)
    L = dist * 1.5
    m0 = u_robot * L
    m1 = u_in * L

    num_curve = int(traj_num_samples * 0.7)
    num_line = traj_num_samples - num_curve

    t = np.linspace(0.0, 1.0, num_curve, dtype=np.float64).reshape(-1, 1)
    t2 = t**2
    t3 = t**3
    h00 = 2 * t3 - 3 * t2 + 1
    h10 = t3 - 2 * t2 + t
    h01 = -2 * t3 + 3 * t2
    h11 = t3 - t2
    pts_curve = h00 * p0 + h10 * m0 + h01 * p_pre + h11 * m1

    t_line = np.linspace(0.0, 1.0, num_line, dtype=np.float64).reshape(-1, 1)
    pts_line = (1 - t_line) * p_pre + t_line * p_goal

    pts = np.concatenate([pts_curve, pts_line], axis=0)
    tangents = _compute_path_tangents(pts, u_robot)
    root_to_fc = fork_reach_m - FORK_CENTER_BACKOFF_M
    root_path = pts - root_to_fc * tangents
    return fork_center_xy, p_pre, p_goal, pts, tangents, root_path


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

    ax.plot(pts[:, 0], pts[:, 1], color="#1f77b4", lw=2.0, label="fork-center reference trajectory")
    ax.plot(root_path[:, 0], root_path[:, 1], color="#444444", lw=1.6, ls="--", alpha=0.9, label="implied root trajectory")
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
        "fork_center entry geometry only",
        "NO wheelbase / NO min-turn-radius",
        "NO chassis sweep / NO drive-wheel check",
        "",
        "Expectation",
        "s_start < s_pre < s_goal",
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
):
    pallet_yaw = math.radians(pallet_yaw_deg)
    u_in = np.array([math.cos(pallet_yaw), math.sin(pallet_yaw)], dtype=np.float64)

    fig, ax = plt.subplots(figsize=(9, 9))

    s_vals = [case.s_pre for case in cases] + [case.s_goal for case in cases]
    axis_s = np.linspace(min(s_vals) - 0.8, max(s_vals) + 0.8, 200)
    axis_pts = pallet_xy + axis_s[:, None] * u_in
    ax.plot(axis_pts[:, 0], axis_pts[:, 1], "--", color="#999999", lw=1.2, label="pallet s-axis")

    any_bad = False
    for case, payload in zip(cases, overlay_payloads, strict=True):
        root_xy, fork_center_xy, p_pre, p_goal, pts, _, root_path = payload
        ax.plot(pts[:, 0], pts[:, 1], color="#1f77b4", alpha=0.22, lw=1.5)
        ax.plot(root_path[:, 0], root_path[:, 1], color="#444444", alpha=0.10, lw=1.2)
        marker_color = "#2ca02c" if case.entry_ok else "#d62728"
        if not case.entry_ok:
            any_bad = True
        ax.scatter(fork_center_xy[0], fork_center_xy[1], color=marker_color, s=26)
        ax.text(fork_center_xy[0], fork_center_xy[1], case.case_id, fontsize=7, color=marker_color)
        ax.scatter(p_pre[0], p_pre[1], color="#ff7f0e", s=12, alpha=0.35)
        ax.scatter(p_goal[0], p_goal[1], color="#2ca02c", s=12, alpha=0.35)

    n_bad = sum(1 for case in cases if not case.entry_ok)
    status = f"{n_bad}/{len(cases)} cases with s_start >= s_pre"
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


def build_case_grid(cfg: dict[str, object], pallet_xy: np.ndarray, pallet_yaw_deg: float) -> tuple[list[CaseMetrics], list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]]:
    x_vals = [
        float(cfg["stage1_init_x_min_m"]),
        0.5 * (float(cfg["stage1_init_x_min_m"]) + float(cfg["stage1_init_x_max_m"])),
        float(cfg["stage1_init_x_max_m"]),
    ]
    y_vals = [
        float(cfg["stage1_init_y_min_m"]),
        0.0,
        float(cfg["stage1_init_y_max_m"]),
    ]
    yaw_vals = [
        float(cfg["stage1_init_yaw_deg_min"]),
        0.0,
        float(cfg["stage1_init_yaw_deg_max"]),
    ]

    cases: list[CaseMetrics] = []
    payloads: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []

    idx = 1
    for root_x in x_vals:
        for root_y in y_vals:
            for yaw_deg in yaw_vals:
                root_xy = np.array([root_x, root_y], dtype=np.float64)
                fork_center_xy, p_pre, p_goal, pts, tangents, root_path = build_reference_trajectory(
                    root_xy=root_xy,
                    yaw_deg=yaw_deg,
                    pallet_xy=pallet_xy,
                    pallet_yaw_deg=pallet_yaw_deg,
                    fork_reach_m=float(cfg["fork_reach_m"]),
                    pallet_depth_m=float(cfg["pallet_depth_m"]),
                    traj_pre_dist_m=float(cfg["traj_pre_dist_m"]),
                    traj_num_samples=int(cfg["traj_num_samples"]),
                    insert_fraction=float(cfg["insert_fraction"]),
                    traj_goal_mode=str(cfg["exp83_traj_goal_mode"]),
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
        default=PROJECT_ROOT / "outputs" / "reference_trajectory_stage1_viz",
        help="Directory to write PNGs and manifest JSON.",
    )
    parser.add_argument("--pallet-x", type=float, default=0.0)
    parser.add_argument("--pallet-y", type=float, default=0.0)
    parser.add_argument("--pallet-yaw-deg", type=float, default=0.0)
    parser.add_argument(
        "--cfg-path",
        type=Path,
        default=CFG_PATH,
        help="Env cfg path to read. Defaults to current IsaacLab env_cfg.py, falls back to legacy project patch if missing.",
    )
    args = parser.parse_args()

    cfg = load_cfg_defaults(args.cfg_path)
    pallet_xy = np.array([args.pallet_x, args.pallet_y], dtype=np.float64)
    pallet_yaw_deg = float(args.pallet_yaw_deg)

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    cases, payloads = build_case_grid(cfg, pallet_xy, pallet_yaw_deg)
    for case, payload in zip(cases, payloads, strict=True):
        root_xy, fork_center_xy, p_pre, p_goal, pts, _, root_path = payload
        draw_case(
            out_path=out_dir / f"{case.case_id}.png",
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
        )

    draw_overlay(
        out_path=out_dir / "overlay_all_cases.png",
        cases=cases,
        overlay_payloads=payloads,
        pallet_xy=pallet_xy,
        pallet_yaw_deg=pallet_yaw_deg,
    )

    summary = {
        "cfg_path": str(args.cfg_path),
        "validation_scope": [
            "fork_center entry geometry only",
            "does_not_check_wheelbase",
            "does_not_check_min_turn_radius",
            "does_not_check_chassis_sweep_or_drive_wheel_path",
        ],
        "cfg_path_used": str(args.cfg_path),
        "parsed_cfg": cfg,
        "pallet_xy": pallet_xy.tolist(),
        "pallet_yaw_deg": pallet_yaw_deg,
        "num_cases": len(cases),
        "num_entry_ok": sum(1 for case in cases if case.entry_ok),
        "num_entry_bad": sum(1 for case in cases if not case.entry_ok),
        "delta_s_min": min(case.delta_s for case in cases),
        "delta_s_max": max(case.delta_s for case in cases),
        "delta_s_mean": sum(case.delta_s for case in cases) / len(cases),
        "root_y_abs_max_max": max(case.root_y_abs_max for case in cases),
        "root_heading_change_deg_max": max(case.root_heading_change_deg for case in cases),
        "root_curvature_max_max": max(case.root_curvature_max for case in cases),
        "cases": [asdict(case) for case in cases],
    }
    manifest_path = out_dir / "reference_trajectory_stage1_manifest.json"
    manifest_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[traj_viz] wrote {len(cases)} case PNGs to: {out_dir}")
    print(f"[traj_viz] overlay: {out_dir / 'overlay_all_cases.png'}")
    print(f"[traj_viz] manifest: {manifest_path}")
    print(
        "[traj_viz] delta_s stats: "
        f"min={summary['delta_s_min']:+.4f}, "
        f"max={summary['delta_s_max']:+.4f}, "
        f"mean={summary['delta_s_mean']:+.4f}"
    )


if __name__ == "__main__":
    main()
