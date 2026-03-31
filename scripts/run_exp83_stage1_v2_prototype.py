#!/usr/bin/env python3
"""Build an offline Stage1-v2 geometry prototype around a 2.72 m upstream anchor.

This prototype does not modify the runtime env. It uses the current Stage1 reset
grid, but swaps in a direct upstream alignment anchor:

    root_align_s = root_goal_s - align_start_dist_m

Then it visualizes the bounded-curvature `dubins_to_pre_straight` family and
summarizes whether the paths still show near-field full-loop behavior.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime
import importlib.util
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
SCAN_SCRIPT = REPO_ROOT / "scripts" / "run_exp83_arc_pre_goal_push_scan.py"
OUT_ROOT = REPO_ROOT / "outputs" / "exp83_stage1_v2_prototype"


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


def draw_stage1_v2_overlay(
    *,
    out_path: Path,
    cases: list[object],
    payloads: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    pallet_xy: np.ndarray,
    pallet_yaw_deg: float,
    align_start_s: float,
    goal_s: float,
    curvature_limit: float,
    align_start_dist_m: float,
) -> None:
    pallet_yaw = math.radians(pallet_yaw_deg)
    u_in = np.array([math.cos(pallet_yaw), math.sin(pallet_yaw)], dtype=np.float64)
    align_pt = pallet_xy + align_start_s * u_in
    goal_pt = pallet_xy + goal_s * u_in

    fig, ax = plt.subplots(figsize=(9.6, 8.6))
    axis_s = np.linspace(min(align_start_s, goal_s) - 0.8, max(align_start_s, goal_s) + 0.8, 200)
    axis_pts = pallet_xy + axis_s[:, None] * u_in
    ax.plot(axis_pts[:, 0], axis_pts[:, 1], "--", color="#8e8e8e", lw=1.1, label="pallet s-axis")

    bbox_points = [align_pt.reshape(1, 2), goal_pt.reshape(1, 2), pallet_xy.reshape(1, 2), axis_pts[[0, -1]]]
    for case, payload in zip(cases, payloads, strict=True):
        root_xy, fork_center_xy, p_pre, p_goal, pts, _, root_path = payload
        ax.plot(root_path[:, 0], root_path[:, 1], color="#404040", alpha=0.18, lw=1.4)
        ax.plot(pts[:, 0], pts[:, 1], color="#1f77b4", alpha=0.22, lw=1.4)
        ax.scatter(root_xy[0], root_xy[1], color="#404040", s=14, alpha=0.55)
        ax.scatter(fork_center_xy[0], fork_center_xy[1], color="#d62728", s=16, alpha=0.55)
        bbox_points.extend(
            [
                root_xy.reshape(1, 2),
                fork_center_xy.reshape(1, 2),
                p_pre.reshape(1, 2),
                p_goal.reshape(1, 2),
                pts,
                root_path,
            ]
        )

    ax.scatter(align_pt[0], align_pt[1], color="#ff7f0e", s=72, marker="s", label="Stage1-v2 align start")
    ax.scatter(goal_pt[0], goal_pt[1], color="#2ca02c", s=72, marker="o", label="final insert goal")

    num_heading_gt_180 = sum(case.root_heading_change_deg > 180.0 for case in cases)
    num_length_gt_10m = sum(case.root_total_length_m > 10.0 for case in cases)
    title = (
        f"Stage1-v2 prototype | direct upstream align-start = {align_start_dist_m:.2f} m | "
        f">180deg: {num_heading_gt_180}/{len(cases)} | >10m: {num_length_gt_10m}/{len(cases)} | "
        f"kappa_limit={curvature_limit:.3f} 1/m"
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


def draw_stage1_v2_case(
    *,
    out_path: Path,
    case: object,
    payload: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    pallet_xy: np.ndarray,
    pallet_yaw_deg: float,
    cfg_path: Path,
    align_start_dist_m: float,
    root_goal_s: float,
) -> None:
    root_xy, fork_center_xy, p_pre, p_goal, pts, _, root_path = payload
    pallet_yaw = math.radians(pallet_yaw_deg)
    u_in = np.array([math.cos(pallet_yaw), math.sin(pallet_yaw)], dtype=np.float64)
    v_lat = np.array([-math.sin(pallet_yaw), math.cos(pallet_yaw)], dtype=np.float64)

    fig = plt.figure(figsize=(12.6, 7.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[4.2, 1.8], wspace=0.12)
    ax = fig.add_subplot(gs[0, 0])
    ax_info = fig.add_subplot(gs[0, 1])
    ax_info.axis("off")

    axis_s = np.linspace(case.s_goal - align_start_dist_m - 0.8, case.s_goal + 0.8, 200)
    axis_pts = pallet_xy + axis_s[:, None] * u_in
    lat_line = np.stack([pallet_xy - 0.25 * v_lat, pallet_xy + 0.25 * v_lat], axis=0)
    ax.plot(axis_pts[:, 0], axis_pts[:, 1], "--", color="#888888", lw=1.2, label="pallet s-axis")
    ax.plot(lat_line[:, 0], lat_line[:, 1], "-", color="#bbbbbb", lw=1.0)

    ax.plot(root_path[:, 0], root_path[:, 1], color="#444444", lw=2.0, ls="--", alpha=0.95, label="vehicle/root reference")
    ax.plot(pts[:, 0], pts[:, 1], color="#1f77b4", lw=1.8, label="fork-center mapping")
    ax.scatter(root_xy[0], root_xy[1], color="#444444", s=36, label="root start")
    ax.scatter(fork_center_xy[0], fork_center_xy[1], color="#d62728", s=42, label="fork_center start")
    ax.scatter(p_pre[0], p_pre[1], color="#ff7f0e", s=42, label="align start")
    ax.scatter(p_goal[0], p_goal[1], color="#2ca02c", s=42, label="insert goal")
    ax.scatter(pallet_xy[0], pallet_xy[1], color="#9467bd", s=40, label="pallet center")

    title = (
        f"{case.case_id} | len={case.root_total_length_m:.3f} m | "
        f"dpsi={case.root_heading_change_deg:.1f} deg | "
        f"kappa_max={case.root_curvature_max:.3f} 1/m"
    )
    fig.suptitle(title, fontsize=13)

    info_lines = [
        "Stage1-v2 Prototype",
        f"root = ({case.root_x:+.3f}, {case.root_y:+.3f})",
        f"yaw = {case.yaw_deg:+.1f} deg",
        f"align_start_dist = {align_start_dist_m:.2f} m",
        f"root_goal_s = {root_goal_s:+.3f}",
        f"s_start = {case.s_start:+.3f}",
        f"s_align = {case.s_pre:+.3f}",
        f"s_goal = {case.s_goal:+.3f}",
        f"path_mode = {case.path_mode}",
        f"root_len = {case.root_total_length_m:.3f} m",
        f"root_dpsi = {case.root_heading_change_deg:.2f} deg",
        f"root_kappa_max = {case.root_curvature_max:.3f} 1/m",
        f"root|y|_max = {case.root_y_abs_max:.3f} m",
        f"legacy entry_ok = {case.entry_ok}",
        "",
        "Meaning",
        "Direct upstream align-start prototype.",
        "legacy entry_ok=false is expected here,",
        "because align-start is upstream of current Stage1 band.",
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
        fontsize=9.2,
        family="monospace",
        linespacing=1.22,
        bbox={"boxstyle": "round", "facecolor": "#f7f7f7", "edgecolor": "#d9d9d9", "alpha": 0.98},
    )

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


def draw_axis_geometry(
    *,
    out_path: Path,
    current_x_min: float,
    current_x_max: float,
    current_pre_s: float,
    proposed_align_s: float,
    goal_s: float,
    align_start_dist_m: float,
) -> None:
    fig, ax = plt.subplots(figsize=(11.2, 2.8))

    ax.hlines(0.0, proposed_align_s - 0.6, goal_s + 0.6, color="#7f7f7f", lw=1.2)
    ax.fill_between([current_x_min, current_x_max], -0.16, 0.16, color="#d62728", alpha=0.22, label="current Stage1 start band")
    ax.axvline(current_pre_s, color="#ff7f0e", lw=2.0, label="current env nominal root_pre")
    ax.axvline(proposed_align_s, color="#1f77b4", lw=2.2, label="proposed Stage1-v2 align start")
    ax.axvline(goal_s, color="#2ca02c", lw=2.2, label="final root goal")

    ax.scatter([current_pre_s, proposed_align_s, goal_s], [0.0, 0.0, 0.0], s=44, color=["#ff7f0e", "#1f77b4", "#2ca02c"])
    ax.annotate("current Stage1 band", xy=((current_x_min + current_x_max) * 0.5, 0.15), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=9)
    ax.annotate(f"root_pre={current_pre_s:.2f}", xy=(current_pre_s, 0.0), xytext=(0, 12), textcoords="offset points", ha="center", fontsize=9, color="#ff7f0e")
    ax.annotate(f"align start={proposed_align_s:.2f}", xy=(proposed_align_s, 0.0), xytext=(0, -18), textcoords="offset points", ha="center", fontsize=9, color="#1f77b4")
    ax.annotate(f"root_goal={goal_s:.2f}", xy=(goal_s, 0.0), xytext=(0, 12), textcoords="offset points", ha="center", fontsize=9, color="#2ca02c")

    ax.set_title(
        f"Stage1-v2 geometry sketch | direct upstream align-start = {align_start_dist_m:.2f} m before root_goal",
        fontsize=12,
    )
    ax.set_xlabel("pallet-axis s (m)")
    ax.set_yticks([])
    ax.grid(True, axis="x", alpha=0.25)
    ax.legend(loc="upper left", ncol=4, frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def select_representative_case_ids(cases: list[object]) -> list[str]:
    if not cases:
        return []
    x_mid = statistics.mean(case.root_x for case in cases)
    center_case = min(cases, key=lambda case: (abs(case.root_x - x_mid) + abs(case.root_y) + abs(case.yaw_deg)))
    worst_length = max(cases, key=lambda case: case.root_total_length_m)
    worst_heading = max(cases, key=lambda case: case.root_heading_change_deg)
    positive_corner = max(cases, key=lambda case: (case.root_y, case.yaw_deg, -case.root_x))
    negative_corner = min(cases, key=lambda case: (case.root_y, case.yaw_deg, case.root_x))

    ordered = [center_case.case_id, worst_length.case_id, worst_heading.case_id, positive_corner.case_id, negative_corner.case_id]
    unique_ids: list[str] = []
    for case_id in ordered:
        if case_id not in unique_ids:
            unique_ids.append(case_id)
    return unique_ids


def write_markdown(path: Path, summary: dict) -> None:
    lines = [
        "# Exp8.3 Stage1-v2 Prototype",
        "",
        f"- cfg_path: `{summary['cfg_path']}`",
        f"- align_start_dist_m: `{summary['align_start_dist_m']:.2f}`",
        f"- scan_mode: `{summary['scan_mode']}`",
        f"- overlay: `{summary['overlay_path']}`",
        f"- geometry_sketch: `{summary['geometry_sketch_path']}`",
        "",
        "## Geometry",
        "",
        f"- root_goal_s: `{summary['root_goal_s']:.3f}`",
        f"- current_env_nominal_root_pre_s: `{summary['current_env_nominal_root_pre_s']:.3f}`",
        f"- proposed_align_start_s: `{summary['proposed_align_start_s']:.3f}`",
        f"- current_stage1_x_range: `[{summary['current_stage1_x_min_m']:.3f}, {summary['current_stage1_x_max_m']:.3f}]`",
        f"- current_stage1_y_range: `[{summary['current_stage1_y_min_m']:.3f}, {summary['current_stage1_y_max_m']:.3f}]`",
        f"- current_stage1_yaw_deg_range: `[{summary['current_stage1_yaw_deg_min']:.1f}, {summary['current_stage1_yaw_deg_max']:.1f}]`",
        "",
        "## Path Audit",
        "",
        f"- num_cases: `{summary['num_cases']}`",
        f"- legacy_entry_ok: `{summary['num_entry_ok']}`",
        f"- root_total_length_mean: `{summary['root_total_length_mean']:.3f} m`",
        f"- root_total_length_max: `{summary['root_total_length_max']:.3f} m`",
        f"- root_heading_change_mean: `{summary['root_heading_change_mean']:.3f} deg`",
        f"- root_heading_change_max: `{summary['root_heading_change_max']:.3f} deg`",
        f"- root_curvature_max_max: `{summary['root_curvature_max_max']:.6f} 1/m`",
        f"- curvature_limit: `{summary['curvature_limit']:.6f} 1/m`",
        f"- num_heading_gt_180: `{summary['num_heading_gt_180']}`",
        f"- num_heading_gt_270: `{summary['num_heading_gt_270']}`",
        f"- num_length_gt_10m: `{summary['num_length_gt_10m']}`",
        "",
        "## Interpretation",
        "",
        "- This is a geometry prototype, not a drop-in env config.",
        "- Legacy `entry_ok` stays false because the direct align-start sits upstream of the current near-field start band.",
        "- The useful signal here is whether bounded-curvature paths stop doing full loops while staying within the physical curvature limit.",
        "",
        "## Representative Cases",
        "",
    ]
    for item in summary["representative_cases"]:
        lines.append(f"- `{item['case_id']}`: `{item['image_path']}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the Stage1-v2 offline geometry prototype.")
    parser.add_argument("--cfg-path", type=Path, default=CFG_PATH)
    parser.add_argument("--align-start-dist-m", type=float, default=2.72)
    parser.add_argument("--grid-count-x", type=int, default=5)
    parser.add_argument("--grid-count-y", type=int, default=5)
    parser.add_argument("--grid-count-yaw", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, default=OUT_ROOT)
    parser.add_argument("--timestamp-tag", type=str, default=None)
    args = parser.parse_args()

    viz = load_module("exp83_viz_cases_stage1_v2", VIZ_SCRIPT)
    scan = load_module("exp83_scan_stage1_v2", SCAN_SCRIPT)
    cfg = viz.load_cfg_defaults(args.cfg_path)

    pallet_xy = np.array([0.0, 0.0], dtype=np.float64)
    pallet_yaw_deg = 0.0
    run_timestamp = args.timestamp_tag or datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    cases, payloads = scan.build_direct_pre_goal_grid(
        mod=viz,
        cfg=cfg,
        pallet_xy=pallet_xy,
        pallet_yaw_deg=pallet_yaw_deg,
        pre_dist_m=float(args.align_start_dist_m),
        grid_count_x=int(args.grid_count_x),
        grid_count_y=int(args.grid_count_y),
        grid_count_yaw=int(args.grid_count_yaw),
    )

    root_to_fc = float(cfg["fork_reach_m"]) - viz.FORK_CENTER_BACKOFF_M
    s_goal = viz.exp83_traj_goal_s(
        pallet_depth_m=float(cfg["pallet_depth_m"]),
        insert_fraction=float(cfg["insert_fraction"]),
        mode=str(cfg["exp83_traj_goal_mode"]),
    )
    p_goal = pallet_xy + s_goal * np.array([1.0, 0.0], dtype=np.float64)
    root_goal = p_goal - root_to_fc * np.array([1.0, 0.0], dtype=np.float64)
    root_goal_s, _ = viz.project_axis(root_goal, pallet_xy, pallet_yaw_deg)
    current_env_nominal_root_pre_s = root_goal_s - float(cfg["traj_pre_dist_m"])
    proposed_align_start_s = root_goal_s - float(args.align_start_dist_m)
    curvature_limit = 1.0 / float(cfg["traj_rs_min_turn_radius_m"])

    overlay_path = out_dir / stamped_name("overlay_stage1_v2_prototype", run_timestamp, ".png")
    geometry_sketch_path = out_dir / stamped_name("geometry_stage1_v2_prototype", run_timestamp, ".png")
    manifest_path = out_dir / stamped_name("stage1_v2_prototype_manifest", run_timestamp, ".json")
    summary_json_path = out_dir / stamped_name("stage1_v2_prototype_summary", run_timestamp, ".json")
    summary_md_path = out_dir / stamped_name("stage1_v2_prototype_summary", run_timestamp, ".md")

    draw_stage1_v2_overlay(
        out_path=overlay_path,
        cases=cases,
        payloads=payloads,
        pallet_xy=pallet_xy,
        pallet_yaw_deg=pallet_yaw_deg,
        align_start_s=proposed_align_start_s,
        goal_s=root_goal_s,
        curvature_limit=curvature_limit,
        align_start_dist_m=float(args.align_start_dist_m),
    )
    draw_axis_geometry(
        out_path=geometry_sketch_path,
        current_x_min=float(cfg["stage1_init_x_min_m"]),
        current_x_max=float(cfg["stage1_init_x_max_m"]),
        current_pre_s=current_env_nominal_root_pre_s,
        proposed_align_s=proposed_align_start_s,
        goal_s=root_goal_s,
        align_start_dist_m=float(args.align_start_dist_m),
    )

    payload_by_id = {case.case_id: payload for case, payload in zip(cases, payloads, strict=True)}
    representative_cases = []
    for case_id in select_representative_case_ids(cases):
        case = next(case for case in cases if case.case_id == case_id)
        image_path = out_dir / stamped_name(case.case_id, run_timestamp, ".png")
        draw_stage1_v2_case(
            out_path=image_path,
            case=case,
            payload=payload_by_id[case_id],
            pallet_xy=pallet_xy,
            pallet_yaw_deg=pallet_yaw_deg,
            cfg_path=args.cfg_path,
            align_start_dist_m=float(args.align_start_dist_m),
            root_goal_s=root_goal_s,
        )
        representative_cases.append({"case_id": case_id, "image_path": str(image_path)})

    heading_vals = [case.root_heading_change_deg for case in cases]
    length_vals = [case.root_total_length_m for case in cases]
    curvature_vals = [case.root_curvature_max for case in cases]
    path_mode_counts: dict[str, int] = {}
    for case in cases:
        path_mode_counts[case.path_mode] = path_mode_counts.get(case.path_mode, 0) + 1

    summary = {
        "tool": "run_exp83_stage1_v2_prototype.py",
        "cfg_path": str(args.cfg_path),
        "scan_mode": "direct_pre_goal_geometry_prototype",
        "align_start_dist_m": float(args.align_start_dist_m),
        "overlay_path": str(overlay_path),
        "geometry_sketch_path": str(geometry_sketch_path),
        "current_stage1_x_min_m": float(cfg["stage1_init_x_min_m"]),
        "current_stage1_x_max_m": float(cfg["stage1_init_x_max_m"]),
        "current_stage1_y_min_m": float(cfg["stage1_init_y_min_m"]),
        "current_stage1_y_max_m": float(cfg["stage1_init_y_max_m"]),
        "current_stage1_yaw_deg_min": float(cfg["stage1_init_yaw_deg_min"]),
        "current_stage1_yaw_deg_max": float(cfg["stage1_init_yaw_deg_max"]),
        "root_goal_s": float(root_goal_s),
        "current_env_nominal_root_pre_s": float(current_env_nominal_root_pre_s),
        "proposed_align_start_s": float(proposed_align_start_s),
        "curvature_limit": float(curvature_limit),
        "num_cases": len(cases),
        "num_entry_ok": sum(1 for case in cases if case.entry_ok),
        "root_total_length_mean": statistics.mean(length_vals),
        "root_total_length_max": max(length_vals),
        "root_total_length_min": min(length_vals),
        "root_heading_change_mean": statistics.mean(heading_vals),
        "root_heading_change_max": max(heading_vals),
        "root_curvature_max_max": max(curvature_vals),
        "num_heading_gt_180": sum(value > 180.0 for value in heading_vals),
        "num_heading_gt_270": sum(value > 270.0 for value in heading_vals),
        "num_length_gt_10m": sum(value > 10.0 for value in length_vals),
        "path_mode_counts": path_mode_counts,
        "representative_cases": representative_cases,
    }

    manifest = {
        "summary": summary,
        "cases": [asdict(case) for case in cases],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    summary_json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(summary_md_path, summary)

    print(f"[stage1_v2_prototype] overlay: {overlay_path}")
    print(f"[stage1_v2_prototype] geometry: {geometry_sketch_path}")
    print(f"[stage1_v2_prototype] manifest: {manifest_path}")
    print(f"[stage1_v2_prototype] summary_json: {summary_json_path}")
    print(f"[stage1_v2_prototype] summary_md: {summary_md_path}")


if __name__ == "__main__":
    main()
