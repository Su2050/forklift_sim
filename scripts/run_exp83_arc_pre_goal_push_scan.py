#!/usr/bin/env python3
"""Scan how far the pre-goal must move upstream for bounded-curvature paths.

This uses a direct pre-goal sweep rather than the current env-side `traj_pre_dist_m`
logic, because Stage1's existing pre-goal construction clamps `root_pre` near the
start pose. The sweep here explicitly sets:

    root_pre_s = root_goal_s - pre_dist_m

Then it audits whether bounded-curvature `dubins_to_pre_straight` paths stop doing
large near-field loops.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path
import statistics
import sys
from types import ModuleType

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = REPO_ROOT / "forklift_pallet_insert_lift_project"
VIZ_SCRIPT = PROJECT_ROOT / "scripts" / "visualize_reference_trajectory_cases.py"
CFG_PATH = PROJECT_ROOT / "isaaclab_patch/source/isaaclab_tasks/isaaclab_tasks/direct/forklift_pallet_insert_lift/env_cfg.py"
OUT_ROOT = REPO_ROOT / "outputs" / "exp83_arc_pre_goal_push_scan"


def load_viz_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("exp83_viz_cases", VIZ_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load module from {VIZ_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_markdown(path: Path, summary: dict) -> None:
    lines = [
        "# Exp8.3 Arc Pre-Goal Push Scan",
        "",
        f"- cfg_path: `{summary['cfg_path']}`",
        f"- traj_model: `{summary['traj_model']}`",
        f"- scan_mode: `{summary['scan_mode']}`",
        f"- note: `{summary['note']}`",
        "",
        "## Sweep Rows",
        "",
    ]
    for row in summary["rows"]:
        lines.extend(
            [
                f"### pre_dist = {row['traj_pre_dist_m']:.2f} m",
                "",
                f"- overlay: `{row['overlay_path']}`",
                f"- num_cases: `{row['num_cases']}`",
                f"- entry_ok: `{row['num_entry_ok']}`",
                f"- root_total_length_mean: `{row['root_total_length_mean']:.3f} m`",
                f"- root_total_length_max: `{row['root_total_length_max']:.3f} m`",
                f"- root_heading_change_mean: `{row['root_heading_change_mean']:.3f} deg`",
                f"- root_heading_change_max: `{row['root_heading_change_max']:.3f} deg`",
                f"- num_heading_gt_180: `{row['num_heading_gt_180']}`",
                f"- num_heading_gt_270: `{row['num_heading_gt_270']}`",
                f"- num_length_gt_10m: `{row['num_length_gt_10m']}`",
                f"- worst_length_case: `{row['worst_length_case_id']}`",
                "",
            ]
        )
    lines.extend(
        [
            "## First Good-ish Rows",
            "",
            f"- first row with `num_heading_gt_270 == 0`: `{summary['first_no_gt_270']}`",
            f"- first row with `num_heading_gt_180 == 0`: `{summary['first_no_gt_180']}`",
            f"- first row with `num_length_gt_10m == 0`: `{summary['first_no_len_gt_10']}`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_direct_pre_goal_grid(
    mod: ModuleType,
    *,
    cfg: dict[str, object],
    pallet_xy: np.ndarray,
    pallet_yaw_deg: float,
    pre_dist_m: float,
    grid_count_x: int,
    grid_count_y: int,
    grid_count_yaw: int,
) -> tuple[list[object], list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]]:
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
    root_to_fc = float(cfg["fork_reach_m"]) - float(mod.FORK_CENTER_BACKOFF_M)
    s_goal = mod.exp83_traj_goal_s(
        pallet_depth_m=float(cfg["pallet_depth_m"]),
        insert_fraction=float(cfg["insert_fraction"]),
        mode=str(cfg["exp83_traj_goal_mode"]),
    )
    p_goal = pallet_xy + s_goal * u_in
    root_goal = p_goal - root_to_fc * u_in
    root_goal_s, _ = mod.project_axis(root_goal, pallet_xy, pallet_yaw_deg)
    root_pre_s = root_goal_s - float(pre_dist_m)
    root_pre = pallet_xy + root_pre_s * u_in
    p_pre = root_pre + root_to_fc * u_in

    cases: list[object] = []
    payloads: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    idx = 1
    for root_x in x_vals:
        for root_y in y_vals:
            for yaw_deg in yaw_vals:
                yaw = math.radians(yaw_deg)
                u_robot = np.array([math.cos(yaw), math.sin(yaw)], dtype=np.float64)
                root_xy = np.array([root_x, root_y], dtype=np.float64)
                fork_center_xy = root_xy + root_to_fc * u_robot

                dubins_dense = mod.sample_forward_only_rs_dense_path_np(
                    root_start_xy=root_xy,
                    root_start_yaw=yaw,
                    root_goal_xy=root_pre,
                    root_goal_yaw=pallet_yaw,
                    min_turn_radius_m=float(cfg["traj_rs_min_turn_radius_m"]),
                    sample_step_m=float(cfg["traj_rs_sample_step_m"]),
                )
                if dubins_dense is None:
                    raise RuntimeError("forward-only RS/Dubins candidate set is empty for direct pre-goal scan")
                dense_xy, dense_yaw, family = dubins_dense
                dense_xy, dense_yaw = mod.append_straight_segment_np(
                    xy=dense_xy,
                    yaw=dense_yaw,
                    goal_xy=root_goal,
                    goal_yaw=pallet_yaw,
                    sample_step_m=float(cfg["traj_rs_sample_step_m"]),
                )
                root_path = dense_xy
                tangents = np.stack([np.cos(dense_yaw), np.sin(dense_yaw)], axis=1)
                pts = root_path + root_to_fc * tangents

                s_start, y_start = mod.project_axis(fork_center_xy, pallet_xy, pallet_yaw_deg)
                s_pre, _ = mod.project_axis(p_pre, pallet_xy, pallet_yaw_deg)
                s_goal_case, _ = mod.project_axis(p_goal, pallet_xy, pallet_yaw_deg)
                root_y_abs_max = max(
                    abs(mod.project_axis(point_xy, pallet_xy, pallet_yaw_deg)[1]) for point_xy in root_path
                )
                root_heading_change_deg, root_curvature_max = mod.compute_path_heading_curvature(
                    root_path,
                    fallback_dir=u_robot,
                    tangents=tangents,
                )
                case = mod.CaseMetrics(
                    case_id=mod.format_case_id(idx, root_x, root_y, yaw_deg),
                    root_x=root_x,
                    root_y=root_y,
                    yaw_deg=yaw_deg,
                    s_start=s_start,
                    s_pre=s_pre,
                    s_goal=s_goal_case,
                    delta_s=s_start - s_pre,
                    y_start=y_start,
                    root_total_length_m=mod.compute_path_length_np(root_path),
                    root_y_abs_max=root_y_abs_max,
                    root_heading_change_deg=root_heading_change_deg,
                    root_curvature_max=root_curvature_max,
                    entry_ok=(s_start < s_pre < s_goal_case),
                    path_mode=f"direct_pre_goal_scan_{family}",
                )
                cases.append(case)
                payloads.append((root_xy, fork_center_xy, p_pre, p_goal, pts, tangents, root_path))
                idx += 1
    return cases, payloads


def main() -> None:
    parser = argparse.ArgumentParser(description="Scan pre-goal push distance for bounded-curvature paths.")
    parser.add_argument("--cfg-path", type=Path, default=CFG_PATH)
    parser.add_argument("--traj-model", type=str, default="dubins_to_pre_straight")
    parser.add_argument(
        "--pre-dist-values",
        type=float,
        nargs="+",
        default=[1.05, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0],
        help="Explicit pre-goal offsets where root_pre_s = root_goal_s - pre_dist_m.",
    )
    parser.add_argument("--grid-count-x", type=int, default=5)
    parser.add_argument("--grid-count-y", type=int, default=5)
    parser.add_argument("--grid-count-yaw", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, default=OUT_ROOT)
    args = parser.parse_args()

    mod = load_viz_module()
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg_base = mod.load_cfg_defaults(args.cfg_path)
    pallet_xy = np.array([0.0, 0.0], dtype=np.float64)
    pallet_yaw_deg = 0.0
    rows: list[dict] = []

    for pre_dist in args.pre_dist_values:
        cfg = dict(cfg_base)
        cfg["traj_model"] = str(args.traj_model)
        cases, payloads = build_direct_pre_goal_grid(
            mod,
            cfg=cfg,
            pallet_xy=pallet_xy,
            pallet_yaw_deg=pallet_yaw_deg,
            pre_dist_m=float(pre_dist),
            grid_count_x=int(args.grid_count_x),
            grid_count_y=int(args.grid_count_y),
            grid_count_yaw=int(args.grid_count_yaw),
        )
        tag = f"pre_{pre_dist:.2f}".replace(".", "p")
        overlay_path = out_dir / f"overlay_{tag}.png"
        mod.draw_overlay(
            out_path=overlay_path,
            cases=cases,
            overlay_payloads=payloads,
            pallet_xy=pallet_xy,
            pallet_yaw_deg=pallet_yaw_deg,
            traj_model=f"{args.traj_model}_pre{pre_dist:.2f}",
        )

        heading_vals = [case.root_heading_change_deg for case in cases]
        length_vals = [case.root_total_length_m for case in cases]
        worst_length_case = max(cases, key=lambda case: case.root_total_length_m)
        row = {
            "traj_pre_dist_m": float(pre_dist),
            "overlay_path": str(overlay_path),
            "num_cases": len(cases),
            "num_entry_ok": sum(1 for case in cases if case.entry_ok),
            "root_total_length_mean": statistics.mean(length_vals),
            "root_total_length_max": max(length_vals),
            "root_heading_change_mean": statistics.mean(heading_vals),
            "root_heading_change_max": max(heading_vals),
            "num_heading_gt_180": sum(value > 180.0 for value in heading_vals),
            "num_heading_gt_270": sum(value > 270.0 for value in heading_vals),
            "num_length_gt_10m": sum(value > 10.0 for value in length_vals),
            "worst_length_case_id": worst_length_case.case_id,
        }
        rows.append(row)

    def first_match(key: str, target: int) -> float | None:
        for row in rows:
            if int(row[key]) == int(target):
                return float(row["traj_pre_dist_m"])
        return None

    summary = {
        "tool": "run_exp83_arc_pre_goal_push_scan.py",
        "cfg_path": str(args.cfg_path),
        "traj_model": str(args.traj_model),
        "scan_mode": "direct_pre_goal",
        "note": "This sweep bypasses the current env pre-goal clamp and directly places root_pre upstream of root_goal.",
        "rows": rows,
        "first_no_gt_270": first_match("num_heading_gt_270", 0),
        "first_no_gt_180": first_match("num_heading_gt_180", 0),
        "first_no_len_gt_10": first_match("num_length_gt_10m", 0),
    }

    json_path = out_dir / "pre_goal_push_scan_summary.json"
    md_path = out_dir / "pre_goal_push_scan_summary.md"
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(md_path, summary)

    print(f"[arc_pre_goal_push_scan] rows={len(rows)}")
    print(f"[arc_pre_goal_push_scan] first_no_gt_270={summary['first_no_gt_270']}")
    print(f"[arc_pre_goal_push_scan] first_no_gt_180={summary['first_no_gt_180']}")
    print(f"[arc_pre_goal_push_scan] first_no_len_gt_10={summary['first_no_len_gt_10']}")
    print(f"[arc_pre_goal_push_scan] json: {json_path}")
    print(f"[arc_pre_goal_push_scan] md: {md_path}")


if __name__ == "__main__":
    main()
