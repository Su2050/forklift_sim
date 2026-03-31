#!/usr/bin/env python3
"""Run a curvature-bounded arc preflight against the current Stage1 case grid.

This tool keeps the experiment offline:
- no PPO or reward changes
- no env patching
- just batch-generate a bounded-curvature trajectory family and audit 125 cases
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = REPO_ROOT / "forklift_pallet_insert_lift_project"
VIZ_SCRIPT = PROJECT_ROOT / "scripts" / "visualize_reference_trajectory_cases.py"
CURRENT_CFG = REPO_ROOT / "IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/forklift_pallet_insert_lift/env_cfg.py"
PATCH_CFG = PROJECT_ROOT / "isaaclab_patch/source/isaaclab_tasks/isaaclab_tasks/direct/forklift_pallet_insert_lift/env_cfg.py"
DEFAULT_CFG = PATCH_CFG
OUT_ROOT = REPO_ROOT / "outputs" / "exp83_arc_preflight_v1"


def run_viz(*, cfg_path: Path, out_dir: Path, traj_model: str, grid_count_x: int, grid_count_y: int, grid_count_yaw: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(VIZ_SCRIPT),
        "--cfg-path",
        str(cfg_path),
        "--traj-model",
        traj_model,
        "--grid-count-x",
        str(grid_count_x),
        "--grid-count-y",
        str(grid_count_y),
        "--grid-count-yaw",
        str(grid_count_yaw),
        "--output-dir",
        str(out_dir),
    ]
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def load_latest_manifest(out_dir: Path) -> dict:
    manifests = sorted(out_dir.glob("reference_trajectory_stage1_manifest*.json"))
    if not manifests:
        raise FileNotFoundError(f"no manifest found in {out_dir}")
    return json.loads(manifests[-1].read_text(encoding="utf-8"))


def write_markdown(path: Path, summary: dict) -> None:
    lines = [
        "# Exp8.3 Arc Preflight V1",
        "",
        f"- traj_model: `{summary['traj_model']}`",
        f"- cfg_path: `{summary['cfg_path']}`",
        f"- overlay: `{summary['overlay_path']}`",
        f"- curvature_limit: `{summary['curvature_limit_m_inv']:.6f} /m`",
        f"- num_cases: `{summary['num_cases']}`",
        f"- num_entry_ok: `{summary['num_entry_ok']}`",
        f"- num_curvature_ok: `{summary['num_curvature_ok']}`",
        f"- num_curvature_over_limit: `{summary['num_curvature_over_limit']}`",
        f"- root_total_length_m_max: `{summary['root_total_length_m_max']:.3f}`",
        f"- root_total_length_m_mean: `{summary['root_total_length_m_mean']:.3f}`",
        f"- root_heading_change_deg_max: `{summary['root_heading_change_deg_max']:.3f}`",
        f"- root_curvature_max_max: `{summary['root_curvature_max_max']:.6f}`",
        "",
        "## Path Mode Counts",
        "",
    ]
    for mode, count in summary["path_mode_counts"].items():
        lines.append(f"- `{mode}`: `{count}`")
    lines.extend(
        [
            "",
            "## Worst Cases",
            "",
            f"- longest_root_path: `{summary['worst_cases']['longest_root_path']['case_id']}` -> `{summary['worst_cases']['longest_root_path']['value']:.3f} m`",
            f"- max_curvature: `{summary['worst_cases']['max_curvature']['case_id']}` -> `{summary['worst_cases']['max_curvature']['value']:.6f} /m`",
            f"- max_heading_change: `{summary['worst_cases']['max_heading_change']['case_id']}` -> `{summary['worst_cases']['max_heading_change']['value']:.3f} deg`",
        ]
    )
    if summary["curvature_over_limit_cases"]:
        lines.extend(["", "## Curvature Over Limit", ""])
        for row in summary["curvature_over_limit_cases"]:
            lines.append(
                f"- `{row['case_id']}`: kappa=`{row['root_curvature_max']:.6f}` path_len=`{row['root_total_length_m']:.3f}` mode=`{row['path_mode']}`"
            )
    else:
        lines.extend(["", "## Curvature Over Limit", "", "- none"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def summarize(manifest: dict) -> dict:
    cases = manifest["cases"]
    cfg = manifest["parsed_cfg"]
    curvature_limit_m_inv = 1.0 / float(cfg["traj_rs_min_turn_radius_m"])
    curvature_over_limit_cases = [
        {
            "case_id": case["case_id"],
            "root_curvature_max": case["root_curvature_max"],
            "root_total_length_m": case["root_total_length_m"],
            "path_mode": case["path_mode"],
        }
        for case in cases
        if float(case["root_curvature_max"]) > curvature_limit_m_inv + 1e-4
    ]
    longest_root_path = max(cases, key=lambda case: case["root_total_length_m"])
    max_curvature = max(cases, key=lambda case: case["root_curvature_max"])
    max_heading = max(cases, key=lambda case: case["root_heading_change_deg"])

    return {
        "tool": "run_exp83_arc_preflight_v1.py",
        "traj_model": manifest["validation_scope"][0].split("=", 1)[1],
        "cfg_path": manifest["cfg_path_used"],
        "overlay_path": manifest["overlay_path"],
        "curvature_limit_m_inv": curvature_limit_m_inv,
        "num_cases": manifest["num_cases"],
        "num_entry_ok": manifest["num_entry_ok"],
        "num_curvature_ok": manifest["num_cases"] - len(curvature_over_limit_cases),
        "num_curvature_over_limit": len(curvature_over_limit_cases),
        "path_mode_counts": manifest["path_mode_counts"],
        "root_total_length_m_max": manifest["root_total_length_m_max"],
        "root_total_length_m_mean": manifest["root_total_length_m_mean"],
        "root_heading_change_deg_max": manifest["root_heading_change_deg_max"],
        "root_curvature_max_max": manifest["root_curvature_max_max"],
        "worst_cases": {
            "longest_root_path": {
                "case_id": longest_root_path["case_id"],
                "value": longest_root_path["root_total_length_m"],
            },
            "max_curvature": {
                "case_id": max_curvature["case_id"],
                "value": max_curvature["root_curvature_max"],
            },
            "max_heading_change": {
                "case_id": max_heading["case_id"],
                "value": max_heading["root_heading_change_deg"],
            },
        },
        "curvature_over_limit_cases": curvature_over_limit_cases,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Exp8.3 bounded-curvature arc preflight.")
    parser.add_argument("--cfg-path", type=Path, default=DEFAULT_CFG)
    parser.add_argument("--traj-model", type=str, default="dubins_to_pre_straight")
    parser.add_argument("--grid-count-x", type=int, default=5)
    parser.add_argument("--grid-count-y", type=int, default=5)
    parser.add_argument("--grid-count-yaw", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, default=OUT_ROOT / "dubins_to_pre_straight")
    args = parser.parse_args()

    run_viz(
        cfg_path=args.cfg_path,
        out_dir=args.output_dir,
        traj_model=str(args.traj_model),
        grid_count_x=int(args.grid_count_x),
        grid_count_y=int(args.grid_count_y),
        grid_count_yaw=int(args.grid_count_yaw),
    )

    manifest = load_latest_manifest(args.output_dir)
    summary = summarize(manifest)
    json_path = args.output_dir / "arc_preflight_v1_summary.json"
    md_path = args.output_dir / "arc_preflight_v1_summary.md"
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(md_path, summary)

    print(
        "[arc_preflight_v1] "
        f"cases={summary['num_cases']} "
        f"entry_ok={summary['num_entry_ok']} "
        f"curvature_ok={summary['num_curvature_ok']} "
        f"curvature_over_limit={summary['num_curvature_over_limit']}"
    )
    print(f"[arc_preflight_v1] overlay: {summary['overlay_path']}")
    print(f"[arc_preflight_v1] summary_json: {json_path}")
    print(f"[arc_preflight_v1] summary_md: {md_path}")


if __name__ == "__main__":
    main()
