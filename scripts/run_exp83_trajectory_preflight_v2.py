#!/usr/bin/env python3
"""Run Exp8.3 trajectory preflight v2 against current and legacy cfgs.

This wrapper keeps trajectory auditing separate from PPO training:
- regenerates the stage1 reference-trajectory visualizations
- evaluates each case with proxy sanity thresholds
- writes a compact json + markdown summary for controlled comparisons

The thresholds here are deliberately proxy-level sanity checks, not hard
vehicle-kinematic guarantees. They exist to catch "obviously too aggressive"
trajectory families before any RL training is resumed.
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = REPO_ROOT / "forklift_pallet_insert_lift_project"
VIZ_SCRIPT = PROJECT_ROOT / "scripts" / "visualize_reference_trajectory_cases.py"
CURRENT_CFG = REPO_ROOT / "IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/forklift_pallet_insert_lift/env_cfg.py"
LEGACY_CFG = PROJECT_ROOT / "isaaclab_patch/source/isaaclab_tasks/isaaclab_tasks/direct/forklift_pallet_insert_lift/env_cfg.py"

OUT_ROOT = REPO_ROOT / "outputs" / "exp83_trajectory_preflight_v2"
CURRENT_OUT = OUT_ROOT / "current"
LEGACY_OUT = OUT_ROOT / "legacy"


@dataclass(frozen=True)
class ProxyThresholds:
    root_y_warn_factor: float = 2.5
    root_y_bad_factor: float = 4.0
    root_heading_warn_deg: float = 45.0
    root_heading_bad_deg: float = 90.0
    root_curvature_warn: float = 4.0
    root_curvature_bad: float = 10.0


def run_viz(*, cfg_path: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(VIZ_SCRIPT),
        "--cfg-path",
        str(cfg_path),
        "--output-dir",
        str(out_dir),
    ]
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def load_manifest(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def classify_case(case: dict, *, stage1_y_abs_max: float, th: ProxyThresholds) -> tuple[str, list[str]]:
    problems: list[str] = []
    severity = "ok"

    root_y_warn = stage1_y_abs_max * th.root_y_warn_factor
    root_y_bad = stage1_y_abs_max * th.root_y_bad_factor

    def bump(level: str, message: str) -> None:
        nonlocal severity
        order = {"ok": 0, "warn": 1, "bad": 2}
        if order[level] > order[severity]:
            severity = level
        problems.append(message)

    if not case["entry_ok"]:
        bump("bad", "entry_ok=false")
    if case["root_y_abs_max"] > root_y_bad:
        bump("bad", f"root_y_abs_max>{root_y_bad:.3f}m")
    elif case["root_y_abs_max"] > root_y_warn:
        bump("warn", f"root_y_abs_max>{root_y_warn:.3f}m")

    if case["root_heading_change_deg"] > th.root_heading_bad_deg:
        bump("bad", f"root_heading_change_deg>{th.root_heading_bad_deg:.1f}deg")
    elif case["root_heading_change_deg"] > th.root_heading_warn_deg:
        bump("warn", f"root_heading_change_deg>{th.root_heading_warn_deg:.1f}deg")

    if case["root_curvature_max"] > th.root_curvature_bad:
        bump("bad", f"root_curvature_max>{th.root_curvature_bad:.1f}/m")
    elif case["root_curvature_max"] > th.root_curvature_warn:
        bump("warn", f"root_curvature_max>{th.root_curvature_warn:.1f}/m")

    return severity, problems


def summarize_scope(name: str, manifest: dict, th: ProxyThresholds) -> dict:
    parsed_cfg = manifest["parsed_cfg"]
    stage1_y_abs_max = max(abs(float(parsed_cfg["stage1_init_y_min_m"])), abs(float(parsed_cfg["stage1_init_y_max_m"])))
    rows: list[dict] = []

    for case in manifest["cases"]:
        severity, problems = classify_case(case, stage1_y_abs_max=stage1_y_abs_max, th=th)
        row = dict(case)
        row["proxy_severity"] = severity
        row["proxy_problems"] = problems
        rows.append(row)

    by_severity = {
        level: sum(1 for row in rows if row["proxy_severity"] == level)
        for level in ("ok", "warn", "bad")
    }

    worst_root_y = max(rows, key=lambda row: row["root_y_abs_max"])
    worst_heading = max(rows, key=lambda row: row["root_heading_change_deg"])
    worst_curvature = max(rows, key=lambda row: row["root_curvature_max"])

    return {
        "scope": name,
        "cfg_path": manifest["cfg_path_used"],
        "stage1_y_abs_max": stage1_y_abs_max,
        "thresholds": {
            "root_y_warn_m": stage1_y_abs_max * th.root_y_warn_factor,
            "root_y_bad_m": stage1_y_abs_max * th.root_y_bad_factor,
            "root_heading_warn_deg": th.root_heading_warn_deg,
            "root_heading_bad_deg": th.root_heading_bad_deg,
            "root_curvature_warn": th.root_curvature_warn,
            "root_curvature_bad": th.root_curvature_bad,
        },
        "counts": {
            "num_cases": len(rows),
            "num_entry_ok": sum(1 for row in rows if row["entry_ok"]),
            "num_proxy_ok": by_severity["ok"],
            "num_proxy_warn": by_severity["warn"],
            "num_proxy_bad": by_severity["bad"],
        },
        "worst_cases": {
            "root_y_abs_max": {
                "case_id": worst_root_y["case_id"],
                "value": worst_root_y["root_y_abs_max"],
            },
            "root_heading_change_deg": {
                "case_id": worst_heading["case_id"],
                "value": worst_heading["root_heading_change_deg"],
            },
            "root_curvature_max": {
                "case_id": worst_curvature["case_id"],
                "value": worst_curvature["root_curvature_max"],
            },
        },
        "bad_cases": [
            {
                "case_id": row["case_id"],
                "root_x": row["root_x"],
                "root_y": row["root_y"],
                "yaw_deg": row["yaw_deg"],
                "proxy_problems": row["proxy_problems"],
            }
            for row in rows
            if row["proxy_severity"] == "bad"
        ],
        "rows": rows,
    }


def write_markdown(path: Path, summary: dict) -> None:
    lines = [
        "# Exp8.3 Trajectory Preflight V2 Summary",
        "",
        "This report is generated by `scripts/run_exp83_trajectory_preflight_v2.py`.",
        "",
    ]
    for scope in summary["scopes"]:
        counts = scope["counts"]
        th = scope["thresholds"]
        lines.extend(
            [
                f"## {scope['scope']}",
                "",
                f"- cfg_path: `{scope['cfg_path']}`",
                f"- num_cases: `{counts['num_cases']}`",
                f"- entry_ok: `{counts['num_entry_ok']}/{counts['num_cases']}`",
                f"- proxy_ok: `{counts['num_proxy_ok']}`",
                f"- proxy_warn: `{counts['num_proxy_warn']}`",
                f"- proxy_bad: `{counts['num_proxy_bad']}`",
                f"- thresholds: `root_y_warn={th['root_y_warn_m']:.3f}m`, `root_y_bad={th['root_y_bad_m']:.3f}m`, `heading_warn={th['root_heading_warn_deg']:.1f}deg`, `heading_bad={th['root_heading_bad_deg']:.1f}deg`, `curvature_warn={th['root_curvature_warn']:.1f}/m`, `curvature_bad={th['root_curvature_bad']:.1f}/m`",
                f"- worst root_y_abs_max: `{scope['worst_cases']['root_y_abs_max']['case_id']} -> {scope['worst_cases']['root_y_abs_max']['value']:.4f}m`",
                f"- worst root_heading_change_deg: `{scope['worst_cases']['root_heading_change_deg']['case_id']} -> {scope['worst_cases']['root_heading_change_deg']['value']:.2f}deg`",
                f"- worst root_curvature_max: `{scope['worst_cases']['root_curvature_max']['case_id']} -> {scope['worst_cases']['root_curvature_max']['value']:.3f}/m`",
                "",
                "### Bad Cases",
                "",
            ]
        )
        if not scope["bad_cases"]:
            lines.append("- none")
        else:
            for case in scope["bad_cases"]:
                problems = ", ".join(case["proxy_problems"])
                lines.append(
                    f"- `{case['case_id']}` at `(x={case['root_x']:+.3f}, y={case['root_y']:+.3f}, yaw={case['yaw_deg']:+.1f})`: {problems}"
                )
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    th = ProxyThresholds()
    run_viz(cfg_path=CURRENT_CFG, out_dir=CURRENT_OUT)
    run_viz(cfg_path=LEGACY_CFG, out_dir=LEGACY_OUT)

    current_manifest = load_manifest(CURRENT_OUT / "reference_trajectory_stage1_manifest.json")
    legacy_manifest = load_manifest(LEGACY_OUT / "reference_trajectory_stage1_manifest.json")

    summary = {
        "tool": "run_exp83_trajectory_preflight_v2.py",
        "threshold_policy": "proxy sanity thresholds, not full kinematic guarantees",
        "scopes": [
            summarize_scope("current", current_manifest, th),
            summarize_scope("legacy", legacy_manifest, th),
        ],
    }

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    json_path = OUT_ROOT / "preflight_v2_summary.json"
    md_path = OUT_ROOT / "preflight_v2_summary.md"
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(md_path, summary)

    for scope in summary["scopes"]:
        counts = scope["counts"]
        print(
            f"[preflight_v2] {scope['scope']}: "
            f"entry_ok={counts['num_entry_ok']}/{counts['num_cases']} "
            f"proxy_ok={counts['num_proxy_ok']} "
            f"proxy_warn={counts['num_proxy_warn']} "
            f"proxy_bad={counts['num_proxy_bad']}"
        )
    print(f"[preflight_v2] json: {json_path}")
    print(f"[preflight_v2] md: {md_path}")


if __name__ == "__main__":
    main()
