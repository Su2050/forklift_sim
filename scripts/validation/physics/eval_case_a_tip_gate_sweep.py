#!/usr/bin/env python3
"""Case A tip gate sweep under fixed pre-hold controllers.

目的：
- 保持 controller 逻辑不变
- 只放宽诊断侧 tip gate（例如 0.12 -> 0.14/0.16/0.18）
- 直接回答 Case A 是否属于 gate-vs-physics mismatch
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from datetime import datetime
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[3]


def _resolve_isaaclab_root() -> Path:
    candidates = []
    isaaclab_dir = os.environ.get("ISAACLAB_DIR")
    if isaaclab_dir:
        candidates.append(Path(isaaclab_dir))
    candidates.append(REPO_ROOT / "IsaacLab")
    candidates.append(Path("/home/uniubi/projects/forklift_sim/IsaacLab"))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("Could not locate IsaacLab root.")


ISAACLAB_ROOT = _resolve_isaaclab_root()
sys.path.insert(0, str(ISAACLAB_ROOT / "source"))
task_patch_path = (
    REPO_ROOT
    / "forklift_pallet_insert_lift_project"
    / "isaaclab_patch"
    / "source"
    / "isaaclab_tasks"
)
sys.path.insert(0, str(task_patch_path))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Case A tip gate sweep")
parser.add_argument("--task", type=str, default="Isaac-Forklift-PalletInsertLift-Direct-v0")
parser.add_argument("--steps", type=int, default=240, help="Control steps per combo.")
parser.add_argument(
    "--tip-gates",
    type=str,
    default="0.17,0.175,0.18",
    help="Comma-separated diagnostic tip gate thresholds (m).",
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

if getattr(args, "enable_cameras", False):
    print("[INFO] eval_case_a_tip_gate_sweep.py 不需要相机，已忽略 --enable_cameras。", flush=True)
    args.enable_cameras = False

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

from _case_a_prehold_common import (
    CASE_A,
    PREHOLD_VARIANTS,
    build_actions,
    build_env_cfg,
    compute_metrics,
    teleport_case,
)
from isaaclab_tasks.direct.forklift_pallet_insert_lift.env import ForkliftPalletInsertLiftEnv


SELECTED_VARIANT_NAMES = ["phaseb_ref", "deep_pullout", "balanced_slow", "tip_priority"]
SELECTED_VARIANTS = [variant for variant in PREHOLD_VARIANTS if variant.name in SELECTED_VARIANT_NAMES]


def parse_tip_gates(raw: str) -> list[float]:
    values = []
    for token in raw.split(","):
        token = token.strip()
        if token:
            values.append(float(token))
    if not values:
        raise ValueError("tip gate sweep 不能为空")
    return values


def evaluate_custom_gate(
    env,
    metrics: dict[str, torch.Tensor],
    hold_counter: torch.Tensor,
    tip_entry_thresholds: torch.Tensor,
    tip_exit_thresholds: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    cfg = env._hold_logic_cfg
    exit_y = cfg.max_lateral_err_m * cfg.hysteresis_ratio
    exit_yaw = cfg.max_yaw_err_deg * cfg.hysteresis_ratio
    insert_exit_thresh = max(0.0, cfg.insert_thresh - cfg.insert_exit_epsilon)

    align_entry = (
        (metrics["center_y_abs"] <= cfg.max_lateral_err_m)
        & (metrics["yaw_err_deg_abs"] <= cfg.max_yaw_err_deg)
    )
    align_exit_exceeded = (
        (metrics["center_y_abs"] > exit_y)
        | (metrics["yaw_err_deg_abs"] > exit_yaw)
    )
    insert_entry = metrics["insert_depth"] >= cfg.insert_thresh
    insert_exit_exceeded = metrics["insert_depth"] < insert_exit_thresh
    tip_gate_active = metrics["dist_front"] <= cfg.tip_align_near_dist
    tip_entry = (~tip_gate_active) | (metrics["tip_y_abs"] <= tip_entry_thresholds)
    tip_exit_exceeded = tip_gate_active & (metrics["tip_y_abs"] > tip_exit_thresholds)

    custom_ready = insert_entry & align_entry & tip_entry & metrics["valid_insert_z"]
    any_exit_exceeded = align_exit_exceeded | insert_exit_exceeded | tip_exit_exceeded
    grace_zone = (~custom_ready) & (~any_exit_exceeded)
    hold_counter_next = torch.where(
        custom_ready,
        hold_counter + 1.0,
        torch.where(grace_zone, hold_counter, hold_counter * cfg.hold_counter_decay),
    )
    return custom_ready, tip_entry, insert_entry, hold_counter_next


def main() -> int:
    tip_gates = parse_tip_gates(args.tip_gates)
    combos = [(variant, gate) for gate in tip_gates for variant in SELECTED_VARIANTS]
    num_envs = len(combos)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = REPO_ROOT / "outputs" / "validation" / "manual_runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{timestamp}_case_a_tip_gate_sweep.csv"

    print("\n" + "=" * 120, flush=True)
    print("Case A tip gate sweep", flush=True)
    print("=" * 120, flush=True)
    print(
        f"Case A: insert_depth={CASE_A.insert_depth_m:.2f}m, lateral={CASE_A.lateral_m:+.2f}m, yaw={CASE_A.yaw_deg:+.1f}deg",
        flush=True,
    )
    print(f"Tip gates: {tip_gates}", flush=True)
    print("Selected variants:", flush=True)
    for variant in SELECTED_VARIANTS:
        print(f"  - {variant.label}", flush=True)

    env = ForkliftPalletInsertLiftEnv(build_env_cfg(num_envs))
    try:
        env.reset()
        env._hold_steps = 10_000_000
        env.cfg.paper_out_of_bounds_dist = 1e6
        env.cfg.max_roll_pitch_rad = math.pi

        teleport_case(env, CASE_A)
        initial_metrics = compute_metrics(env)

        drive_dir = -torch.ones((num_envs,), device=env.device)
        custom_hold_counter = torch.zeros((num_envs,), dtype=torch.float32, device=env.device)
        best_tip = initial_metrics["tip_y_abs"].clone()
        best_center = initial_metrics["center_y_abs"].clone()
        best_yaw = initial_metrics["yaw_err_deg_abs"].clone()
        best_insert = initial_metrics["insert_norm"].clone()
        first_tip_ok_step = torch.full((num_envs,), -1, dtype=torch.long, device=env.device)
        first_hold_step = torch.full((num_envs,), -1, dtype=torch.long, device=env.device)
        ever_hold = torch.zeros((num_envs,), dtype=torch.bool, device=env.device)

        gate_tensor = torch.tensor([gate for _, gate in combos], device=env.device, dtype=torch.float32)
        gate_exit_tensor = gate_tensor + 0.04

        for step in range(args.steps):
            metrics = compute_metrics(env)
            actions, drive_dir = build_actions(env, metrics, drive_dir, [variant for variant, _ in combos])
            env.step(actions)
            after = compute_metrics(env)

            custom_ready, tip_entry_custom, insert_entry_custom, custom_hold_counter = evaluate_custom_gate(
                env,
                after,
                custom_hold_counter,
                gate_tensor,
                gate_exit_tensor,
            )
            tip_ok_now = (
                insert_entry_custom
                & after["align_entry"]
                & tip_entry_custom
                & after["valid_insert_z"]
            )
            new_tip_ok = (first_tip_ok_step < 0) & tip_ok_now
            new_hold = (first_hold_step < 0) & custom_ready
            first_tip_ok_step = torch.where(
                new_tip_ok,
                torch.full_like(first_tip_ok_step, step),
                first_tip_ok_step,
            )
            first_hold_step = torch.where(
                new_hold,
                torch.full_like(first_hold_step, step),
                first_hold_step,
            )
            ever_hold |= custom_ready

            best_tip = torch.minimum(best_tip, after["tip_y_abs"])
            best_center = torch.minimum(best_center, after["center_y_abs"])
            best_yaw = torch.minimum(best_yaw, after["yaw_err_deg_abs"])
            best_insert = torch.maximum(best_insert, after["insert_norm"])

        final_metrics = compute_metrics(env)

        rows = []
        print("\n" + "=" * 150, flush=True)
        print("Sweep results", flush=True)
        print("=" * 150, flush=True)
        print(
            f"{'Gate':>7} | {'Variant':>16} | {'tip init->best->final':>23} | {'center init->best->final':>26} | "
            f"{'yaw init->best->final':>23} | {'tip_ok':>6} | {'hold':>6} | {'first hit':>9}",
            flush=True,
        )
        print(
            f"{'-' * 7}-+-{'-' * 16}-+-{'-' * 23}-+-{'-' * 26}-+-{'-' * 23}-+-{'-' * 6}-+-{'-' * 6}-+-{'-' * 9}",
            flush=True,
        )
        for env_id, (variant, gate) in enumerate(combos):
            first_hit = int(first_hold_step[env_id].item() if first_hold_step[env_id].item() >= 0 else first_tip_ok_step[env_id].item())
            row = {
                "tip_gate_m": gate,
                "variant": variant.label,
                "init_tip": float(initial_metrics["tip_y_abs"][env_id].item()),
                "best_tip": float(best_tip[env_id].item()),
                "final_tip": float(final_metrics["tip_y_abs"][env_id].item()),
                "init_center": float(initial_metrics["center_y_abs"][env_id].item()),
                "best_center": float(best_center[env_id].item()),
                "final_center": float(final_metrics["center_y_abs"][env_id].item()),
                "init_yaw": float(initial_metrics["yaw_err_deg_abs"][env_id].item()),
                "best_yaw": float(best_yaw[env_id].item()),
                "final_yaw": float(final_metrics["yaw_err_deg_abs"][env_id].item()),
                "best_insert": float(best_insert[env_id].item()),
                "tip_ok": bool(first_tip_ok_step[env_id].item() >= 0),
                "hold": bool(ever_hold[env_id].item()),
                "first_hit_step": first_hit,
            }
            rows.append(row)
            hit_text = str(first_hit) if first_hit >= 0 else "-"
            print(
                f"{gate:7.3f} | {variant.label:>16} | "
                f"{row['init_tip']:.3f}->{row['best_tip']:.3f}->{row['final_tip']:.3f} | "
                f"{row['init_center']:.3f}->{row['best_center']:.3f}->{row['final_center']:.3f} | "
                f"{row['init_yaw']:.2f}->{row['best_yaw']:.2f}->{row['final_yaw']:.2f} | "
                f"{str(row['tip_ok']):>6} | {str(row['hold']):>6} | {hit_text:>9}",
                flush=True,
            )

        with csv_path.open("w", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "tip_gate_m",
                    "variant",
                    "init_tip",
                    "best_tip",
                    "final_tip",
                    "init_center",
                    "best_center",
                    "final_center",
                    "init_yaw",
                    "best_yaw",
                    "final_yaw",
                    "best_insert",
                    "tip_ok",
                    "hold",
                    "first_hit_step",
                ],
            )
            writer.writeheader()
            writer.writerows(rows)

        print(f"\nCSV saved to: {csv_path}", flush=True)

        by_gate: dict[float, list[dict]] = {}
        for row in rows:
            by_gate.setdefault(row["tip_gate_m"], []).append(row)

        print("\nSummary by gate:", flush=True)
        first_any_unlock_gate = None
        first_all_unlock_gate = None
        for gate in tip_gates:
            gate_rows = by_gate[gate]
            tip_ok_count = sum(1 for row in gate_rows if row["tip_ok"])
            hold_count = sum(1 for row in gate_rows if row["hold"])
            best_row = min(gate_rows, key=lambda row: row["best_tip"])
            if first_any_unlock_gate is None and tip_ok_count > 0:
                first_any_unlock_gate = gate
            if first_all_unlock_gate is None and tip_ok_count == len(gate_rows):
                first_all_unlock_gate = gate
            print(
                f"  - gate={gate:.3f}: tip_ok={tip_ok_count}/{len(gate_rows)}, hold={hold_count}/{len(gate_rows)}, "
                f"best={best_row['variant']} tip {best_row['init_tip']:.3f}->{best_row['best_tip']:.3f}",
                flush=True,
            )

        if first_any_unlock_gate is not None:
            print(
                f"\nBoundary summary: first gate that unlocks at least one controller = {first_any_unlock_gate:.3f} m",
                flush=True,
            )
        else:
            print(
                "\nBoundary summary: none of the tested gates unlock any controller.",
                flush=True,
            )
        if first_all_unlock_gate is not None:
            print(
                f"Boundary summary: first gate that unlocks all tested controllers = {first_all_unlock_gate:.3f} m",
                flush=True,
            )
        else:
            print(
                "Boundary summary: none of the tested gates unlock all tested controllers.",
                flush=True,
            )

        strict_rows = by_gate[min(tip_gates)]
        loose_rows = by_gate[max(tip_gates)]
        if sum(1 for row in strict_rows if row["tip_ok"]) == 0 and sum(1 for row in loose_rows if row["tip_ok"]) > 0:
            print(
                "\nConclusion: relaxed tip gate unlocks Case A for at least one controller. "
                "This is strong evidence for gate-vs-physics mismatch.",
                flush=True,
            )
        elif sum(1 for row in loose_rows if row["tip_ok"]) == 0:
            print(
                "\nConclusion: even with relaxed tip gates, Case A still does not cross tip/hold. "
                "This points away from pure gate mismatch and toward a deeper controllability/physics bottleneck.",
                flush=True,
            )
        else:
            print(
                "\nConclusion: relaxed tip gates help, but the effect is partial rather than binary.",
                flush=True,
            )
        return 0
    finally:
        env.close()
        simulation_app.close()


if __name__ == "__main__":
    raise SystemExit(main())
