#!/usr/bin/env python3
"""Time-series replay diagnostic for Case A pre-hold correction.

保存固定 Case A 在若干 controller 下的时间序列，帮助回答：
- 车体有没有明显横向移动？
- fork center / fork tip 的横向误差是否真的在跟着变小？
- 是“车在动但 tip 不跟着修正”，还是“连车体都几乎没横移”？
"""

from __future__ import annotations

import argparse
import csv
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


parser = argparse.ArgumentParser(description="Replay Case A pre-hold diagnostic")
parser.add_argument("--task", type=str, default="Isaac-Forklift-PalletInsertLift-Direct-v0")
parser.add_argument("--steps", type=int, default=240, help="Replay steps.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

if getattr(args, "enable_cameras", False):
    print("[INFO] replay_case_a_prehold_diagnostic.py 不需要相机，已忽略 --enable_cameras。", flush=True)
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


SELECTED_VARIANT_NAMES = ["phaseb_ref", "balanced_slow", "deep_pullout"]
SELECTED_VARIANTS = [variant for variant in PREHOLD_VARIANTS if variant.name in SELECTED_VARIANT_NAMES]


def main() -> int:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = REPO_ROOT / "outputs" / "validation" / "manual_runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{timestamp}_case_a_prehold_replay.csv"
    plot_path = out_dir / f"{timestamp}_case_a_prehold_replay.png"

    print("\n" + "=" * 120, flush=True)
    print("Case A pre-hold replay diagnostic", flush=True)
    print("=" * 120, flush=True)
    print(
        f"Case A: insert_depth={CASE_A.insert_depth_m:.2f}m, lateral={CASE_A.lateral_m:+.2f}m, yaw={CASE_A.yaw_deg:+.1f}deg",
        flush=True,
    )
    print("Replay variants:", flush=True)
    for variant in SELECTED_VARIANTS:
        print(f"  - {variant.label}", flush=True)

    env = ForkliftPalletInsertLiftEnv(build_env_cfg(len(SELECTED_VARIANTS)))
    try:
        env.reset()
        env._hold_steps = 10_000_000
        env.cfg.paper_out_of_bounds_dist = 1e6
        env.cfg.max_roll_pitch_rad = float("inf")

        teleport_case(env, CASE_A)
        drive_dir = -torch.ones((env.num_envs,), device=env.device)

        records: list[dict[str, float | str | int | bool]] = []

        def append_records(step: int, metrics: dict[str, torch.Tensor], actions: torch.Tensor | None) -> None:
            for env_id, variant in enumerate(SELECTED_VARIANTS):
                records.append(
                    {
                        "step": step,
                        "variant": variant.label,
                        "drive_cmd": float(actions[env_id, 0].item()) if actions is not None else 0.0,
                        "steer_cmd": float(actions[env_id, 1].item()) if actions is not None else 0.0,
                        "root_y": float(metrics["root_y"][env_id].item()),
                        "center_y_signed": float(metrics["center_y_signed"][env_id].item()),
                        "center_y_abs": float(metrics["center_y_abs"][env_id].item()),
                        "tip_y_signed": float(metrics["tip_y_signed"][env_id].item()),
                        "tip_y_abs": float(metrics["tip_y_abs"][env_id].item()),
                        "yaw_err_deg": float(metrics["yaw_err_deg_abs"][env_id].item()),
                        "insert_norm": float(metrics["insert_norm"][env_id].item()),
                        "dist_front": float(metrics["dist_front"][env_id].item()),
                        "hold_entry": bool(metrics["hold_entry"][env_id].item()),
                        "tip_entry": bool(metrics["tip_entry"][env_id].item()),
                        "align_entry": bool(metrics["align_entry"][env_id].item()),
                    }
                )

        current = compute_metrics(env)
        append_records(0, current, None)

        for step in range(1, args.steps + 1):
            actions, drive_dir = build_actions(env, current, drive_dir, SELECTED_VARIANTS)
            env.step(actions)
            current = compute_metrics(env)
            append_records(step, current, actions)

        with csv_path.open("w", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "step",
                    "variant",
                    "drive_cmd",
                    "steer_cmd",
                    "root_y",
                    "center_y_signed",
                    "center_y_abs",
                    "tip_y_signed",
                    "tip_y_abs",
                    "yaw_err_deg",
                    "insert_norm",
                    "dist_front",
                    "hold_entry",
                    "tip_entry",
                    "align_entry",
                ],
            )
            writer.writeheader()
            writer.writerows(records)

        print(f"CSV saved to: {csv_path}", flush=True)

        summary_rows = []
        for variant in SELECTED_VARIANTS:
            rows = [row for row in records if row["variant"] == variant.label]
            init = rows[0]
            best_tip = min(row["tip_y_abs"] for row in rows)
            best_center = min(row["center_y_abs"] for row in rows)
            best_yaw = min(row["yaw_err_deg"] for row in rows)
            root_y_values = [row["root_y"] for row in rows]
            root_y_span = max(root_y_values) - min(root_y_values)
            hold_steps = [row["step"] for row in rows if row["hold_entry"]]
            tip_steps = [row["step"] for row in rows if row["tip_entry"] and row["align_entry"]]
            summary_rows.append(
                {
                    "variant": variant.label,
                    "init_tip": init["tip_y_abs"],
                    "best_tip": best_tip,
                    "tip_gain": init["tip_y_abs"] - best_tip,
                    "init_center": init["center_y_abs"],
                    "best_center": best_center,
                    "center_gain": init["center_y_abs"] - best_center,
                    "init_yaw": init["yaw_err_deg"],
                    "best_yaw": best_yaw,
                    "yaw_gain": init["yaw_err_deg"] - best_yaw,
                    "root_y_span": root_y_span,
                    "tip_hit_step": min(tip_steps) if tip_steps else -1,
                    "hold_hit_step": min(hold_steps) if hold_steps else -1,
                }
            )

        print("\nSummary:", flush=True)
        for row in summary_rows:
            tip_hit = row["tip_hit_step"] if row["tip_hit_step"] >= 0 else "-"
            hold_hit = row["hold_hit_step"] if row["hold_hit_step"] >= 0 else "-"
            print(
                f"  - {row['variant']}: tip {row['init_tip']:.3f}->{row['best_tip']:.3f} "
                f"(gain {row['tip_gain']:.3f}), center {row['init_center']:.3f}->{row['best_center']:.3f} "
                f"(gain {row['center_gain']:.3f}), yaw {row['init_yaw']:.2f}->{row['best_yaw']:.2f} "
                f"(gain {row['yaw_gain']:.2f}), root_y_span={row['root_y_span']:.3f}, "
                f"tip_hit={tip_hit}, hold_hit={hold_hit}",
                flush=True,
            )

        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
            for variant in SELECTED_VARIANTS:
                rows = [row for row in records if row["variant"] == variant.label]
                steps = [row["step"] for row in rows]
                axes[0].plot(steps, [row["tip_y_abs"] for row in rows], label=variant.label)
                axes[1].plot(steps, [row["center_y_abs"] for row in rows], label=variant.label)
                axes[2].plot(steps, [row["yaw_err_deg"] for row in rows], label=variant.label)
                axes[3].plot(steps, [row["insert_norm"] for row in rows], label=variant.label)

            axes[0].axhline(0.12, color="black", linestyle="--", linewidth=1)
            axes[1].axhline(0.15, color="black", linestyle="--", linewidth=1)
            axes[2].axhline(8.0, color="black", linestyle="--", linewidth=1)
            axes[0].set_ylabel("tip_y_abs (m)")
            axes[1].set_ylabel("center_y_abs (m)")
            axes[2].set_ylabel("yaw_err_deg")
            axes[3].set_ylabel("insert_norm")
            axes[3].set_xlabel("step")
            axes[0].legend(loc="best")
            fig.suptitle("Case A pre-hold replay")
            fig.tight_layout()
            fig.savefig(plot_path, dpi=140)
            plt.close(fig)
            print(f"Plot saved to: {plot_path}", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] Could not render plot: {exc}", flush=True)

        return 0
    finally:
        env.close()
        simulation_app.close()


if __name__ == "__main__":
    raise SystemExit(main())
