#!/usr/bin/env python3
"""Focused pre-hold controller ablation for Case A.

目标：
- 固定在 Case A（已深插，但 tip/hold gate 尚未通过）上做更强的 pre-hold correction 诊断
- 用一组更激进的控制器对照，回答：
  1. 这个 case 在当前物理下能否被 controller 救回 hold 区？
  2. 如果能，哪类 controller 最有效？
  3. 如果不能，当前瓶颈更像控制不足还是物理/几何可纠偏性不足？

说明：
- 为了只看 pre-hold controllability，本脚本对所有 variant 都禁用 hold-freeze。
- 这不是训练 reward 结果，而是固定 case 的“上限型”诊断。
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from dataclasses import dataclass
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
    raise FileNotFoundError(
        "Could not locate IsaacLab root. Set ISAACLAB_DIR or provide a local IsaacLab checkout."
    )


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

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Focused pre-hold Case A controller ablation")
parser.add_argument("--task", type=str, default="Isaac-Forklift-PalletInsertLift-Direct-v0")
parser.add_argument("--steps", type=int, default=240, help="Number of control steps to evaluate.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

if getattr(args, "enable_cameras", False):
    print("[INFO] eval_prehold_case_a_ablation.py 不需要相机，已忽略 --enable_cameras。", flush=True)
    args.enable_cameras = False

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

from isaaclab_tasks.direct.forklift_pallet_insert_lift.env import ForkliftPalletInsertLiftEnv
from isaaclab_tasks.direct.forklift_pallet_insert_lift.env_cfg import ForkliftPalletInsertLiftEnvCfg
from isaaclab_tasks.direct.forklift_pallet_insert_lift.hold_logic import compute_hold_logic


@dataclass(frozen=True)
class CaseSpec:
    name: str
    insert_depth_m: float
    lateral_m: float
    yaw_deg: float


@dataclass(frozen=True)
class ControllerVariant:
    name: str
    label: str
    corridor_low: float
    corridor_high: float
    drive_fwd: float
    drive_rev: float
    k_center: float
    k_tip: float
    k_yaw: float
    hold_margin_tip: float = 0.0


CASE_A = CaseSpec(
    name="case_a",
    insert_depth_m=1.00,
    lateral_m=0.18,
    yaw_deg=4.0,
)


VARIANTS = [
    ControllerVariant(
        name="phaseb_ref",
        label="Phase-B ref",
        corridor_low=0.44,
        corridor_high=0.58,
        drive_fwd=0.18,
        drive_rev=0.16,
        k_center=3.4,
        k_tip=1.6,
        k_yaw=0.35,
    ),
    ControllerVariant(
        name="tip_priority",
        label="Tip priority",
        corridor_low=0.32,
        corridor_high=0.48,
        drive_fwd=0.12,
        drive_rev=0.22,
        k_center=2.0,
        k_tip=3.5,
        k_yaw=0.25,
    ),
    ControllerVariant(
        name="aggr_tip",
        label="Aggressive tip",
        corridor_low=0.26,
        corridor_high=0.44,
        drive_fwd=0.10,
        drive_rev=0.26,
        k_center=2.0,
        k_tip=5.0,
        k_yaw=0.35,
    ),
    ControllerVariant(
        name="tip_yaw",
        label="Tip + yaw",
        corridor_low=0.30,
        corridor_high=0.46,
        drive_fwd=0.10,
        drive_rev=0.24,
        k_center=1.8,
        k_tip=4.0,
        k_yaw=0.80,
    ),
    ControllerVariant(
        name="center_priority",
        label="Center priority",
        corridor_low=0.30,
        corridor_high=0.46,
        drive_fwd=0.12,
        drive_rev=0.22,
        k_center=5.0,
        k_tip=1.2,
        k_yaw=0.25,
    ),
    ControllerVariant(
        name="deep_pullout",
        label="Deep pull-out",
        corridor_low=0.20,
        corridor_high=0.38,
        drive_fwd=0.14,
        drive_rev=0.30,
        k_center=2.5,
        k_tip=4.5,
        k_yaw=0.70,
    ),
    ControllerVariant(
        name="balanced_slow",
        label="Balanced slow",
        corridor_low=0.34,
        corridor_high=0.46,
        drive_fwd=0.08,
        drive_rev=0.18,
        k_center=2.6,
        k_tip=2.8,
        k_yaw=0.45,
    ),
    ControllerVariant(
        name="yaw_first",
        label="Yaw first",
        corridor_low=0.30,
        corridor_high=0.46,
        drive_fwd=0.10,
        drive_rev=0.22,
        k_center=1.5,
        k_tip=2.0,
        k_yaw=1.20,
    ),
]


def build_env_cfg(num_envs: int) -> ForkliftPalletInsertLiftEnvCfg:
    cfg = ForkliftPalletInsertLiftEnvCfg()
    cfg.scene.num_envs = num_envs
    cfg.use_camera = False
    cfg.use_asymmetric_critic = False
    cfg.wait_for_textures = False
    cfg.use_reference_trajectory = False
    cfg.episode_length_s = max(float(getattr(cfg, "episode_length_s", 0.0)), 3600.0)
    cfg.paper_out_of_bounds_dist = 1e6
    cfg.max_roll_pitch_rad = math.pi
    return cfg


def _quat_to_yaw(quat: torch.Tensor) -> torch.Tensor:
    w, x, y, z = quat.unbind(-1)
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _reset_internal_buffers(env: ForkliftPalletInsertLiftEnv, env_ids: torch.Tensor) -> None:
    env.actions[env_ids] = 0.0
    if hasattr(env, "previous_actions"):
        env.previous_actions[env_ids] = 0.0
    env._last_insert_depth[env_ids] = 0.0
    env._hold_counter[env_ids] = 0
    env._last_hold_entry[env_ids] = False
    env._is_first_step[env_ids] = True
    env._lift_pos_target[env_ids] = 0.0
    env._milestone_flags[env_ids] = False
    env._fork_tip_z0[env_ids] = 0.03
    env.episode_length_buf[env_ids] = 0

    optional_zeros = [
        "_fly_counter",
        "_stall_counter",
        "_early_stop_fly",
        "_early_stop_stall",
        "_dz_stuck_counter",
        "_prev_y_err",
        "_prev_yaw_err_deg",
        "_early_stop_dz_stuck",
        "_dz_stuck_fired",
        "_prev_phi_align",
        "_prev_phi_lift_progress",
        "_prev_insert_norm",
        "_prev_in_dead_zone",
        "_prev_phi_lat",
        "_milestone_lift_10cm",
        "_milestone_lift_20cm",
        "_milestone_lift_50cm",
        "_milestone_lift_75cm",
        "_prev_y_err_far",
        "_global_stall_counter",
        "_prev_phi_total_stall",
        "_insert_norm_window",
        "_window_ptr",
        "_window_filled",
        "_prev_phi_total",
        "_prev_dist_front",
        "_last_lift_pos",
    ]
    for name in optional_zeros:
        if hasattr(env, name):
            buffer = getattr(env, name)
            if buffer.dtype == torch.bool:
                buffer[env_ids] = False
            else:
                buffer[env_ids] = 0


def teleport_case(env: ForkliftPalletInsertLiftEnv, case: CaseSpec) -> None:
    device = env.device
    num_envs = env.num_envs
    env_ids = torch.arange(num_envs, device=device, dtype=torch.long)

    pallet_pos = env.pallet.data.root_pos_w[env_ids].clone()
    pallet_quat = env.pallet.data.root_quat_w[env_ids].clone()
    pallet_yaw = _quat_to_yaw(pallet_quat)
    fork_offset = float(env._fork_forward_offset)
    s_front = -0.5 * float(env.cfg.pallet_depth_m)

    pos_rows = []
    quat_rows = []
    for index in range(num_envs):
        py = pallet_yaw[index].item()
        cp = math.cos(py)
        sp = math.sin(py)
        v_lat_x = -sp
        v_lat_y = cp

        robot_yaw = py + math.radians(case.yaw_deg)
        desired_s_tip = s_front + case.insert_depth_m
        desired_tip_x = pallet_pos[index, 0].item() + desired_s_tip * cp + case.lateral_m * v_lat_x
        desired_tip_y = pallet_pos[index, 1].item() + desired_s_tip * sp + case.lateral_m * v_lat_y
        root_x = desired_tip_x - fork_offset * math.cos(robot_yaw)
        root_y = desired_tip_y - fork_offset * math.sin(robot_yaw)
        root_z = 0.03

        pos_rows.append([root_x, root_y, root_z])
        quat_rows.append([math.cos(robot_yaw * 0.5), 0.0, 0.0, math.sin(robot_yaw * 0.5)])

    pos = torch.tensor(pos_rows, device=device, dtype=torch.float32)
    quat = torch.tensor(quat_rows, device=device, dtype=torch.float32)

    env._write_root_pose(env.robot, pos, quat, env_ids)
    zeros3 = torch.zeros((num_envs, 3), device=device)
    env._write_root_vel(env.robot, zeros3, zeros3, env_ids)
    joint_pos = env.robot.data.default_joint_pos[env_ids].clone()
    joint_vel = torch.zeros_like(joint_pos)
    env._write_joint_state(env.robot, joint_pos, joint_vel, env_ids)

    env.scene.write_data_to_sim()
    env.sim.step(render=False)
    env.scene.update(env.cfg.sim.dt)

    env._joint_pos[:] = env.robot.root_physx_view.get_dof_positions()
    env._joint_vel[:] = env.robot.root_physx_view.get_dof_velocities()
    _reset_internal_buffers(env, env_ids)


def compute_metrics(env: ForkliftPalletInsertLiftEnv) -> dict[str, torch.Tensor]:
    env._joint_pos[:] = env.robot.root_physx_view.get_dof_positions()
    env._joint_vel[:] = env.robot.root_physx_view.get_dof_velocities()

    root_pos = env.robot.data.root_pos_w
    root_quat = env.robot.data.root_quat_w
    pallet_pos = env.pallet.data.root_pos_w
    pallet_quat = env.pallet.data.root_quat_w
    tip = env._compute_fork_tip()
    fork_center = env._compute_fork_center()

    pallet_yaw = _quat_to_yaw(pallet_quat)
    robot_yaw = _quat_to_yaw(root_quat)
    yaw_err_signed = torch.atan2(
        torch.sin(robot_yaw - pallet_yaw),
        torch.cos(robot_yaw - pallet_yaw),
    )
    yaw_err_deg = torch.abs(yaw_err_signed) * (180.0 / math.pi)

    cp = torch.cos(pallet_yaw)
    sp = torch.sin(pallet_yaw)
    u_in = torch.stack([cp, sp], dim=-1)
    v_lat = torch.stack([-sp, cp], dim=-1)

    rel_tip = tip[:, :2] - pallet_pos[:, :2]
    rel_fc = fork_center[:, :2] - pallet_pos[:, :2]
    s_tip = torch.sum(rel_tip * u_in, dim=-1)
    tip_y_signed = torch.sum(rel_tip * v_lat, dim=-1)
    center_y_signed = torch.sum(rel_fc * v_lat, dim=-1)

    s_front = -0.5 * float(env.cfg.pallet_depth_m)
    insert_depth = torch.clamp(s_tip - s_front, min=0.0)
    insert_norm = torch.clamp(insert_depth / (float(env.cfg.pallet_depth_m) + 1e-6), 0.0, 1.0)
    dist_front = torch.clamp(s_front - s_tip, min=0.0)

    lift_height = tip[:, 2] - env._fork_tip_z0
    pallet_lift_height = pallet_pos[:, 2] - env.cfg.pallet_cfg.init_state.pos[2]
    z_err = torch.abs(lift_height - pallet_lift_height)
    valid_insert_z = z_err < env.cfg.max_insert_z_err

    hold_state = compute_hold_logic(
        center_y_err=torch.abs(center_y_signed),
        yaw_err_deg=yaw_err_deg,
        insert_depth=insert_depth,
        lift_height=lift_height,
        tip_y_err=torch.abs(tip_y_signed),
        dist_front=dist_front,
        hold_counter=env._hold_counter,
        cfg=env._hold_logic_cfg,
    )

    return {
        "insert_depth": insert_depth,
        "insert_norm": insert_norm,
        "center_y_signed": center_y_signed,
        "center_y_abs": torch.abs(center_y_signed),
        "tip_y_signed": tip_y_signed,
        "tip_y_abs": torch.abs(tip_y_signed),
        "yaw_err_signed_rad": yaw_err_signed,
        "yaw_err_deg_abs": yaw_err_deg,
        "hold_entry": hold_state.hold_entry,
        "align_entry": hold_state.align_entry,
        "tip_entry": hold_state.tip_entry,
        "insert_entry": hold_state.insert_entry,
        "valid_insert_z": valid_insert_z,
        "hold_counter": env._hold_counter.clone(),
    }


def build_actions(
    env: ForkliftPalletInsertLiftEnv,
    metrics: dict[str, torch.Tensor],
    drive_dir: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    actions = torch.zeros((env.num_envs, 3), device=env.device)
    next_drive_dir = drive_dir.clone()

    tip_target = float(env.cfg.tip_align_entry_m)
    center_target = float(env.cfg.max_lateral_err_m)
    yaw_target = float(env.cfg.max_yaw_err_deg)

    for env_id, variant in enumerate(VARIANTS):
        # 禁用 hold-freeze，隔离 pre-hold 可控性。
        env._last_hold_entry[env_id] = False

        insert_norm = float(metrics["insert_norm"][env_id].item())
        center_abs = float(metrics["center_y_abs"][env_id].item())
        tip_abs = float(metrics["tip_y_abs"][env_id].item())
        yaw_abs = float(metrics["yaw_err_deg_abs"][env_id].item())

        target_tip = max(tip_target - variant.hold_margin_tip, 0.02)
        hold_ready = (
            bool(metrics["insert_entry"][env_id].item())
            and bool(metrics["align_entry"][env_id].item())
            and bool(metrics["tip_entry"][env_id].item())
            and bool(metrics["valid_insert_z"][env_id].item())
        )

        if hold_ready:
            drive = 0.0
        else:
            if insert_norm > variant.corridor_high:
                next_drive_dir[env_id] = -1.0
            elif insert_norm < variant.corridor_low:
                next_drive_dir[env_id] = 1.0

            if tip_abs > target_tip:
                if insert_norm > variant.corridor_low:
                    drive = -variant.drive_rev
                    next_drive_dir[env_id] = -1.0
                elif insert_norm < max(variant.corridor_low - 0.05, 0.12):
                    drive = variant.drive_fwd * 0.65
                    next_drive_dir[env_id] = 1.0
                else:
                    drive = 0.0
            elif center_abs > center_target or yaw_abs > yaw_target:
                if insert_norm > variant.corridor_low:
                    drive = -variant.drive_rev * 0.65
                    next_drive_dir[env_id] = -1.0
                elif insert_norm < variant.corridor_high:
                    drive = variant.drive_fwd * 0.50
                    next_drive_dir[env_id] = 1.0
                else:
                    drive = 0.0
            else:
                if insert_norm < variant.corridor_high:
                    drive = variant.drive_fwd
                    next_drive_dir[env_id] = 1.0
                else:
                    drive = 0.0

        raw = (
            variant.k_center * metrics["center_y_signed"][env_id]
            + variant.k_tip * metrics["tip_y_signed"][env_id]
            + variant.k_yaw * metrics["yaw_err_signed_rad"][env_id]
        )
        if abs(drive) < 1e-6:
            steer = 0.0
        else:
            steer = float(torch.sign(torch.tensor(drive)).item()) * float(raw.item())
        steer = max(min(steer, 0.60), -0.60)

        actions[env_id, 0] = drive
        actions[env_id, 1] = steer

    return actions, next_drive_dir


def main() -> int:
    print("\n" + "=" * 108, flush=True)
    print("Focused pre-hold controller ablation: Case A", flush=True)
    print("=" * 108, flush=True)
    print(
        f"Case A: insert_depth={CASE_A.insert_depth_m:.2f}m, lateral={CASE_A.lateral_m:+.2f}m, yaw={CASE_A.yaw_deg:+.1f}deg",
        flush=True,
    )
    print(f"Controller variants: {len(VARIANTS)}", flush=True)

    env = ForkliftPalletInsertLiftEnv(build_env_cfg(len(VARIANTS)))
    try:
        env.reset()
        nominal_hold_steps = int(env._hold_steps)
        env._hold_steps = 10_000_000
        env.cfg.paper_out_of_bounds_dist = 1e6
        env.cfg.max_roll_pitch_rad = math.pi

        teleport_case(env, CASE_A)
        initial_metrics = compute_metrics(env)

        drive_dir = -torch.ones((env.num_envs,), device=env.device)
        best_center = initial_metrics["center_y_abs"].clone()
        best_tip = initial_metrics["tip_y_abs"].clone()
        best_yaw = initial_metrics["yaw_err_deg_abs"].clone()
        best_insert = initial_metrics["insert_norm"].clone()
        max_hold_counter = initial_metrics["hold_counter"].clone()
        ever_hold_entry = initial_metrics["hold_entry"].clone()
        first_hold_step = torch.full((env.num_envs,), -1, dtype=torch.long, device=env.device)
        first_tip_ok_step = torch.full((env.num_envs,), -1, dtype=torch.long, device=env.device)

        print(
            f"Thresholds: center <= {env.cfg.max_lateral_err_m:.2f}m, "
            f"tip <= {env.cfg.tip_align_entry_m:.2f}m, yaw <= {env.cfg.max_yaw_err_deg:.1f}deg",
            flush=True,
        )

        for step in range(args.steps):
            metrics = compute_metrics(env)
            actions, drive_dir = build_actions(env, metrics, drive_dir)
            env.step(actions)
            after = compute_metrics(env)

            best_center = torch.minimum(best_center, after["center_y_abs"])
            best_tip = torch.minimum(best_tip, after["tip_y_abs"])
            best_yaw = torch.minimum(best_yaw, after["yaw_err_deg_abs"])
            best_insert = torch.maximum(best_insert, after["insert_norm"])
            max_hold_counter = torch.maximum(max_hold_counter, after["hold_counter"])
            ever_hold_entry |= after["hold_entry"]

            tip_ok_now = after["tip_entry"] & after["insert_entry"] & after["align_entry"] & after["valid_insert_z"]
            hold_now = after["hold_entry"]
            new_tip_ok = (first_tip_ok_step < 0) & tip_ok_now
            new_hold = (first_hold_step < 0) & hold_now
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

        final_metrics = compute_metrics(env)

        print("\n" + "=" * 150, flush=True)
        print("Controller ablation results", flush=True)
        print("=" * 150, flush=True)
        print(
            f"{'Variant':>18} | {'center init->best->final':>26} | {'tip init->best->final':>23} | "
            f"{'yaw init->best->final':>23} | {'max ins':>7} | {'tip_ok':>6} | {'hold':>6} | {'first hit':>9}",
            flush=True,
        )
        print(
            f"{'-' * 18}-+-{'-' * 26}-+-{'-' * 23}-+-{'-' * 23}-+-{'-' * 7}-+-{'-' * 6}-+-{'-' * 6}-+-{'-' * 9}",
            flush=True,
        )

        results = []
        for env_id, variant in enumerate(VARIANTS):
            row = {
                "variant": variant,
                "init_center": float(initial_metrics["center_y_abs"][env_id].item()),
                "best_center": float(best_center[env_id].item()),
                "final_center": float(final_metrics["center_y_abs"][env_id].item()),
                "init_tip": float(initial_metrics["tip_y_abs"][env_id].item()),
                "best_tip": float(best_tip[env_id].item()),
                "final_tip": float(final_metrics["tip_y_abs"][env_id].item()),
                "init_yaw": float(initial_metrics["yaw_err_deg_abs"][env_id].item()),
                "best_yaw": float(best_yaw[env_id].item()),
                "final_yaw": float(final_metrics["yaw_err_deg_abs"][env_id].item()),
                "max_insert": float(best_insert[env_id].item()),
                "ever_tip_ok": bool(first_tip_ok_step[env_id].item() >= 0),
                "ever_hold": bool(ever_hold_entry[env_id].item()),
                "first_hit_step": int(first_hold_step[env_id].item() if first_hold_step[env_id].item() >= 0 else first_tip_ok_step[env_id].item()),
                "max_hold_counter": float(max_hold_counter[env_id].item()),
                "nominal_success": bool(max_hold_counter[env_id].item() >= nominal_hold_steps),
            }
            results.append(row)
            hit_text = str(row["first_hit_step"]) if row["first_hit_step"] >= 0 else "-"
            print(
                f"{variant.label:>18} | "
                f"{row['init_center']:.3f}->{row['best_center']:.3f}->{row['final_center']:.3f} | "
                f"{row['init_tip']:.3f}->{row['best_tip']:.3f}->{row['final_tip']:.3f} | "
                f"{row['init_yaw']:.2f}->{row['best_yaw']:.2f}->{row['final_yaw']:.2f} | "
                f"{row['max_insert']:.3f} | {str(row['ever_tip_ok']):>6} | {str(row['ever_hold']):>6} | {hit_text:>9}",
                flush=True,
            )

        best_by_tip = min(results, key=lambda item: item["best_tip"])
        best_by_hold = max(results, key=lambda item: (item["ever_hold"], -item["best_tip"]))

        print("\nSummary:", flush=True)
        print(
            f"  - Best tip reduction: {best_by_tip['variant'].label} "
            f"(tip {best_by_tip['init_tip']:.3f} -> {best_by_tip['best_tip']:.3f}, "
            f"center {best_by_tip['init_center']:.3f} -> {best_by_tip['best_center']:.3f}, "
            f"yaw {best_by_tip['init_yaw']:.2f} -> {best_by_tip['best_yaw']:.2f}).",
            flush=True,
        )
        if best_by_hold["ever_hold"]:
            print(
                f"  - Hold-capable variant exists: {best_by_hold['variant'].label} "
                f"(first hit step={best_by_hold['first_hit_step']}, max_hold_counter={best_by_hold['max_hold_counter']:.1f}).",
                flush=True,
            )
        else:
            print("  - No controller variant reached hold on Case A within this sweep.", flush=True)

        tip_target = float(env.cfg.tip_align_entry_m)
        strong_variants = [item for item in results if item["best_tip"] <= tip_target + 1e-6]
        if strong_variants:
            labels = ", ".join(item["variant"].label for item in strong_variants)
            print(f"  - Variants that crossed tip gate: {labels}", flush=True)
            print("  - Interpretation: Case A is rescuable with a stronger pre-hold controller; current loop is not using enough pull-out / tip-focused correction.", flush=True)
        else:
            print("  - No variant crossed the tip gate.", flush=True)
            print("  - Interpretation: even aggressive pull-out + tip/yaw focused control could not rescue Case A in this sweep.", flush=True)
            print("    This pushes the diagnosis closer to a real pre-hold controllability / physics bottleneck, not just weak shaping.", flush=True)

        print("\nController settings:", flush=True)
        for variant in VARIANTS:
            print(
                f"  - {variant.label}: corridor=[{variant.corridor_low:.2f}, {variant.corridor_high:.2f}], "
                f"drive_fwd={variant.drive_fwd:.2f}, drive_rev={variant.drive_rev:.2f}, "
                f"k_center={variant.k_center:.2f}, k_tip={variant.k_tip:.2f}, k_yaw={variant.k_yaw:.2f}",
                flush=True,
            )
        return 0
    finally:
        env.close()
        simulation_app.close()


if __name__ == "__main__":
    raise SystemExit(main())
