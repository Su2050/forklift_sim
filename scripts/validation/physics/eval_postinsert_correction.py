#!/usr/bin/env python3
"""Post-insert correction diagnostic with freeze ablations.

这个脚本回答两个问题：

1. 当前代码是否会在“刚插入”时就把 drive/steer 过早冻结？
2. 对于“已插入但偏心/偏航”的固定 case，是否还能在孔内继续纠偏？

实现方式：
- Phase A：动作透传探针
  同一个 inserted-but-not-hold case，对比三种冻结策略：
  - current_hold_freeze: 当前环境真实行为（仅上一拍 hold_entry 才冻结）
  - freeze_on_insert: 人为 ablation，insert_entry 后立刻冻结 drive/steer
  - no_freeze: 人为 ablation，禁止 hold_entry 触发冻结
- Phase B：固定 case 纠偏诊断
  多个 post-insert case 上运行简单闭环控制器，观察误差是否下降、hold_entry 是否出现、
  以及不同冻结策略的差异。

Usage:
    ISAACLAB_DIR=/home/uniubi/projects/forklift_sim/IsaacLab \
    ./isaaclab.sh -p /home/uniubi/projects/forklift_sim_exp9/scripts/validation/physics/eval_postinsert_correction.py --headless
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

parser = argparse.ArgumentParser(description="Fixed-case post-insert correction diagnostic")
parser.add_argument("--task", type=str, default="Isaac-Forklift-PalletInsertLift-Direct-v0")
parser.add_argument("--probe_steps", type=int, default=24, help="Steps for the pass-through probe.")
parser.add_argument("--controller_steps", type=int, default=180, help="Steps for the closed-loop correction run.")
parser.add_argument(
    "--probe_drive",
    type=float,
    default=-0.18,
    help="Normalized drive action for the pass-through probe.",
)
parser.add_argument(
    "--probe_steer",
    type=float,
    default=0.35,
    help="Absolute normalized steer magnitude for the pass-through probe.",
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

if getattr(args, "enable_cameras", False):
    print("[INFO] eval_postinsert_correction.py 不需要相机，已忽略 --enable_cameras。", flush=True)
    args.enable_cameras = False

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

from isaaclab_tasks.direct.forklift_pallet_insert_lift.env import ForkliftPalletInsertLiftEnv
from isaaclab_tasks.direct.forklift_pallet_insert_lift.env_cfg import ForkliftPalletInsertLiftEnvCfg
from isaaclab_tasks.direct.forklift_pallet_insert_lift.hold_logic import compute_hold_logic


@dataclass(frozen=True)
class FreezeRegime:
    name: str
    label: str
    synthetic_insert_freeze: bool
    disable_hold_freeze: bool


@dataclass(frozen=True)
class CorrectionCase:
    name: str
    label: str
    insert_depth_m: float
    lateral_m: float
    yaw_deg: float


REGIMES = [
    FreezeRegime(
        name="current_hold_freeze",
        label="Current (hold freeze)",
        synthetic_insert_freeze=False,
        disable_hold_freeze=False,
    ),
    FreezeRegime(
        name="freeze_on_insert",
        label="Ablation (freeze on insert)",
        synthetic_insert_freeze=True,
        disable_hold_freeze=False,
    ),
    FreezeRegime(
        name="no_freeze",
        label="Ablation (never freeze)",
        synthetic_insert_freeze=False,
        disable_hold_freeze=True,
    ),
]


PROBE_CASE = CorrectionCase(
    name="probe_lateral",
    label="Probe: inserted + lateral bias",
    insert_depth_m=1.00,
    lateral_m=0.18,
    yaw_deg=4.0,
)

CORRECTION_CASES = [
    CorrectionCase(
        name="lateral_bias",
        label="Case A: deeper insert + lateral bias",
        insert_depth_m=1.00,
        lateral_m=0.18,
        yaw_deg=4.0,
    ),
    CorrectionCase(
        name="yaw_bias",
        label="Case B: deeper insert + yaw bias",
        insert_depth_m=1.00,
        lateral_m=0.05,
        yaw_deg=9.0,
    ),
    CorrectionCase(
        name="near_hold",
        label="Case C: near-hold but not stable",
        insert_depth_m=0.94,
        lateral_m=0.11,
        yaw_deg=7.0,
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


def make_env(num_envs: int) -> tuple[ForkliftPalletInsertLiftEnv, int]:
    env = ForkliftPalletInsertLiftEnv(build_env_cfg(num_envs))
    env.reset()
    nominal_hold_steps = int(env._hold_steps)
    env._hold_steps = 10_000_000
    env.cfg.paper_out_of_bounds_dist = 1e6
    env.cfg.max_roll_pitch_rad = math.pi
    return env, nominal_hold_steps


def _quat_to_yaw(quat: torch.Tensor) -> torch.Tensor:
    w, x, y, z = quat.unbind(-1)
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _yaw_to_quat(yaw: torch.Tensor) -> torch.Tensor:
    half = yaw * 0.5
    return torch.stack(
        [torch.cos(half), torch.zeros_like(half), torch.zeros_like(half), torch.sin(half)],
        dim=-1,
    )


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


def teleport_cases(
    env: ForkliftPalletInsertLiftEnv,
    env_ids: torch.Tensor,
    cases: list[CorrectionCase],
) -> None:
    device = env.device

    pallet_pos = env.pallet.data.root_pos_w[env_ids].clone()
    pallet_quat = env.pallet.data.root_quat_w[env_ids].clone()
    pallet_yaw = _quat_to_yaw(pallet_quat)
    fork_offset = float(env._fork_forward_offset)
    s_front = -0.5 * float(env.cfg.pallet_depth_m)

    pos_rows = []
    quat_rows = []
    joint_pos = env.robot.data.default_joint_pos[env_ids].clone()
    joint_vel = torch.zeros_like(joint_pos)

    for index, case in enumerate(cases):
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
    zeros3 = torch.zeros((len(cases), 3), device=device)
    env._write_root_vel(env.robot, zeros3, zeros3, env_ids)
    env._write_joint_state(env.robot, joint_pos, joint_vel, env_ids)

    env.scene.write_data_to_sim()
    env.sim.step(render=False)
    env.scene.update(env.cfg.sim.dt)

    env._joint_pos[:] = env.robot.root_physx_view.get_dof_positions()
    env._joint_vel[:] = env.robot.root_physx_view.get_dof_velocities()
    _reset_internal_buffers(env, env_ids)


def slice_metrics(metrics: dict[str, torch.Tensor], env_ids: torch.Tensor) -> dict[str, torch.Tensor]:
    return {key: value[env_ids] for key, value in metrics.items()}


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
        "root_pos": root_pos.clone(),
        "robot_yaw": robot_yaw.clone(),
        "insert_depth": insert_depth,
        "insert_norm": insert_norm,
        "dist_front": dist_front,
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


def build_probe_actions(
    metrics: dict[str, torch.Tensor],
) -> torch.Tensor:
    num_envs = len(metrics["insert_norm"])
    actions = torch.zeros((num_envs, 3), device=metrics["insert_norm"].device)
    drive = torch.full((num_envs,), args.probe_drive, device=metrics["insert_norm"].device)
    raw = metrics["center_y_signed"] + 0.35 * metrics["yaw_err_signed_rad"]
    steer_sign = torch.sign(raw)
    steer_sign = torch.where(steer_sign == 0, torch.ones_like(steer_sign), steer_sign)
    steer = torch.sign(drive) * steer_sign * args.probe_steer
    actions[:, 0] = drive
    actions[:, 1] = steer
    return actions


def build_controller_actions(
    metrics: dict[str, torch.Tensor],
    drive_direction: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    corridor_low = 0.44
    corridor_high = 0.58
    drive_direction = torch.where(
        metrics["insert_norm"] >= corridor_high,
        -torch.ones_like(drive_direction),
        drive_direction,
    )
    drive_direction = torch.where(
        metrics["insert_norm"] <= corridor_low,
        torch.ones_like(drive_direction),
        drive_direction,
    )

    need_correction = (
        (metrics["center_y_abs"] > 0.04)
        | (metrics["tip_y_abs"] > 0.04)
        | (metrics["yaw_err_deg_abs"] > 1.5)
    )

    drive = torch.where(
        need_correction,
        torch.where(drive_direction > 0, torch.full_like(drive_direction, 0.18), torch.full_like(drive_direction, -0.16)),
        torch.zeros_like(drive_direction),
    )

    raw = (
        3.4 * metrics["center_y_signed"]
        + 1.6 * metrics["tip_y_signed"]
        + 0.35 * metrics["yaw_err_signed_rad"]
    )
    steer = torch.sign(drive) * raw
    steer = torch.clamp(steer, -0.55, 0.55)
    steer = torch.where(torch.abs(drive) > 1e-6, steer, torch.zeros_like(steer))

    actions = torch.zeros((len(drive), 3), device=drive.device)
    actions[:, 0] = drive
    actions[:, 1] = steer
    return actions, drive_direction


def apply_regime_overrides(
    env: ForkliftPalletInsertLiftEnv,
    env_ids: torch.Tensor,
    metrics: dict[str, torch.Tensor],
    actions: torch.Tensor,
    regimes: list[FreezeRegime],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    actions = actions.clone()
    external_mask = torch.zeros((len(regimes),), device=env.device, dtype=torch.bool)
    internal_mask = torch.zeros((len(regimes),), device=env.device, dtype=torch.bool)

    for local_id, regime in enumerate(regimes):
        env_id = int(env_ids[local_id].item())
        if regime.disable_hold_freeze:
            env._last_hold_entry[env_id] = False
        else:
            internal_mask[local_id] = bool(env._last_hold_entry[env_id].item())

        if regime.synthetic_insert_freeze and bool(metrics["insert_entry"][local_id].item()):
            actions[local_id, :2] = 0.0
            external_mask[local_id] = True

    return actions, external_mask, internal_mask


def run_freeze_probe(
    env: ForkliftPalletInsertLiftEnv,
    probe_env_ids: torch.Tensor,
) -> dict[str, dict[str, float | bool]]:
    teleport_cases(env, probe_env_ids, [PROBE_CASE] * len(REGIMES))
    initial_metrics = slice_metrics(compute_metrics(env), probe_env_ids)
    initial_root = initial_metrics["root_pos"][:, :2].clone()
    initial_yaw = initial_metrics["robot_yaw"].clone()

    external_mask_steps = torch.zeros(len(REGIMES), device=env.device)
    internal_mask_steps = torch.zeros(len(REGIMES), device=env.device)

    probe_actions = build_probe_actions(initial_metrics)

    for _ in range(args.probe_steps):
        current_metrics = slice_metrics(compute_metrics(env), probe_env_ids)
        local_actions, external_mask, internal_mask = apply_regime_overrides(
            env, probe_env_ids, current_metrics, probe_actions, REGIMES
        )
        external_mask_steps += external_mask.float()
        internal_mask_steps += internal_mask.float()

        full_actions = torch.zeros((env.num_envs, 3), device=env.device)
        full_actions[probe_env_ids] = local_actions
        env.step(full_actions)

    final_metrics = slice_metrics(compute_metrics(env), probe_env_ids)
    final_root = final_metrics["root_pos"][:, :2]
    final_yaw = final_metrics["robot_yaw"]

    results: dict[str, dict[str, float | bool]] = {}
    print("\n" + "=" * 88, flush=True)
    print("Phase A: 动作透传探针（固定 inserted case）", flush=True)
    print("=" * 88, flush=True)
    print(
        f"{'Regime':>28} | {'root disp':>9} | {'|Δyaw|':>8} | {'hold hit':>8} | {'ext mask':>8} | {'int mask':>8}",
        flush=True,
    )
    print(f"{'-' * 28}-+-{'-' * 9}-+-{'-' * 8}-+-{'-' * 8}-+-{'-' * 8}-+-{'-' * 8}", flush=True)
    for local_id, regime in enumerate(REGIMES):
        root_disp = torch.linalg.norm(final_root[local_id] - initial_root[local_id]).item()
        yaw_delta = abs(math.degrees((final_yaw[local_id] - initial_yaw[local_id]).item()))
        hold_hit = bool(final_metrics["hold_entry"][local_id].item())
        results[regime.name] = {
            "root_disp": root_disp,
            "yaw_delta_deg": yaw_delta,
            "hold_hit": hold_hit,
            "external_mask_steps": external_mask_steps[local_id].item(),
            "internal_mask_steps": internal_mask_steps[local_id].item(),
        }
        print(
            f"{regime.label:>28} | {root_disp:9.4f} | {yaw_delta:8.2f} | "
            f"{str(hold_hit):>8} | {external_mask_steps[local_id].item():8.0f} | {internal_mask_steps[local_id].item():8.0f}",
            flush=True,
        )

    print("\nProbe interpretation:", flush=True)
    current_motion = results["current_hold_freeze"]["root_disp"]
    insert_freeze_motion = results["freeze_on_insert"]["root_disp"]
    no_freeze_motion = results["no_freeze"]["root_disp"]
    if current_motion > 0.05 and insert_freeze_motion < 0.005:
        print("  - 当前代码在 inserted-but-not-hold case 上仍能透传 drive/steer，不是 insert_entry 就冻结。", flush=True)
    else:
        print("  - 当前代码在该 probe 上未表现出明显的 post-insert 可动性，需结合 Phase B 继续看。", flush=True)
    if abs(current_motion - no_freeze_motion) < 0.03:
        print("  - current 与 no-freeze 几乎一致，说明当前冻结逻辑至少没有在这个阶段提前介入。", flush=True)
    else:
        print("  - current 与 no-freeze 存在可见差异，说明 hold 阶段冻结仍可能影响最后微调。", flush=True)
    return results


def run_correction_matrix(
    env: ForkliftPalletInsertLiftEnv,
    combo_env_ids: torch.Tensor,
    nominal_hold_steps: int,
) -> dict[str, dict[str, dict[str, float | bool]]]:
    combos: list[tuple[CorrectionCase, FreezeRegime]] = []
    for case in CORRECTION_CASES:
        for regime in REGIMES:
            combos.append((case, regime))

    teleport_cases(env, combo_env_ids, [case for case, _ in combos])
    initial_metrics = slice_metrics(compute_metrics(env), combo_env_ids)
    drive_direction = -torch.ones((len(combos),), device=env.device)

    best_center = initial_metrics["center_y_abs"].clone()
    best_tip = initial_metrics["tip_y_abs"].clone()
    best_yaw = initial_metrics["yaw_err_deg_abs"].clone()
    max_hold_counter = initial_metrics["hold_counter"].clone()
    ever_hold_entry = initial_metrics["hold_entry"].clone()
    external_mask_steps = torch.zeros(len(combos), device=env.device)
    internal_mask_steps = torch.zeros(len(combos), device=env.device)

    for _ in range(args.controller_steps):
        current_metrics = slice_metrics(compute_metrics(env), combo_env_ids)
        local_actions, drive_direction = build_controller_actions(current_metrics, drive_direction)
        step_actions, external_mask, internal_mask = apply_regime_overrides(
            env,
            combo_env_ids,
            current_metrics,
            local_actions,
            [regime for _, regime in combos],
        )
        external_mask_steps += external_mask.float()
        internal_mask_steps += internal_mask.float()

        full_actions = torch.zeros((env.num_envs, 3), device=env.device)
        full_actions[combo_env_ids] = step_actions
        env.step(full_actions)

        after_metrics = slice_metrics(compute_metrics(env), combo_env_ids)
        best_center = torch.minimum(best_center, after_metrics["center_y_abs"])
        best_tip = torch.minimum(best_tip, after_metrics["tip_y_abs"])
        best_yaw = torch.minimum(best_yaw, after_metrics["yaw_err_deg_abs"])
        max_hold_counter = torch.maximum(max_hold_counter, after_metrics["hold_counter"])
        ever_hold_entry |= after_metrics["hold_entry"]

    final_metrics = slice_metrics(compute_metrics(env), combo_env_ids)
    grouped: dict[str, dict[str, dict[str, float | bool]]] = {}

    print("\n" + "=" * 120, flush=True)
    print("Phase B: 固定 case 纠偏诊断（闭环控制器）", flush=True)
    print("=" * 120, flush=True)
    for case in CORRECTION_CASES:
        print(f"\n[{case.label}]", flush=True)
        print(
            f"{'Regime':>28} | {'center: init->best->final':>27} | {'tip: init->best->final':>24} | "
            f"{'yaw: init->best->final':>24} | {'hold max':>8} | {'hold hit':>8} | {'ext/int mask':>12}",
            flush=True,
        )
        print(
            f"{'-' * 28}-+-{'-' * 27}-+-{'-' * 24}-+-{'-' * 24}-+-{'-' * 8}-+-{'-' * 8}-+-{'-' * 12}",
            flush=True,
        )
        grouped[case.name] = {}
        for env_id, (combo_case, regime) in enumerate(combos):
            if combo_case.name != case.name:
                continue
            init_center = initial_metrics["center_y_abs"][env_id].item()
            init_tip = initial_metrics["tip_y_abs"][env_id].item()
            init_yaw = initial_metrics["yaw_err_deg_abs"][env_id].item()
            result = {
                "init_center": init_center,
                "best_center": best_center[env_id].item(),
                "final_center": final_metrics["center_y_abs"][env_id].item(),
                "init_tip": init_tip,
                "best_tip": best_tip[env_id].item(),
                "final_tip": final_metrics["tip_y_abs"][env_id].item(),
                "init_yaw": init_yaw,
                "best_yaw": best_yaw[env_id].item(),
                "final_yaw": final_metrics["yaw_err_deg_abs"][env_id].item(),
                "max_hold_counter": max_hold_counter[env_id].item(),
                "nominal_success": bool(max_hold_counter[env_id].item() >= nominal_hold_steps),
                "ever_hold_entry": bool(ever_hold_entry[env_id].item()),
                "external_mask_steps": external_mask_steps[env_id].item(),
                "internal_mask_steps": internal_mask_steps[env_id].item(),
            }
            grouped[case.name][regime.name] = result
            print(
                f"{regime.label:>28} | "
                f"{result['init_center']:.3f}->{result['best_center']:.3f}->{result['final_center']:.3f} | "
                f"{result['init_tip']:.3f}->{result['best_tip']:.3f}->{result['final_tip']:.3f} | "
                f"{result['init_yaw']:.2f}->{result['best_yaw']:.2f}->{result['final_yaw']:.2f} | "
                f"{result['max_hold_counter']:8.1f} | {str(result['ever_hold_entry']):>8} | "
                f"{result['external_mask_steps']:.0f}/{result['internal_mask_steps']:.0f}",
                flush=True,
            )

        current = grouped[case.name]["current_hold_freeze"]
        insert_freeze = grouped[case.name]["freeze_on_insert"]
        no_freeze = grouped[case.name]["no_freeze"]
        print("  Summary:", flush=True)
        if current["best_center"] + 1e-6 < insert_freeze["best_center"]:
            print("    - current 比 insert-freeze 更能压低 center error，说明“按 insert 冻结”会明显伤害纠偏。", flush=True)
        else:
            print("    - current 与 insert-freeze 在 center error 上差异不大。", flush=True)
        if abs(current["best_center"] - no_freeze["best_center"]) < 0.02 and abs(current["best_yaw"] - no_freeze["best_yaw"]) < 1.0:
            print("    - current 与 no-freeze 很接近，冻结逻辑不是这个 case 的主限制。", flush=True)
        else:
            print("    - no-freeze 明显优于/不同于 current，hold_entry 冻结可能仍影响最后微调。", flush=True)
        if not current["ever_hold_entry"] and not no_freeze["ever_hold_entry"]:
            print("    - 两者都没真正进 hold，瓶颈更像控制/物理可纠偏性，而不是冻结。", flush=True)

    return grouped


def main() -> int:
    print("\n" + "=" * 100, flush=True)
    print("Post-insert correction diagnostic", flush=True)
    print("=" * 100, flush=True)
    print(f"IsaacLab root: {ISAACLAB_ROOT}", flush=True)
    print(f"Probe case: {PROBE_CASE.label}", flush=True)
    print("Correction cases:", flush=True)
    for case in CORRECTION_CASES:
        print(
            f"  - {case.label}: insert_depth={case.insert_depth_m:.2f}m, lateral={case.lateral_m:+.2f}m, yaw={case.yaw_deg:+.1f}deg",
            flush=True,
        )
    print("Freeze regimes:", flush=True)
    for regime in REGIMES:
        print(f"  - {regime.label}", flush=True)

    total_envs = len(CORRECTION_CASES) * len(REGIMES)
    env, nominal_hold_steps = make_env(total_envs)
    try:
        probe_env_ids = torch.arange(len(REGIMES), device=env.device, dtype=torch.long)
        combo_env_ids = torch.arange(total_envs, device=env.device, dtype=torch.long)
        probe_results = run_freeze_probe(env, probe_env_ids)
        correction_results = run_correction_matrix(env, combo_env_ids, nominal_hold_steps)
    finally:
        env.close()

    print("\n" + "=" * 100, flush=True)
    print("Overall conclusion", flush=True)
    print("=" * 100, flush=True)
    current_probe = probe_results["current_hold_freeze"]
    insert_probe = probe_results["freeze_on_insert"]
    no_freeze_probe = probe_results["no_freeze"]
    if current_probe["root_disp"] > 0.05 and insert_probe["root_disp"] < 0.005:
        print("1. 当前主线代码没有在 insert_entry 时冻结 drive/steer；人为改成 insert-freeze 会立即让 inserted case 基本失去动作透传。", flush=True)
    else:
        print("1. 这个 probe 没有给出足够强的动作透传对比，需要结合 Phase B 表格继续判断。", flush=True)
    if abs(current_probe["root_disp"] - no_freeze_probe["root_disp"]) < 0.03:
        print("2. current 与 no-freeze 在 probe 上接近，说明至少在 inserted-but-not-hold 阶段，冻结不是当前主矛盾。", flush=True)
    else:
        print("2. no-freeze 在 probe 上明显更活跃，hold_entry 冻结仍值得继续盯。", flush=True)

    for case in CORRECTION_CASES:
        current = correction_results[case.name]["current_hold_freeze"]
        insert_freeze = correction_results[case.name]["freeze_on_insert"]
        no_freeze = correction_results[case.name]["no_freeze"]
        center_gain_vs_insert = insert_freeze["best_center"] - current["best_center"]
        center_gap_vs_no = current["best_center"] - no_freeze["best_center"]
        yaw_gap_vs_no = current["best_yaw"] - no_freeze["best_yaw"]
        print(
            f"3. {case.label}: current 相对 insert-freeze 的 best-center 改善={center_gain_vs_insert:+.3f}m，"
            f"相对 no-freeze 的差距 center={center_gap_vs_no:+.3f}m / yaw={yaw_gap_vs_no:+.2f}deg。",
            flush=True,
        )

    print("\nInterpretation guide:", flush=True)
    print("  - 如果 current 明显优于 freeze_on_insert：说明“按插入立刻冻结”会伤害纠偏，而当前代码已避免这一点。", flush=True)
    print("  - 如果 current 仍接近 no-freeze：说明主要瓶颈在控制/可达性，不在冻结逻辑。", flush=True)
    print("  - 如果 no-freeze 明显优于 current：说明 hold_entry 冻结可能仍会抑制最后几步微调。", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        simulation_app.close()
