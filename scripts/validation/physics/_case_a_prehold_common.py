from __future__ import annotations

import math
from dataclasses import dataclass

import torch

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


PREHOLD_VARIANTS = [
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


def quat_to_yaw(quat: torch.Tensor) -> torch.Tensor:
    w, x, y, z = quat.unbind(-1)
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def reset_internal_buffers(env, env_ids: torch.Tensor) -> None:
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


def teleport_case(env, case: CaseSpec, env_ids: torch.Tensor | None = None) -> None:
    device = env.device
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=device, dtype=torch.long)

    pallet_pos = env.pallet.data.root_pos_w[env_ids].clone()
    pallet_quat = env.pallet.data.root_quat_w[env_ids].clone()
    pallet_yaw = quat_to_yaw(pallet_quat)
    fork_offset = float(env._fork_forward_offset)
    s_front = -0.5 * float(env.cfg.pallet_depth_m)

    pos_rows = []
    quat_rows = []
    for index in range(len(env_ids)):
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
    zeros3 = torch.zeros((len(env_ids), 3), device=device)
    env._write_root_vel(env.robot, zeros3, zeros3, env_ids)
    joint_pos = env.robot.data.default_joint_pos[env_ids].clone()
    joint_vel = torch.zeros_like(joint_pos)
    env._write_joint_state(env.robot, joint_pos, joint_vel, env_ids)

    env.scene.write_data_to_sim()
    env.sim.step(render=False)
    env.scene.update(env.cfg.sim.dt)

    env._joint_pos[:] = env.robot.root_physx_view.get_dof_positions()
    env._joint_vel[:] = env.robot.root_physx_view.get_dof_velocities()
    reset_internal_buffers(env, env_ids)


def compute_metrics(env) -> dict[str, torch.Tensor]:
    env._joint_pos[:] = env.robot.root_physx_view.get_dof_positions()
    env._joint_vel[:] = env.robot.root_physx_view.get_dof_velocities()

    root_pos = env.robot.data.root_pos_w
    root_quat = env.robot.data.root_quat_w
    pallet_pos = env.pallet.data.root_pos_w
    pallet_quat = env.pallet.data.root_quat_w
    tip = env._compute_fork_tip()
    fork_center = env._compute_fork_center()

    pallet_yaw = quat_to_yaw(pallet_quat)
    robot_yaw = quat_to_yaw(root_quat)
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
        "root_y": root_pos[:, 1].clone(),
        "tip_world": tip.clone(),
        "fork_center_world": fork_center.clone(),
        "robot_yaw": robot_yaw.clone(),
        "pallet_yaw": pallet_yaw.clone(),
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


def build_actions(env, metrics: dict[str, torch.Tensor], drive_dir: torch.Tensor, variants: list[ControllerVariant]) -> tuple[torch.Tensor, torch.Tensor]:
    actions = torch.zeros((env.num_envs, 3), device=env.device)
    next_drive_dir = drive_dir.clone()

    tip_target = float(env.cfg.tip_align_entry_m)
    center_target = float(env.cfg.max_lateral_err_m)
    yaw_target = float(env.cfg.max_yaw_err_deg)

    for env_id, variant in enumerate(variants):
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
