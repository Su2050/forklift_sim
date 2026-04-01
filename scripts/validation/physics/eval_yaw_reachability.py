#!/usr/bin/env python3
"""S1.0P Phase V1: Yaw 可达性曲线评估脚本。

将叉车初始化到 pre-insert 位置 (dist_front=0.25m, lateral=0.02m)，
设置不同 yaw 初值，低速直进，记录最大可达 insert_norm 和是否卡死。

产出物:
  - 控制台输出: yaw_init vs max_insert_norm 表格
  - 阈值点结论

Usage:
    isaaclab.sh -p ../scripts/validation/physics/eval_yaw_reachability.py --headless
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
ISAACLAB_ROOT = REPO_ROOT / "IsaacLab"
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


parser = argparse.ArgumentParser(description="V1: Yaw reachability curve evaluation")
parser.add_argument("--task", type=str, default="Isaac-Forklift-PalletInsertLift-Direct-v0")
parser.add_argument(
    "--num_envs",
    type=int,
    default=6,
    help="Requested number of envs. The script will override this to match yaw_angles.",
)
parser.add_argument("--max_steps", type=int, default=300, help="Max simulation steps per evaluation")
parser.add_argument("--drive_strength", type=float, default=0.3, help="Normalized drive action (0-1)")
parser.add_argument("--dist_front", type=float, default=0.25, help="Initial distance from fork tip to pallet front (m)")
parser.add_argument("--lateral", type=float, default=0.02, help="Initial lateral offset (m)")
parser.add_argument(
    "--yaw_angles",
    type=str,
    default="0.5,1.0,2.0,3.0,4.0,5.0",
    help="Comma-separated yaw angles in degrees to test",
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# 这个脚本只做纯物理 reachability，不需要相机；强制关掉可避免旧命令携带
# --enable_cameras 时把 Isaac Sim 启到更不稳定的 rendering 路径。
if getattr(args, "enable_cameras", False):
    print("[INFO] eval_yaw_reachability.py 不需要相机，已忽略 --enable_cameras。", flush=True)
    args.enable_cameras = False

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch

from isaaclab_tasks.direct.forklift_pallet_insert_lift.env import ForkliftPalletInsertLiftEnv
from isaaclab_tasks.direct.forklift_pallet_insert_lift.env_cfg import ForkliftPalletInsertLiftEnvCfg


def parse_yaw_angles(raw: str) -> list[float]:
    values = [token.strip() for token in raw.split(",")]
    angles = [float(token) for token in values if token]
    if not angles:
        raise ValueError("yaw_angles 不能为空")
    return angles


def build_env_cfg(num_envs: int) -> ForkliftPalletInsertLiftEnvCfg:
    cfg = ForkliftPalletInsertLiftEnvCfg()
    cfg.scene.num_envs = num_envs
    cfg.use_camera = False
    cfg.use_asymmetric_critic = False
    cfg.wait_for_textures = False
    cfg.episode_length_s = max(float(getattr(cfg, "episode_length_s", 0.0)), 3600.0)
    return cfg


def teleport_to_pre_insert(
    env: ForkliftPalletInsertLiftEnv,
    dist_front: float,
    lateral: float,
    yaw_angles_deg: list[float],
) -> None:
    """Teleport each env to pre-insert position with a specific yaw."""
    device = env.device
    num_envs = len(yaw_angles_deg)
    env_ids = torch.arange(num_envs, device=device, dtype=torch.long)

    pallet_pos = env.pallet.data.root_pos_w[0]
    pallet_depth = float(env.cfg.pallet_depth_m)
    fork_offset = float(env._fork_forward_offset)
    s_front = -0.5 * pallet_depth
    desired_s_tip = s_front - dist_front

    positions = []
    quats = []
    for yaw_deg in yaw_angles_deg:
        yaw_rad = math.radians(yaw_deg)
        tip_x = pallet_pos[0].item() + desired_s_tip
        root_x = tip_x - fork_offset * math.cos(yaw_rad)
        root_y = pallet_pos[1].item() + lateral - fork_offset * math.sin(yaw_rad)
        root_z = 0.03
        positions.append([root_x, root_y, root_z])

        half = yaw_rad * 0.5
        quats.append([math.cos(half), 0.0, 0.0, math.sin(half)])

    pos_tensor = torch.tensor(positions, device=device, dtype=torch.float32)
    quat_tensor = torch.tensor(quats, device=device, dtype=torch.float32)

    env._write_root_pose(env.robot, pos_tensor, quat_tensor, env_ids)

    zeros3 = torch.zeros((num_envs, 3), device=device)
    env._write_root_vel(env.robot, zeros3, zeros3, env_ids)

    joint_pos = env.robot.data.default_joint_pos[env_ids].clone()
    joint_vel = torch.zeros_like(joint_pos)
    env._write_joint_state(env.robot, joint_pos, joint_vel, env_ids)

    # 同步写回仿真，避免后续 step 读取到旧缓存。
    env.scene.write_data_to_sim()
    env.sim.reset()
    env.scene.update(env.cfg.sim.dt)
    env.robot.reset(env_ids)

    env.actions[env_ids] = 0.0
    if hasattr(env, "previous_actions"):
        env.previous_actions[env_ids] = 0.0
    env._fork_tip_z0[env_ids] = 0.03
    env._last_insert_depth[env_ids] = 0.0
    env._hold_counter[env_ids] = 0
    env._is_first_step[env_ids] = True
    env._lift_pos_target[env_ids] = 0.0
    env._milestone_flags[env_ids] = False
    env._fly_counter[env_ids] = 0
    env._stall_counter[env_ids] = 0
    env._early_stop_fly[env_ids] = False
    env._early_stop_stall[env_ids] = False
    env._prev_phi_align[env_ids] = 0.0
    env._prev_phi_lift_progress[env_ids] = 0.0
    if hasattr(env, "_prev_insert_norm"):
        env._prev_insert_norm[env_ids] = 0.0
    if hasattr(env, "_prev_in_dead_zone"):
        env._prev_in_dead_zone[env_ids] = False
    if hasattr(env, "_prev_phi_lat"):
        env._prev_phi_lat[env_ids] = 0.0
    if hasattr(env, "_global_stall_counter"):
        env._global_stall_counter[env_ids] = 0
    env.episode_length_buf[env_ids] = 0

    print(f"已将 {num_envs} 个环境传送到 pre-insert 位置:", flush=True)
    for index, yaw_deg in enumerate(yaw_angles_deg):
        print(
            f"  env[{index}]: yaw={yaw_deg:5.1f}°, "
            f"root=({positions[index][0]:.3f}, {positions[index][1]:.3f})",
            flush=True,
        )


def compute_insert_norm(env: ForkliftPalletInsertLiftEnv) -> torch.Tensor:
    tip = env._compute_fork_tip()
    pallet_pos = env.pallet.data.root_pos_w
    pallet_quat = env.pallet.data.root_quat_w
    w_p, x_p, y_p, z_p = pallet_quat.unbind(-1)
    pallet_yaw = torch.atan2(2.0 * (w_p * z_p + x_p * y_p), 1.0 - 2.0 * (y_p * y_p + z_p * z_p))
    cp = torch.cos(pallet_yaw)
    sp = torch.sin(pallet_yaw)
    u_in = torch.stack([cp, sp], dim=-1)

    rel_tip = tip[:, :2] - pallet_pos[:, :2]
    s_tip = torch.sum(rel_tip * u_in, dim=-1)
    s_front = -0.5 * env.cfg.pallet_depth_m
    insert_depth = torch.clamp(s_tip - s_front, min=0.0)
    return torch.clamp(insert_depth / (env.cfg.pallet_depth_m + 1e-6), 0.0, 1.0)


def main() -> int:
    yaw_angles_deg = parse_yaw_angles(args.yaw_angles)
    actual_num_envs = len(yaw_angles_deg)

    if args.num_envs != actual_num_envs:
        print(
            f"[INFO] num_envs={args.num_envs} 与 yaw_angles 数量不一致，"
            f"已自动改为 {actual_num_envs}。",
            flush=True,
        )

    print(f"\n{'=' * 70}", flush=True)
    print("S1.0P Phase V1: Yaw 可达性曲线评估", flush=True)
    print(f"{'=' * 70}", flush=True)
    print(f"测试 yaw 角度: {yaw_angles_deg}°", flush=True)
    print(f"初始距离: dist_front={args.dist_front}m, lateral={args.lateral}m", flush=True)
    print(f"驱动强度: {args.drive_strength}, 最大步数: {args.max_steps}", flush=True)
    print(f"{'=' * 70}\n", flush=True)

    env = ForkliftPalletInsertLiftEnv(build_env_cfg(actual_num_envs))
    try:
        env.reset()
        teleport_to_pre_insert(env, args.dist_front, args.lateral, yaw_angles_deg)

        max_insert_norm = torch.zeros(actual_num_envs, device=env.device)
        max_insert_step = torch.zeros(actual_num_envs, dtype=torch.long, device=env.device)
        stuck_flags = torch.zeros(actual_num_envs, dtype=torch.bool, device=env.device)
        collision_flags = torch.zeros(actual_num_envs, dtype=torch.bool, device=env.device)
        prev_insert_norm = torch.zeros(actual_num_envs, device=env.device)
        no_progress_count = torch.zeros(actual_num_envs, dtype=torch.long, device=env.device)

        action = torch.zeros((actual_num_envs, 3), device=env.device)
        action[:, 0] = args.drive_strength

        print(f"\n开始直进测试 ({args.max_steps} 步)...\n", flush=True)
        for step in range(args.max_steps):
            _, _, terminated, truncated, _ = env.step(action)

            insert_norm = compute_insert_norm(env)

            improved = insert_norm > max_insert_norm
            max_insert_norm = torch.where(improved, insert_norm, max_insert_norm)
            max_insert_step = torch.where(
                improved,
                torch.full_like(max_insert_step, step),
                max_insert_step,
            )

            progress = insert_norm - prev_insert_norm
            no_progress_count = torch.where(
                progress < 0.001,
                no_progress_count + 1,
                torch.zeros_like(no_progress_count),
            )
            stuck_flags = stuck_flags | (no_progress_count > 50)
            prev_insert_norm = insert_norm.clone()
            collision_flags = collision_flags | terminated | truncated

            if (step + 1) % 50 == 0:
                values = ", ".join(f"{value:.4f}" for value in insert_norm.tolist())
                print(f"  Step {step + 1:3d}: insert_norm = [{values}]", flush=True)

        print(f"\n{'=' * 70}", flush=True)
        print("结果: Yaw 初值 vs 最大插入深度", flush=True)
        print(f"{'=' * 70}", flush=True)
        print(
            f"{'Yaw (°)':>8} | {'Max Insert Norm':>15} | {'Max Step':>9} | {'Stuck':>6} | {'Collision':>10}",
            flush=True,
        )
        print(f"{'-' * 8}-+-{'-' * 15}-+-{'-' * 9}-+-{'-' * 6}-+-{'-' * 10}", flush=True)

        for index, yaw_deg in enumerate(yaw_angles_deg):
            ins = max_insert_norm[index].item()
            max_step = max_insert_step[index].item()
            stuck = "Yes" if stuck_flags[index].item() else "No"
            collision = "Yes" if collision_flags[index].item() else "No"
            print(f"{yaw_deg:8.1f} | {ins:15.4f} | {max_step:9d} | {stuck:>6} | {collision:>10}", flush=True)

        print(f"\n{'=' * 70}", flush=True)
        print("阈值分析", flush=True)
        print(f"{'=' * 70}", flush=True)
        thresholds = [0.5, 0.3, 0.1, 0.05]
        insert_values = max_insert_norm.tolist()
        for threshold in thresholds:
            above = [index for index, value in enumerate(insert_values) if value >= threshold]
            if above:
                max_yaw = yaw_angles_deg[above[-1]]
                print(f"  insert_norm >= {threshold:.2f}: 最大可用 yaw = {max_yaw}°", flush=True)
            else:
                print(f"  insert_norm >= {threshold:.2f}: 所有 yaw 均不可达", flush=True)

        best_idx = max_insert_norm.argmax().item()
        worst_idx = max_insert_norm.argmin().item()
        print("\n结论:", flush=True)
        print(
            f"  最佳: yaw={yaw_angles_deg[best_idx]}° → insert_norm={max_insert_norm[best_idx]:.4f}",
            flush=True,
        )
        print(
            f"  最差: yaw={yaw_angles_deg[worst_idx]}° → insert_norm={max_insert_norm[worst_idx]:.4f}",
            flush=True,
        )

        worst_value = max_insert_norm[worst_idx].item()
        if worst_value > 0.3:
            print(f"\n  判断: yaw={yaw_angles_deg[-1]}° 仍能深插 (>0.3)", flush=True)
            print("  => 碰撞/几何太宽松，1° 目标无物理约束支撑", flush=True)
        elif worst_value < 0.05:
            print(f"\n  判断: yaw={yaw_angles_deg[worst_idx]}° 基本无法插入", flush=True)
            for index, value in enumerate(insert_values):
                if value < 0.1:
                    print(
                        f"  => yaw >= {yaw_angles_deg[index]}° 明显卡死，精度目标有物理约束支撑",
                        flush=True,
                    )
                    break
        else:
            baseline = max_insert_norm[0].item()
            for index in range(1, len(max_insert_norm)):
                if max_insert_norm[index].item() < baseline * 0.5:
                    print(
                        f"\n  判断: yaw={yaw_angles_deg[index]}° 时插入深度显著衰减 "
                        f"(降至 {baseline * 0.5:.2f} 以下)",
                        flush=True,
                    )
                    print(
                        f"  => 精度目标在 {yaw_angles_deg[index - 1]}°~{yaw_angles_deg[index]}° 之间存在物理约束",
                        flush=True,
                    )
                    break

        print(f"\n{'=' * 70}", flush=True)
        return 0
    finally:
        env.close()
        simulation_app.close()


if __name__ == "__main__":
    raise SystemExit(main())
