"""S1.0P Phase V1: Yaw 可达性曲线评估脚本。

将叉车初始化到 pre-insert 位置 (dist_front=0.25m, lateral=0.02m)，
设置不同 yaw 初值 (0.5°/1°/2°/3°/4°/5°)，低速直进，
记录最大可达 insert_norm 和是否卡死。

产出物:
  - 控制台输出: yaw_init vs max_insert_norm 表格
  - 阈值点结论

Usage:
    isaaclab.sh -p scripts/eval_yaw_reachability.py --headless --num_envs 6
"""
from __future__ import annotations

import argparse
import math

parser = argparse.ArgumentParser(description="V1: Yaw reachability curve evaluation")
parser.add_argument("--task", type=str, default="Isaac-Forklift-PalletInsertLift-Direct-v0")
parser.add_argument("--num_envs", type=int, default=6,
                    help="Number of parallel envs (one per yaw angle)")
parser.add_argument("--max_steps", type=int, default=300,
                    help="Max simulation steps per evaluation")
parser.add_argument("--drive_strength", type=float, default=0.3,
                    help="Normalized drive action (0-1)")
parser.add_argument("--dist_front", type=float, default=0.25,
                    help="Initial distance from fork tip to pallet front (m)")
parser.add_argument("--lateral", type=float, default=0.02,
                    help="Initial lateral offset (m)")
parser.add_argument("--yaw_angles", type=str, default="0.5,1.0,2.0,3.0,4.0,5.0",
                    help="Comma-separated yaw angles in degrees to test")

from isaaclab.app import AppLauncher
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# ---- IsaacLab imports (after AppLauncher) ----
import torch
import gymnasium as gym
import isaaclab_tasks  # noqa: F401 – register tasks

# ---- Parse yaw angles ----
YAW_ANGLES_DEG = [float(x) for x in args.yaw_angles.split(",")]
num_yaw = len(YAW_ANGLES_DEG)

# Adjust num_envs to match yaw angles
actual_num_envs = num_yaw

print(f"\n{'='*70}")
print(f"S1.0P Phase V1: Yaw 可达性曲线评估")
print(f"{'='*70}")
print(f"测试 yaw 角度: {YAW_ANGLES_DEG}°")
print(f"初始距离: dist_front={args.dist_front}m, lateral={args.lateral}m")
print(f"驱动强度: {args.drive_strength}, 最大步数: {args.max_steps}")
print(f"{'='*70}\n")

# ---- Create environment ----
env = gym.make(args.task, cfg={"num_envs": actual_num_envs})
obs, info = env.reset()

# Get the unwrapped env for direct access
raw_env = env.unwrapped

# ---- Teleport each env to its specific yaw init position ----
def teleport_to_pre_insert(raw_env, dist_front: float, lateral: float, yaw_angles_deg: list[float]):
    """Teleport each env to pre-insert position with specific yaw."""
    device = raw_env.device
    n = len(yaw_angles_deg)
    env_ids = torch.arange(n, device=device)

    # Pallet state
    pallet_pos = raw_env.pallet.data.root_pos_w[0]  # (3,) - same for all envs
    pallet_depth = raw_env.cfg.pallet_depth_m
    fork_offset = raw_env._fork_forward_offset

    # s_front (scalar) along pallet insertion axis
    # Pallet at origin facing +X, pocket opens toward -X
    s_front = -0.5 * pallet_depth

    # Desired fork tip position: s_tip = s_front - dist_front (fork tip is dist_front before pallet)
    desired_s_tip = s_front - dist_front

    results = []
    positions = []
    quats = []

    for i, yaw_deg in enumerate(yaw_angles_deg):
        yaw_rad = math.radians(yaw_deg)

        # Fork tip world x: pallet_x + desired_s_tip * cos(pallet_yaw)
        # Since pallet_yaw ≈ 0 (default), tip_x = pallet_x + desired_s_tip
        tip_x = pallet_pos[0].item() + desired_s_tip

        # Robot root position: tip = root + fork_offset * cos/sin(robot_yaw)
        root_x = tip_x - fork_offset * math.cos(yaw_rad)
        root_y = pallet_pos[1].item() + lateral - fork_offset * math.sin(yaw_rad)
        root_z = 0.03

        positions.append([root_x, root_y, root_z])

        # Quaternion from yaw
        half = yaw_rad * 0.5
        quats.append([math.cos(half), 0.0, 0.0, math.sin(half)])

    pos_tensor = torch.tensor(positions, device=device, dtype=torch.float32)
    quat_tensor = torch.tensor(quats, device=device, dtype=torch.float32)

    raw_env._write_root_pose(raw_env.robot, pos_tensor, quat_tensor, env_ids)

    # Zero velocities and joints
    zeros3 = torch.zeros((n, 3), device=device)
    raw_env._write_root_vel(raw_env.robot, zeros3, zeros3, env_ids)

    joint_pos = raw_env.robot.data.default_joint_pos[env_ids].clone()
    joint_vel = torch.zeros_like(joint_pos)
    raw_env._write_joint_state(raw_env.robot, joint_pos, joint_vel, env_ids)

    # Reset fork tip baseline
    raw_env._fork_tip_z0[env_ids] = 0.03

    # Reset internal counters
    raw_env._last_insert_depth[env_ids] = 0.0
    raw_env._hold_counter[env_ids] = 0
    raw_env._is_first_step[env_ids] = True
    raw_env._lift_pos_target[env_ids] = 0.0
    raw_env._milestone_flags[env_ids] = False
    raw_env._fly_counter[env_ids] = 0
    raw_env._stall_counter[env_ids] = 0
    raw_env._early_stop_fly[env_ids] = False
    raw_env._early_stop_stall[env_ids] = False

    # Reset phi caches
    raw_env._prev_phi_align[env_ids] = 0.0
    raw_env._prev_phi_lift_progress[env_ids] = 0.0

    print(f"已将 {n} 个环境传送到 pre-insert 位置:")
    for i, yaw_deg in enumerate(yaw_angles_deg):
        print(f"  env[{i}]: yaw={yaw_deg:5.1f}°, root=({positions[i][0]:.3f}, {positions[i][1]:.3f})")


teleport_to_pre_insert(raw_env, args.dist_front, args.lateral, YAW_ANGLES_DEG)

# ---- Run forward drive and record metrics ----
max_insert_norm = torch.zeros(actual_num_envs, device=raw_env.device)
max_insert_step = torch.zeros(actual_num_envs, dtype=torch.long, device=raw_env.device)
stuck_flags = torch.zeros(actual_num_envs, dtype=torch.bool, device=raw_env.device)
collision_flags = torch.zeros(actual_num_envs, dtype=torch.bool, device=raw_env.device)

# Track insert_norm history for stuck detection
prev_insert_norm = torch.zeros(actual_num_envs, device=raw_env.device)
no_progress_count = torch.zeros(actual_num_envs, dtype=torch.long, device=raw_env.device)

# Constant action: drive forward, no steer, no lift
action = torch.zeros((actual_num_envs, 3), device=raw_env.device)
action[:, 0] = args.drive_strength  # forward drive

print(f"\n开始直进测试 ({args.max_steps} 步)...\n")

for step in range(args.max_steps):
    obs, reward, terminated, truncated, info = env.step(action)

    # Extract current insert_norm from env logs
    if "log" in raw_env.extras:
        cur_insert = raw_env.extras["log"].get("err/insert_norm_mean", None)

    # Compute insert_norm directly from env state
    tip = raw_env._compute_fork_tip()
    pallet_pos = raw_env.pallet.data.root_pos_w
    pallet_quat = raw_env.pallet.data.root_quat_w
    w_p, x_p, y_p, z_p = pallet_quat.unbind(-1)
    pallet_yaw = torch.atan2(2.0 * (w_p * z_p + x_p * y_p),
                             1.0 - 2.0 * (y_p * y_p + z_p * z_p))
    cp = torch.cos(pallet_yaw)
    sp = torch.sin(pallet_yaw)
    u_in = torch.stack([cp, sp], dim=-1)

    rel_tip = tip[:, :2] - pallet_pos[:, :2]
    s_tip = torch.sum(rel_tip * u_in, dim=-1)
    s_front = -0.5 * raw_env.cfg.pallet_depth_m
    insert_depth = torch.clamp(s_tip - s_front, min=0.0)
    insert_norm = torch.clamp(insert_depth / (raw_env.cfg.pallet_depth_m + 1e-6), 0.0, 1.0)

    # Update max
    improved = insert_norm > max_insert_norm
    max_insert_norm = torch.where(improved, insert_norm, max_insert_norm)
    max_insert_step = torch.where(improved, torch.tensor(step, device=raw_env.device), max_insert_step)

    # Stuck detection: if insert_norm hasn't improved by > 0.001 in 50 steps
    progress = insert_norm - prev_insert_norm
    no_progress_count = torch.where(progress < 0.001, no_progress_count + 1, torch.zeros_like(no_progress_count))
    stuck_flags = stuck_flags | (no_progress_count > 50)
    prev_insert_norm = insert_norm.clone()

    # Early termination detection
    collision_flags = collision_flags | terminated

    # Print progress every 50 steps
    if (step + 1) % 50 == 0:
        print(f"  Step {step+1:3d}: insert_norm = [{', '.join(f'{v:.4f}' for v in insert_norm.tolist())}]")

# ---- Results ----
print(f"\n{'='*70}")
print(f"结果: Yaw 初值 vs 最大插入深度")
print(f"{'='*70}")
print(f"{'Yaw (°)':>8} | {'Max Insert Norm':>15} | {'Max Step':>9} | {'Stuck':>6} | {'Collision':>10}")
print(f"{'-'*8}-+-{'-'*15}-+-{'-'*9}-+-{'-'*6}-+-{'-'*10}")

for i, yaw_deg in enumerate(YAW_ANGLES_DEG):
    ins = max_insert_norm[i].item()
    ms = max_insert_step[i].item()
    stk = "Yes" if stuck_flags[i].item() else "No"
    col = "Yes" if collision_flags[i].item() else "No"
    print(f"{yaw_deg:8.1f} | {ins:15.4f} | {ms:9d} | {stk:>6} | {col:>10}")

# ---- Threshold analysis ----
print(f"\n{'='*70}")
print(f"阈值分析")
print(f"{'='*70}")

# Find the yaw where insert_norm drops below useful thresholds
thresholds = [0.5, 0.3, 0.1, 0.05]
for thresh in thresholds:
    above = [i for i, v in enumerate(max_insert_norm.tolist()) if v >= thresh]
    if above:
        max_yaw_for_thresh = YAW_ANGLES_DEG[above[-1]]
        print(f"  insert_norm >= {thresh:.2f}: 最大可用 yaw = {max_yaw_for_thresh}°")
    else:
        print(f"  insert_norm >= {thresh:.2f}: 所有 yaw 均不可达")

# Conclusion
best_idx = max_insert_norm.argmax().item()
worst_idx = max_insert_norm.argmin().item()
print(f"\n结论:")
print(f"  最佳: yaw={YAW_ANGLES_DEG[best_idx]}° → insert_norm={max_insert_norm[best_idx]:.4f}")
print(f"  最差: yaw={YAW_ANGLES_DEG[worst_idx]}° → insert_norm={max_insert_norm[worst_idx]:.4f}")

if max_insert_norm[worst_idx].item() > 0.3:
    print(f"\n  判断: yaw={YAW_ANGLES_DEG[-1]}° 仍能深插 (>{0.3})")
    print(f"  => 碰撞/几何太宽松，1° 目标无物理约束支撑")
elif max_insert_norm[worst_idx].item() < 0.05:
    print(f"\n  判断: yaw={YAW_ANGLES_DEG[worst_idx]}° 基本无法插入")
    # Find first yaw that can't reach 0.1
    for i, v in enumerate(max_insert_norm.tolist()):
        if v < 0.1:
            print(f"  => yaw >= {YAW_ANGLES_DEG[i]}° 明显卡死，精度目标有物理约束支撑")
            break
else:
    # Find degradation point
    for i in range(1, len(max_insert_norm)):
        if max_insert_norm[i].item() < max_insert_norm[0].item() * 0.5:
            print(f"\n  判断: yaw={YAW_ANGLES_DEG[i]}° 时插入深度显著衰减 (降至 {max_insert_norm[0].item()*0.5:.2f} 以下)")
            print(f"  => 精度目标在 {YAW_ANGLES_DEG[i-1]}°~{YAW_ANGLES_DEG[i]}° 之间存在物理约束")
            break

print(f"\n{'='*70}")

# ---- Cleanup ----
env.close()
simulation_app.close()
