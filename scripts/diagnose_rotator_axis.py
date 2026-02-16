#!/usr/bin/env python3
"""
诊断脚本：验证左右 rotator_joint 的关节轴方向是否镜像。

测试方法：
  1. 只给 left_rotator_joint 设正角度（right 保持 0），观察左轮转向
  2. 只给 right_rotator_joint 设正角度（left 保持 0），观察右轮转向
  3. 两个都设正角度，观察是否同向
  4. 一正一负（当前代码的做法），观察是否同向

如果测试 3 两轮同向 → 轴不是镜像 → 代码取反是错的
如果测试 4 两轮同向 → 轴是镜像   → 代码取反是对的
"""

import sys
import math
from pathlib import Path

isaaclab_path = Path(__file__).resolve().parent.parent / "IsaacLab"
sys.path.insert(0, str(isaaclab_path / "source"))

import torch
from isaaclab.app import AppLauncher
import argparse

parser = argparse.ArgumentParser(description="诊断 rotator 关节轴方向")
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from isaaclab_tasks.direct.forklift_pallet_insert_lift.env_cfg import ForkliftPalletInsertLiftEnvCfg
from isaaclab_tasks.direct.forklift_pallet_insert_lift.env import ForkliftPalletInsertLiftEnv


import sys as _sys
def P(msg):
    print(msg, flush=True)

def main():
    P("[DIAG] 创建环境...")
    cfg = ForkliftPalletInsertLiftEnvCfg()
    cfg.scene.num_envs = 1
    cfg.episode_length_s = 3600.0
    env = ForkliftPalletInsertLiftEnv(cfg)
    P("[DIAG] 环境创建完成")

    joint_names = env.robot.joint_names
    left_rot_id = env._left_rotator_id
    right_rot_id = env._right_rotator_id

    P("\n" + "=" * 80)
    P("  Rotator 关节轴方向诊断")
    P("=" * 80)
    P(f"  left_rotator_id  = {left_rot_id}  ({joint_names[left_rot_id[0]]})")
    P(f"  right_rotator_id = {right_rot_id} ({joint_names[right_rot_id[0]]})")

    test_angle = 0.3  # rad, ~17 degrees

    tests = [
        ("A: 仅左轮 +0.3 rad", +test_angle, 0.0),
        ("B: 仅右轮 +0.3 rad", 0.0, +test_angle),
        ("C: 两轮都 +0.3 rad（同号）", +test_angle, +test_angle),
        ("D: 左-0.3 右+0.3（当前代码做法）", -test_angle, +test_angle),
        ("E: 两轮都 -0.3 rad（同号负）", -test_angle, -test_angle),
    ]

    for ti, (name, left_target, right_target) in enumerate(tests):
        P(f"\n{'─' * 70}")
        P(f"  测试 {name}")
        P(f"  left_target={left_target:.3f} rad ({math.degrees(left_target):.1f}°)")
        P(f"  right_target={right_target:.3f} rad ({math.degrees(right_target):.1f}°)")
        P(f"{'─' * 70}")

        P(f"  [DIAG] reset #{ti}...")
        env.reset()
        P(f"  [DIAG] reset #{ti} done, stepping...")

        # 设置转向目标，不设驱动（drive=0）
        for step in range(120):
            actions = torch.tensor([[0.0, 0.0, 0.0]], device=env.device)
            env._pre_physics_step(actions)

            # 手动覆盖 rotator 目标
            env.robot.set_joint_position_target(
                torch.tensor([[left_target]], device=env.device),
                joint_ids=left_rot_id,
            )
            env.robot.set_joint_position_target(
                torch.tensor([[right_target]], device=env.device),
                joint_ids=right_rot_id,
            )
            # 所有轮子速度=0
            env.robot.set_joint_velocity_target(
                torch.zeros((1, len(env._front_wheel_ids)), device=env.device),
                joint_ids=env._front_wheel_ids,
            )
            env.robot.set_joint_velocity_target(
                torch.zeros((1, len(env._back_wheel_ids)), device=env.device),
                joint_ids=env._back_wheel_ids,
            )
            env.robot.write_data_to_sim()
            env.sim.step(render=False)
            env.scene.update(dt=env.cfg.sim.dt)

        P(f"  [DIAG] stepping done")
        # 读取实际关节角度
        joint_pos = env.robot.data.joint_pos[0]
        left_actual = joint_pos[left_rot_id[0]].item()
        right_actual = joint_pos[right_rot_id[0]].item()

        P(f"  实际角度:")
        P(f"    left_rotator  = {left_actual:.4f} rad ({math.degrees(left_actual):.2f}°)")
        P(f"    right_rotator = {right_actual:.4f} rad ({math.degrees(right_actual):.2f}°)")
        P(f"  符号: left={'正' if left_actual > 0.01 else '负' if left_actual < -0.01 else '零'}"
          f"  right={'正' if right_actual > 0.01 else '负' if right_actual < -0.01 else '零'}")

        same_sign = (left_actual > 0.01 and right_actual > 0.01) or \
                    (left_actual < -0.01 and right_actual < -0.01)
        opp_sign = (left_actual > 0.01 and right_actual < -0.01) or \
                   (left_actual < -0.01 and right_actual > 0.01)

        if left_target != 0 and right_target != 0:
            if same_sign:
                P(f"  → 两轮实际角度同号")
            elif opp_sign:
                P(f"  → 两轮实际角度异号")

    # 汇总判断
    P("\n" + "=" * 80)
    P("  诊断结论")
    P("=" * 80)
    P("  请对比测试 C 和 D 的结果：")
    P("    - 如果 C（同号输入）两轮物理上转向同一方向 → 关节轴不是镜像")
    P("      → 当前代码 steer_left=-steer 是错误的，应改为 steer_left=steer")
    P("    - 如果 D（异号输入）两轮物理上转向同一方向 → 关节轴是镜像")
    P("      → 当前代码 steer_left=-steer 是正确的")
    P("=" * 80)

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
