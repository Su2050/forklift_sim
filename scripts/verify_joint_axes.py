#!/usr/bin/env python3
"""USD 关节轴方向验证 — 最小可验证实验。

三阶段验证:
  Phase 1: USD 静态属性读取（配置一致性检查，仅供参考）
  Phase 2: 物理单轴测试（物理真理验证）
  Phase 3: 转向对称性冒烟测试（端到端验证）

用法:
  conda deactivate && cd IsaacLab && \\
  bash isaaclab.sh -p ../scripts/verify_joint_axes.py --headless \\
    > ../logs/$(date +%%Y%%m%%d_%%H%%M%%S)_sanity_check_joint_axes.log 2>&1

输出:
  [PASS] / [FAIL] 标记，可用 grep FAIL 快速检查。
  存在任何 FAIL 时以非零退出码退出。
"""

import sys
import math
from pathlib import Path

isaaclab_path = Path(__file__).resolve().parent.parent / "IsaacLab"
sys.path.insert(0, str(isaaclab_path / "source"))

import torch
from isaaclab.app import AppLauncher
import argparse

parser = argparse.ArgumentParser(description="USD 关节轴方向验证")
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from pxr import Usd, UsdPhysics
from isaaclab_tasks.direct.forklift_pallet_insert_lift.env_cfg import (
    ForkliftPalletInsertLiftEnvCfg,
)
from isaaclab_tasks.direct.forklift_pallet_insert_lift.env import (
    ForkliftPalletInsertLiftEnv,
    _quat_to_yaw,
)


# ─── Helpers ────────────────────────────────────────────────────────

SIGN_THRESH = 0.01


def P(msg):
    print(msg, flush=True)


def sign_label(v):
    if v > SIGN_THRESH:
        return "+"
    elif v < -SIGN_THRESH:
        return "-"
    return "0"


results: list[tuple[str, str, bool, str]] = []


def record(phase: str, name: str, passed: bool, msg: str):
    tag = "[PASS]" if passed else "[FAIL]"
    results.append((phase, name, passed, msg))
    P(f"  {tag} {name}: {msg}")


def _zero_all_wheels(env):
    """Set all wheel velocity targets to zero to prevent movement."""
    dev = env.device
    env.robot.set_joint_velocity_target(
        torch.zeros((1, len(env._front_wheel_ids)), device=dev),
        joint_ids=env._front_wheel_ids,
    )
    env.robot.set_joint_velocity_target(
        torch.zeros((1, len(env._back_wheel_ids)), device=dev),
        joint_ids=env._back_wheel_ids,
    )


def _step_raw(env, n):
    """Low-level sim stepping (bypass RL logic)."""
    for _ in range(n):
        env.robot.write_data_to_sim()
        env.sim.step(render=False)
        env.scene.update(dt=env.cfg.sim.dt)


# ═════════════════════════════════════════════════════════════════════
# Phase 1: USD 静态属性读取
# ═════════════════════════════════════════════════════════════════════

def phase1_static(env):
    P("\n" + "=" * 78)
    P("  Phase 1: USD 静态属性读取（配置一致性检查）")
    P("=" * 78)

    stage = env.sim.stage
    robot_prim_path = env.cfg.robot_cfg.prim_path.replace("env_.*", "env_0")
    robot_prim = stage.GetPrimAtPath(robot_prim_path)
    if not robot_prim.IsValid():
        P(f"  [ERROR] 找不到 robot prim: {robot_prim_path}")
        return

    joints = {}
    for prim in Usd.PrimRange(robot_prim):
        if not prim.IsA(UsdPhysics.Joint):
            continue
        name = prim.GetName()

        joint_type = "unknown"
        axis = "?"
        lower = upper = None

        if prim.IsA(UsdPhysics.RevoluteJoint):
            joint_type = "revolute"
            api = UsdPhysics.RevoluteJoint(prim)
            a = api.GetAxisAttr()
            if a:
                axis = a.Get() or "?"
            lo = api.GetLowerLimitAttr()
            hi = api.GetUpperLimitAttr()
            if lo:
                lower = lo.Get()
            if hi:
                upper = hi.Get()
        elif prim.IsA(UsdPhysics.PrismaticJoint):
            joint_type = "prismatic"
            api = UsdPhysics.PrismaticJoint(prim)
            a = api.GetAxisAttr()
            if a:
                axis = a.Get() or "?"
            lo = api.GetLowerLimitAttr()
            hi = api.GetUpperLimitAttr()
            if lo:
                lower = lo.Get()
            if hi:
                upper = hi.Get()

        drive_kind = "angular" if joint_type == "revolute" else "linear"
        stiff = damp = maxf = "N/A"
        drv = UsdPhysics.DriveAPI.Get(prim, drive_kind)
        if drv:
            s = drv.GetStiffnessAttr()
            d = drv.GetDampingAttr()
            f = drv.GetMaxForceAttr()
            if s:
                stiff = s.Get()
            if d:
                damp = d.Get()
            if f:
                maxf = f.Get()

        joints[name] = dict(type=joint_type, axis=axis, lower=lower, upper=upper)
        lim = f"[{lower}, {upper}]" if lower is not None else "N/A"
        P(f"  [STATIC] {name:30s}: {joint_type:10s} axis={axis}  "
          f"limits={lim}  stiff={stiff}  damp={damp}  maxF={maxf}")

    # detect left/right pairs
    P("")
    seen = set()
    for name in sorted(joints):
        if not name.startswith("left_") or name in seen:
            continue
        right_name = "right_" + name[5:]
        if right_name not in joints:
            continue
        seen.update([name, right_name])
        la = joints[name]["axis"]
        ra = joints[right_name]["axis"]
        base = name[5:]
        if la == ra:
            P(f"  [PAIR] {base}: axis 定义相同 ({la})")
            P(f"         [WARN] 相同 axis 不能断定物理方向相同——"
              f"Prim Transform 可能含镜像，需 Phase 2 验证")
        else:
            P(f"  [PAIR] {base}: axis 不同 (left={la}, right={ra})")
            P(f"         [WARN] 不同 axis 暗示可能有镜像，需 Phase 2 验证")


# ═════════════════════════════════════════════════════════════════════
# Phase 2: 物理单轴测试
# ═════════════════════════════════════════════════════════════════════

def _run_rotator_test(env, left_target, right_target, n_steps=100):
    """Reset, set rotator targets, step, return (left_actual, right_actual)."""
    dev = env.device
    left_id = env._left_rotator_id
    right_id = env._right_rotator_id

    env.reset()
    for _ in range(n_steps):
        env.robot.set_joint_position_target(
            torch.tensor([[left_target]], device=dev), joint_ids=left_id)
        env.robot.set_joint_position_target(
            torch.tensor([[right_target]], device=dev), joint_ids=right_id)
        env.robot.set_joint_position_target(
            torch.tensor([[0.0]], device=dev), joint_ids=[env._lift_id])
        _zero_all_wheels(env)
        _step_raw(env, 1)

    jp = env.robot.data.joint_pos[0]
    return jp[left_id[0]].item(), jp[right_id[0]].item()


def phase2_physical(env):
    P("\n" + "=" * 78)
    P("  Phase 2: 物理单轴测试（物理真理验证）")
    P("=" * 78)

    tv = 0.3  # test value in rad

    # ── A: left only ──
    P(f"\n  ── A: 仅 left_rotator +{tv} rad ──")
    la, _ = _run_rotator_test(env, +tv, 0.0)
    P(f"    actual = {la:.4f} rad ({math.degrees(la):.2f}°)  sign={sign_label(la)}")
    record("Phase2", "left_rotator 正值响应",
           la > SIGN_THRESH,
           f"input=+{tv}, actual={la:+.4f} ({sign_label(la)})")

    # ── B: right only ──
    P(f"\n  ── B: 仅 right_rotator +{tv} rad ──")
    _, ra = _run_rotator_test(env, 0.0, +tv)
    P(f"    actual = {ra:.4f} rad ({math.degrees(ra):.2f}°)  sign={sign_label(ra)}")
    record("Phase2", "right_rotator 正值响应",
           ra > SIGN_THRESH,
           f"input=+{tv}, actual={ra:+.4f} ({sign_label(ra)})")

    # ── C: 同号输入 ──
    P(f"\n  ── C: 两轮同号 +{tv} rad ──")
    lc, rc = _run_rotator_test(env, +tv, +tv)
    ls, rs = sign_label(lc), sign_label(rc)
    same_sign = ls == rs and ls != "0"
    P(f"    left={lc:+.4f} ({ls})  right={rc:+.4f} ({rs})")

    axes_mirrored = not same_sign
    if same_sign:
        P(f"    → 同号输入 → 同号输出: 关节轴物理方向相同（不是镜像）")
        P(f"    → 控制代码应对左右轮使用 **相同符号**")
    else:
        P(f"    → 同号输入 → 异号输出: 关节轴物理方向镜像")
        P(f"    → 控制代码应对左右轮使用 **相反符号**")

    record("Phase2", "配对关节物理方向",
           True,
           f"mirrored={axes_mirrored}  left={lc:+.4f} right={rc:+.4f}")

    # ── D: 异号输入（当前代码做法: left=-steer, right=+steer）──
    P(f"\n  ── D: 异号输入 left=-{tv} right=+{tv}（当前 env.py 做法）──")
    ld, rd = _run_rotator_test(env, -tv, +tv)
    lds, rds = sign_label(ld), sign_label(rd)
    opp_sign_d = lds != rds and lds != "0" and rds != "0"
    P(f"    left={ld:+.4f} ({lds})  right={rd:+.4f} ({rds})")
    if opp_sign_d:
        P(f"    → 异号输入 → 异号输出: 两轮物理上指向相反方向（八字形）")
    else:
        P(f"    → 异号输入 → 同号输出: 两轮物理上仍同向")

    record("Phase2", "当前代码异号输入效果",
           not opp_sign_d,
           f"left={ld:+.4f} right={rd:+.4f}, "
           f"{'八字形(两轮反向)→转向失效' if opp_sign_d else '两轮同向→转向正常'}")

    return axes_mirrored


# ═════════════════════════════════════════════════════════════════════
# Phase 3: 转向对称性冒烟测试
# ═════════════════════════════════════════════════════════════════════

def _run_steer_episode(env, drive_val, steer_val, n_steps=90):
    """Run env through _apply_action with given RL actions, return Δyaw."""
    env.reset()
    yaw0 = _quat_to_yaw(env.robot.data.root_quat_w)[0].item()
    actions = torch.tensor([[drive_val, steer_val, 0.0]], device=env.device)
    for _ in range(n_steps):
        env._pre_physics_step(actions)
        env._apply_action()
        env.sim.step(render=False)
        env.scene.update(dt=env.cfg.sim.dt)
    yaw1 = _quat_to_yaw(env.robot.data.root_quat_w)[0].item()
    return yaw1 - yaw0


def phase3_symmetry(env):
    P("\n" + "=" * 78)
    P("  Phase 3: 转向对称性冒烟测试（端到端，经过 _apply_action）")
    P("=" * 78)

    drive, steer = 0.5, 0.3

    P(f"\n  ── steer=+{steer}, drive={drive} ──")
    dyaw_pos = _run_steer_episode(env, drive, +steer)
    P(f"    Δyaw = {math.degrees(dyaw_pos):+.2f}°")

    P(f"\n  ── steer=-{steer}, drive={drive} ──")
    dyaw_neg = _run_steer_episode(env, drive, -steer)
    P(f"    Δyaw = {math.degrees(dyaw_neg):+.2f}°")

    eps = 1e-6
    signs_opposite = (
        (dyaw_pos > eps and dyaw_neg < -eps)
        or (dyaw_pos < -eps and dyaw_neg > eps)
    )
    ratio = abs(dyaw_pos) / (abs(dyaw_neg) + eps)
    symmetric = 0.5 < ratio < 2.0

    P(f"\n  符号相反: {signs_opposite}")
    P(f"  幅度比: {ratio:.2f} (期望 0.5~2.0)")

    record("Phase3", "转向符号反转",
           signs_opposite,
           f"Δyaw(+steer)={math.degrees(dyaw_pos):+.2f}°  "
           f"Δyaw(-steer)={math.degrees(dyaw_neg):+.2f}°")
    record("Phase3", "转向幅度对称",
           symmetric,
           f"ratio={ratio:.2f}")


# ═════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════

def main():
    P("=" * 78)
    P("  USD 关节轴方向验证 — 最小可验证实验")
    P("=" * 78)

    P("\n[INIT] 创建环境 (num_envs=1) ...")
    cfg = ForkliftPalletInsertLiftEnvCfg()
    cfg.scene.num_envs = 1
    cfg.episode_length_s = 3600.0
    env = ForkliftPalletInsertLiftEnv(cfg)
    P("[INIT] 环境创建完成\n")

    phase1_static(env)
    phase2_physical(env)
    phase3_symmetry(env)

    # ── Summary ──
    P("\n" + "=" * 78)
    P("  汇总报告")
    P("=" * 78)
    n_pass = sum(1 for *_, p, _ in results if p)
    n_fail = sum(1 for *_, p, _ in results if not p)
    for phase, name, passed, msg in results:
        tag = "[PASS]" if passed else "[FAIL]"
        P(f"  {tag} [{phase}] {name}: {msg}")
    P(f"\n  总计: {n_pass} PASS, {n_fail} FAIL")
    if n_fail > 0:
        P("\n  *** 存在 FAIL 项 ***")
        P("  检查 Phase2 的物理方向判定，对照 env.py _apply_action() 中的符号处理。")
        P("  如果 Phase2 显示轴不是镜像，而 env.py 有 steer_left=-steer，则去掉取反。")
    else:
        P("\n  所有检查通过。关节轴方向与控制代码一致。")
    P("=" * 78)

    env.close()
    simulation_app.close()
    sys.exit(1 if n_fail > 0 else 0)


if __name__ == "__main__":
    main()
