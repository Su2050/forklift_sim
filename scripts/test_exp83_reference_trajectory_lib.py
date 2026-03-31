#!/usr/bin/env python3
"""Minimal tests for Exp8.3 standalone reference-trajectory library."""

from __future__ import annotations

import math
import sys

import numpy as np

from exp83_reference_trajectory_lib import (
    Pose2D,
    compute_aligned_front_goal_pose,
    plan_root_path_first_to_front_goal,
    plan_rs_exact,
    plan_rs_to_front_goal,
    vehicle_to_fork_center,
)


def assert_close(name: str, actual: float, expected: float, tol: float) -> None:
    if abs(actual - expected) > tol:
        raise AssertionError(f"{name}: actual={actual} expected={expected} tol={tol}")


def main() -> None:
    pallet_xy = np.array([0.0, 0.0], dtype=np.float64)
    pallet_yaw = 0.0
    pallet_depth_m = 0.8
    fork_reach_m = 1.87
    vehicle_to_fc = vehicle_to_fork_center(fork_reach_m)
    stop_buffer_m = 1.05

    goal_pose, fork_goal_xy = compute_aligned_front_goal_pose(
        pallet_xy=pallet_xy,
        pallet_yaw=pallet_yaw,
        pallet_depth_m=pallet_depth_m,
        vehicle_to_fork_center_m=vehicle_to_fc,
        fork_front_stop_buffer_m=stop_buffer_m,
    )
    assert_close("goal_y", goal_pose.y, 0.0, 1e-9)
    assert_close("goal_yaw", goal_pose.yaw, 0.0, 1e-9)
    assert_close("fork_goal_x", fork_goal_xy[0], -0.5 * pallet_depth_m - stop_buffer_m, 1e-9)

    start = Pose2D(-3.45, -0.15, math.radians(-6.0))
    root_plan = plan_root_path_first_to_front_goal(
        start=start,
        pallet_xy=pallet_xy,
        pallet_yaw=pallet_yaw,
        pallet_depth_m=pallet_depth_m,
        fork_reach_m=fork_reach_m,
        fork_front_stop_buffer_m=stop_buffer_m,
        curve_min_span_m=0.35,
        final_straight_min_m=0.10,
        num_samples=64,
    )
    assert_close("root_goal_x", root_plan.vehicle_xy[-1, 0], goal_pose.x, 1e-6)
    assert_close("root_goal_y", root_plan.vehicle_xy[-1, 1], goal_pose.y, 1e-6)
    assert_close("root_goal_yaw", root_plan.vehicle_yaw[-1], goal_pose.yaw, 1e-6)

    rs_plan = plan_rs_exact(
        start=start,
        goal=goal_pose,
        min_turn_radius_m=0.55,
        vehicle_to_fork_center_m=vehicle_to_fc,
    )
    pos_err = math.hypot(rs_plan.vehicle_xy[-1, 0] - goal_pose.x, rs_plan.vehicle_xy[-1, 1] - goal_pose.y)
    yaw_err = abs(math.degrees(math.atan2(math.sin(rs_plan.vehicle_yaw[-1] - goal_pose.yaw), math.cos(rs_plan.vehicle_yaw[-1] - goal_pose.yaw))))
    if pos_err > 0.12:
        raise AssertionError(f"rs pos_err too large: {pos_err}")
    if yaw_err > 8.0:
        raise AssertionError(f"rs yaw_err too large: {yaw_err}")

    rs_front_plan = plan_rs_to_front_goal(
        start=start,
        pallet_xy=pallet_xy,
        pallet_yaw=pallet_yaw,
        pallet_depth_m=pallet_depth_m,
        fork_reach_m=fork_reach_m,
        fork_front_stop_buffer_m=stop_buffer_m,
        min_turn_radius_m=0.55,
    )
    assert_close("rs_front_goal_x", rs_front_plan.goal_pose.x, goal_pose.x, 1e-9)
    assert_close("rs_front_goal_y", rs_front_plan.goal_pose.y, goal_pose.y, 1e-9)

    print("PASS: exp83 reference trajectory library")
    print(f"  root-path-first end: x={root_plan.vehicle_xy[-1, 0]:+.3f}, y={root_plan.vehicle_xy[-1, 1]:+.3f}, yaw={math.degrees(root_plan.vehicle_yaw[-1]):+.2f}deg")
    print(f"  rs-exact end err: pos={pos_err:.4f}m, yaw={yaw_err:.2f}deg")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"FAIL: {exc}")
        sys.exit(1)
