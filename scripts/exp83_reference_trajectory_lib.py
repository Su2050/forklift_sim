#!/usr/bin/env python3
"""Standalone reference-trajectory generators for Exp8.3.

This module keeps trajectory generation outside the environment so we can:
- compare models without touching PPO/reward code
- unit-test trajectory endpoints and geometry
- visualize explicit start/goal pose planning

It currently provides two main generators:
- `plan_root_path_first_to_front_goal`: current vehicle-aware cubic alignment path
- `plan_rs_exact`: exact Reeds-Shepp sampling via the repository's `rs/rs.py`

The older `plan_rs_lattice` helper is kept as a debugging baseline, but should
not be used as the primary "RS model" result anymore.
"""

from __future__ import annotations

from dataclasses import dataclass
import heapq
import math
from pathlib import Path
import sys
from typing import Iterable

import numpy as np

RS_DIR = Path(__file__).resolve().parents[1] / "rs"
if str(RS_DIR) not in sys.path:
    sys.path.append(str(RS_DIR))
import rs as exact_rs


FORK_CENTER_BACKOFF_M = 0.6


@dataclass(frozen=True)
class Pose2D:
    x: float
    y: float
    yaw: float  # radians


@dataclass
class TrajectoryPlan:
    model: str
    vehicle_xy: np.ndarray
    vehicle_yaw: np.ndarray
    fork_center_xy: np.ndarray
    fork_center_yaw: np.ndarray
    goal_pose: Pose2D
    metadata: dict


@dataclass(frozen=True)
class LatticeConfig:
    primitive_length_m: float = 0.18
    integration_step_m: float = 0.03
    xy_resolution_m: float = 0.08
    yaw_resolution_deg: float = 5.0
    pos_tolerance_m: float = 0.08
    yaw_tolerance_deg: float = 4.0
    reverse_penalty: float = 1.15
    steer_penalty: float = 0.03
    steer_change_penalty: float = 0.04
    direction_change_penalty: float = 0.05
    max_nodes: int = 60000
    bounds_margin_m: float = 1.2


def wrap_angle(theta: float) -> float:
    return math.atan2(math.sin(theta), math.cos(theta))


def rotation_from_yaw(yaw: float) -> np.ndarray:
    return np.array([math.cos(yaw), math.sin(yaw)], dtype=np.float64)


def lateral_from_yaw(yaw: float) -> np.ndarray:
    return np.array([-math.sin(yaw), math.cos(yaw)], dtype=np.float64)


def vehicle_to_fork_center(fork_reach_m: float, fork_center_backoff_m: float = FORK_CENTER_BACKOFF_M) -> float:
    return max(float(fork_reach_m) - float(fork_center_backoff_m), 0.0)


def map_vehicle_path_to_fork_center(
    vehicle_xy: np.ndarray,
    vehicle_yaw: np.ndarray,
    *,
    vehicle_to_fork_center_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    headings = np.stack([np.cos(vehicle_yaw), np.sin(vehicle_yaw)], axis=1)
    fork_xy = vehicle_xy + vehicle_to_fork_center_m * headings
    return fork_xy, vehicle_yaw.copy()


def compute_aligned_front_goal_pose(
    *,
    pallet_xy: np.ndarray,
    pallet_yaw: float,
    pallet_depth_m: float,
    vehicle_to_fork_center_m: float,
    fork_front_stop_buffer_m: float,
) -> tuple[Pose2D, np.ndarray]:
    """Return a vehicle goal pose aligned in front of the pallet.

    The goal is not inside the pallet. Instead, the vehicle stops such that the
    fork center sits `fork_front_stop_buffer_m` in front of the pallet front face.
    """
    u_in = rotation_from_yaw(pallet_yaw)
    s_front = -0.5 * float(pallet_depth_m)
    fork_goal_s = s_front - float(fork_front_stop_buffer_m)
    fork_goal_xy = np.asarray(pallet_xy, dtype=np.float64) + fork_goal_s * u_in
    vehicle_goal_xy = fork_goal_xy - vehicle_to_fork_center_m * u_in
    return Pose2D(float(vehicle_goal_xy[0]), float(vehicle_goal_xy[1]), float(pallet_yaw)), fork_goal_xy


def _sample_direct_cubic_vehicle_path(
    *,
    start: Pose2D,
    goal: Pose2D,
    tangent_scale_m: float,
    num_samples: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Direct cubic Hermite fallback for very-near front-goal cases.

    When the front-goal is already close in the goal frame, the old `pre_s`
    construction can collapse to a near-zero span. In that regime, a direct
    start-pose -> goal-pose cubic is a more stable standalone baseline.
    """
    p0 = np.array([start.x, start.y], dtype=np.float64)
    p1 = np.array([goal.x, goal.y], dtype=np.float64)
    m0 = tangent_scale_m * rotation_from_yaw(start.yaw)
    m1 = tangent_scale_m * rotation_from_yaw(goal.yaw)

    t = np.linspace(0.0, 1.0, max(2, num_samples), dtype=np.float64)
    t2 = t * t
    t3 = t2 * t
    h00 = 2.0 * t3 - 3.0 * t2 + 1.0
    h10 = t3 - 2.0 * t2 + t
    h01 = -2.0 * t3 + 3.0 * t2
    h11 = t3 - t2
    vehicle_xy = (
        h00[:, None] * p0[None, :]
        + h10[:, None] * m0[None, :]
        + h01[:, None] * p1[None, :]
        + h11[:, None] * m1[None, :]
    )

    dh00 = 6.0 * t2 - 6.0 * t
    dh10 = 3.0 * t2 - 4.0 * t + 1.0
    dh01 = -6.0 * t2 + 6.0 * t
    dh11 = 3.0 * t2 - 2.0 * t
    deriv = (
        dh00[:, None] * p0[None, :]
        + dh10[:, None] * m0[None, :]
        + dh01[:, None] * p1[None, :]
        + dh11[:, None] * m1[None, :]
    )
    vehicle_yaw = np.arctan2(deriv[:, 1], deriv[:, 0])
    vehicle_yaw[0] = start.yaw
    vehicle_yaw[-1] = goal.yaw
    return vehicle_xy, vehicle_yaw, {"mode": "direct_cubic_fallback", "tangent_scale_m": tangent_scale_m}


def _sample_cubic_alignment_vehicle_path(
    *,
    start: Pose2D,
    goal: Pose2D,
    curve_min_span_m: float,
    final_straight_min_m: float,
    num_samples: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    u_goal = rotation_from_yaw(goal.yaw)
    v_goal = lateral_from_yaw(goal.yaw)
    start_xy = np.array([start.x, start.y], dtype=np.float64)
    goal_xy = np.array([goal.x, goal.y], dtype=np.float64)
    rel = start_xy - goal_xy
    start_s = float(np.dot(rel, u_goal))
    start_y = float(np.dot(rel, v_goal))
    yaw_rel = wrap_angle(start.yaw - goal.yaw)

    # Goal frame convention:
    # - x/s increases toward the pallet-facing direction
    # - goal pose is at s=0, y=0, yaw=0
    pre_s = max(start_s + float(curve_min_span_m), -10.0)
    pre_s = min(pre_s, -float(final_straight_min_m))
    span_s = max(pre_s - start_s, 1e-6)

    near_goal_threshold = max(0.12, 0.5 * float(curve_min_span_m))
    if start_s >= -float(final_straight_min_m) or span_s <= near_goal_threshold:
        goal_dist = math.hypot(goal.x - start.x, goal.y - start.y)
        tangent_scale_m = max(goal_dist * 0.7, float(curve_min_span_m), float(final_straight_min_m))
        vehicle_xy, vehicle_yaw, meta = _sample_direct_cubic_vehicle_path(
            start=start,
            goal=goal,
            tangent_scale_m=tangent_scale_m,
            num_samples=num_samples,
        )
        meta.update(
            {
                "start_s": start_s,
                "start_y": start_y,
                "pre_s": pre_s,
                "curve_span_s": span_s,
            }
        )
        return vehicle_xy, vehicle_yaw, meta

    slope0 = math.tan(yaw_rel)
    a = (slope0 * span_s + 2.0 * start_y) / (span_s ** 3)
    b = (-2.0 * slope0 * span_s - 3.0 * start_y) / (span_s ** 2)

    num_curve = max(2, int(num_samples * 0.7))
    num_line = max(1, num_samples - num_curve)

    ds_curve = np.linspace(0.0, span_s, num_curve, dtype=np.float64)
    s_curve = start_s + ds_curve
    y_curve = a * ds_curve**3 + b * ds_curve**2 + slope0 * ds_curve + start_y
    dy_ds = 3.0 * a * ds_curve**2 + 2.0 * b * ds_curve + slope0

    curve_xy = goal_xy + np.outer(s_curve, u_goal) + np.outer(y_curve, v_goal)
    curve_dirs = u_goal.reshape(1, 2) + dy_ds.reshape(-1, 1) * v_goal.reshape(1, 2)
    curve_yaw = np.arctan2(curve_dirs[:, 1], curve_dirs[:, 0])
    curve_yaw[0] = start.yaw
    curve_yaw[-1] = goal.yaw

    line_alpha = np.linspace(0.0, 1.0, num_line + 1, dtype=np.float64)[1:]
    pre_xy = goal_xy + pre_s * u_goal
    line_xy = (1.0 - line_alpha[:, None]) * pre_xy + line_alpha[:, None] * goal_xy
    line_yaw = np.full((num_line,), goal.yaw, dtype=np.float64)

    vehicle_xy = np.vstack([curve_xy, line_xy])
    vehicle_yaw = np.concatenate([curve_yaw, line_yaw])
    meta = {
        "start_s": start_s,
        "start_y": start_y,
        "pre_s": pre_s,
        "curve_span_s": span_s,
    }
    return vehicle_xy, vehicle_yaw, meta


def _world_to_rs_goal(start: Pose2D, goal: Pose2D) -> tuple[float, float, float]:
    dx = goal.x - start.x
    dy = goal.y - start.y
    cos_t = math.cos(start.yaw)
    sin_t = math.sin(start.yaw)
    lx = dx * cos_t + dy * sin_t
    ly = -dx * sin_t + dy * cos_t
    lphi = wrap_angle(goal.yaw - start.yaw)
    x_rs_goal = -lx
    y_rs_goal = -ly
    return x_rs_goal, y_rs_goal, lphi


def plan_root_path_first_to_front_goal(
    *,
    start: Pose2D,
    pallet_xy: np.ndarray,
    pallet_yaw: float,
    pallet_depth_m: float,
    fork_reach_m: float,
    fork_front_stop_buffer_m: float,
    curve_min_span_m: float = 0.35,
    final_straight_min_m: float = 0.10,
    num_samples: int = 64,
) -> TrajectoryPlan:
    vehicle_to_fc = vehicle_to_fork_center(fork_reach_m)
    goal_pose, fork_goal_xy = compute_aligned_front_goal_pose(
        pallet_xy=pallet_xy,
        pallet_yaw=pallet_yaw,
        pallet_depth_m=pallet_depth_m,
        vehicle_to_fork_center_m=vehicle_to_fc,
        fork_front_stop_buffer_m=fork_front_stop_buffer_m,
    )
    vehicle_xy, vehicle_yaw, meta = _sample_cubic_alignment_vehicle_path(
        start=start,
        goal=goal_pose,
        curve_min_span_m=curve_min_span_m,
        final_straight_min_m=final_straight_min_m,
        num_samples=num_samples,
    )
    fork_xy, fork_yaw = map_vehicle_path_to_fork_center(
        vehicle_xy,
        vehicle_yaw,
        vehicle_to_fork_center_m=vehicle_to_fc,
    )
    return TrajectoryPlan(
        model="root_path_first",
        vehicle_xy=vehicle_xy,
        vehicle_yaw=vehicle_yaw,
        fork_center_xy=fork_xy,
        fork_center_yaw=fork_yaw,
        goal_pose=goal_pose,
        metadata={
            "fork_goal_xy": fork_goal_xy.tolist(),
            "vehicle_to_fork_center_m": vehicle_to_fc,
            **meta,
        },
    )


def plan_rs_exact(
    *,
    start: Pose2D,
    goal: Pose2D,
    min_turn_radius_m: float,
    vehicle_to_fork_center_m: float,
    sample_step_m: float = 0.03,
) -> TrajectoryPlan:
    """Generate an exact Reeds-Shepp path using the repository's RS solver."""
    world_pts = exact_rs.rs_sample_path(
        start.x,
        start.y,
        start.yaw,
        goal.x,
        goal.y,
        goal.yaw,
        min_turn_radius_m,
        step=sample_step_m,
    )
    if not world_pts:
        raise RuntimeError("exact RS planner failed to sample a path")

    vehicle_xy = np.asarray([[pt[0], pt[1]] for pt in world_pts], dtype=np.float64)
    vehicle_yaw = np.asarray([wrap_angle(pt[2]) for pt in world_pts], dtype=np.float64)
    x_rs_goal, y_rs_goal, th_rs_goal = _world_to_rs_goal(start, goal)
    segs = exact_rs.rs_best_path(x_rs_goal, y_rs_goal, th_rs_goal, min_turn_radius_m)
    total_length_m = float(sum(abs(seg_len) for _, seg_len in segs))
    fork_xy, fork_yaw = map_vehicle_path_to_fork_center(
        vehicle_xy,
        vehicle_yaw,
        vehicle_to_fork_center_m=vehicle_to_fork_center_m,
    )
    pos_err = math.hypot(vehicle_xy[-1, 0] - goal.x, vehicle_xy[-1, 1] - goal.y)
    yaw_err = abs(math.degrees(wrap_angle(vehicle_yaw[-1] - goal.yaw)))
    reverse_length_m = float(sum(abs(seg_len) for _, seg_len in segs if seg_len < 0.0))
    return TrajectoryPlan(
        model="rs_exact",
        vehicle_xy=vehicle_xy,
        vehicle_yaw=vehicle_yaw,
        fork_center_xy=fork_xy,
        fork_center_yaw=fork_yaw,
        goal_pose=goal,
        metadata={
            "planner": "rs_exact",
            "segments": [{"type": seg_type, "length_m": float(seg_len)} for seg_type, seg_len in segs],
            "segment_count": len(segs),
            "total_length_m": total_length_m,
            "reverse_length_m": reverse_length_m,
            "sample_step_m": float(sample_step_m),
            "pos_err": pos_err,
            "yaw_err_deg": yaw_err,
        },
    )


def plan_rs_to_front_goal(
    *,
    start: Pose2D,
    pallet_xy: np.ndarray,
    pallet_yaw: float,
    pallet_depth_m: float,
    fork_reach_m: float,
    fork_front_stop_buffer_m: float,
    min_turn_radius_m: float,
    sample_step_m: float = 0.03,
) -> TrajectoryPlan:
    vehicle_to_fc = vehicle_to_fork_center(fork_reach_m)
    goal_pose, fork_goal_xy = compute_aligned_front_goal_pose(
        pallet_xy=pallet_xy,
        pallet_yaw=pallet_yaw,
        pallet_depth_m=pallet_depth_m,
        vehicle_to_fork_center_m=vehicle_to_fc,
        fork_front_stop_buffer_m=fork_front_stop_buffer_m,
    )
    plan = plan_rs_exact(
        start=start,
        goal=goal_pose,
        min_turn_radius_m=min_turn_radius_m,
        vehicle_to_fork_center_m=vehicle_to_fc,
        sample_step_m=sample_step_m,
    )
    plan.metadata.update(
        {
            "fork_goal_xy": fork_goal_xy.tolist(),
            "vehicle_to_fork_center_m": vehicle_to_fc,
        }
    )
    return plan


@dataclass
class _Node:
    pose: Pose2D
    g_cost: float
    f_cost: float
    key: tuple[int, int, int]
    parent: "_Node | None"
    motion_xy: np.ndarray
    motion_yaw: np.ndarray
    last_steer: int
    last_direction: int


def _discretize_pose(pose: Pose2D, *, xy_resolution_m: float, yaw_resolution_deg: float) -> tuple[int, int, int]:
    yaw_bins = 360.0 / yaw_resolution_deg
    return (
        int(round(pose.x / xy_resolution_m)),
        int(round(pose.y / xy_resolution_m)),
        int(round(math.degrees(wrap_angle(pose.yaw)) / yaw_resolution_deg)) % int(round(yaw_bins)),
    )


def _integrate_motion(
    start: Pose2D,
    *,
    steer: int,
    direction: int,
    min_turn_radius_m: float,
    primitive_length_m: float,
    integration_step_m: float,
) -> tuple[np.ndarray, np.ndarray, Pose2D]:
    step = float(integration_step_m) * float(direction)
    n_steps = max(1, int(math.ceil(abs(primitive_length_m) / max(float(integration_step_m), 1e-6))))
    xy = np.zeros((n_steps, 2), dtype=np.float64)
    yaw = np.zeros((n_steps,), dtype=np.float64)

    x = start.x
    y = start.y
    th = start.yaw
    curvature = float(steer) / max(float(min_turn_radius_m), 1e-6)

    for i in range(n_steps):
        if abs(curvature) < 1e-9:
            x += step * math.cos(th)
            y += step * math.sin(th)
        else:
            dth = curvature * step
            x += (math.sin(th + dth) - math.sin(th)) / curvature
            y += (-math.cos(th + dth) + math.cos(th)) / curvature
            th += dth
        th = wrap_angle(th)
        xy[i] = [x, y]
        yaw[i] = th
    return xy, yaw, Pose2D(x, y, th)


def _heuristic(a: Pose2D, b: Pose2D, *, min_turn_radius_m: float) -> float:
    pos = math.hypot(a.x - b.x, a.y - b.y)
    yaw = abs(wrap_angle(a.yaw - b.yaw)) * float(min_turn_radius_m)
    return pos + 0.25 * yaw


def _iter_motion_primitives() -> Iterable[tuple[int, int]]:
    for direction in (1, -1):
        for steer in (-1, 0, 1):
            yield steer, direction


def _reconstruct_path(node: _Node, *, vehicle_to_fork_center_m: float, goal_pose: Pose2D) -> TrajectoryPlan:
    seg_xy: list[np.ndarray] = []
    seg_yaw: list[np.ndarray] = []
    cur = node
    while cur.parent is not None:
        seg_xy.append(cur.motion_xy)
        seg_yaw.append(cur.motion_yaw)
        cur = cur.parent
    seg_xy.reverse()
    seg_yaw.reverse()

    vehicle_xy = [np.array([[cur.pose.x, cur.pose.y]], dtype=np.float64)]
    vehicle_yaw = [np.array([cur.pose.yaw], dtype=np.float64)]
    vehicle_xy.extend(seg_xy)
    vehicle_yaw.extend(seg_yaw)
    vehicle_xy_arr = np.vstack(vehicle_xy)
    vehicle_yaw_arr = np.concatenate(vehicle_yaw)
    fork_xy, fork_yaw = map_vehicle_path_to_fork_center(
        vehicle_xy_arr,
        vehicle_yaw_arr,
        vehicle_to_fork_center_m=vehicle_to_fork_center_m,
    )
    return TrajectoryPlan(
        model="rs_lattice",
        vehicle_xy=vehicle_xy_arr,
        vehicle_yaw=vehicle_yaw_arr,
        fork_center_xy=fork_xy,
        fork_center_yaw=fork_yaw,
        goal_pose=goal_pose,
        metadata={},
    )


def plan_rs_lattice(
    *,
    start: Pose2D,
    goal: Pose2D,
    min_turn_radius_m: float,
    vehicle_to_fork_center_m: float,
    config: LatticeConfig | None = None,
) -> TrajectoryPlan:
    """Plan a feasible car-like path with forward/reverse primitives.

    This is a Reeds-Shepp-style state-lattice planner. It is not an exact
    shortest-path closed-form solver, but it respects the same key model
    assumptions: bounded curvature and forward/reverse motions.
    """
    cfg = config or LatticeConfig()
    start_key = _discretize_pose(start, xy_resolution_m=cfg.xy_resolution_m, yaw_resolution_deg=cfg.yaw_resolution_deg)
    open_heap: list[tuple[float, int, _Node]] = []
    counter = 0
    start_node = _Node(
        pose=start,
        g_cost=0.0,
        f_cost=_heuristic(start, goal, min_turn_radius_m=min_turn_radius_m),
        key=start_key,
        parent=None,
        motion_xy=np.zeros((0, 2), dtype=np.float64),
        motion_yaw=np.zeros((0,), dtype=np.float64),
        last_steer=0,
        last_direction=1,
    )
    heapq.heappush(open_heap, (start_node.f_cost, counter, start_node))
    best_cost: dict[tuple[int, int, int], float] = {start_key: 0.0}

    min_x = min(start.x, goal.x) - cfg.bounds_margin_m
    max_x = max(start.x, goal.x) + cfg.bounds_margin_m
    min_y = min(start.y, goal.y) - cfg.bounds_margin_m
    max_y = max(start.y, goal.y) + cfg.bounds_margin_m

    expanded = 0
    while open_heap:
        _, _, node = heapq.heappop(open_heap)
        expanded += 1
        if expanded > cfg.max_nodes:
            break

        pos_err = math.hypot(node.pose.x - goal.x, node.pose.y - goal.y)
        yaw_err = abs(math.degrees(wrap_angle(node.pose.yaw - goal.yaw)))
        if pos_err <= cfg.pos_tolerance_m and yaw_err <= cfg.yaw_tolerance_deg:
            plan = _reconstruct_path(node, vehicle_to_fork_center_m=vehicle_to_fork_center_m, goal_pose=goal)
            plan.metadata.update(
                {
                    "expanded_nodes": expanded,
                    "pos_err": pos_err,
                    "yaw_err_deg": yaw_err,
                    "planner": "rs_state_lattice",
                }
            )
            return plan

        for steer, direction in _iter_motion_primitives():
            motion_xy, motion_yaw, next_pose = _integrate_motion(
                node.pose,
                steer=steer,
                direction=direction,
                min_turn_radius_m=min_turn_radius_m,
                primitive_length_m=cfg.primitive_length_m,
                integration_step_m=cfg.integration_step_m,
            )
            if (
                np.any(motion_xy[:, 0] < min_x)
                or np.any(motion_xy[:, 0] > max_x)
                or np.any(motion_xy[:, 1] < min_y)
                or np.any(motion_xy[:, 1] > max_y)
            ):
                continue

            key = _discretize_pose(
                next_pose,
                xy_resolution_m=cfg.xy_resolution_m,
                yaw_resolution_deg=cfg.yaw_resolution_deg,
            )
            step_cost = cfg.primitive_length_m * (cfg.reverse_penalty if direction < 0 else 1.0)
            if steer != 0:
                step_cost += cfg.steer_penalty
            if steer != node.last_steer:
                step_cost += cfg.steer_change_penalty
            if direction != node.last_direction:
                step_cost += cfg.direction_change_penalty
            g_cost = node.g_cost + step_cost

            if g_cost >= best_cost.get(key, float("inf")):
                continue
            best_cost[key] = g_cost
            f_cost = g_cost + _heuristic(next_pose, goal, min_turn_radius_m=min_turn_radius_m)
            counter += 1
            heapq.heappush(
                open_heap,
                (
                    f_cost,
                    counter,
                    _Node(
                        pose=next_pose,
                        g_cost=g_cost,
                        f_cost=f_cost,
                        key=key,
                        parent=node,
                        motion_xy=motion_xy,
                        motion_yaw=motion_yaw,
                        last_steer=steer,
                        last_direction=direction,
                    ),
                ),
            )

    raise RuntimeError(
        "RS lattice planner failed to find a path. "
        "Try increasing bounds margin, max_nodes, or relaxing tolerances."
    )
