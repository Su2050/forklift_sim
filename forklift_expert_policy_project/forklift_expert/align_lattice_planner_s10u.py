"""
S1.0U-friendly lattice / Hybrid-A* style planner for the *ALIGN* layer.

Why this exists
---------------
The current s1.0u expert is mostly "FSM + local PD". When the safety gate is a
BBox/corridor check (tip_lat), purely local steering can oscillate into
"docking <-> retreat" loops.

This planner adds a thin planning layer:
- Plan a *sequence of unit actions* (motion primitives): constant steer + constant gear
  for a short duration.
- Hard-check safety constraints that match s1.0u's BBox logic (corridor + hard_wall).
- Output a primitive list you can execute open-loop (with replan / MPC-style refresh).

Coordinate convention (IMPORTANT)
--------------------------------
We plan in the *pallet frame* used by s1.0u diagnostics:

- x := dist_front (meters): distance from forklift reference point to pallet FRONT plane.
       Approaching the pallet (drive forward) **DECREASES x**.
- y := lat_true (meters): lateral error in pallet center-line frame.
       +y means forklift is to the RIGHT (same sign as s1.0u).
- yaw := yaw_err (radians): atan2(sin_dyaw, cos_dyaw) from obs.
       Negative means "nose to the LEFT" in the s1.0u report.

With this convention, if we approximate the kinematics using yaw_err directly:
    x_dot ≈ -v * cos(yaw)
    y_dot ≈ -v * sin(yaw)
    yaw_dot ≈  v * tan(steer) / wheelbase

(Yes, it's a bit non-standard; it's chosen to match the observed s1.0u signals.)

No external dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass
from heapq import heappush, heappop
import math
from typing import List, Optional, Tuple, Dict, Iterable


# -------------------------
# Small helpers
# -------------------------

def _clip(x: float, lo: float, hi: float) -> float:
    return float(min(max(x, lo), hi))


def _wrap_pi(a: float) -> float:
    # Wrap to [-pi, pi]
    a = (a + math.pi) % (2 * math.pi) - math.pi
    return float(a)


# -------------------------
# Planner definitions
# -------------------------

@dataclass(frozen=True)
class Pose2D:
    x: float
    y: float
    yaw: float


@dataclass(frozen=True)
class Primitive:
    """A unit action: constant steer, constant gear/speed, fixed duration."""
    steer: float      # normalized steer command (matches s1.0u action sign)
    v: float          # signed speed in m/s (+forward toward pallet, -reverse away)
    duration: float   # seconds


@dataclass
class PlannerParams:
    # Kinematics
    wheelbase: float = 1.6
    dt: float = 1.0/30.0            # match env control dt (sim 1/120, decimation 4)

    # Primitive set
    max_steer: float = 0.65          # match cfg.max_steer_far by default
    steer_levels: Tuple[float, ...] = (-1.0, -0.5, 0.0, 0.5, 1.0)  # scaled by max_steer

    v_fwd: float = 0.25              # m/s, representative approach speed
    v_rev: float = -0.25             # m/s, representative reverse speed
    durations: Tuple[float, ...] = (0.33, 0.50, 0.67)  # shorter options for tight-space maneuvers

    # Discretization (for visited / closed set)
    x_res: float = 0.07
    y_res: float = 0.07
    yaw_bins: int = 72

    # Search
    max_expansions: int = 12000
    w_yaw_h: float = 0.45
    reverse_penalty: float = 1.5
    steer_change_penalty: float = 0.30

    # Bounding window (avoid infinite search); in meters in this pallet-front frame
    x_min: float = 0.0
    x_max: float = 3.0
    y_max_abs: float = 3.0

    # -------------------------
    # Safety constraints (match s1.0u BBox logic)
    # -------------------------
    use_corridor_gate: bool = True

    fork_reach: float = 1.87
    fork_tip_lat_ok: float = 0.15
    corridor_slope: float = 0.8

    pre_insert_dist: float = 2.05
    hard_wall: float = 1.92

    # If you want to allow the planner to go as close as hard_wall (rare for ALIGN),
    # set forbid_inside_hard_wall=False and keep the alignment gates.
    forbid_inside_hard_wall: bool = True
    final_lat_ok: float = 0.15
    final_yaw_ok: float = math.radians(10.0)

    # Goal region ("align box") in this frame
    goal_x: float = 2.50
    goal_y: float = 0.00
    goal_dx: float = 0.12
    goal_dy: float = 0.12
    goal_dyaw: float = math.radians(5.0)

    # Sampling for collision checking along a primitive
    collision_check_substeps: int = 6


def safe_corridor(dist_front: float, prm: PlannerParams) -> float:
    """Same corridor shape as s1.0u:
        fork_tip_lat_ok + max(0, dist - fork_reach) * corridor_slope
    """
    return prm.fork_tip_lat_ok + max(0.0, dist_front - prm.fork_reach) * prm.corridor_slope


def tip_lat(p: Pose2D, prm: PlannerParams) -> float:
    # Same as s1.0u: tip_lat = lat - fork_reach*sin(yaw)
    return p.y - prm.fork_reach * math.sin(p.yaw)


def in_goal_region(p: Pose2D, prm: PlannerParams) -> bool:
    if abs(p.x - prm.goal_x) > prm.goal_dx:
        return False
    if abs(p.y - prm.goal_y) > prm.goal_dy:
        return False
    if abs(_wrap_pi(p.yaw)) > prm.goal_dyaw:
        return False
    return True


def pose_to_key(p: Pose2D, prm: PlannerParams) -> Tuple[int, int, int]:
    xi = int(round(p.x / prm.x_res))
    yi = int(round(p.y / prm.y_res))
    yaw = _wrap_pi(p.yaw)
    b = int(round(((yaw + math.pi) / (2 * math.pi)) * prm.yaw_bins)) % prm.yaw_bins
    return xi, yi, b


def _collision_free_point(p: Pose2D, prm: PlannerParams) -> bool:
    """Point-wise safety (checked at each substep)."""
    # Window limits
    if p.x < prm.x_min or p.x > prm.x_max:
        return False
    if abs(p.y) > prm.y_max_abs:
        return False

    # Hard wall (ALIGN generally shouldn't go inside)
    if prm.forbid_inside_hard_wall and p.x <= prm.hard_wall:
        return False

    if not prm.use_corridor_gate:
        return True

    # Corridor gate only matters in near-field
    if p.x < prm.pre_insert_dist:
        tl = tip_lat(p, prm)
        if abs(tl) > safe_corridor(p.x, prm):
            return False

    # Optional: if you do allow going inside hard_wall, enforce alignment gates
    if (not prm.forbid_inside_hard_wall) and p.x <= prm.hard_wall:
        tl = tip_lat(p, prm)
        body_ok = (abs(p.y) <= prm.final_lat_ok and abs(p.yaw) <= prm.final_yaw_ok)
        tip_ok = (abs(tl) <= prm.fork_tip_lat_ok)
        if not (body_ok and tip_ok):
            return False

    return True


def rollout_with_path(p: Pose2D, prim: Primitive, prm: PlannerParams) -> Tuple[Pose2D, List[Pose2D]]:
    """Rollout under constant steer and v. Returns (final_pose, sampled_path_including_final)."""

    # Use fixed substeps (dt) but we also sub-sample for collision checks.
    x, y, yaw = p.x, p.y, p.yaw
    t = 0.0

    path: List[Pose2D] = []
    # Keep a coarse sampling for collision: collision_check_substeps per primitive duration.
    sample_every = max(1, int(round((prim.duration / prm.dt) / max(prm.collision_check_substeps, 1))))

    step_i = 0
    while t < prim.duration - 1e-9:
        v = prim.v
        kappa = math.tan(prim.steer) / prm.wheelbase

        # Yaw error dynamics (match s1.0u sign convention)
        yaw = _wrap_pi(yaw + v * kappa * prm.dt)

        # Position update in dist_front frame:
        # Approach (v>0) decreases x and decreases y by sin(yaw).
        x = x - v * math.cos(yaw) * prm.dt
        y = y - v * math.sin(yaw) * prm.dt

        t += prm.dt
        step_i += 1

        if step_i % sample_every == 0:
            path.append(Pose2D(x, y, yaw))

    if not path:
        path.append(Pose2D(x, y, yaw))
    else:
        # ensure final pose is included
        if path[-1].x != x or path[-1].y != y or path[-1].yaw != yaw:
            path.append(Pose2D(x, y, yaw))

    return Pose2D(x, y, yaw), path


def collision_free_path(path: Iterable[Pose2D], prm: PlannerParams) -> bool:
    for p in path:
        if not _collision_free_point(p, prm):
            return False
    return True


def heuristic(p: Pose2D, prm: PlannerParams) -> float:
    dx = p.x - prm.goal_x
    dy = p.y - prm.goal_y
    d = math.hypot(dx, dy)
    # Soft penalty: discourage backing too far away from goal_x
    if p.x > prm.goal_x + 0.5:
        d += 0.3 * (p.x - prm.goal_x - 0.5)
    yaw_pen = prm.w_yaw_h * abs(_wrap_pi(p.yaw))
    return d + yaw_pen


def make_primitives(prm: PlannerParams) -> List[Primitive]:
    steers = [s * prm.max_steer for s in prm.steer_levels]
    prims: List[Primitive] = []
    for dur in prm.durations:
        for s in steers:
            prims.append(Primitive(steer=s, v=prm.v_fwd, duration=dur))
            prims.append(Primitive(steer=s, v=prm.v_rev, duration=dur))
    return prims


@dataclass
class PlanResult:
    primitives: List[Primitive]
    final_pose: Pose2D
    expansions: int
    reason: str = ""
    # Optional: a coarse predicted path (poses) for debugging
    predicted_path: Optional[List[Pose2D]] = None


def plan_align(start: Pose2D, prm: PlannerParams) -> Optional[PlanResult]:
    """A* search over motion primitives. Returns PlanResult or None."""

    prims = make_primitives(prm)

    # Early safety check
    if not _collision_free_point(start, prm):
        return None

    start_key = pose_to_key(start, prm)

    open_heap: List[Tuple[float, int, Tuple[int, int, int]]] = []
    g_cost: Dict[Tuple[int, int, int], float] = {start_key: 0.0}
    parent: Dict[Tuple[int, int, int], Tuple[Tuple[int, int, int], Primitive, Pose2D]] = {}

    heappush(open_heap, (heuristic(start, prm), 0, start_key))

    rep_pose: Dict[Tuple[int, int, int], Pose2D] = {start_key: start}

    expansions = 0
    tie = 0

    while open_heap and expansions < prm.max_expansions:
        _, _, key = heappop(open_heap)
        cur = rep_pose[key]

        if in_goal_region(cur, prm):
            # Reconstruct primitive sequence
            seq: List[Primitive] = []
            k = key
            while k != start_key:
                pk, prim, _pose = parent[k]
                seq.append(prim)
                k = pk
            seq.reverse()

            # Also build a coarse predicted path (optional but handy)
            p = start
            pred: List[Pose2D] = [p]
            for prim in seq:
                p, path = rollout_with_path(p, prim, prm)
                pred.extend(path)

            return PlanResult(
                primitives=seq,
                final_pose=cur,
                expansions=expansions,
                reason="ok",
                predicted_path=pred,
            )

        expansions += 1

        for prim in prims:
            nxt, path = rollout_with_path(cur, prim, prm)

            # Hard safety along the arc
            if not collision_free_path(path, prm):
                continue

            nk = pose_to_key(nxt, prm)

            # Cost: distance traveled (approx)
            step_cost = abs(prim.v) * prim.duration

            # Reverse penalty
            if prim.v < 0:
                step_cost += prm.reverse_penalty * abs(prim.v) * prim.duration

            # Penalize steer flips (smoother plans)
            if key in parent:
                prev_prim = parent[key][1]
                step_cost += prm.steer_change_penalty * abs(prim.steer - prev_prim.steer)

            ng = g_cost[key] + step_cost

            if nk not in g_cost or ng < g_cost[nk]:
                g_cost[nk] = ng
                rep_pose[nk] = nxt
                parent[nk] = (key, prim, nxt)

                tie += 1
                nf = ng + heuristic(nxt, prm)
                heappush(open_heap, (nf, tie, nk))

    return None


# -------------------------
# Quick sanity demo
# -------------------------
if __name__ == "__main__":
    prm = PlannerParams()
    # Use the same "trouble case" as the s1.0u report:
    start = Pose2D(x=2.865, y=0.382, yaw=math.radians(-5.3))
    res = plan_align(start, prm)
    if res is None:
        print("No plan found.")
    else:
        print(f"Plan length: {len(res.primitives)} primitives, expansions={res.expansions}")
        for i, prim in enumerate(res.primitives[:12]):
            print(i, prim)
        print("final:", res.final_pose)
