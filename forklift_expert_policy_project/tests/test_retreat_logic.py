#!/usr/bin/env python3
"""
Unit tests for expert policy retreat logic (v5-A: lat_true).

Tests can run WITHOUT IsaacLab/Isaac Sim — only needs numpy.
Validates: lat_true computation, steer direction, proportional control,
           alignment-based exit, false-trigger removal, and cooldown blocking.

Usage:
    PYTHONPATH=forklift_expert_policy_project:$PYTHONPATH python3 -m pytest tests/test_retreat_logic.py -v
    # or without pytest:
    PYTHONPATH=forklift_expert_policy_project:$PYTHONPATH python3 tests/test_retreat_logic.py
"""
import math
import json
import os
import sys
import numpy as np

# ---- Import policy ----
from forklift_expert.expert_policy import ForkliftExpertPolicy, ExpertConfig

# ---- Load specs ----
_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
with open(os.path.join(_BASE, "forklift_expert", "obs_spec.json")) as f:
    OBS_SPEC = json.load(f)
with open(os.path.join(_BASE, "forklift_expert", "action_spec.json")) as f:
    ACTION_SPEC = json.load(f)

FIELDS = OBS_SPEC["fields"]

# pallet_half_depth from ExpertConfig
_HALF_D = ExpertConfig().pallet_half_depth  # 1.08


def _make_obs(
    d_x: float = 3.0,
    d_y: float = 0.0,
    cos_dyaw: float = 1.0,
    sin_dyaw: float = 0.0,
    v_forward: float = 0.0,
    yaw_rate: float = 0.0,
    insert_norm: float = 0.0,
    y_err_obs: float = 0.0,
    yaw_err_obs: float = 0.0,
    lift_pos: float = 0.0,
) -> np.ndarray:
    """Build a 15-D obs vector with specified semantic values.

    Note (v5-A): lat is now computed from d_x/d_y/cos_dyaw/sin_dyaw,
    NOT from y_err_obs.  When dyaw=0: lat_true = -d_y.
    """
    obs = np.zeros(15, dtype=np.float32)
    obs[FIELDS["d_xy_r_x"]] = d_x
    obs[FIELDS["d_xy_r_y"]] = d_y
    obs[FIELDS["cos_dyaw"]] = cos_dyaw
    obs[FIELDS["sin_dyaw"]] = sin_dyaw
    obs[FIELDS["v_forward"]] = v_forward
    obs[FIELDS["yaw_rate"]] = yaw_rate
    obs[FIELDS["insert_norm"]] = insert_norm
    obs[FIELDS["y_err_obs"]] = y_err_obs
    obs[FIELDS["yaw_err_obs"]] = yaw_err_obs
    obs[FIELDS["lift_pos"]] = lift_pos
    return obs


def _make_obs_for_lat(
    lat_desired: float,
    dist_front: float = 2.0,
    dyaw: float = 0.0,
    insert_norm: float = 0.0,
) -> np.ndarray:
    """Helper: build obs that produces a specific lat_true and dist_front.

    When dyaw=0: lat_true = sin(0)*d_x - cos(0)*d_y = -d_y
    So d_y = -lat_desired.

    Also sets y_err_obs consistently (clipped version).
    """
    d_x = dist_front + _HALF_D
    c = math.cos(dyaw)
    s = math.sin(dyaw)
    # lat_true = s * d_x - c * d_y = lat_desired
    # => d_y = (s * d_x - lat_desired) / c  (when c != 0)
    if abs(c) > 1e-6:
        d_y = (s * d_x - lat_desired) / c
    else:
        d_y = 0.0

    # Compute y_err_obs (clipped version for consistency)
    lat_clipped = max(-0.5, min(0.5, lat_desired))
    y_err_obs = lat_clipped / 0.5  # normalized, clipped to [-1, 1]

    return _make_obs(
        d_x=d_x, d_y=d_y,
        cos_dyaw=c, sin_dyaw=s,
        insert_norm=insert_norm,
        y_err_obs=y_err_obs,
    )


def _make_policy() -> ForkliftExpertPolicy:
    return ForkliftExpertPolicy(OBS_SPEC, ACTION_SPEC, ExpertConfig())


# =====================================================================
# Test Cases
# =====================================================================

def test_case_0_lat_true_computation():
    """Verify lat_true is computed correctly from d_x/d_y/dyaw."""
    policy = _make_policy()

    # Case A: dyaw=0, d_y=-0.3 -> lat_true = +0.3
    obs_a = _make_obs(d_x=2.0, d_y=-0.3, cos_dyaw=1.0, sin_dyaw=0.0, y_err_obs=0.6)
    _, info_a = policy.act(obs_a)
    assert abs(info_a["lat"] - 0.3) < 0.01, f"Expected lat=0.3, got {info_a['lat']:.4f}"
    assert abs(info_a["lat_clipped"] - 0.3) < 0.01

    # Case B: saturated — d_y=-1.2 -> lat_true = +1.2, but lat_clipped = 0.5
    policy.reset()
    obs_b = _make_obs(d_x=2.0, d_y=-1.2, cos_dyaw=1.0, sin_dyaw=0.0, y_err_obs=1.0)
    _, info_b = policy.act(obs_b)
    assert abs(info_b["lat"] - 1.2) < 0.01, f"Expected lat_true=1.2, got {info_b['lat']:.4f}"
    assert abs(info_b["lat_clipped"] - 0.5) < 0.01, f"Expected lat_clipped=0.5, got {info_b['lat_clipped']:.4f}"

    # Case C: with yaw offset (dyaw=30deg)
    policy.reset()
    dyaw = math.radians(30)
    c, s = math.cos(dyaw), math.sin(dyaw)
    obs_c = _make_obs(d_x=2.0, d_y=0.5, cos_dyaw=c, sin_dyaw=s)
    _, info_c = policy.act(obs_c)
    expected = s * 2.0 - c * 0.5
    assert abs(info_c["lat"] - expected) < 0.01, f"Expected lat={expected:.4f}, got {info_c['lat']:.4f}"

    print(f"  PASS: lat_true computation verified (aligned, saturated, yaw-offset)")


def test_case_1_retreat_steer_direction():
    """lat=+0.5 (right offset) during retreat -> steer should be > 0."""
    policy = _make_policy()

    # dist_front=0.8, lat_true=+0.5
    obs = _make_obs_for_lat(lat_desired=0.5, dist_front=0.8)
    action, info = policy.act(obs)
    assert info["stage"] == "retreat", f"Expected retreat, got {info['stage']}"
    assert info["raw_steer"] > 0, f"Expected positive steer for lat>0, got {info['raw_steer']:.3f}"
    print(f"  PASS: lat=+0.5 -> retreat steer = {info['raw_steer']:.3f} (positive)")


def test_case_2_retreat_steer_proportional():
    """Larger |lat| should produce larger |steer| during retreat.
    v5-A: lat_true can exceed 0.5, so we test 0.49 vs 0.8."""
    # Policy with lat=0.49 (just above 0.48 threshold)
    p1 = _make_policy()
    obs_small = _make_obs_for_lat(lat_desired=0.49, dist_front=0.8)
    _, info_small = p1.act(obs_small)

    # Policy with lat=0.8 (would be clipped to 0.5 in v4, now seen as 0.8)
    p2 = _make_policy()
    obs_large = _make_obs_for_lat(lat_desired=0.8, dist_front=0.8)
    _, info_large = p2.act(obs_large)

    assert info_small["stage"] == "retreat", f"Expected retreat for lat=0.49, got {info_small['stage']}"
    assert info_large["stage"] == "retreat", f"Expected retreat for lat=0.8, got {info_large['stage']}"
    assert abs(info_large["raw_steer"]) > abs(info_small["raw_steer"]), (
        f"Expected |steer| for lat=0.8 ({abs(info_large['raw_steer']):.3f}) > "
        f"lat=0.49 ({abs(info_small['raw_steer']):.3f})"
    )
    print(f"  PASS: steer(lat=0.8)={info_large['raw_steer']:.3f} > steer(lat=0.49)={info_small['raw_steer']:.3f}")


def test_case_3_retreat_exit_on_alignment():
    """Retreat should exit early when lateral alignment improves 40%+.
    v5-A: entry lat can be > 0.5 (e.g., 0.7), exit when drops to 0.3."""
    policy = _make_policy()

    # Step 1: trigger retreat with lat=0.7, dist_front=0.8
    obs_start = _make_obs_for_lat(lat_desired=0.7, dist_front=0.8)
    _, info = policy.act(obs_start)
    assert info["stage"] == "retreat", f"Should start in retreat, got {info['stage']}"
    assert policy._retreat_entry_lat > 0.65, f"Entry lat should be ~0.7, got {policy._retreat_entry_lat}"

    # Step 2: lat improves to 0.25, dist increases to 1.3
    # 0.25 < 0.7 * 0.6 = 0.42 (40%+ improvement)
    # 0.25 < 0.30 (absolute range ok)
    # dist 1.3 > 1.2 (enough room)
    obs_improved = _make_obs_for_lat(lat_desired=0.25, dist_front=1.3)
    _, info2 = policy.act(obs_improved)

    assert info2["stage"] != "retreat", (
        f"Expected exit from retreat (lat improved 0.7->0.25), but stage={info2['stage']}"
    )
    print(f"  PASS: retreat exited early when lat improved from 0.7 to 0.25 (entry_lat unsaturated!)")


def test_case_4_no_false_trigger():
    """lat_true=0.3 (below 0.48 threshold) should NOT trigger retreat,
    even with dist < 1.0."""
    policy = _make_policy()

    # dist_front=0.5, lat_true=0.3 (below 0.48)
    obs = _make_obs_for_lat(lat_desired=0.3, dist_front=0.5)
    _, info = policy.act(obs)

    assert info["stage"] != "retreat", (
        f"Should NOT trigger retreat for lat=0.3 (below 0.48 threshold), "
        f"but got stage={info['stage']}"
    )
    print(f"  PASS: lat_true=0.3 -> no retreat trigger (stage={info['stage']})")


def test_case_5_cooldown_blocks_retrigger():
    """After retreat ends, cooldown should block immediate re-triggering."""
    policy = _make_policy()
    cfg = policy.cfg

    # Step 1: trigger and run retreat to completion (max_retreat_steps)
    obs_trigger = _make_obs_for_lat(lat_desired=0.6, dist_front=0.8)
    for i in range(cfg.max_retreat_steps + 5):
        _, info = policy.act(obs_trigger)
        if info["stage"] != "retreat" and i > 0:
            break

    assert policy._retreat_cooldown_remaining > 0, (
        f"Cooldown should be active after retreat, got {policy._retreat_cooldown_remaining}"
    )

    # Step 2: immediately present same trigger conditions
    _, info_after = policy.act(obs_trigger)
    assert info_after["stage"] != "retreat", (
        f"Cooldown should block retreat re-trigger, but got stage={info_after['stage']}"
    )
    print(f"  PASS: cooldown={policy._retreat_cooldown_remaining} blocks re-trigger (stage={info_after['stage']})")


def test_case_6_large_lat_triggers_stronger_response():
    """v5-A key benefit: lat_true=1.0 should produce stronger docking steer
    than lat_true=0.5 (which was the old v4 ceiling)."""
    # Policy with lat=0.5 (v4 maximum)
    p1 = _make_policy()
    obs_v4max = _make_obs_for_lat(lat_desired=0.5, dist_front=2.0)
    _, info_v4 = p1.act(obs_v4max)

    # Policy with lat=1.0 (v5-A can see this)
    p2 = _make_policy()
    obs_v5 = _make_obs_for_lat(lat_desired=1.0, dist_front=2.0)
    _, info_v5 = p2.act(obs_v5)

    assert info_v4["stage"] == "docking"
    assert info_v5["stage"] == "docking"
    assert abs(info_v5["raw_steer"]) > abs(info_v4["raw_steer"]), (
        f"v5-A should produce stronger steer for lat=1.0 ({abs(info_v5['raw_steer']):.3f}) "
        f"vs lat=0.5 ({abs(info_v4['raw_steer']):.3f})"
    )
    print(f"  PASS: steer(lat=1.0)={info_v5['raw_steer']:.3f} > steer(lat=0.5)={info_v4['raw_steer']:.3f}")


# =====================================================================
# Main (fallback for no-pytest environments)
# =====================================================================
def main():
    tests = [
        test_case_0_lat_true_computation,
        test_case_1_retreat_steer_direction,
        test_case_2_retreat_steer_proportional,
        test_case_3_retreat_exit_on_alignment,
        test_case_4_no_false_trigger,
        test_case_5_cooldown_blocks_retrigger,
        test_case_6_large_lat_triggers_stronger_response,
    ]
    passed = 0
    failed = 0
    for t in tests:
        name = t.__name__
        try:
            t()
            passed += 1
            print(f"  [OK] {name}")
        except AssertionError as e:
            failed += 1
            print(f"  [FAIL] {name}: {e}")
        except Exception as e:
            failed += 1
            print(f"  [ERROR] {name}: {type(e).__name__}: {e}")

    print(f"\n{'='*60}")
    print(f"  Results: {passed} passed, {failed} failed, {passed+failed} total")
    print(f"{'='*60}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
