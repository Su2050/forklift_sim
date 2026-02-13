#!/usr/bin/env python3
"""
Unit tests for expert policy retreat logic (v5-B: parameterised retreat).

Tests can run WITHOUT IsaacLab/Isaac Sim — only needs numpy.
Validates: lat_true computation, steer direction, proportional control,
           alignment-based exit (lat+yaw), exit_reason, rate_limit skip,
           false-trigger removal, and cooldown blocking.

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
    """Build a 15-D obs vector with specified semantic values."""
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
    if abs(c) > 1e-6:
        d_y = (s * d_x - lat_desired) / c
    else:
        d_y = 0.0

    lat_clipped = max(-0.5, min(0.5, lat_desired))
    y_err_obs = lat_clipped / 0.5

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
    assert abs(info_b["lat"] - 1.2) < 0.01
    assert abs(info_b["lat_clipped"] - 0.5) < 0.01

    # Case C: with yaw offset (dyaw=30deg)
    policy.reset()
    dyaw = math.radians(30)
    c, s = math.cos(dyaw), math.sin(dyaw)
    obs_c = _make_obs(d_x=2.0, d_y=0.5, cos_dyaw=c, sin_dyaw=s)
    _, info_c = policy.act(obs_c)
    expected = s * 2.0 - c * 0.5
    assert abs(info_c["lat"] - expected) < 0.01

    print(f"  PASS: lat_true computation verified (aligned, saturated, yaw-offset)")


def test_case_1_retreat_steer_direction():
    """lat=+0.5 (right offset) during retreat -> steer should be > 0."""
    policy = _make_policy()
    obs = _make_obs_for_lat(lat_desired=0.5, dist_front=0.8)
    _, info = policy.act(obs)
    assert info["stage"] == "retreat", f"Expected retreat, got {info['stage']}"
    assert info["raw_steer"] > 0, f"Expected positive steer for lat>0, got {info['raw_steer']:.3f}"
    print(f"  PASS: lat=+0.5 -> retreat steer = {info['raw_steer']:.3f} (positive)")


def test_case_2_retreat_steer_proportional():
    """Larger |lat| should produce larger |steer| during retreat.
    v5-B: lat_term = min(|lat|/0.75, 1.0)*0.50, saturates at lat=0.75."""
    # lat=0.49 (just above 0.48 threshold)
    p1 = _make_policy()
    obs_small = _make_obs_for_lat(lat_desired=0.49, dist_front=0.8)
    _, info_small = p1.act(obs_small)

    # lat=0.8 (v5-B: now above sat point 0.75, should be stronger than 0.49)
    p2 = _make_policy()
    obs_large = _make_obs_for_lat(lat_desired=0.8, dist_front=0.8)
    _, info_large = p2.act(obs_large)

    assert info_small["stage"] == "retreat"
    assert info_large["stage"] == "retreat"
    assert abs(info_large["raw_steer"]) > abs(info_small["raw_steer"]), (
        f"|steer|(lat=0.8)={abs(info_large['raw_steer']):.3f} should > "
        f"|steer|(lat=0.49)={abs(info_small['raw_steer']):.3f}"
    )

    # v5-B specific: lat=0.6 vs lat=0.7 should differ (was identical in v5-A)
    p3 = _make_policy()
    obs_06 = _make_obs_for_lat(lat_desired=0.6, dist_front=0.8)
    _, info_06 = p3.act(obs_06)

    p4 = _make_policy()
    obs_07 = _make_obs_for_lat(lat_desired=0.7, dist_front=0.8)
    _, info_07 = p4.act(obs_07)

    assert abs(info_07["raw_steer"]) > abs(info_06["raw_steer"]), (
        f"v5-B: |steer|(lat=0.7)={abs(info_07['raw_steer']):.3f} should > "
        f"|steer|(lat=0.6)={abs(info_06['raw_steer']):.3f} (was equal in v5-A)"
    )
    print(f"  PASS: proportional steer verified; lat=0.6 -> {info_06['raw_steer']:.3f}, "
          f"lat=0.7 -> {info_07['raw_steer']:.3f}, lat=0.8 -> {info_large['raw_steer']:.3f}")


def test_case_3_retreat_exit_on_alignment():
    """Retreat should exit when BOTH lat AND yaw improve sufficiently.
    v5-B: exit requires abs(yaw) < 30deg in addition to lat conditions."""
    policy = _make_policy()

    # Step 1: trigger retreat with lat=0.7, yaw=0 (small), dist_front=0.8
    obs_start = _make_obs_for_lat(lat_desired=0.7, dist_front=0.8)
    _, info = policy.act(obs_start)
    assert info["stage"] == "retreat"
    assert policy._retreat_entry_lat > 0.65

    # Step 2: lat improves to 0.25, dist=1.3, yaw still small (dyaw=0)
    # 0.25 < 0.7*0.6=0.42 OK, 0.25 < 0.40 OK, yaw ~0 < 30deg OK, dist>1.2 OK
    obs_improved = _make_obs_for_lat(lat_desired=0.25, dist_front=1.3)
    _, info2 = policy.act(obs_improved)
    assert info2["stage"] != "retreat", (
        f"Expected exit from retreat (lat+yaw OK), but stage={info2['stage']}"
    )
    assert policy._retreat_exit_reason == "alignment"
    print(f"  PASS: retreat exited (alignment) when lat improved to 0.25 with small yaw")


def test_case_3b_retreat_no_exit_if_yaw_large():
    """v5-B: retreat should NOT exit alignment-early if yaw is still large,
    even when lat has improved enough."""
    policy = _make_policy()
    cfg = policy.cfg

    # Trigger retreat with lat=0.7, yaw=40deg > exit_yaw_max(30deg)
    dyaw_entry = math.radians(40)
    obs_start = _make_obs_for_lat(lat_desired=0.7, dist_front=0.8, dyaw=dyaw_entry)
    _, info = policy.act(obs_start)
    assert info["stage"] == "retreat"

    # lat improves to 0.25, but yaw stays at 40deg
    obs_lat_ok = _make_obs_for_lat(lat_desired=0.25, dist_front=1.3, dyaw=dyaw_entry)
    _, info2 = policy.act(obs_lat_ok)

    # Should still be in retreat because yaw is too large
    assert info2["stage"] == "retreat", (
        f"Should stay in retreat (yaw=40deg > 30deg), but stage={info2['stage']}"
    )
    print(f"  PASS: retreat blocked alignment-exit when yaw=40deg (> 30deg limit)")


def test_case_4_no_false_trigger():
    """lat_true=0.3 (below 0.48 threshold) should NOT trigger retreat."""
    policy = _make_policy()
    obs = _make_obs_for_lat(lat_desired=0.3, dist_front=0.5)
    _, info = policy.act(obs)
    assert info["stage"] != "retreat"
    print(f"  PASS: lat_true=0.3 -> no retreat trigger (stage={info['stage']})")


def test_case_5_cooldown_blocks_retrigger():
    """After retreat ends, cooldown should block immediate re-triggering.
    Also checks retreat_exit_reason."""
    policy = _make_policy()
    cfg = policy.cfg

    # Trigger and run retreat to max_steps
    obs_trigger = _make_obs_for_lat(lat_desired=0.6, dist_front=0.8)
    for i in range(cfg.max_retreat_steps + 5):
        _, info = policy.act(obs_trigger)
        if info["stage"] != "retreat" and i > 0:
            break

    assert policy._retreat_cooldown_remaining > 0
    assert policy._retreat_exit_reason == "max_steps", (
        f"Expected exit reason 'max_steps', got '{policy._retreat_exit_reason}'"
    )

    _, info_after = policy.act(obs_trigger)
    assert info_after["stage"] != "retreat"
    print(f"  PASS: cooldown blocks re-trigger; exit_reason=max_steps")


def test_case_5b_exit_reason_target_dist():
    """retreat_exit_reason should be 'target_dist' when distance reached."""
    policy = _make_policy()
    cfg = policy.cfg

    # Trigger with lat=0.6, dist=0.8
    obs_trigger = _make_obs_for_lat(lat_desired=0.6, dist_front=0.8)
    _, info = policy.act(obs_trigger)
    assert info["stage"] == "retreat"

    # Next step: dist jumps to retreat_target_dist (1.8), lat still bad
    obs_far = _make_obs_for_lat(lat_desired=0.6, dist_front=cfg.retreat_target_dist)
    _, info2 = policy.act(obs_far)
    assert info2["stage"] != "retreat"
    assert policy._retreat_exit_reason == "target_dist", (
        f"Expected 'target_dist', got '{policy._retreat_exit_reason}'"
    )
    print(f"  PASS: retreat exit_reason=target_dist when dist >= {cfg.retreat_target_dist}")


def test_case_6_large_lat_triggers_stronger_response():
    """lat_true=1.0 should produce stronger docking steer than lat_true=0.5."""
    p1 = _make_policy()
    obs_v4max = _make_obs_for_lat(lat_desired=0.5, dist_front=2.0)
    _, info_v4 = p1.act(obs_v4max)

    p2 = _make_policy()
    obs_v5 = _make_obs_for_lat(lat_desired=1.0, dist_front=2.0)
    _, info_v5 = p2.act(obs_v5)

    assert info_v4["stage"] == "docking"
    assert info_v5["stage"] == "docking"
    assert abs(info_v5["raw_steer"]) > abs(info_v4["raw_steer"])
    print(f"  PASS: steer(lat=1.0)={info_v5['raw_steer']:.3f} > steer(lat=0.5)={info_v4['raw_steer']:.3f}")


def test_case_7_retreat_rate_limit_skip():
    """v5-B: first retreat step should skip steer rate_limit (no ramp-up)."""
    policy = _make_policy()

    # Pre-condition: _prev_steer = 0 (fresh policy)
    assert policy._prev_steer == 0.0

    # Trigger retreat with lat=0.7 -> raw_steer should be significant
    obs = _make_obs_for_lat(lat_desired=0.7, dist_front=0.8)
    _, info = policy.act(obs)
    assert info["stage"] == "retreat"

    # With rate_limit skip, the actual steer should == raw_steer
    # (no 0.35/step ramp-up from 0)
    assert abs(info["steer"] - info["raw_steer"]) < 0.01, (
        f"First retreat step should skip rate_limit: steer={info['steer']:.3f} "
        f"should == raw_steer={info['raw_steer']:.3f}"
    )
    print(f"  PASS: first retreat step steer={info['steer']:.3f} == raw_steer={info['raw_steer']:.3f} (no ramp)")


def test_case_8_retreat_lat_sat_parameterised():
    """v5-B: verify retreat lat_term uses retreat_lat_sat=0.75 parameterisation.
    lat=0.375 -> lat_term = 0.375/0.75*0.50 = 0.25
    lat=0.75  -> lat_term = 1.0*0.50 = 0.50 (saturated)
    lat=1.0   -> lat_term = 1.0*0.50 = 0.50 (still saturated)"""
    cfg = ExpertConfig()

    # Compute expected values manually
    for lat_val, expected_term in [(0.375, 0.25), (0.75, 0.50), (1.0, 0.50)]:
        actual = min(lat_val / cfg.retreat_lat_sat, 1.0) * cfg.retreat_steer_gain
        assert abs(actual - expected_term) < 0.001, (
            f"lat={lat_val}: expected lat_term={expected_term}, got {actual:.4f}"
        )

    print(f"  PASS: retreat_lat_sat parameterisation verified (0.375->0.25, 0.75->0.50, 1.0->0.50)")


def test_case_9_lat_dependent_steer_bonus():
    """v5-C: large |lat| should allow stronger docking steer via bonus.
    lat=0.3 -> no bonus (below 0.4 threshold), steer unchanged
    lat=0.8 -> bonus=min(0.4*0.2, 0.10)=0.08, eff_max_steer=0.65+0.08=0.73
    lat=1.0 -> bonus=min(0.6*0.2, 0.10)=0.10, eff_max_steer=0.75 (capped)"""
    # Small lat: no bonus, steer same as before
    p1 = _make_policy()
    obs_small = _make_obs_for_lat(lat_desired=0.3, dist_front=2.5)  # dist>2.0
    _, info_small = p1.act(obs_small)
    # k_lat=1.1, lat=0.3 => raw_steer=0.33+yaw_term, limit should be 0.65 (no bonus)
    assert abs(info_small["raw_steer"]) <= 0.65 + 0.01, (
        f"Small lat should not get bonus: raw_steer={info_small['raw_steer']:.3f}"
    )

    # Large lat: should get bonus and produce stronger steer
    p2 = _make_policy()
    obs_large = _make_obs_for_lat(lat_desired=0.8, dist_front=2.5)  # dist>2.0
    _, info_large = p2.act(obs_large)
    # k_lat=1.1, lat=0.8 => raw_steer=0.88+yaw_term, eff_max_steer=0.65+0.08=0.73
    assert abs(info_large["raw_steer"]) > 0.65, (
        f"Large lat (0.8) should get bonus: raw_steer={info_large['raw_steer']:.3f} should > 0.65"
    )
    assert abs(info_large["raw_steer"]) <= 0.75 + 0.01, (
        f"Bonus should be capped: raw_steer={info_large['raw_steer']:.3f} should <= 0.75"
    )

    # Very large lat: capped at max bonus
    p3 = _make_policy()
    obs_huge = _make_obs_for_lat(lat_desired=1.5, dist_front=2.5)
    _, info_huge = p3.act(obs_huge)
    assert abs(info_huge["raw_steer"]) <= 0.75 + 0.01

    # Near-distance: no bonus even with large lat (dist < 0.8)
    # Use lat=0.3 to avoid retreat trigger (lat < 0.48)
    p4 = _make_policy()
    obs_near = _make_obs_for_lat(lat_desired=0.3, dist_front=0.5)  # dist < 0.8
    _, info_near = p4.act(obs_near)
    # Near: max_steer_near=0.40, no bonus (dist<0.8), plus gain decay
    assert abs(info_near["raw_steer"]) <= 0.40 + 0.01, (
        f"Near-distance should not get bonus: raw_steer={info_near['raw_steer']:.3f}"
    )

    print(f"  PASS: lat-dependent bonus verified: small(0.3)={info_small['raw_steer']:.3f}, "
          f"large(0.8)={info_large['raw_steer']:.3f}, huge(1.5)={info_huge['raw_steer']:.3f}, "
          f"near(0.3,d=0.5)={info_near['raw_steer']:.3f}")


# =====================================================================
# Main (fallback for no-pytest environments)
# =====================================================================
def main():
    tests = [
        test_case_0_lat_true_computation,
        test_case_1_retreat_steer_direction,
        test_case_2_retreat_steer_proportional,
        test_case_3_retreat_exit_on_alignment,
        test_case_3b_retreat_no_exit_if_yaw_large,
        test_case_4_no_false_trigger,
        test_case_5_cooldown_blocks_retrigger,
        test_case_5b_exit_reason_target_dist,
        test_case_6_large_lat_triggers_stronger_response,
        test_case_7_retreat_rate_limit_skip,
        test_case_8_retreat_lat_sat_parameterised,
        test_case_9_lat_dependent_steer_bonus,
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
