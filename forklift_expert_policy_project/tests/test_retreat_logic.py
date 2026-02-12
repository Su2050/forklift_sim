#!/usr/bin/env python3
"""
Unit tests for expert policy retreat logic (v4 redesign).

Tests can run WITHOUT IsaacLab/Isaac Sim — only needs numpy.
Validates: steer direction, proportional control, alignment-based exit,
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


def _make_policy() -> ForkliftExpertPolicy:
    return ForkliftExpertPolicy(OBS_SPEC, ACTION_SPEC, ExpertConfig())


# =====================================================================
# Test Cases
# =====================================================================

def test_case_1_retreat_steer_direction():
    """lat=+0.5 (right offset) during retreat -> steer should be > 0.
    Ackermann reverse: positive steer + backward drive swings rear left,
    which moves the forklift left to reduce positive lateral error."""
    policy = _make_policy()

    # Force into retreat: dist_front < 1.0, |lat| >= 0.48
    # d_x = 1.08 + 0.8 = 1.88 -> dist_front = 0.8
    obs = _make_obs(d_x=1.88, y_err_obs=1.0)  # lat = 0.5m

    # First call: trigger retreat
    action, info = policy.act(obs)
    assert info["stage"] == "retreat", f"Expected retreat, got {info['stage']}"
    # Steer should be positive for positive lat
    assert info["raw_steer"] > 0, f"Expected positive steer for lat>0, got {info['raw_steer']:.3f}"
    print(f"  PASS: lat=+0.5 -> retreat steer = {info['raw_steer']:.3f} (positive)")


def test_case_2_retreat_steer_proportional():
    """Larger |lat| should produce larger |steer| during retreat."""
    # Policy with lat ~= 0.485 (just above 0.48 threshold, accounting for float32)
    p1 = _make_policy()
    obs_small = _make_obs(d_x=1.88, y_err_obs=0.97)  # lat=0.485
    _, info_small = p1.act(obs_small)

    # Policy with lat=0.5
    p2 = _make_policy()
    obs_large = _make_obs(d_x=1.88, y_err_obs=1.0)  # lat=0.5
    _, info_large = p2.act(obs_large)

    assert info_small["stage"] == "retreat", f"Expected retreat for lat=0.485, got {info_small['stage']}"
    assert info_large["stage"] == "retreat", f"Expected retreat for lat=0.5, got {info_large['stage']}"
    assert abs(info_large["raw_steer"]) >= abs(info_small["raw_steer"]), (
        f"Expected |steer| for lat=0.5 ({abs(info_large['raw_steer']):.3f}) >= "
        f"lat=0.485 ({abs(info_small['raw_steer']):.3f})"
    )
    print(f"  PASS: steer(lat=0.5)={info_large['raw_steer']:.3f} >= steer(lat=0.485)={info_small['raw_steer']:.3f}")


def test_case_3_retreat_exit_on_alignment():
    """Retreat should exit early when lateral alignment improves 40%+."""
    policy = _make_policy()
    cfg = policy.cfg

    # Step 1: trigger retreat with lat~=0.49, dist_front=0.8
    obs_start = _make_obs(d_x=1.88, y_err_obs=0.98)  # lat=0.49
    _, info = policy.act(obs_start)
    assert info["stage"] == "retreat", f"Should start in retreat, got {info['stage']}"
    assert policy._retreat_entry_lat > 0.45, f"Entry lat should be ~0.49, got {policy._retreat_entry_lat}"

    # Step 2: simulate a few retreat steps where lat improves
    # After backing up, suppose lat drops to 0.25 and dist increases to 1.3
    # (0.25 < 0.49 * 0.6 = 0.294, and 0.25 < 0.30, and dist > 1.2)
    # d_x for dist_front=1.3: d_x = 1.3 + 1.08 = 2.38
    obs_improved = _make_obs(d_x=2.38, y_err_obs=0.50)  # lat=0.25
    _, info2 = policy.act(obs_improved)

    # Should have exited retreat (alignment_improved = True)
    assert info2["stage"] != "retreat", (
        f"Expected exit from retreat (lat improved to 0.25), but stage={info2['stage']}"
    )
    print(f"  PASS: retreat exited early when lat improved from 0.48 to 0.25")


def test_case_4_no_false_trigger():
    """lat=0.3 (below 0.48 threshold) but d_y=0.7 (large robot-frame offset).
    Old code would trigger retreat via lat_unsaturated; v4 should NOT."""
    policy = _make_policy()

    # dist_front=0.5 (close), lat=0.3 (below 0.48), d_y=0.7 (large)
    obs = _make_obs(d_x=1.58, d_y=0.7, y_err_obs=0.6)  # lat=0.3
    _, info = policy.act(obs)

    assert info["stage"] != "retreat", (
        f"Should NOT trigger retreat for lat=0.3 (below 0.48 threshold), "
        f"but got stage={info['stage']}"
    )
    print(f"  PASS: lat=0.3, d_y=0.7 -> no false retreat trigger (stage={info['stage']})")


def test_case_5_cooldown_blocks_retrigger():
    """After retreat ends, cooldown should block immediate re-triggering."""
    policy = _make_policy()
    cfg = policy.cfg

    # Step 1: trigger and run retreat to completion (max_retreat_steps)
    obs_trigger = _make_obs(d_x=1.88, y_err_obs=1.0)  # lat=0.5, dist_front=0.8
    for i in range(cfg.max_retreat_steps + 5):
        _, info = policy.act(obs_trigger)
        if info["stage"] != "retreat" and i > 0:
            break

    # After retreat ends, cooldown should be active
    assert policy._retreat_cooldown_remaining > 0, (
        f"Cooldown should be active after retreat, got {policy._retreat_cooldown_remaining}"
    )

    # Step 2: immediately present same trigger conditions
    _, info_after = policy.act(obs_trigger)
    assert info_after["stage"] != "retreat", (
        f"Cooldown should block retreat re-trigger, but got stage={info_after['stage']}"
    )
    print(f"  PASS: cooldown={policy._retreat_cooldown_remaining} blocks re-trigger (stage={info_after['stage']})")


# =====================================================================
# Main (fallback for no-pytest environments)
# =====================================================================
def main():
    tests = [
        test_case_1_retreat_steer_direction,
        test_case_2_retreat_steer_proportional,
        test_case_3_retreat_exit_on_alignment,
        test_case_4_no_false_trigger,
        test_case_5_cooldown_blocks_retrigger,
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
