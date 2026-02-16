#!/usr/bin/env python3
"""
Unit tests for expert policy v7 FSM architecture.

Tests can run WITHOUT IsaacLab/Isaac Sim — only needs numpy.
Validates: Stanley controller, FSM transitions with hysteresis,
           HardAbort reverse steer polarity, Straighten sub-phase,
           insert_norm progress detection.

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

from forklift_expert.expert_policy import ForkliftExpertPolicy, ExpertConfig

# ---- Load specs ----
_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
with open(os.path.join(_BASE, "forklift_expert", "obs_spec.json")) as f:
    OBS_SPEC = json.load(f)
with open(os.path.join(_BASE, "forklift_expert", "action_spec.json")) as f:
    ACTION_SPEC = json.load(f)

FIELDS = OBS_SPEC["fields"]
_HALF_D = ExpertConfig().pallet_half_depth  # 1.08
_FORK_LEN = ExpertConfig().fork_length      # 1.87


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
    v_forward: float = 0.5,
) -> np.ndarray:
    """Build obs that produces a specific lat_true and dist_front.

    When dyaw=0: lat_true = sin(0)*d_x - cos(0)*d_y = -d_y
    So d_y = -lat_desired.
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
        v_forward=v_forward,
    )


def _make_policy() -> ForkliftExpertPolicy:
    return ForkliftExpertPolicy(OBS_SPEC, ACTION_SPEC, ExpertConfig())


# =====================================================================
# 1. Stanley controller tests
# =====================================================================

def test_stanley_direction_lat_positive():
    """lat>0 (vehicle right of path) → steer should be negative (turn left).
    Stanley: delta_rad = yaw + atan2(k_e * lat, v + k_soft) → positive
    raw_steer = -delta_rad * k_steer → negative."""
    policy = _make_policy()
    obs = _make_obs_for_lat(lat_desired=0.3, dist_front=3.0, v_forward=0.8)
    _, info = policy.act(obs)
    assert info["fsm_stage"] == "Approach"
    assert info["raw_steer"] < 0, (
        f"lat>0 in Approach: steer should be negative, got {info['raw_steer']:.3f}"
    )
    print(f"  PASS: Stanley direction lat>0 → steer={info['raw_steer']:.3f} (negative)")


def test_stanley_direction_lat_negative():
    """lat<0 (vehicle left of path) → steer should be positive."""
    policy = _make_policy()
    obs = _make_obs_for_lat(lat_desired=-0.3, dist_front=3.0, v_forward=0.8)
    _, info = policy.act(obs)
    assert info["fsm_stage"] == "Approach"
    assert info["raw_steer"] > 0, (
        f"lat<0 in Approach: steer should be positive, got {info['raw_steer']:.3f}"
    )
    print(f"  PASS: Stanley direction lat<0 → steer={info['raw_steer']:.3f} (positive)")


def test_stanley_saturation():
    """Large lat should NOT produce steer > eff_max_steer due to atan2 saturation."""
    policy = _make_policy()
    obs = _make_obs_for_lat(lat_desired=2.0, dist_front=3.0, v_forward=0.8)
    _, info = policy.act(obs)
    assert info["fsm_stage"] == "Approach"
    eff = info["eff_max_steer"]
    assert abs(info["raw_steer"]) <= eff + 0.001, (
        f"Stanley should be capped at eff_max_steer={eff:.3f}, "
        f"got |raw_steer|={abs(info['raw_steer']):.3f}"
    )
    print(f"  PASS: Stanley saturation: lat=2.0 → |steer|={abs(info['raw_steer']):.3f} <= {eff:.3f}")


def test_stanley_unit_scaling():
    """Verify k_steer converts radians to normalised steer correctly.
    At small angles: raw_steer ≈ -delta_rad * k_steer
    delta_rad should be in physical radians; k_steer = 1/0.6 ≈ 1.667."""
    cfg = ExpertConfig()
    assert abs(cfg.k_steer - 1.0 / cfg.steer_angle_rad) < 0.01, (
        f"k_steer should be 1/steer_angle_rad = {1.0/cfg.steer_angle_rad:.3f}, "
        f"got {cfg.k_steer:.3f}"
    )
    print(f"  PASS: k_steer = {cfg.k_steer:.3f} ≈ 1/{cfg.steer_angle_rad}")


def test_stanley_low_speed_protection():
    """effective_v = max(|v_forward|, 0.2) prevents divergence at v=0.
    Same lat/yaw at v=0 vs v=0.8 should produce different steer magnitudes.
    Use small lat to avoid hitting eff_max_steer clamp."""
    p1 = _make_policy()
    obs_slow = _make_obs_for_lat(lat_desired=0.05, dist_front=3.0, v_forward=0.0)
    _, info_slow = p1.act(obs_slow)

    p2 = _make_policy()
    obs_fast = _make_obs_for_lat(lat_desired=0.05, dist_front=3.0, v_forward=0.8)
    _, info_fast = p2.act(obs_fast)

    assert info_slow["fsm_stage"] == "Approach"
    assert info_fast["fsm_stage"] == "Approach"
    assert abs(info_slow["raw_steer"]) > abs(info_fast["raw_steer"]), (
        f"Low speed should produce stronger correction: "
        f"|steer(v=0)|={abs(info_slow['raw_steer']):.3f} should > "
        f"|steer(v=0.8)|={abs(info_fast['raw_steer']):.3f}"
    )
    print(f"  PASS: Stanley v=0 → |steer|={abs(info_slow['raw_steer']):.3f} > "
          f"v=0.8 → |steer|={abs(info_fast['raw_steer']):.3f}")


# =====================================================================
# 2. FSM transition tests
# =====================================================================

def test_fsm_approach_to_straighten():
    """When dtc <= hard_wall and aligned → transition to Straighten (not FinalInsert)."""
    policy = _make_policy()
    cfg = policy.cfg
    dtc_target = cfg.hard_wall - 0.01
    dist_front = dtc_target + cfg.fork_length
    obs = _make_obs_for_lat(lat_desired=0.0, dist_front=dist_front)
    _, info = policy.act(obs)
    assert info["fsm_stage"] == "Straighten", (
        f"Aligned at hard_wall should go to Straighten, got {info['fsm_stage']}"
    )
    print(f"  PASS: Approach → Straighten when dtc={dtc_target:.3f}, aligned")


def test_fsm_approach_to_hard_abort():
    """When dtc <= hard_wall and NOT aligned → HardAbort."""
    policy = _make_policy()
    cfg = policy.cfg
    dtc_target = cfg.hard_wall - 0.01
    dist_front = dtc_target + cfg.fork_length
    obs = _make_obs_for_lat(lat_desired=0.3, dist_front=dist_front)
    _, info = policy.act(obs)
    assert info["fsm_stage"] == "HardAbort", (
        f"Misaligned at hard_wall should go to HardAbort, got {info['fsm_stage']}"
    )
    assert info["abort_reason"] == "pose_not_aligned"
    print(f"  PASS: Approach → HardAbort when lat=0.3 at hard_wall")


def test_fsm_straighten_to_final_insert():
    """Straighten → FinalInsert when |prev_steer| < steer_straight_ok."""
    policy = _make_policy()
    cfg = policy.cfg
    policy._fsm_stage = "Straighten"
    policy._prev_steer = 0.01
    dtc_target = cfg.hard_wall - 0.01
    dist_front = dtc_target + cfg.fork_length
    obs = _make_obs_for_lat(lat_desired=0.0, dist_front=dist_front)
    _, info = policy.act(obs)
    assert info["fsm_stage"] == "FinalInsert", (
        f"Straighten with small prev_steer should go to FinalInsert, got {info['fsm_stage']}"
    )
    print(f"  PASS: Straighten → FinalInsert when prev_steer={policy._prev_steer:.3f}")


def test_fsm_straighten_to_hard_abort():
    """Straighten → HardAbort when alignment lost."""
    policy = _make_policy()
    cfg = policy.cfg
    policy._fsm_stage = "Straighten"
    policy._prev_steer = 0.3
    dtc_target = cfg.hard_wall - 0.01
    dist_front = dtc_target + cfg.fork_length
    obs = _make_obs_for_lat(lat_desired=0.3, dist_front=dist_front)
    _, info = policy.act(obs)
    assert info["fsm_stage"] == "HardAbort", (
        f"Straighten with large lat should go to HardAbort, got {info['fsm_stage']}"
    )
    assert info["abort_reason"] == "alignment_lost"
    print(f"  PASS: Straighten → HardAbort when lat=0.3 > abort_lat={cfg.abort_lat}")


def test_fsm_final_insert_steer_zero():
    """FinalInsert must produce steer=0 (blind push)."""
    policy = _make_policy()
    policy._fsm_stage = "FinalInsert"
    policy._prev_steer = 0.0
    cfg = policy.cfg
    dtc_target = cfg.hard_wall - 0.01
    dist_front = dtc_target + cfg.fork_length
    obs = _make_obs_for_lat(lat_desired=0.02, dist_front=dist_front, insert_norm=0.3)
    _, info = policy.act(obs)
    assert info["fsm_stage"] == "FinalInsert"
    assert info["raw_steer"] == 0.0, (
        f"FinalInsert raw_steer must be 0, got {info['raw_steer']:.4f}"
    )
    print(f"  PASS: FinalInsert steer=0 (blind push)")


def test_fsm_final_insert_to_hard_abort_alignment():
    """FinalInsert → HardAbort on alignment loss (loose thresholds)."""
    policy = _make_policy()
    cfg = policy.cfg
    policy._fsm_stage = "FinalInsert"
    policy._prev_steer = 0.0
    dtc_target = cfg.hard_wall - 0.01
    dist_front = dtc_target + cfg.fork_length
    obs = _make_obs_for_lat(
        lat_desired=cfg.abort_lat + 0.01, dist_front=dist_front, insert_norm=0.3)
    _, info = policy.act(obs)
    assert info["fsm_stage"] == "HardAbort", (
        f"Expected HardAbort on alignment loss, got {info['fsm_stage']}"
    )
    assert info["abort_reason"] == "alignment_lost"
    print(f"  PASS: FinalInsert → HardAbort when lat > abort_lat")


def test_fsm_hysteresis():
    """Strict entry, loose exit: lat=0.10 should NOT trigger HardAbort from FinalInsert
    (abort_lat=0.15), but lat=0.10 > final_lat_ok=0.08 would prevent entry."""
    cfg = ExpertConfig()
    assert cfg.final_lat_ok < cfg.abort_lat, (
        f"Hysteresis requires final_lat_ok({cfg.final_lat_ok}) < abort_lat({cfg.abort_lat})"
    )
    policy = _make_policy()
    policy._fsm_stage = "FinalInsert"
    policy._prev_steer = 0.0
    dtc_target = cfg.hard_wall - 0.01
    dist_front = dtc_target + cfg.fork_length
    obs = _make_obs_for_lat(lat_desired=0.10, dist_front=dist_front, insert_norm=0.3)
    _, info = policy.act(obs)
    assert info["fsm_stage"] == "FinalInsert", (
        f"lat=0.10 within hysteresis band should stay FinalInsert, got {info['fsm_stage']}"
    )
    print(f"  PASS: FSM hysteresis — lat=0.10 stays in FinalInsert "
          f"(entry<{cfg.final_lat_ok}, exit>{cfg.abort_lat})")


def test_fsm_hard_abort_state_lock():
    """HardAbort stays locked until dist >= retreat_target_dist, ignoring dtc."""
    policy = _make_policy()
    cfg = policy.cfg
    policy._fsm_stage = "HardAbort"
    policy._retreat_steps = 0
    dist_front = cfg.retreat_target_dist - 0.5
    obs = _make_obs_for_lat(lat_desired=0.0, dist_front=dist_front)
    _, info = policy.act(obs)
    assert info["fsm_stage"] == "HardAbort", (
        f"HardAbort should be locked until dist >= target, got {info['fsm_stage']}"
    )
    policy.reset()
    policy._fsm_stage = "HardAbort"
    obs_far = _make_obs_for_lat(lat_desired=0.0, dist_front=cfg.retreat_target_dist + 0.1)
    _, info2 = policy.act(obs_far)
    assert info2["fsm_stage"] == "Approach", (
        f"HardAbort should exit to Approach at retreat_target_dist, got {info2['fsm_stage']}"
    )
    print(f"  PASS: HardAbort state lock and exit at retreat_target_dist={cfg.retreat_target_dist}")


def test_fsm_to_lift():
    """insert_norm >= lift_on_insert_norm → Lift from any FSM stage."""
    for stage in ("Approach", "Straighten", "FinalInsert", "HardAbort"):
        policy = _make_policy()
        policy._fsm_stage = stage
        cfg = policy.cfg
        obs = _make_obs(insert_norm=cfg.lift_on_insert_norm + 0.01)
        _, info = policy.act(obs)
        assert info["fsm_stage"] == "Lift", (
            f"From {stage}: insert_norm >= threshold should go to Lift, got {info['fsm_stage']}"
        )
    print(f"  PASS: Lift transition from all FSM stages on insert_norm >= {cfg.lift_on_insert_norm}")


# =====================================================================
# 3. HardAbort reverse steer polarity (TDD critical)
# =====================================================================

def test_reverse_steer_polarity():
    """HardAbort: lat>0 → steer>0, lat<0 → steer<0 (hard-coded signs).
    Verified by v5-BC test_case_1: drive=-1 + steer=sign(lat) correctly
    corrects lateral offset during reverse driving."""
    policy = _make_policy()
    cfg = policy.cfg
    policy._fsm_stage = "HardAbort"
    policy._retreat_steps = 1
    dist_front = cfg.retreat_target_dist - 0.5
    obs_right = _make_obs_for_lat(lat_desired=0.5, dist_front=dist_front)
    _, info = policy.act(obs_right)
    assert info["fsm_stage"] == "HardAbort"
    assert info["raw_steer"] > 0, (
        f"lat>0 in HardAbort: steer MUST be >0, got {info['raw_steer']:.3f}"
    )

    policy2 = _make_policy()
    policy2._fsm_stage = "HardAbort"
    policy2._retreat_steps = 1
    obs_left = _make_obs_for_lat(lat_desired=-0.5, dist_front=dist_front)
    _, info2 = policy2.act(obs_left)
    assert info2["fsm_stage"] == "HardAbort"
    assert info2["raw_steer"] < 0, (
        f"lat<0 in HardAbort: steer MUST be <0, got {info2['raw_steer']:.3f}"
    )
    print(f"  PASS: reverse steer polarity: lat>0 → steer={info['raw_steer']:.3f}, "
          f"lat<0 → steer={info2['raw_steer']:.3f}")


def test_reverse_steer_proportional():
    """Larger |lat| should produce larger |steer| in HardAbort."""
    p1 = _make_policy()
    p1._fsm_stage = "HardAbort"
    p1._retreat_steps = 1
    cfg = p1.cfg
    dist_front = cfg.retreat_target_dist - 0.5

    obs_small = _make_obs_for_lat(lat_desired=0.2, dist_front=dist_front)
    _, info_small = p1.act(obs_small)

    p2 = _make_policy()
    p2._fsm_stage = "HardAbort"
    p2._retreat_steps = 1
    obs_large = _make_obs_for_lat(lat_desired=0.8, dist_front=dist_front)
    _, info_large = p2.act(obs_large)

    assert abs(info_large["raw_steer"]) > abs(info_small["raw_steer"]), (
        f"|steer(lat=0.8)|={abs(info_large['raw_steer']):.3f} should > "
        f"|steer(lat=0.2)|={abs(info_small['raw_steer']):.3f}"
    )
    print(f"  PASS: reverse proportional: lat=0.2→{info_small['raw_steer']:.3f}, "
          f"lat=0.8→{info_large['raw_steer']:.3f}")


def test_hard_abort_drive_negative():
    """HardAbort must always produce drive = retreat_drive (-1.0)."""
    policy = _make_policy()
    cfg = policy.cfg
    policy._fsm_stage = "HardAbort"
    dist_front = cfg.retreat_target_dist - 0.5
    obs = _make_obs_for_lat(lat_desired=0.3, dist_front=dist_front)
    _, info = policy.act(obs)
    assert info["drive"] < 0, (
        f"HardAbort drive should be negative, got {info['drive']:.3f}"
    )
    print(f"  PASS: HardAbort drive={info['drive']:.3f} (negative)")


# =====================================================================
# 4. Straighten sub-phase tests
# =====================================================================

def test_straighten_drive_zero():
    """Straighten stage: drive should be 0 (or near-zero)."""
    policy = _make_policy()
    policy._fsm_stage = "Straighten"
    policy._prev_steer = 0.3
    cfg = policy.cfg
    dtc_target = cfg.hard_wall - 0.01
    dist_front = dtc_target + cfg.fork_length
    obs = _make_obs_for_lat(lat_desired=0.0, dist_front=dist_front)
    _, info = policy.act(obs)
    assert info["fsm_stage"] in ("Straighten", "FinalInsert")
    assert abs(info["raw_steer"]) < 0.001, (
        f"Straighten raw_steer should be ~0, got {info['raw_steer']:.4f}"
    )
    print(f"  PASS: Straighten raw_steer=0, drive targets 0")


def test_straighten_converges_to_final_insert():
    """After enough steps in Straighten, prev_steer should converge to 0
    and transition to FinalInsert."""
    policy = _make_policy()
    cfg = policy.cfg
    policy._fsm_stage = "Straighten"
    policy._prev_steer = 0.30

    dtc_target = cfg.hard_wall - 0.01
    dist_front = dtc_target + cfg.fork_length
    obs = _make_obs_for_lat(lat_desired=0.0, dist_front=dist_front, insert_norm=0.2)

    reached_final = False
    for step in range(20):
        _, info = policy.act(obs)
        if info["fsm_stage"] == "FinalInsert":
            reached_final = True
            break

    assert reached_final, (
        f"Straighten should converge to FinalInsert within 20 steps, "
        f"still in {policy._fsm_stage}, prev_steer={policy._prev_steer:.4f}"
    )
    print(f"  PASS: Straighten → FinalInsert in {step+1} steps")


# =====================================================================
# 5. insert_norm progress detection tests
# =====================================================================

def test_no_progress_triggers_abort():
    """FinalInsert: if insert_norm doesn't grow for no_progress_steps → HardAbort."""
    policy = _make_policy()
    cfg = policy.cfg
    policy._fsm_stage = "FinalInsert"
    policy._prev_steer = 0.0

    dtc_target = cfg.hard_wall - 0.01
    dist_front = dtc_target + cfg.fork_length
    static_ins = 0.30

    for i in range(cfg.no_progress_steps + 2):
        obs = _make_obs_for_lat(
            lat_desired=0.0, dist_front=dist_front, insert_norm=static_ins)
        _, info = policy.act(obs)
        if info["fsm_stage"] == "HardAbort":
            assert info["abort_reason"] == "no_progress"
            print(f"  PASS: no_progress abort triggered at step {i+1} "
                  f"(threshold={cfg.no_progress_steps})")
            return

    assert False, (
        f"no_progress abort should have triggered within {cfg.no_progress_steps+2} steps"
    )


def test_progress_resets_counter():
    """If insert_norm grows, the no_progress counter resets."""
    policy = _make_policy()
    cfg = policy.cfg
    policy._fsm_stage = "FinalInsert"
    policy._prev_steer = 0.0
    policy._prev_insert_norm = 0.30

    dtc_target = cfg.hard_wall - 0.01
    dist_front = dtc_target + cfg.fork_length

    for i in range(cfg.no_progress_steps - 2):
        obs = _make_obs_for_lat(
            lat_desired=0.0, dist_front=dist_front, insert_norm=0.30)
        _, info = policy.act(obs)
        assert info["fsm_stage"] == "FinalInsert"

    obs_progress = _make_obs_for_lat(
        lat_desired=0.0, dist_front=dist_front, insert_norm=0.35)
    _, info = policy.act(obs_progress)
    assert info["fsm_stage"] == "FinalInsert"

    for i in range(cfg.no_progress_steps - 2):
        obs = _make_obs_for_lat(
            lat_desired=0.0, dist_front=dist_front, insert_norm=0.35)
        _, info = policy.act(obs)
        assert info["fsm_stage"] == "FinalInsert", (
            f"Counter should have reset after progress; "
            f"abort at step {i+1} is premature"
        )

    print(f"  PASS: progress resets no_progress counter")


# =====================================================================
# 6. dist_to_contact computation test
# =====================================================================

def test_dist_to_contact():
    """dtc = max(dist_front - fork_length, 0)."""
    policy = _make_policy()
    cfg = policy.cfg

    obs_far = _make_obs(d_x=5.0)
    _, info_far = policy.act(obs_far)
    expected_dist_front = 5.0 - cfg.pallet_half_depth
    expected_dtc = max(expected_dist_front - cfg.fork_length, 0.0)
    assert abs(info_far["dist_to_contact"] - expected_dtc) < 0.01, (
        f"dtc should be {expected_dtc:.3f}, got {info_far['dist_to_contact']:.3f}"
    )

    policy.reset()
    obs_close = _make_obs(d_x=cfg.pallet_half_depth + cfg.fork_length - 0.1)
    _, info_close = policy.act(obs_close)
    assert info_close["dist_to_contact"] == 0.0, (
        f"dtc should be 0 when dist_front < fork_length, got {info_close['dist_to_contact']:.3f}"
    )
    print(f"  PASS: dist_to_contact computation: far={info_far['dist_to_contact']:.3f}, close=0.0")


# =====================================================================
# 7. Info fields present
# =====================================================================

def test_info_fields():
    """All required v7 info fields should be present."""
    policy = _make_policy()
    obs = _make_obs(d_x=3.0)
    _, info = policy.act(obs)

    required = [
        "stage", "fsm_stage", "dist_front", "dist_to_contact",
        "lat", "yaw", "insert_norm", "raw_steer", "steer", "drive",
        "eff_max_steer", "steer_sat_ratio", "abort_reason",
        "in_retreat", "retreat_steps", "retreat_cooldown",
    ]
    missing = [k for k in required if k not in info]
    assert not missing, f"Missing info fields: {missing}"
    print(f"  PASS: all {len(required)} required info fields present")


# =====================================================================
# Main
# =====================================================================
def main():
    tests = [
        test_stanley_direction_lat_positive,
        test_stanley_direction_lat_negative,
        test_stanley_saturation,
        test_stanley_unit_scaling,
        test_stanley_low_speed_protection,
        test_fsm_approach_to_straighten,
        test_fsm_approach_to_hard_abort,
        test_fsm_straighten_to_final_insert,
        test_fsm_straighten_to_hard_abort,
        test_fsm_final_insert_steer_zero,
        test_fsm_final_insert_to_hard_abort_alignment,
        test_fsm_hysteresis,
        test_fsm_hard_abort_state_lock,
        test_fsm_to_lift,
        test_reverse_steer_polarity,
        test_reverse_steer_proportional,
        test_hard_abort_drive_negative,
        test_straighten_drive_zero,
        test_straighten_converges_to_final_insert,
        test_no_progress_triggers_abort,
        test_progress_resets_counter,
        test_dist_to_contact,
        test_info_fields,
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
