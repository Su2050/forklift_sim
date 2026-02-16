"""
Rule-based expert policy for forklift pallet-insertion task.

Adapted for the **actual 15-D observation vector** produced by
``env._get_observations()`` in the IsaacLab forklift environment:

  [0-1]  d_xy_r          robot-frame relative position to pallet center (m)
  [2-3]  cos_dyaw, sin_dyaw   yaw difference encoding
  [4-5]  v_xy_r          robot-frame linear velocity (m/s)
  [6]    yaw_rate         yaw angular velocity (rad/s)
  [7-8]  lift_pos, lift_vel   lift joint position / velocity
  [9]    insert_norm      insertion depth normalised 0-1
  [10-12] prev actions    (drive, steer, lift) from last step
  [13]   y_err_obs        lateral error in pallet center-line frame,
                          normalised by 0.5 m, clipped [-1, 1]
  [14]   yaw_err_obs      yaw error in pallet center-line frame,
                          normalised by 15 deg, clipped [-1, 1]
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple
import json
import math
import numpy as np


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def _clip(x: float, lo: float, hi: float) -> float:
    return float(min(max(x, lo), hi))


def _wrap_pi(angle: float) -> float:
    """Wrap angle to [-pi, pi]."""
    a = (angle + math.pi) % (2 * math.pi) - math.pi
    return float(a)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class ExpertConfig:
    """Tunable knobs -- calibrated for the 15-D obs produced by the IsaacLab
    forklift environment.

    Key units
    ---------
    * ``dist``  : metres  (from ``d_xy_r[0]``, forward distance to pallet
                  *center*; subtract ``pallet_half_depth`` to approximate
                  distance to pallet front opening).
    * ``lat``   : metres  (true pallet-frame lateral error, computed from
                  ``d_xy_r`` + ``cos/sin(dyaw)``; NOT clipped like ``y_err_obs``).
    * ``yaw``   : radians (recovered via ``atan2(sin_dyaw, cos_dyaw)``).
    """

    # ---- Geometry: pallet size ----
    # pallet_depth_m = 2.16 (from env_cfg); half is used to estimate
    # distance-to-front from distance-to-center.
    pallet_half_depth: float = 1.08

    # ---- Docking (approach + align) ----
    k_lat: float = 1.1          # steering gain for lateral error — reduced from 1.5
                                # to prevent lateral overshoot (stress-test: 26% drift pattern)
    k_yaw: float = 0.9          # steering gain for yaw error   — reduced from 1.2
    k_damp: float = 0.20        # NEW: yaw-rate damping to suppress oscillation
    k_dist: float = 0.6         # throttle gain for distance
    v_max: float = 0.95         # max forward command — near full speed
    v_min: float = 0.80         # strong forward drive is critical for Ackermann steering
    max_steer: float = 0.55     # (legacy, used only as fallback)
    max_steer_far: float = 0.65  # steer limit when dist > 2.0m (room to correct)
    max_steer_near: float = 0.40 # steer limit when dist < 0.8m (prevent overshoot)
    # v5-C: lat-dependent steer bonus (only when dist >= 0.8m)
    max_steer_lat_bonus_start: float = 0.4   # |lat| > this -> add bonus to eff_max_steer
    max_steer_lat_bonus_max: float = 0.10    # cap: far limit goes from 0.65 to max 0.75
    slow_dist: float = 0.5      # only slow very close to pallet front
    stop_dist: float = 0.3      # docking "arrived" gate (m, to pallet front)

    # alignment thresholds  (used to compute misalign ratio for speed scaling)
    lat_ok: float = 0.20        # 20 cm
    yaw_ok: float = math.radians(15.0)  # 15 deg

    # ---- Retreat ----
    # Only trigger retreat when VERY close AND severely misaligned.
    # Stress-test showed docking-retreat cycling is the #1 failure mode;
    # the docking controller alone can correct |lat| up to 0.48 given
    # enough steps.  Retreat is a last resort for extreme cases.
    retreat_lat_thresh: float = 0.48    # near-saturated lat (metres)
    retreat_yaw_thresh: float = math.radians(35.0)  # large yaw
    retreat_dist_thresh: float = 1.0    # only retreat when < 1m to pallet front
    retreat_target_dist: float = 2.5    # v7: increased from 1.8 for longer correction runway
    retreat_drive: float = -1.0         # full backward speed
    retreat_steer_gain: float = 0.50    # v5-B legacy (unused in v7 FSM)
    retreat_k_yaw: float = 0.30        # v5-B legacy (unused in v7 FSM)
    retreat_max_steer: float = 0.80    # v7: reduced from 0.90
    retreat_lat_sat: float = 0.75      # v5-B: lat at which retreat_lat_term saturates (was implicit 0.5)
    retreat_exit_lat_abs: float = 0.40 # v5-B: was 0.30 — abs lat threshold for alignment exit
    retreat_exit_yaw_max: float = math.radians(30)  # v5-B: yaw must also be OK to exit retreat
    max_retreat_steps: int = 80         # hard cap per single retreat
    retreat_cooldown: int = 80          # v7: reduced from 150 (FSM state-locking handles cycling)

    # ---- Insertion ----
    # Stress-test showed max_ins ≈ 0.43-0.48 even after 300+ insertion steps
    # with vf0 ≈ 60%. The forklift stalls inside the pallet due to friction.
    # Solution: much higher insertion drive to overcome pallet resistance.
    ins_v_max: float = 0.80         # was 0.40 — doubled for faster insertion progress
    ins_v_min: float = 0.20         # was 0.08 — strong minimum to prevent stalling
    ins_lat_ok: float = 0.15        # 15 cm (was 10) -- more forgiving
    ins_yaw_ok: float = math.radians(12.0)  # 12 deg (was 8) -- more forgiving

    # Alignment gate: insertion stage is only entered when BOTH insert_norm
    # exceeds the threshold AND alignment is within these gates.
    ins_stage_lat_gate: float = 0.25    # tightened to 25cm (longer episode allows better alignment)
    ins_stage_yaw_gate: float = math.radians(15.0)  # tightened to 15 deg

    # Contact / slip backoff -- **disabled** by default because the 15-D obs
    # does NOT include contact_flag or slip_flag.
    backoff_on_contact: bool = False
    backoff_throttle: float = -0.20
    backoff_steps: int = 6

    # ---- Lift ----
    lift_on_insert_norm: float = 0.75
    lift_cmd: float = 0.60

    # ---- Safety / smoothness ----
    steer_rate_limit: float = 0.35     # max delta-steer per step
    throttle_rate_limit: float = 0.50  # max delta-throttle per step — faster acceleration
    deadband_steer: float = 0.02

    # ---- Stage heuristic (v5 legacy, unused by v7 FSM) ----
    use_insert_norm_for_stage: bool = True
    insert_enter_stage: float = 0.15

    # ---- v7 FSM: Geometry ----
    fork_length: float = 1.87          # fork length (metres)

    # ---- v7 FSM: Distance thresholds ----
    pre_insert: float = 0.5            # soft zone: tighten alignment but still Approach
    hard_wall: float = 0.2             # hard FSM boundary (dtc < this → FinalInsert or HardAbort)

    # ---- v7 FSM: Alignment thresholds (hysteresis) ----
    final_lat_ok: float = 0.08         # strict entry to FinalInsert (metres)
    final_yaw_ok: float = math.radians(5.0)   # strict entry to FinalInsert
    abort_lat: float = 0.15            # loose exit from FinalInsert (metres)
    abort_yaw: float = math.radians(10.0)     # loose exit from FinalInsert

    # ---- v7 FSM: FinalInsert ----
    final_insert_drive: float = 0.50   # constant forward push in FinalInsert

    # ---- v7 FSM: HardAbort (Smart Retreat) ----
    k_lat_rev: float = 0.50            # reverse steer lateral gain (POSITIVE: lat>0 → steer>0)
    k_yaw_rev: float = 0.30            # reverse steer yaw gain (POSITIVE)

    # ---- v7 FSM: Stanley controller (A1+, unused in A0) ----
    k_e: float = 3.0                   # crosstrack gain
    k_soft: float = 0.4                # low-speed protection
    k_steer: float = 1.667             # rad→normalised scaling (1/steer_angle_rad)
    steer_angle_rad: float = 0.6       # env steer angle in radians

    # ---- v7 FSM: Straighten sub-phase (A2, unused in A0/A1) ----
    steer_straight_ok: float = 0.05    # |prev_steer| < this → wheels are straight

    # ---- v7 FSM: Progress detection (A2, unused in A0/A1) ----
    no_progress_steps: int = 20        # consecutive steps without ins growth
    no_progress_eps: float = 0.01      # min insert_norm delta to count as progress


# ---------------------------------------------------------------------------
# Expert policy
# ---------------------------------------------------------------------------
class ForkliftExpertPolicy:
    """
    A rule-based expert policy (v7: FSM architecture).

    It consumes the 15-D obs vector from the IsaacLab forklift env and emits
    a 3-D action vector ``[drive, steer, lift]``.

    v7 FSM Stages
    -------------
    * **Approach**     : far-field alignment + approach (PD or Stanley steering)
    * **Straighten**   : (A2) wheels straight before final push
    * **FinalInsert**  : steer-blind constant push into pallet
    * **HardAbort**    : Smart Retreat — backward + active steering correction
    * **Lift**         : lift after sufficient insertion depth
    """

    def __init__(
        self,
        obs_spec: Dict[str, Any],
        action_spec: Dict[str, Any],
        cfg: Optional[ExpertConfig] = None,
    ) -> None:
        self.obs_spec = obs_spec
        self.action_spec = action_spec
        self.cfg = cfg or ExpertConfig()

        self._prev_steer: float = 0.0
        self._prev_throttle: float = 0.0

        # v7 FSM state
        self._fsm_stage: str = "Approach"
        self._abort_reason: str = ""
        self._retreat_steps: int = 0
        self._retreat_cooldown_remaining: int = 0

        # v7 A2: progress detection (unused in A0/A1)
        self._prev_insert_norm: float = 0.0
        self._no_progress_count: int = 0

        # Validate specs
        assert "fields" in self.obs_spec, "obs_spec missing 'fields'"
        assert "fields" in self.action_spec, "action_spec missing 'fields'"
        self.action_dim = int(self.action_spec.get("action_dim", 0))
        if self.action_dim <= 0:
            raise ValueError("action_dim must be > 0 in action_spec")

        # Cache obs field indices for fast look-up
        f = self.obs_spec["fields"]
        self._idx_d_xy_r_x   = int(f.get("d_xy_r_x", -1))
        self._idx_d_xy_r_y   = int(f.get("d_xy_r_y", -1))
        self._idx_cos_dyaw   = int(f.get("cos_dyaw", -1))
        self._idx_sin_dyaw   = int(f.get("sin_dyaw", -1))
        self._idx_v_forward  = int(f.get("v_forward", -1))
        self._idx_yaw_rate   = int(f.get("yaw_rate", -1))
        self._idx_lift_pos   = int(f.get("lift_pos", -1))
        self._idx_insert_norm = int(f.get("insert_norm", -1))
        self._idx_y_err_obs  = int(f.get("y_err_obs", -1))
        self._idx_yaw_err_obs = int(f.get("yaw_err_obs", -1))

    # ------------------------------------------------------------------ IO
    @staticmethod
    def load_json(path: str) -> Dict[str, Any]:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def reset(self) -> None:
        self._prev_steer = 0.0
        self._prev_throttle = 0.0
        self._fsm_stage = "Approach"
        self._abort_reason = ""
        self._retreat_steps = 0
        self._retreat_cooldown_remaining = 0
        self._prev_insert_norm = 0.0
        self._no_progress_count = 0

    # -------------------------------------------------------------- helpers
    @staticmethod
    def _safe_read(obs: np.ndarray, idx: int, default: float = 0.0) -> float:
        """Read a single element from obs by index; return *default* if
        the index is negative (field not mapped) or out of range."""
        if idx < 0 or idx >= obs.shape[-1]:
            return float(default)
        return float(obs[idx])

    def _decode_obs(self, obs: np.ndarray) -> Dict[str, float]:
        """Decode the raw 15-D obs vector into semantic fields that the
        expert control logic operates on.

        Returns a dict with at least:
          dist_front, lat_true, lat_clipped, yaw_err, insert_norm,
          v_forward, yaw_rate, lift_pos, contact_flag, slip_flag
        """
        _r = self._safe_read  # shorthand

        # --- Forward distance (robot frame) ---------------------------------
        d_x = _r(obs, self._idx_d_xy_r_x, default=2.0)
        d_y = _r(obs, self._idx_d_xy_r_y, default=0.0)
        dist_to_center = math.sqrt(d_x ** 2 + d_y ** 2)
        dist_front = max(d_x - self.cfg.pallet_half_depth, 0.0)

        # --- Yaw error (full-range radians) --------------------------------
        cos_dy = _r(obs, self._idx_cos_dyaw, default=1.0)
        sin_dy = _r(obs, self._idx_sin_dyaw, default=0.0)
        yaw_err = math.atan2(sin_dy, cos_dy)  # [-pi, pi]

        # --- True lateral error (metres, pallet center-line frame) ---------
        # Rotate robot-frame d_xy_r to pallet-frame, take lateral component.
        # dyaw = pallet_yaw - robot_yaw; cos_dy/sin_dy encode this angle.
        # This is the UNSATURATED equivalent of env's y_signed_obs (before clip).
        lat_true = sin_dy * d_x - cos_dy * d_y

        # Clipped version (from y_err_obs) for reference / logging
        y_err_norm = _r(obs, self._idx_y_err_obs, default=0.0)
        lat_clipped = y_err_norm * 0.5  # metres, clipped to [-0.5, +0.5]

        # --- Other scalars --------------------------------------------------
        insert_norm = _r(obs, self._idx_insert_norm, default=0.0)
        v_forward   = _r(obs, self._idx_v_forward, default=0.0)
        yaw_rate    = _r(obs, self._idx_yaw_rate, default=0.0)
        lift_pos    = _r(obs, self._idx_lift_pos, default=0.0)

        # contact / slip are NOT present in the 15-D obs -- always 0
        contact_flag = 0.0
        slip_flag    = 0.0

        dist_to_contact = max(dist_front - self.cfg.fork_length, 0.0)

        return {
            "dist_front": dist_front,
            "dist_to_contact": dist_to_contact,
            "dist_to_center": dist_to_center,
            "d_x": d_x,
            "d_y": d_y,
            "lat_true": lat_true,
            "lat_clipped": lat_clipped,
            "yaw_err": yaw_err,
            "insert_norm": insert_norm,
            "v_forward": v_forward,
            "yaw_rate": yaw_rate,
            "lift_pos": lift_pos,
            "contact_flag": contact_flag,
            "slip_flag": slip_flag,
        }

    def _rate_limit(self, val: float, prev: float, limit: float) -> float:
        dv = _clip(val - prev, -limit, limit)
        return prev + dv

    def _build_action(self, drive: float, steer: float, lift: float) -> np.ndarray:
        """Pack scalar commands into the action vector according to
        ``action_spec``."""
        a = np.zeros((self.action_dim,), dtype=np.float32)
        f = self.action_spec["fields"]
        c = self.action_spec.get("clip", {})
        for key in ("drive", "throttle"):
            if key in f:
                lo, hi = c.get(key, [-1.0, 1.0])
                a[int(f[key])] = _clip(drive, lo, hi)
                break
        if "steer" in f:
            lo, hi = c.get("steer", [-1.0, 1.0])
            a[int(f["steer"])] = _clip(steer, lo, hi)
        if "lift" in f:
            lo, hi = c.get("lift", [-1.0, 1.0])
            a[int(f["lift"])] = _clip(lift, lo, hi)
        return a

    # --------------------------------------------------------------- main
    def act(self, obs: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Compute expert action from a single 15-D observation.

        v7 FSM: Approach → (Straighten →) FinalInsert → Lift
                 ↕ HardAbort ↔ Approach
        """
        cfg = self.cfg

        # ---- Decode obs ----
        s = self._decode_obs(obs)
        dist = s["dist_front"]
        dtc  = s["dist_to_contact"]
        lat  = s["lat_true"]
        lat_clipped = s["lat_clipped"]
        yaw  = s["yaw_err"]
        insert_norm = s["insert_norm"]
        yaw_rate = s["yaw_rate"]
        v_forward = s["v_forward"]

        # ---- Cooldown tick ----
        if self._retreat_cooldown_remaining > 0:
            self._retreat_cooldown_remaining -= 1

        # ================================================================
        # FSM state transitions  (priority order)
        # ================================================================
        if insert_norm >= cfg.lift_on_insert_norm:
            self._fsm_stage = "Lift"

        elif self._fsm_stage == "HardAbort":
            if (dist >= cfg.retreat_target_dist
                    or self._retreat_steps >= cfg.max_retreat_steps):
                self._fsm_stage = "Approach"
                self._retreat_cooldown_remaining = cfg.retreat_cooldown
                self._retreat_steps = 0

        elif self._fsm_stage == "Straighten":
            if abs(lat) > cfg.abort_lat or abs(yaw) > cfg.abort_yaw:
                self._fsm_stage = "HardAbort"
                self._abort_reason = "alignment_lost"
                self._retreat_steps = 0
            elif abs(self._prev_steer) < cfg.steer_straight_ok:
                self._fsm_stage = "FinalInsert"

        elif self._fsm_stage == "FinalInsert":
            if abs(lat) > cfg.abort_lat or abs(yaw) > cfg.abort_yaw:
                self._fsm_stage = "HardAbort"
                self._abort_reason = "alignment_lost"
                self._retreat_steps = 0

        elif self._fsm_stage == "Approach":
            if dtc <= cfg.hard_wall:
                aligned = (abs(lat) < cfg.final_lat_ok
                           and abs(yaw) < cfg.final_yaw_ok)
                if aligned:
                    self._fsm_stage = "FinalInsert"
                else:
                    self._fsm_stage = "HardAbort"
                    self._abort_reason = "pose_not_aligned"
                    self._retreat_steps = 0

        # ================================================================
        # Per-stage control
        # ================================================================
        drive = 0.0
        raw_steer = 0.0
        lift = 0.0
        eff_max_steer = 1.0

        if self._fsm_stage == "Lift":
            drive = 0.0
            raw_steer = 0.0
            lift = cfg.lift_cmd

        elif self._fsm_stage == "HardAbort":
            drive = cfg.retreat_drive
            eff_max_steer = cfg.retreat_max_steer
            raw_steer = _clip(
                cfg.k_lat_rev * lat + cfg.k_yaw_rev * yaw,
                -eff_max_steer, eff_max_steer,
            )
            self._retreat_steps += 1
            if self._retreat_steps == 1:
                self._prev_steer = raw_steer

        elif self._fsm_stage == "Straighten":
            drive = 0.0
            raw_steer = 0.0

        elif self._fsm_stage == "FinalInsert":
            drive = cfg.final_insert_drive
            raw_steer = 0.0

        else:
            # ---- Approach: Stanley steering (A1) ----
            eff_max_steer = cfg.max_steer_far  # 0.80 in Approach

            effective_v = max(abs(v_forward), 0.2)
            crosstrack_term = math.atan2(cfg.k_e * lat, effective_v + cfg.k_soft)
            delta_rad = yaw + crosstrack_term + cfg.k_damp * yaw_rate
            raw_steer = _clip(-delta_rad * cfg.k_steer,
                              -eff_max_steer, eff_max_steer)

            # ---- Approach: speed control ----
            v = _clip(cfg.k_dist * dist, cfg.v_min, cfg.v_max)
            if dist <= cfg.slow_dist:
                v *= _clip(dist / max(cfg.slow_dist, 1e-3), 0.15, 1.0)

            misalign = max(
                abs(lat) / max(cfg.lat_ok, 1e-6),
                abs(yaw) / max(cfg.yaw_ok, 1e-6),
            )
            misalign = _clip(misalign, 0.0, 3.0)
            if dist < 1.5:
                speed_scale = 1.0 / (1.0 + 0.20 * misalign)
                speed_scale = max(speed_scale, 0.65)
            else:
                speed_scale = 1.0 / (1.0 + 0.08 * misalign)
                speed_scale = max(speed_scale, 0.90)
            drive = v * speed_scale

        # ---- Rate-limit for smoothness ----
        steer = self._rate_limit(
            raw_steer, self._prev_steer, cfg.steer_rate_limit
        )
        drive = self._rate_limit(
            drive, self._prev_throttle, cfg.throttle_rate_limit
        )

        self._prev_steer = steer
        self._prev_throttle = drive

        action = self._build_action(drive=drive, steer=steer, lift=lift)

        steer_sat_ratio = (abs(raw_steer) / max(eff_max_steer, 1e-6)
                           if eff_max_steer > 0 else 0.0)

        info = {
            "stage": self._fsm_stage,
            "fsm_stage": self._fsm_stage,
            "dist_front": dist,
            "dist_to_contact": dtc,
            "dist_to_center": s["dist_to_center"],
            "d_x": s["d_x"],
            "d_y": s["d_y"],
            "lat": lat,
            "lat_clipped": lat_clipped,
            "yaw": yaw,
            "insert_norm": insert_norm,
            "v_forward": v_forward,
            "contact_flag": s["contact_flag"],
            "slip_flag": s["slip_flag"],
            "raw_steer": raw_steer,
            "steer": steer,
            "drive": drive,
            "lift": lift,
            "eff_max_steer": eff_max_steer,
            "steer_sat_ratio": steer_sat_ratio,
            "abort_reason": self._abort_reason,
            "in_retreat": self._fsm_stage == "HardAbort",
            "retreat_steps": self._retreat_steps,
            "retreat_cooldown": self._retreat_cooldown_remaining,
        }
        return action, info
