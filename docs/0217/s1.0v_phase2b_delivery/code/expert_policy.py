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


# Optional planning layer for ALIGN stage (motion primitives + A*).
# Safe fallback: if the module is missing, policy behaves exactly as before.
try:
    from .align_lattice_planner_s10u import Pose2D as _AlignPose2D
    from .align_lattice_planner_s10u import PlannerParams as _AlignPlannerParams
    from .align_lattice_planner_s10u import plan_align as _plan_align_primitives
except Exception:
    _AlignPose2D = None
    _AlignPlannerParams = None
    _plan_align_primitives = None


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

    # ---- S1.0U: BBox / Fork Tip Geometry ----
    # fork_forward_offset = 1.8667m measured from USD mesh (see
    # docs/diagnostic_reports/success_sanity_check_2026-02-10.md L73).
    # Rounded up to 1.87m for safety margin.
    fork_reach: float = 1.87           # root to fork tip forward distance (m)
    fork_tip_lat_ok: float = 0.15      # max fork tip lateral error at contact (m)
                                       # relaxed 0.10→0.15: pallet opening has margin

    # ---- S1.0U: Pre-contact Safety Gates ----
    hard_wall: float = 1.92            # absolute safety line = fork_reach + 5cm
    hard_wall_hyst: float = 0.05       # Schmitt hysteresis for hard wall re-entry
    pre_insert_dist: float = 2.05      # corridor/decel zone start (m)
                                       # 2.40→2.05: Ackermann geometry self-aligns tip_lat
                                       # at hard_wall; corridor at 2.1-2.4 was false-positive
    bbox_abort_hold_steps: int = 8     # min steps to hold abort-triggered retreat
    bbox_retreat_target_dist: float = 2.5  # Phase-2b: from 2.7, more OOB margin
                                           # must be > pre_insert_dist to give full approach run

    # ---- S1.0U Stage-B: Near-field tip_lat fusion ----
    # Blend steering target from body-lat to fork-tip-lat as vehicle approaches.
    # Ensures controller optimises the SAME lateral error that the safety gate
    # checks, and produces stronger steer when tip_lat > lat (yaw lever arm).
    tip_blend_dist: float = 2.5             # start blending at this dist (m)
                                            # full tip_lat at hard_wall (1.92m)

    # ---- S1.0U: Contact boundary alignment (stricter than far-field) ----
    final_lat_ok: float = 0.15         # max root lateral error at hard wall (m)
    final_yaw_ok: float = 0.175        # max yaw error at hard wall (~10 deg)

    # ---- Docking (approach + align) ----
    k_lat: float = 1.1          # steering gain for lateral error — reduced from 1.5
                                # to prevent lateral overshoot (stress-test: 26% drift pattern)
    k_yaw: float = 0.9          # steering gain for yaw error   — reduced from 1.2
    k_damp: float = 0.20        # NEW: yaw-rate damping to suppress oscillation
    # ---- Stage-E Step 2: yaw_ref tracking ----
    k_yaw_ref: float = 2.0      # yaw_ref tracking gain (replaces k_lat+k_yaw coupling)
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

    # ---- S1.0U Stage-C: Alignment-based speed modulation ----
    # Linear lerp from full-speed to crawl based on alignment quality.
    # When |lat| or |yaw| exceed the "ok" threshold, speed begins to drop;
    # at the "slow" threshold, speed is fully reduced to crawl.
    align_lat_ok: float = 0.15                    # |lat| below -> no penalty
    align_lat_slow: float = 0.50                  # |lat| above -> full penalty (crawl)
    align_yaw_ok: float = math.radians(5.0)       # |yaw| below -> no penalty (~0.087 rad)
    align_yaw_slow: float = math.radians(15.0)    # |yaw| above -> full penalty (~0.262 rad)
    align_crawl_speed: float = 0.15               # floor speed (Ackermann needs some motion)

    # ---- S1.0V: Optional ALIGN planner (motion primitives) ----
    # If enabled, the policy can switch from local PD to a short-horizon
    # primitive plan when it detects repeated bbox_abort or severe misalignment.
    use_align_planner: bool = True
    align_goal_x: float = 2.10           # was 2.50; reduce handover gap to insertion zone
    align_goal_dx: float = 0.15          # was 0.12; slightly more tolerance
    align_goal_dy: float = 0.18          # was 0.12; more tolerance for handover
    align_goal_dyaw: float = math.radians(5.0)
    # Planner-internal goal box (can be looser than handover box to ease search)
    align_plan_goal_dx: float = 0.20
    align_plan_goal_dy: float = 0.20
    align_plan_goal_dyaw: float = math.radians(8.0)

    align_trigger_bbox_abort_count: int = 1
    align_fwd_drive: float = 0.30        # throttle during planned forward primitives
    align_rev_drive: float = -0.25       # Phase-2: from -0.35, limit reverse displacement
    align_planner_dt: float = 1.0/30.0   # was 0.02; MUST match env control dt (sim 1/120, dec 4)
    align_x_max_abs: float = 2.85        # Phase-2: from 3.0, 0.15m margin before env OOB
    align_x_headroom: float = 0.3        # Phase-2: from 0.6, tighter reverse leash
    align_trigger_cooldown_steps: int = 24  # after successful align, block re-trigger (anti ping-pong)

    # Planner parameter pass-through
    align_planner_max_expansions: int = 12000
    align_planner_v_rev: float = -0.20
    align_planner_reverse_penalty: float = 2.5
    align_planner_steer_levels: Tuple[float, ...] = (-1.0, -0.5, 0.0, 0.5, 1.0)

    align_max_replan_fails: int = 3      # consecutive fails -> freeze planner this episode

    # Drift monitoring: periodic check of actual vs predicted pose
    align_drift_check_every: int = 5    # check every N steps (not replan, just check)
    align_drift_x_tol: float = 0.15     # x (dist) drift tolerance before replan
    align_drift_y_tol: float = 0.10     # y (lat) drift tolerance before replan
    align_drift_yaw_tol: float = 0.10   # yaw drift tolerance (rad) before replan

    # ---- Retreat ----
    # Only trigger retreat when VERY close AND severely misaligned.
    # Stress-test showed docking-retreat cycling is the #1 failure mode;
    # the docking controller alone can correct |lat| up to 0.48 given
    # enough steps.  Retreat is a last resort for extreme cases.
    retreat_lat_thresh: float = 0.48    # near-saturated lat (metres)
    retreat_yaw_thresh: float = math.radians(35.0)  # large yaw
    retreat_dist_thresh: float = 2.0    # S1.0U: 1.0 -> 2.0, cover pre-contact zone
    retreat_target_dist: float = 1.8    # was 2.5 → 1.5 too short → 1.8 compromise
    retreat_drive: float = -1.0         # full backward speed
    retreat_steer_gain: float = 0.40    # Stage-E: restored (0.05→0.40) — sign fix makes
                                        # lat & yaw terms cooperate, no need to suppress
    retreat_k_yaw: float = 0.80        # Stage-E: lowered (1.50→0.80) — avoid tire saturation
    retreat_max_steer: float = 0.90    # v5-B: was hardcoded 0.8 — clip limit for retreat steer
    retreat_lat_sat: float = 0.75      # v5-B: lat at which retreat_lat_term saturates (was implicit 0.5)
    retreat_exit_lat_abs: float = 0.40 # v5-B: was 0.30 — abs lat threshold for alignment exit
    retreat_exit_yaw_max: float = math.radians(30)  # v5-B: yaw must also be OK to exit retreat
    max_retreat_steps: int = 200        # 80→200: allow deeper retreat to fully correct alignment
    retreat_cooldown: int = 150         # was 80 — give docking more time to self-correct

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
    lift_on_insert_norm: float = 0.40  # Phase-2b: was 0.75, physical max ~0.477
    lift_cmd: float = 0.60

    # ---- Safety / smoothness ----
    steer_rate_limit: float = 0.35     # max delta-steer per step
    throttle_rate_limit: float = 0.50  # max delta-throttle per step — faster acceleration
    deadband_steer: float = 0.02

    # ---- Stage heuristic ----
    use_insert_norm_for_stage: bool = True
    insert_enter_stage: float = 0.15   # reverted from 0.05; premature entry caused pushing against pallet side


# ---------------------------------------------------------------------------
# Expert policy
# ---------------------------------------------------------------------------
class ForkliftExpertPolicy:
    """
    A rule-based expert policy.

    It consumes the 15-D obs vector from the IsaacLab forklift env and emits
    a 3-D action vector ``[drive, steer, lift]``.

    Stages
    ------
    * **Retreat**   : back up when too close + severely misaligned
    * **Docking**   : align + approach to the pallet front
    * **Insertion** : low-speed insertion with stricter alignment gates
    * **Lift**      : lift after sufficient insertion depth
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
        self._backoff_countdown: int = 0

        # Retreat state
        self._in_retreat: bool = False
        self._retreat_steps: int = 0
        self._retreat_cooldown_remaining: int = 0
        self._retreat_entry_lat: float = 0.0  # |lat| when retreat started
        self._retreat_entry_yaw: float = 0.0  # |yaw| when retreat started
        self._retreat_exit_reason: str = ""   # last exit reason for logging

        # S1.0U: BBox abort state
        self._bbox_abort_hold: int = 0       # remaining hold steps for abort retreat
        self._bbox_abort_reason: str = ""    # last abort reason
        self._bbox_abort_count: int = 0      # cumulative abort count (for logging)
        self._align_armed: bool = False      # S1.0V: armed/disarm hysteresis for planner trigger
        self._passed_hard_wall: bool = False # Schmitt: once passed, use relaxed exit
        self._retreat_is_bbox: bool = False  # current retreat triggered by bbox_abort?

        # Stage-C: episode-boundary detector
        self._prev_dist_front: float | None = None

        # S1.0V: ALIGN planner state (optional)
        self._align_active: bool = False
        self._align_plan: list | None = None
        self._align_plan_i: int = 0
        self._align_steps_left: int = 0
        self._align_cur_drive: float = 0.0
        self._align_cur_steer: float = 0.0
        self._align_replan_fail_streak: int = 0
        self._align_trigger_cooldown: int = 0  # S1.0V: post-align cooldown counter
        self._align_plan_snapshot: dict | None = None  # S1.0V: one-shot snapshot for logging
        self._align_predicted_path: list | None = None  # predicted (x,y,yaw) tuples from planner
        self._align_path_cursor: int = 0        # index into predicted_path for drift check
        self._align_drift_check_cd: int = 0     # countdown to next drift check
        self._align_drift: tuple | None = None  # latest (drift_x, drift_y, drift_yaw) or None
        self._align_replan_reason: str = ""     # why last replan happened

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
        self._backoff_countdown = 0
        self._in_retreat = False
        self._retreat_steps = 0
        self._retreat_cooldown_remaining = 0
        self._retreat_entry_lat = 0.0
        self._retreat_entry_yaw = 0.0
        self._retreat_exit_reason = ""
        # S1.0U
        self._bbox_abort_hold = 0
        self._bbox_abort_reason = ""
        self._bbox_abort_count = 0
        self._align_armed = False
        self._passed_hard_wall = False
        self._retreat_is_bbox = False
        # Stage-C: clear episode-boundary detector
        self._prev_dist_front = None

        # S1.0V: ALIGN planner
        self._align_active = False
        self._align_plan = None
        self._align_plan_i = 0
        self._align_steps_left = 0
        self._align_cur_drive = 0.0
        self._align_cur_steer = 0.0
        self._align_replan_fail_streak = 0
        self._align_trigger_cooldown = 0
        self._align_plan_snapshot = None
        self._align_predicted_path = None
        self._align_path_cursor = 0
        self._align_drift_check_cd = 0
        self._align_drift = None
        self._align_replan_reason = ""

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
        lat_clipped = y_err_norm * 0.8  # metres, clipped to [-0.8, +0.8] (env_cfg.y_err_obs_scale=0.8)

        # --- Other scalars --------------------------------------------------
        insert_norm = _r(obs, self._idx_insert_norm, default=0.0)
        v_forward   = _r(obs, self._idx_v_forward, default=0.0)
        yaw_rate    = _r(obs, self._idx_yaw_rate, default=0.0)
        lift_pos    = _r(obs, self._idx_lift_pos, default=0.0)

        # contact / slip are NOT present in the 15-D obs -- always 0
        contact_flag = 0.0
        slip_flag    = 0.0

        return {
            "dist_front": dist_front,
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
        """Compute expert action from a single 15-D observation."""
        cfg = self.cfg

        # ---- Decode obs into semantic fields ----
        s = self._decode_obs(obs)
        dist = s["dist_front"]

        # S1.0V: align planner trigger cooldown (prevents align<->docking ping-pong)
        if self._align_trigger_cooldown > 0:
            self._align_trigger_cooldown -= 1

        # S1.0U Stage-C: Implicit reset on episode boundary.
        # IsaacLab resets the env without calling policy.reset(); detect
        # this by a sudden large increase in dist (new episode spawns far
        # from pallet) and flush stale internal state.
        if self._prev_dist_front is not None and dist > self._prev_dist_front + 0.5:
            self.reset()
        self._prev_dist_front = dist

        lat  = s["lat_true"]       # v5-A: use unsaturated pallet-frame lat
        lat_clipped = s["lat_clipped"]  # for logging only
        yaw  = s["yaw_err"]
        insert_norm  = s["insert_norm"]
        contact_flag = s["contact_flag"]
        slip_flag    = s["slip_flag"]

        # Unsaturated lateral signal from robot-frame d_y.
        lat_unsaturated = abs(s["d_y"])

        # ---- S1.0U: Fork tip lateral projection ----
        # Lever-arm effect: fork tip sweeps laterally when yaw != 0.
        # yaw = pallet_yaw - robot_yaw, so robot heading in pallet frame = -yaw.
        # tip_lat = lat + reach * sin(-yaw) = lat - reach * sin(yaw).
        tip_lat = lat - cfg.fork_reach * math.sin(yaw)

        # Dynamic safety corridor: narrows as forklift approaches contact.
        safe_corridor = cfg.fork_tip_lat_ok + max(0.0, dist - cfg.fork_reach) * 0.8

        # Is fork tip aligned at the contact boundary?
        tip_aligned = abs(tip_lat) < cfg.fork_tip_lat_ok
        body_aligned = abs(lat) < cfg.final_lat_ok and abs(yaw) < cfg.final_yaw_ok

        # ---- Stage decision (insertion) ----
        # S1.0U: also require tip_lat alignment to enter insertion.
        in_insertion = False
        if cfg.use_insert_norm_for_stage:
            in_insertion = (
                insert_norm >= cfg.insert_enter_stage
                and abs(lat) < cfg.ins_stage_lat_gate
                and abs(yaw) < cfg.ins_stage_yaw_gate
                and abs(tip_lat) < cfg.fork_tip_lat_ok  # S1.0U
            )
        else:
            in_insertion = (
                dist <= cfg.stop_dist + 0.05
                and abs(lat) <= cfg.lat_ok * 1.5
                and abs(yaw) <= cfg.yaw_ok * 1.5
            )

        # ---- Backoff trigger ----
        if cfg.backoff_on_contact and (contact_flag > 0.5 or slip_flag > 0.5):
            if in_insertion:
                self._backoff_countdown = max(
                    self._backoff_countdown, cfg.backoff_steps
                )

        # ---- S1.0U: BBox collision prediction ----
        # Decrement abort hold counter each step.
        bbox_abort = False
        if self._bbox_abort_hold > 0:
            self._bbox_abort_hold -= 1
            bbox_abort = True  # still in forced retreat from prior abort

        if not in_insertion and not bbox_abort:
            # Layer 1: Dynamic safety corridor check
            if dist < cfg.pre_insert_dist and abs(tip_lat) > safe_corridor:
                bbox_abort = True
                self._bbox_abort_hold = cfg.bbox_abort_hold_steps
                self._bbox_abort_reason = "corridor"
                self._bbox_abort_count += 1
                self._align_armed = True

            # Layer 2: Hard wall gate (Schmitt trigger)
            # Check both tip AND body alignment at the entry boundary.
            # tip_aligned alone can pass a severely yawed body, causing
            # "diagonal insertion" jam (fork sides hit pallet slot walls).
            wall_enter = cfg.hard_wall
            wall_exit = cfg.hard_wall + cfg.hard_wall_hyst
            if dist <= wall_enter:
                self._passed_hard_wall = True
            if self._passed_hard_wall:
                if dist > wall_exit:
                    self._passed_hard_wall = False
                elif not (tip_aligned and body_aligned):
                    bbox_abort = True
                    self._bbox_abort_hold = cfg.bbox_abort_hold_steps
                    self._bbox_abort_reason = "hard_wall"
                    self._bbox_abort_count += 1
                    self._align_armed = True

        # ---- Retreat trigger ----
        # Cooldown prevents rapid retreat-dock cycling that wastes steps.
        if self._retreat_cooldown_remaining > 0:
            self._retreat_cooldown_remaining -= 1

        need_retreat = (
            not in_insertion
            and insert_norm < cfg.lift_on_insert_norm
            and dist < cfg.retreat_dist_thresh
            and self._retreat_cooldown_remaining <= 0
            and (abs(lat) >= cfg.retreat_lat_thresh
                 or abs(yaw) >= cfg.retreat_yaw_thresh)
        )

        # S1.0U: bbox_abort also forces retreat (bypass cooldown).
        if bbox_abort and not self._in_retreat:
            need_retreat = True
            self._retreat_is_bbox = True

        # ---- Compute steer (shared across non-retreat stages) ----
        # SIGN (Stage-E fix): positive lat (偏左) -> positive steer (向右修正)
        # RWS Ackermann forward: positive steer rotates vehicle rightward.
        # PD controller: proportional on lat+yaw, derivative (damping) on yaw_rate
        yaw_rate = s["yaw_rate"]

        # Stage-E Step 2: yaw_ref tracking architecture.
        # Replaces tip_lat fusion + coupled PD with a decoupled design:
        #   1) Compute distance-weighted fork-tip target
        #   2) Geometrically invert to target heading (yaw_ref)
        #   3) Track yaw_ref with pure angle PD (no k_lat term → no coupling)
        #
        # This eliminates the k_yaw_eff sign-flip that GPT proved occurs
        # when w_tip > 0.44 in the old linear-blend approach.

        # 1. Distance blending weight: 0 far away, 1 at hard_wall
        alpha = _clip((cfg.pre_insert_dist - dist)
                      / (cfg.pre_insert_dist - cfg.hard_wall), 0.0, 1.0)

        # 2. Fork-tip lateral target:
        #    Far (alpha~0): tip_target~0 → yaw_ref drives lat toward 0
        #    Near (alpha~1): tip_target relaxes to ±tip_ok, easing final entry
        tip_target = alpha * _clip(lat, -cfg.fork_tip_lat_ok, cfg.fork_tip_lat_ok)

        # 3. Geometric inverse: what yaw makes tip_lat == tip_target?
        #    tip_lat = lat - fork_reach * sin(yaw) = tip_target
        #    → sin(yaw_ref) = (lat - tip_target) / fork_reach
        sin_yaw_ref = _clip((lat - tip_target) / cfg.fork_reach, -0.999, 0.999)
        yaw_ref = math.asin(sin_yaw_ref)

        # 4. Pure angle PD tracking (fully decoupled from lat)
        raw_steer = cfg.k_yaw_ref * (yaw_ref - yaw) - cfg.k_damp * yaw_rate

        # Fix C: near-distance gain decay to prevent overshoot
        if dist < 1.0:
            gain_scale = max(0.4, dist / 1.0)
            raw_steer *= gain_scale

        if abs(raw_steer) < cfg.deadband_steer:
            raw_steer = 0.0

        # Fix B: distance-adaptive steer limit (aggressive far, gentle near)
        if dist > 2.0:
            eff_max_steer = cfg.max_steer_far
        elif dist < 0.8:
            eff_max_steer = cfg.max_steer_near
        else:
            t = (dist - 0.8) / 1.2
            eff_max_steer = cfg.max_steer_near + t * (cfg.max_steer_far - cfg.max_steer_near)

        # v5-C: lat-dependent bonus (only when dist >= 0.8m to protect near-pallet)
        if dist >= 0.8 and abs(lat) > cfg.max_steer_lat_bonus_start:
            lat_excess = abs(lat) - cfg.max_steer_lat_bonus_start
            bonus = min(lat_excess * 0.2, cfg.max_steer_lat_bonus_max)
            eff_max_steer += bonus

        # S1.0U: tighten steer limit inside decel zone to reduce tip sweep rate
        if dist < cfg.pre_insert_dist:
            decel_t = _clip((cfg.pre_insert_dist - dist)
                            / (cfg.pre_insert_dist - cfg.fork_reach), 0.0, 1.0)
            eff_max_steer *= (1.0 - 0.4 * decel_t)  # shrink up to 40%

        raw_steer = _clip(raw_steer, -eff_max_steer, eff_max_steer)

        # ---- Compute drive + lift + steer by stage ----
        drive = 0.0
        lift = 0.0
        align_penalty = 0.0   # Stage-C diagnostic; overwritten in docking stage
        stage = "docking"

        if insert_norm >= cfg.lift_on_insert_norm:
            # -------- Lift stage --------
            stage = "lift"
            drive = 0.15      # Phase-2b: small forward push to prevent physics jitter pullback
            raw_steer = 0.0   # Phase-2b: suppress lateral force during lift
            lift = cfg.lift_cmd

        elif self._in_retreat or need_retreat:
            # -------- Retreat stage --------
            if not self._in_retreat:
                self._in_retreat = True
                self._retreat_steps = 0
                self._retreat_entry_lat = abs(lat)
                self._retreat_entry_yaw = abs(yaw)  # v5-B: record entry yaw
                self._retreat_exit_reason = ""

            # Exit conditions: alignment improved OR distance/step budget reached
            # v5-B: added yaw constraint to prevent "early-exit + cooldown = drift"
            alignment_improved = (
                abs(lat) < self._retreat_entry_lat * 0.6   # lat improved 40%+
                and abs(lat) < cfg.retreat_exit_lat_abs     # v5-B: was 0.30, now config (0.40)
                and abs(yaw) < cfg.retreat_exit_yaw_max     # v5-B: yaw must also be OK
                and dist > 1.2                              # enough room to re-approach
            )

            # S1.0U: if retreat was triggered by bbox_abort, also require
            # tip_lat to be within corridor before allowing exit.
            if self._bbox_abort_hold > 0:
                alignment_improved = False  # force hold until counter expires
            elif self._retreat_is_bbox:
                # bbox-triggered retreat must reach eff_retreat_target distance;
                # alignment_improved alone exits too early (lat crosses zero
                # during retreat), causing bbox_abort re-trigger oscillation.
                alignment_improved = False

            # Determine exit reason
            # bbox_abort retreats use a longer target distance to allow full re-approach.
            eff_retreat_target = (cfg.bbox_retreat_target_dist
                                 if self._retreat_is_bbox
                                 else cfg.retreat_target_dist)
            if alignment_improved:
                retreat_done = True
                self._retreat_exit_reason = "alignment"
            elif dist >= cfg.align_x_max_abs:
                # Phase-2b: hard OOB guard — env terminates at dist>3.0 for 30 steps,
                # so stop retreat right at align_x_max_abs (2.85) to prevent accumulation
                retreat_done = True
                self._retreat_exit_reason = "oob_guard"
            elif dist >= eff_retreat_target:
                retreat_done = True
                self._retreat_exit_reason = "target_dist"
            elif self._retreat_steps >= cfg.max_retreat_steps:
                retreat_done = True
                self._retreat_exit_reason = "max_steps"
            else:
                retreat_done = False

            if retreat_done:
                self._in_retreat = False
                self._retreat_steps = 0
                self._retreat_is_bbox = False
                self._retreat_cooldown_remaining = cfg.retreat_cooldown
            else:
                stage = "retreat"
                drive = cfg.retreat_drive
                # Phase-2b: deceleration band near OOB boundary
                if dist > 2.3:
                    fade = min((dist - 2.3) / 0.4, 1.0)  # 2.3->2.7: 0->1
                    drive = drive * (1.0 - 0.7 * fade)    # -1.0 -> -0.3
                # v5-B: parameterised lat_term with retreat_lat_sat
                # (decouples slope from saturation point)
                # Stage-E: negated so lat>0 (偏左) produces negative steer
                # during reverse, making lat & yaw terms cooperate instead of cancel.
                retreat_lat_term = (
                    -math.copysign(min(abs(lat) / cfg.retreat_lat_sat, 1.0), lat)
                    * cfg.retreat_steer_gain
                )
                # Stage-E: negated — during reverse, negative steer increases yaw,
                # so yaw>0 needs negative steer to push yaw toward zero.
                retreat_yaw_term = -yaw * cfg.retreat_k_yaw
                retreat_steer = retreat_lat_term + retreat_yaw_term
                raw_steer = _clip(retreat_steer,
                                  -cfg.retreat_max_steer, cfg.retreat_max_steer)
                self._retreat_steps += 1

                # v5-B: skip steer rate_limit on first retreat step
                # to eliminate ~3-step ramp-up delay
                if self._retreat_steps == 1:
                    self._prev_steer = raw_steer

        if stage not in ("lift", "retreat"):
            if in_insertion:
                # -------- Insertion stage --------
                stage = "insertion"

                if self._backoff_countdown > 0:
                    drive = cfg.backoff_throttle
                    self._backoff_countdown -= 1
                else:
                    aligned = (
                        abs(lat) <= cfg.ins_lat_ok
                        and abs(yaw) <= cfg.ins_yaw_ok
                    )
                    if aligned:
                        base = cfg.ins_v_max
                        slow = 1.0
                        if dist <= cfg.slow_dist:
                            slow = _clip(
                                dist / max(cfg.slow_dist, 1e-3), 0.2, 1.0
                            )
                        drive = _clip(base * slow, cfg.ins_v_min, cfg.ins_v_max)
                    else:
                        # Not aligned -- keep a meaningful creep speed so
                        # Ackermann steering can still correct the heading.
                        drive = 0.30

            else:
                # -------- Docking stage --------
                stage = "docking"

                # S1.0V: Optional ALIGN planning layer.
                # Triggered by repeated bbox_abort OR severe misalignment near approach zone.
                if cfg.use_align_planner and _plan_align_primitives is not None:
                    severe_misalign = (abs(lat) > cfg.align_lat_ok or abs(yaw) > cfg.align_yaw_ok)
                    near_zone = dist <= (cfg.tip_blend_dist + 0.6)

                    # S1.0V Phase-1: triple-gated trigger
                    within_x = dist <= (cfg.align_x_max_abs - 0.05)
                    trigger = within_x and (
                        (self._align_armed and self._bbox_abort_count >= cfg.align_trigger_bbox_abort_count)
                        or (near_zone and severe_misalign)
                    ) and self._align_trigger_cooldown <= 0

                    # Enter planner mode
                    if trigger and not self._align_active and self._align_replan_fail_streak < cfg.align_max_replan_fails:
                        self._align_active = True
                        self._align_plan = None
                        self._align_plan_i = 0
                        self._align_steps_left = 0

                    # Exit planner mode once parked in the align box
                    # Single-sided dist check: allow being closer than goal_x (don't force reverse)
                    in_align_box = (
                        dist <= (cfg.align_goal_x + cfg.align_goal_dx)
                        and abs(lat) <= cfg.align_goal_dy
                        and abs(yaw) <= cfg.align_goal_dyaw
                    )
                    if self._align_active and in_align_box:
                        self._align_active = False
                        self._align_plan = None
                        self._align_replan_fail_streak = 0
                        self._align_armed = False
                        self._align_trigger_cooldown = cfg.align_trigger_cooldown_steps

                    # Execute (or refresh) a primitive plan
                    if self._align_active:
                        # --- Event-driven replan decision ---
                        self._align_drift = None
                        need_replan = False
                        replan_reason = ""

                        if self._align_plan is None:
                            need_replan = True
                            replan_reason = "init"
                        elif self._align_plan_i >= len(self._align_plan):
                            need_replan = True
                            replan_reason = "plan_done"
                        elif bbox_abort:
                            need_replan = True
                            replan_reason = "bbox_abort"

                        # Drift check: compare actual pose vs predicted path
                        if not need_replan and self._align_predicted_path:
                            self._align_drift_check_cd -= 1
                            if self._align_drift_check_cd <= 0:
                                self._align_drift_check_cd = cfg.align_drift_check_every
                                idx = min(self._align_path_cursor,
                                          len(self._align_predicted_path) - 1)
                                px, py, pyaw = self._align_predicted_path[idx]
                                drift_x = abs(dist - px)
                                drift_y = abs(lat - py)
                                drift_yaw = abs(_wrap_pi(yaw - pyaw))
                                self._align_drift = (drift_x, drift_y, drift_yaw)
                                if (drift_x > cfg.align_drift_x_tol
                                        or drift_y > cfg.align_drift_y_tol
                                        or drift_yaw > cfg.align_drift_yaw_tol):
                                    need_replan = True
                                    replan_reason = "drift"

                        if need_replan:
                            self._align_replan_reason = replan_reason
                            prm = _AlignPlannerParams(
                                dt=cfg.align_planner_dt,
                                steer_angle_rad=0.6,  # Phase-2b: match env steer scaling
                                max_steer=cfg.max_steer_far,
                                fork_reach=cfg.fork_reach,
                                fork_tip_lat_ok=cfg.fork_tip_lat_ok,
                                pre_insert_dist=cfg.pre_insert_dist,
                                hard_wall=cfg.hard_wall,
                                final_lat_ok=cfg.final_lat_ok,
                                final_yaw_ok=cfg.final_yaw_ok,
                                goal_x=cfg.align_goal_x,
                                goal_dx=cfg.align_plan_goal_dx,
                                goal_dy=cfg.align_plan_goal_dy,
                                goal_dyaw=cfg.align_plan_goal_dyaw,
                                max_expansions=cfg.align_planner_max_expansions,
                                steer_levels=cfg.align_planner_steer_levels,
                                v_rev=cfg.align_planner_v_rev,
                                reverse_penalty=cfg.align_planner_reverse_penalty,
                                x_max=min(cfg.align_x_max_abs, dist + cfg.align_x_headroom),
                                y_max_abs=3.0,
                            )
                            res = _plan_align_primitives(
                                _AlignPose2D(dist, lat, yaw), prm
                            )
                            if res is not None and len(res.primitives) > 0:
                                self._align_plan = res.primitives
                                self._align_plan_i = 0
                                self._align_replan_fail_streak = 0
                                self._align_predicted_path = [
                                    (p.x, p.y, p.yaw) for p in res.predicted_path
                                ] if res.predicted_path else None
                                self._align_path_cursor = 0
                                self._align_drift_check_cd = cfg.align_drift_check_every
                                self._align_plan_snapshot = {
                                    "start": (dist, lat, yaw),
                                    "final": (res.final_pose.x, res.final_pose.y, res.final_pose.yaw),
                                    "expansions": res.expansions,
                                    "x_max": prm.x_max,
                                    "reason": replan_reason,
                                    "primitives": [
                                        {"steer": p.steer, "v": p.v, "dur": p.duration}
                                        for p in res.primitives
                                    ],
                                }
                            else:
                                self._align_replan_fail_streak += 1
                                if self._align_replan_fail_streak >= cfg.align_max_replan_fails:
                                    self._align_active = False
                                    self._align_plan = None

                        # Load next primitive when needed
                        if self._align_active and self._align_plan is not None:
                            if self._align_steps_left <= 0 and self._align_plan_i < len(self._align_plan):
                                prim = self._align_plan[self._align_plan_i]
                                self._align_plan_i += 1
                                self._align_steps_left = max(
                                    1, int(math.ceil(prim.duration / cfg.align_planner_dt - 1e-9))
                                )
                                self._align_cur_steer = prim.steer
                                self._align_cur_drive = (
                                    cfg.align_fwd_drive if prim.v >= 0.0
                                    else cfg.align_rev_drive
                                )
                                self._align_path_cursor += self._align_steps_left

                            if self._align_steps_left > 0:
                                stage = "align_plan"
                                drive = self._align_cur_drive
                                raw_steer = self._align_cur_steer
                                self._align_steps_left -= 1

                # If not in align_plan, use original docking controller
                if stage != "align_plan":
                    v = _clip(cfg.k_dist * dist, cfg.v_min, cfg.v_max)
                    if dist <= cfg.slow_dist:
                        v *= _clip(dist / max(cfg.slow_dist, 1e-3), 0.15, 1.0)

                    # S1.0U: deceleration zone
                    if dist < cfg.pre_insert_dist:
                        decel_t = _clip((cfg.pre_insert_dist - dist)
                                        / (cfg.pre_insert_dist - cfg.fork_reach),
                                        0.0, 1.0)
                        v_cap = cfg.v_max * (1.0 - 0.5 * decel_t)
                        v = min(v, v_cap)

                    # S1.0U Stage-C: Alignment-based speed modulation
                    lat_p = _clip(
                        (abs(lat) - cfg.align_lat_ok)
                        / (cfg.align_lat_slow - cfg.align_lat_ok),
                        0.0, 1.0,
                    )
                    yaw_p = _clip(
                        (abs(yaw) - cfg.align_yaw_ok)
                        / (cfg.align_yaw_slow - cfg.align_yaw_ok),
                        0.0, 1.0,
                    )
                    align_penalty = max(lat_p, yaw_p)
                    v = v * (1.0 - align_penalty) + cfg.align_crawl_speed * align_penalty
                    drive = v

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

        info = {
            "stage": stage,
            "dist_front": dist,
            "dist_to_center": s["dist_to_center"],
            "d_x": s["d_x"],
            "d_y": s["d_y"],
            "lat": lat,              # lat_true (unsaturated, for control)
            "lat_clipped": lat_clipped,  # y_err_obs * 0.5 (for comparison)
            "yaw": yaw,
            "insert_norm": insert_norm,
            "v_forward": s["v_forward"],
            "contact_flag": contact_flag,
            "slip_flag": slip_flag,
            "raw_steer": raw_steer,
            "steer": steer,
            "drive": drive,
            "lift": lift,
            "backoff_countdown": self._backoff_countdown,
            "in_retreat": self._in_retreat,
            "retreat_steps": self._retreat_steps,
            "retreat_exit_reason": self._retreat_exit_reason,
            # S1.0U: bbox safety diagnostics
            "tip_lat": tip_lat,
            "safe_corridor": safe_corridor,
            "tip_aligned": tip_aligned,
            "body_aligned": body_aligned,
            "bbox_abort": bbox_abort,
            "bbox_abort_reason": self._bbox_abort_reason,
            "bbox_abort_count": self._bbox_abort_count,
            "passed_hard_wall": self._passed_hard_wall,
            # S1.0U Stage-C: alignment speed modulation diagnostic
            "align_penalty": align_penalty,
            # S1.0V: planner diagnostics
            "align_active": self._align_active,
            "align_plan_len": len(self._align_plan) if self._align_plan else 0,
            "align_plan_i": self._align_plan_i,
            "align_replan_fail_streak": self._align_replan_fail_streak,
            "align_armed": self._align_armed,
            "align_trigger_cooldown": self._align_trigger_cooldown,
            "align_plan_snapshot": self._align_plan_snapshot,
            "align_drift": self._align_drift,
            "align_replan_reason": self._align_replan_reason,
        }
        self._align_plan_snapshot = None
        return action, info
