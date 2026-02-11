"""
Rule-based expert policy for forklift pallet-insertion task (路线 A).

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
    """Tunable knobs — calibrated for the 15-D obs produced by the IsaacLab
    forklift environment.

    Key units
    ---------
    * ``dist``  : metres  (from ``d_xy_r[0]``, forward distance to pallet
                  *center*; subtract ``pallet_half_depth`` to approximate
                  distance to pallet front opening).
    * ``lat``   : metres  (de-normalised from ``y_err_obs * 0.5``).
    * ``yaw``   : radians (recovered via ``atan2(sin_dyaw, cos_dyaw)``).
    """

    # ---- Geometry: pallet size ----
    # pallet_depth_m = 2.16 (from env_cfg); half is used to estimate
    # distance-to-front from distance-to-center.
    pallet_half_depth: float = 1.08

    # ---- Docking (approach + align) ----
    k_lat: float = 2.0          # steering gain for lateral error (m → normalised)
    k_yaw: float = 1.2          # steering gain for yaw error   (rad → normalised)
    k_dist: float = 0.5         # throttle gain for distance     (m → normalised)
    v_max: float = 0.80         # max forward command (normalised action)
    v_min: float = 0.08         # min forward command when moving forward
    slow_dist: float = 1.5      # start slowing when closer than this (m, to pallet center)
    stop_dist: float = 0.5      # docking "arrived" gate (m, to pallet center)

    # alignment thresholds  (used to decide "advance vs. adjust")
    lat_ok: float = 0.06        # 6 cm  — acceptable lateral error for docking
    yaw_ok: float = math.radians(5.0)  # 5 deg

    # ---- Insertion ----
    ins_v_max: float = 0.35
    ins_v_min: float = 0.05
    ins_lat_ok: float = 0.03    # 3 cm  — must be tighter than docking
    ins_yaw_ok: float = math.radians(3.0)  # 3 deg

    # Contact / slip backoff — **disabled** by default because the 15-D obs
    # does NOT include contact_flag or slip_flag.
    backoff_on_contact: bool = False
    backoff_throttle: float = -0.20
    backoff_steps: int = 6

    # ---- Lift ----
    lift_on_insert_norm: float = 0.75
    lift_cmd: float = 0.60

    # ---- Safety / smoothness ----
    steer_rate_limit: float = 0.20     # max delta-steer per step
    throttle_rate_limit: float = 0.25  # max delta-throttle per step
    deadband_steer: float = 0.02

    # ---- Stage heuristic ----
    use_insert_norm_for_stage: bool = True
    insert_enter_stage: float = 0.15   # insert_norm threshold to enter insertion stage


# ---------------------------------------------------------------------------
# Expert policy
# ---------------------------------------------------------------------------
class ForkliftExpertPolicy:
    """
    A rule-based expert policy (路线 A).

    It consumes the 15-D obs vector from the IsaacLab forklift env and emits
    a 3-D action vector ``[drive, steer, lift]``.

    Stages
    ------
    * **Docking** : align + approach to the pallet front
    * **Insertion** : low-speed insertion with stricter alignment gates
    * **Lift** : lift after sufficient insertion depth
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
          dist_front, lateral_err, yaw_err, insert_norm,
          v_forward, yaw_rate, lift_pos, contact_flag, slip_flag
        """
        _r = self._safe_read  # shorthand

        # --- Forward distance (robot frame) ---------------------------------
        # d_xy_r is the 2-D vector from robot to pallet *center* in robot
        # frame.  d_xy_r[0] (x, forward) is positive when the pallet is
        # ahead.  We subtract half-pallet-depth to estimate distance to the
        # pallet front opening (where the forks enter).
        d_x = _r(obs, self._idx_d_xy_r_x, default=2.0)
        d_y = _r(obs, self._idx_d_xy_r_y, default=0.0)
        dist_to_center = math.sqrt(d_x ** 2 + d_y ** 2)
        dist_front = max(d_x - self.cfg.pallet_half_depth, 0.0)

        # --- Yaw error (full-range radians) --------------------------------
        cos_dy = _r(obs, self._idx_cos_dyaw, default=1.0)
        sin_dy = _r(obs, self._idx_sin_dyaw, default=0.0)
        yaw_err = math.atan2(sin_dy, cos_dy)  # [-pi, pi]

        # --- Lateral error (metres, from pallet center-line frame) ---------
        # y_err_obs is normalised by 0.5 m and clipped to [-1, 1].
        # Denormalise to get approximate raw metres (capped at ±0.5 m).
        y_err_norm = _r(obs, self._idx_y_err_obs, default=0.0)
        lateral_err = y_err_norm * 0.5  # metres

        # --- Other scalars --------------------------------------------------
        insert_norm = _r(obs, self._idx_insert_norm, default=0.0)
        v_forward   = _r(obs, self._idx_v_forward, default=0.0)
        yaw_rate    = _r(obs, self._idx_yaw_rate, default=0.0)
        lift_pos    = _r(obs, self._idx_lift_pos, default=0.0)

        # contact / slip are NOT present in the 15-D obs — always 0
        contact_flag = 0.0
        slip_flag    = 0.0

        return {
            "dist_front": dist_front,
            "dist_to_center": dist_to_center,
            "d_x": d_x,
            "d_y": d_y,
            "lateral_err": lateral_err,
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
        # drive (was "throttle" in earlier spec versions)
        for key in ("drive", "throttle"):
            if key in f:
                lo, hi = c.get(key, [-1.0, 1.0])
                a[int(f[key])] = _clip(drive, lo, hi)
                break
        # steer
        if "steer" in f:
            lo, hi = c.get("steer", [-1.0, 1.0])
            a[int(f["steer"])] = _clip(steer, lo, hi)
        # lift
        if "lift" in f:
            lo, hi = c.get("lift", [-1.0, 1.0])
            a[int(f["lift"])] = _clip(lift, lo, hi)
        return a

    # --------------------------------------------------------------- main
    def act(self, obs: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Compute expert action from a single 15-D observation.

        Args:
            obs: shape ``(obs_dim,)``

        Returns:
            action: shape ``(action_dim,)``
            info: dict with debug scalars
        """
        cfg = self.cfg

        # ---- Decode obs into semantic fields ----
        s = self._decode_obs(obs)
        dist = s["dist_front"]
        lat  = s["lateral_err"]
        yaw  = s["yaw_err"]
        insert_norm  = s["insert_norm"]
        contact_flag = s["contact_flag"]
        slip_flag    = s["slip_flag"]

        # ---- Stage decision ----
        in_insertion = False
        if cfg.use_insert_norm_for_stage:
            in_insertion = insert_norm >= cfg.insert_enter_stage
        else:
            in_insertion = (
                dist <= cfg.stop_dist + 0.05
                and abs(lat) <= cfg.lat_ok * 1.5
                and abs(yaw) <= cfg.yaw_ok * 1.5
            )

        # ---- Backoff trigger (only if contact/slip obs are available) ----
        if cfg.backoff_on_contact and (contact_flag > 0.5 or slip_flag > 0.5):
            if in_insertion:
                self._backoff_countdown = max(
                    self._backoff_countdown, cfg.backoff_steps
                )

        # ---- Compute steer (shared across stages) ----
        raw_steer = cfg.k_lat * lat + cfg.k_yaw * yaw
        if abs(raw_steer) < cfg.deadband_steer:
            raw_steer = 0.0
        raw_steer = _clip(raw_steer, -1.0, 1.0)

        # ---- Compute drive + lift by stage ----
        drive = 0.0
        lift = 0.0
        stage = "docking"

        if insert_norm >= cfg.lift_on_insert_norm:
            # -------- Lift stage --------
            stage = "lift"
            drive = 0.0
            lift = cfg.lift_cmd

        elif in_insertion:
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
                    # not aligned → stop forward, only steer
                    drive = 0.0

        else:
            # -------- Docking stage --------
            stage = "docking"

            # Speed profile: proportional to distance, with slow-down zone
            v = _clip(cfg.k_dist * dist, cfg.v_min, cfg.v_max)
            if dist <= cfg.slow_dist:
                v *= _clip(dist / max(cfg.slow_dist, 1e-3), 0.15, 1.0)

            # Reduce speed when misaligned to leave room for steering
            misalign = max(
                abs(lat) / max(cfg.lat_ok, 1e-6),
                abs(yaw) / max(cfg.yaw_ok, 1e-6),
            )
            misalign = _clip(misalign, 0.0, 3.0)
            speed_scale = 1.0 / (1.0 + 0.7 * misalign)
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

        info = {
            "stage": stage,
            "dist_front": dist,
            "dist_to_center": s["dist_to_center"],
            "d_x": s["d_x"],
            "d_y": s["d_y"],
            "lat": lat,
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
        }
        return action, info
