"""S2.0a: GPU-batched Bezier reference trajectory generation.

Provides two functions used by the forklift environment:
  - generate_bezier_path: compute cubic Bezier waypoints + tangent angles
  - closest_point_on_path: find nearest waypoint and heading error
"""

from __future__ import annotations

import math
import torch


def generate_bezier_path(
    start_xy: torch.Tensor,
    start_yaw: torch.Tensor,
    end_xy: torch.Tensor,
    end_yaw: torch.Tensor,
    num_points: int = 64,
    ctrl_scale: float = 0.4,
    ctrl_arm_max: float = 2.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate cubic Bezier reference paths for N environments.

    Args:
        start_xy:  (N, 2) fork tip initial positions.
        start_yaw: (N,)   robot initial headings (rad).
        end_xy:    (N, 2) pallet front-face center positions.
        end_yaw:   (N,)   pallet headings / insertion direction (rad).
        num_points: number of samples along the curve.
        ctrl_scale: fraction of straight-line distance used for control-point offset.
        ctrl_arm_max: upper clamp on control-point arm length (m) to prevent
                      self-intersecting curves when start-to-end distance is large.

    Returns:
        waypoints:     (N, num_points, 2)
        tangent_angles: (N, num_points)  heading of the curve tangent at each sample.
    """
    N = start_xy.shape[0]
    device = start_xy.device

    d = torch.norm(end_xy - start_xy, dim=-1, keepdim=True)  # (N, 1)
    arm = torch.clamp(ctrl_scale * d, max=ctrl_arm_max)       # (N, 1)

    cos_s = torch.cos(start_yaw).unsqueeze(-1)  # (N, 1)
    sin_s = torch.sin(start_yaw).unsqueeze(-1)
    cos_e = torch.cos(end_yaw).unsqueeze(-1)
    sin_e = torch.sin(end_yaw).unsqueeze(-1)

    P0 = start_xy                                                     # (N, 2)
    P1 = P0 + arm * torch.cat([cos_s, sin_s], dim=-1)                # (N, 2)
    P2 = end_xy - arm * torch.cat([cos_e, sin_e], dim=-1)            # (N, 2)
    P3 = end_xy                                                       # (N, 2)

    t = torch.linspace(0.0, 1.0, num_points, device=device)          # (num_points,)
    t = t.unsqueeze(0)                                                 # (1, num_points)
    u = 1.0 - t                                                        # (1, num_points)

    # B(t) = u^3 P0 + 3 u^2 t P1 + 3 u t^2 P2 + t^3 P3
    w0 = (u ** 3).unsqueeze(-1)            # (1, num_points, 1)
    w1 = (3.0 * u ** 2 * t).unsqueeze(-1)
    w2 = (3.0 * u * t ** 2).unsqueeze(-1)
    w3 = (t ** 3).unsqueeze(-1)

    P0e = P0.unsqueeze(1)  # (N, 1, 2)
    P1e = P1.unsqueeze(1)
    P2e = P2.unsqueeze(1)
    P3e = P3.unsqueeze(1)

    waypoints = w0 * P0e + w1 * P1e + w2 * P2e + w3 * P3e           # (N, num_points, 2)

    # B'(t) = 3 u^2 (P1-P0) + 6 u t (P2-P1) + 3 t^2 (P3-P2)
    d01 = (P1 - P0).unsqueeze(1)  # (N, 1, 2)
    d12 = (P2 - P1).unsqueeze(1)
    d23 = (P3 - P2).unsqueeze(1)

    tw0 = (3.0 * u ** 2).unsqueeze(-1)
    tw1 = (6.0 * u * t).unsqueeze(-1)
    tw2 = (3.0 * t ** 2).unsqueeze(-1)

    tangent_vec = tw0 * d01 + tw1 * d12 + tw2 * d23                  # (N, num_points, 2)
    tangent_angles = torch.atan2(tangent_vec[..., 1], tangent_vec[..., 0])  # (N, num_points)

    return waypoints, tangent_angles


def closest_point_on_path(
    query_xy: torch.Tensor,
    waypoints: torch.Tensor,
    tangent_angles: torch.Tensor,
    query_yaw: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Find the closest waypoint on the reference path for each environment.

    Args:
        query_xy:       (N, 2) current fork-tip positions.
        waypoints:      (N, num_points, 2) pre-computed path waypoints.
        tangent_angles:  (N, num_points) tangent heading at each waypoint.
        query_yaw:      (N,) current robot heading.

    Returns:
        r_cd:   (N,) Euclidean distance to the nearest waypoint.
        r_cpsi: (N,) absolute heading difference w.r.t. the tangent at the
                nearest waypoint, in [0, pi].
    """
    diff = query_xy.unsqueeze(1) - waypoints                          # (N, num_points, 2)
    dists = torch.norm(diff, dim=-1)                                   # (N, num_points)
    idx = torch.argmin(dists, dim=1)                                   # (N,)

    r_cd = dists.gather(1, idx.unsqueeze(1)).squeeze(1)               # (N,)

    nearest_tan = tangent_angles.gather(1, idx.unsqueeze(1)).squeeze(1)  # (N,)
    angle_diff = query_yaw - nearest_tan
    angle_diff = torch.atan2(torch.sin(angle_diff), torch.cos(angle_diff))
    r_cpsi = torch.abs(angle_diff)                                     # (N,)

    return r_cd, r_cpsi
