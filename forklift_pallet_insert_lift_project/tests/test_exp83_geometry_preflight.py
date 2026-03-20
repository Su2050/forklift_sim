"""Exp8.3 预检单测 U0/U1（纯 torch，无需 Isaac）。

- U0：参考轨迹离散化后弧长参数非递减、端点与显式输入一致（与 env 中 Hermite+直线段结构一致）。
- U1：insert_norm / success 深度与托盘 yaw 旋转下的投影一致。
"""

from __future__ import annotations

import math

import torch


# --- 与 env 中单测计划一致的容差 ---
EPS_POS = 1e-3
EPS_YAW = math.radians(0.5)
EPS_INS = 1e-3


def _build_traj_pts_batch(
    p0: torch.Tensor,
    yaw0: torch.Tensor,
    pallet_pos: torch.Tensor,
    pallet_yaw: torch.Tensor,
    *,
    pallet_depth_m: float,
    traj_pre_dist_m: float,
    num_samples: int,
    device: torch.device,
) -> torch.Tensor:
    """与 `ForkliftPalletInsertLiftEnv._build_reference_trajectory` 相同的点列生成（2D）。"""
    u_in = torch.stack([torch.cos(pallet_yaw), torch.sin(pallet_yaw)], dim=-1)
    t0 = torch.stack([torch.cos(yaw0), torch.sin(yaw0)], dim=-1)
    s_front = -0.5 * pallet_depth_m
    p_goal = pallet_pos + s_front * u_in
    p_pre = pallet_pos + (s_front - traj_pre_dist_m) * u_in

    dist = torch.norm(p_pre - p0, dim=-1, keepdim=True)
    L = dist * 1.5
    m0 = t0 * L
    m1 = u_in * L

    num_curve = int(num_samples * 0.7)
    num_line = num_samples - num_curve

    t = torch.linspace(0.0, 1.0, num_curve, device=device).view(1, -1, 1)
    t2 = t**2
    t3 = t**3
    h00 = 2 * t3 - 3 * t2 + 1
    h10 = t3 - 2 * t2 + t
    h01 = -2 * t3 + 3 * t2
    h11 = t3 - t2

    p0_exp = p0.unsqueeze(1)
    m0_exp = m0.unsqueeze(1)
    p1_exp = p_pre.unsqueeze(1)
    m1_exp = m1.unsqueeze(1)
    pts_curve = h00 * p0_exp + h10 * m0_exp + h01 * p1_exp + h11 * m1_exp

    t_line = torch.linspace(0.0, 1.0, num_line, device=device).view(1, -1, 1)
    pts_line = (1 - t_line) * p_pre.unsqueeze(1) + t_line * p_goal.unsqueeze(1)
    return torch.cat([pts_curve, pts_line], dim=1)


def _traj_s_norm(pts: torch.Tensor) -> torch.Tensor:
    diffs = pts[:, 1:, :] - pts[:, :-1, :]
    dists = torch.norm(diffs, dim=-1)
    s_cum = torch.cat(
        [torch.zeros((pts.shape[0], 1), device=pts.device), torch.cumsum(dists, dim=-1)],
        dim=1,
    )
    s_total = s_cum[:, -1:] + 1e-6
    return s_cum / s_total


def test_u0_traj_endpoints_and_monotone_s():
    device = torch.device("cpu")
    M = 3
    p0 = torch.tensor([[0.0, 0.0], [-1.0, 0.5], [2.0, -1.0]], device=device)
    yaw0 = torch.zeros(M, device=device)
    pallet_pos = torch.tensor([[5.0, 0.0], [4.0, 1.0], [3.0, -2.0]], device=device)
    pallet_yaw = torch.zeros(M, device=device)

    pts = _build_traj_pts_batch(
        p0,
        yaw0,
        pallet_pos,
        pallet_yaw,
        pallet_depth_m=2.16,
        traj_pre_dist_m=1.2,
        num_samples=32,
        device=device,
    )
    s_norm = _traj_s_norm(pts)
    assert torch.all(s_norm[:, 1:] + 1e-9 >= s_norm[:, :-1])

    u_in = torch.stack([torch.cos(pallet_yaw), torch.sin(pallet_yaw)], dim=-1)
    s_front = -0.5 * 2.16
    p_goal = pallet_pos + s_front * u_in

    d0 = torch.norm(pts[:, 0, :] - p0, dim=-1)
    d1 = torch.norm(pts[:, -1, :] - p_goal, dim=-1)
    assert torch.all(d0 < EPS_POS)
    assert torch.all(d1 < EPS_POS)


def test_u0_query_d_traj_zero_at_start():
    """查询点在 p0 时 d_traj≈0；切线取离散边 p1-p0，与名义 yaw0 允许离散化误差（与 env 一致）。"""
    device = torch.device("cpu")
    M = 1
    p0 = torch.tensor([[0.0, 0.0]], device=device)
    yaw0 = torch.tensor([0.3], device=device)
    pallet_pos = torch.tensor([[4.0, 0.0]], device=device)
    pallet_yaw = torch.tensor([-0.2], device=device)

    pts = _build_traj_pts_batch(
        p0,
        yaw0,
        pallet_pos,
        pallet_yaw,
        pallet_depth_m=2.16,
        traj_pre_dist_m=1.2,
        num_samples=24,
        device=device,
    )
    diffs = pts[:, 1:, :] - pts[:, :-1, :]
    dists = torch.norm(diffs, dim=-1)
    tangents = diffs / (dists.unsqueeze(-1) + 1e-6)
    tangents = torch.cat([tangents, tangents[:, -1:, :]], dim=1)

    fork_xy = p0
    robot_yaw = yaw0
    dists_q = torch.norm(pts - fork_xy.unsqueeze(1), dim=-1)
    min_d, min_ix = torch.min(dists_q, dim=1)
    closest_t = tangents[0, min_ix[0]]
    traj_yaw = torch.atan2(closest_t[1], closest_t[0])
    yaw_err = torch.atan2(
        torch.sin(robot_yaw[0] - traj_yaw),
        torch.cos(robot_yaw[0] - traj_yaw),
    )
    assert min_d.item() < EPS_POS
    # 首段为 Hermite 离散化后的弦向，与 t0(yaw0) 可有数度偏差；Env 中同样用差分切线
    assert abs(yaw_err.item()) < math.radians(15.0)


def _insert_norm_tip(
    pallet_xy: torch.Tensor,
    pallet_yaw: torch.Tensor,
    tip_xy: torch.Tensor,
    *,
    pallet_depth_m: float,
) -> torch.Tensor:
    """与 env._get_rewards 中 insert_norm 一致（tip 投影）。"""
    cp = torch.cos(pallet_yaw)
    sp = torch.sin(pallet_yaw)
    u_in = torch.stack([cp, sp], dim=-1)
    s_front = -0.5 * pallet_depth_m
    rel_tip = tip_xy - pallet_xy
    s_tip = torch.sum(rel_tip * u_in, dim=-1)
    insert_depth = torch.clamp(s_tip - s_front, min=0.0)
    return torch.clamp(insert_depth / (pallet_depth_m + 1e-6), 0.0, 1.0)


def test_u1_insert_norm_at_success_center_yaw0():
    D = 2.16
    ins_f = 0.4
    s_front = -0.5 * D
    s_center = s_front + (ins_f * D - 0.6)
    pallet_xy = torch.zeros(1, 2)
    pallet_yaw = torch.zeros(1)
    # yaw=0：fork_center 在 (s_center, 0)，tip 在 (s_center+0.6, 0)
    tip_xy = torch.tensor([[s_center + 0.6, 0.0]])
    ins = _insert_norm_tip(pallet_xy, pallet_yaw, tip_xy, pallet_depth_m=D)
    assert abs(ins.item() - ins_f) < EPS_INS


def test_u1_insert_norm_rotated_pallet():
    D = 2.16
    ins_f = 0.4
    s_front = -0.5 * D
    s_center = s_front + (ins_f * D - 0.6)
    p_yaw = math.pi / 6.0
    cp, sp = math.cos(p_yaw), math.sin(p_yaw)
    u_in = torch.tensor([cp, sp])
    # fork_center on axis at s_center
    cxy = torch.tensor([s_center * cp, s_center * sp]).unsqueeze(0)
    pallet_yaw = torch.tensor([p_yaw])
    # robot aligned with pallet, tip 沿 u_in 伸出 0.6m
    tip_xy = cxy + 0.6 * u_in.unsqueeze(0)
    ins = _insert_norm_tip(torch.zeros(1, 2), pallet_yaw, tip_xy, pallet_depth_m=D)
    assert abs(ins.item() - ins_f) < EPS_INS

    # 深于 rd 目标（front+0.6）应显著大于 insert_fraction
    s_rd = s_front + 0.6
    cxy_deep = torch.tensor([s_rd * cp, s_rd * sp]).unsqueeze(0)
    tip_deep = cxy_deep + 0.6 * u_in.unsqueeze(0)
    ins_deep = _insert_norm_tip(torch.zeros(1, 2), pallet_yaw, tip_deep, pallet_depth_m=D)
    assert ins_deep.item() > ins_f + EPS_INS


if __name__ == "__main__":
    test_u0_traj_endpoints_and_monotone_s()
    test_u0_query_d_traj_zero_at_start()
    test_u1_insert_norm_at_success_center_yaw0()
    test_u1_insert_norm_rotated_pallet()
    print("exp83_geometry_preflight: OK")
