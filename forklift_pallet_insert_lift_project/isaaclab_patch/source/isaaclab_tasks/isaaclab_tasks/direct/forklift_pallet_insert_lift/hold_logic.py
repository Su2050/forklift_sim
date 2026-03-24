from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class HoldLogicConfig:
    insert_thresh: float
    max_lateral_err_m: float
    max_yaw_err_deg: float
    hysteresis_ratio: float
    insert_exit_epsilon: float
    lift_delta_m: float
    lift_exit_epsilon: float
    hold_counter_decay: float
    tip_align_entry_m: float
    tip_align_exit_m: float
    tip_align_near_dist: float
    require_lift: bool


@dataclass(frozen=True)
class HoldLogicState:
    align_entry: torch.Tensor
    align_exit_exceeded: torch.Tensor
    insert_entry: torch.Tensor
    insert_exit_exceeded: torch.Tensor
    tip_gate_active: torch.Tensor
    tip_entry: torch.Tensor
    tip_exit_exceeded: torch.Tensor
    lift_entry: torch.Tensor
    lift_exit_exceeded: torch.Tensor
    hold_entry: torch.Tensor
    any_exit_exceeded: torch.Tensor
    grace_zone: torch.Tensor
    hold_counter_next: torch.Tensor


def compute_hold_logic(
    *,
    center_y_err: torch.Tensor,
    yaw_err_deg: torch.Tensor,
    insert_depth: torch.Tensor,
    lift_height: torch.Tensor,
    tip_y_err: torch.Tensor,
    dist_front: torch.Tensor,
    hold_counter: torch.Tensor,
    cfg: HoldLogicConfig,
) -> HoldLogicState:
    """Compute cfg-driven hold and success gates with hysteresis and decay."""

    exit_y = cfg.max_lateral_err_m * cfg.hysteresis_ratio
    exit_yaw = cfg.max_yaw_err_deg * cfg.hysteresis_ratio
    insert_exit_thresh = max(0.0, cfg.insert_thresh - cfg.insert_exit_epsilon)
    lift_exit_thresh = max(0.0, cfg.lift_delta_m - cfg.lift_exit_epsilon)

    align_entry = (center_y_err <= cfg.max_lateral_err_m) & (yaw_err_deg <= cfg.max_yaw_err_deg)
    align_exit_exceeded = (center_y_err > exit_y) | (yaw_err_deg > exit_yaw)

    insert_entry = insert_depth >= cfg.insert_thresh
    insert_exit_exceeded = insert_depth < insert_exit_thresh

    tip_gate_active = dist_front <= cfg.tip_align_near_dist
    tip_entry = (~tip_gate_active) | (tip_y_err <= cfg.tip_align_entry_m)
    tip_exit_exceeded = tip_gate_active & (tip_y_err > cfg.tip_align_exit_m)

    if cfg.require_lift:
        lift_entry = lift_height >= cfg.lift_delta_m
        lift_exit_exceeded = lift_height < lift_exit_thresh
    else:
        lift_entry = torch.ones_like(align_entry, dtype=torch.bool)
        lift_exit_exceeded = torch.zeros_like(align_entry, dtype=torch.bool)

    hold_entry = insert_entry & align_entry & tip_entry & lift_entry
    any_exit_exceeded = (
        align_exit_exceeded
        | insert_exit_exceeded
        | tip_exit_exceeded
        | lift_exit_exceeded
    )
    grace_zone = (~hold_entry) & (~any_exit_exceeded)

    decayed_counter = hold_counter * cfg.hold_counter_decay
    hold_counter_next = torch.where(
        hold_entry,
        hold_counter + 1.0,
        torch.where(grace_zone, hold_counter, decayed_counter),
    )

    return HoldLogicState(
        align_entry=align_entry,
        align_exit_exceeded=align_exit_exceeded,
        insert_entry=insert_entry,
        insert_exit_exceeded=insert_exit_exceeded,
        tip_gate_active=tip_gate_active,
        tip_entry=tip_entry,
        tip_exit_exceeded=tip_exit_exceeded,
        lift_entry=lift_entry,
        lift_exit_exceeded=lift_exit_exceeded,
        hold_entry=hold_entry,
        any_exit_exceeded=any_exit_exceeded,
        grace_zone=grace_zone,
        hold_counter_next=hold_counter_next,
    )
