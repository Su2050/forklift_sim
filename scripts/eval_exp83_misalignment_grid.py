"""Evaluate one checkpoint on a fixed near-field misalignment grid.

Each grid point fixes stage1 reset to a single (x, y, yaw) tuple, then runs a
deterministic episode. Optional --force_zero_steer compares whether success is
coming from actual steering or simply driving straight in a nearly aligned reset.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import signal
import sys
import time
import traceback
from pathlib import Path

import numpy as np

parser = argparse.ArgumentParser(description="Misalignment-grid eval for exp8.3 checkpoints")
parser.add_argument("--task", type=str, default="Isaac-Forklift-PalletInsertLift-Direct-v0")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--label", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--seed", type=int, default=20260327)
parser.add_argument("--x_root", type=float, default=-3.40)
parser.add_argument("--y_values", type=str, default="-0.15,-0.10,-0.05,0.0,0.05,0.10,0.15")
parser.add_argument("--yaw_deg_values", type=str, default="-6,-4,-2,0,2,4,6")
parser.add_argument("--episodes_per_point", type=int, default=1)
parser.add_argument("--force_zero_steer", action="store_true")
parser.add_argument("--steer_scale", type=float, default=1.0)
parser.add_argument(
    "--output_dir",
    type=str,
    default="/home/uniubi/projects/forklift_sim/outputs/exp83_misalignment_grid_eval",
)

from isaaclab.app import AppLauncher

AppLauncher.add_app_launcher_args(parser)
args, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

from rsl_rl.runners import OnPolicyRunner

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

from isaaclab_tasks.direct.forklift_pallet_insert_lift.hold_logic import compute_hold_logic


TERMINATION_REQUESTED = False
TERMINATION_REASON = ""


def _request_termination(signum, _frame):
    global TERMINATION_REQUESTED, TERMINATION_REASON
    if TERMINATION_REQUESTED:
        return
    TERMINATION_REQUESTED = True
    TERMINATION_REASON = signal.Signals(signum).name
    print(
        f"[WARN] Received {TERMINATION_REASON}; stopping after the current step and writing partial outputs.",
        flush=True,
    )


signal.signal(signal.SIGTERM, _request_termination)
signal.signal(signal.SIGINT, _request_termination)


def _quat_to_yaw(q: torch.Tensor) -> torch.Tensor:
    w, x, y, z = q.unbind(-1)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return torch.atan2(siny_cosp, cosy_cosp)


def _parse_list(spec: str) -> list[float]:
    return [float(v.strip()) for v in spec.split(",") if v.strip()]


def _safe_mean(rows: list[dict[str, float | int | str]], key: str) -> float:
    if not rows:
        return 0.0
    first = rows[0]
    if key not in first:
        return 0.0
    return sum(float(row[key]) for row in rows) / len(rows)


def _mode_tag(force_zero_steer: bool, steer_scale: float) -> str:
    effective_scale = 0.0 if force_zero_steer else steer_scale
    if abs(effective_scale) < 1e-8:
        return "zero_steer"
    if abs(effective_scale - 1.0) < 1e-8:
        return "normal"
    if abs(effective_scale - 0.5) < 1e-8:
        return "half_steer"
    if abs(effective_scale + 1.0) < 1e-8:
        return "flip_steer"
    return f"steer_scale_{effective_scale:+.2f}".replace("+", "p").replace("-", "m").replace(".", "p")


def _write_rows(rows_path: Path, rows: list[dict[str, float | int | str]]):
    if not rows:
        rows_path.write_text("")
        return
    fieldnames = list(rows[0].keys())
    with rows_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _build_summary(
    rows: list[dict[str, float | int | str]],
    *,
    label: str,
    checkpoint: str,
    mode_tag: str,
    total_grid_points: int,
    total_point_repeats: int,
    completed_point_repeats: int,
    x_root: float,
    y_values: list[float],
    yaw_values: list[float],
    start_time_s: float,
    complete: bool,
    termination_requested: bool,
    termination_reason: str,
    error: str | None,
) -> dict[str, float | int | str | bool | list[float] | None]:
    return {
        "label": label,
        "checkpoint": str(Path(checkpoint).resolve()),
        "mode": mode_tag,
        "steer_scale": 0.0 if args.force_zero_steer else args.steer_scale,
        "num_envs": args.num_envs,
        "episodes_per_point": args.episodes_per_point,
        "total_grid_points": total_grid_points,
        "total_point_repeats": total_point_repeats,
        "completed_point_repeats": completed_point_repeats,
        "planned_episodes": total_point_repeats * args.num_envs,
        "total_episodes": len(rows),
        "x_root": x_root,
        "y_values": y_values,
        "yaw_deg_values": yaw_values,
        "success_rate_ep": _safe_mean(rows, "success"),
        "ever_inserted_rate": _safe_mean(rows, "ever_inserted"),
        "ever_inserted_push_free_rate": _safe_mean(rows, "ever_inserted_push_free"),
        "ever_hold_entry_rate": _safe_mean(rows, "ever_hold_entry"),
        "ever_clean_insert_ready_rate": _safe_mean(rows, "ever_clean_insert_ready"),
        "ever_dirty_insert_rate": _safe_mean(rows, "ever_dirty_insert"),
        "timeout_frac": _safe_mean(rows, "timeout"),
        "mean_episode_length": _safe_mean(rows, "episode_length"),
        "mean_max_pallet_disp_xy": _safe_mean(rows, "max_pallet_disp_xy"),
        "mean_max_hold_counter": _safe_mean(rows, "max_hold_counter"),
        "mean_min_dist_front": _safe_mean(rows, "min_dist_front"),
        "mean_abs_steer_raw": _safe_mean(rows, "mean_abs_steer_raw"),
        "mean_max_abs_steer_raw": _safe_mean(rows, "max_abs_steer_raw"),
        "mean_steer_raw": _safe_mean(rows, "mean_steer_raw"),
        "mean_steer_applied": _safe_mean(rows, "mean_steer_applied"),
        "mean_abs_steer_applied": _safe_mean(rows, "mean_abs_steer_applied"),
        "mean_max_abs_steer_applied": _safe_mean(rows, "max_abs_steer_applied"),
        "complete": complete,
        "termination_requested": termination_requested,
        "termination_reason": termination_reason or "",
        "error": error,
        "elapsed_sec": time.time() - start_time_s,
    }


def _reseed_fixed_stage1_reset(raw_env, seed_val: int, x_root: float, y_m: float, yaw_deg: float):
    torch.manual_seed(seed_val)
    np.random.seed(seed_val)
    raw_env.cfg.stage1_init_x_min_m = x_root
    raw_env.cfg.stage1_init_x_max_m = x_root
    raw_env.cfg.stage1_init_y_min_m = y_m
    raw_env.cfg.stage1_init_y_max_m = y_m
    raw_env.cfg.stage1_init_yaw_deg_min = yaw_deg
    raw_env.cfg.stage1_init_yaw_deg_max = yaw_deg
    with torch.inference_mode():
        all_ids = torch.arange(raw_env.num_envs, device=raw_env.device)
        raw_env._reset_idx(all_ids)


def _compute_step_state(raw_env):
    pallet_pos = raw_env.pallet.data.root_pos_w
    tip = raw_env._compute_fork_tip()
    fork_center = raw_env._compute_fork_center()

    robot_yaw = _quat_to_yaw(raw_env.robot.data.root_quat_w)
    pallet_yaw = _quat_to_yaw(raw_env.pallet.data.root_quat_w)
    yaw_err = torch.atan2(torch.sin(robot_yaw - pallet_yaw), torch.cos(robot_yaw - pallet_yaw))
    yaw_err_deg = torch.abs(yaw_err) * (180.0 / math.pi)

    cp = torch.cos(pallet_yaw)
    sp = torch.sin(pallet_yaw)
    u_in = torch.stack([cp, sp], dim=-1)
    v_lat = torch.stack([-sp, cp], dim=-1)

    rel_tip = tip[:, :2] - pallet_pos[:, :2]
    s_tip = torch.sum(rel_tip * u_in, dim=-1)
    s_front = -0.5 * raw_env.cfg.pallet_depth_m
    dist_front = torch.clamp(s_front - s_tip, min=0.0)
    insert_depth = torch.clamp(s_tip - s_front, min=0.0)

    rel_fc = fork_center[:, :2] - pallet_pos[:, :2]
    center_y_err = torch.abs(torch.sum(rel_fc * v_lat, dim=-1))
    tip_y_err = torch.abs(torch.sum(rel_tip * v_lat, dim=-1))
    lift_height = tip[:, 2] - raw_env._fork_tip_z0

    hold_state = compute_hold_logic(
        center_y_err=center_y_err,
        yaw_err_deg=yaw_err_deg,
        insert_depth=insert_depth,
        lift_height=lift_height,
        tip_y_err=tip_y_err,
        dist_front=dist_front,
        hold_counter=raw_env._hold_counter,
        cfg=raw_env._hold_logic_cfg,
    )

    pallet_init_pos_xy = torch.tensor(raw_env.cfg.pallet_cfg.init_state.pos[:2], device=raw_env.device)
    pallet_disp_xy = torch.norm(pallet_pos[:, :2] - pallet_init_pos_xy, dim=-1)
    push_free = pallet_disp_xy < raw_env.cfg.push_free_disp_thresh_m

    inserted = hold_state.insert_entry
    hold_entry = hold_state.hold_entry
    clean_insert_ready = hold_entry & push_free
    dirty_insert = inserted & ~push_free
    success = raw_env._hold_counter >= raw_env._hold_steps

    return {
        "inserted": inserted,
        "push_free": push_free,
        "inserted_push_free": inserted & push_free,
        "hold_entry": hold_entry,
        "clean_insert_ready": clean_insert_ready,
        "dirty_insert": dirty_insert,
        "success": success,
        "pallet_disp_xy": pallet_disp_xy,
        "dist_front": dist_front,
        "yaw_err_deg": yaw_err_deg,
        "center_y_err": center_y_err,
    }


@hydra_task_config(args.task, args.agent)
def main(env_cfg, agent_cfg):
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.seed = args.seed
    env_cfg.sim.device = "cuda:0"
    print(
        "[INFO] Effective env overrides before gym.make: "
        f"use_camera={getattr(env_cfg, 'use_camera', None)} "
        f"use_asymmetric_critic={getattr(env_cfg, 'use_asymmetric_critic', None)} "
        f"stage_1_mode={getattr(env_cfg, 'stage_1_mode', None)} "
        f"camera=({getattr(env_cfg, 'camera_width', None)}x{getattr(env_cfg, 'camera_height', None)})",
        flush=True,
    )

    env = gym.make(args.task, cfg=env_cfg)
    env_wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    raw_env = env.unwrapped

    runner = OnPolicyRunner(env_wrapped, agent_cfg.to_dict(), log_dir=None, device="cuda:0")
    runner.load(args.checkpoint)
    try:
        policy_nn = runner.alg.policy
    except AttributeError:
        policy_nn = runner.alg.actor_critic

    y_values = _parse_list(args.y_values)
    yaw_values = _parse_list(args.yaw_deg_values)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    mode_tag = _mode_tag(args.force_zero_steer, args.steer_scale)
    rows_path = output_dir / f"{args.label}_{mode_tag}_rows.csv"
    summary_path = output_dir / f"{args.label}_{mode_tag}_summary.json"
    partial_rows_path = output_dir / f"{args.label}_{mode_tag}_partial_rows.csv"
    partial_summary_path = output_dir / f"{args.label}_{mode_tag}_partial_summary.json"

    rows: list[dict[str, float | int | str]] = []
    max_steps = int(raw_env.max_episode_length) + 5
    total_grid_points = len(y_values) * len(yaw_values)
    total_point_repeats = total_grid_points * args.episodes_per_point
    completed_point_repeats = 0
    start_time_s = time.time()
    error_msg: str | None = None

    point_specs: list[tuple[int, float, float, int, int]] = []
    point_idx = 0
    for yaw_deg in yaw_values:
        for y_m in y_values:
            for rep in range(args.episodes_per_point):
                point_idx += 1
                seed_val = args.seed + rep
                point_specs.append((point_idx, yaw_deg, y_m, rep, seed_val))

    try:
        for point_idx, yaw_deg, y_m, rep, seed_val in point_specs:
            if TERMINATION_REQUESTED:
                print(
                    f"[WARN] Stop requested before point {point_idx}/{total_point_repeats}; writing partial outputs.",
                    flush=True,
                )
                break
            point_start_s = time.time()
            print(
                (
                    f"[PROGRESS] point {point_idx}/{total_point_repeats} start "
                    f"mode={mode_tag} y={y_m:+.3f} yaw_deg={yaw_deg:+.1f} rep={rep} seed={seed_val}"
                ),
                flush=True,
            )
            try:
                _reseed_fixed_stage1_reset(raw_env, seed_val, args.x_root, y_m, yaw_deg)
                obs = env_wrapped.get_observations()
                done_mask = torch.zeros(args.num_envs, dtype=torch.bool, device=raw_env.device)
                ep_inserted = torch.zeros(args.num_envs, dtype=torch.bool, device=raw_env.device)
                ep_inserted_push_free = torch.zeros(args.num_envs, dtype=torch.bool, device=raw_env.device)
                ep_hold_entry = torch.zeros(args.num_envs, dtype=torch.bool, device=raw_env.device)
                ep_clean_insert_ready = torch.zeros(args.num_envs, dtype=torch.bool, device=raw_env.device)
                ep_dirty_insert = torch.zeros(args.num_envs, dtype=torch.bool, device=raw_env.device)
                ep_success = torch.zeros(args.num_envs, dtype=torch.bool, device=raw_env.device)
                ep_max_pallet_disp = torch.zeros(args.num_envs, dtype=torch.float32, device=raw_env.device)
                ep_max_hold_counter = torch.zeros(args.num_envs, dtype=torch.float32, device=raw_env.device)
                ep_min_dist_front = torch.full((args.num_envs,), float("inf"), dtype=torch.float32, device=raw_env.device)
                ep_max_abs_steer_raw = torch.zeros(args.num_envs, dtype=torch.float32, device=raw_env.device)
                ep_mean_abs_steer_raw = torch.zeros(args.num_envs, dtype=torch.float32, device=raw_env.device)
                ep_mean_steer_raw = torch.zeros(args.num_envs, dtype=torch.float32, device=raw_env.device)
                ep_max_abs_steer_applied = torch.zeros(args.num_envs, dtype=torch.float32, device=raw_env.device)
                ep_mean_abs_steer_applied = torch.zeros(args.num_envs, dtype=torch.float32, device=raw_env.device)
                ep_mean_steer_applied = torch.zeros(args.num_envs, dtype=torch.float32, device=raw_env.device)
                ep_mean_abs_drive_raw = torch.zeros(args.num_envs, dtype=torch.float32, device=raw_env.device)
                ep_len = torch.zeros(args.num_envs, dtype=torch.int32, device=raw_env.device)
                ep_done_len = torch.full((args.num_envs,), -1, dtype=torch.int32, device=raw_env.device)

                step_count = 0
                while not done_mask.all() and step_count < max_steps:
                    if TERMINATION_REQUESTED:
                        break
                    with torch.inference_mode():
                        raw_actions = policy_nn.act_inference(obs)
                        applied_actions = raw_actions.clone()
                        if args.force_zero_steer:
                            applied_actions[:, 1] = 0.0
                        else:
                            applied_actions[:, 1] = raw_actions[:, 1] * args.steer_scale
                        obs, _, dones, _ = env_wrapped.step(applied_actions)

                    step_count += 1
                    state = _compute_step_state(raw_env)
                    active = ~done_mask
                    if int(active.sum().item()) > 0:
                        raw_steer = torch.abs(raw_actions[:, 1])
                        signed_raw_steer = raw_actions[:, 1]
                        applied_steer = torch.abs(applied_actions[:, 1])
                        signed_applied_steer = applied_actions[:, 1]
                        raw_drive = torch.abs(raw_actions[:, 0])

                        ep_inserted[active] |= state["inserted"][active]
                        ep_inserted_push_free[active] |= state["inserted_push_free"][active]
                        ep_hold_entry[active] |= state["hold_entry"][active]
                        ep_clean_insert_ready[active] |= state["clean_insert_ready"][active]
                        ep_dirty_insert[active] |= state["dirty_insert"][active]
                        ep_success[active] |= state["success"][active]
                        ep_max_pallet_disp[active] = torch.maximum(ep_max_pallet_disp[active], state["pallet_disp_xy"][active])
                        ep_max_hold_counter[active] = torch.maximum(ep_max_hold_counter[active], raw_env._hold_counter[active])
                        ep_min_dist_front[active] = torch.minimum(ep_min_dist_front[active], state["dist_front"][active])
                        ep_max_abs_steer_raw[active] = torch.maximum(ep_max_abs_steer_raw[active], raw_steer[active])
                        ep_mean_abs_steer_raw[active] += raw_steer[active]
                        ep_mean_steer_raw[active] += signed_raw_steer[active]
                        ep_max_abs_steer_applied[active] = torch.maximum(ep_max_abs_steer_applied[active], applied_steer[active])
                        ep_mean_abs_steer_applied[active] += applied_steer[active]
                        ep_mean_steer_applied[active] += signed_applied_steer[active]
                        ep_mean_abs_drive_raw[active] += raw_drive[active]
                        ep_len[active] += 1

                    if isinstance(dones, torch.Tensor):
                        newly_done = dones.bool() & ~done_mask
                    else:
                        newly_done = torch.tensor(dones, dtype=torch.bool, device=raw_env.device) & ~done_mask
                    ep_done_len[newly_done] = ep_len[newly_done]
                    done_mask |= newly_done

                point_success_count = 0
                point_inserted_count = 0
                point_clean_count = 0
                point_hold_count = 0
                point_timeout_count = 0

                for env_id in range(args.num_envs):
                    ep_length = int(ep_done_len[env_id].item())
                    if ep_length < 0:
                        ep_length = int(ep_len[env_id].item())
                    denom = max(ep_length, 1)
                    timeout = int(ep_length >= int(raw_env.max_episode_length) - 1 and not bool(ep_success[env_id].item()))
                    init_pos = raw_env.robot.data.root_pos_w[env_id]
                    init_yaw = math.degrees(float(_quat_to_yaw(raw_env.robot.data.root_quat_w[env_id:env_id + 1])[0].item()))
                    row = {
                        "label": args.label,
                        "mode": mode_tag,
                        "rep": rep,
                        "seed": seed_val,
                        "grid_point_index": point_idx,
                        "grid_point_total": total_point_repeats,
                        "grid_x_root": args.x_root,
                        "grid_y_m": y_m,
                        "grid_yaw_deg": yaw_deg,
                        "init_x_root_actual": float(init_pos[0].item()),
                        "init_y_root_actual": float(init_pos[1].item()),
                        "init_yaw_deg_actual": init_yaw,
                        "success": int(ep_success[env_id].item()),
                        "ever_inserted": int(ep_inserted[env_id].item()),
                        "ever_inserted_push_free": int(ep_inserted_push_free[env_id].item()),
                        "ever_hold_entry": int(ep_hold_entry[env_id].item()),
                        "ever_clean_insert_ready": int(ep_clean_insert_ready[env_id].item()),
                        "ever_dirty_insert": int(ep_dirty_insert[env_id].item()),
                        "timeout": timeout,
                        "episode_length": ep_length,
                        "max_pallet_disp_xy": float(ep_max_pallet_disp[env_id].item()),
                        "max_hold_counter": float(ep_max_hold_counter[env_id].item()),
                        "min_dist_front": float(ep_min_dist_front[env_id].item()),
                        "mean_abs_drive_raw": float((ep_mean_abs_drive_raw[env_id] / denom).item()),
                        "mean_abs_steer_raw": float((ep_mean_abs_steer_raw[env_id] / denom).item()),
                        "mean_steer_raw": float((ep_mean_steer_raw[env_id] / denom).item()),
                        "max_abs_steer_raw": float(ep_max_abs_steer_raw[env_id].item()),
                        "mean_steer_applied": float((ep_mean_steer_applied[env_id] / denom).item()),
                        "mean_abs_steer_applied": float((ep_mean_abs_steer_applied[env_id] / denom).item()),
                        "max_abs_steer_applied": float(ep_max_abs_steer_applied[env_id].item()),
                        "steer_scale": 0.0 if args.force_zero_steer else args.steer_scale,
                        "terminated_early": int(TERMINATION_REQUESTED),
                    }
                    rows.append(row)
                    point_success_count += int(row["success"])
                    point_inserted_count += int(row["ever_inserted"])
                    point_clean_count += int(row["ever_clean_insert_ready"])
                    point_hold_count += int(row["ever_hold_entry"])
                    point_timeout_count += int(row["timeout"])

                completed_point_repeats += 1
                print(
                    (
                        f"[PROGRESS] point {point_idx}/{total_point_repeats} done "
                        f"steps={step_count} success={point_success_count}/{args.num_envs} "
                        f"inserted={point_inserted_count}/{args.num_envs} "
                        f"clean={point_clean_count}/{args.num_envs} "
                        f"hold={point_hold_count}/{args.num_envs} "
                        f"timeout={point_timeout_count}/{args.num_envs} "
                        f"elapsed_s={time.time() - point_start_s:.1f}"
                    ),
                    flush=True,
                )
            except Exception:
                print(
                    f"[ERROR] point {point_idx}/{total_point_repeats} crashed; writing partial outputs.",
                    flush=True,
                )
                raise
            if TERMINATION_REQUESTED:
                print(
                    f"[WARN] Stop requested after point {point_idx}/{total_point_repeats}; writing partial outputs.",
                    flush=True,
                )
                break
    except Exception as exc:
        error_msg = f"{type(exc).__name__}: {exc}"
        print("[ERROR] Eval aborted with exception:", flush=True)
        print(traceback.format_exc(), flush=True)
    finally:
        complete = (error_msg is None) and (not TERMINATION_REQUESTED) and (completed_point_repeats == total_point_repeats)
        out_rows_path = rows_path if complete else partial_rows_path
        out_summary_path = summary_path if complete else partial_summary_path
        _write_rows(out_rows_path, rows)
        summary = _build_summary(
            rows,
            label=args.label,
            checkpoint=args.checkpoint,
            mode_tag=mode_tag,
            total_grid_points=total_grid_points,
            total_point_repeats=total_point_repeats,
            completed_point_repeats=completed_point_repeats,
            x_root=args.x_root,
            y_values=y_values,
            yaw_values=yaw_values,
            start_time_s=start_time_s,
            complete=complete,
            termination_requested=TERMINATION_REQUESTED,
            termination_reason=TERMINATION_REASON,
            error=error_msg,
        )
        with out_summary_path.open("w") as f:
            json.dump(summary, f, indent=2)
        if complete:
            partial_rows_path.unlink(missing_ok=True)
            partial_summary_path.unlink(missing_ok=True)
        print(json.dumps(summary, indent=2), flush=True)
        print(f"[INFO] Wrote rows to: {out_rows_path}", flush=True)
        print(f"[INFO] Wrote summary to: {out_summary_path}", flush=True)
        env.close()
        if error_msg is not None:
            raise RuntimeError(error_msg)


if __name__ == "__main__":
    main()
    simulation_app.close()
