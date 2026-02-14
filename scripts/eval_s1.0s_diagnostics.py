"""S1.0S 离线 per-episode 诊断脚本（基于 eval_s1.0q_diagnostics.py 扩展）。

新增功能:
  - 初始位姿捕获 (init_x, init_y, init_yaw)
  - 难度分层分类 (Easy / Medium / Hard / Extreme)
  - 多 seed 批量评估 (--seeds 1 2 3 ... 或 --seed_start/--seed_end)
  - 区域成功率统计 + Bootstrap 95% CI
  - 分层 JSON 摘要

输出:
  - 控制台报告（漏斗 / 分布 / 死区分析 / 长尾分析 / 区域分层）
  - CSV 文件 per-episode 统计（含 init_x, init_y, init_yaw, difficulty_tier）
  - JSON 摘要（含区域分层统计 + lift height 分布）

Usage:
    cd IsaacLab && CONDA_PREFIX="" TERM=xterm bash isaaclab.sh -p \
        ../scripts/eval_s1.0s_diagnostics.py \
        --task Isaac-Forklift-PalletInsertLift-Direct-v0 \
        --headless --num_envs 1024 \
        --checkpoint logs/rsl_rl/forklift_pallet_insert_lift/2026-02-13_18-40-18/model_3595.pt \
        --experiment_name phase0_baseline \
        --max_steps 2000 \
        --seeds 1 2 3 4 5 6 7 8 9 10

    # 或使用 seed 范围:
    cd IsaacLab && CONDA_PREFIX="" TERM=xterm bash isaaclab.sh -p \
        ../scripts/eval_s1.0s_diagnostics.py \
        --task Isaac-Forklift-PalletInsertLift-Direct-v0 \
        --headless --num_envs 1024 \
        --checkpoint logs/rsl_rl/forklift_pallet_insert_lift/2026-02-13_18-40-18/model_3595.pt \
        --experiment_name phase0_baseline \
        --max_steps 2000 \
        --seed_start 1 --seed_end 10
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time as _time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="S1.0S per-episode diagnostics (multi-seed)")
AppLauncher.add_app_launcher_args(parser)
parser.add_argument("--task", type=str, default="Isaac-Forklift-PalletInsertLift-Direct-v0")
parser.add_argument("--num_envs", type=int, default=1024)
# seed 参数: 优先使用 --seeds，其次使用 --seed_start/--seed_end，最后默认 42
parser.add_argument("--seed", type=int, default=42, help="Single seed (overridden by --seeds)")
parser.add_argument("--seeds", type=int, nargs="+", default=None,
                    help="Multiple seeds for batch evaluation (e.g. --seeds 1 2 3 4 5)")
parser.add_argument("--seed_start", type=int, default=None,
                    help="Start of seed range (inclusive)")
parser.add_argument("--seed_end", type=int, default=None,
                    help="End of seed range (inclusive)")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--max_steps", type=int, default=2000,
                    help="Max env.step() calls per seed (1024 envs × ~340 steps/ep → ~6000 episodes)")
parser.add_argument("--checkpoint", type=str, required=True,
                    help="Path to model checkpoint (relative to IsaacLab dir or absolute)")
parser.add_argument("--experiment_name", type=str, default="unknown",
                    help="Experiment label for output")
parser.add_argument("--output_dir", type=str, default="../data/s1.0s_eval",
                    help="Base output directory")
args = parser.parse_args()

# Resolve seed list
if args.seeds is not None:
    SEED_LIST = args.seeds
elif args.seed_start is not None and args.seed_end is not None:
    SEED_LIST = list(range(args.seed_start, args.seed_end + 1))
else:
    SEED_LIST = [args.seed]

# Clear sys.argv for Hydra
sys.argv = [sys.argv[0]]

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# ---- IsaacLab imports (after AppLauncher) ----
import torch
import numpy as np
import gymnasium as gym

from rsl_rl.runners import OnPolicyRunner
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config


# ---- 几何计算工具 ----
def _quat_to_yaw(q: torch.Tensor) -> torch.Tensor:
    w, x, y, z = q.unbind(-1)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return torch.atan2(siny_cosp, cosy_cosp)


def _quat_to_pitch(q: torch.Tensor) -> torch.Tensor:
    """从四元数提取 pitch 角 (rad)，与 env.py 倾翻检测同源。"""
    w, x, y, z = q.unbind(-1)
    sinp = 2.0 * (w * y - z * x)
    return torch.asin(torch.clamp(sinp, -1.0, 1.0))


def compute_geometry(raw_env):
    """从 raw_env 直接计算 per-env 几何指标，与 _get_rewards() 同源。"""
    root_pos = raw_env.robot.data.root_pos_w
    pallet_pos = raw_env.pallet.data.root_pos_w
    robot_quat = raw_env.robot.data.root_quat_w
    robot_yaw = _quat_to_yaw(robot_quat)
    pallet_yaw = _quat_to_yaw(raw_env.pallet.data.root_quat_w)

    cp = torch.cos(pallet_yaw)
    sp = torch.sin(pallet_yaw)
    u_in = torch.stack([cp, sp], dim=-1)
    v_lat = torch.stack([-sp, cp], dim=-1)

    rel_robot = root_pos[:, :2] - pallet_pos[:, :2]
    y_signed = torch.sum(rel_robot * v_lat, dim=-1)
    y_err = torch.abs(y_signed)

    yaw_err = torch.atan2(
        torch.sin(robot_yaw - pallet_yaw),
        torch.cos(robot_yaw - pallet_yaw),
    )
    yaw_err_deg = torch.abs(yaw_err) * (180.0 / math.pi)

    # insert_norm
    tip = raw_env._compute_fork_tip()
    rel_tip = tip[:, :2] - pallet_pos[:, :2]
    s_tip = torch.sum(rel_tip * u_in, dim=-1)
    s_front = -0.5 * raw_env.cfg.pallet_depth_m
    insert_depth = torch.clamp(s_tip - s_front, min=0.0)
    insert_norm = torch.clamp(insert_depth / (raw_env.cfg.pallet_depth_m + 1e-6), 0.0, 1.0)

    # lift
    lift_height = tip[:, 2] - raw_env._fork_tip_z0

    # z_err and pitch
    z_err = tip[:, 2] - pallet_pos[:, 2]
    pitch_deg = _quat_to_pitch(robot_quat) * (180.0 / math.pi)

    return {
        "y_err": y_err,
        "yaw_err_deg": yaw_err_deg,
        "insert_norm": insert_norm,
        "insert_depth": insert_depth,
        "lift_height": lift_height,
        "hold_counter": raw_env._hold_counter.clone(),
        "z_err": z_err,
        "pitch_deg": pitch_deg,
    }


# ---- 难度分层 ----
def classify_difficulty(init_x: float, init_y: float, init_yaw_deg: float) -> str:
    """根据初始位姿分类难度等级。

    定义（与 s1.0s 计划 Section 3.7 一致）:
    - Easy:    |y|<0.3 AND |yaw|<7deg AND x<-3.0
    - Hard:    |y|>0.4 AND |yaw|>10deg
    - Extreme: |y|>0.4 AND |yaw|>10deg AND x>-2.8
    - Medium:  其余
    """
    abs_y = abs(init_y)
    abs_yaw = abs(init_yaw_deg)

    if abs_y > 0.4 and abs_yaw > 10.0 and init_x > -2.8:
        return "extreme"
    elif abs_y > 0.4 and abs_yaw > 10.0:
        return "hard"
    elif abs_y < 0.3 and abs_yaw < 7.0 and init_x < -3.0:
        return "easy"
    else:
        return "medium"


# ==============================================================================
# Per-episode 收集（单 seed）
# ==============================================================================
def run_eval_single_seed(env_wrapped, raw_env, policy_nn, device, max_steps, seed_val):
    """收集 per-episode 统计，扩展初始位姿捕获。"""
    N = raw_env.num_envs
    print(f"\n[EVAL] seed={seed_val}, {N} envs, max_steps={max_steps}", flush=True)

    # -- Per-env episode buffers --
    ep_total_reward = torch.zeros(N, device=device)
    ep_steps = torch.zeros(N, dtype=torch.int32, device=device)
    ep_max_insert = torch.zeros(N, device=device)
    ep_max_hold = torch.zeros(N, device=device)
    ep_success = torch.zeros(N, dtype=torch.bool, device=device)
    ep_max_lift = torch.zeros(N, device=device)

    # Funnel flags
    ep_ever_near = torch.zeros(N, dtype=torch.bool, device=device)
    ep_ever_deep = torch.zeros(N, dtype=torch.bool, device=device)
    ep_ever_yaw_ok = torch.zeros(N, dtype=torch.bool, device=device)
    ep_ever_lat_ok = torch.zeros(N, dtype=torch.bool, device=device)
    ep_ever_both_ok = torch.zeros(N, dtype=torch.bool, device=device)
    ep_ever_grace = torch.zeros(N, dtype=torch.bool, device=device)
    ep_ever_half_hold = torch.zeros(N, dtype=torch.bool, device=device)
    ep_ever_dead_zone = torch.zeros(N, dtype=torch.bool, device=device)

    # First-near step tracking
    ep_first_near_step = torch.full((N,), -1, dtype=torch.int32, device=device)

    # Near-field accumulation
    ep_yaw_near_sum = torch.zeros(N, device=device)
    ep_yaw_near_count = torch.zeros(N, dtype=torch.int32, device=device)
    ep_lat_near_sum = torch.zeros(N, device=device)
    ep_near_steps = torch.zeros(N, dtype=torch.int32, device=device)

    # Terminal state
    ep_final_insert = torch.zeros(N, device=device)
    ep_final_y_err = torch.zeros(N, device=device)
    ep_final_yaw_err = torch.zeros(N, device=device)

    # Shadow stuck counter
    ep_shadow_stuck_counter = torch.zeros(N, dtype=torch.int32, device=device)
    ep_max_stuck_counter = torch.zeros(N, dtype=torch.int32, device=device)
    ep_prev_insert_norm = torch.zeros(N, device=device)
    ep_prev_y_err = torch.zeros(N, device=device)

    # Extended fields
    ep_min_y_err = torch.full((N,), float('inf'), device=device)
    ep_min_yaw_err = torch.full((N,), float('inf'), device=device)
    ep_deep_steps = torch.zeros(N, dtype=torch.int32, device=device)
    ep_min_abs_z_err = torch.full((N,), float('inf'), device=device)
    ep_max_pitch_deg = torch.zeros(N, device=device)
    ep_action_flips = torch.zeros(N, dtype=torch.int32, device=device)
    ep_prev_actions = None

    # S1.0S: 初始位姿捕获（每 episode 首步记录）
    ep_init_x = torch.zeros(N, device=device)
    ep_init_y = torch.zeros(N, device=device)
    ep_init_yaw = torch.zeros(N, device=device)
    ep_init_recorded = torch.zeros(N, dtype=torch.bool, device=device)

    completed = []

    obs = env_wrapped.get_observations()
    total_steps = 0

    while total_steps < max_steps:
        with torch.inference_mode():
            actions = policy_nn.act_inference(obs)
            obs, rewards, dones, infos = env_wrapped.step(actions)

        total_steps += 1
        geo = compute_geometry(raw_env)
        insert_norm = geo["insert_norm"]
        yaw_err_deg = geo["yaw_err_deg"]
        y_err = geo["y_err"]
        hold_counter = geo["hold_counter"]
        lift_height = geo["lift_height"]
        z_err = geo["z_err"]
        pitch_deg = geo["pitch_deg"]

        # S1.0S: 捕获初始位姿（仅首步）
        need_init = ~ep_init_recorded
        if need_init.any():
            root_pos = raw_env.robot.data.root_pos_w
            robot_yaw = _quat_to_yaw(raw_env.robot.data.root_quat_w)
            ep_init_x = torch.where(need_init, root_pos[:, 0], ep_init_x)
            ep_init_y = torch.where(need_init, root_pos[:, 1], ep_init_y)
            ep_init_yaw = torch.where(need_init, robot_yaw, ep_init_yaw)
            ep_init_recorded |= need_init

        # -- Accumulate stats --
        rew_1d = rewards.squeeze(-1) if rewards.dim() > 1 else rewards
        ep_total_reward += rew_1d
        ep_steps += 1
        ep_max_insert = torch.maximum(ep_max_insert, insert_norm)
        ep_max_hold = torch.maximum(ep_max_hold, hold_counter)
        ep_max_lift = torch.maximum(ep_max_lift, lift_height)
        ep_success |= (hold_counter >= raw_env._hold_steps)

        near_mask = insert_norm > 0.1
        deep_mask = insert_norm > 0.3
        yaw_ok_mask = near_mask & (yaw_err_deg < raw_env.cfg.max_yaw_err_deg)
        lat_ok_mask = near_mask & (y_err < raw_env.cfg.max_lateral_err_m)
        both_ok_mask = yaw_ok_mask & lat_ok_mask
        dead_zone_mask = deep_mask & (y_err > 0.20)

        ep_ever_near |= near_mask
        ep_ever_deep |= deep_mask
        ep_ever_yaw_ok |= yaw_ok_mask
        ep_ever_lat_ok |= lat_ok_mask
        ep_ever_both_ok |= both_ok_mask
        ep_ever_grace |= (hold_counter > 0)
        ep_ever_half_hold |= (hold_counter >= 5)
        ep_ever_dead_zone |= dead_zone_mask

        # First near step
        newly_near = near_mask & (ep_first_near_step < 0)
        ep_first_near_step = torch.where(newly_near, ep_steps, ep_first_near_step)

        # Near-field accumulation
        ep_yaw_near_sum += torch.where(near_mask, yaw_err_deg, torch.zeros_like(yaw_err_deg))
        ep_yaw_near_count += near_mask.int()
        ep_lat_near_sum += torch.where(near_mask, y_err, torch.zeros_like(y_err))
        ep_near_steps += near_mask.int()

        # Track final state
        ep_final_insert = insert_norm.clone()
        ep_final_y_err = y_err.clone()
        ep_final_yaw_err = yaw_err_deg.clone()

        # Shadow stuck counter
        shadow_stuck_cond = dead_zone_mask & (
            torch.abs(insert_norm - ep_prev_insert_norm) < 0.005
        ) & (
            torch.abs(y_err - ep_prev_y_err) < 0.005
        )
        ep_shadow_stuck_counter = torch.where(
            shadow_stuck_cond, ep_shadow_stuck_counter + 1,
            torch.zeros_like(ep_shadow_stuck_counter))
        ep_max_stuck_counter = torch.maximum(ep_max_stuck_counter, ep_shadow_stuck_counter)
        ep_prev_insert_norm = insert_norm.clone()
        ep_prev_y_err = y_err.clone()

        # Extended fields
        ep_min_y_err = torch.minimum(ep_min_y_err, y_err)
        ep_min_yaw_err = torch.minimum(ep_min_yaw_err, yaw_err_deg)
        ep_deep_steps += deep_mask.int()
        ep_min_abs_z_err = torch.minimum(ep_min_abs_z_err, torch.abs(z_err))
        ep_max_pitch_deg = torch.maximum(ep_max_pitch_deg, torch.abs(pitch_deg))
        if ep_prev_actions is None:
            ep_prev_actions = torch.zeros_like(actions)
        sign_flips = (actions * ep_prev_actions < 0).any(dim=-1)
        ep_action_flips += sign_flips.int()
        ep_prev_actions = actions.clone()

        # -- Detect resets --
        if isinstance(dones, torch.Tensor):
            done_mask = dones.bool()
        else:
            done_mask = torch.tensor(dones, dtype=torch.bool, device=device)

        done_ids = torch.where(done_mask)[0]
        for idx in done_ids:
            i = idx.item()
            yaw_near_mean = (ep_yaw_near_sum[i] / max(ep_yaw_near_count[i].item(), 1)).item()
            lat_near_mean = (ep_lat_near_sum[i] / max(ep_yaw_near_count[i].item(), 1)).item()

            is_success = bool(ep_success[i].item())
            is_dz_stuck = bool(hasattr(raw_env, '_early_stop_dz_stuck') and raw_env._early_stop_dz_stuck[i].item())
            steps_i = ep_steps[i].item()
            is_timeout = (steps_i >= raw_env.max_episode_length) and not is_success
            if is_success:
                term_reason = "success"
            elif is_dz_stuck:
                term_reason = "early_dz_stuck"
            elif is_timeout:
                term_reason = "timeout"
            else:
                term_reason = "other"

            # S1.0S: 初始位姿 + 难度分类
            init_x_val = ep_init_x[i].item()
            init_y_val = ep_init_y[i].item()
            init_yaw_val = ep_init_yaw[i].item() * (180.0 / math.pi)  # -> deg
            difficulty = classify_difficulty(init_x_val, init_y_val, init_yaw_val)

            completed.append({
                "seed": seed_val,
                "total_reward": ep_total_reward[i].item(),
                "steps": steps_i,
                "max_insert_norm": ep_max_insert[i].item(),
                "max_hold_counter": ep_max_hold[i].item(),
                "max_lift_height": ep_max_lift[i].item(),
                "success": int(is_success),
                "ever_near": int(ep_ever_near[i].item()),
                "ever_deep": int(ep_ever_deep[i].item()),
                "ever_yaw_ok": int(ep_ever_yaw_ok[i].item()),
                "ever_lat_ok": int(ep_ever_lat_ok[i].item()),
                "ever_both_ok": int(ep_ever_both_ok[i].item()),
                "ever_grace": int(ep_ever_grace[i].item()),
                "ever_half_hold": int(ep_ever_half_hold[i].item()),
                "ever_dead_zone": int(ep_ever_dead_zone[i].item()),
                "first_near_step": ep_first_near_step[i].item(),
                "yaw_near_mean": yaw_near_mean,
                "lat_near_mean": lat_near_mean,
                "near_steps": ep_near_steps[i].item(),
                "near_frac": ep_near_steps[i].item() / max(steps_i, 1),
                "final_insert_norm": ep_final_insert[i].item(),
                "final_y_err": ep_final_y_err[i].item(),
                "final_yaw_err_deg": ep_final_yaw_err[i].item(),
                "term_reason": term_reason,
                "max_stuck_counter": ep_max_stuck_counter[i].item(),
                "min_y_err": ep_min_y_err[i].item(),
                "min_yaw_err_deg": ep_min_yaw_err[i].item(),
                "deep_steps": ep_deep_steps[i].item(),
                "min_abs_z_err": ep_min_abs_z_err[i].item(),
                "max_pitch_deg": ep_max_pitch_deg[i].item(),
                "action_flip_rate": ep_action_flips[i].item() / max(steps_i, 1),
                # S1.0S: 新增字段
                "init_x": init_x_val,
                "init_y": init_y_val,
                "init_yaw_deg": init_yaw_val,
                "difficulty_tier": difficulty,
            })

            # Reset per-episode buffers
            ep_total_reward[i] = 0
            ep_steps[i] = 0
            ep_max_insert[i] = 0
            ep_max_hold[i] = 0
            ep_max_lift[i] = 0
            ep_success[i] = False
            ep_ever_near[i] = False
            ep_ever_deep[i] = False
            ep_ever_yaw_ok[i] = False
            ep_ever_lat_ok[i] = False
            ep_ever_both_ok[i] = False
            ep_ever_grace[i] = False
            ep_ever_half_hold[i] = False
            ep_ever_dead_zone[i] = False
            ep_first_near_step[i] = -1
            ep_yaw_near_sum[i] = 0
            ep_yaw_near_count[i] = 0
            ep_lat_near_sum[i] = 0
            ep_near_steps[i] = 0
            ep_shadow_stuck_counter[i] = 0
            ep_max_stuck_counter[i] = 0
            ep_prev_insert_norm[i] = 0
            ep_prev_y_err[i] = 0
            ep_min_y_err[i] = float('inf')
            ep_min_yaw_err[i] = float('inf')
            ep_deep_steps[i] = 0
            ep_min_abs_z_err[i] = float('inf')
            ep_max_pitch_deg[i] = 0
            ep_action_flips[i] = 0
            if ep_prev_actions is not None:
                ep_prev_actions[i] = 0
            # S1.0S: 重置初始位姿
            ep_init_recorded[i] = False
            ep_init_x[i] = 0
            ep_init_y[i] = 0
            ep_init_yaw[i] = 0

        if total_steps % 200 == 0:
            print(f"  step {total_steps}/{max_steps}, episodes={len(completed)}", flush=True)

    print(f"[EVAL] seed={seed_val} 完成: {len(completed)} episodes", flush=True)
    return completed


# ==============================================================================
# Bootstrap 置信区间
# ==============================================================================
def bootstrap_ci(arr, n_boot=2000, alpha=0.05):
    """计算 Bootstrap 95% CI for mean。"""
    if len(arr) == 0:
        return 0.0, 0.0, 0.0
    arr = np.asarray(arr, dtype=float)
    n = len(arr)
    means = np.array([np.mean(np.random.choice(arr, size=n, replace=True)) for _ in range(n_boot)])
    lo = np.percentile(means, 100 * alpha / 2)
    hi = np.percentile(means, 100 * (1 - alpha / 2))
    return float(arr.mean()), float(lo), float(hi)


# ==============================================================================
# 分析与输出
# ==============================================================================
def analyze_and_output(episodes, experiment_name, output_dir, seed_list):
    """分析 per-episode 数据并输出报告 + CSV + JSON。"""
    os.makedirs(output_dir, exist_ok=True)
    n_ep = len(episodes)

    if n_ep == 0:
        print(f"[WARN] 没有完成的 episode，跳过分析")
        return {"experiment": experiment_name, "n_episodes": 0}

    sep = "=" * 70
    print(f"\n{sep}")
    print(f"S1.0S 离线诊断报告: {experiment_name} ({n_ep} episodes, {len(seed_list)} seeds)")
    print(f"Seeds: {seed_list}")
    print(f"{sep}")

    # ======== 1. 核心 KPI ========
    success_arr = np.array([e["success"] for e in episodes])
    rew_arr = np.array([e["total_reward"] for e in episodes])
    ep_len_arr = np.array([e["steps"] for e in episodes])
    lift_arr = np.array([e["max_lift_height"] for e in episodes])

    success_rate = success_arr.mean()
    ep_reward_p50 = np.percentile(rew_arr, 50)
    p95_ep_len = np.percentile(ep_len_arr, 95)

    fail_mask = ~success_arr.astype(bool)
    fail_ep_lens = ep_len_arr[fail_mask]
    total_steps_all = ep_len_arr.sum()
    fail_step_share = fail_ep_lens.sum() / max(total_steps_all, 1)
    mean_fail_ep_len = fail_ep_lens.mean() if len(fail_ep_lens) > 0 else 0.0
    max_ep_len = int(ep_len_arr.max())
    timeout_frac = (ep_len_arr == max_ep_len).mean()

    lat_near_vals = np.array([e["lat_near_mean"] for e in episodes if e["lat_near_mean"] > 0])
    yaw_near_vals = np.array([e["yaw_near_mean"] for e in episodes if e["yaw_near_mean"] > 0])
    lateral_near_p90 = np.percentile(lat_near_vals, 90) if len(lat_near_vals) > 0 else float('inf')
    yaw_near_p90 = np.percentile(yaw_near_vals, 90) if len(yaw_near_vals) > 0 else float('inf')

    # Bootstrap CI for success rate
    sr_mean, sr_lo, sr_hi = bootstrap_ci(success_arr)

    print(f"\n--- 核心 KPI ---")
    print(f"  success_rate_ep     = {success_rate*100:.2f}%  (95% CI: [{sr_lo*100:.2f}%, {sr_hi*100:.2f}%])")
    print(f"  ep_reward_p50       = {ep_reward_p50:.2f}")
    print(f"  ep_reward_mean      = {rew_arr.mean():.2f}")
    print(f"  p50_ep_len          = {np.percentile(ep_len_arr, 50):.0f}")
    print(f"  p95_ep_len          = {p95_ep_len:.0f}")
    print(f"  mean_ep_len         = {ep_len_arr.mean():.0f}")
    print(f"  fail_step_share     = {fail_step_share*100:.2f}%")
    print(f"  mean_fail_ep_len    = {mean_fail_ep_len:.0f}")
    print(f"  P(ep_len==max)      = {timeout_frac*100:.2f}%  (max_ep_len={max_ep_len})")
    print(f"  lateral_near_p90    = {lateral_near_p90:.3f} m")
    print(f"  yaw_near_p90        = {yaw_near_p90:.2f}°")

    # Throughput
    success_per_1e6 = success_arr.sum() / max(total_steps_all, 1) * 1e6
    timeout_per_1e6 = (ep_len_arr == max_ep_len).sum() / max(total_steps_all, 1) * 1e6
    print(f"  success_per_1e6_steps = {success_per_1e6:.1f}")
    print(f"  timeout_per_1e6_steps = {timeout_per_1e6:.1f}")

    # S1.0S: lift height 统计
    print(f"\n--- 举升高度统计 ---")
    success_lifts = lift_arr[success_arr.astype(bool)]
    print(f"  max_lift_height (all): p50={np.percentile(lift_arr, 50):.3f}, p90={np.percentile(lift_arr, 90):.3f}, max={lift_arr.max():.3f}")
    if len(success_lifts) > 0:
        print(f"  max_lift_height (success): p50={np.percentile(success_lifts, 50):.3f}, p90={np.percentile(success_lifts, 90):.3f}")

    # ======== 2. 漏斗分析 ========
    funnel = {
        "P(insert>0.1) [near]": np.mean([e["ever_near"] for e in episodes]),
        "P(insert>0.3) [deep]": np.mean([e["ever_deep"] for e in episodes]),
        "P(near & yaw<5°)":     np.mean([e["ever_yaw_ok"] for e in episodes]),
        "P(near & y<0.15m)":    np.mean([e["ever_lat_ok"] for e in episodes]),
        "P(near & yaw<5 & y<0.15)": np.mean([e["ever_both_ok"] for e in episodes]),
        "P(hold>0) [grace]":    np.mean([e["ever_grace"] for e in episodes]),
        "P(hold>=5) [half]":    np.mean([e["ever_half_hold"] for e in episodes]),
        "P(success)":           success_rate,
    }
    print(f"\n--- 漏斗分析 (per-episode) ---")
    for k, v in funnel.items():
        bar = "#" * int(v * 50)
        print(f"  {k:40s}  {v*100:6.2f}%  |{bar}")

    # ======== 3. 死区分析 ========
    dead_zone_arr = np.array([e["ever_dead_zone"] for e in episodes])
    dz_rate = dead_zone_arr.mean()

    failed_eps = [e for e in episodes if not e["success"]]
    n_failed = len(failed_eps)
    if n_failed > 0:
        dz_among_failed = np.mean([e["ever_dead_zone"] for e in failed_eps])
    else:
        dz_among_failed = 0.0

    success_eps = [e for e in episodes if e["success"]]
    n_success = len(success_eps)
    if n_success > 0:
        dz_among_success = np.mean([e["ever_dead_zone"] for e in success_eps])
    else:
        dz_among_success = 0.0

    print(f"\n--- 死区分析 (insert>0.3 & y_err>0.2) ---")
    print(f"  P(ever_dead_zone) 全部         = {dz_rate*100:.2f}%  ({int(dead_zone_arr.sum())}/{n_ep})")
    print(f"  P(ever_dead_zone | failed)     = {dz_among_failed*100:.2f}%  (失败 {n_failed} ep)")
    print(f"  P(ever_dead_zone | success)    = {dz_among_success*100:.2f}%  (成功 {n_success} ep)")

    # ======== 4. S1.0S 区域分层分析 ========
    tiers = ["easy", "medium", "hard", "extreme"]
    tier_data = {t: [e for e in episodes if e["difficulty_tier"] == t] for t in tiers}

    print(f"\n--- S1.0S 区域分层分析 ---")
    tier_stats = {}
    for t in tiers:
        t_eps = tier_data[t]
        n_t = len(t_eps)
        if n_t > 0:
            t_success = [e["success"] for e in t_eps]
            sr_m, sr_l, sr_h = bootstrap_ci(t_success)
            tier_stats[t] = {
                "n_episodes": n_t,
                "frac": n_t / n_ep,
                "success_rate": round(sr_m * 100, 2),
                "success_rate_ci_lo": round(sr_l * 100, 2),
                "success_rate_ci_hi": round(sr_h * 100, 2),
            }
            print(f"  {t:8s}: N={n_t:5d} ({n_t/n_ep*100:5.1f}%), success={sr_m*100:6.2f}% "
                  f"(95% CI: [{sr_l*100:.2f}%, {sr_h*100:.2f}%])")
        else:
            tier_stats[t] = {
                "n_episodes": 0, "frac": 0.0,
                "success_rate": 0.0, "success_rate_ci_lo": 0.0, "success_rate_ci_hi": 0.0,
            }
            print(f"  {t:8s}: N=    0 (0.0%)")

    # Success gap
    sr_easy = tier_stats["easy"]["success_rate"]
    sr_hard = tier_stats["hard"]["success_rate"]
    sr_extreme = tier_stats["extreme"]["success_rate"]
    success_gap = sr_easy - sr_hard
    print(f"\n  success_gap (easy - hard) = {success_gap:.2f}pp")
    print(f"  success_rate_extreme      = {sr_extreme:.2f}%")

    # ======== 5. 长尾分析 ========
    sorted_by_len = sorted(episodes, key=lambda e: e["steps"], reverse=True)
    top5pct_n = max(1, int(n_ep * 0.05))
    top5pct_eps = sorted_by_len[:top5pct_n]
    top5pct_success = np.mean([e["success"] for e in top5pct_eps])
    top5pct_dz = np.mean([e["ever_dead_zone"] for e in top5pct_eps])
    top5pct_mean_len = np.mean([e["steps"] for e in top5pct_eps])

    print(f"\n--- 长尾分析 (episode length top 5%, N={top5pct_n}) ---")
    print(f"  mean_ep_len         = {top5pct_mean_len:.0f}")
    print(f"  success_rate        = {top5pct_success*100:.2f}%")
    print(f"  P(ever_dead_zone)   = {top5pct_dz*100:.2f}%")

    # ======== 6. Per-seed 摘要 ========
    if len(seed_list) > 1:
        print(f"\n--- Per-seed 摘要 ---")
        for s in seed_list:
            s_eps = [e for e in episodes if e["seed"] == s]
            if len(s_eps) > 0:
                s_sr = np.mean([e["success"] for e in s_eps])
                print(f"  seed={s:4d}: N={len(s_eps):5d}, success={s_sr*100:.2f}%")

    # ======== 7. CSV 输出 ========
    csv_path = os.path.join(output_dir, f"s1.0s_{experiment_name}_episodes.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(episodes[0].keys()))
        writer.writeheader()
        writer.writerows(episodes)
    print(f"\n[CSV] Per-episode 数据: {csv_path}")

    # ======== 8. JSON 摘要 ========
    summary = {
        "experiment": experiment_name,
        "n_seeds": len(seed_list),
        "seeds": seed_list,
        "n_episodes": n_ep,
        "success_rate_ep": round(success_rate * 100, 2),
        "success_rate_ci_lo": round(sr_lo * 100, 2),
        "success_rate_ci_hi": round(sr_hi * 100, 2),
        "ep_reward_p50": round(ep_reward_p50, 2),
        "ep_reward_mean": round(rew_arr.mean(), 2),
        "p50_ep_len": round(np.percentile(ep_len_arr, 50), 0),
        "p95_ep_len": round(p95_ep_len, 0),
        "mean_ep_len": round(ep_len_arr.mean(), 0),
        "P_ever_near": round(funnel["P(insert>0.1) [near]"] * 100, 2),
        "P_ever_deep": round(funnel["P(insert>0.3) [deep]"] * 100, 2),
        "P_ever_both_ok": round(funnel["P(near & yaw<5 & y<0.15)"] * 100, 2),
        "P_ever_grace": round(funnel["P(hold>0) [grace]"] * 100, 2),
        "P_ever_dead_zone": round(dz_rate * 100, 2),
        "P_dz_given_failed": round(dz_among_failed * 100, 2),
        "fail_step_share": round(fail_step_share * 100, 2),
        "mean_fail_ep_len": round(float(mean_fail_ep_len), 0),
        "timeout_frac": round(float(timeout_frac) * 100, 2),
        "lateral_near_p90": round(float(lateral_near_p90), 3),
        "yaw_near_p90": round(float(yaw_near_p90), 2),
        "success_per_1e6_steps": round(float(success_per_1e6), 1),
        "timeout_per_1e6_steps": round(float(timeout_per_1e6), 1),
        # S1.0S: 区域分层
        "tier_stats": tier_stats,
        "success_gap_easy_hard": round(success_gap, 2),
        "success_rate_easy": sr_easy,
        "success_rate_hard": sr_hard,
        "success_rate_extreme": sr_extreme,
        # S1.0S: lift height 分布
        "lift_height_p50": round(float(np.percentile(lift_arr, 50)), 3),
        "lift_height_p90": round(float(np.percentile(lift_arr, 90)), 3),
        "lift_height_max": round(float(lift_arr.max()), 3),
    }

    json_path = os.path.join(output_dir, f"s1.0s_{experiment_name}_summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[JSON] 摘要数据: {json_path}")

    print(f"\n[SUMMARY_JSON] {json.dumps(summary)}")

    return summary


# ==============================================================================
# Main
# ==============================================================================
@hydra_task_config(args.task, args.agent)
def main(env_cfg, agent_cfg):
    ckpt = args.checkpoint
    experiment_name = args.experiment_name
    output_dir = os.path.abspath(args.output_dir)

    env_cfg.scene.num_envs = args.num_envs
    env_cfg.sim.device = "cuda:0"

    all_episodes = []

    for seed_idx, seed_val in enumerate(SEED_LIST):
        print(f"\n{'='*70}")
        print(f"[MULTI-SEED] Running seed {seed_val} ({seed_idx+1}/{len(SEED_LIST)})")
        print(f"{'='*70}")

        env_cfg.seed = seed_val
        env = gym.make(args.task, cfg=env_cfg)
        env_wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        raw_env = env.unwrapped
        device = raw_env.device

        print(f"[INFO] Experiment: {experiment_name}, seed={seed_val}", flush=True)
        print(f"[INFO] Loading checkpoint: {ckpt}", flush=True)
        runner = OnPolicyRunner(env_wrapped, agent_cfg.to_dict(), log_dir=None, device="cuda:0")
        runner.load(ckpt)
        try:
            policy_nn = runner.alg.policy
        except AttributeError:
            policy_nn = runner.alg.actor_critic

        t0 = _time.time()
        episodes = run_eval_single_seed(
            env_wrapped, raw_env, policy_nn, device, args.max_steps, seed_val,
        )
        elapsed = _time.time() - t0
        print(f"[INFO] seed={seed_val} eval 耗时: {elapsed:.1f}s", flush=True)
        env.close()

        all_episodes.extend(episodes)

    print(f"\n[MULTI-SEED] 总计: {len(all_episodes)} episodes across {len(SEED_LIST)} seeds")
    analyze_and_output(all_episodes, experiment_name, output_dir, SEED_LIST)


if __name__ == "__main__":
    main()
    simulation_app.close()
