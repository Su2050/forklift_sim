"""S1.0Q 离线 per-episode 诊断脚本。

基于 eval_s1.0p_diagnostics.py 改造，支持通过 --checkpoint 指定不同实验的模型。

输出:
  - 控制台报告（漏斗 / 分布 / 死区分析 / 长尾分析）
  - CSV 文件 per-episode 统计

Usage:
    cd IsaacLab && CONDA_PREFIX="" TERM=xterm bash isaaclab.sh -p \
        ../scripts/eval_s1.0q_diagnostics.py \
        --task Isaac-Forklift-PalletInsertLift-Direct-v0 \
        --headless --num_envs 1024 \
        --checkpoint logs/rsl_rl/forklift_pallet_insert_lift/2026-02-13_13-20-41/model_3595.pt \
        --experiment_name A1 \
        --max_steps 2000
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time as _time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="S1.0Q per-episode diagnostics")
AppLauncher.add_app_launcher_args(parser)
parser.add_argument("--task", type=str, default="Isaac-Forklift-PalletInsertLift-Direct-v0")
parser.add_argument("--num_envs", type=int, default=1024)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--max_steps", type=int, default=2000,
                    help="Max env.step() calls (1024 envs × ~340 steps/ep → ~6000 episodes)")
parser.add_argument("--checkpoint", type=str, required=True,
                    help="Path to model checkpoint (relative to IsaacLab dir or absolute)")
parser.add_argument("--experiment_name", type=str, default="unknown",
                    help="Experiment label for output")
parser.add_argument("--output_dir", type=str, default="../data/s1.0q_eval",
                    help="Base output directory")
args = parser.parse_args()

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
    y_err = torch.abs(torch.sum(rel_robot * v_lat, dim=-1))

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

    # Batch-4 补充: z_err (叉齿尖端与托盘的高度差, 带符号) 和 pitch
    z_err = tip[:, 2] - pallet_pos[:, 2]          # >0 偏高, <0 偏低
    pitch_deg = _quat_to_pitch(robot_quat) * (180.0 / math.pi)  # 带符号度数

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


# ==============================================================================
# Per-episode 收集
# ==============================================================================
def run_eval(env_wrapped, raw_env, policy_nn, device, max_steps):
    """收集 per-episode 统计，重点关注死区和长尾指标。"""
    N = raw_env.num_envs
    print(f"\n[EVAL] {N} envs, max_steps={max_steps}", flush=True)

    # -- Per-env episode buffers --
    ep_total_reward = torch.zeros(N, device=device)
    ep_steps = torch.zeros(N, dtype=torch.int32, device=device)
    ep_max_insert = torch.zeros(N, device=device)
    ep_max_hold = torch.zeros(N, device=device)
    ep_success = torch.zeros(N, dtype=torch.bool, device=device)

    # Funnel flags
    ep_ever_near = torch.zeros(N, dtype=torch.bool, device=device)
    ep_ever_deep = torch.zeros(N, dtype=torch.bool, device=device)
    ep_ever_yaw_ok = torch.zeros(N, dtype=torch.bool, device=device)
    ep_ever_lat_ok = torch.zeros(N, dtype=torch.bool, device=device)
    ep_ever_both_ok = torch.zeros(N, dtype=torch.bool, device=device)
    ep_ever_grace = torch.zeros(N, dtype=torch.bool, device=device)
    ep_ever_half_hold = torch.zeros(N, dtype=torch.bool, device=device)

    # Dead zone flags (S1.0Q 新增)
    ep_ever_dead_zone = torch.zeros(N, dtype=torch.bool, device=device)  # insert>0.3 AND y_err>0.2

    # First-near step tracking (anti-self-deception)
    ep_first_near_step = torch.full((N,), -1, dtype=torch.int32, device=device)

    # Near-field accumulation
    ep_yaw_near_sum = torch.zeros(N, device=device)
    ep_yaw_near_count = torch.zeros(N, dtype=torch.int32, device=device)
    ep_lat_near_sum = torch.zeros(N, device=device)
    ep_near_steps = torch.zeros(N, dtype=torch.int32, device=device)

    # Terminal state (最终 step 的状态)
    ep_final_insert = torch.zeros(N, device=device)
    ep_final_y_err = torch.zeros(N, device=device)
    ep_final_yaw_err = torch.zeros(N, device=device)

    # S1.0Q Batch-4: shadow stuck counter (eval 侧独立实现)
    ep_shadow_stuck_counter = torch.zeros(N, dtype=torch.int32, device=device)
    ep_max_stuck_counter = torch.zeros(N, dtype=torch.int32, device=device)
    ep_prev_insert_norm = torch.zeros(N, device=device)
    ep_prev_y_err = torch.zeros(N, device=device)

    # S1.0Q Batch-4: 扩展诊断字段
    ep_min_y_err = torch.full((N,), float('inf'), device=device)
    ep_min_yaw_err = torch.full((N,), float('inf'), device=device)
    ep_deep_steps = torch.zeros(N, dtype=torch.int32, device=device)

    # S1.0Q Batch-4 补充: z_err / pitch / action chattering
    ep_min_abs_z_err = torch.full((N,), float('inf'), device=device)
    ep_max_pitch_deg = torch.zeros(N, device=device)
    ep_action_flips = torch.zeros(N, dtype=torch.int32, device=device)
    ep_prev_actions = None  # 延迟初始化（需要知道 action_dim）

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
        z_err = geo["z_err"]
        pitch_deg = geo["pitch_deg"]

        # -- Accumulate stats --
        rew_1d = rewards.squeeze(-1) if rewards.dim() > 1 else rewards
        ep_total_reward += rew_1d
        ep_steps += 1
        ep_max_insert = torch.maximum(ep_max_insert, insert_norm)
        ep_max_hold = torch.maximum(ep_max_hold, hold_counter)
        ep_success |= (hold_counter >= raw_env._hold_steps)

        near_mask = insert_norm > 0.1
        deep_mask = insert_norm > 0.3
        yaw_ok_mask = near_mask & (yaw_err_deg < raw_env.cfg.max_yaw_err_deg)
        lat_ok_mask = near_mask & (y_err < raw_env.cfg.max_lateral_err_m)
        both_ok_mask = yaw_ok_mask & lat_ok_mask

        # Dead zone: deep insertion with large lateral error
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

        # S1.0Q Batch-4: shadow stuck counter
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

        # S1.0Q Batch-4: 扩展字段
        ep_min_y_err = torch.minimum(ep_min_y_err, y_err)
        ep_min_yaw_err = torch.minimum(ep_min_yaw_err, yaw_err_deg)
        ep_deep_steps += deep_mask.int()

        # S1.0Q Batch-4 补充: z_err / pitch / action chattering
        ep_min_abs_z_err = torch.minimum(ep_min_abs_z_err, torch.abs(z_err))
        ep_max_pitch_deg = torch.maximum(ep_max_pitch_deg, torch.abs(pitch_deg))
        # action chattering: 任一维度发生符号翻转即计入
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

            # S1.0Q Batch-4: 判断 term_reason
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

            completed.append({
                "total_reward": ep_total_reward[i].item(),
                "steps": steps_i,
                "max_insert_norm": ep_max_insert[i].item(),
                "max_hold_counter": ep_max_hold[i].item(),
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
                # S1.0Q Batch-4: 新增字段
                "term_reason": term_reason,
                "max_stuck_counter": ep_max_stuck_counter[i].item(),
                "min_y_err": ep_min_y_err[i].item(),
                "min_yaw_err_deg": ep_min_yaw_err[i].item(),
                "deep_steps": ep_deep_steps[i].item(),
                # S1.0Q Batch-4 补充: z / pitch / chattering
                "min_abs_z_err": ep_min_abs_z_err[i].item(),
                "max_pitch_deg": ep_max_pitch_deg[i].item(),
                "action_flip_rate": ep_action_flips[i].item() / max(steps_i, 1),
            })

            # Reset per-episode buffers
            ep_total_reward[i] = 0
            ep_steps[i] = 0
            ep_max_insert[i] = 0
            ep_max_hold[i] = 0
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
            # S1.0Q Batch-4: shadow counter + 扩展字段
            ep_shadow_stuck_counter[i] = 0
            ep_max_stuck_counter[i] = 0
            ep_prev_insert_norm[i] = 0
            ep_prev_y_err[i] = 0
            ep_min_y_err[i] = float('inf')
            ep_min_yaw_err[i] = float('inf')
            ep_deep_steps[i] = 0
            # S1.0Q Batch-4 补充: z / pitch / chattering
            ep_min_abs_z_err[i] = float('inf')
            ep_max_pitch_deg[i] = 0
            ep_action_flips[i] = 0
            if ep_prev_actions is not None:
                ep_prev_actions[i] = 0

        if total_steps % 200 == 0:
            print(f"  step {total_steps}/{max_steps}, episodes={len(completed)}", flush=True)

    print(f"[EVAL] 完成: {len(completed)} episodes", flush=True)
    return completed


# ==============================================================================
# 分析与输出
# ==============================================================================
def analyze_and_output(episodes, experiment_name, output_dir):
    """分析 per-episode 数据并输出报告 + CSV。"""
    os.makedirs(output_dir, exist_ok=True)
    n_ep = len(episodes)

    if n_ep == 0:
        print(f"[WARN] 没有完成的 episode，跳过分析")
        return {"experiment": experiment_name, "n_episodes": 0}

    sep = "=" * 70
    print(f"\n{sep}")
    print(f"S1.0Q 离线诊断报告: {experiment_name} ({n_ep} episodes)")
    print(f"{sep}")

    # ======== 1. 核心 KPI ========
    success_arr = np.array([e["success"] for e in episodes])
    rew_arr = np.array([e["total_reward"] for e in episodes])
    ep_len_arr = np.array([e["steps"] for e in episodes])

    success_rate = success_arr.mean()
    ep_reward_p50 = np.percentile(rew_arr, 50)
    p95_ep_len = np.percentile(ep_len_arr, 95)

    # S1.0Q Batch-2: fail_step_share / mean_fail_episode_len / timeout 占比
    fail_mask = ~success_arr.astype(bool)
    fail_ep_lens = ep_len_arr[fail_mask]
    total_steps_all = ep_len_arr.sum()
    fail_step_share = fail_ep_lens.sum() / max(total_steps_all, 1)
    mean_fail_ep_len = fail_ep_lens.mean() if len(fail_ep_lens) > 0 else 0.0
    # timeout: episode 走满最大步数（env max_episode_length=1079）
    max_ep_len = int(ep_len_arr.max())
    timeout_frac = (ep_len_arr == max_ep_len).mean()

    # lateral / yaw p90 (near-field per-episode mean)
    lat_near_vals = np.array([e["lat_near_mean"] for e in episodes if e["lat_near_mean"] > 0])
    yaw_near_vals = np.array([e["yaw_near_mean"] for e in episodes if e["yaw_near_mean"] > 0])
    lateral_near_p90 = np.percentile(lat_near_vals, 90) if len(lat_near_vals) > 0 else float('inf')
    yaw_near_p90 = np.percentile(yaw_near_vals, 90) if len(yaw_near_vals) > 0 else float('inf')

    print(f"\n--- 核心 KPI ---")
    print(f"  success_rate_ep     = {success_rate*100:.2f}%")
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

    # S1.0Q Batch-4: 吞吐量指标
    success_per_1e6 = success_arr.sum() / max(total_steps_all, 1) * 1e6
    timeout_per_1e6 = (ep_len_arr == max_ep_len).sum() / max(total_steps_all, 1) * 1e6
    print(f"  success_per_1e6_steps = {success_per_1e6:.1f}")
    print(f"  timeout_per_1e6_steps = {timeout_per_1e6:.1f}")

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

    # ======== 3. 死区分析 (S1.0Q 重点) ========
    dead_zone_arr = np.array([e["ever_dead_zone"] for e in episodes])
    dz_rate = dead_zone_arr.mean()

    # 在失败 episode 中，多少比例进过死区
    failed_eps = [e for e in episodes if not e["success"]]
    n_failed = len(failed_eps)
    if n_failed > 0:
        dz_among_failed = np.mean([e["ever_dead_zone"] for e in failed_eps])
    else:
        dz_among_failed = 0.0

    # 成功 episode 中，多少比例曾进过死区
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

    # ======== 4. 长尾分析 ========
    # 按 episode length 排序，看 top 5% 的情况
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

    # ======== 5. 反自欺指标 ========
    first_near_steps = [e["first_near_step"] for e in episodes if e["first_near_step"] >= 0]
    if first_near_steps:
        mean_first_near = np.mean(first_near_steps)
    else:
        mean_first_near = float('inf')

    print(f"\n--- 反自欺指标 ---")
    print(f"  P(ever_near)              = {funnel['P(insert>0.1) [near]']*100:.2f}%")
    print(f"  P(ever_deep)              = {funnel['P(insert>0.3) [deep]']*100:.2f}%")
    print(f"  mean_steps_to_first_near  = {mean_first_near:.1f}")

    # ======== 6. 分布统计 ========
    print(f"\n--- 分布统计 ---")

    # hold_counter_max
    hold_arr = np.array([e["max_hold_counter"] for e in episodes])
    buckets = [(0, 0), (0.01, 1), (1, 3), (3, 5), (5, 7), (7, 9), (9, 9.99), (10, 999)]
    bucket_labels = ["= 0", "0-1", "1-3", "3-5", "5-7", "7-9", "9-10", ">= 10"]
    print(f"\n  hold_counter_max 分布:")
    for (lo, hi), label in zip(buckets, bucket_labels):
        if label == "= 0":
            cnt = np.sum(hold_arr == 0)
        elif label == ">= 10":
            cnt = np.sum(hold_arr >= 10)
        else:
            cnt = np.sum((hold_arr > lo) & (hold_arr <= hi))
        pct = cnt / n_ep * 100
        bar = "#" * int(pct / 2)
        print(f"    {label:8s}  {cnt:5d} ({pct:5.1f}%)  |{bar}")

    # max_insert_norm 分位数
    ins_arr = np.array([e["max_insert_norm"] for e in episodes])
    print(f"\n  max_insert_norm 分位数:")
    for p in [25, 50, 75, 90, 95]:
        print(f"    p{p}: {np.percentile(ins_arr, p):.3f}")

    # yaw/lateral in near-field (reuse arrays from KPI section)
    if len(yaw_near_vals) > 0:
        print(f"\n  yaw_deg (near-field mean), N={len(yaw_near_vals)}:")
        for p in [50, 75, 90, 95]:
            print(f"    p{p}: {np.percentile(yaw_near_vals, p):.2f}°")
    if len(lat_near_vals) > 0:
        print(f"\n  lateral (near-field mean), N={len(lat_near_vals)}:")
        for p in [50, 75, 90, 95]:
            print(f"    p{p}: {np.percentile(lat_near_vals, p):.3f} m")

    # episode reward 分位数
    print(f"\n  total_reward 分位数:")
    for p in [10, 25, 50, 75, 90]:
        print(f"    p{p}: {np.percentile(rew_arr, p):.2f}")

    # ======== 7. CSV 输出 ========
    csv_path = os.path.join(output_dir, f"s1.0q_{experiment_name}_episodes.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(episodes[0].keys()))
        writer.writeheader()
        writer.writerows(episodes)
    print(f"\n[CSV] Per-episode 数据: {csv_path}")

    # ======== 8. 摘要 JSON（方便后续自动汇总）========
    summary = {
        "experiment": experiment_name,
        "n_episodes": n_ep,
        "success_rate_ep": round(success_rate * 100, 2),
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
        "mean_steps_first_near": round(mean_first_near, 1),
        # S1.0Q Batch-2 新增指标
        "fail_step_share": round(fail_step_share * 100, 2),
        "mean_fail_ep_len": round(float(mean_fail_ep_len), 0),
        "timeout_frac": round(float(timeout_frac) * 100, 2),
        "lateral_near_p90": round(float(lateral_near_p90), 3),
        "yaw_near_p90": round(float(yaw_near_p90), 2),
        # S1.0Q Batch-4: 吞吐量指标
        "success_per_1e6_steps": round(float(success_per_1e6), 1),
        "timeout_per_1e6_steps": round(float(timeout_per_1e6), 1),
    }

    import json
    json_path = os.path.join(output_dir, f"s1.0q_{experiment_name}_summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[JSON] 摘要数据: {json_path}")

    # Print machine-readable summary line for batch collection
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
    env_cfg.seed = args.seed
    env_cfg.sim.device = "cuda:0"

    env = gym.make(args.task, cfg=env_cfg)
    env_wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    raw_env = env.unwrapped
    device = raw_env.device

    print(f"[INFO] Experiment: {experiment_name}", flush=True)
    print(f"[INFO] Loading checkpoint: {ckpt}", flush=True)
    runner = OnPolicyRunner(env_wrapped, agent_cfg.to_dict(), log_dir=None, device="cuda:0")
    runner.load(ckpt)
    try:
        policy_nn = runner.alg.policy
    except AttributeError:
        policy_nn = runner.alg.actor_critic

    t0 = _time.time()
    episodes = run_eval(
        env_wrapped, raw_env, policy_nn, device, args.max_steps,
    )
    elapsed = _time.time() - t0
    print(f"[INFO] Eval 耗时: {elapsed:.1f}s", flush=True)
    env.close()

    analyze_and_output(episodes, experiment_name, output_dir)


if __name__ == "__main__":
    main()
    simulation_app.close()
