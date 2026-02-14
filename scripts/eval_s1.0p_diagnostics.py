"""S1.0P 诊断脚本：漏斗分析 / 分布统计 / 奖励分解 / 代表性 rollout。

回复 s1.0p_feedback.md 的数据请求 3/4/5。

Phase 1: 1024 envs × ~300 episodes，收集 per-episode 统计
Phase 2: 16 envs × ~50 episodes，收集 step-level rollout

Usage:
    cd IsaacLab && CONDA_PREFIX="" TERM=xterm bash isaaclab.sh -p \
        ../scripts/eval_s1.0p_diagnostics.py \
        --task Isaac-Forklift-PalletInsertLift-Direct-v0 \
        --headless --num_envs 1024
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="S1.0P diagnostics")
AppLauncher.add_app_launcher_args(parser)
parser.add_argument("--task", type=str, default="Isaac-Forklift-PalletInsertLift-Direct-v0")
parser.add_argument("--num_envs", type=int, default=1024)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--max_steps", type=int, default=2000,
                    help="Max env.step() calls for Phase 1 (1024 envs × ~340 steps/ep → ~2000 steps ≈ 6000 episodes)")
parser.add_argument("--rollout_envs", type=int, default=16)
parser.add_argument("--rollout_episodes", type=int, default=100)
parser.add_argument("--output_dir", type=str, default="../data")
args = parser.parse_args()

# Hydra (used by hydra_task_config) re-reads sys.argv and rejects unknown args.
# Clear sys.argv after argparse so Hydra only sees the script name.
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


def compute_geometry(raw_env):
    """从 raw_env 直接计算 per-env 几何指标，与 _get_rewards() 同源。"""
    root_pos = raw_env.robot.data.root_pos_w
    pallet_pos = raw_env.pallet.data.root_pos_w
    robot_yaw = _quat_to_yaw(raw_env.robot.data.root_quat_w)
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

    return {
        "y_err": y_err,
        "yaw_err_deg": yaw_err_deg,
        "insert_norm": insert_norm,
        "insert_depth": insert_depth,
        "lift_height": lift_height,
        "hold_counter": raw_env._hold_counter.clone(),
    }


# ==============================================================================
# 合并运行: Per-episode 统计 + 前 N_ROLLOUT 个 env 的 step-level rollout
# ==============================================================================
def run_combined(env_wrapped, raw_env, policy_nn, device, max_steps, n_rollout_envs):
    """单次循环同时收集 per-episode 统计 + step-level rollout。

    所有 N 个 env 收集 per-episode 统计（漏斗 + 分布 + 奖励）。
    前 n_rollout_envs 个 env 额外记录逐步状态（用于代表性 episode 分析）。
    """
    N = raw_env.num_envs
    R = min(n_rollout_envs, N)  # rollout tracking env 数
    print(f"\n[RUN] {N} envs, max_steps={max_steps}, rollout_envs={R}", flush=True)

    # -- Per-env episode buffers (all envs) --
    ep_total_reward = torch.zeros(N, device=device)
    ep_steps = torch.zeros(N, dtype=torch.int32, device=device)
    ep_max_insert = torch.zeros(N, device=device)
    ep_max_hold = torch.zeros(N, device=device)
    ep_success = torch.zeros(N, dtype=torch.bool, device=device)

    # Per-env "best achieved" for funnel
    ep_ever_near = torch.zeros(N, dtype=torch.bool, device=device)
    ep_ever_deep = torch.zeros(N, dtype=torch.bool, device=device)
    ep_ever_yaw_ok = torch.zeros(N, dtype=torch.bool, device=device)
    ep_ever_lat_ok = torch.zeros(N, dtype=torch.bool, device=device)
    ep_ever_both_ok = torch.zeros(N, dtype=torch.bool, device=device)
    ep_ever_grace = torch.zeros(N, dtype=torch.bool, device=device)
    ep_ever_half_hold = torch.zeros(N, dtype=torch.bool, device=device)

    # Per-env near-field accumulation
    ep_yaw_near_sum = torch.zeros(N, device=device)
    ep_yaw_near_count = torch.zeros(N, dtype=torch.int32, device=device)
    ep_lat_near_sum = torch.zeros(N, device=device)
    ep_near_steps = torch.zeros(N, dtype=torch.int32, device=device)

    completed = []  # per-episode stats for ALL envs

    # -- Step-level rollout buffers (first R envs only) --
    ep_logs = {i: [] for i in range(R)}
    rollouts = []  # list of (steps_data, summary)

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

        # -- Accumulate per-episode stats (all envs) --
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

        ep_ever_near |= near_mask
        ep_ever_deep |= deep_mask
        ep_ever_yaw_ok |= yaw_ok_mask
        ep_ever_lat_ok |= lat_ok_mask
        ep_ever_both_ok |= both_ok_mask
        ep_ever_grace |= (hold_counter > 0)
        ep_ever_half_hold |= (hold_counter >= 5)

        ep_yaw_near_sum += torch.where(near_mask, yaw_err_deg, torch.zeros_like(yaw_err_deg))
        ep_yaw_near_count += near_mask.int()
        ep_lat_near_sum += torch.where(near_mask, y_err, torch.zeros_like(y_err))
        ep_near_steps += near_mask.int()

        # -- Step-level logging (first R envs) --
        for i in range(R):
            ep_logs[i].append({
                "step": len(ep_logs[i]),
                "yaw_err_deg": yaw_err_deg[i].item(),
                "y_err": y_err[i].item(),
                "insert_norm": insert_norm[i].item(),
                "lift_height": geo["lift_height"][i].item(),
                "hold_counter": hold_counter[i].item(),
                "drive": raw_env.actions[i, 0].item(),
                "steer": raw_env.actions[i, 1].item(),
                "lift_act": raw_env.actions[i, 2].item(),
                "reward": rew_1d[i].item(),
            })

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

            completed.append({
                "total_reward": ep_total_reward[i].item(),
                "steps": ep_steps[i].item(),
                "max_insert_norm": ep_max_insert[i].item(),
                "max_hold_counter": ep_max_hold[i].item(),
                "success": ep_success[i].item(),
                "ever_near": ep_ever_near[i].item(),
                "ever_deep": ep_ever_deep[i].item(),
                "ever_yaw_ok": ep_ever_yaw_ok[i].item(),
                "ever_lat_ok": ep_ever_lat_ok[i].item(),
                "ever_both_ok": ep_ever_both_ok[i].item(),
                "ever_grace": ep_ever_grace[i].item(),
                "ever_half_hold": ep_ever_half_hold[i].item(),
                "yaw_near_mean": yaw_near_mean,
                "lat_near_mean": lat_near_mean,
                "near_steps": ep_near_steps[i].item(),
                "near_frac": ep_near_steps[i].item() / max(ep_steps[i].item(), 1),
            })

            # Save step-level rollout for first R envs
            if i < R and len(ep_logs[i]) > 0:
                summary = {
                    "max_insert_norm": ep_max_insert[i].item(),
                    "max_hold_counter": ep_max_hold[i].item(),
                    "success": ep_success[i].item(),
                    "steps": len(ep_logs[i]),
                }
                rollouts.append((list(ep_logs[i]), summary))
                ep_logs[i] = []

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
            ep_yaw_near_sum[i] = 0
            ep_yaw_near_count[i] = 0
            ep_lat_near_sum[i] = 0
            ep_near_steps[i] = 0

        if total_steps % 200 == 0:
            print(f"  step {total_steps}/{max_steps}, episodes={len(completed)}, rollouts={len(rollouts)}", flush=True)

    print(f"[RUN] 完成: {len(completed)} episodes, {len(rollouts)} rollouts collected", flush=True)
    return completed, rollouts


# ==============================================================================
# 分析与输出
# ==============================================================================
def analyze_and_output(episodes, rollouts, output_dir):
    """分析 Phase 1/2 数据并输出 CSV + 控制台报告。"""
    os.makedirs(output_dir, exist_ok=True)
    n_ep = len(episodes)
    print(f"\n{'='*70}")
    print(f"S1.0P 诊断分析报告 ({n_ep} episodes)")
    print(f"{'='*70}")

    # ---- 漏斗分析 ----
    funnel = {
        "P(insert>0.1) [near]": sum(1 for e in episodes if e["ever_near"]) / n_ep,
        "P(insert>0.3) [deep]": sum(1 for e in episodes if e["ever_deep"]) / n_ep,
        "P(near AND yaw<5)": sum(1 for e in episodes if e["ever_yaw_ok"]) / n_ep,
        "P(near AND y<0.15)": sum(1 for e in episodes if e["ever_lat_ok"]) / n_ep,
        "P(near AND yaw<5 AND y<0.15)": sum(1 for e in episodes if e["ever_both_ok"]) / n_ep,
        "P(hold_counter>0) [ever grace]": sum(1 for e in episodes if e["ever_grace"]) / n_ep,
        "P(hold_counter>=5) [half hold]": sum(1 for e in episodes if e["ever_half_hold"]) / n_ep,
        "P(success)": sum(1 for e in episodes if e["success"]) / n_ep,
    }

    print("\n--- 漏斗分析 (per-episode best) ---")
    for k, v in funnel.items():
        bar = "#" * int(v * 50)
        print(f"  {k:42s}  {v*100:6.2f}%  |{bar}")

    # ---- 分布统计 ----
    print("\n--- 分布统计 ---")

    # hold_counter_max distribution
    hold_maxes = [e["max_hold_counter"] for e in episodes]
    hold_arr = np.array(hold_maxes)

    buckets = [(0, 0), (0.1, 1), (1, 3), (3, 5), (5, 7), (7, 9), (9, 9.99), (10, 999)]
    bucket_labels = ["= 0", "0-1", "1-3", "3-5", "5-7", "7-9", "9-10", ">= 10"]
    print("\n  hold_counter_max 分布:")
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

    # yaw/lateral in near-field (per-episode mean)
    yaw_near_vals = [e["yaw_near_mean"] for e in episodes if e["yaw_near_mean"] > 0]
    lat_near_vals = [e["lat_near_mean"] for e in episodes if e["lat_near_mean"] > 0]

    if yaw_near_vals:
        yaw_arr = np.array(yaw_near_vals)
        print(f"\n  yaw_deg (near-field, per-episode mean), N={len(yaw_arr)}:")
        for p in [50, 75, 90, 95]:
            print(f"    p{p}: {np.percentile(yaw_arr, p):.2f}°")

    if lat_near_vals:
        lat_arr = np.array(lat_near_vals)
        print(f"\n  lateral (near-field, per-episode mean), N={len(lat_arr)}:")
        for p in [50, 75, 90, 95]:
            print(f"    p{p}: {np.percentile(lat_arr, p):.3f} m")

    # max insert_norm distribution
    ins_arr = np.array([e["max_insert_norm"] for e in episodes])
    print(f"\n  max_insert_norm 分位数:")
    for p in [25, 50, 75, 90, 95]:
        print(f"    p{p}: {np.percentile(ins_arr, p):.3f}")

    # ---- 奖励分解（近场 vs 远场步数比）----
    near_fracs = [e["near_frac"] for e in episodes]
    total_rews = [e["total_reward"] for e in episodes]
    print(f"\n--- 奖励统计 ---")
    print(f"  total_reward per episode:")
    rew_arr = np.array(total_rews)
    print(f"    mean={rew_arr.mean():.2f}, std={rew_arr.std():.2f}")
    for p in [25, 50, 75, 90]:
        print(f"    p{p}: {np.percentile(rew_arr, p):.2f}")

    nf_arr = np.array(near_fracs)
    print(f"\n  near_field 步数占比:")
    print(f"    mean={nf_arr.mean():.3f}, p50={np.percentile(nf_arr, 50):.3f}, p90={np.percentile(nf_arr, 90):.3f}")

    # Episode length
    ep_lens = np.array([e["steps"] for e in episodes])
    print(f"\n  episode_length:")
    print(f"    mean={ep_lens.mean():.0f}, p50={np.percentile(ep_lens, 50):.0f}, p90={np.percentile(ep_lens, 90):.0f}")

    # ---- CSV 输出 ----
    csv_path = os.path.join(output_dir, "s1.0p_episode_stats.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(episodes[0].keys()))
        writer.writeheader()
        writer.writerows(episodes)
    print(f"\n[CSV] Per-episode 数据已写入: {csv_path}")

    # ---- Rollout 分析 ----
    if rollouts:
        print(f"\n--- 代表性 Rollout 分析 ({len(rollouts)} episodes) ---")

        # Classify episodes
        type_a = []  # deep insert but no success
        type_b = []  # grace zone but hold failed
        type_c = []  # success
        type_d = []  # highest hold (fallback)

        for steps_data, summary in rollouts:
            if summary["success"]:
                type_c.append((steps_data, summary))
            elif summary["max_hold_counter"] > 0 and summary["max_hold_counter"] < 10:
                type_b.append((steps_data, summary))
            elif summary["max_insert_norm"] > 0.3:
                type_a.append((steps_data, summary))

        # Sort type_b by hold_counter desc
        type_b.sort(key=lambda x: x[1]["max_hold_counter"], reverse=True)
        type_a.sort(key=lambda x: x[1]["max_insert_norm"], reverse=True)

        # Select best examples
        selected = {}
        if type_c:
            selected["C_success"] = type_c[0]
            print(f"  类型 C (成功): 找到 {len(type_c)} 个")
        if type_b:
            selected["B_grace_fail"] = type_b[0]
            print(f"  类型 B (grace zone + hold 失败): 找到 {len(type_b)} 个, 最高 hold={type_b[0][1]['max_hold_counter']:.1f}")
        if type_a:
            selected["A_deep_no_success"] = type_a[0]
            print(f"  类型 A (深插无成功): 找到 {len(type_a)} 个, 最深 insert={type_a[0][1]['max_insert_norm']:.3f}")

        if not type_c and not type_b:
            # Fallback: highest hold overall
            all_sorted = sorted(rollouts, key=lambda x: x[1]["max_hold_counter"], reverse=True)
            selected["D_best_hold"] = all_sorted[0]
            print(f"  类型 D (最高 hold fallback): hold={all_sorted[0][1]['max_hold_counter']:.1f}")

        # Write rollout CSVs
        for label, (steps_data, summary) in selected.items():
            csv_path_r = os.path.join(output_dir, f"s1.0p_rollout_{label}.csv")
            with open(csv_path_r, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(steps_data[0].keys()))
                writer.writeheader()
                writer.writerows(steps_data)
            print(f"  [CSV] {label}: {csv_path_r} ({len(steps_data)} steps, "
                  f"max_insert={summary['max_insert_norm']:.3f}, "
                  f"max_hold={summary['max_hold_counter']:.1f}, "
                  f"success={summary['success']})")

        # Print key moments from best rollout
        for label, (steps_data, summary) in selected.items():
            print(f"\n  === {label} 关键时刻 ===")
            # Find step with max insert_norm
            max_ins_step = max(steps_data, key=lambda s: s["insert_norm"])
            # Find step with max hold_counter
            max_hold_step = max(steps_data, key=lambda s: s["hold_counter"])
            # First step entering near field
            near_steps = [s for s in steps_data if s["insert_norm"] > 0.1]
            first_near = near_steps[0] if near_steps else None

            if first_near:
                print(f"    进入近场 (step {first_near['step']}): "
                      f"yaw={first_near['yaw_err_deg']:.1f}°, y={first_near['y_err']:.3f}m, "
                      f"insert={first_near['insert_norm']:.3f}")
            print(f"    最深插入 (step {max_ins_step['step']}): "
                  f"yaw={max_ins_step['yaw_err_deg']:.1f}°, y={max_ins_step['y_err']:.3f}m, "
                  f"insert={max_ins_step['insert_norm']:.3f}, hold={max_ins_step['hold_counter']:.1f}")
            print(f"    最大 hold (step {max_hold_step['step']}): "
                  f"yaw={max_hold_step['yaw_err_deg']:.1f}°, y={max_hold_step['y_err']:.3f}m, "
                  f"insert={max_hold_step['insert_norm']:.3f}, hold={max_hold_step['hold_counter']:.1f}")

    return funnel


# ==============================================================================
# Main
# ==============================================================================
@hydra_task_config(args.task, args.agent)
def main(env_cfg, agent_cfg):
    ckpt = os.path.join("logs", "rsl_rl", "forklift_pallet_insert_lift",
                        "2026-02-12_11-14-55", "model_3296.pt")
    output_dir = os.path.abspath(args.output_dir)

    env_cfg.scene.num_envs = args.num_envs
    env_cfg.seed = args.seed
    env_cfg.sim.device = "cuda:0"

    env = gym.make(args.task, cfg=env_cfg)
    env_wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    raw_env = env.unwrapped
    device = raw_env.device

    print(f"[INFO] Loading checkpoint: {ckpt}", flush=True)
    runner = OnPolicyRunner(env_wrapped, agent_cfg.to_dict(), log_dir=None, device="cuda:0")
    runner.load(ckpt)
    try:
        policy_nn = runner.alg.policy
    except AttributeError:
        policy_nn = runner.alg.actor_critic

    # 单次合并运行
    episodes, rollouts = run_combined(
        env_wrapped, raw_env, policy_nn, device,
        args.max_steps, args.rollout_envs,
    )
    env.close()

    # 分析与输出
    analyze_and_output(episodes, rollouts, output_dir)


if __name__ == "__main__":
    main()
    simulation_app.close()
