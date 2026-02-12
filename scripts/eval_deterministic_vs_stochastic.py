"""S1.0P Phase V2: Deterministic vs Stochastic 评估。

加载 s1.0o 最佳模型 (model_3296)，分别用确定性策略和随机策略跑 N 个 episode，
比较 success rate / yaw / insert_norm / hold_counter，定位转化率瓶颈。

产出物:
  - 控制台输出: deterministic vs stochastic 对比表
  - 差值百分比

Usage:
    cd IsaacLab && isaaclab.sh -p ../scripts/eval_deterministic_vs_stochastic.py \
        --task Isaac-Forklift-PalletInsertLift-Direct-v0 \
        --checkpoint /path/to/model_3296.pt \
        --headless --num_envs 128 --episodes 5
"""
from __future__ import annotations

import argparse
import os
import sys

parser = argparse.ArgumentParser(description="V2: Deterministic vs Stochastic evaluation")
parser.add_argument("--task", type=str, default="Isaac-Forklift-PalletInsertLift-Direct-v0")
parser.add_argument("--num_envs", type=int, default=128)
parser.add_argument("--checkpoint", type=str, required=True,
                    help="Path to model checkpoint (e.g. model_3296.pt)")
parser.add_argument("--episodes", type=int, default=5,
                    help="Number of episode rollouts per mode (each rollout covers num_envs parallel envs)")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point",
                    help="Agent config entry point")

from isaaclab.app import AppLauncher
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# ---- IsaacLab imports (after AppLauncher) ----
import torch
import gymnasium as gym

from rsl_rl.runners import OnPolicyRunner
from isaaclab.envs import DirectRLEnvCfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config


print(f"\n{'='*70}")
print(f"S1.0P Phase V2: Deterministic vs Stochastic 评估")
print(f"{'='*70}")
print(f"Checkpoint: {args.checkpoint}")
print(f"Num envs: {args.num_envs}, Episodes: {args.episodes}")
print(f"{'='*70}\n")


@hydra_task_config(args.task, args.agent)
def main(env_cfg, agent_cfg):
    """Run deterministic and stochastic evaluation."""
    # Setup
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.seed = args.seed
    env_cfg.sim.device = "cuda:0"

    # Create env
    env = gym.make(args.task, cfg=env_cfg)
    env_wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    raw_env = env.unwrapped
    device = raw_env.device

    # Load model
    print(f"[INFO] Loading model from: {args.checkpoint}")
    runner = OnPolicyRunner(env_wrapped, agent_cfg.to_dict(), log_dir=None, device="cuda:0")
    runner.load(args.checkpoint)

    # Get policy
    policy = runner.get_inference_policy(device=device)

    # Get the underlying actor for stochastic mode
    try:
        policy_nn = runner.alg.policy
    except AttributeError:
        policy_nn = runner.alg.actor_critic

    def run_episodes(mode: str, n_episodes: int):
        """Run n_episodes rollouts and collect metrics.

        mode: 'deterministic' or 'stochastic'
        Each rollout runs until all envs terminate or timeout once.
        """
        is_stochastic = (mode == "stochastic")

        all_metrics = {
            "success_count": 0,
            "total_envs": 0,
            "yaw_deg_sum": 0.0,
            "yaw_deg_near_sum": 0.0,
            "yaw_deg_near_count": 0,
            "insert_norm_sum": 0.0,
            "hold_counter_max_sum": 0.0,
            "hold_counter_mean_sum": 0.0,
            "grace_zone_frac_sum": 0.0,
            "episode_length_sum": 0,
            "n_steps": 0,
            # Per-env success tracking
            "deep_insert_frac_sum": 0.0,
            "yaw_deg_deep_sum": 0.0,
            "yaw_deg_deep_count": 0,
        }

        for ep in range(n_episodes):
            obs = env_wrapped.get_observations()
            done_mask = torch.zeros(args.num_envs, dtype=torch.bool, device=device)
            ep_success = torch.zeros(args.num_envs, dtype=torch.bool, device=device)
            ep_max_insert = torch.zeros(args.num_envs, device=device)
            ep_max_hold = torch.zeros(args.num_envs, device=device)
            step_count = 0
            max_steps = int(raw_env.max_episode_length) + 10

            while not done_mask.all() and step_count < max_steps:
                with torch.inference_mode():
                    if is_stochastic:
                        # act() → sample from Normal(mean, std)
                        actions = policy_nn.act(obs)
                    else:
                        # act_inference() → deterministic mean
                        actions = policy_nn.act_inference(obs)

                    obs, _, dones, infos = env_wrapped.step(actions)

                step_count += 1

                # Extract logs from raw env
                log = raw_env.extras.get("log", {})

                # Track per-step metrics (mean across active envs)
                active = ~done_mask
                n_active = active.sum().item()
                if n_active > 0:
                    if "err/yaw_deg_mean" in log:
                        all_metrics["yaw_deg_sum"] += log["err/yaw_deg_mean"].item()
                    if "err/insert_norm_mean" in log:
                        all_metrics["insert_norm_sum"] += log["err/insert_norm_mean"].item()
                    if "phase/hold_counter_max" in log:
                        all_metrics["hold_counter_max_sum"] += log["phase/hold_counter_max"].item()
                    if "phase/hold_counter_mean" in log:
                        all_metrics["hold_counter_mean_sum"] += log["phase/hold_counter_mean"].item()
                    if "phase/grace_zone_frac" in log:
                        all_metrics["grace_zone_frac_sum"] += log["phase/grace_zone_frac"].item()
                    if "err/yaw_deg_near_success" in log:
                        all_metrics["yaw_deg_near_sum"] += log["err/yaw_deg_near_success"].item()
                        all_metrics["yaw_deg_near_count"] += 1
                    if "diag/deep_insert_frac" in log:
                        all_metrics["deep_insert_frac_sum"] += log["diag/deep_insert_frac"].item()
                    if "err/yaw_deg_deep_mean" in log:
                        all_metrics["yaw_deg_deep_sum"] += log["err/yaw_deg_deep_mean"].item()
                        all_metrics["yaw_deg_deep_count"] += 1

                    all_metrics["n_steps"] += 1

                # Track success
                if "phase/frac_success" in log:
                    success_now = raw_env._hold_counter >= raw_env._hold_steps
                    ep_success = ep_success | success_now

                # Detect episode termination
                if isinstance(dones, torch.Tensor):
                    newly_done = dones.bool() & ~done_mask
                else:
                    newly_done = torch.tensor(dones, dtype=torch.bool, device=device) & ~done_mask
                done_mask = done_mask | newly_done

            # Aggregate per-episode results
            all_metrics["success_count"] += ep_success.sum().item()
            all_metrics["total_envs"] += args.num_envs

            print(f"  [{mode}] Episode {ep+1}/{n_episodes}: "
                  f"success={ep_success.sum().item()}/{args.num_envs} "
                  f"({ep_success.float().mean().item()*100:.1f}%), "
                  f"steps={step_count}")

        # Compute averages
        n = all_metrics["n_steps"] if all_metrics["n_steps"] > 0 else 1
        results = {
            "mode": mode,
            "success_rate": all_metrics["success_count"] / max(all_metrics["total_envs"], 1),
            "yaw_deg_mean": all_metrics["yaw_deg_sum"] / n,
            "yaw_deg_near_mean": all_metrics["yaw_deg_near_sum"] / max(all_metrics["yaw_deg_near_count"], 1),
            "insert_norm_mean": all_metrics["insert_norm_sum"] / n,
            "hold_counter_max": all_metrics["hold_counter_max_sum"] / n,
            "hold_counter_mean": all_metrics["hold_counter_mean_sum"] / n,
            "grace_zone_frac": all_metrics["grace_zone_frac_sum"] / n,
            "deep_insert_frac": all_metrics["deep_insert_frac_sum"] / n,
            "yaw_deg_deep_mean": all_metrics["yaw_deg_deep_sum"] / max(all_metrics["yaw_deg_deep_count"], 1),
            "total_envs": all_metrics["total_envs"],
            "success_count": all_metrics["success_count"],
        }
        return results

    # ---- Run both modes ----
    print(">>> 运行确定性评估 (Deterministic)...")
    det_results = run_episodes("deterministic", args.episodes)

    # Reset env for fair comparison
    obs = env_wrapped.get_observations()
    torch.manual_seed(args.seed)

    print("\n>>> 运行随机策略评估 (Stochastic)...")
    sto_results = run_episodes("stochastic", args.episodes)

    # ---- Print comparison ----
    print(f"\n{'='*70}")
    print(f"对比结果: Deterministic vs Stochastic")
    print(f"{'='*70}")

    def fmt_pct(val):
        return f"{val*100:.2f}%"

    def diff_pct(det_val, sto_val):
        if abs(sto_val) < 1e-10:
            return "N/A"
        diff = (det_val - sto_val) / abs(sto_val) * 100
        sign = "+" if diff > 0 else ""
        return f"{sign}{diff:.1f}%"

    metrics_to_compare = [
        ("success_rate",     "成功率",             fmt_pct),
        ("yaw_deg_mean",     "yaw_deg_mean (°)",   lambda v: f"{v:.3f}"),
        ("yaw_deg_near_mean","yaw_near_success (°)",lambda v: f"{v:.3f}"),
        ("yaw_deg_deep_mean","yaw_deep_mean (°)",   lambda v: f"{v:.3f}"),
        ("insert_norm_mean", "insert_norm_mean",    lambda v: f"{v:.4f}"),
        ("hold_counter_max", "hold_counter_max",    lambda v: f"{v:.2f}"),
        ("hold_counter_mean","hold_counter_mean",   lambda v: f"{v:.4f}"),
        ("grace_zone_frac",  "grace_zone_frac",     lambda v: f"{v:.4f}"),
        ("deep_insert_frac", "deep_insert_frac",    lambda v: f"{v:.4f}"),
    ]

    print(f"\n{'指标':^25} | {'Deterministic':^15} | {'Stochastic':^15} | {'差值%':^10}")
    print(f"{'-'*25}-+-{'-'*15}-+-{'-'*15}-+-{'-'*10}")

    for key, label, fmt in metrics_to_compare:
        d_val = det_results.get(key, 0)
        s_val = sto_results.get(key, 0)
        d_str = fmt(d_val)
        s_str = fmt(s_val)
        diff = diff_pct(d_val, s_val)
        print(f"{label:^25} | {d_str:^15} | {s_str:^15} | {diff:^10}")

    # ---- Diagnosis ----
    print(f"\n{'='*70}")
    print(f"诊断结论")
    print(f"{'='*70}")

    sr_det = det_results["success_rate"]
    sr_sto = sto_results["success_rate"]
    yaw_det = det_results["yaw_deg_mean"]
    yaw_sto = sto_results["yaw_deg_mean"]

    if sr_det > sr_sto * 1.3:
        print("  [瓶颈]: 动作噪声是转化率的显著负面因素")
        print(f"    确定性成功率 ({sr_det*100:.1f}%) >> 随机成功率 ({sr_sto*100:.1f}%)")
        print("    => 建议: 优先改善 hold 稳定性 (S1/S2), 减少抖动出 hold 区间")
        if yaw_det < yaw_sto * 0.8:
            print(f"    yaw 精度也受噪声影响: det={yaw_det:.2f}° vs sto={yaw_sto:.2f}°")
    elif sr_det < sr_sto * 1.1 and sr_det < 0.3:
        print("  [瓶颈]: 成功率均低，且确定性策略无明显优势")
        print("    => 问题不在噪声，而在策略本身未学到精确对齐")
        print("    => 建议: 优先加强精度信号 (P 组实验)")
    else:
        print(f"  两种模式成功率接近: det={sr_det*100:.1f}%, sto={sr_sto*100:.1f}%")
        if sr_det > 0.5:
            print("    => 策略稳健，噪声不是主要瓶颈")
        else:
            print("    => 两种模式都需要改善，可能是奖励函数本身的局限")

    # Hold counter analysis
    hc_det = det_results["hold_counter_mean"]
    hc_sto = sto_results["hold_counter_mean"]
    if hc_det > hc_sto * 1.5 and hc_det > 0.01:
        print(f"\n  [hold 稳定性]: 确定性 hold_mean ({hc_det:.4f}) >> 随机 ({hc_sto:.4f})")
        print("    => 噪声导致 hold 计数器频繁被重置，印证 'S 组优先' 方向")

    print(f"\n{'='*70}")

    # Cleanup
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
