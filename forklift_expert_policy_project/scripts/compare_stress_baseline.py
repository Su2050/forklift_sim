#!/usr/bin/env python3
"""
Automated baseline comparison between stress test runs.
Parses per-episode logs and computes aggregate metrics, then outputs a PASS/FAIL verdict.

Usage:
    python3 compare_stress_baseline.py <baseline_dir> <candidate_dir> [--seeds SEED1,SEED2,...]

Example:
    python3 compare_stress_baseline.py logs/stress_test/large_v3 logs/stress_test/v4_smoke --seeds 1337,6789,99999
"""
import argparse
import glob
import os
import re
import sys
from dataclasses import dataclass, field
from typing import List, Dict, Optional


@dataclass
class EpisodeStats:
    steps: int = 0
    reason: str = ""
    max_ins: float = 0.0
    retreat_steps: int = 0
    had_retreat: bool = False


def parse_log_file(path: str, max_episodes: Optional[int] = None) -> List[EpisodeStats]:
    """Parse a seed log file and extract per-episode statistics.

    Parses the summary lines like:
      [EP   0] 1079 steps  reason=truncated  ...  max_ins=0.422  ...  stages={'docking': 998, 'retreat': 80}
    """
    episodes = []

    with open(path, "r") as f:
        for line in f:
            line = line.strip()

            # Match episode summary line
            m_ep = re.match(r"\[EP\s+\d+\]\s+(\d+)\s+steps\s+reason=(\w+)", line)
            if not m_ep:
                continue

            ep = EpisodeStats()
            ep.steps = int(m_ep.group(1))
            ep.reason = m_ep.group(2)

            # Extract max_ins
            m_ins = re.search(r"max_ins=([\d.]+)", line)
            if m_ins:
                ep.max_ins = float(m_ins.group(1))

            # Extract retreat steps from stages dict
            m_stages = re.search(r"stages=\{([^}]+)\}", line)
            if m_stages:
                stages_str = m_stages.group(1)
                m_retreat = re.search(r"'retreat':\s*(\d+)", stages_str)
                if m_retreat:
                    ep.retreat_steps = int(m_retreat.group(1))
                    ep.had_retreat = True

            episodes.append(ep)
            if max_episodes and len(episodes) >= max_episodes:
                break

    return episodes


@dataclass
class AggMetrics:
    total_episodes: int = 0
    avg_retreat_steps: float = 0.0
    retreat_trigger_rate: float = 0.0
    avg_max_ins: float = 0.0
    zero_ins_count: int = 0


def compute_metrics(log_dir: str, seeds: List[int], max_ep_per_seed: Optional[int] = None) -> AggMetrics:
    """Compute aggregate metrics from all seed logs in a directory."""
    all_episodes = []
    for seed in seeds:
        pattern = os.path.join(log_dir, f"seed_{seed}.log")
        files = glob.glob(pattern)
        if not files:
            # Try alternative naming
            pattern2 = os.path.join(log_dir, f"*seed{seed}*.log")
            files = glob.glob(pattern2)
        for f in files:
            eps = parse_log_file(f, max_episodes=max_ep_per_seed)
            all_episodes.extend(eps)

    if not all_episodes:
        return AggMetrics()

    m = AggMetrics()
    m.total_episodes = len(all_episodes)
    m.avg_retreat_steps = sum(e.retreat_steps for e in all_episodes) / len(all_episodes)
    m.retreat_trigger_rate = sum(1 for e in all_episodes if e.had_retreat) / len(all_episodes)
    m.avg_max_ins = sum(e.max_ins for e in all_episodes) / len(all_episodes)
    m.zero_ins_count = sum(1 for e in all_episodes if e.max_ins < 0.01)
    return m


def main():
    parser = argparse.ArgumentParser(description="Compare stress test baseline vs candidate")
    parser.add_argument("baseline_dir", help="Path to baseline log directory (e.g. large_v3)")
    parser.add_argument("candidate_dir", help="Path to candidate log directory (e.g. v4_smoke)")
    parser.add_argument("--seeds", default="1337,6789,99999",
                        help="Comma-separated seeds to compare")
    parser.add_argument("--max_ep", type=int, default=None,
                        help="Max episodes per seed to compare (default: all)")
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    print(f"Seeds: {seeds}")
    print(f"Baseline: {args.baseline_dir}")
    print(f"Candidate: {args.candidate_dir}")
    print()

    base = compute_metrics(args.baseline_dir, seeds, args.max_ep)
    cand = compute_metrics(args.candidate_dir, seeds, args.max_ep)

    print(f"{'Metric':<30} {'Baseline':>12} {'Candidate':>12} {'Change':>12}")
    print("-" * 70)

    def fmt_change(old, new, lower_better=True):
        if old == 0:
            return "N/A"
        pct = (new - old) / old * 100
        arrow = "↓" if pct < 0 else "↑"
        good = (pct < 0) if lower_better else (pct > 0)
        tag = "GOOD" if good else "BAD"
        return f"{pct:+.1f}% {arrow} [{tag}]"

    print(f"{'total_episodes':<30} {base.total_episodes:>12} {cand.total_episodes:>12} {'':>12}")
    print(f"{'avg_retreat_steps':<30} {base.avg_retreat_steps:>12.1f} {cand.avg_retreat_steps:>12.1f} {fmt_change(base.avg_retreat_steps, cand.avg_retreat_steps, lower_better=True):>12}")
    print(f"{'retreat_trigger_rate':<30} {base.retreat_trigger_rate:>12.1%} {cand.retreat_trigger_rate:>12.1%} {fmt_change(base.retreat_trigger_rate, cand.retreat_trigger_rate, lower_better=True):>12}")
    print(f"{'avg_max_ins':<30} {base.avg_max_ins:>12.3f} {cand.avg_max_ins:>12.3f} {fmt_change(base.avg_max_ins, cand.avg_max_ins, lower_better=False):>12}")
    print(f"{'zero_ins_episodes':<30} {base.zero_ins_count:>12} {cand.zero_ins_count:>12} {fmt_change(base.zero_ins_count, cand.zero_ins_count, lower_better=True):>12}")
    print()

    # ---- Gate verdict ----
    failures = []

    # 1. avg_retreat_steps should decrease by 25%+
    if base.avg_retreat_steps > 0:
        retreat_change = (cand.avg_retreat_steps - base.avg_retreat_steps) / base.avg_retreat_steps
        if retreat_change > -0.25:
            failures.append(f"avg_retreat_steps did not drop 25%+ (change: {retreat_change:+.1%})")

    # 2. avg_max_ins should not decrease more than 5%
    if base.avg_max_ins > 0:
        ins_change = (cand.avg_max_ins - base.avg_max_ins) / base.avg_max_ins
        if ins_change < -0.05:
            failures.append(f"avg_max_ins dropped >5% (change: {ins_change:+.1%})")

    # 3. retreat_trigger_rate should not increase more than 10%
    if base.retreat_trigger_rate > 0:
        trigger_change = (cand.retreat_trigger_rate - base.retreat_trigger_rate) / base.retreat_trigger_rate
        if trigger_change > 0.10:
            failures.append(f"retreat_trigger_rate increased >10% (change: {trigger_change:+.1%})")

    # 4. zero_ins_count should not increase by more than 1 (small sample noise)
    if cand.zero_ins_count > base.zero_ins_count + 1:
        failures.append(f"zero_ins episodes increased by >1: {base.zero_ins_count} -> {cand.zero_ins_count}")

    if failures:
        print("=" * 70)
        print("  VERDICT: FAIL")
        print("=" * 70)
        for f in failures:
            print(f"  - {f}")
        return 1
    else:
        print("=" * 70)
        print("  VERDICT: PASS")
        print("=" * 70)
        return 0


if __name__ == "__main__":
    sys.exit(main())
