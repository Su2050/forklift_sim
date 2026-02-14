#!/usr/bin/env python3
"""S1.0Q Batch-3 统计分析：Bootstrap CI + 失败模式拆分。

纯 Python 实现（无第三方依赖）。

从 per-episode CSV 精确计算：
1. Bootstrap 95% CI（success_rate, timeout_frac, fail_step_share, lateral_near_p90）
2. 失败模式拆分 P(timeout AND ever_dead_zone) / P(timeout AND NOT ever_dead_zone)

输出 JSON 供复盘文档引用。
"""
import csv
import json
import math
import os
import random

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "s1.0q_eval_batch3")
OUTPUT_JSON = os.path.join(DATA_DIR, "batch3_stats_analysis.json")

EXPERIMENTS = [
    ("A1_B1a_baseline", "s1.0q_A1_B1a_baseline_episodes.csv"),
    ("B3_C1_lat_fine",  "s1.0q_B3_C1_lat_fine_episodes.csv"),
    ("B3_B1b_gate",     "s1.0q_B3_B1b_gate_episodes.csv"),
    ("B3_floor015",     "s1.0q_B3_floor015_episodes.csv"),
    ("B3_stuck_det",    "s1.0q_B3_stuck_det_episodes.csv"),
]

N_BOOTSTRAP = 2000
SEED = 42


def load_csv(path):
    """加载 CSV 为 list of dicts，数值自动转换。"""
    rows = []
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            parsed = {}
            for k, v in row.items():
                try:
                    parsed[k] = float(v)
                except (ValueError, TypeError):
                    parsed[k] = v
            rows.append(parsed)
    return rows


def percentile(sorted_vals, p):
    """计算已排序数组的百分位数（线性插值）。"""
    n = len(sorted_vals)
    if n == 0:
        return 0.0
    k = (p / 100.0) * (n - 1)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_vals[int(k)]
    d = k - f
    return sorted_vals[int(f)] * (1 - d) + sorted_vals[int(c)] * d


def bootstrap_ci(values, stat_fn, n_boot=N_BOOTSTRAP, ci=0.95):
    """Bootstrap 置信区间（纯 Python）。

    Args:
        values: list of numbers
        stat_fn: 统计函数 (list -> scalar)
        n_boot: bootstrap 次数
        ci: 置信水平
    Returns:
        (point_estimate, ci_low, ci_high)
    """
    rng = random.Random(SEED)
    n = len(values)
    point = stat_fn(values)
    boot_stats = []
    for _ in range(n_boot):
        sample = [values[rng.randint(0, n - 1)] for _ in range(n)]
        boot_stats.append(stat_fn(sample))
    boot_stats.sort()
    alpha = (1 - ci) / 2
    ci_low = percentile(boot_stats, alpha * 100)
    ci_high = percentile(boot_stats, (1 - alpha) * 100)
    return float(point), float(ci_low), float(ci_high)


def compute_stats(rows):
    """对一组 episode 数据计算所有统计量。"""
    n = len(rows)

    success_vals = [r["success"] for r in rows]
    steps_vals = [r["steps"] for r in rows]
    ever_dz_vals = [r["ever_dead_zone"] for r in rows]
    lat_near_vals = [r["lat_near_mean"] for r in rows]
    ever_near_vals = [r["ever_near"] for r in rows]

    max_steps = int(max(steps_vals))
    timeout_vals = [1.0 if s == max_steps else 0.0 for s in steps_vals]

    # --- Bootstrap CI ---

    # 1. success_rate
    def sr_fn(vals):
        return sum(vals) / len(vals) * 100 if vals else 0
    sr_point, sr_lo, sr_hi = bootstrap_ci(success_vals, sr_fn)

    # 2. timeout_frac
    def tf_fn(vals):
        return sum(vals) / len(vals) * 100 if vals else 0
    tf_point, tf_lo, tf_hi = bootstrap_ci(timeout_vals, tf_fn)

    # 3. fail_step_share（对 episode 索引重采样）
    # 将每个 episode 打包为 (steps, success) tuple
    ep_tuples = list(zip(steps_vals, success_vals))

    def fss_fn(tuples):
        total = sum(t[0] for t in tuples)
        if total == 0:
            return 0.0
        fail_total = sum(t[0] for t in tuples if t[1] == 0)
        return fail_total / total * 100

    fss_point, fss_lo, fss_hi = bootstrap_ci(ep_tuples, fss_fn)

    # 4. lateral_near_p90（只对 ever_near==1 的 episode）
    lat_valid = [lat_near_vals[i] for i in range(n) if ever_near_vals[i] == 1]
    if lat_valid:
        def lp90_fn(vals):
            s = sorted(vals)
            return percentile(s, 90)
        lp90_point, lp90_lo, lp90_hi = bootstrap_ci(lat_valid, lp90_fn)
    else:
        lp90_point, lp90_lo, lp90_hi = 0, 0, 0

    # --- 失败模式拆分 ---
    n_timeout = sum(1 for t in timeout_vals if t == 1)
    timeout_and_dz = sum(1 for i in range(n)
                         if timeout_vals[i] == 1 and ever_dz_vals[i] == 1)
    timeout_and_no_dz = sum(1 for i in range(n)
                            if timeout_vals[i] == 1 and ever_dz_vals[i] == 0)

    pct_timeout_dz = timeout_and_dz / n * 100 if n > 0 else 0
    pct_timeout_no_dz = timeout_and_no_dz / n * 100 if n > 0 else 0
    frac_timeout_dz = timeout_and_dz / n_timeout * 100 if n_timeout > 0 else 0
    frac_timeout_no_dz = timeout_and_no_dz / n_timeout * 100 if n_timeout > 0 else 0

    # 失败 episode 统计
    failed_steps = [steps_vals[i] for i in range(n) if success_vals[i] == 0]
    n_failed = len(failed_steps)
    mean_fail_ep_len = sum(failed_steps) / n_failed if n_failed > 0 else 0

    return {
        "n_episodes": n,
        "n_timeout": n_timeout,
        "n_failed": n_failed,
        "max_steps": max_steps,
        "mean_fail_ep_len": round(mean_fail_ep_len),
        "bootstrap_ci": {
            "success_rate": {
                "point": round(sr_point, 2),
                "ci_lo": round(sr_lo, 2),
                "ci_hi": round(sr_hi, 2),
            },
            "timeout_frac": {
                "point": round(tf_point, 2),
                "ci_lo": round(tf_lo, 2),
                "ci_hi": round(tf_hi, 2),
            },
            "fail_step_share": {
                "point": round(fss_point, 2),
                "ci_lo": round(fss_lo, 2),
                "ci_hi": round(fss_hi, 2),
            },
            "lateral_near_p90": {
                "point": round(lp90_point, 3),
                "ci_lo": round(lp90_lo, 3),
                "ci_hi": round(lp90_hi, 3),
            },
        },
        "failure_mode_split": {
            "timeout_and_dz_count": timeout_and_dz,
            "timeout_and_no_dz_count": timeout_and_no_dz,
            "pct_timeout_dz_of_all": round(pct_timeout_dz, 2),
            "pct_timeout_no_dz_of_all": round(pct_timeout_no_dz, 2),
            "frac_timeout_with_dz": round(frac_timeout_dz, 1),
            "frac_timeout_without_dz": round(frac_timeout_no_dz, 1),
        },
    }


def main():
    results = {}
    for name, csv_file in EXPERIMENTS:
        path = os.path.join(DATA_DIR, csv_file)
        if not os.path.exists(path):
            print(f"[WARN] {path} not found, skipping {name}")
            continue
        print(f"[{name}] Loading {csv_file}...", end=" ")
        rows = load_csv(path)
        stats = compute_stats(rows)
        results[name] = stats
        ci = stats["bootstrap_ci"]
        fm = stats["failure_mode_split"]
        print(f"{stats['n_episodes']} episodes")
        print(f"  success_rate: {ci['success_rate']['point']:.2f}% "
              f"[{ci['success_rate']['ci_lo']:.2f}, {ci['success_rate']['ci_hi']:.2f}]")
        print(f"  timeout_frac: {ci['timeout_frac']['point']:.2f}% "
              f"[{ci['timeout_frac']['ci_lo']:.2f}, {ci['timeout_frac']['ci_hi']:.2f}]")
        print(f"  fail_step_share: {ci['fail_step_share']['point']:.2f}% "
              f"[{ci['fail_step_share']['ci_lo']:.2f}, {ci['fail_step_share']['ci_hi']:.2f}]")
        print(f"  lateral_near_p90: {ci['lateral_near_p90']['point']:.3f} "
              f"[{ci['lateral_near_p90']['ci_lo']:.3f}, {ci['lateral_near_p90']['ci_hi']:.3f}]")
        print(f"  timeout split: {fm['timeout_and_dz_count']} dz + "
              f"{fm['timeout_and_no_dz_count']} non-dz = {stats['n_timeout']} total "
              f"({fm['frac_timeout_with_dz']:.1f}% / {fm['frac_timeout_without_dz']:.1f}%)")
        print()

    # 保存 JSON
    with open(OUTPUT_JSON, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"[DONE] Results saved to {OUTPUT_JSON}")

    # --- 打印汇总对比表 ---
    print()
    print("=" * 110)
    print(f"{'Experiment':22s} {'Succ%':>12s} {'Timeout%':>14s} {'FailStep%':>14s} {'LatP90':>14s}")
    print("-" * 110)
    for name, _ in EXPERIMENTS:
        if name not in results:
            continue
        ci = results[name]["bootstrap_ci"]
        print(f"{name:22s} "
              f"{ci['success_rate']['point']:5.2f} [{ci['success_rate']['ci_lo']:5.2f},{ci['success_rate']['ci_hi']:5.2f}] "
              f"{ci['timeout_frac']['point']:5.2f} [{ci['timeout_frac']['ci_lo']:5.2f},{ci['timeout_frac']['ci_hi']:5.2f}] "
              f"{ci['fail_step_share']['point']:5.2f} [{ci['fail_step_share']['ci_lo']:5.2f},{ci['fail_step_share']['ci_hi']:5.2f}] "
              f"{ci['lateral_near_p90']['point']:5.3f} [{ci['lateral_near_p90']['ci_lo']:5.3f},{ci['lateral_near_p90']['ci_hi']:5.3f}]")
    print("=" * 110)

    # --- 打印失败模式拆分表 ---
    print()
    print("=" * 90)
    print(f"{'Experiment':22s} {'N_ep':>6s} {'N_timeout':>10s} {'DZ_timeout':>11s} {'NoDZ_timeout':>13s} {'DZ%':>6s} {'NoDZ%':>7s}")
    print("-" * 90)
    for name, _ in EXPERIMENTS:
        if name not in results:
            continue
        s = results[name]
        fm = s["failure_mode_split"]
        print(f"{name:22s} {s['n_episodes']:6d} {s['n_timeout']:10d} "
              f"{fm['timeout_and_dz_count']:11d} {fm['timeout_and_no_dz_count']:13d} "
              f"{fm['frac_timeout_with_dz']:5.1f}% {fm['frac_timeout_without_dz']:6.1f}%")
    print("=" * 90)


if __name__ == "__main__":
    main()
