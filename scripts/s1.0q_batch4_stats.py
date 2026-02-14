#!/usr/bin/env python3
"""S1.0Q Batch-4 统计分析：Bootstrap CI + 失败模式拆分 + 吞吐量指标 + Shadow multi-threshold trade-off。

纯 Python 实现（无第三方依赖）。

从 per-episode CSV 计算：
1. Bootstrap 95% CI（success_rate, timeout_frac, fail_step_share, lateral_near_p90, success_per_1e6_steps）
2. 失败模式拆分 P(timeout AND ever_dead_zone) / P(timeout AND NOT ever_dead_zone)
3. Shadow multi-threshold 分析（false_positive_rate 和 coverage vs 阈值）

输出 JSON 供复盘文档引用。
"""
import csv
import json
import math
import os
import random

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "s1.0q_eval_batch4")
OUTPUT_JSON = os.path.join(DATA_DIR, "batch4_stats_analysis.json")

EXPERIMENTS = [
    ("B4_baseline",        "s1.0q_B4_baseline_episodes.csv"),
    ("B4_SD60",            "s1.0q_B4_SD60_episodes.csv"),
    ("B4_SD80",            "s1.0q_B4_SD80_episodes.csv"),
    ("B4_SD100",           "s1.0q_B4_SD100_episodes.csv"),
    ("B4_SD80_penOnly",    "s1.0q_B4_SD80_penOnly_episodes.csv"),
    ("B4_SD80_doneOnly",   "s1.0q_B4_SD80_doneOnly_episodes.csv"),
]

# Shadow baseline 用同一个 baseline eval 的 CSV（含 max_stuck_counter 列）
SHADOW_CSV = "s1.0q_B4_baseline_episodes.csv"

SHADOW_THRESHOLDS = [30, 40, 50, 60, 70, 80, 90, 100]

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
    """Bootstrap 置信区间（纯 Python）。"""
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

    # 3. fail_step_share
    ep_tuples = list(zip(steps_vals, success_vals))

    def fss_fn(tuples):
        total = sum(t[0] for t in tuples)
        if total == 0:
            return 0.0
        fail_total = sum(t[0] for t in tuples if t[1] == 0)
        return fail_total / total * 100

    fss_point, fss_lo, fss_hi = bootstrap_ci(ep_tuples, fss_fn)

    # 4. lateral_near_p90
    lat_valid = [lat_near_vals[i] for i in range(n) if ever_near_vals[i] == 1]
    if lat_valid:
        def lp90_fn(vals):
            s = sorted(vals)
            return percentile(s, 90)
        lp90_point, lp90_lo, lp90_hi = bootstrap_ci(lat_valid, lp90_fn)
    else:
        lp90_point, lp90_lo, lp90_hi = 0, 0, 0

    # 5. success_per_1e6_steps (Batch-4 新增)
    def spm_fn(tuples):
        total = sum(t[0] for t in tuples)
        if total == 0:
            return 0.0
        n_success = sum(1 for t in tuples if t[1] == 1)
        return n_success / total * 1e6

    spm_point, spm_lo, spm_hi = bootstrap_ci(ep_tuples, spm_fn)

    # 6. timeout_per_1e6_steps (Batch-4 新增)
    ep_triples = list(zip(steps_vals, success_vals, timeout_vals))

    def tpm_fn(triples):
        total = sum(t[0] for t in triples)
        if total == 0:
            return 0.0
        n_timeout = sum(1 for t in triples if t[2] == 1)
        return n_timeout / total * 1e6

    tpm_point, tpm_lo, tpm_hi = bootstrap_ci(ep_triples, tpm_fn)

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
            "success_per_1e6_steps": {
                "point": round(spm_point, 1),
                "ci_lo": round(spm_lo, 1),
                "ci_hi": round(spm_hi, 1),
            },
            "timeout_per_1e6_steps": {
                "point": round(tpm_point, 1),
                "ci_lo": round(tpm_lo, 1),
                "ci_hi": round(tpm_hi, 1),
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


def shadow_multi_threshold(rows, thresholds):
    """从 baseline eval CSV 的 max_stuck_counter 列做多阈值 trade-off 分析。

    Returns:
        list of dicts, one per threshold, with:
        - threshold, false_positive_rate, false_positive_count,
          coverage, coverage_count, n_dz_timeout, n_episodes
    """
    n = len(rows)
    if n == 0 or "max_stuck_counter" not in rows[0]:
        return []

    max_steps = int(max(r["steps"] for r in rows))
    results = []
    for T in thresholds:
        would_trigger = 0
        would_trigger_and_success = 0
        n_dz_timeout = 0
        would_trigger_and_dz_timeout = 0

        for r in rows:
            counter = int(r["max_stuck_counter"])
            is_success = r["success"] == 1
            is_timeout = r["steps"] == max_steps and not is_success
            is_dz = r["ever_dead_zone"] == 1

            triggered = counter >= T
            if triggered:
                would_trigger += 1
                if is_success:
                    would_trigger_and_success += 1
            if is_timeout and is_dz:
                n_dz_timeout += 1
                if triggered:
                    would_trigger_and_dz_timeout += 1

        fp_rate = would_trigger_and_success / n * 100 if n > 0 else 0
        coverage = would_trigger_and_dz_timeout / n_dz_timeout * 100 if n_dz_timeout > 0 else 0

        results.append({
            "threshold": T,
            "would_trigger_count": would_trigger,
            "false_positive_count": would_trigger_and_success,
            "false_positive_rate_pct": round(fp_rate, 2),
            "n_dz_timeout": n_dz_timeout,
            "coverage_count": would_trigger_and_dz_timeout,
            "coverage_pct": round(coverage, 1),
            "n_episodes": n,
        })

    return results


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
        print(f"  success/1M steps: {ci['success_per_1e6_steps']['point']:.1f} "
              f"[{ci['success_per_1e6_steps']['ci_lo']:.1f}, {ci['success_per_1e6_steps']['ci_hi']:.1f}]")
        print(f"  timeout/1M steps: {ci['timeout_per_1e6_steps']['point']:.1f} "
              f"[{ci['timeout_per_1e6_steps']['ci_lo']:.1f}, {ci['timeout_per_1e6_steps']['ci_hi']:.1f}]")
        print(f"  timeout split: {fm['timeout_and_dz_count']} dz + "
              f"{fm['timeout_and_no_dz_count']} non-dz = {stats['n_timeout']} total "
              f"({fm['frac_timeout_with_dz']:.1f}% / {fm['frac_timeout_without_dz']:.1f}%)")
        print()

    # --- Shadow multi-threshold trade-off ---
    shadow_path = os.path.join(DATA_DIR, SHADOW_CSV)
    shadow_results = []
    if os.path.exists(shadow_path):
        print("[Shadow] Multi-threshold trade-off analysis")
        shadow_rows = load_csv(shadow_path)
        if shadow_rows and "max_stuck_counter" in shadow_rows[0]:
            shadow_results = shadow_multi_threshold(shadow_rows, SHADOW_THRESHOLDS)
            print(f"  {'Threshold':>10s}  {'Trigger':>8s}  {'FP_count':>9s}  {'FP_rate':>8s}  "
                  f"{'DZ_TO':>6s}  {'Coverage':>9s}  {'Cov%':>6s}")
            print("  " + "-" * 70)
            for r in shadow_results:
                print(f"  {r['threshold']:>10d}  {r['would_trigger_count']:>8d}  "
                      f"{r['false_positive_count']:>9d}  {r['false_positive_rate_pct']:>7.2f}%  "
                      f"{r['n_dz_timeout']:>6d}  {r['coverage_count']:>9d}  {r['coverage_pct']:>5.1f}%")
            print()
        else:
            print("  [WARN] max_stuck_counter column not found in baseline CSV, skipping shadow analysis")
    else:
        print(f"[WARN] Shadow CSV not found: {shadow_path}")

    # 保存 JSON
    output = {
        "experiments": results,
        "shadow_tradeoff": shadow_results,
    }
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"[DONE] Results saved to {OUTPUT_JSON}")

    # --- 打印汇总对比表 ---
    print()
    print("=" * 140)
    print(f"{'Experiment':22s} {'Succ%':>12s} {'Timeout%':>14s} {'FailStep%':>14s} "
          f"{'LatP90':>14s} {'Succ/1M':>14s} {'TO/1M':>12s}")
    print("-" * 140)
    for name, _ in EXPERIMENTS:
        if name not in results:
            continue
        ci = results[name]["bootstrap_ci"]
        print(f"{name:22s} "
              f"{ci['success_rate']['point']:5.2f} [{ci['success_rate']['ci_lo']:5.2f},{ci['success_rate']['ci_hi']:5.2f}] "
              f"{ci['timeout_frac']['point']:5.2f} [{ci['timeout_frac']['ci_lo']:5.2f},{ci['timeout_frac']['ci_hi']:5.2f}] "
              f"{ci['fail_step_share']['point']:5.2f} [{ci['fail_step_share']['ci_lo']:5.2f},{ci['fail_step_share']['ci_hi']:5.2f}] "
              f"{ci['lateral_near_p90']['point']:5.3f} [{ci['lateral_near_p90']['ci_lo']:5.3f},{ci['lateral_near_p90']['ci_hi']:5.3f}] "
              f"{ci['success_per_1e6_steps']['point']:7.1f} "
              f"{ci['timeout_per_1e6_steps']['point']:7.1f}")
    print("=" * 140)

    # --- 打印失败模式拆分表 ---
    print()
    print("=" * 90)
    print(f"{'Experiment':22s} {'N_ep':>6s} {'N_timeout':>10s} {'DZ_timeout':>11s} "
          f"{'NoDZ_timeout':>13s} {'DZ%':>6s} {'NoDZ%':>7s}")
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
