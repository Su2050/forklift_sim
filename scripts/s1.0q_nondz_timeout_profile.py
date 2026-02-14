#!/usr/bin/env python3
"""S1.0Q 非死区 timeout 画像分析。

从 Batch-4 baseline eval 的 per-episode CSV 中筛选 "timeout AND NOT ever_dead_zone" 的 episode，
对其进行分类诊断：
  - near 徘徊型：max_insert_norm < 0.3（从未到 deep）
  - deep 精修不足型：max_insert_norm >= 0.3 但 max_hold_counter < 3
  - 接近成功型：max_hold_counter >= 3 但未达到成功

输出各类占比 + 特征分布，为 Batch-5 方向选择提供数据支撑。

用法:
    python scripts/s1.0q_nondz_timeout_profile.py [--csv path/to/episodes.csv]
"""
import csv
import math
import os
import sys

# 默认使用 Batch-4 baseline eval CSV
DEFAULT_CSV = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "data", "s1.0q_eval_batch4", "s1.0q_B4_baseline_episodes.csv"
)


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


def percentile(values, p):
    """计算百分位数（线性插值）。"""
    if not values:
        return 0.0
    s = sorted(values)
    n = len(s)
    k = (p / 100.0) * (n - 1)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return s[int(k)]
    d = k - f
    return s[int(f)] * (1 - d) + s[int(c)] * d


def print_distribution(name, values, unit=""):
    """打印一组值的分布统计。"""
    if not values:
        print(f"  {name}: (no data)")
        return
    print(f"  {name} (N={len(values)}):")
    for p in [25, 50, 75, 90, 95]:
        v = percentile(values, p)
        print(f"    p{p}: {v:.3f}{unit}")
    print(f"    mean: {sum(values)/len(values):.3f}{unit}")
    print(f"    min:  {min(values):.3f}{unit}, max: {max(values):.3f}{unit}")


def main():
    # 解析参数
    csv_path = DEFAULT_CSV
    for i, arg in enumerate(sys.argv[1:], 1):
        if arg == "--csv" and i < len(sys.argv) - 1:
            csv_path = sys.argv[i + 1]

    if not os.path.exists(csv_path):
        print(f"[ERROR] CSV 文件不存在: {csv_path}")
        print("  请先运行 Batch-4 baseline eval 生成包含新字段的 CSV")
        sys.exit(1)

    print(f"[INFO] 加载: {csv_path}")
    rows = load_csv(csv_path)
    n_total = len(rows)
    print(f"[INFO] 总 episode 数: {n_total}")

    # 找到 max_steps（timeout 判断条件）
    max_steps = max(r["steps"] for r in rows)
    print(f"[INFO] max_steps (timeout 阈值): {int(max_steps)}")

    # 筛选非死区 timeout
    nondz_timeout = [
        r for r in rows
        if r["steps"] == max_steps
        and r["success"] == 0
        and r["ever_dead_zone"] == 0
    ]
    n_nondz = len(nondz_timeout)

    # 同时计算死区 timeout 作为对照
    dz_timeout = [
        r for r in rows
        if r["steps"] == max_steps
        and r["success"] == 0
        and r["ever_dead_zone"] == 1
    ]
    n_all_timeout = sum(1 for r in rows if r["steps"] == max_steps and r["success"] == 0)

    print(f"\n{'='*70}")
    print(f"非死区 timeout 画像分析")
    print(f"{'='*70}")
    print(f"  总 timeout: {n_all_timeout} ({n_all_timeout/n_total*100:.1f}% of all episodes)")
    print(f"  死区 timeout: {len(dz_timeout)} ({len(dz_timeout)/max(n_all_timeout,1)*100:.1f}%)")
    print(f"  非死区 timeout: {n_nondz} ({n_nondz/max(n_all_timeout,1)*100:.1f}%)")

    if n_nondz == 0:
        print("\n  没有非死区 timeout episode，分析结束。")
        return

    # ---- 分类 ----
    near_hovering = []    # max_insert_norm < 0.3，从未到 deep
    deep_misaligned = []  # max_insert_norm >= 0.3 但 max_hold_counter < 3
    near_success = []     # max_hold_counter >= 3

    for r in nondz_timeout:
        if r["max_insert_norm"] < 0.3:
            near_hovering.append(r)
        elif r["max_hold_counter"] < 3:
            deep_misaligned.append(r)
        else:
            near_success.append(r)

    print(f"\n--- 分类结果 ---")
    print(f"  near 徘徊型 (max_insert < 0.3):     {len(near_hovering):4d} ({len(near_hovering)/n_nondz*100:.1f}%)")
    print(f"  deep 精修不足型 (deep but hold < 3): {len(deep_misaligned):4d} ({len(deep_misaligned)/n_nondz*100:.1f}%)")
    print(f"  接近成功型 (hold >= 3):               {len(near_success):4d} ({len(near_success)/n_nondz*100:.1f}%)")

    # ---- 各类特征分布 ----
    categories = [
        ("全部非死区 timeout", nondz_timeout),
        ("near 徘徊型", near_hovering),
        ("deep 精修不足型", deep_misaligned),
        ("接近成功型", near_success),
    ]

    for cat_name, cat_rows in categories:
        if not cat_rows:
            continue
        print(f"\n--- {cat_name} (N={len(cat_rows)}) ---")

        print_distribution("max_insert_norm", [r["max_insert_norm"] for r in cat_rows])
        print_distribution("max_hold_counter", [r["max_hold_counter"] for r in cat_rows])
        print_distribution("lat_near_mean", [r["lat_near_mean"] for r in cat_rows if r["lat_near_mean"] > 0], " m")
        print_distribution("yaw_near_mean", [r["yaw_near_mean"] for r in cat_rows if r["yaw_near_mean"] > 0], " deg")
        print_distribution("near_frac", [r["near_frac"] for r in cat_rows])

        # Batch-4 扩展字段（如果存在）
        if "min_y_err" in cat_rows[0]:
            min_y_vals = [r["min_y_err"] for r in cat_rows if r["min_y_err"] < 1e6]
            print_distribution("min_y_err", min_y_vals, " m")
        if "min_yaw_err_deg" in cat_rows[0]:
            min_yaw_vals = [r["min_yaw_err_deg"] for r in cat_rows if r["min_yaw_err_deg"] < 1e6]
            print_distribution("min_yaw_err_deg", min_yaw_vals, " deg")
        if "deep_steps" in cat_rows[0]:
            print_distribution("deep_steps", [r["deep_steps"] for r in cat_rows])
        if "max_stuck_counter" in cat_rows[0]:
            print_distribution("max_stuck_counter", [r["max_stuck_counter"] for r in cat_rows])

        # Batch-4 补充字段: z_err / pitch / action chattering
        if "min_abs_z_err" in cat_rows[0]:
            print_distribution("min_abs_z_err", [r["min_abs_z_err"] for r in cat_rows if r["min_abs_z_err"] < 1e6], " m")
        if "max_pitch_deg" in cat_rows[0]:
            print_distribution("max_pitch_deg", [r["max_pitch_deg"] for r in cat_rows], " deg")
        if "action_flip_rate" in cat_rows[0]:
            print_distribution("action_flip_rate", [r["action_flip_rate"] for r in cat_rows])

        # 终端状态
        print_distribution("final_insert_norm", [r["final_insert_norm"] for r in cat_rows])
        print_distribution("final_y_err", [r["final_y_err"] for r in cat_rows], " m")
        print_distribution("final_yaw_err_deg", [r["final_yaw_err_deg"] for r in cat_rows], " deg")

    print(f"\n{'='*70}")
    print(f"画像分析完成。")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
