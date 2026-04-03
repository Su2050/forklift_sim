import re, statistics, os

path = "/home/uniubi/projects/forklift_sim_exp9/logs/20260402_093917_train_exp9_0_no_reference_rewardfix_o3_seed42.log"
with open(path) as f:
    text = f.read()

iter_blocks = re.split(r'Learning iteration (\d+)/2000', text)
metrics = [
    "phase/frac_inserted", "phase/frac_aligned", "phase/frac_tip_constraint_ok",
    "phase/frac_hold_entry", "phase/frac_success",
    "err/center_lateral_inserted_mean", "err/tip_lateral_inserted_mean",
    "err/yaw_deg_inserted_mean", "paper_reward/r_postinsert_align",
    "paper_reward/R_plus", "paper_reward/r_d", "diag/max_hold_counter"
]
data = {}
for i in range(1, len(iter_blocks)-1, 2):
    it = int(iter_blocks[i])
    block = iter_blocks[i+1]
    for m in metrics:
        pat = re.escape(m) + r':\s*([-\d.eE+]+)'
        match = re.search(pat, block)
        if match:
            data.setdefault(m, []).append((it, float(match.group(1))))

if not data:
    print("No data found.")
    exit(1)

last_iter = max(it for vals in data.values() for it, _ in vals)

md_content = f"# Exp9.0 O3 训练结果总结\n\n"
md_content += f"**日志**: `{os.path.basename(path)}`\n"
md_content += f"**最终迭代**: {last_iter}/2000\n\n"

md_content += "## 1. 终态指标 (Last 50 均值)\n\n"
md_content += "| Metric | Last 50 Mean |\n|---|---|\n"
for m in metrics:
    vals = data.get(m, [])
    if vals:
        l50 = statistics.mean([v for _, v in vals[-50:]])
        md_content += f"| `{m}` | {l50:.4f} |\n"

md_content += "\n## 2. 阶段趋势\n\n"
windows = [(0, 500, "0-500"), (500, 1000, "500-1000"), (1000, 1500, "1000-1500"), (1500, 2001, "1500-2000")]
md_content += "| Metric | " + " | ".join([l for _, _, l in windows]) + " |\n"
md_content += "|---|" + "|".join(["---"] * len(windows)) + "|\n"
for m in ["phase/frac_inserted", "phase/frac_aligned", "err/center_lateral_inserted_mean", "err/tip_lateral_inserted_mean", "err/yaw_deg_inserted_mean", "diag/max_hold_counter", "paper_reward/r_postinsert_align"]:
    vals = data.get(m, [])
    row = f"| `{m}` |"
    for lo, hi, _ in windows:
        wvals = [v for it, v in vals if lo <= it < hi]
        if wvals:
            row += f" {statistics.mean(wvals):.4f} |"
        else:
            row += f" n/a |"
    md_content += row + "\n"

succ_vals = data.get("phase/frac_success", [])
succ_positive = sum(1 for _, v in succ_vals if v > 0)
md_content += f"\n**Success > 0 迭代数**: {succ_positive}/{len(succ_vals)}\n"

out_path = "/home/uniubi/projects/forklift_sim_exp9/docs/exp9_0/exp9_0_rewardfix_o3_result_20260402.md"
with open(out_path, "w") as f:
    f.write(md_content)
print(f"Summary written to {out_path}")