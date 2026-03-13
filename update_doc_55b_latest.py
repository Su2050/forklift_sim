import re

file_path = "/home/uniubi/projects/forklift_sim/docs/0310-0313experiments/system_analysis_and_iterative_plan_20260311.md"

with open(file_path, "r", encoding="utf-8") as f:
    content = f.read()

old_text = """**结果 (Exp 5.5b 成功)**：Agent 完美适应了 0.3m 的新难度。`rg` 成功率稳定在 50%~60%，且**偏航角误差显著下降**（从 15° 降至 8°），托盘位移也被压制在 0.3m 左右。确立了“平滑退火 + 动态惩罚”的 SOP。"""

new_text = """**结果 (Exp 5.5b 成功)**：Agent 完美适应了 0.3m 的新难度。
   - **成功率 (`rg`)**：稳定在 **50%** 左右（Iteration 200+ 持续保持）。
   - **姿态精度显著提升**：偏航角误差从 15° 稳步下降至 **8.9°** 左右，横向误差控制在 **0.15m**。
   - **推盘行为被有效控制**：托盘位移被压制在 **0.39m** 左右（相比于 0.2m 阈值时失控的 0.95m，`alpha_5=3.0` 的惩罚权重起到了完美的平衡作用）。
   - **结论**：确立了“平滑退火 + 动态惩罚”的 SOP，证明了在保证能拿到大奖的前提下，逐步收紧阈值可以有效逼迫 Agent 提高精度。"""

if old_text in content:
    content = content.replace(old_text, new_text)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)
    print("Successfully updated 5.5b latest results in system_analysis_and_iterative_plan_20260311.md")
else:
    print("Could not find the target text to replace.")
