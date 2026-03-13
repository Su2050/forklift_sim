import re

with open('docs/0310-0313experiments/system_analysis_and_iterative_plan_20260311.md', 'r', encoding='utf-8') as f:
    content = f.read()

# Update Section 6.6 and add 6.7
new_section_6_6 = r"""### 6.6 实验 5.4：放宽 rg 阈值，打破认知障碍 (重大突破)
**背景**：Agent 变成了推土机，因为 `rg` 触发条件（误差 < 0.1m）在随机探索下物理不可达。
**行动**：将 `paper_rg_dist_thresh` 放宽到 0.4m（插进一半就算赢）。
**结果**：**巨大成功！** Agent 偶然触发大奖后瞬间开窍，`rg` 触发率飙升至 60% 以上，回合长度大幅缩短。确立了后续“基于 Checkpoint 逐步收紧阈值”的课程学习路线。

### 6.7 实验 5.5 vs 5.5b：课程退火的艺术 (步子迈大了容易扯着蛋)
**背景**：在 0.4m 阈值稳定后，我们试图直接将阈值收紧到 0.2m (Exp 5.5)。
**结果 (Exp 5.5 失败)**：`rg` 瞬间跌回 0，Agent 发现大奖太难拿，退化回了“推土机”状态（托盘位移飙升至 0.95m，偏航角恶化至 15°）。
**修正 (Exp 5.5b 平滑退火)**：
1. 将阈值退回到 **0.3m**。
2. 将推盘惩罚 `alpha_5` 从 1.0 稍微提高到 **3.0**，抑制推土机行为。
3. 重新加载 0.4m 时的 Checkpoint。
**结果 (Exp 5.5b 成功)**：Agent 完美适应了 0.3m 的新难度。`rg` 成功率稳定在 50%~60%，且**偏航角误差显著下降**（从 15° 降至 8°），托盘位移也被压制在 0.3m 左右。确立了“平滑退火 + 动态惩罚”的 SOP。
"""

# Replace the old 6.6 with the new 6.6 and 6.7
start_marker = "### 6.6 实验 5.4：放宽 rg 阈值，打破认知障碍 (重大突破)"
end_marker = "## 7. 实验 0"

start_idx = content.find(start_marker)
end_idx = content.find(end_marker)

if start_idx != -1 and end_idx != -1:
    content = content[:start_idx] + new_section_6_6 + '\n' + content[end_idx:]

# Update Section 18.4 (Curriculum Learning)
new_section_18_4 = r"""### 18.4 沉淀的突破性经验
11. **`rg` 终局奖励的物理不可达阈值与“宽容篮筐”效应 (在实验 5.4 中突破)**
    - **现象**：在修复了所有漏洞后，Agent 变成了“推土机”，顶着托盘往前推，死活不插进去。
    - **根本原因**：终局奖励 `rg` 的触发条件太苛刻（`dist_center < 0.1`，要求完美插到底）。在随机探索阶段，Agent 根本不可能偶然达到这个精度，导致 Critic 网络永远不知道“插入”有 200 分的大奖。
    - **突破性经验 (动态阈值/课程学习)**：将 `rg` 的触发阈值放宽到 `0.4`（只要插进去一半就算成功）。这相当于把“篮筐”放低放大。修改后，Agent 在几代之内就偶然触发了奖励，瞬间“开窍”，成功率飙升至 60% 以上。

12. **课程退火的艺术：步子迈大了容易灾难性遗忘 (在实验 5.5b 中修正)**
    - **现象**：在 0.4m 阈值成功后，直接收紧到 0.2m，Agent 瞬间崩溃，退化回推土机。
    - **原因**：从插 2/3 到几乎插到底，对姿态精度的要求呈指数级上升，直接跳跃导致了灾难性遗忘。
    - **经验 (动态退火 SOP)**：确立了近场插入的标准操作流程：
      1. **宽容破冰**：用极宽阈值（0.4m）激活 Critic。
      2. **平滑退火**：以 0.1m 为步长（0.4 -> 0.3 -> 0.2 -> 0.1）逐步收紧。
      3. **继承记忆**：每次收紧必须加载上一阶段稳定的 Checkpoint。
      4. **动态惩罚**：在收紧阈值的同时，缓慢增加推盘惩罚（如 1.0 -> 3.0），逼迫其走向精细控制。

13. **轨迹奖励在近场的“喧宾夺主”效应 (在实验 5.2 中修正)**
"""

start_marker_18 = "### 18.4 沉淀的突破性经验"
end_marker_18 = "13. **视觉表征的“伪瓶颈”与 RRL 范式的确立 (Exp 1 vs Exp 2)**"

start_idx_18 = content.find(start_marker_18)
end_idx_18 = content.find(end_marker_18)

if start_idx_18 != -1 and end_idx_18 != -1:
    # Need to adjust the numbering for the rest of the list
    rest_of_content = content[end_idx_18:]
    rest_of_content = rest_of_content.replace("13. **视觉表征", "14. **视觉表征")
    rest_of_content = rest_of_content.replace("14. **动作空间", "15. **动作空间")
    rest_of_content = rest_of_content.replace("15. **势函数差分", "16. **势函数差分")
    
    content = content[:start_idx_18] + new_section_18_4 + rest_of_content

with open('docs/0310-0313experiments/system_analysis_and_iterative_plan_20260311.md', 'w', encoding='utf-8') as f:
    f.write(content)

print("Updated main plan doc.")
