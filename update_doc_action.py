import re

with open('docs/0310-0313experiments/system_analysis_and_iterative_plan_20260311.md', 'r', encoding='utf-8') as f:
    content = f.read()

# Add the action space insight to Section 18.3
action_insight = r"""
8. **动作空间的降维打击 (Exp 1 & Exp 4)**
   - **现象**：最初我们让 Agent 同时控制 `drive` (驱动), `steer` (转向) 和 `lift` (举升) 三个动作。结果 Agent 在接近托盘时，经常因为胡乱举升导致叉车重心不稳、翻车，或者叉尖高度不对而撞击托盘。
   - **根本原因**：将“导航接近”和“举升操作”混在一个 RL 策略中，极大地增加了探索空间的维度和噪声。在还没有学会“走到门口”之前，就让它乱动叉子，是典型的“还没学会走就想跑”。
   - **突破性经验**：在实验 1 和实验 4 中，我们坚决贯彻了论文的思路，将动作空间从 3 维降到 2 维 (`approach-only` 策略，只控制 `drive` 和 `steer`)，强制 `lift` 关节固定为 0。这极大地收敛了探索空间，使得 Agent 能够专心学习平面上的对齐和插入。这证明了**在复杂机器人任务中，合理的动作空间解耦比端到端“大力出奇迹”更重要**。

9. **势函数差分奖励的“薅羊毛”陷阱 (Exp 3.x)**
   - **现象**：在实验 3.x 中，我们尝试使用复杂的势函数差分（如 `dist_front` 的下降量）来奖励插入。结果 Agent 学会了“斜着浅插推着走”：它发现只要保持在托盘边缘摩擦，就能不断产生微小的“距离下降”，从而源源不断地薅取差分奖励，而这笔收益竟然超过了推盘的惩罚。
   - **根本原因**：差分奖励（基于状态变化量）在连续控制中极易被利用。Agent 可以通过高频的微小震荡或卡在某个临界点，将单次任务变成“无限提款机”。
   - **突破性经验**：在实验 4 中，我们果断废弃了差分奖励，回归论文原生的**绝对状态驱动奖励**（如 $exp(-dist)$）。绝对状态奖励只看“你现在离目标有多近”，而不是“你刚才靠近了多少”，从数学上彻底封死了“原地摩擦刷分”的漏洞。
"""

# Insert the new insight before the end of the file
content = content + action_insight

with open('docs/0310-0313experiments/system_analysis_and_iterative_plan_20260311.md', 'w', encoding='utf-8') as f:
    f.write(content)

print("Document updated with action space and reward shaping insights.")
