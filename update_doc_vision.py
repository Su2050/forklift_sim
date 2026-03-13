import re

with open('docs/0310-0313experiments/system_analysis_and_iterative_plan_20260311.md', 'r', encoding='utf-8') as f:
    content = f.read()

# Add the vision backbone insight to Section 18.3
vision_insight = r"""
7. **视觉表征的“伪瓶颈”与 RRL 范式的确立 (Exp 1 vs Exp 2)**
   - **现象**：在早期实验中，我们花费了大量精力进行 MobileNetV3 的坐标回归预训练，试图让 CNN 直接输出精确的 `(x, y, yaw)`。但无论怎么训练，Y 轴误差始终降不下去（约 16cm），且最终 RL 成功率卡死在 16.5%。
   - **根本原因（认知偏差）**：我们错误地认为“如果 CNN 不能输出精确的物理坐标，RL 就学不会”。但实际上，论文采用的是 **RRL (ResNet as Representation for RL)** 范式，即直接使用标准的 ImageNet 预训练特征，并且**全程冻结骨干网络**。
   - **突破性经验**：在实验 2 中，我们直接换用冻结的 ResNet18 (ImageNet 权重)，不加任何物理坐标预训练，结果发现其表现与辛辛苦苦预训练的 MobileNetV3 完全一致！这证明了**当前的瓶颈根本不在视觉感知端，而在于 RL 的探索策略和 Reward 塑造**。将“理解视觉特征的物理含义”的任务交给下游的 MLP (Actor/Critic) 去学习，是更简洁、更鲁棒的路线。
"""

# Insert the new insight before the end of the file
content = content + vision_insight

with open('docs/0310-0313experiments/system_analysis_and_iterative_plan_20260311.md', 'w', encoding='utf-8') as f:
    f.write(content)

print("Document updated with vision insight.")
