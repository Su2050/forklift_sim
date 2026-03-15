# 2026-03-11 对话核心要点整理

## 1. 文档目的

本文件用于汇总本次围绕以下主题的对话结论，方便后续实验决策时快速回看：

- 外部论文 `Visual-Based Forklift Learning System Enabling Zero-Shot Sim2Real Without Real-World Data` 对当前项目的启发
- 可借鉴的公开研究与开源工作
- 当前视觉端到端训练线的阶段性判断
- 结合最近实验结果形成的具体建议

---

## 2. 外部论文的关键信息

### 2.1 论文里用的 ResNet 是什么

根据论文正文可确认的信息：

- 使用的是 **标准 ResNet**
- 做了 **ImageNet 预训练**
- 没有做额外的 domain adaptation
- 输入图像先从 `352x288` resize 到 `224x224`
- 左右两路相机分别提取 `512` 维特征，再与速度、偏航角速度、历史动作一起送入 PPO policy

论文原文的明确表述是：

> images are then converted into feature vectors ... using a ResNet pretrained on ImageNet  
> we use a standard pretrained ResNet without domain adaptation

### 2.2 哪个具体型号

论文正文里**没有明确写出**是 `ResNet18`、`ResNet34` 还是 `ResNet50`。

当前能给出的最稳妥结论是：

- **能确认它用了 ImageNet 预训练的标准 ResNet**
- **不能确认具体层数型号**

从每路输出 `512` 维特征这一点看，**更像 `ResNet18/34` 风格**，但这只是推测，不能当成论文明示结论。

### 2.3 这篇论文对当前项目最重要的启发

1. **视觉预训练路线是对的**
   - 论文不是让 RL 从零学视觉，而是直接使用预训练视觉表征。
   - 这强烈支持当前项目的 `视觉预训练 + RL finetune` 主线。

2. **任务拆解是合理的**
   - 论文把任务拆成 `approach policy` 和 `loading decision`。
   - 这和当前项目里把“接近/插入/对齐保持”与“lift/最终完成”拆开看的思路一致。

3. **近场几何信息很关键**
   - 论文用了双侧相机，强调司机在真实操作中会观察叉齿与托盘开口关系。
   - 对当前项目而言，这意味着如果后续还卡在精对齐，问题不一定只是 backbone，可能也和近场视角设计有关。

4. **纯视觉 actor + 特权 critic 的结构是合理的**
   - 论文的 actor 依赖视觉与可测量速度，reward/训练使用仿真中的 privileged state。
   - 这与当前项目的 asymmetric actor-critic 方向一致。

5. **domain randomization 不应只做外观随机化**
   - 论文还对观测速度和动作做扰动，而不只是灯光/颜色随机化。
   - 后续若要考虑 sim2real，这一点值得补上。

---

## 3. 可借鉴的公开研究与开源工作

### 3.1 更值得借鉴的论文/方法

1. **RRL: ResNet as Representation for Reinforcement Learning**
   - 直接支持“强 backbone 先做表征，再交给 RL”这条路线。

2. **DrQ-v2**
   - 很适合借鉴像素输入 RL 的低成本增强策略，例如随机裁剪、轻量 augmentation。

3. **RMA / privileged adaptation 系列**
   - 适合参考 `teacher -> student`、history encoder、在线适应这类方法论。

4. **Learn to Teach (L2T)**
   - 值得参考 teacher-student 是否能边学边蒸馏，而不是严格两阶段切分。

5. **机器人足球/egocentric vision sim2real 类工作**
   - 虽然不是叉车，但在“第一视角 + 精对位 + sim2real”上有相似结构。

### 3.2 更值得翻的开源仓库/工程

1. [`IsaacSim-Autonomous-Forklift`](https://github.com/iminolee/isaacsim-autonomous-forklift)
   - 值得看场景组织、ROS/仿真桥接、托盘检测与对接工程拆分。

2. [`facebookresearch/drqv2`](https://github.com/facebookresearch/drqv2)
   - 值得借像素 RL 训练增强和稳定化策略。

3. [`penn-pal-lab/scaffolder`](https://github.com/penn-pal-lab/scaffolder)
   - 值得借 privileged sensing / teacher-student 的方法框架。

4. [`IsaacLab`](https://github.com/isaac-sim/IsaacLab)
   - 重点看 sim2real、gear assembly、synthetic data 相关示例。

5. 托盘检测/姿态估计相关 ROS 仓库
   - 例如 `auto_forklift_pallet_detection`
   - 适合作为感知 debug baseline，而不是直接替代 RL 主线

---

## 4. 当前实验线的阶段性判断

### 4.1 已基本确认的结论

1. **`64x64 + MobileNet scratch`**
   - 能明显优于最初简单 CNN，但最终大约卡在 `20%` 平台。
   - 说明 backbone 升级有效，但纯 RL scratch 仍学不到稳定精对齐。

2. **`64x64 + 预训练 + RL finetune`**
   - 成功率提升到约 `26%`。
   - 更关键的是，训练早期横向误差和偏航误差一度很好，说明预训练视觉表征确实有效。
   - 真正的问题是：**后续 RL 微调没有把这份精度稳定保住。**

3. **`256x256 scratch`**
   - 在当前任务上基本不可行。
   - 这进一步证明：**高分辨率本身不能替代预训练。**

4. **`256x256 + 预训练 + 强推盘惩罚`**
   - 早期会插，但后面逐渐变成极端保守。
   - 说明“强惩罚”可以压住推盘，但也会把策略吓成“不敢插”。

5. **`256x256 + 预训练 + 正常推盘惩罚`**
   - 表面成功率约 `22%~25%`，但托盘位移很大。
   - 本质上是**推土机式假成功**，不是可靠的精确插入。

### 4.2 当前最重要的判断

目前更可信的主结论不是“换更大的 backbone 就能解决问题”，而是：

- **视觉预训练主线是对的**
- **当前主瓶颈在微调策略与 reward gate**
- **近场精对齐和插入保持仍未真正打通**

### 4.3 关于 `freeze50` 的说明

本次对话里对 `freeze50` 出现过一个需要特别标记的点：

1. 从我们快速查看的**早期日志**看：
   - `diag/pallet_disp_xy_mean` 很低
   - `err/yaw_deg_near_success` 和 `err/lateral_near_success` 一度很好
   - 说明它早期至少不像“推土机”那样失控

2. 但当前已有实验文档 `vision_e2e_rl_baseline_guide.md` 中，对 `freeze50` 的阶段性表述是：
   - 后续发生了灾难性遗忘
   - 最终效果很差

因此现阶段对 `freeze50` 的稳妥记录应为：

- **早期几何指标看起来有希望**
- **但最终结论仍应以完整实验结果文档和完整日志为准**

---

## 5. 当前形成的一致建议

### 5.1 不要因为论文用了 ResNet，就立刻把当前主线改成 ResNet

原因：

- 当前实验已经证明 `256x256 + 预训练` 能显著改善早期几何表现
- 现在的主问题不像是“完全看不见”
- 更像是“看见了，但在 RL 中学成了推盘 / 保守 / 不愿 commit 插入”

所以当前 ROI 更高的，不是立即 `MobileNet -> ResNet`，而是先修：

- freeze/unfreeze 策略
- reward gate
- 近场插入激励
- 举升阶段拆分

### 5.2 当前更值得做的是“对准后前插激励”

结合最近实验，建议优先补一类更对症的奖励设计：

1. 当横向误差和偏航误差已进入 fine gate 时：
   - 显式奖励 `dist_front` 下降
   - 或显式奖励 `insert_norm` 上升

2. 在这个 gate 内：
   - 适度减弱“停滞”类惩罚
   - 避免策略在对准后因为怕碰撞而不敢往前 commit

### 5.3 推盘惩罚不要再只在“太松”和“太狠”之间切换

更建议使用条件化或平滑化的方式：

1. 当横向/偏航还没进 gate 时：
   - 推盘惩罚重一些

2. 当已经进精对齐 gate 时：
   - 仍保留一个小的非零惩罚
   - 不要因为浅插入就过早把惩罚清零

3. 最好把“惩罚是否降低”与 `lift` 是否开始绑定，而不只是和“插入深度超过某值”硬绑定

### 5.4 举升阶段应被单独当成子问题

目前多个实验里：

- `phase/frac_lifted` 基本仍是 `0`

因此更合理的阶段目标应是：

1. 先打通 `接近 -> 精对齐 -> 插入 -> 保持`
2. 再单独打通 `lift`

后续可考虑：

- 单独课程
- 单独 decision head
- 类似论文中的 `loading decision` 拆分

### 5.5 如果后面仍然“看得差不多但插不进去”，优先升级预训练目标，而不是先换更大 backbone

优先建议：

1. 从粗 `(x, y, yaw)` 回归
2. 升级为孔位/开口关键点
3. 或局部 heatmap / dense supervision

原因是：

- 当前预训练已经能提供一定几何感知能力
- 真正缺的可能是“最后几厘米”的结构化几何线索
- 这往往比单纯换更大的 backbone 更直接

### 5.6 视角设计仍值得后续重新评估

如果后续 reward 和预训练目标都修过，仍然卡在近场精对齐：

- 值得重新评估双相机
- 值得评估更低、更前、更贴近叉齿的视角
- 也值得评估是否需要单独服务近场对齐的辅助视角

---

## 6. 推荐的后续执行顺序

按当前对话形成的优先级，建议顺序如下：

1. **先核实 `freeze50` 的完整结果**
   - 不要只看早期日志
   - 以完整实验文档和完整日志为准

2. **优先修 reward gate，而不是先换 backbone**
   - 重点解决“对准后不敢前插”和“浅插后推土机”两个问题

3. **如果 reward 修正后仍然不稳，再升级预训练目标**
   - 从粗位姿回归升级到关键点/heatmap

4. **只有在上述问题基本理顺后，再评估是否切 ResNet18**
   - 这时再做 `MobileNetV3-Small vs ResNet18` 才更有解释力

5. **最后再把 lift 阶段单独打通**

---

## 7. 一句话结论

本次对话形成的最核心判断是：

**论文证明了“ImageNet 预训练视觉 backbone + RL”这条路是成立的；而当前项目最近实验也表明，眼下最该修的不是 backbone 型号本身，而是微调策略、reward gate、近场精对齐监督以及 lift 阶段拆分。**
