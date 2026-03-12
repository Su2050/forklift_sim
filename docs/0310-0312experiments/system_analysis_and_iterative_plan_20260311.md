# 视觉叉车系统分析与迭代实验总计划 (2026-03-11)

## 1. 目的

本文件用于把当前视觉叉车训练线的判断、目标、实验路线和记录规范统一下来，按“**假设 -> 验证 -> 修正假设 -> 再验证**”的方式持续推进，避免：

- 已经证伪的方向重复踩坑
- 不同实验之间缺乏可比性
- 只盯单个成功率指标，忽略“假成功”

本文件应作为后续一段时间内的**总实验计划与实验账本入口**。每完成一轮实验，应同步更新本文件，并将详细结果写入独立实验 md。

---

## 2. 当前最新结论

基于以下材料综合判断：

- `docs/0310-0311experiments/mobilenet_baseline_result_20260310.md`
- `docs/0310-0311experiments/mobilenet_finetune_result_20260310.md`
- `docs/0310-0311experiments/scratch_baseline_256x256_result_20260311.md`
- `docs/0310-0311experiments/finetune_256x256_strong_pen_result_20260311.md`
- `docs/0310-0311experiments/finetune_256x256_normal_pen_result_20260311.md`
- `docs/0310-0311experiments/finetune_256x256_freeze50_result_20260311.md`
- 最新日志 `logs/20260311_154823_train_finetune_256x256_freeze50_env64.log`

### 2.1 当前没有一条可以算“论文式成功训练”的路线

原因如下：

1. **`256x256 scratch` 全程为 0**
   - 说明纯 RL 从零开始学习高分辨率视觉输入在当前任务上不可行。

2. **`256x256 + pretrain + strong penalty`**
   - 早期会插，但很快被强惩罚压成极端保守策略。
   - 说明惩罚过强会把策略训练成“不敢插”。

3. **`256x256 + pretrain + normal penalty`**
   - 表面成功率约 `22%~25%`，但托盘位移达到 `~2m`。
   - 这是典型“推着托盘走”的**假成功**，不能视为真正的精确插入。

4. **`256x256 + pretrain + freeze50`**
   - 最新日志到 `iteration 310` 左右，`success_rate_total` 约 `0.25%`，`phase/frac_inserted` 回到 `0.0`。
   - 说明短冻结期 + 端到端 RL 微调导致了明显退化，符合“灾难性遗忘”判断。

### 2.2 当前真正的系统状态

当前更准确的系统状态不是“完全看不见”，而是：

- 能一定程度靠近托盘
- 能在部分时刻出现较好的横向/偏航误差
- 但**无法稳定变成 push-free 的插入保持成功**
- 更不会自然打通 `lift`

### 2.3 当前最重要的负面证据

1. **RL 不能可靠地修预训练误差**
   - `freeze50` 说明：当 Actor 还不成熟时，RL 传回给 backbone 的梯度质量太差，会破坏预训练特征。

2. **当前 success 指标会被 reward hacking 污染**
   - `normal_pen` 已证明：只看 `success_rate_total` 会把“推土机”误判为进步。

3. **任务拆分方式与论文不一致**
   - 当前 Stage 1 虽然 success 已不再要求 lift，但**动作空间仍是 3 维**：
     - `drive`
     - `steer`
     - `lift`
   - 这意味着策略在“只考接近/插入”的阶段，仍要同时探索一个不必要的 `lift` 动作维度。
   - 论文的 `approach policy` 只输出 `throttle + steering`，`lift` 决策是后续单独的监督策略。

---

## 3. 与论文《Visual-Based Forklift Learning System Enabling Zero-Shot Sim2Real Without Real-World Data》的差距

### 3.1 论文做对了什么

论文的关键成功点不是单一一个 ResNet，而是以下组合：

1. **使用标准 ImageNet 预训练 ResNet**
2. **使用双侧相机，强调叉齿与托盘开口的近场几何关系**
3. **把任务拆成 approach policy 和 loading decision**
4. **approach policy 只学驱动与转向，不把 lift 混进同一个 RL policy**
5. **使用 photorealistic sim + domain randomization**
6. **reward/critic 允许使用 privileged information**

### 3.2 我们与论文的核心差距

当前项目与论文相比，主要差在 4 个地方：

1. **预训练范式完全不同 (最致命差距)**
   - 当前：试图让 CNN 回归物理坐标 (x,y,yaw)，导致陷入“预训练误差瓶颈”。
   - 论文：使用 RRL 范式，直接拿 ImageNet 预训练的 ResNet，**全程冻结**，只输出通用视觉特征，把“理解物理含义”的任务交给下游 FC 层在 RL 中学习。

2. **任务拆分不够彻底**
   - 当前 Stage 1 的 success 已经不要求 lift，但 actor 仍输出 3 维动作。

3. **相机几何仍与论文不同**
   - 当前主线是单个 `60°` 俯视相机。
   - 论文更强调从侧边观察叉齿与托盘开口关系。

4. **评估指标还没完全区分“真插入”和“推土机”**
   - 当前日志里虽然有 `pallet_disp_xy_mean`，但还没有把“push-free success”提升为一等 KPI。

---

## 4. 系统拆解：影响因素与关系图

为了避免胡乱试超参，先把系统拆开。

### 4.1 影响最终成功率的 6 个主因素

1. **观测几何**
   - 相机数量
   - 相机高度/俯角/前向偏移
   - 是否能稳定看到叉齿和托盘开口的相对关系

2. **视觉表征精度**
   - backbone 容量
   - 分辨率
   - 数据量与数据分布
   - 监督目标是否足够密集

3. **任务拆分方式**
   - 是否将 `approach` 与 `lift/decision` 拆开
   - 是否让早期 RL 去承担过多子任务

4. **RL 优化动力学**
   - freeze/unfreeze 策略
   - backbone 学习率
   - 是否全程冻结
   - 是否延迟解冻

5. **reward / curriculum / success 判据**
   - 是否鼓励“对准后继续往前插”
   - 是否允许“推托盘式假成功”
   - 是否过早把策略训练成保守

6. **sim realism / randomization**
   - 材质、光照、颜色
   - 动作噪声、观测噪声
   - 后续 sim2real 时尤为重要

### 4.2 因果链条

当前系统更符合下面这条因果链：

`观测几何`
-> `视觉表征质量`
-> `是否能进入精对齐区域`
-> `reward 是否引导继续插入而不是推盘/停住`
-> `是否能稳定 hold`
-> `是否值得再学 lift`

这意味着：

- 如果前面的视觉表征和观测几何不过关，后面的 RL 微调和 reward 修修补补只会不断撞墙。
- 如果 reward 判据不对，即便视觉表征够了，也会被“推土机”污染。

---

## 5. 新的总目标与验收口径

为了避免再次被“假成功”误导，后续不再把 `success_rate_total` 作为唯一目标。

### 5.1 一级目标：先复现论文的 approach 训练逻辑

短期目标不是“端到端学完整 lift”，而是先做到：

- **纯视觉 approach policy**
- **只学 drive + steer**
- **在不推托盘的情况下完成接近 + 对齐 + 插入 + hold**

### 5.2 二级目标：再做 loading decision / lift

当一级目标稳定后，再做：

- 是否触发 lift 的判定
- lift 动作执行

### 5.3 后续统一主 KPI

建议新增并统一关注以下指标：

1. `push_free_success_rate`
   - 成功且托盘位移未超过阈值

2. `push_free_insert_rate`
   - 达到插入阈值且托盘位移未超过阈值

3. `pallet_disp_xy_mean / p95 / max`

4. `err/lateral_near_success`

5. `err/yaw_deg_near_success`

6. `phase/frac_inserted`

7. `phase/hold_counter_max`

8. `phase/frac_lifted`

### 5.4 建议的“真成功”阈值

在论文式 approach 训练阶段，建议至少满足：

- `push_free_success_rate >= 0.40` 作为中期门槛
- `push_free_success_rate >= 0.60` 作为进入下一阶段的门槛
- `pallet_disp_xy_mean <= 0.05m`
- `err/lateral_near_success <= 0.10m`
- `err/yaw_deg_near_success <= 5.0°`

说明：

- 这比当前日志里的“总成功率”更难，但也更接近真正可迁移的策略。

---

## 6. 迭代实验总路线

下面的实验顺序遵循：

- 先做**单因素验证**
- 再做**组合验证**
- 每一步只回答一个核心问题

---

## 7. 实验 0：先修实验基础设施与评估口径

### 假设 H0

如果评估口径不修，后续所有实验都可能再次被“假成功”污染。

### 只改一个因素

不是改训练逻辑，而是改**日志与评估指标**。

### 要做的事情

1. 新增 `push_free_success_rate`
2. 新增 `push_free_insert_rate`
3. 增加 `pallet_disp_xy_p95`
4. 固定每轮实验都导出：
   - 训练尾段关键指标表
   - 至少 3 个评估视频
   - 一段失败案例视频

### 验证标准

- 从此以后，任何实验结论必须同时报告：
  - `success_rate_total`
  - `push_free_success_rate`
  - `pallet_disp_xy_mean/p95`

### 决策规则

- 若没有 `push_free` 相关指标，则该轮实验**不算有效比较**。

---

## 8. 实验 1：按论文方式彻底拆开 approach 与 lift

### 假设 H1

当前 Stage 1 虽然 success 不要求 lift，但 actor 仍输出 `lift` 动作，这会明显增加探索噪声，降低 sample efficiency。  
如果把 Stage 1 改成真正的 **2 动作 approach-only policy**，效果会优于当前 3 动作联合策略。

### 只改一个因素

- 将 Stage 1 的 actor 动作改为仅 `drive + steer`
- `lift` 固定为 `0`

### 保持不变

- 当前最佳相机
- 当前最佳预训练权重
- 当前 reward（先不改）
- 当前课程

### 验证指标

- `phase/frac_inserted`
- `push_free_insert_rate`
- `push_free_success_rate`
- `pallet_disp_xy_mean`

### 预期

若 H1 成立，应出现：

- 插入率上升
- 推盘位移下降
- 成功率更稳定

### 决策规则

- 如果该实验显著优于现有 3 动作 Stage 1，则后续所有 approach 实验都统一改成 2 动作。
- 如果收益很小，再考虑是观测与表征而非任务拆分在主导。

---

## 9. 实验 2 (新主线)：基于 RRL 范式的 ImageNet 冻结骨干实验

### 假设 H2_RRL

**重大发现与假设修正**：
论文《Visual-Based Forklift Learning System...》中明确提到：*"No special processing is required because we use a standard pretrained ResNet without domain adaptation."*
这意味着，他们**根本没有做特定任务的物理坐标回归预训练**！
他们的核心范式是 **RRL (ResNet as Representation for RL)**：直接使用通用的 ImageNet 预训练特征（如 ResNet-18 输出的 512 维特征），并且**全程冻结骨干网络**。将“理解这些视觉特征代表什么物理含义”的任务，完全交给下游没有被冻结的 FC 层（Actor/Critic head）在 RL 探索中去学习。

因此，我们之前纠结的“预训练 Y 误差 16cm 降不下去”可能是一个伪命题。我们不需要让 CNN 输出精确坐标，只需要它输出高质量的通用视觉特征即可。

### 只改一个因素

- 替换 Backbone 为 `ResNet-18`
- 加载标准的 `ImageNet` 预训练权重
- **全程冻结** Backbone (`freeze_backbone=True`)
- 仅让 RL 训练下游的 MLP 层

### 保持不变

- 实验 1 确立的 2 动作 `approach-only` 策略
- 当前 reward 与课程

### 验证指标

- `push_free_success_rate`
- `push_free_insert_rate`
- `pallet_disp_xy_mean`

### 预期

若 H2_RRL 成立，策略应能在没有特定任务预训练的情况下，通过 RL 逐渐学会在近场进行精细对齐，且不会破坏视觉特征（因为已冻结）。

### 决策规则

- 如果该实验能稳步提升 `push_free_success_rate`，则正式确立 RRL 范式为主线，废弃之前的“坐标回归预训练”路线。
- 如果效果不佳，再考虑是否是视角或奖励函数的问题。

---

## 10. [已降级/备用路线] 试图让骨干网络理解物理坐标的特定任务预训练

> ⚠️ **[2026-03-11 晚间修正] 本路线已降级为备用路线**。
> 原因：根据对论文的重新审视，发现其采用的是 RRL 范式（直接使用冻结的 ImageNet 特征），并未进行特定任务的坐标回归预训练。因此，本节中关于“扩大数据集”、“升级密集几何监督”的计划暂时搁置，优先执行**实验 2 (新主线)**。

### 10.1 备用实验 A：提升预训练数据分布

**假设**：当前预训练失败的主因，不一定只是模型容量，而是数据分布不够贴近“最后几厘米”的难点。如果扩大数据集并显式提升近场/硬例比例，预训练误差会明显下降。

### 10.2 备用实验 B：升级预训练目标，从粗位姿回归变成更密集的几何监督

**假设**：当前 `(x, y, yaw)` 粗回归不够支撑最后几厘米的精对齐。如果引入孔位/开口关键点、heatmap 或多任务监督，视觉 backbone 会学到更稳定的近场几何特征。

### 10.3 备用实验 C：在监督目标和数据分布固定后，比较 backbone 容量

**假设**：如果数据分布和监督目标都已经更对了，但误差仍高，瓶颈才更可能在 backbone 容量。

---

## 11. 实验 3：重塑 Reward 函数，突破 16.5% 瓶颈 (当前最高优先级)

### 假设 H3_Reward

**重大发现与假设修正**：
经过 2026-03-11 晚间的实验 1 和实验 2 对比，我们发现无论使用复杂的 MobileNetV3 坐标回归预训练，还是使用简单的 ResNet18 冻结通用特征 (RRL 范式)，最终的无碰撞成功率都死死卡在 **16.5%** 左右。
这说明：**当前的瓶颈已经不在视觉感知端，而在于 RL 的探索策略和 Reward 函数的塑造。** 智能体在接近目标时，缺乏足够的引导来完成最后的“精准插入”动作。

### 只改一个因素

在实验 2 (ResNet18 冻结 RRL 范式) 的基础上，只修改 `env.py` 中的 Reward 函数：
1. **增加“保持对齐”奖励**：鼓励叉车在对准后稳定姿态，不要乱动。
2. **强化“插入深度”奖励**：在最后插入阶段（`insert_depth` > 0），给予更强的梯度奖励，引导智能体勇敢向前。
3. **微调“推托盘”惩罚**：寻找探索（允许轻微碰撞）与保守（严格防推）之间的最佳平衡点。

### 保持不变

- 实验 1 确立的 2 动作 `approach-only` 策略
- 实验 2 确立的 RRL 范式 (ResNet18 + ImageNet权重 + 全程冻结)

### 验证指标

- `push_free_success_rate` (目标：突破 16.5%，冲击 30%+)
- `err/lateral_near_success` 和 `err/yaw_deg_near_success` (观察对齐精度是否保持)

### 预期

若 H3_Reward 成立，在更合理的奖励引导下，智能体应该能学会勇敢且精准地完成最后的插入动作，成功率将迎来新一轮的爆发。

---

## 12. 实验 4：相机视角 ablation，向论文的侧视近场几何靠拢

### 假设 H4

当前单个 `60°` 俯视相机虽然能看见货叉和托盘，但不一定是最有利于近场插孔对齐的几何视角。  
论文使用侧视双相机，说明视角本身可能就是主因之一。

### 只改一个因素

在最佳预训练 + 最佳 Stage 1 设置基础上，只比较相机方案：

1. 当前单俯视相机
2. 单个更低、更前的 fork-centric 相机
3. 论文风格双侧相机

### 验证指标

- 近场预训练误差
- frozen RL 的 push-free 插入率
- 视频中叉齿与插孔相对关系是否更清楚

### 决策规则

- 若双侧或 fork-centric 明显提升，则进入主线替换。

---

## 14. 实验 5：按论文方式拆出 loading decision / lift

### 假设 H5

`lift` 不应该被当成 approach 训练的“自然副产物”。  
如果先把 approach-only 打通，再单独训练 `loading decision` 或 `lift trigger`，整体会更稳定。

### 只改一个因素

在已有 approach-only 成果上：

- 用仿真自动生成成功/失败样本
- 训练二分类 decision head：
  - 是否已满足 lift 条件

### 验证指标

- decision classifier 的 `acc / precision / recall`
- 结合 approach policy 后的最终 lift 成功率

### 决策规则

- 若 classifier 稳定，后续就不再要求主 RL policy 同时学会 approach + lift。

---

## 15. 实验 6：只有在前面都稳定后，才做组合验证

### 条件

只有当下面 3 项都分别验证有效后，才组合：

1. 2 动作 `approach-only` 有效
2. 新预训练方案有效
3. 新相机方案有效

### 组合目标

- 最佳相机
- 最佳预训练
- 2 动作 Stage 1
- 单独 decision / lift 头

### 最终 sim 目标

- `push_free_success_rate >= 0.60`
- `pallet_disp_xy_mean <= 0.05m`
- `phase/frac_lifted > 0`

这时才有资格谈“朝论文结果靠近”。

---

## 14. 我对下一步最具体的建议

按优先级，建议下一步只做下面 3 件事，不要同时乱改：

### 建议 1：立刻做实验 3 (重塑 Reward 函数)

原因：
- 昨晚的实验证明，视觉特征已经足够支撑对齐，但最后插入动作缺乏引导。
- 必须通过 Reward Shaping 打破 16.5% 的成功率天花板。

### 建议 2：全面拥抱 RRL 范式 (实验 2 的结论)

原因：
- 实验 2 已经证明了 RRL 范式（ResNet18 + ImageNet冻结）的极高效率和稳定性。
- 后续所有实验都应基于此基座，不再浪费时间进行任务特定的坐标回归预训练。

### 建议 3：暂缓特定任务预训练的数据收集

原因：
- 既然 RRL 范式已经跑通并追平了预训练方案，我们根本不需要去死磕“预训练 Y 误差 16cm”的问题。
- 只有在 RRL 范式彻底失败后，我们才需要回到备用路线。

---

## 15. 记录规范：后续如何持续迭代

后续每轮实验都必须新增一个独立 md，格式固定为：

```md
## 实验名
- 假设：
- 只改一个因素：
- 保持不变：
- 运行命令：
- 日志路径：
- 模型路径：
- 关键指标：
- 视频观察：
- 结论：
- 是否进入下一轮：
```

同时，本总计划文件应至少更新以下 3 项：

1. 当前最新结论
2. 哪个假设已经被证伪
3. 下一个最高优先级实验

---

## 16. 一句话总结

当前要想接近论文的成功路径，**不是死磕视觉预训练，而是先把问题拆对，并拥抱 RRL 范式**：

1. **先把 approach-only 训练做干净**（已在实验 1 验证有效）
2. **把“真成功”与“推土机假成功”区分开**（已在实验 0 验证有效）
3. **彻底拥抱 RRL 范式**：直接使用冻结的 ImageNet 通用特征，把物理理解交给下游 RL（已在实验 2 验证有效，成功追平复杂预训练）。
4. **死磕 Reward Shaping**：目前卡在 16.5% 的瓶颈，说明视觉已非短板，下一步必须在 RL 探索和最后插入阶段的奖励引导上做文章。
