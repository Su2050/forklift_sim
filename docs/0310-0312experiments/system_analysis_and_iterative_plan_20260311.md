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

- `docs/0310-0312experiments/scratch_baseline_256x256_result_20260311.md`
- `docs/0310-0312experiments/finetune_256x256_freeze50_result_20260311.md`
- `docs/0310-0312experiments/experiment_log_20260311_night.md`
- `docs/0310-0312experiments/experiment_log_20260312_morning.md`
- `docs/0310-0312experiments/experiment_3x_reference_trajectory_reward_plan_20260312.md`
- 最新日志 `logs/20260312_100059_train_exp3_reward_shaping_rrl.log`

### 2.1 当前没有一条可以算“论文式成功训练”的路线

原因如下：

1. **`256x256 scratch` 全程为 0**
   - 说明纯 RL 从零开始学习高分辨率视觉输入在当前任务上不可行。

2. **“坐标回归预训练”路线已正式降级**
   - `MobileNetV3 任务预训练+解冻微调` 与 `ResNet18 ImageNet特征+全程冻结` 的最终上限几乎一致，二者都卡在 `push-free ≈ 16.5%`。
   - 这说明特定任务的坐标回归预训练不是当前主瓶颈，`RRL` 才是更合理、更简洁的主线。

3. **`exp3 reward shaping v1` 还没有打通 `approach -> insert -> hold`**
   - 到 `iteration 348` 左右，`push_free_success_rate_total` 仅约 `0.53%`，`push_free_insert_rate_total` 仅约 `0.60%`。
   - 同时 `diag/pallet_disp_xy_mean` 仍约 `0.08m ~ 0.11m`，说明虽然出现了极少量近成功或浅插入，但距离“稳定、无推盘的插入保持成功”还很远。

4. **`loading decision / lift` 仍未开始独立打通**
   - 当前即便在 `approach-only` 主线中，也还没有形成论文那种“接近策略 + 单独决策举升”的完整 pipeline。

### 2.2 当前真正的系统状态

当前更准确的系统状态不是“完全看不见”，而是：

- 已确认 `2 动作 approach-only` 与 `RRL (ResNet18 + ImageNet + 全程冻结)` 是当前正确主线
- 能在相当一部分 episode 中形成中等质量对齐，`phase/frac_aligned` 可达到 `~0.35 ~ 0.63`
- 在 `exp3 reward shaping v1` 中，`err/dist_front_mean` 已下降到约 `0.22m`，说明叉尖能够更频繁地开到托盘口前
- 但这种接近还**不能稳定转化为插入与保持**，`phase/frac_inserted` 基本仍接近 `0`，`phase/hold_counter_max` 仍为 `0`
- 托盘位移仍不可忽略，`diag/pallet_disp_xy_mean` 约 `0.08m ~ 0.11m`
- `lift / loading decision` 仍完全没打通

### 2.3 当前最重要的负面证据

1. **特定任务预训练不是当前主要矛盾**
   - 经过实验 1 与实验 2 对比，已经可以把“要不要继续死磕坐标回归预训练”视为已回答的问题：当前主线应固定为 `RRL`。

2. **当前 Reward v1 只解决了“更靠近”，没有解决“沿可插入路径 commit”**
   - 最新 `exp3` 日志表明：近场误差和局部接近确实比之前更好，但稳定插入与保持几乎没有发生。
   - 这说明当前 Reward 仍缺少“参考走廊 -> commit -> insert”的显式分解。

3. **问题已不再是动作维度，而是 Reward 结构与诊断粒度**
   - `2 动作 approach-only` 已验证有效，不应继续把“3 动作探索噪声”当成当前主障碍。
   - 当前真正缺的是：更像论文那样的几何先验、分阶段 Reward、以及能区分 `approach / commit / insert` 的过程诊断。

---

## 3. 与论文《Visual-Based Forklift Learning System Enabling Zero-Shot Sim2Real Without Real-World Data》的差距

### 3.1 论文做对了什么

论文的关键成功点不是单一一个 ResNet，而是以下组合：

1. **使用标准 ImageNet 预训练 ResNet**
2. **使用双侧相机，强调叉齿与托盘开口的近场几何关系**
3. **把任务拆成 approach policy 和 loading decision**
4. **approach policy 只学驱动与转向，不把 lift 混进同一个 RL policy**
5. **在 Reward 中使用基于参考轨迹的几何先验**
   - 原文明确使用了基于 `clothoid` 近似的 reference trajectory 来计算正奖励。
6. **使用 photorealistic sim + domain randomization**
7. **reward/critic 允许使用 privileged information**

### 3.2 我们与论文的核心差距

经过实验 1 和实验 2 的修正后，我们已经在两点上向论文靠拢：

- `2 动作 approach-only`
- `RRL` 范式（ImageNet 特征 + 冻结骨干）

当前项目与论文相比，主要还差在 6 个地方：

1. **Reward 仍缺少“参考轨迹走廊”这一层全局几何先验**
   - 当前主线 Reward 仍以局部中心线误差、局部门控和插入深度为主。
   - 论文则显式利用 reference trajectory 引导 forklift 以“可插入的方式”接近托盘。

2. **相机几何仍与论文不同**
   - 当前主线是单个 `60°` 俯视相机。
   - 论文更强调从侧边观察叉齿与托盘开口关系。

3. **`approach -> commit -> insert` 还没有真正打通**
   - 当前 `exp3 v1` 说明：我们已经不只是“看不见”，而是“能靠近、能对齐一些，但不会稳定 commit 到插入”。

4. **`loading decision / lift` 仍未单独建模**
   - 论文是 `approach policy` 和 `loading decision` 两阶段。
   - 我们目前仍停留在先把 `approach` 打通的阶段。

5. **过程诊断还不够细**
   - `push-free` 指标已经补上，但还缺少能直接回答“卡在走廊、卡在 commit、还是卡在 insert”的过程指标。

6. **domain randomization 范围仍较窄**
   - 论文除了外观随机化，还对观测速度、动作执行等做扰动。
   - 这部分对后续 sim2real 仍有借鉴价值。

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

4. `phase/frac_aligned`

5. `phase/frac_inserted`

6. `phase/hold_counter_max`

7. `diag/near_success_frac`

8. `err/lateral_near_success`

9. `err/yaw_deg_near_success`

10. `err/dist_front_mean`

11. `milestone/hit_approach`

12. 若启用 `3.x` 参考轨迹方案，增加：
    - `traj/corridor_frac`
    - `traj/commit_gate_mean`
    - `traj/d_traj_mean`

13. `phase/frac_lifted`

说明：

- `1~3` 是最终结果指标
- `4~12` 是过程诊断指标，用于判断策略到底卡在 `approach`、`commit` 还是 `insert`

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

## 11. 实验 3：Reward 重构主线 (已启动，需拆成 3.1 ~ 3.4)

### 假设 H3_Reward

**重大发现与假设修正**：
经过 2026-03-11 晚间的实验 1 和实验 2 对比，我们发现无论使用复杂的 MobileNetV3 坐标回归预训练，还是使用简单的 ResNet18 冻结通用特征 (RRL 范式)，最终的无碰撞成功率都死死卡在 **16.5%** 左右。
这说明：**当前的瓶颈已经不在视觉感知端，而在于 RL 的探索策略和 Reward 函数的塑造。** 智能体在接近目标时，缺乏足够的引导来完成最后的“精准插入”动作。

### 11.1 当前状态与判断修正

最新运行中的 `exp3 reward shaping` 日志 `logs/20260312_100059_train_exp3_reward_shaping_rrl.log` 表明：

- 到 `iteration 348` 左右，`push_free_success_rate_total` 仅约 `0.53%`
- `push_free_insert_rate_total` 仅约 `0.60%`
- `diag/pallet_disp_xy_mean` 仍约 `0.08m ~ 0.11m`
- `err/lateral_near_success` 约 `0.14m`
- `err/yaw_deg_near_success` 约 `8° ~ 9°`
- `phase/frac_inserted` 基本仍接近 `0`

这说明：

- 当前 Reward v1 **不是完全无效**
- 它确实让叉尖更常到达托盘前沿附近，也让 near-success 区域偶发出现
- 但它还没有把“更靠近”稳定转化成“可插入、可保持、可复现的 push-free 成功”

因此，**实验 3 不应继续作为一个“大包 Reward 实验”推进，而应拆成严格单因素的 `3.1 ~ 3.4`**。

详细设计见：

- `docs/0310-0312experiments/experiment_3x_reference_trajectory_reward_plan_20260312.md`

### 保持不变

- 实验 1 确立的 2 动作 `approach-only` 策略
- 实验 2 确立的 RRL 范式 (ResNet18 + ImageNet权重 + 全程冻结)

### 11.2 实验 3.1：参考轨迹走廊替代远场距离带

#### 假设

当前 Reward 的远场引导仍然过于局部，导致策略学会了“对齐一些”，却没有学会“沿可插入路径推进”。  
如果引入类似论文中 reference trajectory 的“走廊先验”，`approach` 与后续 `commit` 会更顺。

#### 只改一个因素

- 用参考轨迹走廊 shaping 替代当前远场 `phi1`
- 第一版采用 `clothoid-lite / trajectory-lite`，不强求严格数学 clothoid

#### 保持不变

- 当前 `2 动作 approach-only`
- 当前 `RRL` 主线
- 当前相机方案
- 当前 success / hold 判据

#### 验证指标

- `phase/frac_aligned`
- `err/dist_front_mean`
- `milestone/hit_approach`
- `traj/corridor_frac`（新增）
- `traj/d_traj_mean`（新增）

#### 决策规则

- 若 `approach` 与进入走廊相关指标明显改善，再继续做 `实验 3.2`
- 若几乎无改善，先排查轨迹几何与走廊宽度，而不是立即切相机

### 11.3 实验 3.2：近场 commit 奖励

#### 假设

即使进入了走廊，智能体仍可能在托盘前沿犹豫不前。  
如果在近场显式奖励 `dist_front` 下降与 `insert_norm` 上升，`insert` 会更容易破零。

#### 只改一个因素

- 在 `实验 3.1` 最佳配置上，只新增近场 `commit` 奖励

#### 保持不变

- 参考轨迹走廊
- 当前相机与动作空间
- 当前 success / hold 判据

#### 验证指标

- `phase/frac_inserted`
- `push_free_insert_rate`
- `err/dist_front_mean`
- `traj/commit_gate_mean`（新增）

#### 决策规则

- 若 `insert` 仍完全不破零，说明当前主要问题仍是“看见了但不敢插”
- 若 `insert` 破零但 `push_free` 不升，则进入 `实验 3.3`

### 11.4 实验 3.3：条件化推盘惩罚

#### 假设

一旦 `commit` 打开，策略很容易再次回到“推着托盘走”的老问题。  
如果把推盘惩罚改成“远场重、近场轻、死区再加重、但永远非零”的条件化惩罚，能够在探索与防 bulldozer 之间取得更好平衡。

#### 只改一个因素

- 只重构 `pen_pallet_push`

#### 保持不变

- `实验 3.1` 的轨迹走廊
- `实验 3.2` 的近场 commit 奖励
- 其他 success / hold 逻辑

#### 验证指标

- `push_free_insert_rate`
- `push_free_success_rate`
- `pallet_disp_xy_mean / p95`
- 视频中的托盘位移

#### 决策规则

- 若插入率保住、位移下降，则条件化惩罚成立
- 若插入再次消失，说明惩罚仍然过重

### 11.5 实验 3.4：死区撤退 / 重试奖励

#### 假设

即使前 3 步都有效，策略仍可能卡在“浅插错位 -> 顶住 -> 不退不进”的死区。  
如果显式奖励从死区退出并重试，最终 `hold` 成功率会更容易提升。

#### 只改一个因素

- 只新增死区撤退 / 重试奖励

#### 保持不变

- `实验 3.1 ~ 3.3` 最佳配置

#### 验证指标

- `diag/dead_zone_frac`（新增）
- `diag/dead_zone_escape_frac`（新增）
- `phase/hold_counter_max`
- `push_free_success_rate`

#### 决策规则

- 若视频中开始出现“退一点再对准再插”的行为，且 `hold_counter_max` 提升，则该项成立
- 若 `3.1 ~ 3.4` 都无明显帮助，再提高相机视角 ablation 的优先级

---

## 12. 实验 4：相机视角 ablation，向论文的侧视近场几何靠拢

### 假设 H4

当前单个 `60°` 俯视相机虽然能看见货叉和托盘，但不一定是最有利于近场插孔对齐的几何视角。  
论文使用侧视双相机，说明视角本身可能就是主因之一。

### 只改一个因素

在 `实验 3.x` 的最佳 Reward 配置 + `2 动作 approach-only` + `RRL` 冻结骨干基础上，只比较相机方案：

1. 当前单俯视相机
2. 单个更低、更前的 fork-centric 相机
3. 论文风格双侧相机

### 验证指标

- `push_free_insert_rate / push_free_success_rate`
- `phase/frac_inserted`
- `err/lateral_near_success / err/yaw_deg_near_success`
- 视频中叉齿与插孔相对关系是否更清楚

### 决策规则

- 只有在 `实验 3.x` 分解后仍然卡住时，才提高本实验优先级
- 若双侧或 fork-centric 明显提升，则进入主线替换

---

## 13. 实验 5：按论文方式拆出 loading decision / lift

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

## 14. 实验 6：只有在前面都稳定后，才做组合验证

### 条件

只有当下面 4 项都分别验证有效后，才组合：

1. 2 动作 `approach-only` 有效
2. `实验 3.x` 中至少一项 Reward 主改动稳定有效
3. 新相机方案有效（若做了视角 ablation）
4. `loading decision / lift` 头有效

### 组合目标

- 最佳相机
- `RRL` 冻结骨干
- 最佳 `3.x` Reward 配置
- 2 动作 Stage 1
- 单独 decision / lift 头

### 最终 sim 目标

- `push_free_success_rate >= 0.60`
- `pallet_disp_xy_mean <= 0.05m`
- `phase/frac_lifted > 0`

这时才有资格谈“朝论文结果靠近”。

---

## 15. 我对下一步最具体的建议

按优先级，建议下一步只做下面 3 件事，不要同时乱改：

### 建议 1：立刻做实验 3.1（参考轨迹走廊）

原因：
- 最新 `exp3 v1` 已经证明，单纯“保持对齐 + 插入深度 + 推盘惩罚”的局部 shaping 不足以稳定打通插入。
- 当前更缺的是“沿可插入路径接近”的全局几何先验。

### 建议 2：实验 3 必须严格按 `3.1 -> 3.2 -> 3.3 -> 3.4` 分拆

原因：
- 现在最怕的不是 reward 太弱，而是一次改太多后无法判断到底哪一项在起作用。
- 只有严格做单因素，后面的相机和 lift 决策实验才有解释力。

### 建议 3：继续锁定 `RRL + 2 动作 approach-only` 为统一基座，暂缓无关分支

原因：
- 当前不应再回头扩张“任务特定预训练”路线。
- 相机 ablation 与 `loading decision` 也应排在 `3.x` Reward 主线后面。

---

## 16. 记录规范：后续如何持续迭代

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

## 17. 一句话总结

当前要想接近论文的成功路径，**不是死磕视觉预训练，而是先把问题拆对，并拥抱 RRL 范式**：

1. **先把 approach-only 训练做干净**（已在实验 1 验证有效）
2. **把“真成功”与“推土机假成功”区分开**（已在实验 0 验证有效）
3. **彻底拥抱 RRL 范式**：直接使用冻结的 ImageNet 通用特征，把物理理解交给下游 RL（已在实验 2 验证有效，成功追平复杂预训练）。
4. **把 Reward 主线从“大包调参”升级为 `3.1 ~ 3.4` 分阶段路线**：先补参考轨迹走廊，再补近场 commit、条件化推盘惩罚和死区重试。
5. **只有当 `3.x` 这条线仍失败时，再提高相机视角 ablation 的优先级**。
