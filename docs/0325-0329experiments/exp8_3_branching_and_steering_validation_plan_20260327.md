# `exp8.3` 下一阶段分支策略与 `steering` 验证执行计划

> 日期: `2026-03-27`
> 当前主分支: `exp/exp8_3_clean_insert_hold`
> 当前基线提交: `fcd76121f2c392d361aa5d8c7222778728525d3b`
> 关联文档:
> - `docs/0325-0329experiments/exp8_3_bonusw1p0_repro_iter100_unified_eval_20260327.md`
> - `docs/0325-0329experiments/exp8_3_bonusw1p0_iter100_early_bifurcation_analysis_20260327.md`

---

## 1. 先给结论

**要建立新分支，而且不应该把接下来的实验继续直接堆在 `exp/exp8_3_clean_insert_hold` 上。**

原因不是当前主线完全错，而是下一阶段要回答的问题已经变了：

- 之前的问题是: `bonusw=1.0` 这条路能不能产出强 checkpoint
- 现在的问题是: 为什么只有部分 `seed / repeat` 能稳定进好解

这个新问题需要两类互相解耦的工作:

1. **诊断类实验**
   不改 reward 主逻辑，先回答“策略到底有没有学会 steering / early bifurcation 到底怎么发生”
2. **干预类实验**
   在诊断结论基础上，再去改 curriculum / reward 入口 / PPO 方差

如果这两类工作继续混在一个分支里做，后面会很难归因:

- 是因为 reward 改对了？
- 还是因为 reset 变了？
- 还是因为 PPO 方差降了？
- 还是只是新加的 eval 脚本让我们看起来“更明白了”？

所以接下来推荐的做法是:

- 保留 `exp/exp8_3_clean_insert_hold` 作为当前主线历史
- 从当前基线切出**诊断分支**
- 诊断完成后，再从同一冻结基线切出**干预分支**

---

## 2. 当前最重要的新怀疑

你刚刚提出的怀疑非常关键:

**当前策略到底有没有真的学会 steering，还是只是学会了“在非常对齐的初始化里往前叉”？**

这是必须优先验证的，因为当前 `stage_1` 初始化确实非常窄:

- `x ∈ [-3.55, -3.25]`
- `y ∈ [-0.05, 0.05]`
- `yaw ∈ [-2°, 2°]`

在这种分布下，强 checkpoint 完全有可能靠“前推主导”拿到很好的 near-field 成绩，而并没有真正学会:

- 用 steering 修正横向误差
- 用 steering 修正偏航误差
- 在更宽的近场扰动下重新找回 clean insert 入口

这件事之所以重要，是因为它会直接改变我们对“第一主因”的判断：

- 如果强 checkpoint 在 `zero-steer` 下几乎不掉，那么当前第一问题就不是 PPO 超参数，而是**任务入口没有逼出 steering 技能**
- 如果强 checkpoint 明显依赖 steer，但仍然高方差，那么主问题更像是**reward 入口太窄 + PPO 方差放大**

所以，下一阶段的第一个目标不是立刻继续改 reward，而是先回答:

**当前强 checkpoint 的成功，究竟有多少来自 steering 能力，多少来自初始化已经几乎对准。**

---

## 3. 分支策略

### 3.1 Phase 0: 冻结当前基线

在开新分支之前，先把当前研究状态冻结成基线。

建议动作:

1. 把当前 `docs/0325-0329experiments/` 下的新文档、`scripts/run_exp83_bonusw1p0_repro_*`、`scripts/experiments/play_and_record_policy_input.py` 这类实验工具先整理并提交。
2. 不要提交 `outputs/` 里的产物。
3. 在提交后，打一个基线 tag，例如:

`exp8_3_bonusw1p0_iter100_baseline_20260327`

这样后面所有新分支都从同一个、可复现的点切出。

### 3.2 Phase 1: 诊断分支

推荐新分支:

`exp/exp8_3_steer_usage_diagnostics`

职责:

- 只做诊断和评估工具
- 不改 reward 主逻辑
- 不改 PPO 超参数
- 不改正式训练默认配置

这个分支的目标是回答:

- 强 checkpoint 是否真的使用 steering
- 相同 checkpoint / 相同 seed 的 replay 是否高度一致
- 当前 good checkpoint 的 basin 宽度有多大
- early bifurcation 到底是“不会 steering”还是“会 steering 但优化掉进了别的 basin”

### 3.3 Phase 2: 干预分支 A

如果诊断结论是:

**策略主要靠前推，真正 steering 学得不够**

则从 baseline 再切:

`exp/exp8_3_stage1_steering_curriculum`

职责:

- 只改 near-field reset / curriculum
- 不先改 reward
- 目标是逼出 steering 技能

### 3.4 Phase 3: 干预分支 B

如果诊断结论是:

**策略已经在用 steering，但 early reward 入口仍太窄**

则从 baseline 再切:

`exp/exp8_3_preinsert_entry_widen`

职责:

- 只改 pre-insert 阶段 shaping
- 不再回到强 gate / 强 penalty
- 目标是扩大 `good-clean` basin 的入口

### 3.5 Phase 4: 方差控制分支

只有在 A/B 里有一条路线明显更好之后，再切:

`exp/exp8_3_lowvar_ppo`

职责:

- 只动 PPO 方差相关超参数
- 不再同时改 reward / reset

---

## 4. 哪些改动是在“扩大 good basin”

下面这些属于**扩大好 basin 吸引域**的改动。

### 4.1 轻度放宽 near-field 的 `y / yaw`，保持 `x` 不动

目标:

- 让策略必须使用 steering
- 但不要把任务一下变成 wide reset

建议候选:

- `x` 保持 `[-3.55, -3.25]`
- `y: ±0.05 -> ±0.10`
- `yaw: ±2° -> ±4°`

为什么不先动 `x`:

- 当前最值得验证的是 steering，不是远距离 approach
- 改 `x` 会把“接近能力”和“纠偏能力”混在一起

### 4.2 在 pre-insert 阶段补更早的 alignment shaping

当前 `clean_insert_push_free_bonus` 生效偏晚，只有插入且 push-free 才明显给好 signal。

下一步更有价值的是在 **尚未插入，但已经 near-field** 时就鼓励:

- `dist_front` 持续下降
- `lateral / yaw` 误差持续下降
- 不 retreat

目标不是加大最终 success reward，而是让 run 在 `iter 10~15` 更容易进入 `good-clean` basin。

### 4.3 对 retreat attractor 做轻度抑制

从 `r2_seed44` 看，失败不是 dirty，而是 approach 崩掉后退远。

可以考虑轻度 shaping:

- 在 near-field 内，如果 `dist_front_mean` 明显回升，削弱正奖励
- 或对“已接近后又退回”的行为加很轻的惩罚

重点:

- 轻
- 只在 near-field 生效
- 不要变成新一轮强 penalty

### 4.4 继续保留 `bonusw=1.0` 作为主线参考

当前证据已经显示:

- `bonusw=1.0` 可以产出非常强的 checkpoint
- 问题在复现率，不在“有没有好解”

所以当前不应该又回到完全新的 reward 家族，而应该围绕这条线做 basin widening。

---

## 5. 哪些改动是在“降低 PPO 方差”

这些不是当前第一主因，但属于重要放大器。

### 5.1 增大 rollout 批量

优先候选:

- `num_envs: 64 -> 128`
或
- `num_steps_per_env: 64 -> 96 / 128`

目标:

- 降低 early update 的梯度噪声
- 减少不同 seed / repeat 在前 10~15 iter 就分流到不同 basin

### 5.2 降低更新激进度

候选:

- `learning_rate: 3e-4 -> 1.5e-4`
- `desired_kl: 0.008 -> 0.006`

目的:

- 减少一轮 update 把策略从“接近好 basin”直接推到 bad basin

### 5.3 保留多 seed 短训 + unified eval

这不是直接让策略更强，但它能更可靠地判断:

- 哪些配置真正在扩大 good basin
- 哪些配置只是偶然给某个 seed 提供了好运

---

## 6. 哪些改动现在不值得先做

### 6.1 不值得先继续加重 `gate / dirty penalty`

这条路线已经多次显示:

- 可以压 dirty insert
- 也很容易把 insertion 一起压没

### 6.2 不值得先回 wide reset

原因:

- 当前连 steering 是否真的学会都还没确认
- 现在回 wide reset，会把问题重新缠在一起

### 6.3 不值得先继续扫很多 `bonus weight`

在 steering 问题没厘清前，再细扫 `0.8 / 1.2 / 1.4` 这类权重，信息增益不高。

### 6.4 不值得直接上更长 `400 / 800 iter`

现在缺的是:

- 早期为什么分叉
- steering 到底有没有学会

不是更长的训练时间。

---

## 7. 诊断分支的完整执行计划

## 7.1 目标

回答 4 个问题:

1. `r1_seed42 / r1_seed44` 这类强 checkpoint 是否真的在用 steering？
2. 如果把 `steer` 强行置零，性能会掉多少？
3. 如果增加一点 `y / yaw` 偏差，强 checkpoint 还能纠回来吗？
4. 同 checkpoint + 同 seed 重放时，初始化和轨迹是否高度一致？

## 7.2 建议改动文件

建议新增，不要直接污染当前主 eval 脚本:

- `scripts/eval_exp83_zero_steer_checkpoint.py`
- `scripts/eval_exp83_misalignment_grid.py`
- `scripts/eval_exp83_action_usage.py`
- `scripts/run_exp83_steer_usage_diag_suite.sh`

如果需要复用现有 unified eval 逻辑，也可以从:

- `scripts/eval_exp83_checkpoint.py`

拷贝出诊断版脚本，而不是直接在原脚本里堆很多分支。

## 7.3 具体实验

### D1. Zero-Steer Eval

对象:

- 强 run:
  - `r1_seed42`
  - `r1_seed44`
- 失败 run:
  - `r2_seed44`
  - `r1_seed43`
  - `r2_seed43`

做法:

- 正常 deterministic eval
- 再做一版 `steer = 0` 的 deterministic eval

关键指标:

- `success_rate_ep`
- `ever_inserted_push_free_rate`
- `ever_clean_insert_ready_rate`
- `mean_max_pallet_disp_xy`

判据:

- 如果强 run 在 `zero-steer` 下几乎不掉，说明当前策略主要靠前推
- 如果强 run 大幅掉，说明 steering 已经是关键技能

### D2. Misalignment Grid Eval

对象:

- `r1_seed42`
- `r1_seed44`

grid 建议:

- `y ∈ {0, ±0.05, ±0.10, ±0.15}`
- `yaw ∈ {0°, ±2°, ±4°, ±6°}`

注意:

- `x` 保持当前近场区间中心附近
- 先只测 near-field 下的纠偏能力，不测远距离

关键输出:

- success heatmap
- push_free heatmap
- zero-steer vs normal-steer 对照 heatmap

判据:

- 如果 normal-steer 比 zero-steer 的可行区明显大，说明 steering 真有学到
- 如果两者差不多，说明策略主要依赖 forward-only

### D3. Action Usage Logging

对象:

- 同 D2

记录:

- `mean_abs_drive`
- `mean_abs_steer`
- `max_abs_steer`
- `steer sign flip count`
- `corr(steer, y_err)`
- `corr(steer, yaw_err)`

目的:

- 给“到底会不会 steering”提供直接证据

### D4. Replay Repro Check

对象:

- `r1_seed42`

做法:

- 同 checkpoint
- 同可视化脚本
- 同 `--seed`
- 连续录两次

比较:

- 初始 `x / y / yaw`
- 前 `50~100` 步动作
- 前 `50~100` 步关键状态

目的:

- 明确可视化 replay 的一致性边界
- 避免后续误把 replay 偶然差异当成策略问题

## 7.4 诊断分支的停止条件

满足以下任一结论，就可以停止 Phase 1:

### 结论 A

强 checkpoint 在 `zero-steer` 下也很强。

解释:

- 当前 stage1 主要学到的是“往前叉”
- steering 技能没有被课程真正逼出来

后续进入:

`exp/exp8_3_stage1_steering_curriculum`

### 结论 B

强 checkpoint 明显依赖 steering，但 basin 仍然窄。

解释:

- steering 学到了一部分
- 主问题转为 reward 入口太窄 / PPO 方差大

后续进入:

`exp/exp8_3_preinsert_entry_widen`

---

## 8. 干预分支 A: steering curriculum 分支计划

触发条件:

- D1/D2 显示强 checkpoint 对 steer 依赖弱
或
- zero-steer 和 normal-steer 差异小

分支:

`exp/exp8_3_stage1_steering_curriculum`

### A1. 只放宽 `y / yaw`

配置候选:

- `y: ±0.05 -> ±0.10`
- `yaw: ±2° -> ±4°`

不动:

- `x`
- reward
- PPO 超参数

### A2. 训练计划

1. `3 seeds x 50 iter`
2. unified eval
3. 如果 steering 使用明显提升，再上 `3 seeds x 100 iter`

### A3. 成功判据

- `zero-steer` 对比明显变差
- normal-steer 的 grid 可行区明显扩大
- 同时 unified eval 不比当前 `bonusw=1.0` 最强 run 差太多

---

## 9. 干预分支 B: pre-insert entry widen 分支计划

触发条件:

- D1/D2 显示 steering 已经真实使用
- 但 early bifurcation 仍然严重

分支:

`exp/exp8_3_preinsert_entry_widen`

### B1. 只改 pre-insert shaping

重点:

- near-field 时更早奖励 `dist_front` 持续下降
- near-field 时更早奖励 `lateral / yaw` 误差收敛
- 抑制明显 retreat

不做:

- 重启强 gate / 强 penalty
- 大改 clean bonus 权重

### B2. 训练计划

1. `seed43 + seed44` stress short run: `50 iter`
2. 如果 stress seed 改善，再补 `seed42`
3. 再做 unified eval

### B3. 成功判据

- `r2_seed44` 这类 no-insert attractor 减少
- `r1_seed43` 这类 dirty attractor 比例下降
- good run 数量增加，而不是只把所有 run 压成保守

---

## 10. 干预分支 C: PPO 方差控制计划

触发条件:

- A 或 B 有明显更好的训练主线
- 但跨 repeat 方差依然大

分支:

`exp/exp8_3_lowvar_ppo`

### C1. 先试两类轻改动

- `num_envs 64 -> 128`
- `learning_rate 3e-4 -> 1.5e-4`
- `desired_kl 0.008 -> 0.006`

注意:

- 一次只动一个小包
- 不再同时改 reward / reset

### C2. 判据

- 强 run 保持
- 失败 run 比例下降
- `seed43 / seed44` 的 bifurcation 减轻

---

## 11. 推荐实际执行顺序

### Step 0

整理并提交当前未提交的文档和脚本，打 baseline tag。

### Step 1

切 `exp/exp8_3_steer_usage_diagnostics`

### Step 2

做:

- `D1 zero-steer eval`
- `D2 misalignment grid eval`
- `D3 action logging`
- `D4 replay repro check`

### Step 3

按诊断结果二选一:

- steering 不足 -> 切 `exp/exp8_3_stage1_steering_curriculum`
- steering 已学会 -> 切 `exp/exp8_3_preinsert_entry_widen`

### Step 4

只有在主线明显改善后，再切 `exp/exp8_3_lowvar_ppo`

---

## 12. 一句话判断

**当前最合理的策略不是继续在主分支上直接试新 reward，而是先切一个诊断分支，把“有没有真正学会 steering”这件事查清楚；然后再按诊断结果，去做 curriculum widening 或 pre-insert entry widening。**
