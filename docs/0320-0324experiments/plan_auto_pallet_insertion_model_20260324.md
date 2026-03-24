# 接下来怎么推进到“能自动插托盘”的模型

> 日期: `2026-03-24`
> 适用范围: `exp8.3` 当前主线
> 当前建议主线: `progress_potential + disp_gate`

---

## 0. 分支策略

这份计划里涉及的修改，**不要直接在 `master` 上做**。

推荐的基线分支是:

- `exp/exp8_3_geom_validation_b0prime`
- 基线提交建议固定为: `aa7e32a303225fa44a396992f99be3913bf135b0`

原因:

- 这是当前正在工作的分支
- 它已经包含了 `exp8.3` 的几何修正、runtime U0/U1 检查和最近分析所依赖的诊断能力
- 后续要修的 `hold/success` 接线问题，本质上属于当前 `exp8.3` 主线的问题，放在这个分支上继续最顺

在正式开始切分支之前，建议先做两件事:

1. 确保工作树干净
2. 把当前基线固定到一个明确的 commit 或 tag

推荐做法:

- 先在 `exp/exp8_3_geom_validation_b0prime` 上整理工作树
- 确认基线提交为 `aa7e32a303225fa44a396992f99be3913bf135b0`
- 可选地打一个轻量 tag，例如 `exp8_3_plan_baseline_20260324`

这样后续每个实验分支都能明确回答:

- 它是从哪个基线切出来的
- 它和其他实验分支是否 truly 可比

不建议把后续修改放回这些历史实验分支中继续推进:

- `exp/exp8_3_g2_rd_target_success_center`
- `exp/exp8_3_g2b_target_family_success_center`
- `exp/exp8_3_g3_traj_and_target_success_center`

这些分支更适合作为**对照和结果留档**，不适合作为当前主开发线。

### 推荐执行方式

如果希望保持提交历史清晰，建议采用**分层分支**，而不是把 3 个分支都直接从同一基线切出来。

推荐关系如下:

`exp/exp8_3_geom_validation_b0prime @ aa7e32a303225fa44a396992f99be3913bf135b0`
-> `exp/exp8_3_hold_success_fix`
-> `exp8_3_phase1_ready` (tag 或固定 commit)
-> `exp/exp8_3_clean_insert_hold`
-> `exp/exp8_3_teacher_nonvisual`

解释:

1. `exp/exp8_3_hold_success_fix`
   - 用于修 `_hold_counter` / `success` / cfg 接线
   - 对应本计划的 Phase 1

2. `exp8_3_phase1_ready`
   - 不是开发分支，而是一个“冻结基线”
   - 表示 Phase 1 已完成，success/hold 逻辑已校准
   - 后续所有 reward / teacher 分流实验都应该从这里切

3. `exp/exp8_3_clean_insert_hold`
   - 必须从 `exp8_3_phase1_ready` 切出
   - 用于继续做 `progress_potential + disp_gate` 主线上的 reward / gate / near-field curriculum 试验
   - 对应本计划的 Phase 2 和 Phase 4

4. `exp/exp8_3_teacher_nonvisual`
   - 也必须从 `exp8_3_phase1_ready` 切出
   - 用于做 non-visual teacher 分流实验
   - 对应本计划的 Phase 3

注意:

- `clean_insert_hold` 不应直接从 `exp/exp8_3_geom_validation_b0prime` 切出
- `teacher_nonvisual` 也不应直接从 `exp/exp8_3_geom_validation_b0prime` 切出
- 两者都应该建立在“Phase 1 已经修好逻辑”的固定基线上

如果你想减少分支数量，也可以采用更简化的方式:

- Phase 1 先直接在 `exp/exp8_3_geom_validation_b0prime` 上修代码
- Phase 1 完成后立刻打一个固定 tag（例如 `exp8_3_phase1_ready`）
- 再从这个 tag 切出 `clean_insert_hold` 和 `teacher_nonvisual`

### 每个建议分支的第一批提交

下面不是最终唯一方案，而是建议的**第一批提交粒度**。目标是让每个分支的前几次 commit 都能独立验证，不把“逻辑修复”“reward 试验”“teacher 分流”混在一起。

#### `exp/exp8_3_hold_success_fix`

这个分支只做一件事:

- 把当前运行中的 hold / success / done 逻辑真正接到 cfg

建议第一批提交拆成 3 个 commit:

1. `wire hold/success gates to cfg`
   - 让 `_hold_counter` 真正使用:
     - `max_lateral_err_m`
     - `max_yaw_err_deg`
     - `stage1_success_without_lift`
     - `tip gate`
   - 清理当前硬编码的 `y_err < 0.1`、`yaw_err_deg < 5.0` 逻辑
   - 明确 Stage 1 与后续阶段的 success 判据差异

2. `enable hold decay and hysteresis in runtime logic`
   - 把 cfg 里已有但未真正接线的逻辑接上:
     - `hold_counter_decay`
     - `hysteresis_ratio`
     - `insert_exit_epsilon`
     - `lift_exit_epsilon`
     - `tip_align_exit_m`
   - 保证 hold 不再因为单步物理抖动直接硬清零

3. `add sanity checks and logging for hold/success`
   - 增加最小必要的日志或断言
   - 确保以下指标与实际逻辑一致:
     - `phase/frac_success`
     - `phase/frac_success_geom_strict`
     - `diag/max_hold_counter`
     - `diag/success_term_frac`
   - 如果可能，补一个纯逻辑单测或轻量 preflight 检查

这个分支第一批提交完成后的验收标准:

- 不追求 success 立刻上升
- 只要求“配置怎么写，运行就怎么判”
- 日志指标与代码定义不再打架

#### `exp/exp8_3_clean_insert_hold`

这个分支**必须建立在 `exp8_3_phase1_ready` 之后**，目标不是修逻辑，而是开始针对:

- clean insert
- push-free insert
- stable hold

来改训练信号。

建议第一批提交拆成 3 个 commit:

1. `gate post-insert progress reward by clean alignment`
   - 对插入后的 `r_d`、progress 或 potential 奖励加 gate
   - gate 至少绑定:
     - `center alignment`
     - `tip constraint`
     - 必要时加 `push_free` 约束
   - 原则是只奖励“干净插入”，不奖励“斜着顶进去”

2. `add near-field curriculum config and run scripts`
   - 增加 near-field curriculum 的独立配置或启动脚本
   - 把课程目标收敛到:
     - clean insert
     - hold 起步
   - 不在第一批里就恢复 wide reset

3. `add focused diagnostics for clean insert vs push-insert`
   - 补充或强化对这几类指标的监控:
     - `diag/pallet_disp_xy_mean`
     - `phase/frac_inserted`
     - `phase/frac_tip_constraint_ok`
     - `diag/max_hold_counter`
     - 近场下的 lateral / yaw 误差
   - 让日志能区分“真插入”与“推盘型假插入”

这个分支第一批提交完成后的验收标准:

- 不再只看“是否首次出现 `hold_counter > 0`”
- 至少要满足以下更稳的组合判据:
  - `diag/max_hold_counter` 连续多个日志窗口非零
  - `phase/frac_success_geom_strict` 连续非零，而不是单点抖动
  - `diag/pallet_disp_xy_mean` 不随插入提升而明显恶化
- 插入增加时，推盘位移不能同步明显恶化
- reward 改动是否把行为推向更干净的插入，而不是更激进的撞盘

#### `exp/exp8_3_teacher_nonvisual`

这个分支不应该和 reward 主线混着改，它的目标非常单一:

- 尽快判断当前问题主要在任务定义，还是在视觉表征

这个分支**也必须从 `exp8_3_phase1_ready` 切出**，不要从 `clean_insert_hold` 再继续切。

原因:

- 它的目的不是继承 reward 主线改动
- 而是用与视觉主线**同一份逻辑基线**做分流诊断
- 如果它建立在 `clean_insert_hold` 之后，就会把“reward 改动”和“视觉分流”混在一起，结论不干净

建议第一批提交拆成 3 个 commit:

1. `add non-visual teacher training config`
   - 基于修正后的 env 增加 non-visual teacher 配置
   - 保持 success / hold / reward 逻辑与视觉分支一致
   - 唯一变化应尽量集中在 observation 侧

2. `add teacher train and eval entrypoints`
   - 增加 teacher 训练脚本
   - 增加最小评估脚本或日志提取脚本
   - 保证 teacher 与视觉 run 的指标可直接对照

3. `add teacher-vs-vision comparison template`
   - 增加一份对比记录模板或结果汇总脚本
   - 至少对齐以下指标:
     - `frac_inserted`
     - `max_hold_counter`
     - `frac_success`
     - `pallet_disp_xy_mean`
   - 保证实验跑完后能立刻回答“teacher 会不会”

这个分支第一批提交完成后的验收标准:

- 不是 teacher 一定要立刻成功
- 而是要尽快得到一个清晰结论:
  - teacher 也学不出来 -> 问题主要在 task / reward / geometry
  - teacher 能学出来 -> 问题更多在视觉或表征

时序要求:

- 这个分支不应该等到 Phase 2 完全结束后再开始
- 更合理的安排是: **Phase 1 完成后，Teacher 分流与 clean-insert 主线并行推进**
- 否则“尽快判断问题归因”这个目标就会被拖慢

### 结论

这份计划默认采用的分支顺序是:

`exp/exp8_3_geom_validation_b0prime`
-> `exp/exp8_3_hold_success_fix`
-> `exp8_3_phase1_ready`
-> `exp/exp8_3_clean_insert_hold`
-> `exp/exp8_3_teacher_nonvisual`

其中:

- `clean_insert_hold` 与 `teacher_nonvisual` 是**并行分支**
- 二者共享同一个 `Phase 1 ready` 基线
- `Phase 4` 只建立在 `clean_insert_hold` 的结果之上，不依赖 teacher 分支

---

## 1. 当前判断

最近这批实验已经说明一个关键事实:

- 现在的主问题已经不是“完全不会靠近托盘”
- 而是“到了近场后，插入不够干净，hold/success 没真正形成稳定闭环”

因此，下一阶段不应该继续把主要精力放在“更快靠近”上，而应该转向:

1. 修正环境逻辑，让 success / hold 的定义与配置一致
2. 强化“干净插入、稳定保持”的训练信号
3. 用更有控制性的课程，把 clean insert + hold 单独打通
4. 尽快分离“视觉问题”与“任务定义问题”

---

## 2. 总目标

训练出一个可以**自动完成托盘插入**的模型，并且这个模型满足以下要求:

- 能稳定进入近场
- 能以较小偏航和较小横向误差完成插入
- 不依赖推盘式“假插入”
- 能触发并维持 hold
- 后续可以自然扩展到 lift / loading decision

当前阶段的短期目标不是“直接端到端完成举升”，而是先把:

`approach -> clean insert -> hold`

这条链路打通。

---

## 3. 推进路线

### 第一步：先修代码，不先加新 reward

这是当前最优先的事情。

要做的不是继续调 reward，而是先把 actual hold / success 逻辑接到 cfg 上，确保训练运行时真正使用这些配置项:

- `max_lateral_err_m`
- `max_yaw_err_deg`
- `stage1_success_without_lift`
- `tip gate`
- `decay / hysteresis`

核心原因:

- 如果 `_hold_counter` 还在走硬编码条件，而不是走 cfg
- 那么后续所有“放宽阈值”“课程学习”“门控实验”都会变成空转
- 日志里看到的配置，和训练时真正生效的逻辑就不是一回事

这一阶段的目标不是提升指标，而是保证:

1. 配置写了什么，运行就真的用什么
2. `phase/frac_success`、`diag/max_hold_counter` 的解释和代码一致
3. Stage 1 的 `without_lift` 逻辑真的可控

### 第二步：不回退分支，继续沿 `progress_potential + disp_gate` 往前走

不建议退回 `unify_traj_and_target_family` 之前的旧路线。

原因很直接:

- 最近日志已经证明 `progress_potential + disp_gate` 比前面的方案更能产生插入信号
- 它虽然还没成功，但方向上已经比老方案更接近正确行为

但训练重点需要变化:

- 不再主要关注“怎么更快靠近”
- 改为关注“插入后如何稳定 hold、如何避免推盘”

建议的 reward 调整方向:

1. 插入后对 `r_d` 或 progress 奖励增加 alignment gate
2. 插入后对 tip 对齐增加 gate
3. 只奖励“干净插入”
4. 不奖励“斜着顶进去”“靠推盘拿进展”

最核心的一条原则是:

> 插入后的正奖励必须和 clean alignment 绑定，而不是只和前进深度绑定。

---

## 4. 训练流程建议

### 第三步：训练流程分两段

建议把训练拆成两个阶段，而不是继续直接 wide reset 长跑。

#### 阶段 A：近场课程，先打通 clean insert + hold

目标:

- 从更容易的初始分布出发
- 先让策略学会“干净地插进去并保持住”

这一阶段重点看:

- `phase/frac_inserted`
- `phase/frac_center_aligned_cfg`
- `phase/frac_tip_constraint_ok`
- `diag/max_hold_counter`
- `phase/frac_success`
- `diag/pallet_disp_xy_mean`

判断标准:

- 如果 near-field 下仍然完全没有 hold
- 说明问题主要不在 reset 太远
- 而在 reward / success 定义 / 几何一致性

#### 阶段 B：回到 wide reset

在近场课程打通以后，再放回当前的 wide reset，例如:

- 距离 `1.5 ~ 2.5m`
- 横向 `±0.5m`
- 偏航 `±15°`

原因:

- 从最近日志看，wide reset 现在不会阻止策略接近托盘
- 但它会明显放大近场不稳定
- 如果 clean insert + hold 本身还没学会，就不适合直接 wide reset 长跑

因此更合理的顺序是:

`先学会近场 clean insert + hold -> 再恢复 wide reset`

---

## 5. 诊断分流

### 第四步：尽快区分“视觉问题”还是“任务定义问题”

这是非常值得马上做的一步。

建议在**当前修正后的环境**上，先训练一个 `non-visual teacher`。

目的不是替代视觉模型，而是快速回答:

- 如果 teacher 也学不出来:
  - 说明问题主要在 reward / geometry / success definition
  - 此时继续打磨视觉输入意义不大

- 如果 teacher 能学出来，但视觉模型学不出来:
  - 说明任务定义大体是通的
  - 主要矛盾转到 perception / representation / camera geometry

这一实验的价值非常高，因为它能快速切断很多不必要的争论。

这里的“尽快”在执行上应理解为:

- **Phase 1 一结束就启动**
- 而不是等 reward 主线 Phase 2 做完以后再启动

推荐时序:

- Phase 1: 修 success/hold 逻辑
- Phase 2: `clean_insert_hold` 主线
- Phase 3: `teacher_nonvisual` 分流

其中 Phase 2 与 Phase 3 应并行推进。

---

## 6. 建议的执行顺序

### Phase 1：环境与逻辑校准

1. 修 `_hold_counter` 与 success 逻辑接线
2. 确认 `stage1_success_without_lift` 真实生效
3. 确认 tip gate / decay / hysteresis 真实生效
4. 重新校对日志指标与代码定义
5. 冻结一个 `Phase 1 ready` 基线 commit / tag

交付物:

- 一次代码修复
- 一次 sanity run
- 一份“配置与实际逻辑一致”的确认记录
- 一个固定基线（推荐 tag: `exp8_3_phase1_ready`）

### Phase 2：近场 clean insert + hold 主线

1. 保持 `progress_potential + disp_gate`
2. 对插入后 `r_d / progress` 加 alignment / tip gate
3. 跑 near-field curriculum
4. 观察是否出现**持续性的** hold 与 clean insert 信号

交付物:

- 一组 near-field 对照实验
- 一份针对“clean insert / push-insert”区别的日志分析
- 一份是否达到 Phase 4 放行标准的判断

### Phase 3：teacher 分流实验

1. 在修正后的 env 上跑 non-visual teacher
2. 判断 teacher 是否能打通 `clean insert + hold`
3. 用 teacher 结果决定后续重点放在任务定义还是视觉表征

交付物:

- 一次 teacher 训练结果
- 一份“问题主要在 perception 还是 task definition”的结论

### Phase 4：恢复 wide reset

1. 当 near-field 主线稳定后，再恢复 `1.5~2.5m` wide reset
2. 观察推进能力能否保留到远场
3. 只在这一步讨论“远场课程如何设计”

交付物:

- near-field 到 wide-reset 的迁移验证
- 一份是否具备继续做 full approach 训练的判断

### Phase 4 放行标准

只有当 `clean_insert_hold` 分支至少满足以下条件时，才建议进入 wide reset:

1. `diag/max_hold_counter` 不再只是偶发单点非零，而是连续多个窗口稳定非零
2. `phase/frac_success_geom_strict` 连续非零，说明不是单纯“推进去一下”
3. `diag/pallet_disp_xy_mean` 没有随着插入率上升而显著恶化
4. 多个 seed 下趋势一致，至少不是单 seed 偶然现象

如果这些条件还不满足，就不应该进入 Phase 4。

---

## 7. 每阶段重点监控指标

接下来不建议再把 `Mean reward` 当作主指标，而应重点盯以下行为指标:

- `phase/frac_inserted`
- `phase/frac_center_aligned_cfg`
- `phase/frac_tip_constraint_ok`
- `diag/max_hold_counter`
- `phase/frac_success`
- `phase/frac_success_geom_strict`
- `diag/pallet_disp_xy_mean`
- `err/yaw_deg_mean`
- `err/center_lateral_mean`

解释:

- `frac_inserted` 只能说明“进去了多少”
- `max_hold_counter` 才能说明“能不能稳住”
- `pallet_disp_xy_mean` 才能区分“干净插入”还是“推盘式插入”

---

## 8. 一句话策略

下一阶段最重要的不是继续堆 reward，而是先把环境逻辑校正干净，然后用:

`修正后的 success/hold 逻辑 + progress_potential + disp_gate + near-field curriculum + non-visual teacher 分流`

这条路线，把 `clean insert + hold` 先单独打通。
