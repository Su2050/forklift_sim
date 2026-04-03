# Exp9.0 下一个 A/B 实验计划：Gate Curriculum vs Strict Baseline

日期：`2026-04-03`

基于对 `20260403_exp9_0_recent_result_analysis_by_methodology.md` 的深度分析，当前的核心矛盾已经非常清晰：**策略能频繁到达 `0.175m` 的 near-pass 区，但深插入后的局部纠偏能力不足，导致无法跨越 `0.12m` 的 strict gate。** 纯粹放宽 gate 只能“粉饰”数据，无法提升真实的几何对齐质量。

基于分析文档中的“下一步探索方向”，本次实验将聚焦于最高优先级的方向：**Training Gate Curriculum（动态门限课程学习）**。

---

## 1. 实验目的

验证“动态收紧容忍度（Curriculum）”能否作为训练桥梁，引导策略跨越 `0.175m -> 0.12m` 的最后纠偏瓶颈，最终提升严格标准下的成功率（Strict Success）。

## 2. A/B 分组设置

建议跑一个双 seed（如 seed 42, 43）的 A/B/C 三组对比，总计 6 个 run，跑到 2000 iter：

*   **Group A (对照组 - Strict Baseline):** 
    *   保持当前的 Strict 设置。
    *   `tip_gate` 全程固定为 `0.12m`。
*   **Group B (实验组 1 - Gate Curriculum):** 
    *   引入动态门限退火。
    *   训练初期 `tip_gate` 设为 `0.175m`（接住 near-pass 样本，提供正反馈）。
    *   随着训练步数（例如在前 1000 或 1500 iter 内），将 `tip_gate` 线性退火收紧至 `0.12m`。
    *   *注意：评估口径（Diagnostic）必须始终保留一组严格按 `0.12m` 计算的 `success_strict`。*
*   **Group C (实验组 2 - Gap Shaping / 备选):** 
    *   保持 `tip_gate = 0.12m` 不变。
    *   在 Reward 中增加针对 `0.175m -> 0.12m` 区间的显式 Dense Reward（例如：当处于 inserted 状态且 tip error 在 0.175 以内时，error 每缩小一点给予额外奖励）。

## 3. 核心观测指标 (Metrics)

不要只盯总 reward，必须看漏斗转化和几何误差（取 Last 20/50 均值）：

1.  **终极目标指标：**
    *   `phase/frac_success_strict` （无论训练 gate 怎么变，最终都必须看这个严格标准下的成功率）
2.  **漏斗转化卡点指标：**
    *   `phase/frac_inserted` （确保前期插入能力没有退化）
    *   `phase/frac_prehold_reachable_band` （0.175m 宽容带的到达率）
    *   `phase/frac_hold_entry` （进入 0.12m 严格 hold 的比例）
3.  **几何对齐质量指标：**
    *   `err/center_lateral_inserted_mean`
    *   `err/tip_lateral_inserted_mean`
    *   `err/yaw_deg_inserted_mean`

## 4. 判读标准 (Evaluation Criteria)

实验结束后，通过对比 Group B/C 与 Group A 的指标，按以下标准进行判读：

*   🟢 **大捷 (Curriculum 破局成功)：**
    *   **现象：** Group B 的 `frac_success_strict` 显著大于 0 且高于 Group A。同时，Group B 的 `tip_lateral_inserted_mean` 呈现明显的下降趋势（优于之前的 Relaxed 组）。
    *   **结论：** 动态门限成功搭建了桥梁，策略借此学会了深插入后的最后一段精细纠偏。问题解决。
*   🟡 **假象 / 延后卡点 (Curriculum 无效)：**
    *   **现象：** Group B 在退火前期（gate 较宽时）有较高的 success，但随着 gate 收紧到 0.12m，success 重新跌回 0（或极低）。`frac_prehold_reachable_band` 再次出现大量堆积，且几何误差 `tip_lateral_inserted_mean` 停留在 0.35~0.4 左右降不下去。
    *   **结论：** 策略依然无法克服深插入后的局部 controllability（可控性）问题。Curriculum 只是推迟了卡点，说明底层物理/动作空间在深插入时确实缺乏横向纠偏能力，下一步必须转向“修改环境/控制器”或“强化 Case A 初始状态采样”。
*   🔴 **负面恶化 (Curriculum 破坏了学习)：**
    *   **现象：** Group B 连 `frac_inserted` 都出现了明显下降。
    *   **结论：** 变动的 gate 目标导致了策略的灾难性遗忘或梯度混乱，破坏了原本已经学好的前序插入动作。需要放弃 Curriculum，转向 Group C 的 Gap Shaping 方案。

---
**执行建议：** 优先实现 Group B 的 Curriculum 逻辑（需要修改 `env.py` 或 `env_cfg.py` 中关于 success 判定的阈值，使其随 `env.step_count` 或外部传入的 progress 衰减），并确保 `success_strict` 的 logging 逻辑不受退火影响，然后即可启动多 seed 训练。