# Exp8.3 Confirm800 阶段 G4 进入决策

**日期**：2026-03-22  
**来源计划**：`/home/uniubi/.cursor/plans/exp8.3后续推进完善版_3834ad36.plan.md`  
**决策窗口**：`iter 700-799`

## 决策结论

**结论：暂不进入 `G4`。**

原因不是 runtime U0 未通过，也不是长跑没完成；相反，这一轮 `confirm800` 的执行链已经完整跑通。真正卡住进入 `G4` 的，是主候选 `G2b` 在长 horizon 下没有保住推进性，因此不满足计划里定义的量化门槛。

## 输入依据

- runtime U0 验证：`docs/0310-0319experiments/exp8_paper_native_refine/exp8_3_b0prime_validation/runtime_u0_sanity_validation.md`
- `G2b-800` 结果：`forklift_sim_wt_g2b/docs/0310-0319experiments/exp8_paper_native_refine/exp8_3_b0prime_validation/g2b_target_center_family_success_center_confirm800_validation.md`
- `G3-800` 结果：`forklift_sim_wt_g3/docs/0310-0319experiments/exp8_paper_native_refine/exp8_3_b0prime_validation/g3_unify_traj_and_target_family_confirm800_validation.md`
- `G2b-800 vs G3-800` 对比：`docs/0310-0319experiments/exp8_paper_native_refine/exp8_3_b0prime_validation/g2b_target_center_family_success_center_vs_g3_unify_traj_and_target_family_confirm800_validation.md`
- `G2b-400` 历史参考：`forklift_sim_wt_g2b/logs/20260321_190257_train_exp8_3_g2b_target_center_family_success_center_baseline.log`

## 计划门槛逐项核对

### A. 相比 `G3-800`

计划要求 `G2b` 同时满足：

1. `G2b err/dist_front_mean` 更低  
   - `G2b-800 = 2.1469`
   - `G3-800 = 1.1089`
   - **结果：失败**

2. `G2b phase/frac_inserted` 不低于 `G3`
   - `G2b-800 = 0.0000`
   - `G3-800 = 0.0541`
   - **结果：失败**

3. `G2b phase/frac_rg` 不低于 `G3`
   - `G2b-800 = 0.0000`
   - `G3-800 = 0.0214`
   - **结果：失败**

### B. 相比 `G2b-400`

计划要求 `G2b-800` 同时满足：

1. `diag/pallet_disp_xy_mean` 不恶化超过 `+0.05 m` 且绝对值 `<= 0.18`
   - `G2b-400 = 0.1032`
   - `G2b-800 = 0.0000`
   - `delta = -0.1032`
   - **结果：通过**

2. `err/yaw_deg_mean` 不恶化超过 `+2.0 deg` 且绝对值 `<= 11.5 deg`
   - `G2b-400 = 9.1831 deg`
   - `G2b-800 = 7.9905 deg`
   - `delta = -1.1926 deg`
   - **结果：通过**

3. `err/dist_front_mean` 不高于 `G2b-400` 的 `1.15x`
   - `G2b-400 = 0.7151`
   - `G2b-800 = 2.1469`
   - `ratio = 3.0023x`
   - **结果：失败**

## 为什么不能进入 G4

1. **主候选 `G2b` 已经不再优于 `G3`。**  
   在计划定义的主窗口 `700-799` 内，`G2b` 的 `dist_front` 更差，`inserted/rg` 全部归零，而 `G3` 仍保留明显非零推进信号。也就是说，“把 `G2b` 作为主候选继续往下加新改动”的前提已经被 confirm800 结果推翻。

2. **`G2b` 的失败形态不是“脏”，而是“推进性塌缩”。**  
   `G2b-800` 的 `diag/pallet_disp_xy_mean` 和 `err/yaw_deg_mean` 都通过了门槛，说明它不是因为后期失控才被卡住；真正的问题是它在后段完全失去了入口接近与插入倾向。这种失败同样不适合直接叠加 `G4` 新改动，因为会把问题来源混在一起。

3. **`G3` 虽然比 `G2b` 更有 late-window 推进保持力，但仍不是放行态。**  
   `G3-800` 依旧 `phase/frac_success = 0`，而且 `diag/pallet_disp_xy_mean = 0.1675` 仍高于 `G2b-800`。因此当前不能简单把结论替换成“改跑 `G3` 进入 G4”；更合理的是先承认 confirm800 给出的主结论：**当前 family 还没有到“可加下一层机制”的成熟度。**

## 本轮正式决策

### 决策

- **不进入 `G4`**

### 当前最准确的结论口径

- runtime U0 已经在真实 env 路径通过，训练链路可信。
- `G2b-800` 不满足计划定义的 `G4` 进入门槛。
- `G3-800` 在后段比 `G2b-800` 更能保留推进，但仍未闭环 success。
- 因此，下一轮工作重点应是：**先处理长 horizon 下“推进性保持 vs 行为干净度”的折中问题，而不是直接叠加 `G4`。**

## 一句话决策

**confirm800 的量化结果已经给出明确答案：当前不应进入 `G4`；应先回头处理 `G2b` 的后段推进性塌缩，并把 `G3` 作为“能保留更多 late-window 推进信号”的保守参照，而不是把新的 success-layer 改动继续往上堆。**
