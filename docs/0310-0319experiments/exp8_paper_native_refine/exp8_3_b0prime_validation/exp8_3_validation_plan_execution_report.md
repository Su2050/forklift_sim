# Exp8.3 验证计划执行完整报告

**日期**：2026-03-22  
**来源计划**：`/home/uniubi/.cursor/plans/exp8.3验证计划_6965ee26.plan.md`  
**覆盖范围**：`B0 -> B0′` 早窗验证、`B0′ / G1 / G2 / G2b / G3` smoke 与 `400 iter` 初筛、worktree 串行总控与过程问题处置  
**总控脚本**：`scripts/run_exp8_3_worktree_smokes.sh`  
**总控日志**：`logs/20260321_153838_sanity_check_exp8_3_worktree_baseline400.log`

## 结论先行

- 这份计划的**第一轮几何验证已经实际跑完**，并且从 `B0′` 一直推进到了 `G3-400` 收尾。
- 计划最关键的结论是：**build-order bug 已确认修复，`B0′` 成为了干净新基线；第一轮当前最优候选不是 `G3`，而是 `G2b`。**
- `G1` 证明 trajectory terminal geometry package 过浅是重要矛盾之一；`G2` 证明只改 `r_d` 不够；`G2b` 证明 `target_center family` 作为 `reward+done` 联合干预是有效方向；`G3` 证明“再把 trajectory 也统一进去”会让行为更干净，但在 `400 iter` 内变得更保守。
- 当前仍未完全闭环的事项主要有两条：
  - `forklift_pallet_insert_lift_project/tests/test_exp83_geometry_preflight.py` 的**纯 torch 预检**已落地并可独立运行，但**真实 env 内直接调用 `_query_reference_trajectory()` 的 runtime U0** 仍未补。
  - 全部结论目前仍是 **single-seed + 400 iter** 口径，尚未进入 `700-800 iter` 确认阶段，也未做双 seed。

## 计划 todo 执行状态

| todo id | 计划 frontmatter | 实际执行结果 | 报告判断 |
| --- | --- | --- | --- |
| `fix-build-order-baseline` | completed | 已按 `B0′` 方案修复 `_build_reference_trajectory()` 的 build-order / fresh reset tensor 接线，并完成 `B0′` smoke 与 `B0′-400` 验证 | 完成 |
| `add-preflight-unit-tests` | pending | 已落地 `forklift_pallet_insert_lift_project/tests/test_exp83_geometry_preflight.py`，并补了可直接运行入口；纯 torch U0/U1 可跑，但 runtime U0 仍未补 | 部分完成，维持 pending 合理 |
| `audit-current-geometry` | completed | 已在计划与验证文档中整理五层几何定义、统一标量终点公式，并用新增日志对齐 root / center / tip | 完成 |
| `split-target-center-family` | pending | 已实际拆成 `G2` 与 `G2b` 两条代码/实验分支，并完成 smoke + `400 iter`；文档也明确了 `G2b` 是 `reward+done`，不是纯 reward | **执行上已完成**，只是 plan frontmatter 还没回写同步 |
| `add-diagnostic-logs` | completed | `geom/s_traj_end`、`geom/s_rd_target`、`geom/s_success_center`、`err/root_lateral_mean`、`err/center_lateral_mean`、`err/tip_lateral_mean`、`phase/frac_rg`、`phase/frac_success` 等已稳定进入 stdout | 完成 |
| `define-metrics-and-horizons` | completed | 实际执行中已按 `iter 0-50`、smoke、`iter 300-399` 三层口径进行判断与文档落盘 | 完成 |
| `register-g5` | completed | `G5a / G5b` 已在计划中登记为第二轮假设，但本轮按计划未开跑 | 完成（登记完成，执行延后） |
| `document-exp83-gap` | completed | 已回写 `exp8_3_fork_center_traj_fix.md`，并新增多份阶段性验证文档 | 完成 |

## 实施过程时间线

1. **先修 build-order，再做几何实验。**  
   在 patch 源目录中把 `B0′` 作为新基线落地：`_build_reference_trajectory()` 改为能吃 fresh reset tensors，并确保 reset 后的轨迹使用当前 episode 的 pallet / robot / joint 状态，而不是旧缓存。

2. **补运行前几何预检与诊断日志。**  
   增加 `forklift_pallet_insert_lift_project/tests/test_exp83_geometry_preflight.py` 作为低成本预检；同时在任务代码里补齐 `root / center / tip` 横向误差、`s_center_mean / s_tip_mean`、`phase/frac_rg`、`phase/frac_success`、`diag/out_of_bounds_frac`、`diag/success_term_frac` 等日志。

3. **完成 `B0 -> B0′` 早窗验证。**  
   使用历史 `B0` 与 `B0′ smoke` 做 `iter 0-50` 对照，确认 `traj/d_traj_mean` 起点量级从双位数回落到接近 `0`，证明 build-order bug 会真实污染 `r_cd / r_cpsi` 主路径。

4. **完成 `B0′-400` 新基线。**  
   结果显示 `B0′` 已经是干净、稳定、不推盘的新基线，但仍长期停在远场 approach，没有形成有效插入/成功信号。

5. **完成 `G1 smoke` 与 `G1-400`。**  
   `G1` 只改 trajectory terminal geometry package 到 `s_success_center`。结果表明它明显增强了入口接近与部分插入，但代价是偏航、横向误差与推盘显著增大。

6. **建立 `G2 / G2b / G3` 三个独立 branch + worktree。**  
   三条分支分别实现、分别 smoke、分别提交，避免在同一工作树里反复切换代码状态。  
   - `G2`：`reward-only`，只改 `r_d`  
   - `G2b`：`reward+done`，统一 `target_center family`  
   - `G3`：`trajectory-only + reward+done`，统一 `p_goal + target_center family`

7. **回到主工作树，补串行总控脚本。**  
   `scripts/run_exp8_3_worktree_smokes.sh` 最终支持：
   - `smoke` / `baseline400` 两种模式
   - 顺序执行 `G2 -> G2b -> G3`
   - 每个 worktree 启动前自动把 patch 安装进共享 `IsaacLab`
   - `--detach` 彻底脱离终端，整条串行链条用 `nohup` 跑在后台

8. **处理一次真实的运行中断。**  
   首轮 `baseline400` 总控中，`G2-400` 在约 `iter 227` 附近出现“进程仍在、日志不再增长”的挂住现象。处理方式是：
   - 手动 kill 挂住的 `G2-400`
   - 确认 GPU / 内存资源释放
   - 重新以 `--detach baseline400` 启动整条总控链

9. **detach 总控最终完整收尾。**  
   重启后的串行总控按顺序完成了：
   - `G2-400`
   - `G2b-400`
   - `G3-400`

   总控日志末尾已明确写出：
   - `[OK] g3 baseline400 passed`
   - `[DONE] All requested Exp8.3 worktree baseline400 runs completed.`

## 已落盘的阶段性文档

- `docs/0310-0319experiments/exp8_paper_native_refine/exp8_3_b0prime_validation/b0_to_b0prime_iter0_50_validation.md`
- `docs/0310-0319experiments/exp8_paper_native_refine/exp8_3_b0prime_validation/b0prime_400_baseline_validation.md`
- `docs/0310-0319experiments/exp8_paper_native_refine/exp8_3_b0prime_validation/g1_traj_terminal_s_success_center_smoke_validation.md`
- `docs/0310-0319experiments/exp8_paper_native_refine/exp8_3_b0prime_validation/g1_traj_terminal_s_success_center_baseline_validation.md`
- `docs/0310-0319experiments/exp8_paper_native_refine/exp8_3_b0prime_validation/g2b_target_center_family_success_center_vs_g3_unify_traj_and_target_family_baseline_validation.md`

说明：

- `G2-400` 与 `G2b/G3` smoke 的结果此前尚未单独成文，本文已把它们作为完整计划执行结果统一归档。

## smoke 阶段结果摘要

| 实验 | 日志 | 结果 | 主要现象 |
| --- | --- | --- | --- |
| `B0′ smoke` | `logs/20260320_101115_train_exp8_3_b0prime_smoke.log` | 通过 | build-order 早窗污染被明显消除，新增日志 key 全部接线成功 |
| `G1 smoke` | `logs/20260320_154622_train_exp8_3_g1_traj_terminal_s_success_center_smoke.log` | 通过 | 比 `B0′` 更接近入口并首次出现非零插入，但偏航/推盘迅速抬升 |
| `G2 smoke` | `forklift_sim_wt_g2/logs/20260320_222816_smoke_train_exp8_3_g2_rd_target_s_success_center.log` | 通过 | `reward-only` 接线正常，但末段仍无非零插入/`rg`，说明只改 `r_d` 不足 |
| `G2b smoke` | `forklift_sim_wt_g2b/logs/20260320_225859_smoke_train_exp8_3_g2b_target_center_family_success_center.log` | 通过 | 行为较干净、推盘小，但近场推进还不强；为后续 `400 iter` 留下观察空间 |
| `G3 smoke` | `forklift_sim_wt_g3/logs/20260320_232819_smoke_train_exp8_3_g3_unify_traj_and_target_family.log` | 通过 | 早期推进最积极，已出现非零 `inserted/rg`，但伴随更大的偏航与推盘 |

## `400 iter` 结果总表（`iter 300-399` 窗口均值）

| 实验 | 干预类型 | `err/dist_front_mean` | `err/yaw_deg_mean` | `diag/pallet_disp_xy_mean` | `phase/frac_inserted` | `phase/frac_rg` | `phase/frac_success` | `traj/d_traj_mean` | 主结论 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `B0′-400` | `—` | 2.8740 | 8.1466 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.9021 | build-order 修复后形成干净基线，但仍长期停在远场 |
| `G1-400` | `trajectory-only` | 0.5397 | 13.8867 | 0.2534 | 0.0775 | 0.0000 | 0.0000 | 0.5210 | 推进最强，但偏航/推盘最重，不是最终解 |
| `G2-400` | `reward-only` | 2.0241 | 8.1255 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.1670 | 只改 `r_d` 不足以把策略拉进近场 |
| `G2b-400` | `reward+done` | 0.7151 | 9.1831 | 0.1032 | 0.0242 | 0.0128 | 0.0000 | 0.3703 | 当前最均衡，既比 `B0′/G2` 更能进去，也比 `G1` 更干净 |
| `G3-400` | `trajectory-only + reward+done` | 1.2718 | 8.6356 | 0.0345 | 0.0083 | 0.0055 | 0.0000 | 0.2623 | 更干净、更贴走廊，但明显更保守，不如 `G2b` 会推进 |

## 对计划原始假设的回答

1. **build-order bug 是真实主变量，不是日志层小问题。**  
   `B0 -> B0′` 早窗已经给出足够证据：如果参考轨迹不是用当前 reset 张量重建，`traj/d_traj_mean` 起点会被旧 episode 轨迹严重污染。

2. **trajectory terminal geometry package 过浅，确实是主矛盾之一。**  
   `G1` 相比 `B0′` 大幅改善入口接近与插入倾向，说明“轨迹终点太浅”不是边角因素。

3. **但只改 trajectory terminal geometry package 会把策略拉得过猛、过歪。**  
   `G1` 的 `yaw`、横向误差与托盘位移显著恶化，说明它更像“会往里冲”，而不是“已经学会高质量插入”。

4. **只改 `r_d` 目标点不够。**  
   `G2-400` 明确说明 `reward-only` 强度不足，无法独立把策略从远场推进到稳定近场。

5. **`target_center family` 的 `reward+done` 联合统一是有效方向。**  
   `G2b-400` 比 `G2-400` 明显更接近入口、也更常出现非零 `inserted/rg`，这支持了计划里“若 G2 不明显，再补跑 G2b”的判断。

6. **第一轮结果不支持把 `G3` 作为当前冠军。**  
   `G3` 虽然更干净、更少推盘、更贴走廊，但在 `400 iter` 内明显比 `G2b` 更保守，推进性不足。因此这轮并没有得到“只有 trajectory + target family 全统一才明显有效”的结论。

7. **成功判定 / 终止层仍然是下一轮问题，而不是本轮已解决问题。**  
   所有 `400 iter` run 的 `phase/frac_success` 都没有在决策窗口内形成稳定非零，因此 `G4 / G5a / G5b` 仍保留为后续阶段，而不是可以现在提前下结论说“已经不需要看”。

## 过程中的偏差与修正

1. **预检脚本运行环境问题**  
   初始尝试依赖 `pytest`，但当前 agent 环境缺少该命令。后续为预检脚本补了直接运行入口，避免把是否能跑预检绑定到 `pytest` 是否安装。

2. **训练启动环境问题**  
   早期 smoke 启动时出现过 `TERM=dumb`、conda 污染、日志为空或进程直接退出的问题。后续统一在启动脚本里做了：
   - `export TERM=xterm`
   - conda 全部 deactivate / unset
   - `PYTHONUNBUFFERED=1`
   - `nohup ... &`

3. **worktree 共享 IsaacLab 的路径问题**  
   worktree 内部的 `IsaacLab` 目录不是完整 checkout，导致直接安装 patch 失败。后续改成：若 worktree 本地 `IsaacLab` 不可用，就自动回落到共享的 `/home/uniubi/projects/forklift_sim/IsaacLab`。

4. **首轮 `G2-400` 挂住**  
   首次 `baseline400` 总控里，`G2-400` 出现“进程活着、日志不再增长”的典型 hang。后续已通过 kill 残留进程、确认资源释放、再用 `--detach baseline400` 方式重启总控解决。

## 当前状态与建议

- **第一轮 `400 iter` 机制初筛已经完成。**
- **当前最值得进入更长 horizon 的候选是 `G2b`。**
- `G3` 可以保留为“更干净、更少推盘”的保守对照，但它不应替代 `G2b` 成为本轮一号候选。
- 若继续执行原计划，推荐优先级是：
  1. `G2b` 进入 `700-800 iter`
  2. 视资源决定是否保留 `G3` 作为第二条长跑对照
  3. 在第一轮更长 horizon 结论出来后，再进入 `G4 / G5a / G5b`
- 另一个尚未闭环的技术债是：把 **runtime U0** 真正补进真实 env 路径，避免预检仍停留在纯 torch 级别。

## 一句话总报告

**这份计划已经把“先修 build-order，再分离 trajectory / `r_d` / `target_center family` 的几何干预”完整执行了一遍；真实跑出来的第一轮答案是：`B0′` 建立了干净基线，`G1` 证明 trajectory 终点太浅是主矛盾之一，`G2` 证明只改 `r_d` 不够，`G2b` 成为当前最优候选，而 `G3` 虽更干净却在 `400 iter` 内过于保守，因此下一步最值得继续追的是 `G2b`，不是直接转向 `G3` 或过早进入 `G4/G5`。**
