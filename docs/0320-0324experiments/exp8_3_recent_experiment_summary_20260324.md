# Exp8.3 最近实验结果汇总

> 日期范围: `2026-03-20 ~ 2026-03-24`
> 数据来源: `/home/uniubi/projects/forklift_sim_wt_g3/logs` 下最近 10 个实验日志
> 说明 1: 不同 reward family 的 `Mean reward` 量级不同, 不能直接横向比较绝对值
> 说明 2: 本文以日志末尾指标为准, 更关注 `frac_inserted / frac_success / frac_lifted_enough` 这类行为指标

---

## 1. 一页结论

- `unify_traj_and_target_family` 系列整体未成功收敛: reward 很高, 但 success 始终为 0, 插入信号不能稳定保持到训练末尾。
- `realwt_confirm400` 仍未打开插入: 到末尾 `frac_inserted = 0.0000`, 说明只切到 real-weight 还不够。
- `realwt + progress_potential` 开始出现可重复的插入信号, 虽然很弱, 但方向是对的。
- `realwt + progress_potential + disp_gate` 是最近最有希望的方向。最佳 run `20260324_090023...` 末尾达到 `phase/frac_inserted = 0.1406` 和 `phase/frac_success_geom_strict = 0.0312`, 说明策略已经明显更接近“几何上接近成功”的状态。
- 但当前真正的瓶颈已经从“接近托盘”转移到“稳定 hold/举升完成”: 最佳 run 仍然 `phase/frac_success = 0.0000`, `phase/frac_success_strict = 0.0000`, `phase/frac_lifted_enough = 0.0000`。
- `2026-03-24` 这批 `disp_gate` 启动里有 2 个空日志, 1 个只跑到 `6/200`, 说明除了 reward 设计, 启动和收尾稳定性也需要单独排查。
- 最新长跑 `20260324_090023...` 日志停在 `199/200`, 未见 `200/200` 或 `Saving model`, 需要检查最终收尾和落盘逻辑。

---

## 2. 有效训练日志汇总

| 日期 | 日志 | 变体 | 末尾迭代 | Mean reward | frac_inserted | frac_success_geom_strict | frac_success | frac_lifted_enough | 结果判定 |
|------|------|------|-----------|-------------|---------------|--------------------------|--------------|--------------------|----------|
| 2026-03-20 | `20260320_232819_smoke_train_exp8_3_g3_unify_traj_and_target_family.log` | smoke / unify traj family | `49/50` | `115043.00` | `0.0781` | `-` | `0.0000` | `-` | smoke 可跑通, 但仅说明训练链路正常, 不代表已学会任务 |
| 2026-03-21 | `20260321_223243_train_exp8_3_g3_unify_traj_and_target_family_baseline.log` | baseline | `399/400` | `130061.51` | `0.0000` | `-` | `0.0000` | `-` | 长跑后末尾无插入, baseline 失败 |
| 2026-03-22 | `20260322_093455_sanity_check_exp8_3_runtime_u0_g3.log` | sanity check | `1/2` | `-` | `0.0000` | `-` | `0.0000` | `-` | 仅做运行时 sanity check, 不参与训练优劣比较 |
| 2026-03-22 | `20260322_163444_train_exp8_3_g3_unify_traj_and_target_family_confirm.log` | confirm | `799/800` | `98619.26` | `0.0000` | `-` | `0.0000` | `-` | 更长训练仍未打开成功, 说明该 family 不足以支撑完整任务 |
| 2026-03-23 | `20260323_172210_train_exp8_3_g3_realwt_confirm400.log` | realwt confirm400 | `399/400` | `147255.41` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | 切换到 real-weight 后仍未打开插入/举升 |
| 2026-03-23 | `20260323_224329_train_exp8_3_g3_realwt_confirm200_progress_potential.log` | realwt + progress potential | `199/200` | `4717.05` | `0.0156` | `0.0000` | `0.0000` | `0.0000` | 首次出现弱插入信号, 说明 progress potential 有正向作用 |
| 2026-03-24 | `20260324_090023_train_exp8_3_g3_realwt_confirm200_progress_potential_disp_gate.log` | realwt + progress potential + disp gate | `199/200` | `2544.70` | `0.1406` | `0.0312` | `0.0000` | `0.0000` | 最近最佳 run, 插入显著提升, 但仍卡在成功闭环前最后一步 |

---

## 3. 异常或无效 run

| 日期 | 日志 | 状态 | 现象 | 判断 |
|------|------|------|------|------|
| 2026-03-24 | `20260324_085446_train_exp8_3_g3_realwt_confirm200_progress_potential_disp_gate.log` | 空日志 | `0` 行, 无任何训练输出 | 可视为启动失败或启动后立即退出 |
| 2026-03-24 | `20260324_085521_train_exp8_3_g3_realwt_confirm200_progress_potential_disp_gate.log` | 早停 | 仅跑到 `6/200`, `Mean reward = 178.22`, `frac_inserted = 0.0000` | 更像早期退出或人工中断, 不宜与完整 run 直接比较 |
| 2026-03-24 | `20260324_085941_train_exp8_3_g3_realwt_confirm200_progress_potential_disp_gate.log` | 空日志 | `0` 行, 无任何训练输出 | 可视为启动失败或启动后立即退出 |

---

## 4. 按实验家族的结论

### 4.1 `unify_traj_and_target_family` 系列

- 从 smoke 到 baseline 再到 confirm, reward 始终很高, 但末尾 `frac_success` 一直为 0。
- smoke run 曾短暂出现 `frac_inserted = 0.0781`, 但在后续 baseline 和 confirm 的长跑里并没有保留下来。
- 这说明该 family 更像是在优化 dense reward 的局部行为, 但没有形成稳定的插入-举升闭环。

### 4.2 `realwt_confirm400`

- 切到 real-weight 后, 日志末尾依然 `frac_inserted = 0.0000`, `frac_success = 0.0000`, `frac_lifted_enough = 0.0000`。
- 结论是: 单独切 reward weight 不足以把策略从“接近但不插入”推到“可插入”阶段。

### 4.3 `realwt + progress_potential`

- `20260323_224329...` 首次在末尾给出 `frac_inserted = 0.0156`。
- 数值不高, 但这是一个重要转折点: 表明 progress-style shaping 开始为插入提供可观测的正梯度。

### 4.4 `realwt + progress_potential + disp_gate`

- 最佳 run `20260324_090023...` 将 `frac_inserted` 提升到 `0.1406`, 同时出现 `frac_success_geom_strict = 0.0312`。
- 这说明 agent 已经能更频繁地靠近正确几何位姿, 相比前一天有明显进步。
- 但 `frac_lifted_enough = 0.0000`, `frac_success_strict = 0.0000`, `frac_success = 0.0000` 说明策略仍然没有越过“稳定 hold + 举升完成”的门槛。
- 因而当前最合理的判断是: 最近实验已经把问题从“完全不会插入”推进到“偶尔能插入但不会完成举升闭环”。

---

## 5. 当前最佳 run 与核心瓶颈

当前最佳 run 是:

- `20260324_090023_train_exp8_3_g3_realwt_confirm200_progress_potential_disp_gate.log`

其意义不在于已经成功, 而在于它是最近最清楚地展示“插入能力明显抬升”的一次:

- `phase/frac_inserted = 0.1406`
- `phase/frac_success_geom_strict = 0.0312`
- `phase/frac_success = 0.0000`
- `phase/frac_lifted_enough = 0.0000`

因此, 当前瓶颈可以概括为:

1. 接近与部分插入已经开始出现。
2. 真正的成功闭环仍然断在 hold / lift 阶段。
3. 启动与收尾链路不够稳定, 会干扰实验效率和结果判读。

---

## 6. 建议的下一步

1. 继续沿 `progress_potential + disp_gate` 方向迭代, 不建议退回 `unify_traj_and_target_family` 系列。
2. 将下一轮 reward 调整重心转到 post-insert 的 hold 与 lift, 而不是继续加强远端接近奖励。
3. 单独排查 `disp_gate` 的启动稳定性, 因为同一天已出现 2 个空日志和 1 个早停 run。
4. 检查训练收尾与模型落盘逻辑, 解决多次停在 `N-1/N` 的问题, 避免最佳 run 结束时没有明确保存记录。

---

## 7. 附注

- 本文是日志级别的统一汇总, 不是离线评估报告。
- 若后续拿到了对应 checkpoint 的 rollout 视频或 eval 结果, 建议把这份文档继续补成“训练日志 + 可视化行为”的联合结论。
