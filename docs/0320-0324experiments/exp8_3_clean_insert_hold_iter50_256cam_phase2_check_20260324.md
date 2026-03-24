# `exp8_3_clean_insert_hold_iter50_256cam` 对照 `Phase 2` 验收标准简表

> 日期: `2026-03-24`
> 对照文档: `plan_auto_pallet_insertion_model_20260324.md`
> 训练目录: `IsaacLab/logs/rsl_rl/forklift_pallet_insert_lift/2026-03-24_14-47-30_exp8_3_clean_insert_hold_iter50_256cam`
> 说明: 本次 run 目标为 `50` iter, TensorBoard 实际记录到 `step 46`。训练随后卡住并被人工终止, 以下判断基于 `step 0 ~ 46` 的已落盘事件数据。

---

## 1. 一页结论

- 这次 run 说明 `Phase 2` 主线方向是对的: near-field 下已经打开明显插入信号, `phase/frac_inserted` 末尾为 `0.2031`, 近 5 个窗口均值为 `0.2063`。
- `hold` 相关信号不再只是“首次碰巧非零”: `diag/max_hold_counter` 在 `47` 个窗口里有 `30` 个非零, 最长连续非零达到 `11` 个窗口, 末尾连续 5 个窗口都保持非零。
- 但这次 run 还**不能算通过 `Phase 2` 验收**: `phase/frac_success_geom_strict` 虽然出现过, 但最长只连续 `3` 个窗口, 末尾仍有明显断档; `phase/frac_success` 很稀疏, `phase/frac_lifted_enough = 0.0000`。
- 最大问题仍是“插进去了, 但不够干净也不够稳”: `diag/pallet_disp_xy_mean` 近 5 个窗口均值从起始 `0.0031 m` 升到 `0.0858 m`, `diag/pallet_disp_xy_inserted_mean` 末尾达到 `0.1577 m`, 而 `phase/frac_inserted_push_free` 末尾只有 `0.0156`。
- 因此当前状态可以概括为: **`Phase 2` 已经拿到可重复的近场插入/hold 弱信号, 但仍未达到 `clean insert + stable hold` 的验收线, 也还不具备进入 `Phase 4` wide reset 的条件。**

---

## 2. 关键指标摘录

| 指标 | 本次结果 | 备注 |
|------|----------|------|
| `phase/frac_inserted` | `0.2031` | 近 5 个窗口均值 `0.2063` |
| `phase/frac_clean_insert_ready` | `0.0156` | 有弱信号, 但不稳定 |
| `phase/frac_hold_entry` | `0.0156` | 有弱信号, 但不稳定 |
| `phase/frac_success_geom_strict` | `0.0156` | 非零 `20/47` 窗口, 最长连续 `3` 窗口 |
| `phase/frac_success` | `0.0156` | 非零 `4/47` 窗口, 最长连续 `1` 窗口 |
| `phase/frac_lifted_enough` | `0.0000` | 尚未打开 |
| `diag/max_hold_counter` | `10` | 非零 `30/47` 窗口, 最长连续 `11` 窗口 |
| `phase/frac_tip_constraint_ok` | `0.9688` | tip 约束总体较强 |
| `phase/frac_inserted_push_free` | `0.0156` | 与 `frac_inserted` 差距很大 |
| `phase/frac_push_free_success` | `0.0000` | push-free success 仍为零 |
| `diag/pallet_disp_xy_mean` | `0.0824 m` | 近 5 个窗口均值 `0.0858 m` |
| `diag/pallet_disp_xy_inserted_mean` | `0.1577 m` | 插入态位移仍偏大 |
| `err/dist_front_mean` | `0.3295 -> 0.0652 m` | 前向接近明显改善 |
| `err/yaw_deg_mean` | `1.2822 -> 5.2359 deg` | 整体偏航恶化 |
| `err/yaw_deg_inserted_mean` | `9.2015 deg` | 插入态偏航仍偏大 |

---

## 3. 对照 `Phase 2` 验收标准

| 文档标准 | 本次观测 | 判断 |
|----------|----------|------|
| 不再只看“是否首次出现 `hold_counter > 0`” | `diag/max_hold_counter` 非零 `30/47` 窗口, 最长连续 `11` 窗口, 末尾连续 `5` 窗口非零 | 已达成 |
| `diag/max_hold_counter` 连续多个日志窗口非零 | `step 9 ~ 19` 连续非零, 尾部 `step 42 ~ 46` 也连续非零 | 基本达成 |
| `phase/frac_success_geom_strict` 连续非零, 而不是单点抖动 | 非零 `20/47` 窗口, 但最长连续仅 `3` 窗口; 尾部 `42 ~ 46` 中有 `3` 个零窗口 | 未达成 |
| `diag/pallet_disp_xy_mean` 不随插入提升而明显恶化 | `frac_inserted` 近 5 窗口均值从 `0.0000` 升到 `0.2063`, 同期 `pallet_disp_xy_mean` 从 `0.0031 m` 升到 `0.0858 m` | 未达成 |
| 插入增加时, 推盘位移不能同步明显恶化 | `diag/pallet_disp_xy_inserted_mean = 0.1577 m`, `phase/frac_inserted_push_free = 0.0156`, 远低于 `phase/frac_inserted = 0.2031` | 未达成 |
| reward 改动是否把行为推向更干净的插入, 而不是更激进的撞盘 | 正面: `frac_inserted` 提升, `tip_constraint_ok = 0.9688`, `dist_front_mean` 明显下降。负面: `yaw` 恶化, 插入态位移偏大, `push_free_success = 0` | 部分达成 |

---

## 4. `Phase 2` 目前已经达成了什么

1. `near-field + progress_potential + disp_gate` 确实已经把策略从“几乎不插入”推进到“能较稳定地产生插入信号”。
2. `hold` 相关计数器已经不是偶发单点事件, 说明当前 reward/gate 至少开始把策略带到可短暂保持的区域。
3. `tip constraint` 指标总体较好, 说明当前策略不是完全失控地乱撞。
4. `err/dist_front_mean` 大幅下降, 说明 agent 已能更稳定地把货叉推进到接近插入区域。

---

## 5. `Phase 2` 还缺什么

1. 还没有形成**持续性的** `success_geom_strict`。目前只能说“偶尔接近严格成功”, 不能说“已经稳定到位”。
2. `clean_insert_ready`、`hold_entry`、`success` 都还偏稀疏, 真正的稳定 hold 闭环还没建立起来。
3. `push-free` 证据明显不够: `frac_inserted` 已到 `0.2031`, 但 `frac_inserted_push_free` 只有 `0.0156`, 说明大量插入更像“带推盘成分的插入”。
4. 插入态的位移和偏航仍偏大: `pallet_disp_xy_inserted_mean = 0.1577 m`, `yaw_deg_inserted_mean = 9.2015 deg`, 这与“clean insert”目标仍有明显距离。
5. 这只是一次未跑完的单 seed smoke run。训练卡在 `46/50` 后被人工终止, 还缺完整结束 run 和多 seed 可比性。

---

## 6. 当前结论对应的后续动作

1. 继续停留在 `Phase 2`, 不进入 `Phase 4` wide reset。
2. 下一轮优先强化 post-insert 的 `alignment / tip / push-free` gate, 而不是继续单纯放大奖励深度。
3. 把 `frac_inserted_push_free`、`pallet_disp_xy_inserted_mean`、`yaw_deg_inserted_mean` 作为 near-field 主监控指标。
4. 先解决 run 收尾卡住的问题, 再补至少 `2 ~ 3` 个 seed 的 near-field 对照, 判断当前趋势是否稳定。

---

## 7. 一句话判断

**这次 `iter50_256cam` 已经证明 `Phase 2` 方向有用, 但目前更像“打开了插入和弱 hold 信号”, 还不能算“clean insert + stable hold 已经打通”。**
