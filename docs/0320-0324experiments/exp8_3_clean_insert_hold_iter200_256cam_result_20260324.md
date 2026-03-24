# `exp8_3_clean_insert_hold_iter200_256cam` 结果记录

> 日期: `2026-03-24`
> 分支: `exp/exp8_3_clean_insert_hold`
> 训练目录: `IsaacLab/logs/rsl_rl/forklift_pallet_insert_lift/2026-03-24_16-14-51_exp8_3_clean_insert_hold_iter200_256cam`
> 日志文件: `logs/20260324_161520_train_exp8_3_clean_insert_hold_iter200_256cam.log`
> 说明: 本次 run 为 `near-field + 256x256 + clean_insert_hold` 的 `200 iter` 正式短训练。训练目录中已保存到 `model_199.pt`，但文本日志停在 `199/200`，未看到 `200/200` 的正常收尾打印。

---

## 1. 一页结论

- 这次 run 证明 `Phase 2` 主线不是假信号。`clean insert / hold / success_geom_strict` 都已经不是偶发单点，而是在整轮里反复出现。
- 这次 run 的最大新进展是: **尾窗首次打出了非零 `phase/frac_success`**。在 `iter 199`，`phase/frac_success = 0.0156`，同时 `phase/frac_clean_insert_ready = 0.0156`、`phase/frac_hold_entry = 0.0156`、`diag/max_hold_counter = 9.0`。
- 但这次 run 仍然**不能算通过 `Phase 2`**。原因不是“完全不会插”，而是“插入大多仍然不干净，推盘过大，导致 clean insert 和稳定 hold 没有真正站住”。
- 整轮里 `phase/frac_inserted` 已经很高，最佳到 `0.6719`，尾部 5 个窗口均值也有 `0.5156`；但 `phase/frac_inserted_push_free` 最好只有 `0.0625`，尾部 5 个窗口均值只有 `0.0125`。
- 推盘问题仍然很重。`diag/pallet_disp_xy_mean` 全程最高到 `0.5713 m`，尾部 5 个窗口均值约 `0.4811 m`；`diag/pallet_disp_xy_inserted_mean` 在尾部甚至到 `0.5104 m`。这说明当前大量“插入”仍然是带明显推盘的 dirty insert。
- 因此这轮 run 的最准确定义是: **已经把 `clean_insert_hold` 从“概念方向”推进到“可重复出现弱成功信号”，但离 `clean insert + stable hold` 还有明显距离。**

---

## 2. 关键指标摘录

| 指标 | 最佳值 | 尾窗最后值 | 尾部 5 窗口均值 | 备注 |
|------|--------|------------|------------------|------|
| `phase/frac_inserted` | `0.6719` (`iter 38`) | `0.5000` | `0.5156` | 插入能力已明显打开 |
| `phase/frac_inserted_push_free` | `0.0625` (`iter 71`) | `0.0156` | `0.0125` | push-free 插入仍很弱 |
| `phase/frac_clean_insert_ready` | `0.0625` (`iter 71`) | `0.0156` | `0.0125` | clean insert 信号已出现, 但很稀疏 |
| `phase/frac_hold_entry` | `0.0625` (`iter 71`) | `0.0156` | `0.0125` | hold 入口可重复出现 |
| `phase/frac_success_geom_strict` | `0.0625` (`iter 71`) | `0.0156` | `0.0125` | 严格几何成功已有重复信号 |
| `phase/frac_success` | `0.0312` (`iter 29`) | `0.0156` | `0.0031` | 真实 success 仍很稀疏 |
| `diag/max_hold_counter` | `10` (`iter 21`) | `9` | `5.6` | hold 计数器已明显打开 |
| `diag/pallet_disp_xy_mean` | `0.5713 m` (`iter 88`) | `0.4835 m` | `0.4811 m` | 推盘仍然明显 |
| `diag/pallet_disp_xy_inserted_mean` | `0.5241 m` (`iter 198`) | `0.5104 m` | `0.4514 m` | 插入态推盘尤其严重 |

---

## 3. 整轮里发生了什么

1. `inserted` 基本已经稳定打开。`phase/frac_inserted` 在 `200` 个窗口里有 `194` 个窗口非零，说明策略已经能经常把货叉推进到插入区域。

2. `hold` 相关信号也已经不是一次性噪声。`diag/max_hold_counter` 在 `200` 个窗口里有 `167` 个窗口非零；`phase/frac_hold_entry` 和 `phase/frac_success_geom_strict` 都各自有 `92` 个窗口非零。

3. `clean insert` 虽然弱，但确实反复出现了。`phase/frac_clean_insert_ready` 与 `phase/frac_inserted_push_free` 都有 `73` 个窗口非零，说明当前 gate 不是完全没起作用。

4. `phase/frac_success` 也已经不是永远为零。整轮有 `22` 个窗口出现非零 success，最后 10 次非零出现在 `iter 126, 132, 150, 152, 167, 173, 182, 184, 193, 199`。

5. 但 clean insert 信号始终远弱于 inserted 信号。也就是说，模型已经学会“插进去”，但还没有学会“干净地插进去并稳住”。

---

## 4. 尾窗判断

最后 5 个窗口 (`iter 195 ~ 199`) 的关键信号如下:

| Iter | `frac_inserted` | `frac_inserted_push_free` | `frac_clean_insert_ready` | `frac_hold_entry` | `frac_success_geom_strict` | `frac_success` | `max_hold_counter` | `pallet_disp_xy_mean` |
|------|-----------------|---------------------------|----------------------------|-------------------|----------------------------|----------------|--------------------|------------------------|
| `195` | `0.5469` | `0.0156` | `0.0156` | `0.0156` | `0.0156` | `0.0000` | `5` | `0.4686 m` |
| `196` | `0.5312` | `0.0156` | `0.0156` | `0.0156` | `0.0156` | `0.0000` | `7` | `0.4878 m` |
| `197` | `0.5000` | `0.0156` | `0.0156` | `0.0156` | `0.0156` | `0.0000` | `4` | `0.4966 m` |
| `198` | `0.5000` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `3` | `0.4688 m` |
| `199` | `0.5000` | `0.0156` | `0.0156` | `0.0156` | `0.0156` | `0.0156` | `9` | `0.4835 m` |

尾窗最重要的信息有两条:

1. `success` 在最后一轮重新打开了, 这说明当前策略已经能够在尾段再次回到可成功区域, 不是完全漂掉。
2. 但推盘量依旧很大, 所以这个 success 更像“脏插里的弱成功点”, 不是已经得到稳定 clean insert。

---

## 5. 对照 `Phase 2` 验收标准

| 文档标准 | 本次观测 | 判断 |
|----------|----------|------|
| `hold_counter` 需连续多个窗口非零 | `diag/max_hold_counter` 非零 `167/200` 窗口, 尾部 5 窗口均值 `5.6` | 已达成 |
| `phase/frac_success_geom_strict` 需要反复出现, 不是单点抖动 | 非零 `92/200` 窗口, 最佳 `0.0625` | 已基本达成 |
| 不能只会“插进去”, 还要能出现 `clean_insert_ready` | 非零 `73/200` 窗口, 但尾窗仍稀疏 | 部分达成 |
| `phase/frac_success` 应该从极稀疏走向可稳定复现 | 非零 `22/200` 窗口, 尾窗仅最后一轮非零 | 未达成 |
| `diag/pallet_disp_xy_mean` 不应随着插入提升而显著恶化 | 尾部均值 `0.4811 m`, 插入态位移尾部均值 `0.4514 m` | 未达成 |
| `frac_inserted_push_free` 应该与 `frac_inserted` 的差距收敛 | 尾窗 `0.5156` vs `0.0125`，差距仍极大 | 未达成 |

---

## 6. 这轮实验带来的新结论

1. `clean_insert_hold` 这条线值得继续，不应该回退。因为它已经把 `success` 从“几乎只在中段偶发”推进到了“尾窗也能再出现”。

2. 当前主瓶颈已经很清楚，不是“不会接近”也不是“hold 逻辑没接通”，而是**dirty insert 太多**。大部分插入都伴随明显推盘，所以 clean insert 与 success 很难稳定下来。

3. 现在单独 gate `r_d` 还不够强。虽然 gate 已经在工作，但它还没有把“脏插策略”压到足够低。

4. 下一轮重点应该继续压 post-insert 的 dirty insert，而不是再去加速远场接近。因为 `inserted` 已经很多，再继续强化接近，收益不会是当前最大项。

---

## 7. 建议的下一步

1. 继续停留在 `exp/exp8_3_clean_insert_hold`，不要切回旧分支。

2. 做第二批 `clean_insert_hold` 改动，重点是:
   - 不只 gate `r_d`，把 `r_cd` 也纳入 post-insert clean gate。
   - 视情况把近场 `r_cpsi` 也纳入 gate，减少“带偏航推进也能拿奖励”的空间。
   - 收紧 `clean_insert_gate_floor` 和 `clean_insert_push_sigma_m`，让 dirty insert 时正奖励塌得更明显。
   - 考虑加入显式 `dirty insert / push pallet` 负奖励，而不是只靠 gate 衰减。

3. 第二批改动后，优先再跑同口径的 `near-field 50 iter / 256x256` 做快速筛查；只有当尾窗里的 `frac_inserted_push_free`、`frac_clean_insert_ready`、`frac_success` 明显变稳，才值得再上 `200 iter`。

4. 另外单独记录一个工程问题: 本次文本日志又停在 `199/200`，虽然已经保存到 `model_199.pt`，但仍没有看到完整收尾打印。这个收尾问题后面仍建议单独查一下。

---

## 8. 一句话判断

**这轮 `iter200_256cam` 已经把 `Phase 2` 推进到“可重复出现弱 clean-insert / hold / success 信号”的阶段，但核心瓶颈仍是 dirty insert 和推盘过大，因此还不能算完成 `clean insert + stable hold`。**
