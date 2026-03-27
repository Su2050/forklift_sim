# `exp8.3` `bonusw=1.0` 同口径复现实验 `100 iter` 统一 `eval` 结果

> 日期: `2026-03-27`
> 分支: `exp/exp8_3_clean_insert_hold`
> 范围: 汇总 `bonusw=1.0` 同口径复现实验 `seed42/43/44 x r1/r2` 共 `6` 个 `100 iter` checkpoint 的统一 deterministic eval 结果
> 说明: 本文是 `training tail summary` 的配套文档，口径以统一 `eval` 为准

---

## 1. 实验设置

- 训练脚本: `scripts/run_exp83_bonusw1p0_repro_multiseed_batch.sh`
- 训练启动脚本: `scripts/run_exp83_bonusw1p0_repro_multiseed_batch_nohup.sh`
- 统一 eval 脚本: `scripts/run_exp83_bonusw1p0_repro_eval_suite.sh`
- 统一 eval 启动脚本: `scripts/run_exp83_bonusw1p0_repro_eval_suite_nohup.sh`
- 输出目录: `outputs/exp83_eval_bonusw1p0_repro_iter100`

统一 eval 口径固定为:

- `num_envs = 16`
- `rollouts = 4`
- `64` 个 episode / checkpoint
- deterministic policy
- near-field
- `256x256 camera`

因此，这 `6` 个 checkpoint 之间是同口径可直接比较的。

---

## 2. 一页结论

- 这批 `100 iter` 复现实验的统一 eval 已经 `6/6` 全部完成。
- 最直观的结果是:
  - `2` 条 run 几乎满分成功
  - `1` 条 run 中等偏好
  - `3` 条 run 完全失败
- 最强的两个 run 是:
  - `r1_seed42`: `success_rate_ep = 0.9844`
  - `r1_seed44`: `success_rate_ep = 0.9844`
- 次优 run 是:
  - `r2_seed42`: `success_rate_ep = 0.7656`
- 完全失败的三个 run 是:
  - `r1_seed43`: `success_rate_ep = 0.0000`
  - `r2_seed43`: `success_rate_ep = 0.0000`
  - `r2_seed44`: `success_rate_ep = 0.0000`

最重要的新认识有两个:

1. `100 iter` 并不是“整体还是一般”，而是出现了非常明显的**两极分化**。
2. 训练尾段文档低估了 `r1_seed44`，统一 eval 显示它实际上是强 checkpoint，这再次说明不能只靠训练尾窗判断最终质量。

一句话总结:

**这批 `100 iter` 复现实验已经证明，`bonusw=1.0` 可以在部分 run 上达到接近最近最佳水平，但跨 seed / repeat 的稳定性仍然很差，问题已经从“有没有好解”变成了“为什么只有部分 run 能进好解”。**

---

## 3. Per-Run 统一 `eval` 结果

| run | success_ep | ever_inserted | ever_push_free | ever_hold | ever_clean_ready | ever_dirty | timeout | mean_ep_len | mean_max_disp | mean_max_hold |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `r1_seed42` | `0.9844` | `1.0000` | `0.8594` | `0.9844` | `0.8594` | `0.1406` | `0.0156` | `202.9` | `0.0841` | `8.8594` |
| `r1_seed43` | `0.0000` | `0.7969` | `0.0000` | `0.0000` | `0.0000` | `0.7969` | `0.7969` | `1067.7` | `1.2568` | `0.0000` |
| `r1_seed44` | `0.9844` | `1.0000` | `0.9375` | `0.9844` | `0.9375` | `0.0938` | `0.0000` | `160.9` | `0.0489` | `8.8594` |
| `r2_seed42` | `0.7656` | `1.0000` | `0.6094` | `0.7812` | `0.6094` | `0.4219` | `0.0312` | `294.0` | `0.2656` | `6.9688` |
| `r2_seed43` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `1.0000` | `1079.0` | `0.0000` | `0.0000` |
| `r2_seed44` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `354.9` | `0.0000` | `0.0000` |

从 per-run 结果看:

- `r1_seed42` 和 `r1_seed44` 都已经进入 very strong policy 区间
- `r2_seed42` 能明显插入并多次 hold，但 dirty insert 仍然偏高
- `r1_seed43` 是典型的“会插但全是 dirty insert”
- `r2_seed43` 是典型的“完全不插且超时”
- `r2_seed44` 则是“完全不插，但也不是 timeout 型失败”

---

## 4. Family-Level Aggregate

### 4.1 全部 `6` 条 run 总体均值

| family | success_ep | ever_inserted | ever_push_free | ever_hold | ever_dirty | timeout | mean_max_disp | mean_max_hold |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `all 6 mean` | `0.4557` | `0.6328` | `0.4010` | `0.4583` | `0.2422` | `0.3073` | `0.2759` | `4.1146` |

这个均值本身已经比此前的 `50 iter` 完整 sweep 更高，但它掩盖了内部强烈的 bimodal 结构，所以只能作为“总体水平”参考，不能替代 per-run 判断。

### 4.2 按 `repeat` 聚合

| repeat | success_ep mean+-std | ever_inserted mean+-std | ever_push_free mean+-std | ever_hold mean+-std | ever_dirty mean+-std | timeout mean+-std | mean_max_disp mean+-std | mean_max_hold mean+-std |
|---|---|---|---|---|---|---|---|---|
| `r1` | `0.6563 +- 0.4640` | `0.9323 +- 0.0958` | `0.5990 +- 0.4247` | `0.6563 +- 0.4640` | `0.3438 +- 0.3210` | `0.2708 +- 0.3720` | `0.4633 +- 0.5613` | `5.9062 +- 4.1763` |
| `r2` | `0.2552 +- 0.3609` | `0.3333 +- 0.4714` | `0.2031 +- 0.2873` | `0.2604 +- 0.3683` | `0.1406 +- 0.1989` | `0.3438 +- 0.4642` | `0.0885 +- 0.1252` | `2.3229 +- 3.2851` |

按 `repeat` 看，`r1` 明显强于 `r2`：

- `r1` 更容易把 insertion 真正转成 `push_free / hold / success`
- `r2` 平均更保守，插入与 hold 都更差
- 但 `r2` 的低位移并不代表更好，而是有不少 case 根本没插

### 4.3 按 `seed` 聚合

| seed | success_ep mean+-std | ever_inserted mean+-std | ever_push_free mean+-std | ever_hold mean+-std | ever_dirty mean+-std | timeout mean+-std | mean_max_disp mean+-std | mean_max_hold mean+-std |
|---|---|---|---|---|---|---|---|---|
| `42` | `0.8750 +- 0.1094` | `1.0000 +- 0.0000` | `0.7344 +- 0.1250` | `0.8828 +- 0.1016` | `0.2812 +- 0.1406` | `0.0234 +- 0.0078` | `0.1749 +- 0.0908` | `7.9141 +- 0.9453` |
| `43` | `0.0000 +- 0.0000` | `0.3984 +- 0.3984` | `0.0000 +- 0.0000` | `0.0000 +- 0.0000` | `0.3984 +- 0.3984` | `0.8984 +- 0.1016` | `0.6284 +- 0.6284` | `0.0000 +- 0.0000` |
| `44` | `0.4922 +- 0.4922` | `0.5000 +- 0.5000` | `0.4688 +- 0.4688` | `0.4922 +- 0.4922` | `0.0469 +- 0.0469` | `0.0000 +- 0.0000` | `0.0245 +- 0.0245` | `4.4297 +- 4.4297` |

按 `seed` 看，差异比按 `repeat` 还大:

- `seed42` 很稳，是当前最可靠的 seed
- `seed43` 两次都完全失败，是当前最差 seed
- `seed44` 呈现出一次极强、一次完全失败的两极结构

---

## 5. 关键发现

### 5.1 `r1_seed44` 是这轮最大的惊喜

训练尾段文档里，`r1_seed44` 看起来只是“有一些 clean signal，但没有成功”。  
但统一 eval 结果显示它实际上非常强:

- `success_rate_ep = 0.9844`
- `ever_inserted_push_free_rate = 0.9375`
- `ever_hold_entry_rate = 0.9844`
- `mean_max_pallet_disp_xy = 0.0489`

这说明:

- 单看训练尾窗，可能会严重低估某些 checkpoint
- 统一 eval 仍然是判断 checkpoint 真正价值的必要步骤

### 5.2 `seed42` 仍然是当前最稳定的正例

`seed42` 两次 repeat 都不错:

- `r1_seed42`: 近乎满分
- `r2_seed42`: 虽然更脏，但仍有 `0.7656` success

这说明 `bonusw=1.0 + 100 iter` 的这条路并不是偶然单点成功，而是至少对某些 seed 已经能重复进入高质量解。

### 5.3 `seed43` 依然是核心难点

`seed43` 两次都失败，但失败方式不同:

- `r1_seed43`: 大量 dirty insert，`success=0`
- `r2_seed43`: 完全不插，且 `timeout=1.0`

这说明 `seed43` 仍然是最能暴露系统不稳定性的 case。  
当前 reward 和训练动力学仍然没有把它稳定推入 clean-insert basin。

### 5.4 `seed44` 暗示当前系统是“双峰解”

`seed44` 的两次结果几乎是两个极端:

- `r1_seed44`: 非常强
- `r2_seed44`: 完全失败

这进一步支持一个判断:

**当前系统不是围绕单一均值轻微波动，而是在两个 attractor 之间跳。**

一个 attractor 对应:

- clean insert
- hold
- high success

另一个 attractor 对应:

- 不插
- 或非常有限的尝试
- 最终无成功

### 5.5 `100 iter` 的价值主要体现在“允许一部分 run 真正成型”，但没有消除方差

和之前 `50 iter` 的结果相比，这批 `100 iter` 的最好 run 已经很强。  
所以把迭代数拉到 `100` 并不是没用。

但它没有解决的仍然是:

- 为什么只有部分 run 进入好 attractor
- 为什么另一些 run 还会停留在 dirty 或不插 attractor

所以当前阶段最该回答的问题已经不再是:

- “100 iter 有没有提升？”

而是:

- “决定 run 落进哪一个 attractor 的关键早期因素是什么？”

---

## 6. 与最近最佳结果的关系

这批 `100 iter` 复现实验已经出现了接近最近最佳水平的 checkpoint:

- `r1_seed42`: `0.9844`
- `r1_seed44`: `0.9844`

它们已经接近此前最强的:

- `exp83_eval_bonus_seed42`: `1.0000`
- `exp83_eval_bonus_seed43`: `1.0000`

所以结论不是“这批还不行”，而是:

- **这批里已经有很强的 checkpoint**
- **但还没有把强结果稳定推广到全部 seed / repeat**

---

## 7. 当前最合理的判断

基于统一 eval，当前最稳妥的判断是:

1. `bonusw=1.0` 这条路线已经足以产出接近最近最佳水平的强 checkpoint。
2. `100 iter` 比 `50 iter` 更容易让部分 run 成型。
3. 但系统仍然高度不稳定，尤其是 `seed43`。
4. 当前最重要的问题已经不是“这个 reward 行不行”，而是“如何让更多 run 稳定落在高成功 attractor 上”。

---

## 8. 一句话判断

**这 6 个统一 eval 结果表明，`bonusw=1.0 + 100 iter` 已经能稳定地产生部分非常强的 checkpoint，但整体仍然是明显的双峰分布: 两条 run 接近满分，一条中等可用，三条完全失败。当前瓶颈已经不是有没有好解，而是怎么提高好解的复现率。**
