# Exp9.0 Tip-Gate Short A/B Result

日期：`2026-04-02 20:04:19`

## 1. 运行设置

| Run | Log | Final iter |
| --- | --- | ---: |
| strict (`tip_entry=0.12`) | `20260402_181030_seed43_train_exp9_0_tipgate_ab_long_o3_dualband_strict_seed43_iter200.log` | `199/200` |
| relaxed (`tip_entry=0.175`) | `20260402_181030_seed43_train_exp9_0_tipgate_ab_long_o3_dualband_relaxed0175_seed43_iter200.log` | `199/200` |

## 2. Last-N Mean 对比

窗口：最后 `20` 个 iteration

| Metric | strict | relaxed 0.175 | delta (relaxed-strict) |
| --- | ---: | ---: | ---: |
| `phase/frac_inserted` | 0.4984 | 0.5492 | 0.0508 |
| `phase/frac_prehold_reachable_band` | 0.0109 | 0.0000 | -0.0109 |
| `phase/frac_prehold_reachable_band_companion` | 0.0109 | 0.0000 | -0.0109 |
| `diag/prehold_reachable_band_frac_of_inserted` | 0.0211 | 0.0000 | -0.0211 |
| `diag/prehold_reachable_band_companion_frac_of_inserted` | 0.0211 | 0.0000 | -0.0211 |
| `phase/frac_hold_entry` | 0.0008 | 0.0039 | 0.0031 |
| `phase/frac_success` | 0.0000 | 0.0008 | 0.0008 |
| `phase/frac_success_strict` | 0.0000 | 0.0000 | 0.0000 |
| `err/center_lateral_inserted_mean` | 0.3391 | 0.4419 | 0.1028 |
| `err/tip_lateral_inserted_mean` | 0.3375 | 0.4411 | 0.1037 |
| `err/yaw_deg_inserted_mean` | 6.6484 | 7.0892 | 0.4408 |
| `diag/max_hold_counter` | 0.6500 | 1.4502 | 0.8002 |

## 3. 首次命中与峰值

| Metric | strict first>0 | relaxed first>0 | strict peak@iter | relaxed peak@iter |
| --- | ---: | ---: | --- | --- |
| `phase/frac_prehold_reachable_band` | 0 | 80 | 0.0469 @ 183 | 0.0156 @ 80 |
| `phase/frac_prehold_reachable_band_companion` | 0 | 80 | 0.0469 @ 183 | 0.0156 @ 80 |
| `phase/frac_hold_entry` | 0 | 2 | 0.0312 @ 127 | 0.0312 @ 104 |
| `phase/frac_success` | 53 | 2 | 0.0156 @ 53 | 0.0156 @ 2 |
| `phase/frac_success_strict` | n/a | n/a | 0.0000 @ 0 | 0.0000 @ 0 |

## 4. 事件计数

| Event | strict | relaxed 0.175 |
| --- | ---: | ---: |
| `phase/frac_prehold_reachable_band > 0` iterations | 84 | 2 |
| `phase/frac_prehold_reachable_band_companion > 0` iterations | 86 | 2 |
| `phase/frac_hold_entry > 0` iterations | 9 | 28 |
| `phase/frac_success > 0` iterations | 2 | 4 |
| `phase/frac_prehold_reachable_band > phase/frac_hold_entry` iterations | 82 | 0 |
| `phase/frac_prehold_reachable_band_companion > phase/frac_hold_entry` iterations | 83 | 0 |

## 5. 关键判读

- strict 组 `phase/frac_prehold_reachable_band - phase/frac_hold_entry` 的 last-20 平均差为 `0.0101`。
- relaxed 组同一差值的 last-20 平均差为 `-0.0039`。
- 这更像 strict 组里确实存在“到达 0.17 带但没进 hold”的堆积，而 relaxed gate 把其中一部分转化成了 hold。
- relaxed 组的 `phase/frac_success` 更高，说明放宽 tip gate 至少在短跑阶段对成功闭环有正向作用。

## 6. 快速结论

- strict last-20: `band017=0.0109`, `band0175=0.0109`, `hold=0.0008`, `success=0.0000`
- relaxed last-20: `band017=0.0000`, `band0175=0.0000`, `hold=0.0039`, `success=0.0008`
