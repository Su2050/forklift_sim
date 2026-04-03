# Exp9.0 Tip-Gate Short A/B Result

日期：`2026-04-02 19:07:23`

## 1. 运行设置

| Run | Log | Final iter |
| --- | --- | ---: |
| strict (`tip_entry=0.12`) | `20260402_181030_seed42_train_exp9_0_tipgate_ab_long_o3_dualband_strict_seed42_iter200.log` | `199/200` |
| relaxed (`tip_entry=0.175`) | `20260402_181030_seed42_train_exp9_0_tipgate_ab_long_o3_dualband_relaxed0175_seed42_iter200.log` | `199/200` |

## 2. Last-N Mean 对比

窗口：最后 `20` 个 iteration

| Metric | strict | relaxed 0.175 | delta (relaxed-strict) |
| --- | ---: | ---: | ---: |
| `phase/frac_inserted` | 0.6922 | 0.5813 | -0.1109 |
| `phase/frac_prehold_reachable_band` | 0.0070 | 0.0008 | -0.0062 |
| `phase/frac_prehold_reachable_band_companion` | 0.0086 | 0.0016 | -0.0070 |
| `diag/prehold_reachable_band_frac_of_inserted` | 0.0097 | 0.0014 | -0.0083 |
| `diag/prehold_reachable_band_companion_frac_of_inserted` | 0.0117 | 0.0026 | -0.0091 |
| `phase/frac_hold_entry` | 0.0008 | 0.0023 | 0.0016 |
| `phase/frac_success` | 0.0000 | 0.0016 | 0.0016 |
| `phase/frac_success_strict` | 0.0000 | 0.0000 | 0.0000 |
| `err/center_lateral_inserted_mean` | 0.3795 | 0.3818 | 0.0023 |
| `err/tip_lateral_inserted_mean` | 0.3848 | 0.3947 | 0.0099 |
| `err/yaw_deg_inserted_mean` | 6.0541 | 7.2754 | 1.2212 |
| `diag/max_hold_counter` | 0.2000 | 1.2522 | 1.0522 |

## 3. 首次命中与峰值

| Metric | strict first>0 | relaxed first>0 | strict peak@iter | relaxed peak@iter |
| --- | ---: | ---: | --- | --- |
| `phase/frac_prehold_reachable_band` | 8 | 21 | 0.0312 @ 57 | 0.0156 @ 21 |
| `phase/frac_prehold_reachable_band_companion` | 8 | 21 | 0.0312 @ 57 | 0.0156 @ 21 |
| `phase/frac_hold_entry` | 0 | 0 | 0.0312 @ 7 | 0.0156 @ 0 |
| `phase/frac_success` | 12 | 7 | 0.0156 @ 12 | 0.0156 @ 7 |
| `phase/frac_success_strict` | n/a | n/a | 0.0000 @ 0 | 0.0000 @ 0 |

## 4. 事件计数

| Event | strict | relaxed 0.175 |
| --- | ---: | ---: |
| `phase/frac_prehold_reachable_band > 0` iterations | 87 | 5 |
| `phase/frac_prehold_reachable_band_companion > 0` iterations | 90 | 6 |
| `phase/frac_hold_entry > 0` iterations | 31 | 28 |
| `phase/frac_success > 0` iterations | 6 | 8 |
| `phase/frac_prehold_reachable_band > phase/frac_hold_entry` iterations | 73 | 0 |
| `phase/frac_prehold_reachable_band_companion > phase/frac_hold_entry` iterations | 76 | 0 |

## 5. 关键判读

- strict 组 `phase/frac_prehold_reachable_band - phase/frac_hold_entry` 的 last-20 平均差为 `0.0062`。
- relaxed 组同一差值的 last-20 平均差为 `-0.0016`。
- 这更像 strict 组里确实存在“到达 0.17 带但没进 hold”的堆积，而 relaxed gate 把其中一部分转化成了 hold。
- relaxed 组的 `phase/frac_success` 更高，说明放宽 tip gate 至少在短跑阶段对成功闭环有正向作用。

## 6. 快速结论

- strict last-20: `band017=0.0070`, `band0175=0.0086`, `hold=0.0008`, `success=0.0000`
- relaxed last-20: `band017=0.0008`, `band0175=0.0016`, `hold=0.0023`, `success=0.0016`
