# Exp9.0 Tip-Gate Short A/B Result

日期：`2026-04-02 17:37:56`

## 1. 运行设置

| Run | Log | Final iter |
| --- | --- | ---: |
| strict (`tip_entry=0.12`) | `20260402_multiseed_debug_seed42_train_exp9_0_tipgate_ab_multiseed_debug_strict_seed42_iter1.log` | `0/1` |
| relaxed (`tip_entry=0.175`) | `20260402_multiseed_debug_seed42_train_exp9_0_tipgate_ab_multiseed_debug_relaxed0175_seed42_iter1.log` | `0/1` |

## 2. Last-N Mean 对比

窗口：最后 `20` 个 iteration

| Metric | strict | relaxed 0.175 | delta (relaxed-strict) |
| --- | ---: | ---: | ---: |
| `phase/frac_inserted` | 1.0000 | 1.0000 | 0.0000 |
| `phase/frac_prehold_reachable_band` | 0.0000 | 0.0000 | 0.0000 |
| `phase/frac_prehold_reachable_band_companion` | 0.0000 | 0.0000 | 0.0000 |
| `diag/prehold_reachable_band_frac_of_inserted` | 0.0000 | 0.0000 | 0.0000 |
| `diag/prehold_reachable_band_companion_frac_of_inserted` | 0.0000 | 0.0000 | 0.0000 |
| `phase/frac_hold_entry` | 0.0000 | 0.0000 | 0.0000 |
| `phase/frac_success` | 0.0000 | 0.0000 | 0.0000 |
| `phase/frac_success_strict` | 0.0000 | 0.0000 | 0.0000 |
| `err/center_lateral_inserted_mean` | 0.8027 | 0.8137 | 0.0110 |
| `err/tip_lateral_inserted_mean` | 0.9446 | 0.9638 | 0.0192 |
| `err/yaw_deg_inserted_mean` | 13.6798 | 14.4846 | 0.8048 |
| `diag/max_hold_counter` | 0.0000 | 0.0000 | 0.0000 |

## 3. 首次命中与峰值

| Metric | strict first>0 | relaxed first>0 | strict peak@iter | relaxed peak@iter |
| --- | ---: | ---: | --- | --- |
| `phase/frac_prehold_reachable_band` | n/a | n/a | 0.0000 @ 0 | 0.0000 @ 0 |
| `phase/frac_prehold_reachable_band_companion` | n/a | n/a | 0.0000 @ 0 | 0.0000 @ 0 |
| `phase/frac_hold_entry` | n/a | n/a | 0.0000 @ 0 | 0.0000 @ 0 |
| `phase/frac_success` | n/a | n/a | 0.0000 @ 0 | 0.0000 @ 0 |
| `phase/frac_success_strict` | n/a | n/a | 0.0000 @ 0 | 0.0000 @ 0 |

## 4. 事件计数

| Event | strict | relaxed 0.175 |
| --- | ---: | ---: |
| `phase/frac_prehold_reachable_band > 0` iterations | 0 | 0 |
| `phase/frac_prehold_reachable_band_companion > 0` iterations | 0 | 0 |
| `phase/frac_hold_entry > 0` iterations | 0 | 0 |
| `phase/frac_success > 0` iterations | 0 | 0 |
| `phase/frac_prehold_reachable_band > phase/frac_hold_entry` iterations | 0 | 0 |
| `phase/frac_prehold_reachable_band_companion > phase/frac_hold_entry` iterations | 0 | 0 |

## 5. 关键判读

- strict 组 `phase/frac_prehold_reachable_band - phase/frac_hold_entry` 的 last-20 平均差为 `0.0000`。
- relaxed 组同一差值的 last-20 平均差为 `0.0000`。
- 目前还没有看到非常强的“reachable band -> hold 转化”证据，可能需要更长一点的短跑或更多 seed。
- relaxed 组暂时没有在 `phase/frac_success` 上形成明显优势，后续要重点看 hold 和 success_strict 的分叉。

## 6. 快速结论

- strict last-20: `band017=0.0000`, `band0175=0.0000`, `hold=0.0000`, `success=0.0000`
- relaxed last-20: `band017=0.0000`, `band0175=0.0000`, `hold=0.0000`, `success=0.0000`
