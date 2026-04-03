# Exp9.0 Tip-Gate Short A/B Result

日期：`2026-04-02 17:30:09`

## 1. 运行设置

| Run | Log | Final iter |
| --- | --- | ---: |
| strict (`tip_entry=0.12`) | `20260402_165500_train_exp9_0_tipgate_ab_short_o3_strict_seed42_iter80.log` | `79/80` |
| relaxed (`tip_entry=0.175`) | `20260402_165500_train_exp9_0_tipgate_ab_short_o3_relaxed0175_seed42_iter80.log` | `79/80` |

## 2. Last-N Mean 对比

窗口：最后 `20` 个 iteration

| Metric | strict | relaxed 0.175 | delta (relaxed-strict) |
| --- | ---: | ---: | ---: |
| `phase/frac_inserted` | 0.6375 | 0.6547 | 0.0172 |
| `phase/frac_prehold_reachable_band` | 0.0008 | 0.0000 | -0.0008 |
| `diag/prehold_reachable_band_frac_of_inserted` | 0.0013 | 0.0000 | -0.0013 |
| `phase/frac_hold_entry` | 0.0016 | 0.0008 | -0.0008 |
| `phase/frac_success` | 0.0008 | 0.0000 | -0.0008 |
| `phase/frac_success_strict` | 0.0000 | 0.0000 | 0.0000 |
| `err/center_lateral_inserted_mean` | 0.3820 | 0.3633 | -0.0186 |
| `err/tip_lateral_inserted_mean` | 0.3863 | 0.3655 | -0.0208 |
| `err/yaw_deg_inserted_mean` | 7.1080 | 6.6011 | -0.5069 |
| `diag/max_hold_counter` | 2.9781 | 0.7001 | -2.2781 |

## 3. 首次命中与峰值

| Metric | strict first>0 | relaxed first>0 | strict peak@iter | relaxed peak@iter |
| --- | ---: | ---: | --- | --- |
| `phase/frac_prehold_reachable_band` | 2 | n/a | 0.0312 @ 14 | 0.0000 @ 0 |
| `phase/frac_hold_entry` | 2 | 2 | 0.0156 @ 2 | 0.0156 @ 2 |
| `phase/frac_success` | 18 | 38 | 0.0156 @ 18 | 0.0156 @ 38 |
| `phase/frac_success_strict` | n/a | n/a | 0.0000 @ 0 | 0.0000 @ 0 |

## 4. 事件计数

| Event | strict | relaxed 0.175 |
| --- | ---: | ---: |
| `phase/frac_prehold_reachable_band > 0` iterations | 19 | 0 |
| `phase/frac_hold_entry > 0` iterations | 10 | 9 |
| `phase/frac_success > 0` iterations | 3 | 1 |
| `phase/frac_prehold_reachable_band > phase/frac_hold_entry` iterations | 16 | 0 |

## 5. 关键判读

- strict 组 `phase/frac_prehold_reachable_band - phase/frac_hold_entry` 的 last-20 平均差为 `-0.0008`。
- relaxed 组同一差值的 last-20 平均差为 `-0.0008`。
- 目前还没有看到非常强的“reachable band -> hold 转化”证据，可能需要更长一点的短跑或更多 seed。
- relaxed 组暂时没有在 `phase/frac_success` 上形成明显优势，后续要重点看 hold 和 success_strict 的分叉。

## 6. 快速结论

- strict last-20: `band=0.0008`, `hold=0.0016`, `success=0.0008`
- relaxed last-20: `band=0.0000`, `hold=0.0008`, `success=0.0000`
