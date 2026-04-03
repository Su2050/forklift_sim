# Exp9.0 Tip-Gate A/B Multiseed Result

Seeds: `seed42, seed43`

## 1. Per-Seed Quick View

| Seed | strict hold | relaxed hold | strict success | relaxed success | strict band0175 | relaxed band0175 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| seed42 | 0.0008 | 0.0023 | 0.0000 | 0.0016 | 0.0086 | 0.0016 |
| seed43 | 0.0008 | 0.0039 | 0.0000 | 0.0008 | 0.0109 | 0.0000 |

## 2. Mean Last-N Across Seeds

窗口：最后 `20` 个 iteration

| Metric | strict mean | relaxed 0.175 mean | delta (relaxed-strict) |
| --- | ---: | ---: | ---: |
| `phase/frac_inserted` | 0.5953 | 0.5652 | -0.0301 |
| `phase/frac_prehold_reachable_band` | 0.0090 | 0.0004 | -0.0086 |
| `phase/frac_prehold_reachable_band_companion` | 0.0098 | 0.0008 | -0.0090 |
| `diag/prehold_reachable_band_frac_of_inserted` | 0.0154 | 0.0007 | -0.0147 |
| `diag/prehold_reachable_band_companion_frac_of_inserted` | 0.0164 | 0.0013 | -0.0151 |
| `phase/frac_hold_entry` | 0.0008 | 0.0031 | 0.0023 |
| `phase/frac_success` | 0.0000 | 0.0012 | 0.0012 |
| `phase/frac_success_strict` | 0.0000 | 0.0000 | 0.0000 |
| `err/center_lateral_inserted_mean` | 0.3593 | 0.4118 | 0.0525 |
| `err/tip_lateral_inserted_mean` | 0.3611 | 0.4179 | 0.0568 |
| `err/yaw_deg_inserted_mean` | 6.3513 | 7.1823 | 0.8310 |
| `diag/max_hold_counter` | 0.4250 | 1.3512 | 0.9262 |
