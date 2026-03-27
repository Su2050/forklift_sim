## exp8.3 steer usage diagnostics (2026-03-27)

### Scope
We ran two diagnostics on strong checkpoints to test whether steering is actually being used:

1. **Zero-steer eval**: set steer action to `0` before stepping.
2. **Misalignment grid eval**: fixed-stage1 grid of `y ∈ {-0.10,-0.05,0,0.05,0.10}`, `yaw ∈ {-4,-2,0,2,4}` at `x=-3.40`.

Checkpoints:
- `r1_seed42` -> `.../2026-03-27_01-16-53_exp83_bonusw1p0_repro_r1_seed42_iter100_256cam/model_99.pt`
- `r1_seed44` -> `.../2026-03-27_03-04-32_exp83_bonusw1p0_repro_r1_seed44_iter100_256cam/model_99.pt`

Outputs:
- Zero-steer summaries:  
  `outputs/exp83_steer_usage_diagnostics/zero_steer/exp83_zero_steer_r1_seed42_iter100_summary.json`  
  `outputs/exp83_steer_usage_diagnostics/zero_steer/exp83_zero_steer_r1_seed44_iter100_summary.json`
- Grid summaries / rows:  
  `outputs/exp83_steer_usage_diagnostics/misalignment_grid/exp83_grid_r1_seed42_iter100_*`  
  `outputs/exp83_steer_usage_diagnostics/misalignment_grid/exp83_grid_r1_seed44_iter100_*`

---

### A. Zero-steer eval (near-field)
Both strong checkpoints remain almost unchanged when steering is forced to `0`.

| Checkpoint | success_rate_ep | ever_inserted_push_free_rate | mean_abs_steer_raw |
|---|---:|---:|---:|
| r1_seed42 (zero-steer) | 0.9531 | 0.8750 | 0.0322 |
| r1_seed44 (zero-steer) | 0.9688 | 0.9219 | 0.00624 |

**Interpretation:** steering is not a critical contributor for these strong checkpoints under current near-field reset. The policy is largely “drive-forward dominant.”

---

### B. Misalignment grid (normal vs zero-steer)

Summary-level metrics are **identical** between normal and zero-steer:

| Checkpoint | Mode | success_rate_ep | ever_inserted_push_free_rate | timeout_frac | mean_max_pallet_disp_xy |
|---|---|---:|---:|---:|---:|
| r1_seed42 | normal | 0.60 | 0.40 | 0.40 | 2.3159 |
| r1_seed42 | zero-steer | 0.60 | 0.44 | 0.40 | 2.2238 |
| r1_seed44 | normal | 0.60 | 0.44 | 0.40 | 2.8603 |
| r1_seed44 | zero-steer | 0.60 | 0.44 | 0.40 | 2.8269 |

Grid success points are **identical** between normal and zero-steer, for both checkpoints.  
There are `15/25` success points, concentrated in a narrow band:

Success grid (y, yaw):
- `y ∈ {-0.10,-0.05}` with `yaw ∈ {0,2,4}`
- `y ∈ {0.00,0.05}` with `yaw ∈ {-2,0,2}`
- `y ∈ {0.10}` with `yaw ∈ {-4,-2,0}`

This pattern is **unchanged** when steering is forced to zero.

**Interpretation:** the “success basin” is narrow and aligned with the near-field reset; steering corrections are not expanding the basin at all.

---

### Conclusion (direct answer)
**Normal vs zero-steer shows essentially no difference in correction range.**  
This is strong evidence that the current policy’s success is **not driven by steering**, but by forward motion under a very aligned reset distribution.

---

### Immediate implications
1. The current near-field curriculum is too “pre-aligned,” allowing a forward-dominant policy to succeed.
2. If we want robust insertions, we must **force the policy to use steering** by widening `y/yaw` in stage1 or adding pre-insert alignment shaping.
3. Further reward tuning without fixing this will keep producing fragile success that does not generalize.
