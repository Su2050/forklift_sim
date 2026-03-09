# Stage1 s1.0zc Experiment Summary

## Context

- Branch: `exp/stage1-cv-s1.0zc`
- Baseline reference log: `/home/uniubi/projects/forklift_sim/logs/20260308_204352_train_s1.0zc.log`
- Goal of this round: use control-variable smoke runs to identify which Stage 1 reset factors are worth promoting into longer confirmation runs.

## Baseline Reference

Reference window: tail 50 iterations of `20260308_204352_train_s1.0zc.log`

- `success_rate_ema`: `0.2767`
- `success_rate_total`: `0.2858`
- `Mean episode length`: `792.1428`
- `phase/frac_inserted`: `0.2981`
- `phase/frac_aligned`: `0.0170`
- `err/yaw_deg_near_success`: `7.6409`
- `err/lateral_near_success`: `0.1776`

Interpretation:

- Baseline can insert, but alignment and hold are still weak.
- Main bottleneck remains near-success yaw/alignment, not a pure visibility problem.

## Smoke Results

### `initYaw_narrow`

- Log: `/home/uniubi/projects/forklift_sim/logs/20260308_224555_smoke_train_s1.0zc.log`
- `success_rate_ema` mean: `0.2482`
- `Mean episode length` mean: `432.2043`
- `phase/frac_aligned` mean: `0.1082`
- `err/yaw_deg_near_success` mean: `3.6264`
- `err/lateral_near_success` mean: `0.1411`

Conclusion:

- Strong positive direction.
- This is still the best factor found in the smoke matrix.

### `initYaw_wide`

- Log: `/home/uniubi/projects/forklift_sim/logs/20260308_225011_smoke_train_s1.0zc.log`
- `success_rate_ema` mean: `0.0647`
- `Mean episode length` mean: `599.4337`
- `phase/frac_aligned` mean: `0.0375`
- `err/yaw_deg_near_success` mean: `10.8056`
- `err/lateral_near_success` mean: `0.2018`

Conclusion:

- Clearly negative.
- Eliminate from shortlist.

### `initY_narrow`

- Log: `/home/uniubi/projects/forklift_sim/logs/20260308_230123_smoke_train_s1.0zc.log`
- `success_rate_ema` mean: `0.1651`
- `Mean episode length` mean: `460.9023`
- `phase/frac_aligned` mean: `0.0482`
- `err/yaw_deg_near_success` mean: `6.6928`
- `err/lateral_near_success` mean: `0.1192`

Conclusion:

- Positive, but weaker than `initYaw_narrow`.
- Keep as secondary shortlist candidate.

### `initY_wide`

- Log: `/home/uniubi/projects/forklift_sim/logs/20260308_230742_smoke_train_s1.0zc.log`
- `success_rate_ema` mean: `0.0972`
- `Mean episode length` mean: `571.6837`
- `phase/frac_aligned` mean: `0.0243`
- `err/yaw_deg_near_success` mean: `7.7937`
- `err/lateral_near_success` mean: `0.1926`

Conclusion:

- Negative direction.
- Eliminate from shortlist.

### `tipGate_tight`

- Log: `/home/uniubi/projects/forklift_sim/logs/20260308_231503_smoke_train_s1.0zc.log`
- `success_rate_ema` mean: `0.1234`
- `Mean episode length` mean: `514.2860`
- `phase/frac_inserted` mean: `0.3011`
- `phase/frac_aligned` mean: `0.0352`
- `err/yaw_deg_near_success` mean: `7.1221`
- `err/lateral_near_success` mean: `0.1597`

Conclusion:

- Not obviously harmful.
- Some intermediate metrics improved, but overall leverage is weaker than `initYaw_narrow` and `initY_narrow`.

### `tipGate_loose`

- Log: `/home/uniubi/projects/forklift_sim/logs/20260308_232114_smoke_train_s1.0zc.log`
- `success_rate_ema` mean: `0.1132`
- `Mean episode length` mean: `530.2703`
- `phase/frac_inserted` mean: `0.3122`
- `phase/frac_aligned` mean: `0.0217`
- `err/yaw_deg_near_success` mean: `7.9386`
- `err/lateral_near_success` mean: `0.1732`

Conclusion:

- Similar to `tipGate_tight`, but slightly weaker overall.
- Does not justify promotion ahead of the reset narrowing factors.

## Current Shortlist

Keep:

- `initYaw_narrow`
- `initY_narrow`

Eliminate:

- `initYaw_wide`
- `initY_wide`
- `tipGate_tight`
- `tipGate_loose`

## Decision

- The `tip_y_gate3` line is closed for now.
- It is not the next high-leverage direction.
- Next execution step should be longer confirmation runs for:
  - `initYaw_narrow`
  - `initY_narrow`

## Formal Confirmation

### `initYaw_narrow_formal`

- Log: `/home/uniubi/projects/forklift_sim/logs/20260309_112057_train_s1.0zc.log`
- Window used for comparison: tail 50 iterations
- `success_rate_ema`: `0.4883`
- `success_rate_total`: `0.4942`
- `Mean episode length`: `577.1876`
- `phase/frac_inserted`: `0.3031`
- `phase/frac_aligned`: `0.0316`
- `phase/hold_counter_max`: `6.7000`
- `phase/hold_counter_mean`: `0.0959`
- `err/yaw_deg_near_success`: `4.7896`
- `err/lateral_near_success`: `0.1783`

Relative to baseline tail 50:

- `success_rate_ema`: `+76.5%`
- `success_rate_total`: `+72.9%`
- `Mean episode length`: `-27.1%`
- `phase/frac_inserted`: `+1.7%`
- `phase/frac_aligned`: `+85.9%`
- `err/yaw_deg_near_success`: `-37.3%`
- `err/lateral_near_success`: roughly flat

Conclusion:

- Formal confirmation passed.
- `initYaw_narrow` is now the best validated candidate on the current branch.
- Improvement is not only in smoke metrics; it persists in a longer 300-iteration confirmation run.

### `initY_narrow_formal`

- Log: `/home/uniubi/projects/forklift_sim/logs/20260309_115659_train_s1.0zc.log`
- Window used for comparison: tail 50 iterations
- `success_rate_ema`: `0.3470`
- `success_rate_total`: `0.3686`
- `Mean episode length`: `724.6794`
- `phase/frac_inserted`: `0.2315`
- `phase/frac_aligned`: `0.0340`
- `phase/hold_counter_max`: `5.1200`
- `phase/hold_counter_mean`: `0.0341`
- `err/yaw_deg_near_success`: `7.1272`
- `err/lateral_near_success`: `0.1223`

Relative to baseline tail 50:

- `success_rate_ema`: `+25.4%`
- `success_rate_total`: `+29.0%`
- `Mean episode length`: `-8.5%`
- `phase/frac_inserted`: `-22.3%`
- `phase/frac_aligned`: `+100.0%`
- `err/yaw_deg_near_success`: `-6.7%`
- `err/lateral_near_success`: `-31.1%`

Relative to `initYaw_narrow_formal` tail 50:

- `success_rate_ema`: `-28.9%`
- `success_rate_total`: `-25.4%`
- `Mean episode length`: `+25.6%`
- `phase/frac_inserted`: `-23.6%`
- `phase/frac_aligned`: `+7.6%`
- `err/yaw_deg_near_success`: `+48.8%`
- `err/lateral_near_success`: `-31.4%`

Conclusion:

- Formal confirmation passed against baseline.
- `initY_narrow` remains a valid secondary candidate, mainly by improving lateral alignment quality.
- It is clearly weaker than `initYaw_narrow_formal` on success rate, episode length, inserted fraction, and yaw quality.

## Current Decision

- Stage 2 formal confirmation is complete.
- Best validated single factor: `initYaw_narrow`
- Secondary validated single factor: `initY_narrow`

## Combination Test

### `initYaw_narrow + initY_narrow`

- Log: `/home/uniubi/projects/forklift_sim/logs/20260309_123758_train_s1.0zc.log`
- Window used for comparison: tail 50 iterations
- `success_rate_ema`: `0.5916`
- `success_rate_total`: `0.5974`
- `Mean episode length`: `451.6906`
- `phase/frac_inserted`: `0.3759`
- `phase/frac_aligned`: `0.0527`
- `phase/hold_counter_max`: `8.5200`
- `phase/hold_counter_mean`: `0.1634`
- `err/yaw_deg_near_success`: `10.6148`
- `err/lateral_near_success`: `0.2109`

Relative to `initYaw_narrow_formal` tail 50:

- `success_rate_ema`: `+21.2%`
- `success_rate_total`: `+20.9%`
- `Mean episode length`: `-21.7%`
- `phase/frac_inserted`: `+24.0%`
- `phase/frac_aligned`: `+66.8%`
- `phase/hold_counter_max`: `+27.2%`
- `phase/hold_counter_mean`: `+70.4%`
- `err/yaw_deg_near_success`: worse
- `err/lateral_near_success`: worse

Conclusion:

- This combination is the current best recipe on the branch.
- It clearly beats both validated single-factor runs on the plan's primary decision metrics: success rate, episode length, inserted fraction, aligned fraction, and hold statistics.
- The `near_success` yaw/lateral error metrics became worse, so the combination should be treated as a strong candidate, but still worth one more confirmation run before promoting as the stable default.

## Updated Decision

- Stage 3 first-priority combination test passed.
- Current best candidate:
  - `initYaw_narrow + initY_narrow`
- There is still no reason to revive `tip_y_gate3`.
- Next step should be one of:
  - repeat this same combination once more for stability confirmation
  - or promote it to a longer `1000 iter` confirmation run

## Next Gate

If the combination run still shows:

- `inserted` not low
- `aligned` still low
- `hold_counter` still weak

then the next priority should switch from reset randomization to Stage 1 success/hold logic:

- relax `max_yaw_err_deg`
- reduce `hold_time_s`
- enable or increase `k_lat_fine`
