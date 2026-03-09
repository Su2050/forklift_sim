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

### `initYaw_narrow + initY_narrow` long confirmation (`resume 450 -> 1000`)

- Log: `/home/uniubi/projects/forklift_sim/logs/20260309_152411_train_s1.0zc.log`
- Window used for comparison: tail 50 iterations
- `success_rate_ema`: `0.4820`
- `success_rate_total`: `0.4841`
- `Mean episode length`: `567.8368`
- `phase/frac_inserted`: `0.5173`
- `phase/frac_aligned`: `0.0397`
- `phase/hold_counter_max`: `8.0800`
- `phase/hold_counter_mean`: `0.2220`
- `err/yaw_deg_near_success`: `10.2928`
- `err/lateral_near_success`: `0.2028`

Relative to baseline tail 50:

- `success_rate_ema`: `+74.2%`
- `success_rate_total`: `+69.4%`
- `Mean episode length`: `-28.3%`
- `phase/frac_inserted`: `+73.5%`
- `phase/frac_aligned`: `+133.5%`
- `phase/hold_counter_max`: `+52.5%`
- `phase/hold_counter_mean`: `+462.0%`
- `err/yaw_deg_near_success`: worse
- `err/lateral_near_success`: worse

Relative to `initYaw_narrow_formal` tail 50:

- `success_rate_ema`: `-1.3%`
- `success_rate_total`: `-2.0%`
- `Mean episode length`: `-1.6%`
- `phase/frac_inserted`: `+70.7%`
- `phase/frac_aligned`: `+25.6%`
- `phase/hold_counter_max`: `+20.6%`
- `phase/hold_counter_mean`: `+131.5%`
- `err/yaw_deg_near_success`: worse
- `err/lateral_near_success`: worse

Relative to first combo confirmation tail 50:

- `success_rate_ema`: `-18.5%`
- `success_rate_total`: `-19.0%`
- `Mean episode length`: `+25.7%`
- `phase/frac_inserted`: `+37.6%`
- `phase/frac_aligned`: `-24.7%`
- `phase/hold_counter_max`: `-5.2%`
- `phase/hold_counter_mean`: `+35.9%`
- `err/yaw_deg_near_success`: slightly better
- `err/lateral_near_success`: slightly better

Conclusion:

- The combination is still clearly better than baseline and remains strong on inserted fraction and hold statistics.
- However, the long confirmation run did not reproduce the earlier `0.5916` success-rate peak.
- On the primary decision metric, it no longer shows a stable advantage over `initYaw_narrow_formal`.
- This means the reset-randomization line has likely reached a plateau on the current branch.

## Updated Decision

- Stage 3 is now considered closed for this branch.
- `initYaw_narrow + initY_narrow` remains a useful recipe, but it is not yet validated as a stable default over `initYaw_narrow_formal`.
- There is still no reason to revive `tip_y_gate3`.
- The next priority should switch from reset randomization to Stage 1 success/hold logic.

## Stage 1 Logic Line

Base recipe for the next line:

- keep `initYaw_narrow + initY_narrow`
- keep camera / asymmetric critic / Stage 1 setup unchanged
- only change one Stage 1 success-hold variable at a time

Recommended execution order:

1. Relax `max_yaw_err_deg`: `5.0 -> 8.0`
2. Reduce `hold_time_s`: `0.33 -> 0.20`
3. Enable `k_lat_fine`: `0.0 -> 0.8`

Why this order:

- Current long-run logs show `inserted` is already high and `hold_counter` is not zero, but hold is still frequently broken by yaw/lateral quality near success.
- `diag_hold/yaw_margin` stays clearly negative in the long run, so yaw tolerance is the most direct first lever.
- `hold_time_s` should be tested only after checking whether the success gate itself is still too strict.
- `k_lat_fine` is a shaping lever and should come after success-gate tests, otherwise it becomes harder to tell whether gains come from reward shaping or from the success definition.

Smoke-screening rule for this line:

- Run each Stage 1 logic change as a single-factor `smoke_train` first.
- Promote only if it improves at least one primary metric (`success_rate_ema` or `success_rate_total`) without clearly damaging `phase/frac_aligned` or `phase/hold_counter_mean`.

### `max_yaw_err_deg: 5 -> 8` (`yawRelax_smoke`)

- Log: `/home/uniubi/projects/forklift_sim/logs/20260309_163116_smoke_train_s1.0zc.log`
- Window used for comparison: full 30-iteration smoke window
- `success_rate_ema`: `0.4559`
- `Mean episode length`: `224.4770`
- `phase/frac_inserted`: `0.1896`
- `phase/frac_aligned`: `0.1526`
- `phase/hold_counter_mean`: `0.1570`
- `err/yaw_deg_near_success`: `4.3767`
- `err/lateral_near_success`: `0.1144`

Relative to `initYaw_narrow` smoke:

- `success_rate_ema`: strongly better
- `Mean episode length`: much shorter
- `phase/frac_aligned`: better
- `phase/hold_counter_mean`: much better
- `err/yaw_deg_near_success`: slightly worse than the very best yaw-only smoke, but still strong
- `err/lateral_near_success`: better
- `phase/frac_inserted`: lower

Relative to baseline tail 30:

- `success_rate_ema`: clearly better
- `Mean episode length`: much shorter
- `phase/frac_aligned`: much better
- `phase/hold_counter_mean`: much better
- `err/yaw_deg_near_success`: clearly better
- `err/lateral_near_success`: clearly better

Conclusion:

- Relaxing the Stage 1 yaw hold gate is a strong positive direction.
- This is the first clear confirmation that the current bottleneck is in Stage 1 success/hold logic rather than reset randomization.
- The lower `inserted` fraction means this change is not a free improvement on every metric, but the much better success, alignment, and hold statistics make it worth immediate formal confirmation.

Next action:

- Keep `max_yaw_err_deg=8.0`.
- Run a longer formal confirmation before deciding whether to stack `hold_time_s`.

### `max_yaw_err_deg: 5 -> 8` (`yawRelax_formal`)

- Log: `/home/uniubi/projects/forklift_sim/logs/20260309_163730_train_s1.0zc.log`
- Window used for comparison: tail 50 iterations
- `success_rate_ema`: `0.6790`
- `success_rate_total`: `0.6845`
- `Mean episode length`: `370.8524`
- `phase/frac_inserted`: `0.2163`
- `phase/frac_aligned`: `0.0855`
- `phase/hold_counter_mean`: `0.1265`
- `phase/hold_counter_max`: `8.5600`
- `err/yaw_deg_near_success`: `4.5571`
- `err/lateral_near_success`: `0.1219`

Relative to baseline tail 50:

- `success_rate_ema`: `+145.4%`
- `success_rate_total`: `+139.5%`
- `Mean episode length`: `-53.2%`
- `phase/frac_inserted`: lower
- `phase/frac_aligned`: `+402.9%`
- `phase/hold_counter_mean`: `+220.3%`
- `phase/hold_counter_max`: `+61.5%`
- `err/yaw_deg_near_success`: `-40.4%`
- `err/lateral_near_success`: `-31.4%`

Relative to `initYaw_narrow_formal` tail 50:

- `success_rate_ema`: `+39.1%`
- `success_rate_total`: `+38.5%`
- `Mean episode length`: `-35.7%`
- `phase/frac_inserted`: lower
- `phase/frac_aligned`: `+170.6%`
- `phase/hold_counter_mean`: `+31.9%`
- `phase/hold_counter_max`: `+27.8%`
- `err/yaw_deg_near_success`: `-4.9%`
- `err/lateral_near_success`: `-31.6%`

Relative to first combo confirmation tail 50:

- `success_rate_ema`: `+14.8%`
- `success_rate_total`: `+14.6%`
- `Mean episode length`: `-17.9%`
- `phase/frac_inserted`: lower
- `phase/frac_aligned`: `+62.2%`
- `phase/hold_counter_mean`: `-22.6%`
- `phase/hold_counter_max`: roughly flat
- `err/yaw_deg_near_success`: much better
- `err/lateral_near_success`: much better

Conclusion:

- Relaxing the Stage 1 yaw hold gate is formally confirmed and is now the strongest validated logic-side improvement on the branch.
- This result outperforms the previous best reset-only and reset-combination confirmations on the main decision metrics.
- The lower `inserted` fraction means the gain is not coming from deeper insertion, but from a much healthier success/hold transition once the agent reaches the near-success region.

Next action:

- Keep `max_yaw_err_deg=8.0` as the current best logic-side setting.
- Continue with the next single-factor logic smoke: reduce `hold_time_s`.

### `hold_time_s: 0.33 -> 0.20` (`holdTimeRelax_smoke`)

- Log: `/home/uniubi/projects/forklift_sim/logs/20260309_171654_smoke_train_s1.0zc.log`
- Window used for comparison: full 30-iteration smoke window
- `success_rate_ema`: `0.4590`
- `Mean episode length`: `220.9360`
- `phase/frac_inserted`: `0.1941`
- `phase/frac_aligned`: `0.1529`
- `phase/hold_counter_mean`: `0.0671`
- `err/yaw_deg_near_success`: `4.1787`
- `err/lateral_near_success`: `0.1107`

Relative to `yawRelax_smoke`:

- `success_rate_ema`: nearly flat, slightly better
- `Mean episode length`: slightly shorter
- `phase/frac_inserted`: slightly better
- `phase/frac_aligned`: essentially flat
- `err/yaw_deg_near_success`: slightly better
- `err/lateral_near_success`: slightly better
- `phase/hold_counter_mean`: lower

Interpretation:

- Reducing `hold_time_s` from `0.33` to `0.20` does not show a large extra gain on top of `max_yaw_err_deg=8.0`.
- The main metrics moved only slightly, so this is not yet strong evidence of a real task-closure improvement.
- Because shorter hold time directly relaxes the success condition, part of the observed gain could be a surface-level success-gate effect rather than a genuinely more stable policy.

Conclusion:

- `holdTimeRelax_smoke` is not clearly negative.
- But it is also not strong enough to promote as a new default from smoke alone.
- If this line is pursued, it should be labeled as a validation run for “whether the gain is only from a looser success gate”.

Recommended next action:

- Conservative path: keep `max_yaw_err_deg=8.0`, revert `hold_time_s` to `0.33`, and treat yaw relaxation as the current best logic-side change.
- Aggressive path: run one formal confirmation with `hold_time_s=0.20`, explicitly treating it as a validation of whether the gain is superficial or truly stable.

### `k_lat_fine: 0.0 -> 0.8` (`kLatFine_smoke`)

- Log: `/home/uniubi/projects/forklift_sim/logs/20260309_173129_smoke_train_s1.0zc.log`
- Window used for comparison: full 30-iteration smoke window
- `success_rate_ema`: `0.4497`
- `Mean episode length`: `238.9730`
- `phase/frac_inserted`: `0.2022`
- `phase/frac_aligned`: `0.1674`
- `phase/hold_counter_mean`: `0.1770`
- `err/yaw_deg_near_success`: `4.2221`
- `err/lateral_near_success`: `0.1092`

Relative to `yawRelax_smoke`:

- `success_rate_ema`: slightly lower
- `Mean episode length`: slightly longer
- `phase/frac_inserted`: slightly better
- `phase/frac_aligned`: better
- `phase/hold_counter_mean`: better
- `err/yaw_deg_near_success`: slightly better
- `err/lateral_near_success`: slightly better

Relative to `holdTimeRelax_smoke`:

- `success_rate_ema`: slightly lower
- `Mean episode length`: slightly longer
- `phase/frac_aligned`: better
- `phase/hold_counter_mean`: much better
- `err/yaw_deg_near_success`: roughly flat to slightly worse
- `err/lateral_near_success`: slightly better

Interpretation:

- Enabling `k_lat_fine` does not immediately improve the headline success-rate metric over `yawRelax_smoke`.
- However, it consistently improves the process metrics most related to near-success quality: alignment, hold-counter behavior, and lateral/yaw quality.
- This makes it a good candidate for a combination confirmation with the already validated `max_yaw_err_deg=8.0` setting.

Conclusion:

- `kLatFine_smoke` is a meaningful positive candidate, but its benefit appears more in process quality than in raw smoke success rate.
- The next useful question is whether those process gains convert into stable success-rate gains in a longer formal confirmation run.

Next action:

- Keep `max_yaw_err_deg=8.0`.
- Keep `hold_time_s=0.33`.
- Run a longer formal confirmation for `max_yaw_err_deg=8.0 + k_lat_fine=0.8`.
