# Exp8.3 Pre-Dock Goal-Set Audit

- cfg_path: `/home/uniubi/projects/forklift_sim/forklift_pallet_insert_lift_project/isaaclab_patch/source/isaaclab_tasks/isaaclab_tasks/direct/forklift_pallet_insert_lift/env_cfg.py`
- scan_mode: `current_stage1_near_field_to_predock_goal_set`
- note: `Goal-set search uses world-motion-aware reverse stats from sampled RS paths, not only library segment signs.`

## Best Combo

- d_pre_m: `1.05`
- y_tol_m: `0.10`
- yaw_tol_deg: `6.0`
- overlay: `/home/uniubi/projects/forklift_sim/outputs/exp83_predock_goal_set_audit_smoke/overlay_predock_goal_set_best_20260331_181405.png`
- geometry: `/home/uniubi/projects/forklift_sim/outputs/exp83_predock_goal_set_audit_smoke/geometry_predock_goal_set_best_20260331_181405.png`
- reverse_free: `125/125`
- mean_reverse_frac: `0.000`
- mean_length: `13.791 m`
- mean_heading_change: `330.816 deg`

## Top Rows

| rank | d_pre | y_tol | yaw_tol | reverse_free | mean_rev_frac | >180deg | >10m | mean_len |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1.05 | 0.10 | 6.0 | 125/125 | 0.000 | 116 | 116 | 13.791 |

## Representative Cases

- `c01_xm3p600_ym0p150_yawm6p000`: `/home/uniubi/projects/forklift_sim/outputs/exp83_predock_goal_set_audit_smoke/c01_xm3p600_ym0p150_yawm6p000_20260331_181405.png`
- `c103_xm3p450_ym0p150_yawp0p000`: `/home/uniubi/projects/forklift_sim/outputs/exp83_predock_goal_set_audit_smoke/c103_xm3p450_ym0p150_yawp0p000_20260331_181405.png`
- `c113_xm3p450_yp0p000_yawp0p000`: `/home/uniubi/projects/forklift_sim/outputs/exp83_predock_goal_set_audit_smoke/c113_xm3p450_yp0p000_yawp0p000_20260331_181405.png`
- `c25_xm3p600_yp0p150_yawp6p000`: `/home/uniubi/projects/forklift_sim/outputs/exp83_predock_goal_set_audit_smoke/c25_xm3p600_yp0p150_yawp6p000_20260331_181405.png`
