# Exp8.3 Pre-Dock Goal-Set Audit

- cfg_path: `/home/uniubi/projects/forklift_sim/forklift_pallet_insert_lift_project/isaaclab_patch/source/isaaclab_tasks/isaaclab_tasks/direct/forklift_pallet_insert_lift/env_cfg.py`
- scan_mode: `current_stage1_near_field_to_predock_goal_set`
- note: `Goal-set search uses world-motion-aware reverse stats from sampled RS paths, not only library segment signs.`

## Best Combo

- d_pre_m: `1.25`
- y_tol_m: `0.15`
- yaw_tol_deg: `6.0`
- overlay: `/home/uniubi/projects/forklift_sim/outputs/exp83_predock_goal_set_audit/overlay_predock_goal_set_best_20260331_181445.png`
- geometry: `/home/uniubi/projects/forklift_sim/outputs/exp83_predock_goal_set_audit/geometry_predock_goal_set_best_20260331_181445.png`
- reverse_free: `125/125`
- mean_reverse_frac: `0.000`
- mean_length: `11.735 m`
- mean_heading_change: `284.712 deg`

## Top Rows

| rank | d_pre | y_tol | yaw_tol | reverse_free | mean_rev_frac | >180deg | >10m | mean_len |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1.25 | 0.15 | 6.0 | 125/125 | 0.000 | 100 | 100 | 11.735 |
| 2 | 1.05 | 0.15 | 8.0 | 125/125 | 0.000 | 100 | 100 | 11.902 |
| 3 | 1.05 | 0.15 | 6.0 | 125/125 | 0.000 | 100 | 100 | 11.902 |
| 4 | 1.05 | 0.15 | 3.0 | 125/125 | 0.000 | 100 | 100 | 11.902 |
| 5 | 1.25 | 0.15 | 3.0 | 125/125 | 0.000 | 110 | 110 | 12.928 |
| 6 | 1.05 | 0.10 | 8.0 | 125/125 | 0.000 | 116 | 116 | 13.791 |
| 7 | 1.05 | 0.10 | 6.0 | 125/125 | 0.000 | 116 | 116 | 13.791 |
| 8 | 1.05 | 0.05 | 8.0 | 125/125 | 0.000 | 116 | 116 | 13.791 |
| 9 | 1.05 | 0.05 | 6.0 | 125/125 | 0.000 | 116 | 116 | 13.791 |
| 10 | 1.25 | 0.15 | 8.0 | 125/125 | 0.000 | 120 | 120 | 14.085 |

## Representative Cases

- `c01_xm3p600_ym0p150_yawm6p000`: `/home/uniubi/projects/forklift_sim/outputs/exp83_predock_goal_set_audit/c01_xm3p600_ym0p150_yawm6p000_20260331_181445.png`
- `c33_xm3p562_ym0p075_yawp0p000`: `/home/uniubi/projects/forklift_sim/outputs/exp83_predock_goal_set_audit/c33_xm3p562_ym0p075_yawp0p000_20260331_181445.png`
- `c25_xm3p600_yp0p150_yawp6p000`: `/home/uniubi/projects/forklift_sim/outputs/exp83_predock_goal_set_audit/c25_xm3p600_yp0p150_yawp6p000_20260331_181445.png`
