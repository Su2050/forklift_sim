# Exp8.3 Arc Pre-Goal Push Scan

- cfg_path: `/home/uniubi/projects/forklift_sim/forklift_pallet_insert_lift_project/isaaclab_patch/source/isaaclab_tasks/isaaclab_tasks/direct/forklift_pallet_insert_lift/env_cfg.py`
- traj_model: `dubins_to_pre_straight`
- scan_mode: `direct_pre_goal`
- note: `This sweep bypasses the current env pre-goal clamp and directly places root_pre upstream of root_goal.`

## Sweep Rows

### pre_dist = 1.05 m

- overlay: `/home/uniubi/projects/forklift_sim/outputs/exp83_arc_pre_goal_push_scan/overlay_pre_1p05.png`
- num_cases: `125`
- entry_ok: `125`
- root_total_length_mean: `15.772 m`
- root_total_length_max: `16.330 m`
- root_heading_change_mean: `356.880 deg`
- root_heading_change_max: `366.000 deg`
- num_heading_gt_180: `125`
- num_heading_gt_270: `125`
- num_length_gt_10m: `125`
- worst_length_case: `c86_xm3p488_yp0p000_yawm6p000`

### pre_dist = 2.00 m

- overlay: `/home/uniubi/projects/forklift_sim/outputs/exp83_arc_pre_goal_push_scan/overlay_pre_2p00.png`
- num_cases: `125`
- entry_ok: `0`
- root_total_length_mean: `11.770 m`
- root_total_length_max: `17.615 m`
- root_heading_change_mean: `218.160 deg`
- root_heading_change_max: `360.000 deg`
- num_heading_gt_180: `76`
- num_heading_gt_270: `76`
- num_length_gt_10m: `76`
- worst_length_case: `c103_xm3p450_ym0p150_yawp0p000`

### pre_dist = 3.00 m

- overlay: `/home/uniubi/projects/forklift_sim/outputs/exp83_arc_pre_goal_push_scan/overlay_pre_3p00.png`
- num_cases: `125`
- entry_ok: `0`
- root_total_length_mean: `4.829 m`
- root_total_length_max: `4.909 m`
- root_heading_change_mean: `3.600 deg`
- root_heading_change_max: `6.000 deg`
- num_heading_gt_180: `0`
- num_heading_gt_270: `0`
- num_length_gt_10m: `0`
- worst_length_case: `c105_xm3p450_ym0p150_yawp6p000`

### pre_dist = 4.00 m

- overlay: `/home/uniubi/projects/forklift_sim/outputs/exp83_arc_pre_goal_push_scan/overlay_pre_4p00.png`
- num_cases: `125`
- entry_ok: `0`
- root_total_length_mean: `6.827 m`
- root_total_length_max: `6.906 m`
- root_heading_change_mean: `3.600 deg`
- root_heading_change_max: `6.000 deg`
- num_heading_gt_180: `0`
- num_heading_gt_270: `0`
- num_length_gt_10m: `0`
- worst_length_case: `c105_xm3p450_ym0p150_yawp6p000`

### pre_dist = 5.00 m

- overlay: `/home/uniubi/projects/forklift_sim/outputs/exp83_arc_pre_goal_push_scan/overlay_pre_5p00.png`
- num_cases: `125`
- entry_ok: `0`
- root_total_length_mean: `8.827 m`
- root_total_length_max: `8.904 m`
- root_heading_change_mean: `3.600 deg`
- root_heading_change_max: `6.000 deg`
- num_heading_gt_180: `0`
- num_heading_gt_270: `0`
- num_length_gt_10m: `0`
- worst_length_case: `c105_xm3p450_ym0p150_yawp6p000`

### pre_dist = 6.00 m

- overlay: `/home/uniubi/projects/forklift_sim/outputs/exp83_arc_pre_goal_push_scan/overlay_pre_6p00.png`
- num_cases: `125`
- entry_ok: `0`
- root_total_length_mean: `10.826 m`
- root_total_length_max: `10.903 m`
- root_heading_change_mean: `3.600 deg`
- root_heading_change_max: `6.000 deg`
- num_heading_gt_180: `0`
- num_heading_gt_270: `0`
- num_length_gt_10m: `125`
- worst_length_case: `c105_xm3p450_ym0p150_yawp6p000`

### pre_dist = 8.00 m

- overlay: `/home/uniubi/projects/forklift_sim/outputs/exp83_arc_pre_goal_push_scan/overlay_pre_8p00.png`
- num_cases: `125`
- entry_ok: `0`
- root_total_length_mean: `14.826 m`
- root_total_length_max: `14.902 m`
- root_heading_change_mean: `3.600 deg`
- root_heading_change_max: `6.000 deg`
- num_heading_gt_180: `0`
- num_heading_gt_270: `0`
- num_length_gt_10m: `125`
- worst_length_case: `c105_xm3p450_ym0p150_yawp6p000`

### pre_dist = 10.00 m

- overlay: `/home/uniubi/projects/forklift_sim/outputs/exp83_arc_pre_goal_push_scan/overlay_pre_10p00.png`
- num_cases: `125`
- entry_ok: `0`
- root_total_length_mean: `18.826 m`
- root_total_length_max: `18.902 m`
- root_heading_change_mean: `3.600 deg`
- root_heading_change_max: `6.000 deg`
- num_heading_gt_180: `0`
- num_heading_gt_270: `0`
- num_length_gt_10m: `125`
- worst_length_case: `c105_xm3p450_ym0p150_yawp6p000`

## First Good-ish Rows

- first row with `num_heading_gt_270 == 0`: `3.0`
- first row with `num_heading_gt_180 == 0`: `3.0`
- first row with `num_length_gt_10m == 0`: `3.0`
