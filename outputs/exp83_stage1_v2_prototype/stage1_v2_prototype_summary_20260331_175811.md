# Exp8.3 Stage1-v2 Prototype

- cfg_path: `/home/uniubi/projects/forklift_sim/forklift_pallet_insert_lift_project/isaaclab_patch/source/isaaclab_tasks/isaaclab_tasks/direct/forklift_pallet_insert_lift/env_cfg.py`
- align_start_dist_m: `2.72`
- scan_mode: `direct_pre_goal_geometry_prototype`
- overlay: `/home/uniubi/projects/forklift_sim/outputs/exp83_stage1_v2_prototype/overlay_stage1_v2_prototype_20260331_175811.png`
- geometry_sketch: `/home/uniubi/projects/forklift_sim/outputs/exp83_stage1_v2_prototype/geometry_stage1_v2_prototype_20260331_175811.png`

## Geometry

- root_goal_s: `-2.350`
- current_env_nominal_root_pre_s: `-3.400`
- proposed_align_start_s: `-5.070`
- current_stage1_x_range: `[-3.600, -3.450]`
- current_stage1_y_range: `[-0.150, 0.150]`
- current_stage1_yaw_deg_range: `[-6.0, 6.0]`

## Path Audit

- num_cases: `125`
- legacy_entry_ok: `0`
- root_total_length_mean: `4.270 m`
- root_total_length_max: `4.352 m`
- root_heading_change_mean: `3.600 deg`
- root_heading_change_max: `6.000 deg`
- root_curvature_max_max: `0.427359 1/m`
- curvature_limit: `0.427350 1/m`
- num_heading_gt_180: `0`
- num_heading_gt_270: `0`
- num_length_gt_10m: `0`

## Interpretation

- This is a geometry prototype, not a drop-in env config.
- Legacy `entry_ok` stays false because the direct align-start sits upstream of the current near-field start band.
- The useful signal here is whether bounded-curvature paths stop doing full loops while staying within the physical curvature limit.

## Representative Cases

- `c63_xm3p525_yp0p000_yawp0p000`: `/home/uniubi/projects/forklift_sim/outputs/exp83_stage1_v2_prototype/c63_xm3p525_yp0p000_yawp0p000_20260331_175811.png`
- `c105_xm3p450_ym0p150_yawp6p000`: `/home/uniubi/projects/forklift_sim/outputs/exp83_stage1_v2_prototype/c105_xm3p450_ym0p150_yawp6p000_20260331_175811.png`
- `c01_xm3p600_ym0p150_yawm6p000`: `/home/uniubi/projects/forklift_sim/outputs/exp83_stage1_v2_prototype/c01_xm3p600_ym0p150_yawm6p000_20260331_175811.png`
- `c25_xm3p600_yp0p150_yawp6p000`: `/home/uniubi/projects/forklift_sim/outputs/exp83_stage1_v2_prototype/c25_xm3p600_yp0p150_yawp6p000_20260331_175811.png`
