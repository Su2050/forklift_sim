# Exp8.3 Forward-Preferred RS Extra-Length Probe 2026-03-31

## 目的

上一轮 `forward-preferred RS` 审计显示：

- `125/125` 个 case 都 fallback 回了 `root_path_first`
- 最主要怀疑项是 `traj_rs_forward_preferred_max_extra_length_m = 1.50` 太紧

因此本轮按控制变量原则，只验证这一项：

- 代码里临时把 `max_extra_length_m: 1.50 -> 3.00`
- 其它筛选条件全部不动
- 重新跑同一套 `5 x 5 x 5` Stage1 审计

之后再做一轮纯离线阈值扫描，判断“要放到多大才会真正放出 RS”

## 审计命令

```bash
python forklift_pallet_insert_lift_project/scripts/visualize_reference_trajectory_cases.py \
  --traj-model rs_forward_preferred \
  --grid-count-x 5 \
  --grid-count-y 5 \
  --grid-count-yaw 5 \
  --output-dir forklift_pallet_insert_lift_project/outputs/reference_trajectory_stage1_viz_rs_forward_preferred
```

## 新产物

- overlay:
  [overlay_all_cases_20260331_162943.png](/home/uniubi/projects/forklift_sim/forklift_pallet_insert_lift_project/outputs/reference_trajectory_stage1_viz_rs_forward_preferred/overlay_all_cases_20260331_162943.png)
- manifest:
  [reference_trajectory_stage1_manifest_20260331_162943.json](/home/uniubi/projects/forklift_sim/forklift_pallet_insert_lift_project/outputs/reference_trajectory_stage1_viz_rs_forward_preferred/reference_trajectory_stage1_manifest_20260331_162943.json)

上一轮用于对比的时间戳产物：

- [reference_trajectory_stage1_manifest_20260331_162531.json](/home/uniubi/projects/forklift_sim/forklift_pallet_insert_lift_project/outputs/reference_trajectory_stage1_viz_rs_forward_preferred/reference_trajectory_stage1_manifest_20260331_162531.json)

## 结果

把 `max_extra_length_m` 放到 `3.00` 之后，结果完全没有变化：

- `num_cases = 125`
- `num_entry_ok = 125`
- `path_mode_counts = {"rs_forward_preferred_fallback_root_path_first": 125}`
- `root_heading_change_deg_max = 10.39`
- `root_curvature_max_max = 7.18`

也就是说：

**这不是“再稍微放宽一点就能用”的问题。**

## 离线阈值扫描

为了避免继续盲调，我又做了一个不改代码的离线统计扫描，固定其它门槛不变，只看 `max_extra_length_m` 需要放到多大，当前 125 个 case 才会开始接受真实 RS：

- `1.5 -> 0 / 125` accepted
- `3.0 -> 0 / 125`
- `5.0 -> 0 / 125`
- `8.0 -> 0 / 125`
- `12.0 -> 0 / 125`
- `20.0 -> 118 / 125`

更关键的是最优候选的 extra-length 分布：

- `best_extra_min = 13.26 m`
- `best_extra_p50 = 13.79 m`
- `best_extra_p90 = 14.05 m`
- `best_extra_max = 14.15 m`

reverse 比例反而不是主要问题：

- `best_rev_min = 0.0`
- `best_rev_p50 = 0.0`
- `best_rev_p90 = 0.0`
- `best_rev_max = 0.464`

## 结论

当前 near-field 设定下，`forward-preferred RS` 想在保持：

- `max_reverse_frac <= 0.35`
- `direction_switches <= 1`
- `require_final_forward = True`

这些条件不变的前提下选出真实 RS，所需的额外路径长度不是 `2m` 或 `3m` 量级，而是 **约 `13~14m`**。

这已经说明两件事：

1. 当前 `front-goal` 几何对 exact RS family 很不友好
2. 如果靠把 `max_extra_length_m` 一路放大到 `13m+` 才能选出 RS，那这个“forward-preferred”就失去原本意义了

所以本轮的判断是：

**不建议把 `max_extra_length_m` 真的改到很大。**

## 后续建议

下一步更值得做的，不是继续放宽 `max_extra_length_m`，而是改更上游的几何定义：

1. 调整 vehicle goal，让目标停位离托盘前方更远一些
2. 或者给 RS 单独定义一个更远的 pre-alignment goal，再映射到 fork-center
3. 继续保持控制变量，一次只改 goal geometry，不同时改 RS 筛选条件

## 备注

本轮验证后，代码中的 `traj_rs_forward_preferred_max_extra_length_m` 已恢复到 `1.50`，避免把一个已经验证“无效”的试探值留在项目源默认配置里。
