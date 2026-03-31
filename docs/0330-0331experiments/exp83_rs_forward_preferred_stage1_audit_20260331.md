# Exp8.3 Forward-Preferred RS Stage1 Audit 2026-03-31

## 目的

在项目源代码中引入 `traj_model = "rs_forward_preferred"` 后，先不接训练，先按当前 Stage1 初始化范围做纯几何批量审计，确认：

- forward-preferred RS 是否真的能在 near-field 中选出比 `rs_exact` 更自然的参考轨迹
- 还是说当前筛选条件过严，最终全部回退到 `root_path_first`

同时，本轮把可视化脚本改成了“每次输出文件名自带统一时间戳”，避免重复运行覆盖旧图。

## 代码改动

- 可视化脚本：
  [visualize_reference_trajectory_cases.py](/home/uniubi/projects/forklift_sim/forklift_pallet_insert_lift_project/scripts/visualize_reference_trajectory_cases.py)

本轮新增行为：

- 每次运行自动生成 `run_timestamp = YYYYMMDD_HHMMSS`
- 每个 case 图文件名变为 `cXX_..._<timestamp>.png`
- overlay 文件名变为 `overlay_all_cases_<timestamp>.png`
- manifest 文件名变为 `reference_trajectory_stage1_manifest_<timestamp>.json`

## 运行命令

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
  [overlay_all_cases_20260331_162531.png](/home/uniubi/projects/forklift_sim/forklift_pallet_insert_lift_project/outputs/reference_trajectory_stage1_viz_rs_forward_preferred/overlay_all_cases_20260331_162531.png)
- manifest:
  [reference_trajectory_stage1_manifest_20260331_162531.json](/home/uniubi/projects/forklift_sim/forklift_pallet_insert_lift_project/outputs/reference_trajectory_stage1_viz_rs_forward_preferred/reference_trajectory_stage1_manifest_20260331_162531.json)

## 结果

- `num_cases = 125`
- `num_entry_ok = 125`
- `path_mode_counts = {"rs_forward_preferred_fallback_root_path_first": 125}`
- `root_heading_change_deg_max = 10.39`
- `root_curvature_max_max = 7.18`

和上一轮 `rs_exact` 对比：

- `rs_exact`:
  - `root_heading_change_deg_max = 178.42`
  - `root_curvature_max_max = 229.75`
- `rs_forward_preferred` 本轮：
  - 轨迹几何指标明显温和
  - 但原因不是“选出了更好的 RS”
  - 而是 `125/125` 全部 fallback 回了 `root_path_first`

所以这轮的真实结论不是“forward-preferred RS 已经成功”，而是：

**当前 forward-preferred 筛选门槛太严，导致 near-field 下没有任何一个 RS 候选通过筛选。**

## 补充诊断

我额外对 125 个 case 做了候选筛选统计，结果如下：

- `accepted_cases = 0`
- `total_length` 超限拒绝：`860`
- `reverse_frac` 超限拒绝：`140`
- `direction_switches` 超限拒绝：`0`
- `final_forward` 不满足：`0`

这说明当前最主要的卡点是：

1. `traj_rs_forward_preferred_max_extra_length_m = 1.50` 太紧
2. 其次是 `traj_rs_forward_preferred_max_reverse_frac = 0.35`

反而不是：

- 方向切换次数太多
- 末段必须前进这个条件太苛刻

## 当前判断

- `rs_exact` 不能直接作为 near-field 默认训练参考轨迹
- `rs_forward_preferred` 这个方向是合理的，但当前参数还没有真正跑出 RS 轨迹
- 现在还不能把训练默认切到 RS family

## 下一步

建议继续做单因素调参，不要同时改多项：

1. 只放宽 `traj_rs_forward_preferred_max_extra_length_m`
   - 例如 `1.50 -> 3.00`
2. 保持其它门槛不变，再跑同样的 `5x5x5` 审计
3. 目标不是先追求“全部都用 RS”，而是先让一部分 near-field case 能稳定选到非 fallback 的 forward-preferred RS
4. 只有在 `path_mode_counts` 里开始出现实质性的 `rs_forward_preferred` case 后，才值得继续讨论接训练
