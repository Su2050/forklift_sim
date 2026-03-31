# Exp8.3 Trajectory Preflight V2 Batch1 Result

日期：2026-03-31

分支：

- 顶层：`exp/exp8_3_traj_preflight_v2`
- IsaacLab：`exp/exp8_3_traj_preflight_v2`

## 1. 这轮具体改了什么

这不是继续调 reward，也不是继续扫 PPO。

这轮只改了 trajectory generator 本身，而且只做了第一批修复：

1. 参考轨迹从 `fork_center -> p_pre -> p_goal` 直接生成
   改成
   `vehicle/root path first -> 再映射成 fork_center path`

2. `vehicle pre-pose` 不再直接等于旧版 `fork_pre - offset`
   而是加入了两个新约束：
   - `traj_vehicle_curve_min_span_m = 0.35`
   - `traj_vehicle_final_straight_min_m = 0.10`

3. root 路径生成不再用 xy-Hermite 直接硬连
   而是改成在托盘坐标系中：
   - `s` 单调前进
   - `y(s)` 三次曲线收敛到 0
   - 末段保留一段直插

4. reward/query 的外部接口先不改
   - `d_traj`
   - `yaw_traj_err`
   - signed traj obs

仍沿用旧接口，但底层路径已经换成新生成器。

## 2. 输出产物

统一 preflight 输出目录：

- [exp83_trajectory_preflight_v2](/home/uniubi/projects/forklift_sim/outputs/exp83_trajectory_preflight_v2)

关键文件：

- [preflight_v2_summary.json](/home/uniubi/projects/forklift_sim/outputs/exp83_trajectory_preflight_v2/preflight_v2_summary.json)
- [preflight_v2_summary.md](/home/uniubi/projects/forklift_sim/outputs/exp83_trajectory_preflight_v2/preflight_v2_summary.md)
- [current overlay](/home/uniubi/projects/forklift_sim/outputs/exp83_trajectory_preflight_v2/current/overlay_all_cases.png)
- [current c19](/home/uniubi/projects/forklift_sim/outputs/exp83_trajectory_preflight_v2/current/c19_xm3p450_ym0p150_yawm6p000.png)

## 3. 结果

### 3.1 current

新的 `Preflight V2` 结果：

- `entry_ok = 27/27`
- `proxy_ok = 9/27`
- `proxy_warn = 18/27`
- `proxy_bad = 0/27`

worst-case 指标：

- `root_y_abs_max = 0.1500 m`
- `root_heading_change_deg = 10.39 deg`
- `root_curvature_max = 7.182 /m`

### 3.2 legacy

新的 `Preflight V2` 结果：

- `entry_ok = 27/27`
- `proxy_ok = 27/27`
- `proxy_warn = 0/27`
- `proxy_bad = 0/27`

worst-case 指标：

- `root_y_abs_max = 0.0800 m`
- `root_heading_change_deg = 5.41 deg`
- `root_curvature_max = 3.829 /m`

## 4. 和上一版相比，发生了什么变化

这轮最关键的变化不是 `entry_ok`，而是：

- 原来 current 基本是 `3 ok / 24 bad`
- 现在 current 变成了 `9 ok / 18 warn / 0 bad`

也就是说：

- 这第一刀还没有把 current 完全修到“全绿”
- 但已经把“明显离谱”的 `bad` 全部消掉了

这说明：

- `vehicle/root path first`
- 再加上
  - 最小整车曲线长度
  - 最末端直插段

确实把 trajectory 从“明显不靠谱”往“可以继续调”推进了一大步。

## 5. 单 case 代表图解读

最值得看的是：

- [current c19](/home/uniubi/projects/forklift_sim/outputs/exp83_trajectory_preflight_v2/current/c19_xm3p450_ym0p150_yawm6p000.png)

这一例里：

- `delta_s = -0.357`
- `root|y|_max = 0.150`
- `root_dpsi = 0.24 deg`
- `root_kappa_max = 7.182 /m`

更直观的一点是：

- 同一个 `c19`
- `fork_center` 轨迹最大外摆大约从 `1.23m`
  降到了
- 大约 `0.64m`

说明新的 vehicle-aware 生成器，确实压掉了旧版那种非常夸张的外摆。

## 6. 当前判断

这轮不能说“trajectory 已经彻底修好”，但可以明确说：

1. 改 trajectory generator 这条路是对的
2. `root-path-first + minimum vehicle curve span` 是有效的
3. 当前已经不再是“27/27 entry 全过，但 proxy 一片 bad”
4. 下一步不该回到 PPO，而应该继续做 Trajectory Preflight V2 的第二批细化

## 7. 下一步

下一步建议仍然只改 trajectory，不碰 PPO：

1. 把 `warn` 继续压到更少
   - 重点看 current 的 18 个 warn

2. 优先排查两件事
   - `traj_vehicle_curve_min_span_m` 是否还需要进一步拉长
   - current reset (`y ±0.15 / yaw ±6`) 下，`p_pre` 是否还应该再往前收一点

3. 等 current 至少大部分变成 `proxy_ok`
   再恢复 `approach_only` PPO 训练

