# Exp8.3 RS 参考轨迹独立化与首轮验证

日期：2026-03-31

## 这次做了什么

把参考轨迹生成从环境里抽成了可独立调用的脚本链，分成 4 个入口：

- 轨迹库：[exp83_reference_trajectory_lib.py](/home/uniubi/projects/forklift_sim/scripts/exp83_reference_trajectory_lib.py)
- 单独生成脚本：[generate_exp83_reference_trajectory.py](/home/uniubi/projects/forklift_sim/scripts/generate_exp83_reference_trajectory.py)
- 对比可视化脚本：[visualize_exp83_reference_path_models.py](/home/uniubi/projects/forklift_sim/scripts/visualize_exp83_reference_path_models.py)
- 最小测试：[test_exp83_reference_trajectory_lib.py](/home/uniubi/projects/forklift_sim/scripts/test_exp83_reference_trajectory_lib.py)

现在支持两类生成器：

- `root_path_first`
  先规划 vehicle/root 路径，再映射出 `fork_center` 路径。
- `rs_exact`
  直接调用仓库里的 [rs.py](/home/uniubi/projects/forklift_sim/rs/rs.py) 做精确 Reeds-Shepp 采样。

## 目标状态如何定义

这次没有把目标设成“托盘中心”，而是按下面的方式定义成“托盘前方、姿态对准”的 vehicle goal：

1. 先算 pallet insert axis。
2. 在 pallet front face 前方留出 `fork_front_stop_buffer_m`。
3. 再减去 `vehicle_to_fork_center`，得到 vehicle/root 应该停住的目标位姿。

也就是说，reference trajectory 的终点现在是：

- 不是托盘中心
- 不是 fork tip 进到托盘里
- 而是“车体应停在托盘前方，且 yaw 对准托盘”的 pose

## 首轮验证怎么做的

样例：

- start = `(-3.45, -0.15, -6deg)`
- pallet = `(0, 0, 0deg)`
- min_turn_radius = `0.55m`

运行命令：

```bash
python scripts/test_exp83_reference_trajectory_lib.py
python scripts/visualize_exp83_reference_path_models.py \
  --tag c19_like_rs_exact \
  --start-x -3.45 \
  --start-y -0.15 \
  --start-yaw-deg -6.0 \
  --min-turn-radius-m 0.55
python scripts/generate_exp83_reference_trajectory.py \
  --model rs_exact \
  --goal-mode front_of_pallet \
  --start-x -3.45 \
  --start-y -0.15 \
  --start-yaw-deg -6.0 \
  --min-turn-radius-m 0.55 \
  --tag c19_like_single
```

## 结果

测试通过：

- [test_exp83_reference_trajectory_lib.py](/home/uniubi/projects/forklift_sim/scripts/test_exp83_reference_trajectory_lib.py)

对比图：

- [exp83_rs_front_goal_c19_like_rs_exact.png](/home/uniubi/projects/forklift_sim/outputs/exp83_reference_path_models/exp83_rs_front_goal_c19_like_rs_exact.png)
- [exp83_rs_front_goal_c19_like_rs_exact.json](/home/uniubi/projects/forklift_sim/outputs/exp83_reference_path_models/exp83_rs_front_goal_c19_like_rs_exact.json)

单独生成器输出：

- [exp83_rs_exact_c19_like_single.png](/home/uniubi/projects/forklift_sim/outputs/exp83_reference_path_generator/exp83_rs_exact_c19_like_single.png)
- [exp83_rs_exact_c19_like_single_rows.csv](/home/uniubi/projects/forklift_sim/outputs/exp83_reference_path_generator/exp83_rs_exact_c19_like_single_rows.csv)
- [exp83_rs_exact_c19_like_single_summary.json](/home/uniubi/projects/forklift_sim/outputs/exp83_reference_path_generator/exp83_rs_exact_c19_like_single_summary.json)

exact RS 这次给出的段序列是：

- `L +0.53m`
- `R -0.09m`
- `L -0.56m`

总长约 `1.18m`，终点误差约 `0`。

## 这次最重要的发现

右图之前“完全不能用”，主要原因不是 RS 思想本身不行，而是我之前拿去画图的是一版 state-lattice 近似，不是仓库里已有的精确 RS 实现。

这次换成 [rs.py](/home/uniubi/projects/forklift_sim/rs/rs.py) 以后，终点误差已经收到了 `0`，所以现在这个脚本链可以作为后续 reference trajectory 研究的干净基线。

但这次还有一个更关键的发现：

- 当前样例里，front-goal 对应的 vehicle goal 是 `(-3.40, 0.0, 0deg)`
- 而 start 是 `(-3.45, -0.15, -6deg)`

也就是说，当前这类 stage1 near-field case 里，起点离“托盘前方对准位姿”已经非常近了。

这会直接带来一个现象：

- exact RS 的最短路不一定是“单调前进”
- 它可能包含很短的倒车段，来更快修正姿态

所以如果后面看到 exact RS 有小段 reverse，不能立刻判成“轨迹错了”。
更需要先分清：

- 是 RS 模型本身不合理
- 还是当前 goal 设得离 start 太近，导致最短路天然包含 reverse

## 对 root_path_first 的补丁

这次顺手也补了一个必要修复：

- 当 front-goal 离起点过近时，旧版 `root_path_first` 的 `pre_s` 会退化到几乎零 span
- 现在加了 `direct_cubic_fallback`

这不是最终方案，只是为了让 standalone compare 在 near-goal case 上不至于直接塌掉。

## 下一步建议

第一步，不要马上把 exact RS 直接接回 reward。

先做两类独立审计：

1. 用 `rs_exact` 扫一组当前 stage1 reset case，统计：
   - 是否频繁出现 reverse
   - reverse 长度占比
   - goal 与 start 的几何距离分布
2. 把 `goal_stop_buffer_m` 当成单因素变量扫几档，观察：
   - goal 稍微前移后，exact RS 是否更自然地变成前进主导

如果这两步做完，发现 exact RS 仍然稳定更合理，再考虑把它接到 env 的 reference trajectory / shaping 口径里。
