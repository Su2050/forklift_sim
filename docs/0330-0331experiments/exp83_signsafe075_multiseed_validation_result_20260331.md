# exp8.3 `sign-safe + stage1_steer_action_scale=0.75` 多 seed 验证结果（进行中）

## 1. 目的

本轮的目标不是继续扫 reward，而是验证当前最优 baseline

- `stage1_clip_wrong_sign_steer_enable = True`
- `stage1_steer_action_scale = 0.75`

是否只是在 `seed42` 上偶然有效，还是已经对多个 seed 都起作用。

这轮坚持控制变量：

- 不改 reward
- 不改 reset
- 不改 camera 分辨率，保持 `256x256`
- 不改训练步数，保持 `50 iter`
- 不改 eval 网格，统一使用 `3x3`：
  - `x_root = -3.40`
  - `y = [-0.10, 0.00, +0.10]`
  - `yaw = [-4°, 0°, +4°]`
- 每个 checkpoint 都同时评估
  - `normal`
  - `zero-steer`

验收重点仍然是：

1. `normal` 是否至少不再输给 `zero-steer`
2. `normal` 是否出现更多 `push_free / clean_insert_ready / hold_entry`
3. 这种改善是否能跨 seed 复现

## 2. 当前基线

当前固定基线来自前序单因素实验：

- 纯调幅度到 `0.75` 后，`normal` 从 `1/9` 提到 `4/9`，但仍输给 `zero-steer = 5/9`
- 再加上 `sign-safe clip` 后，`seed42` 第一次达到 `normal = 5/9`，追平 `zero-steer = 5/9`

因此本轮多 seed 验证要回答的问题是：

> `sign-safe + 0.75` 只是把 `seed42` 修好了，还是已经把整个训练配方推到“多数 seed 至少不再被 steering 拖后腿”的状态？

## 3. 统一协议

### 3.1 训练协议

- task: `Isaac-Forklift-PalletInsertLift-Direct-v0`
- mode: `stage_1_mode = True`
- camera: `256x256`
- envs: `64`
- train length: `50 iter`

### 3.2 评估协议

- 对每个 seed 的 `model_49.pt` 跑统一 `3x3` misalignment grid
- 输出两个 summary：
  - `normal`
  - `zero-steer`

### 3.3 判据

- 若 `normal < zero-steer`：说明当前 baseline 仍未把 steering 变成正资产
- 若 `normal = zero-steer`：说明 baseline 至少已经把“有害 steering”压住，但 steering 优势仍未建立
- 若 `normal > zero-steer`：说明 baseline 开始真正扩大 good basin，可考虑继续做更长训练或更多 seed 验证

## 4. 结果

### 4.1 seed42

训练 run:

- `IsaacLab/logs/rsl_rl/forklift_pallet_insert_lift/2026-03-31_12-04-55_exp83_stage1_signsafe_clip_seed42_iter50_256cam`

训练尾窗：

- `phase/frac_inserted = 0.0000`
- `phase/frac_inserted_push_free = 0.0000`
- `phase/frac_clean_insert_ready = 0.0000`
- `phase/frac_hold_entry = 0.0000`
- `phase/frac_success = 0.0000`
- `phase/frac_dirty_insert = 0.0000`
- `diag/preinsert_wrong_sign_clipped_frac = 0.1250`
- `diag/pallet_disp_xy_mean = 0.1421`
- `traj/d_traj_mean = 0.1157`
- `traj/yaw_traj_deg_mean = 2.8154`

统一 `3x3` eval：

- `normal = 5/9`
- `zero-steer = 5/9`

关键 summary：

- `normal clean_insert_ready = 4/9`
- `zero-steer clean_insert_ready = 4/9`
- `normal dirty_insert = 2/9`
- `zero-steer dirty_insert = 1/9`

结论：

- `seed42` 上，`sign-safe + 0.75` 已经把 `normal` 从“落后”推到“追平”
- 但还不能说 steering 已经带来净优势

### 4.2 seed43

训练 run:

- `IsaacLab/logs/rsl_rl/forklift_pallet_insert_lift/2026-03-31_13-03-01_exp83_stage1_signsafe_clip_seed43_iter50_256cam`

训练尾窗：

- `phase/frac_inserted = 0.0156`
- `phase/frac_inserted_push_free = 0.0156`
- `phase/frac_clean_insert_ready = 0.0156`
- `phase/frac_hold_entry = 0.0156`
- `phase/frac_success = 0.0000`
- `phase/frac_dirty_insert = 0.0000`
- `diag/preinsert_wrong_sign_clipped_frac = 0.1719`
- `diag/pallet_disp_xy_mean = 0.2415`
- `traj/d_traj_mean = 0.2344`
- `traj/yaw_traj_deg_mean = 3.9574`

统一 `3x3` eval：

- `normal = 5/9`
- `zero-steer = 5/9`

关键 summary：

- `normal clean_insert_ready = 5/9`
- `zero-steer clean_insert_ready = 4/9`
- `normal dirty_insert = 1/9`
- `zero-steer dirty_insert = 1/9`

逐点差异：

- 在 `(yaw=0°, y=+0.10m)` 这一格，`normal` 把成功从 `dirty` 修成了 `clean`
- 但在 `(yaw=-4°, y=0.00m)` 这一格，`normal` 仍然是 `dirty timeout`，没有优于 `zero-steer`

结论：

- `seed43` 复现了 `seed42` 的主趋势：`normal` 至少不再落后
- 同时比 `seed42` 多出了一个很重要的新现象：`normal` 开始在个别格点上把 `dirty success` 修成 `clean success`
- 这说明 `sign-safe` 的作用不只是“保住成功率”，还可能在逐步改善成功的质量

### 4.3 seed44

当前状态：

- 训练中
- 仍将使用完全相同的 baseline、训练长度与统一 `3x3` eval

待补内容：

- 训练尾窗
- `3x3 normal`
- `3x3 zero-steer`
- 跨 seed 总结

## 5. 中期判断

在 `seed42` 和 `seed43` 上，当前 baseline 已经稳定呈现出同一个模式：

- `normal` 不再输给 `zero-steer`
- 但 `normal` 还没有稳定强于 `zero-steer`

这意味着当前最合理的判断不是“steering 已经学出来了”，而是：

> `sign-safe + 0.75` 已经基本压住了“错误 steering 伤害策略”这件事，但还没有把“正确 steering 带来额外收益”这件事充分做大。

换句话说，当前 baseline 更像是从“坏 steering”迈到了“无害 steering / 边界有益 steering”，还没有迈到“明显有益 steering”。

## 6. seed44 之后的决策门槛

seed44 补完后，按下面的分流继续：

1. 如果 `3/3 seed` 都满足 `normal >= zero-steer`
   - 说明当前 baseline 具有跨 seed 稳定性
   - 下一步优先考虑在该 baseline 上继续压 dirty insert / 扩大 clean hold，而不是回去重做 sign 线

2. 如果 `seed44` 回到 `normal < zero-steer`
   - 说明当前 baseline 只是“部分 seed 有效”
   - 下一步仍应沿 steering 入口继续改，而不是直接长训

3. 如果 `seed44` 首次达到 `normal > zero-steer`
   - 说明当前 baseline 已经接近可用
   - 下一步可以考虑：
     - `3 seeds x 100 iter`
     - 或保持该 baseline，专门做 clean/dirty 分流改进
