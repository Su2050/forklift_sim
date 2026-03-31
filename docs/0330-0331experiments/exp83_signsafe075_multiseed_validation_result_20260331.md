# exp8.3 `sign-safe + stage1_steer_action_scale=0.75` 多 seed 验证结果

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

训练 run:

- `IsaacLab/logs/rsl_rl/forklift_pallet_insert_lift/2026-03-31_13-45-39_exp83_stage1_signsafe_clip_seed44_iter50_256cam`

训练尾窗：

- `phase/frac_inserted = 0.5781`
- `phase/frac_inserted_push_free = 0.0000`
- `phase/frac_clean_insert_ready = 0.0000`
- `phase/frac_hold_entry = 0.0156`
- `phase/frac_success = 0.0000`
- `phase/frac_dirty_insert = 0.5781`
- `diag/preinsert_wrong_sign_clipped_frac = 0.0781`
- `diag/pallet_disp_xy_mean = 0.2771`
- `traj/d_traj_mean = 0.4102`
- `traj/yaw_traj_deg_mean = 8.0373`

统一 `3x3` eval：

- `normal = 2/9`
- `zero-steer = 4/9`

关键 summary：

- `normal clean_insert_ready = 0/9`
- `zero-steer clean_insert_ready = 2/9`
- `normal dirty_insert = 3/9`
- `zero-steer dirty_insert = 3/9`
- `normal mean_abs_steer_applied = 0.4571`
- `zero-steer mean_abs_steer_applied = 0.0000`

逐点差异：

- 在 `(yaw=-4°, y=0.00m)` 这一格，`normal` 打成了 `dirty success`，而 `zero-steer` 直接 timeout
- 但 `zero-steer` 在 3 个关键格点明显更强：
  - `(yaw=0°, y=0.00m)`：`zero-steer` 是 `clean success`，`normal` 直接失败
  - `(yaw=0°, y=+0.10m)`：`zero-steer` 是 `dirty success`，`normal` 直接失败
  - `(yaw=+4°, y=-0.10m)`：`zero-steer` 是 `clean success`，`normal` 直接失败

结论：

- `seed44` 没有复现 `seed42/43` 的“至少追平”
- 在这条 seed 上，当前 baseline 仍然存在明显的 steering 伤害
- 更直观地说，`sign-safe + 0.75` 把一部分坏 steering 压住了，但还没把 steering 约束到足够温和；一旦这条 seed 学出大幅单边转向，`normal` 仍会明显弱于 `zero-steer`

## 5. 跨 seed 总结

三条 seed 的统一结果如下：

| seed | train tail `frac_inserted` | train tail `frac_dirty_insert` | `3x3 normal` | `3x3 zero` | 结论 |
| --- | ---: | ---: | ---: | ---: | --- |
| `42` | `0.0000` | `0.0000` | `5/9` | `5/9` | 追平 |
| `43` | `0.0156` | `0.0000` | `5/9` | `5/9` | 追平，且 clean 质量更好 |
| `44` | `0.5781` | `0.5781` | `2/9` | `4/9` | 明显落后 |

这轮 multi-seed validation 给出的主结论非常明确：

- `sign-safe + 0.75` 不是无效改动
- 它确实能在部分 seed 上把 `normal` 从“落后”推到“追平”
- 但它**还没有强到能跨 seed 稳定保证 `normal >= zero-steer`**

因此，这条 baseline 目前最多只能算：

- `normal` 不再输给 `zero-steer`
  只在一部分 seed 上成立
- `normal` 还没有稳定强于 `zero-steer`
  在 `seed44` 上明确不成立

这意味着当前最合理的判断不是“steering 已经学出来了”，而是：

> `sign-safe + 0.75` 已经部分压住了“错误 steering 伤害策略”，但还没有把 steering 通道稳定约束到一个跨 seed 都安全的工作区间。

换句话说，当前 baseline 更像是从“总是有害 steering”迈到了“有时无害、有时仍然有害 steering”，还没有迈到“稳定有益 steering”。

## 6. 下一步怎么做

基于这轮控制变量结果，我认为下一步不应该直接做 `3 seeds x 100 iter`，也不应该继续沿 `sign-safe` 这条线只靠加时长硬推。更合理的是继续做单因素，但把变量收缩到“降低 stage1 steering 幅度的上界”：

1. 固定 `stage1_clip_wrong_sign_steer_enable = True`
   - 不动 reward
   - 不动 reset
   - 不动 `256x256`
   - 不动训练长度和 eval 网格

2. 只扫一个变量：`stage1_steer_action_scale`
   - 当前证据表明：
     - `1.0` 太大
     - `0.75` 对 `seed42/43` 有帮助，但对 `seed44` 仍不够稳
   - 所以下一步最自然的是补中间点：
     - `0.60`
     - `0.65`
     - 或 `0.70`

3. 推荐的最小下一实验
   - 先选一个中间点，例如 `stage1_steer_action_scale = 0.65`
   - 只跑 `seed44 x 50 iter`
   - 然后继续统一 `3x3 normal / zero-steer`
   - 目标非常具体：
     - `normal` 至少追平 `zero-steer`
     - `mean_abs_steer_applied` 明显低于当前 `0.4571`
     - `clean_insert_ready` 不再是 `0/9`

4. 只有当 `seed44` 也能被拉回到 `normal >= zero-steer`
   - 才值得把新 scale 再做成 `3 seeds x 50 iter`
   - 再之后才值得谈更长训练或 clean/dirty 的下一层改动
