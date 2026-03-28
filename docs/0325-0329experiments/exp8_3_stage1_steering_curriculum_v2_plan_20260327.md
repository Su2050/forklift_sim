## exp8.3 stage1 steering curriculum v2 plan (2026-03-27)

### 背景

`v1` 已经证明两件事：

1. 当前主问题确实不是“只差再拉长训练”，而是 `stage1` 原始 reset 太对齐，策略容易学成“往前叉”。
2. 直接把 `y/yaw` 放宽到 `±0.10m / ±4°` 再加第一版 pre-insert shaping，虽然能打掉一部分前推捷径，但 `3 seeds x 50 iter` 结果仍然分流成：
   - `dirty-insert`
   - `no-insert`
   - `weak-success`

因此，下一阶段不应直接把 `v1` 拉到几百 iter，而是先做一个更平衡的 `v2`：

- 比旧的对齐课程更需要 steering
- 比 `v1` 更不容易一上来就掉进 `no-insert`
- 继续保留 `bonusw=1.0`，不把问题重新混成 reward 主线改动

---

### 分支策略

- 基线来源：`exp/exp8_3_stage1_steering_curriculum`
- 基线提交：`cad5c7f1` (`Add stage1 steering curriculum v1`)
- 新分支：`exp/exp8_3_stage1_steering_curriculum_v2`

`v2` 必须独立分支推进，原因是：

1. `v1` 已经形成完整实验闭环，后续需要保留它作为“第一次打掉前推捷径”的参考线。
2. `v2` 的目标不是继续扩大扰动，而是重新平衡“必须 steering”和“不要直接塌成 no-insert”。
3. 这轮如果有效，后面才值得把同一配置拉到 `200 iter`。

---

### v2 核心思路

`v2` 不再继续把课程做得更激进，而是改成：

1. reset 比旧版更宽，但比 `v1` 更温和
2. pre-insert shaping 更偏向“先摆正”，更少偏向“继续往前顶”
3. 保留 `bonusw=1.0`
4. 验收标准从“能不能成功”切换到“有没有真正学会 steering”

---

### v2 参数调整

#### 1. stage1 reset

保持 `x` 不变：

- `stage1_init_x_min_m = -3.55`
- `stage1_init_x_max_m = -3.25`

把 `y/yaw` 从 `v1` 的：

- `y = ±0.10`
- `yaw = ±4°`

收回到更平衡的：

- `y = ±0.08`
- `yaw = ±3°`

设计意图：

- 仍然让 steering 成为必要技能
- 但避免把早期训练直接推入 `no-insert / retreat`

#### 2. pre-insert shaping

保留：

- `preinsert_align_reward_enable = true`
- `preinsert_y_err_delta_weight = 1.0`

调整为：

- `preinsert_yaw_err_delta_weight: 0.5 -> 1.0`
- `preinsert_dist_front_delta_weight: 0.75 -> 0.30`
- `preinsert_retreat_penalty_weight: 1.0 -> 0.50`

设计意图：

- 更强调“横向/朝向变好”
- 更少奖励“继续前冲”
- 仍然保留一定 retreat 惩罚，但避免把策略过早压成保守不插

其余 clean bonus/gate 配置保持与当前 `bonusw=1.0` 主线一致，不额外引入新的强 gate / dirty penalty。

---

### 执行步骤

#### Phase A：代码落地

1. 在 `env_cfg.py` 中实现：
   - `stage1 y/yaw` 调整到 `±0.08 / ±3°`
   - `preinsert` 权重调整到 `yaw=1.0 / dist=0.30 / retreat=0.50`
2. 不改 `x`
3. 不引入新的强 gate / dirty penalty

#### Phase B：最小验证

1. `py_compile`
2. `install_into_isaaclab.sh`
3. `2 iter smoke`
4. 确认新日志项仍正常打出：
   - `paper_reward/r_preinsert_align`
   - `paper_reward/r_preinsert_retreat`
   - `diag/preinsert_*`

#### Phase C：短训筛选

1. `3 seeds x 50 iter`
2. 固定口径：
   - near-field
   - `256x256`
   - `64 envs`
   - `bonusw=1.0`
3. 主要看：
   - `phase/frac_inserted_push_free`
   - `phase/frac_hold_entry`
   - `phase/frac_success`
   - `phase/frac_dirty_insert`
   - `diag/pallet_disp_xy_mean`

#### Phase D：grid 回归

对每个 seed 的末尾 checkpoint 立刻做：

1. `normal misalignment grid`
2. `zero-steer misalignment grid`

并计算：

- `normal_success_rate_grid`
- `zero_steer_success_rate_grid`
- `steer_gap = normal - zero_steer`

---

### 硬验收标准

只有同时满足下面两个条件，才值得把这条配置往几百 iter 推：

1. `normal` 明显强于 `zero-steer`
   - 也就是 `steer_gap` 要明显大于 `0`
2. 至少 `2/3` 个 seed 在 `50 iter` 后出现稳定的：
   - `push_free`
   - `hold`
   - 非零 `success`

这里的“稳定”不要求最终 success 特别高，但不能只是单点抖动。

---

### 通过后怎么推

如果 `v2` 通过硬验收标准：

1. 先推到 `200 iter`
2. 不直接上 `400 / 800 iter`
3. `200 iter` 后再做统一 eval 和新一轮 grid

原因：

- `200 iter` 已足够判断 steering-based basin 是否开始成型
- 当前阶段最缺的是“方向验证”，不是更长训练账单

---

### 当前不建议做的事

1. 不建议继续推旧的 aligned `bonusw=1.0` 训练
2. 不建议直接把 `v1` 拉到 `200 / 400 iter`
3. 不建议继续扫更多 `bonus weight`
4. 不建议重新加重 `gate / dirty penalty`
5. 不建议先回 `wide reset`

---

### 一句话目标

`v2` 的目标不是立刻拿最高 success，而是：

**先让策略必须靠 steering 才能成功，并且让这种 steering-based 成功至少在 `2/3` seed 上开始稳定出现。**
