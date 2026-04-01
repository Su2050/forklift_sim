# Exp9.0 Rewardfix O1 终态总结与下一步优化方案

日期：`2026-04-01`

## 1. 训练概况

| 项目 | 基线 (400 iter) | O1 (2000 iter) |
| --- | ---: | ---: |
| run name | `exp9_0_no_reference_master_init_seed42_iter400` | `exp9_0_no_reference_rewardfix_o1_seed42` |
| iterations | 400 | 2000 |
| 总步数 | 1,638,400 | 8,192,000 |
| 训练时长 | ~56 min | ~4h41m |
| seed | 42 | 42 |
| num_envs | 64 | 64 |

## 2. 终态对比（最后 50 轮均值）

| 指标 | 基线 last50 | O1 last50 | 变化 |
| --- | ---: | ---: | :--- |
| `frac_inserted` | 0.5391 | 0.5525 | 基本持平 |
| `frac_aligned` | 0.0425 | **0.0591** | +39% |
| `frac_tip_constraint_ok` | 0.1334 | **0.1913** | +43% |
| `frac_hold_entry` | 0.0025 | 0.0000 | 变差 |
| `frac_success` | 0.0009 | 0.0000 | 变差 |
| `center_lateral_inserted_mean` | 0.4241m | **0.3723m** | -12% |
| `tip_lateral_inserted_mean` | 0.4301m | **0.3807m** | -11% |
| `yaw_deg_inserted_mean` | 6.3491° | 8.1135° | 变差 |
| `hold_exit_exceeded_frac` | 0.9675 | 0.9860 | 略差 |
| `R_plus` | 5.2494 | 3.1987 | 被压低（符合预期） |
| `clean_insert_gate_inserted_mean` | 0.1507 | **0.0504** | 被压到 floor 附近 |
| `r_clean_insert_bonus` | 0.0844 | 0.0000 | 已关闭 |

## 3. 全程统计

| 指标 | 基线 (400 iter) | O1 (2000 iter) |
| --- | ---: | ---: |
| `success > 0` 的 iteration 数 | 16 | 35 |
| `hold_entry > 0` 的 iteration 数 | 47 | 186 |
| `aligned > 0` 的 iteration 数 | 355 | 1908 |
| `max_hold_counter` 峰值 | 10.0 | 10.0 |
| 最后 100 轮 `yaw <= 8°` 的轮数 | 45/50 | 71/100 |
| 最后 100 轮 `center <= 0.15m` 的轮数 | 0/50 | 0/100 |
| 最后 100 轮 `tip <= 0.12m` 的轮数 | 0/50 | 0/100 |

## 4. 趋势分析

按阶段看 O1 的关键指标均值：

| 阶段 | aligned | tip_ok | hold_entry | success | center_err | tip_err | yaw_err |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0-99 | 0.0336 | 0.1392 | 0.0022 | 0.0005 | 0.4122 | 0.4306 | 7.67 |
| 100-399 | 0.0311 | 0.1296 | 0.0017 | 0.0002 | 0.3836 | 0.3945 | 6.49 |
| 400-799 | 0.0524 | 0.1447 | 0.0019 | 0.0004 | 0.3826 | 0.3927 | 6.86 |
| 800-1199 | 0.0589 | 0.1444 | 0.0011 | 0.0001 | 0.3847 | 0.3971 | 7.14 |
| 1200-1599 | 0.0498 | 0.1470 | 0.0015 | 0.0003 | 0.3796 | 0.3911 | 7.48 |
| 1600-1799 | 0.0461 | 0.1573 | 0.0019 | 0.0005 | 0.3847 | 0.4012 | 6.97 |
| 1800-1999 | 0.0536 | 0.1487 | 0.0005 | 0.0001 | 0.3850 | 0.3978 | 7.57 |

最后 500 轮线性斜率：

- `aligned`：+2.1e-5/iter（几乎平）
- `tip_constraint_ok`：+2.4e-5/iter（几乎平）
- `hold_entry`：-3.2e-6/iter（微降）
- `success`：-1.5e-6/iter（微降）
- `center_err`：-2.0e-5/iter（几乎平）
- `tip_err`：-4.8e-5/iter（几乎平）

**结论：O1 在 iter 400 左右就已经平台化，后续 1600 轮没有实质性进步。**

## 5. 判读

### 5.1 O1 做对了什么

1. **成功压低了"脏插入挣钱"**：`clean_insert_gate_inserted_mean` 从 `0.15` 压到 `0.05`，`R_plus` 从 `5.25` 降到 `3.20`。
2. **中间指标有改善**：`aligned` +39%，`tip_constraint_ok` +43%，`center/tip` 误差各降 ~11%。
3. **preinsert shaping 延长生效**：`preinsert_active_frac` 从基线的约 `0.02` 提升到约 `0.5`，说明更多样本在接收纠偏信号。

### 5.2 O1 没解决什么

1. **center/tip 误差仍然远高于 gate**：`center ≈ 0.37m`（阈值 `0.15m`，2.5x），`tip ≈ 0.38m`（阈值 `0.12m`，3.2x）。
2. **hold_entry/success 没有被抬起来**：尾段甚至比基线更差。
3. **yaw 反而略微恶化**：从 `6.35°` 升到 `8.11°`，可能是因为 `preinsert_yaw_err_delta_weight` 被降低了。

### 5.3 根因

O1 的三条改动本质上都是在"减少脏插入的正奖励"或"延长已有的 delta shaping"。但当前的核心问题是：

> **策略在插入后缺少一个直接的、持续的"把 center/tip 横向误差往 0 压"的密集奖励信号。**

- `r_d`（距离 target_center）是标量距离奖励，不直接约束横向分量。
- `preinsert_align` 用的是 root `y_err` 的 delta，不是 `center_y_err` 或 `tip_y_err`，而且只在 `insert_norm < 0.45` 时生效。
- `clean_insert_gate` 只是衰减 `r_d`，不是给一个正向的"横向误差减小就加分"的信号。
- 一旦插深了（`insert_norm > 0.45`），策略就只剩 `r_d`（被 gate 压到很低）和极稀疏的 `rg`，没有任何密集信号引导它继续横移修正。

## 6. 下一步优化方案

### 方案 O2：新增 post-insert 横向/tip 密集 shaping（需要改代码）

这是我认为最值得试的一条路。核心思路是：在 `inserted` 之后，给一个直接奖励 `center_y_err` 和 `tip_y_err` 减小的密集信号。

#### 6.1 新增 `r_postinsert_align`（改 `env.py`）

在 reward 计算中，`inserted` 且 `insert_norm >= preinsert_insert_frac_max` 的样本上，新增：

```python
postinsert_active = (insert_depth >= self._insert_thresh).float()

center_y_shaping = torch.exp(
    -(center_y_err / cfg.postinsert_center_sigma_m) ** 2
)
tip_y_shaping = torch.where(
    dist_front <= cfg.tip_align_near_dist,
    torch.exp(-(tip_y_err / cfg.postinsert_tip_sigma_m) ** 2),
    torch.ones_like(tip_y_err),
)

r_postinsert_align = postinsert_active * cfg.postinsert_align_weight * (
    cfg.postinsert_center_weight * center_y_shaping
    + cfg.postinsert_tip_weight * tip_y_shaping
)
```

然后把 `r_postinsert_align` 加到 `R_plus` 里。

#### 6.2 对应 config 参数（改 `env_cfg.py`）

```python
postinsert_align_enable: bool = True
postinsert_align_weight: float = 3.0
postinsert_center_sigma_m: float = 0.20
postinsert_tip_sigma_m: float = 0.15
postinsert_center_weight: float = 1.0
postinsert_tip_weight: float = 1.0
```

`sigma` 的选择逻辑：
- `center_sigma = 0.20m`：当前均值 `0.37m` 时 gate 值约 `exp(-(0.37/0.20)^2) ≈ 0.03`，但 `0.15m` 时约 `0.57`，形成从当前位置到阈值的持续梯度。
- `tip_sigma = 0.15m`：类似逻辑，让 `0.12m` 时约 `0.57`。

#### 6.3 同时保留 O1 的 override

O1 的"压低脏插入正奖励"和"关闭过早 bonus"是对的，应该保留：

```bash
env.clean_insert_gate_floor=0.05
env.clean_insert_gate_start_frac=0.15
env.clean_insert_gate_ramp_frac=0.25
env.clean_insert_push_free_bonus_enable=false
```

#### 6.4 恢复 preinsert yaw 权重

O1 里把 `preinsert_yaw_err_delta_weight` 降到了 `0.6`，导致 yaw 略微恶化。建议恢复到 `1.0`：

```bash
env.preinsert_yaw_err_delta_weight=1.0
```

### 方案 O2b（备选）：把 bonus 改成 hold_entry 后才发

如果不想新增 reward 项，可以改一行代码：把 `inserted_push_free_reward` 的条件从 `inserted && push_free` 改成 `hold_state.hold_entry && push_free`，然后重新打开 bonus：

```bash
env.clean_insert_push_free_bonus_enable=true
env.clean_insert_push_free_bonus_weight=3.0
```

这样 bonus 就变成了"进入 hold 区才给"的稀疏但有意义的正反馈，而不是"只要插进去就给"的过早正反馈。

### 方案 O2c（备选）：放宽 gate 阈值做课程

如果上面两条都不想改代码，可以先把 gate 阈值放宽，让策略先学会"进 hold"，再逐步收紧：

```bash
env.max_lateral_err_m=0.30
env.tip_align_entry_m=0.25
```

但这条我不太推荐作为第一选择，因为它本质上是在降低任务难度，而不是在教策略学会精对齐。

## 7. 推荐执行顺序

1. **优先试 O2**（新增 `r_postinsert_align`）：这是最直接解决"插入后缺少横向密集信号"的方案。
2. 如果 O2 效果不够，**叠加 O2b**（bonus 改成 hold_entry 后才发）。
3. 如果前两条都不够，**再考虑 O2c**（放宽 gate 做课程）。

## 8. 预期观察点

如果 O2 有效，最先应该看到的是：

1. `center_lateral_inserted_mean` 和 `tip_lateral_inserted_mean` 开始持续下降（不再平台化在 `0.37-0.38m`）
2. `aligned` 和 `tip_constraint_ok` 同步上升
3. `hold_entry` 从接近 `0` 开始有持续的非零值
4. `success` 在 `hold_entry` 稳定后开始出现

如果跑了 `500` 轮还没看到 `center/tip` 误差下降的趋势，说明 `postinsert_align_weight` 或 `sigma` 需要调整。
