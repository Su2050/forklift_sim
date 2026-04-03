# Exp9.0 O1/O2/O2b 全面复盘与 O3 方案设计

日期：`2026-04-02`

## 1. 实验概况

| 项目 | 基线 | O1 | O2 | O2b |
| --- | ---: | ---: | ---: | ---: |
| run name | `..master_init_seed42_iter400` | `..rewardfix_o1_seed42` | `..rewardfix_o2_seed42` | `..rewardfix_o2b_seed42` |
| iterations | 400 | 2000 | 2000 | 550 (提前停止) |
| seed / num_envs | 42 / 64 | 42 / 64 | 42 / 64 | 42 / 64 |
| 核心改动 | — | 压低脏插入奖励 + 延长 preinsert shaping | 新增 postinsert_align Gaussian shaping | push_free bonus 改挂 hold_entry |

### 各实验 override 差异

| 参数 | 基线 | O1 | O2 | O2b |
| --- | :---: | :---: | :---: | :---: |
| `clean_insert_gate_floor` | 0.15 | **0.05** | **0.05** | **0.05** |
| `clean_insert_push_free_bonus_enable` | true | **false** | **false** | **true (weight=3.0)** |
| `preinsert_insert_frac_max` | 0.20 | **0.40** | **0.40** | **0.45** |
| `preinsert_y_err_delta_weight` | 1.0 | **2.0** | **2.0** | **2.0** |
| `preinsert_yaw_err_delta_weight` | 1.0 | **0.6** | **1.0** | **1.0** |
| `preinsert_dist_front_delta_weight` | 0.30 | **0.15** | **0.15** | **0.15** |
| `postinsert_align_enable` | — | — | **true** | false |
| `postinsert_center_sigma_m` | — | — | **0.20** | — |
| `postinsert_tip_sigma_m` | — | — | **0.15** | — |
| `postinsert_align_weight` | — | — | **3.0** | — |
| push_free bonus 触发条件 | inserted | — | — | **hold_entry (代码改动)** |

## 2. 终态对比（各实验 last 50 轮均值）

| 指标 | 基线 | O1 | O2 | O2b | 最优 |
| --- | ---: | ---: | ---: | ---: | :--- |
| `frac_inserted` | 53.9% | 55.3% | **57.9%** | 50.7% | O2 |
| `frac_aligned` | 4.3% | **5.9%** | 4.5% | 3.9% | O1 |
| `frac_tip_constraint_ok` | 13.3% | **19.1%** | 14.0% | 15.0% | O1 |
| `frac_hold_entry` | **0.25%** | 0.00% | 0.06% | 0.12% | 基线 |
| `frac_success` | **0.09%** | 0.00% | 0.00% | 0.06% | 基线 |
| `center_y_err (ins)` | 0.424m | **0.372m** | 0.397m | 0.397m | O1 |
| `tip_y_err (ins)` | 0.430m | **0.381m** | 0.412m | 0.430m | O1 |
| `yaw_err (ins)` | **6.35°** | 8.11° | 8.51° | 8.79° | 基线 |
| `max_hold_counter` | 1.38 | 0.18 | **2.71** | 0.52 | O2 |
| `R_plus` | 5.25 | 3.20 | 4.40 | 4.48 | — |
| `pallet_disp (ins)` | **0.232** | 0.288 | 0.245 | 0.369 | 基线 |

### 早期学习活跃期对比（iter 50-200）

| 指标 | 基线 | O1 | O2 | O2b |
| --- | ---: | ---: | ---: | ---: |
| `center_y_err (ins)` | 0.410m | 0.386m | **0.367m** | 0.385m |
| `tip_y_err (ins)` | 0.408m | 0.399m | **0.371m** | 0.386m |
| `yaw_err (ins)` | **6.99°** | 7.38° | 7.45° | 8.61° |
| `hold_entry` | 0.14% | 0.14% | **0.23%** | 0.09% |
| `max_hold_counter` | 0.65 | 0.51 | **1.60** | 0.43 |

**关键发现**：O2 在早期（iter 50-200）的 center/tip 误差和 hold_entry 其实是四轮中最好的，但后期退化了。

## 3. O2 失败的根因：Gaussian sigma 选择错误

### 3.1 梯度分析

O2 使用 `postinsert_center_sigma_m=0.20`、`postinsert_tip_sigma_m=0.15`。在当前误差水平下：

| 参数 | 当前误差 | shaping 值 | 梯度 (d/d_err) |
| --- | ---: | ---: | ---: |
| center: σ=0.20 | 0.39m | **0.022** | -0.44 |
| tip: σ=0.15 | 0.41m | **0.0006** | **-0.02** |

**tip 的 shaping 值只有 0.0006，梯度几乎为零。** 策略在当前误差水平上根本感受不到 tip 方向的奖励梯度。center 的也只有 0.022，信号极弱。

这解释了为什么 O2 的 `r_postinsert_align` 全程输出 ~0.55-0.67 却没有驱动误差下降：奖励值主要来自 `postinsert_active` 的 baseline 值（所有 inserted 样本都有），而不是来自误差减小的梯度。

### 3.2 不同 sigma 下的梯度对比

以 `err=0.39m` 为例：

| sigma | shaping 值 | 梯度 | 相对 σ=0.20 |
| ---: | ---: | ---: | :--- |
| 0.20 | 0.022 | -0.44 | 基准 |
| 0.30 | 0.185 | -1.60 | 梯度 **3.6x** |
| **0.40** | **0.387** | **-1.88** | 梯度 **4.3x** |
| 0.50 | 0.544 | -1.70 | 梯度 3.9x |
| 0.60 | 0.655 | -1.42 | 梯度 3.2x |

**最优梯度点在 σ ≈ 0.40**（对应 `err/σ ≈ 1.0`，Gaussian 梯度峰值）。

以 `err=0.41m`（tip）为例：

| sigma | shaping 值 | 梯度 |
| ---: | ---: | ---: |
| 0.15 | 0.0006 | -0.02 |
| **0.40** | **0.350** | **-1.79** |

sigma 从 0.15 放大到 0.40，**tip 的梯度提升 86 倍**。

### 3.3 为什么 O2 早期有效但后期退化

O2 在 iter 50-200 时 center/tip 误差确实在下降（center: 0.41→0.37, tip: 0.42→0.37），这是因为早期部分样本的误差较小（分布尾部），能感受到 Gaussian 梯度。但随着策略收敛到 ~0.38m 的均值，所有样本都落在 Gaussian 的平坦区，梯度消失，学习停滞。

## 4. 所有实验的共性问题：yaw 恶化

| 实验 | yaw_err (last50) | 相对基线 |
| --- | ---: | :--- |
| 基线 | **6.35°** | — |
| O1 | 8.11° | +1.76° |
| O2 | 8.51° | +2.16° |
| O2b | 8.79° | +2.44° |

所有改进实验的 yaw 都比基线差。原因：

1. O1 把 `preinsert_yaw_err_delta_weight` 从 1.0 降到 0.6
2. O2/O2b 恢复到 1.0 但不够：postinsert shaping 只约束 center/tip，策略发现"横移不管角度"也能拿到 postinsert 奖励
3. `aligned` 的 yaw 阈值是 8°，当前 8.5° 刚好卡在门槛外

**yaw 是 hold_entry 的隐性瓶颈**：即使 center/tip 降到阈值以下，yaw > 8° 也会阻止 aligned → hold_entry。

## 5. O3 方案设计

### 5.1 核心思路

修正 O2 的两个缺陷：
1. **放大 sigma** 使 Gaussian 梯度在当前误差水平有效
2. **加入 yaw shaping** 防止策略牺牲角度换横移

### 5.2 代码改动（env.py）

在现有 `r_postinsert_align` 计算中加入 yaw 项：

```python
# O2 → O3: 加入 yaw shaping
if self.cfg.postinsert_align_enable:
    postinsert_active = (insert_depth >= self._insert_thresh).float()
    center_y_shaping = torch.exp(
        -(center_y_err / max(self.cfg.postinsert_center_sigma_m, 1e-6)) ** 2
    )
    tip_y_shaping = torch.where(
        dist_front <= self.cfg.tip_align_near_dist,
        torch.exp(
            -(tip_y_err / max(self.cfg.postinsert_tip_sigma_m, 1e-6)) ** 2
        ),
        torch.ones_like(tip_y_err),
    )
    yaw_shaping = torch.exp(
        -(yaw_err_deg / max(self.cfg.postinsert_yaw_sigma_deg, 1e-6)) ** 2
    )
    r_postinsert_align = postinsert_active * self.cfg.postinsert_align_weight * (
        self.cfg.postinsert_center_weight * center_y_shaping
        + self.cfg.postinsert_tip_weight * tip_y_shaping
        + self.cfg.postinsert_yaw_weight * yaw_shaping
    )
```

### 5.3 新增 config 参数（env_cfg.py）

```python
postinsert_yaw_sigma_deg: float = 10.0
postinsert_yaw_weight: float = 0.5
```

### 5.4 O3 完整 override

```bash
# --- O1 有效 override（保留）---
env.clean_insert_gate_floor=0.05
env.clean_insert_push_free_bonus_enable=false
env.preinsert_align_reward_enable=true
env.preinsert_insert_frac_max=0.40
env.preinsert_y_err_delta_weight=2.0
env.preinsert_yaw_err_delta_weight=1.5    # 从 1.0 提到 1.5，加强 yaw 纠偏
env.preinsert_dist_front_delta_weight=0.15

# --- O3: 修正后的 postinsert shaping ---
env.postinsert_align_enable=true
env.postinsert_align_weight=4.0           # 从 3.0 → 4.0
env.postinsert_center_sigma_m=0.40        # 从 0.20 → 0.40（梯度 4.3x）
env.postinsert_tip_sigma_m=0.40           # 从 0.15 → 0.40（梯度 86x）
env.postinsert_center_weight=1.0
env.postinsert_tip_weight=1.0
env.postinsert_yaw_sigma_deg=10.0         # 新增：yaw shaping
env.postinsert_yaw_weight=0.5             # 新增：yaw 权重（低于 center/tip）
```

### 5.5 sigma 选择依据

| 分量 | 当前误差 | sigma | err/σ | shaping | 梯度 | 目标误差处 shaping |
| --- | ---: | ---: | ---: | ---: | ---: | :--- |
| center | 0.39m | 0.40m | 0.98 | 0.387 | -1.88 | 0.15m → 0.87 |
| tip | 0.41m | 0.40m | 1.03 | 0.350 | -1.79 | 0.12m → 0.91 |
| yaw | 8.5° | 10.0° | 0.85 | 0.486 | -0.83 | 5.0° → 0.78 |

所有分量在当前误差水平都处于 Gaussian 的最大梯度区间（err/σ ≈ 0.85-1.03），确保策略能感受到强信号。

### 5.6 预期观察点

如果 O3 有效，应在 **200-400 iter** 内看到：

1. `center_lateral_inserted_mean` 持续下降突破 0.30m（O2 停在 0.37m）
2. `tip_lateral_inserted_mean` 持续下降突破 0.30m
3. `yaw_deg_inserted_mean` 不再恶化，保持 ≤ 7°
4. `aligned` 从 ~5% 开始上升
5. `hold_entry` 出现持续非零值

如果 **500 iter 后 center/tip 仍停在 > 0.35m**，说明 postinsert shaping 的权重不够，考虑：
- 提高 `postinsert_align_weight` 到 6.0-8.0
- 或缩小 sigma 到 0.35（此时误差已经下降到更低水平）

### 5.7 风险与备选

| 风险 | 应对 |
| --- | --- |
| postinsert 奖励过大，策略不再前进只顾横移 | 观察 `frac_inserted` 是否大幅下降；如是，降低 weight |
| yaw shaping 和 lateral shaping 冲突 | yaw_weight=0.5 已经较低；如 yaw 过度约束，降到 0.3 |
| sigma=0.40 在误差降到 0.20m 后梯度变弱 | 此时 shaping=0.78，梯度仍有 -1.56，足够；真正需要关注的是 < 0.15m |

## 6. 执行计划

1. 改 `env.py`：在 postinsert_align 块中加入 yaw_shaping
2. 改 `env_cfg.py`：新增 `postinsert_yaw_sigma_deg` 和 `postinsert_yaw_weight`
3. 创建 `scripts/run_exp90_rewardfix_o3_remote.sh`
4. commit → sync patch → 启动训练（2000 iter）
