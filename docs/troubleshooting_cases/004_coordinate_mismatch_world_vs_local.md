# 踩坑记录：多环境训练中世界坐标与局部坐标混用导致指标全面异常

> 版本：s1.0h | 日期：2026-02-08 | 严重程度：**致命** — 策略完全无法学习

---

## 1. 现象

启动 1024 环境并行训练后，日志中出现以下**全面异常**的指标：

| 指标 | 异常值 | 正常范围（参考 s1.0f） |
|---|---|---|
| `insert_norm_mean` | **10.98** | 0 ~ 1 |
| `dist_front_mean` | **24.28 m** | 2 ~ 4 m |
| `frac_inserted` | **0.5000**（恒定） | 初期 0.0，逐步增长 |
| `lateral_mean` | **0.0000** | ~0.3 m（初始有随机化） |
| `yaw_deg_mean` | **0.0003** | ~8°（初始有随机化） |
| `Mean episode length` | **1.00** | 30 ~ 300 步 |
| `frac_timeout` | **1.0000** | 逐步下降 |
| `Computation` | **760 steps/s** | 数千 steps/s |

关键特征：**所有指标从第 1 个 iteration 起就异常，且完全不随训练变化**。这不是策略"学歪了"，而是观测/奖励计算本身就有系统性错误。

---

## 2. 排查过程

### 2.1 交叉对比历史日志

首先确认是否为 s1.0h 特有的问题：

- **s0.6 日志**：`lateral_mean: 0.34`, `insert_norm_mean: 0.0`, `dist_front_mean: 2.5~3.1`, episode length 逐步增长 — **正常**
- **s1.0f 日志**：`lateral_mean: 0.33`, `insert_norm_mean: 0.0`, `dist_front_mean: 3.0~3.5`, episode length 31→256 — **正常**
- **s1.0h 日志**：如上表，**全面异常**

结论：**bug 是 s1.0h 重构时引入的**。

### 2.2 分析异常值的数学含义

`insert_norm_mean ≈ 11` 意味着 `insert_depth / pallet_depth_m ≈ 11`，即 `insert_depth ≈ 11 × 2.16 ≈ 23.8 m`。这远超任何物理可能的插入深度，说明**不是物理仿真问题，而是纯计算错误**。

`frac_inserted = 0.5000` 恒定，暗示恰好一半的环境被判定为"已插入"。这与 1024 个环境在网格中排列、约一半 x > 0 约一半 x < 0 的分布一致。

### 2.3 定位到坐标系不匹配

检查 `insert_depth` 的计算公式：

```python
# _get_observations() / _get_rewards() 中（修复前）
tip = self._compute_fork_tip()                          # 返回世界坐标
insert_depth = torch.clamp(tip[:, 0] - self._pallet_front_x, min=0.0)
```

其中：
- `_compute_fork_tip()` 使用 `self.robot.data.root_pos_w`，返回**世界坐标**
- `self._pallet_front_x` 来自 `self.cfg.pallet_cfg.init_state.pos[0] - pallet_depth_m * 0.5 = 0.0 - 1.08 = -1.08`，是**局部坐标标量**

两者坐标系不同！

---

## 3. 根因分析

### 3.1 Isaac Lab 多环境布局

Isaac Lab 将 N 个环境以 `spacing=6.0m` 排列在世界空间的网格中。对于 1024 个环境（32×32 网格），世界 x 坐标范围大约为 **-96m 到 +96m**。

### 3.2 bug 的数学推导

以第 500 个环境为例：
- 该环境世界 x 偏移 ≈ +90m
- `tip[:, 0]`（世界坐标） ≈ 90 - 1.87 = 88.13m（叉车在环境局部偏移 -1.87m）
- `self._pallet_front_x`（局部坐标） = -1.08m

```
insert_depth = tip_x - _pallet_front_x
             = 88.13 - (-1.08)
             = 89.21 m    ← 完全错误！实际应为 ~0m
```

而对于 x 为负的环境：
- `tip[:, 0]` ≈ -90 - 1.87 = -91.87m
- `insert_depth = -91.87 - (-1.08) = -90.79` → clamp 到 0

这解释了 `frac_inserted = 0.5` — x > 0 的一半环境全部被判定为深度插入，x < 0 的一半 clamp 到 0。

### 3.3 为什么 `lateral_mean = 0` 和 `yaw_deg_mean ≈ 0`

虽然初始化有 ±0.6m 横向和 ±14.3° 偏航随机化，但 episode 只持续 1 步。第 1 步时所有环境刚被同时 reset，reset 过程会将状态写入 sim 并步进一步物理——此时观测值几乎等于 reset 时写入的均值（随机化正负抵消）。

### 3.4 因果链：坐标 bug → 策略崩溃 → 速度暴降

```
坐标系不匹配
    → insert_depth 计算出错（insert_norm ≈ 11，应为 0~1）
    → 观测向量包含垃圾值
    → 奖励计算全面失真（r_insert: 163 vs 正常 0~5）
    → 策略收到无意义的信号，梯度混乱
    → 策略崩溃，随机动作
    → episode 1 步后超时终止
    → 每个 simulation step 重置全部 1024 个环境
    → _reset_idx() 包含 write_data_to_sim + sim.reset，开销巨大
    → 训练速度仅 760 steps/s（正常应为数千 steps/s）
```

**速度慢的主因不是碰撞计算或 maxConvexHulls，而是每步全量重置 1024 环境。**

对比数据：
- **s1.0f**（正常）: episode length 31→256 步，每步仅重置 ~5 个环境，速度 ~17000 steps/s
- **s1.0h**（异常）: episode length 1 步，每步重置全部 1024 环境，速度 760 steps/s

---

## 4. 修复方案

在 `_get_observations()` 和 `_get_rewards()` 中，将固定标量 `self._pallet_front_x` 替换为**从 pallet 实际世界位置动态计算的逐环境张量**：

```python
# 修复后
pallet_pos = self.pallet.data.root_pos_w
pallet_front_x = pallet_pos[:, 0] - self.cfg.pallet_depth_m * 0.5  # (N,) 世界坐标
insert_depth = torch.clamp(tip[:, 0] - pallet_front_x, min=0.0)
```

替换位置：
- `_get_observations()`：insert_depth 计算（第 759~761 行）
- `_get_rewards()`：insert_depth 和 dist_front 计算（第 820~827 行）

**`_reset_idx()` 中的 `self._pallet_front_x` 不需要修改**，因为 reset 时操作的也是局部坐标，两者一致。

---

## 5. 修复后预期效果

| 指标 | 修复前 | 修复后预期 |
|---|---|---|
| `insert_norm_mean` | ~11 | 0~1（初期接近 0） |
| `dist_front_mean` | ~24m | 2~4m |
| `frac_inserted` | 0.5（恒定） | 0.0（初期），逐步增长 |
| `lateral_mean` | 0.0 | ~0.3m（初始随机化值） |
| `Mean episode length` | 1.00 | 30~300 步 |
| 训练速度 | 760 steps/s | 数千 steps/s |

---

## 6. 经验教训

### 6.1 核心规则：多环境 RL 中永远注意坐标系

在 Isaac Lab 的多环境并行训练中：

- **`xxx.data.root_pos_w`** 返回的是**世界坐标**，每个环境不同
- **`cfg.xxx.init_state.pos`** 定义的是**相对于环境原点的局部坐标**，所有环境共享同一值
- 两者**绝对不能直接做差**

单环境（`num_envs=1`）测试时 bug 不会暴露，因为 env_0 的世界原点就是局部原点（偏移 = 0）。**只有多环境并行时才会出现。**

### 6.2 快速识别此类 bug 的信号

如果你看到以下组合，几乎可以确定是世界/局部坐标混用：

1. **归一化指标远超 [0, 1] 范围**（如 insert_norm > 2）
2. **距离指标远超物理合理范围**（如 dist_front > 10m，而场景只有几米大）
3. **分数指标恒定在 0.5**（对称网格布局的特征）
4. **Episode length = 1**（观测/奖励全乱，策略立即崩溃）
5. **上述异常从第 1 个 iteration 就存在**（不是训练过程中逐渐恶化）

### 6.3 防范措施

1. **添加断言/日志**：在关键计算处加 sanity check，例如 `assert insert_norm.max() < 2.0`
2. **少量环境冒烟测试**：在 `num_envs=4` 下运行几个 iteration，打印每个环境的 insert_depth、dist_front 等，确认跨环境一致
3. **重构时逐步验证**：修改观测/奖励计算后，先对比前后版本在相同初始状态下的输出，再启动全量训练
4. **代码审查重点**：任何 `self._xxx`（标量缓存）与 `self.xxx.data.root_pos_w`（张量）做运算的地方，都要检查坐标系是否匹配

---

## 7. 快速诊断 Checklist

遇到训练指标异常时，按以下顺序排查：

- [ ] **Mean episode length 是否 = 1？** 如果是，说明 done 条件或观测/奖励有严重错误
- [ ] **归一化指标是否在合理范围？** insert_norm 应在 0~1，lateral/yaw 应与初始随机化匹配
- [ ] **异常是从第 1 个 iteration 就存在还是逐渐出现？** 第 1 个就异常 = 计算bug；逐渐恶化 = 策略/奖励设计问题
- [ ] **对比历史正常版本的日志**，定位到具体哪个版本引入了 bug
- [ ] **检查所有坐标运算**：`_compute_fork_tip()` 返回世界坐标，与之做差的量是否也是世界坐标？
- [ ] **检查 `frac_inserted` 是否恒定在 0.5**：这是世界/局部坐标混用的标志性特征（对称网格布局）
