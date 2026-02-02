# 复盘：Mean episode length 长期=1.00（每步重开）如何定位与修复

本文记录一次真实排障：训练“能跑”，但 `Mean episode length` 长期贴着 `1.00`，导致策略几乎学不到多步行为。最终发现**根因不止一个**：既有“翻车判定/出生嵌地”的物理问题，也有“DirectRLEnv reset 流程用错”的代码问题。

适用人群：

- 你看到训练日志里 `Mean episode length` 经常=1 或很小（1~5）
- 你在任务里依赖 tipped/success/time_out 等 done 条件
- 你自己实现了 Direct workflow 环境（`DirectRLEnv`），写了 `_reset_idx()`

---

## 1. 现象与危害：为什么 episode length = 1.00 很危险

### 1.1 现象

训练日志中反复出现：

- `Mean episode length: 1.00`
- `Mean reward` 长期接近某个常数（不一定大/小）

这代表：**环境平均每个 episode 只走 1 个 step 就被 reset**。

### 1.2 为什么这会让叉车任务学不到东西

叉车插托盘需要多步策略链条：

1) 靠近托盘  
2) 对准（横向误差 + yaw 误差）  
3) 插入（插入深度增长）  
4) 抬升（fork tip 高度提升）  

如果 episode 几乎每步就结束，策略只能“学到第一步的局部奖励/惩罚”，无法形成多步规划。

### 1.3 快速健康判断标准（强烈建议记住）

- **必须满足**：`Mean episode length` 明显大于 10，并且整体趋势不应长期卡在 1~2
- **最好满足**：随着训练推进，逐步到几十/上百（视最大 episode 长度而定）

---

## 2. 第一次假设：99% 的“1步一死”来自 tipped（翻车）过度触发

当 done 条件包含 tipped 时，最常见的原因是：

- 物理上真的翻车（初始嵌地、碰撞爆炸、动作过猛）
- 或者 roll/pitch 计算/四元数顺序/姿态写入有问题，导致“看起来没翻但代码判翻”

### 2.1 最短闭环：A/B 测试确认 tipped 是否元凶

目标：只定位来源，不追求正确性。

在 `_get_dones()` 中，把：

```python
terminated = tipped | success
```

临时改成：

```python
terminated = success
```

观察 2~3 个 iteration：

- **如果 `Mean episode length` 立刻变大**：基本确认 tipped 是主要来源
- **如果仍≈1**：说明还有其他 reset/终止逻辑在影响（继续往第 3 节查）

### 2.2 加 debug 的方法：打印 tipped 比例与 roll/pitch 的统计

建议在 `_get_dones()` 临时打印（每 N 步一次）：

- `tipped_ratio`（被判翻车的 env 占比）
- `roll/pitch` 的 max/mean
- 阈值 `max_roll_pitch_rad`

这样能区分两类情况：

- **reset 后 roll/pitch 就超阈值**：姿态写入/初始 pose 有问题
- **走几步后突然超阈值**：更像碰撞爆炸或动作过猛

---

## 3. 第一次修复：从“翻车/物理”侧把系统稳住（让 tipped_ratio 下降）

这次排障中，tipped 很快被压到 0%，关键动作是：

### 3.1 提高 reset 初始高度（最关键）

将初始 z 从 `0.03m` 提高到 `0.1m`。

直觉解释：

- 很多机器人 USD 的实际几何尺寸/碰撞体高度比你想象的大
- z 太低会导致车身/轮子“嵌入地面”
- 物理引擎会强行把刚体“弹出”嵌入 → 产生巨大冲击 → 瞬间翻车 → tipped 触发

### 3.2 降低动作幅度（辅助稳定）

将 `wheel_speed_rad_s`、`steer_angle_rad` 降低，让初期随机策略不会产生大冲击。

### 3.3 放宽翻车阈值（临时）

将 `max_roll_pitch_rad` 从 ~25° 放宽到 ~34°，给 early training 更大容忍度。后续策略稳定后再逐步收紧。

### 3.4 结果：tipped_ratio=0% 但 episode length 仍会回到 1

这一步很关键：它说明“翻车”不是唯一问题。

当日志里出现类似现象：

- `tipped_ratio = 0%`
- `success_ratio = 0%`
- 但 `Mean episode length` 仍频繁掉到 `1.00`

就要立刻怀疑：**reset 流程本身写错了**（见下一节）。

---

## 4. 真正的根因：DirectRLEnv 的 reset 流程用错（没调 super + 误用 sim.reset）

### 4.1 关键知识：DirectRLEnv._reset_idx() 在做什么

在 IsaacLab 的 direct workflow 中，`DirectRLEnv._reset_idx(env_ids)` 会做两件非常关键的事：

1) `self.scene.reset(env_ids)`：按 env_ids 重置场景内部 buffer  
2) `self.episode_length_buf[env_ids] = 0`：重置 episode 计数器  

因此，你在自定义环境里写 `_reset_idx()` 时，**必须**在合适位置调用：

```python
super()._reset_idx(env_ids)
```

否则会出现非常诡异的表现：看起来没有触发 tipped/success/time_out，但环境仍频繁 reset，episode length 统计异常。

### 4.2 另一个大坑：不要在 _reset_idx 里调用 sim.reset()

`self.sim.reset()` 属于“全局仿真 reset”，会破坏并行环境（vectorized env）中按 env_ids 做局部 reset 的语义，导致：

- 非目标 env 被意外 reset
- 物理状态/scene buffer 不一致
- 训练统计出现离谱现象（包括 episode length 异常）

官方环境的 direct reset 实现（例如 AnyMal、Quadcopter）都不会在 `_reset_idx()` 中调用 `sim.reset()`。

### 4.3 规范 reset 顺序（建议照抄官方模式）

推荐顺序：

1) `self.robot.reset(env_ids)`：清理 articulation 内部 buffer  
2) `super()._reset_idx(env_ids)`：scene reset + episode_length_buf reset  
3) 写入 root pose/vel、关节状态等  
4) 清理自定义缓存（例如 `actions`、`_hold_counter`、`_last_insert_depth` 等）  

---

## 5. 顺带发现：观测维度配置不一致（14 vs 13）

日志里出现：

- Actor/Critic MLP `in_features=13`

但配置中 `observation_space = 14`。

这类不一致有时会被框架“自动兜底”，但会埋雷（尤其当你后续改观测、加噪声模型、或换训练框架时）。建议保持一致：

- 要么把配置改成 13
- 要么补齐观测到 14（并同步更新网络/配置）

---

## 6. 最终验证：什么才算“真的修好了”

修复 reset 流程后，本次训练指标出现立刻改善：

- `Mean episode length` 从 1 附近快速上升到 **几十/上百**
- `steps/s` 也变得非常高且稳定（不再出现 reset 抖动带来的异常）

典型健康信号：

- `Mean episode length`：持续 > 10，并逐步上升
- reward：可以是负的（早期常见），但不应“永远卡死在某个常数”

---

## 7. 可复用的排障流程清单（Checklist）

当你再次遇到 `Mean episode length` 异常（1~2）时，按以下顺序查：

1) **先判定是不是 done 条件触发**  
   - A/B：暂时禁用 tipped（或 success）观察 episode length 是否立刻变长
2) **打印 done 的分解统计**  
   - tipped_ratio / success_ratio / time_out_ratio（按 env 平均）
3) **检查 reset 是否按 direct workflow 正确实现**  
   - `_reset_idx()` 是否调用了 `super()._reset_idx(env_ids)`  
   - 是否误用 `sim.reset()`  
   - env_ids 处理是否正确（tensor/list/ALL_INDICES）
4) **检查初始 pose 是否合理**  
   - 初始 z 是否会嵌入地面  
   - 动作幅度是否太大（early training 先收敛到安全范围）
5) **检查观测/配置一致性**  
   - observation_space 与实际 concat 维度一致  
   - 网络输入维度与实际 obs 一致  

---

## 8. 关联阅读

- 常见现象总入口：`docs/06_troubleshooting.md`
- 任务设计（观测/奖励/终止）：`docs/03_task_design_rl.md`

