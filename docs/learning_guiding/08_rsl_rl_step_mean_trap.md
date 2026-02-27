# RSL-RL 日志的 Step 级均值陷阱：为什么 0.5% 的成功率实际上是 100%？

> 日期：2026-02-27
> 标签：`RSL-RL`, `Logging`, `Metrics`, `Success Rate`

在 RL 训练中，我们经常会遇到指标不符合直觉的情况。本文记录了一个极其隐蔽的工程陷阱：**RSL-RL 的日志系统如何用 0.5% 的数据掩盖了 Agent 100% 的完美表现**。

---

## 1. 现象：令人绝望的“低成功率”

在经过多轮奖励函数优化和参数调整后，我们的叉车插入举升任务训练日志显示：
- `frac_inserted` (插入率) 稳定在 ~15%
- `frac_lifted` (举升率) 稳定在 ~8%
- **`frac_success` (成功率) 却死死卡在 0.4% ~ 0.6% 之间**

直觉上，这意味着 Agent 尝试了 1000 次，只有 5 次成功。我们因此制定了庞大的诊断计划，试图找出 “Hold 阶段（保持举升状态）” 失败的原因，怀疑是物理引擎抖动、判定条件过于苛刻，甚至打算引入复杂的课程学习。

---

## 2. 破案过程：我是如何发现这个问题的？

为了排查 Hold 阶段到底哪里出了问题，我在 `env.py` 中加入了极度细致的诊断日志：

```python
# 记录在 Hold 阶段中，具体是哪个子条件（插入/对齐/举升）失败了
self.extras["log"]["diag_hold/fail_ins_frac"] = ((~insert_entry) & (self._hold_counter > 0)).float().mean()
self.extras["log"]["diag_hold/fail_align_frac"] = ((~align_entry) & (self._hold_counter > 0)).float().mean()
self.extras["log"]["diag_hold/fail_lift_frac"] = ((~lift_entry) & (self._hold_counter > 0)).float().mean()

# 记录 Hold 计数器的最大值
self.extras["log"]["phase/hold_counter_max"] = self._hold_counter.float().max()
```

跑完训练后，我用 `grep` 提取了这些日志，结果让我大吃一惊：
- `fail_ins_frac` 永远是 **0.0000**
- `fail_align_frac` 永远是 **0.0000**
- `fail_lift_frac` 只有极少数 iteration 是 0.0010，绝大多数是 **0.0000**
- `hold_counter_max` 几乎每个 iteration 都能达到满分 **10.0000**

**矛盾出现了：**
如果成功率只有 0.5%，意味着 Agent 绝大多数时候在 Hold 阶段失败了。但诊断日志却说：**Agent 一旦进入 Hold 阶段，就再也没有失败过！它稳稳地拿到了满分。**

这促使我重新审视 `frac_success` 这个指标的计算方式。

---

## 3. 根因分析：RSL-RL 的 Step 级求均值机制

在 RSL-RL 的架构中，我们在 `env.py` 里写入 `self.extras["log"]` 的数据，是**在每个 Environment Step 被收集，并在整个 Rollout（通常是 24 步）结束后求均值**。

让我们来看看 `success` 是怎么计算和重置的：

```python
# 1. 判断是否成功
success = self._hold_counter >= self._hold_steps

# 2. 写入日志
self.extras["log"]["phase/frac_success"] = success.float().mean()

# 3. 触发环境重置（在 _get_dones 和 _reset_idx 中）
if success:
    self._hold_counter = 0  # 重置计数器
```

由于 `DirectRLEnv.step()` 的调用顺序是 `_get_dones()` -> `_get_rewards()` -> `_reset_idx()`，当 `success` 达成时：
1. 当前 Step，`success` 为 True，被记录到日志。
2. 紧接着，环境被 Reset，叉车回到起点，`_hold_counter` 清零。
3. 下一个 Step，`success` 变为 False。

**结论：在一次长达数百步的完美 Episode 中，`success` 标志位仅仅只会亮起 1 到 2 个 Step。**

---

## 4. 数学推导：为什么乘以 Mean_episode_length？

如果一个 Agent 是**完美的大师**，每次都能成功，它的数据会是怎样的？

假设 `Mean_episode_length` 为 400 步（即 Agent 花了 400 步完成任务并触发 success 重置）。
在这 400 步中：
- 只有最后 2 步的 `success` 是 1.0
- 前面 398 步的 `success` 都是 0.0

RSL-RL 计算的均值 `frac_success` = `(2 * 1.0 + 398 * 0.0) / 400` = **0.005（即 0.5%）**。

这就是为什么我们要用以下公式来还原真实的 Episode 成功率：

```text
真实 Episode 成功率 = (frac_success / 2) * Mean_episode_length
```

注：除以 2 是因为在我们这套代码逻辑中，`success` 状态会在 `_get_dones` 和 `_get_rewards` 之间存活，通常会占据 2 个 step 的记录。具体除以 1 还是 2 取决于底层 reset 时序，但乘以 `Mean_episode_length` 是必须的。

### 代入真实数据验证

| 实验 | frac_success | Mean_episode_length | 还原后的真实成功率 |
|------|-------------|-------------------|------------------|
| Baseline | 0.0005 | 350 | `0.0005 / 2 * 350` = **8.75%** |
| Exp-A（后期） | 0.0049 | 409 | `0.0049 / 2 * 409` = **100%** |
| Exp-A（均值） | 0.0042 | ~400 | `0.0042 / 2 * 400` = **84%** |

我们以为的 0.5% 失败率，实际上是 85%~100% 的大师级表现。

---

## 5. 经验教训与最佳实践

1. **永远不要用 Step 级的 Flag 均值来评估 Episode 级事件。**  
   像 `success`、`timeout` 这种只在 Episode 结束时触发一次的布尔值，其 Step 均值会被 Episode 长度严重稀释。

2. **正确做法是记录 Episode 级指标。**  
   在环境内部维护 Episode 级计数器，在 reset 时写入成功/失败原因，例如：

```python
self._ep_success_count += success.sum().item()
self._ep_total_count += reset_buf.sum().item()

if self._ep_total_count > 0:
    self.extras["log"]["episode/success_rate"] = self._ep_success_count / self._ep_total_count
```

3. **当指标违背直觉时，先怀疑指标定义，再怀疑算法本身。**  
   如果不是先加了 `fail_ins_frac` / `fail_align_frac` / `fail_lift_frac` 这些诊断日志并发现矛盾，我们很可能会在错误方向（改判定、做课程学习）上继续浪费大量时间。
