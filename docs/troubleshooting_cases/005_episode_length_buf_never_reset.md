# 踩坑记录：_reset_idx 未调用 super() 导致 episode_length_buf 永不归零

> 版本：s1.0h | 日期：2026-02-08 | 严重程度：**致命** — 训练 5 轮后策略完全瘫痪

---

## 1. 现象

训练启动后前 2~3 个 iteration 指标看似正常（episode length 1~3），随后迅速退化：

| Iteration | episode_length | frac_timeout | action_noise_std |
|---|---|---|---|
| 0 | 1.63 | 0.17 | 2.96 |
| 1 | 3.54 | 0.35 | 2.92 |
| 2 | 2.91 | 0.53 | 2.87 |
| 3 | **1.00** | 0.70 | 2.86 |
| 4~21+ | **1.00** | **1.00** | 0.58 |

从 iter 3 开始，**所有 1024 个环境在每步都被判定为超时**，episode 永远只有 1 步，所有奖励分量归零，策略完全无法学习。

---

## 2. 排查过程

### 2.1 排除已知问题

- `insert_norm_mean: 0.0000`、`dist_front_mean: 0.5537` — 坐标系修复已生效，不是坐标 bug
- `frac_tipped: 0.0000` — 没有翻车终止
- `frac_success: 0.0000` — 没有成功终止
- 唯一的终止来源是 **timeout**（`frac_timeout` 从 0.17 线性增长到 1.0）

### 2.2 分析 frac_timeout 的增长模式

`frac_timeout` 每 iteration 增长约 18%，5 轮后达到 100%。这个增长率 **与 `num_steps_per_env / max_episode_length` 精确吻合**：

- `num_steps_per_env = 65536 / 1024 = 64`
- `max_episode_length = ceil(12.0 / 0.03333) = 360`
- `64 / 360 = 17.8%` -- 每轮约 18% 的环境新加入"永久超时"

### 2.3 检查 _get_dones() 的超时条件

```python
time_out = self.episode_length_buf >= self.max_episode_length - 1  # buf >= 359
```

`max_episode_length = 360`，timeout 需要 `episode_length_buf >= 359`。正常情况下一个 episode 最多跑 360 步才会超时。但日志显示 episode 仅 1 步就超时，说明 **`episode_length_buf` 的值异常大**。

### 2.4 追踪 episode_length_buf 的生命周期

Isaac Lab 基类 `DirectRLEnv.step()` 的流程：

```
episode_length_buf += 1          # 每步 +1
_get_dones()                     # 检查终止/超时
_get_rewards()                   # 计算奖励
_reset_idx(reset_env_ids)        # 重置结束的环境
_get_observations()              # 计算观测
```

基类 `DirectRLEnv._reset_idx()` 中有关键一行：

```python
self.episode_length_buf[env_ids] = 0  # 重置 episode 计数器
```

**但用户的 `_reset_idx()` 完全覆盖了基类方法，且没有调用 `super()._reset_idx()`！**

```python
def _reset_idx(self, env_ids):
    if env_ids is None:
        env_ids = torch.arange(...)
    # ... 各种自定义重置逻辑 ...
    # 没有 super()._reset_idx(env_ids)
    # 没有 self.episode_length_buf[env_ids] = 0
```

### 2.5 发现 RSL-RL 的初始随机化

RSL-RL 的 `train.py` 调用 `runner.learn(init_at_random_ep_len=True)`，这会在训练开始前随机初始化 `episode_length_buf`：

```python
# RSL-RL on_policy_runner.py (line 67)
self.env.episode_length_buf = torch.randint_like(
    self.env.episode_length_buf, high=int(self.env.max_episode_length)  # [0, 360)
)
```

**目的**：打散各环境的重置时机，避免所有环境同时重置。

**但由于 `_reset_idx()` 不归零 `episode_length_buf`，这些随机初始值永远不会被清除！**

---

## 3. 根因分析

### 3.1 完整因果链

```
RSL-RL 随机化 episode_length_buf 为 [0, 359]
    → 每步 +1，初始值高的环境很快达到 >= 359
    → _get_dones() 返回 time_out = True
    → 基类调用 _reset_idx()
    → 用户覆盖的 _reset_idx() 不归零 episode_length_buf
    → buf 保持在 359+，下一步变成 360+
    → 又触发 timeout → 又调用 _reset_idx() → 又不归零
    → 该环境永久卡在 1 步超时
    → 每 iteration 约 18% 的环境加入"永久超时"
    → ~5.6 iteration 后全部 1024 环境陷入死循环
    → episode length = 1，策略完全无法学习
```

### 3.2 数值验证

`episode_length_buf` 初始随机值为 [0, 359]。每步 +1，经过 N 步后值为 [N, N+359]。其中 >= 359 的环境数 = `min(N+1, 360) / 360`：

| 总步数 (N) | iteration | 理论 frac_timeout | 实际值 |
|---|---|---|---|
| 64 | 0 | 65/360 = 18.1% | **17%** |
| 128 | 1 | 129/360 = 35.8% | **35%** |
| 192 | 2 | 193/360 = 53.6% | **53%** |
| 256 | 3 | 257/360 = 71.4% | **70%** |
| 360+ | 5+ | 100% | **100%** |

理论值与实际值完全吻合，确认根因。

---

## 4. 修复方案

在 `_reset_idx()` 开头调用 `super()._reset_idx(env_ids)`：

```python
def _reset_idx(self, env_ids):
    if env_ids is None:
        env_ids = torch.arange(self.num_envs, device=self.device)

    # 必须调用基类 _reset_idx()！否则 episode_length_buf 永远不归零
    super()._reset_idx(env_ids)

    # ... 后续自定义重置逻辑 ...
```

`super()._reset_idx(env_ids)` 会执行：
- `self.scene.reset(env_ids)` — 重置场景资产
- 事件管理器 / 噪声模型重置
- **`self.episode_length_buf[env_ids] = 0`** — 关键！重置 episode 计数器

---

## 5. 经验教训

### 5.1 核心规则：覆盖基类方法时必须考虑 super()

在 Isaac Lab 的 `DirectRLEnv` 中，`_reset_idx()` 基类实现包含多个关键操作：

```python
def _reset_idx(self, env_ids):
    self.scene.reset(env_ids)           # 重置场景
    # ... 事件/噪声重置 ...
    self.episode_length_buf[env_ids] = 0  # 重置 episode 计数器
```

如果子类覆盖 `_reset_idx()` 却不调用 `super()`，以上所有操作都会丢失。**`episode_length_buf` 归零是其中最致命的遗漏。**

### 5.2 为什么单环境测试发现不了

- 单环境（`num_envs=1`）时 `init_at_random_ep_len` 仍然生效
- 但只有 1 个环境，随机到的初始值可能较小，需要很多步才触发
- verify 脚本（`verify_forklift_insert_lift.py`）通常只跑几百步，可能恰好没到触发阈值
- **只有在多环境长时间训练时问题才暴露**

### 5.3 快速识别此类 bug 的信号

如果你看到以下组合，几乎可以确定是 `episode_length_buf` 未重置：

1. **`frac_timeout` 每 iteration 线性增长**（增长率 = num_steps_per_env / max_episode_length）
2. **增长到 1.0 后永远不下降**
3. **`Mean episode length` 在 `frac_timeout` 到达 1.0 后锁定在 1.00**
4. **`frac_tipped = 0.0`**（不是翻车导致的）
5. **所有奖励分量接近 0**（策略没有时间做任何事）

### 5.4 防范措施

1. **覆盖 `_reset_idx()` 时始终以 `super()._reset_idx(env_ids)` 开头**
2. **在 `_reset_idx()` 末尾加断言验证**：`assert (self.episode_length_buf[env_ids] == 0).all()`
3. **训练前 3 轮必须检查 `frac_timeout` 趋势** — 如果线性增长，立即停训排查
4. **代码审查重点**：任何覆盖基类方法的地方，都要确认是否需要调用 `super()`

---

## 6. 快速诊断 Checklist

遇到 `Mean episode length` 异常小且 `frac_timeout` 线性增长时：

- [ ] `_reset_idx()` 是否调用了 `super()._reset_idx(env_ids)`？
- [ ] `episode_length_buf` 在 reset 后是否归零？
- [ ] `init_at_random_ep_len` 是否为 True？（如果是，episode_length_buf 的初始值是随机的）
- [ ] `frac_timeout` 的增长率是否等于 `num_steps_per_env / max_episode_length`？
- [ ] 在 `_reset_idx()` 中是否有 `self.sim.reset()` 等全局操作可能干扰其他环境？
