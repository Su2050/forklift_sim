# 奖励函数优化方案 v2

## 1. 问题诊断：4个体检题

### 1.1 时间惩罚无效且过猛

**问题**：`rew_time_penalty=-0.05` 在当前阶段（几乎都 timeout）基本无用，且数量级破坏奖励排序。

**计算验证**：
- `dt = 1/120`，`decimation = 4`，`step_dt = 1/30 ≈ 0.03333s`
- `episode_length_s = 45s`，`max_episode_length = 45 / 0.03333 ≈ 1350 steps`
- 时间惩罚累计：`-0.05 × 1350 = -67.5`
- 成功奖励：`10`
- **结果**：即使成功，净回报仍是 `-57.5`，成功在回报排序里没有优势

**根因**：
- Episode 基本都跑满（长度固定），固定的每步惩罚对所有回合等价于常数偏置
- 不会产生"快点干完"的梯度信号，因为 agent 还没学会提前结束
- 更阴险：如果存在"失败可提前终止"的条件，会诱导"快速触发失败来少扣时间费"

### 1.2 Progress 虽是 delta 但无 gate

**问题**：允许"歪着硬怼"也算进步，且可能出现"负负得正"。

**当前代码**：
```python
progress = insert_depth - self._last_insert_depth  # delta型 ✓
rew += self.cfg.rew_progress * progress  # 但无gate机制 ✗
```

**根因**：
- 不管对齐与否都给 progress 奖励
- Agent 会学会"歪着硬怼也能拿分"
- 如果惩罚方式不对（负×负=正），还可能"后退赚钱"

### 1.3 action_l2=0 会导致抖动

**问题**：去掉 L2 惩罚，常见后果是动作幅度变大且频率变高（抖动）。

**根因**：
- L2 惩罚的是"动作大小"，但真实问题是"来回改主意"（高频变化）
- 应该惩罚动作变化率（action rate）而非幅度

### 1.4 success=10 太小

**问题**：无法在总回报里建立压倒性优势，制度养出形式主义。

**根因**：
- 终极目标如果在"总回报"里没有压倒性优势，agent 会变成最懂制度漏洞的基层干部
- 策略优化重点转向"少扣分"而非"完成任务"

---

## 2. 奖励函数设计原则

基于诊断，确立以下设计原则：

1. **成功必须在回报排序里有压倒性优势**
   - `rew_success` 应是单步 shaping 的 10-100 倍
   - 成功回合的 return 必须稳定显著高于失败回合

2. **进度奖励必须 gate 到对齐达标**
   - 对齐时给正奖励，未对齐时惩罚正向推进
   - 避免"歪着硬怼"和"后退赚钱"

3. **惩罚动作变化率而非幅度**
   - 允许"大动作"（方向盘一次打够）
   - 不允许"抽搐式大动作"（左一下右一下）

4. **时间压力写进成功奖励而非 per-step penalty**
   - 成功越快奖金越高
   - 失败不会因为早结束而占便宜

---

## 3. 新奖励函数架构

### 3.1 奖励公式

```python
def _get_rewards(self) -> torch.Tensor:
    # 1. Gated Progress（对齐达标才给推进奖励）
    gate_aligned = (lateral_err <= gate_lateral_err_m) & (yaw_err <= gate_yaw_err_deg)
    wrong_progress = torch.clamp(progress, min=0.0)  # 只惩罚正向推进
    progress_reward = torch.where(
        gate_aligned,
        rew_progress * progress,           # 对齐时：奖励推进
        -rew_wrong_progress * wrong_progress  # 未对齐时：惩罚正向推进
    )

    # 2. Action Rate Penalty（惩罚动作变化率，首步不惩罚）
    action_rate = ||a_t - a_{t-1}||²
    action_rate_penalty = rew_action_rate * action_rate  # 负值

    # 3. Base Shaping Rewards
    rew = progress_reward
    rew += rew_align * lateral_err       # 负值惩罚
    rew += rew_yaw * yaw_err             # 负值惩罚
    rew += rew_lift * clamp(lift_delta, min=0)  # 正值奖励
    rew += action_rate_penalty

    # 4. Success with Time Bonus（成功奖励 + 时间奖金）
    time_ratio = step / max_steps
    time_bonus = rew_success_time * (1 - time_ratio)  # 越早成功奖金越高
    success_total = rew_success + time_bonus
    rew += success_total if success else 0

    # 5. Failure Terminal Penalty（失败终止惩罚，包含timeout）
    is_failure = (terminated | timeout) & (~success)
    rew += rew_failure_terminal if is_failure else 0

    return rew
```

### 3.2 参数配置表

| 参数 | 原值 | 新值 | 说明 |
|------|------|------|------|
| `rew_progress` | 2.0 | **4.0** | 提高（对齐时给予） |
| `rew_align` | -1.0 | -1.0 | 保持 |
| `rew_yaw` | -0.2 | -0.2 | 保持 |
| `rew_lift` | 1.0 | 1.0 | 保持 |
| `rew_success` | 10.0 | **100.0** | 大幅提升（10倍） |
| `rew_action_l2` | -0.01 | **0.0** | 移除（用action_rate替代） |
| `rew_time_penalty` | -0.05 | **0.0** | 移除 |
| `rew_wrong_progress` | - | **2.0** | 新增：惩罚未对齐时的正向推进 |
| `rew_action_rate` | - | **-0.02** | 新增：动作变化率惩罚 |
| `rew_success_time` | - | **30.0** | 新增：时间奖金 |
| `rew_failure_terminal` | - | **-30.0** | 新增：失败终止惩罚 |
| `gate_lateral_err_m` | - | **0.05** | 新增：对齐gate阈值 |
| `gate_yaw_err_deg` | - | **5.0** | 新增：角度gate阈值 |

### 3.3 参数定标方法

**success_total 定标**：
- 目标完成时间：600 steps（约20秒）
- 成功基础奖励：100
- 时间奖金（600步完成）：`30 × (1 - 600/1350) ≈ 16.7`
- 成功总奖励：`100 + 16.7 = 116.7`
- 失败惩罚：`-30`
- **成功/失败差距**：`116.7 - (-30) = 146.7`（显著差距）

**action_rate 定标**：
- 从 `-0.01` 开始（比原 action_l2 略小）
- 观察 action 输出方差，如果动作僵化则减小

---

## 4. 关键实现细节

### 4.1 首步 action rate 处理

```python
# __init__
self._last_actions = torch.zeros((num_envs, action_space), device=device)
self._is_first_step = torch.ones((num_envs,), dtype=torch.bool, device=device)

# _get_rewards
action_rate_penalty = torch.where(
    self._is_first_step,
    torch.zeros_like(action_rate),  # 首步不惩罚
    self.cfg.rew_action_rate * action_rate
)
self._last_actions = self.actions.clone()
self._is_first_step[:] = False

# _reset_idx
self._last_actions[env_ids] = 0.0
self._is_first_step[env_ids] = True
```

### 4.2 错误推进惩罚的符号处理

```python
# 只惩罚未对齐时的正向推进，避免"后退赚钱"
wrong_progress = torch.clamp(progress, min=0.0)  # 只取正值
progress_reward = torch.where(
    gate_aligned,
    rew_progress * progress,           # 对齐：正常奖励/惩罚
    -rew_wrong_progress * wrong_progress  # 未对齐：只惩罚前进
)
```

### 4.3 failure penalty 覆盖 timeout

```python
# timeout 也视为失败并惩罚
is_terminated = self.reset_terminated
is_timeout = self.reset_time_outs
is_failure = (is_terminated | is_timeout) & (~success)
rew += torch.where(is_failure, rew_failure_terminal, 0.0)
```

---

## 5. 实验方案

### 5.1 Phase 1：单变量测试（500 iterations）

**P1.1 Gated Progress**
```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Forklift-PalletInsertLift-Direct-v0 \
  --num_envs 1024 --headless --max_iterations 500 \
  env.rew_progress=4.0 env.rew_wrong_progress=2.0 \
  env.gate_lateral_err_m=0.05 env.gate_yaw_err_deg=5.0 \
  agent.run_name=exp_gated_progress
```

**P1.2 Action Rate**
```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Forklift-PalletInsertLift-Direct-v0 \
  --num_envs 1024 --headless --max_iterations 500 \
  env.rew_action_l2=0.0 env.rew_action_rate=-0.01 \
  agent.run_name=exp_action_rate
```

**P1.3 New Success Structure**
```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Forklift-PalletInsertLift-Direct-v0 \
  --num_envs 1024 --headless --max_iterations 500 \
  env.rew_success=100.0 env.rew_success_time=30.0 \
  env.rew_failure_terminal=-30.0 env.rew_time_penalty=0.0 \
  agent.run_name=exp_success_structure
```

### 5.2 Phase 2：完整组合（2000 iterations）

```bash
cd /home/uniubi/projects/forklift_sim/IsaacLab

nohup ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Forklift-PalletInsertLift-Direct-v0 \
  --num_envs 1024 \
  --headless \
  --max_iterations 2000 \
  env.rew_progress=4.0 \
  env.rew_wrong_progress=2.0 \
  env.gate_lateral_err_m=0.05 \
  env.gate_yaw_err_deg=5.0 \
  env.rew_action_l2=0.0 \
  env.rew_action_rate=-0.02 \
  env.rew_success=100.0 \
  env.rew_success_time=30.0 \
  env.rew_failure_terminal=-30.0 \
  env.rew_time_penalty=0.0 \
  agent.run_name=exp_full_rewrite_v1 > train_full_v1.log 2>&1 &
```

---

## 6. 验收标准

### 6.1 核心指标

1. **成功回合 return >> 失败回合 return**
   - 抽样 100 个 episode，计算成功/失败回合的平均 return
   - 差距要大到肉眼可见（建议 > 50）

2. **Episode length 分布有双峰**
   - 成功回合：快速完成（< 600 steps）
   - 失败回合：timeout（~1350 steps）或提前终止

3. **Action 输出不饱和且无高频抖动**
   - Action 均值不接近 ±1
   - Action 方差适中（非接近0也非极大）

### 6.2 统计口径

在 env 中或评估脚本中统计：
- `success_count`：成功次数
- `timeout_count`：超时次数
- `failure_count`：翻车等硬失败次数
- `success_return_mean`：成功回合平均回报
- `failure_return_mean`：失败回合平均回报

---

## 7. 风险与缓解

| 风险 | 缓解措施 |
|------|---------|
| success 过大导致探索不足 | Phase 1 先测试，观察前 100k steps 的探索情况 |
| gated progress 导致初期无法学习推进 | gate 阈值设置略宽于成功阈值（0.05 vs 0.03） |
| action_rate 导致动作僵化 | 从 -0.01 开始，观察 action 输出方差 |
| failure penalty 导致"活着就好" | 确保 success_total >> \|failure_terminal\| |

---

## 8. 回滚计划

```bash
# Git 备份
git add -A
git commit -m "Baseline before reward function rewrite"
git tag baseline-before-rewrite

# 回滚
git checkout baseline-before-rewrite
```

所有 Hydra 参数都可以命令行覆盖，无需修改代码即可快速回滚到旧参数。

---

**文档版本**: v2.0
**创建日期**: 2026-02-02
**最后更新**: 2026-02-03
