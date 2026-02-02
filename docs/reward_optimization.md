# 奖励函数优化方案

## 1. 问题现象

### 1.1 训练日志分析

基于训练日志 `logs/rsl_rl/forklift_pallet_insert_lift/2026-02-02_20-25-04`：

- **Mean episode length**: ~1349 steps
  - 最大值：1350 steps = 45s ÷ (0.02s × 4 decimation)
  - 策略几乎每次都跑满 timeout
- **Mean reward**: -20 左右（稳定但负值）
- **训练步数**: 2M steps（2000 iterations）

### 1.2 视频观察

观察视频 `rl-video-step-0.mp4` 发现：

- ✗ **动作幅度极小**：叉车在微调姿态，每次调整角度不到 1 度
- ✗ **调整过程缓慢**：整个 episode 都在"磨洋工"
- ✗ **缺乏紧迫感**：策略没有快速完成任务的动力

## 2. 根因分析

### 2.1 当前奖励函数的激励偏差

```python
# 当前奖励配置（env_cfg.py）
rew_progress = 2.0        # 插入进度奖励
rew_align = -1.0          # 横向对齐惩罚
rew_yaw = -0.2            # 角度对齐惩罚
rew_lift = 1.0            # 提升奖励
rew_success = 10.0        # 成功奖励
rew_action_l2 = -0.01     # 动作 L2 惩罚 ⚠️
# 缺少时间惩罚 ⚠️
```

**问题点**：

1. **`rew_action_l2 = -0.01`**：惩罚大动作，鼓励保守策略
   - 每步惩罚 = -0.01 × Σ(action²)
   - 策略学会使用微小动作来最小化这个惩罚
   
2. **缺少时间惩罚**：慢速完成任务不扣分
   - 在 45 秒内慢慢调整也能积累正向奖励
   - 没有快速完成任务的激励

3. **episode 45 秒**：给了策略足够的"磨蹭"空间

### 2.2 策略学习的结果

- 为了最小化 `action_l2` 惩罚，选择微小动作
- 在时间充足的情况下，慢速推进也能积累正向奖励
- 最终形成"磨洋工"行为模式

## 3. 解决方案

### 3.1 方案对比

| 方案 ID | 修改内容 | Hydra 参数 | 预期效果 | 风险 |
|---------|----------|------------|---------|------|
| **A** | 移除 action L2 惩罚 | `env.rew_action_l2=0.0` | 允许大幅度动作 | 可能出现抖动 |
| **B** | 提高进度奖励 | `env.rew_progress=4.0` | 强调快速推进 | 可能牺牲对齐精度 |
| **C** | 新增时间惩罚 | 需修改代码添加 `rew_time_penalty` | 制造紧迫感 | 过大会压制探索 |
| **D** | 缩短 episode 长度 | `env.episode_length_s=30.0` | 减少可用时间 | 降低成功率 |
| **推荐组合** | A + B + C | 见下文 | 综合改善 | 需实验验证 |

### 3.2 推荐组合方案（A + B + C）

**修改参数**：
- `rew_action_l2 = 0.0`（移除动作惩罚）
- `rew_progress = 4.0`（提高进度奖励）
- `rew_time_penalty = -0.05`（新增时间惩罚）

**预期效果**：
- 策略可以使用较大幅度的动作
- 快速推进获得更多奖励
- 每步都有时间压力，鼓励高效完成

## 4. 参数建议范围与风险评估

| 参数 | 当前值 | 建议范围 | 推荐起点 | 过大风险 | 过小风险 |
|------|--------|---------|---------|---------|---------|
| `rew_action_l2` | -0.01 | [0, -0.005] | **0.0** | 保守动作 | 抖动/不稳定 |
| `rew_progress` | 2.0 | [3.0, 6.0] | **4.0** | 忽略对齐 | 推进动力不足 |
| `rew_time_penalty` | 无 | [-0.02, -0.1] | **-0.05** | 压制探索 | 缺乏紧迫感 |
| `episode_length_s` | 45.0 | [25, 45] | 45.0（保持） | 降低成功率 | 磨洋工空间大 |

## 5. 实施方法

### 5.1 方案 C 的代码修改（已实现）

**修改 1: `env_cfg.py`**

```python
# reward scales
rew_progress = 2.0
rew_align = -1.0
rew_yaw = -0.2
rew_lift = 1.0
rew_success = 10.0
rew_action_l2 = -0.01
rew_time_penalty = -0.05  # 新增：每步时间惩罚，鼓励快速完成任务
```

**修改 2: `env.py` 的 `_get_rewards()` 方法**

```python
def _get_rewards(self) -> torch.Tensor:
    # ... 现有奖励计算 ...
    
    # reward components
    rew = torch.zeros((self.num_envs,), device=self.device)
    rew += self.cfg.rew_progress * progress
    rew += self.cfg.rew_align * lateral_err
    rew += self.cfg.rew_yaw * yaw_err
    rew += self.cfg.rew_lift * torch.clamp(lift_delta, min=0.0)
    rew += self.cfg.rew_action_l2 * (self.actions**2).sum(dim=1)
    rew += self.cfg.rew_time_penalty  # 每步固定时间惩罚，鼓励快速完成
    rew += torch.where(success, torch.full_like(rew, self.cfg.rew_success), torch.zeros_like(rew))
    
    return rew
```

### 5.2 使用 Hydra 参数覆盖训练

**推荐组合训练命令**：

```bash
cd /home/uniubi/projects/forklift_sim/IsaacLab

./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Forklift-PalletInsertLift-Direct-v0 \
  --num_envs 16 \
  --headless \
  --max_iterations 2000 \
  env.rew_action_l2=0.0 \
  env.rew_progress=4.0 \
  env.rew_time_penalty=-0.05 \
  agent.run_name=exp_combo_ABC
```

**单独测试各方案**：

```bash
# 方案 A：移除 action L2 惩罚
env.rew_action_l2=0.0

# 方案 B：提高进度奖励
env.rew_progress=4.0

# 方案 D：缩短 episode 长度
env.episode_length_s=30.0
```

## 6. Baseline 备份策略

### 6.1 代码备份（Git）

```bash
cd /home/uniubi/projects/forklift_sim
git add IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/forklift_pallet_insert_lift/
git commit -m "Baseline: 奖励函数优化前的配置（episode_length=45s, rew_action_l2=-0.01）"
git tag baseline-reward-v1
```

### 6.2 Baseline 性能指标

记录当前最佳模型的性能：

- **模型路径**: `logs/rsl_rl/forklift_pallet_insert_lift/2026-02-02_20-25-04/model_1999.pt`
- **Mean episode length**: ~1349 steps（接近最大值）
- **Mean reward**: -20 左右
- **成功率**: 待测量（需补充 success metric）
- **视频样本**: `rl-video-step-0.mp4`

### 6.3 Baseline 训练命令

```bash
cd /home/uniubi/projects/forklift_sim/IsaacLab

./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Forklift-PalletInsertLift-Direct-v0 \
  --num_envs 16 \
  --headless \
  --max_iterations 2000 \
  agent.run_name=baseline_v1
```

## 7. 评估指标

### 7.1 训练过程指标（TensorBoard）

```bash
cd /home/uniubi/projects/forklift_sim/IsaacLab
./isaaclab.sh -p -m tensorboard.main \
  --logdir logs/rsl_rl/forklift_pallet_insert_lift \
  --port 6006
```

关注指标：
- **Mean reward**：越高越好（期望从 -20 提升到 0 附近）
- **Mean episode length**：期望降低到 500-800 steps（约 20-30 秒）
- **Mean value_function loss**：收敛情况
- **steps/s**：训练效率，确保无性能退化

### 7.2 策略质量指标（play.py 测试）

```bash
cd /home/uniubi/projects/forklift_sim/IsaacLab

./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
  --task Isaac-Forklift-PalletInsertLift-Direct-v0 \
  --num_envs 1 \
  --load_run <实验目录名> \
  --checkpoint model_2000.pt \
  --enable_cameras \
  --rendering_mode performance
```

评估指标：
- **成功率**：在 20 个 episode 中完成任务的比例
- **成功时间分布**：完成任务的平均 episode length
- **视频观察**：动作幅度、调整速度、是否仍"磨洋工"

## 8. 回滚计划

如果实验结果不理想：

### 8.1 恢复代码

```bash
git checkout baseline-reward-v1
```

### 8.2 恢复训练

使用 baseline 训练命令重新训练，或从已有 checkpoint 继续：

```bash
cd /home/uniubi/projects/forklift_sim/IsaacLab

./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Forklift-PalletInsertLift-Direct-v0 \
  --num_envs 16 \
  --headless \
  --resume \
  --load_run 2026-02-02_20-25-04
```

## 9. 实验建议

### 9.1 快速验证（推荐）

先用较少的训练步数快速验证方案是否有效：

```bash
# 500k steps 快速验证（约 2-3 小时）
--max_iterations 500
```

观察前 500k steps 的 TensorBoard 曲线：
- Mean episode length 是否下降？
- Mean reward 是否上升？

### 9.2 完整训练

验证有效后，进行完整训练：

```bash
# 2M steps 完整训练
--max_iterations 2000
```

### 9.3 参数微调

如果推荐组合效果不理想，可以尝试：

- **时间惩罚过大**：降低 `rew_time_penalty` 到 -0.02
- **动作不稳定**：恢复小的 action L2 惩罚，如 -0.005
- **忽略对齐**：降低 `rew_progress` 到 3.0

## 10. 预期改善

优化后的策略应表现为：

✓ **动作幅度增大**：每次调整角度 > 5 度  
✓ **调整速度加快**：episode length 降低到 500-800 steps  
✓ **保持成功率**：不牺牲任务完成质量  
✓ **奖励提升**：Mean reward 从 -20 提升到接近 0  

---

**文档版本**: v1.0  
**创建日期**: 2026-02-02  
**最后更新**: 2026-02-02
