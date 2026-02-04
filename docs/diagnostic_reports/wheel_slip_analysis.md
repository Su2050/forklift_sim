# 货叉怼托盘时轮子打滑问题诊断报告

## 问题现象

从训练视频和日志观察到：
- **货叉怼到托盘时，轮子出现打滑**
- 训练日志显示 `obs/insert_norm_mean: 0.0000`（插入深度几乎为0）
- `err/lateral_mean: 0.0525`（横向误差约5.25cm，超过成功阈值3cm）
- `err/yaw_deg_mean: 2.9088`度（接近阈值3度）

## 诊断步骤执行结果

### 步骤1：验证对齐精度

**训练结束时的对齐指标**（迭代1999/2000）：
- `err/lateral_mean: 0.0525` (5.25cm) - **超标**（阈值：3cm）
- `err/yaw_deg_mean: 2.9088` (2.9度) - **接近阈值**（阈值：3度）
- `obs/insert_norm_mean: 0.0000` - **插入深度为0**

**训练过程中的对齐指标变化**（最后20次迭代）：
- 横向误差稳定在 **5.1-5.4cm** 之间，持续超标
- 偏航误差稳定在 **2.8-3.0度** 之间，接近阈值上限
- 插入深度始终为 **0.0000**，从未成功插入

**对齐完成度**：
- `phase/frac_aligned: 0.7012` (70%) - 70%的环境达到对齐条件
- `phase/frac_inserted: 0.0000` (0%) - **没有任何环境成功插入**

### 步骤2：检查奖励函数

**当前使用的奖励函数**（`env.py` 第253-293行）：

```python
rew = 0
rew += self.cfg.rew_progress * progress          # 插入深度增量奖励
rew += self.cfg.rew_align * lateral_err          # 横向对齐惩罚（负值）
rew += self.cfg.rew_yaw * yaw_err                # 偏航角惩罚（负值）
rew += self.cfg.rew_lift * clamp(lift_delta, min=0.0)  # 抬升奖励
rew += self.cfg.rew_action_l2 * ||actions||²     # 动作平滑惩罚
rew += rew_success (if success)                  # 成功奖励
```

**奖励系数配置**（`env_cfg.py`）：
- `rew_progress = 2.0` - 插入进度奖励
- `rew_align = -1.0` - 横向对齐惩罚系数
- `rew_yaw = -0.2` - 偏航角惩罚系数
- `rew_success = 10.0` - 成功奖励

**训练日志中的奖励组件**（迭代1999/2000）：
- `rew/r_align: 0.0025` - 对齐奖励很小（因为lateral_err约5.4cm，惩罚 = -1.0 × 0.054 = -0.054）
- `rew/r_insert: 0.0000` - 插入奖励为0（因为progress为0）
- `rew/success: 0.0000` - 成功奖励为0（因为没有成功）

**关键发现**：
- 奖励函数**没有gate机制**，对齐惩罚在所有距离都生效
- 插入进度奖励**没有对齐gate**，即使未对齐也能获得推进奖励
- 这可能导致策略学会"歪着硬怼"而非精确对齐

### 步骤3：物理机制分析

**托盘配置**（`env_cfg.py` 第150行）：
- `kinematic_enabled = True` - 托盘是**固定不动**的
- `disable_gravity = True` - 禁用重力
- `max_depenetration_velocity = 1.0` - 最大穿透恢复速度

**插入深度计算**（`env.py` 第235-237行）：
```python
tip = self._compute_fork_tip()
insert_depth = torch.clamp(tip[:, 0] - self._pallet_front_x, min=0.0)
insert_norm = (insert_depth / (self.cfg.pallet_depth_m + 1e-6)).unsqueeze(-1)
```

**安全机制**（`env.py` 第166-169行）：
```python
# two-stage safety: if already inserted enough, suppress driving and let it lift
inserted = self._last_insert_depth >= self._insert_thresh
drive = torch.where(inserted, torch.zeros_like(drive), drive)
steer = torch.where(inserted, torch.zeros_like(steer), steer)
```

**物理交互机制**：
1. 当货叉**没有对准孔位**时，会撞到托盘的前面（而非插入孔位）
2. 托盘是kinematic固定物体，会**阻挡叉车前进**
3. 轮子继续转动（因为策略仍在输出drive动作），但叉车被托盘阻挡
4. 这导致**轮子打滑**（轮子转动但车体不前进）

## 问题定位总结

### 核心问题

**货叉没有对准托盘孔位，导致撞到托盘前面而非插入孔位，引发轮子打滑。**

### 根本原因

1. **对齐精度不足**：
   - 横向位置偏差过大（5.25cm vs 3cm要求）
   - 偏航角度接近临界值（2.9° vs 3°要求）
   - 策略没有学会精确对齐

2. **奖励函数设计问题**：
   - 缺少对齐gate机制，对齐惩罚在所有距离都生效
   - 插入进度奖励没有对齐gate，允许未对齐时推进
   - 对齐惩罚系数可能不够强（`rew_align = -1.0`）

3. **物理交互机制**：
   - 托盘是kinematic固定物体，碰撞后产生阻挡力
   - 当货叉撞到托盘前面时，轮子继续转动但车体被阻挡
   - 导致轮子打滑现象

### 关键证据

1. **插入深度为0**：`obs/insert_norm_mean: 0.0000` 表明货叉没有真正插入
2. **横向对齐超标**：`err/lateral_mean: 0.0525`（5.25cm）超过成功阈值3cm
3. **偏航误差接近阈值**：`err/yaw_deg_mean: 2.9088`度 接近3°上限
4. **无成功案例**：`phase/frac_inserted: 0.0000` 表明训练过程中没有任何环境成功插入
5. **对齐完成度不准确**：`phase/frac_aligned: 0.7012` 显示70%环境达到对齐条件，但实际横向误差仍然超标，说明对齐判断可能不够严格

### 结论

**这不是物理参数问题，而是策略学习问题**：
- 策略在训练过程中没有学会精确对齐
- 横向误差从训练初期的~30cm降低到训练结束时的~5cm，说明策略在学习对齐
- 但精度仍然不够（5.25cm > 3cm阈值），导致货叉无法插入孔位
- 当货叉撞到托盘前面时，由于托盘固定不动，产生阻挡力，轮子打滑

## 相关文件

- 环境配置：`forklift_pallet_insert_lift_project/isaaclab_patch/source/isaaclab_tasks/isaaclab_tasks/direct/forklift_pallet_insert_lift/env_cfg.py`
- 环境实现：`forklift_pallet_insert_lift_project/isaaclab_patch/source/isaaclab_tasks/isaaclab_tasks/direct/forklift_pallet_insert_lift/env.py`
- 训练日志：`train_insert_norm_fix_v2.log`
- 训练结果目录：`IsaacLab/logs/rsl_rl/forklift_pallet_insert_lift/2026-02-04_07-37-47_exp_insert_norm_fix_v2/`

## 诊断时间

2026-02-04
