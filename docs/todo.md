# 待办事项清单

## 重要提示

本文档列出了项目当前需要完成的关键任务。请按顺序完成，每个任务完成后更新状态。

---

## 任务列表

### 1. 修复转向控制问题 ⚠️ 高优先级

**任务描述**：
- 确认并修复后轮转向控制逻辑问题
- 参考诊断报告：`docs/diagnostic_reports/steering_control_analysis.md`

**问题根源**：
- USD 文件设计为后轮转向，但代码实现假设了前轮转向的行为模式
- 缺少后轮转向的运动学模型
- 导致转向角度与运动方向不匹配

**修复方案**（参考诊断报告）：
- **方案 A（推荐）**：实现后轮转向的运动学模型
  - 根据后轮转向角度计算转向半径
  - 根据转向半径调整前轮速度
  - 实现差速控制
- **方案 B**：修改 USD 文件为前轮转向
- **方案 C**：混合方案

**相关文件**：
- `IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/forklift_pallet_insert_lift/env.py` - `_apply_action()` 方法
- `IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/forklift_pallet_insert_lift/env_cfg.py` - 可能需要添加物理参数配置

**验证方法**：
- 运行诊断脚本：`scripts/diagnose_pallet_orientation.py`
- 确认转向角度与运动方向一致
- 检查转向控制是否符合物理规律

**状态**：⏳ 待开始

---

### 2. 重新训练模型

**任务描述**：
- 在修复转向控制问题后，使用修复后的环境重新训练模型
- 使用当前的 S0.2 奖励函数配置

**训练配置**：
- 任务 ID: `Isaac-Forklift-PalletInsertLift-Direct-v0`
- 奖励函数版本: S0.2
- 环境数量: 1024（根据之前的配置）

**训练命令**（参考）：
```bash
cd /home/uniubi/projects/forklift_sim/IsaacLab
nohup ./isaaclab.sh scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Forklift-PalletInsertLift-Direct-v0 \
  --num_envs 1024 \
  > ../train_reward_s0_2_fixed.log 2>&1 &
```

**预期结果**：
- 训练日志显示转向控制正常
- 策略能够正确学习转向行为
- 训练效率提升（相比之前）

**状态**：⏳ 等待任务 1 完成

---

### 3. 根据训练结果重新设计奖励函数优化

**任务描述**：
- 分析修复后的训练结果
- 根据新的训练数据重新评估奖励函数效果
- 优化奖励函数参数或设计

**分析内容**：
- 训练日志分析（`train_reward_s0_2_fixed.log`）
- 策略行为分析（视频回放）
- 奖励函数各分量效果评估
- 识别新的局部最优或训练瓶颈

**可能的优化方向**：
- 调整奖励系数（`k_align`, `k_approach`, `k_insert`, `k_lift` 等）
- 优化软门控权重（`w_ready`, `w_lift`）
- 调整时间惩罚（`rew_time_penalty`）
- 重新设计奖励结构（如果需要）

**相关文件**：
- `IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/forklift_pallet_insert_lift/env_cfg.py` - 奖励函数配置
- `IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/forklift_pallet_insert_lift/env.py` - 奖励函数实现
- `docs/rewards/reward_function_s0_2_2026-02-03_15-40-50.md` - 当前奖励函数文档

**状态**：⏳ 等待任务 2 完成

---

### 4. 实现更鲁棒的 insert_norm 计算方案（托盘坐标系）

**任务描述**：
- 当前已修复 `insert_norm` 的几何定义方向错误（最小修复方案）
- 实现更鲁棒的方案：在托盘坐标系中计算插入深度
- 支持托盘位置和朝向的随机化

**当前状态**：
- ✅ 已完成最小修复：将 `_pallet_front_x` 计算从 `+ 0.5*depth` 改为 `- 0.5*depth`
- ⏳ 待实现：托盘坐标系计算方案

**问题背景**：
- 当前实现假设托盘固定在 `(0, 0, 0)` 且朝向固定
- 如果未来需要随机化托盘位置和 yaw，当前的世界坐标系计算会失效
- 需要在托盘本地坐标系中计算插入深度

**修复方案**（更鲁棒）：

在 `_get_observations()` 和 `_get_rewards()` 中，将插入深度计算改为托盘坐标系：

```python
# 当前实现（世界坐标系，仅适用于固定托盘）
tip = self._compute_fork_tip()
insert_depth = torch.clamp(tip[:, 0] - self._pallet_front_x, min=0.0)
insert_norm = (insert_depth / (self.cfg.pallet_depth_m + 1e-6)).unsqueeze(-1)

# 改进实现（托盘坐标系，支持位置和朝向随机化）
tip = self._compute_fork_tip()  # (N, 3) in world frame
pallet_pos = self.pallet.data.root_pos_w  # (N, 3)
pallet_yaw = _quat_to_yaw(self.pallet.data.root_quat_w)  # (N,)

# 将叉尖位置转换到托盘坐标系
tip_rel = tip[:, :2] - pallet_pos[:, :2]  # (N, 2) relative position in world XY
cos_yaw = torch.cos(-pallet_yaw)  # 旋转到托盘坐标系
sin_yaw = torch.sin(-pallet_yaw)
tip_x_pallet = cos_yaw * tip_rel[:, 0] - sin_yaw * tip_rel[:, 1]  # (N,)

# 托盘近端面在托盘坐标系中是 x = -depth/2
pallet_near_face_x = -self.cfg.pallet_depth_m * 0.5
insert_depth = torch.clamp(tip_x_pallet - pallet_near_face_x, min=0.0, max=self.cfg.pallet_depth_m)
insert_norm = (insert_depth / (self.cfg.pallet_depth_m + 1e-6)).unsqueeze(-1)
```

**优势**：
- ✅ 支持托盘位置随机化（`pallet_pos` 可以是任意值）
- ✅ 支持托盘朝向随机化（`pallet_yaw` 可以是任意值）
- ✅ 几何意义更清晰：在托盘本地坐标系中计算
- ✅ 更符合物理直觉：插入深度是相对于托盘本身的

**相关文件**：
- `forklift_pallet_insert_lift_project/isaaclab_patch/source/isaaclab_tasks/isaaclab_tasks/direct/forklift_pallet_insert_lift/env.py`
  - `_get_observations()` 方法（第234-237行）
  - `_get_rewards()` 方法（第255-256行）
  - 可以移除 `__init__` 中的 `self._pallet_front_x`（不再需要）

**实施步骤**：
1. 在 `_get_observations()` 中实现托盘坐标系计算
2. 在 `_get_rewards()` 中同步更新计算逻辑
3. 移除 `__init__` 中的 `self._pallet_front_x` 初始化（可选）
4. 测试验证：确保与当前固定托盘场景结果一致
5. 未来扩展：在 `_reset_idx()` 中添加托盘位置和朝向随机化

**验证方法**：
- 运行当前训练，确认修复后的效果
- 对比最小修复方案和鲁棒方案的计算结果（固定托盘场景下应该一致）
- 如果未来添加托盘随机化，验证鲁棒方案仍然有效

**状态**：⏳ 待开始（可选改进，优先级较低）

---

## 任务依赖关系

```
任务 1 (修复转向控制)
    ↓
任务 2 (重新训练)
    ↓
任务 3 (优化奖励函数)
```

## 注意事项

1. **任务 1 是基础**：必须完成转向控制修复，否则后续训练可能无效
2. **任务 2 需要时间**：训练可能需要较长时间，建议使用 `nohup` 后台运行
3. **任务 3 需要数据**：需要等待训练完成并收集足够的数据才能进行分析

## 更新日志

- 2026-02-03: 创建待办事项清单
- 2026-02-03: 添加任务 4 - 实现更鲁棒的 insert_norm 计算方案（托盘坐标系）

---

**最后更新**: 2026-02-03
