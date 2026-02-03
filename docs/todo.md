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

---

**最后更新**: 2026-02-03
