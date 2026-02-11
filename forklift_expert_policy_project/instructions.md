# forklift_expert_policy_project 说明文档

> 目标：解释这个项目的**背景**、当前遇到的**问题**、为什么要引入 **expert（专家策略）机制**、它具体做了什么、以及接下来要怎么把它接到 **BC → RL** 的训练链路里。

---

## 1. 背景：我们在做什么任务？

当前任务是一个典型的"精密位姿 + 接触插入"的连续控制问题：

* **叉车**需要在仿真（IsaacLab / Isaac Sim）中完成**叉取托盘**（fork insertion / pallet insertion + lift）
* 任务通常隐含 3 个阶段：

  1. **Docking**：到托盘正前方并对齐（横向误差、航向误差变小）
  2. **Insertion**：微调后低速插入（接触/摩擦/卡滞风险最大）
  3. **Lift**：插入到一定深度后举升

目前环境观测为 **15 维向量 obs**（并非图像），动作为 3 维连续向量（drive/steer/lift）。

---

## 2. 现在遇到了什么问题？（为什么"纯 RL 从零学"很痛）

在这个任务上，**纯 RL 从随机策略起步**容易遇到几类"工程级难题"：

### 2.1 成功信号稀疏 / 成功阈值不可达

* "成功"往往要求横向误差、航向误差、插入深度同时达标，阈值又很严
* 随机探索几乎碰不到成功 → `frac_success` 长期为 0 或极低
* 没有稳定的正反馈，PPO 很容易只学到"局部刷分行为"（例如怼、蹭、抖）

### 2.2 阶段链路很长：先对齐，再插入，再举升

* 任何一个阶段卡住都会导致后续阶段完全学不到
* 最常见的卡点：**对齐阶段不够好**，导致插入阶段永远触发不了有效学习

### 2.3 接触导致动力学变复杂：打滑 / 卡滞 / 反复顶

* 插入阶段一旦接触，摩擦、碰撞、约束会使系统变得很"非线性 + 不稳定"
* RL 会出现：斜着硬怼、蛇形打舵、靠噪声探索把精细控制淹没等

### 2.4 训练成本高：大量步数消耗在"学会基本开车"

* 你真正关心的难点是插入/微调，但 RL 很多算力先花在"学会接近 + 对齐"的基础技能上

---

## 3. 为什么要引入专家机制（Expert / Demonstration）？

引入 expert 的动机是非常工程化的：
**先让系统有一个"像样能用"的驾驶行为作为起点，再用 RL 去精雕细琢。**

核心收益：

1. **减少探索难度**：不从随机策略起步，避免"黑屋摸钥匙孔"
2. **把基础技能外包给监督学习（BC）**：让策略先学会"怎么对齐、怎么减速、怎么不乱抖"
3. **更稳定的训练起点**：BC 预训练后的 policy 初始化能显著提升早期成功概率
4. **可作为 baseline / 调试工具**：如果 expert 都跑不动，通常说明 obs 映射/符号/单位有问题（比 RL 更容易暴露）
5. **天然支持 curriculum**：先只做 Docking 示范 → 再扩展到 Insertion/Lift

> 一句话：**expert 是"老师示范" + "训练轮子"**。最终目标不是用它上线，而是用它把 RL 拉到正确轨道上。

---

## 4. 这个项目具体做了什么？

这个项目交付的是一个"路线 A"最小闭环：

> **规则专家（expert）自动开叉车 → 采集大量示范（demos）→ 可选 BC 预训练 → 再接 RL 微调**

### 4.1 项目文件结构

* `forklift_expert/expert_policy.py`
  规则专家策略：输入 obs（15D），通过 `_decode_obs()` 解码，输出 action（3D）。
* `forklift_expert/obs_spec.json`
  **obs 映射表**：15 维 obs 各字段名 → index 的映射（已按实际 env 填好）。
* `forklift_expert/action_spec.json`
  action 定义（drive/steer/lift 三维）及 clip 范围。
* `scripts/collect_demos.py`
  运行 expert 采集示范数据，输出 `data/demos_*.npz`。已适配 IsaacLab 向量环境。
* `scripts/analyze_demos.py`
  统计 demos 基本分布（均值、方差、done 比例、episode 数等）。
* `scripts/bc_train.py`
  一个最小版 BC（MLP 回归动作）训练脚本（可选，用来快速验证"示范可学"）。

---

## 5. Expert 策略是怎么工作的？（核心逻辑）

这个 expert 是一个**三阶段规则控制器**：Docking → Insertion → Lift。

### 5.1 输入：15 维 obs（已适配实际 env）

15 维观测向量来自 `env._get_observations()`，expert 通过 `_decode_obs()` 解码为语义字段：

| 维度 | 字段 | 含义 |
|------|------|------|
| 0-1 | `d_xy_r` | 机器人→托盘中心的 2D 相对位置（robot frame, 米）|
| 2-3 | `cos_dyaw`, `sin_dyaw` | 偏航差的三角函数编码 |
| 4-5 | `v_xy_r` | 机器人线速度（robot frame, m/s）|
| 6 | `yaw_rate` | 偏航角速度（rad/s）|
| 7-8 | `lift_pos`, `lift_vel` | lift 关节位置/速度 |
| 9 | `insert_norm` | 插入深度归一化（0~1）|
| 10-12 | `prev actions` | 上一步动作（drive, steer, lift）|
| 13 | `y_err_obs` | 横向误差（pallet center-line frame, 归一化 ÷0.5m, clip [-1,1]）|
| 14 | `yaw_err_obs` | 偏航误差（pallet center-line frame, 归一化 ÷15°, clip [-1,1]）|

expert 内部的 `_decode_obs()` 会做以下派生计算：

* `dist_front` = `d_xy_r[0] - pallet_half_depth`（到托盘前端开口的前向距离）
* `lateral_err` = `y_err_obs × 0.5`（反归一化为米）
* `yaw_err` = `atan2(sin_dyaw, cos_dyaw)`（完整弧度 [-π, π]）

> 注意：`contact_flag` / `slip_flag` 在 15 维 obs 中**不存在**，倒车重试功能默认禁用。

### 5.2 输出：动作向量（3 维）

* `drive`：驱动/车轮角速度（-1~1, 归一化）
* `steer`：转向/前轮转角（-1~1, 归一化）
* `lift`：举升位置增量（-1~1, 归一化）

### 5.3 阶段判定（Stage）

* **Lift**：`insert_norm >= 0.75` → `lift=0.60`，停止前进
* **Insertion**：`insert_norm >= 0.15` → 进入插入阶段（更严格门控）
* **Docking**：否则就是对齐 + 接近

> 这里用 `insert_norm` 切阶段，是因为它通常是最稳定的"进度条"。

### 5.4 转向控制（所有阶段共享）

使用一个简单的几何组合（并带死区、限幅、速率限制防抖）：

```
steer = k_lat * lateral_err + k_yaw * yaw_err
```

并且：

* `deadband_steer`：小于阈值的转向直接归零（防抖）
* `steer_rate_limit`：每步转向变化不超过固定值（防蛇形抖动）

### 5.5 Docking：对齐 + 接近（越近越慢、偏差大就慢）

* 速度基本与距离成比例：`v ~ k_dist * dist`
* 近距离（`dist < slow_dist = 1.5m`）进一步减速
* 横向/航向偏差越大，速度越小（给转向留时间）

目的：看起来像"老司机"——先把方向修正，再靠近，不会斜着猛冲。

### 5.6 Insertion：低速插入（严格门控，防硬怼）

Insertion 阶段只在"对齐达标"时才给前进：

* `|lateral| <= 0.03 m`
* `|yaw| <= 3°`

达标 → 低速前进（`ins_v_min ~ ins_v_max`）
不达标 → `drive=0`，只打方向修正

目的：禁止"斜着硬顶进去"导致打滑卡死。

### 5.7 接触/打滑处理（当前禁用）

当前 15 维 obs 不包含 `contact_flag` / `slip_flag`，所以 `backoff_on_contact` 默认为 `False`。
如果未来在 obs 中添加了接触/滑移标志（扩展为 17 维），可以启用此功能：

* 在 insertion 阶段一旦检测到接触/滑移，会触发固定步数倒车重试
* `backoff_throttle = -0.20`、`backoff_steps = 6`

---

## 6. `obs_spec.json` 映射层（已适配完成）

`obs_spec.json` 已按实际 `env._get_observations()` 的 15 维输出填好了所有字段映射（index 0-14）。

expert 的 `_decode_obs()` 方法通过 `obs_spec.json` 中的 index 读取原始值，再做必要的转换：

* cos/sin → `atan2` 恢复完整弧度
* `y_err_obs` × 0.5 反归一化为米
* `d_xy_r[0]` - `pallet_half_depth` 得到到托盘前端的距离

如果 env 的观测顺序在未来版本中发生变化，只需更新 `obs_spec.json` 中的 index，不需要改 expert 代码。

---

## 7. expert 做完之后，后面要做什么？（完整训练链路）

下面是推荐的"工程闭环"路线（最容易成功）：

### Step 0：obs 映射（已完成）

`obs_spec.json` 已按 `env._get_observations()` 的实际字段填好。`_decode_obs()` 会自动处理 cos/sin → atan2、y_err_obs → 米等转换。

---

### Step 1：跑 expert 采集 demos（先从 Docking 开始）

用 `scripts/collect_demos.py` 自动跑 N 局并保存数据：

```bash
./isaaclab.sh -p /ABS/PATH/TO/scripts/collect_demos.py \
  --task Isaac-Forklift-PalletInsertLift-Direct-v0 \
  --num_envs 64 \
  --episodes 3000 \
  --headless
```

输出：`data/demos_YYYYMMDD_HHMMSS.npz`
包含：

* `obs`：每步观测（N, obs_dim）
* `act`：每步动作（N, act_dim）
* `done`：每步是否结束
* `episode_id` / `env_id`
* `meta`：json 字符串，记录任务/维度/参数等

> `collect_demos.py` 已适配 IsaacLab：自动处理 dict obs（`{"policy": tensor}`）、torch tensor 转换、向量环境 auto-reset。每个 env 有独立的 expert 实例（不共享 rate-limit 状态）。

建议策略：

* **先只做 Docking demos**（不追求插入/举升），把对齐行为打稳
* 之后再扩展 Insertion/Lift

---

### Step 2：快速 sanity check（示范是不是靠谱）

用 `scripts/analyze_demos.py` 看分布是否正常：

```bash
python scripts/analyze_demos.py data/demos_xxx.npz
```

关注：

* obs/act 均值方差是否离谱
* done 比例是否合理
* episode 数是否按预期增长

---

### Step 3（可选但推荐）：BC 预训练（让策略"先像专家"）

用 `scripts/bc_train.py` 训练一个 MLP 回归动作：

```bash
python scripts/bc_train.py --demos data/demos_xxx.npz --out data/bc_actor.pt
```

这一步不是最终结构，只是为了：

* 验证"示范是可学习的"
* 得到一个可用于 RL 初始化的策略权重（或作为参考）

---

### Step 4：接 RL 微调（让策略超越专家）

推荐做法（按改动成本从低到高）：

**方案 A：RL 直接加载 BC actor 作为初始化**

* PPO/SAC 开训前，actor 权重从 `bc_actor.pt` 初始化
* 训练会更快进入"非零成功率"区域

**方案 B：RL 训练时加一个衰减的 imitation loss（更稳）**

* 前期 `λ` 大：防止策略学坏
* 后期 `λ` 衰减：让 RL 超越 expert

**方案 C：离策略（SAC/TD3）把 demos 放进 replay buffer**

* demo 作为经验回放的一部分，前期占比高，后期降低

---

## 8. 这个项目的定位边界（别误解它能"一步到位"）

* 这不是一个"完美 autopilot"，它是一个**稳定可重复的示范生成器**
* expert 的目标是：

  * **产出"像样能用"的 Docking/Insertion 基本行为**
  * 帮 BC 把 policy 拉到合理初始化点
* 真正的高精度、强鲁棒性、复杂接触微操，仍然要靠后续 RL 训练去学

---

## 9. 行动清单

1. 单 env 非 headless 跑 5 个 episode 目视检查 expert 行为（验证 `_decode_obs()` 和阈值是否合理）
2. 批量采集 3k episodes（先 Docking）
3. 跑 `analyze_demos.py` 看分布
4. 跑最小 BC 看能否拟合（loss 是否正常下降）
5. 把 BC 权重接入 RL（PPO actor init），并逐步收紧成功阈值做 curriculum
6. （可选）给 collect_demos 加 debug 曲线保存（dist/lat/yaw/insert_norm/steer/drive 的 time series），用于可视化诊断

### 已知限制

* `y_err_obs` 被 clip 到 [-1,1]（对应 ±0.5m），Docking 远距离时超过 0.5m 的横向偏移信息会丢失
* `yaw_err_obs` 被 clip 到 [-1,1]（对应 ±15°），但 expert 使用 `atan2(sin_dyaw, cos_dyaw)` 恢复完整角度，不受此限制
* `contact_flag` / `slip_flag` 不在 obs 中，倒车重试功能默认禁用
