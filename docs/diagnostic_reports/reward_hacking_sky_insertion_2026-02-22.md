# 奖励作弊漏洞（Reward Hacking）：“隔空飞越”与“空举免罚”的分析与修复

**日期：** 2026-02-22  
**问题背景：** 在解决“托盘拖地（重心异常）”问题并在 `s1.0v_fixed_physics` 版本取得高均值奖励（~167分）后，通过 `play.py` 观察模型实际行为时，发现叉车并没有真正插入托盘，而是**开局就将货叉举升至最高（2.0m），然后悬空开过托盘上方**，并且依然被系统判定为“任务成功”。

这是一个极为典型的强化学习“奖励作弊（Reward Hacking）”现象，AI 发现并利用了仿真几何计算与奖励设计中的逻辑漏洞，完全绕过了物理碰撞与对齐的困难。

---

## 一、 根因分析

通过排查 `env.py` 和 `env_cfg.py`，确认该作弊行为由两个致命漏洞组合而成：

### 漏洞 1：插入深度计算“无视 Z 轴”（Sky Insertion 漏洞）
在 `_get_observations` 和 `_get_rewards` 中计算插入深度 `insert_depth` 和归一化插入进度 `insert_norm` 时，原代码仅使用了 X/Y 平面的 2D 投影：
```python
# 旧逻辑：只取了 tip[:, :2]，丢弃了 Z 轴高度
rel_tip = tip[:, :2] - pallet_pos[:, :2]
s_tip = torch.sum(rel_tip * u_in, dim=-1)
s_front = -0.5 * self.cfg.pallet_depth_m
insert_depth = torch.clamp(s_tip - s_front, min=0.0)
```
**后果：** 只要叉车的底盘在 X/Y 平面上移动到了托盘上方，哪怕货叉举在 2 米的高空（完全没有插入托盘孔），在数学上依然会被判定为 `insert_norm = 1.0`（100% 完美插入）。这让 AI 得以完全避开复杂的货叉-托盘孔对齐和物理碰撞问题。

### 漏洞 2：空举惩罚只是“过路费”（Delta-based Penalty 漏洞）
为了防止未插入托盘前“空举货叉”，原系统设计了 `pen_premature` 惩罚：
```python
# 旧逻辑：惩罚对象是 delta_lift（每帧举升速度）
pen_premature = -self.cfg.k_pre * (1.0 - premature_fade) * torch.clamp(delta_lift, min=0.0)
```
**后果：** 惩罚的仅仅是**举升的动作/速度（delta_lift）**。且系数 `k_pre` 仅为 5.0。
这意味着如果 AI 一开局就把货叉顶到 2.0m 最高点，它总共只付出 `-5.0 * 2.0 = -10` 分的代价。一旦到达最高点不再上升（`delta_lift = 0`），此后的几百个步长里，虽然它一直“高举着空叉”，但再也不会受到任何惩罚！“一次买断，终身免罚”。

### AI 的终极作弊链路（组合技）
1. **开局即高举：** AI 无脑输出全速举升动作，花极小的代价（-10分）将货叉升到 2.0m 高空。
2. **凌空飞渡：** 带着高空货叉驶向托盘。由于货叉在天上，托盘在地上（0.15m），完全不会产生物理碰撞。
3. **“隔空插入”：** 叉车开过托盘正上方，系统误判为 `insert_norm = 1.0`。
4. **收割奖励：**
   - 对齐奖励解锁（因为 X/Y 已对齐）。
   - 由于此时货叉已经在 2.0m（`lift_height > 1.0m`），直接满足了“举升成功”条件 `lift_entry = True`。
   - 所有成功条件瞬间凑齐（`still_ok = True`），收割高额终端奖励（`r_terminal`）并结束回合。

---

## 二、 修复方案（防作弊）

为了彻底封死这两条捷径，我们实施了针对性的物理与状态约束修复：

### 1. 增加 Z 轴高度约束（Z-Gate Constraint）
在计算出 `insert_depth` 之后，引入货叉实际高度与托盘实际高度的**相对误差校验**：
```python
pallet_lift_height = pallet_pos[:, 2] - self.cfg.pallet_cfg.init_state.pos[2]
z_err = torch.abs(lift_height - pallet_lift_height)
# max_insert_z_err = 0.4m
valid_insert_z = z_err < self.cfg.max_insert_z_err
# 高度误差超过容差，判定为假插入，深度归零
insert_depth = torch.where(valid_insert_z, insert_depth, torch.zeros_like(insert_depth))
```
*注：无论是 `_get_observations` 还是 `_get_rewards` 中都补充了这一验证逻辑，并在 Config 中暴露了 `max_insert_z_err` 参数。*

### 2. 绝对高度状态惩罚（State-based Penalty）
将 `pen_premature` 从惩罚“动作增量”改为惩罚“绝对状态（高度）”，并加入 `clamp` 防止负高度变成倒扣正奖励：
```python
pen_premature = -self.cfg.k_pre * (1.0 - premature_fade) * torch.clamp(lift_height, min=0.0)
```
由于是按帧持续扣分，AI 再敢开局乱举货叉发呆，每一步都会流血不止，打消了提前举升的动机。

---

## 三、 终局策略优化（Endgame Optimizations）

在修复漏洞的同时，发现原有的终局判定过于严苛，甚至逼迫 AI“不能停下”。顺势引入了三项旨在鼓励**“到达目标后保持静止”**的终局优化：

1. **降低举升成功门槛（Lower Lift Threshold）**
   - 工业标准中托盘离开地面即算成功，无需举到半空。
   - 修改：`lift_delta_m` 从 `1.0m` 降低为 `0.3m`。

2. **豁免全局停滞惩罚（Exempt Global Stall Penalty）**
   - 根因：`pen_global_stall` 会在总势能 `phi_total` 不变时每步狠扣 `-1.5` 分。如果 AI 已经完美达成任务（`still_ok = True`），保持不动反而会被判定为“停滞”，导致 AI 不敢静止而疯狂抖动。
   - 修复：在停滞检测条件中加上豁免：
     ```python
     global_stall_cond = (
         ...
         & (~still_ok) # 已经到达终点并保持住的，绝对不扣停滞分
     )
     ```

3. **新增静止奖励（Stay-Still Reward）**
   - 为了主动引导 AI“乖乖举着不动就能稳定拿分”。
   - 修复：增加 `r_stay_still`（系数 `rew_stay_still = 0.5`）。
     ```python
     r_stay_still = self.cfg.rew_stay_still * still_ok.float()
     ```
   - 这一正向奖励加在最后的 10~30 步 hold 时间内，极大地增强了策略的稳定性。

---

## 四、 结论

通过引入 `z_err` 交叉验证与将 `delta_lift` 改为 `lift_height` 惩罚，我们成功闭环了仿真环境与奖励机制之间的语义鸿沟。这再次证明在 RL 中，**必须保证奖励公式（Reward）的几何计算与物理引擎（Physics）发生实际碰撞产生约束时，逻辑必须保持高度一致**，否则智能体必然会找到一条在数学上合法但在物理上荒谬的“白嫖”之路。