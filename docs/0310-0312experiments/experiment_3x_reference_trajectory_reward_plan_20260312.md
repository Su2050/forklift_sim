# 实验 3.x：参考轨迹引导 Reward 重构方案 (2026-03-12)

## 1. 文档目的

本文用于把“参考论文中的参考轨迹 / clothoid 近似路线”这个思路，整理成**适合当前项目环境**的一套可执行方案。

目标不是一次性重写全部 Reward，而是：

1. 在当前主线基座上，补上一条“**可插入的接近路径先验**”
2. 把 Reward 明确拆成几个彼此可单独验证的模块
3. 形成 `实验 3.1 ~ 3.4` 的单因素消融路线

---

## 2. 当前基线与问题定位

当前应固定不动的主线基座：

- `approach-only`，只输出 `drive + steer`
- `RRL` 范式：`ResNet18 + ImageNet pretrained + 全程冻结`
- 当前单相机 `60°` 俯视视角
- 成功评估以 `push_free` 相关指标为主，而不是只看 `success_rate_total`

结合：

- `docs/0310-0312experiments/system_analysis_and_iterative_plan_20260311.md`
- `docs/0310-0312experiments/experiment_log_20260312_morning.md`
- `logs/20260312_084658_train_exp3_reward_shaping_rrl.log`

当前 Reward 的早期表现可以概括为：

- `phase/frac_aligned` 已能到 `~0.53`
- `err/lateral_mean` 约 `0.15m`
- `err/yaw_deg_mean` 约 `5°`
- 但 `phase/frac_inserted = 0`
- `push_free_success_rate_total = 0`
- `stage_dist_front_mean` 仍在 `3.4m ~ 3.6m`
- `milestone/hit_approach = 0`

这说明当前策略出现了一个很明确的局部最优：

**它学会了“远处对齐”，但没有学会“沿着一条可插入的路径向前 commit”。**

换句话说，当前 Reward 的主要问题不是“完全没有对齐信号”，而是：

1. 远场缺少一条明确的、单调的“走哪里”引导
2. 近场缺少“现在可以放心往前顶一点”的 commit 信号
3. 防推盘逻辑和死区逻辑虽然存在，但还没有和“可插入路径”耦合起来

---

## 3. 核心假设 H3.x

### H3.x

如果在当前环境中引入一条**每回合基于起点与托盘姿态构造的参考接近轨迹**，并将 Reward 拆成：

- 远场轨迹走廊引导
- 近场 commit 奖励
- 条件化推盘惩罚
- 死区撤退 / 重试奖励

那么智能体会更容易学会：

`接近 -> 进入走廊 -> 对准 -> commit -> 插入`

而不是停留在：

`远处对齐 -> 不敢前插 / 盲目前顶`

### 这条假设与论文的关系

论文真正值得借鉴的，不是“必须精确复现 clothoid 数学形式”，而是：

- 给叉车 approach 阶段一个**几何上合理的参考路线**
- Reward 不只奖励“离目标近”，还奖励“以正确姿态沿合理路径接近”

对我们当前环境，**第一版不需要强求严格 clothoid**。  
更务实的做法是先实现一个 **clothoid-lite / trajectory-lite** 版本，只验证“路径先验本身是否有用”。

---

## 4. 设计原则

本轮 `3.x` 方案必须遵守以下原则：

### 4.1 只动 Reward，不动主线架构

本轮不改：

- backbone
- 相机方案
- action 维度
- success 判据主逻辑

避免再次把问题混成“到底是视觉、动作空间、视角还是奖励”。

### 4.2 参考轨迹只服务于 approach，不替代 success

参考轨迹只是一个**引导先验**，不能替代真正成功定义。  
真正成功仍然必须由：

- `insert`
- `align`
- `hold`
- `push_free`

共同决定。

### 4.3 尽量沿用当前 env 的几何量

当前环境里已经有很多高质量几何量，不要另起炉灶：

- `y_err`
- `tip_y_err`
- `yaw_err_deg`
- `stage_dist_front`
- `dist_front`
- `insert_depth`
- `insert_norm`
- `pallet_disp_xy`

参考轨迹方案应尽量基于这些已有量搭建，而不是引入另一套完全平行的几何定义。

### 4.4 奖励尽量平滑、连续、可诊断

优先使用：

- potential difference
- 平滑 gate
- 限幅后的 delta reward

避免：

- 生硬 if-else
- 一步触发过大脉冲奖励
- 远场几乎无梯度、近场突然奖励爆炸

### 4.5 推盘惩罚不能再“过早归零”

当前项目已经反复证明：

- 惩罚太强会变成“不敢插”
- 惩罚太弱或过早关闭会变成“推土机”

因此新的推盘惩罚必须是**条件化衰减**，不是“插入超过某阈值后直接归零”。

---

## 5. 当前环境里可直接复用的状态量

在 `env.py` 的 `_get_rewards()` 中，当前已经有以下可直接复用的量：

- `root_pos[:, :2]`：车体 base 平面位置
- `tip[:, :2]`：叉齿尖端平面位置
- `pallet_pos[:, :2]`：托盘中心位置
- `robot_yaw`
- `pallet_yaw`
- `u_in`：托盘插入方向单位向量
- `v_lat`：托盘横向单位向量
- `y_err`
- `tip_y_err`
- `yaw_err_deg`
- `stage_dist_front`
- `dist_front`
- `insert_depth`
- `insert_norm`
- `pallet_disp_xy`

因此，`3.x` 不需要额外改 observation。  
第一版只需要：

1. 在 reset 时缓存初始位姿
2. 生成每个 env 的参考轨迹
3. 每步计算当前点到轨迹的最近点、切线方向与轨迹进度

---

## 6. 参考轨迹的建议实现：clothoid-lite，而不是严格 clothoid

## 6.1 为什么不建议第一版直接上严格 clothoid

严格 clothoid 的问题不是不能做，而是：

- 首版实现复杂度偏高
- 调参成本高
- 一旦无效，很难判断是“轨迹先验没用”还是“clothoid 实现有问题”

所以第一版建议先做一个**近似版参考轨迹**，只验证路径先验有没有帮助。

## 6.2 建议的轨迹形式

建议使用：

1. **一段三次 Bézier 曲线**：从当前回合起点连接到“预对位点”
2. **一段直线**：从预对位点沿托盘中心线进入托盘前沿

这样可以同时满足：

- 起点方向连续
- 终点方向与托盘插入方向一致
- 近场仍然回到“沿托盘中心线插入”的清晰几何

## 6.3 几何定义

设：

- `p0`：reset 时车体 base 的 2D 位置
- `h0`：reset 时车头朝向单位向量
- `pallet_xy`：托盘中心 2D 位置
- `u_in`：托盘插入方向单位向量
- `s_front = -0.5 * pallet_depth`

定义：

- `p_goal = pallet_xy + s_front * u_in`
- `p_pre = pallet_xy + (s_front - d_pre) * u_in`

其中：

- `p_goal`：托盘前沿中心点
- `p_pre`：进入托盘前的一点“预对位点”
- `d_pre`：建议起始值 `1.0m ~ 1.3m`

三次 Bézier 控制点建议为：

```text
B0 = p0
B1 = p0 + l0 * h0
B2 = p_pre - l1 * u_in
B3 = p_pre
```

其中：

- `l0`：起点切线长度，建议 `0.6m ~ 1.0m`
- `l1`：终点切线长度，建议 `0.8m ~ 1.2m`

然后再接：

```text
p_pre -> p_goal
```

这条直线段就是近场插入参考中心线。

## 6.4 第一版实现建议

不要每步做解析最短距离求解。  
建议在 reset 时把轨迹离散成 `16 ~ 32` 个点，缓存成 polyline：

1. 轨迹点 `traj_pts`
2. 相邻点切线 `traj_tangents`
3. 归一化累计弧长 `traj_s_norm`

每步只需：

1. 找当前位置到 polyline 最近点
2. 读取对应切线方向
3. 得到 `d_traj`、`yaw_traj_err_deg`、`s_traj_norm`

这样实现最稳妥，也最好调试。

---

## 7. Reward 拆分总览

我建议把 `实验 3.x` 的 Reward 拆成 5 层：

### 7.1 保留层：继续沿用

这些先保留：

- `phi_ins`
- `phi_lift`
- `pen_premature`
- `pen_dense`
- `r_terminal`
- `hold counter`
- `tip alignment` 成功门控
- `push_free` KPI 统计

### 7.2 替换层：远场接近

把当前偏“距离带”的 `phi1`，替换成“**参考轨迹走廊 + 轨迹进度**”。

### 7.3 新增层：近场 commit

在“已经基本进走廊且姿态可接受”时，额外奖励：

- `dist_front` 下降
- `insert_norm` 上升

解决“看见了但不敢往前插”的问题。

### 7.4 重构层：条件化防推盘

根据是否在可插入走廊里，动态调整推盘惩罚，而不是简单按 `insert_norm` 粗暴开关。

### 7.5 修复层：死区撤退 / 重试

当出现“浅插错位 + 卡住”时，允许并鼓励有控制地后退重试，而不是持续顶死。

---

## 8. 具体 Reward 项怎么拆

## 8.1 远场参考轨迹走廊势函数 `phi_traj`

### 目的

解决当前策略“远处对齐但不向前接近”的问题。

### 每步新增几何量

- `d_traj`：当前 `base` 到参考轨迹最近点的法向距离
- `yaw_traj_err_deg`：车头朝向与该最近点切线方向的误差
- `s_traj_norm`：当前位置在参考轨迹上的归一化进度，范围 `[0, 1]`

### 推荐形式

```text
phi_traj_center =
    exp(-(d_traj / sigma_traj_d)^2)
  * exp(-(yaw_traj_err_deg / sigma_traj_yaw)^2)

phi_traj_progress =
    s_traj_norm * phi_traj_center

phi_traj =
    k_traj_center * phi_traj_center
  + k_traj_progress * phi_traj_progress
```

对应 shaping：

```text
r_traj = gamma * phi_traj_t - phi_traj_{t-1}
```

### 解释

- `phi_traj_center`：鼓励进入轨迹走廊并对准切线方向
- `phi_traj_progress`：鼓励沿走廊往前，而不是只在某个角度上站住

### 为什么它比当前 `phi1` 更适合

当前 `phi1` 更像“进一个距离带就行”。  
参考轨迹方案更像：

- 先进入一条可插入的走廊
- 再沿走廊稳定推进

这更符合叉车任务，而不是普通点机器人导航。

---

## 8.2 近场 commit 奖励 `r_commit`

### 目的

解决“已经基本对准，但不敢往前插”的问题。

### 激活门控

只有当下面条件逐渐满足时才打开：

```text
gate_commit =
    smoothstep((d_commit_open - stage_dist_front) / d_commit_open)
  * exp(-(tip_y_err / sigma_commit_tip)^2)
  * exp(-(yaw_err_deg / sigma_commit_yaw)^2)
```

建议首版：

- `d_commit_open = 1.0m`
- `sigma_commit_tip = 0.12m`
- `sigma_commit_yaw = 8°`

### 奖励形式

建议直接用**限幅 delta 奖励**，不要做太复杂：

```text
r_commit_front =
    k_commit_front
  * clip(prev_dist_front - dist_front, 0, delta_front_clip)
  * gate_commit

r_commit_insert =
    k_commit_insert
  * clip(insert_norm - prev_insert_norm, 0, delta_insert_clip)
  * gate_commit
```

建议首版：

- `delta_front_clip = 0.05`
- `delta_insert_clip = 0.03`

### 解释

这两项分别鼓励：

- 已经基本对准后继续向前
- 一旦开始浅插，继续把插入深度做出来

### 与现有 `phi_ins` 的关系

两者不是重复关系：

- `phi_ins`：更像“你已经开始插了，我给你持续势函数”
- `r_commit_insert`：更像“你不要在门口停住，赶紧把浅插变成稳定插入”

---

## 8.3 条件化推盘惩罚 `pen_push_cond`

### 目的

解决“不是不敢插，就是推着走”的两难。

### 问题点

推盘惩罚如果只按 `insert_norm` 衰减，会有两个风险：

1. 还没真正对准，只是碰到了边缘，就开始减罚
2. 近场浅插时仍可能靠“硬顶”获得更高回报

### 建议重构

推盘惩罚不再只看 `insert_norm`，而是看“是否处于可插入状态”：

```text
w_push_relax = gate_commit

k_push_eff =
    k_push_far * (1 - w_push_relax)
  + k_push_near * w_push_relax
  + k_push_deadzone_bonus * dead_zone

pen_push_cond = -k_push_eff * push_excess
```

其中建议满足：

- `k_push_far > k_push_near > 0`
- 即使在近场也**不允许降到 0**
- 进入死区时惩罚重新加重

### 建议起始值

- `k_push_far = 1.0`
- `k_push_near = 0.30 ~ 0.40`
- `k_push_deadzone_bonus = 0.6`
- `pallet_push_deadband_m = 0.05 ~ 0.06`

### 解释

这等价于：

- 远场乱撞：重罚
- 近场已进可插入走廊：允许非常轻微试探，但仍保留非零代价
- 深插错位卡死：再次重罚

---

## 8.4 死区撤退 / 重试奖励 `r_escape`

### 目的

解决“半插不插、横偏过大、持续顶死”的情况。

### 死区定义建议

不要只看 `insert_norm` 和 `y_err`，建议把 `tip_y_err` 也纳入：

```text
dead_zone =
    (insert_norm > dead_zone_insert_thresh)
  & (
        (y_err > dead_zone_lat_thresh)
      | (tip_y_err > dead_zone_tip_thresh)
    )
```

### 奖励形式

```text
r_escape =
    k_escape
  * clip(prev_insert_norm - insert_norm, 0, delta_escape_clip)
  * prev_dead_zone
```

即：

- 只有上一时刻已经在死区
- 当前真的退出了一点点
- 才给奖励

### 解释

它不是鼓励“后退”，而是鼓励“**从错位深插中退出**”，给策略一个重试通道。

### 建议起始值

- `dead_zone_insert_thresh = 0.20`
- `dead_zone_lat_thresh = 0.16`
- `dead_zone_tip_thresh = 0.12`
- `k_escape = 2.0`
- `delta_escape_clip = 0.04`

---

## 8.5 继续保留但建议弱化的项

### `pen_global_stall`

当前早期日志里，这项负值较大，容易在 agent 还没学会接近时就形成“常驻背景惩罚”。

建议：

- 先保留
- 但推迟触发时间
- 或降低绝对值

建议首版：

- `global_stall_steps: 120 -> 240`
- `rew_global_stall: -1.5 -> -0.5`

### `phi1`

在上轨迹走廊后，建议：

- `phi1` 直接置弱甚至置零
- 避免“距离带”和“轨迹走廊”两套远场信号彼此冲突

我更建议：

- `实验 3.1` 中直接让 `phi_traj` 替代 `phi1`

而不是把两者同时开大。

---

## 9. 建议新增的诊断项

如果做 `3.x`，日志里必须新增下面这些量，否则会很难判断到底哪里失效。

### 9.1 轨迹几何诊断

- `traj/d_traj_mean`
- `traj/d_traj_p95`
- `traj/yaw_traj_deg_mean`
- `traj/s_traj_norm_mean`
- `traj/corridor_frac`
- `traj/pre_point_reached_frac`

### 9.2 commit 诊断

- `traj/commit_gate_mean`
- `traj/commit_gate_p95`
- `reward/r_commit_front`
- `reward/r_commit_insert`

### 9.3 防推 / 死区诊断

- `reward/pen_push_cond`
- `reward/r_escape`
- `diag/dead_zone_frac`
- `diag/dead_zone_escape_frac`

### 9.4 仍然必须保留的主 KPI

- `episode/push_free_success_rate_total`
- `episode/push_free_insert_rate_total`
- `diag/pallet_disp_xy_mean`
- `diag/pallet_disp_xy_p95`
- `phase/frac_inserted`
- `phase/hold_counter_max`
- `err/lateral_near_success`
- `err/yaw_deg_near_success`

---

## 10. 建议的实验顺序：3.1 到 3.4

不要一次把全部东西都打开。  
建议严格按下面顺序做单因素实验。

---

## 11. 实验 3.1：参考轨迹走廊替代远场距离带

### 假设

当前 Reward 的远场引导太弱或太模糊，导致策略学会“远处对齐”而不是“沿正确路径接近”。  
如果用参考轨迹走廊替代 `phi1`，`approach` 会明显变得更主动。

### 只改一个因素

- 新增 `phi_traj`
- 用 `phi_traj` 替代当前 `phi1`
- 其余全部保持不变

### Reward 改动

```text
phi_total =
    phi_traj
  + phi2
  + phi_ins
  + phi_lift
```

建议：

- `k_phi1 = 0`
- 保留 `phi2 / phi_ins / phi_lift`

### 保持不变

- `RRL` 主线
- 单相机
- `approach-only`
- `push_free` 评估
- 当前 success / hold 逻辑

### 关键指标

- `traj/corridor_frac`
- `traj/s_traj_norm_mean`
- `milestone/hit_approach`
- `err/stage_dist_front_mean`
- `phase/frac_aligned`

### 成功判据

至少满足以下 3 条中的 2 条：

1. `milestone/hit_approach` 明显大于当前基线
2. `stage_dist_front_mean` 在前 `100 ~ 150 iter` 内显著下降
3. `phase/frac_aligned` 不低于当前 baseline

### 失败解释

如果 `3.1` 失败，优先怀疑：

1. 轨迹几何定义有问题
2. 轨迹离散/最近点匹配不稳定
3. 轨迹走廊太窄，导致远场几乎拿不到正信号

---

## 12. 实验 3.2：在走廊内增加近场 commit 奖励

### 假设

只靠轨迹走廊可以把车带到门口，但仍不足以让策略果断插入。  
如果在“已进入可插入状态”时显式奖励 `dist_front` 下降和 `insert_norm` 上升，`frac_inserted` 会突破 0。

### 只改一个因素

在 `3.1` 最佳配置基础上，只新增：

- `r_commit_front`
- `r_commit_insert`

### Reward 改动

```text
rew =
    rew_base_from_3_1
  + r_commit_front
  + r_commit_insert
```

### 保持不变

- 轨迹几何
- success 逻辑
- push penalty 逻辑
- dead-zone 逻辑

### 关键指标

- `phase/frac_inserted`
- `episode/push_free_insert_rate_total`
- `traj/commit_gate_mean`
- `reward/r_commit_front`
- `reward/r_commit_insert`

### 成功判据

至少出现下面 2 个变化：

1. `phase/frac_inserted` 从 0 明显抬升
2. `push_free_insert_rate_total` 破零
3. 视频里能看到“对准后继续前插”，而不是门口停住

### 失败解释

如果 `3.2` 失败，优先怀疑：

1. `gate_commit` 太苛刻，实际上很少打开
2. `r_commit_front` 太小，不足以压过时间惩罚与停滞惩罚
3. `phi_ins` 启动太晚，导致 commit 与插入势函数之间有空档

---

## 13. 实验 3.3：把推盘惩罚改成“条件化非零惩罚”

### 假设

当 `3.2` 开始推动插入后，系统很可能再次回到“推着托盘走”的老问题。  
如果把推盘惩罚改成“远场重、近场轻、死区再加重、但永远非零”，可以兼顾探索与防 bulldozer。

### 只改一个因素

只替换 `pen_pallet_push` 为 `pen_push_cond`。

### Reward 改动

```text
pen_push_cond = -k_push_eff * push_excess
```

其中：

```text
k_push_eff =
    k_push_far * (1 - gate_commit)
  + k_push_near * gate_commit
  + k_push_deadzone_bonus * dead_zone
```

### 保持不变

- 轨迹走廊
- commit 奖励
- 其余终局与 hold 逻辑

### 关键指标

- `episode/push_free_insert_rate_total`
- `episode/push_free_success_rate_total`
- `diag/pallet_disp_xy_mean`
- `diag/pallet_disp_xy_p95`
- `reward/pen_push_cond`

### 成功判据

满足：

1. `frac_inserted` 不显著下降
2. `pallet_disp_xy_mean/p95` 明显受控
3. `push_free_insert_rate_total` 和 `push_free_success_rate_total` 开始上升

### 失败解释

如果 `3.3` 失败：

- 插入掉光：说明惩罚仍然太重
- 插入有了但托盘乱飞：说明近场 relax 太宽

---

## 14. 实验 3.4：死区撤退 / 重试机制

### 假设

即使 `3.1 ~ 3.3` 生效，策略仍可能卡在“浅插错位 -> 顶住 -> 既不前进也不后退”的死区。  
如果显式奖励从死区退出并重试，最终 hold 成功率会提高。

### 只改一个因素

只新增：

- `r_escape`

并适度放缓：

- `pen_global_stall`

### Reward 改动

```text
r_escape =
    k_escape
  * clip(prev_insert_norm - insert_norm, 0, delta_escape_clip)
  * prev_dead_zone
```

### 保持不变

- 轨迹走廊
- commit 奖励
- 条件化 push penalty

### 关键指标

- `diag/dead_zone_frac`
- `diag/dead_zone_escape_frac`
- `phase/hold_counter_max`
- `episode/push_free_success_rate_total`

### 成功判据

1. 死区卡死比例下降
2. 视频中开始出现“退一点再对准再插”的行为
3. `hold_counter_max` 与最终 `push_free_success` 继续上升

### 失败解释

如果 `3.4` 失败，说明：

- 当前最核心瓶颈可能已不是 Reward，而是视角 / 观测几何
- 或 PPO 探索参数需要同步调整

---

## 15. 推荐的初始参数表（仅作起始值）

下面是一组**建议起始值**，不是最终值：

| 参数 | 建议值 | 用途 |
| :--- | :--- | :--- |
| `traj_pre_dist_m` | `1.2` | 预对位点距离 |
| `traj_ctrl_start_m` | `0.8` | Bézier 起点切线长度 |
| `traj_ctrl_goal_m` | `1.0` | Bézier 终点切线长度 |
| `traj_num_samples` | `21` | 轨迹离散点数 |
| `sigma_traj_d` | `0.35` | 轨迹走廊宽度 |
| `sigma_traj_yaw_deg` | `15.0` | 轨迹切线偏航尺度 |
| `k_traj_center` | `4.0` | 走廊居中奖励强度 |
| `k_traj_progress` | `6.0` | 沿轨迹推进奖励强度 |
| `d_commit_open` | `1.0` | commit 门控打开距离 |
| `sigma_commit_tip` | `0.12` | commit 时 tip 横向尺度 |
| `sigma_commit_yaw_deg` | `8.0` | commit 时偏航尺度 |
| `k_commit_front` | `4.0` | 近场前插奖励 |
| `k_commit_insert` | `8.0` | 插入增量奖励 |
| `k_push_far` | `1.0` | 远场推盘惩罚 |
| `k_push_near` | `0.35` | 近场推盘惩罚 |
| `k_push_deadzone_bonus` | `0.6` | 死区加重项 |
| `pallet_push_deadband_m` | `0.06` | 推盘死区 |
| `dead_zone_insert_thresh` | `0.20` | 深插死区阈值 |
| `dead_zone_lat_thresh` | `0.16` | 横偏死区阈值 |
| `dead_zone_tip_thresh` | `0.12` | tip 横偏死区阈值 |
| `k_escape` | `2.0` | 死区撤退奖励 |
| `delta_escape_clip` | `0.04` | 撤退奖励限幅 |
| `global_stall_steps` | `240` | 延后停滞惩罚触发 |
| `rew_global_stall` | `-0.5` | 降低停滞惩罚强度 |

---

## 16. 实现落点建议

## 16.1 `env_cfg.py`

建议新增一组清晰参数前缀，避免和现有参数混在一起：

- `traj_*`
- `commit_*`
- `push_*`
- `escape_*`

## 16.2 `env.py`

建议只改 3 个位置：

1. `_reset_idx()`
   - 缓存每个 env 的初始 base 位姿
   - 生成参考轨迹 polyline
2. 新增 helper
   - `_build_reference_trajectory()`
   - `_query_reference_trajectory()`
3. `_get_rewards()`
   - 加入 `phi_traj`
   - 加入 `r_commit_*`
   - 重构 `pen_pallet_push`
   - 加入 `r_escape`

## 16.3 新增缓存变量

建议新增：

- `_traj_pts`
- `_traj_tangents`
- `_traj_s_norm`
- `_prev_phi_traj`
- `_prev_dist_front`
- `_prev_insert_norm`
- `_prev_dead_zone`

其中：

- `_prev_insert_norm` 目前已有，可直接复用
- 新增缓存应尽量只服务于 Reward，不污染 observation

---

## 17. 决策规则

做完 `3.1 ~ 3.4` 后，按下面逻辑决策：

### 情况 A：`3.1` 就明显有效

说明论文的“参考轨迹先验”在你们环境里是成立的。  
后续可以继续打磨轨迹形式，不急着换视角。

### 情况 B：`3.1` 无效，但 `3.2` 有效

说明核心不是“轨迹本身”，而是“近场 commit 信号不够”。  
后续应优先继续做 commit / insert shaping。

### 情况 C：`3.2` 有效但 `3.3` 一上就坏

说明当前真正难点是“探索与防推”的平衡。  
后续重点应在条件化 push penalty，而不是更复杂路径。

### 情况 D：`3.1 ~ 3.4` 都无明显帮助

那就应提高对“相机几何才是主因”的怀疑度。  
此时才更有理由进入相机视角 ablation，而不是继续在 Reward 上微调。

---

## 18. 一句话总结

`实验 3.x` 的核心不是“用数学曲线替代 RL”，而是：

**给当前已经具备基本对齐能力的视觉策略，再补上一条“可插入接近路径”的 Reward 先验，并把“前插 commit、推盘约束、死区撤退”拆成独立可验证的模块。**

如果这条路线有效，它最可能先带来的不是最终成功率瞬间暴涨，而是：

1. `approach` 率先恢复
2. `frac_inserted` 先破零
3. 然后才是 `push_free_success` 往上走

这才是最符合当前系统状态的推进顺序。
