# Success 判定 Sanity Check 诊断报告

> **日期**：2026-02-10
>
> **版本**：S1.0k
>
> **触发原因**：训练跑了近 6500 万步（~1000/2000 iterations），`frac_success` 始终为 0

---

## 1. 功能介绍

### 1.1 问题背景

S1.0k 训练日志显示：
- `frac_success = 0.0000`：从未有任何环境触发成功
- `frac_inserted = 0.0000`：从未有任何环境达到插入阈值
- `insert_norm_mean = 0.02~0.06`：实际插入深度远低于阈值
- `lift_height_mean = -0.03`：负值，叉车在下沉

需要排查：**是"策略学不到"还是"成功判定不可能触发"还是"物理上就到不了"**。

### 1.2 工具说明

在 `scripts/verify_forklift_insert_lift.py` 中新增了 `--sanity-check` 模式，执行两层诊断：

**A 层：纯判定逻辑验证（传送绕开物理）**
- A1：将叉车传送到"理论上应该成功"的完美位姿，step 1 步，检查 `success_now` 的三个分量（`inserted_enough`、`aligned_enough`、`lifted_enough`）是否全部为 True
- A2：以零动作连续 step 40 步，检查 `hold_counter` 能否累积到 30（即连续保持成功状态 1 秒），以及 `terminated` 是否变为 True

**B 层：物理可达性验证（真实控制推进）**
- B1：从完美对齐位置以 `drive=0.3` 持续前进最多 800 步，记录最大可达 `insert_depth`，判断碰撞体是否阻止深插入
- B2：在已插入状态下以 `lift=1.0` 持续举升最多 400 步，记录最大可达 `lift_height`

两层均带**早停机制**（连续 N 步无增长则停止）和**卡住诊断**（输出轮速/车体速度，区分打滑 vs 碰撞阻挡）。

### 1.3 运行命令

```bash
cd /home/uniubi/projects/forklift_sim/IsaacLab
./isaaclab.sh -p ../scripts/verify_forklift_insert_lift.py --headless --sanity-check
```

### 1.4 关键实现细节

**`teleport_to_success_pose()`** — 传送叉车到完美成功位姿：
- 沿托盘插入轴（S1.0k 中心线几何）计算目标 tip 位置，通用 yaw 公式反推 root
- **不调用 `sim.reset()`**（会覆盖写入的位姿），改用 `write_data_to_sim()` → `sim.step()` → `scene.update()`
- 同步所有内部缓存：`_lift_pos_target`（防止控制器拉回 0）、`_fork_tip_z0`（lift_height 基准）、`_last_phi_total`、`_hold_counter`、`_is_first_step` 等

**`_read_success_components()`** — 独立复现 S1.0k 的几何计算，读取 insert_depth / y_err / yaw_err / lift_height 及其判定结果。

---

## 2. 成功判定条件

成功需要**同时且连续 30 步**满足：

- `inserted_enough`: `insert_depth >= insert_thresh`
- `aligned_enough`: `y_err <= 0.03m` 且 `yaw_err <= 3deg`
- `lifted_enough`: `lift_height >= 0.12m`

成功判定**完全基于几何量**，不依赖碰撞/接触事件。

关键参数：

- ~~`insert_thresh = 0.6667 * 2.16 = 1.44m`~~（修复前，物理不可达）
- `insert_thresh = 0.40 * 2.16 = 0.864m`（**修复后**，详见第 7 节）
- `max_lateral_err_m = 0.03m`
- `max_yaw_err_deg = 3.0deg`
- `lift_delta_m = 0.12m`
- `hold_steps = 30`（`hold_time_s=1.0s`）
- `fork_forward_offset = 1.8667m`
- `fork_z_base = 0.0m`

---

## 3. 运行结果（完整日志）

### 3.1 环境信息

```
环境数量: 1
Physics step-size: 0.008333s (120Hz)
Environment step-size: 0.03333s (30Hz, decimation=4)
GPU: NVIDIA Tegra NVIDIA GB10 (92GB)
```

### 3.2 A1：传送到完美位姿 → step 1 步

```
[teleport] root=(-1.4467, -0.0000, 0.0300)
[teleport] tip =(0.4199, -0.0001, 0.1785)
[teleport] pallet=(0.0000, 0.0000, 0.1480)
[teleport] lift_joint=0.1500m, _lift_pos_target=0.1500
[teleport] _fork_tip_z0=0.0300, _last_phi_total=15.4732

[A1] step 1 后各分量:
  insert_depth: 1.4904m (阈值=1.4400m) → PASS
  y_err: 0.000237m (阈值=0.0300m) → PASS
  yaw_err_deg: 0.0034deg (阈值=3.00deg) → PASS
  lift_height: 0.1637m (阈值=0.1200m) → PASS
  success_now: TRUE
  hold_counter: 1/30
```

**结论：判定逻辑本身正确。** 当几何量满足条件时，`success_now` 确实为 True。

### 3.3 A2：连续 step 40 步检查 hold_counter

```
step   2: hold_counter=  2, success_now=T  ins=1.4831 y=0.00082 yaw=0.141 lift=0.1692
step   3: hold_counter=  3, success_now=T  ins=1.4724 y=0.00351 yaw=0.343 lift=0.1895
step   4: hold_counter=  4, success_now=T  ins=1.4407 y=0.00078 yaw=0.021 lift=0.2015
step   5: hold_counter=  0, success_now=F  ins=1.4078 y=0.00057 yaw=0.019 lift=0.2125
  ↑ hold 中断！insert_depth 从 1.4407 跌到 1.4078 < 1.44m 阈值
step   6: hold_counter=  0, success_now=F  ins=1.4001 ...
  ...
step  41: hold_counter=  0, success_now=F  ins=1.2516 y=0.02353 yaw=1.979 lift=0.1223

[A2] hold_counter 最高: 4/30
[A2] success (terminated): FALSE
[A2] hold 中断于 step 5: inserted_enough=F (insert_depth=1.4078)
```

**分析：** 传送把叉车"塞进"了托盘（几何重叠），PhysX 碰撞解算立刻把机器人往外推，每步后退约 0.02m。4 步后 insert_depth 跌破 1.44m。这是物理引擎正常行为——不允许穿透。hold_counter 逻辑本身没有 bug。

### 3.4 B1：物理插入深度可达性

```
step    0: insert_depth=0.0000m, max=0.0000m, stall=1
step  100: insert_depth=0.3159m, max=0.3159m, stall=0
step  200: insert_depth=1.0313m, max=1.0312m, stall=17
step  263: insert_depth=1.0313m, max=1.0312m, stall=80
[B1] 卡住！连续 80 步无增长
     轮速=0.7718 rad/s, 车体速度=0.2508 m/s
     → 车在动但插入深度不增 → 可能滑移/偏航

[B1] 最大插入深度: 1.0312m (阈值=1.4400m)
[B1] 结论: 不可达
[B1] 缺口: 0.4088m
```

**关键发现：物理上最多只能插入 1.03m，而成功阈值要求 1.44m。缺口 0.41m，差 28%。**

卡住时轮速 0.77 rad/s、车体速度 0.25 m/s，说明轮子在转、车也在动，但插入深度不增——**叉车被碰撞体挡住后开始侧滑/偏转**。

### 3.5 B2：物理举升高度可达性

```
step    0: lift_height=-0.0291m, lift_joint=0.0013m
step   60: lift_height=0.4622m,  lift_joint=0.4925m
step  120: lift_height=0.9622m,  lift_joint=0.9925m
step  180: lift_height=1.4622m,  lift_joint=1.4925m
step  240: lift_height=1.9601m,  lift_joint=1.9905m
step  302: 卡住（joint 达到 2.0m 上限）

[B2] 最大举升高度: 1.9644m (阈值=0.1200m)
[B2] 结论: 可达
```

**举升完全正常。** 最大 1.96m >> 阈值 0.12m。

---

## 4. 综合诊断

```
================================================================
          SUCCESS SANITY CHECK 诊断报告
================================================================

--- A 层：判定逻辑验证 ---
[A1] success_now:     PASS     ← 判定公式/阈值/单位正确
[A2] hold_counter:    4/30     ← 碰撞解算导致位姿漂移，非逻辑 bug

--- B 层：物理可达性验证 ---
[B1] 最大插入: 1.0312m < 阈值 1.4400m   ← 不可达（根因）
[B2] 最大举升: 1.9644m > 阈值 0.1200m   ← 可达

================================================================
根因：碰撞体阻止叉车插入超过 ~1.03m，但 insert_fraction=2/3
      要求 1.44m。训练无论跑多久，success 都不可能触发。
================================================================
```

---

## 5. 根因链路

```
托盘碰撞近似 = convexDecomposition
    → PhysX 将托盘内部空间也视为碰撞体
    → 叉齿最多只能物理推进 ~1.03m
    → insert_depth 永远 < 1.44m (insert_thresh)
    → inserted_enough 永远为 False
    → success_now 永远为 False
    → frac_success = 0.0000
    → 训练无法获得成功奖励
    → 策略无法学到有效行为
```

关键证据：
- B1 测试中轮子在转（0.77 rad/s）、车在动（0.25 m/s）但插入深度不增 → 被碰撞体卡住
- 托盘碰撞近似从 `boundingCube` 被动态修改为 `convexDecomposition`（见环境初始化日志）
- convexDecomposition 会将 U 形托盘的内部凹陷空间也"填满"为凸碰撞体

---

## 6. 修复方案

### 方案 A：降低 `insert_fraction`（快速修复，立即可用）

将 `env_cfg.py` 中的 `insert_fraction` 从 `2/3`（1.44m）降低到可达范围内：

```python
# 当前（不可达）
insert_fraction: float = 2.0 / 3.0    # → 1.44m（物理上限 1.03m）

# 建议修改（留 15% 安全余量）
insert_fraction: float = 0.40          # → 0.864m（物理上限 1.03m，余量 0.17m）
```

**优点**：改一行配置即可，无需修改碰撞体
**缺点**：任务变简单了，"浅插入"可能不是真实场景需要的深度

文件位置：`forklift_pallet_insert_lift_project/isaaclab_patch/source/isaaclab_tasks/isaaclab_tasks/direct/forklift_pallet_insert_lift/env_cfg.py`

### 方案 B：修改托盘碰撞体（根本修复，需要测试）

当前问题是 `convexDecomposition` 将 U 形托盘的内部空间填满了。需要：

1. **在 USD 中手动编辑碰撞体**：将托盘的碰撞近似从 `convexDecomposition` 改为手工构建的多个 box collider（底板 + 左侧板 + 右侧板），留出叉齿插入的通道
2. **或在 `env.py` 的 `_setup_scene()` 中修改碰撞近似类型**：
   - 从 `convexDecomposition` 改为 `meshSimplification`（保留凹陷）
   - 或使用 `triangleMesh`（精确但性能差）

**优点**：物理上真实，插入深度可达全深
**缺点**：需要修改 USD 资产或碰撞配置，需要测试碰撞稳定性

### 方案 C：A+B 组合（推荐）

1. **先执行方案 A**：立即降低 `insert_fraction` 到 0.40，验证训练流程能跑通、策略能学到成功
2. **再执行方案 B**：修改碰撞体，恢复更高的 `insert_fraction`，让任务更接近真实

### 修改后验证

无论选择哪个方案，修改后重新运行 sanity check 验证：

```bash
cd /home/uniubi/projects/forklift_sim/IsaacLab
./isaaclab.sh -p ../scripts/verify_forklift_insert_lift.py --headless --sanity-check
```

预期结果：
- B1 最大插入深度 > 新的 insert_thresh
- A1 success_now = TRUE
- A2 hold_counter 达到 30/30（如果用方案 B 根本修复了碰撞体）

---

## 7. 方案 A 修复验证

### 7.1 修改内容

**配置变更**：将 `env_cfg.py` 中 `insert_fraction` 从 `2/3`（1.44m）降低为 `0.40`（0.864m）。

```python
# ===== KPI（成功判定指标）=====
# S1.0k sanity check 发现：convexDecomposition 碰撞体阻止叉齿插入超过 ~1.03m，
# 原 2/3 (1.44m) 阈值物理不可达。降低到 0.40 (0.864m)，留 15% 安全余量。
# 详见 docs/diagnostic_reports/success_sanity_check_2026-02-10.md
insert_fraction: float = 0.40
```

**同步步骤**：

```bash
# 1. 修改 patch 源
#    forklift_pallet_insert_lift_project/isaaclab_patch/.../env_cfg.py

# 2. 重新安装到 IsaacLab 运行时目录
bash forklift_pallet_insert_lift_project/scripts/install_into_isaaclab.sh \
     /home/uniubi/projects/forklift_sim/IsaacLab

# 3. 验证同步生效
#    IsaacLab/source/isaaclab_tasks/.../env_cfg.py 中 insert_fraction 应为 0.40
```

> **注意**：仅修改 patch 源目录不会生效，必须执行 `install_into_isaaclab.sh` 将变更复制到 IsaacLab 运行时目录。首次验证时因遗漏此步骤导致 sanity check 仍读到旧值 `2/3`。

### 7.2 验证结果（完整日志）

```
================================================================================
  关键阈值
================================================================================
  insert_thresh: 0.8640m (insert_fraction=0.4000 * pallet_depth=2.1600)
  max_lateral_err_m: 0.0300m
  max_yaw_err_deg: 3.00deg
  lift_delta_m: 0.1200m
  hold_steps: 30 (hold_time_s=1.00s)
  fork_forward_offset: 1.8667m
  fork_z_base: 0.0000m

================================================================================
  A1: 传送到完美位姿 → step 1 步
================================================================================
  [A1] step 1 后各分量:
  insert_depth: 1.4904m (阈值=0.8640m) → PASS
  y_err: 0.000237m (阈值=0.0300m) → PASS
  yaw_err_deg: 0.0034deg (阈值=3.00deg) → PASS
  lift_height: 0.1637m (阈值=0.1200m) → PASS
  success_now: TRUE
  hold_counter: 1/30

================================================================================
  A2: 连续 step 40 步（零动作）检查 hold_counter
================================================================================
  step   2: hold_counter=  2, success_now=T  ins=1.4831 y=0.00082 yaw=0.141 lift=0.1692
  step   3: hold_counter=  3, success_now=T  ins=1.4724 y=0.00351 yaw=0.343 lift=0.1895
  step   4: hold_counter=  4, success_now=T  ins=1.4407 y=0.00078 yaw=0.021 lift=0.2015
  step   5: hold_counter=  5, success_now=T  ins=1.4078 y=0.00057 yaw=0.019 lift=0.2125
  step   6: hold_counter=  6, success_now=T  ins=1.4001 y=0.00038 yaw=0.025 lift=0.2094
  step  12: hold_counter= 12, success_now=T  ins=1.3904 y=0.00376 yaw=0.159 lift=0.1725
  step  22: hold_counter= 22, success_now=T  ins=1.3785 y=0.00399 yaw=0.112 lift=0.1289
  → terminated=True at step 30, hold_counter=30

  [A2] hold_counter 最高: 30/30
  [A2] success (terminated): TRUE

================================================================================
  B1: 物理插入深度可达性（从对齐位置前进）
================================================================================
  step    0: insert_depth=0.0000m, max=0.0000m, stall=1
  step  100: insert_depth=0.3159m, max=0.3159m, stall=0
  step  200: insert_depth=0.9386m, max=0.9384m, stall=1
  step  279: insert_depth=0.9392m, max=0.9384m, stall=80
  [B1] 卡住！连续 80 步无增长
       轮速=-0.0000 rad/s, 车体速度=0.0001 m/s
       → 轮子和车都不动 → 可能被完全卡死

  [B1] 最大插入深度: 0.9384m (阈值=0.8640m)
  [B1] 结论: 可达

================================================================================
  B2: 物理举升高度可达性（在当前插入状态下举升）
================================================================================
  step    0: lift_height=-0.0290m, lift_joint=0.0014m
  step   60: lift_height=0.0465m,  lift_joint=0.0769m
  step  120: lift_height=0.5466m,  lift_joint=0.5769m
  step  180: lift_height=1.0477m,  lift_joint=1.0781m
  step  240: lift_height=1.5477m,  lift_joint=1.5781m
  step  300: lift_height=1.9676m,  lift_joint=1.9979m
  step  354: 卡住（joint 达到 2.0m 上限）

  [B2] 最大举升高度: 1.9673m (阈值=0.1200m)
  [B2] 结论: 可达

================================================================
          综合诊断
================================================================

  >>> 判定逻辑和物理可达性均正常，训练 success=0 是策略问题 <<<
  >>> 建议：调整奖励/课程学习/超参数 <<<
  所有检查通过！
================================================================
```

### 7.3 修复前后对比

| 测试项 | 修复前（insert_fraction=2/3） | 修复后（insert_fraction=0.40） | 变化 |
|--------|-------------------------------|--------------------------------|------|
| insert_thresh | 1.4400m | **0.8640m** | -0.576m |
| A1 success_now | PASS | PASS | 不变 |
| A2 hold_counter | **4/30 (FAIL)** | **30/30 (PASS)** | 逻辑判定可持续通过 |
| A2 terminated | FALSE | **TRUE** | 成功终止 |
| B1 最大插入 | 1.0312m < 1.44m (**不可达**) | 0.9384m > 0.864m (**可达**) | 阈值降低后物理可达 |
| B2 最大举升 | 1.9644m (可达) | 1.9673m (可达) | 不变 |

### 7.4 余量分析

- **B1 余量**：`0.9384m - 0.8640m = 0.0744m`（约 **8.6%**）
- 训练中随机初始化会引入轻微偏航/侧偏，实际可达深度可能略低于 0.94m
- 8.6% 余量较紧但可用，如果训练中发现 `frac_inserted` 仍偏低，可考虑进一步降低 `insert_fraction` 至 0.35

### 7.5 A2 通过的原因分析

修复前 A2 在 step 5 就失败了，因为 PhysX 碰撞解算将叉车向外推，insert_depth 从 1.49m 快速下降：
- step 4: `insert_depth=1.4407m` > 1.44m → PASS
- step 5: `insert_depth=1.4078m` < 1.44m → FAIL（中断 hold）

修复后阈值降为 0.864m，即使 insert_depth 持续下降（step 41 时降到 1.25m），仍远高于 0.864m，因此 hold_counter 持续累积到 30/30 触发成功终止。

### 7.6 结论

方案 A 修复有效：
1. 判定逻辑验证（A 层）全部通过，hold_counter 可达 30/30 触发成功
2. 物理可达性验证（B 层）全部通过，插入深度和举升高度均超过阈值
3. 综合诊断结论："判定逻辑和物理可达性均正常，所有检查通过"
4. **可以重新启动 S1.0k 训练**，预期 `frac_success` 将不再为 0

---

## 8. S1.0L 奖励整改落地记录（2026-02-10）

### 8.1 本次代码变更

已在 `env.py` / `env_cfg.py` 落地以下 S1.0L 改动：

1. **Stage1/2 距离参考点改为 base**
   - `e_band`、`w_band`、`E2` 的距离项使用 `dist_front_base`
   - `insert_depth/insert_norm` 仍基于 `tip`，成功判定不变
2. **推迟 Stage3 接管**
   - `ins_start: 0.02 -> 0.10`
   - `ins_ramp: 0.05 -> 0.15`
3. **纯差分 shaping**
   - `gamma: 0.99 -> 1.0`
4. **默认不压制 phi1/phi2**
   - 默认 `phi_total = phi1 + phi2 + phi_ins + phi_lift`
   - 保留回滚开关 `suppress_preinsert_phi_with_w3`
5. **Stage3 门控放宽 + 插入势函数 baseline 提升**
   - `y_gate3: 0.10 -> 0.18`
   - `yaw_gate3: 8 -> 12`
   - `phi_ins` baseline 从 `0.2` 提升到 `0.4`
6. **新增里程碑奖励与失败早停**
   - milestones: `approach/align/insert10/insert30`
   - early-stop: `fly/stall`（惩罚在 reward，done 在 dones）

### 8.2 冒烟训练验证（30 iterations）

运行命令：

```bash
cd /home/uniubi/projects/forklift_sim/IsaacLab
CONDA_PREFIX= CONDA_DEFAULT_ENV= TERM=xterm ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Forklift-PalletInsertLift-Direct-v0 \
  --headless --num_envs 64 --max_iterations 30 --run_name s1.0l_smoke
```

日志文件：`logs/s1.0l_smoke_train.log`

关键观测（iter 0 -> iter 29）：

- `s0/w_band`: `0.6171 -> 0.9654`（不再是旧版那种长期焊死 1.0）
- `s0/e_band`: `0.0829 -> 0.3018`（进入可学习量级，不再是 1.6+ 的失配区间）
- `s0/r_pot`: `0.0004 -> 0.0019`（均值接近 0，已摆脱固定负底噪）
- `err/insert_norm_mean`: `0.0196 -> 0.0799`
- `phase/frac_inserted`: 从长期 `0.0000` 提升到阶段性非零（最高见到 `0.0938`）
- `phase/frac_aligned`: 从 `~0.01` 抬升到 `0.0312` 量级

新增日志项可正常产出：

- `err/dist_front_base_mean`
- `err/stage_dist_front_mean`
- `s0/r_milestone`
- `s0/pen_early_stop`
- `term/frac_early_fly`
- `term/frac_early_stall`

### 8.3 当前结论

S1.0L 已完成代码落地与短训冒烟，结果显示：

1. 奖励结构从“尺度失配 + 负常数底噪”状态中解耦成功；
2. 插入相关统计已出现非零（`frac_inserted > 0`）；
3. 仍未在 30 iter 内出现 `frac_success > 0`，属于预期（训练轮次不足，且 lift 阶段尚未形成稳定策略）；
4. 建议进入下一轮中等规模训练（例如 200~500 iter）继续观测 `frac_success` 与 `hold_counter_max`。
