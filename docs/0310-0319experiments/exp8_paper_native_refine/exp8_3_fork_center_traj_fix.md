# Exp 8.3：轨迹跟踪对象修正（fork_center 替换 root_pos）

**日期**：2026-03-19（文档 2026-03-20 修订：结论与建议与代码、日志对齐）
**实验分支**：`exp/vision_cnn/exp8_paper_native_refine` (Run: `exp8_3_fork_center_traj`)
**日志**：`logs/20260319_215340_train_exp8_3_fork_center_traj.log`

**Exp8.3 验证计划实施分支（B0′ + 诊断日志 + U0/U1）**：`exp/exp8_3_geom_validation_b0prime`  
- `_reset_idx`：参考轨迹改为在写入 pallet / robot / joint **之后**生成，且 `_build_reference_trajectory` 使用 reset 张量显式传入，避免旧 episode 污染 `r_cd`/`r_cpsi`。  
- `_get_rewards`：新增 `s_center_mean`、`s_tip_mean`、`err/root|center|tip_lateral_mean`、`phase/frac_rg`、`phase/frac_success`、`diag/out_of_bounds_frac`、`diag/success_term_frac` 及首步 `geom/s_*` 常量。  
- 预检单测（无 Isaac）：`forklift_pallet_insert_lift_project/tests/test_exp83_geometry_preflight.py`（可直接 `python .../test_exp83_geometry_preflight.py`）。
- B0′ smoke（需本机 Isaac 环境）：`bash scripts/run_exp8_3_b0prime_smoke.sh`（日志带时间戳写入 `logs/`）；脚本在 `TERM=dumb`（如 nohup）下会自动设为 `xterm-256color`，避免 `tabs: dumb` 报错。跑前执行 `install_into_isaaclab.sh`。

## 1. 问题发现

在对 Exp 8.2 进行深度诊断时，通过 ResNet34 特征相似度分析排除了「视觉盲区」假设后，我们用 `check_traj_tangent.py` 脚本发现了一个**轨迹几何 Bug**：参考轨迹的生成与查询使用了 **车体中心（`root_pos`）**，而论文要求的是 **叉臂中心（center of the forks）**。

### Bug 描述

```python
# _build_reference_trajectory: 起点用的是车体中心
p0 = self.robot.data.root_pos_w[env_ids, :2]

# _query_reference_trajectory: 查询点用的也是车体中心
root_pos = self.robot.data.root_pos_w[:, :2]
```

> "$r_d$ and $r_{cd}$ are the distances from the **center of the forks** to the pallet and clothoid curve"

### Bug 后果（Exp 8.2）

终点前 `traj_pre_dist_m = 1.2m` 为直线段，以外为 Hermite 曲线段。当叉尖到达托盘前沿时，叉臂中心约在前沿后方 ~0.6m（处于直线段、切线 0°），而车体中心仍在曲线段，切线约 -17°——Agent 按轨迹拿满 $r_{c\psi}$ 却难以真正插入。

## 2. 修复内容

仅修改 `env.py` 中轨迹起点与查询点为 `fork_center`（见 patch 源目录 `forklift_pallet_insert_lift/.../env.py`）。

## 3. 早期训练结果（对照 Exp 8.2 @ 100 iter）

| 指标 | Exp 8.2 @ 100 代 | Exp 8.3 @ 11 代 | 说明 |
| :--- | ---: | ---: | :--- |
| `yaw_deg_mean` | ~15.5° | **7.3°** | 早期对齐明显改善 |
| `pallet_disp_xy` | ~0.10 m | **0.04 m** | 推盘更小 |
| `r_cpsi` | ~5.0 | **8.2** | 轨迹切线信号与 fork 一致 |
| `R_plus` | ~59.9 | **110.7** | 与上表同一日志块 |

**结论（早期）**：`fork_center` 修复使近端切线方向与论文定义一致，前段训练偏航与推盘优于 Bug 版。

## 4. 其他配置（与 Exp 8.2 一致）

- 奖励：`1/x` + `clip(20)`；$\alpha_1=\alpha_2=\alpha_3=5.0$（近场 `dist_front<0.5m` 时 $\alpha_3$×3），$\alpha_4=50.0$（`rg`）
- 惩罚：$r_p$=0.5, $r_a$=0.1, $r_{ini}$=5.0, $r_{bound}$=0.5
- 轨迹：Hermite + 直线；**`p_goal` = 托盘前沿中心**
- 视觉：ResNet34 冻结；64 env；目标 2000 iter（本 run 至少记录至 1104 iter）

**几何注记**：「`fork_center` 到前沿 ≈ tip 插入 0.6m」仅在 **车体/托盘偏航近似对齐（Δψ≈0）** 时成立；大偏航时沿托盘轴有效插入为 $0.6\cos(\Delta\psi)$，不能用 center 几何直接替代 tip 质量。

---

## 5. 完整训练过程（日志实测至 1104 iter）

### 5.1 关键指标纵向对比

（`frac_inserted` 等为 `phase/frac_inserted` 的百分比写法；与 `20260319_215340_train_exp8_3_fork_center_traj.log` 抽样一致。）

| 迭代 | Mean Reward | yaw_deg_mean | dist_front | frac_inserted | pallet_disp | r_cpsi | traj/yaw_traj_deg |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | ~1,200 | 7.8° | 1.75 m | 0% | 0.00 m | 8.95 | 8° |
| 10 | ~19,601 | 7.1° | 0.87 m | 0% | 0.02 m | 7.75 | 26° |
| 50 | ~76,633 | 7.8° | 0.81 m | 0% | 0.04 m | 6.58 | 42° |
| 100 | ~80,801 | 9.3° | 0.83 m | 3.1% | 0.05 m | 4.82 | 49° |
| 200 | ~79,993 | 9.3° | 0.75 m | 1.6% | 0.10 m | 6.56 | 37° |
| 300 | ~79,094 | 10.2° | 0.68 m | 0% | 0.11 m | 5.97 | 46° |
| 400 | ~71,782 | 15.2° | 0.60 m | 4.7% | 0.29 m | 5.64 | 50° |
| 500 | ~63,776 | **22.6°** | 0.38 m | 6.3% | 0.48 m | 3.44 | 50° |
| 600 | ~64,377 | 15.3° | 0.61 m | 3.1% | 0.17 m | 4.95 | 59° |
| 700 | ~43,536 | 31.1° | 0.44 m | **10.9%** | 0.90 m | 2.95 | 62° |
| 800 | ~39,670 | 40.3° | 0.71 m | **14.1%** | 1.03 m | 2.51 | 71° |
| 900 | ~40,851 | 31.2° | 0.76 m | 6.3% | 0.66 m | 2.98 | 58° |
| 1000 | ~30,813 | 47.5° | 0.96 m | 1.6% | 0.33 m | 2.04 | 69° |
| 1050 | ~32,500 | 62.1° | 1.24 m | 1.6% | 0.58 m | 1.34 | 76° |
| **1103** | **~30,962** | **46.5°** | 0.95 m | 4.7% | 0.60 m | 2.25 | 68° |

### 5.2 阶段划分（描述性，非唯一解释）

- **Iter ~1–200**：`yaw_deg_mean` 约 7–10°，推盘较低；`R_plus` 高。`frac_inserted` 仍多为 0–3%。
- **Iter ~200–400**：偏航与横向误差渐增，`traj/yaw_traj_deg_mean` 上升，轨迹跟随变差。
- **Iter ~400–800**：`frac_inserted` 阶段性升高，但常伴 `pallet_disp` 很大、`yaw_deg_mean` 高——更像**大偏航下的接触/推盘**，不宜单独称为「成功插入」。
- **Iter ~800–1103**：`yaw_deg_mean` 常处 45–60°，`r_cpsi` 很低，Mean Reward 相对前期明显下降。

### 5.3 关于 `rg` 与 `success`（重要纠偏）

- **`rg` 并非全程为 0**。例如 iter **798–802** 日志中出现 `paper_reward/rg: 0.0156`（batch 内少量 env 满足门控）。早期多数 iter 为 `rg: 0.0000` 仍成立。
- **`rg` ≠ `success`**。代码中：
  - `rg`：`dist_center < paper_rg_dist_thresh` 且 `tip_y_err < 0.20` 且 `yaw_err_deg < 15°`。
  - `success`：`insert_norm >= insert_fraction` 且 `y_err < 0.1` 且 `yaw_err_deg < 5.0`，并需 `hold_counter` 累积（与 `env_cfg` 中放宽的 `max_lateral_err_m` / `max_yaw_err_deg` **未统一**，success 仍为硬编码严门槛）。
- 因此：**不能用「`rg` 几乎为 0」直接推出「从未出现任何深插」**；只能说明 **未稳定满足 `rg` 门控**，且 **episode 级 success（若日志未单独统计）需另查**。

---

## 6. 根因分析（按优先级）

### 6.1 主因：三套几何终点不一致 + 有限长度轨迹

在同一 approach 阶段，至少存在：

| 名称 | 代码含义（托盘轴 `s`，中心为原点，前沿 `s_front=-D/2`） |
| :--- | :--- |
| **轨迹终点 `p_goal`** | `s = s_front`（前沿） |
| **`r_d` 的 `target_center`** | 前沿沿 `u_in` 再深入 **0.6 m**（`s ≈ s_front + 0.6`） |
| **success 深度** | `tip` 插入 `insert_fraction * D`（如 0.40×2.16=**0.864 m**），对应 **Δψ≈0** 时 `fork_center` 约在 **`s_front + 0.264 m`**，与 `p_goal`、`r_d` 目标均不同 |

此外，`d_traj` 是到**整条离散轨迹最近点**的距离；轨迹终点停在前沿时，继续深入托盘内部后最近点易仍卡在前沿附近，**可能与 `r_d` 拉深形成张力**。

### 6.2 次因：`r_d` 为欧氏距离、对偏航不敏感

`r_d = clip(1/(dist_center+ε), 0, max)` 不惩罚大偏航；与大偏航下仍可出现较高 `frac_inserted`、高推盘的现象**相容**，但不宜单独升格为「唯一根因」。

### 6.3 次因：`r_cpsi` 的 `clip(20)` 在近零误差处饱和

误差 &lt; ~0.05 rad 时 `r_cpsi` 已顶格，对「继续压小偏航」梯度弱。但退化段均值偏航常在 **10°→40°+**，**不能单靠**「小角饱和」解释全程崩溃，更适合作为**辅助因素**。

---

## 7. 实验结论（严谨表述）

| 维度 | 结论 |
| :--- | :--- |
| **`fork_center` 修复** | 有效消除 Exp 8.2 的「root 跟踪导致的错误切线」；前 ~200 iter 偏航与推盘优于典型 Bug 行为。 |
| **是否解决任务** | **未**：中后期仍出现大偏航、高推盘、轨迹跟随崩坏；**不宜**用「`rg` 全程为 0」概括。 |
| **失败形态** | 接触区出现「斜冲 / 推盘 + 偶发 `frac_inserted` 升高」与 reward 结构相容。 |
| **优先改进杠杆** | **统一** `p_goal`、`r_d` 的 `target_center` 与 success 所隐含的 **`fork_center` 终点**（并理清 tip/center 在 Δψ 下的关系）；其次再考虑 `r_d` 门控、轨迹延长与 `clip` 形状。 |

---

## 8. 下一步改进方向（修正几何错误后的建议）

1. **统一终点（推荐先做）**  
   - 将 **`p_goal` 与 `target_center`** 对齐到 **同一** success 对应的 **`fork_center` 平面位置**：在 Δψ≈0 下，相对前沿向内 **`insert_fraction * D - 0.6`** m（当前参数约 **0.264 m**），**不是**把 `p_goal` 直接平移 **0.864 m**（0.864 是 **tip** 插入深度，不是 `fork_center` 轨迹终点的位移）。  
   - 托盘轴标量：`s_unified ≈ s_front + (insert_fraction * D - 0.6)`（与 `D=2.16`、`insert_fraction=0.40` 时 `s ≈ -0.816` 一致）。

2. **单因素验证**：先只改 `target_center`，或只延长 `p_goal`，或两者同时改，对照 `8.3` 日志，避免与权重调参混谈（见项目内 `exp8.3` 验证计划）。

3. **`r_d` 偏航门控**（在终点统一后仍斜冲时再上）：偏航超阈时削弱或关闭 `r_d`。

4. **`success` 与 `env_cfg` 对齐**：将 `hold` 用的 `y_err` / `yaw` 阈值与 `env_cfg` 中课程参数统一，否则「成功率」解读会与配置注释不一致。

5. **谨慎**：单纯加大 `alpha_3` 或 `paper_reward_max` 可能放大近场数值与不稳定，应在几何一致后再调。

---

## 9. 反思

- **传感器先于算法**：参考轨迹几何错误会让策略「学对错误任务」；`fork_center` 修复是必要的。  
- **仅修跟踪点不够**：若轨迹终点、`r_d` 目标与 success 深度仍不一致，单阶段 RL 在接触区仍易出现目标冲突与可利用行为。  
- **表述需与代码一致**：`rg`、`success`、`tip` 插入与 `fork_center` 目标必须分条写清，避免混用导致错误结论。
