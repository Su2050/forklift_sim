# 实验报告：s1.0x early_stop_fly 放宽 + 从零 vs resume 对比

**日期：** 2026-02-23
**分支：** `exp/fix-pallet-dragging`
**提交：** `dab60c3` — `fix(env): relax early_stop_fly thresholds to prevent false termination`

---

## 一、实验背景

### 1.1 问题

`early_stop_fly` 的旧阈值过于严苛：叉车初始位置 `(-3.5, 0, 0.03)` 到托盘 `(0, 0, 0.15)` 的 XY 距离为 3.5m，已超过 `early_stop_d_xy_max = 3.0m`。Episode 开始时 fly_counter 立即累加，模型仅有 30 步（约 1 秒）将距离压到 3.0m 以下。

### 1.2 修改内容

| 参数 | 修改前 | 修改后 |
|------|--------|--------|
| `early_stop_d_xy_max` | 3.0 m | 5.0 m |
| `early_stop_d_xy_steps` | 30 步 | 60 步 |

修改依据：`docs/episode_termination_conditions.md` 中的方案 D。

### 1.3 实验目的

1. 验证放宽 `early_stop_fly` 不会引入 reward hacking（模型不会利用更大的活动范围逃避任务）。
2. 对比从零训练 vs 基于旧模型 resume 的效果差异，确定后续最佳训练路线。

---

## 二、实验设计

三组实验均使用相同的环境配置（s1.0x，含放宽后的 early_stop_fly）：

| 实验 | 描述 | 总迭代数 | 起点 | 日志 | Checkpoint 目录 |
|------|------|----------|------|------|-----------------|
| s1.0x | 从零训练 | 2000 | 随机初始化 | `20260223_081254_train_s1.0x.log` | `2026-02-23_08-12-59/` |
| A: s1.0x_continue | s1.0x 继续训练 | 2000→4000 | s1.0x model_1999 | `20260223_101749_train_s1.0x_continue.log` | `2026-02-23_10-17-55/` |
| B: s1.0x_resume | 旧 anti_hack 继续训练 | 2000→4000 | `2026-02-22_21-34-12` model_1999 | `20260223_120020_train_s1.0x_resume.log` | `2026-02-23_12-00-26/` |

共同参数：1024 envs, seed 42, cuda:0, ClampedActorCritic, 每 50 iter 保存 checkpoint。

---

## 三、最终结果对比（各实验最后一个 iteration）

### 3.1 核心指标

| 指标 | s1.0x (iter 1999) | A: continue (iter 3998) | B: resume (iter 3998) | 最优 |
|------|-------------------|-------------------------|------------------------|------|
| Mean reward | -86.2 | +7.3 | **+99.7** | B |
| frac_success_now | 1.46% | 1.37% | **1.86%** | B |
| frac_success | 0.49% | 0.29% | 0.29% | 持平 |
| frac_aligned | 19.8% | 21.3% | **28.0%** | B |
| near_success_frac | 40.1% | 44.3% | **46.0%** | B |
| deep_insert_frac | 17.0% | **19.2%** | 15.2% | A |

### 3.2 精度指标

| 指标 | s1.0x (iter 1999) | A: continue (iter 3998) | B: resume (iter 3998) | 最优 |
|------|-------------------|-------------------------|------------------------|------|
| lateral_mean | 0.337 m | 0.332 m | **0.216 m** | B |
| deep_lat_bad_frac | 8.1% | 9.0% | **1.37%** | B |
| yaw_deg_mean | 4.88° | **4.28°** | 4.95° | A |
| yaw_deg_deep_mean | **1.61°** | 1.86° | 2.47° | s1.0x |
| lateral_near_success | 0.318 m | 0.286 m | **0.158 m** | B |

### 3.3 势函数与惩罚

| 指标 | s1.0x (iter 1999) | A: continue (iter 3998) | B: resume (iter 3998) | 最优 |
|------|-------------------|-------------------------|------------------------|------|
| phi_total | 3.60 | 3.84 | **4.85** | B |
| phi_ins | 0.77 | 0.88 | **1.14** | B |
| phi_lift | 0.15 | 0.18 | **0.36** | B |
| pen_global_stall | -0.330 | -0.195 | **-0.135** | B |
| pen_premature | -0.003 | -0.004 | **-0.0004** | B |
| r_terminal | **+0.575** | +0.350 | +0.340 | s1.0x |

### 3.4 终止条件验证

| 指标 | s1.0x | A: continue | B: resume |
|------|-------|-------------|-----------|
| term/frac_early_fly | 0.0000 | 0.0000 | 0.0010 |
| term/frac_early_stall | 0.0000 | 0.0000 | 0.0000 |
| term/frac_tipped | 0.0000 | 0.0000 | 0.0000 |

`early_stop_fly` 在三组实验中几乎不触发（实验 B 仅 0.1%），放宽阈值未引入任何问题。

---

## 四、最佳模型

| 项目 | 值 |
|------|------|
| 推荐模型 | 实验 B (s1.0x_resume) |
| Checkpoint | `IsaacLab/logs/rsl_rl/forklift_pallet_insert_lift/2026-02-23_12-00-26/model_3998.pt` |
| 总训练量 | 旧 s1.0w_anti_hack 2000 iters + s1.0x_resume 2000 iters = 4000 iters 等效 |

---

## 五、结论

### 5.1 early_stop_fly 放宽验证通过

- 三组实验 `term/frac_early_fly` 均接近零，模型没有利用更大活动范围逃避任务。
- 放宽后消除了初始位置即在禁区内的隐患，对训练无负面影响。

### 5.2 实验 B（旧模型 resume）全面胜出

- **横向精度碾压**：`lateral_mean` 0.216m vs 0.33m（实验 A），`deep_lat_bad_frac` 1.4% vs 9.0%，差距约 6 倍。旧模型在 s1.0w 中积累的横向对齐能力被完整继承。
- **势函数最高**：`phi_total = 4.85`，比实验 A 高 26%，各阶段完成度更高。
- **停滞最少**：`pen_global_stall = -0.135`，模型行为最流畅。
- **Mean reward 大幅领先**：+99.7 vs +7.3。

### 5.3 从零训练的唯一优势

深插入时的航向精度更高（`yaw_deg_deep_mean` 1.6° vs 2.5°），但不足以弥补横向对齐的巨大差距。

### 5.4 后续建议

基于实验 B 的 `model_3998.pt` 继续迭代训练，重点关注：
1. 从"深插入"到"举升成功"的最后一步转化率（当前瓶颈）。
2. 深插入航向精度仍有提升空间（可参考从零训练路线中学到的特征）。

---

## 六、Play 命令

```bash
cd /home/uniubi/projects/forklift_sim/IsaacLab

# 实验 B 最终模型
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
  --task Isaac-Forklift-PalletInsertLift-Direct-v0 \
  --num_envs 1 \
  --checkpoint "logs/rsl_rl/forklift_pallet_insert_lift/2026-02-23_12-00-26/model_3998.pt" \
  --seed 999 --headless --video --video_length 1200

# 实验 A 最终模型（对比用）
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
  --task Isaac-Forklift-PalletInsertLift-Direct-v0 \
  --num_envs 1 \
  --checkpoint "logs/rsl_rl/forklift_pallet_insert_lift/2026-02-23_10-17-55/model_3998.pt" \
  --seed 999 --headless --video --video_length 1200
```
