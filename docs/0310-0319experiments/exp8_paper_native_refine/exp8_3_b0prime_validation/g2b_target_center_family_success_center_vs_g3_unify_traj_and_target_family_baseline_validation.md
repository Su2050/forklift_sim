# Exp8.3 G2b vs G3 正式对比验证（400 iter）

**日期**：2026-03-22  
**对比窗口**：`iter 300-399`

**G2b 分支**：`exp/exp8_3_g2b_target_family_success_center`  
**G2b git rev**：`f7cebb73df8ae730ea05fdc5ffa40e0bc27051cc`  
**G2b 日志**：`forklift_sim_wt_g2b/logs/20260321_190257_train_exp8_3_g2b_target_center_family_success_center_baseline.log`  
**G2b 启动脚本**：`forklift_sim_wt_g2b/scripts/run_exp8_3_g2b_baseline.sh`  
**G2b IsaacLab run 目录**：`IsaacLab/logs/rsl_rl/forklift_pallet_insert_lift/2026-03-21_19-03-06_exp8_3_g2b_target_center_family_success_center_baseline`

**G3 分支**：`exp/exp8_3_g3_traj_and_target_success_center`  
**G3 git rev**：`51240643855cc0a460f2f43273498e7eac807fba`  
**G3 日志**：`forklift_sim_wt_g3/logs/20260321_223243_train_exp8_3_g3_unify_traj_and_target_family_baseline.log`  
**G3 启动脚本**：`forklift_sim_wt_g3/scripts/run_exp8_3_g3_baseline.sh`  
**G3 IsaacLab run 目录**：`IsaacLab/logs/rsl_rl/forklift_pallet_insert_lift/2026-03-21_22-32-52_exp8_3_g3_unify_traj_and_target_family_baseline`

## 对比目的

- `G2b` 的定义是：只把 `target_center family` 统一到 `s_success_center`，即 `r_d + rg + done 侧 out_of_bounds` 共同改到 `success_center`；trajectory terminal geometry package 仍保持 `front`。
- `G3` 的定义是：在 `G2b` 的基础上，再把 trajectory terminal geometry package 也统一到 `s_success_center`。
- 本文要回答的问题是：**在 `400 iter` 这一轮机制初筛 horizon 下，`G3` 是否能在 `G2b` 已有收益之上继续带来更强的入口接近与插入能力。**

## 实际训练 CLI override

两条 run 共享相同的训练骨架：

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Forklift-PalletInsertLift-Direct-v0 \
  --headless \
  --enable_cameras \
  --num_envs 64 \
  --max_iterations 400 \
  env.use_camera=true \
  env.use_asymmetric_critic=true \
  env.stage_1_mode=false \
  env.camera_width=256 \
  env.camera_height=256 \
  agent.policy.class_name=rsl_rl.modules.VisionActorCritic \
  agent.policy.backbone_type=resnet34 \
  'agent.obs_groups.policy=[image, proprio]' \
  'agent.obs_groups.critic=[critic]' \
  agent.policy.imagenet_backbone_init=true \
  agent.policy.freeze_backbone=true
```

真正区分 `G2b` 与 `G3` 的 override 只有：

```bash
# G2b
agent.run_name=exp8_3_g2b_target_center_family_success_center_baseline
env.exp83_traj_goal_mode=front
env.exp83_target_center_family_mode=success_center

# G3
agent.run_name=exp8_3_g3_unify_traj_and_target_family_baseline
env.exp83_traj_goal_mode=success_center
env.exp83_target_center_family_mode=success_center
```

## run 级几何常量

| run | `geom/s_traj_end` | `geom/s_rd_target` | `geom/s_success_center` | 含义 |
| --- | ---: | ---: | ---: | --- |
| `G2b-400` | -1.0800 | -0.8160 | -0.8160 | 只统一 `target_center family`，trajectory 终点仍停在前沿 |
| `G3-400` | -0.8160 | -0.8160 | -0.8160 | trajectory terminal geometry package 与 `target_center family` 全统一到 `success_center` |

## `iter 300-399` 窗口均值对比

| 指标 | `G2b-400` | `G3-400` | 直接读法 |
| --- | ---: | ---: | --- |
| `err/dist_front_mean` | 0.7151 | 1.2718 | `G2b` 明显更接近入口 |
| `err/lateral_mean` | 0.2928 | 0.2855 | `G3` 略好，但差距很小；该量当前仍是 root-based |
| `err/yaw_deg_mean` | 9.1831 | 8.6356 | `G3` 略好 |
| `diag/pallet_disp_xy_mean` | 0.1032 | 0.0345 | `G3` 明显更少推盘 |
| `phase/frac_inserted` | 0.0242 | 0.0083 | `G2b` 的插入倾向更强 |
| `phase/frac_rg` | 0.0128 | 0.0055 | `G2b` 更常进入 `rg` |
| `phase/frac_success` | 0.0000 | 0.0000 | 两者都未形成稳定成功 |
| `traj/d_traj_mean` | 0.3703 | 0.2623 | `G3` 更贴近自身轨迹走廊 |
| `traj/yaw_traj_deg_mean` | 12.7054 | 12.0622 | `G3` 略好 |
| `phase/frac_inserted` 非零计数 | 77 / 100 | 46 / 100 | `G2b` 更频繁进入插入态 |
| `phase/frac_rg` 非零计数 | 60 / 100 | 34 / 100 | `G2b` 更频繁触达 `rg` |

## 窗口锚点

| iter | run | `err/dist_front_mean` | `err/yaw_deg_mean` | `diag/pallet_disp_xy_mean` | `phase/frac_inserted` | `phase/frac_rg` | `traj/d_traj_mean` | `traj/yaw_traj_deg_mean` |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 300 | `G2b-400` | 0.7078 | 8.6612 | 0.0904 | 0.0156 | 0.0156 | 0.3556 | 14.6078 |
| 300 | `G3-400` | 1.1209 | 7.1154 | 0.0207 | 0.0000 | 0.0156 | 0.2201 | 13.0980 |
| 350 | `G2b-400` | 0.8969 | 8.6061 | 0.0325 | 0.0312 | 0.0156 | 0.2923 | 11.1046 |
| 350 | `G3-400` | 1.3241 | 8.6188 | 0.0373 | 0.0000 | 0.0000 | 0.2466 | 12.2755 |
| 399 | `G2b-400` | 0.8036 | 10.2940 | 0.1040 | 0.0000 | 0.0000 | 0.4034 | 14.6346 |
| 399 | `G3-400` | 1.5674 | 8.2429 | 0.0157 | 0.0000 | 0.0000 | 0.2513 | 11.1662 |

## 对比结论

1. **在 `400 iter` 这一 horizon 下，`G2b` 明显比 `G3` 更能把策略往入口和插入态推进。**  
   `G2b` 的 `err/dist_front_mean` 明显更低，`phase/frac_inserted` 与 `phase/frac_rg` 也稳定高于 `G3`。这说明当 `target_center family` 已经统一到 `s_success_center` 后，再把 trajectory terminal geometry package 同时推到 `success_center`，并没有进一步增强近场推进，反而削弱了它。

2. **`G3` 的优势是“更干净”，不是“更能进去”。**  
   `G3` 在 `err/yaw_deg_mean`、`diag/pallet_disp_xy_mean`、`traj/d_traj_mean` 上都优于 `G2b`，说明它更贴近自己的轨迹走廊、姿态更稳、推盘更少；但这组收益是以明显更弱的前向接近和更弱的插入倾向为代价换来的。

3. **因此，第一轮数据并不支持“只有 `G3` 才能解决主矛盾”这一判断。**  
   如果“trajectory 终点浅 + target_center family 过深的叠加不一致”是唯一主因，那么 `G3` 应该在 `400 iter` 就至少不弱于 `G2b`。但真实结果恰好相反：`G3` 更像把行为约束得更保守、更规整，却没有把策略推进到更强的入口接近状态。

4. **两条方案都还没有形成稳定成功闭环。**  
   在 `iter 300-399` 窗口里，两者的 `phase/frac_success` 都是 `0.0000`。因此本轮不能把 `success` 当赢家判据，只能把它们视作两种不同的几何行为趋势：
   - `G2b`：更有推进性，更像“有机会继续往成功方向长”
   - `G3`：更干净，但当前过于保守

## 一句话结论

**`G2b-400` 是这组对比里更强的候选：它比 `G3-400` 更能接近入口并更频繁进入插入/`rg` 相；`G3-400` 虽然更稳、更少推盘、更贴走廊，但在 `400 iter` 内明显牺牲了推进性，因此第一轮主结论应优先保留 `G2b`，而不是把 `G3` 作为当前最优方案。**

## 对下一步的含义

- 若继续按计划进入 `700-800 iter` 机制确认，**首选应是 `G2b`**。
- `G3` 可以保留为“更保守、更干净”的对照，但不应作为本轮一号候选。
- 由于当前仍是 **single-seed + 400 iter**，最终口径仍应写成「待更长 horizon / 双 seed 进一步确认」。
