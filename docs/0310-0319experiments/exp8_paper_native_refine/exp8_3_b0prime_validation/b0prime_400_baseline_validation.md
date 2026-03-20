# Exp8.3 B0′ 正式基线验证（400 iter）

**日期**：2026-03-20  
**分支**：`exp/exp8_3_geom_validation_b0prime`  
**git rev**：`d741c915257b47fda9527802bd06df3bbabe7337`  
**日志**：`logs/20260320_111306_train_exp8_3_b0prime_baseline.log`  
**启动脚本**：`scripts/run_exp8_3_b0prime_baseline.sh`  
**IsaacLab run 目录**：`IsaacLab/logs/rsl_rl/forklift_pallet_insert_lift/2026-03-20_11-13-15_exp8_3_b0prime_baseline`

## 运行说明

- 本次 run 采用 `B0′` 基线代码：只修 reference trajectory build-order / fresh reset tensor 接线，不改 trajectory / `r_d` / success 终点定义。
- 日志按 runner 的 **0-based iteration** 记法输出，最终 summary 停在 `iter 399/400`；结合训练进程已退出，可视为本次 `400 iter` 正式基线已完成。

## 实际训练 CLI override

核心 override 与 `scripts/run_exp8_3_b0prime_baseline.sh` 一致：

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Forklift-PalletInsertLift-Direct-v0 \
  --headless \
  --enable_cameras \
  --num_envs 64 \
  --max_iterations 400 \
  agent.run_name=exp8_3_b0prime_baseline \
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

## `iter 300-399` 窗口锚点

| iter | `traj/d_traj_mean` | `traj/yaw_traj_deg_mean` | `err/yaw_deg_mean` | `err/root_lateral_mean` | `err/center_lateral_mean` | `err/tip_lateral_mean` | `err/dist_front_mean` | `diag/pallet_disp_xy_mean` | `s_center_mean` | `phase/frac_inserted` | `phase/frac_rg` | `phase/frac_success` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 300 | 1.0567 | 4.0011 | 7.9502 | 0.2619 | 0.2410 | 0.2697 | 3.0533 | 0.0000 | -4.7255 | 0.0000 | 0.0000 | 0.0000 |
| 320 | 0.9906 | 4.1590 | 8.7903 | 0.3252 | 0.2561 | 0.2761 | 2.9577 | 0.0000 | -4.6279 | 0.0000 | 0.0000 | 0.0000 |
| 340 | 0.8718 | 3.9325 | 7.6955 | 0.2465 | 0.2530 | 0.2817 | 2.8946 | 0.0000 | -4.5677 | 0.0000 | 0.0000 | 0.0000 |
| 360 | 0.8206 | 3.6230 | 7.7918 | 0.2609 | 0.2543 | 0.2816 | 2.8608 | 0.0000 | -4.5325 | 0.0000 | 0.0000 | 0.0000 |
| 380 | 0.8007 | 4.4150 | 8.5274 | 0.2973 | 0.2855 | 0.3097 | 2.8058 | 0.0000 | -4.4764 | 0.0000 | 0.0000 | 0.0000 |
| 399 | 0.9779 | 4.5021 | 8.7002 | 0.2761 | 0.2667 | 0.3025 | 2.9094 | 0.0000 | -4.5801 | 0.0000 | 0.0000 | 0.0000 |

## 窗口结论

1. **`B0′` 作为新基线是干净且稳定的。**  
   `traj/d_traj_mean` 在 `iter 300-399` 窗口维持在约 `0.80-1.06`，`traj/yaw_traj_deg_mean` 维持在约 `3.62-4.50°`，没有出现旧 `B0` 那种 build-order 污染导致的起步错轨迹主导现象。

2. **但 `B0′` 并没有学会真正接近入口。**  
   `err/dist_front_mean` 在 `iter 300-399` 仍维持约 `2.81-3.05m`，`s_center_mean` 维持约 `-4.73 ~ -4.48`，而当前几何常量是：
   - `geom/s_traj_end = -1.0800`
   - `geom/s_rd_target = -0.4800`
   - `geom/s_success_center = -0.8160`

   这说明策略长期停留在托盘前沿很远的位置，并没有推进到参考轨迹终点附近，更没有进入 success 等效深度附近。

3. **`B0′` 也没有进入任何有效插入/成功相。**  
   整份日志中未观察到非零的：
   - `phase/frac_inserted`
   - `phase/frac_rg`
   - `phase/frac_success`

   因此，这次 `400 iter` run 的结论不是“build-order 修复后任务已解决”，而是“build-order 修复后，训练不再被错轨迹污染，但仍卡在远场 approach”。

4. **当前策略没有表现出推盘恶化，但这不是成功信号。**  
   `diag/pallet_disp_xy_mean` 全窗口维持 `0.0000`，说明没有明显把托盘推走；但结合 `err/dist_front_mean` 与 `s_center_mean`，更合理的解释是“它根本还没有有效触达入口”，而不是“已经学会了高质量无推盘插入”。

## 一句话结论

**`B0′-400` 证明了 build-order 修复足以建立一个干净、稳定、不推盘的新基线；但它没有解决几何终点不一致问题，策略依然停留在远场 approach，未产生任何插入/成功信号。**

## 对第一轮主矩阵的意义

- 后续第一轮主矩阵应继续按计划只跑：`G1 / G2 / G3`
- 其中：
  - `G1`：只改 trajectory terminal geometry package 到 `s_success_center`
  - `G2`：只改 `r_d` 目标到 `s_success_center`
  - `G3`：统一 `p_goal + target_center family` 到 `s_success_center`
- `G4 / G5a / G5b` 继续暂缓，不与第一轮几何终点实验混跑

## 直接下一步

- 先进入 `G1`：只改 trajectory terminal geometry package，先跑 `50-100 iter` smoke
- 若 smoke 正常，再跑 `G1-400 iter` 机制初筛
