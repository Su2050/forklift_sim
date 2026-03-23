# Exp8.3 G2b vs G3 strict diagnostics 对比验证（200 iter）

**日期**：2026-03-23  
**工作树分支**：`exp/exp8_3_geom_validation_b0prime`  
**git rev**：`c6484b49321daf8101d72b6a5ebca492eb8aeb69`  
**对比窗口**：`iter 150-199`

**说明**：本次不是直接在 `forklift_sim_wt_g2b / forklift_sim_wt_g3` 中跑 confirm，而是在当前主工作树中补入 `strict success diagnostics` 后，通过 profile override 复现 `G2b / G3` 两种几何设置，回答“在更严格的诊断口径下，下一步更该沿哪条线继续”。

**启动脚本**：`scripts/run_exp8_3_strict_diag_compare.sh`

**G2b 日志**：`/home/uniubi/projects/forklift_sim/logs/20260323_130004_train_exp8_3_strict_diag_compare_g2b_iter200.log`  
**G2b IsaacLab run 目录**：`/home/uniubi/projects/forklift_sim/IsaacLab/logs/rsl_rl/forklift_pallet_insert_lift/2026-03-23_13-00-13_exp8_3_strict_diag_compare_g2b_iter200`

**G3 日志**：`/home/uniubi/projects/forklift_sim/logs/20260323_144523_train_exp8_3_strict_diag_compare_g3_iter200.log`  
**G3 IsaacLab run 目录**：`/home/uniubi/projects/forklift_sim/IsaacLab/logs/rsl_rl/forklift_pallet_insert_lift/2026-03-23_14-45-31_exp8_3_strict_diag_compare_g3_iter200`

## 对比目的

- `G2b`：`traj_goal=front`，`target_center_family=success_center`
- `G3`：`traj_goal=success_center`，`target_center_family=success_center`
- 在当前主工作树里，新增了 `strict diagnostics`：
  - `phase/frac_inserted_z_valid`
  - `phase/frac_center_aligned_cfg`
  - `phase/frac_tip_constraint_ok`
  - `phase/frac_success_geom_strict`
  - `phase/frac_success_strict`
  - `phase/frac_push_free_success`
- 本文要回答的问题是：**strict 口径是否改变我们对 `G2b` vs `G3` 的排序判断，以及 low-level 漏洞主要暴露在 z-valid、near-field alignment，还是 lift/hold 闭环。**

## 实际训练 CLI override

两条 run 共享相同的训练骨架：

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Forklift-PalletInsertLift-Direct-v0 \
  --headless \
  --enable_cameras \
  --num_envs 64 \
  --max_iterations 200 \
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
agent.run_name=exp8_3_strict_diag_compare_g2b_iter200
env.exp83_traj_goal_mode=front
env.exp83_target_center_family_mode=success_center

# G3
agent.run_name=exp8_3_strict_diag_compare_g3_iter200
env.exp83_traj_goal_mode=success_center
env.exp83_target_center_family_mode=success_center
```

## run 级几何常量

| run | `geom/s_traj_end` | `geom/s_rd_target` | `geom/s_success_center` | 含义 |
| --- | ---: | ---: | ---: | --- |
| `G2b-200` | -1.0800 | -0.8160 | -0.8160 | trajectory 仍停在前沿，但 `target_center family` 已切到 `success_center` |
| `G3-200` | -0.8160 | -0.8160 | -0.8160 | trajectory 与 `target_center family` 全统一到 `success_center` |

## `iter 150-199` strict 尾窗均值对比

| 指标 | `G2b-200` | `G3-200` | 直接读法 |
| --- | ---: | ---: | --- |
| `err/dist_front_mean` | 0.5996 | 0.5786 | `G3` 略更接近入口 |
| `err/yaw_deg_mean` | 11.1133 | 12.1573 | `G2b` yaw 略更稳 |
| `diag/pallet_disp_xy_mean` | 0.2266 | 0.2162 | `G3` 略更干净 |
| `phase/frac_inserted` | 0.0387 | 0.0659 | `G3` 保留了更强的插入信号 |
| `phase/frac_inserted_z_valid` | 0.0387 | 0.0659 | `G3` 在 z-valid 口径下仍更强，排序未反转 |
| `phase/frac_center_aligned_cfg` | 0.1444 | 0.1291 | `G2b` 在 center/yaw 严格对齐上略好 |
| `phase/frac_tip_constraint_ok` | 0.1722 | 0.1706 | 两者几乎持平 |
| `phase/frac_success_geom_strict` | 0.0056 | 0.0066 | `G3` 略多触发严格几何成功 |
| `phase/frac_lifted_enough` | 0.0000 | 0.0000 | 两者都没有 lift closure |
| `phase/frac_success_strict` | 0.0000 | 0.0000 | 两者都没有 strict success |
| `phase/frac_push_free` | 0.6309 | 0.6519 | `G3` 反而略更 push-free |
| `phase/frac_push_free_success` | 0.0000 | 0.0000 | 两者都没有 push-free strict success |
| `phase/frac_rg` | 0.0259 | 0.0309 | `G3` 更常进入 `rg` |
| `traj/d_traj_mean` | 0.4498 | 0.4523 | 两者近似持平，`G2b` 略贴近自身轨迹 |
| `traj/yaw_traj_deg_mean` | 12.4191 | 12.7373 | `G2b` 略贴近自身轨迹切线 |

## 尾窗非零计数

| 指标 | `G2b-200` | `G3-200` | 直接读法 |
| --- | --- | --- | --- |
| `phase/frac_inserted > 0` | `47 / 50` | `50 / 50` | `G3` 的插入信号更连续 |
| `phase/frac_inserted_z_valid > 0` | `47 / 50` | `50 / 50` | z-valid 口径下排序不变 |
| `phase/frac_success_geom_strict > 0` | `17 / 50` | `18 / 50` | 两者都偶尔触达严格几何成功，`G3` 略多 |
| `phase/frac_success_strict > 0` | `0 / 50` | `0 / 50` | 两者都没有 strict success 闭环 |
| `phase/frac_push_free_success > 0` | `0 / 50` | `0 / 50` | 两者都没有 push-free strict success |
| `phase/frac_rg > 0` | `42 / 50` | `47 / 50` | `G3` 更常触达近场 `rg` |
| `diag/max_hold_counter > 0` | `1 / 50` | `0 / 50` | `G2b` 只有一次短暂 hold blip，未形成闭环 |

补充说明：

- `G2b` 尾窗 `diag/max_hold_counter` 最大值为 `8.0`
- `G3` 尾窗 `diag/max_hold_counter` 最大值为 `0.0`
- 两者全程 `phase/frac_success` 窗口均值都为 `0.0`

## 窗口锚点

| iter | run | `err/dist_front_mean` | `diag/pallet_disp_xy_mean` | `phase/frac_inserted_z_valid` | `phase/frac_success_geom_strict` | `phase/frac_rg` | `traj/d_traj_mean` |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 50 | `G2b-200` | 0.5755 | 0.1267 | 0.0938 | 0.0000 | 0.0312 | 0.4568 |
| 50 | `G3-200` | 0.6159 | 0.1627 | 0.0469 | 0.0156 | 0.0312 | 0.4154 |
| 100 | `G2b-200` | 0.6199 | 0.2027 | 0.0469 | 0.0000 | 0.0156 | 0.4377 |
| 100 | `G3-200` | 0.6054 | 0.1632 | 0.0781 | 0.0000 | 0.0156 | 0.4410 |
| 150 | `G2b-200` | 0.4866 | 0.2203 | 0.0469 | 0.0156 | 0.0312 | 0.4425 |
| 150 | `G3-200` | 0.6000 | 0.1790 | 0.0312 | 0.0000 | 0.0156 | 0.3693 |
| 199 | `G2b-200` | 0.5996 | 0.1676 | 0.0469 | 0.0000 | 0.0312 | 0.5216 |
| 199 | `G3-200` | 0.5234 | 0.2174 | 0.0469 | 0.0000 | 0.0000 | 0.4662 |

## 结果解读

1. **strict diagnostics 没有推翻“更适合沿 `G3` 继续”的方向判断。**  
   尾窗 `150-199` 中，`G3` 在 `frac_inserted / frac_inserted_z_valid / frac_rg` 上都略优，而且 `diag/pallet_disp_xy_mean` 与 `phase/frac_push_free` 也没有比 `G2b` 更差，说明“统一 trajectory 到 success_center”在 strict 口径下依旧更像值得继续的底座。

2. **z-valid 不是当前 `G2b / G3` 排序差异的主因。**  
   两条 run 的 `phase/frac_inserted` 与 `phase/frac_inserted_z_valid` 在尾窗里几乎完全同速同序，说明这次 strict compare 并没有看到“raw inserted 很高、但 z-valid 后大幅塌掉”的情况。也就是说，当前 `G2b` 的核心问题不是 z-valid 漏洞，而更像是几何推进保持力本身较弱。

3. **真正暴露出来的瓶颈在 strict success 的后半段，而不是 insert 本身。**  
   两条 run 都能偶尔打出 `phase/frac_success_geom_strict > 0`，但 `phase/frac_lifted_enough` 全程为 `0`，于是 `phase/frac_success_strict` 与 `phase/frac_push_free_success` 全程都为 `0`。这说明目前 `200 iter` compare 更像是在比较“approach + near-field geometry quality”，还没有进入“insert-lift 闭环已经打通”的阶段。

4. **当前 active success 也没有闭环，只是 `G2b` 在尾窗里出现过一次短暂 hold blip。**  
   `phase/frac_success` 的尾窗均值两者都是 `0`。`G2b` 尾窗里有一次 `diag/max_hold_counter=8` 的瞬时非零，但距离真正的 hold-based success 还差得很远；`G3` 在尾窗里则连这种短暂 hold 都没有保住。这个现象说明当前 reward/done 口径下，策略还没形成稳定的近场 hold 行为。

5. **因此，这轮 strict compare 的主结论不是“成功标准已经够严”，而是“strict 指标已经足以支持我们在 `G3` 上继续排查”。**  
   下一步如果目标是继续比较几何底座，应当沿 `G3` 继续；如果目标是尽快逼近真正 success 闭环，则应当在 `G3` 上优先检查 why `lifted_enough` 永远为 `0`，以及当前 active success 与 strict success 的分工是否需要重新整理。

## 一句话结论

**在 `strict diagnostics + 200 iter` 的同工作树对比里，`G3` 依旧比 `G2b` 更能稳定保住 `inserted_z_valid / rg` 信号，而且没有更脏；真正卡住两者的不是 z-valid，而是 strict success 的后半段 (`lifted_enough / hold` 闭环) 仍然完全没有建立，因此下一步更适合沿 `G3` 继续，而不是回到 `G2b`。**
