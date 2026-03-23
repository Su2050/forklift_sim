# Exp8.3 G2b vs G3 长跑对比验证（800 iter）

**日期**：2026-03-22  
**对比窗口**：`iter 700-799`

**G2b 分支**：`exp/exp8_3_g2b_target_family_success_center`  
**G2b git rev**：`c4a7b2790b8da8d198502f9f6f2091f2cee27e77`  
**G2b 日志**：`forklift_sim_wt_g2b/logs/20260322_093811_train_exp8_3_g2b_target_center_family_success_center_confirm.log`  
**G2b 启动脚本**：`forklift_sim_wt_g2b/scripts/run_exp8_3_g2b_confirm800.sh`  
**G2b IsaacLab run 目录**：`IsaacLab/logs/rsl_rl/forklift_pallet_insert_lift/2026-03-22_09-38-20_exp8_3_g2b_target_center_family_success_center_confirm`

**G3 分支**：`exp/exp8_3_g3_traj_and_target_success_center`  
**G3 git rev**：`cb6c34e0fbb8d96946c18847cbf470d0957e7d3b`  
**G3 日志**：`forklift_sim_wt_g3/logs/20260322_163444_train_exp8_3_g3_unify_traj_and_target_family_confirm.log`  
**G3 启动脚本**：`forklift_sim_wt_g3/scripts/run_exp8_3_g3_confirm800.sh`  
**G3 IsaacLab run 目录**：`IsaacLab/logs/rsl_rl/forklift_pallet_insert_lift/2026-03-22_16-34-53_exp8_3_g3_unify_traj_and_target_family_confirm`

## 对比目的

- `G2b` 的定义是：只把 `target_center family` 统一到 `s_success_center`，trajectory terminal geometry package 仍保持 `front`。
- `G3` 的定义是：在 `G2b` 的基础上，再把 trajectory terminal geometry package 也统一到 `s_success_center`。
- 本文要回答的问题是：**在 `800 iter` 的 confirm horizon 下，`G2b` 是否仍然是主候选，还是 `G3` 在后段更能保住推进性。**

## 实际训练 CLI override

两条 run 共享相同的训练骨架：

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Forklift-PalletInsertLift-Direct-v0 \
  --headless \
  --enable_cameras \
  --num_envs 64 \
  --max_iterations 800 \
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
agent.run_name=exp8_3_g2b_target_center_family_success_center_confirm
env.exp83_traj_goal_mode=front
env.exp83_target_center_family_mode=success_center

# G3
agent.run_name=exp8_3_g3_unify_traj_and_target_family_confirm
env.exp83_traj_goal_mode=success_center
env.exp83_target_center_family_mode=success_center
```

## run 级几何常量

| run | `geom/s_traj_end` | `geom/s_rd_target` | `geom/s_success_center` | 含义 |
| --- | ---: | ---: | ---: | --- |
| `G2b-800` | -1.0800 | -0.8160 | -0.8160 | 只统一 `target_center family`，trajectory 终点仍停在前沿 |
| `G3-800` | -0.8160 | -0.8160 | -0.8160 | trajectory terminal geometry package 与 `target_center family` 全统一到 `success_center` |

## `iter 700-799` 窗口均值对比

| 指标 | `G2b-800` | `G3-800` | 直接读法 |
| --- | ---: | ---: | --- |
| `err/dist_front_mean` | 2.1469 | 1.1089 | `G3` 明显更接近入口 |
| `err/yaw_deg_mean` | 7.9905 | 10.4152 | `G2b` 更稳 |
| `diag/pallet_disp_xy_mean` | 0.0000 | 0.1675 | `G2b` 更干净，`G3` 仍有一定推盘 |
| `phase/frac_inserted` | 0.0000 | 0.0541 | `G3` 仍保留插入倾向，`G2b` 已完全丢失 |
| `phase/frac_rg` | 0.0000 | 0.0214 | `G3` 更常进入 `rg` |
| `phase/frac_success` | 0.0000 | 0.0000 | 两者都未形成稳定 success |
| `traj/d_traj_mean` | 0.3689 | 0.4858 | `G2b` 更贴近自身轨迹走廊 |
| `traj/yaw_traj_deg_mean` | 7.8107 | 10.9538 | `G2b` 更贴近自身轨迹切线 |
| `phase/frac_inserted` 非零计数 | `0 / 100` | `66 / 100` | `G3` 长期保住了推进信号 |
| `phase/frac_rg` 非零计数 | `0 / 100` | `56 / 100` | `G3` 更常触达近场 `rg` 区间 |

## 窗口锚点

| iter | run | `err/dist_front_mean` | `err/yaw_deg_mean` | `diag/pallet_disp_xy_mean` | `phase/frac_inserted` | `phase/frac_rg` | `traj/d_traj_mean` | `traj/yaw_traj_deg_mean` |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 700 | `G2b-800` | 1.4780 | 7.2158 | 0.0000 | 0.0000 | 0.0000 | 0.1149 | 19.5814 |
| 700 | `G3-800` | 0.4805 | 9.5226 | 0.2889 | 0.0469 | 0.0156 | 0.4139 | 10.4471 |
| 750 | `G2b-800` | 2.7395 | 7.1607 | 0.0000 | 0.0000 | 0.0000 | 0.7671 | 2.3208 |
| 750 | `G3-800` | 0.4514 | 12.8261 | 0.2440 | 0.1094 | 0.0625 | 0.5315 | 13.1950 |
| 799 | `G2b-800` | 2.4913 | 7.3715 | 0.0000 | 0.0000 | 0.0000 | 0.4597 | 2.5031 |
| 799 | `G3-800` | 2.8340 | 7.6655 | 0.0000 | 0.0000 | 0.0000 | 0.8704 | 2.3778 |

## 对比结论

1. **长 horizon 下，`G2b` 与 `G3` 的相对排序发生了变化。**  
   在 `400 iter` 的 `300-399` 窗口里，`G2b` 比 `G3` 更会推进；但到了 `800 iter` 的 `700-799` 窗口，`G2b` 已完全失去 `inserted/rg`，而 `G3` 仍保留了稳定的非零推进信号。也就是说，`G2b` 的主问题不是“不够干净”，而是“后段推进性塌缩”。

2. **`G3` 的优势从“更干净”扩大成了“后段更有推进保持力”。**  
   `G3` 虽然在 `diag/pallet_disp_xy_mean` 与 `err/yaw_deg_mean` 上仍不如 `G2b` 干净，但它至少把一部分接近入口、进入 `rg`、触发 `inserted` 的能力保留到了 `700-799`。这说明“把 trajectory terminal geometry package 也统一到 `success_center`”在长 horizon 下并非只是保守化，它对保持近场行为可能是有益的。

3. **但 `G3` 也还不是可以直接放行 `G4` 的方案。**  
   它的 `phase/frac_success` 依旧为 `0`，`diag/pallet_disp_xy_mean` 仍有 `0.1675`，而且 `iter 799` 的单点也已经出现回撤。因此 confirm800 的答案不是“`G3` 已经赢了”，而是“`G3` 比 `G2b` 更像一个还保留推进性的候选，但两者都还不够成熟”。  

4. **因此，这轮 `confirm800` 的真正主结论是：当前不能直接进入 `G4`。**  
   `G2b` 不再满足“主候选应优于 `G3`”的前提；`G3` 虽然在后段更有推进保持力，但也没有形成 success 闭环。下一步更合理的是先解决“长 horizon 下推进性与干净度的折中”，而不是立刻引入 `G4` 的新改动。

## 一句话结论

**`G2b-800` 在后段退成了“干净但不推进”的保守策略，而 `G3-800` 虽然还不够干净，却保住了明显更强的 late-window `inserted/rg` 信号；因此 confirm800 的结果不支持直接进入 `G4`，而是要求先回头处理长 horizon 下的推进性保持问题。**
