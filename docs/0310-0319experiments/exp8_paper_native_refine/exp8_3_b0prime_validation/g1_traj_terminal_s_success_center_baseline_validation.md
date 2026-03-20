# Exp8.3 G1 正式基线验证（400 iter）

**日期**：2026-03-20  
**分支**：`exp/exp8_3_geom_validation_b0prime`  
**任务代码 git rev**：`7e1de7bca5a69b2817568bc5431f56aaba08b087`  
**日志**：`logs/20260320_171930_train_exp8_3_g1_traj_terminal_s_success_center_baseline.log`  
**启动脚本**：`scripts/run_exp8_3_g1_baseline.sh`  
**IsaacLab run 目录**：`IsaacLab/logs/rsl_rl/forklift_pallet_insert_lift/2026-03-20_17-19-39_exp8_3_g1_traj_terminal_s_success_center_baseline`

## 运行说明

- 本次 run 采用 `G1` 代码：只改 trajectory terminal geometry package 到 `s_success_center`，不改 `r_d` 目标点，不改 `rg` / done 侧 `out_of_bounds`，也不改 success / hold gate。
- 日志按 runner 的 **0-based iteration** 记法输出，最终 summary 停在 `iter 399/400`；当前训练进程已退出，因此可视为本次 `400 iter` 正式基线已完成。
- 启动脚本 `scripts/run_exp8_3_g1_baseline.sh` 在本次 run 时是工作区文件；任务代码本身对应的 git 快照为上面的 `git rev`。

## 实际训练 CLI override

核心 override 与 `scripts/run_exp8_3_g1_baseline.sh` 一致：

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Forklift-PalletInsertLift-Direct-v0 \
  --headless \
  --enable_cameras \
  --num_envs 64 \
  --max_iterations 400 \
  agent.run_name=exp8_3_g1_traj_terminal_s_success_center_baseline \
  env.use_camera=true \
  env.use_asymmetric_critic=true \
  env.stage_1_mode=false \
  env.camera_width=256 \
  env.camera_height=256 \
  env.exp83_traj_goal_mode=success_center \
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
| 300 | 0.4822 | 15.4194 | 14.4890 | 0.3739 | 0.4278 | 0.5450 | 0.6258 | 0.2041 | -2.1662 | 0.0469 | 0.0000 | 0.0000 |
| 320 | 0.5678 | 12.5347 | 13.5874 | 0.4155 | 0.4867 | 0.5881 | 0.5278 | 0.2378 | -2.0543 | 0.0312 | 0.0000 | 0.0000 |
| 340 | 0.6376 | 15.4749 | 15.2162 | 0.4352 | 0.5260 | 0.6654 | 0.4003 | 0.2677 | -1.8944 | 0.1094 | 0.0000 | 0.0000 |
| 360 | 0.5586 | 13.2872 | 16.7812 | 0.4210 | 0.4277 | 0.5639 | 0.4965 | 0.3170 | -1.9767 | 0.1094 | 0.0000 | 0.0000 |
| 380 | 0.4597 | 13.0732 | 12.2697 | 0.3233 | 0.4041 | 0.5096 | 0.6211 | 0.2217 | -2.1224 | 0.0938 | 0.0000 | 0.0000 |
| 399 | 0.5479 | 13.0326 | 15.6732 | 0.4078 | 0.4628 | 0.5909 | 0.5962 | 0.2800 | -2.1454 | 0.0469 | 0.0000 | 0.0000 |

## 与 `B0′-400` 的直接对照

相对 `B0′-400`，`G1-400` 在 `iter 300-399` 窗口里表现出如下稳定差异：

- **更接近入口**：
  - `G1 err/dist_front_mean ≈ 0.40-0.63m`
  - `B0′ err/dist_front_mean ≈ 2.81-3.05m`
- **沿托盘轴更靠前**：
  - `G1 s_center_mean ≈ -2.17 ~ -1.89`
  - `B0′ s_center_mean ≈ -4.73 ~ -4.48`
- **持续出现部分插入**：
  - `G1 phase/frac_inserted ≈ 0.03-0.11`
  - `B0′ phase/frac_inserted = 0.0000`
- **但姿态/横向/推盘副作用更大**：
  - `G1 err/yaw_deg_mean ≈ 12.27-16.78°`
  - `B0′ err/yaw_deg_mean ≈ 7.70-8.79°`
  - `G1 err/center_lateral_mean ≈ 0.40-0.53`
  - `B0′ err/center_lateral_mean ≈ 0.24-0.29`
  - `G1 err/tip_lateral_mean ≈ 0.51-0.67`
  - `B0′ err/tip_lateral_mean ≈ 0.27-0.31`
  - `G1 diag/pallet_disp_xy_mean ≈ 0.20-0.32`
  - `B0′ diag/pallet_disp_xy_mean = 0.0000`

## 运行期现象补充

1. **`phase/frac_inserted` 并非偶发单点，而是贯穿 run 的持续非零现象。**  
   从早中期开始，`phase/frac_inserted` 多次出现非零，峰值达到 `0.1562`。这与 `B0′-400` 的“全程 0”形成了清晰对照。

2. **`phase/frac_rg` 仍全程为 0。**  
   即使 `G1` 已能稳定进入部分插入，`rg` 依然没有被触发，说明仅改 trajectory terminal geometry package 还不足以满足当前 `target_center family + yaw/tip lateral` 的联合门控。

3. **`phase/frac_success` / `diag/success_term_frac` 出现过两次极稀疏非零。**  
   在约 `iter 24` 和 `iter 62` 附近，日志各出现一次：
   - `phase/frac_success = 0.0156`
   - `diag/success_term_frac = 0.0156`

   但这些信号没有持续到 `iter 300-399` 决策窗口，因此不能据此判定 `G1` 已经学成稳定成功策略。

## 窗口结论

1. **`G1` 明确验证了：trajectory terminal geometry package 过浅，是 `B0′` 卡在远场的重要原因之一。**  
   只改 trajectory 终点后，策略就从 `B0′` 的“长期远场徘徊”变成了“稳定接近入口并持续出现部分插入”。

2. **但 `G1` 同时稳定带来了更大的偏航、横向误差和推盘。**  
   在 `iter 300-399` 窗口里，`err/yaw_deg_mean`、`err/center_lateral_mean`、`err/tip_lateral_mean`、`diag/pallet_disp_xy_mean` 全都显著高于 `B0′`，说明当前策略更像是“更愿意进去，但进去得更歪、更推盘”。

3. **因此 `G1` 不是最终解，而是第一条强机制证据。**  
   它证明轨迹终点浅不是边角问题，而是主矛盾之一；但剩余副作用说明“轨迹终点”之外仍有未统一的目标在拉策略，最直接的候选就是 `r_d` 与 `target_center family`。

4. **`G1` 没有形成稳定成功闭环。**  
   最终窗口中：
   - `phase/frac_inserted` 仍非零
   - `phase/frac_rg = 0.0000`
   - `phase/frac_success = 0.0000`

   这说明它解决的是“能不能进去”的一部分，而不是“能不能以正确姿态、低推盘地稳定达成任务”。

## 一句话结论

**`G1-400` 正式确认：把 trajectory terminal geometry package 推到 `s_success_center` 后，策略会显著更接近入口并持续出现部分插入；但同时会稳定带来更大的偏航、横向误差和推盘，且没有形成稳定 `rg/success`，因此 `G1` 是强正向证据，但不是最终方案。**

## 对下一步的意义

- `G1` 已经足够证明“轨迹终点过浅”是主矛盾之一
- 下一步应继续按计划推进：
  - **`G2`**：只改 `r_d` 目标点到 `s_success_center`
  - **`G3`**：统一 `p_goal + target_center family`
- 若 `G2` 不明显，再补 **`G2b`** 检查 `rg + done侧 out_of_bounds` 是否共同在拉偏

当前不建议跳过 `G2 / G3` 直接把 `G1` 当最终解，因为 `G1-400` 已经清楚表明：**轨迹终点改深后，入口接近问题缓解了，但“进去的质量”问题仍然严重。**
