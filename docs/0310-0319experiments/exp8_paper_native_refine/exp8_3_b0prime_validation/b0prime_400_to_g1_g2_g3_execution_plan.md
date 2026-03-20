# Exp8.3 B0′-400 完成后到 G1/G2/G3 的执行计划

**日期**：2026-03-20  
**当前正式基线日志**：`logs/20260320_111306_train_exp8_3_b0prime_baseline.log`  
**适用阶段**：`B0′-400 iter` 正式基线跑完后

## 1. B0′-400 跑完后先做什么

在进入 `G1 / G2 / G3` 之前，先固定一版 `B0′-400` 结论，避免后面实验太多后回头补记录。

必须落盘的内容：

- `git rev-parse HEAD`
- 分支名
- 完整训练命令行（actual CLI override）
- 日志文件名
- IsaacLab run 目录
- `iter 300-400` 窗口主指标
- 一句话结论：`B0′` 是否只证明了 build-order 修复，还是已经在几何质量上也有明确改善

## 2. B0′-400 需要抄表的指标

主判据：

- `traj/d_traj_mean`
- `traj/yaw_traj_deg_mean`
- `err/yaw_deg_mean`
- `err/root_lateral_mean`
- `err/center_lateral_mean`
- `err/tip_lateral_mean`
- `err/dist_front_mean`
- `diag/pallet_disp_xy_mean`

辅助现象：

- `phase/frac_inserted`
- `phase/frac_rg`

旁证：

- `phase/frac_success`
- `diag/success_term_frac`
- `phase/frac_aligned`

## 3. 第一轮主矩阵（只跑这三条）

### G1

- **干预类型**：`trajectory-only`
- **唯一改动**：把 trajectory terminal geometry package 改到 `s_success_center`
- **解释口径**：不是纯 endpoint-only，而是 `p_goal` 改动带来的整条 terminal geometry package 变化
- **建议 run_name**：`exp8_3_g1_traj_terminal_s_success_center`

### G2

- **干预类型**：`reward-only`
- **唯一改动**：只改 `r_d` 目标点到 `s_success_center`
- **不改**：`rg`、done 侧 `out_of_bounds`、`r_out`
- **建议 run_name**：`exp8_3_g2_rd_target_s_success_center`

### G3

- **干预类型**：`trajectory-only + reward+done`
- **唯一改动**：统一 `p_goal + target_center family` 到 `s_success_center`
- **解释口径**：这是联合统一实验，不是纯 endpoint-only
- **建议 run_name**：`exp8_3_g3_unify_traj_and_target_family`

## 4. G2b 什么时候补跑

`G2b` 不进第一轮主矩阵，只有在下面条件同时满足时才补跑：

- `G2` 改善不明显
- 你怀疑 `rg / done 侧 out_of_bounds` 也在帮着拉偏

对应口径：

- **干预类型**：`reward+done`
- **说明**：不是纯 reward ablation

## 5. 每条实验开跑前统一检查

每次都按同一套顺序做：

1. 清理旧训练进程与 GPU 残留
2. 执行  
   `bash forklift_pallet_insert_lift_project/scripts/install_into_isaaclab.sh /home/uniubi/projects/forklift_sim/IsaacLab`
3. 记录 `git rev-parse HEAD`
4. 确认 `seed / num_envs / camera / backbone / CLI override` 与 `B0′` 一致
5. 确认日志名遵守  
   `logs/YYYYMMDD_HHMMSS_train_<version>.log`

## 6. 每条实验的统一 horizon

- smoke：`50-100 iter`
- 机制初筛：`400 iter`
- 机制确认：`700-800 iter`

第一轮现在不要扩矩阵，只按：

- `B0′`
- `G1`
- `G2`
- `G3`

## 7. 每条实验的统一判断窗口

- `B0 -> B0′`：只看 `iter 0-50`
- `B0′ / G1 / G2 / G3` 初筛：看 `iter 300-400`
- 入围确认：看 `iter 700-800`

## 8. 第一轮结束后的动作

当 `G1 / G2 / G3` 都跑完 `400 iter` 后：

1. 用 `iter 300-400` 做并排对表
2. 选出最好的 `1-2` 个方案
3. 再把入围方案跑到 `700-800 iter`

## 9. 现在不要做什么

在第一轮主矩阵结论出来之前，先不要进入：

- `G4`
- `G5a`
- `G5b`

原因：

- 这些属于**成功判定 / 终止层**改动
- 会改变训练分布
- 不应该和第一轮几何终点实验混在一起
