# Exp8.3 B0 vs B0+bonus Unified Eval (2026-03-25)

## 1. Purpose

在 `near-field + 256x256 + 50 iter` 口径下，给 `B0 baseline` 和 `B0 + clean_insert_push_free_bonus` 补统一的 checkpoint eval，避免继续只靠训练尾窗日志判断方向。

这次 eval 的目标是回答两个问题：

1. `B0 + bonus` 是否真的比 `B0` 更好，而不只是训练日志看起来更顺眼？
2. 如果答案是“是”，下一步是否应该只扫 `bonus weight`，而不是继续碰强 `gate / penalty`？

## 2. Eval Setup

- 评估脚本: [eval_exp83_checkpoint.py](/home/uniubi/projects/forklift_sim/scripts/eval_exp83_checkpoint.py)
- 批量脚本: [run_exp83_b0_bonus_eval_suite.sh](/home/uniubi/projects/forklift_sim/scripts/run_exp83_b0_bonus_eval_suite.sh)
- 模式: deterministic policy eval
- 环境口径:
  - `stage_1_mode=true`
  - `use_camera=true`
  - `camera_width=256`
  - `camera_height=256`
  - `num_envs=32`
  - `rollouts=2`
- 每个 checkpoint 总共评估 `64` 个 episode

本次统一 eval 的输出目录在 [outputs/exp83_eval](/home/uniubi/projects/forklift_sim/outputs/exp83_eval)。

## 3. Compared Checkpoints

### B0

- [b0_seed42](/home/uniubi/projects/forklift_sim/outputs/exp83_eval/exp83_eval_b0_seed42_summary.json)
- [b0_seed43](/home/uniubi/projects/forklift_sim/outputs/exp83_eval/exp83_eval_b0_seed43_summary.json)
- [b0_seed44](/home/uniubi/projects/forklift_sim/outputs/exp83_eval/exp83_eval_b0_seed44_summary.json)

### B0 + bonus

- [bonus_seed42](/home/uniubi/projects/forklift_sim/outputs/exp83_eval/exp83_eval_bonus_seed42_summary.json)
- [bonus_seed43](/home/uniubi/projects/forklift_sim/outputs/exp83_eval/exp83_eval_bonus_seed43_summary.json)
- [bonus_seed44](/home/uniubi/projects/forklift_sim/outputs/exp83_eval/exp83_eval_bonus_seed44_summary.json)

## 4. Per-Seed Comparison

| run | success_ep | ever_inserted | ever_push_free | ever_dirty | mean_max_disp | mean_max_hold | timeout |
|---|---:|---:|---:|---:|---:|---:|---:|
| b0_seed42 | 0.2188 | 0.9688 | 0.1250 | 0.8750 | 0.4280 | 2.0156 | 0.6094 |
| b0_seed43 | 0.5938 | 0.9844 | 0.3906 | 0.6094 | 0.3773 | 5.4688 | 0.1406 |
| b0_seed44 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.9612 | 0.0000 | 0.0469 |
| bonus_seed42 | 1.0000 | 1.0000 | 0.9688 | 0.0312 | 0.0067 | 9.0000 | 0.0000 |
| bonus_seed43 | 1.0000 | 1.0000 | 0.9219 | 0.0781 | 0.0121 | 9.0000 | 0.0000 |
| bonus_seed44 | 0.0625 | 0.8594 | 0.0469 | 0.8125 | 0.8929 | 0.5625 | 0.9375 |

## 5. Family-Level Aggregate

| family | success_ep mean+-std | ever_inserted mean+-std | ever_push_free mean+-std | ever_dirty mean+-std | mean_max_disp mean+-std | mean_max_hold mean+-std | timeout mean+-std |
|---|---|---|---|---|---|---|---|
| B0 | 0.2708 +- 0.2452 | 0.6510 +- 0.4604 | 0.1719 +- 0.1629 | 0.4948 +- 0.3663 | 0.9222 +- 0.7350 | 2.4948 +- 2.2582 | 0.2656 +- 0.2461 |
| B0+bonus | 0.6875 +- 0.4419 | 0.9531 +- 0.0663 | 0.6458 +- 0.4240 | 0.3073 +- 0.3577 | 0.3039 +- 0.4165 | 6.1875 +- 3.9775 | 0.3125 +- 0.4419 |

## 6. Main Findings

### 6.1 Unified eval changed the previous picture

只看训练尾窗会低估 `B0 + bonus` 的真实表现。

最明显的是 `bonus_seed43`：训练尾窗看起来并不突出，但统一 eval 下却是非常强的 checkpoint：

- `success_rate_ep = 1.0000`
- `ever_inserted_push_free_rate = 0.9219`
- `mean_max_pallet_disp_xy = 0.0121`
- `mean_max_hold_counter = 9.0`

这说明之前“只看最后几轮训练窗口”的判断口径不够稳，统一 eval 更值得作为下一步实验分流依据。

### 6.2 B0 + bonus is clearly better than B0 on average

按统一 eval 的 family mean 看，`B0 + bonus` 在关键指标上明显优于 `B0`：

- `success_rate_ep`: `0.6875` vs `0.2708`
- `ever_inserted_push_free_rate`: `0.6458` vs `0.1719`
- `mean_max_pallet_disp_xy`: `0.3039` vs `0.9222`
- `mean_max_hold_counter`: `6.1875` vs `2.4948`

直观上，这说明轻量 `push_free bonus` 比 `B0` 更能把“插进去”转成“干净插入 + hold + success”。

### 6.3 But variance is still large

`bonus_seed42` 和 `bonus_seed43` 都非常强，但 `bonus_seed44` 明显偏脏插、超时多：

- `success_rate_ep = 0.0625`
- `ever_dirty_insert_rate = 0.8125`
- `timeout_frac = 0.9375`

也就是说，`B0 + bonus` 还不能算已经稳定收敛成可靠配置，只能说它已经比 `B0` 更值得继续做主线。

### 6.4 Strong gate / penalty should stay paused

结合前面做过的单因素、双因素和 `800 iter` 长跑，这次统一 eval 进一步支持一个结论：

- 当前阶段最该继续的是 `B0` 附近的轻量 shaping
- 不该再回去碰强 `gate_r_cpsi`、强 `tight gate package`、强 `dirty penalty`

原因不是这些想法永远错，而是它们在当前实现和当前强度范围内已经表现出“容易把 insertion 一起压掉”。

## 7. Decision

结论是：**可以开始只扫 `bonus weight`，但要保持其余配置不动。**

更准确地说，下一步应该做的是：

1. 固定 `B0` 为主干，不再改别的 reward / gate 项。
2. 只扫 `clean_insert_push_free_bonus_weight`。
3. 每个 weight 继续用相同口径：
   - `near-field`
   - `256x256`
   - `50 iter`
   - `3 seeds`
   - 跑完后做同一套 unified eval
4. 先用统一 eval 选 weight，再决定是否上 `100/200 iter`。

## 8. Recommended Bonus-Weight Sweep

第一轮建议不要扫太宽，只围绕当前 `1.0` 做窄范围扫描：

- `0.5`
- `1.0`
- `1.5`

理由：

- `1.0` 已经证明这条线是活的
- `0.5` 可以检验当前 bonus 是否偏强
- `1.5` 可以检验是否能把 `seed44` 这种脏插 case 往 clean insert 再推一点

不建议一开始就上更大的 weight，也不建议同时再改别的项，否则又会回到归因不清的老问题。

## 9. Current Best Interpretation

到目前为止，最合理的判断是：

**真正有希望的主线不是“更强的抑制 dirty insert”，而是“保留 insertion 动机，再用很轻的 clean bonus 把策略往 push-free / hold / success 拉”。**

统一 eval 已经给出了足够强的证据，说明下一步最值得做的是 `bonus weight` 小范围扫描，而不是继续发明新的 gate 组合。
