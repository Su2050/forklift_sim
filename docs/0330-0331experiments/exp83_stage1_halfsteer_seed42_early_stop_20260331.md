# Exp8.3 Stage1 Half-Steer 训练早停记录

日期：2026-03-31

## 1. 实验目的

这个实验要验证的是：

> 把 `half-steer` 从 frozen-checkpoint eval 里的人工缩放，直接改成训练环境里的真实默认配置后，能不能把这份优势转成训练结果。

对应代码提交：

- IsaacLab `4eee480` `Dampen stage1 steering amplitude`

核心改动：

- `stage1_steer_action_scale = 0.5`

## 2. 固定条件

- 分支：`exp/exp8_3_force_steering_curriculum`
- seed：`42`
- camera：`256x256`
- 训练长度原计划：`50 iter`
- run dir：
  - `IsaacLab/logs/rsl_rl/forklift_pallet_insert_lift/2026-03-31_10-51-47_exp83_stage1_halfsteer_seed42_iter50_256cam`

## 3. 早停前的最新有效指标

在 `iter 22`，最新标量是：

- `phase/frac_inserted = 0.0`
- `phase/frac_inserted_push_free = 0.0`
- `phase/frac_clean_insert_ready = 0.0`
- `phase/frac_hold_entry = 0.0`
- `phase/frac_success = 0.0`
- `diag/preinsert_wrong_sign_abort_frac = 0.0`
- `paper_reward/r_preinsert_wrong_sign_abort = 0.0`
- `diag/preinsert_steer_wrong_sign_frac = 0.375`
- `diag/preinsert_steer_signed_action_mean = 0.1930`
- `paper_reward/preinsert_progress_align_gate = 0.2974`

## 4. 为什么提前停掉

提前停掉的原因不是 crash，也不是训练脚本异常，而是：

- 到 `iter 22` 仍然完全没有长出 insertion
- 和前面的 `wrong-sign abort` 版本相比，这已经明显偏慢
- 继续把它从 `22` 拉到 `50`，大概率只是在继续验证“过于保守”

所以这里的 early-stop 是节省算力，不是中断有效上升趋势。

## 5. 这次失败的真正意义

这次实验很重要，因为它证明了一件容易混淆的事：

> `frozen checkpoint + test-time half-steer 有效`
>
> 不等于
>
> `把 half-steer 直接作为训练默认配置也有效`

也就是说：

- frozen-checkpoint eval 说明：当前 learned policy 的 applied steer 过大，缩小后能减少伤害
- 但 train-time 直接把 steering 默认缩小到 `0.5`，会把训练过程本身压得过于保守，导致 insertion 起不来

所以：

- `0.5` 可能是不错的 deployment / inference 修正
- 但它未必是好的 train-time 默认参数

## 6. 当前最合理的控制变量分流

这次 early-stop 之后，不应该马上继续跑更多训练版本。

更合理的下一步是先做 frozen-checkpoint 的 `steer_scale` 额外扫描，把“推理时最优 scale 区间”再钉清楚。

### 6.1 下一步固定不动

- checkpoint 固定为：
  - `IsaacLab/logs/rsl_rl/forklift_pallet_insert_lift/2026-03-31_09-48-47_exp83_wrong_sign_abort_seed42_iter50_256cam/model_49.pt`
- 3x3 grid 固定：
  - `x_root = -3.40`
  - `y ∈ {-0.10, 0.0, +0.10}`
  - `yaw ∈ {-4°, 0°, +4°}`

### 6.2 下一步只改一个变量

- `steer_scale`

建议优先补：

- `0.25`
- `0.75`

目的：

- 看 frozen-checkpoint 的性能曲线是单调下降、单峰，还是只有 `0.5` 特别有效
- 为下一步 train-time scale 选择提供更可靠的区间判断

## 7. 一句话总结

这次 early-stop 的核心价值是：

> 它把问题进一步拆清楚了：`half-steer` 的好处目前更像“推理时减伤”，还不能直接当作“训练默认配置”；因此下一步应该先做 frozen-checkpoint 的 scale 扫描，而不是立刻再开更多训练长跑。
