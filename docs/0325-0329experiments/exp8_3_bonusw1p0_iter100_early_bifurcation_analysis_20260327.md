# `exp8.3` `bonusw=1.0` `iter100` 早期分叉分析

> 日期: `2026-03-27`
> 分支: `exp/exp8_3_clean_insert_hold`
> 目的: 对比 `r1_seed44` 与 `r2_seed44`、`seed43` 的早期训练差异，找出 run 落入不同 attractor 的分叉点
> 数据源:
> - 统一 eval 总结: `docs/0325-0329experiments/exp8_3_bonusw1p0_repro_iter100_unified_eval_20260327.md`
> - 训练 event:
>   - `r1_seed44`: `IsaacLab/logs/rsl_rl/forklift_pallet_insert_lift/2026-03-27_03-04-32_exp83_bonusw1p0_repro_r1_seed44_iter100_256cam/`
>   - `r2_seed44`: `IsaacLab/logs/rsl_rl/forklift_pallet_insert_lift/2026-03-27_05-44-39_exp83_bonusw1p0_repro_r2_seed44_iter100_256cam/`
>   - `r1_seed43`: `IsaacLab/logs/rsl_rl/forklift_pallet_insert_lift/2026-03-27_02-10-36_exp83_bonusw1p0_repro_r1_seed43_iter100_256cam/`
>   - `r2_seed43`: `IsaacLab/logs/rsl_rl/forklift_pallet_insert_lift/2026-03-27_04-51-37_exp83_bonusw1p0_repro_r2_seed43_iter100_256cam/`

---

## 1. 一页结论

- `bonusw=1.0` 当前不是“平均一般”，而是明显存在多个 attractor。
- `r1_seed44` 走进的是 **clean-insert -> hold -> success** attractor。
- `r2_seed44` 走进的是 **approach 崩掉 -> 远离托盘 -> 永不插入** attractor。
- `r1_seed43` 走进的是 **会插入，但迅速变成 dirty insert** attractor。
- `r2_seed43` 走进的是 **短暂尝试插入 -> 放弃推进 -> 近场徘徊/退回** attractor。

最关键的分叉时间窗不在很后面，而是在 **`iter 10 ~ 15`**:

- `r1_seed44` 在这个窗口第一次出现 `inserted_push_free / hold_entry`，之后保持低位移并继续推进。
- `r2_seed44` 在这个窗口 `dist_front` 和 `d_traj` 直接爆到 `> 1.7 ~ 2.7 m`，说明 approach 已经崩掉。
- `r1_seed43` 在这个窗口虽然高比例插入，但 `aligned` 迅速塌掉，`dirty_insert` 和 `pallet_disp` 同步上升。
- `r2_seed43` 在这个窗口出现过轻微 `push_free / hold`，但很快全部消失，随后进入“近场但不再承诺插入”的保守 attractor。

一句话总结:

**当前的决定性问题不是奖励有没有把正确行为定义出来，而是优化过程在 `iter 10~15` 左右会把 run 推向三种完全不同的 basin: clean success、dirty insert、或者 no-insert。**

---

## 2. 统一 eval 终局对照

这 4 条 run 的统一 eval 终局分别是:

| run | success_ep | ever_inserted | ever_push_free | ever_dirty | timeout | mean_max_disp |
|---|---:|---:|---:|---:|---:|---:|
| `r1_seed44` | `0.9844` | `1.0000` | `0.9375` | `0.0938` | `0.0000` | `0.0489` |
| `r2_seed44` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `0.0000` |
| `r1_seed43` | `0.0000` | `0.7969` | `0.0000` | `0.7969` | `0.7969` | `1.2568` |
| `r2_seed43` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `1.0000` | `0.0000` |

所以早期分叉分析要回答的核心问题其实是:

- 为什么 `r1_seed44` 能把早期的 clean signal 接成 success？
- 为什么 `r2_seed44` 连插入都没有打开？
- 为什么 `seed43` 会更容易落进 dirty/no-insert attractor？

---

## 3. 首次事件对比

| run | first inserted | first push_free | first hold | first success | first dirty |
|---|---:|---:|---:|---:|---:|
| `r1_seed44` | `13` | `13` | `13` | `22` | `16` |
| `r2_seed44` | `None` | `None` | `None` | `None` | `None` |
| `r1_seed43` | `6` | `6` | `6` | `None` | `8` |
| `r2_seed43` | `7` | `7` | `7` | `None` | `7` |

这里有两个非常重要的现象:

- `r1_seed44` 不是“先大量脏插，后面慢慢洗干净”，而是**第一次真正插入时就同时出现 `push_free` 和 `hold`**。
- `seed43` 两条 run 都在很早就摸到过 `push_free / hold`，但都没保住，说明问题不是“完全没有正确信号”，而是**信号太短暂，很快被坏 attractor 吞掉**。

---

## 4. `iter 10 / 15 / 20 / 30` 关键窗口

### 4.1 `r1_seed44`: 好 attractor 在 `iter 13~15` 被锁住

| metric | `iter10` | `iter15` | `iter20` | `iter30` |
|---|---:|---:|---:|---:|
| `frac_inserted` | `0.0000` | `0.0312` | `0.0781` | `0.1562` |
| `frac_inserted_push_free` | `0.0000` | `0.0312` | `0.0156` | `0.0156` |
| `frac_hold_entry` | `0.0000` | `0.0312` | `0.0156` | `0.0312` |
| `frac_dirty_insert` | `0.0000` | `0.0000` | `0.0625` | `0.1406` |
| `frac_success` | `0.0000` | `0.0000` | `0.0000` | `0.0156` |
| `pallet_disp_xy_mean` | `0.0027` | `0.0173` | `0.1009` | `0.0923` |
| `dist_front_mean` | `0.1405` | `0.0869` | `0.0758` | `0.1147` |
| `frac_near_field` | `1.0000` | `1.0000` | `1.0000` | `1.0000` |
| `frac_aligned` | `1.0000` | `0.9844` | `0.9375` | `0.8594` |

解释:

- `iter 10` 时它还没真正插入，但已经稳定处在近场，`dist_front` 很低。
- 到 `iter 15`，**第一次插入就伴随 `push_free + hold`**，这基本就是“进入好 attractor”的标志。
- 后面虽然会出现少量 dirty insert，但 clean signal 始终没丢，所以最终能接出 success。

### 4.2 `r2_seed44`: `iter 10` 就已经崩成 no-insert attractor

| metric | `iter10` | `iter15` | `iter20` | `iter30` |
|---|---:|---:|---:|---:|
| `frac_inserted` | `0.0000` | `0.0000` | `0.0000` | `0.0000` |
| `frac_inserted_push_free` | `0.0000` | `0.0000` | `0.0000` | `0.0000` |
| `frac_hold_entry` | `0.0000` | `0.0000` | `0.0000` | `0.0000` |
| `pallet_disp_xy_mean` | `0.0001` | `0.0000` | `0.0000` | `0.0000` |
| `dist_front_mean` | `2.7550` | `2.3252` | `1.7953` | `2.5817` |
| `frac_near_field` | `0.1875` | `0.2812` | `0.8594` | `0.3281` |
| `frac_aligned` | `0.9844` | `1.0000` | `1.0000` | `1.0000` |
| `traj/d_traj_mean` | `2.1439` | `1.7149` | `1.1857` | `1.9878` |

解释:

- 这条 run 的关键不是“脏插”，而是**approach 本身在 `iter 10` 附近被打飞**。
- `aligned` 很高，但 `dist_front_mean` 和 `d_traj_mean` 同时爆到 `> 2m`，说明它不是姿态坏，而是**离托盘太远、不再承诺前推**。
- 这是一种“看起来很规矩，但根本不插”的 attractor。

最重要的分叉点:

**`r1_seed44` 和 `r2_seed44` 在同一个 seed 下，真正的区别在 `iter 10~15` 的 approach 稳定性，而不是后期 hold 逻辑。**

### 4.3 `r1_seed43`: 不是 no-insert，而是 dirty-insert attractor

| metric | `iter10` | `iter15` | `iter20` | `iter30` |
|---|---:|---:|---:|---:|
| `frac_inserted` | `0.2188` | `0.8125` | `0.8438` | `0.7188` |
| `frac_inserted_push_free` | `0.0312` | `0.0000` | `0.0000` | `0.0000` |
| `frac_hold_entry` | `0.0312` | `0.0000` | `0.0000` | `0.0000` |
| `frac_dirty_insert` | `0.1875` | `0.8125` | `0.8438` | `0.7188` |
| `pallet_disp_xy_mean` | `0.0888` | `0.2140` | `0.2382` | `0.7428` |
| `dist_front_mean` | `0.0203` | `0.0108` | `0.0151` | `0.0454` |
| `frac_near_field` | `1.0000` | `1.0000` | `1.0000` | `1.0000` |
| `frac_aligned` | `0.5469` | `0.1250` | `0.1406` | `0.0000` |
| `traj/d_traj_mean` | `0.1972` | `0.4267` | `0.4524` | `0.8808` |

解释:

- 这条 run 其实非常敢插，而且很早就插得很深。
- 但从 `iter 10 -> 15` 开始，`aligned` 迅速塌掉，`push_free / hold` 同时归零。
- 之后 `dirty_insert` 和 `pallet_disp` 一路上升，说明它已经进入了**“顶着托盘插”的 basin**。

最重要的分叉点:

**`seed43` 的问题不是“不敢接近”，而是“接近之后几乎立刻滑向 dirty insert”。**

### 4.4 `r2_seed43`: 短暂摸到正确信号，但很快退回保守 attractor

| metric | `iter10` | `iter15` | `iter20` | `iter30` |
|---|---:|---:|---:|---:|
| `frac_inserted` | `0.0938` | `0.0312` | `0.0469` | `0.0000` |
| `frac_inserted_push_free` | `0.0156` | `0.0000` | `0.0000` | `0.0000` |
| `frac_hold_entry` | `0.0156` | `0.0000` | `0.0000` | `0.0000` |
| `frac_dirty_insert` | `0.0781` | `0.0312` | `0.0469` | `0.0000` |
| `pallet_disp_xy_mean` | `0.0542` | `0.0387` | `0.0677` | `0.0039` |
| `dist_front_mean` | `0.1147` | `0.0804` | `0.1369` | `0.5724` |
| `frac_near_field` | `1.0000` | `1.0000` | `1.0000` | `1.0000` |
| `frac_aligned` | `0.8281` | `0.9062` | `0.8125` | `0.9688` |
| `traj/d_traj_mean` | `0.1268` | `0.1093` | `0.1050` | `0.1034` |

解释:

- 它和 `r1_seed43` 不一样，不是一路越顶越脏。
- 它在 `iter 10` 一度摸到过一点 `push_free / hold`，但随后完全消失。
- 到 `iter 30`，`inserted = 0`、`disp ≈ 0`，而 `dist_front` 回升到 `0.5724`，更像是**退回到“近场但不继续插”的保守 basin**。

---

## 5. 分叉点总结

当前最关键的早期分叉点可以概括成 3 类:

### 5.1 类型 A: `iter 10~15` 进入 clean basin

代表 run: `r1_seed44`

特征:

- `dist_front_mean < 0.1`
- `frac_near_field = 1.0`
- 第一次 `inserted` 几乎同时伴随 `push_free + hold`
- `pallet_disp_xy_mean` 仍然很低

这类 run 后面大概率会接成 success。

### 5.2 类型 B: `iter 10` 左右 approach 直接崩掉

代表 run: `r2_seed44`

特征:

- `dist_front_mean` 突然飙到 `> 1m`
- `traj/d_traj_mean` 同步飙高
- `near_field` 开始掉
- `inserted / push_free / hold` 长期全零

这类 run 很早就已经没救，不需要等到 `100 iter` 才知道失败。

### 5.3 类型 C: 能接近甚至插入，但 `iter 10~15` 滑向 dirty 或保守 basin

代表 run:

- `r1_seed43`: dirty basin
- `r2_seed43`: 保守 basin

特征:

- 一开始 `dist_front` 其实不差
- 甚至能短暂触发 `push_free / hold`
- 但很快出现下面两种分化之一:
  - `aligned` 崩掉、`dirty_insert` 爆涨、`disp` 增长
  - `inserted` 消失、`disp` 回落、`dist_front` 回升

---

## 6. 对后续实验最有价值的判断

### 6.1 `Train/mean_reward` 不能用来判断 attractor

失败 run 早期 reward 甚至可能更高。例如:

- `r2_seed43` 在 `iter 30` 的 `Train/mean_reward = 260376`
- `r1_seed44` 在 `iter 30` 的 `Train/mean_reward = 86364`

但前者最后完全失败，后者几乎满分。

所以后续如果要做早停筛选，不能看 reward，应该看:

- `frac_inserted_push_free`
- `frac_hold_entry`
- `frac_dirty_insert`
- `diag/pallet_disp_xy_mean`
- `err/dist_front_mean`

### 6.2 `iter 10~15` 已经足够做早期 run triage

从这批 run 看，到了 `iter 10~15` 左右，三种失败/成功模式已经基本可区分:

- `dist_front` 爆掉型: no-insert
- `dirty_insert` 爆掉型: dirty attractor
- `push_free + hold` 保住型: good attractor

这意味着后续可以考虑做更便宜的 early triage:

- 先跑到 `iter 15`
- 用上述诊断指标给 run 打标签
- 再决定哪些 run 值得继续拉长到 `100/200 iter`

### 6.3 `seed43` 应该被当成 stress seed

`seed43` 两次都失败，但失败模式不同，说明它最能暴露系统不稳定性。

后续如果要做“提高复现率”的实验，`seed43` 应该作为固定 stress seed 保留，而不是只看 `seed42/44`。

---

## 7. 下一步建议

基于这份早期分叉分析，最合理的下一步不是继续盲拉长训，而是:

1. 对 `iter 10~15` 建一个自动早期诊断脚本。
2. 把 run 分成:
   - good-clean
   - dirty-insert
   - no-insert / retreat
3. 对 `r1_seed44` vs `r2_seed44` 进一步对比更底层的 actor 输出或 action 统计，确认 `dist_front` 爆掉是“策略后退”还是“转向导致轨迹偏离”。
4. 以后新实验先用 `seed43` + `seed44` 做 stress check，不只看 `seed42`。

一句话说:

**当前最值得做的事情，是把“好 attractor / 脏 attractor / 不插 attractor”在 `iter 10~15` 的分叉信号固化成筛选规则，而不是继续仅靠最终 `success` 事后判断。**
