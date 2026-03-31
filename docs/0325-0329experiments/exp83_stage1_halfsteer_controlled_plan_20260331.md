# Exp8.3 Stage1 Half-Steer 发现与控制变量计划

日期：2026-03-31

## 1. 这轮新发现，先说结论

### 1.1 `wrong-sign abort` 本身不是无效改动，但它不是主解

- 已验证：`preinsert wrong-sign abort` 的逻辑已经真正接上训练，不是死代码。
- 证据：在 `exp83_wrong_sign_abort_smoke_v3` 里，`diag/preinsert_wrong_sign_abort_frac` 和 `paper_reward/r_preinsert_wrong_sign_abort` 都出现过非零。
- 但在完整 `50 iter` 短训里，它带来的主要效果是：
  - 插入更多了；
  - 但主要是 dirty insert；
  - `clean_insert_ready / hold / success` 没有稳定展开。

对应 run：

- `wrong-sign abort` 训练：
  - `IsaacLab/logs/rsl_rl/forklift_pallet_insert_lift/2026-03-31_09-48-47_exp83_wrong_sign_abort_seed42_iter50_256cam`

### 1.2 当前更像“steering 幅度过大”，而不是“完全不该 steering”

在同一个 checkpoint 上做 `3x3` 网格对照后，结果非常关键：

- `normal`
  - summary: `outputs/exp83_wrong_sign_abort_seed42_3x3/exp83_wrong_sign_abort_seed42_iter50_recheck3x3_normal_normal_summary.json`
  - `success_rate = 1/9 = 0.1111`
  - `inserted = 3/9`
  - `push_free = 0/9`
  - `clean_insert_ready = 0/9`
- `zero-steer`
  - summary: `outputs/exp83_wrong_sign_abort_seed42_3x3/exp83_wrong_sign_abort_seed42_iter50_recheck3x3_zero_zero_steer_summary.json`
  - `success_rate = 4/9 = 0.4444`
  - `inserted = 5/9`
  - `push_free = 3/9`
  - `clean_insert_ready = 3/9`
- `half-steer`
  - summary: `outputs/exp83_wrong_sign_abort_seed42_3x3/exp83_wrong_sign_abort_seed42_iter50_recheck3x3_half_half_steer_summary.json`
  - `success_rate = 4/9 = 0.4444`
  - `inserted = 4/9`
  - `push_free = 3/9`
  - `clean_insert_ready = 3/9`
  - `dirty_insert = 1/9`
  - `timeout = 0/9`

这组对照说明：

- `normal << zero-steer`
- 但 `half-steer >> normal`
- 所以当前主问题更像：
  - 不是“策略一转就全错”
  - 而是“当前 applied steer 在 stage1 太猛，很多本来可以 clean 的轨迹被 steering 自己打坏了”

### 1.3 因此下一刀应该先改“applied steer 幅度”，而不是继续扫 reward weight

基于上面的单因素结果，已经把新的环境改成：

- `stage1_steer_action_scale = 0.5`

对应 commit：

- IsaacLab: `4eee480` `Dampen stage1 steering amplitude`

这个改动是最小可解释改动：

- 只动一个变量：`stage1` 下的 steering 幅度
- 不动 reward
- 不动 reset
- 不动 observation
- 不动 hold/success 逻辑

所以它非常适合做下一轮控制变量验证。

## 2. 当前实验链条怎么理解

### 2.1 已经完成并固定下来的结论

#### A. mirrored audit 已经把“target sign 是不是算错了”这件事基本排掉

- 早期 step 上，env steering target 的符号是会在镜像点之间翻转的。
- 但 policy raw steer 在镜像点之间不翻转。
- 这说明早期大问题不是 target 定义错，而是 policy 自己学出了偏置。

对应文档：

- `docs/0325-0329experiments/exp83_force_steering_guidance_v3_seed42_mirrored_audit_20260330.md`

#### B. `wrong-sign abort` 证明了“护栏有用”，但证明不了“steering 学出来了”

- 它能拦住部分持续反号 steering。
- 但不能自动把策略送进 good-clean basin。
- 所以它更像 safety barrier，不是主导解。

#### C. `half-steer` 是这轮最有信息量的对照

- 因为它把问题从“该不该 steering”进一步压缩成：
  - “该 steering，但目前 steering 的 applied gain 太大”

## 3. 当前正在跑的实验

当前正在运行：

- `IsaacLab/logs/rsl_rl/forklift_pallet_insert_lift/2026-03-31_10-51-47_exp83_stage1_halfsteer_seed42_iter50_256cam`

它的意义不是再证明一次能不能插，而是验证：

- 当 `stage1` steering 幅度直接减半后，
- 训练出来的新 checkpoint，
- 在 `3x3 normal vs zero-steer` 下，
- `normal` 是否终于不再明显弱于 `zero-steer`。

## 4. 接下来严格按控制变量怎么做

### 4.1 固定不动的东西

下一轮对比里，以下内容全部固定：

- 分支：`exp/exp8_3_force_steering_curriculum`
- 摄像头：`256x256`
- 训练长度：`50 iter`
- seed：先固定 `seed42`
- reset 分布：保持当前 stage1 配置不变
- reward：保持当前 `wrong-sign abort` 版本不变
- 验收：统一都用同一套 `3x3 normal / zero-steer`
  - `x_root = -3.40`
  - `y in {-0.10, 0.0, 0.10}`
  - `yaw in {-4, 0, 4}`

### 4.2 当前只允许变化的变量

在这一小段实验里，只允许动一个变量：

- `stage1_steer_action_scale`

当前已经完成或正在做的版本：

- `1.0`
  - 已完成
  - 即 `wrong-sign abort` 版本
- `0.5`
  - 已在 checkpoint eval 中验证 `half-steer` 有优势
  - 已在训练环境里接成真实 config
  - 正在跑 `50 iter`

### 4.3 当前这轮的验收标准

`stage1_steer_action_scale = 0.5` 这轮只有在同时满足下面条件时，才算通过：

1. `3x3 normal` 不再明显弱于 `3x3 zero-steer`
2. `normal` 至少达到 `zero-steer` 的 success 水平，最好更高
3. `ever_inserted_push_free` 和 `ever_clean_insert_ready` 必须是非零
4. 训练日志末尾不能只剩 `dirty_insert` 增长

### 4.4 这轮之后的单因素分流

#### 如果 `0.5` 版通过

下一步：

- 不再动 reward
- 不再动 reset
- 直接做 `3 seeds x 50 iter`
- 看结论是不是 seed-stable

#### 如果 `0.5` 版比 `1.0` 好，但仍然 `normal < zero-steer`

下一步只扫 steer scale，不动别的：

- `0.35`
- `0.65`

仍然是：

- 同一分支
- 同一 reward
- 同一 reset
- 同一 seed42
- 同一 `3x3 normal / zero-steer` 验收

这一步的目的，是确认最优 steering gain 大致在哪个区间。

#### 如果 `0.5` 版几乎没有改善

那么下一步不建议继续扫更多 reward weight。

应该转向新的单因素问题：

- 不是“steer 多大”
- 而是“什么时候允许 steering 生效”

那时应切换到下一条控制变量线：

- 固定 gain
- 只改 steering 生效门控

例如：

- 只在 `target_abs` 超过某阈值时允许 steering
- 或只在 near-field / preinsert-active 里允许 steering

但这一步要等 `0.5` 版跑完再决定。

### 4.5 最小执行模板

训练模板：

```bash
cd /home/uniubi/projects/forklift_sim/IsaacLab
env TERM=xterm PYTHONUNBUFFERED=1 CONDA_PREFIX= CONDA_DEFAULT_ENV= \
  ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Forklift-PalletInsertLift-Direct-v0 \
  --headless --enable_cameras \
  --num_envs 64 --max_iterations 50 --seed 42 \
  agent.run_name=<run_name>
```

`3x3 normal` 验收模板：

```bash
cd /home/uniubi/projects/forklift_sim/IsaacLab
env TERM=xterm PYTHONUNBUFFERED=1 CONDA_PREFIX= CONDA_DEFAULT_ENV= \
  ./isaaclab.sh -p ../scripts/eval_exp83_misalignment_grid.py \
  --task Isaac-Forklift-PalletInsertLift-Direct-v0 \
  --headless --enable_cameras \
  --checkpoint <model_49.pt> \
  --label <label>_normal \
  --x_root -3.40 \
  --y_values=-0.10,0.0,0.10 \
  --yaw_deg_values=-4,0,4 \
  --episodes_per_point 1 \
  --output_dir /home/uniubi/projects/forklift_sim/outputs/<output_dir>
```

`3x3 zero-steer` 验收模板：

```bash
cd /home/uniubi/projects/forklift_sim/IsaacLab
env TERM=xterm PYTHONUNBUFFERED=1 CONDA_PREFIX= CONDA_DEFAULT_ENV= \
  ./isaaclab.sh -p ../scripts/eval_exp83_misalignment_grid.py \
  --task Isaac-Forklift-PalletInsertLift-Direct-v0 \
  --headless --enable_cameras \
  --checkpoint <model_49.pt> \
  --label <label>_zero \
  --x_root -3.40 \
  --y_values=-0.10,0.0,0.10 \
  --yaw_deg_values=-4,0,4 \
  --episodes_per_point 1 \
  --force_zero_steer \
  --output_dir /home/uniubi/projects/forklift_sim/outputs/<output_dir>
```

## 5. 现在明确不做的事

为了避免变量爆炸，下面这些现在都不建议先做：

- 不继续扫 bonus weight
- 不继续改 reward 权重
- 不继续改 reset 分布
- 不回 wide reset
- 不动 visual / non-visual 切换
- 不同时改 steering gain 和 steering gate

## 6. 当前最简洁的判断

如果把这一轮的判断压缩成一句话：

> 当前最像的主因已经从“steering sign 完全错了”进一步收缩为“stage1 applied steering 幅度偏大，导致正常 policy 比 zero-steer 更容易把自己打出 clean corridor”；因此现在最合理的下一步是只做 steering gain 的单因素验证，而不是再扩 reward/课程变量。
