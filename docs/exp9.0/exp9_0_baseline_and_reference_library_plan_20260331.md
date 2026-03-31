# Exp9.0 Baseline And Reference Library Plan

## 1. 当前分支改动

`exp/exp9_0` 先做了两件事：

1. 把 Stage 1 的初始 `x / y / yaw` 范围改回 `master` 的原始分布：
   - `x ∈ [-2.5, -1.0]`
   - `y ∈ [-0.6, 0.6]`
   - `yaw ∈ [-0.25, 0.25] rad ≈ ±14.3239°`
2. 新增 `env.use_reference_trajectory` 开关：
   - `false` 时 reset 不生成轨迹
   - `r_cd / r_cpsi` 不参与 reward
   - `traj/*` 日志置零

这意味着“无参考轨迹训练”现在是真正的 ablation，而不是只把权重调成 0 但仍然重复生成轨迹。

## 2. 基准怎么跑

直接运行：

```bash
bash scripts/run_exp90_no_reference_baseline.sh
```

可选环境变量：

```bash
SEED=42 MAX_ITERATIONS=400 NUM_ENVS=64 bash scripts/run_exp90_no_reference_baseline.sh
```

脚本默认：

- `env.use_reference_trajectory=false`
- `env.alpha_2=0.0`
- `env.alpha_3=0.0`
- 保留当前 clean-insert / preinsert shaping
- 使用新的 master 风格初始位姿范围

因此这个 baseline 测到的是：

“在更宽初始分布下，只靠当前视觉观测 + 非轨迹 shaping，策略能学到什么程度”。

建议重点盯这些指标：

- `push_free_success_rate_total`
- `push_free_insert_rate_total`
- `phase/frac_inserted`
- `err/dist_front_mean`
- `err/yaw_deg_mean`

## 3. 用 master 训练好的模型生成参考轨迹，可不可行

结论：**可行，但更适合做离线 teacher/reference library，不建议每次训练时在线生成。**

原因：

1. `master` 模型可以提供“在给定初始位姿下，策略大概率会怎么走”的经验路径，这能当作 teacher signal。
2. 但直接在线 rollout 生成有两个明显问题：
   - reset 时会重复跑很多遍，训练时间被轨迹生成吃掉
   - teacher 一旦在某些初始位姿上本身不稳定，就会把噪声直接注入训练
3. 更稳妥的用法是：
   - 先固定一批初始位姿
   - 用 `master` 模型离线 rollout 一次
   - 只保留质量通过的轨迹
   - 训练时直接查表读取

也就是说，`master` 模型更适合做“离线数据生产器”，不适合做“在线实时参考轨迹服务”。

## 4. 初始位置做成离散选项，可不可行

结论：**非常可行，而且这条路我认为比在线生成更合理。**

推荐设计：

1. 先离线生成 `N=1000` 个初始位姿 case。
2. 每个 case 保存：
   - `case_id`
   - `init_x / init_y / init_yaw`
   - `traj_pts`
   - `traj_tangents`
   - `traj_s_norm`
   - 可选质量标签，比如 `teacher_success / push_free / min_clearance`
3. 训练 reset 时不再做连续采样，而是从 `1000` 个 case 里采样一个索引。
4. 选中 case 后，直接把对应轨迹拷到 env cache。

这样有几个直接好处：

1. 轨迹只生成一次，不在训练里重复算。
2. 训练分布完全可复现，A/B 实验更干净。
3. 可以对 case 做分层采样，比如 easy / medium / hard。
4. 可以提前把坏轨迹过滤掉，不让 teacher 噪声污染训练。

## 5. 这套离散库的成本其实很低

按当前配置：

- `traj_num_samples = 21`
- 每个 sample 存 `pts(2) + tangents(2) + s_norm(1) = 5` 个 float

粗略内存：

- 单条轨迹：`21 * 5 * 4 bytes = 420 bytes`
- `1000` 条轨迹：约 `420 KB`

即使再加上一些 metadata，也远小于图像数据和训练日志的体量。

所以从工程角度看，“1000 个离散初始位姿 + 预生成轨迹库”几乎没有存储压力，主要工作量在于：

- 定义 case 采样分布
- 设计离线生成脚本
- 做 teacher 轨迹质量过滤

## 6. 建议推进顺序

建议按下面的顺序走：

1. 先跑 `exp9.0` 无参考轨迹 baseline，确认纯视觉 + 非轨迹 shaping 的上限。
2. 如果 baseline 明显掉得厉害，再做“离散初始位姿 + 离线轨迹库”版本。
3. 轨迹库优先用离线生成，不要在训练时重复生成。
4. `master` 模型可以作为轨迹库来源之一，但最好加质量过滤；如果 teacher 不稳，可以改成几何规划器或 teacher+planner 混合方案。

我的当前判断是：

- “在线生成参考轨迹”不优
- “离散初始位姿 + 预生成轨迹库”很值得做
- `master` 模型可以参与生成，但应该是离线、可审计、可过滤的方式
