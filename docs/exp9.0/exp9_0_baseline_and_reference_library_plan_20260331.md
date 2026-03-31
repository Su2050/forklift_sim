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

## 7. 最终保留的目标是什么

结论：**最终保留下来的目标仍然应该是 `success`，而不是“轨迹像不像 teacher”。**

更准确地说，这条路线应该分成两层目标：

1. 主训练 KPI 继续看任务成功：
   - `phase/frac_success`
   - `push_free_success_rate_total`
   - `push_free_insert_rate_total`
2. 最终验收 KPI 更看重严格成功：
   - `phase/frac_success_strict`
   - `phase/frac_push_free_success`

这里要特别强调：

- teacher/reference 只能是辅助信号
- 不能把“贴近参考轨迹”本身当成最终目标
- 也不建议为了 teacher 方案去修改 success 定义

否则实验最后会变成“更会模仿 teacher”，而不是“更会完成插入任务”。

## 8. 指标分层

建议把所有指标分成三层来读。

### 8.1 最终指标

这层决定方案是否值得保留：

- `push_free_success_rate_total`
- `phase/frac_success`
- `phase/frac_success_strict`
- `phase/frac_push_free_success`

### 8.2 过程指标

这层用于判断模型卡在哪个阶段：

- `push_free_insert_rate_total`
- `phase/frac_inserted`
- `phase/frac_inserted_push_free`
- `phase/frac_aligned`
- `phase/frac_hold_entry`
- `err/yaw_deg_mean`
- `diag/pallet_disp_xy_mean`

### 8.3 解释指标

这层只用于解释，不作为最终结论：

- `traj/d_traj_mean`
- `traj/yaw_traj_deg_mean`
- `paper_reward/r_cd*`
- `paper_reward/r_cpsi*`
- teacher 覆盖率
- teacher 轨迹质量标签

如果某个方案只让解释指标变好，但没有让 success 变好，那这个方案就不应该保留。

## 9. 详细执行计划

### Phase A：建立基线

目标：

- 确认“完全不用参考轨迹”时，在 master 初始分布下能到什么水平

执行：

1. 跑当前 `exp9.0` 的 no-reference baseline。
2. 固定至少 `3` 个 seed，建议 `42 / 43 / 44`。
3. 汇总：
   - `push_free_success_rate_total`
   - `push_free_insert_rate_total`
   - `phase/frac_success`
   - `phase/frac_success_strict`
   - `phase/frac_push_free_success`
   - `err/yaw_deg_mean`
   - `diag/pallet_disp_xy_mean`

产出：

- `A` 组结果：连续初始分布 + no-reference

### Phase B：建立固定 case 底座

目标：

- 把训练分布固定下来，后续所有实验共享同一套 case

执行：

1. 离线生成 `1000` 个初始位姿 case。
2. 每个 case 至少保存：
   - `case_id`
   - `init_x`
   - `init_y`
   - `init_yaw`
   - `difficulty_tag`
3. 不建议纯随机一次生成完就直接用。
4. 建议在 `1000` 个 case 中人为保证 easy / medium / hard 都有覆盖。

建议分层思路：

- easy：较小 `|y| / |yaw|`、较合适 `x`
- medium：中等横偏和偏航
- hard：较大 `|y| / |yaw|` 或更远/更近的 `x`

产出：

- `case_library_v1.json` 或 `case_library_v1.npz`

### Phase C：先验证“离散化本身”有没有价值

目标：

- 把“case 离散化”的收益和“teacher/reference”的收益分开

执行：

1. 用同样的 `1000` case，跑一组不带参考轨迹的训练。
2. reset 时不再连续采样，而是从 case 库随机抽 index。
3. success 定义完全不改。

产出：

- `B` 组结果：离散 `1000 case` + no-reference

判定：

- 如果 `B` 明显好于 `A`，说明固定 case 集本身就有价值
- 如果 `B` 和 `A` 接近，说明后续提升更可能来自 teacher/reference，而不是离散化本身

### Phase D：筛选 teacher checkpoint

目标：

- 找到能稳定产出高质量轨迹的 `master` 模型

执行：

1. 从 `master` 选 `1` 到 `3` 个最强 checkpoint。
2. 用这几个 checkpoint 在同一套 `1000 case` 上离线 rollout。
3. 记录每个 case 的 teacher 结果：
   - `teacher_success`
   - `teacher_push_free`
   - `final_center_err`
   - `final_yaw_err`
   - `pallet_disp_xy`
   - 可选 `trajectory_length`
   - 可选 `min_clearance`

产出：

- `teacher_eval_summary_v1`
- 每个 teacher checkpoint 的 case 级统计表

判定：

- 只保留在固定 case 集上质量最稳定的 checkpoint
- 如果没有 checkpoint 稳定，就先不要继续 teacher 路线

### Phase E：做 teacher 轨迹质量过滤

目标：

- 不把坏 teacher 轨迹带进训练

执行：

1. 为每个 case 设置过滤条件。
2. 第一版建议保守一点，只保留高质量 teacher 样本。

推荐第一版过滤条件：

- `teacher_push_free = true`
- `final_center_err` 过线
- `final_yaw_err` 过线
- `pallet_disp_xy` 足够小

可选：

- 如果 teacher 在某些 case 上明显抖动或绕远，也直接剔除

产出：

- `teacher_case_mask_v1`
- `teacher_good_cases_v1`

判定：

- 如果高质量样本覆盖率太低，比如不到全部 case 的三分之一，teacher 路线应暂停

### Phase F：生成离线 reference library

目标：

- 把 teacher 轨迹一次性缓存下来，训练时直接查表

执行：

1. 对通过过滤的 case，保存：
   - `traj_pts`
   - `traj_tangents`
   - `traj_s_norm`
2. 再保存 metadata：
   - `case_id`
   - `init_x / init_y / init_yaw`
   - `teacher_checkpoint`
   - `teacher_success`
   - `teacher_push_free`
   - `quality_score`
3. 文件格式建议：
   - 主数据：`.npz` 或 `.pt`
   - 元信息：`.json`

产出：

- `reference_library_teacher_v1.npz`
- `reference_library_teacher_v1_manifest.json`

### Phase G：同时做一个“离线几何库”对照组

目标：

- 区分“收益来自缓存/离线化”还是“收益来自 teacher 本身”

执行：

1. 用相同的 `1000 case`，离线生成当前几何 planner 的参考轨迹。
2. 保存成与 teacher library 相同的数据结构。

产出：

- `reference_library_geometric_v1.npz`

这样后续至少能形成三组公平对比：

- `B`：离散 case + no-reference
- `C`：离散 case + offline geometric library
- `D`：离散 case + offline teacher library

### Phase H：接入环境开关

目标：

- 在同一套 env 中支持多种 reference 来源

建议新增配置：

- `reset_sampling_mode = continuous | discrete_case_library`
- `reference_source = none | geometric_online | offline_geometric_library | offline_teacher_library`
- `case_library_path`
- `reference_library_path`

要求：

- success 逻辑不改
- hold 逻辑不改
- 只改 reset 数据来源和 reference corridor 来源

### Phase I：先做最小改动版训练

目标：

- 先验证 teacher/reference 是否真的能提高 success

执行：

1. 第一版不要改 actor 输入。
2. 只复用现有 `r_cd / r_cpsi` 的接口。
3. 也就是让 teacher/library 只作为 reward corridor，而不是额外监督头。

这样做的好处：

- 变量最少
- 结论最容易归因
- 如果没收益，能更快止损

### Phase J：正式实验矩阵

至少做下面四组：

1. `A`：连续分布 + no-reference
2. `B`：离散 `1000 case` + no-reference
3. `C`：离散 `1000 case` + offline geometric library
4. `D`：离散 `1000 case` + offline teacher library

所有组统一要求：

- 相同 seed 集
- 相同训练步数
- 相同成功定义
- 相同日志统计口径

### Phase K：统一验收标准

建议按下面顺序判断：

1. 先看 `push_free_success_rate_total`
2. 再看 `phase/frac_success_strict`
3. 再看 `push_free_insert_rate_total`
4. 最后才看轨迹相关解释指标

判断规则：

- 如果 `D > C > B`，说明 teacher 真正有增益
- 如果 `C ≈ D > B`，说明离线 reference 库有价值，但 teacher 不是关键
- 如果 `B ≈ C ≈ D`，说明参考轨迹不是主矛盾，应回到 reward / obs / action / curriculum
- 如果 `D` 只提高插入率，不提高 `push_free_success`，说明 teacher 可能在鼓励脏插入，这条线不能直接保留

## 10. Stop/Go 门槛

为了避免 teacher 路线无限投入，建议明确 stop/go 门槛。

### 继续推进（Go）

满足任一条即可继续：

1. `D` 相比 `B`，`push_free_success_rate_total` 有稳定提升
2. `D` 相比 `C`，严格成功指标也有稳定提升
3. teacher 高质量覆盖率足够高，且多 seed 下趋势一致

### 暂停或终止（Stop）

满足任一条就应暂停：

1. teacher 高质量覆盖率过低
2. `D` 只提升插入率，不提升 success
3. 多 seed 下 `D` 不稳定，只有单 seed 偶然好
4. `C` 与 `D` 几乎一致，teacher 额外复杂度不值得

## 11. 推荐实施顺序

如果按性价比排序，我建议这样落地：

1. 跑完 `A`
2. 做 `1000 case` 库
3. 跑 `B`
4. 做 teacher checkpoint 离线审核
5. 做 teacher 质量过滤
6. 生成 `C` 和 `D` 两套离线 reference 库
7. 接入环境开关
8. 跑 `C`
9. 跑 `D`
10. 只按 success 结论决定是否保留 teacher 路线

## 12. 一句话原则

**这条方案最终留下来的标准，不是“teacher 轨迹生成得出来”，也不是“corridor loss 更漂亮”，而是“能否稳定提升 success，尤其是 push-free success”。**
