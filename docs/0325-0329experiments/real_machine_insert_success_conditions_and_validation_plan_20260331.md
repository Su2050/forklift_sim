# 叉车“自动插入托盘（不举升）”成功条件与验证总计划（基于 `forklift_pallet_insert_lift_project`）

> 版本日期：2026-03-31  
> 适用范围：`forklift_pallet_insert_lift_project` 目录内实现（含 `isaaclab_patch`、`tests`、`scripts`）  
> 说明：本文只引用该项目目录内已有代码、配置、测试和脚手架；对于尚未在项目内落地的能力，会明确标注“需新增”。

---

## 0. 先统一一句话目标（避免训练方向跑偏）

**当前项目真正训练的目标（Stage 1 / no-lift）**：

在当前定义的近场初始扰动范围内，叉车通过视觉 actor + 特权 critic 的训练配置，稳定完成“对齐 + 插入（不举升）”，并且其成功判定、观测契约、轨迹几何和 reward/curriculum 接线彼此一致。

**注意**：

- 本项目当前默认不是“最终实机发布口径”，而是一个 `stage1_success_without_lift=True` 的阶段性任务定义；
- 因此，所有计划与验收都必须先以 `env_cfg.py`、`env.py`、`hold_logic.py` 为准，不能直接拿 `README.md` 里较早期的 insert+lift KPI 当成当前默认口径。

---

## 1. “成功”定义（不举升版）

你最担心的是“什么叫成功会不会判断错”。在这个项目里，这件事必须拆成 3 层，而且第一层要**完全对齐当前代码实现**。

### 1.1 当前项目默认的单回合成功（以代码实现为准）

当前 `stage1` 默认配置来自：

- `isaaclab_patch/.../env_cfg.py`
- `isaaclab_patch/.../env.py`
- `isaaclab_patch/.../hold_logic.py`

在当前项目里，单个 episode 的“成功”应理解为：

1. **当前阶段不要求举升**：`stage1_success_without_lift=True`；
2. **插入深度达标**：`insert_depth >= insert_fraction * pallet_depth_m = 0.40 * 2.16 = 0.864 m`；
3. **中心对齐达标**：`center_y_err <= 0.15 m` 且 `yaw_err <= 8.0 deg`；
4. **近场 tip 约束达标**：当 `dist_front <= 2.2 m` 时，还要求 `tip_y_err <= 0.12 m`；
5. **保持时间达标**：`hold_counter >= hold_steps`，其中 `hold_steps = int(hold_time_s / ctrl_dt)`，当前 `hold_time_s = 0.33 s`，`ctrl_dt = sim.dt * decimation = 1/120 * 4 = 1/30 s`；
6. **去抖与退出逻辑生效**：不是简单“一步不满足就清零”，而是使用 `hysteresis_ratio=1.2`、`insert_exit_epsilon=0.02`、`lift_exit_epsilon=0.08`、`hold_counter_decay=0.8`；
7. **终止条件未触发异常**：没有 `tipped`、没有 `out_of_bounds`，也没有直接因 timeout 结束。

这意味着：

- 当前项目的 no-lift success 不是“深度 + lateral + yaw + hold”这么简单；
- 它还明确依赖 `tip gate`、`hysteresis`、`grace zone` 和 `hold counter decay`；
- 所以文档里若继续写“3 cm / 3° / 0.8~1.0 s”这样的口径，会和当前项目默认实现不一致。

### 1.2 指标口径注意事项（当前项目里非常重要）

当前环境日志里已经写出了：

- `phase/frac_success`
- `phase/frac_success_strict`
- `phase/frac_success_geom_strict`
- `phase/frac_push_free_success`

但这些目前仍是**step 级均值**，不是标准 episode success rate。

因此，在当前项目范围内：

- 还不能直接把 `phase/frac_success=...` 解释成“90% episode 成功率”；
- 也不能只凭这些 step mean 就给 checkpoint 贴“可部署”标签；
- 如果后面要写 `SR >= 90%` 这种门槛，必须先补一个 episode 级评估口径或专门 evaluator。

### 1.3 Checkpoint 成功（建议口径）

在当前项目里，checkpoint 是否算“过门禁”，建议按以下顺序收紧：

1. **先过实现一致性门禁**：几何、obs 契约、success 逻辑、runtime sanity 全绿；
2. **再看固定域内行为指标**：`frac_inserted`、`frac_hold_entry`、`frac_clean_insert_ready`、`frac_success_strict`、`frac_push_free_success` 是否一起变好；
3. **最后再补 episode 级评估器**：固定种子、固定工作域、固定 checkpoint，输出真正的 episode success / timeout / out_of_bounds / push-free success。

在 episode 级 evaluator 尚未补齐前，不建议把“`SR >= 90%`”写成当前项目的硬门禁。

---

## 2. 成功所需条件总表（只基于当前项目）

下面把条件整理成 8 类。每一类都只引用当前项目目录里已经有的实现或明确缺口。

## C1. 几何定义与成功几何口径一致

**必须满足**：

- `insert_depth` 的坐标系定义与托盘 yaw 旋转下的投影一致；
- `pallet_front` 方向、`success_center`、`target_center_family`、`traj_goal` 这些几何定义彼此不打架；
- 文档中的成功口径与 `env_cfg.py` 当前默认值一致。

**当前项目里怎么验证**：

- 跑 `tests/test_exp83_geometry_preflight.py`；
- 开启 `exp83_runtime_u0_enable=true`、`exp83_runtime_u1_enable=true` 做 sanity run；
- 关注 `env.py` 启动时写出的 `geom/s_traj_end`、`geom/s_rd_target`、`geom/s_success_center`。

**当前项目内可用入口**：

- `forklift_pallet_insert_lift_project/tests/test_exp83_geometry_preflight.py`
- `forklift_pallet_insert_lift_project/isaaclab_patch/.../env_cfg.py` 中的 `exp83_runtime_u0_*`
- `forklift_pallet_insert_lift_project/isaaclab_patch/.../env_cfg.py` 中的 `exp83_runtime_u1_*`

**通过标准（示例）**：

- 预检脚本通过；
- runtime U0/U1 在 sanity run 中不 fail-fast；
- `insert_norm` 在 `success_center` 下与 `insert_fraction=0.40` 对上。

**常见误判**：

- 继续沿用 `README.md` 里“2/3 插入 + 举升 + 3cm/3°”的旧 KPI；
- 以为参考轨迹终点、target center、success center 是同一个东西。

---

## C2. 当前训练工作域是明确的，而且先只在这个域里验证

**必须满足**：

- 当前训练不是“全域叉车插托盘”，而是 `stage1` 的近场工作域；
- 初始随机化范围要被当作当前任务定义的一部分，而不是默认忽略。

**当前项目里怎么验证**：

- 直接以 `env_cfg.py` 中 `stage1_init_x/y/yaw_*` 为当前工作域定义；
- 用固定 seed 的训练/回放先验证这个窄域，而不是先扩域；
- 如果后面要扩域，先补 dedicated sweep 工具，再谈“可达性”。

**当前项目事实**：

- 当前默认工作域大致是：
- `x in [-3.60, -3.45]`
- `y in [-0.08, 0.08]`
- `yaw in [-3 deg, 3 deg]`

**通过标准（示例）**：

- 团队明确承认“这就是当前训练域”；
- 在这个域内先把 success logic、obs、trajectory、reward 接线跑通；
- 没有在当前项目里把“窄域成功”误写成“全域可部署成功”。

**常见误判**：

- 把当前项目的 `stage1` 结果直接外推到更大偏航、更大横向误差或更远距离；
- 在没有 sweep/eval 工具时就讨论“理论可达率 95%”。

---

## C3. 参考轨迹与 target-center family 接线正确

**必须满足**：

- reset 后生成的参考轨迹端点、切向和托盘轴定义一致；
- 轨迹入口几何真实存在，也就是不能出现 `fork_center start` 已经跑到 `p_pre` 前方的情况；
- `exp83_traj_goal_mode`、`exp83_target_center_family_mode` 与 reward / out_of_bounds / rg 接线一致；
- 轨迹不是“看起来合理”，而是和当前 `env.py` 的几何定义同源。

**需要特别追踪的已知发现**：

- 前几天的 runtime 可视化已经明确暴露过一个关键问题：
- 在旧的 `stage1 v2` 近场课程里，`fork_center start` 在托盘轴向坐标上系统性落在 `p_pre` 前方，典型表现为 `delta_s = s_start - s_pre > 0`；
- 当前项目后来通过“把 `stage1_init_x_*` 后移 + 把 `traj_pre_dist_m` 调到 `1.05`”来尝试修正这个问题；
- 但这件事**不能因为配置已经改过就默认视为解决**，必须重新做回归验证。

**当前项目里怎么验证**：

- `test_exp83_geometry_preflight.py` 的 U0 预检；
- 开启 `exp83_runtime_u0_enable=true` 做运行时轨迹 sanity；
- 开启 `exp83_runtime_u1_enable=true` 做 `target_center_family` 接线 sanity。
- 跑 `forklift_pallet_insert_lift_project/scripts/visualize_reference_trajectory_cases.py`，生成代表性 case 的 top-down PNG 和 manifest；
- 人工查看 `fork_center start / p_pre / p_goal / trajectory` 的相对位置，而不是只看数值。

**通过标准（示例）**：

- 轨迹弧长单调；
- 起点、终点、查询点 yaw 误差都在预期容差内；
- 代表性 case 中都满足 `s_start < s_pre < s_goal`；
- manifest 中 `delta_s = s_start - s_pre` 全部为负；
- 至少要人工复核：
- `overlay_all_cases.png`
- 中心 case 单图
- 边界 case 单图（例如最小 x / 最大 y / 最大 yaw）
- `front` / `success_center` / `front_center` 这些模式切换时，相关日志和行为变化符合设计。

**常见误判**：

- 以为 runtime U0 通过，就已经证明 trajectory entry 合理；
- 以为“轨迹画出来了”就等于 reward / done 也跟它一致；
- 只看自动数值，不看 top-down 图；
- 忘了 `traj_goal` 和 `target_center_family` 是两套相关但不完全相同的接线。

---

## C4. 观测、动作、时序链路与当前项目契约一致

**必须满足**：

- 视觉 actor 的输入真的是 `image + proprio`；
- critic 真的是 15 维 privileged obs；
- `stage1` 动作空间语义明确，不举升不是“忽略 lift”，而是环境会自动补零并屏蔽它；
- 相机 shape、数值范围、NaN/Inf 保护都符合当前代码。

**当前项目里怎么验证**：

- 跑 `scripts/test_camera_obs_contract.py`；
- 直接核对 `_get_camera_image()`、`_get_easy8()`、`_get_observations()`；
- 核对 `_pre_physics_step()` 与 `_apply_action()` 的 stage1 行为。

**当前项目事实**：

- `policy` 看到的是 `image + proprio`；
- `critic` 看到的是 15 维低维状态；
- `action_space=2`，但环境内部会自动补齐第 3 维 lift；
- `stage1` 下 lift 会被强制为 0；
- 插入足够深后，drive/steer 也会被抑制。

**通过标准（示例）**：

- `test_camera_obs_contract.py` 通过；
- 相机图像 shape 正确，且无 NaN/Inf；
- 训练日志和动作语义与代码一致，没有“以为是 3 动作，其实 actor 只在学 2 动作”的理解偏差。

**常见误判**：

- 仍按低维 state-only 或 3 动作 lift-enabled 的旧直觉理解当前训练；
- 把 `phase/frac_success` 的变化归因到 lift，而当前 stage1 实际并不要求 lift 进入 success。

---

## C5. Reward/Curriculum 接线没有把行为带偏

**必须满足**：

- clean insert gate、preinsert shaping、trajectory reward、rg/out_of_bounds 等机制一致；
- 不会出现“指标好看，但实际只是更会顶盘/擦盘/撞盘”的情况；
- 课程收紧的前提是前一阶段已经稳定，而不是在 success 还没打通时继续扩难度。

**当前项目里怎么验证**：

- 以 `env_cfg.py` 中当前 reward/curriculum 开关为主线做审计；
- 结合训练日志观察：
- `phase/frac_inserted`
- `phase/frac_hold_entry`
- `phase/frac_clean_insert_ready`
- `phase/frac_success_strict`
- `phase/frac_push_free_success`
- `diag/pallet_disp_xy_mean`
- `err/tip_lateral_inserted_mean`

**当前项目缺口**：

- 项目内还没有专门的 reward audit CLI；
- 目前主要依赖训练日志、sanity 回放和人工对照。

**通过标准（示例）**：

- 插入、hold、clean insert、strict success、push-free success 是同向改善；
- 托盘位移和 dirty insert 没有在“成功提升”的同时恶化。

**常见误判**：

- 只看某个 reward 分量变大；
- 只看 `phase/frac_success` 而不看 `clean_insert_ready` 与 `push_free_success`。

---

## C6. Success Logic 必须以 `hold_logic.py` 为唯一事实来源

这是当前项目里最不能模糊的一项。

**必须满足**：

- 文档、配置、代码三者对 success 的描述一致；
- `tip gate`、`hysteresis`、`grace zone`、`hold decay` 都被视为 success logic 的正式组成部分；
- `stage1_success_without_lift=True` 的语义写清楚，不要和 README 旧版本定义混用。

**当前项目里怎么验证**：

- 直接看 `hold_logic.py`；
- 跑 `tests/test_exp83_geometry_preflight.py` 中的：
- `test_phase1_hold_logic_skips_lift_but_keeps_tip_gate`
- `test_hold_logic_hysteresis_and_decay`

**通过标准（示例）**：

- 预检测试通过；
- 文档中的阈值与 `env_cfg.py` 一致；
- 没有人再把 README 中“2/3 + 1s + 3cm/3° + lift”当作当前 stage1 默认 success。

**当前项目特别提醒**：

- 环境里虽然已经有 `_ep_success_count` / `_ep_total_count` 变量，但当前项目内还没有把 episode success 作为标准日志输出；
- 因此，在这一步里，除了校公式，还要明确“哪些是 step metric，哪些未来才是 episode metric”。

---

## C7. 训练可复现性：当前项目已有基础，但还没完全闭环

**必须满足**：

- seed 真能从命令行进到环境；
- 每次 run 的 `env.yaml`、`agent.yaml` 会被落盘；
- 代码版本能随训练一起追溯；
- 如果要宣称 multi-seed 稳定，必须先补一个项目内 batch 驱动。

**当前项目里怎么验证**：

- `train.py` 支持 `--seed`；
- 训练时会 dump `params/env.yaml` 和 `params/agent.yaml`；
- runner 会写 git repo state 到日志。

**当前项目缺口**：

- 项目内目前没有现成 multiseed batch shell；
- 也没有项目内的标准 checkpoint eval suite。

**通过标准（示例）**：

- 单次 run 能明确还原：代码版本、seed、env 配置、agent 配置；
- 在需要 multi-seed 结论前，先补最小的 seed-loop 脚本或明确人工执行流程。

**常见误判**：

- 只凭单 seed 好结果就宣称“这个配置稳定”；
- 没有保存 `env.yaml/agent.yaml` 就开始横向比较不同实验。

---

## C8. 实机闭环与安全：当前项目里还主要停留在计划层

**必须满足**：

- 未来实机部署时，观测契约要和当前训练契约一致；
- 控制周期要与当前项目 `ctrl_dt = 1/30 s` 对齐；
- no-lift 实机测试也必须显式说明 lift 接口如何处理；
- 安全 SOP、人工接管、急停、速度上限要在项目外单独落地。

**当前项目里怎么处理**：

- 这部分目前更多是**后续工程门禁**，不是项目内现成脚本；
- 当前项目能提供的是：训练定义、obs/action 契约、success logic 和基本日志口径；
- 真正的实机 adapter、SOP、硬件级 fail-safe 仍需新增。

**通过标准（示例）**：

- 在进入实机前，先把项目内的 stage1 定义冻结；
- 再单独输出实机观测映射、控制映射、安全清单；
- 不直接拿当前项目训练日志替代实机安全验证。

**常见误判**：

- 以为仿真里 stage1 成功，就已经等于“真实机器可以跑”；
- 在没有实机观测映射文档的情况下直接部署视觉策略。

---

## 3. 分阶段执行计划（按当前项目真实能力重排）

## Phase A：定义冻结与口径澄清（1 天）

**执行**：

1. 以 `env_cfg.py + env.py + hold_logic.py` 为当前事实来源；
2. 明确写出当前 stage1 no-lift success 口径；
3. 明确区分：
   - 当前 step 级日志指标；
   - 未来需要补的 episode 级评估指标；
4. 在文档中明确：`README.md` 里的旧 KPI 不作为当前 stage1 默认成功定义。

**验收**：

- 团队对“当前默认 success”无歧义；
- 没有人再混用旧 README KPI 与当前 cfg 口径。

---

## Phase B：几何与轨迹预检（1~2 天）

**执行**：

1. 跑：
   `python forklift_pallet_insert_lift_project/tests/test_exp83_geometry_preflight.py`
2. 做一次启用 `exp83_runtime_u0_enable=true` 的 sanity run；
3. 做一次启用 `exp83_runtime_u1_enable=true` 的 sanity run；
4. 跑：
   `python forklift_pallet_insert_lift_project/scripts/visualize_reference_trajectory_cases.py`
5. 记录：
   - `geom/s_traj_end`
   - `geom/s_rd_target`
   - `geom/s_success_center`
   - 可视化 manifest 中的 `delta_s_min / delta_s_max / delta_s_mean`
6. 由人人工复核 `overlay_all_cases.png` 和几个代表性单图，确认 `fork_center start` 没有重新跑到 `p_pre` 前方。

**门禁**：

- 如果 U0/U1 预检不过，禁止进入长训；
- 如果 `delta_s` 仍有正值，禁止进入长训；
- 如果 top-down 图上看起来轨迹入口仍然不存在，禁止进入长训；
- 如果 runtime sanity fail-fast，先修 geometry/trajectory 接线。

---

## Phase C：观测/动作契约与 reward 接线审计（1~2 天）

**执行**：

1. 跑：
   `python forklift_pallet_insert_lift_project/scripts/test_camera_obs_contract.py`
2. 核对当前 stage1 的 obs/action 语义：
   - actor 是 `image + proprio`
   - critic 是 `critic(15d)`
   - action space 外部 2 维、内部补 lift 维
3. 做一轮短 sanity train 或 play，重点看：
   - `phase/frac_inserted`
   - `phase/frac_hold_entry`
   - `phase/frac_clean_insert_ready`
   - `phase/frac_success_strict`
   - `phase/frac_push_free_success`

**门禁**：

- obs contract 未通过，不进入训练；
- 如果 reward 让 dirty insert / push pallet 上升，先修接线，不扩训练。

---

## Phase D：有记录的 seeded training（2~5 天）

**执行**：

1. 用 `train.py --seed ...` 启动训练；
2. 每个 run 保留 `env.yaml`、`agent.yaml`、git state；
3. 先在当前窄域里训练，不先扩 `stage1_init_*`；
4. 如果要做 multi-seed，对当前项目先补最小 seed-loop 驱动。

**门禁**：

- 在 episode evaluator 未补齐前，不把 step mean 当最终成功率；
- 在当前窄域未稳定前，不扩域。

---

## Phase E：补 episode evaluator，再谈 checkpoint gate（后续）

**执行**：

1. 为当前项目补一个 episode 级 evaluator；
2. 固定 checkpoint、固定 seeds、固定工作域输出真正的：
   - episode success
   - timeout
   - out_of_bounds
   - push-free success
3. 只有到这一步，才适合讨论 `SR >= xx%` 的 checkpoint 门槛。

**门禁**：

- 没有 episode evaluator，就不把“可复现高成功率”写成结论。

---

## Phase F：实机准备（独立工程项）

**执行**：

1. 设计真实机器的观测映射，确保与当前项目 obs 契约一致；
2. 设计 no-lift 部署时的动作映射与安全约束；
3. 建立急停、人工接管、限速、录像复核 SOP；
4. 先低速、空场、窄域，再逐步扩域。

**门禁**：

- 当前项目内训练通过，不等于可以跳过这一步。

---

## 4. 你现在就能做的“最小可执行清单”（只用当前项目）

1. 跑：
   `python forklift_pallet_insert_lift_project/tests/test_exp83_geometry_preflight.py`
2. 跑：
   `python forklift_pallet_insert_lift_project/scripts/test_camera_obs_contract.py`
3. 跑：
   `python forklift_pallet_insert_lift_project/scripts/visualize_reference_trajectory_cases.py`
4. 先人工看：
   - `overlay_all_cases.png`
   - 至少 3 张代表性单图
   再确认当前是否真的满足 `s_start < s_pre < s_goal`
5. 把当前 success 定义从 `env_cfg.py + hold_logic.py` 摘出来冻结成单页说明；
6. 做一次打开 `exp83_runtime_u0_enable=true` 和 `exp83_runtime_u1_enable=true` 的 sanity run；
7. 再用显式 `--seed` 启动训练，并保存 `env.yaml / agent.yaml`。

如果前 1~6 任一步不通过，不建议继续“多训几天看看”。

---

## 5. 真实机器相关的特别提醒（基于当前项目现状）

1. **当前项目默认控制周期约 30 Hz**：`ctrl_dt = 1/120 * 4 = 1/30 s`。
2. **当前项目默认是 no-lift stage1 success**：如果上实机先做“不举升插入”，部署接口也要保持这一点，不要让软件口径和硬件行为分裂。
3. **视觉策略不是只喂一张图就行**：当前 actor 需要 `image + proprio`，critic 虽然只用于训练，但它反映了项目对状态定义的真实口径。
4. **当前项目还没有实机 episode evaluator**：所以实机前必须单独做结果判定工具，不能直接复用 `phase/frac_success`。
5. **当前项目还没有实机安全脚手架**：急停、人工接管、限速、录像留存、失败复盘都需要单独工程化。

---

## 6. 建议的文档拆分方式（贴合当前项目）

建议至少维护以下 4 份：

1. `success_definition_no_lift.md`
   当前 `stage1` 成功定义，直接抄自 `env_cfg.py + hold_logic.py`
2. `preflight_and_runtime_sanity_report.md`
   记录 `test_exp83_geometry_preflight.py`、camera obs contract、runtime U0/U1 结果
3. `seeded_training_report.md`
   记录 seed、`env.yaml`、`agent.yaml`、关键阶段指标
4. `episode_evaluator_plan.md`
   记录如何把当前 step metrics 升级成真正的 checkpoint gate

---

## 7. 一句话总结

如果只看 `forklift_pallet_insert_lift_project`，你现在最需要做的不是“继续训更久”，而是先把**当前 stage1 no-lift 的成功定义、几何/轨迹预检、obs/action 契约、success logic 和 seed 追溯**这五件事做成硬门禁；只有这些口径先统一，后面的训练结果才有资格被认真解释。
