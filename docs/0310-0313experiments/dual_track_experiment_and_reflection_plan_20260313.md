---
name: 双线实验与反思机制计划
overview: 先建立共同基座与失败复盘规则，再把 master 势函数作为对照线、论文绝对状态 Reward 作为主线，按单因素和课程化方式推进。
todos:
  - id: audit-common-base
    content: 先做共同基座审计，统一 backbone、冻结策略、动作空间、成功判据和指标口径
    status: pending
  - id: audit-reward-gate-consistency
    content: 审计 rg、is_inserted、hold_cond、push_free 的一致性，防止出现奖励触发但真实成功未发生
    status: pending
  - id: create-reflection-rule
    content: 创建 .cursor/rules/experiment-reflection.mdc，并固化失败后全链路复盘 checklist
    status: pending
  - id: create-branches
    content: 创建并切换到对应的 Git 分支（分支 A: exp/master_potential_approach_only，分支 B: exp/vision_cnn/paper_native_reward_v2）
    status: pending
  - id: branch-a-master-control
    content: 建立 master 势函数对照线，只回滚 reward family，不回滚已验证必要的底层修复
    status: pending
  - id: branch-b-paper-mainline
    content: 建立论文 Reward 主线，并先校准到真实的 ResNet18 + ImageNet + 全程冻结基座
    status: pending
  - id: define-curriculum-and-kpi
    content: 为论文 Reward 主线制定 rg 阈值课程、统一 KPI 和分支判废规则
    status: pending
isProject: false
---

# 双线实验与失败复盘计划

## 1. 目标

- 建立一条 `master 势函数对照线` 和一条 `论文 Reward 主线`，但两条线必须先共享同一个经过审计的共同基座。
- 任何实验未达预期，都必须先完成全链路复盘，再决定是否继续调参或换方向。

## 2. Step 0：共同基座审计

- 代码落点统一为 [forklift_pallet_insert_lift_project/isaaclab_patch/source/isaaclab_tasks/isaaclab_tasks/direct/forklift_pallet_insert_lift/env.py](/home/uniubi/projects/forklift_sim/forklift_pallet_insert_lift_project/isaaclab_patch/source/isaaclab_tasks/isaaclab_tasks/direct/forklift_pallet_insert_lift/env.py)、[forklift_pallet_insert_lift_project/isaaclab_patch/source/isaaclab_tasks/isaaclab_tasks/direct/forklift_pallet_insert_lift/env_cfg.py](/home/uniubi/projects/forklift_sim/forklift_pallet_insert_lift_project/isaaclab_patch/source/isaaclab_tasks/isaaclab_tasks/direct/forklift_pallet_insert_lift/env_cfg.py)、[forklift_pallet_insert_lift_project/isaaclab_patch/source/isaaclab_tasks/isaaclab_tasks/direct/forklift_pallet_insert_lift/vision_actor_critic.py](/home/uniubi/projects/forklift_sim/forklift_pallet_insert_lift_project/isaaclab_patch/source/isaaclab_tasks/isaaclab_tasks/direct/forklift_pallet_insert_lift/vision_actor_critic.py)、[forklift_pallet_insert_lift_project/isaaclab_patch/source/isaaclab_tasks/isaaclab_tasks/direct/forklift_pallet_insert_lift/agents/rsl_rl_ppo_cfg.py](/home/uniubi/projects/forklift_sim/forklift_pallet_insert_lift_project/isaaclab_patch/source/isaaclab_tasks/isaaclab_tasks/direct/forklift_pallet_insert_lift/agents/rsl_rl_ppo_cfg.py)。
- 为了“回归 master”只允许回滚 `reward family`，不允许回滚已经证实必要的底层修复，如几何、buffer 初始化、done/reset、日志键同步等实现修复。
- 统一非 reward 变量：`2 动作 approach-only`、`success 不含 lift`、`critic=15 维 privileged state`、相机方案、统一日志指标、统一评估视频导出。
- 先审计启动脚本与配置是否真的一致：[run_experiment_2_rrl.sh](/home/uniubi/projects/forklift_sim/run_experiment_2_rrl.sh) 显式覆盖了 `agent.policy.backbone_type="resnet18"`，但 [run_experiment_4_paper_native.sh](/home/uniubi/projects/forklift_sim/run_experiment_4_paper_native.sh) 当前没有这条 override，而 [forklift_pallet_insert_lift_project/isaaclab_patch/source/isaaclab_tasks/isaaclab_tasks/direct/forklift_pallet_insert_lift/agents/rsl_rl_ppo_cfg.py](/home/uniubi/projects/forklift_sim/forklift_pallet_insert_lift_project/isaaclab_patch/source/isaaclab_tasks/isaaclab_tasks/direct/forklift_pallet_insert_lift/agents/rsl_rl_ppo_cfg.py) 默认仍是 `freeze_backbone_updates=50`。计划执行前必须先把“实际 backbone/冻结策略”校准成单一真相。
- 输出物：一份简短 audit 结论，明确两条线共享什么、只允许哪一个因素不同。

## 2.1 Step 0.5：奖励门控与真实成功一致性审计

- 在任何继续调参之前，先单独核对 `paper_reward/rg`、`is_inserted`、`hold_cond`、`push_free_success_rate`、`push_free_insert_rate` 的逻辑关系。
- 目标是避免出现“`rg` 已明显触发，但 `phase/frac_inserted` 仍为 0，或托盘位移持续扩大”的假突破。
- 若发现奖励门槛与真实插入/保持条件不一致，则优先修正判据和日志，再继续课程实验；此类问题视为 `底层漏洞`，不归类为“调参失败”。
- 输出物：一份 reward-gate consistency 结论，明确哪些量代表“接近”、哪些量代表“真实插入”、哪些量代表“无推盘真成功”。

## 3. 失败复盘规则

- 新增 `.cursor/rules/experiment-reflection.mdc`。
- 每次实验失败必须按固定 checklist 复盘五个环节：物理环境与 reset、观测与相机、actor/critic 与价值函数、reward/done/reset/buffer/log key、指标与视频。
- 在 checklist 没完成前，禁止直接继续调超参。
- 每轮实验都必须保存：日志尾段关键指标、至少 3 个评估视频、1 个失败视频、以及“是否存在底层漏洞”的结论。
- `value_function loss` 必须作为显式观察项进入每轮实验记录；若出现异常放大或与行为指标明显脱钩，需要优先排查 reward scale、critic target 和成功/失败门控是否失真，而不是直接继续收紧课程。

## 4. 分支 A：master 势函数对照线

- **分支创建**：基于当前修复好的基座，拉取新分支 `exp/master_potential_approach_only`。
- 目标：回答“在保留现有底层修复、只把 reward family 切回 master 风格势函数后，`2 动作 approach-only + 视觉 actor` 是否仍然可行”。
- 这条线是 `对照线`，不是把整个代码库原样回滚到 `master`。
- 保持不变：共同基座中的全部非 reward 项。
- 唯一主改动：恢复 master 风格的 `phi1/phi2/phi_ins` 势函数差分和对应门控；初始化分布按你的要求回到 `master` 一致。
- 预算限制：先做 `smoke_train`，再做一轮短训；如果再次出现“死区薅羊毛 / 推土机 / 原地刷分”，立即降级，不投入长训。
- 判废条件：`push_free_insert_rate` 与 `push_free_success_rate` 无法破零，或视频再次清楚复现 reward hacking。

## 5. 分支 B：论文 Reward 主线

- **分支创建**：基于当前论文奖励分支，拉取或重命名为 `exp/vision_cnn/paper_native_reward_v2`（或继续在当前分支推进，但需明确 commit 节点）。
- 目标：把当前已出现突破的论文式绝对状态奖励继续做成唯一主线。
- 起点：从当前论文奖励分支继续，但先通过共同基座审计，确保它真的在跑 `ResNet18 + ImageNet + 全程冻结`。
- 第一批实验：**动态退火课程学习 (SOP)**。按 `0.4 -> 0.3 -> 0.2 -> 0.1` 的 0.1m 步长平滑收紧 `rg` 阈值。每次收紧必须加载上一阶段稳定的 Checkpoint。
- 在收紧 `rg` 阈值的同时，可进行**动态惩罚**微调（如将推盘惩罚 `alpha_5` 从 1.0 缓慢增加到 3.0 等），逼迫策略从粗放走向精细控制。
- 不允许把 `rg`、`rini`、`rp`、初始化范围、相机几何一起改。
- 必须显式定义课程退出路径：当某个 `rg` 阈值在近场/容易初始化上稳定后，要逐步恢复更宽的初始化分布，最终回到 `master` 分布或同级难度，避免只在 easy setup 上形成局部策略。
- 若课程收紧后 `rg` 继续高触发，但 `phase/frac_inserted`、`push_free_*`、视频行为不同步改善，则暂停后续课程，回到 `Step 0.5` 重新审计奖励与成功判据。

## 6. 统一验收与决策

- 统一主 KPI：`push_free_success_rate`、`push_free_insert_rate`、`phase/frac_inserted`、`phase/hold_counter_max`、`diag/pallet_disp_xy_mean/p95`。
- 分支 A 额外关注：`phi/phi1`、`phi/phi2`、`phi/phi_ins` 是否在“正确行为”上升，而不是在死区刷分。
- 分支 B 额外关注：`paper_reward/rg` 是否真实触发，`err/lateral_mean`、`err/yaw_deg_mean` 是否随课程收紧同步改善。
- 决策规则：如果分支 A 只能稳定复现 hacking，而分支 B 能通过课程持续收紧阈值，就正式停止把势函数当主线；如果分支 A 在统一基座下也能稳定跑通，再保留为备用线。
- 两条线都要同时看 `reward 指标`、`过程指标`、`最终 KPI` 和 `视频行为` 四类证据，禁止仅凭单个奖励项上升就判定“实验成功”。

## 6.1 固定执行顺序

- 固定顺序为：`创建分支 -> 共同基座审计 -> 奖励门控一致性审计 -> 分支 B 第一批课程实验 -> 分支 A smoke 对照 -> 分支 B 第二批/第三批实验`。
- 分支 A 只承担“验证势函数在统一基座下是否仍可行”的职责，不得抢在分支 B 主线之前消耗长训预算。
- 若分支 B 第一批实验已经暴露判据失配、critic 漂移或明显假突破，则暂停分支 A，对底层逻辑先做复盘与修正。

## 7. 执行约束

- 训练日志命名遵循 `logs/YYYYMMDD_HHMMSS_<type>_<version>.log`。
- 修改任务代码时，只改 patch 源目录，再执行 `forklift_pallet_insert_lift_project/scripts/install_into_isaaclab.sh` 同步到 IsaacLab。
- 每轮实验都必须新增独立实验记录，并明确写出“这轮实验真正回答了什么问题”。
