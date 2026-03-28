# Exp8.3 四类同质错误修复分支映射

日期：2026-03-28

## 目的

把当前已经识别出来的 4 类“研究流程同质错误”映射成独立修复分支，避免后续修复和主实验线相互污染。

---

## 1. Branch A：actor 观测 / 信号审计

分支名：

- `exp/exp8_3_actor_obs_signal_audit`

要修的问题：

- 以为 actor 在使用某个几何信号
- 实际上这个信号只在 reward 或 critic 中存在

这一支主要要做：

1. 列出 steering 相关信号在 actor / critic / reward 中的分布
2. 明确当前 actor 到底直接看到了什么
3. 设计最小 actor-side steering signal 注入方案

优先产物：

- actor/critic/reward 信号表
- actor steering signal 最小增量实验方案

---

## 2. Branch B：steering gap 验收门

分支名：

- `exp/exp8_3_steering_gap_acceptance_gate`

要修的问题：

- 以为 success 提升就等于 steering 学出来了

这一支主要要做：

1. 固化 `normal vs zero-steer` 作为硬验收标准
2. 固化 grid 输出与统计表
3. 补 action-based steering usage 指标

优先产物：

- steering gap 统一评估脚本
- steer usage 诊断指标

---

## 3. Branch C：多 seed / 统一评估可靠性

分支名：

- `exp/exp8_3_multiseed_eval_reliability`

要修的问题：

- 以为单 seed / 尾窗 / reward 足够代表方向
- eval / grid 队列又容易卡死

这一支主要要做：

1. 统一 eval / grid 的 timeout、失败不中断、重试和状态清单
2. 固化多 seed 短训与 unified eval 的套件
3. 固化 early triage 指标

优先产物：

- 更稳的 eval/grid runner
- 多 seed 对照自动汇总

---

## 4. Branch D：几何 / 可视化 gate

分支名：

- `exp/exp8_3_geometry_visualization_gate`

要修的问题：

- 以为一个代表点几何没问题，就等于整个机制没问题

这一支主要要做：

1. 固化 geometry-only sweep
2. 固化 runtime 双视图
3. 固化 `s_start < s_pre < s_goal` 检查

优先产物：

- geometry gate 脚本
- 几何验收模板

---

## 5. 与当前主线的关系

当前主实验线仍然是：

- `exp/exp8_3_stage1_entry_geometry_v3`

它负责回答：

- entry geometry 修正后，是否能真正打开 steering gap

上面 4 个修复分支负责把研究流程本身补牢，避免后面继续被：

- actor 信号错觉
- steering 验收错觉
- 单 seed / eval 假象
- 几何边界盲区

这些问题反复拖住。
