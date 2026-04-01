# Exp9.0 Low-Level Validation Recheck

日期：`2026-04-01`

## 1. 背景

这轮复核的目标不是重新发散讨论训练策略，而是把 `scripts/validation` 里已经暴露出的 low-level 问题逐条分清：

1. 哪些是底层定义真有问题。
2. 哪些是验证脚本本身读错了资产、沿用了过期口径，或者量测不稳定。
3. 哪些修完之后，`exp9.0` 可以放心把“success=0 更像策略问题”当成当前主线假设继续推进。

本页对应的最终 smoke runner 结果在：

- `outputs/validation/manual_runs/20260401_124246_verify_geometry_compatibility.log`
- `outputs/validation/manual_runs/20260401_124246_verify_joint_axes.log`
- `outputs/validation/manual_runs/20260401_124246_eval_yaw_reachability.log`
- `outputs/validation/manual_runs/20260401_124246_test_camera_output.log`
- `outputs/validation/manual_runs/20260401_124246_verify_trajectory_and_fov.log`
- `outputs/validation/manual_runs/20260401_124246_verify_forklift_insert_lift_sanity.log`

对应总览：

- `outputs/validation/manual_runs/20260401_124246_*`
- `scripts/validation/run_smoke_validation.sh`

## 2. 原始发现与复核结论

### 2.1 `verify_joint_axes`

原始发现：

- 早先日志里有 `当前代码异号输入效果: FAIL`
- 有时还会出现 `转向幅度对称 FAIL`

复核结论：

- `异号输入 FAIL` 不是当前环境控制逻辑的真实问题，而是脚本把“旧的左右反号映射”误写成了“当前 env.py 做法”。
- 当前环境实际控制映射已经是“左右 rotator 同号”，这和 Phase 2 的物理轴验证是一致的。
- Phase 3 的剩余 FAIL 也不是稳定的底层错误，而是测试起点不固定时，单次 reset 会让正负 steer 的 yaw 量测落入低信号/高波动区，导致 smoke gate 偶发误报。

现在的处理：

- Phase 2 改为直接验证“当前 env 控制映射”，不再把旧的异号映射当成当前实现。
- 旧的异号映射保留为 `legacy-check` 信息输出，只用于说明为什么它会形成八字形。
- Phase 3 每次都把叉车重置到固定标准起点，再做 steering symmetry smoke，去掉 reset 噪声。
- `verify_joint_axes.py` 改成 camera-free，不再依赖 `--enable_cameras`。
- `run_smoke_validation.sh` 同步改为 camera-free 调用 `verify_joint_axes.py`。

最终结果：

- `outputs/validation/manual_runs/20260401_124246_verify_joint_axes.log`
- 最终为 `6 PASS / 0 FAIL`

结论：

- 当前 steering low-level 没有发现需要继续修 env 控制符号的真问题。

### 2.2 `verify_geometry_compatibility`

原始发现：

- 早先日志给出 `高度不兼容`、`间距不匹配`
- 但同时 success/reachability 类验证又显示物理上并不是完全插不进去

复核结论：

- 这次几何 mismatch 的主因不是资产真的不兼容，而是脚本当时读取了错误的托盘资产基准：
  - 脚本读的是未缩放的 Nucleus `pallet.usd`
  - 环境实际使用的是 `assets/pallet_com_shifted.usd`
  - 并且在 env 里还额外施加了 `scale=(1.8, 1.8, 1.8)`
- 因此旧脚本拿 `1.2m x 0.8m x 0.142m` 的托盘去和环境里实际 `1.8x` 放大的托盘做比较，结论天然会偏向假阳性。

现在的处理：

- `verify_geometry_compatibility.py` 改为直接读取 `env_cfg` 中真实使用的 `forklift/pallet` 资产路径与 `spawn.scale`。
- 几何边界框和尺寸诊断统一乘上实际 `spawn.scale`。
- 输出里明确标注：
  - 当前托盘仍然是单 Mesh 资产
  - 插入孔尺寸属于“缩放后的比例估算”
  - 最终应结合 success/physics 类验证解读
- 原来的“实际碰撞测试”改名为“插入路径冒烟检查”，避免把当前 root-pose 推进检查误写成完整动力学证明。

最终结果：

- `outputs/validation/manual_runs/20260401_124246_verify_geometry_compatibility.log`
- 关键尺寸变为：
  - 单孔宽度 `409.77 mm`
  - 插入孔间距 `722.07 mm`
  - 插入孔高度 `178.13 mm`
- 与货叉估计尺寸比较后，三项兼容性检查均通过

结论：

- 这次 geometry warning 的主体是脚本读错资产/缩放导致的误报，已修复。
- 但资产正确性仍然只能记为 `smoke-pass / partial confidence`，因为托盘 pocket 尺寸仍来自单 Mesh 比例估算，而不是显式 pocket 子结构解析。

### 2.3 `eval_yaw_reachability`

原始发现：

- 之前脚本曾在运行中 `Segmentation fault`

复核结论：

- 当前脚本已经稳定可跑，并能稳定产出结果表，不再是 startup crash 类问题。

当前结果（`outputs/validation/manual_runs/20260401_124246_eval_yaw_reachability.log`）：

- `0.5° -> max_insert_norm=0.2196`
- `2.0° -> max_insert_norm=0.2199`
- `5.0° -> max_insert_norm=0.2199`

解释：

- 这说明“当前 straight-drive 局部 probe”本身比较保守，不代表整套任务在物理上不可达。
- 这个结论应与 `verify_forklift_insert_lift --sanity-check` 一起解读；后者已经验证过完整 success 判定和物理 reachability 没有底层堵死。

## 3. 这次实际修改的文件

- `scripts/validation/physics/verify_joint_axes.py`
- `scripts/validation/assets/verify_geometry_compatibility.py`
- `scripts/validation/run_smoke_validation.sh`

## 4. 最终 smoke 状态

最终一轮 runner：

- `scripts/validation/run_smoke_validation.sh`
- 时间戳：`20260401_124246`
- 结果：`Overall: PASS (PASS=6, WARN=0, FAIL=0)`

对应六项：

- 资产正确性：`Smoke-pass, but still partial-confidence`
- 物理可达性：`Yes at smoke level`
- 观测准确性：`Pass`
- success 口径可靠性：`Pass`
- 训练前置诊断：`Pass`
- 回归 / Stop-Go：`已有可执行 smoke runner，可做轻量 gate`

## 5. 对 `exp9.0` 的直接意义

本轮复核之后，可以更有把握地把下面这句话当成当前工作假设：

- 现在 `exp9.0` 里 success 仍然起不来，主要矛盾更像策略/训练闭环，而不是 low-level 定义已经错到训练无意义。

但仍保留两个边界条件：

- `verify_geometry_compatibility.py` 里的托盘 pocket 尺寸仍是单 Mesh 估算，不应把它当成最终 CAD 级真值。
- `eval_yaw_reachability.py` 当前反映的是一个受限 straight-drive probe，不应直接外推成“整任务 yaw 不可达”。
