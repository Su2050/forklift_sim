# Validation Scripts

这个目录集中管理“任务定义可信验证体系”相关脚本，目标是在正式训练前先验证底层定义是否可信，而不是把问题留到长时训练里暴露。

## Purpose

这里要解决的核心问题，不是“再调一次 reward / obs 看 success 会不会涨”，而是先把仿真训练栈里最底层、最容易偷偷出错的部分做成一组可验证、可复现、可回归的真值检查，确认我们优化的是任务本身，而不是资产错误、物理假象、观测失真或指标 bug。

换句话说，这套脚本的目标是建立一套“任务定义可信性验证体系”，覆盖资产、物理、观测、成功口径四个底层面，确保后续训练结果建立在一个正确、可解释、可追溯的仿真基础上。

## Validation Goals

当前这套验证体系面向 6 类诉求：

1. 资产正确性
   - 验证尺寸、缩放、坐标系、朝向、几何定义和碰撞体是否与任务语义一致。
   - 关注 fork tip / fork center / pallet entry 的几何定义是否和 success 判定一致。
   - 希望产出资产真值表、几何示意图、collider / COM / friction 检查报告。

2. 物理可达性
   - 验证当前 success 条件在仿真中是否真实可达，而不是训练在追一个不可达目标。
   - 关注最大可插入深度、举升阈值、hold 时间、误差阈值与物理抖动是否兼容。
   - 希望产出 reachability case 集和“理论成功位姿 -> 仿真实测结果”对照表。

3. 观测准确性
   - 验证 policy 真正看到的图像和低维观测是否与环境状态一致。
   - 关注相机位姿、FOV、obs 管线、reset 首帧、刷新时序、图像数值范围。
   - 希望产出 obs 真值对照表、随机 case 的 obs-vs-state 校验样例。

4. 成功指标与日志口径可靠性
   - 验证 `phase/frac_success`、`phase/frac_success_strict`、`phase/frac_push_free_success` 等定义是否与预期一致。
   - 关注 train-time success 与 strict diagnostic success 是否被混用，hold counter 和 gate 是否存在误判。
   - 希望产出 success 逻辑说明、边界 case 回放和真值测试。

5. 训练前置诊断能力
   - 目标是在不训练或很短训练下就快速分辨是资产、物理、对齐、观测还是 success gate 出问题。
   - 希望有固定 case、固定 checkpoint、固定可视化流程做 smoke validation。

6. 回归与版本可信度
   - 目标是让 reward / obs / success / 资产修改后，底层真值不会被悄悄改坏。
   - 希望把“经验判断”变成 checklist、Stop/Go gate 和最小回归套件。

当前采用按验证对象分层的结构：

- `assets/`: 3D 资产、几何兼容性、USD 层级与方向检查
- `physics/`: 关节轴、可达性、运动学与物理 sanity check
- `observations/`: 相机、视野、观测采样与初始化画面检查
- `success/`: success gate、hold 逻辑与任务闭环验证
- `playback/`: 回放、录屏、策略输入可视化

## Coverage Assessment

下面是当前 `scripts/validation` 的覆盖情况判断：

| 验证类目 | 当前覆盖 | 现有脚本 | 判断 |
| --- | --- | --- | --- |
| 资产正确性 | `Partial` | `assets/verify_geometry_compatibility.py`, `assets/check_forklift.py`, `assets/check_pallet_mesh.py`, `assets/check_pallet_mesh2.py`, `assets/check_usd_hierarchy.py`, `assets/check_usd_steering.py`, `assets/diagnose_pallet_orientation.py`, `assets/scan_nucleus_pallets.py` | 已经能覆盖几何、层级、方向、碰撞形状和插入兼容性，但还没有独立的 COM / friction / mass / material 真值表生成器。 |
| 物理可达性 | `Mostly yes` | `physics/eval_yaw_reachability.py`, `physics/verify_joint_axes.py`, `physics/diagnose_rotator_axis.py`, `physics/check_joints_simple.py`, `success/verify_forklift_insert_lift.py --sanity-check` | 已经具备“关节轴方向 + reachability + success 位姿 sanity check”的核心能力，足够做 smoke 级物理可达性验证。 |
| 观测准确性 | `Partial` | `observations/camera_eval.py`, `observations/capture_init_frames.py`, `observations/test_camera_output.py`, `observations/test_double_env.py`, `observations/verify_trajectory_and_fov.py`, `playback/play_and_record_policy_input.py` | 已能验证相机视角、首帧、policy image 提取和简单 FOV 假设，但还缺低维 proprio / critic 的系统性真值对照脚本。 |
| 成功口径可靠性 | `Partial to strong` | `success/verify_forklift_insert_lift.py` | 这个脚本已经覆盖 success gate、hold counter、物理插入/举升 sanity check，是当前最接近“真值回放”的脚本；但还缺更轻量的边界 case 单元测试。 |
| 训练前置诊断 | `Yes (smoke)` | `success/verify_forklift_insert_lift.py`, `physics/eval_yaw_reachability.py`, `observations/capture_init_frames.py`, `playback/play_and_record.py` | 已经能在长训练前做 smoke 诊断，但还没有一个统一入口把这些检查串成固定套件。 |
| 回归与 Stop/Go | `No / weak` | 无统一脚本 | 目前还没有统一 checklist、自动 gate、最小回归 runner，这部分仍需要补。 |

## Current Conclusion

以现在的脚本集合来看：

- 可以实现一版“低层可信性 smoke validation”。
- 不能说已经实现了完整的“可回归验证框架”。
- 当前最强的能力集中在几何兼容性、关节轴/可达性、相机观测和 success gate sanity check。
- 当前最大的缺口在于：
  - 缺少统一 runner，把多类检查串成固定的 pre-train validation 套件。
  - 缺少 COM / friction / material / inertia 的专门报告脚本。
  - 缺少 proprio / critic 数值对照检查。
  - 缺少 success 边界 case 的轻量单元测试。
  - 缺少 Stop/Go gate 和回归 checklist。

## Recommended Smoke Order

如果只是先回答“这套环境定义今天值不值得继续往上训练”，建议按这个顺序跑：

1. `assets/verify_geometry_compatibility.py`
2. `physics/verify_joint_axes.py`
3. `physics/eval_yaw_reachability.py`
4. `observations/test_camera_output.py`
5. `observations/capture_init_frames.py`
6. `success/verify_forklift_insert_lift.py --sanity-check`

如果要直接跑当前的一键 smoke 套件，使用：

- `scripts/validation/run_smoke_validation.sh`

当前 runner 的默认检查集合是：

- `assets/verify_geometry_compatibility.py`
- `physics/verify_joint_axes.py`
- `physics/eval_yaw_reachability.py`
- `observations/test_camera_output.py`
- `observations/verify_trajectory_and_fov.py`
- `success/verify_forklift_insert_lift.py --sanity-check`

约定如下：

- 这里的脚本是新的 canonical 入口。
- 历史路径 `scripts/*.py`、`scripts/tools/*.py`、`scripts/experiments/*.py` 保留轻量包装脚本，避免旧文档和旧命令失效。
- 新脚本尽量不要硬编码仓库绝对路径，统一从当前文件位置推导 `REPO_ROOT`。
- 产出物优先写到 `outputs/validation/` 或 `docs/diagnostic_assets/`，不要再落到仓库根目录。
- 带实验假设的运行脚本不要内嵌特定 checkpoint；优先改成显式参数。
- 当前不少基于 `ForkliftPalletInsertLiftEnv` 的脚本在实际运行时需要附带 `--enable_cameras`，否则会在相机传感器初始化阶段失败。
- 目前 `assets/verify_geometry_compatibility.py` 和 `physics/eval_yaw_reachability.py` 已在脚本内部显式关闭相机依赖；runner 会只给仍然依赖相机链路的脚本自动补 `--enable_cameras`。

这套结构目前是“低迁移成本、易继续演进”的折中方案：

- 优点是迁移快、兼容旧文档、便于按验证主题继续扩充。
- 暂时没有把这些脚本进一步封装成 Python package，因为当前大部分脚本仍以 `isaaclab.sh -p <script>` 方式独立运行，保持扁平入口更稳妥。
- 如果后续脚本数量继续增长，可以再补一层公共工具，例如 `common/paths.py`、`common/reporting.py`、`common/cases.py`。
