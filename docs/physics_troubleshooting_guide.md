# Isaac Sim 物理排查标准指南

在强化学习训练过程中，如果遇到智能体无法完成特定任务（如无法插入、无法举升等），首先需要排查的是**物理几何限制**和**环境逻辑设定**是否冲突。

为此，我们在 `scripts/` 下提供了一个专门的验证脚本，用于在不依赖任何神经网络策略的情况下，通过直接控制关节来测试物理系统的真实上限。

## 核心排查工具：`verify_forklift_insert_lift.py`

此脚本可以直接控制叉车并进行极端的物理极限测试（Sanity Check），它会帮你回答诸如“货叉最多能插进托盘多深”或“最大举升高度是多少”等物理底层问题。

### 1. 运行物理极限排查 (Sanity Check)

**运行命令（注意必须脱离 Conda 环境，使用 IsaacLab 自带的 Python）：**

```bash
cd /home/uniubi/projects/forklift_sim/IsaacLab
env TERM=xterm PYTHONUNBUFFERED=1 CONDA_PREFIX="" CONDA_DEFAULT_ENV="" ./isaaclab.sh -p ../scripts/verify_forklift_insert_lift.py --sanity-check --headless
```

**排查报告解读：**

脚本运行完毕后，会输出一份详细的 `SUCCESS SANITY CHECK 诊断报告`。

*   **A 层：判定逻辑验证 (A1/A2)**
    *   该阶段脚本会直接把叉车**瞬间传送**到理论上“完美插入且举升”的位置。
    *   如果 A 层报告出现 `FAIL`，说明代码里关于“什么是成功”的判定公式（例如坐标系转换、奖励判定等）有 Bug。
*   **B 层：物理可达性验证 (B1/B2) —— 最关键！**
    *   **B1 物理最大插入深度**：脚本会自动驾驶叉车全速冲向托盘孔，直到物理引擎判定发生刚体碰撞导致完全卡死。
        *   这里打印出来的 `最大插入深度` 就是当前 USD 模型和 `scale` 缩放比例下的**物理绝对上限**。
        *   如果你设定的 `insert_thresh` (如 `pallet_depth_m * insert_fraction`) 大于这个上限，就说明任务在物理上绝对不可能完成。
    *   **B2 物理最大举升高度**：测试升降关节在负载情况下的行程极限。

### 2. 手动键盘控制验证

如果你需要自己看着画面开一开叉车，感受一下碰撞体积：

**运行命令（去掉 `--headless`，并在有图形界面的终端运行）：**

```bash
cd /home/uniubi/projects/forklift_sim/IsaacLab
env TERM=xterm PYTHONUNBUFFERED=1 CONDA_PREFIX="" CONDA_DEFAULT_ENV="" ./isaaclab.sh -p ../scripts/verify_forklift_insert_lift.py --manual
```

**操作说明：**
*   `W/S`：前进/后退
*   `A/D`：左转/右转
*   `R/F`：货叉上升/下降
*   `SPACE`：急停
*   `G`：瞬间传送到托盘正前方完美对齐的理想位置（极大方便测试插入）
*   `P`：在终端打印当前插入深度、误差等详细数值状态

## 常见物理问题排查案例：尺寸与缩放比例冲突

以**托盘缩放比例**为例：

1.  如果在 `env_cfg.py` 中将托盘的 `scale` 设置得过小（比如从 `1.8` 降到 `1.5`），托盘的插孔宽度也会同比例缩小。
2.  此时运行 `--sanity-check` 会在 `B1` 阶段报错：最大插入深度为 `0.000m`。
3.  这就表明货叉的物理宽度已经大于插孔的宽度，叉车连门都进不去，直接被挡在了外面。

**排查金标准：** 任何涉及环境大更新、修改 `env_cfg.py` 里的模型尺寸（`scale`）、任务阈值（如 `insert_fraction`）之后，**必须先跑一次 `--sanity-check`**，确保 `B1` 和 `B2` 的结论均为 `可达`，然后才能启动 RL 训练，否则会白白浪费算力。

## 附加排查工具箱 (Scripts Troubleshooting Toolkit)

除了最核心的动态物理验证工具 `verify_forklift_insert_lift.py` 之外，`scripts/` 目录下还提供了一系列非常有用的排查和辅助脚本，主要分为以下几类：

### 1. 静态几何与网格诊断工具 (Geometry & Mesh Diagnostics)
这类脚本无需运行复杂的物理仿真，而是直接读取 USD 文件的底层数据来进行诊断：
*   **`verify_geometry_compatibility.py`**：**几何尺寸适配检查器。** 它会分别读取叉车和托盘的 Bounding Box（包围盒），并计算“货叉有多宽”、“插孔有多大”，直接在终端输出它们在尺寸上是否兼容。
*   **`check_pallet_mesh2.py`**：**托盘内部结构“X光”扫描仪。** 它可以读取托盘的底层网格顶点（Vertices），并在终端画出 ASCII 字符画（俯视图和正视图），让你无需打开 3D 软件就能一眼看穿托盘内部有没有隐藏的木块或立柱。
*   **`check_forklift.py`**：专门用来单独读取叉车 USD 并测量其货叉真实物理长度的脚本。

### 2. 资产属性修改工具 (Asset Hack Tools)
当发现模型物理属性有问题，又不想/无法使用 Blender 等第三方 3D 软件修改时：
*   **`shift_pallet_com.py`**：**物理重心平移器。** 它通过调用 `UsdPhysics.MassAPI`，强行修改现成 USD 文件的重心位置（Center of Mass）并导出新文件，非常适合解决因尺寸错配导致的“头轻脚重”拖地问题。

### 3. 关节与底层 USD 树排查 (Joints & USD Hierarchy)
如果在配置新机器人资产时发现轮子不转、或者关节动作异常，可使用以下脚本排查：
*   **`check_joints_simple.py`** & **`verify_joint_axes.py`**：用来读取和验证机器人所有关节（Joints）的名称、类型（旋转/平移）、驱动轴、极限范围（Limits）和刚度/阻尼。
*   **`check_usd_hierarchy.py`**：用来打印 USD 文件的层级树结构（Prim Tree），帮你确认 `RigidBody` 和 `ArticulationRoot` 标签是否绑在了正确的节点上。
*   **`check_usd_steering.py`**：专门用来排查叉车转向节（Steering）问题的独立测试工具。

### 4. 强化学习策略行为分析 (Policy Behavior Eval)
当训练结果不如预期，需要细致分析模型到底在哪个阶段、什么姿态下犯错时：
*   **`eval_s1.0s_diagnostics.py`** (及其他 `eval_*.py`)：**策略详细诊断器。** 它们会加载训练好的权重，自动跑上几百个 Episode，然后输出非常详细的统计报告（如：多少次是因为超时失败的、多少次是卡在死区的、偏航角分布是怎样的等），为奖励函数的下一步修改提供数据支撑。