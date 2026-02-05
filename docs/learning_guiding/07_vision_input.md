# 视觉输入与相机配置：从"状态向量"到"图像观测"的进阶路线

本页适合：已经跑通了叉车任务，想了解"为什么当前没有相机"、"怎么加 RGB 相机"、"把输入改成图像要改哪些地方"、以及"一步步迭代到更真实/可迁移"的读者。

---

## 1. 为什么当前任务没有相机也能训练？

这是一个**状态向量（state-based）任务**，不是视觉任务。

策略输入不是图像，而是仿真里直接读取的一组"低维状态"：

- 托盘相对叉车的位置/朝向
- 叉车自身速度/角速度
- 升降关节状态
- 货叉插入深度
- 上一步动作

这些数据在仿真里**可以直接获得**（不需要相机），是强化学习最容易训练的输入形式。

> **"没有相机"不代表不能加。**
> 只是当前任务选择了"先用状态向量让策略学会核心动作"，相机/视觉属于进阶改造。

---

## 2. 当前策略输入是什么（14 维逐项解释）

配置里：

```python
observation_space = 14
```

代码在 `forklift_pallet_insert_lift/env.py` 的 `_get_observations()` 里，最终拼成 14 维向量：

| 维度 | 名称 | 含义 |
|------|------|------|
| 2 | `d_xy_r` | 托盘相对叉车的 x/y 位移（在叉车坐标系） |
| 2 | `cos(dyaw)`, `sin(dyaw)` | 托盘与叉车朝向差的三角函数表示 |
| 2 | `v_xy_r` | 叉车平面速度（叉车坐标系） |
| 1 | `yaw_rate` | 偏航角速度 |
| 1 | `lift_pos` | 升降关节位置 |
| 1 | `lift_vel` | 升降关节速度 |
| 1 | `insert_norm` | 归一化插入深度（插入深度 / 托盘深度） |
| 3 | `actions` | 上一步执行的动作 |

> **插入深度怎么估算？**
> 环境每步会找叉车所有刚体里"沿叉车前进方向投影最大"的点，当作货叉尖端 tip，再计算 tip 与托盘前沿的距离。

---

## 3. 两个 RGB 相机怎么加（Left / Right）

### 3.1 传感器类型选择

IsaacLab 提供两类相机传感器：

| 类型 | 说明 | 适用场景 |
|------|------|----------|
| `CameraCfg` / `Camera` | 通用相机，支持多种数据类型 | 单环境/少量环境调试 |
| `TiledCameraCfg` / `TiledCamera` | 基于 tiled rendering 的相机，大规模并行更高效 | **多环境训练推荐** |

对于 128+ 环境训练，**推荐使用 `TiledCameraCfg`**。

### 3.2 配置示例（在 `env_cfg.py` 中新增）

```python
from isaaclab.sensors import TiledCameraCfg
import isaaclab.sim as sim_utils

# ... 在 ForkliftPalletInsertLiftEnvCfg 类中增加 ...

# 左侧相机（挂在叉车车体左前方）
camera_left: TiledCameraCfg = TiledCameraCfg(
    prim_path="/World/envs/env_.*/Robot/CameraLeft",
    offset=TiledCameraCfg.OffsetCfg(
        pos=(0.5, 0.3, 1.2),  # 相对叉车根节点的偏移（前/左/上）
        rot=(0.5, -0.5, 0.5, -0.5),  # 朝向前方（ROS 约定）
        convention="ros",
    ),
    data_types=["rgb"],
    spawn=sim_utils.PinholeCameraCfg(
        focal_length=24.0,
        focus_distance=400.0,
        horizontal_aperture=20.955,
        clipping_range=(0.1, 20.0),
    ),
    width=96,
    height=96,
)

# 右侧相机（挂在叉车车体右前方）
camera_right: TiledCameraCfg = TiledCameraCfg(
    prim_path="/World/envs/env_.*/Robot/CameraRight",
    offset=TiledCameraCfg.OffsetCfg(
        pos=(0.5, -0.3, 1.2),  # 相对叉车根节点的偏移（前/右/上）
        rot=(0.5, -0.5, 0.5, -0.5),
        convention="ros",
    ),
    data_types=["rgb"],
    spawn=sim_utils.PinholeCameraCfg(
        focal_length=24.0,
        focus_distance=400.0,
        horizontal_aperture=20.955,
        clipping_range=(0.1, 20.0),
    ),
    width=96,
    height=96,
)
```

### 3.3 在 `env.py` 中注册传感器

在 `_setup_scene()` 里：

```python
from isaaclab.sensors import TiledCamera

# ... 在 _setup_scene() 中增加 ...
self._camera_left = TiledCamera(self.cfg.camera_left)
self._camera_right = TiledCamera(self.cfg.camera_right)

# 注册到 scene
self.scene.sensors["camera_left"] = self._camera_left
self.scene.sensors["camera_right"] = self._camera_right
```

### 3.4 读取图像数据

在 `_get_observations()` 或其他需要图像的地方：

```python
# 获取 RGB 图像（形状：N × H × W × 3，uint8）
rgb_left = self._camera_left.data.output["rgb"]
rgb_right = self._camera_right.data.output["rgb"]

# 归一化到 [0, 1]
rgb_left_norm = rgb_left.float() / 255.0
rgb_right_norm = rgb_right.float() / 255.0
```

---

## 4. 把输入改成"图像/图像+状态"要改哪些地方

### 路线对比

| 路线 | 描述 | 难度 | 推荐度 |
|------|------|------|--------|
| **A（推荐）** | 图像→轻量编码器→embedding→与14维状态拼接 | 中 | ★★★★★ |
| **B（进阶）** | 直接用图像张量作为 policy 输入 | 高 | ★★☆☆☆ |
| **C（工程化）** | 多模态 obs 分组（policy/critic 不同输入） | 高 | ★★★☆☆ |

### 路线 A（推荐）：图像 → embedding → 与状态向量拼接

**思路**：在环境内部把两路 RGB 图像通过一个轻量 CNN 编码成固定维度向量（比如 32 维），再与现有 14 维状态拼接，最终输出 14+32+32=78 维观测。

**优点**：
- RSL-RL 的默认 MLP policy 仍可用
- 最容易训练起来
- 可以先冻结 encoder 验证链路，再逐步联合训练

**需要改的地方**：

1. **`env_cfg.py`**：增加两路相机配置（见上面 3.2）+ 更新 `observation_space = 14 + 32 + 32`
2. **`env.py`**：
   - 新增一个轻量 CNN encoder（可放在 `__init__` 里初始化）
   - `_setup_scene()` 注册相机
   - `_get_observations()` 里：图像→encoder→embedding→与原有 obs 拼接
3. **`agents/rsl_rl_ppo_cfg.py`**：可能需要调整 `actor_hidden_dims` / `critic_hidden_dims`

**参考示例**：`IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/shadow_hand/shadow_hand_vision_env.py`

### 路线 B（进阶）：直接用图像张量作为观测

**思路**：把 `observations["policy"]` 直接设为图像张量（N × H × W × C 或 N × C × H × W）。

**问题**：
- RSL-RL 默认的 `ActorCritic` 是 MLP，无法处理图像张量
- 需要自定义 CNN policy 或使用其他框架（如 SKRL、SB3）

**参考示例**：`IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/cartpole/cartpole_camera_env.py`

### 路线 C（工程化）：多模态 obs 分组

**思路**：利用 RSL-RL 的 `obs_groups` 机制，让 policy 和 critic 接收不同的观测组合。

**示例配置**（在 PPO runner cfg 中）：

```python
obs_groups = {
    "policy": ["low_dim", "vision_embedding"],
    "critic": ["low_dim", "privileged"],
}
```

**前提**：图像需要先变成向量（embedding），才能与其他组 concat。

---

## 5. 训练与评估命令的关键开关

### 5.1 启用相机传感器

如果环境里配置了相机，训练时需要启用相机渲染：

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Forklift-PalletInsertLift-Direct-v0 \
  --headless \
  --enable_cameras \
  --num_envs 32
```

> **注意**：`--enable_cameras` 和 `--video` 是两回事：
> - `--enable_cameras`：启用传感器相机，让环境能读取图像数据
> - `--video`：启用训练过程录制，会额外消耗资源

### 5.2 建议开关组合

| 场景 | 命令开关 |
|------|----------|
| 状态向量训练（无相机） | `--headless --num_envs 128` |
| 有相机但 embedding 融合 | `--headless --enable_cameras --num_envs 32` |
| 调试/可视化 | 去掉 `--headless`，加 `--num_envs 4` |

---

## 6. 性能与收敛影响

### 6.1 影响因素

| 因素 | 影响 | 建议 |
|------|------|------|
| **图像分辨率** | 分辨率越高，显存/计算越大 | 起步用 84×84 或 96×96 |
| **相机数量** | 两路比一路约多 50-100% 渲染开销 | 先验证一路，再加第二路 |
| **update_period** | 相机更新频率 | 可设为与控制步长相同或更低（如 0.033s） |
| **num_envs** | 环境越多，总渲染越重 | 有相机时先降到 16~32 验证 |
| **data_types** | 同时要 rgb+depth 会更慢 | 先只用 rgb |

### 6.2 经验值参考

| 配置 | 预估 steps/s（RTX 3090/A100 单卡） |
|------|-----------------------------------|
| 无相机，128 envs | 2000~3000 |
| 两路 RGB 96×96，32 envs | 400~800 |
| 两路 RGB 84×84，64 envs | 600~1000 |

> 以上数值仅供参考，实际取决于具体硬件与场景复杂度。

---

## 7. 进阶迭代路线图

### Phase 0：保持现状基线（状态向量 PPO）

| 项目 | 内容 |
|------|------|
| **目标** | 确保当前任务稳定、奖励/终止/成功率基线可复现 |
| **改动点** | 无 |
| **推荐参数** | `--num_envs 128`，`--max_iterations 2000` |
| **验收标准** | 固定种子下 1-2 小时内能出现成功轨迹（或成功率明显上升） |
| **常见坑** | 如果一直不收敛，先检查奖励/终止逻辑、初始随机范围 |

---

### Phase 1：加两路 RGB，但先不让策略用（只验证传感器链路）

| 项目 | 内容 |
|------|------|
| **目标** | 相机能稳定输出、性能可接受（不崩、不丢帧/不黑屏） |
| **改动点** | `env_cfg.py` 增加两路 `TiledCameraCfg`；`env.py` 的 `_setup_scene()` 注册 sensors；训练脚本加 `--enable_cameras` |
| **推荐参数** | 分辨率 84×84 或 96×96；`update_period` 设为 `1/30` 或与 sim dt × decimation 对齐 |
| **验收标准** | headless 训练/回放都能拿到两路图像（可打印 shape 或保存几帧）；steps/s 降幅在可接受范围（比如不低于 400） |
| **常见坑** | 忘记 `--enable_cameras`；prim_path 写错导致相机没挂上；分辨率太高导致显存爆 |

---

### Phase 2（推荐起步）：图像 → embedding → 与 14 维融合

| 项目 | 内容 |
|------|------|
| **目标** | 在不改 RSL-RL MLP policy 的前提下，让视觉信息"进入策略" |
| **改动点** | 在 env 中加一个轻量 CNN encoder（如 3-4 层卷积 + flatten）；`_get_observations()` 里把两路图像过 encoder 得到 embedding（如 32 维），与原 14 维拼接；更新 `observation_space` |
| **推荐参数** | encoder 输出维度 32~64；先冻结 encoder 权重验证链路，再逐步放开联合训练 |
| **验收标准** | 与 Phase 0 相比，至少不显著退化；能出现可解释的视觉相关行为（如更少横向偏差、更早对准） |
| **常见坑** | encoder 输出维度与 `observation_space` 不一致；图像没归一化导致训练不稳定；encoder 与 policy 的设备不一致 |

---

### Phase 3：域随机化（Domain Randomization）+ 视觉鲁棒性

| 项目 | 内容 |
|------|------|
| **目标** | 让视觉不依赖固定光照/纹理/相机姿态，增强泛化能力 |
| **改动点** | 在 reset 或场景初始化时随机：灯光强度/颜色、托盘/地面纹理、相机姿态轻微抖动（±几度/几厘米）、托盘初始位置/朝向扰动 |
| **推荐参数** | 灯光强度 ±30%、纹理随机（如果资产支持）、相机姿态抖动 ±2°/±0.02m |
| **验收标准** | 在随机化下成功率不坍塌（与非随机相比掉不超过 30%）；对 unseen 种子仍可完成插入+抬升 |
| **常见坑** | 随机范围太大导致无法学习；忘记在 eval 时关闭部分随机化；纹理随机导致训练速度骤降 |

---

### Phase 4：把托盘从 kinematic 变成动态（更真实）

| 项目 | 内容 |
|------|------|
| **目标** | 让插入与抬升更符合真实物理（托盘会被推/会滑） |
| **改动点** | 修改 `pallet_cfg` 的 rigid_props：`kinematic_enabled=False`，`disable_gravity=False`；可能需要重新调整奖励系数和成功阈值 |
| **推荐参数** | 托盘质量/摩擦按真实值设定；成功判定的"保持时间"可能需要放宽 |
| **验收标准** | 能稳定插入且不把托盘撞飞；成功 hold 仍可达成（哪怕成功率下降） |
| **常见坑** | 托盘太轻被撞飞；插入后托盘滑动导致 hold 失败；奖励噪声变大导致训练不稳定 |

---

### Phase 5：端到端视觉（可选，工程成本高）

| 项目 | 内容 |
|------|------|
| **目标** | 策略直接消费两路 RGB（CNN policy），减少对"特权状态"的依赖 |
| **改动点** | 自定义 RSL-RL policy（如 CNN + MLP fusion）或更换 RL 框架（如 SKRL、SB3）；`observation_space` 改为图像维度 |
| **推荐参数** | 图像分辨率 84×84；CNN 参考经典结构（如 Nature DQN 的卷积部分）；`num_envs` 降到 16~32 |
| **验收标准** | 在相同随机化下达到 Phase 2/3 接近的成功率 |
| **风险** | 采样慢（约 10 倍）；显存需求大（约 2-4 倍）；收敛更难（可能需要更多迭代或更好的超参） |

---

### Phase 6：Sim2Real 前的"现实约束"清单

如果你最终要把策略部署到真实叉车上，需要提前考虑这些：

| 类别 | 要点 |
|------|------|
| **相机** | 内参/畸变标定、曝光/白平衡噪声、分辨率与仿真对齐、图像延迟（lag） |
| **控制** | 控制频率（仿真 30Hz → 真实 10~20Hz？）、动作延迟、执行器响应曲线 |
| **物理** | 摩擦系数/质量不确定性、托盘形状变异、货物重量变化 |
| **观测缺失** | 遮挡/丢帧处理、状态估计误差（如果真实中没有"托盘位姿"直接可用） |
| **安全** | 碰撞检测、紧急停止、速度限制 |

**建议**：在仿真里加入与上述对应的"域随机化"（Domain Randomization），让策略对这些不确定性有一定鲁棒性。

---

## 8. 下一步读什么

- 想了解当前任务的奖励/终止设计：`docs/03_task_design_rl.md`
- 想看训练命令与日志目录：`docs/04_training_and_artifacts.md`
- 遇到问题想排障：`docs/06_troubleshooting.md`
