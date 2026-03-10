# 视觉端到端强化学习 (Vision-E2E RL) 实验文档

## 1. 实验背景与目标

本项目旨在通过强化学习（RL）训练叉车完成**“接近 -> 对齐 -> 插入 -> 举升”**托盘的完整任务。
在早期的实验中（`master` 分支），我们已经成功训练出了基于**低维物理状态**（叉车与托盘的相对位置、偏航角等 15 维数据）的专家策略。

本次实验（`exp/vision_cnn/scratch_rl_baseline` 分支）的目标是：**从低维状态输入切换到纯视觉输入（端到端）**。
即：Actor 网络不再直接读取托盘的精确坐标，而是通过叉车上搭载的**第一人称摄像头画面（RGB 图像）**，结合自身的本体感觉（速度、转向等），直接输出控制动作。

本实验作为视觉方案的 **Baseline（基线）**，最初采用了简单的 3 层 CNN 架构从零开始训练。在随后的迭代中（`exp/vision_cnn/cnn_pretrain` 分支），我们升级到了 **MobileNetV3-Small** 作为视觉主干网络，以提取更丰富的特征，并验证了其在 60 度俯视视角和远距离初始化下的可行性。

---

## 2. 核心架构设计

为了降低视觉策略的训练难度，我们采用了 **不对称 Actor-Critic (Asymmetric Actor-Critic)** 架构：
- **Actor（策略网络）**：只能看到“现实中能获取的信息”（图像 + 本体感觉），负责输出动作。
- **Critic（价值网络）**：在仿真训练阶段，可以“作弊”看到全局真实的低维物理状态（15维），负责评估当前状态的价值，指导 Actor 更新。

### 2.1 观测空间 (Observation Space)

#### Actor 输入 (`obs["policy"]`)
1. **`image`**: `[3, 64, 64]` 的 RGB 图像，归一化到 `[0, 1]`。
2. **`proprio` (本体感觉)**: 8 维向量，包含：
   - `v_x_r, v_y_r`: 叉车局部坐标系下的线速度 (2)
   - `yaw_rate`: 偏航角速度 (1)
   - `lift_pos, lift_vel`: 货叉的举升高度和速度 (2)
   - `prev_actions`: 上一帧的动作 (3)

#### Critic 输入 (`obs["critic"]`)
**`privileged_obs`**: 15 维低维物理状态（与 `master` 分支的专家策略输入完全一致），包含：
- 相对位置、相对偏航角、线速度、角速度、举升状态、插入深度归一化、上一帧动作、以及严格的横向/偏航误差。

### 2.2 网络结构 (VisionActorCritic)

网络在 `vision_actor_critic.py` 中定义。根据不同的分支，视觉编码器有所不同：

**方案 A：基础 3 层 CNN (位于 `scratch_rl_baseline` 分支)**
```python
# 1. 视觉编码器 (Nature CNN 变体)
self.image_encoder = nn.Sequential(
    nn.Conv2d(3, 32, kernel_size=8, stride=4),
    ELU(),
    nn.Conv2d(32, 64, kernel_size=4, stride=2),
    ELU(),
    nn.Conv2d(64, 128, kernel_size=3, stride=1),
    ELU(),
    nn.Flatten(), # 输出 2048 维
)

# 2. 视觉特征投影
self.image_proj = nn.Sequential(
    nn.Linear(2048, 256), ELU(),
    nn.Linear(256, 256), ELU(),
)
```

**方案 B：MobileNetV3-Small (位于 `cnn_pretrain` 分支，当前使用)**
```python
# 1. 视觉编码器 (MobileNetV3-Small)
# 位于 vision_backbone.py 中定义
self.image_encoder = MobileNetVisionBackbone(imagenet_init=False)
# 内部结构：tv_models.mobilenet_v3_small 的 features 层 + AdaptiveAvgPool2d
# 输出维度: 576 维

# 2. 视觉特征投影
self.image_proj = nn.Sequential(
    nn.Linear(576, 256), ELU(),
    nn.Linear(256, 256), ELU(),
)
```

**公共部分：本体感觉与 Actor/Critic Head**
```python
# 3. 本体感觉编码器
self.proprio_encoder = MLP(input_dim=8, output_dim=128, hidden_dims=[128, 128], activation="elu")

# 4. Actor Head (融合视觉与本体感觉)
# 输入: 256 (视觉投影后) + 128 (本体) = 384 维
self.actor = MLP(input_dim=384, output_dim=3, hidden_dims=[256, 256, 128], activation="elu")

# 5. Critic MLP (直接处理 15 维特权状态)
self.critic = MLP(input_dim=15, output_dim=1, hidden_dims=[256, 256, 128], activation="elu")
```

---

## 3. 环境与物理配置优化

在本次 Baseline 实验中，我们对环境（`env_cfg.py` 和 `env.py`）进行了两项关键的物理层优化，以适配视觉训练：

### 3.1 摄像头视角调整 (60度俯视)
为了让网络能同时看清**前方的托盘**和**自身的货叉**（提供对齐的视觉参考），我们将摄像头的俯仰角（Pitch）从 45 度下压到了 **60 度**。
```python
# env_cfg.py
camera_pos_local: tuple[float, float, float] = (130.0, 0.0, 250.0) # 挂载在车体上方
camera_rpy_local_deg: tuple[float, float, float] = (0.0, 60.0, 0.0) # Pitch=60度，向下俯视
```

### 3.2 初始出生点安全距离 (防止穿透)
在 `Stage 1`（近距离课程）中，叉车初始位置如果离托盘太近，长达 `1.87m` 的货叉会在初始化瞬间直接穿透托盘，导致 PhysX 物理引擎为了去穿透而将托盘弹飞。
我们修改了 `env.py` 中的初始化逻辑，留出了安全裕度：
```python
# env.py - _reset_idx()
if self._stage_1_mode:
    # 托盘开口在 X ≈ -1.08m，货叉前伸 1.87m。
    # X 必须小于 -2.95m 才能保证不穿透。
    x = sample_uniform(-3.5, -3.0, (len(env_ids), 1), device=self.device)
```

---

## 4. 训练超参数配置 (PPO)

训练配置位于 `agents/rsl_rl_ppo_cfg.py`，针对视觉任务进行了精调：

```python
# 基础运行参数
num_envs = 256              # 降低并行环境数以防止显存 OOM (带有相机渲染极耗显存)
num_steps_per_env = 64      # 每次 rollout 步数
max_iterations = 2000       # 总迭代次数

# 策略配置
init_noise_std = 0.4        # 初始动作噪声 (降低探索压力)
noise_std_type = "log"      # 噪声类型
actor_obs_normalization = False  # 图像已归一化，关闭 Actor 全局归一化
critic_obs_normalization = True  # Critic 的低维状态保持归一化

# PPO 算法参数
learning_rate = 3e-4
entropy_coef = 0.0005       # 较低的熵系数，适合精修模式
desired_kl = 0.008          # 较小的 KL 散度目标，保守更新
```

---

## 5. 如何复现项目

### 5.1 环境准备
确保已安装 IsaacLab，并处于项目的根目录下。
如果修改了 `forklift_pallet_insert_lift_project` 目录下的代码，必须先运行同步脚本将其 Patch 到 IsaacLab 中：
```bash
bash forklift_pallet_insert_lift_project/scripts/install_into_isaaclab.sh IsaacLab
```

### 5.2 验证摄像头视角
在正式训练前，可以使用评估脚本单独查看摄像头的画面，确保能看清货叉和托盘：
```bash
cd IsaacLab
env TERM=xterm PYTHONUNBUFFERED=1 CONDA_PREFIX="" CONDA_DEFAULT_ENV="" \
  ./isaaclab.sh -p ../scripts/tools/camera_eval.py \
  --cam-name test_60deg \
  --cam-x 130 --cam-y 0 --cam-z 250 \
  --pitch-deg 60 \
  --headless --enable_cameras
```
输出的视频和关键帧会保存在 `outputs/camera_eval/test_60deg/` 目录下。

### 5.3 启动训练
使用以下命令启动带相机渲染的 PPO 训练（注意必须加 `--enable_cameras` 参数）：
```bash
cd IsaacLab
nohup env TERM=xterm PYTHONUNBUFFERED=1 CONDA_PREFIX="" CONDA_DEFAULT_ENV="" \
  bash isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Forklift-PalletInsertLift-Direct-v0 \
  --headless \
  --num_envs 256 \
  --max_iterations 2000 \
  --enable_cameras \
  > ../logs/$(date +%Y%m%d_%H%M%S)_train_vision_baseline.log 2>&1 &
```
*注：`num_envs` 设为 256 是为了在单张 24GB 显存的 GPU 上稳定运行。如果显存充足，可以适当调大以加快收集速度。*

### 5.4 查看训练过程
可以通过 `tail -f` 查看日志，重点关注以下指标：
- `episode/success_rate_total`: 总成功率
- `phi/phi_ins`: 插入深度奖励（如果该值开始显著大于 0，说明网络已经学会了对齐并插入）
- `diag/near_success_frac`: 接近成功的比例

### 5.5 播放训练好的策略
训练完成后，使用 `play.py` 查看策略表现：
```bash
cd IsaacLab
env TERM=xterm PYTHONUNBUFFERED=1 CONDA_PREFIX="" CONDA_DEFAULT_ENV="" \
  ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
  --task Isaac-Forklift-PalletInsertLift-Direct-v0 \
  --num_envs 1 \
  --checkpoint "<你的模型路径/model_1999.pt>" \
  --headless --video --video_length 600
```

---

## 6. 实验演进与分支说明

### 6.1 `exp/vision_cnn/scratch_rl_baseline` 分支
这是最初的视觉 Baseline，验证了最基础的 3 层 CNN 在 64x64 分辨率下的可行性。

### 6.2 `exp/vision_cnn/cnn_pretrain` 分支
在 Baseline 的基础上进行了两项重要升级：
1. **升级 Backbone**：将简单的 3 层 CNN 替换为 **MobileNetV3-Small**，以提取更丰富的空间与语义特征。
2. **支持预训练权重加载**：在 `vision_actor_critic.py` 中增加了对预训练 Backbone 权重的加载支持。
*注：该分支跑出了纯 RL 从零训练 MobileNetV3 的 Baseline，最终成功率卡在 20% 左右。*

### 6.3 `exp/vision_cnn/expert_data_collection` 分支
为了突破 20% 的成功率瓶颈，我们拉取了该分支用于**视觉预训练 (Visual Pretraining)**：
1. **数据采集**：编写了 `collect_expert_data.py`，利用 15 维输入的专家策略（`model_1999.pt`，成功率 ~70%），在开启摄像头的环境中跑仿真，收集了 64,000 帧 `(RGB图像, 物理状态)` 的 HDF5 数据集。
2. **监督学习预训练**：编写了 `train_visual_pretrain.py`，让 MobileNetV3 学习直接从图像预测托盘的 `(x, y, yaw)`。经过 30 个 Epoch 的训练，横向误差 (Y) 降至 0.16m，偏航角误差降至 2.9°。最佳权重保存在 `outputs/vision_pretrain/best_backbone.pt`。

### 6.4 `exp/vision_cnn/rl_finetune` 分支 (当前)
在完成预训练后拉取此分支，用于**阶段三：强化学习微调 (RL Finetuning)**。
计划在此分支中：
1. 修改 RL 训练配置，加载预训练好的 `best_backbone.pt`。
2. 冻结（或以极小学习率微调）MobileNetV3 的特征层。
3. 重新启动 PPO 训练，观察是否能突破 20% 的成功率瓶颈。
