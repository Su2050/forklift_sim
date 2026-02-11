# 正式 BC 训练脚本设计

## 核心目标

训练出的 BC 权重能**直接通过 `--resume` 加载到 rsl_rl PPO 训练中**，无需任何格式转换。

## 关键兼容性约束

BC 脚本必须精确复现 rsl_rl 的网络结构和 checkpoint 格式：

- **Actor MLP** 结构：`Linear(15,256) + ELU + Linear(256,256) + ELU + Linear(256,128) + ELU + Linear(128,3)`
  - state_dict keys: `actor.0.weight` [256,15], `actor.2.weight` [256,256], `actor.4.weight` [128,256], `actor.6.weight` [3,128]
- **log_std** 参数：`nn.Parameter` shape [3]，初始化为 `log(0.5)` 与训练配置一致
- **Obs normalizer**：`EmpiricalNormalization(15)`，从 demo 数据统计 mean/var/std
  - buffers: `actor_obs_normalizer._mean` [1,15], `._var` [1,15], `._std` [1,15], `.count`
- **Critic MLP**：BC 不训练 critic，但 checkpoint 中需包含随机初始化的 critic 权重
  - 结构：`Linear(15,256) + ELU + ... + Linear(128,1)`
- **Checkpoint 格式**：`{"model_state_dict": ..., "optimizer_state_dict": ..., "iter": 0, "infos": None}`

## rsl_rl 源码关键参考

### ActorCritic 类（`rsl_rl.modules.actor_critic`）

```python
class ActorCritic(nn.Module):
    def __init__(self, obs, obs_groups, num_actions, ...):
        self.actor = MLP(num_actor_obs, num_actions, actor_hidden_dims, activation)
        self.critic = MLP(num_critic_obs, 1, critic_hidden_dims, activation)
        # noise_std_type="log" 时:
        self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        # actor_obs_normalization=True 时:
        self.actor_obs_normalizer = EmpiricalNormalization(num_actor_obs)
        self.critic_obs_normalizer = EmpiricalNormalization(num_critic_obs)

    def act_inference(self, obs):
        obs = self.get_actor_obs(obs)
        obs = self.actor_obs_normalizer(obs)
        return self.actor(obs)
```

### MLP 类（`rsl_rl.networks.mlp`）

```python
class MLP(nn.Sequential):
    def __init__(self, input_dim, output_dim, hidden_dims, activation="elu"):
        # hidden_dims=[256,256,128], activation="elu" 时:
        # layer 0: Linear(15, 256)    → key: actor.0.weight/bias
        # layer 1: ELU()              → key: actor.1 (无参数)
        # layer 2: Linear(256, 256)   → key: actor.2.weight/bias
        # layer 3: ELU()
        # layer 4: Linear(256, 128)   → key: actor.4.weight/bias
        # layer 5: ELU()
        # layer 6: Linear(128, 3)     → key: actor.6.weight/bias
        for idx, layer in enumerate(layers):
            self.add_module(f"{idx}", layer)
```

### EmpiricalNormalization 类（`rsl_rl.networks.normalization`）

```python
class EmpiricalNormalization(nn.Module):
    def __init__(self, shape, eps=1e-2):
        self.register_buffer("_mean", torch.zeros(shape).unsqueeze(0))  # [1, obs_dim]
        self.register_buffer("_var", torch.ones(shape).unsqueeze(0))
        self.register_buffer("_std", torch.ones(shape).unsqueeze(0))
        self.register_buffer("count", torch.tensor(0, dtype=torch.long))

    def forward(self, x):
        return (x - self._mean) / (self._std + self.eps)
```

### OnPolicyRunner checkpoint 格式

```python
# save:
saved_dict = {
    "model_state_dict": self.alg.policy.state_dict(),
    "optimizer_state_dict": self.alg.optimizer.state_dict(),
    "iter": self.current_learning_iteration,
    "infos": infos,
}
torch.save(saved_dict, path)

# load:
loaded_dict = torch.load(path)
self.alg.policy.load_state_dict(loaded_dict["model_state_dict"])
```

### 当前训练配置（rsl_rl_ppo_cfg.py）

```python
policy = RslRlPpoActorCriticCfg(
    class_name="rsl_rl.modules.ClampedActorCritic",
    init_noise_std=0.5,
    noise_std_type="log",
    actor_obs_normalization=True,
    critic_obs_normalization=True,
    actor_hidden_dims=[256, 256, 128],
    critic_hidden_dims=[256, 256, 128],
    activation="elu",
)
```

## 脚本功能设计

文件路径：`forklift_expert_policy_project/scripts/bc_train.py`（替换原最小版）

### 核心功能

1. **构建与 rsl_rl 完全一致的 Actor MLP**
   - 直接使用 `rsl_rl.networks.MLP` 类（如果可 import），否则自行复现等效结构
   - 确保 `nn.Sequential` + `add_module(f"{idx}", layer)` 的命名方式一致

2. **Obs 归一化**
   - 从 demo 数据全量计算 mean/var/std
   - 填入 `EmpiricalNormalization` 的 buffer
   - 训练时先 normalize obs 再送入 actor（与 rsl_rl `act()` 流程一致）

3. **训练**
   - Loss: MSE (obs → action)
   - Optimizer: Adam, lr=3e-4
   - 带 cosine LR scheduler + warmup
   - Gradient clipping (max_norm=1.0)
   - Early stopping (patience=10, 基于 val loss)
   - 可选: 按 stage 加权 loss（insertion/lift 阶段权重更高）

4. **保存为 rsl_rl 兼容 checkpoint**
   - 组装完整 `model_state_dict`（actor + critic + normalizer + log_std）
   - 写入 `{"model_state_dict": ..., "optimizer_state_dict": dummy, "iter": 0, "infos": {"bc_val_loss": ...}}`
   - 输出路径命名: `bc_model_0.pt`（可直接作为 `--checkpoint bc_model_0.pt` 加载）

### 额外实用功能

5. **数据过滤选项**
   - `--stage_filter docking` : 只用 docking 阶段的 transitions
   - `--min_episode_len N` : 过滤太短的 episode

6. **诊断输出**
   - 每 epoch 打印 per-action-dim MSE（drive/steer/lift 分别的 loss）
   - 打印 obs normalizer 统计值（确认数据合理性）
   - 可选 TensorBoard 日志

7. **验证工具** (`--verify` 模式)
   - 加载保存的 checkpoint，跑一小批 demo obs，对比 BC 输出 vs expert 原始 action
   - 打印 action 误差统计，确认 checkpoint 可正常加载

8. **BC verify 视频录制** (`--verify ... --video`)
   - 在 `--verify` 模式下，加载 BC checkpoint 创建 env，用 BC actor 驱动仿真
   - headless + `render_mode="rgb_array"` + `gym.wrappers.RecordVideo` 录制视频
   - 视频保存到 `data/videos/bc_verify/` 目录
   - 可设置 `--video_length` 控制录制步数（默认 600）

## 完整 model_state_dict key 结构（预期）

```
log_std                                  [3]
actor.0.weight                           [256, 15]
actor.0.bias                             [256]
actor.2.weight                           [256, 256]
actor.2.bias                             [256]
actor.4.weight                           [128, 256]
actor.4.bias                             [128]
actor.6.weight                           [3, 128]
actor.6.bias                             [3]
actor_obs_normalizer._mean               [1, 15]
actor_obs_normalizer._var                [1, 15]
actor_obs_normalizer._std                [1, 15]
actor_obs_normalizer.count               scalar (int64)
critic.0.weight                          [256, 15]
critic.0.bias                            [256]
critic.2.weight                          [256, 256]
critic.2.bias                            [256]
critic.4.weight                          [128, 256]
critic.4.bias                            [128]
critic.6.weight                          [1, 128]
critic.6.bias                            [1]
critic_obs_normalizer._mean              [1, 15]
critic_obs_normalizer._var               [1, 15]
critic_obs_normalizer._std               [1, 15]
critic_obs_normalizer.count              scalar (int64)
```

## 新增脚本：Expert 策略视频录制

文件路径：`forklift_expert_policy_project/scripts/play_expert.py`（新建）

### 功能

用规则 expert 策略驱动仿真环境，headless 模式下录制视频，用于目视验证 expert 行为。

### 实现要点

- 创建 env 时使用 `render_mode="rgb_array"` + `enable_cameras=True`
- 用 `gym.wrappers.RecordVideo` 包裹 env
- 每 step 调用 expert.act() 生成动作，送入 env.step()
- 视频保存到 `data/videos/expert_play/` 目录
- 支持参数：`--num_envs 1 --episodes 5 --video_length 600 --headless`

### 使用方式

```bash
./isaaclab.sh -p forklift_expert_policy_project/scripts/play_expert.py \
  --task Isaac-Forklift-PalletInsertLift-Direct-v0 \
  --num_envs 1 --headless \
  --video_length 600
```

---

## 使用方式总览

```bash
# ========== 1. Expert 策略视频回放（验证规则控制器行为） ==========
./isaaclab.sh -p forklift_expert_policy_project/scripts/play_expert.py \
  --task Isaac-Forklift-PalletInsertLift-Direct-v0 \
  --num_envs 1 --headless \
  --video_length 600
# 输出: data/videos/expert_play/*.mp4

# ========== 2. Expert 采集 demo 数据 ==========
./isaaclab.sh -p forklift_expert_policy_project/scripts/collect_demos.py \
  --task Isaac-Forklift-PalletInsertLift-Direct-v0 \
  --num_envs 64 --episodes 3000 --headless
# 输出: data/demos_YYYYMMDD_HHMMSS.npz

# ========== 3. BC 训练 ==========
./isaaclab.sh -p forklift_expert_policy_project/scripts/bc_train.py \
  --demos data/demos_xxx.npz \
  --out data/bc_model_0.pt \
  --epochs 200 --batch_size 2048

# ========== 4. BC 验证（统计 + 视频） ==========
./isaaclab.sh -p forklift_expert_policy_project/scripts/bc_train.py \
  --demos data/demos_xxx.npz \
  --verify data/bc_model_0.pt \
  --video --video_length 600 --headless
# 输出: 误差统计打印 + data/videos/bc_verify/*.mp4

# ========== 5. 接入 RL 训练（直接作为 resume checkpoint） ==========
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Forklift-PalletInsertLift-Direct-v0 \
  --resume --load_run bc_pretrain --checkpoint bc_model_0.pt \
  --headless --num_envs 1024 --max_iterations 2000

# ========== 6. RL checkpoint 回放视频（IsaacLab 已有） ==========
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
  --task Isaac-Forklift-PalletInsertLift-Direct-v0 \
  --headless --video --video_length 600 \
  --load_run xxx --checkpoint model_xxx.pt
# 输出: logs/rsl_rl/.../videos/play/*.mp4
```

## 待确认事项

- [ ] 是否需要从已有训练 checkpoint 中提取 normalizer 统计值（而非从 demo 数据重新计算）
- [ ] BC 训练后 RL 续训时，是否需要冻结 normalizer（防止统计量被在线数据覆盖）
- [ ] 是否需要支持 `--stage_filter` 数据过滤（需要 demo 数据中有 stage 标注）
