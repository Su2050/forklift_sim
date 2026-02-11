# Forklift Expert Policy（路线 A：规则专家 → 示范数据 → BC → RL 微调）

完整的"专家示范 → BC 预训练 → RL 微调"工程链路，适用于 IsaacLab 叉车叉托盘任务。

---

## 快速开始（6 步闭环）

```bash
# 在 IsaacLab 仓库根目录执行（./isaaclab.sh -p ...）

# 1. 目视验证 expert 行为（录制视频）
./isaaclab.sh -p forklift_expert_policy_project/scripts/play_expert.py \
  --task Isaac-Forklift-PalletInsertLift-Direct-v0 \
  --num_envs 1 --headless --video --video_length 600

# 2. 批量采集 demo 数据
./isaaclab.sh -p forklift_expert_policy_project/scripts/collect_demos.py \
  --task Isaac-Forklift-PalletInsertLift-Direct-v0 \
  --num_envs 64 --episodes 3000 --headless

# 3. 检查 demo 数据分布
python scripts/analyze_demos.py data/demos_xxx.npz

# 4. BC 预训练（输出 rsl_rl 兼容 checkpoint）
./isaaclab.sh -p forklift_expert_policy_project/scripts/bc_train.py \
  --demos data/demos_xxx.npz --out data/bc_model_0.pt --epochs 200

# 5. 验证 BC checkpoint
./isaaclab.sh -p forklift_expert_policy_project/scripts/bc_train.py \
  --demos data/demos_xxx.npz --verify data/bc_model_0.pt

# 6. 接入 RL 训练（BC 权重作为 PPO actor 初始化）
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Forklift-PalletInsertLift-Direct-v0 \
  --resume --load_run bc_pretrain --checkpoint bc_model_0.pt \
  --headless --num_envs 1024 --max_iterations 2000
```

---

## 项目结构

```
forklift_expert_policy_project/
├── forklift_expert/               # 核心 expert 模块
│   ├── __init__.py
│   ├── expert_policy.py           # 规则专家策略（三阶段控制器）
│   ├── obs_spec.json              # 15 维 obs 字段 → index 映射（已按实际 env 填好）
│   └── action_spec.json           # 3 维 action 定义（drive/steer/lift）
├── scripts/
│   ├── play_expert.py             # Expert 回放 + headless 视频录制
│   ├── collect_demos.py           # 批量采集 demo 数据（适配 IsaacLab 向量环境）
│   ├── analyze_demos.py           # demo 数据分布统计
│   └── bc_train.py                # 正式 BC 训练（rsl_rl 兼容 checkpoint 输出）
├── bc_train_design.md             # BC 脚本设计文档（含 rsl_rl 源码参考）
├── instructions.md                # 详细说明文档（背景、原理、完整链路）
├── pyproject.toml
├── setup.cfg
└── README.md                      # ← 你在看的文件
```

---

## 15 维观测空间（已适配）

观测来自 `env._get_observations()`，`obs_spec.json` 已填好全部映射：

| 维度 | 字段 | 含义 |
|------|------|------|
| 0-1 | `d_xy_r` | 机器人→托盘中心的 2D 相对位置（robot frame, m）|
| 2-3 | `cos_dyaw`, `sin_dyaw` | 偏航差的 cos/sin 编码 |
| 4-5 | `v_xy_r` | 机器人线速度（robot frame, m/s）|
| 6 | `yaw_rate` | 偏航角速度（rad/s）|
| 7-8 | `lift_pos`, `lift_vel` | lift 关节位置/速度 |
| 9 | `insert_norm` | 插入深度归一化（0~1）|
| 10-12 | `prev actions` | 上一步动作（drive, steer, lift）|
| 13 | `y_err_obs` | 横向误差（pallet frame, 归一化 ÷0.5m, clip [-1,1]）|
| 14 | `yaw_err_obs` | 偏航误差（pallet frame, 归一化 ÷15deg, clip [-1,1]）|

Expert 的 `_decode_obs()` 自动处理：
- `dist_front` = `d_xy_r[0] - pallet_half_depth`
- `lateral_err` = `y_err_obs * 0.5`（反归一化为米）
- `yaw_err` = `atan2(sin_dyaw, cos_dyaw)`（完整弧度）

---

## Expert 策略逻辑

三阶段规则控制器（Docking → Insertion → Lift），基于 `insert_norm` 切换阶段：

**Docking**（`insert_norm < 0.15`）
- `steer = k_lat * lateral_err + k_yaw * yaw_err`
- `drive` 与距离成比例，近距离减速，偏差大时减速留出转向余量

**Insertion**（`0.15 <= insert_norm < 0.75`）
- 严格对齐门控：`|lateral| <= 3cm` 且 `|yaw| <= 3deg` 才允许前进
- 不达标时 `drive=0`，只打方向修正

**Lift**（`insert_norm >= 0.75`）
- 停止前进，执行举升（`lift=0.60`）

---

## BC 训练脚本特性

`scripts/bc_train.py` 是正式的 BC 训练器，输出的 checkpoint 可直接被 rsl_rl 加载：

- 网络结构与 `rsl_rl.modules.ActorCritic` 完全一致（MLP [256,256,128] + ELU）
- state_dict 25 个 key 精确匹配（actor + critic + normalizer + log_std）
- Obs 归一化从 demo 数据统计，填入 `EmpiricalNormalization` buffer
- Cosine LR scheduler + warmup + gradient clipping + early stopping
- 每 epoch 打印 per-action-dim MSE（drive/steer/lift）
- `--verify` 模式：加载 checkpoint 对比 BC 输出 vs demo action

---

## Headless 视频录制

所有可视化均支持 SSH 远程 headless 模式：

| 场景 | 命令 |
|------|------|
| Expert 行为验证 | `play_expert.py --video --headless` |
| BC checkpoint 回放 | `bc_train.py --verify ... --video --headless`（规划中）|
| RL checkpoint 回放 | IsaacLab `play.py --video --headless` |

---

## 建议

- **先只做 Docking 的 demos**，把对齐行为打稳，再扩展到 Insertion/Lift
- 如果 expert 在某些初始状态经常失败，优先调 expert 规则（限速、转向增益等），不要硬堆 episodes
- BC 训练后用 `--verify` 确认误差合理，再接入 RL 训练

---

## 已知限制

- `y_err_obs` clip 在 [-1,1]（±0.5m），远距离大偏移信息会丢失
- `contact_flag` / `slip_flag` 不在 15 维 obs 中，expert 倒车重试功能默认禁用
- BC 只训练 actor，critic 为随机初始化（RL 续训会很快覆盖）
