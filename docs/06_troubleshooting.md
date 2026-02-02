# 常见问题排障（按现象查）

本页适合：遇到“报错/不动/很慢/不收敛/一直翻车”等问题的人。
建议从上到下按现象查。

---

## 1. 报错：Environment doesn’t exist / task not found

### 现象

- 运行 `train.py` 或 `play.py` 时提示：找不到 `Isaac-Forklift-PalletInsertLift-Direct-v0`

### 原因（最常见）

- 任务没被 import，导致 gym 注册没执行
- 你在错误的目录里运行（不是 IsaacLab 根目录），导致 `isaaclab_tasks` 没被正确加载

### 解决

1) 确认任务目录存在：
   - `IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/forklift_pallet_insert_lift/`
2) 确认 `direct/__init__.py` 有这一行（或至少包含 `forklift_pallet_insert_lift`）：
   - `from . import forklift_pallet_insert_lift  # noqa: F401`
3) 如果你是“外部 patch 安装到另一份 IsaacLab”，重新执行：

```bash
bash scripts/install_into_isaaclab.sh /path/to/IsaacLab
```

---

## 2. 现象：叉车不动 / 只抖动 / 原地转圈

### 原因候选

- 叉车 USD 的**关节名字和代码不一致**
- 动作被夹住（[-1,1]），但缩放参数过小导致动力不够
- 训练初期策略随机输出，表现像“乱动”是正常的

### 如何快速确认关节名是否匹配

用随机策略跑 1 个环境，看是否有控制输出（并结合 UI 观察）：

```bash
cd <你的IsaacLab目录>
./isaaclab.sh -p scripts/environments/random_agent.py \
  --task Isaac-Forklift-PalletInsertLift-Direct-v0 \
  --num_envs 1
```

如果关节名不匹配，你会看到明显异常（例如完全无运动、控制目标无法应用）。\n此时需要回到环境代码中核对 `find_joints([...])` 的关节名称列表。

---

## 3. 现象：训练很慢 / steps/s 很低

### 常见原因

- 没开 `--headless`（UI 开着会慢很多）
- `--num_envs` 太大导致显存溢出或频繁换页
- 机器 CPU/GPU 资源不足、驱动/电源策略限制

### 建议操作

- 先用 headless：

```bash
--headless
```

- 降低并行环境数：

```bash
--num_envs 32
```

- 先冒烟验证链路是否通，再逐步加大 `num_envs`。

---

## 4. 现象：一直倾翻（episode 很短），几乎学不会

### 可能原因

- 初始随机范围太大（出生姿态偏航太大、离托盘太近/太歪）
- 轮速/转向/升降上限过大，导致动作过猛
- 奖励里“动作惩罚”太弱，策略学会猛冲

### 你可以先做的“非改代码”排查

- 降低 `--num_envs`，开 `--video` 观察具体翻车方式
- 训练先跑少量迭代，看看是否有进步趋势

### 进阶（需要改配置/代码时）

可以从这些方向调参（这部分我后续也可以单独给一份“调参指南”）：

- 降低 `wheel_speed_rad_s` 或 `steer_angle_rad`
- 增大 `rew_action_l2`（更强的动作惩罚）
- 缩小 reset 时 yaw 采样范围
- 放宽成功阈值（先学会大致对齐再收紧）

---

## 5. 现象：reward 一直是负的，是不是没在学习？

不一定。\n这个任务包含多项惩罚（对齐误差、偏航误差、动作惩罚），早期负值很常见。\n
建议看这些更可靠的信号：

- episode length 是否变长
- 回放时是否出现“对准→插入”的稳定趋势
- 是否开始出现成功（哪怕很低）

---

## 6. 现象：pallet 相关报错（rigid body / physics API）

该任务环境里专门有一段“给 pallet 自动补物理 API”的逻辑，目的就是：\n有些 `pallet.usd` 只是视觉模型，不带刚体 API，会导致 RigidObject 初始化失败。\n
如果你仍遇到 pallet 初始化问题，优先确认：

- 你使用的 Isaac Sim 资产路径是否正确
- `ISAAC_NUCLEUS_DIR` 是否可访问
- 任务在 `_setup_scene()` 中是否执行到了 pallet patch 的逻辑

---

## 7. 下一步读什么

- 想了解"任务目标/奖励/观测是怎么设计的"：`docs/03_task_design_rl.md`
- 想给叉车加 RGB 相机、改成图像输入、做进阶迭代：`docs/07_vision_input.md`
- 想看一次真实排障复盘（Mean episode length 长期=1.00 的定位与修复）：`docs/troubleshooting_cases/2026-02-02_episode_length_reset_postmortem.md`