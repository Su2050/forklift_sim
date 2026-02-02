# DGX Spark：Isaac Sim + Isaac Lab 安装与叉车任务推进记录

> 目标：在 DGX Spark 上跑通 **Isaac Sim + Isaac Lab + RSL-RL**，并进一步训练 **叉车“对准→插入→抬升”** 任务。

---

## 已完成（刚刚做过的事情）

### 0）环境体检
- 确认 GPU/驱动/系统版本：
  - NVIDIA Driver：`580.95.05`
  - CUDA：`13.0`
  - Ubuntu：`24.04.3 LTS`
- 发现并解决：
  - `git lfs` 未安装 → 安装并初始化

### 1）编译工具链对齐（按 Spark 官方建议）
- 安装并切换默认编译器到 GCC/G++ 11：
  - `gcc/g++ -> 11.5`

### 2）获取并构建 Isaac Sim（源码构建）
- 解决 GitHub HTTPS TLS 不稳定：
  - 设置 Git 使用 HTTP/1.1（避免 TLS/HTTP2 抖动导致 clone 失败）
- 克隆 IsaacSim 仓库（shallow）并拉取 LFS：
  - `git lfs install` + `git lfs pull`
- 确认该版本无 submodule（没有 `.gitmodules`）
- 执行构建：
  - `./build.sh | tee build.log`
  - **构建成功：`BUILD (RELEASE) SUCCEEDED`**

### 3）导出 Isaac Sim 路径（后续 IsaacLab 需要）
- 设置环境变量并验收 `python.sh` 存在：
  - `ISAACSIM_PATH=.../_build/linux-aarch64/release`
  - `ISAACSIM_PYTHON_EXE=$ISAACSIM_PATH/python.sh`

### 4）获取 Isaac Lab（用瘦身 clone）
- 由于全量 clone 太慢，改用 shallow + single-branch + filter：
  - 分支：`release/2.3.0`
  - `--depth=1 --single-branch --filter=blob:none`
- Submodule 更新（若有）：
  - `git submodule update --init --recursive --depth 1 --jobs 8`

### 5）把 IsaacLab 指向本地构建的 IsaacSim
- 创建软链接 `_isaac_sim`，并验收 `python.sh` 可见：
  - `ln -sfn $ISAACSIM_PATH ./_isaac_sim`

### 6）安装 IsaacLab + RL 组件
- 运行 `./isaaclab.sh --install`：
  - 过程中完成 torch/torchvision(CUDA13) 等依赖安装
  - 遇到 pip 解析/下载慢 & GitHub TLS 抖动（robomimic）问题
  - 结论：**RL 训练不依赖 robomimic，可先跳过**
- 验证核心可用：
  - `./_isaac_sim/python.sh -c "import isaaclab; print('isaaclab OK')"` ✅
- 安装 RSL-RL（PPO 训练所需）：
  - `./isaaclab.sh -i rsl_rl | tee rsl_rl_install.log`

---

## 当前状态（你现在已经拥有）
- ✅ Isaac Sim 已在 Spark 上成功 build（可用 release 产物）
- ✅ Isaac Lab 已拉取并链接到本地 IsaacSim
- ✅ IsaacLab Python 环境可 import（`isaaclab OK`）
- ✅ RSL-RL 训练框架组件已安装（可跑 PPO）

---

## 接下来要做（下一步计划）

### Step A：跑官方 sample RL「冒烟测试」
目的：确认 **Sim + Lab + RL** 的训练链路完全打通。

在 `~/projects/forklift_sim/IsaacLab` 执行：
```bash
cd ~/projects/forklift_sim/IsaacLab
export LD_PRELOAD="$LD_PRELOAD:/lib/aarch64-linux-gnu/libgomp.so.1"

./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task=Isaac-Velocity-Rough-H1-v0 \
  --headless \
  --num_envs 64 \
  2>&1 | tee smoke_rl.log
```
验收标准：
- 终端开始持续输出训练日志
- 生成 logs/runs 目录与 checkpoint

---

### Step B：安装“叉车对准→插入→抬升”任务 patch
目的：把自定义任务注册到 IsaacLab tasks 中。

1）把 `forklift_pallet_insert_lift_project_v2.zip` 传到 Spark（放到 `~/projects/` 之类）
2）解压并安装 patch：
```bash
cd ~/projects
unzip forklift_pallet_insert_lift_project_v2.zip
cd forklift_pallet_insert_lift_project

bash scripts/install_into_isaaclab.sh ~/projects/forklift_sim/IsaacLab
```

---

### Step C：开训叉车任务（PPO / headless）
```bash
cd ~/projects/forklift_sim/IsaacLab
export LD_PRELOAD="$LD_PRELOAD:/lib/aarch64-linux-gnu/libgomp.so.1"

./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Forklift-PalletInsertLift-Direct-v0 \
  --headless \
  --num_envs 128 \
  2>&1 | tee forklift_train.log
```

---

### Step D：评估 / 回放 / 录视频（可选）
训练有 checkpoint 后：
```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
  --task Isaac-Forklift-PalletInsertLift-Direct-v0 \
  --headless --video --video_length 300 \
  --num_envs 16 \
  --load_run <RUN_NAME> \
  --checkpoint <PATH/TO/model_XXXX.pt>
```

---

## 经验记录（踩坑点与解决方案）
- GitHub TLS 抖动：
  - 建议保持：`git config --global http.version HTTP/1.1`
- pip 下载/解析慢：
  - 可设：`PIP_DEFAULT_TIMEOUT=300`、`PIP_RETRIES=20`
- robomimic 拉取失败：
  - 与 PPO/RSL-RL 主线无关，可先跳过；需要模仿学习时再补。

---

## 建议的“里程碑”
1) `smoke_rl.log` 能跑起来（冒烟测试通过）
2) `Isaac-Forklift-PalletInsertLift-Direct-v0` 开训并产出 checkpoint
3) play 回放能看到“对准→插入→抬升”雏形



---

## 进展更新（已完成冒烟测试）
- 你已成功跑起官方 RSL-RL 训练循环（冒烟测试通过）：训练日志持续输出、迭代正常推进。
- 关键观察：
  - `steps/s` ~ 2600：采样与学习链路跑通且性能正常
  - `Episode_Termination/base_contact: 1.0000`：当前阶段几乎每个 episode 都因“机体底盘触地/摔倒”终止（早期很常见，不影响“链路是否通”的判断）
  - `Mean reward` 为负：该任务包含多项惩罚项，早期负值正常；更关注是否随迭代上升、episode length 是否变长

### 冒烟测试到此即可
- 只要你确认训练在稳定迭代、日志/ckpt 能生成，就可以 `Ctrl+C` 停止，进入叉车任务。

---

## 下一步（进入叉车对准→插入→抬升任务）
1) 将 `forklift_pallet_insert_lift_project_v2.zip` 传到 Spark 的 `~/projects/`
2) 解压并安装 task patch：
```bash
cd ~/projects
unzip forklift_pallet_insert_lift_project_v2.zip
cd forklift_pallet_insert_lift_project
bash scripts/install_into_isaaclab.sh ~/projects/forklift_sim/IsaacLab
```
3) 开训叉车任务（PPO/headless）：
```bash
cd ~/projects/forklift_sim/IsaacLab
export LD_PRELOAD="$LD_PRELOAD:/lib/aarch64-linux-gnu/libgomp.so.1"

./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Forklift-PalletInsertLift-Direct-v0 \
  --headless \
  --num_envs 128 \
  2>&1 | tee forklift_train.log
```

