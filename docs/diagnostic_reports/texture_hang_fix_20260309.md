# 诊断与修复报告：Isaac Sim 视觉训练启动卡死问题

**日期**: 2026-03-09
**环境**: `exp/vision_cnn/scratch_rl_baseline`
**问题描述**: 在启动带有视觉传感器 (`--enable_cameras`) 的 `scratch-rl` 训练时，程序在输出 `[INFO]: Starting the simulation. This may take a few seconds. Please wait...` 后永久卡死，无法进入 `Learning iteration`。

## 1. 现象分析与对比

在排查过程中，我们对比了不同运行模式的表现：

*   **`sanity_check` (无相机)**: 能够正常启动并完成迭代。因为没有携带 `--enable_cameras` 参数，RTX 渲染器未激活，不加载材质。
*   **`collect-data` (有相机，早期)**: 能够启动。虽然开启了相机，但当时 Omniverse 的本地缓存 (`~/.cache/ov`) 完好，遇到不支持的贴图格式（PNG）时会迅速报错跳过（0.5秒内）。
*   **`formal scratch-rl` (有相机，当前)**: 永久卡死。伴随大量 `PNG not supported: unknown PNG chunk type` 的报错，并且报错间隔长达几十秒。

## 2. 根本原因 (Root Causes)

导致卡死的直接原因和深层原因结合在一起，形成了一个死锁链：

1.  **贴图格式不支持 (直接触发点)**:
    Isaac Sim 的 RTX 渲染器在加载 ForkliftC 资产时，尝试从 S3 下载并解析某些 PNG 贴图（如 `T_Forklift_C01_Normal.1005.png`），但这些文件包含未知的 PNG Chunk，导致解析失败报错。
2.  **僵尸进程导致缓存死锁 (核心原因)**:
    之前被中断或崩溃的运行遗留了后台的 `kit` 和 `train.py` 僵尸进程。这些僵尸进程**锁死了 Omniverse 的本地缓存数据库 (`omni.kvdb.plugin`)**。
3.  **并发下载与超时雪崩 (卡死表现)**:
    由于缓存被锁死，渲染器无法读取本地已有的缓存（即使是损坏的缓存），被迫为 **128 个环境** 重新发起海量的并发下载请求（每个贴图约 12MB）。网络延迟加上解析失败，导致单个贴图的报错周期长达 70 多秒。几百个贴图排队超时，使得仿真初始化过程看起来像永久卡死。
4.  **`wait_for_textures` 默认行为**:
    `DirectRLEnvCfg` 默认开启了 `wait_for_textures = True`。这要求环境在 `reset()` 阶段必须等待所有贴图加载完毕才能继续。由于贴图一直在超时重试，环境被永久阻塞。

## 3. 解决与修复方法

为了彻底解决这个问题并防止未来再次发生，我们采取了以下三个层面的修复：

### 3.1 代码层修复 (Bypass)
在 `ForkliftPalletInsertLiftEnvCfg` 中显式禁用等待贴图加载：
```python
# 禁用等待贴图加载，避免因 PNG 解析报错导致仿真启动挂起
wait_for_textures: bool = False
```
*提交记录*: `fix(vision-cnn): disable wait_for_textures to prevent training hang`

### 3.2 系统状态清理 (Cleanup)
编写并执行了一个清理脚本，强制杀死所有可能占用缓存的僵尸进程：
```bash
#!/bin/bash
echo "Killing zombie train.py and kit processes..."
pgrep -af "train.py" | awk '{print $1}' | xargs -r kill -9
pgrep -af "kit" | awk '{print $1}' | xargs -r kill -9
echo "Done."
```
*提交记录*: `chore(vision-cnn): add script to clean up zombie kit processes`

### 3.3 缓存重置与重建 (Reset)
清理了损坏的 Omniverse 本地缓存，并让 Isaac Sim 重新拉取扩展包：
```bash
rm -rf ~/.cache/ov ~/.local/share/ov ~/.nvidia-omniverse/
cd IsaacLab && ./isaaclab.sh --install
```

## 4. 训练结果验证

修复完成后，我们重新启动了 `scratch-rl` 训练 (`20260309_233639_train_s1.0zd.log`)。
*   **是否正常进入迭代**: 是。不再卡在 `Starting the simulation`，成功进入了 `Learning iteration 1/2000`。
*   **是否运行完毕**: 是。训练已成功执行到 `Learning iteration 1999/2000`，总步数达到 16,384,000，耗时约 2小时41分钟。
*   **训练结果简报**:
    *   `episode/success_rate_total`: **0.3502** (35.02%)
    *   `diag/near_success_frac`: 1.0000 (100% 能够接近托盘)
    *   `diag/deep_insert_frac`: 0.6172 (61.72% 能够深度插入)
    *   `err/yaw_deg_near_success`: 8.76度
    *   `err/lateral_near_success`: 0.18米

**结论**: 从头训练的视觉基线 (Scratch RL Baseline) 已经成功跑通，证明基于 MobileNetV3-Small 的视觉 Actor 架构在当前环境中是可行的，并取得了 35% 的初步成功率。这为后续的预训练权重微调 (Pre-trained RL Fine-tune) 提供了可靠的对比基线。