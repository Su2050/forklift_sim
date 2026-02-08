# GPU崩溃问题诊断 - 2026-02-07

## 问题描述

运行 `play.py` 脚本时，GPU崩溃并报错：
```
[Error] [carb.graphics-vulkan.plugin] VkResult: ERROR_DEVICE_LOST
[Error] [gpu.foundation.plugin] A GPU crash occurred. Exiting the application...
Reasons for the failure: a device lost, out of memory, or an unexpected bug.
```

## 环境信息

- **系统架构**: ARM64 (NVIDIA Tegra GB10)
- **GPU**: NVIDIA Tegra NVIDIA GB10
- **GPU内存**: 91927 MB (~90GB)
- **CUDA Capability**: 12.1 (超出PyTorch支持的12.0上限)
- **运行模式**: Headless渲染模式 (`isaaclab.python.headless.rendering.kit`)
- **命令参数**: `--headless --enable_cameras --video --video_length 600`

## 错误特征

1. **多个PNG纹理加载失败**：
   - `T_Forklift_C01_Normal.1005.png`
   - `T_Forklift_C01_Normal.1004.png`
   - `T_Forklift_C01_Normal.1006.png`
   - `T_Forklift_C01_Normal.1002.png`
   - 错误信息：`PNG not supported: unknown PNG chunk type`

2. **GPU设备丢失**：
   - Vulkan驱动报错：`ERROR_DEVICE_LOST`
   - 发生在纹理加载过程中

3. **CUDA兼容性警告**：
   - GPU CUDA capability 12.1 超出PyTorch支持范围 (8.0-12.0)

## 可能原因

1. **资源消耗过大**
   - 同时启用 `--enable_cameras` 和 `--video` 会显著增加GPU内存和计算负担
   - 视频录制600步（约20秒）需要持续渲染和编码

2. **纹理加载失败导致GPU驱动错误**
   - PNG纹理文件损坏或格式不兼容
   - Vulkan驱动在处理失败纹理时崩溃

3. **ARM架构+Vulkan驱动兼容性问题**
   - NVIDIA Tegra在ARM架构上的Vulkan驱动可能存在稳定性问题
   - 特别是在处理复杂渲染场景时

4. **CUDA Capability兼容性问题**
   - 虽然只是警告，但可能在特定操作时导致问题

## 解决方案

### 方案1: 移除 `--enable_cameras`（推荐）

如果只是测试模型性能，不需要摄像头输入，移除该参数：

```bash
cd /home/uniubi/projects/forklift_sim/IsaacLab
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
  --task Isaac-Forklift-PalletInsertLift-Direct-v0 \
  --num_envs 1 \
  --checkpoint "/home/uniubi/projects/forklift_sim/IsaacLab/logs/rsl_rl/forklift_pallet_insert_lift/2026-02-07_07-24-08_exp_s1.0e/model_1999.pt" \
  --headless \
  --video --video_length 600
```

**关键变化**: 移除 `--enable_cameras` 参数

### 方案2: 减少视频长度

如果必须使用摄像头，减少视频录制长度：

```bash
cd /home/uniubi/projects/forklift_sim/IsaacLab
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
  --task Isaac-Forklift-PalletInsertLift-Direct-v0 \
  --num_envs 1 \
  --checkpoint "/home/uniubi/projects/forklift_sim/IsaacLab/logs/rsl_rl/forklift_pallet_insert_lift/2026-02-07_07-24-08_exp_s1.0e/model_1999.pt" \
  --headless \
  --enable_cameras \
  --video --video_length 100
```

**关键变化**: 将 `--video_length` 从600减少到100（约3.3秒）

### 方案3: 禁用视频录制（如果不需要视频）

如果只需要测试模型，不需要录制视频：

```bash
cd /home/uniubi/projects/forklift_sim/IsaacLab
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
  --task Isaac-Forklift-PalletInsertLift-Direct-v0 \
  --num_envs 1 \
  --checkpoint "/home/uniubi/projects/forklift_sim/IsaacLab/logs/rsl_rl/forklift_pallet_insert_lift/2026-02-07_07-24-08_exp_s1.0e/model_1999.pt" \
  --headless \
  --enable_cameras
```

**关键变化**: 移除 `--video` 和 `--video_length` 参数

### 方案4: 检查GPU状态和清理资源

在运行前检查GPU状态：

```bash
# 检查GPU状态
nvidia-smi

# 检查是否有其他进程占用GPU
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv

# 如果发现占用GPU的进程，可以kill掉（谨慎操作）
# kill <PID>

# 检查Vulkan驱动
vulkaninfo | head -50
```

### 方案5: 降低渲染质量（如果支持）

如果Isaac Sim支持，可以尝试降低渲染分辨率或质量设置。

## 参数说明

- `--headless`: 无头模式，不显示GUI（已使用）
- `--enable_cameras`: 启用传感器相机，用于视觉输入（会增加GPU负担）
- `--video`: 启用视频录制（会增加GPU负担）
- `--video_length`: 录制视频的步数（600步≈20秒）

**注意**: `--enable_cameras` 和 `--video` 都会增加GPU负担，同时使用可能导致资源不足。

## 推荐测试顺序

1. **首先尝试方案1**（移除 `--enable_cameras`）
   - 如果成功，说明是资源不足问题
   - 如果仍然失败，继续下一步

2. **尝试方案2**（减少视频长度）
   - 如果成功，说明是视频录制资源问题

3. **尝试方案3**（禁用视频录制）
   - 如果成功，确认是视频录制导致的问题

4. **如果所有方案都失败**
   - 检查GPU驱动和Vulkan支持（方案4）
   - 查看系统日志：`sudo dmesg | tail -100 | grep -i "gpu\|vulkan\|memory"`

## 技术说明

### 资源消耗对比

| 模式 | GPU内存消耗 | 稳定性 |
|------|------------|--------|
| 仅headless | 低 | 高 |
| headless + video | 中 | 中 |
| headless + cameras | 中-高 | 中 |
| headless + cameras + video | 高 | 低（可能崩溃） |

### 纹理加载失败的影响

PNG纹理加载失败本身不应该导致GPU崩溃，但在ARM架构+Vulkan环境下，驱动可能无法正确处理错误，导致设备丢失。

## 相关文档

- [进程被杀死问题诊断](./2026-02-02_process_killed_during_play.md)
- [Isaac Lab 故障排除指南](../learning_guiding/06_troubleshooting.md)
- [视频录制文档](../learning_guiding/05_evaluation_and_export.md)
