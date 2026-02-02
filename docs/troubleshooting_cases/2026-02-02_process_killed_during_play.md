# 进程被杀死问题诊断 - 2026-02-02

## 问题描述

运行 `play.py` 脚本时，进程被系统杀死（killed），错误信息：
```
/home/uniubi/projects/forklift_sim/IsaacLab/_isaac_sim/python.sh: 第 73 行： 220908 已杀死
There was an error running python
```

## 环境信息

- **系统架构**: ARM64 (NVIDIA Tegra GB10)
- **GPU**: NVIDIA Tegra NVIDIA GB10
- **GPU内存**: ~90GB
- **系统内存**: 119GB 总内存，111GB 可用
- **运行模式**: 渲染模式 (`isaaclab.python.rendering.kit`)
- **命令参数**: `--video --video_length 50`

## 可能原因

1. **渲染模式在ARM架构上的兼容性问题**
   - 当前使用渲染模式（非headless），可能在ARM架构上存在驱动或兼容性问题
   - Vulkan驱动可能存在问题

2. **GPU内存分配失败**
   - 虽然GPU显示有90GB内存，但在启动时可能无法正确分配
   - 其他进程可能占用了GPU资源

3. **视频录制功能导致资源问题**
   - `--video` 参数启用了摄像头和渲染，可能消耗过多资源

## 解决方案

### 方案1: 使用Headless模式（推荐）

即使需要录制视频，headless模式也支持渲染和视频录制：

```bash
cd /home/uniubi/projects/forklift_sim/IsaacLab
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
  --task Isaac-Forklift-PalletInsertLift-Direct-v0 \
  --num_envs 1 \
  --checkpoint "/home/uniubi/projects/forklift_sim/IsaacLab/logs/rsl_rl/forklift_pallet_insert_lift/2026-02-02_18-13-10/model_1999.pt" \
  --headless \
  --video --video_length 50
```

**关键变化**: 添加 `--headless` 参数

### 方案2: 禁用视频录制（如果不需要视频）

如果不需要录制视频，可以移除 `--video` 参数：

```bash
cd /home/uniubi/projects/forklift_sim/IsaacLab
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
  --task Isaac-Forklift-PalletInsertLift-Direct-v0 \
  --num_envs 1 \
  --checkpoint "/home/uniubi/projects/forklift_sim/IsaacLab/logs/rsl_rl/forklift_pallet_insert_lift/2026-02-02_18-13-10/model_1999.pt"
```

### 方案3: 检查GPU驱动和Vulkan支持

```bash
# 检查GPU状态
nvidia-smi

# 检查Vulkan支持
vulkaninfo | head -50

# 检查是否有其他进程占用GPU
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
```

### 方案4: 减少资源使用

如果问题仍然存在，可以尝试：

1. **减少环境数量**（已经是1，无法再减少）
2. **减少视频长度**：将 `--video_length` 从50减少到更小的值
3. **使用更简单的渲染模式**

## 技术说明

根据 `app_launcher.py` 的实现：

- **默认模式**（无 `--headless` + `--video`）: 使用 `isaaclab.python.rendering.kit`
- **Headless模式**（`--headless` + `--video`）: 使用 `isaaclab.python.headless.rendering.kit`
- **Headless模式**（`--headless` + 无 `--video`）: 使用 `isaaclab.python.headless.kit`

Headless模式仍然支持渲染和视频录制，只是不显示GUI窗口，这在服务器环境或ARM架构上通常更稳定。

## 验证步骤

1. 先尝试方案1（添加 `--headless`）
2. 如果成功，说明是渲染模式的兼容性问题
3. 如果仍然失败，检查系统日志：
   ```bash
   # 需要sudo权限
   sudo dmesg | tail -100 | grep -i "killed\|oom\|memory"
   ```

## 相关文档

- [Isaac Lab 故障排除指南](../06_troubleshooting.md)
- [视频录制文档](../05_evaluation_and_export.md)

