# 诊断报告：s1.0zd 视觉基线训练 (20260309_233639) 启动前排障全记录

**日期**: 2026-03-09  
**分支**: `exp/vision_cnn/scratch_rl_baseline`  
**最终成功日志**: `20260309_233639_train_s1.0zd.log`  
**耗时**: 从首次尝试到成功启动约 **4 小时**（19:46 → 23:36）

---

## 概述

在正式启动 3 层 CNN baseline 的 scratch-RL 训练之前，经历了多轮排障。问题涉及 4 个独立故障：

1. **Python 环境/模块导入失败** — 脚本未在 IsaacLab 环境下运行
2. **观测数据格式不匹配** — collect-data 脚本取值方式与新 observation dict 不兼容
3. **仿真启动永久挂起** — PNG 纹理解析 + 僵尸进程锁缓存 + `wait_for_textures` 三因素叠加
4. **GPU OOM (进程被 Killed)** — 相机渲染占用过多显存

---

## 时间线

### Phase 1：MobileNet 短迭代验证（19:46 – 21:44）✅ 正常

在 `exp/vision_cnn/cnn_pretrain` 分支上用 MobileNetV3-Small 做短迭代测试，一切正常：

| 日志 | 类型 | 迭代 | 结果 |
|---|---|---|---|
| `194633_sanity_check_s1.0zd` | sanity_check | 51/100 | 正常，ema=24.4% |
| `205832_sanity_check_s1.0zd` | sanity_check | 9/10 | 正常 |
| `210356_smoke_train_s1.0zd` | smoke_train | 24/30 | 正常，~15.5s/iter |
| `211138_smoke_train_s1.0zd` | smoke_train | 29/30 | 正常，~15.7s/iter |

此阶段**相机渲染正常**，纹理缓存完好，Omniverse 进程干净。

---

### Phase 2：数据采集脚本调试（21:44 – 22:11）

切换到 `scratch_rl_baseline` 分支后，开始跑 `scripts/collect_detection_data.py` 脚本采集检测训练数据。

#### 故障 2a：ModuleNotFoundError: No module named 'isaaclab'

**日志**: `20260309_214414_sanity_check_s1.0zd.log`  
**原因**: 直接用系统 Python 运行脚本，未通过 `isaaclab.sh -p` 启动。  
**修复**: 使用 `env CONDA_PREFIX="" CONDA_DEFAULT_ENV="" bash isaaclab.sh -p scripts/collect_detection_data.py ...`

#### 故障 2b：ModuleNotFoundError: No module named 'omni.client'

**日志**: `20260309_214806_sanity_check_s1.0zd.log`  
**原因**: IsaacLab 的 `isaaclab.sim.converters` 在 `AppLauncher` 之前被 import，而 Omniverse 模块需要 AppLauncher 初始化后才可用。  
**修复**: 调整 import 顺序，确保 `AppLauncher` 先于 `isaaclab.sim` 被 import。

#### 故障 2c：TypeError: tuple indices must be integers or slices, not str

**日志**: `20260309_214833_sanity_check_s1.0zd.log`  
**原因**: `collect_detection_data.py` 用 `obs["image"]` 取值，但视觉环境返回的 observation 是一个嵌套 dict（`{"policy": {"image": ..., "proprio": ...}, "critic": ...}`），不能直接用字符串索引。  
**修复**: 修改数据采集脚本，按新的 observation dict 结构正确提取 image tensor。

#### 数据采集首次成功

**日志**: `20260309_214946_sanity_check_s1.0zd.log`  
首次成功采集检测数据（64 envs），但已开始出现 PNG 纹理错误：
```
[Error] Couldn't process ...T_Forklift_C01_Normal.1002.png
Reason: Failed to load image: PNG not supported: unknown PNG chunk type
```
此时错误尚可快速跳过（<1s），未造成卡死。

---

### Phase 3：仿真启动挂死（22:04 – 23:02）

#### 故障 3a：OOM Killed

**日志**: `20260309_220224_sanity_check_s1.0zd.log`  
32 envs + 相机渲染，进程在 `Starting the simulation` 后被系统 OOM Killer 杀死。

#### 故障 3b：训练启动永久挂死（核心问题）

**日志**:
- `20260309_224121_train_s1.0zd.log` — 挂死 ~5min（PNG 错误间隔 ~50s）
- `20260309_225326_train_s1.0zd.log` — 挂死 ~4min（PNG 错误间隔 ~200s）
- `20260309_230201_train_s1.0zd.log` — 挂死

**现象**: 程序输出 `[INFO]: Starting the simulation...` 后永远不会进入 `Learning iteration`。日志中反复出现：
```
[Error] [gpu.foundation.plugin] Couldn't process
  https://omniverse-content-production.s3-us-west-2.amazonaws.com/.../T_Forklift_C01_Normal.1005.png
  Reason: Failed to load image: PNG not supported: unknown PNG chunk type
```
且错误间隔从 Phase 2 的 <1s 恶化为 **50 – 300 秒**。

**根因分析**（三因素叠加）:

1. **僵尸进程锁死缓存**：Phase 1 – 2 被中断的 `kit` / `train.py` 进程残留在后台，持有 Omniverse 本地缓存 (`omni.kvdb.plugin`) 的排他锁。新进程无法读取已有缓存。
2. **S3 下载超时雪崩**：缓存不可用后，渲染器为 256 个环境并发下载海量贴图（每个 ~12MB），网络延迟 + 解析失败导致单贴图超时周期 >50s，数百贴图排队。
3. **`wait_for_textures` 阻塞**：`DirectRLEnvCfg` 默认 `wait_for_textures=True`，要求所有贴图加载完毕才能 `reset()`。由于贴图一直在超时重试，环境被永久阻塞。

---

### Phase 4：修复（23:00 – 23:35）

#### 修复 A：禁用 wait_for_textures

**commit**: `571f7d46` (23:00:52)  
**文件**: `env_cfg.py`
```python
wait_for_textures: bool = False
```
使 environment 的 `reset()` 不再等待贴图就绪，即使纹理加载失败也能继续仿真。对 RL 训练无影响——headless 模式下只需要相机 RGB 数据，叉车表面纹理的完整性不影响相机画面内容。

#### 修复 B：僵尸进程清理脚本

**commit**: `40ce9a78` (23:34:30)  
**文件**: `scripts/experiments/cleanup_zombies.sh`
```bash
pgrep -af "train.py" | awk '{print $1}' | xargs -r kill -9
pgrep -af "kit" | awk '{print $1}' | xargs -r kill -9
```
确保没有残留进程锁住缓存。

#### 修复 C：缓存重置

手动清理了 Omniverse 本地缓存：
```bash
rm -rf ~/.cache/ov ~/.local/share/ov ~/.nvidia-omniverse/
```

---

### Phase 5：验证测试（23:11 – 23:35）

修复后的中间验证日志：

| 日志 | 目的 | 结果 |
|---|---|---|
| `test_32_envs.log` | 32 envs 测试 | 仍挂（修复 A 之前） |
| `test_32_envs_320.log` | 再次测试 | 有 PNG 错误但未挂 |
| `test_unbuffered*.log` | 多种配置测试 | 部分成功，部分挂死 |
| `test_env_cfg.log` | 完整配置测试 | OOM Killed（num_envs 过大） |

---

### Phase 6：正式训练成功启动（23:36）✅

**日志**: `20260309_233639_train_s1.0zd.log`

训练配置:
- Backbone: 3 层 Nature CNN（非 MobileNet）
- num_envs: 256
- max_iterations: 2000
- `wait_for_textures: False`

启动特征:
```
wait_for_textures: False
[INFO]: Time taken for simulation start : 11.231268 seconds  ← 正常（之前挂死时 >300s）
[INFO]: Completed setting up the environment...
```

训练结果:
- 完整跑完 2000 iterations
- `episode/success_rate_total`: **35.02%**
- `diag/near_success_frac`: 100%
- `diag/deep_insert_frac`: 61.72%
- 总耗时: ~2h 41min

---

## 经验总结

| # | 教训 | 防范措施 |
|---|---|---|
| 1 | 被中断的 Isaac Sim 进程会锁死 Omniverse 缓存，导致后续启动雪崩 | 每次训练前运行 `cleanup_zombies.sh` |
| 2 | `wait_for_textures=True` 在纹理加载异常时会永久阻塞 | 对 RL 训练默认设为 `False` |
| 3 | 相机渲染大幅增加 GPU 显存消耗 | 视觉训练 num_envs 不超过 256（24GB GPU） |
| 4 | `isaaclab.sh -p` 必须在干净 Python 环境下运行 | 前缀 `env CONDA_PREFIX="" CONDA_DEFAULT_ENV=""` |
| 5 | 视觉环境的 observation 结构与纯状态环境不同 | 取值时按 `obs["policy"]["image"]` 层级访问 |

---

## 关联文件

- 修复 commit: `571f7d46`, `40ce9a78`
- 已有细节报告: `docs/diagnostic_reports/texture_hang_fix_20260309.md`
- 成功训练日志: `logs/20260309_233639_train_s1.0zd.log`
- 清理脚本: `scripts/experiments/cleanup_zombies.sh`
