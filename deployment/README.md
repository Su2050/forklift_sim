# 叉车插盘举升策略 — 部署指南

## 目录结构

```
deployment/
├── README.md                           ← 本文件（部署说明）
├── model_io_and_deployment_analysis.md ← 模型 I/O 规格与部署路径分析
├── infer.py                            ← 推理脚本（零依赖，仅需 torch + numpy）
├── requirements.txt                    ← Python 依赖
└── model_1999.pt                       ← 模型权重（需自行复制，见下方说明）
```

## 快速开始

### 1. 准备模型文件

```bash
cp IsaacLab/logs/rsl_rl/forklift_pallet_insert_lift/2026-02-27_17-43-22/model_1999.pt deployment/
```

### 2. 安装依赖

```bash
pip install -r deployment/requirements.txt
```

### 3. 运行推理测试

```bash
cd deployment
python infer.py --model model_1999.pt --test
```

预期输出：

```
[ForkliftPolicy] loaded from model_1999.pt
  obs normalizer count: 131072000
  action noise std: [0.01326 0.02426 0.03051]

============================================================
  推理管线测试
============================================================

输入观测 (15 dim):
  [ 0] d_x_r        = +3.5000
  [ 1] d_y_r        = +0.2000
  ...

输出动作 (3 dim):
  [0] drive  = +0.xxxx  →  +x.xx rad/s
  [1] steer  = +0.xxxx  →  +x.xx rad
  [2] lift   = -0.xxxx  →  -x.xxxx m/s

推理延迟: ~0.05 ms / step
等效控制频率: ~20000 Hz

============================================================
  测试通过 ✓
============================================================
```

### 4. 导出 ONNX（可选，用于嵌入式设备部署）

```bash
python infer.py --model model_1999.pt --export-onnx policy.onnx
```

导出后可在 TensorRT / ONNX Runtime / OpenVINO 等推理引擎上运行。

---

## 集成到实车控制系统

### 方式一：Python 直接集成

```python
from infer import ForkliftPolicy, build_obs, map_action_to_physical
import numpy as np

# 初始化（只做一次）
policy = ForkliftPolicy("model_1999.pt", device="cpu")  # 或 "cuda:0"

# --- 控制循环 (30 Hz) ---
prev_actions = np.zeros(3)

while running:
    # 1. 从传感器获取数据
    pallet_pos_robot = stereo_camera.get_pallet_position()  # (2,) [x, y] in robot frame
    pallet_yaw_robot = stereo_camera.get_pallet_yaw()       # float, rad
    robot_vel = imu.get_velocity_robot_frame()               # (2,) [vx, vy]
    yaw_rate = imu.get_yaw_rate()                            # float, rad/s
    lift_pos = encoder.get_lift_position()                   # float, m
    lift_vel = encoder.get_lift_velocity()                   # float, m/s
    insert_depth = tof_sensor.get_insert_depth()             # float, m
    y_signed = stereo_camera.get_lateral_offset()            # float, m (带符号)
    dyaw = stereo_camera.get_yaw_offset()                    # float, rad (带符号)

    # 2. 构建观测向量
    obs = build_obs(
        pallet_pos_robot=pallet_pos_robot,
        pallet_yaw_robot=pallet_yaw_robot,
        robot_vel_xy_robot=robot_vel,
        yaw_rate=yaw_rate,
        lift_pos=lift_pos,
        lift_vel=lift_vel,
        insert_depth_m=insert_depth,
        prev_actions=prev_actions,
        y_signed_m=y_signed,
        dyaw_rad=dyaw,
    )

    # 3. 推理
    action = policy.infer(obs)  # (3,) in [-1, 1]
    prev_actions = action.copy()

    # 4. 映射到物理指令并发送
    cmd = map_action_to_physical(action)
    can_bus.send_drive(cmd["drive_rad_s"])    # 车轮角速度
    can_bus.send_steer(cmd["steer_rad"])      # 转向角
    can_bus.send_lift(cmd["lift_m_s"])        # 升降速度

    sleep_until_next_cycle()  # 30 Hz
```

### 方式二：ONNX Runtime 集成（C++ / 嵌入式）

```cpp
#include <onnxruntime_cxx_api.h>

// 加载模型
Ort::Session session(env, "policy.onnx", session_options);

// 推理
float obs_raw[15] = { /* 填入 15 维观测 */ };
float action[3];
// ... 标准 ONNX Runtime 推理调用 ...

// 注意：ONNX 模型已内置归一化层，直接输入原始观测即可
```

---

## 观测向量说明（15 维）

| 索引 | 名称 | 含义 | 数据来源 |
|------|------|------|---------|
| 0-1 | `d_xy_r` | 托盘相对叉车位置 (robot frame, m) | 双目/深度相机 |
| 2 | `cos_dyaw` | 偏航角差余弦 | 双目/深度相机 |
| 3 | `sin_dyaw` | 偏航角差正弦 | 双目/深度相机 |
| 4-5 | `v_xy_r` | 叉车平面速度 (robot frame, m/s) | IMU + 轮速计 |
| 6 | `yaw_rate` | 偏航角速度 (rad/s) | IMU 陀螺仪 |
| 7 | `lift_pos` | 升降关节位置 (m) | 升降编码器 |
| 8 | `lift_vel` | 升降关节速度 (m/s) | 升降编码器差分 |
| 9 | `insert_norm` | 归一化插入深度 [0, 1] | ToF 传感器 |
| 10-12 | `prev_actions` | 上一步动作 [-1, 1] | 控制器缓存 |
| 13 | `y_err_obs` | 横向误差 (归一化, [-1, 1]) | 双目/深度相机 |
| 14 | `yaw_err_obs` | 偏航误差 (归一化, [-1, 1]) | 双目/深度相机 |

## 动作向量说明（3 维）

| 索引 | 名称 | 网络输出 | 物理指令 |
|------|------|---------|---------|
| 0 | drive | [-1, 1] | ×20 → 车轮角速度 (rad/s) |
| 1 | steer | [-1, 1] | ×0.6 → 转向角 (rad) |
| 2 | lift | [-1, 1] | ×0.5 → 升降速度 (m/s) |

---

## 性能参数

| 项目 | 值 |
|------|----|
| 模型大小 | 2.4 MB |
| 推理延迟 (CPU) | < 0.1 ms / step |
| 推理延迟 (ONNX) | < 0.05 ms / step |
| 要求控制频率 | 30 Hz（与训练一致） |
| Episode 成功率 | ~89% (EMA) |
| 横向对齐精度 | ~9.7 cm |
| 偏航对齐精度 | ~3.1° |

---

## 注意事项

1. **观测归一化**：模型内置 running mean/std 归一化。`infer.py` 中已自动处理，输入原始物理量即可。如使用 ONNX 导出版本，归一化层已内嵌，同样直接输入原始观测。

2. **控制频率**：训练时为 30 Hz。实车部署建议保持 30 Hz，偏差 ±20% 以内（24–36 Hz）可接受。

3. **坐标系**：
   - `d_xy_r` 和 `v_xy_r` 在 **robot frame**（x=前进方向, y=左侧）
   - `y_err_obs` 和 `yaw_err_obs` 基于 **pallet center-line frame**
   - 相机外参标定后需做坐标变换

4. **insert_norm 计算**：`insert_norm = clip(insert_depth_m / 2.16, 0, 1)`。其中 2.16 是托盘深度（1.2m × 1.8 缩放）。如实车托盘尺寸不同，需调整此参数。

5. **prev_actions**：首步设为 `[0, 0, 0]`，之后每步使用上一步的输出。控制器重启时需重置。

6. **安全制动**：模型训练时内置了插入足够深度后自动停止驱动的逻辑。实车部署建议额外添加硬件级安全限位。
