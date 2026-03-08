# 叉车仿真相机加装与视频流拉取——代码实现指南

**日期：** 2026-03-08  
**关联文件：**

| 文件 | 路径 | 作用 |
|------|------|------|
| 环境配置 | `IsaacLab/source/isaaclab_tasks/.../forklift_pallet_insert_lift/env_cfg.py` | 声明相机参数和 `TiledCameraCfg` |
| 环境实现 | `IsaacLab/source/isaaclab_tasks/.../forklift_pallet_insert_lift/env.py` | 相机初始化、图像采集、obs 组装 |
| 评估脚本 | `scripts/tools/camera_eval.py` | 独立测试工具：驱动叉车 + 录视频 + 导出关键帧 |

---

## 一、整体架构

```
env_cfg.py                     env.py                          camera_eval.py
┌────────────────┐   ┌───────────────────────────┐   ┌──────────────────────┐
│ use_camera      │──▶│ __init__()                │   │ 创建 env, 设参数     │
│ camera_width    │   │   self._camera_enabled    │   │ gym.make(cfg=cfg)    │
│ camera_height   │   │                           │   │                      │
│ camera_hfov_deg │   │ _setup_scene()            │   │ for step in steps:   │
│ camera_mount_body│  │   动态覆盖 tiled_camera   │   │   env.step(action)   │
│ camera_pos_local│   │   RPY→quaternion          │   │   obs["policy"]["image"]
│ camera_rpy_deg  │   │   TiledCamera(cfg)        │   │   保存 frame / video │
│ tiled_camera    │   │                           │   │                      │
│  (TiledCameraCfg)│  │ _get_camera_image()       │   │ 导出 mp4 + png      │
└────────────────┘   │   camera.data.output["rgb"]│   └──────────────────────┘
                     │   → (N,3,H,W) float [0,1]  │
                     │                             │
                     │ _get_observations()         │
                     │   camera_enabled → dict obs │
                     │   {"image":..,"proprio":..} │
                     └───────────────────────────┘
```

---

## 二、env_cfg.py——相机参数声明

### 2.1 新增字段

```python
# IsaacLab/source/isaaclab_tasks/.../env_cfg.py  第 57-95 行

# ===== Step-1 (video-e2e): camera + asymmetric critic scaffolding =====
# 默认关闭，确保与当前 low-dim 基线行为完全一致
use_camera: bool = False
use_asymmetric_critic: bool = False

# 相机参数（第一版推荐）
camera_width: int = 64
camera_height: int = 64
camera_hfov_deg: float = 75.0
camera_mount_body: str = "body"
# 相机相对挂载 body 的位姿 — 注意：forklift_c.usd 使用 cm 单位，
# TiledCamera offset 在 prim 局部坐标系中应用，因此这里的值必须是 cm。
# 物理含义：前方 0.8m、上方 1.7m → 80cm、170cm
camera_pos_local: tuple[float, float, float] = (80.0, 0.0, 170.0)
camera_rpy_local_deg: tuple[float, float, float] = (0.0, -20.0, 0.0)

# easy8 + privileged 维度（供 obs 组装使用）
easy8_dim: int = 8
privileged_dim: int = 22
```

**设计要点：**

- `use_camera = False`：默认关闭，不影响已有的 low-dim 训练
- 位置单位是 **cm**（与 `forklift_c.usd` 的 `metersPerUnit=0.01` 一致）
- `camera_mount_body = "body"`：挂在车体上，不随 lift 运动（门架最上方固定位置）

### 2.2 TiledCameraCfg 实例

```python
# env_cfg.py 第 79-95 行

tiled_camera: TiledCameraCfg = TiledCameraCfg(
    prim_path=f"/World/envs/env_.*/Robot/{camera_mount_body}/Camera",
    offset=TiledCameraCfg.OffsetCfg(
        pos=camera_pos_local,
        rot=(0.9848078, 0.0, -0.1736482, 0.0),  # pitch=-20° 默认值，运行时会被覆盖
        convention="world",  # 关键：必须用 world（+X前、+Z上），不要用 ros
    ),
    data_types=["rgb"],
    spawn=sim_utils.PinholeCameraCfg(
        focal_length=24.0,
        focus_distance=400.0,
        horizontal_aperture=20.955,
        clipping_range=(0.1, 40.0),
    ),
    width=camera_width,
    height=camera_height,
)
```

**关键注意：**

- `convention="world"`——body prim 的局部坐标系是 +X 前方、+Z 上方，必须用 `world` 约定。如果误用 `ros`（+Z 前、-Y 上），画面会旋转 90°。
- `rot` 的值在类定义时写死，运行时会在 `env.py` 的 `_setup_scene()` 中被动态覆盖。
- `width`/`height` 同样在类定义时绑定了 `camera_width`/`camera_height` 的默认值，运行时也需要显式覆盖。

### 2.3 @configclass 的坑

`@configclass` 的行为类似 `dataclass`，**嵌套对象在类定义时就固化了**。例如：

```python
camera_pos_local = (80.0, 0.0, 170.0)
tiled_camera = TiledCameraCfg(
    offset=TiledCameraCfg.OffsetCfg(pos=camera_pos_local, ...)  # 此时值已固定
)
```

运行时即使修改 `cfg.camera_pos_local = (130.0, 0.0, 250.0)`，`cfg.tiled_camera.offset.pos` 仍然是 `(80.0, 0.0, 170.0)`。必须在 `_setup_scene()` 中**显式赋值到子字段**。

---

## 三、env.py——相机初始化与图像采集

### 3.1 `__init__()` 中的相机状态初始化

```python
# env.py 第 130-137 行

class ForkliftPalletInsertLiftEnv(DirectRLEnv):
    def __init__(self, cfg, render_mode=None, **kwargs):
        # _setup_scene() 会在 super().__init__ 内被调用，
        # 因此相机开关/状态字段必须在 super() 之前初始化。
        self._camera_enabled = bool(getattr(cfg, "use_camera", False))
        self._asym_enabled = bool(getattr(cfg, "use_asymmetric_critic", False))
        self._camera_initialized = False
        self._camera = None
        self._warned_camera_fallback = False

        super().__init__(cfg, render_mode, **kwargs)
```

**为什么在 super() 之前？** 因为 `DirectRLEnv.__init__()` 内部会调用 `_setup_scene()`，而 `_setup_scene()` 需要读取 `self._camera_enabled` 来决定是否创建相机。如果顺序反了，会报 `AttributeError`。

### 3.2 `_setup_scene()` 中的相机创建

```python
# env.py 第 622-664 行

if self._camera_enabled:
    mount_body = str(getattr(self.cfg, "camera_mount_body", "base_link"))
    mount_prim = f"/World/envs/env_0/Robot/{mount_body}"
    stage = self.sim.stage
    if not stage.GetPrimAtPath(mount_prim).IsValid():
        # 严格模式：body 不存在直接报错，列出可用 children 辅助排查
        robot_prim = stage.GetPrimAtPath('/World/envs/env_0/Robot')
        candidates = [child.GetName() for child in robot_prim.GetChildren()] if robot_prim.IsValid() else []
        raise RuntimeError(f"[camera] mount body prim not found: {mount_prim}. available={candidates}")

    # ① 动态覆盖 tiled_camera 的所有子字段
    self.cfg.tiled_camera.prim_path = f"/World/envs/env_.*/Robot/{mount_body}/Camera"
    self.cfg.tiled_camera.offset.pos = self.cfg.camera_pos_local
    self.cfg.tiled_camera.width = self.cfg.camera_width
    self.cfg.tiled_camera.height = self.cfg.camera_height

    # ② 根据 hfov 和 horizontal_aperture 反算 focal_length
    import math
    hfov_rad = math.radians(self.cfg.camera_hfov_deg)
    horizontal_aperture = self.cfg.tiled_camera.spawn.horizontal_aperture
    focal_length = horizontal_aperture / (2.0 * math.tan(hfov_rad / 2.0))
    self.cfg.tiled_camera.spawn.focal_length = focal_length

    # ③ 将 Euler 角（RPY）转为四元数
    roll_deg, pitch_deg, yaw_deg = self.cfg.camera_rpy_local_deg
    cr = math.cos(math.radians(roll_deg) * 0.5)
    sr = math.sin(math.radians(roll_deg) * 0.5)
    cp = math.cos(math.radians(pitch_deg) * 0.5)
    sp = math.sin(math.radians(pitch_deg) * 0.5)
    cy = math.cos(math.radians(yaw_deg) * 0.5)
    sy = math.sin(math.radians(yaw_deg) * 0.5)
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    self.cfg.tiled_camera.offset.rot = (w, x, y, z)

    # ④ 创建 TiledCamera 实例
    self._camera = TiledCamera(self.cfg.tiled_camera)
    self._camera_initialized = True
```

**三步覆盖流程：**

1. **pos / width / height / prim_path**：直接赋值
2. **focal_length**：根据 `hfov_deg` 和 `horizontal_aperture` 反算（`f = aperture / (2 * tan(hfov/2))`）
3. **rot**：将用户友好的 Euler 角 (roll, pitch, yaw) 转为四元数 (w, x, y, z)

最后把 `self._camera` 注册到 `self.scene.sensors`，让 Isaac Lab 框架自动处理环境克隆和数据同步：

```python
# env.py 第 688-689 行
if self._camera_enabled and self._camera is not None:
    self.scene.sensors["tiled_camera"] = self._camera
```

### 3.3 `_get_camera_image()` 图像采集

```python
# env.py 第 820-852 行

def _get_camera_image(self) -> torch.Tensor:
    """返回真实相机图像张量，统一为 (N,3,H,W), float32, [0,1]。"""
    h = int(getattr(self.cfg, "camera_height", 64))
    w = int(getattr(self.cfg, "camera_width", 64))

    if not self._camera_initialized or self._camera is None:
        raise RuntimeError("[camera] camera requested but not initialized")

    rgb = self._camera.data.output["rgb"]

    # 支持两种常见布局: (N,H,W,3) 或 (N,3,H,W)
    if rgb.shape[-1] == 3:
        rgb = rgb.permute(0, 3, 1, 2)   # (N,H,W,3) → (N,3,H,W)
    elif rgb.shape[1] == 3:
        pass                              # 已经是 (N,3,H,W)

    rgb = rgb.float()
    if rgb.max() > 1.0:
        rgb = rgb / 255.0                # uint8 → [0,1]
    rgb = torch.clamp(rgb, 0.0, 1.0)

    # 安全检查
    if torch.isnan(rgb).any() or torch.isinf(rgb).any():
        raise RuntimeError("[camera] rgb contains NaN/Inf")
    if rgb.shape[2] != h or rgb.shape[3] != w:
        raise RuntimeError(f"[camera] rgb shape mismatch: {tuple(rgb.shape)} vs expect (*,3,{h},{w})")

    return rgb
```

**关键点：**

- `TiledCamera.data.output["rgb"]` 可能返回 `(N,H,W,3)` 或 `(N,3,H,W)`，这里做了自动适配
- 自动处理 `uint8 [0,255]` → `float32 [0,1]` 的转换
- 有 NaN/Inf 和 shape 保护，出问题时快速定位

### 3.4 `_get_observations()` 观测分发

```python
# env.py 第 993-1006 行

if self._camera_enabled:
    obs_dict = {
        "policy": {
            "image": self._get_camera_image(),   # (N, 3, H, W)
            "proprio": self._get_easy8(),         # (N, 8)
        }
    }
else:
    obs_dict = {"policy": obs}                    # (N, 15) 向量观测

if self._asym_enabled:
    obs_dict["critic"] = self._get_privileged_obs(obs)
```

**两种模式：**

| `use_camera` | `obs["policy"]` 结构 | 用途 |
|---|---|---|
| `False` | `Tensor(N, 15)` | 原始 low-dim 训练 |
| `True` | `{"image": Tensor(N,3,H,W), "proprio": Tensor(N,8)}` | video-e2e 训练 |

`easy8` 包含：`[v_x_r, v_y_r, yaw_rate, lift_pos, lift_vel, prev_drive, prev_steer, prev_lift]`

---

## 四、camera_eval.py——独立测试与视频录制脚本

### 4.1 文件位置

```
scripts/tools/camera_eval.py
```

### 4.2 命令行参数

```python
parser.add_argument("--cam-name", type=str, required=True)     # 输出目录名
parser.add_argument("--cam-x", type=float, required=True)      # X 偏移 (cm)
parser.add_argument("--cam-y", type=float, required=True)      # Y 偏移 (cm)
parser.add_argument("--cam-z", type=float, required=True)      # Z 偏移 (cm)
parser.add_argument("--pitch-deg", type=float, required=True)  # Pitch 角度
parser.add_argument("--yaw-deg", type=float, default=0.0)      # Yaw 角度
parser.add_argument("--roll-deg", type=float, default=0.0)     # Roll 角度
parser.add_argument("--mount-body", type=str, default="body")  # 挂载体
parser.add_argument("--hfov-deg", type=float, default=75.0)    # 水平视场角
parser.add_argument("--resolution", type=int, default=320)     # 分辨率
parser.add_argument("--steps", type=int, default=200)          # 仿真步数
```

### 4.3 核心流程

```python
def main():
    out_dir = Path(f".../outputs/camera_eval/{args.cam_name}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # ① 创建并配置环境
    cfg = ForkliftPalletInsertLiftEnvCfg()
    cfg.scene.num_envs = 1
    cfg.use_camera = True
    cfg.camera_width = args.resolution
    cfg.camera_height = args.resolution
    cfg.camera_hfov_deg = args.hfov_deg
    cfg.camera_mount_body = args.mount_body
    cfg.camera_pos_local = (args.cam_x, args.cam_y, args.cam_z)
    cfg.camera_rpy_local_deg = (args.roll_deg, args.pitch_deg, args.yaw_deg)
    cfg.robot_cfg.init_state.pos = (-2.0, 0.0, 0.03)  # 拉近到托盘 2m

    env = gym.make("Isaac-Forklift-PalletInsertLift-Direct-v0", cfg=cfg)
    obs, _ = env.reset()

    # ② 生成可视化标记方块（红色=货叉尖端，蓝色=托盘）
    stage = env.unwrapped.sim.stage
    spawn_marker_cube(stage, ".../TipMarker",    (-0.2, 0, 0.1), 0.2, (1,0,0), ...)
    spawn_marker_cube(stage, ".../PalletMarker", (0, 0, 0.3),    0.3, (0,0.2,1), ...)

    # ③ 驱动叉车并逐帧采集图像
    frames = []
    for step in range(args.steps):
        action = torch.zeros((1, 3), device=env.unwrapped.device)
        if step <= 60:
            action[:, 0] = 1.0     # 前进
        else:
            action[:, 2] = 1.0     # 举升

        obs, _, _, _, _ = env.step(action)

        # 从 obs 中提取图像
        img_tensor = obs["policy"]["image"][0]         # [3, H, W]
        img_np = img_tensor.cpu().numpy()
        img_np = np.transpose(img_np, (1, 2, 0))      # [H, W, 3]
        if img_np.dtype != np.uint8:
            img_np = (img_np * 255).clip(0, 255).astype(np.uint8)
        frames.append(img_np)

        # 保存关键帧
        if step == 0:
            Image.fromarray(img_np).save(out_dir / "frame_start.png")
        elif step == 50:
            Image.fromarray(img_np).save(out_dir / "frame_mid.png")
        elif step == args.steps - 1:
            Image.fromarray(img_np).save(out_dir / "frame_end.png")

    env.close()

    # ④ 用 OpenCV 合成 mp4 视频
    video_path = out_dir / "video.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(video_path), fourcc, 30.0, (w, h))
    for frame in frames:
        out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    out.release()

    # ⑤ 计算背景稳定性指标
    diffs = []
    for i in range(1, min(10, len(frames))):
        diff = np.mean(np.abs(frames[i].astype(float) - frames[0].astype(float)))
        diffs.append(diff)
    avg_diff = np.mean(diffs)

    # ⑥ 保存指标文件
    with open(out_dir / "metrics.txt", "w") as f:
        f.write(f"Background stability (avg diff 0-10): {avg_diff:.2f}\n")
        f.write(f"Camera pos: {args.cam_x}, {args.cam_y}, {args.cam_z}\n")
        f.write(f"Camera rot: {args.roll_deg}, {args.pitch_deg}, {args.yaw_deg}\n")
        f.write(f"Camera hfov: {args.hfov_deg}\n")
        f.write(f"Camera mount: {args.mount_body}\n")
```

### 4.4 标记方块生成

用 `UsdGeom.Cube` 和 `UsdShade.Material` 在场景中动态创建带颜色的标记方块，用于人眼快速判断相机视角是否正确：

```python
from pxr import UsdGeom, UsdShade, Sdf, Gf

def ensure_preview_material(stage, mat_path, color):
    material = UsdShade.Material.Define(stage, mat_path)
    shader = UsdShade.Shader.Define(stage, f"{mat_path}/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.2)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return material

def spawn_marker_cube(stage, prim_path, pos_xyz, size, color, mat_path):
    cube = UsdGeom.Cube.Define(stage, prim_path)
    cube.GetSizeAttr().Set(size)
    cube.ClearXformOpOrder()
    cube.AddTranslateOp().Set(Gf.Vec3d(*pos_xyz))
    cube.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    material = ensure_preview_material(stage, mat_path, color)
    UsdShade.MaterialBindingAPI(cube.GetPrim()).Bind(material)
```

- 设置了 `emissiveColor`（自发光），保证标记在任何光照条件下都可见
- `DisplayColor` 作为 fallback，给不支持 PBR 的渲染模式用

---

## 五、完整测试流程

### 5.1 前置条件

```bash
# 必须退出 conda，否则 isaaclab.sh 会用错误的 Python
unset CONDA_PREFIX && unset CONDA_DEFAULT_ENV

cd /home/uniubi/projects/forklift_sim/IsaacLab
```

### 5.2 运行评估脚本

```bash
./isaaclab.sh -p /home/uniubi/projects/forklift_sim/scripts/tools/camera_eval.py \
  --headless --enable_cameras \
  --cam-name world_conv_test \
  --cam-x 130 --cam-y 0 --cam-z 250 \
  --pitch-deg 45 --hfov-deg 90 \
  --resolution 320 --steps 150
```

**必须带 `--enable_cameras`**，否则会报 `RuntimeError: A camera was spawned without the --enable_cameras flag`。

### 5.3 检查输出

```bash
ls outputs/camera_eval/world_conv_test/
# frame_start.png  frame_mid.png  frame_end.png  video.mp4  metrics.txt
```

### 5.4 验证清单

| 检查项 | 判断方法 | 预期 |
|--------|----------|------|
| 画面方向正确 | 打开 frame_start.png，地面在底部 | 地面在画面底部，天空/远处在顶部 |
| 能看到货叉或标记 | 画面中应有红色/蓝色色块或叉车结构 | 有明显的场景元素 |
| 叉车在运动 | 对比 frame_start 和 frame_end | 两帧内容有显著差异 |
| 视频流畅 | 播放 video.mp4 | 30fps 流畅，无卡顿 |
| 指标文件完整 | 查看 metrics.txt | 包含背景稳定性指标和相机参数 |

### 5.5 批量测试多组参数

由于 Isaac Sim 不支持同一进程内反复创建环境（会卡死），必须**串行执行**：

```bash
#!/bin/bash
cd /home/uniubi/projects/forklift_sim/IsaacLab
unset CONDA_PREFIX && unset CONDA_DEFAULT_ENV

for name in test_a test_b test_c; do
  echo "=== Running $name ==="
  ./isaaclab.sh -p .../camera_eval.py \
    --headless --enable_cameras \
    --cam-name $name \
    --cam-x 130 --cam-y 0 --cam-z 250 \
    --pitch-deg 45 --hfov-deg 90 \
    --steps 150
  echo "=== $name done ==="
done
```

不要用 `&` 并行启动——GPU 只能服务一个 Isaac Sim 实例。

---

## 六、在训练中启用相机观测

当前 `use_camera` 默认 `False`。要在训练中启用：

```python
# 在训练脚本或 hydra 配置中
cfg.use_camera = True
cfg.use_asymmetric_critic = True   # 可选：使用非对称 critic
cfg.camera_width = 64              # 训练时可用低分辨率
cfg.camera_height = 64
```

启用后 `obs["policy"]` 变为嵌套 dict：

```python
obs["policy"]["image"]    # Tensor (N, 3, 64, 64) float32 [0,1]
obs["policy"]["proprio"]  # Tensor (N, 8) float32
```

Policy 网络需要相应调整：CNN 编码器处理 `image`，MLP 处理 `proprio`，最后 concat。

---

## 七、常见错误与排查

| 错误 | 原因 | 解决 |
|------|------|------|
| `ModuleNotFoundError: No module named 'isaaclab'` | conda base 环境没有 Isaac Lab | `unset CONDA_PREFIX && unset CONDA_DEFAULT_ENV` |
| `RuntimeError: A camera was spawned without the --enable_cameras flag` | 缺少启动参数 | 命令行加 `--enable_cameras` |
| `RuntimeError: [camera] mount body prim not found` | `camera_mount_body` 名字拼错 | 查看报错里的 available children 列表 |
| `IndexError: index 1 is out of bounds for dimension 1 with size 1` | action 维度不对 | action 必须是 `(N, 3)` 不是 `(N, 1)` |
| `TypeError: Cannot handle this data type: (1, 1, 320), |u1` | PIL 接收到了错误的 array shape | `np.transpose(img_np, (1, 2, 0))` 把 `[3,H,W]` 转成 `[H,W,3]` |
| 画面全黑（mean<0.05） | 相机在模型内部（cm/m 单位混淆） | `camera_pos_local` 值必须是 cm |
| 画面旋转 90° | `convention="ros"` 与 body 坐标系不匹配 | 改为 `convention="world"` |
| 画面全白（mean>0.9） | 相机穿过门架钢结构，在模型内部看天空 | 降低 Z 偏移或增大前向偏移 |
| 进程卡死无输出 | 同一进程内反复创建/销毁环境 | 每个测试用独立进程，串行执行 |
