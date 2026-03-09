叉车插盘举升 Teacher-Student 视觉蒸馏 — 完整工程实施指南


---

0. 总览方案图

┌──────────────────────────────────────────────────────────────────────────────┐
│                          Phase 1: 加相机 + 验证视野                          │
│                                                                              │
│  forklift_c.usd                                                             │
│     ├── Robot root                                                           │
│     │    └── mast_camera (TiledCamera, 224×224 RGB)                          │
│     │         pos=(0.7, 0, 2.1)  rot=pitch_down_15°                         │
│     └── Pallet                                                               │
│                                                                              │
│  验证: 10 envs + GUI → 截图确认视野覆盖远场/近场/插入全流程                     │
└───────────────────────────────────────┬──────────────────────────────────────┘
                                        ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                    Phase 2: Teacher 跑仿真 + 采集数据集                      │
│                                                                              │
│  collect_data.py (64 envs + teacher policy + TiledCamera)                    │
│     每 step 记录:                                                            │
│     ┌─────────────┬───────────────────────┬──────────────────────────────┐   │
│     │ RGB 224×224  │ easy_states (8 dim)   │ label: [d_x,d_y,dyaw,ins_d] │   │
│     │             │ v_xy, yr, lp, lv, pa  │ + teacher_action (3 dim)     │   │
│     └─────────────┴───────────────────────┴──────────────────────────────┘   │
│     目标: ~200K 帧 (v0.1), ~500K 帧 (v1.0)                                  │
└───────────────────────────────────────┬──────────────────────────────────────┘
                                        ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                    Phase 3: 离线训练 Student                                 │
│                                                                              │
│  MobileNetV3-Small                                                           │
│     RGB (224×224)  ──→  [backbone 576-d]  ──┐                                │
│                                              ├──→ MLP(576+8, 256, 4)         │
│     easy_states (8)  ────────────────────────┘     ↓                         │
│                                              [d_x, d_y, dyaw, ins_d]         │
│                                                    ↓                         │
│                                        geometric_to_obs() 变换               │
│                                                    ↓                         │
│                                  7 missing obs dims → 拼回 15 dim            │
│                                                    ↓                         │
│                                     (可选) action distillation loss          │
└───────────────────────────────────────┬──────────────────────────────────────┘
                                        ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                    Phase 4: 闭环评估                                         │
│                                                                              │
│  Level 1: 离线回放 → 标签误差分布分析                                         │
│  Level 2: 半闭环 → 替换部分 obs, 其余保持真值                                 │
│  Level 3: 全闭环 → student(image) + easy_states → 15d obs → teacher actor     │
│                                                                              │
│  目标: L3 闭环成功率 ≥ 80% (teacher ~89%)                                    │
│  后续: DAgger / DR / 微调 / 边缘部署                                         │
└──────────────────────────────────────────────────────────────────────────────┘


---

第 1 部分：在 Isaac Lab 上加相机传感器

1.1 相机数量与安装位置

第一版建议：单相机。理由：

1. 单相机是实车部署成本最低的方案，也是 DV500 级芯片能处理的上限
2. 多相机融合是高风险变量，应在单相机 baseline 验证后再考虑
3. 单相机需仔细选位置，但任务几何（前向接近+插入）天然适合前视相机
  
安装位置：门架（mast）顶部偏前方，向下俯视约 12–18°

具体参数（基于 forklift_c.usd 几何测量）：

parent prim:   Robot root (base_link)
offset pos:    (0.7, 0.0, 2.1)  — 相对于 robot root origin
               x=0.7m: 略前于驾驶室（门架前缘附近）
               y=0.0m: 中心线上
               z=2.1m: 门架顶部（overhead guard 高度）
offset rot:    pitch down 15° → quat (w,x,y,z) ≈ (0.9914, 0.0, 0.1305, 0.0)
               （绕 Y 轴正方向旋转 -15°）

为什么这个位置好：

距离阶段
画面内容
可估计的量
远场 (3-4m)
托盘完整轮廓，地面参考线
d_x, d_y, dyaw
中场 (1-2m)
托盘开口清晰，叉齿前端部分可见
d_x, d_y, dyaw, 初步 insert
近场 (<1m)
托盘开口填满画面下半部，叉齿进入
dyaw, y_err, insert_depth
插入中
托盘上表面，叉齿隐没部分
insert_depth（通过可见长度变化）
举升
托盘升起，地面距离变化
lift confirmation

注意：相机固定在 robot root（不随 lift joint 移动），这保证了近场视角稳定。如果挂在 fork carriage 上，举升时视角会剧烈变化，增加 student 学习难度。

1.2 相机参数设定

参数
第一版推荐值
理由
分辨率
224 × 224
MobileNet 标准输入；DV500 部署友好；GPU 显存可控
FOV (水平)
80°
覆盖 ±40°，近场能看到叉齿两侧 + 托盘宽度
focal_length / aperture
24mm / 20.955mm
对应 ~80° HFOV（Isaac Sim PinholeCameraCfg 标准）
clipping range
(0.05, 15.0) m
近裁面 5cm 防穿模，远裁面 15m 覆盖远场
更新周期
与控制频率同步: 1/30 s
即 update_period = self.cfg.sim.dt * self.cfg.decimation

数据类型选择：

数据类型
第一版 student 输入
数据采集时记录
说明
rgb
✅
✅
student 唯一视觉输入
distance_to_image_plane
❌
✅
深度图，用于 debug 和未来 RGBD student
semantic_segmentation
❌
✅ (可选)
分割 mask，验证相机能"看到"什么
instance_id_segmentation
❌
❌
第一版不需要
关键点 / bbox
❌
❌
第一版不做检测，直接回归

采集时记录深度和分割的理由：

- 深度图可以做"oracle baseline"——用深度图直接几何算 d_x, d_y, dyaw，不训练任何网络，先验证管线正确性
- 分割 mask 可以验证相机是否真的看到了托盘 / 叉齿
- 存储成本低：每帧多 ~100KB，不影响总数据量
  
1.3 Isaac Lab 项目改动点清单

⚠️ 以下基于 Isaac Lab release/2.3.0 的常见项目结构。如版本不同，具体 API 名可能有差异。

改动文件列表：

isaaclab_patch/source/isaaclab_tasks/isaaclab_tasks/direct/forklift_pallet_insert_lift/
├── env_cfg.py      ← 新增 camera_cfg 配置项（可选字段，默认 None）
├── env.py          ← _setup_scene() 中条件创建相机；新增 _get_camera_obs() 方法
├── env_data_collection.py  ← [新文件] 继承 env，覆写 _get_observations 附加图像
└── agents/
    └── rsl_rl_ppo_cfg.py   ← 不改（训练配置不变）

scripts/
├── collect_data.py         ← [新文件] 数据采集主脚本
└── verify_camera_view.py   ← [新文件] 快速验证相机视野

文件 1: env_cfg.py 改动

在 ForkliftPalletInsertLiftEnvCfg 末尾新增：

from isaaclab.sensors import TiledCameraCfg
import isaaclab.sim as sim_utils

# ===== 相机配置（数据采集用，训练时不启用）=====
# 设为 None 表示不创建相机（保持原有训练速度）
camera_cfg: TiledCameraCfg | None = None

# 供 collect_data.py 使用的预设配置
@staticmethod
def make_camera_cfg() -> TiledCameraCfg:
    """返回数据采集用的 TiledCamera 配置。"""
    return TiledCameraCfg(
        prim_path="/World/envs/env_.*/Robot/mast_camera",
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.7, 0.0, 2.1),
            rot=(0.9914, 0.0, 0.1305, 0.0),  # pitch down ~15°
            convention="world",  # Isaac Lab >=2.x: "world" or "ros"
        ),
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.05, 15.0),
        ),
        width=224,
        height=224,
        data_types=["rgb", "distance_to_image_plane"],
    )

关键设计决策：camera_cfg 默认为 None。只有数据采集脚本显式启用相机。这样原有训练流程零改动、零性能损失。

文件 2: env.py 改动

在 _setup_scene() 中，clone 环境之后、添加灯光之前，加入条件创建相机的逻辑：

def _setup_scene(self):
    # ... （原有 robot + pallet + ground + clone 逻辑不变）...

    # ---- 可选：相机传感器 ----
    self._has_camera = self.cfg.camera_cfg is not None
    if self._has_camera:
        from isaaclab.sensors import TiledCamera
        self.camera = TiledCamera(self.cfg.camera_cfg)
        self.scene.sensors["mast_camera"] = self.camera
        print(f"[INFO] TiledCamera 已创建: {self.cfg.camera_cfg.prim_path}, "
              f"resolution={self.cfg.camera_cfg.width}x{self.cfg.camera_cfg.height}")

    # lights ...

新增一个获取相机数据的方法（不影响原有 _get_observations）：

def get_camera_images(self) -> torch.Tensor | None:
    """返回当前帧的 RGB 图像 (N, H, W, 3) uint8，如果无相机返回 None。
    
    注意: TiledCamera 返回 (N, H, W, 4) RGBA, 需截取前 3 通道。
    """
    if not self._has_camera:
        return None
    # TiledCamera.data.output["rgb"] 返回 (N, H, W, 4) float [0,1] 或 uint8
    # 具体格式取决于 Isaac Lab 版本，这里做兼容处理
    rgba = self.camera.data.output["rgb"]  # (N, H, W, 4)
    rgb = rgba[..., :3]  # 去掉 alpha
    if rgb.dtype == torch.float32:
        rgb = (rgb * 255).to(torch.uint8)
    return rgb  # (N, H, W, 3) uint8

文件 3: verify_camera_view.py（新文件）

"""快速验证相机视野的脚本。

用法:
    cd IsaacLab
    ./isaaclab.sh -p scripts/verify_camera_view.py --num_envs 4
    
运行后保存 4 张截图到 camera_debug/ 目录，人工确认视野覆盖。
"""
import argparse
import torch
import numpy as np
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_envs", type=int, default=4)
    parser.add_argument("--out_dir", type=str, default="camera_debug")
    args = parser.parse_args()
    
    # 导入并配置环境（启用相机）
    from isaaclab_tasks.direct.forklift_pallet_insert_lift.env_cfg import (
        ForkliftPalletInsertLiftEnvCfg,
    )
    from isaaclab_tasks.direct.forklift_pallet_insert_lift.env import (
        ForkliftPalletInsertLiftEnv,
    )
    from isaaclab.envs import DirectRLEnvCfg
    
    cfg = ForkliftPalletInsertLiftEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.camera_cfg = cfg.make_camera_cfg()  # 启用相机
    
    env = ForkliftPalletInsertLiftEnv(cfg)
    
    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)
    
    # Reset 并截图
    env.reset()
    # 跑几步让物理稳定
    for _ in range(30):
        dummy_action = torch.zeros(args.num_envs, 3, device=env.device)
        env.step(dummy_action)
    
    images = env.get_camera_images()  # (N, H, W, 3)
    if images is None:
        print("[ERROR] 相机未创建！检查 camera_cfg")
        return
    
    # 保存截图
    from PIL import Image
    for i in range(min(args.num_envs, 8)):
        img = images[i].cpu().numpy()
        Image.fromarray(img).save(out_dir / f"env_{i}_initial.png")
        print(f"[OK] 保存 env_{i}_initial.png: shape={img.shape}, "
              f"min={img.min()}, max={img.max()}")
    
    # 手动前进到不同阶段截图
    # 让叉车前进一段距离
    drive_action = torch.tensor([[0.3, 0.0, 0.0]] * args.num_envs, device=env.device)
    for step in range(300):  # ~10 秒
        env.step(drive_action)
        if step in [60, 150, 250]:  # 约 2s, 5s, 8s
            images = env.get_camera_images()
            for i in range(min(args.num_envs, 4)):
                img = images[i].cpu().numpy()
                Image.fromarray(img).save(
                    out_dir / f"env_{i}_step{step}.png")
    
    print(f"\n[DONE] 截图已保存到 {out_dir}/")
    print("请人工检查:")
    print("  1. 远场: 能否看到完整托盘轮廓？")
    print("  2. 中场: 能否看到托盘开口？")
    print("  3. 近场: 叉齿进入时能否看到插入过程？")
    print("  4. 是否有遮挡（门架、驾驶室）？")
    
    env.close()

if __name__ == "__main__":
    from isaaclab.app import AppLauncher
    app_launcher = AppLauncher(headless=True)
    main()
    app_launcher.close()

1.4 数据同步方案

关键对齐关系：

Physics timestep:     dt = 1/120 s
Decimation:           4
Control step:         dt_ctrl = 4/120 = 1/30 s ≈ 33.3 ms
Camera update:        1/30 s（与控制步同步）
Episode length:       最长 1080 control steps = 36s

一个控制步的时序：
  t                 t + dt_ctrl
  |----|----|----|----|
  ↑phy  phy  phy  phy↑
                      ↑
                camera.update() → RGB_t
                _get_observations() → obs_t (15d), obs_gt_t
                _get_rewards() → rew_t
                _pre_physics_step(action_t) → 缓存 action_t

时间戳定义（每个样本）：

@dataclass
class SampleMeta:
    env_id: int          # 并行环境索引 [0, num_envs)
    episode_id: int      # 该 env 的第几个 episode（自增）
    step_id: int         # episode 内步数 [0, max_ep_len)
    global_step: int     # 全局步数（所有 env 累计）
    sim_time_s: float    # 仿真时间 = step_id * dt_ctrl

同步保证：Isaac Lab 的 step() 内部按顺序调用 _pre_physics_step → physics × decimation → _get_observations。在 _get_observations() 返回时，PhysX 状态和相机渲染已经完成。因此在 step() 返回后读取相机数据和观测，二者是同步的。

推荐的采集时机：

# 在 collect_data.py 的主循环中
obs = env.reset()  # type: dict["policy": Tensor(N,15)]
done = False

while not done:
    # 1. 用 teacher 计算动作
    action = teacher.infer_batch(obs["policy"].cpu().numpy())
    action_tensor = torch.from_numpy(action).to(env.device)
    
    # 2. step（内部完成 physics + camera render + obs 计算）
    obs, reward, terminated, truncated, info = env.step(action_tensor)
    
    # 3. 此时读取的一切都与当前 step 同步
    images = env.get_camera_images()       # RGB_t
    obs_15d = obs["policy"]                # obs_t (包含 teacher 真值)
    # ... 记录样本 ...

1.5 最小可运行版本的验证 Checklist

阶段 A：相机创建验证（~2 小时）

[] env_cfg.py 中 make_camera_cfg() 编译无错
[] 4 envs headless 模式启动不崩溃
[] get_camera_images() 返回 (4, 224, 224, 3) uint8 张量
[] 图像不全黑 / 不全白（灯光和相机都正确创建）
[] 保存为 PNG 后人眼可见叉车/托盘几何

阶段 B：视野覆盖验证（~2 小时）

[] 远场 (d_x ≈ 3.5m)：画面中能看到完整托盘
[] 中场 (d_x ≈ 1.5m)：托盘开口清晰可见
[] 近场 (d_x ≈ 0.3m)：可以看到叉齿与托盘的空间关系
[] 插入过程中：画面有可辨别的变化（叉齿消失进托盘内部）
[] 偏航 ±15° 时：托盘仍在画面内
[] 横偏 ±0.6m 时：托盘仍在画面内
[] 无严重遮挡（门架/驾驶室不挡住主视野）

阶段 C：标签对齐验证（~3 小时）

[] 记录同步的 (image, obs_15d, teacher_action)
[] 用 obs[0:2] (d_xy_r) 在图像上标注托盘预期位置，检查一致性
[] 用 obs[9] (insert_norm) 与图像中叉齿可见长度对比
[] 跑 10 个完整 episode，确认每个阶段都有覆盖

阶段 D：吞吐量验证（~1 小时）

[] 64 envs + TiledCamera：测量 step/s（预期 > 100 step/s）
[] 128 envs + TiledCamera：测量 step/s 并确认 GPU 显存不溢出
[] 估算采集 200K 帧需要的时间（预期 < 30 分钟）

1.6 第一阶段最容易踩的坑

坑
表现
规避建议
相机坐标系与 robot frame 不一致
图像中叉车朝"错误"方向；标签 d_x 正负号反转
Isaac Sim 相机默认 convention 是 USD (Y-up, -Z forward)，需在 OffsetCfg 中设 convention="world" 或 "ros" 并验证。用 verify_camera_view.py 手动对照。
叉车视觉几何不真实
叉车渲染为纯色方块或没有细节
forklift_c.usd 自带视觉 mesh，但如果环境中 visual_material 未启用或光照不够，可能看起来很假。先确认 DomeLightCfg(intensity=2000) 已开启。
货叉/门架遮挡
近场时门架立柱遮住托盘
调整相机 offset.pos 让相机略高于或绕过遮挡物。大部分 forklift_c 模型的门架是半透明结构或较细，遮挡有限。
相机帧率和控制步不对齐
图像是上一步的
确保 update_period 与 dt * decimation 一致。在 step() 返回后才读取图像。不要在 _pre_physics_step 中读取。
多 env + camera 显存爆炸
CUDA OOM
用 TiledCamera（非 Camera）；先用 64 envs 测试；224×224 分辨率 + 64 envs ≈ 额外 ~0.3 GB 显存（可控）。如果仍 OOM，降到 32 envs。
标签时序偏移
Student 学到的"这张图对应的 insert_norm"其实是上一步的
重点验证：在 step() 返回后，obs["policy"] 和 get_camera_images() 必须都反映 本步 physics 之后 的状态。Isaac Lab 的标准 step 流程保证了这一点，但 custom env 里如果覆写了 step 顺序需小心。
env_spacing 不足导致相邻环境串入画面
图像中出现隔壁 env 的叉车/托盘
当前 env_spacing=6.0m，对 15m 远裁面可能不够。将远裁面设为 8.0m 或增大 env_spacing 到 10.0m。
TiledCamera 在 Isaac Lab 某些版本上 API 不同
AttributeError 或 data.output 结构不同
先打印 dir(camera.data) 和 camera.data.output.keys() 确认。Isaac Lab 2.1 vs 2.3 的 camera 数据结构有变化。


---

第 2 部分：Teacher 跑仿真并采集数据集

2.1 样本 Schema

每个控制步记录的完整数据结构：

@dataclass
class DataSample:
    """单个控制步的完整记录。"""

    # ---- 元数据 ----
    env_id: int                      # 并行环境 index
    episode_id: int                  # 该 env 的 episode 序号
    step_id: int                     # episode 内步数
    global_frame_id: int             # 全局帧 ID（唯一标识）
    sim_time_s: float                # 仿真时间
    seed: int                        # 随机种子

    # ---- 视觉输入 ----
    rgb: np.ndarray                  # (224, 224, 3) uint8
    depth: np.ndarray | None         # (224, 224) float32, meters（可选）

    # ---- Easy states（车载传感器直读）----
    v_xy_r: np.ndarray               # (2,) float32, robot frame 速度
    yaw_rate: float                  # rad/s
    lift_pos: float                  # m
    lift_vel: float                  # m/s
    prev_actions: np.ndarray         # (3,) float32, [-1, 1]
    # 打包为 8 维向量：[v_x, v_y, yr, lp, lv, pa0, pa1, pa2]

    # ---- 几何真值标签（student 需要学习的）----
    pallet_pos_robot: np.ndarray     # (2,) d_x, d_y in robot frame (m)
    dyaw_rad: float                  # pallet_yaw - robot_yaw (rad)
    insert_depth_m: float            # 插入深度 (m, ≥0)
    y_signed_m: float                # 中心线横向偏差 (m, 带符号)

    # ---- 衍生标签（用于验证转换正确性）----
    cos_dyaw: float
    sin_dyaw: float
    insert_norm: float               # clip(insert_depth_m / 2.16, 0, 1)
    y_err_obs: float                 # clip(y_signed_m / 0.8, -1, 1)
    yaw_err_obs: float               # clip(-dyaw_rad / deg2rad(15), -1, 1)

    # ---- 完整 teacher 观测（15d，用于对照）----
    teacher_obs_15d: np.ndarray      # (15,) float32

    # ---- Teacher 动作 ----
    teacher_action: np.ndarray       # (3,) float32, [-1, 1]

    # ---- Episode 状态 ----
    reward: float
    done: bool
    truncated: bool
    success: bool                    # hold_counter >= hold_steps
    insert_norm_gt: float            # 与 insert_norm 应一致（冗余校验）

    # ---- 额外诊断（可选，不必每帧存）----
    pallet_pos_world: np.ndarray     # (3,) 托盘世界坐标
    robot_pos_world: np.ndarray      # (3,) 机器人世界坐标
    fork_tip_world: np.ndarray       # (3,) 叉齿尖端世界坐标
    pallet_displacement_m: float     # 托盘位移

最终存储的核心字段（按优先级）：

级别
字段
大小/帧
必须存
P0
rgb
~10 KB (JPEG)
✅
P0
easy_states (8d)
32 B
✅
P0
geometric labels (4d): [d_x, d_y, dyaw, ins_d]
16 B
✅
P0
teacher_action (3d)
12 B
✅
P0
meta: env_id, ep_id, step_id
12 B
✅
P1
teacher_obs_15d
60 B
✅
P1
derived labels (7d)
28 B
✅
P1
done, success, reward
6 B
✅
P2
depth
~100 KB (float16→PNG)
首次采集
P2
world poses (robot, pallet, tip)
36 B
可选

2.2 文件组织结构

dataset/
├── manifest.json                  # 数据集元信息（版本、采集配置、统计）
├── splits.json                    # train/val/test episode 划分
├── stats.json                     # 标签统计（mean/std/min/max per dim）
│
├── images/
│   ├── ep_0000/                   # 按 episode 组织
│   │   ├── frame_000000.jpg       # env_id=X, step_id=0
│   │   ├── frame_000001.jpg
│   │   └── ...
│   ├── ep_0001/
│   └── ...
│
├── depth/                         # 可选，结构同 images/
│   └── ...
│
├── labels/
│   ├── ep_0000.npz                # 每个 episode 一个 npz 文件
│   │   # 内含:
│   │   #   geometric_labels: (T, 4) float32  [d_x, d_y, dyaw, ins_d]
│   │   #   derived_labels:   (T, 7) float32  [7 missing obs]
│   │   #   easy_states:      (T, 8) float32
│   │   #   teacher_obs:      (T, 15) float32
│   │   #   teacher_actions:  (T, 3) float32
│   │   #   rewards:          (T,) float32
│   │   #   dones:            (T,) bool
│   │   #   success:          bool (episode-level)
│   │   #   meta: dict (env_id, seed, ep_length, ...)
│   ├── ep_0001.npz
│   └── ...
│
└── README.md                      # 数据集说明

为什么按 episode 组织而不是按帧：

1. 保持 episode 内时序完整性，方便按 episode 做 train/val/test 切分
2. 避免同一 episode 的帧泄露到不同 split
3. 方便分析 per-episode 成功率
4. npz 压缩效率高（episode 内标签连续，压缩比好）
  
2.3 Train/Val/Test 划分

# splits.json 示例
{
    "version": "v1.0",
    "split_method": "by_episode",
    "train": [0, 1, 2, ..., 799],        # 80% episodes
    "val":   [800, 801, ..., 899],        # 10% episodes
    "test":  [900, 901, ..., 999],        # 10% episodes
    "hard_cases": [42, 137, 256, ...],    # 横偏 > 0.4m 或 yaw > 10° 的 episode
    "notes": "按 episode_id 固定划分，不按帧。Hard cases 从 train 中抽取，用于 debug。"
}

划分原则：

- 严格按 episode 划分，绝不把同一 episode 的帧分到不同 split
- 按初始条件分层抽样：确保 train/val/test 中都有足够的大横偏、大角偏、接近中心等 case
- Val 和 test 设置不同的随机种子（val 种子 = train 种子 + 10000，test 种子 = train 种子 + 20000）
- Hard cases 子集：从 val 中筛选初始 |y₀| > 0.4m 或 |yaw₀| > 10° 的 episode，单独评估
  
2.4 采样覆盖策略

teacher 的 ~89% 成功率意味着 ~11% 失败。需要确保数据覆盖所有阶段：

阶段
step 占比 (成功ep)
关注点
覆盖策略
远场接近 (d>2m)
~30%
d_x, d_y, dyaw 变化范围大
自然覆盖
中场对齐 (1-2m)
~25%
y_err, yaw_err 需精确
自然覆盖
近场插入 (<1m)
~20%
insert_depth 快速变化，y_err 关键
自然覆盖
深插举升
~15%
insert_norm > 0.4, lift_pos 变化
可能不足*
成功保持
~10%
静止状态
可能过少
失败 case
~11% ep
各种失败模式
需专门保留

具体策略：

1. 基础采集：用 teacher 自然跑 1000+ episodes，不做任何干预
2. 失败保留：不删除失败 episode。失败 case 提供"边界行为"数据，学生需要知道这些区域的观测是什么样的
3. 数据增强覆盖：对于 insert_depth > 0.5 的深插帧可能较少，可以通过：
  - 适当增加 episode 数量
  - 或在 env_cfg 中把 episode_length_s 临时设长（45s），让 teacher 有更多时间完成
4. 初始条件多样化：保持 reset 中的随机化范围（x∈[-4,-3], y∈[-0.6,+0.6], yaw∈[-14°,+14°]），自然产生多样化初始条件
  
帧级分布验证 Checklist：

采集完成后，绘制以下分布直方图：
[] d_x 分布（应覆盖 [-1, 4] m）
[] y_err 分布（应覆盖 [-0.6, 0.6] m）
[] dyaw 分布（应覆盖 [-0.25, 0.25] rad）
[] insert_norm 分布（应覆盖 [0, 0.5+]，深插区域有足够样本）
[] lift_pos 分布（应覆盖 [0, 1.0+] m）
[] 每个阶段的帧数统计

2.5 Teacher Action 是否一起存

必须存。 理由：

用途
何时使用
怎么用
主 loss：缺失 obs 监督
训练第一版 student
image → geometric_labels, 与 GT 做 MSE
辅 loss：action 蒸馏
加强 student 后续迭代
student_obs → teacher_actor → pred_action vs GT action
DAgger 参考
Phase 4 闭环迭代
student 跑偏时用 teacher action 作为纠正信号
分析 baseline
验证管线
"如果标签完美，teacher actor 能达到多少成功率？"（应≈89%）

联合 loss 设计（第一版不用，备用）：

# 主 loss：几何标签回归
loss_obs = MSE(pred_geometric, gt_geometric)  # 权重 1.0

# 辅 loss：action 蒸馏（第二版加入）
# 把 student 预估的 missing obs 拼上 easy states → 15d obs → teacher actor → pred action
# vs teacher GT action
loss_action = MSE(pred_action_via_student, gt_teacher_action)  # 权重 0.1

loss = loss_obs + 0.1 * loss_action

2.6 数据量建议

版本
Episodes
总帧数(估算)
采集时间(64 envs)
磁盘空间
用途
v0.0 (验证)
50
~22K
~2 min
~0.3 GB
验证管线正确性，student 能否 overfit
v0.1 (baseline)
500
~220K
~15 min
~3 GB
训练第一版 student
v1.0 (完整)
2000
~880K
~1 hr
~12 GB
正式训练
v2.0 (增强)
5000+
~2.2M
~3 hr
~30 GB
加 DR 后的大规模数据

执行建议：先采 v0.0 验证管线，通过后采 v0.1 训练 baseline，根据效果决定是否需要 v1.0。

2.7 数据采集脚本（核心伪代码）

"""collect_data.py — 用 teacher 采集视觉训练数据。

用法:
    cd IsaacLab
    ./isaaclab.sh -p scripts/collect_data.py \
        --teacher_model deployment/model_1999.pt \
        --num_envs 64 \
        --num_episodes 500 \
        --out_dir dataset/v0.1
"""

import argparse, json, time
from pathlib import Path
import numpy as np
import torch
from PIL import Image

def main():
    args = parse_args()
    
    # ---- 1. 环境初始化（启用相机）----
    from isaaclab_tasks.direct.forklift_pallet_insert_lift.env_cfg import (
        ForkliftPalletInsertLiftEnvCfg,
    )
    from isaaclab_tasks.direct.forklift_pallet_insert_lift.env import (
        ForkliftPalletInsertLiftEnv,
    )
    
    cfg = ForkliftPalletInsertLiftEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.camera_cfg = cfg.make_camera_cfg()  # 启用相机
    
    env = ForkliftPalletInsertLiftEnv(cfg)
    
    # ---- 2. Teacher 策略 ----
    from infer import ForkliftPolicy
    teacher = ForkliftPolicy(args.teacher_model, device="cpu")
    
    # ---- 3. 数据采集主循环 ----
    out_dir = Path(args.out_dir)
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    (out_dir / "labels").mkdir(parents=True, exist_ok=True)
    
    ep_counts = np.zeros(args.num_envs, dtype=int)  # 每个 env 的 episode 计数
    total_episodes_done = 0
    
    # 每个 env 维护的 episode 缓冲
    ep_buffers = {i: EpisodeBuffer(env_id=i) for i in range(args.num_envs)}
    
    obs_dict = env.reset()
    prev_actions = np.zeros((args.num_envs, 3), dtype=np.float32)
    
    pbar = tqdm(total=args.num_episodes, desc="Episodes")
    
    while total_episodes_done < args.num_episodes:
        # ---- Teacher 推理 ----
        obs_np = obs_dict["policy"].cpu().numpy()  # (N, 15)
        actions = teacher.infer_batch(obs_np)       # (N, 3)
        action_tensor = torch.from_numpy(actions).to(env.device)
        
        # ---- 采集当前帧数据 ----
        images = env.get_camera_images()  # (N, 224, 224, 3) uint8
        
        for i in range(args.num_envs):
            # 提取几何真值标签
            geo_label = extract_geometric_labels(
                obs_15d=obs_np[i],
                env=env,
                env_idx=i,
            )
            
            ep_buffers[i].add_frame(
                rgb=images[i].cpu().numpy(),
                easy_states=obs_np[i, [4,5,6,7,8,10,11,12]],  # 8d
                geometric_labels=geo_label,   # 4d
                teacher_obs=obs_np[i],        # 15d
                teacher_action=actions[i],    # 3d
            )
        
        # ---- Step 环境 ----
        obs_dict, rewards, terminated, truncated, infos = env.step(action_tensor)
        
        # ---- 检查 episode 结束 ----
        dones = terminated | truncated
        for i in range(args.num_envs):
            if dones[i]:
                # 保存完整 episode
                ep_id = f"ep_{total_episodes_done:06d}"
                ep_buffers[i].save(
                    out_dir, ep_id,
                    success=bool(env._hold_counter[i] >= env._hold_steps),
                )
                ep_buffers[i].reset()
                total_episodes_done += 1
                pbar.update(1)
                
                if total_episodes_done >= args.num_episodes:
                    break
        
        prev_actions = actions.copy()
    
    pbar.close()
    
    # ---- 4. 保存统计信息 ----
    compute_and_save_stats(out_dir)
    generate_splits(out_dir, train_frac=0.8, val_frac=0.1)
    
    print(f"\n[DONE] 采集完成: {total_episodes_done} episodes → {out_dir}")
    env.close()


def extract_geometric_labels(obs_15d: np.ndarray, env, env_idx: int) -> np.ndarray:
    """从 teacher 15d 观测中提取 4 维几何真值标签。
    
    Returns:
        [d_x, d_y, dyaw_rad, insert_depth_m]
    """
    d_x = obs_15d[0]   # 已在 robot frame
    d_y = obs_15d[1]
    
    # dyaw = atan2(sin_dyaw, cos_dyaw) = pallet_yaw - robot_yaw
    cos_d = obs_15d[2]
    sin_d = obs_15d[3]
    dyaw = np.arctan2(sin_d, cos_d)
    
    # insert_depth_m = insert_norm * pallet_depth_m
    insert_norm = obs_15d[9]
    insert_depth_m = insert_norm * 2.16  # pallet_depth_m
    
    return np.array([d_x, d_y, dyaw, insert_depth_m], dtype=np.float32)


class EpisodeBuffer:
    """单个 episode 的帧缓冲。"""
    
    def __init__(self, env_id: int):
        self.env_id = env_id
        self.frames_rgb = []
        self.frames_easy = []
        self.frames_geo = []
        self.frames_obs = []
        self.frames_act = []
    
    def add_frame(self, rgb, easy_states, geometric_labels, 
                  teacher_obs, teacher_action):
        self.frames_rgb.append(rgb)
        self.frames_easy.append(easy_states)
        self.frames_geo.append(geometric_labels)
        self.frames_obs.append(teacher_obs)
        self.frames_act.append(teacher_action)
    
    def save(self, out_dir: Path, ep_id: str, success: bool):
        ep_len = len(self.frames_rgb)
        
        # 保存图像
        img_dir = out_dir / "images" / ep_id
        img_dir.mkdir(parents=True, exist_ok=True)
        for t, rgb in enumerate(self.frames_rgb):
            Image.fromarray(rgb).save(
                img_dir / f"frame_{t:06d}.jpg", quality=90)
        
        # 保存标签
        np.savez_compressed(
            out_dir / "labels" / f"{ep_id}.npz",
            geometric_labels=np.stack(self.frames_geo),     # (T, 4)
            easy_states=np.stack(self.frames_easy),         # (T, 8)
            teacher_obs=np.stack(self.frames_obs),          # (T, 15)
            teacher_actions=np.stack(self.frames_act),      # (T, 3)
            success=success,
            ep_length=ep_len,
            env_id=self.env_id,
        )
    
    def reset(self):
        self.frames_rgb.clear()
        self.frames_easy.clear()
        self.frames_geo.clear()
        self.frames_obs.clear()
        self.frames_act.clear()


---

第 3 部分：离线训练 Student

3.1 第一版任务定义排序

优先级
任务定义
推荐理由
不推荐理由
★★★
image + easy_states → 4d 几何中间量
最优平衡：复用 teacher actor；easy states 提供运动先验减少歧义；4d 输出简洁；可通过确定性变换还原 7d obs
—
★★
image → 4d 几何中间量
纯视觉方案，部署最简；但失去运动状态信息，近场单帧歧义大
近场插入阶段 insert_depth 完全依赖视觉，误差可能大
★
image + easy_states → 7d obs 直接
省去后处理，但 cos/sin 约束 + clip 饱和区域梯度消失
输出维度多；需要额外 loss 结构处理约束
✗
image → action (3d)
直接蒸馏行为，但丢失了 teacher actor 的泛化能力和可解释性
不推荐第一版做：teacher actor 已调好，不应丢弃

第一版推荐：image + easy_states → 4d 几何中间量 [d_x, d_y, dyaw_rad, insert_depth_m]

理由：
1. easy_states 不白给：lift_pos 对 insert_depth 估计有直接帮助（叉齿高度约束）；prev_actions 提供运动连续性先验
2. 4 维比 7 维更好学：更少输出 = 更快收敛 = 更小模型
3. 几何中间量语义清晰：每一维都有明确物理意义，误差分析容易
4. 确定性后处理：用 geometric_to_obs() 函数将 4d → 7d，无可训练参数，不引入额外误差源
  
3.2 标签参数化深度对比

Option A: 直接回归 7 维 obs

输出: [d_x, d_y, cos_dyaw, sin_dyaw, insert_norm, y_err_obs, yaw_err_obs]

维度
值域
Loss 设计
问题
d_x, d_y
[-5, 5] m
MSE
✅ 正常
cos_dyaw
[-1, 1]
MSE
⚠️ cos²+sin²=1 约束难强制
sin_dyaw
[-1, 1]
MSE
⚠️ 同上
insert_norm
[0, 1]
MSE
⚠️ 在 0 和 1 处 clip → 梯度消失
y_err_obs
[-1, 1]
MSE
⚠️ 在 ±1 处 clip → 梯度消失
yaw_err_obs
[-1, 1]
MSE
⚠️ 在 ±1 处 clip → 梯度消失

致命问题：cos_dyaw 和 sin_dyaw 应满足 cos² + sin² = 1。如果网络预测 (0.8, 0.9)，范数 = 1.2 ≠ 1。后续 teacher actor 从未见过这种不一致输入，行为不可预测。

虽然可以用 atan2(sin, cos) → normalize 后处理，但这等于在推理时做了一次非线性变换，不如直接预测角度。

Option B: 回归 4 维几何量（推荐）

输出: [d_x, d_y, dyaw_rad, insert_depth_m]

维度
值域（训练数据范围）
Loss 设计
备注
d_x
[-1.5, 4.0] m
MSE
✅ 连续无约束
d_y
[-1.0, 1.0] m
MSE
✅ 连续无约束
dyaw_rad
[-0.3, 0.3] rad
MSE
✅ 小角度范围，无 wrapping 问题
insert_depth_m
[0, 1.1] m
MSE
✅ 连续无约束

后处理函数（确定性，无可训练参数）：

def geometric_to_obs(geo_4d: np.ndarray,
                     pallet_depth_m: float = 2.16,
                     y_err_scale: float = 0.8) -> np.ndarray:
    """将 4 维几何预测转换为 7 维 missing obs。
    
    Args:
        geo_4d: [d_x, d_y, dyaw_rad, insert_depth_m]
    
    Returns:
        missing_obs_7d: [d_x, d_y, cos_dyaw, sin_dyaw, 
                         insert_norm, y_err_obs, yaw_err_obs]
    """
    d_x, d_y, dyaw, ins_d = geo_4d
    
    cos_dyaw = np.cos(dyaw)
    sin_dyaw = np.sin(dyaw)
    insert_norm = np.clip(ins_d / pallet_depth_m, 0.0, 1.0)
    
    # y_signed: 机器人在 pallet center-line frame 中的横向偏差
    # 推导: y_signed = d_x * sin(dyaw) - d_y * cos(dyaw)
    # （从 robot frame 的 pallet 位置和相对朝向推导）
    y_signed = d_x * np.sin(dyaw) - d_y * np.cos(dyaw)
    y_err_obs = np.clip(y_signed / y_err_scale, -1.0, 1.0)
    
    # yaw_err: robot_yaw - pallet_yaw = -dyaw
    yaw_err_obs = np.clip(-dyaw / np.radians(15.0), -1.0, 1.0)
    
    return np.array([d_x, d_y, cos_dyaw, sin_dyaw,
                     insert_norm, y_err_obs, yaw_err_obs],
                    dtype=np.float32)

Option B 优势总结：
- 4 维 < 7 维：更少参数量、更快收敛
- 无约束输出：MSE loss 梯度均匀，不存在饱和死区
- cos/sin 始终精确归一：np.cos/sin 保证 cos²+sin²=1
- y_err_obs 和 yaw_err_obs 由几何推导，不会出现不一致
- insert_norm 的 clip 在后处理中完成，不影响 loss 梯度
  
Option B 风险与缓解：
- y_signed 对 dyaw 的误差敏感：当 d_x ≈ 3m 时，dyaw 误差 1° → y_signed 误差 ≈ 3×0.017 ≈ 5cm。这在 y_err_obs 的 scale (0.8m) 内可接受。
- insert_depth_m 在远场时恒为 0：loss 被 0 值主导。缓解：对 insert_depth_m > 0 的帧使用更高权重（见 3.4 节）。
  
3.3 网络结构推荐

Baseline A（最小，推荐第一版）：MobileNetV3-Small + MLP

       RGB (224×224×3)
            │
   ┌────────▼────────┐
   │ MobileNetV3-Small│  ← pretrained on ImageNet, freeze first 3 stages
   │ (backbone only)  │
   └────────┬────────┘
            │  features: (N, 576)     ← global avg pool 后
            │
   easy_states (8)
            │
   ┌────────▼────────┐
   │ concat → 584     │
   │ Linear(584, 256) │
   │ ReLU              │
   │ Linear(256, 64)  │
   │ ReLU              │
   │ Linear(64, 4)    │  ← [d_x, d_y, dyaw, ins_d]
   └────────┬────────┘
            │
   geometric_to_obs()   ← 确定性后处理
            │
   7 missing obs dims

属性
值
参数量
~2.7M (backbone ~2.5M + head ~0.2M)
FLOPs
~60 MFLOPs
推理延迟 (CPU, ARM)
~15-25 ms
推理延迟 (DV500 NPU)
~5-10 ms (估算)
ONNX 导出
✅ 无自定义算子

Baseline B（稍强）：ResNet-18 + MLP

同上结构，backbone 替换为 ResNet-18
features: (N, 512) → concat easy_states → 520 → MLP → 4

属性
值
参数量
~11.5M
FLOPs
~1.8 GFLOPs
推理延迟 (CPU, ARM)
~30-50 ms
推理延迟 (DV500 NPU)
可能超标

是否需要时序？

方案
第一版
后续
单帧
✅ 推荐
—
2 帧堆叠 (t, t-1)
❌
如果单帧 insert_depth 精度不够
小 GRU (hidden=64)
❌
最后手段，边缘部署困难

单帧优先的理由：
1. 部署最简单（无状态管理）
2. 叉车运动缓慢（30Hz → 每帧移动 ~1cm），单帧信息足够
3. easy_states 中的 prev_actions 和 v_xy_r 已经间接提供了时序信息
4. insert_depth 的视觉估计可能需要运动线索，但这是后续优化，不应成为 V1 的阻碍
  
为什么这样设计对 DV500 友好

DV500 约束：
- NPU 算力有限（~2-4 TOPS），大模型推不动
- 支持 INT8 量化的标准 CNN 算子
- 不支持自定义 CUDA kernel
  
MobileNetV3-Small 的优势：
- 专门为移动/边缘设计
- 所有算子（depthwise conv, SE module, hardswish）都有 INT8 量化对应
- 2.7M 参数量化后 < 3 MB
- 60 MFLOPs @ INT8 → DV500 上 ~5ms
  
3.4 Loss 设计

class StudentLoss(nn.Module):
    def __init__(self):
        super().__init__()
        # 各维度标签的 std（从 stats.json 加载），用于归一化 loss
        # 典型值: d_x~1.5m, d_y~0.3m, dyaw~0.1rad, ins_d~0.3m
        self.label_std = nn.Parameter(
            torch.tensor([1.5, 0.3, 0.1, 0.3]), requires_grad=False)
    
    def forward(self, pred_4d, gt_4d, phase_weights=None):
        """
        Args:
            pred_4d: (B, 4) [d_x, d_y, dyaw, ins_d]
            gt_4d:   (B, 4) [d_x, d_y, dyaw, ins_d]
            phase_weights: (B,) 可选的样本权重
        """
        # 归一化误差：各维度除以 std，使 loss 量纲一致
        err = (pred_4d - gt_4d) / self.label_std
        loss_per_dim = err ** 2  # (B, 4)
        
        # 维度权重
        dim_weights = torch.tensor([1.0, 1.0, 2.0, 2.0])
        # dyaw 和 insert_depth 权重 2x：teacher 对这两个维度更敏感
        
        loss = (loss_per_dim * dim_weights).mean(dim=1)  # (B,)
        
        if phase_weights is not None:
            loss = loss * phase_weights
        
        return loss.mean()

维度权重初始化理由：

维度
权重
理由
d_x
1.0
远场时重要，近场时不太敏感
d_y
1.0
全程重要但变化范围小
dyaw
2.0
teacher 对 yaw 误差极其敏感（5° 就影响成功率）
insert_depth
2.0
控制举升时机的关键量，误差直接导致 premature lift

阶段重加权策略：

def compute_phase_weights(gt_4d: torch.Tensor) -> torch.Tensor:
    """根据任务阶段给不同帧不同权重。"""
    d_x = gt_4d[:, 0]
    ins_d = gt_4d[:, 3]
    
    w = torch.ones_like(d_x)
    
    # 近场帧权重 ×2（d_x < 1.5m）
    w = torch.where(d_x < 1.5, w * 2.0, w)
    
    # 插入阶段帧权重 ×3（insert_depth > 0.1m）
    w = torch.where(ins_d > 0.1, w * 3.0, w)
    
    # 归一化使 mean(w) = 1
    w = w / w.mean()
    
    return w

可选的 Action 蒸馏 Loss（第二版加入）：

def action_distillation_loss(student_geo_4d, easy_states_8d, 
                              gt_teacher_action, teacher_actor, 
                              obs_normalizer):
    """闭环 action 蒸馏 loss。
    
    流程: student_output → geometric_to_obs → 拼 15d → teacher_actor → pred_action
    """
    # 1. 几何后处理
    missing_7d = batch_geometric_to_obs(student_geo_4d)  # (B, 7)
    
    # 2. 重建 15d obs
    obs_15d = reconstruct_full_obs(missing_7d, easy_states_8d)  # (B, 15)
    
    # 3. 过 teacher actor
    obs_norm = obs_normalizer.normalize(obs_15d)
    pred_action = teacher_actor(obs_norm)  # (B, 3)
    pred_action = torch.clamp(pred_action, -1, 1)
    
    # 4. 与 GT action 做 MSE
    return F.mse_loss(pred_action, gt_teacher_action)

⚠️ 注意：这个 loss 需要通过 teacher actor 做前向传播，将梯度传回 student。teacher actor 的权重冻结，但 student 的梯度通过 geometric_to_obs 的可微部分（cos/sin/clip 都可微）回传。第一版建议先不加，先验证纯几何回归的效果。

3.5 训练 Pipeline 组织

# train_student.py 伪代码

# ---- DataLoader ----
class ForkliftDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, split="train", augment=True):
        self.episodes = load_episode_list(data_dir, split)
        self.augment = augment
        self.transform = transforms.Compose([
            transforms.ColorJitter(brightness=0.2, contrast=0.2,
                                   saturation=0.1, hue=0.05),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
        ]) if augment else transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
        ])
    
    def __getitem__(self, idx):
        ep_id, frame_id = self.index_map[idx]
        
        # 加载图像
        img = Image.open(f"images/{ep_id}/frame_{frame_id:06d}.jpg")
        img_tensor = self.transform(img)  # (3, 224, 224)
        
        # 加载标签
        labels = np.load(f"labels/{ep_id}.npz")
        geo_4d = labels["geometric_labels"][frame_id]     # (4,)
        easy_8d = labels["easy_states"][frame_id]          # (8,)
        teacher_action = labels["teacher_actions"][frame_id]  # (3,)
        teacher_obs = labels["teacher_obs"][frame_id]      # (15,)
        
        return {
            "image": img_tensor,          # (3, 224, 224) float32
            "easy_states": easy_8d,       # (8,) float32
            "geo_label": geo_4d,          # (4,) float32
            "teacher_action": teacher_action,
            "teacher_obs": teacher_obs,
        }

# ---- Augmentations ----
# 第一版只做颜色增强（不做几何增强，因为标签与视角强耦合）
# 后续可加: 
#   - Gaussian noise (σ=0.01), 模拟传感器噪声
#   - Random erasing, 模拟遮挡
#   - Brightness/contrast 幅度加大, 模拟光照变化

# ---- Training Loop ----
model = StudentModel()  # MobileNetV3-Small + MLP
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

best_val_loss = float("inf")
patience_counter = 0

for epoch in range(100):
    # Train
    model.train()
    for batch in train_loader:
        pred_4d = model(batch["image"], batch["easy_states"])
        loss = criterion(pred_4d, batch["geo_label"],
                        phase_weights=compute_phase_weights(batch["geo_label"]))
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    
    # Validate
    model.eval()
    val_metrics = evaluate(model, val_loader)
    
    # Early stopping
    if val_metrics["loss"] < best_val_loss:
        best_val_loss = val_metrics["loss"]
        torch.save(model.state_dict(), "best_student.pt")
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= 15:
            break
    
    scheduler.step()
    log_metrics(epoch, val_metrics)

3.6 评估指标定义

离线帧级指标

def compute_frame_metrics(pred_4d, gt_4d, gt_obs_15d):
    """计算每个维度的帧级误差。"""
    metrics = {}
    
    # ---- 几何维度误差 ----
    metrics["d_x_mae"] = np.abs(pred_4d[:, 0] - gt_4d[:, 0]).mean()     # m
    metrics["d_y_mae"] = np.abs(pred_4d[:, 1] - gt_4d[:, 1]).mean()     # m
    metrics["dyaw_mae_deg"] = np.abs(pred_4d[:, 2] - gt_4d[:, 2]).mean() * 180/np.pi  # deg
    metrics["ins_d_mae"] = np.abs(pred_4d[:, 3] - gt_4d[:, 3]).mean()   # m
    
    # ---- 转换后 obs 维度误差 ----
    pred_7d = batch_geometric_to_obs(pred_4d)
    gt_7d = batch_geometric_to_obs(gt_4d)
    
    metrics["cos_dyaw_mae"] = np.abs(pred_7d[:, 2] - gt_7d[:, 2]).mean()
    metrics["sin_dyaw_mae"] = np.abs(pred_7d[:, 3] - gt_7d[:, 3]).mean()
    metrics["insert_norm_mae"] = np.abs(pred_7d[:, 4] - gt_7d[:, 4]).mean()
    metrics["y_err_obs_mae"] = np.abs(pred_7d[:, 5] - gt_7d[:, 5]).mean()
    metrics["yaw_err_obs_mae"] = np.abs(pred_7d[:, 6] - gt_7d[:, 6]).mean()
    
    # ---- 分阶段指标 ----
    far_mask = gt_4d[:, 0] > 2.0    # 远场
    near_mask = gt_4d[:, 0] < 1.0   # 近场
    insert_mask = gt_4d[:, 3] > 0.1  # 插入阶段
    
    for name, mask in [("far", far_mask), ("near", near_mask), ("insert", insert_mask)]:
        if mask.sum() > 0:
            metrics[f"d_x_mae_{name}"] = np.abs(pred_4d[mask, 0] - gt_4d[mask, 0]).mean()
            metrics[f"d_y_mae_{name}"] = np.abs(pred_4d[mask, 1] - gt_4d[mask, 1]).mean()
            metrics[f"dyaw_mae_deg_{name}"] = np.abs(pred_4d[mask, 2] - gt_4d[mask, 2]).mean() * 180/np.pi
            metrics[f"ins_d_mae_{name}"] = np.abs(pred_4d[mask, 3] - gt_4d[mask, 3]).mean()
    
    return metrics

关键阈值（student 必须达到才值得做闭环）：

指标
合格线
优秀线
理由
d_x MAE
< 0.20 m
< 0.10 m
teacher obs[0] 的典型动态范围 ~4m
d_y MAE
< 0.08 m
< 0.04 m
teacher 成功要求 y_err < 0.15m
dyaw MAE
< 3.0°
< 1.5°
teacher 成功要求 yaw_err < 5°
insert_d MAE
< 0.10 m
< 0.05 m
insert_thresh ≈ 0.86m, 10cm 误差可接受
d_x MAE (near)
< 0.10 m
< 0.05 m
近场精度要求更高
d_y MAE (near)
< 0.05 m
< 0.03 m
近场横向对齐是成功关键
dyaw MAE (near)
< 2.0°
< 1.0°
近场偏航对齐是成功关键
ins_d MAE (insert)
< 0.08 m
< 0.04 m
控制举升时机

闭环 Proxy 指标（离线可估算）：

def compute_action_consistency(pred_4d, gt_4d, easy_states, 
                                teacher_actor, obs_normalizer):
    """估算: 如果用 student 的预测替换真值，teacher action 会偏多少？
    
    这是闭环成功率的最佳离线 proxy。
    """
    # 用 student 预测重建 15d obs
    pred_obs = reconstruct_full_obs(
        batch_geometric_to_obs(pred_4d), easy_states)
    
    # 用真值重建 15d obs（基线）
    gt_obs = reconstruct_full_obs(
        batch_geometric_to_obs(gt_4d), easy_states)
    
    # 过 teacher actor
    pred_action = teacher_actor(obs_normalizer.normalize(pred_obs))
    gt_action = teacher_actor(obs_normalizer.normalize(gt_obs))
    
    action_mae = np.abs(pred_action - gt_action).mean(axis=0)
    # action_mae[0] = drive 偏差, [1] = steer 偏差, [2] = lift 偏差
    
    return {
        "action_drive_mae": action_mae[0],  # 合格线 < 0.05
        "action_steer_mae": action_mae[1],  # 合格线 < 0.03（转向最敏感）
        "action_lift_mae": action_mae[2],   # 合格线 < 0.05
    }

3.7 Baseline 优先级

顺序
Baseline
目的
预计耗时
1
几何 Oracle
用 GT 观测（但不用 GT action）重建 15d obs → teacher actor → 闭环。验证"如果 student 完美，系统能到多少成功率"。答案应 ≈89%。
2 小时
2
MLP Baseline（喂真值 4d + easy 8d）
不用图像，直接用真值几何量训练 MLP(12→4)。验证标签参数化和 geometric_to_obs() 转换是否正确。应几乎完美。
半天
3
MobileNetV3-Small + easy states
第一版 student。RGB → backbone → concat easy → MLP → 4d。
1-2 天
4
MobileNetV3-Small（纯视觉，无 easy states）
测量 easy states 的增益。如果差很多 → easy states 必须保留。
1 天
5
ResNet-18 + easy states
测量 backbone 容量的增益。如果 MobileNetV3 已够 → 不需升级。
1 天
6
MobileNetV3 + action distillation loss
加入 teacher action 蒸馏辅助 loss。测量额外监督的增益。
1 天


---

第 4 部分：接回 Teacher 做闭环评估

4.1 Student 接回 ForkliftPolicy 的具体方案

class StudentForkliftPolicy:
    """用 student 视觉网络替换缺失观测的闭环策略。
    
    数据流:
    1. camera → RGB image
    2. 车载传感器 → easy_states (8d)
    3. student_net(image, easy_states) → geometric_pred (4d)
    4. geometric_to_obs(geometric_pred) → missing_obs (7d)
    5. reconstruct_full_obs(missing_obs, easy_states) → obs_15d
    6. teacher_actor(obs_normalizer(obs_15d)) → action (3d)
    7. map_action_to_physical(action) → CAN 指令
    """
    
    def __init__(self, 
                 student_model_path: str,
                 teacher_model_path: str,
                 device: str = "cpu"):
        self.device = torch.device(device)
        
        # 加载 student
        self.student = load_student_model(student_model_path)
        self.student.to(self.device).eval()
        
        # 加载 teacher actor + normalizer
        from infer import ForkliftPolicy
        self.teacher = ForkliftPolicy(teacher_model_path, device=device)
        
        # 状态
        self._prev_actions = np.zeros(3, dtype=np.float32)
    
    @torch.no_grad()
    def infer(self, 
              rgb: np.ndarray,           # (224, 224, 3) uint8
              v_xy_r: np.ndarray,        # (2,) float32
              yaw_rate: float,
              lift_pos: float,
              lift_vel: float) -> np.ndarray:
        """单步推理（闭环调用）。"""
        
        # 1. 构建 easy_states
        easy_states = np.array([
            v_xy_r[0], v_xy_r[1],
            yaw_rate,
            lift_pos, lift_vel,
            self._prev_actions[0],
            self._prev_actions[1],
            self._prev_actions[2],
        ], dtype=np.float32)
        
        # 2. Student 网络推理
        img_tensor = preprocess_image(rgb).unsqueeze(0).to(self.device)
        easy_tensor = torch.from_numpy(easy_states).unsqueeze(0).to(self.device)
        geo_pred = self.student(img_tensor, easy_tensor)  # (1, 4)
        geo_pred = geo_pred.squeeze(0).cpu().numpy()       # (4,)
        
        # 3. 几何后处理
        missing_7d = geometric_to_obs(geo_pred)            # (7,)
        
        # 4. 重建 15d obs
        obs_15d = np.zeros(15, dtype=np.float32)
        obs_15d[0:2] = missing_7d[0:2]     # d_x, d_y
        obs_15d[2]   = missing_7d[2]        # cos_dyaw
        obs_15d[3]   = missing_7d[3]        # sin_dyaw
        obs_15d[4:6] = v_xy_r               # v_xy_r
        obs_15d[6]   = yaw_rate             # yaw_rate
        obs_15d[7]   = lift_pos             # lift_pos
        obs_15d[8]   = lift_vel             # lift_vel
        obs_15d[9]   = missing_7d[4]        # insert_norm
        obs_15d[10:13] = self._prev_actions # prev_actions
        obs_15d[13]  = missing_7d[5]        # y_err_obs
        obs_15d[14]  = missing_7d[6]        # yaw_err_obs
        
        # 5. Teacher actor 推理
        action = self.teacher.infer(obs_15d)  # (3,)
        
        # 6. 更新状态
        self._prev_actions = action.copy()
        
        return action

字段来源汇总：

obs 索引
字段
来源
首步初始化
0-1
d_xy_r
student 预测
student(首帧图像)
2-3
cos/sin_dyaw
student 预测 → cos/sin
同上
4-5
v_xy_r
仿真 PhysX / 车载 IMU
[0, 0]
6
yaw_rate
仿真 PhysX / 车载 IMU
0
7
lift_pos
仿真 PhysX / 编码器
0
8
lift_vel
仿真 PhysX / 编码器
0
9
insert_norm
student 预测 → clip
0（远场）
10-12
prev_actions
控制器自身缓存
[0, 0, 0]
13
y_err_obs
student 预测 → 几何推导
student(首帧图像)
14
yaw_err_obs
student 预测 → 几何推导
student(首帧图像)

4.2 三层闭环评估方案

Level 1：离线回放评估

做法：用 val/test 数据集的帧，逐帧预测并与 GT 对比。

指标：
- 帧级几何误差（3.6 节所有指标）
- Action consistency（pred_obs → teacher_action vs gt_action）
- 分阶段误差分布图
  
通过标准：action_steer_mae < 0.03 且 action_drive_mae < 0.05

耗时：~10 分钟（纯 CPU 推理 + 计算指标）

Level 2：半闭环评估

做法：在 Isaac Lab 仿真中运行，逐维度替换 student 预测，测量对成功率的影响。

实验矩阵:
  a) 全部 15d 用真值（oracle baseline, ≈89%）
  b) 只替换 d_xy_r (obs[0:1])，其余真值
  c) 只替换 cos/sin_dyaw (obs[2:3])，其余真值
  d) 只替换 insert_norm (obs[9])，其余真值
  e) 只替换 y_err + yaw_err (obs[13:14])，其余真值
  f) 替换全部 7 维 missing obs，easy states 用真值

目的：定位"哪个维度的 student 误差对闭环影响最大"。如果 (c) 掉了 20% 成功率但 (b) 只掉 5%，说明 dyaw 估计是瓶颈。

指标：
- 每个配置跑 200 episodes，统计成功率
- 对比各配置的 hold_counter 分布
- 记录失败模式
  
Level 3：全闭环评估

做法：完全用 student(camera) + easy_states(PhysX) → teacher_actor，跑完整 episode。

"""closedloop_eval.py — student 全闭环评估。"""

def run_closedloop_eval(env, student_policy, num_episodes=500):
    """
    env: 带相机的 ForkliftPalletInsertLiftEnv
    student_policy: StudentForkliftPolicy 实例
    """
    success_count = 0
    total_count = 0
    
    obs_dict = env.reset()
    ep_logs = []
    
    while total_count < num_episodes:
        # 获取相机图像和 easy states
        images = env.get_camera_images()     # (N, 224, 224, 3)
        obs_15d = obs_dict["policy"]          # (N, 15) — 作为 easy states 的来源
        
        # Student 推理（逐 env）
        actions = np.zeros((env.num_envs, 3), dtype=np.float32)
        for i in range(env.num_envs):
            actions[i] = student_policy.infer(
                rgb=images[i].cpu().numpy(),
                v_xy_r=obs_15d[i, 4:6].cpu().numpy(),
                yaw_rate=obs_15d[i, 6].item(),
                lift_pos=obs_15d[i, 7].item(),
                lift_vel=obs_15d[i, 8].item(),
            )
        
        action_tensor = torch.from_numpy(actions).to(env.device)
        obs_dict, rew, term, trunc, info = env.step(action_tensor)
        
        # 记录 episode 结束
        dones = term | trunc
        for i in range(env.num_envs):
            if dones[i]:
                is_success = env._hold_counter[i] >= env._hold_steps
                success_count += int(is_success)
                total_count += 1
    
    success_rate = success_count / total_count
    return success_rate

指标：

指标
定义
目标
闭环成功率
hold 10 步 / 总 episodes
≥ 80%
平均完成时间
成功 episode 的 mean step count
< 550 步 (teacher ~450)
失败模式分布
timeout / tipped / fly / stall 各占比
—
横向漂移累积
episode 内
student_y_err - gt_y_err
偏航抖动

student_yaw_err - gt_yaw_err
insert_norm 跟踪误差

student_insert - gt_insert
托盘推移
pallet displacement 的 mean 和 max
mean < 0.10m
崩坏行为
teacher 从未出现但 student 出现的新失败模式
无

4.3 闭环效果不好时的修复优先级

优先级
策略
解决的问题
什么信号触发
预计耗时
1
加 easy states（如果还没加）
近场歧义、insert_depth 精度
Level 2 显示 insert_norm 单维度替换就大幅掉分
半天
2
数据覆盖补齐
特定阶段/初始条件数据不足
分阶段误差分析显示某阶段 MAE 远高于其他
1 天
3
阶段重加权
近场/插入阶段 loss 被远场淹没
远场指标优秀但近场/插入阶段差
2 小时
4
标签参数化调整
geometric_to_obs 转换引入误差放大
d_y 和 dyaw 的 MAE 都不大，但 y_err_obs 误差大
半天
5
2 帧堆叠
单帧无法区分的运动状态
insert_depth 在快速插入时跳变
1 天
6
多任务 loss（加 action 蒸馏）
几何标签精度足够但 action 一致性差
帧级指标达标但闭环掉分
1 天
7
DAgger
分布偏移（student 闭环行为与 teacher 的数据分布不同）
Level 3 闭环出现 teacher 未见过的状态
2-3 天
8
少量微调 teacher
teacher 对 student 噪声不鲁棒
student obs 误差 < 阈值但 teacher action 剧烈振荡
2-3 天（谨慎）

详细说明各策略：

策略 1（加 easy states）：如果第一版实验选了纯视觉路线（无 easy states），这是最先尝试的增强。easy_states 中的 lift_pos 直接约束了 insert_depth 的可行范围；prev_actions 提供运动连续性。实测增益通常 5-15% 这一量级。

策略 4（标签参数化调整）：如果发现 y_err_obs 闭环误差大但 d_y 和 dyaw 的 MAE 都不大，可能是 y_signed = d_x * sin(dyaw) - d_y * cos(dyaw) 中误差被 d_x（较大数）× sin(dyaw误差) 放大。此时考虑：
- 增加 y_signed_m 为 student 的第 5 个直接输出
- 或者直接回归 y_err_obs（Option A），绕过几何推导
  
策略 7（DAgger）：见 4.6 节。

4.4 Student 闭环的新失败模式诊断

teacher 训练时从未出现过但 student 可能触发的行为：

崩坏行为
原因
诊断信号
修复
叉车原地旋转不前进
student 的 d_x 估计跳变，teacher 不断调整方向
高频 steer oscillation + 低 drive
加时序（2帧）或低通滤波 student 输出
已插入但不举升
insert_norm 估计偏低，insert_gate_norm (0.35) 未触发
lift_cmd ≈ 0 且 gt_insert > 0.4
对 insert_depth 标签加权 or 偏移修正
远场偏航过大、永远无法对齐
dyaw 估计有固定偏差（bias）
远场 dyaw 残差分布不居中
数据增强 + 标签去均值（零均值化）
撞击托盘（未对齐就冲进去）
d_x 估计过大（以为还远），teacher 给出大 drive
episode 早期即碰撞
验证 d_x 在近场的精度
举升后叉齿抖动
student 在高 lift_pos 时视野变化大，预测不稳定
举升阶段 geo_pred 方差大
训练数据中举升阶段帧加权

推荐的闭环诊断 log（每步记录，用于事后分析）：

closedloop_log = {
    "step": step_id,
    "student_geo_pred": geo_pred,            # student 的 4d 预测
    "gt_geo": gt_geo,                        # 真值 4d（仿真环境可获得）
    "student_obs_15d": obs_15d_from_student,  # 重建的 15d obs
    "gt_obs_15d": obs_15d_gt,                # 真值 15d obs
    "teacher_action_from_student": action,    # 基于 student obs 的 action
    "teacher_action_from_gt": action_gt,      # 基于真值 obs 的 action（参考）
    "reward": reward,
    "hold_counter": hold_counter,
}

4.5 Domain Randomization 策略

何时引入：Phase 3 student 第一版跑通后（NOT 第一版）

理由：
1. 第一版目标是验证 teacher-student 管线可行性。如果干净数据上 student 都学不好，加 DR 也救不了
2. DR 每新增一种随机化都需要重新采集数据、重新训练，调参成本高
3. 先在无 DR 条件下建立 baseline，有对照组才能衡量 DR 的真实增益
  
DR 引入顺序（按价值递减）：

顺序
随机化类型
实现方式
对 sim2real 的价值
1
光照强度/方向
Isaac Sim DomeLightCfg 中随机 intensity ∈ [500, 5000]，color bias ±0.1
高
2
相机外参微扰
offset.pos ±5cm, offset.rot ±3°
高
3
相机内参微扰
focal_length ±2mm
中
4
图像后处理噪声
Gaussian noise σ∈[0, 0.02], motion blur, JPEG artifact
中
5
托盘纹理随机
替换 pallet USD 材质为随机木纹/贴图
中
6
地面纹理随机
替换 ground plane 材质
低
7
叉车自身观测噪声
v_xy_r ±0.05 m/s, yaw_rate ±0.01 rad/s, lift_pos ±2mm
低（easy states 精度高）
✗
托盘位姿随机
当前 teacher 只在原点训练
先不碰，teacher 的泛化性未验证

先别碰：
- 不要改托盘位姿随机化（teacher 没见过，改了需要重训 teacher）
- 不要做图像风格迁移（GAN-based），成本太高且不稳定
  
4.6 DAgger 最小可执行方案

┌────── Round 0: Teacher 采集 ──────┐
│ teacher policy → 500 episodes      │
│ 记录: (image, easy, geo_label,     │
│        teacher_action)              │
│ 数据集: D_0 (~220K frames)         │
└──────────────────┬─────────────────┘
                   ▼
         训练 Student v0 on D_0
                   │
                   ▼
┌────── Round 1: Student 闭环 ──────┐
│ student v0 → 200 episodes          │
│ 每步同时记录:                       │
│   - student 的观测重建              │
│   - GT 几何标签（仿真中可获得）       │
│   - teacher 的 GT action            │
│     (用真值 obs → teacher actor      │
│      计算的"纠正 action")           │
│ 数据集: D_1 (~88K frames)           │
│ !!关键!!: D_1 反映了 student 闭环   │
│ 时 agent 实际访问的状态分布           │
└──────────────────┬─────────────────┘
                   ▼
         合并: D_agg = D_0 ∪ D_1
         重训 Student v1 on D_agg
                   │
                   ▼
┌────── Round 2: Student 闭环 ──────┐
│ student v1 → 200 episodes          │
│ 数据集: D_2                        │
│ D_agg = D_0 ∪ D_1 ∪ D_2          │
│ 重训 Student v2                    │
└──────────────────┬─────────────────┘
                   ▼
         ...重复直到收敛...

停止条件：
- 连续两轮闭环成功率差 < 2%
- 或成功率已达 ≥ 85%
- 或已迭代 5 轮（通常 2-3 轮即收敛）
  
Round 1 的核心实现：

def dagger_round(env, student_policy, teacher_policy, num_episodes=200):
    """在 student 闭环运行时，同时记录 teacher 的纠正标签。"""
    
    obs_dict = env.reset()
    
    while not done:
        images = env.get_camera_images()
        obs_gt = obs_dict["policy"]  # 真值 15d obs
        
        # Student 闭环动作
        student_action = student_policy.infer_batch(images, obs_gt[:, [4,5,6,7,8,10,11,12]])
        
        # Teacher 纠正动作（用真值 obs）
        teacher_action = teacher_policy.infer_batch(obs_gt.cpu().numpy())
        
        # GT 几何标签（从 obs_gt 提取）
        gt_geo = extract_labels_from_obs(obs_gt)
        
        # Student 预测（用于诊断分布偏移）
        student_geo = student_policy.predict_geo(images, easy_states)
        
        # 记录到 DAgger 数据集
        record(image=images, 
               geo_label=gt_geo,           # GT 标签
               teacher_action=teacher_action,
               student_geo_pred=student_geo,  # 分析用
               distribution_shift=np.abs(student_geo - gt_geo))
        
        # 用 STUDENT 的动作推进仿真（不是 teacher！这是 DAgger 的关键）
        obs_dict, _, _, _, _ = env.step(
            torch.from_numpy(student_action).to(env.device))


---

第 5 部分：综合输出

5.1 Repo / 文件级改动建议

forklift_pallet_insert_lift_project/
├── isaaclab_patch/
│   └── .../forklift_pallet_insert_lift/
│       ├── env_cfg.py                 [修改] 新增 camera_cfg + make_camera_cfg()
│       ├── env.py                     [修改] _setup_scene() 条件创建相机；
│       │                                      新增 get_camera_images()
│       └── ...（其余不改）
│
├── deployment/
│   ├── infer.py                       [不改] 保持原有推理管线
│   ├── model_1999.pt                  [不改]
│   └── ...
│
├── student/                           [新目录]
│   ├── README.md                      # student 训练说明
│   ├── requirements.txt               # torchvision, pillow, tqdm
│   │
│   ├── data/
│   │   ├── collect_data.py            # 数据采集脚本
│   │   ├── verify_camera_view.py      # 相机视野验证
│   │   ├── compute_stats.py           # 标签统计
│   │   └── visualize_dataset.py       # 数据可视化
│   │
│   ├── model/
│   │   ├── student_net.py             # Student 网络定义
│   │   ├── geometric_transform.py     # geometric_to_obs() + reconstruct_full_obs()
│   │   └── loss.py                    # Loss 函数
│   │
│   ├── train/
│   │   ├── train_student.py           # 训练脚本
│   │   ├── evaluate_offline.py        # 离线评估
│   │   └── configs/                   # 训练超参配置
│   │
│   ├── eval/
│   │   ├── closedloop_eval.py         # L3 全闭环评估
│   │   ├── semiclosed_eval.py         # L2 半闭环评估
│   │   ├── student_forklift_policy.py # StudentForkliftPolicy 封装
│   │   └── dagger.py                  # DAgger 闭环迭代
│   │
│   └── export/
│       └── export_student_onnx.py     # 导出整合模型
│
└── scripts/
    └── install_into_isaaclab.sh       [修改] 可选复制 student/ 到 IsaacLab

5.2 Student 网络定义（关键伪代码）

# student/model/student_net.py

import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

class ForkliftStudentNet(nn.Module):
    """视觉感知 Student：image + easy_states → 4d geometric prediction。
    
    Architecture:
        RGB (224×224) → MobileNetV3-Small → 576d features
        concat easy_states (8d) → 584d
        MLP(584 → 256 → 64 → 4)
    
    Output: [d_x, d_y, dyaw_rad, insert_depth_m]
    """
    
    def __init__(self, 
                 easy_state_dim: int = 8,
                 output_dim: int = 4,
                 pretrained_backbone: bool = True,
                 freeze_backbone_stages: int = 3):
        super().__init__()
        
        # ---- Backbone ----
        backbone = mobilenet_v3_small(
            weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1 
            if pretrained_backbone else None
        )
        # 去掉分类头，只保留特征提取
        self.features = backbone.features       # 输出 (N, 576, 7, 7)
        self.avgpool = backbone.avgpool          # 输出 (N, 576, 1, 1)
        self.feature_dim = 576
        
        # 冻结前 N 个 stage（减少过拟合，加速训练）
        if freeze_backbone_stages > 0:
            for i, layer in enumerate(self.features):
                if i < freeze_backbone_stages:
                    for param in layer.parameters():
                        param.requires_grad = False
        
        # ---- Fusion Head ----
        fusion_dim = self.feature_dim + easy_state_dim
        self.head = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, output_dim),
        )
        
        # 输出归一化参数（从 stats.json 加载，使输出接近 0 均值）
        # 训练时设为标签的 mean，推理时减去
        self.register_buffer("output_mean", torch.zeros(output_dim))
        self.register_buffer("output_std", torch.ones(output_dim))
    
    def forward(self, image: torch.Tensor, 
                easy_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: (N, 3, 224, 224) float32, ImageNet-normalized
            easy_states: (N, 8) float32
        Returns:
            pred: (N, 4) [d_x, d_y, dyaw_rad, insert_depth_m]
        """
        # Visual features
        x = self.features(image)       # (N, 576, 7, 7)
        x = self.avgpool(x)           # (N, 576, 1, 1)
        x = x.flatten(1)              # (N, 576)
        
        # Fuse with easy states
        x = torch.cat([x, easy_states], dim=1)  # (N, 584)
        
        # Predict geometric quantities
        pred = self.head(x)           # (N, 4)
        
        # De-normalize（训练时标签已 normalize，推理时还原）
        pred = pred * self.output_std + self.output_mean
        
        return pred
    
    def count_params(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}


# ---- 推理时的完整管线 ----
class StudentInferencePipeline:
    """封装 student + geometric_transform + teacher actor。"""
    
    def __init__(self, student_path, teacher_path, device="cpu"):
        self.device = torch.device(device)
        
        # Student
        self.student = ForkliftStudentNet()
        self.student.load_state_dict(torch.load(student_path, map_location=device))
        self.student.to(self.device).eval()
        
        # Teacher
        from infer import ForkliftPolicy
        self.teacher = ForkliftPolicy(teacher_path, device=device)
        
        # Image preprocessing
        from torchvision import transforms
        self.img_transform = transforms.Compose([
            transforms.ToTensor(),  # HWC uint8 → CHW float [0,1]
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
        ])
        
        self._prev_actions = np.zeros(3, dtype=np.float32)
    
    @torch.no_grad()
    def step(self, rgb_hwc_uint8, v_xy, yr, lp, lv):
        """完整的 student → teacher 推理流程。"""
        from student.model.geometric_transform import (
            geometric_to_obs, reconstruct_full_obs)
        
        # Preprocess image
        img = self.img_transform(rgb_hwc_uint8).unsqueeze(0).to(self.device)
        
        # Easy states
        easy = np.array([v_xy[0], v_xy[1], yr, lp, lv,
                         *self._prev_actions], dtype=np.float32)
        easy_t = torch.from_numpy(easy).unsqueeze(0).to(self.device)
        
        # Student prediction
        geo_pred = self.student(img, easy_t).squeeze(0).cpu().numpy()
        
        # Convert to obs
        missing_7d = geometric_to_obs(geo_pred)
        obs_15d = reconstruct_full_obs(missing_7d, easy)
        
        # Teacher action
        action = self.teacher.infer(obs_15d)
        self._prev_actions = action.copy()
        
        return action

5.3 geometric_transform.py（关键后处理）

# student/model/geometric_transform.py

import numpy as np
import math

def geometric_to_obs(geo_4d: np.ndarray,
                     pallet_depth_m: float = 2.16,
                     y_err_scale: float = 0.8) -> np.ndarray:
    """4d 几何预测 → 7d missing obs。
    
    Input:  [d_x, d_y, dyaw_rad, insert_depth_m]
    Output: [d_x, d_y, cos_dyaw, sin_dyaw, insert_norm, y_err_obs, yaw_err_obs]
    
    ⚠️ 符号约定（必须与 env.py / build_obs 一致）:
    - dyaw = pallet_yaw - robot_yaw（正 = 托盘偏左）
    - yaw_err_obs 使用 robot_yaw - pallet_yaw = -dyaw
    - y_signed = d_x * sin(dyaw) - d_y * cos(dyaw)
    """
    d_x, d_y = geo_4d[0], geo_4d[1]
    dyaw = geo_4d[2]
    ins_d = max(geo_4d[3], 0.0)  # 插入深度非负
    
    cos_dyaw = math.cos(dyaw)
    sin_dyaw = math.sin(dyaw)
    insert_norm = np.clip(ins_d / pallet_depth_m, 0.0, 1.0)
    
    y_signed = d_x * math.sin(dyaw) - d_y * math.cos(dyaw)
    y_err_obs = np.clip(y_signed / y_err_scale, -1.0, 1.0)
    
    yaw_err_obs = np.clip(-dyaw / math.radians(15.0), -1.0, 1.0)
    
    return np.array([d_x, d_y, cos_dyaw, sin_dyaw,
                     insert_norm, y_err_obs, yaw_err_obs], dtype=np.float32)


def reconstruct_full_obs(missing_7d: np.ndarray,
                         easy_states_8d: np.ndarray) -> np.ndarray:
    """重建完整 15 维 teacher obs。
    
    missing_7d: [d_x, d_y, cos_dyaw, sin_dyaw, insert_norm, y_err_obs, yaw_err_obs]
    easy_states_8d: [v_x, v_y, yr, lp, lv, pa0, pa1, pa2]
    
    Output obs 顺序必须与 env.py _get_observations 一致:
    [d_x, d_y, cos_dyaw, sin_dyaw, v_x, v_y, yr, lp, lv, 
     insert_norm, pa0, pa1, pa2, y_err_obs, yaw_err_obs]
    """
    obs = np.zeros(15, dtype=np.float32)
    obs[0:2] = missing_7d[0:2]     # d_x, d_y
    obs[2]   = missing_7d[2]        # cos_dyaw
    obs[3]   = missing_7d[3]        # sin_dyaw
    obs[4]   = easy_states_8d[0]    # v_x
    obs[5]   = easy_states_8d[1]    # v_y
    obs[6]   = easy_states_8d[2]    # yaw_rate
    obs[7]   = easy_states_8d[3]    # lift_pos
    obs[8]   = easy_states_8d[4]    # lift_vel
    obs[9]   = missing_7d[4]        # insert_norm
    obs[10]  = easy_states_8d[5]    # prev_action_drive
    obs[11]  = easy_states_8d[6]    # prev_action_steer
    obs[12]  = easy_states_8d[7]    # prev_action_lift
    obs[13]  = missing_7d[5]        # y_err_obs
    obs[14]  = missing_7d[6]        # yaw_err_obs
    return obs


def batch_geometric_to_obs(geo_batch: np.ndarray, **kwargs) -> np.ndarray:
    """批量版本。输入 (B, 4), 输出 (B, 7)。"""
    return np.stack([geometric_to_obs(g, **kwargs) for g in geo_batch])

5.4 数据集 Schema（结构化定义）

# dataset schema v1.0
SCHEMA = {
    "version": "1.0",
    "frame_fields": {
        # 视觉输入
        "rgb": {"shape": (224, 224, 3), "dtype": "uint8", "storage": "jpeg"},
        "depth": {"shape": (224, 224), "dtype": "float16", "storage": "png16", "optional": True},
        
        # Easy states（车载传感器）
        "easy_states": {
            "shape": (8,), "dtype": "float32",
            "fields": ["v_x", "v_y", "yaw_rate", "lift_pos", "lift_vel",
                       "prev_act_drive", "prev_act_steer", "prev_act_lift"]
        },
        
        # 几何标签（student 学习目标）
        "geometric_labels": {
            "shape": (4,), "dtype": "float32",
            "fields": ["d_x_m", "d_y_m", "dyaw_rad", "insert_depth_m"]
        },
        
        # 衍生标签（用于验证 geometric_to_obs）
        "derived_obs_labels": {
            "shape": (7,), "dtype": "float32",
            "fields": ["d_x", "d_y", "cos_dyaw", "sin_dyaw",
                       "insert_norm", "y_err_obs", "yaw_err_obs"]
        },
        
        # Teacher 完整状态（用于 oracle baseline 和调试）
        "teacher_obs": {"shape": (15,), "dtype": "float32"},
        "teacher_action": {"shape": (3,), "dtype": "float32"},
        
        # Episode 状态
        "reward": {"shape": (), "dtype": "float32"},
        "done": {"shape": (), "dtype": "bool"},
        "success": {"shape": (), "dtype": "bool"},
    },
    
    "episode_fields": {
        "env_id": "int",
        "episode_length": "int",
        "is_success": "bool",
        "initial_d_x": "float32",
        "initial_y_err": "float32",
        "initial_yaw_err_deg": "float32",
        "max_insert_norm": "float32",
        "max_lift_height": "float32",
    }
}

5.5 闭环评估 Checklist

□ Phase 1: 相机验证
  □ 1.1 4 envs 创建不崩溃
  □ 1.2 图像非全黑/全白
  □ 1.3 远场/中场/近场/插入 四个阶段截图覆盖
  □ 1.4 无严重遮挡
  □ 1.5 env_spacing 足够（相邻环境不串入）
  □ 1.6 帧率测试：64 envs + camera step/s > 100

□ Phase 2: 数据采集
  □ 2.1 v0.0 (50 ep) 采集成功，文件完整
  □ 2.2 标签与 GT obs 一致性验证（逐维度 diff < 1e-5）
  □ 2.3 geometric_to_obs() 单元测试通过（对 env.py 算出的 7d 做 diff）
  □ 2.4 数据分布直方图覆盖所有阶段
  □ 2.5 v0.1 (500 ep) 采集完成
  □ 2.6 splits.json 生成，train:val:test = 80:10:10

□ Phase 3: Student 训练
  □ 3.1 Oracle baseline 闭环 ≈ 89%（验证管线正确）
  □ 3.2 MLP baseline（真值输入）train loss → ~0（验证标签正确）
  □ 3.3 MobileNetV3 v0 训练完成
  □ 3.4 离线帧级 MAE 达标（见 3.6 节阈值表）
  □ 3.5 Action consistency MAE 达标
  □ 3.6 分阶段误差分析完成

□ Phase 4: 闭环评估
  □ 4.1 Level 1：离线 val/test 指标全部达标
  □ 4.2 Level 2：半闭环逐维度替换实验完成
       □ 4.2.1 只替换 d_xy_r → 成功率 vs oracle 差值 < 5%
       □ 4.2.2 只替换 cos/sin_dyaw → 差值 < 5%
       □ 4.2.3 只替换 insert_norm → 差值 < 5%
       □ 4.2.4 只替换 y/yaw_err → 差值 < 5%
       □ 4.2.5 替换全部 7d → 差值 < 10%
  □ 4.3 Level 3：全闭环 500 episodes
       □ 成功率 ≥ 80%
       □ 无新失败模式
       □ 横向漂移累积 < 0.10m
       □ 偏航抖动 std < 2°
  □ 4.4 诊断 log 保存并分析完毕

5.6 风险清单和排查路径

风险
概率
影响
排查路径
缓解措施
Isaac Lab 版本差异导致 TiledCamera API 不兼容
中
高
dir(camera.data) 确认接口；查阅 version changelog
准备 Camera (非 Tiled) 回退方案
forklift_c.usd 视觉 mesh 太简化导致图像信息不足
低
高
截图检查；与真实叉车照片对比
使用更高保真度叉车 USD；或加 DR 弥补
相机位置选择不当导致近场全白/过曝
中
中
verify_camera_view.py 逐阶段截图
调整 offset / FOV；加入 medium exposure 纹理
geometric_to_obs 符号错误
高
极高
对 v0.0 数据集逐帧做 diff（derived_labels vs GT obs），diff < 1e-5
单元测试 + 数据集验证（Phase 2.3）
insert_depth 在近场估计不准
高
高
分阶段 MAE 分析；近场 insert MAE 是否远高于远场
加 easy states (lift_pos)；加阶段重加权；2 帧堆叠
Student 训练过拟合（数据量不够）
中
中
train vs val loss 差距；val 指标不下降
增大数据量 → v1.0；加 dropout；加 augmentation
闭环误差累积（snowball effect）
高
高
闭环 episode 内误差随 step 递增
DAgger；低通滤波 student 输出；考虑 EMA 平滑
64 envs + camera + teacher 推理速度太慢
中
低
测量 step/s；如 < 50，减少 envs
减少 envs 到 32；JPEG 异步写入；batch teacher 推理

5.7 建议的执行顺序

优先级  任务                                     预计耗时    依赖
─────────────────────────────────────────────────────────────
P0.1   env_cfg.py 添加 camera_cfg               2h        无
P0.2   env.py 添加 get_camera_images()           2h        P0.1
P0.3   verify_camera_view.py 截图验证            2h        P0.2
P0.4   确认视野覆盖，调整 offset/FOV             2h        P0.3
─── checkpoint: 相机可用 ────────────────────────
P1.1   geometric_transform.py 编写 + 单元测试    3h        无
P1.2   collect_data.py 编写                      4h        P0.4, P1.1
P1.3   v0.0 数据采集 (50 ep) + 标签验证           1h        P1.2
P1.4   标签一致性验证 (diff < 1e-5)               1h        P1.3
─── checkpoint: 数据管线正确 ───────────────────
P2.1   v0.1 数据采集 (500 ep)                    30min     P1.4
P2.2   compute_stats.py + 分布可视化              2h        P2.1
P2.3   生成 splits.json                           30min     P2.2
─── checkpoint: 数据集就绪 ────────────────────
P3.1   student_net.py (MobileNetV3 + MLP)        3h        无
P3.2   loss.py                                   2h        无
P3.3   Oracle baseline 闭环验证（≈89%）           2h        P0.4
P3.4   MLP baseline（真值输入）→ loss ≈ 0         2h        P2.3
P3.5   MobileNetV3 训练 (v0.1 数据集)             8h(GPU)   P2.3, P3.1
P3.6   离线评估 + 分阶段误差分析                   2h        P3.5
─── checkpoint: Student v0 训练完成 ────────────
P4.1   StudentForkliftPolicy 封装                3h        P3.5
P4.2   Level 1：离线 action consistency 评估       1h        P4.1
P4.3   Level 2：半闭环逐维度替换实验               4h        P4.1
P4.4   Level 3：全闭环 500 episodes               2h        P4.3
P4.5   诊断 log 分析 + 失败模式分类                3h        P4.4
─── checkpoint: 首轮闭环评估完成 ────────────────
P5.1   根据 P4.5 诊断结果选择修复策略              1h        P4.5
P5.2   执行修复（数据/权重/结构调整）              1-3d      P5.1
P5.3   DAgger Round 1（如需）                     1d        P5.2
─── checkpoint: Student v1 达标 ────────────────

5.8 week 1 到 week 2 的务实推进计划

Week 1：建管线（Phase 1 + Phase 2 + Phase 3 baseline）

Day
任务
交付物
Mon
P0.1-P0.4: 加相机 + 验证视野
camera_debug/ 截图确认
Tue
P1.1-P1.4: 采集管线 + v0.0 数据 + 标签验证
dataset/v0.0/ + 标签 diff 报告
Wed AM
P2.1-P2.3: v0.1 数据采集 + 统计
dataset/v0.1/ + 分布直方图
Wed PM
P3.1-P3.2: Student 网络 + Loss 定义
student_net.py, loss.py
Thu
P3.3-P3.4: Oracle baseline + MLP baseline
"管线正确性已验证" 的结论
Fri
P3.5 开始: MobileNetV3 训练（跑过夜）
训练启动，监控 loss 曲线

Week 1 末的交付标准：
[] 相机视野覆盖所有阶段（有截图证明）
[] v0.1 数据集就绪（500 ep, ~220K frames）
[] geometric_to_obs 单元测试通过（diff < 1e-5）
[] Oracle baseline 闭环 ≈ 89%
[] MobileNetV3 student 训练中

Week 2：训练 + 闭环评估 + 迭代

Day
任务
交付物
Mon
P3.5-P3.6: student v0 训练完成 + 离线评估
帧级 MAE 表 + 分阶段分析图
Tue
P4.1-P4.2: 封装 + 离线 action consistency
"离线指标达标/不达标" 判断
Wed
P4.3: 半闭环逐维度替换实验
6 个配置的成功率对比表
Thu
P4.4-P4.5: 全闭环 + 诊断分析
"闭环成功率 = X%" + 失败模式报告
Fri
P5.1-P5.2: 根据诊断选修复策略并执行
student v1 or 修改计划

Week 2 末的交付标准：
[] Student v0 离线帧级 MAE 全部达标
[] 半闭环实验完成，识别出最敏感维度
[] 全闭环成功率数字（目标 ≥ 80%）
[] 如未达标：已制定并开始执行修复方案
[] 闭环诊断 log 和失败模式分析完成


---

5.9 最终说明

为什么不推翻 teacher 重训：

1. teacher 已经 89% 成功率，是经过 103 分钟训练验证的可用资产
2. teacher actor 只有 2.4 MB，推理 < 0.1 ms，边缘部署无压力
3. teacher 的 obs 定义清晰，15 维中 8 维是 easy states，只需补 7 维
4. 直接 image → action 蒸馏丢失了 teacher actor 的泛化能力和可解释性
5. 如果 student 观测估计足够好，teacher actor 的成功率（89%）直接继承
  
唯一需要重构 teacher 的情况：
- student 的观测噪声导致 teacher 行为剧烈振荡（teacher 对噪声不鲁棒）
- 这时的修复路径是：在仿真中对 teacher 的 15d obs 添加高斯噪声（模拟 student 误差），微调 teacher 几百个 iteration（不是从零训练），使其适应有噪声的观测
  
这条路线的最大优势是每一步都可验证、可回滚、可解释。你永远知道"是感知错了还是决策错了"。