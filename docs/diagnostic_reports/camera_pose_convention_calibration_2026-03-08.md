# 相机位姿校准与坐标系约定修复——完整排查历程

**日期：** 2026-03-08  
**关联前置报告：** `camera_black_image_root_cause_2026-03-07.md`（相机黑图 cm/m 单位根因）  
**最终成果：** 成功在叉车 `body` 上以 `world` 坐标约定挂载了一台固定相机，俯视角拍到了货叉插入托盘的完整过程，画面正且清晰。

---

## 一、背景

在 3 月 7 日修复了相机黑图根因（cm/m 单位缩放 100 倍）之后，相机终于能出亮图了。但接下来发现：**画面里虽然不黑了，却始终拍不到货叉工作区**。本文记录了从"出亮图"到"画面正确"的完整排查过程，包括我（AI agent）犯的每一次错误、用户的每一次关键反馈、以及最终的修复方案。

---

## 二、完整排查时间线

### 第 1 阶段：黑图修复后首次出图 → 用户指出"根本不朝货叉"

#### 我做了什么

修复 cm/m 单位问题后，相机从 `(-3.49, 0, 0.05)`（车体内部地面）移到了 `(-2.70, 0, 1.73)`（车体前方 0.8m、上方 1.7m），图像亮度从 mean=0.04 提升到 mean=0.69。

我以为问题解决了，保存了 `/tmp/camera_verify_final.png` 并给用户报告"修复成功"。

#### 用户反馈

> **"你看一下吧，我看了，这个根本就不是朝向货叉的"**

#### 实际问题

用户说得完全对。我去分析了那张图，发现：

- 画面中间是**大面积亮背景**（天空/远处场景）
- 左侧只有一条很窄的**黑色竖向结构**（门架/车体边缘的一小部分）
- 定量指标：`center mean ≈ 222`（几乎全亮），`left mean ≈ 48`（暗结构只在左边缘）
- 这说明相机只是**擦着车体边缘拍出去了**，主视角完全不在货叉前方

#### 我的错误

修复黑图时，我只关注了"像素是否变亮"这一个指标，**没有验证画面内容是否正确**。"图像不黑了"不等于"相机视角对了"。

---

### 第 2 阶段：用户要求建立成功标准 → 写了评估脚本

#### 用户反馈

> **"继续做这件事，你需要想考虑下成功的标准是什么？然后分步来实现验证修正"**

#### 我做了什么

1. **定义了量化成功标准：**
   - 货叉尖端必须落在画面下半部
   - 托盘入口/工作区必须落在画面中部
   - 图像不能只是大面积天空/背景

2. **写了 `camera_pose_sweep.py`：** 8 个候选位姿的批量扫描脚本，每个位姿会自动投影货叉尖端和托盘入口到图像上，并计算评分。

3. **写了 `camera_eval_one.py`：** 单候选位姿评估脚本，支持参数化配置。

#### 遇到的问题

- **Isaac Sim 在同一进程中反复创建/销毁环境会卡死**：`camera_pose_sweep.py` 跑完第一个候选后就挂住了（Jetson Orin 上渲染管线无法正确重置）。
- **解决方案：** 改成"每个候选位姿单独起一个进程"的方式，虽然慢但稳定。

---

### 第 3 阶段：创建参数化评估脚本 camera_eval.py → 初始测试

将 `camera_eval_one.py` 发展为功能更完整的 `scripts/tools/camera_eval.py`，增加了：

- 视频录制（mp4）
- 关键帧导出（frame_start/mid/end.png）
- 叉车运动控制（先前进 60 步，再举升）
- 标记方块（红色=货叉尖端，蓝色=托盘）
- 背景稳定性指标

**初始测试 test1/test2/test3**：相机参数 `pos=(80, 0, 170) pitch=-20° FOV=75°`

结果：叉车只跑了很少步数（约 10 步），画面基本没有变化。

---

### 第 4 阶段：4 个候选位姿测试 (cand1-cand4) → 用户指出看不清内容

跑了 4 组参数：

| 候选 | 位置 (cm) | Pitch | FOV |
|------|-----------|-------|-----|
| cand1 | (80, 0, 200) | -30° | 75° |
| cand2 | (80, 0, 220) | -40° | 75° |
| cand3 | (100, 0, 220) | -45° | 75° |
| cand4 | (60, 0, 250) | -50° | 75° |

#### 用户反馈

> **"从 cand1 中看不出来画面里面究竟是啥啊，你现在相机的视场角是多少"**

#### 实际问题

FOV 只有 75°，在 320×320 分辨率下看到的范围太窄，很难判断画面里到底是什么。

#### 我的错误

我应该一开始就用更大的 FOV 来做探索（比如 90-120°），先确认相机方向对不对，再收窄 FOV。这是调参顺序的问题。

---

### 第 5 阶段：加大 FOV + 增加标记方块 → 用户指出两个核心问题

把 FOV 加到 120°，增加了红蓝标记方块，跑了 3 组广角测试：

| 候选 | 位置 (cm) | Pitch | FOV |
|------|-----------|-------|-----|
| cand_wide | (80, 0, 200) | -30° | 120° |
| cand_wide2 | (150, 0, 250) | -45° | 120° |
| cand_wide3 | (20, 0, 280) | -50° | 120° |

#### 用户反馈（这条是整个排查过程最关键的反馈）

> **"start 和 end 的画面基本是一样的啊，cand_wide 是只看到了门架子，cand_wide2 是只看到了托盘，但 start 和 end 几乎一样，是你没有让叉车动吗？cand_wide3 里面看到的是远处的东西，也是 start 和 end 一样。另外为啥相机的画面我感觉是应该旋转 90 度呢"**

#### 用户指出了两个问题

1. **叉车没动：** start 和 end 画面几乎一样
2. **画面旋转了 90 度：** 地面不在画面底部，像是横着拍的

#### 实际原因

1. **步数不足 + 初始位置太远：** 当时测试只跑了 10 步（0.33 秒），叉车完全没时间前进。加上初始位置在 `(-3.5, 0, 0.03)`，离托盘 3.5m。
2. **坐标系约定错误：** `TiledCameraCfg.OffsetCfg` 的 `convention="ros"` 意味着相机坐标系是 +Z 前方、-Y 上方，而 `body` prim 用的是仿真世界坐标系（+X 前方、+Z 上方），两者存在 90° 旋转差异。

#### 我的错误

- 步数 10 步太少是个低级错误——0.33 秒连轮子都转不了多少
- `convention` 参数的影响我完全没有理解，一直在 `ros` 约定下调 Euler 角，是在错误的坐标系里做事

---

### 第 6 阶段：修复运动 + Roll 旋转实验 → 走弯路

#### 修复运动

- 步数从 10 改为 **200 步**（约 6.7 秒）
- 叉车初始位置从 `(-3.5, 0, 0.03)` 改为 **`(-2.0, 0, 0.03)`**
- 动作策略：前 60 步前进，之后举升

#### 尝试用 Roll 修正旋转（走弯路）

为了修正 90° 旋转，我在 `ros` 约定下尝试了 3 组 Roll 角：

| 测试 | Roll | Pitch | 结果 |
|------|------|-------|------|
| close_rot_0 | 0° | -45° | 画面旋转 90°（原始问题） |
| close_rot_90 | 90° | -45° | 画面方向变了，但前后也扭了 |
| close_rot_n90 | -90° | -45° | 同上，另一个方向 |

**这条路是错的。** Roll 虽然能旋转画面，但同时把前后方向也搞乱了。用 Roll 来补偿坐标系约定差异是治标不治本的歪招。

---

### 第 7 阶段：发现 convention 参数 → 切换到 "world" 约定

#### 关键发现

Isaac Lab 的 `TiledCameraCfg.OffsetCfg` 支持三种坐标约定：

| 约定 | 前方轴 | 上方轴 | 适用场景 |
|------|--------|--------|----------|
| `ros` | +Z | -Y | ROS 标准相机模型 |
| `opengl` | -Z | +Y | USD/OpenGL 原生相机 |
| **`world`** | **+X** | **+Z** | **与仿真世界坐标系一致** |

`body` prim 的局部坐标系是 +X 前方、+Z 上方，所以必须用 `world` 约定，否则会出现 90° 旋转。

#### 修改 env_cfg.py

```python
# 修改前
convention="ros",     # +Z 前方、-Y 上方 → 与 body 不一致 → 画面旋转 90°

# 修改后
convention="world",   # +X 前方、+Z 上方 → 与 body 一致 → 画面正确
```

**同时注意：** 切换到 `world` 约定后，pitch 的正负含义也变了。在 `ros` 下 pitch=-45° 是向下看，在 `world` 下需要用 **pitch=+45°** 才是向下看。

---

### 第 8 阶段：world_conv_test → 用户确认成功

用修正后的参数跑了 `world_conv_test`：

```
pos=(130, 0, 250) cm  →  前方 1.3m、上方 2.5m
pitch=+45°           →  向下看 45°（world 约定）
FOV=90°
convention=world
steps=150
```

输出到 `/home/uniubi/projects/forklift_sim/outputs/camera_eval/world_conv_test/`。

#### 用户反馈

> **"这次正确了啊。非常棒"**

画面特征：
- 地面在画面底部，远处背景在画面顶部（**方向正确**）
- 能清晰看到货叉、托盘、地面网格
- 叉车从远处开到托盘处 → 插入 → 举升的**完整过程都拍到了**
- start/mid/end 三帧有明显的内容变化（**叉车确实在动**）

---

### 第 9 阶段：用户要求读取图片验证 → 我自己检查确认

在 `world_conv_test` 之前，用户还有一次反馈：

> **"你自己读取下图片看看吧。。我看了都不对"**

这是在 `close_rot_*` 系列实验期间。我读取了图片后确认画面确实不对——`close_rot_0` 的画面只有几个像素有内容（几乎全黑/全白），`close_rot_n90` 虽然有内容但方向不正确。

这次自检让我意识到不能只看 metrics.txt 的数值就下结论，必须真正看图片本身的内容。

---

## 三、所有犯错记录汇总

| 序号 | 错误 | 错误类型 | 用户纠正 | 正确做法 |
|------|------|----------|----------|----------|
| 1 | 黑图修复后只看亮度不看内容，宣称"修复成功" | 验证不充分 | "这个根本就不是朝向货叉的" | 必须同时验证亮度 + 画面内容 + 场景语义 |
| 2 | 候选位姿用 75° 窄 FOV，看不清内容 | 调参顺序错误 | "看不出来画面里面究竟是啥" | 先用大 FOV（90-120°）确认方向，再收窄 |
| 3 | 测试只跑 10 步，叉车根本没动 | 测试设计缺陷 | "start 和 end 的画面基本是一样的" | 至少 100-200 步，并打印 robot 位置 |
| 4 | 在 `ros` 约定下用 Roll 补偿 90° 旋转 | 治标不治本 | （我自己检查图片后发现不对） | 切换到正确的 `convention="world"` |
| 5 | 没理解 `convention` 参数的含义就直接调角度 | 知识盲区 | "为啥相机的画面我感觉是应该旋转 90 度呢" | 先搞清坐标系约定，再做位姿调整 |
| 6 | 第一轮 sweep 在同一进程内反复创建环境导致卡死 | 技术盲区 | （自己发现） | Isaac Sim 每次只能跑一个环境进程，串行启动 |
| 7 | 叉车初始位置 -3.5m 太远，320×320 下托盘太小 | 测试条件不合理 | （隐含在 start=end 反馈中） | 调试时把叉车拉近到 -2.0m |

---

## 四、用户所有关键反馈汇总

| 时间点 | 用户原话 | 触发了什么修正 |
|--------|----------|----------------|
| 黑图修复后 | "你看一下吧，我看了，这个根本就不是朝向货叉的" | 从"亮度修复"转向"视角修复" |
| 提出方法论 | "继续做这件事，你需要想考虑下成功的标准是什么？然后分步来实现验证修正" | 建立了量化成功标准 + 参数化评估脚本 |
| cand1 测试后 | "从 cand1 中看不出来画面里面究竟是啥啊，你现在相机的视场角是多少" | 把 FOV 从 75° 加到 120°，增加标记方块 |
| cand_wide 系列后 | "start 和 end 的画面基本是一样的...另外为啥相机的画面我感觉是应该旋转 90 度呢" | 修复步数 + 初始位置 + 发现 convention 问题 |
| close_rot 系列时 | "你自己读取下图片看看吧。。我看了都不对" | 我开始真正检查图片而非只看指标 |
| world_conv_test 后 | "这次正确了啊。非常棒" | 确认最终方案正确 |

---

## 五、最终修复内容

### 5.1 env_cfg.py

```python
# convention 从 "ros" 改为 "world"
convention="world",  # +X 前方、+Z 上方，与 body 局部坐标系一致
```

### 5.2 env.py `_setup_scene()` 中的运行时参数覆盖

在创建 `TiledCamera` 之前，动态覆盖所有相关子字段：

```python
self.cfg.tiled_camera.prim_path = f"/World/envs/env_.*/Robot/{mount_body}/Camera"
self.cfg.tiled_camera.offset.pos = self.cfg.camera_pos_local
self.cfg.tiled_camera.width = self.cfg.camera_width
self.cfg.tiled_camera.height = self.cfg.camera_height

# 根据 hfov 和 horizontal_aperture 反算 focal_length
hfov_rad = math.radians(self.cfg.camera_hfov_deg)
horizontal_aperture = self.cfg.tiled_camera.spawn.horizontal_aperture
focal_length = horizontal_aperture / (2.0 * math.tan(hfov_rad / 2.0))
self.cfg.tiled_camera.spawn.focal_length = focal_length

# 将 euler 角（RPY）转为四元数
roll_deg, pitch_deg, yaw_deg = self.cfg.camera_rpy_local_deg
# ... euler → quaternion 转换 ...
self.cfg.tiled_camera.offset.rot = (w, x, y, z)
```

**原因：** `@configclass` 类似 dataclass，类级别的 `tiled_camera = TiledCameraCfg(...)` 在类定义时就固化了内部值。实例化后修改 `self.cfg.camera_pos_local` 不会自动传播到 `self.cfg.tiled_camera.offset.pos`，必须显式赋值。

### 5.3 最终相机参数

```python
camera_mount_body = "body"
camera_pos_local = (130.0, 0.0, 250.0)   # cm：前方 1.3m，上方 2.5m
camera_rpy_local_deg = (0.0, 45.0, 0.0)  # pitch=+45° 向下看（world 约定）
camera_hfov_deg = 90.0
camera_width = 320
camera_height = 320
convention = "world"
```

---

## 六、验证结果

验证输出位于 `/home/uniubi/projects/forklift_sim/outputs/camera_eval/world_conv_test/`：

| 文件 | 内容 |
|------|------|
| `frame_start.png` | 初始帧：俯视地面，前方可见蓝色标记方块和托盘 |
| `frame_mid.png` | 中间帧：叉车已前进，托盘在画面中变大 |
| `frame_end.png` | 末尾帧：货叉已插入托盘并举升 |
| `video.mp4` | 完整 150 帧视频，可见叉车 → 接近托盘 → 插入 → 举升全过程 |
| `metrics.txt` | 背景稳定性指标 5.86（说明画面有显著变化 = 叉车确实在动） |

---

## 七、全部实验清单

以下是整个排查过程中跑过的所有实验，按时间顺序排列：

| 实验名 | 参数摘要 | 发现/结论 |
|--------|----------|-----------|
| test1/2/3 | pos=(80,0,170) pitch=-20° FOV=75° 10步 | 步数太少，画面几乎没变化 |
| cand1 | pos=(80,0,200) pitch=-30° FOV=75° | 用户看不清内容 → FOV 太窄 |
| cand2 | pos=(80,0,220) pitch=-40° FOV=75° | 同上 |
| cand3 | pos=(100,0,220) pitch=-45° FOV=75° | 同上 |
| cand4 | pos=(60,0,250) pitch=-50° FOV=75° | 同上 |
| cand_wide | pos=(80,0,200) pitch=-30° FOV=120° | 只看到门架 → 视角不对 |
| cand_wide2 | pos=(150,0,250) pitch=-45° FOV=120° | 只看到托盘 → start=end 没动 |
| cand_wide3 | pos=(20,0,280) pitch=-50° FOV=120° | 看到远处 → start=end 没动 |
| rot_0 | pos=(130,0,250) pitch=-45° roll=0° FOV=90° | 画面旋转 90° |
| rot_90 | pos=(130,0,250) pitch=-45° roll=90° FOV=90° | Roll 补偿方向不对 |
| rot_n90 | pos=(130,0,250) pitch=-45° roll=-90° FOV=90° | 同上 |
| close_rot_0 | 同 rot_0，步数增加到 200，叉车位置拉近 | 画面仍然旋转 90° |
| close_rot_90 | 同上 + roll=90° | 方向仍不正确 |
| close_rot_n90 | 同上 + roll=-90° | 方向仍不正确 |
| **world_conv_test** | **pos=(130,0,250) pitch=+45° FOV=90° convention=world** | **成功！画面正确** |
| mast_1 | pos=(50,0,250) pitch=+45° FOV=90° convention=world | 更靠近门架的备选位姿 |

---

## 八、核心经验总结

### 8.1 坐标系约定是根因

Isaac Lab `TiledCameraCfg.OffsetCfg` 的 `convention` 参数决定了 Euler 角和旋转的参考系。`body` prim 用的是世界坐标系（+X 前、+Z 上），所以必须用 `convention="world"`，不要用 `ros`。

### 8.2 调试相机的正确顺序

1. **先用大 FOV（90-120°）+ 长步数（150+步）** 确认画面方向
2. **加彩色标记方块** 提供明确的视觉锚点
3. **打印 robot 位置** 验证叉车确实在运动
4. 确认方向正确后再收窄 FOV 和微调位置

### 8.3 不能只看数值指标

`mean=0.69, std=0.36` 看起来很好，但画面可能完全是天空。必须同时看**图片本身**和**语义内容**。

### 8.4 @configclass 嵌套对象不会自动同步

修改 `self.cfg.camera_pos_local` 不会传播到 `self.cfg.tiled_camera.offset.pos`，必须在 `_setup_scene()` 中显式赋值。

### 8.5 forklift_c.usd 使用 cm 单位

所有 `camera_pos_local` 的值都是厘米，不是米。这一点在前一天的黑图排查中已经确认。

### 8.6 Isaac Sim 在 Jetson Orin 上每次只能起一个环境进程

不能在同一进程里反复 `gym.make()` + `env.close()`，会导致渲染管线挂死。必须串行启动独立进程。

### 8.7 用户反馈是最有效的诊断信号

整个排查中最关键的转折点都来自用户的直觉判断：
- "根本不朝货叉" → 停止在黑图修复上庆祝
- "看不出来画面里面究竟是啥" → 加大 FOV + 标记
- "start 和 end 一样" + "旋转 90 度" → 找到 convention 根因
- "你自己看看图片" → 停止纯数值分析，开始真正看图

---

## 九、工具与脚本

| 脚本 | 位置 | 用途 |
|------|------|------|
| `camera_eval.py` | `scripts/tools/camera_eval.py` | 参数化相机评估工具，支持自定义位置/角度/FOV/分辨率，自动导出视频+关键帧+指标 |

**使用示例：**

```bash
cd /home/uniubi/projects/forklift_sim/IsaacLab
unset CONDA_PREFIX && unset CONDA_DEFAULT_ENV
./isaaclab.sh -p /home/uniubi/projects/forklift_sim/scripts/tools/camera_eval.py \
  --headless --enable_cameras \
  --cam-name my_test \
  --cam-x 130 --cam-y 0 --cam-z 250 \
  --pitch-deg 45 --hfov-deg 90 \
  --steps 150
```

输出到 `/home/uniubi/projects/forklift_sim/outputs/camera_eval/my_test/`。
