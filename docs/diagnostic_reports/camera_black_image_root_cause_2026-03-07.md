# 相机黑图问题根因分析与修复

**日期：** 2026-03-07  
**问题背景：** 在 `ForkliftPalletInsertLift` 环境中，已确认相机观测链路存在，`obs["policy"]["image"]` 可以正常拿到张量，但多轮测试中输出图像长期偏黑。此前尝试过调相机位姿、提高分辨率、增强光照、前台抓日志等手段，均未从根本上解决问题。

---

## 一、现象复盘

排查过程中确认了以下现象：

1. **相机观测链路存在**
   - `obs["policy"]["image"]` 可以稳定返回。
   - `TiledCamera` 的 `rgb` / `rgba` 输出也存在，说明不是“相机没创建成功”。

2. **输出图长期接近全黑**
   - 64x64 默认配置下，图像统计约为：
     - `mean ≈ 0.04`
     - `max ≈ 0.09 ~ 0.18`
   - 图像空间分布几乎均匀，仅有极少数亮点，说明不是“正常场景但曝光低”，而是大部分像素根本没有看到有效场景内容。

3. **分辨率问题不是黑图主因**
   - 之前已确认环境中同时存在：
     - `cfg.camera_width / cfg.camera_height`
     - `cfg.tiled_camera.width / cfg.tiled_camera.height`
   - 只改后者会被前者默认值 `64` 影响。
   - 这一点解释了“320 设置未真正生效”，但并不能解释为什么默认 64x64 会长期近黑。

4. **渲染日志存在纹理错误，但不是根因**
   - 日志中长期出现：
     - `unknown PNG chunk type`
     - `FOF_Map_Palette_A_D.tga` / `FOF_Map_Palette_A_N.tga` 缺失
   - 这些问题会影响材质质量，但不会单独导致整幅图几乎全黑。

---

## 二、关键诊断过程

为了定位黑图根因，做了三类关键验证。

### 1. 验证相机是否真的有输出

通过最小诊断脚本确认：

- `obs["policy"]["image"]` 正常返回 `(1, 3, 64, 64)`；
- `TiledCamera.data.output["rgb"]` 为 `(1, 64, 64, 3)`；
- 输出数据不是全 0，也不是 NaN/Inf。

结论：**不是“相机链路断了”，而是“相机看到了错误的位置/方向”。**

### 2. 验证渲染管线

日志中出现：

- `DLSS increasing input dimensions: Render resolution of (37, 37) is below minimal input resolution of 300.`

这说明低分辨率下 DLSS 会把内部渲染分辨率进一步压低到 37x37，确实会影响画质；但即使禁用/调整相关设置，黑图现象仍然存在，说明 **DLSS 不是根因，只是次要画质问题**。

### 3. 直接检查相机世界坐标

这是最终定位根因的关键一步。

原配置中：

- `camera_pos_local = (0.8, 0.0, 1.7)`

按设计语义，这本应表示：

- 向前 0.8m
- 向上 1.7m

但实际诊断到的相机世界位置为：

- `camera WORLD pos = (-3.4920, 0.0000, 0.0470)`

而机器人根位置约为：

- `robot root_pos_w = (-3.5000, 0.0000, 0.0280)`

也就是说，相机相对于车体实际只偏移了约：

- `x ≈ 0.008m`
- `z ≈ 0.017m`

这与预期的 0.8m / 1.7m 正好相差 **100 倍**。

---

## 三、根因确认

根因是 **USD 单位制与相机 offset 单位不一致**。

`forklift_c.usd` 使用的是 **cm 单位**，而 `TiledCameraCfg.OffsetCfg.pos` 是在挂载 prim 的**局部坐标系单位**中生效的。  
因此原来配置的：

```python
camera_pos_local = (0.8, 0.0, 1.7)
```

并不会被解释为：

- 0.8m
- 1.7m

而是会被解释为：

- 0.8cm
- 1.7cm

换算后就是：

- 0.008m
- 0.017m

于是相机实际上被挂在了**机器人内部、接近地面的位置**，几乎只能看到黑暗的模型内部和局部遮挡，因此长期输出“近黑图”。

---

## 四、修复方案

### 1. 相机偏移量改为 cm 单位

将 `camera_pos_local` 从米制语义值改为与 USD 一致的 cm 值：

```python
camera_pos_local = (80.0, 0.0, 170.0)
```

对应物理含义仍然是：

- 向前 0.8m
- 向上 1.7m

同时在配置注释中明确写明：

- `forklift_c.usd` 使用 cm 单位
- `TiledCamera offset` 在 prim 局部坐标系中应用
- 因此这里的值必须写成 cm

### 2. 同步修复两处代码

本次同步更新了：

- `IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/forklift_pallet_insert_lift/env_cfg.py`
- `forklift_pallet_insert_lift_project/isaaclab_patch/source/isaaclab_tasks/isaaclab_tasks/direct/forklift_pallet_insert_lift/env_cfg.py`

避免后续再次出现“patch 目录已改，但实际运行源码未改”的偏差。

### 3. 渲染配置补充说明

排查过程中还观察到低分辨率相机下 DLSS 会把内部渲染分辨率压到 `37x37`。  
这不是黑图根因，但会影响画质，后续若继续做 video-e2e 训练，建议：

- 在低分辨率相机场景下禁用 DLSS，或
- 直接使用更高原始分辨率（例如 320x320），并同时设置：
  - `cfg.camera_width / cfg.camera_height`
  - `cfg.tiled_camera.width / cfg.tiled_camera.height`

---

## 五、修复后验证结果

修复后重新验证，得到如下结果：

1. **相机世界位置恢复正常**
   - 修复后：
     - `camera WORLD pos = (-2.7000, 0.0000, 1.7300)`
   - 这与“前方 0.8m、上方 1.7m”的设计预期一致。

2. **图像亮度和内容显著恢复**
   - 修复前：
     - `mean ≈ 0.04`
     - `>10% 亮度像素占比 ≈ 0.2%`
   - 修复后：
     - `mean ≈ 0.69`
     - `max ≈ 0.93`
     - `std ≈ 0.36`
     - `>10% 亮度像素占比 ≈ 83.8%`

3. **最终结论**
   - 相机观测链路本身一直是通的；
   - 黑图的根本原因不是“没渲染出来”，而是**相机挂载位姿在 cm 单位资产上被缩小了 100 倍**；
   - 修复后已经可以稳定产出可见 RGB 图像。

---

## 六、经验教训

这次问题的关键教训是：

1. **相机 offset 的单位必须和挂载 prim 所在 USD 的局部单位一致。**
2. **不要只看配置语义，要直接检查相机的世界坐标。**
3. **“图像不全黑”不等于“相机位置正确”，必须结合均值、方差、像素分布一起判断。**
4. **当项目同时存在运行源码目录和 patch 目录时，必须先确认实际 import 的模块路径。**

---

## 七、当前结论

**已证实：**

- 相机黑图的根因是 `camera_pos_local` 在 cm 单位 USD 上被缩小了 100 倍；
- 修复后 64x64 相机已能稳定输出可见 RGB；
- `obs["policy"]["image"]` 链路正常。

**仍可后续优化：**

- 320x320 分辨率下的稳定性与耗时；
- 低分辨率场景是否需要彻底禁用 DLSS；
- 远端资产纹理缺失（`FOF_Map_Palette_A_*.tga`）对最终视觉质量的影响。
