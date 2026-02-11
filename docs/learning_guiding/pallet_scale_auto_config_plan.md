# 托盘缩放自动联动配置（待执行计划）

> 状态：待执行  
> 记录时间：2026-02-08

## 概述

在 `env_cfg.py` 内引入单一可修改的托盘缩放配置项，派生计算所有关联数值（含质量、距离阈值、重置范围等），并同步更新 `env.py` 中的硬编码值与注释/文档。

## 目标

在 `env_cfg.py` 内只保留一个可修改的 `pallet_scale`，其余关联参数全部通过公式自动计算。同时将 `env.py` 中的硬编码重置范围也纳入配置体系。

## 可行性

已验证现有代码中 `@configclass` 支持 class-level 引用：
- `sim = SimulationCfg(dt=1/120, render_interval=decimation)` 引用了同类的 `decimation`
- `max_time_s = episode_length_s` 引用了同类的 `episode_length_s`

因此 `pallet_scale` 定义在前，后续字段引用它是安全的。

## 实施步骤

### 1) 定义缩放配置项与基准常量

在 `env_cfg.py` 类顶部，`decimation` 之前，新增：

```python
# ===== 托盘缩放（唯一需要手动修改的参数）=====
pallet_scale: float = 1.6

# 基准值（scale=1.8 时的参数，用于按比例计算）
_BASE_SCALE: float = 1.8
_SR: float = pallet_scale / _BASE_SCALE  # scale ratio
```

注意：现有代码中无下划线前缀字段先例。如果 `@configclass` 对下划线前缀有特殊处理（如跳过序列化），需要改用无下划线命名（如 `base_scale` / `scale_ratio`）。将在静态验证步骤中确认。

### 2) 将 env_cfg.py 中 7 项关联参数改为公式

**几何参数：**
- `pallet_depth_m = 2.16 * _SR`（原始深度 1.2m x pallet_scale，等价于 2.16 x _SR）
- `pallet_cfg.spawn.scale = (pallet_scale, pallet_scale, pallet_scale)`
- `pallet_cfg.init_state.pos = (0.0, 0.0, 0.15 * _SR)`（只缩放 z）
- `pallet_cfg.mass_props.mass = 45.0 * (_SR ** 3)`（按体积缩放）

**距离参数：**
- `robot_cfg.init_state.pos = (-3.5 * _SR, 0.0, 0.03)`（只缩放 x，z 是离地间隙不变）
- `d_far = 2.6 * _SR`
- `d_close = 1.1 * _SR`
- `d_safe_m = 0.7 * _SR`

### 3) 新增重置范围配置项（解决 env.py 硬编码问题）

在 `env_cfg.py` 中新增：

```python
# 重置时叉车初始位姿随机范围（随 pallet_scale 自动缩放）
reset_x_min: float = -4.0 * _SR
reset_x_max: float = -2.5 * _SR
reset_y_half: float = 0.6   # y 范围 [-0.6, 0.6]，与缩放无关
```

### 4) 修改 env.py 使用配置项替换硬编码

在 `env.py` 的 `_reset_idx` 方法中：

```python
# 原：x = sample_uniform(-4.0, -2.5, ...)
x = sample_uniform(self.cfg.reset_x_min, self.cfg.reset_x_max, (len(env_ids), 1), device=self.device)
# 原：y = sample_uniform(-0.6, 0.6, ...)
y = sample_uniform(-self.cfg.reset_y_half, self.cfg.reset_y_half, (len(env_ids), 1), device=self.device)
```

### 5) 更新注释与文档

**env_cfg.py 注释：**
- 顶部 S1.0h docstring 中"托盘缩放 4.0x -> 1.8x"改为"由 pallet_scale 配置"
- 托盘几何参数区块注释更新
- 托盘配置块注释更新（移除"修改后需同步更新"的提示，改为"由 pallet_scale 自动计算"）

**env.py 注释：**
- `_reset_idx` 中"适配 1.8x 缩放"改为"由 cfg.reset_x_min/max 控制"

**docs/learning_guiding/parameter_modification_guide.md：**
- 更新说明：现在只需改 `pallet_scale`，无需手动同步其他参数
- 保留"修改后需重新执行 install_into_isaaclab.sh"的提醒

## 完整改动清单

- `env_cfg.py`：新增 pallet_scale/_BASE_SCALE/_SR，修改 7 项参数为公式，新增 3 项 reset 配置，更新注释
- `env.py`：`_reset_idx` 中 2 行硬编码改为读取 cfg，更新注释
- `parameter_modification_guide.md`：更新文档说明

## 待办事项

- [ ] 在 env_cfg.py 顶部定义 pallet_scale / _BASE_SCALE / _SR 及基准常量
- [ ] 将 env_cfg.py 中 7 项关联参数改为公式计算（pallet_depth_m / d_far / d_close / d_safe_m / pos / mass / scale）
- [ ] 在 env_cfg.py 新增 reset_x_min/max 和 reset_y_half 配置项并随 scale 自动计算
- [ ] 修改 env.py _reset_idx 使用 cfg 中的 reset 范围替换硬编码
- [ ] 更新 env_cfg.py / env.py 中所有 1.8x 硬编码注释
- [ ] 更新 docs/learning_guiding/parameter_modification_guide.md 反映新机制
- [ ] 静态验证 - 实例化 cfg 打印所有计算值确认公式正确
- [ ] 运行时验证 - 用已有 verify 脚本测试完整流程

## 验证方案

### 静态验证（修改后立即执行）

编写一段临时 Python 代码（在 env_cfg.py 末尾或单独脚本），实例化配置并打印所有计算值：

```python
if __name__ == "__main__":
    cfg = ForkliftPalletInsertLiftEnvCfg()
    print(f"pallet_scale     = {cfg.pallet_scale}")
    print(f"pallet_depth_m   = {cfg.pallet_depth_m:.4f}  (期望: 1.2 * {cfg.pallet_scale} = {1.2 * cfg.pallet_scale:.4f})")
    print(f"pallet mass      = {cfg.pallet_cfg.spawn.mass_props.mass:.2f}")
    print(f"pallet scale     = {cfg.pallet_cfg.spawn.scale}")
    print(f"pallet init z    = {cfg.pallet_cfg.init_state.pos[2]:.4f}")
    print(f"robot init x     = {cfg.robot_cfg.init_state.pos[0]:.4f}")
    print(f"d_far            = {cfg.d_far:.4f}")
    print(f"d_close          = {cfg.d_close:.4f}")
    print(f"d_safe_m         = {cfg.d_safe_m:.4f}")
    print(f"reset_x_min      = {cfg.reset_x_min:.4f}")
    print(f"reset_x_max      = {cfg.reset_x_max:.4f}")
    print(f"reset_y_half     = {cfg.reset_y_half:.4f}")
```

验证点：
- 所有值非 None / 非零
- `pallet_depth_m` = `1.2 * pallet_scale`
- `pallet_cfg.spawn.scale` 三个分量均等于 `pallet_scale`
- `mass` 按体积比缩放
- `d_far > d_close > 0`
- `reset_x_min < reset_x_max < 0`

### 运行时验证（安装到 IsaacLab 后执行）

项目已有完整验证脚本 `scripts/verify_forklift_insert_lift.py`，可测试：
- 环境初始化（资产加载、关节识别）
- 货叉尖端计算（`_compute_fork_tip`）
- 托盘前部坐标（`_pallet_front_x`）
- 接近、对齐、插入、举升全流程

执行步骤：
```bash
# 1. 安装到 IsaacLab
bash forklift_pallet_insert_lift_project/scripts/install_into_isaaclab.sh /path/to/IsaacLab

# 2. 自动测试
cd /path/to/IsaacLab
./isaaclab.sh -p ../scripts/verify_forklift_insert_lift.py --headless

# 3. 手动测试（可视化）
./isaaclab.sh -p ../scripts/verify_forklift_insert_lift.py --manual
```

重点观察：
- 托盘大小是否与货叉匹配（1.6x 下插入孔宽度 ~365mm vs 货叉 ~394mm，可能偏紧）
- 插入深度计算是否正常（insert_depth > 0）
- 重置后叉车初始位置是否在合理范围

### 风险点与回退

- 如果 1.6x 缩放导致插入孔宽度不够（365mm < 394mm），可能需要回退到 >= 1.75x
- 如果 `@configclass` 不支持下划线前缀字段，改用 `base_scale` / `scale_ratio` 命名
