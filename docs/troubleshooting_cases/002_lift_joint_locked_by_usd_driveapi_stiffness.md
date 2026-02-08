# 问题：lift_joint 被 USD DriveAPI stiffness 锁死，无法响应速度控制

## 问题描述

**现象**：
- `lift_joint` 位置始终为 `0.0000m`，无论如何设置速度目标
- `set_joint_velocity_target()` 完全无效，关节纹丝不动
- 验证脚本显示：预期速度 `0.125 m/s`，实际速度 `≈ 0 rad/s`
- Isaac Lab 的 `ImplicitActuatorCfg` 显示 `stiffness=0.0`，看起来配置正确

**影响**：
- 叉车无法举升货叉/托盘，RL 训练中 lift 相关奖励始终为 0
- 即使 approach + insert 阶段完成，也无法进入 lift 阶段

## 排查过程

### 1. 检查 Isaac Lab 执行器配置

```python
# env_cfg.py 中的配置
"lift": ImplicitActuatorCfg(
    joint_names_expr=["lift_joint"],
    velocity_limit=1.0,
    effort_limit=5000.0,
    stiffness=0.0,       # ← 看起来是速度控制模式
    damping=1000.0,
),
```

验证脚本输出确认：
```
stiffness: tensor([[0.]], device='cuda:0')   ← Isaac Lab 认为 stiffness=0
damping: tensor([[1000.]], device='cuda:0')
effort_limit: tensor([[5000.]], device='cuda:0')
```

**结论**：Isaac Lab 层面配置正确，但关节不动 → 问题不在 Python 配置层。

### 2. 检查 USD DriveAPI 原始参数（根因）

通过 `UsdPhysics.DriveAPI.Get(prim, "linear")` 直接读取 `forklift_c.usd` 中 `lift_joint` 的 DriveAPI：

```
USD DriveAPI 原始参数:
  stiffness = 100000.0    ← 极高的位置保持刚度！
  damping   = 10000.0
  maxForce  = (默认)
```

**这就是根因**：
- USD 中 `stiffness=100000.0` 表示 **位置控制模式**，PhysX 以 10 万 N/m 的力把关节保持在目标位置（默认为 0）
- Isaac Lab 的 `ImplicitActuatorCfg(stiffness=0.0)` **未能覆盖** USD 中的 DriveAPI 值
- 结果：PhysX 的位置保持力（100000 × 位移）远大于速度控制产生的力，关节被锁死

### 3. 为什么 ImplicitActuatorCfg 没有覆盖 USD DriveAPI？

Isaac Lab 的 `ImplicitActuator` 通过 PhysX Articulation API 设置关节驱动参数。但在某些版本或配置下：

1. **Isaac Lab 可能只设置 PhysX runtime API**，而不修改 USD stage 上的 DriveAPI 属性
2. **PhysX 在场景构建时** bake USD DriveAPI 参数作为初始值
3. 如果 Isaac Lab 的覆盖发生在 PhysX 读取之后，或者覆盖方式不同于 PhysX 期望的方式，原始值就会保留

Isaac Lab 还会发出以下警告，暗示参数传递机制正在变化：
```
[Warning] The <ImplicitActuatorCfg> object has a value for 'effort_limit'.
  This parameter will be removed in the future. To set the effort limit, please use 'effort_limit_sim' instead.
[Warning] The <ImplicitActuatorCfg> object has a value for 'velocity_limit'.
  Previously, although this value was specified, it was not getting used by implicit actuators.
```

## 根本原因

```
USD 文件 (forklift_c.usd)
  └── lift_joint → DriveAPI("linear")
        stiffness = 100000.0   ← PhysX 在场景构建时读取这个值
        damping   = 10000.0

Isaac Lab (env_cfg.py)
  └── ImplicitActuatorCfg
        stiffness = 0.0        ← 可能未能覆盖 USD DriveAPI 的值
        damping   = 1000.0

PhysX 运行时实际使用的值:
  stiffness = 100000.0         ← USD 原始值（未被覆盖）
  → 关节被锁在位置 0，velocity target 无效
```

**核心原则**：PhysX 关节驱动参数的权威来源是 **USD DriveAPI**，而非 Isaac Lab 的 Python 配置。当两者冲突时，USD DriveAPI 的值可能胜出。

## 解决方案

### 方案 A：在 `_setup_scene()` 中直接修改 USD DriveAPI（推荐）

在场景克隆（`clone_environments()`）**之前**修改 USD stage 上的 DriveAPI 参数：

```python
def _setup_scene(self):
    self.robot = Articulation(self.cfg.robot_cfg)
    # ... 其他资产加载 ...
    
    # ★ 在 clone 之前修复 lift_joint DriveAPI
    self._fix_lift_joint_drive()
    
    self.scene.clone_environments(copy_from_source=False)
    # ...

def _fix_lift_joint_drive(self):
    from pxr import Usd, UsdPhysics
    stage = self.sim.stage
    robot_prim = stage.GetPrimAtPath("/World/envs/env_0/Robot")
    
    for prim in Usd.PrimRange(robot_prim):
        if "lift" not in prim.GetName().lower():
            continue
        drive_api = UsdPhysics.DriveAPI.Get(prim, "linear")
        if not drive_api:
            continue
        
        # 覆盖为速度控制模式
        drive_api.GetStiffnessAttr().Set(0.0)       # 无位置保持
        drive_api.GetDampingAttr().Set(10000.0)      # 速度控制阻尼
        drive_api.GetMaxForceAttr().Set(50000.0)     # 最大力 50kN
        return
```

**关键**：必须在 `clone_environments()` 之前调用，这样修改会自动继承到所有克隆环境。

### 方案 B：预处理 USD 文件（最彻底）

直接在 USD 文件中修改 `lift_joint` 的 DriveAPI 参数，保存为新的 USD 文件：

```python
# 一次性脚本，修改后保存
stage = Usd.Stage.Open("forklift_c.usd")
# ... 修改 DriveAPI ...
stage.GetRootLayer().Save()
```

然后在 `env_cfg.py` 中引用修改后的 USD 文件。

### 方案 C：修改 env_cfg.py 使用 effort_limit_sim（待验证）

根据 Isaac Lab 的警告提示，使用新 API：

```python
"lift": ImplicitActuatorCfg(
    joint_names_expr=["lift_joint"],
    velocity_limit_sim=1.0,       # 新参数名
    effort_limit_sim=5000.0,      # 新参数名
    stiffness=0.0,
    damping=1000.0,
),
```

## 修改文件

- `env.py` — 添加 `_fix_lift_joint_drive()` 方法
- 调用位置应在 `_setup_scene()` 中，`clone_environments()` 之前

## 验证方法

运行验证脚本观察：
1. `lift_pos` 在 lift 测试阶段应该持续增长（从 0.0 向上）
2. `fork_tip_z` 应该跟随 `lift_pos` 增长
3. 如果有托盘接触，`pallet_z` 也应该跟随上升

```bash
cd /home/uniubi/projects/forklift_sim/IsaacLab
PYTHONUNBUFFERED=1 ./isaaclab.sh -p ../scripts/verify_forklift_insert_lift.py --headless
```

## 相关问题

- **001_forklift_low_speed_despite_max_throttle.md**：轮子 `effort_limit` 不足导致低速，同样是物理参数不匹配问题
- **pallet_physics_optimization_2026-02-05.md**：记录了 lift `effort_limit` 从 500→5000、`stiffness` 从 2000→0 的修改，但该修改仅在 `ImplicitActuatorCfg` 层面，未触及 USD DriveAPI
- **PhysX Cooking 时机问题**：与碰撞体的 `convexDecomposition` 在运行时修改不生效是同一类问题——PhysX 在场景构建时 bake 参数，之后修改 USD 不会被重新读取

## 经验总结

1. **Isaac Lab `ImplicitActuatorCfg` 不一定能覆盖 USD DriveAPI 参数**——当关节不响应时，第一步应该检查 USD 原始 DriveAPI 值
2. **检查顺序**：USD DriveAPI stiffness → `ImplicitActuatorCfg` 是否生效 → effort_limit 是否足够
3. **PhysX 参数修改的时机至关重要**：必须在 `clone_environments()` / `sim.reset()` 之前完成
4. **`stiffness >> 0` 表示位置控制**，此时 `set_joint_velocity_target()` 几乎无效，因为位置保持力远大于速度控制力
5. 调试时用 `UsdPhysics.DriveAPI.Get(prim, "linear"/"angular")` 直接读取 PhysX 看到的真实值，不要只看 Isaac Lab 的 Python 配置

---

*创建日期：2026-02-08*
*关联版本：S1.0h*
*状态：修复方案已确定，待验证修复时机（需移至 `_setup_scene()`）*
