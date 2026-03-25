# `exp8.3` B0 baseline `seed 43` 启动故障 debug 总结

> 日期: `2026-03-25`
> 分支: `exp/exp8_3_clean_insert_hold`
> 相关实验: `B0 baseline`, `near-field`, `256x256`, `64 envs`, `50 iter`
> 相关 run:
> - 失败日志: `logs/20260325_152215_train_exp83_b0_seed43_iter50_256cam.log`
> - 修复后 smoke: `logs/20260325_161309_sanity_check_exp83_b0_seed43_groundfix.log`
> - 修复后正式补跑: `logs/20260325_161431_train_exp83_b0_seed43_iter50_256cam.log`

---

## 1. 问题现象

这次 `B0 baseline 3 seeds x 50 iter` 批次没有完整跑完。

- `seed 42` 正常完成。
- `seed 43` 没有进入训练迭代。
- 批次在 `seed 43` 处中断，导致 `seed 44` 当时未开始。

失败时最显著的现象有两个：

1. `scene creation` 异常慢  
   日志里记录:
   - `Time taken for scene creation : 1141.796193 seconds`

2. 场景创建结束后立即在 ground plane 绑定物理材质时崩溃  
   核心报错:

```text
Stage.GetPrimAtPath(Stage, NoneType)
```

对应调用链大意是：

- `env._setup_scene()`
- `spawn_ground_plane("/World/ground", cfg=self.cfg.ground_cfg)`
- `bind_physics_material(collision_prim_path, ...)`
- `collision_prim_path` 实际为 `None`

---

## 2. 初步判断

这个故障看起来像是 `seed 43` 导致，但实际并不是 PPO 随机种子影响了环境物理逻辑。

更合理的解释是：

- 当前 `ground_cfg` 使用的是 Isaac 默认的 **远程 ground USD**。
- 这次远程 ground 资源解析异常或返回了不完整结构。
- `spawn_ground_plane()` 内部没有处理“没找到 `Plane` 子 prim”的边界情况。
- 后续 `bind_physics_material()` 收到了 `None`，于是触发了 `Stage.GetPrimAtPath(Stage, NoneType)`。

也就是说，**根因是 ground plane 资源加载/解析链路不稳，而不是 seed 43 本身“学坏了”或配置有分支行为。**

---

## 3. 证据链

### 3.1 同配置下 `seed 42` 正常

对照 `seed 42` 的同口径 B0 日志可以看到：

- `Time taken for scene creation : 3.575577 seconds`

说明：

- 这套训练配置本身可以正常启动；
- 失败并不是 reward / PPO / obs group 配错；
- 问题更像是场景资源层面的偶发启动故障。

### 3.2 `seed 43` 失败点明确落在 ground plane

失败日志里，叉车和托盘已经完成了大部分模板环境初始化：

- pallet 物理属性修复已执行
- lift joint drive 覆盖已执行

之后才在地面创建处崩掉。  
这说明：

- 故障不是托盘物理修复引起的；
- 也不是 camera mount 检查引起的；
- 主要问题集中在 `ground_cfg -> spawn_ground_plane` 这一段。

### 3.3 `scene creation` 超长卡顿与远程资源强相关

当前 `ground_cfg.usd_path` 指向：

```text
https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac/Environments/Grid/default_environment.usd
```

这不是本地文件。  
因此即便最终能成功，也会引入以下风险：

- 网络/远端内容服务抖动
- 资源结构临时异常
- 长时间卡在场景创建阶段
- 上游工具函数没有处理空 prim 的边界 bug

---

## 4. 修复思路

目标不是去改 PPO、奖励或批处理脚本，而是让环境在 ground plane 这一步**变得稳健且可离线运行**。

修复策略分两层：

1. **预判远程 ground USD 不可靠时，直接跳过默认远程地面**
   - 如果 `ground_cfg.usd_path` 不是可验证的本地文件，就不再尝试走默认远程 grid plane。

2. **本地回退为一个简单的 mesh ground plane**
   - 直接在 `/World/ground` 下创建一个本地 box mesh 地面；
   - 保留碰撞；
   - 继续绑定 physics material；
   - 保留地面颜色配置。

这样做的核心收益是：

- 不再依赖远程默认 ground USD；
- 避免 `spawn_ground_plane()` 内部空 prim 路径 bug；
- 大幅降低启动抖动；
- 即使后续再次遇到上游空值异常，也可以自动 fallback。

---

## 5. 代码修改

修改文件：

- `forklift_pallet_insert_lift_project/isaaclab_patch/source/isaaclab_tasks/isaaclab_tasks/direct/forklift_pallet_insert_lift/env.py`

新增了两层逻辑：

1. `_spawn_ground_plane_with_fallback(...)`
   - 优先尝试原本的 `spawn_ground_plane`
   - 若路径不是本地文件，直接走本地 fallback
   - 若命中 `Stage.GetPrimAtPath(Stage, NoneType)` 这类可恢复异常，也走 fallback

2. `_spawn_local_ground_plane(...)`
   - 用本地 mesh 创建一个简单地面
   - 应用 visual material 和 physics material

并把 `_setup_scene()` 中原本的：

```python
spawn_ground_plane(prim_path="/World/ground", cfg=self.cfg.ground_cfg)
```

替换为：

```python
_spawn_ground_plane_with_fallback(prim_path="/World/ground", cfg=self.cfg.ground_cfg)
```

---

## 6. 修复验证

### 6.1 smoke 验证通过

修复后，我先做了一个 `seed 43` 的 `1 iter` 启动 smoke：

- 日志: `logs/20260325_161309_sanity_check_exp83_b0_seed43_groundfix.log`

关键结果：

- 检测到 ground USD 不是本地文件，直接走 fallback
- 成功创建本地 mesh 地面
- `Time taken for scene creation : 1.044440 seconds`
- 已进入 `Learning iteration 0/1`

这说明：

- 修复不是“避免报错但卡更久”
- 而是真正把场景创建恢复到了正常速度量级

### 6.2 正式补跑验证通过

随后按原始 B0 口径启动正式补跑：

- 日志: `logs/20260325_161431_train_exp83_b0_seed43_iter50_256cam.log`

关键结果：

- `Time taken for scene creation : 1.072777 seconds`
- 已进入 `Learning iteration 0/50`
- `Total timesteps: 4096` 已出现

说明修复不仅在 smoke 下有效，在正式训练配置下也有效。

---

## 7. 当前结论

这次 `seed 43` 中断的结论可以明确写成：

1. 不是 PPO seed 导致的训练逻辑异常。
2. 是环境启动阶段对 **远程 ground USD** 的依赖触发了不稳定性。
3. IsaacLab 默认 `spawn_ground_plane()` 在“未找到有效 collision plane”时存在空值边界问题。
4. 用本地 mesh ground plane 回退后，启动恢复正常，且 `scene creation` 从 `1141s` 下降到约 `1s`。

---

## 8. 对后续实验的影响

这次修复的直接影响是：

- `B0 baseline` 的 `seed 43/44` 可以继续补跑；
- 后续同类 headless 训练不会再被远程 ground 资源偶发卡死；
- 批处理脚本的稳定性明显提升；
- 之后如果再遇到启动类问题，优先排查 forklift USD 资源、贴图缺失、相机或其他远程资源，而不是先怀疑 seed。

---

## 9. 后续建议

1. 后续训练环境尽量避免依赖远程基础场景资源，尤其是地面这类完全可以本地替代的资产。
2. 若以后需要保留更复杂的默认 ground 外观，可以考虑把对应 USD 固化到仓库本地，再显式走本地路径。
3. 这次修复解决的是 **ground plane 启动崩溃**，不是 forklift USD 的贴图缺失问题；日志里依然会看到若干贴图找不到的 hydra 报错，但它们目前不阻塞训练进入迭代。
4. 等 `seed 43/44` 补跑结束后，再统一更新 `B0 baseline` 的多 seed 结果对照。
