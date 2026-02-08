# 参数修改指南：托盘缩放与安装同步

本文专门回答“托盘缩放参数是否在 `env_cfg.py` 中，以及修改后要做什么操作”的问题，
并给出可执行的流程与注意事项。

## 结论（简明版）

- 托盘缩放参数 **就在** `env_cfg.py` 中：`pallet_cfg.spawn.scale=(1.8, 1.8, 1.8)`
- 修改 `env_cfg.py` 后 **必须** 重新执行安装脚本：
  - `forklift_pallet_insert_lift_project/scripts/install_into_isaaclab.sh`
- 原因：训练时使用的是 **IsaacLab 安装目录**中的副本，而不是源目录

## 为什么必须重新安装？

IsaacLab 的训练脚本通过固定路径加载任务：

```
IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/forklift_pallet_insert_lift/
```

`forklift_pallet_insert_lift_project/` 只是“源目录”。
安装脚本的作用是把源目录的代码 **复制** 到 IsaacLab 的固定路径。
所以修改源代码后，不运行安装脚本，训练仍然会读取旧版本。

## 托盘缩放参数在哪里？

文件：`forklift_pallet_insert_lift_project/isaaclab_patch/source/.../env_cfg.py`

关键字段：

```python
scale=(1.8, 1.8, 1.8)
```

## 修改托盘缩放时的联动参数

缩放不仅影响几何尺寸，还会影响插入判定、初始高度与奖励分布。建议同步检查：

1. `pallet_cfg.spawn.scale`
   托盘整体缩放比例（核心参数）

2. `pallet_depth_m`
   托盘深度（应按缩放比例线性更新）

3. `pallet_cfg.init_state.pos[2]`
   托盘初始高度（避免穿地或悬空）

4. `robot_cfg.init_state.pos[0]`
   叉车初始距离（与托盘尺寸匹配）

5. `d_far / d_close`
   距离自适应阈值（托盘变大/变小后可能需要调整）

## 完整操作步骤（修改 → 安装 → 训练）

1) 修改源代码  
编辑：  
`forklift_pallet_insert_lift_project/isaaclab_patch/source/.../env_cfg.py`

2) 重新安装到 IsaacLab  
执行：  
```
bash forklift_pallet_insert_lift_project/scripts/install_into_isaaclab.sh /path/to/IsaacLab
```

如果你的 IsaacLab 在本机默认路径：  
`/home/uniubi/projects/forklift_sim/IsaacLab`，可直接替换为该路径。

3) 重新训练 / 运行验证  
进入 IsaacLab 目录后执行训练或验证脚本：
```
cd /path/to/IsaacLab
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Forklift-PalletInsertLift-Direct-v0
```

## 验证建议（可选）

- 运行任务并观察插入是否顺畅、托盘是否与货叉匹配
- 若托盘过大/过小，优先检查 `pallet_depth_m` 与 `init_state.pos`
- 如插入判定异常，检查 `env.py` 内基于 `pallet_depth_m` 的插入逻辑
