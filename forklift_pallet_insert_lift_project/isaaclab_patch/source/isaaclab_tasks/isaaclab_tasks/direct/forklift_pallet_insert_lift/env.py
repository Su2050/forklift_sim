"""Forklift Pallet Insert+Lift 环境实现（DirectRLEnv）。
by sull

本文件是任务的核心逻辑，包含：
- 任务环境类 `ForkliftPalletInsertLiftEnv`
- 观测、奖励、终止与重置逻辑
- 叉车/托盘物理修复与诊断工具

调用流程（Isaac Lab 直接环境）：
1) `_setup_scene()`：创建资产、设置物理、克隆环境
2) `_reset_idx()`：按 env_ids 重置初始状态
3) 每步循环：
   - `_pre_physics_step(actions)`  缓存动作
   - `_apply_action()`             将动作写入仿真
   - `_get_observations()`         计算观测
   - `_get_rewards()`              计算奖励
   - `_get_dones()`                判断终止/超时

坐标约定：
- 机器人朝向以 yaw（Z 轴旋转）描述
- 叉车从 -X 方向接近托盘，插入深度沿 +X 增加
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import spawn_ground_plane
from isaaclab.utils.math import sample_uniform

from .env_cfg import ForkliftPalletInsertLiftEnvCfg


def _quat_to_yaw(q: torch.Tensor) -> torch.Tensor:
    """Extract yaw from quaternion (w,x,y,z). Assumes Z-up and mainly yaw rotations."""
    w, x, y, z = q.unbind(-1)
    # yaw (z-axis rotation)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return torch.atan2(siny_cosp, cosy_cosp)


def _yaw_to_mat2(yaw: torch.Tensor) -> torch.Tensor:
    """2x2 rotation matrix for yaw (world->robot frame uses -yaw)."""
    c = torch.cos(yaw)
    s = torch.sin(yaw)
    # [[c, -s],[s,c]]
    return torch.stack(
        [torch.stack([c, -s], dim=-1), torch.stack([s, c], dim=-1)],
        dim=-2,
    )


def smoothstep(x: torch.Tensor) -> torch.Tensor:
    """Hermite smoothstep: 0→1 with zero-derivative at both ends."""
    x = torch.clamp(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def _force_pallet_dynamic(stage, pallet_prim_path: str):
    """确保托盘是动态刚体（不是 kinematic）
    
    遍历托盘 prim 及其所有子 prim，将所有 RigidBodyAPI 设置为非 kinematic。
    """
    from pxr import Usd, UsdPhysics
    
    root_prim = stage.GetPrimAtPath(pallet_prim_path)
    if not root_prim.IsValid():
        print(f"[警告] 找不到托盘 prim: {pallet_prim_path}")
        return
    
    # 遍历托盘及其所有子 prim（使用 Usd.PrimRange 兼容所有 USD 版本）
    prims_to_process = list(Usd.PrimRange(root_prim))
    
    for prim in prims_to_process:
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            rb_api = UsdPhysics.RigidBodyAPI(prim)
            # 确保是动态刚体
            rb_api.GetRigidBodyEnabledAttr().Set(True)
            rb_api.GetKinematicEnabledAttr().Set(False)
            print(f"[信息] 已设置 {prim.GetPath()} 为动态刚体")


def _force_pallet_convex_decomposition(stage, pallet_prim_path: str):
    """为托盘设置凸分解碰撞体，使 pocket 可以被插入
    
    遍历托盘 prim 及其所有子 prim，为所有带有碰撞体的 prim 设置凸分解。
    """
    from pxr import Usd, UsdPhysics, PhysxSchema, UsdGeom
    
    root_prim = stage.GetPrimAtPath(pallet_prim_path)
    if not root_prim.IsValid():
        print(f"[警告] 找不到托盘 prim: {pallet_prim_path}")
        return
    
    # 遍历托盘及其所有子 prim（使用 Usd.PrimRange 兼容所有 USD 版本）
    prims_to_process = list(Usd.PrimRange(root_prim))
    
    applied_count = 0
    for prim in prims_to_process:
        # 检查是否有 Mesh 几何体或已有碰撞 API
        has_mesh = prim.IsA(UsdGeom.Mesh)
        has_collision = prim.HasAPI(UsdPhysics.CollisionAPI)
        
        if has_mesh or has_collision:
            # 设置碰撞近似为凸分解
            mesh_collision_api = UsdPhysics.MeshCollisionAPI.Apply(prim)
            mesh_collision_api.GetApproximationAttr().Set("convexDecomposition")
            
            # 凸分解参数
            convex_api = PhysxSchema.PhysxConvexDecompositionCollisionAPI.Apply(prim)
            convex_api.GetMaxConvexHullsAttr().Set(8)   # 最大凸体数（从32降到8，平衡精度与性能）
            convex_api.GetHullVertexLimitAttr().Set(64)  # 每个凸体最大顶点数
            applied_count += 1
    
    print(f"[信息] 凸分解已应用到 {applied_count} 个 prim")


class ForkliftPalletInsertLiftEnv(DirectRLEnv):
    cfg: ForkliftPalletInsertLiftEnvCfg

    def __init__(self, cfg: ForkliftPalletInsertLiftEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.robot: Articulation = self.scene.articulations["robot"]
        self.pallet: RigidObject = self.scene.rigid_objects["pallet"]

        # joint indices：将关节名字映射为索引，便于后续批量设置目标
        self._front_wheel_ids, _ = self.robot.find_joints(["left_front_wheel_joint", "right_front_wheel_joint"], preserve_order=True)
        self._back_wheel_ids, _ = self.robot.find_joints(["left_back_wheel_joint", "right_back_wheel_joint"], preserve_order=True)
        self._rotator_ids, _ = self.robot.find_joints(["left_rotator_joint", "right_rotator_joint"], preserve_order=True)
        # separate left/right rotator IDs for order-independent steering
        self._left_rotator_id, _ = self.robot.find_joints(["left_rotator_joint"], preserve_order=True)
        self._right_rotator_id, _ = self.robot.find_joints(["right_rotator_joint"], preserve_order=True)
        self._lift_id, _ = self.robot.find_joints(["lift_joint"], preserve_order=True)
        self._lift_id = self._lift_id[0]

        # ---- 基础缓存 ----
        # actions: 当前步动作缓存（归一化动作）
        # _last_insert_depth: 上一步插入深度（用于安全制动与奖励增量）
        # _fork_tip_z0: reset 时 fork tip 的基准高度
        # _hold_counter: 成功条件保持计数器
        # _lift_pos_target: lift 关节位置目标（位置控制）
        self.actions = torch.zeros((self.num_envs, self.cfg.action_space), device=self.device)
        self._last_insert_depth = torch.zeros((self.num_envs,), device=self.device)
        self._fork_tip_z0 = torch.zeros((self.num_envs,), device=self.device)
        # S1.0O-C2: 使用 float 以支持衰减（原 S1.0N 为 int32）
        self._hold_counter = torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device)
        self._lift_pos_target = torch.zeros((self.num_envs,), device=self.device)

        # ---- S1.0k: 势函数 shaping 缓存 ----
        self._is_first_step = torch.ones((self.num_envs,), dtype=torch.bool, device=self.device)
        self._last_phi_total = torch.zeros((self.num_envs,), device=self.device)
        self._last_lift_pos = torch.zeros((self.num_envs,), device=self.device)
        # S1.0L: milestones + early-stop buffers
        # S1.0M: 从 4 列扩展到 6 列（新增 fine_align + precise_align）
        # S1.0N: 从 6 列扩展到 7 列（新增 gate_align）
        self._milestone_flags = torch.zeros((self.num_envs, 7), dtype=torch.bool, device=self.device)
        # S1.0N: delta hold-align shaping 状态量
        self._prev_phi_align = torch.zeros((self.num_envs,), device=self.device)
        # S1.0O-A3: lift 进度 delta 势函数缓存
        self._prev_phi_lift_progress = torch.zeros((self.num_envs,), device=self.device)
        self._fly_counter = torch.zeros((self.num_envs,), dtype=torch.int32, device=self.device)
        self._stall_counter = torch.zeros((self.num_envs,), dtype=torch.int32, device=self.device)
        self._early_stop_fly = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        self._early_stop_stall = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        # S1.0Q Batch-3: dead-zone stuck detector
        self._dz_stuck_counter = torch.zeros((self.num_envs,), dtype=torch.int32, device=self.device)
        self._prev_y_err = torch.zeros((self.num_envs,), device=self.device)
        self._early_stop_dz_stuck = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        # S1.0Q Batch-4: penOnly 消融首次触发标记
        self._dz_stuck_fired = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        # S1.0Q: 死区撤退 shaping 状态量
        self._prev_insert_norm = torch.zeros((self.num_envs,), device=self.device)
        self._prev_in_dead_zone = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        # S1.0Q-A2v2: 撤退窗口缓冲（环形缓冲区）
        self._insert_norm_window = torch.zeros(
            (self.num_envs, self.cfg.retreat_window_size), device=self.device)
        self._window_ptr = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)
        self._window_filled = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        # S1.0Q: 横向精调 delta shaping 状态量
        self._prev_phi_lat = torch.zeros((self.num_envs,), device=self.device)
        # S1.0S Phase-2: 举升里程碑 flags (10cm, 20cm)
        self._milestone_lift_10cm = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        self._milestone_lift_20cm = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        # S1.0T: 高举升里程碑 flags (50cm, 75cm)
        self._milestone_lift_50cm = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        self._milestone_lift_75cm = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        # S1.0S Phase-R: 远场大横偏修正 delta shaping 状态量
        self._prev_y_err_far = torch.zeros((self.num_envs,), device=self.device)
        # S1.0S Phase-3: 全局进展停滞检测器
        self._global_stall_counter = torch.zeros((self.num_envs,), dtype=torch.int32, device=self.device)
        self._prev_phi_total_stall = torch.zeros((self.num_envs,), device=self.device)

        # S1.0z: Episode 级别成功率统计
        self._ep_success_count = 0
        self._ep_total_count = 0

        # ---- 派生常量 ----
        # S1.0h 修复：符号 + → -
        # _pallet_front_x 指向托盘 pocket 开口（近端面，-X 侧），不是远端面
        # 叉车从 -X 方向接近，insert_depth = tip_x - _pallet_front_x > 0 表示已插入
        self._pallet_front_x = self.cfg.pallet_cfg.init_state.pos[0] - self.cfg.pallet_depth_m * 0.5
        self._insert_thresh = self.cfg.insert_fraction * self.cfg.pallet_depth_m
        # 成功判定需要的 hold 步数
        ctrl_dt = self.cfg.sim.dt * self.cfg.decimation
        self._hold_steps = max(1, int(self.cfg.hold_time_s / ctrl_dt))

        # 便捷引用（从 PhysX view 手动刷新，因为 robot.data.joint_pos 在
        # Fabric clone 失败时不会被 scene.update() 刷新——所有值恒为 0）
        num_joints = len(self.robot.joint_names)
        self._joint_pos = torch.zeros((self.num_envs, num_joints), device=self.device)
        self._joint_vel = torch.zeros((self.num_envs, num_joints), device=self.device)

        # ---- fork tip 运动学偏移量（从 USD mesh 测量或回退到默认值） ----
        # body_pos_w 在 Fabric clone 失败或 body frame origin 重合时无法区分各 link，
        # 因此使用 root_pos + yaw-旋转的固定偏移 + lift_joint_pos 来估算 fork tip。
        self._fork_forward_offset, self._fork_z_base = self._measure_fork_offset_from_usd()

        # 注：_fix_lift_joint_drive() 已移到 _setup_scene() 中 clone_environments() 之前调用，
        # 确保 PhysX 在 sim.reset() 时 bake 到正确的 DriveAPI 参数（stiffness=200000, 位置控制）。

    def _fix_lift_joint_drive(self):
        """覆盖 lift_joint 的 USD DriveAPI 参数为位置控制模式。

        forklift_c.usd 原始 DriveAPI 设置了 stiffness=100000, damping=10000。
        Isaac Lab 的 ImplicitActuatorCfg 会在 sim.reset() 时覆盖 PhysX drive 参数，
        但为确保 clone 前 USD stage 上的值也一致（双保险），这里直接修改。

        logs32 验证可行的参数组合：stiffness=200000, damping=10000, maxForce=50000。
        修改必须在 clone_environments() 之前（模板环境），clone 后自动继承。
        """
        from pxr import Usd, UsdPhysics

        try:
            stage = self.sim.stage
            robot_prim_path = self.cfg.robot_cfg.prim_path.replace("env_.*", "env_0")
            robot_prim = stage.GetPrimAtPath(robot_prim_path)
            if not robot_prim.IsValid():
                print("[lift_drive] 无法找到 robot prim，跳过")
                return

            for prim in Usd.PrimRange(robot_prim):
                if "lift" not in prim.GetName().lower():
                    continue

                # 检查 linear DriveAPI（prismatic joint）
                drive_api = UsdPhysics.DriveAPI.Get(prim, "linear")
                if not drive_api:
                    continue

                # 原始值
                old_stiff = drive_api.GetStiffnessAttr().Get() if drive_api.GetStiffnessAttr() else "N/A"
                old_damp = drive_api.GetDampingAttr().Get() if drive_api.GetDampingAttr() else "N/A"
                old_force = drive_api.GetMaxForceAttr().Get() if drive_api.GetMaxForceAttr() else "N/A"

                # 覆盖为位置控制模式（logs32 验证值）
                drive_api.GetStiffnessAttr().Set(200000.0)    # 位置控制刚度
                drive_api.GetDampingAttr().Set(10000.0)       # 阻尼
                drive_api.GetMaxForceAttr().Set(50000.0)      # 最大力 50kN

                # 新值
                new_stiff = drive_api.GetStiffnessAttr().Get()
                new_damp = drive_api.GetDampingAttr().Get()
                new_force = drive_api.GetMaxForceAttr().Get()

                print(f"[lift_drive] 已覆盖 {prim.GetPath()} DriveAPI(linear):")
                print(f"  stiffness: {old_stiff} → {new_stiff}")
                print(f"  damping:   {old_damp} → {new_damp}")
                print(f"  maxForce:  {old_force} → {new_force}")
                return

            print("[lift_drive] 未找到 lift joint DriveAPI")
        except Exception as e:
            print(f"[lift_drive] 修复 lift drive 失败: {e}")

    def _measure_fork_offset_from_usd(self) -> tuple[float, float]:
        """从 USD mesh 数据测量 fork tip 相对于 articulation root 的前向偏移和基准 z 高度。

        遍历 robot prim 下所有名称含 'lift'/'fork' 的 body 的 mesh 子节点，
        计算所有顶点的 bounding box，取最大 X 作为前向偏移（假设 USD 中 +X = 前进方向）。
        如果无法测量，则回退到保守默认值。

        Returns:
            (fork_forward_offset, fork_z_base):
                fork_forward_offset — 从 root origin 到 fork tip 的前向距离（m）
                fork_z_base — fork tip 在 root frame 中的基准 z 高度（m），不含 lift_joint 位移
        """
        from pxr import Usd, UsdGeom
        import numpy as np

        DEFAULT_FORWARD = 1.5   # 保守默认值（m），略小于典型叉车货叉长度
        DEFAULT_Z_BASE = 0.10   # 保守默认值（m）

        try:
            stage = self.sim.stage
            robot_prim_path = self.cfg.robot_cfg.prim_path.replace("env_.*", "env_0")
            robot_prim = stage.GetPrimAtPath(robot_prim_path)
            if not robot_prim.IsValid():
                print(f"[fork_offset] 无法找到 robot prim: {robot_prim_path}，使用默认偏移")
                return DEFAULT_FORWARD, DEFAULT_Z_BASE

            # ---- 获取 USD 单位缩放（很多 USD 用 cm 而非 m） ----
            meters_per_unit = 1.0
            if stage.HasAuthoredMetadata("metersPerUnit"):
                meters_per_unit = stage.GetMetadata("metersPerUnit")
            elif UsdGeom.GetStageMetersPerUnit(stage) != 0.0:
                meters_per_unit = UsdGeom.GetStageMetersPerUnit(stage)
            print(f"[fork_offset] USD metersPerUnit = {meters_per_unit}")

            # robot root 的 world transform（用于将顶点转到 robot local frame）
            robot_xformable = UsdGeom.Xformable(robot_prim)
            robot_xform = robot_xformable.ComputeLocalToWorldTransform(0.0)
            robot_inv = robot_xform.GetInverse()

            # 收集所有 mesh 顶点（在 robot local frame 中，单位为 m）
            all_points = []
            fork_points = []  # 仅 lift/fork 相关 mesh 的顶点

            for prim in Usd.PrimRange(robot_prim):
                if not prim.IsA(UsdGeom.Mesh):
                    continue
                mesh = UsdGeom.Mesh(prim)
                pts_attr = mesh.GetPointsAttr()
                if pts_attr is None:
                    continue
                pts = pts_attr.Get()
                if pts is None or len(pts) == 0:
                    continue

                # local-to-world transform 已包含层级中的所有 scale
                xformable = UsdGeom.Xformable(prim)
                xform = xformable.ComputeLocalToWorldTransform(0.0)

                # 转到 robot local frame 后再乘以 metersPerUnit 转为 m
                is_fork = any(
                    kw in str(prim.GetPath()).lower()
                    for kw in ("lift", "fork", "tine")
                )
                for pt in pts:
                    world_pt = xform.Transform(pt)
                    local_pt = robot_inv.Transform(world_pt)
                    pt_m = [local_pt[0] * meters_per_unit,
                            local_pt[1] * meters_per_unit,
                            local_pt[2] * meters_per_unit]
                    all_points.append(pt_m)
                    if is_fork:
                        fork_points.append(pt_m)

            if not all_points:
                print("[fork_offset] 未找到任何 mesh 顶点，使用默认偏移")
                return DEFAULT_FORWARD, DEFAULT_Z_BASE

            all_arr = np.array(all_points)
            all_min = all_arr.min(axis=0)
            all_max = all_arr.max(axis=0)
            extent = all_max - all_min

            # ---- 自动检测单位：如果最大维度 > 10m，假设 cm 单位 ----
            unit_scale = 1.0
            if max(extent) > 10.0:
                unit_scale = 0.01  # cm → m
                all_arr *= unit_scale
                all_min = all_arr.min(axis=0)
                all_max = all_arr.max(axis=0)
                extent = all_max - all_min
                print(f"[fork_offset] 检测到 cm 单位（extent>{10.0}），自动转换 ×{unit_scale}")
                if fork_points:
                    fork_points = [[p[0]*unit_scale, p[1]*unit_scale, p[2]*unit_scale]
                                   for p in fork_points]

            print(f"[fork_offset] 模型范围(m): X[{all_min[0]:.4f}, {all_max[0]:.4f}], "
                  f"Y[{all_min[1]:.4f}, {all_max[1]:.4f}], Z[{all_min[2]:.4f}, {all_max[2]:.4f}]")

            if fork_points:
                fork_arr = np.array(fork_points)
                fork_forward = float(fork_arr[:, 0].max())
                fork_z = float(fork_arr[:, 2].min())
                print(f"[fork_offset] lift/fork mesh: forward={fork_forward:.4f}m, z_base={fork_z:.4f}m")
                if 0.3 < fork_forward < 5.0:
                    return fork_forward, max(fork_z, 0.0)
                else:
                    print(f"[fork_offset] 测量值不合理 ({fork_forward:.4f}m)，尝试整体前向")
            else:
                print("[fork_offset] 未找到 lift/fork mesh，尝试整体前向")

            # 回退：使用整体最大前向 X
            overall_forward = float(all_max[0])
            overall_z_min = float(all_min[2])
            print(f"[fork_offset] 整体 forward_max={overall_forward:.4f}m, z_min={overall_z_min:.4f}m")
            if 0.3 < overall_forward < 5.0:
                return overall_forward, max(overall_z_min, 0.0)

            print(f"[fork_offset] 整体值也不合理，使用默认值")
            return DEFAULT_FORWARD, DEFAULT_Z_BASE

        except Exception as e:
            print(f"[fork_offset] USD mesh 测量失败: {e}，使用默认偏移")
            return DEFAULT_FORWARD, DEFAULT_Z_BASE

    def _setup_pallet_physics(self):
        """在环境克隆前，强制设置托盘物理属性

        必须在 clone_environments() 之前调用，确保模板环境的设置能被克隆继承。

        S1.0h 修复：
        - 使用 Isaac Lab 官方 schemas.define_rigid_body_properties() 创建 RigidBodyAPI
          （之前手动 UsdPhysics.RigidBodyAPI.Apply() 在 Nucleus 引用层上不生效）
        - 若根 prim 失败，回退到子 prim 逐个尝试
        - 所有关键 print 加 flush=True，避免 nohup 缓冲导致日志丢失
        """
        from pxr import Usd, UsdPhysics, PhysxSchema, UsdGeom
        from isaaclab.sim import schemas as rl_schemas

        stage = self.sim.stage

        # 诊断：修改前的 USD 状态（只看 env_0）
        diag_pallet_path = self.cfg.pallet_cfg.prim_path.replace("env_.*", "env_0")
        self._log_pallet_usd(stage, diag_pallet_path, label="修改前")
        self._log_pallet_physx(label="修改前")

        # 只修改模板环境（env_0），让克隆继承
        pallet_path = self.cfg.pallet_cfg.prim_path.replace("env_.*", "env_0")
        root_prim = stage.GetPrimAtPath(pallet_path)
        if not root_prim.IsValid():
            print(f"[警告] 找不到托盘 prim: {pallet_path}", flush=True)
            return

        # ---- Step 1: 创建 RigidBodyAPI（多级回退策略） ----
        # Nucleus 的 pallet.usd 不含 RigidBodyAPI。Isaac Lab spawn 时调用的
        # modify_rigid_body_properties 只修改已有 API，不会创建新 API（返回 False）。
        # 使用 define_rigid_body_properties（先创建再修改）来解决。
        rigid_cfg = sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            kinematic_enabled=False,
            disable_gravity=False,
            max_depenetration_velocity=1.0,
        )

        # 方案 A：在根 prim 上调用 Isaac Lab 官方 define API
        try:
            rl_schemas.define_rigid_body_properties(pallet_path, rigid_cfg, stage)
            print(f"[信息] define_rigid_body_properties 已调用: {pallet_path}", flush=True)
        except Exception as e:
            print(f"[警告] define_rigid_body_properties 在根 prim 失败: {e}", flush=True)

        rb_ok = root_prim.HasAPI(UsdPhysics.RigidBodyAPI)
        print(f"[诊断] 根 prim HasAPI(RigidBodyAPI) = {rb_ok}", flush=True)

        # 方案 B：根 prim 失败，逐个尝试子 prim（Xform / Mesh）
        if not rb_ok:
            print("[信息] 根 prim 未成功，尝试子 prim...", flush=True)
            for child in Usd.PrimRange(root_prim):
                if child == root_prim:
                    continue
                child_path = str(child.GetPath())
                child_type = child.GetTypeName()
                try:
                    rl_schemas.define_rigid_body_properties(child_path, rigid_cfg, stage)
                    if child.HasAPI(UsdPhysics.RigidBodyAPI):
                        print(f"[信息] RigidBodyAPI 成功应用到子 prim: "
                              f"{child_path} (type={child_type})", flush=True)
                        rb_ok = True
                        break
                except Exception as e:
                    print(f"[诊断] 子 prim {child_path} 失败: {e}", flush=True)

        # 方案 C：全部失败，用低级 USD API 硬写并打印明确错误
        if not rb_ok:
            print("[警告] define API 均失败，尝试低级 USD API...", flush=True)
            UsdPhysics.RigidBodyAPI.Apply(root_prim)
            rb_api = UsdPhysics.RigidBodyAPI(root_prim)
            rb_api.GetRigidBodyEnabledAttr().Set(True)
            rb_api.GetKinematicEnabledAttr().Set(False)
            rb_ok = root_prim.HasAPI(UsdPhysics.RigidBodyAPI)
            if rb_ok:
                print(f"[信息] 低级 API 成功: {pallet_path}", flush=True)
            else:
                print(f"[错误] 所有方案均无法创建 RigidBodyAPI，训练将失败！", flush=True)

        # 注意：不在根 Xform prim 上添加 CollisionAPI。
        # 碰撞形状只需在子 Mesh prim 上，根 Xform 加碰撞会导致双重碰撞检测，
        # 严重拖慢仿真速度（947 vs 17000 steps/s）。

        # ---- Step 2: 遍历子 prim 设置凸分解碰撞体（跳过根 prim） ----
        prims_to_process = list(Usd.PrimRange(root_prim))
        for prim in prims_to_process:
            # 跳过根 prim，只处理子 prim
            if prim == root_prim:
                continue

            # 已有 RigidBodyAPI 的子 prim 也强制设为动态
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                rb = UsdPhysics.RigidBodyAPI(prim)
                rb.GetRigidBodyEnabledAttr().Set(True)
                rb.GetKinematicEnabledAttr().Set(False)

            # 为 Mesh / 已有碰撞的 prim 设置凸分解
            has_mesh = prim.IsA(UsdGeom.Mesh)
            has_collision = prim.HasAPI(UsdPhysics.CollisionAPI)

            if has_mesh or has_collision:
                # 确保有 CollisionAPI
                if not prim.HasAPI(UsdPhysics.CollisionAPI):
                    UsdPhysics.CollisionAPI.Apply(prim)
                mesh_collision_api = UsdPhysics.MeshCollisionAPI.Apply(prim)
                mesh_collision_api.GetApproximationAttr().Set("convexDecomposition")

                convex_api = PhysxSchema.PhysxConvexDecompositionCollisionAPI.Apply(prim)
                convex_api.GetMaxConvexHullsAttr().Set(8)  # 从32降到8，平衡精度与性能
                convex_api.GetHullVertexLimitAttr().Set(64)

        print("[信息] 托盘物理属性已设置完成（模板环境）", flush=True)
        # 诊断：修改后的 USD 状态（只看 env_0）
        self._log_pallet_usd(stage, diag_pallet_path, label="修改后")
        self._log_pallet_physx(label="修改后")

    def _log_pallet_usd(self, stage, pallet_path: str, label: str):
        """打印托盘 USD 物理属性诊断信息（仅用于排查问题）。"""
        from pxr import Usd, UsdPhysics, UsdGeom

        root_prim = stage.GetPrimAtPath(pallet_path)
        if not root_prim.IsValid():
            print(f"[诊断] {label} 找不到托盘 prim: {pallet_path}")
            return

        prims_to_process = list(Usd.PrimRange(root_prim))

        print("\n" + "=" * 60)
        print(f"[诊断] {label} 托盘 USD 状态: {pallet_path}")
        print(f"[诊断] prim 数量: {len(prims_to_process)}")

        rigid_body_count = 0
        collision_count = 0

        for prim in prims_to_process:
            prim_path = prim.GetPath()
            prim_type = prim.GetTypeName()

            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                rb_api = UsdPhysics.RigidBodyAPI(prim)
                rb_enabled = rb_api.GetRigidBodyEnabledAttr().Get()
                kinematic = rb_api.GetKinematicEnabledAttr().Get()
                print(
                    f"[诊断] RigidBody: {prim_path} type={prim_type} "
                    f"enabled={rb_enabled} kinematic={kinematic}"
                )
                rigid_body_count += 1

            has_collision = prim.HasAPI(UsdPhysics.CollisionAPI)
            has_mesh = prim.IsA(UsdGeom.Mesh)
            if has_collision or has_mesh:
                approx = None
                if prim.HasAPI(UsdPhysics.MeshCollisionAPI):
                    mesh_api = UsdPhysics.MeshCollisionAPI(prim)
                    approx = mesh_api.GetApproximationAttr().Get()
                print(
                    f"[诊断] Collision: {prim_path} type={prim_type} "
                    f"collision_api={has_collision} mesh={has_mesh} approx={approx}"
                )
                collision_count += 1

        if rigid_body_count == 0:
            print("[诊断] 未发现 RigidBodyAPI，托盘可能没有刚体属性")

        print(f"[诊断] RigidBody 数量: {rigid_body_count}")
        print(f"[诊断] Collision/Mesh 数量: {collision_count}")
        print("=" * 60 + "\n")

    def _log_pallet_physx(self, label: str):
        """打印 PhysX 运行时视图的基本信息（可用性诊断）。"""
        if not hasattr(self, "pallet"):
            return
        if not hasattr(self.pallet, "root_physx_view"):
            print(f"[诊断] {label} PhysX view 不可用（root_physx_view 缺失）")
            return

        view = self.pallet.root_physx_view
        count = getattr(view, "count", None)
        print(f"[诊断] {label} PhysX view 已初始化, count={count}")
        if hasattr(view, "get_kinematic_enabled"):
            try:
                kin = view.get_kinematic_enabled()
                print(f"[诊断] {label} PhysX kinematic: {kin}")
            except Exception as exc:
                print(f"[诊断] {label} PhysX kinematic 读取失败: {exc}")
        else:
            print(f"[诊断] {label} PhysX 无直接 kinematic 读取接口")

    # ---------------------------
    # Scene setup
    # ---------------------------
    def _setup_scene(self):
        """构建场景与资产。

        注意顺序：
        1) 创建资产（robot/pallet）
        2) 修改模板环境（env_0）的物理属性
        3) 在克隆前修复 lift joint DriveAPI
        4) 克隆环境并加入场景
        """
        # assets
        self.robot = Articulation(self.cfg.robot_cfg)
        self.pallet = RigidObject(self.cfg.pallet_cfg)

        # 在克隆之前修改模板环境（env_0）的托盘物理属性
        self._setup_pallet_physics()

        # 在克隆之前修复 lift_joint DriveAPI（USD 原始 stiffness=100000 会锁死关节）
        # 必须在 clone_environments() 之前，这样修改会继承到所有克隆环境；
        # 也必须在 sim.reset() 之前，这样 PhysX bake 时读到的就是 stiffness=0。
        self._fix_lift_joint_drive()

        # ground
        spawn_ground_plane(prim_path="/World/ground", cfg=self.cfg.ground_cfg)

        # clone envs
        self.scene.clone_environments(copy_from_source=False)

        # collision filtering (needed for CPU sim)
        # Note: `self.device` may be either a string ("cpu") or torch.device("cpu").
        if getattr(self.device, "type", str(self.device)) == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])

        # add to scene
        self.scene.articulations["robot"] = self.robot
        self.scene.rigid_objects["pallet"] = self.pallet

        # 注意：托盘物理属性在 _setup_scene() 中 clone 之前设置

        # lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    # ---------------------------
    # Actions
    # ---------------------------
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        # store normalized actions
        self.actions = torch.clamp(actions, -1.0, 1.0)

    def _apply_action(self) -> None:
        """将动作写入仿真。

        动作含义：
        - actions[:,0] 驱动（车轮角速度）
        - actions[:,1] 转向（前轮转角）
        - actions[:,2] 举升（lift 位置增量）
        """
        # decode actions
        drive = self.actions[:, 0] * self.cfg.wheel_speed_rad_s
        steer = self.actions[:, 1] * self.cfg.steer_angle_rad
        lift_v = self.actions[:, 2] * self.cfg.lift_speed_m_s

        # two-stage safety: if already inserted enough, suppress driving and let it lift
        inserted = self._last_insert_depth >= self._insert_thresh
        drive = torch.where(inserted, torch.zeros_like(drive), drive)
        steer = torch.where(inserted, torch.zeros_like(steer), steer)

        # set targets
        # wheels: velocity targets
        self.robot.set_joint_velocity_target(drive.unsqueeze(-1).repeat(1, len(self._front_wheel_ids)), joint_ids=self._front_wheel_ids)
        # back wheels follow (optional)
        self.robot.set_joint_velocity_target(drive.unsqueeze(-1).repeat(1, len(self._back_wheel_ids)), joint_ids=self._back_wheel_ids)

        # steering: position targets by joint name (order-independent)
        # left rotator needs SAME sign (verified by scripts/verify_joint_axes.py)
        # Previous assumption of mirrored axis was INCORRECT.
        steer_left = steer
        steer_right = steer
        self.robot.set_joint_position_target(steer_left.unsqueeze(-1), joint_ids=self._left_rotator_id)
        self.robot.set_joint_position_target(steer_right.unsqueeze(-1), joint_ids=self._right_rotator_id)

        # lift: position target (accumulated per substep)
        # logs32 验证：stiffness=200000 + set_joint_position_target 是唯一可行的 drive 控制方式
        # 注意：_apply_action() 每 env step 被调用 decimation 次，必须用 sim.dt 而非 step_dt
        self._lift_pos_target += lift_v * self.cfg.sim.dt
        self._lift_pos_target = torch.clamp(self._lift_pos_target, 0.0, 2.0)
        self.robot.set_joint_position_target(
            self._lift_pos_target.unsqueeze(-1), joint_ids=[self._lift_id]
        )

        # write to sim
        self.robot.write_data_to_sim()

    # ---------------------------
    # Observations / Rewards / Dones
    # ---------------------------
    def _compute_fork_tip(self) -> torch.Tensor:
        """运动学方法估算 fork tip 世界位置。

        使用 root_pos + yaw 旋转的固定前向偏移 + lift_joint 位移来计算。
        这比 body_pos_w 方法更可靠，因为 body_pos_w 在 Fabric clone 失败或
        body frame origin 重合时（如 forklift_c.usd）无法区分各 link。

        前向偏移量 (_fork_forward_offset) 在 __init__ 中从 USD mesh 数据测量，
        或回退到保守默认值。

        Returns:
            tip: (N, 3) tensor — fork tip 的世界坐标
        """
        root_pos = self.robot.data.root_pos_w   # (N, 3)
        yaw = _quat_to_yaw(self.robot.data.root_quat_w)  # (N,)
        lift_pos = self._joint_pos[:, self._lift_id]      # (N,) lift joint 位移

        cos_yaw = torch.cos(yaw)
        sin_yaw = torch.sin(yaw)

        tip_x = root_pos[:, 0] + self._fork_forward_offset * cos_yaw
        tip_y = root_pos[:, 1] + self._fork_forward_offset * sin_yaw
        tip_z = root_pos[:, 2] + self._fork_z_base + lift_pos

        return torch.stack([tip_x, tip_y, tip_z], dim=-1)

    def _compute_phi_align(self, env_ids: torch.Tensor | None = None) -> torch.Tensor:
        """计算对齐势函数 phi_align（与 _get_rewards 同源几何）。

        用于 delta hold-align shaping 的状态缓存初始化（_reset_idx）和
        每步 delta 计算（_get_rewards）。

        Args:
            env_ids: 如果提供，只计算这些 env 的值；否则计算全部。

        Returns:
            phi_align: (len(env_ids),) 或 (N,) — 对齐势函数值 [0, 1]
        """
        if env_ids is not None:
            root_pos = self.robot.data.root_pos_w[env_ids]
            robot_yaw = _quat_to_yaw(self.robot.data.root_quat_w[env_ids])
            pallet_pos = self.pallet.data.root_pos_w[env_ids]
            pallet_yaw = _quat_to_yaw(self.pallet.data.root_quat_w[env_ids])
        else:
            root_pos = self.robot.data.root_pos_w
            robot_yaw = _quat_to_yaw(self.robot.data.root_quat_w)
            pallet_pos = self.pallet.data.root_pos_w
            pallet_yaw = _quat_to_yaw(self.pallet.data.root_quat_w)

        cp = torch.cos(pallet_yaw)
        sp = torch.sin(pallet_yaw)
        v_lat = torch.stack([-sp, cp], dim=-1)

        rel_robot = root_pos[:, :2] - pallet_pos[:, :2]
        y_err = torch.abs(torch.sum(rel_robot * v_lat, dim=-1))

        yaw_err = torch.atan2(
            torch.sin(robot_yaw - pallet_yaw),
            torch.cos(robot_yaw - pallet_yaw),
        )
        yaw_err_deg = torch.abs(yaw_err) * (180.0 / math.pi)

        phi_align = (
            torch.exp(-(y_err / self.cfg.hold_align_sigma_y) ** 2)
            * torch.exp(-(yaw_err_deg / self.cfg.hold_align_sigma_yaw) ** 2)
        )
        return phi_align

    def _get_observations(self) -> dict[str, torch.Tensor]:
        """构造观测向量（长度=15，S1.0N: 13→15）。

        顺序如下：
        1) 机器人到托盘的相对位置（机器人坐标系）d_xy_r (2)
        2) 偏航差 dyaw 的 cos/sin (2)
        3) 机器人线速度（机器人坐标系）v_xy_r (2)
        4) 机器人偏航角速度 yaw_rate (1)
        5) lift 关节位置与速度 (2)
        6) 插入深度归一化 insert_norm (1)
        7) 当前动作 actions (3)
        8) S1.0N: pallet center line frame 横向误差 y_err_obs (1, 带符号, clip [-1,1])
        9) S1.0N: pallet center line frame 偏航误差 yaw_err_obs (1, 带符号, clip [-1,1])
        """
        # ---- 从 PhysX view 刷新关节数据 ----
        # robot.data.joint_pos 在 Fabric clone 失败时不更新（始终为 0），
        # 但 root_physx_view.get_dof_positions() 能正确返回值。
        self._joint_pos[:] = self.robot.root_physx_view.get_dof_positions()
        self._joint_vel[:] = self.robot.root_physx_view.get_dof_velocities()

        # ---- PhysX 直读诊断 ----
        # 条件1: 前 5 步（初始化阶段）
        # 条件2: lift_target > 0 的前 10 次（举升阶段）
        _diag_init = self.common_step_counter < 5
        _diag_lift = (self._lift_pos_target[0] > 0.001
                      and not hasattr(self, '_diag_lift_count'))
        _diag_lift_ongoing = (self._lift_pos_target[0] > 0.001
                              and hasattr(self, '_diag_lift_count')
                              and self._diag_lift_count < 10
                              and self.common_step_counter % 20 == 0)
        if _diag_init or _diag_lift or _diag_lift_ongoing:
            if _diag_lift and not hasattr(self, '_diag_lift_count'):
                self._diag_lift_count = 0
            try:
                dof_pos = self.robot.root_physx_view.get_dof_positions()
                dof_vel = self.robot.root_physx_view.get_dof_velocities()
                print(f"[DIAG step={self.common_step_counter}] "
                      f"physx.dof_pos(lift)={dof_pos[0, self._lift_id]:.5f}, "
                      f"physx.dof_vel(lift)={dof_vel[0, self._lift_id]:.5f}, "
                      f"lab.joint_pos={self._joint_pos[0, self._lift_id]:.5f}, "
                      f"lift_target={self._lift_pos_target[0]:.5f}")
                if hasattr(self, '_diag_lift_count'):
                    self._diag_lift_count += 1
            except Exception as e:
                print(f"[DIAG step={self.common_step_counter}] PhysX 直读失败: {e}")

        # states
        root_pos = self.robot.data.root_pos_w
        root_quat = self.robot.data.root_quat_w
        root_lin_vel = self.robot.data.root_lin_vel_w
        root_ang_vel = self.robot.data.root_ang_vel_w

        yaw = _quat_to_yaw(root_quat)
        R = _yaw_to_mat2(-yaw)  # world->robot 2D

        pallet_pos = self.pallet.data.root_pos_w
        pallet_quat = self.pallet.data.root_quat_w
        pallet_yaw = _quat_to_yaw(pallet_quat)

        # relative position in world
        d_xy_w = (pallet_pos[:, :2] - root_pos[:, :2])
        d_xy_r = torch.einsum("nij,nj->ni", R, d_xy_w)

        dyaw = pallet_yaw - yaw
        cos_dyaw = torch.cos(dyaw)
        sin_dyaw = torch.sin(dyaw)

        # velocities in robot frame
        v_xy_w = root_lin_vel[:, :2]
        v_xy_r = torch.einsum("nij,nj->ni", R, v_xy_w)
        yaw_rate = root_ang_vel[:, 2:3]

        lift_pos = self._joint_pos[:, self._lift_id:self._lift_id + 1]
        lift_vel = self._joint_vel[:, self._lift_id:self._lift_id + 1]

        # insertion depth (normalized)
        # S1.0W: 修复观测坐标系与奖励计算不一致的问题 (Bug 1)。使用托盘局部坐标系投影。
        tip = self._compute_fork_tip()
        cp_obs = torch.cos(pallet_yaw)
        sp_obs = torch.sin(pallet_yaw)
        u_in_obs = torch.stack([cp_obs, sp_obs], dim=-1)
        rel_tip_obs = tip[:, :2] - pallet_pos[:, :2]
        s_tip_obs = torch.sum(rel_tip_obs * u_in_obs, dim=-1)
        s_front_obs = -0.5 * self.cfg.pallet_depth_m
        insert_depth_obs = torch.clamp(s_tip_obs - s_front_obs, min=0.0)
        
        # 新增 Z 轴对齐约束（防止隔空飞越作弊）
        lift_height_obs = tip[:, 2] - self._fork_tip_z0
        pallet_lift_height_obs = pallet_pos[:, 2] - self.cfg.pallet_cfg.init_state.pos[2]
        z_err_obs = torch.abs(lift_height_obs - pallet_lift_height_obs)
        valid_insert_z_obs = z_err_obs < self.cfg.max_insert_z_err
        insert_depth_obs = torch.where(valid_insert_z_obs, insert_depth_obs, torch.zeros_like(insert_depth_obs))
        
        insert_norm = torch.clamp(
            insert_depth_obs / (self.cfg.pallet_depth_m + 1e-6), 0.0, 1.0
        ).unsqueeze(-1)

        # S1.0N: pallet center line frame 误差（与 _get_rewards 同源几何）
        v_lat_obs = torch.stack([-sp_obs, cp_obs], dim=-1)
        y_signed_obs = torch.sum((root_pos[:, :2] - pallet_pos[:, :2]) * v_lat_obs, dim=-1)
        # S1.0S Phase-0.5: 使用可配置尺度替代硬编码 0.5（消除 |y|>scale 时观测饱和）
        y_err_obs = torch.clamp(y_signed_obs / self.cfg.y_err_obs_scale, -1.0, 1.0)

        dyaw_signed_obs = torch.atan2(torch.sin(yaw - pallet_yaw), torch.cos(yaw - pallet_yaw))
        yaw_err_obs = torch.clamp(dyaw_signed_obs / (15.0 * math.pi / 180.0), -1.0, 1.0)

        obs = torch.cat(
            [
                d_xy_r,  # 2
                cos_dyaw.unsqueeze(-1), sin_dyaw.unsqueeze(-1),  # 2
                v_xy_r,  # 2
                yaw_rate,  # 1
                lift_pos / self.cfg.lift_pos_scale, lift_vel,  # 2  S1.0T: obs 归一化
                insert_norm,  # 1
                self.actions,  # 3
                y_err_obs.unsqueeze(-1),    # 1 — S1.0N
                yaw_err_obs.unsqueeze(-1),  # 1 — S1.0N
            ],
            dim=-1,
        )
        # Isaac Lab direct workflow expects a dict with at least the "policy" key.
        # For asymmetric actor-critic, add a "critic" key with privileged observations.
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        """S1.0k 三阶段势函数奖励（严格中心线几何 + 全局势函数 shaping）

        核心改进（相对 S1.0j）：
        ──────────────────────────────────────────────────────────
        A. 严格中心线几何：y_err / yaw_err 基于托盘局部坐标轴，不再随距离变分母
        B. 三阶段势函数 shaping：
           Stage 1 (phi1): 距离带 [d1_min, d1_max] + 粗对齐
           Stage 2 (phi2): 微调接近（带距离带门控 + 粗对齐门控）
           Stage 3 (phi_ins): 插入深化（带严对齐门控）
           + phi_lift: 举升（带插入门控 + 严对齐门控）
        C. 全局 r_pot = gamma * phi_total_t - phi_total_{t-1}
           → 原地抖动净亏损，后退自动扣分
        D. action std 约束（配合 rsl_rl_ppo_cfg.py 修改）
        ──────────────────────────────────────────────────────────

        奖励结构（每 step 叠加）:
          rew = r_pot                                      # 势函数 shaping
               + pen_premature                              # 空举惩罚
               + pen_dense                                  # 时间 + 动作L2 + 距离惩罚
               + r_terminal                                 # 成功/超时终局奖励
        """
        # ---- 从 PhysX view 刷新关节数据 ----
        # _get_rewards() 在 _get_observations() 之前调用，需要先刷新
        self._joint_pos[:] = self.robot.root_physx_view.get_dof_positions()
        self._joint_vel[:] = self.robot.root_physx_view.get_dof_velocities()

        # ==================================================================
        # S1.0k: 三阶段势函数 shaping + 严格中心线几何
        # ==================================================================

        # ---- 基础量 ----
        root_pos = self.robot.data.root_pos_w
        pallet_pos = self.pallet.data.root_pos_w
        tip = self._compute_fork_tip()                                       # (N, 3)

        robot_yaw = _quat_to_yaw(self.robot.data.root_quat_w)               # (N,)
        pallet_yaw = _quat_to_yaw(self.pallet.data.root_quat_w)             # (N,)

        # ---- 严格中心线几何 ----
        cp = torch.cos(pallet_yaw)
        sp = torch.sin(pallet_yaw)
        u_in = torch.stack([cp, sp], dim=-1)                                 # (N,2) 插入方向
        v_lat = torch.stack([-sp, cp], dim=-1)                               # (N,2) 横向

        # 横向误差（中心线重叠度）
        rel_robot = root_pos[:, :2] - pallet_pos[:, :2]
        y_signed = torch.sum(rel_robot * v_lat, dim=-1)
        y_err = torch.abs(y_signed)

        # S1.0U: fork tip 在 pallet center-line frame 中的横向误差
        # tip 已在 L904 通过 _compute_fork_tip() 计算，直接复用。
        rel_tip_lat = tip[:, :2] - pallet_pos[:, :2]
        tip_y_signed = torch.sum(rel_tip_lat * v_lat, dim=-1)
        tip_y_err = torch.abs(tip_y_signed)

        # 偏航误差（中心线平行度）
        yaw_err = torch.atan2(
            torch.sin(robot_yaw - pallet_yaw),
            torch.cos(robot_yaw - pallet_yaw),
        )
        yaw_err_deg = torch.abs(yaw_err) * (180.0 / math.pi)
        yaw_err_rad = torch.abs(yaw_err)

        # 举升
        lift_height = tip[:, 2] - self._fork_tip_z0
        delta_lift = lift_height - self._last_lift_pos
        self._last_lift_pos = lift_height.detach()

        # 沿托盘插入轴的标量坐标
        rel_tip = tip[:, :2] - pallet_pos[:, :2]
        s_tip = torch.sum(rel_tip * u_in, dim=-1)
        s_front = -0.5 * self.cfg.pallet_depth_m

        dist_front = torch.clamp(s_front - s_tip, min=0.0)
        insert_depth = torch.clamp(s_tip - s_front, min=0.0)
        
        # 新增 Z 轴对齐约束（防止隔空飞越作弊）
        pallet_lift_height = pallet_pos[:, 2] - self.cfg.pallet_cfg.init_state.pos[2]
        z_err = torch.abs(lift_height - pallet_lift_height)
        valid_insert_z = z_err < self.cfg.max_insert_z_err
        insert_depth = torch.where(valid_insert_z, insert_depth, torch.zeros_like(insert_depth))
        
        insert_norm = torch.clamp(insert_depth / (self.cfg.pallet_depth_m + 1e-6), 0.0, 1.0)
        # S1.0L: Stage1/2 距离参考点改为 base（插入深度仍基于 tip）。
        rel_base = root_pos[:, :2] - pallet_pos[:, :2]
        s_base = torch.sum(rel_base * u_in, dim=-1)
        dist_front_base = torch.clamp(s_front - s_base, min=0.0)
        stage_dist_front = dist_front_base if self.cfg.stage_distance_ref == "base" else dist_front

        # 仍需 insert_depth 用于 _apply_action 安全制动
        self._last_insert_depth = insert_depth.detach()

        # 2D 距离（用于连续距离惩罚）
        d_xy = torch.sqrt(stage_dist_front ** 2 + y_err ** 2)

        # ---- Stage 1: 距离带 + 粗对齐 (Phi1 >= 0) ----
        e_band = torch.where(
            stage_dist_front < self.cfg.d1_min, self.cfg.d1_min - stage_dist_front,
            torch.where(stage_dist_front > self.cfg.d1_max, stage_dist_front - self.cfg.d1_max,
                        torch.zeros_like(dist_front))
        )
        E1 = (e_band / self.cfg.e_band_scale
              + y_err / self.cfg.y_scale1
              + yaw_err_deg / self.cfg.yaw_scale1)
        phi1 = self.cfg.k_phi1 / (1.0 + E1)

        # ---- Stage 2: 微调接近，带软门控 ----
        E2 = (stage_dist_front / self.cfg.d2_scale
              + y_err / self.cfg.y_scale2
              + yaw_err_deg / self.cfg.yaw_scale2)
        phi2_base = self.cfg.k_phi2 / (1.0 + E2)

        # 距离带权重（从 d1_max 到 d1_min 平滑打开）
        w_band = smoothstep(
            (self.cfg.d1_max - stage_dist_front) / (self.cfg.d1_max - self.cfg.d1_min)
        )
        # 粗对齐门控（指数软门控）
        # S1.0U: distance-weighted fork-tip alignment factor.
        # Far-field (dist > thresh): tip_weight=0, original behaviour.
        # Near-field (dist -> contact): tip_weight -> 1, tip dominates.
        tip_weight = smoothstep(
            (self.cfg.tip_y_weight_dist_thresh - stage_dist_front)
            / max(self.cfg.tip_y_weight_dist_thresh, 1e-6)
        )
        w_align2 = torch.exp(
            -(y_err / self.cfg.y_gate2) ** 2
            - (yaw_err_deg / self.cfg.yaw_gate2) ** 2
            - tip_weight * (tip_y_err / self.cfg.tip_y_gate2) ** 2   # S1.0U
        )
        phi2 = phi2_base * w_band * w_align2

        # ---- Stage 3: 插入深化 ----
        w3 = smoothstep(
            (insert_norm - self.cfg.ins_start) / self.cfg.ins_ramp
        )
        # 严对齐门控
        w_align3 = torch.exp(
            -(y_err / self.cfg.y_gate3) ** 2
            - (yaw_err_deg / self.cfg.yaw_gate3) ** 2
            - tip_weight * (tip_y_err / self.cfg.tip_y_gate3) ** 2   # S1.0U
        )
        # S1.0Q-B1: ins_floor 可配置（B0=0.4，B1a=0.1）+ 横向独立门控 w_lat_gate
        w_lat_gate = torch.exp(-(y_err / self.cfg.ins_lat_gate_sigma) ** 2)
        phi_ins = self.cfg.k_ins * (self.cfg.ins_floor + (1.0 - self.cfg.ins_floor) * w_align3) \
                  * insert_norm * w3 * w_lat_gate

        # ---- 举升势函数 + 空举惩罚 ----
        w_lift_base = smoothstep(
            (insert_norm - self.cfg.insert_gate_norm) / self.cfg.insert_ramp_norm
        )
        w_lift = w_lift_base * w_align3
        phi_lift = self.cfg.k_lift * w_lift * lift_height

        # S1.0O-A3: premature lift 惩罚分段温和化
        # insert_norm < hard_thresh 时全额惩罚，soft_thresh 以上惩罚 → 0
        premature_fade = smoothstep(
            (insert_norm - self.cfg.premature_hard_thresh)
            / (self.cfg.premature_soft_thresh - self.cfg.premature_hard_thresh + 1e-6)
        )
        # 防作弊修复：从动作惩罚(delta_lift)改为绝对高度惩罚(lift_height)，并加上 clamp 防止负高度变成正奖励
        pen_premature = -self.cfg.k_pre * (1.0 - premature_fade) * torch.clamp(lift_height, min=0.0)
        pen_premature = torch.where(self._is_first_step, torch.zeros_like(pen_premature), pen_premature)

        # ---- 总势函数 & shaping ----
        # S1.0L: 默认不压制 phi1/phi2，保留开关可回滚旧逻辑。
        if self.cfg.suppress_preinsert_phi_with_w3:
            phi_total = (phi1 + phi2) * (1.0 - w3) + phi_ins + phi_lift
        else:
            phi_total = phi1 + phi2 + phi_ins + phi_lift

        gamma = self.cfg.gamma
        r_pot = gamma * phi_total - self._last_phi_total
        r_pot = torch.where(self._is_first_step, torch.zeros_like(r_pot), r_pot)
        phi_delta_abs = torch.abs(phi_total - self._last_phi_total)
        self._last_phi_total = phi_total.detach()

        # ---- 常驻惩罚 ----
        pen_dense = (
            self.cfg.rew_time_penalty
            + self.cfg.rew_action_l2 * (self.actions ** 2).sum(dim=1)
            - self.cfg.k_dist_cont * d_xy
        )

        # ---- S1.0U: 碰撞接近惩罚 ----
        # Near pallet + fork tip misaligned -> penalty.
        prox_mask = (stage_dist_front < self.cfg.tip_prox_dist_thresh).float()
        prox_excess = torch.clamp(tip_y_err - self.cfg.tip_prox_lat_thresh, min=0.0)
        pen_tip_proximity = -self.cfg.k_tip_proximity_pen * prox_mask * prox_excess

        # ---- S1.0zB: 托盘位移惩罚 (Exp-A) ----
        # 惩罚在未插稳前推走托盘的行为
        pallet_init_pos_xy = torch.tensor(self.cfg.pallet_cfg.init_state.pos[:2], device=self.device)
        pallet_disp_xy = torch.norm(pallet_pos[:, :2] - pallet_init_pos_xy, dim=-1)
        push_excess = torch.clamp(pallet_disp_xy - self.cfg.pallet_push_deadband_m, min=0.0)
        push_gate = 1.0 - smoothstep(
            (insert_norm - self.cfg.pallet_push_insert_gate) / self.cfg.pallet_push_insert_ramp
        )
        pen_pallet_push = -self.cfg.k_pallet_push_pen * push_gate * push_excess

        # S1.0U: tip alignment flags (used by milestones AND hold counter below).
        is_near = stage_dist_front < self.cfg.tip_align_near_dist
        tip_align_ok = tip_y_err <= self.cfg.tip_align_entry_m
        tip_align_violated = tip_y_err > self.cfg.tip_align_exit_m

        # ---- 里程碑奖励（一次性）----
        milestone_approach = dist_front <= 0.25
        milestone_coarse_align = (y_err <= 0.20) & (yaw_err_deg <= 10.0)
        # S1.0N: gate_align 绑定 approach flag 防远处早触发
        # S1.0U: near-field alignment milestones also require tip alignment.
        milestone_gate_align = (
            (y_err <= 0.15) & (yaw_err_deg <= 8.0)
            & self._milestone_flags[:, 0]  # 必须先触发过 approach
            & (~is_near | tip_align_ok)    # S1.0U
        )
        milestone_insert10 = insert_norm >= 0.10
        milestone_insert30 = insert_norm >= 0.30
        # S1.0M: 新增对齐里程碑（使用与 aligned_enough 同源的 y_err/yaw_err_deg）
        milestone_fine_align = (
            (y_err <= 0.10) & (yaw_err_deg <= 5.0)
            & (~is_near | tip_align_ok)    # S1.0U
        )
        milestone_precise_align = (
            (y_err <= 0.05) & (yaw_err_deg <= 3.0)
            & (~is_near | tip_align_ok)    # S1.0U
        )

        milestone_hits = torch.stack(
            [milestone_approach, milestone_coarse_align, milestone_gate_align,
             milestone_insert10, milestone_insert30,
             milestone_fine_align, milestone_precise_align], dim=-1
        )

        # S1.0W: 修复里程碑永久剥夺 Bug (Bug 2)。
        # 死区判定提前：仅在非死区时才记录 flag 并发放奖励，
        # 死区内的触发不写入 flag，退出死区后仍可重新获得。
        in_dead_zone_now = (insert_norm > self.cfg.dead_zone_insert_thresh) & \
                           (y_err > self.cfg.dead_zone_lat_thresh)
        valid_for_flag = milestone_hits & (~in_dead_zone_now.unsqueeze(-1))
        new_hits = valid_for_flag & (~self._milestone_flags)
        self._milestone_flags |= new_hits

        milestone_reward = (
            new_hits[:, 0].float() * self.cfg.rew_milestone_approach
            + new_hits[:, 1].float() * self.cfg.rew_milestone_coarse_align
            + new_hits[:, 2].float() * self.cfg.rew_milestone_gate_align
            + new_hits[:, 3].float() * self.cfg.rew_milestone_insert_10
            + new_hits[:, 4].float() * self.cfg.rew_milestone_insert_30
            + new_hits[:, 5].float() * self.cfg.rew_milestone_fine_align
            + new_hits[:, 6].float() * self.cfg.rew_milestone_precise_align
        )

        # ---- S1.0T-fix: 举升里程碑（插入门控 + 阈值穿越触发）----
        # 必须在叉齿插入后、且高度从下方穿越阈值时才触发，防止空举刷分。
        # delta_lift 在 L947 已计算（用旧 _last_lift_pos），此处可安全回推 prev。
        prev_lift_height = lift_height - delta_lift
        lift_milestone_gate = insert_norm >= self.cfg.insert_gate_norm  # 0.35
        hit_lift_10cm = (
            (prev_lift_height < 0.10) & (lift_height >= 0.10)
            & lift_milestone_gate & (~self._milestone_lift_10cm)
        )
        hit_lift_20cm = (
            (prev_lift_height < 0.20) & (lift_height >= 0.20)
            & lift_milestone_gate & (~self._milestone_lift_20cm)
        )
        self._milestone_lift_10cm |= hit_lift_10cm
        self._milestone_lift_20cm |= hit_lift_20cm
        # S1.0T: 高举升里程碑（同样需要插入门控 + 穿越）
        hit_lift_50cm = (
            (prev_lift_height < 0.50) & (lift_height >= 0.50)
            & lift_milestone_gate & (~self._milestone_lift_50cm)
        )
        hit_lift_75cm = (
            (prev_lift_height < 0.75) & (lift_height >= 0.75)
            & lift_milestone_gate & (~self._milestone_lift_75cm)
        )
        self._milestone_lift_50cm |= hit_lift_50cm
        self._milestone_lift_75cm |= hit_lift_75cm
        milestone_reward = milestone_reward + (
            hit_lift_10cm.float() * self.cfg.rew_milestone_lift_10cm
            + hit_lift_20cm.float() * self.cfg.rew_milestone_lift_20cm
            + hit_lift_50cm.float() * self.cfg.rew_milestone_lift_50cm
            + hit_lift_75cm.float() * self.cfg.rew_milestone_lift_75cm
        )

        # ---- 失败早停计数（惩罚在 reward，终止在 dones）----
        fly_cond = d_xy > self.cfg.early_stop_d_xy_max
        self._fly_counter = torch.where(fly_cond, self._fly_counter + 1, torch.zeros_like(self._fly_counter))
        self._early_stop_fly = self._fly_counter >= self.cfg.early_stop_d_xy_steps

        action_abs_max = torch.max(torch.abs(self.actions), dim=1).values
        stall_cond = (~self._is_first_step) & (phi_delta_abs < self.cfg.early_stop_stall_phi_eps) & (
            action_abs_max < self.cfg.early_stop_stall_action_eps
        )
        self._stall_counter = torch.where(stall_cond, self._stall_counter + 1, torch.zeros_like(self._stall_counter))
        self._early_stop_stall = self._stall_counter >= self.cfg.early_stop_stall_steps

        # ---- S1.0Q Batch-3: Dead-zone stuck detector ----
        # 在死区内，如果连续 K 步 insert_norm 和 y_err 变化都 < epsilon，
        # 则判定为"卡死"（大幅动作但碰撞卡住），给 penalty 并触发 early done
        dz_stuck_cond = in_dead_zone_now & (
            torch.abs(insert_norm - self._prev_insert_norm) < self.cfg.dz_stuck_ins_eps
        ) & (
            torch.abs(y_err - self._prev_y_err) < self.cfg.dz_stuck_lat_eps
        )
        self._dz_stuck_counter = torch.where(
            dz_stuck_cond, self._dz_stuck_counter + 1,
            torch.zeros_like(self._dz_stuck_counter))
        dz_stuck_triggered = self._dz_stuck_counter >= self.cfg.dz_stuck_steps
        # Batch-4: 拆分 early done 与 penalty
        # early done 只在 dz_stuck_early_done=True 时触发终止
        self._early_stop_dz_stuck = dz_stuck_triggered & self.cfg.dz_stuck_early_done
        # penalty 只在首次触发时给一次（防止 penOnly 模式下每步累积）
        dz_stuck_first_fire = dz_stuck_triggered & (~self._dz_stuck_fired)
        self._dz_stuck_fired = self._dz_stuck_fired | dz_stuck_triggered
        self._prev_y_err = y_err.detach().clone()

        early_stop_penalty = (
            self._early_stop_fly.float() * self.cfg.rew_early_stop_fly
            + self._early_stop_stall.float() * self.cfg.rew_early_stop_stall
            + dz_stuck_first_fire.float() * self.cfg.rew_early_stop_dz_stuck
        )

        # ---- S1.0N: 全维度 Schmitt trigger hold counter ----
        # 对齐 entry/exit
        align_entry = (y_err <= self.cfg.max_lateral_err_m) & (yaw_err_deg <= self.cfg.max_yaw_err_deg)
        exit_y = self.cfg.max_lateral_err_m * self.cfg.hysteresis_ratio      # 0.18m
        exit_yaw = self.cfg.max_yaw_err_deg * self.cfg.hysteresis_ratio      # 9.6°
        align_exit_exceeded = (y_err > exit_y) | (yaw_err_deg > exit_yaw)

        # S1.0U: tip alignment constraint (near-field only).
        # is_near / tip_align_ok / tip_align_violated computed above (before milestones).
        align_entry = align_entry & (~is_near | tip_align_ok)
        align_exit_exceeded = align_exit_exceeded | (is_near & tip_align_violated)

        # 插入 entry/exit（带容差防微弹）
        insert_entry = insert_depth >= self._insert_thresh
        insert_exit_exceeded = insert_depth < (self._insert_thresh - self.cfg.insert_exit_epsilon)

        # 举升 entry/exit（带容差防微降）
        lift_entry = lift_height >= self.cfg.lift_delta_m
        lift_exit_exceeded = lift_height < (self.cfg.lift_delta_m - self.cfg.lift_exit_epsilon)

        # 三段式更新 + S1.0O-C2 越界衰减
        still_ok = insert_entry & align_entry & lift_entry
        any_exit_exceeded = align_exit_exceeded | insert_exit_exceeded | lift_exit_exceeded
        grace_zone = (~still_ok) & (~any_exit_exceeded)

        # S1.0O-C2: 越界时衰减而非清零（float counter）
        decayed = self._hold_counter * self.cfg.hold_counter_decay

        self._hold_counter = torch.where(
            still_ok,
            self._hold_counter + 1,
            torch.where(
                grace_zone,
                self._hold_counter,                        # hold constant: 不加不减
                decayed,                                   # S1.0O-C2: 越界衰减而非归零
            ),
        )
        success = self._hold_counter >= self._hold_steps

        # 保留兼容变量（用于日志）
        inserted_enough = insert_entry
        aligned_enough = align_entry
        lifted_enough = lift_entry
        success_now = still_ok

        time_ratio = self.episode_length_buf.float() / (self.max_episode_length + 1e-6)
        time_bonus = self.cfg.rew_success_time * (1.0 - time_ratio)

        r_terminal = torch.zeros((self.num_envs,), device=self.device)
        r_terminal = torch.where(
            success,
            self.cfg.rew_success + time_bonus,
            r_terminal,
        )
        time_out_now = self.episode_length_buf >= self.max_episode_length - 1
        r_terminal = torch.where(
            time_out_now & ~success,
            r_terminal + self.cfg.rew_timeout,
            r_terminal,
        )

        # ---- S1.0N: delta hold-align shaping ----
        phi_align = (
            torch.exp(-(y_err / self.cfg.hold_align_sigma_y) ** 2)
            * torch.exp(-(yaw_err_deg / self.cfg.hold_align_sigma_yaw) ** 2)
        )
        r_hold_align = self.cfg.k_hold_align * (phi_align - self._prev_phi_align)
        r_hold_align = torch.where(self._is_first_step, torch.zeros_like(r_hold_align), r_hold_align)
        self._prev_phi_align = phi_align.detach()

        # ---- S1.0T-fix: lift 进度 delta shaping（门控势函数差分）----
        # 用 w_lift_base 门控原始势函数后再做差分，防止空举获得 lift progress 奖励，
        # 并保持 potential-difference 语义连续（避免门控开关瞬间产生伪脉冲）。
        lift_target = self.cfg.lift_delta_m   # S1.0T: 1.0m
        lift_err = torch.abs(lift_height - lift_target)
        phi_lift_progress_raw = torch.exp(-(lift_err / self.cfg.sigma_lift) ** 2)
        phi_lift_progress_gated = w_lift_base * phi_lift_progress_raw
        r_lift_progress = self.cfg.k_lift_progress * (phi_lift_progress_gated - self._prev_phi_lift_progress)
        r_lift_progress = torch.where(self._is_first_step, torch.zeros_like(r_lift_progress), r_lift_progress)
        self._prev_phi_lift_progress = phi_lift_progress_gated.detach()

        # ---- S1.0Q-A1: 深插死区惩罚 ----
        dead_zone = (insert_norm > self.cfg.dead_zone_insert_thresh) & \
                    (y_err > self.cfg.dead_zone_lat_thresh)
        pen_dead_zone = -self.cfg.k_dead_zone * dead_zone.float() * \
            (insert_norm - self.cfg.dead_zone_insert_thresh) * \
            (y_err - self.cfg.dead_zone_lat_thresh)
        pen_dead_zone = torch.clamp(pen_dead_zone, min=-self.cfg.dead_zone_pen_clamp, max=0.0)

        # ---- S1.0Q-A2v2: 窗口式死区撤退鼓励 ----
        # 写入当前 insert_norm 到环形缓冲
        env_arange = torch.arange(self.num_envs, device=self.device)
        ptr = self._window_ptr
        self._insert_norm_window[env_arange, ptr] = insert_norm
        self._window_ptr = (ptr + 1) % self.cfg.retreat_window_size
        # 窗口填满后才激活（至少积累满一个完整窗口的数据）
        self._window_filled = self._window_filled | (self._window_ptr == 0)
        # 净下降 = 窗口最老值 - 当前值（正值表示净退出）
        oldest_idx = self._window_ptr  # 下一个要被覆盖的位置就是最老的
        oldest_insert = self._insert_norm_window[env_arange, oldest_idx]
        net_retreat = oldest_insert - insert_norm  # 正 = 退出
        r_retreat = torch.where(
            self._prev_in_dead_zone & self._window_filled & ~self._is_first_step,
            self.cfg.k_retreat * torch.clamp(net_retreat, min=0.0),  # 只奖励净退出
            torch.zeros_like(insert_norm)
        )
        r_retreat = torch.clamp(r_retreat, max=self.cfg.retreat_reward_clamp)
        # 更新 prev 状态
        self._prev_insert_norm = insert_norm.detach()
        self._prev_in_dead_zone = dead_zone.detach().clone()

        # ---- S1.0Q-C1: 横向 delta shaping ----
        lat_fine_active = insert_norm > self.cfg.lat_fine_ins_thresh
        phi_lat = torch.exp(-(y_err / self.cfg.lat_fine_sigma) ** 2)
        r_lat_fine = self.cfg.k_lat_fine * (phi_lat - self._prev_phi_lat)
        r_lat_fine = torch.where(
            lat_fine_active & ~self._is_first_step,
            r_lat_fine,
            torch.zeros_like(r_lat_fine)
        )
        self._prev_phi_lat = phi_lat.detach()

        # ---- S1.0S Phase-R: 远场大横偏修正奖励 ----
        # 仅在 y_err > 阈值 且 insert_norm < 阈值（还没开始插入）时激活
        far_lat_active = (y_err > self.cfg.far_lat_y_thresh) & (insert_norm < self.cfg.far_lat_ins_thresh)
        r_far_lat_fix = self.cfg.k_far_lat * far_lat_active.float() * torch.clamp(
            self._prev_y_err_far - y_err, min=0.0
        )
        r_far_lat_fix = torch.where(self._is_first_step, torch.zeros_like(r_far_lat_fix), r_far_lat_fix)
        self._prev_y_err_far = y_err.detach().clone()

        # ---- S1.0S Phase-3: 全局进展停滞检测 ----
        # 增加豁免条件：如果已经到达目标并保持住（still_ok=True），则不计为停滞
        global_stall_cond = (
            (~self._is_first_step)
            & (torch.abs(phi_total - self._prev_phi_total_stall) < self.cfg.global_stall_phi_eps)
            & (~still_ok)
        )
        self._global_stall_counter = torch.where(
            global_stall_cond,
            self._global_stall_counter + 1,
            torch.zeros_like(self._global_stall_counter)
        )
        pen_global_stall = torch.where(
            self._global_stall_counter >= self.cfg.global_stall_steps,
            torch.full_like(phi_total, self.cfg.rew_global_stall),
            torch.zeros_like(phi_total)
        )
        self._prev_phi_total_stall = phi_total.detach().clone()
        
        # 新增静止奖励：成功后保持不动
        r_stay_still = self.cfg.rew_stay_still * still_ok.float()

        # ---- 总奖励 ----
        rew = (r_pot + pen_premature + pen_dense + r_terminal + milestone_reward
               + r_hold_align + r_lift_progress + early_stop_penalty
               + pen_dead_zone + r_retreat + r_lat_fine
               + r_far_lat_fix + pen_global_stall
               + pen_tip_proximity + pen_pallet_push + r_stay_still)  # S1.0zB + S1.0U + 防作弊修复

        # 清除首步标记
        self._is_first_step[:] = False

        # ==================================================================
        # 日志输出
        # ==================================================================
        if "log" not in self.extras:
            self.extras["log"] = {}

        # 势函数分量
        self.extras["log"]["phi/phi1"] = phi1.mean()
        self.extras["log"]["phi/phi2"] = phi2.mean()
        self.extras["log"]["phi/phi_ins"] = phi_ins.mean()
        self.extras["log"]["phi/phi_lift"] = phi_lift.mean()
        self.extras["log"]["phi/phi_total"] = phi_total.mean()

        # 奖励分量
        self.extras["log"]["s0/r_pot"] = r_pot.mean()
        self.extras["log"]["s0/pen_premature"] = pen_premature.mean()
        self.extras["log"]["s0/pen_dense"] = pen_dense.mean()
        self.extras["log"]["s0/r_terminal"] = r_terminal.mean()
        self.extras["log"]["s0/r_milestone"] = milestone_reward.mean()
        self.extras["log"]["s0/r_hold_align"] = r_hold_align.mean()
        self.extras["log"]["s0/r_lift_progress"] = r_lift_progress.mean()
        self.extras["log"]["s0/pen_early_stop"] = early_stop_penalty.mean()
        # S1.0Q: 新增奖励分量日志
        self.extras["log"]["s0/pen_dead_zone"] = pen_dead_zone.mean()
        self.extras["log"]["s0/r_retreat"] = r_retreat.mean()
        self.extras["log"]["s0/r_lat_fine"] = r_lat_fine.mean()
        # S1.0S: 新增奖励分量日志
        self.extras["log"]["s0/r_far_lat_fix"] = r_far_lat_fix.mean()
        self.extras["log"]["s0/pen_global_stall"] = pen_global_stall.mean()
        self.extras["log"]["s0/r_stay_still"] = r_stay_still.mean()
        self.extras["log"]["s0/pen_pallet_push"] = pen_pallet_push.mean()

        # 门控权重
        self.extras["log"]["s0/w_band"] = w_band.mean()
        self.extras["log"]["s0/w3"] = w3.mean()
        self.extras["log"]["s0/w_align2"] = w_align2.mean()
        self.extras["log"]["s0/w_align3"] = w_align3.mean()
        self.extras["log"]["s0/w_lift_base"] = w_lift_base.mean()
        self.extras["log"]["s0/w_lift"] = w_lift.mean()
        self.extras["log"]["s0/e_band"] = e_band.mean()
        # S1.0Q: 新增门控权重日志
        self.extras["log"]["s0/w_lat_gate"] = w_lat_gate.mean()

        # 误差分布
        self.extras["log"]["err/lateral_mean"] = y_err.mean()
        self.extras["log"]["err/yaw_deg_mean"] = yaw_err_deg.mean()
        self.extras["log"]["err/insert_norm_mean"] = insert_norm.mean()
        self.extras["log"]["err/lift_height_mean"] = lift_height.mean()
        self.extras["log"]["err/dist_front_mean"] = dist_front.mean()
        self.extras["log"]["err/dist_front_base_mean"] = dist_front_base.mean()
        self.extras["log"]["err/stage_dist_front_mean"] = stage_dist_front.mean()
        self.extras["log"]["diag/pallet_disp_xy_mean"] = pallet_disp_xy.mean()
        self.extras["log"]["diag/pallet_disp_xy_max"] = pallet_disp_xy.max()
        self.extras["log"]["err/d_xy_mean"] = d_xy.mean()

        # 阶段命中率
        self.extras["log"]["phase/frac_inserted"] = inserted_enough.float().mean()
        self.extras["log"]["phase/frac_aligned"] = aligned_enough.float().mean()
        self.extras["log"]["phase/frac_lifted"] = lifted_enough.float().mean()
        self.extras["log"]["phase/frac_success_now"] = success_now.float().mean()
        self.extras["log"]["phase/frac_success"] = success.float().mean()
        self.extras["log"]["phase/hold_counter_max"] = self._hold_counter.float().max()
        # S1.0N: hold counter 均值 + grace zone 比例
        self.extras["log"]["phase/hold_counter_mean"] = self._hold_counter.float().mean()
        self.extras["log"]["phase/grace_zone_frac"] = grace_zone.float().mean()
        self.extras["log"]["milestone/hit_approach"] = new_hits[:, 0].float().mean()
        self.extras["log"]["milestone/hit_align"] = new_hits[:, 1].float().mean()
        # S1.0N: gate_align 里程碑日志
        self.extras["log"]["milestone/hit_gate_align"] = new_hits[:, 2].float().mean()
        self.extras["log"]["milestone/hit_insert10"] = new_hits[:, 3].float().mean()
        self.extras["log"]["milestone/hit_insert30"] = new_hits[:, 4].float().mean()
        # S1.0M: 对齐里程碑日志
        self.extras["log"]["milestone/hit_fine_align"] = new_hits[:, 5].float().mean()
        self.extras["log"]["milestone/hit_precise_align"] = new_hits[:, 6].float().mean()
        # S1.0S: 举升里程碑日志
        self.extras["log"]["milestone/lift_10cm_frac"] = self._milestone_lift_10cm.float().mean()
        self.extras["log"]["milestone/lift_20cm_frac"] = self._milestone_lift_20cm.float().mean()
        # S1.0T: 高举升里程碑日志
        self.extras["log"]["milestone/lift_50cm_frac"] = self._milestone_lift_50cm.float().mean()
        self.extras["log"]["milestone/lift_75cm_frac"] = self._milestone_lift_75cm.float().mean()

        # Hold 失败诊断日志
        self.extras["log"]["diag_hold/fail_ins_frac"] = ((~insert_entry) & (self._hold_counter > 0)).float().mean()
        self.extras["log"]["diag_hold/fail_align_frac"] = ((~align_entry) & (self._hold_counter > 0)).float().mean()
        self.extras["log"]["diag_hold/fail_lift_frac"] = ((~lift_entry) & (self._hold_counter > 0)).float().mean()
        self.extras["log"]["diag_hold/ins_margin"] = (insert_depth - self._insert_thresh).mean()
        self.extras["log"]["diag_hold/y_margin"] = (self.cfg.max_lateral_err_m - y_err).mean()
        self.extras["log"]["diag_hold/yaw_margin"] = (self.cfg.max_yaw_err_deg - yaw_err_deg).mean()
        self.extras["log"]["diag_hold/lift_margin"] = (lift_height - self.cfg.lift_delta_m).mean()

        # 终止原因
        q = self.robot.data.root_quat_w
        w, x, y, z = q.unbind(-1)
        sinr_cosp = 2.0 * (w * x + y * z)
        cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
        roll = torch.atan2(sinr_cosp, cosr_cosp)
        sinp = 2.0 * (w * y - z * x)
        pitch = torch.asin(torch.clamp(sinp, -1.0, 1.0))
        tipped = (torch.abs(roll) > self.cfg.max_roll_pitch_rad) | (
            torch.abs(pitch) > self.cfg.max_roll_pitch_rad
        )
        self.extras["log"]["term/frac_tipped"] = tipped.float().mean()
        self.extras["log"]["term/frac_timeout"] = time_out_now.float().mean()
        self.extras["log"]["term/frac_early_fly"] = self._early_stop_fly.float().mean()
        self.extras["log"]["term/frac_early_stall"] = self._early_stop_stall.float().mean()
        self.extras["log"]["term/frac_early_dz_stuck"] = self._early_stop_dz_stuck.float().mean()

        # S1.0z: Episode 级别真实成功率统计
        reset_mask = self.reset_buf
        if reset_mask.any():
            num_resets = reset_mask.sum().item()
            num_successes = (success & reset_mask).sum().item()
            
            self._ep_total_count += num_resets
            self._ep_success_count += num_successes
            
            # 使用 EMA 跟踪最近的成功率 (平滑系数 0.001，约等于过去 1000 次 reset 的平均)
            # 1024 个环境，每个 episode ~400 步，每步平均 ~2.5 次 reset，1000 次 reset 约等于 400 步 (一个完整周期)
            batch_rate = num_successes / num_resets
            if not hasattr(self, "_ep_success_rate_ema"):
                self._ep_success_rate_ema = batch_rate
            else:
                # 动态 alpha，确保批量 reset 时权重正确
                alpha = min(1.0, num_resets * 0.001)
                self._ep_success_rate_ema = (1 - alpha) * self._ep_success_rate_ema + alpha * batch_rate

        if hasattr(self, "_ep_success_rate_ema"):
            self.extras["log"]["episode/success_rate_ema"] = self._ep_success_rate_ema
        
        if self._ep_total_count > 0:
            self.extras["log"]["episode/success_rate_total"] = self._ep_success_count / self._ep_total_count

        # ==================================================================
        # S1.0P: 诊断观测指标（不改训练，只改看结果的方式）
        # ==================================================================

        # --- 近成功区域 yaw 指标 (insert_norm > 0.1) ---
        near_success_mask = insert_norm > 0.1
        near_frac = near_success_mask.float().mean()
        self.extras["log"]["diag/near_success_frac"] = near_frac
        if near_success_mask.any():
            yaw_near = yaw_err_deg[near_success_mask]
            self.extras["log"]["err/yaw_deg_near_success"] = yaw_near.mean()
            self.extras["log"]["err/lateral_near_success"] = y_err[near_success_mask].mean()
            # r_hold_align 信号可见性 (改进 4/D)
            r_align_near = r_hold_align[near_success_mask]
            self.extras["log"]["s0/r_hold_align_near_p50"] = torch.quantile(r_align_near, 0.5)
            self.extras["log"]["s0/r_hold_align_near_p90"] = torch.quantile(r_align_near, 0.9)

        # --- 深插阶段 yaw 指标 (insert_norm > 0.3) (改进 6/C) ---
        deep_insert_mask = insert_norm > 0.3
        self.extras["log"]["diag/deep_insert_frac"] = deep_insert_mask.float().mean()
        if deep_insert_mask.any():
            yaw_deep = yaw_err_deg[deep_insert_mask]
            self.extras["log"]["err/yaw_deg_deep_mean"] = yaw_deep.mean()
            self.extras["log"]["err/yaw_deg_deep_p90"] = torch.quantile(yaw_deep, 0.9)

        # ==================================================================
        # S1.0Q: 新增诊断指标
        # ==================================================================
        # 死区 step 比例（深插 + 横向大偏差）
        self.extras["log"]["diag/deep_lat_bad_frac"] = dead_zone.float().mean()

        return rew

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # time out
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        # success when hold counter reached
        success = self._hold_counter >= self._hold_steps

        # tip-over check via roll/pitch
        q = self.robot.data.root_quat_w
        w, x, y, z = q.unbind(-1)
        # roll
        sinr_cosp = 2.0 * (w * x + y * z)
        cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
        roll = torch.atan2(sinr_cosp, cosr_cosp)
        # pitch
        sinp = 2.0 * (w * y - z * x)
        pitch = torch.asin(torch.clamp(sinp, -1.0, 1.0))
        tipped = (torch.abs(roll) > self.cfg.max_roll_pitch_rad) | (torch.abs(pitch) > self.cfg.max_roll_pitch_rad)

        terminated = tipped | success | self._early_stop_fly | self._early_stop_stall | self._early_stop_dz_stuck
        return terminated, time_out

    # ---------------------------
    # Reset
    # ---------------------------
    def _reset_idx(self, env_ids: torch.Tensor | None):
        """按 env_ids 重置环境。

        包括：
        - 清零奖励/插入缓存
        - 托盘固定在初始位姿
        - 叉车随机化初始位姿（x/y/yaw）
        - 关节与速度归零，并刷新物理状态
        """
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        # ---- 重置基类 episode 计数器（必须！否则 episode_length_buf 永不归零） ----
        # 基类 DirectRLEnv._reset_idx() 中会做 self.episode_length_buf[env_ids] = 0
        # 如果不调用 super()，RSL-RL 的 init_at_random_ep_len 随机初始值将永远不会被清除，
        # 导致环境在训练前几轮就全部进入永久超时（episode length = 1）。
        super()._reset_idx(env_ids)

        # ---- 计数器清零 ----
        self._last_insert_depth[env_ids] = 0.0
        self._hold_counter[env_ids] = 0
        self._is_first_step[env_ids] = True
        self._lift_pos_target[env_ids] = 0.0
        self._milestone_flags[env_ids] = False
        self._fly_counter[env_ids] = 0
        self._stall_counter[env_ids] = 0
        self._early_stop_fly[env_ids] = False
        self._early_stop_stall[env_ids] = False
        # S1.0Q Batch-3: dead-zone stuck detector
        self._dz_stuck_counter[env_ids] = 0
        self._prev_y_err[env_ids] = 0.0
        self._early_stop_dz_stuck[env_ids] = False
        # S1.0Q Batch-4: penOnly 首次触发标记
        self._dz_stuck_fired[env_ids] = False
        # S1.0N: _prev_phi_align 暂时清零，在位姿写入后再初始化为当前 phi_align
        self._prev_phi_align[env_ids] = 0.0
        # S1.0O-A3: lift 进度势函数缓存清零
        self._prev_phi_lift_progress[env_ids] = 0.0
        # S1.0Q: 死区撤退 / 横向精调状态量清零
        self._prev_insert_norm[env_ids] = 0.0
        self._prev_in_dead_zone[env_ids] = False
        self._prev_phi_lat[env_ids] = 0.0
        # S1.0S Phase-2: 举升里程碑清零
        self._milestone_lift_10cm[env_ids] = False
        self._milestone_lift_20cm[env_ids] = False
        # S1.0T: 高举升里程碑清零
        self._milestone_lift_50cm[env_ids] = False
        self._milestone_lift_75cm[env_ids] = False
        # S1.0S Phase-R: 远场修正状态量清零
        self._prev_y_err_far[env_ids] = 0.0
        # S1.0S Phase-3: 全局停滞检测器清零
        self._global_stall_counter[env_ids] = 0
        self._prev_phi_total_stall[env_ids] = 0.0
        # S1.0Q-A2v2: 撤退窗口缓冲清零
        self._insert_norm_window[env_ids] = 0.0
        self._window_ptr[env_ids] = 0
        self._window_filled[env_ids] = False

        # ---- 托盘固定位姿（可选：后续可加随机化） ----
        pallet_pos = torch.tensor(self.cfg.pallet_cfg.init_state.pos, device=self.device).repeat(len(env_ids), 1)
        pallet_quat = torch.tensor(self.cfg.pallet_cfg.init_state.rot, device=self.device).repeat(len(env_ids), 1)
        self._write_root_pose(self.pallet, pallet_pos, pallet_quat, env_ids)

        # ---- 随机化叉车初始位姿 ----
        # _pallet_front_x ≈ -1.08m，fork_forward_offset ≈ 1.87m
        # x_max 需满足: x_max + 1.87*cos(yaw_max) < -1.08 → x_max < -2.89m
        # 取 -3.0m 留安全裕度，避免叉齿穿透托盘导致 PhysX 去穿透弹飞托盘
        x = sample_uniform(-4.0, -3.0, (len(env_ids), 1), device=self.device)
        y = sample_uniform(-0.6, 0.6, (len(env_ids), 1), device=self.device)
        z = torch.full((len(env_ids), 1), 0.03, device=self.device)
        yaw = sample_uniform(-0.25, 0.25, (len(env_ids), 1), device=self.device)

        pos = torch.cat([x, y, z], dim=1)
        half = yaw * 0.5
        quat = torch.cat([torch.cos(half), torch.zeros_like(half), torch.zeros_like(half), torch.sin(half)], dim=1)

        self._write_root_pose(self.robot, pos, quat, env_ids)

        # 速度清零
        zeros3 = torch.zeros((len(env_ids), 3), device=self.device)
        self._write_root_vel(self.robot, zeros3, zeros3, env_ids)

        # 关节归零（lift down, wheels zero, steering zero）
        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        joint_vel = torch.zeros_like(joint_pos)
        self._write_joint_state(self.robot, joint_pos, joint_vel, env_ids)

        # ---- 基线 fork tip 高度 ----
        # S1.0h 修复：不再调用 scene.write_data_to_sim() / sim.reset() / scene.update()
        # sim.reset() 是全局 PhysX 引擎重置，会将所有 1024 个环境的位姿覆盖回 config 默认值，
        # 完全摧毁上面刚写入的随机化位姿（详见 S0.7 postmortem）。
        # _fork_z_base = 0.0（训练日志实测），因此 fork_tip_z0 = root_z = z.squeeze(-1)，误差为零。
        self._fork_tip_z0[env_ids] = z.squeeze(-1)

        # ---- S1.0k: 初始化势函数缓存 ----
        # reset 时 insert_depth ≈ 0、lift_height ≈ 0
        # 精确计算 phi_total 初始值，避免首步异常大的 r_pot
        # 注：_is_first_step 保护会清零首步 r_pot，但精确初始化更安全
        y_err_reset = torch.abs(y.squeeze(-1))
        yaw_err_deg_reset = torch.abs(yaw.squeeze(-1)) * (180.0 / math.pi)
        # S1.0L: reset 初始化与 _get_rewards 保持一致（stage_distance_ref=base）。
        cp_reset = torch.cos(torch.zeros_like(x.squeeze(-1)))
        sp_reset = torch.sin(torch.zeros_like(x.squeeze(-1)))
        u_in_reset = torch.stack([cp_reset, sp_reset], dim=-1)
        pallet_pos_reset = pallet_pos[:, :2]
        rel_base_reset = pos[:, :2] - pallet_pos_reset
        s_base_reset = torch.sum(rel_base_reset * u_in_reset, dim=-1)
        s_front_reset = -0.5 * self.cfg.pallet_depth_m
        if self.cfg.stage_distance_ref == "base":
            dist_front_reset = torch.clamp(s_front_reset - s_base_reset, min=0.0)
        else:
            cos_yaw_reset = torch.cos(yaw.squeeze(-1))
            sin_yaw_reset = torch.sin(yaw.squeeze(-1))
            tip_x_reset = pos[:, 0] + self._fork_forward_offset * cos_yaw_reset
            tip_y_reset = pos[:, 1] + self._fork_forward_offset * sin_yaw_reset
            rel_tip_reset = torch.stack([tip_x_reset, tip_y_reset], dim=-1) - pallet_pos_reset
            s_tip_reset = torch.sum(rel_tip_reset * u_in_reset, dim=-1)
            dist_front_reset = torch.clamp(s_front_reset - s_tip_reset, min=0.0)

        # 计算 phi1 初始值
        e_band_reset = torch.where(
            dist_front_reset < self.cfg.d1_min, self.cfg.d1_min - dist_front_reset,
            torch.where(dist_front_reset > self.cfg.d1_max, dist_front_reset - self.cfg.d1_max,
                        torch.zeros_like(dist_front_reset))
        )
        E1_reset = (e_band_reset / self.cfg.e_band_scale
                     + y_err_reset / self.cfg.y_scale1
                     + yaw_err_deg_reset / self.cfg.yaw_scale1)
        phi1_reset = self.cfg.k_phi1 / (1.0 + E1_reset)

        # 计算 phi2 初始值
        E2_reset = (dist_front_reset / self.cfg.d2_scale
                     + y_err_reset / self.cfg.y_scale2
                     + yaw_err_deg_reset / self.cfg.yaw_scale2)
        phi2_base_reset = self.cfg.k_phi2 / (1.0 + E2_reset)
        w_band_reset = smoothstep(
            (self.cfg.d1_max - dist_front_reset) / (self.cfg.d1_max - self.cfg.d1_min)
        )
        w_align2_reset = torch.exp(
            -(y_err_reset / self.cfg.y_gate2) ** 2
            - (yaw_err_deg_reset / self.cfg.yaw_gate2) ** 2
        )
        phi2_reset = phi2_base_reset * w_band_reset * w_align2_reset

        # reset 时 insert_norm ≈ 0, lift_height ≈ 0 → phi_ins = 0, phi_lift = 0
        if self.cfg.suppress_preinsert_phi_with_w3:
            phi_total_reset = phi1_reset + phi2_reset  # w3≈0 时等价
        else:
            phi_total_reset = phi1_reset + phi2_reset
        self._last_phi_total[env_ids] = phi_total_reset

        # 举升增量缓存
        self._last_lift_pos[env_ids] = 0.0

        # S1.0N: 初始化 _prev_phi_align 为当前位姿的 phi_align，防"开局白嫖"
        # 注意：此时 robot 位姿已写入但 PhysX 尚未 step，
        # 使用 reset 时已知的 y_err/yaw_err 直接计算（避免依赖 PhysX view）
        phi_align_init = (
            torch.exp(-(y_err_reset / self.cfg.hold_align_sigma_y) ** 2)
            * torch.exp(-(yaw_err_deg_reset / self.cfg.hold_align_sigma_yaw) ** 2)
        )
        self._prev_phi_align[env_ids] = phi_align_init.detach()

        # S1.0Q: 初始化 _prev_phi_lat 为 reset 位姿的 phi_lat（防"开局白嫖"）
        phi_lat_init = torch.exp(-(y_err_reset / self.cfg.lat_fine_sigma) ** 2)
        self._prev_phi_lat[env_ids] = phi_lat_init.detach()
        # _prev_insert_norm 在 reset 时 insert_norm ≈ 0，保持清零即可

        # 注：不再额外调用 self.robot.reset(env_ids)，
        # super()._reset_idx() 已通过 scene.reset(env_ids) 调用过一次。

    # ---------------------------
    # Compatibility helpers (API name differences across versions)
    # ---------------------------
    def _write_root_pose(self, asset, pos, quat, env_ids):
        """设置 asset 的根位姿。

        Isaac Lab >=1.x API: write_root_pose_to_sim(root_pose: (N,7), env_ids)
        root_pose = [pos(3), quat(4)]  (quat 格式 w,x,y,z)
        """
        root_pose = torch.cat([pos, quat], dim=-1)  # (N, 7)
        if hasattr(asset, "write_root_pose_to_sim"):
            asset.write_root_pose_to_sim(root_pose, env_ids)
        elif hasattr(asset, "write_root_state_to_sim"):
            root_state = torch.zeros((len(env_ids), 13), device=self.device)
            root_state[:, 0:7] = root_pose
            asset.write_root_state_to_sim(root_state, env_ids)
        else:
            raise AttributeError("Asset has no known root pose writer.")

    def _write_root_vel(self, asset, lin_vel, ang_vel, env_ids):
        """设置 asset 的根速度。

        Isaac Lab >=1.x API: write_root_velocity_to_sim(root_velocity: (N,6), env_ids)
        root_velocity = [lin_vel(3), ang_vel(3)]
        """
        root_vel = torch.cat([lin_vel, ang_vel], dim=-1)  # (N, 6)
        if hasattr(asset, "write_root_velocity_to_sim"):
            asset.write_root_velocity_to_sim(root_vel, env_ids)
        elif hasattr(asset, "write_root_state_to_sim"):
            pass
        else:
            raise AttributeError("Asset has no known root velocity writer.")

    def _write_joint_state(self, articulation, joint_pos, joint_vel, env_ids):
        """设置关节状态（位置 + 速度）。

        注意 write_joint_state_to_sim 的第三个位置参数是 joint_ids，
        必须用关键字参数传 env_ids，否则会被误当作 joint_ids。
        """
        if hasattr(articulation, "write_joint_state_to_sim"):
            articulation.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        elif hasattr(articulation, "write_joint_pos_to_sim") and hasattr(articulation, "write_joint_vel_to_sim"):
            articulation.write_joint_pos_to_sim(joint_pos, env_ids=env_ids)
            articulation.write_joint_vel_to_sim(joint_vel, env_ids=env_ids)
        else:
            raise AttributeError("Articulation has no known joint state writer.")
