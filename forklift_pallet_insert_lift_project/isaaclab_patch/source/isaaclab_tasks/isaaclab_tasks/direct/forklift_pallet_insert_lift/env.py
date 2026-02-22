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
from .ref_trajectory import generate_bezier_path, closest_point_on_path


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

        # joint indices
        self._front_wheel_ids, _ = self.robot.find_joints(["left_front_wheel_joint", "right_front_wheel_joint"], preserve_order=True)
        self._back_wheel_ids, _ = self.robot.find_joints(["left_back_wheel_joint", "right_back_wheel_joint"], preserve_order=True)
        self._rotator_ids, _ = self.robot.find_joints(["left_rotator_joint", "right_rotator_joint"], preserve_order=True)
        self._left_rotator_id, _ = self.robot.find_joints(["left_rotator_joint"], preserve_order=True)
        self._right_rotator_id, _ = self.robot.find_joints(["right_rotator_joint"], preserve_order=True)
        self._lift_id, _ = self.robot.find_joints(["lift_joint"], preserve_order=True)
        self._lift_id = self._lift_id[0]

        N = self.num_envs
        dev = self.device

        self.actions = torch.zeros((N, self.cfg.action_space), device=dev)
        self._last_insert_depth = torch.zeros((N,), device=dev)
        self._fork_tip_z0 = torch.zeros((N,), device=dev)
        self._lift_pos_target = torch.zeros((N,), device=dev)

        # S2.0a buffers
        self._is_first_step = torch.ones((N,), dtype=torch.bool, device=dev)
        self._prev_actions = torch.zeros((N, 3), device=dev)
        self._ref_waypoints = torch.zeros((N, self.cfg.ref_num_waypoints, 2), device=dev)
        self._ref_tangents = torch.zeros((N, self.cfg.ref_num_waypoints), device=dev)
        self._cached_insert_norm = torch.zeros((N,), device=dev)
        self._cached_y_err = torch.zeros((N,), device=dev)
        self._cached_yaw_err_deg = torch.zeros((N,), device=dev)

        self._pallet_front_x = self.cfg.pallet_cfg.init_state.pos[0] - self.cfg.pallet_depth_m * 0.5
        self._insert_thresh = self.cfg.goal_insert_fraction * self.cfg.pallet_depth_m

        num_joints = len(self.robot.joint_names)
        self._joint_pos = torch.zeros((N, num_joints), device=dev)
        self._joint_vel = torch.zeros((N, num_joints), device=dev)

        self._fork_forward_offset, self._fork_z_base = self._measure_fork_offset_from_usd()

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
        """将动作写入仿真。S2.0a: 禁用 lift，移除插入安全制动。"""
        drive = self.actions[:, 0] * self.cfg.wheel_speed_rad_s
        steer = self.actions[:, 1] * self.cfg.steer_angle_rad
        lift_v = torch.zeros_like(drive)

        self.robot.set_joint_velocity_target(drive.unsqueeze(-1).repeat(1, len(self._front_wheel_ids)), joint_ids=self._front_wheel_ids)
        self.robot.set_joint_velocity_target(drive.unsqueeze(-1).repeat(1, len(self._back_wheel_ids)), joint_ids=self._back_wheel_ids)

        steer_left = steer
        steer_right = steer
        self.robot.set_joint_position_target(steer_left.unsqueeze(-1), joint_ids=self._left_rotator_id)
        self.robot.set_joint_position_target(steer_right.unsqueeze(-1), joint_ids=self._right_rotator_id)

        self._lift_pos_target += lift_v * self.cfg.sim.dt
        self._lift_pos_target = torch.clamp(self._lift_pos_target, 0.0, 2.0)
        self.robot.set_joint_position_target(self._lift_pos_target.unsqueeze(-1), joint_ids=[self._lift_id])
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
        """S2.0a: paper-style R+/R- reward.

        R+ = a1/(r_d+e) + a2/(r_cd+e) + a3/(r_cpsi+e) + a4*r_g
        R- = a5*r_p + a6*r_v + a7*r_a + a8*r_ini
        """
        self._joint_pos[:] = self.robot.root_physx_view.get_dof_positions()
        self._joint_vel[:] = self.robot.root_physx_view.get_dof_velocities()

        cfg = self.cfg
        tip = self._compute_fork_tip()
        root_pos = self.robot.data.root_pos_w
        pallet_pos = self.pallet.data.root_pos_w
        robot_yaw = _quat_to_yaw(self.robot.data.root_quat_w)
        pallet_yaw = _quat_to_yaw(self.pallet.data.root_quat_w)

        cp = torch.cos(pallet_yaw)
        sp = torch.sin(pallet_yaw)
        u_in = torch.stack([cp, sp], dim=-1)
        v_lat = torch.stack([-sp, cp], dim=-1)

        rel_robot = root_pos[:, :2] - pallet_pos[:, :2]
        y_err = torch.abs(torch.sum(rel_robot * v_lat, dim=-1))

        yaw_err = torch.atan2(torch.sin(robot_yaw - pallet_yaw), torch.cos(robot_yaw - pallet_yaw))
        yaw_err_deg = torch.abs(yaw_err) * (180.0 / math.pi)

        rel_tip = tip[:, :2] - pallet_pos[:, :2]
        s_tip = torch.sum(rel_tip * u_in, dim=-1)
        s_front = -0.5 * cfg.pallet_depth_m
        insert_depth = torch.clamp(s_tip - s_front, min=0.0)
        insert_norm = torch.clamp(insert_depth / (cfg.pallet_depth_m + 1e-6), 0.0, 1.0)
        self._last_insert_depth = insert_depth.detach()

        self._cached_insert_norm = insert_norm.detach()
        self._cached_y_err = y_err.detach()
        self._cached_yaw_err_deg = yaw_err_deg.detach()

        r_d = torch.norm(tip[:, :2] - pallet_pos[:, :2], dim=-1)
        r_cd, r_cpsi = closest_point_on_path(tip[:, :2], self._ref_waypoints, self._ref_tangents, robot_yaw)

        r_g = (
            (insert_norm >= cfg.goal_insert_fraction)
            & (y_err <= cfg.goal_max_lateral_m)
            & (yaw_err_deg <= cfg.goal_max_yaw_deg)
        )

        R_pos = (
            cfg.alpha_1 / (r_d + cfg.eps_d)
            + cfg.alpha_2 / (r_cd + cfg.eps_cd)
            + cfg.alpha_3 / (r_cpsi + cfg.eps_cpsi)
            + cfg.alpha_4 * r_g.float()
        )

        pallet_vel = torch.norm(self.pallet.data.root_lin_vel_w[:, :2], dim=-1)
        r_p = -(pallet_vel > cfg.pallet_vel_thresh).float()

        robot_speed = torch.norm(self.robot.data.root_lin_vel_w[:, :2], dim=-1)
        excess = torch.clamp(robot_speed - cfg.speed_thresh, min=0.0)
        r_v = -(excess ** 2)

        r_a_raw = -((self.actions - self._prev_actions) ** 2).sum(dim=-1)
        r_a = torch.where(self._is_first_step, torch.zeros_like(r_a_raw), r_a_raw)

        r_ini = -((robot_speed < cfg.idle_speed_thresh) & (r_d > cfg.idle_dist_thresh)).float()

        R_neg = cfg.alpha_5 * r_p + cfg.alpha_6 * r_v + cfg.alpha_7 * r_a + cfg.alpha_8 * r_ini

        time_out = self.episode_length_buf >= self.max_episode_length - 1
        success = r_g
        r_terminal = torch.where(time_out & ~success, torch.full_like(r_d, cfg.rew_timeout), torch.zeros_like(r_d))

        rew = R_pos + R_neg + r_terminal

        self._prev_actions = self.actions.clone()
        self._is_first_step[:] = False

        if "log" not in self.extras:
            self.extras["log"] = {}
        log = self.extras["log"]
        log["r_pos/r_d_inv"] = (cfg.alpha_1 / (r_d + cfg.eps_d)).mean()
        log["r_pos/r_cd_inv"] = (cfg.alpha_2 / (r_cd + cfg.eps_cd)).mean()
        log["r_pos/r_cpsi_inv"] = (cfg.alpha_3 / (r_cpsi + cfg.eps_cpsi)).mean()
        log["r_pos/r_g"] = r_g.float().mean()
        log["r_neg/r_p"] = (cfg.alpha_5 * r_p).mean()
        log["r_neg/r_v"] = (cfg.alpha_6 * r_v).mean()
        log["r_neg/r_a"] = (cfg.alpha_7 * r_a).mean()
        log["r_neg/r_ini"] = (cfg.alpha_8 * r_ini).mean()
        log["err/r_d_mean"] = r_d.mean()
        log["err/r_cd_mean"] = r_cd.mean()
        log["err/r_cpsi_mean"] = r_cpsi.mean()
        log["err/y_err_mean"] = y_err.mean()
        log["err/yaw_deg_mean"] = yaw_err_deg.mean()
        log["err/insert_norm_mean"] = insert_norm.mean()
        log["diag/r_d_min"] = r_d.min()
        log["diag/r_cd_min"] = r_cd.min()
        log["diag/r_cpsi_min"] = r_cpsi.min()
        log["diag/pallet_vel_max"] = pallet_vel.max()
        log["diag/robot_speed_mean"] = robot_speed.mean()
        log["term/frac_success"] = success.float().mean()
        log["term/frac_timeout"] = time_out.float().mean()

        return rew

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        success = (
            (self._cached_insert_norm >= self.cfg.goal_insert_fraction)
            & (self._cached_y_err <= self.cfg.goal_max_lateral_m)
            & (self._cached_yaw_err_deg <= self.cfg.goal_max_yaw_deg)
        )

        q = self.robot.data.root_quat_w
        w, x, y, z = q.unbind(-1)
        sinr_cosp = 2.0 * (w * x + y * z)
        cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
        roll = torch.atan2(sinr_cosp, cosr_cosp)
        sinp = 2.0 * (w * y - z * x)
        pitch = torch.asin(torch.clamp(sinp, -1.0, 1.0))
        tipped = (torch.abs(roll) > self.cfg.max_roll_pitch_rad) | (torch.abs(pitch) > self.cfg.max_roll_pitch_rad)

        if "log" not in self.extras:
            self.extras["log"] = {}
        self.extras["log"]["term/frac_tipped"] = tipped.float().mean()

        terminated = success | tipped
        return terminated, time_out

    # ---------------------------
    # Reset
    # ---------------------------
    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        super()._reset_idx(env_ids)

        self._prev_actions[env_ids] = 0.0
        self._last_insert_depth[env_ids] = 0.0
        self._is_first_step[env_ids] = True
        self._lift_pos_target[env_ids] = 0.0
        self._cached_insert_norm[env_ids] = 0.0
        self._cached_y_err[env_ids] = 0.0
        self._cached_yaw_err_deg[env_ids] = 0.0

        pallet_pos = torch.tensor(self.cfg.pallet_cfg.init_state.pos, device=self.device).repeat(len(env_ids), 1)
        pallet_quat = torch.tensor(self.cfg.pallet_cfg.init_state.rot, device=self.device).repeat(len(env_ids), 1)
        self._write_root_pose(self.pallet, pallet_pos, pallet_quat, env_ids)

        x = sample_uniform(-4.0, -2.5, (len(env_ids), 1), device=self.device)
        y = sample_uniform(-0.6, 0.6, (len(env_ids), 1), device=self.device)
        z = torch.full((len(env_ids), 1), 0.03, device=self.device)
        yaw = sample_uniform(-0.25, 0.25, (len(env_ids), 1), device=self.device)

        pos = torch.cat([x, y, z], dim=1)
        half = yaw * 0.5
        quat = torch.cat([torch.cos(half), torch.zeros_like(half), torch.zeros_like(half), torch.sin(half)], dim=1)

        self._write_root_pose(self.robot, pos, quat, env_ids)

        zeros3 = torch.zeros((len(env_ids), 3), device=self.device)
        self._write_root_vel(self.robot, zeros3, zeros3, env_ids)

        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        joint_vel = torch.zeros_like(joint_pos)
        self._write_joint_state(self.robot, joint_pos, joint_vel, env_ids)

        self._fork_tip_z0[env_ids] = z.squeeze(-1)

        yaw_flat = yaw.squeeze(-1)
        tip_x0 = x.squeeze(-1) + self._fork_forward_offset * torch.cos(yaw_flat)
        tip_y0 = y.squeeze(-1) + self._fork_forward_offset * torch.sin(yaw_flat)
        start_xy = torch.stack([tip_x0, tip_y0], dim=-1)

        pallet_yaw_reset = _quat_to_yaw(pallet_quat)
        cp_p = torch.cos(pallet_yaw_reset)
        sp_p = torch.sin(pallet_yaw_reset)
        s_front = -0.5 * self.cfg.pallet_depth_m
        end_xy = pallet_pos[:, :2] + s_front * torch.stack([cp_p, sp_p], dim=-1)

        wps, tans = generate_bezier_path(
            start_xy,
            yaw_flat,
            end_xy,
            pallet_yaw_reset,
            num_points=self.cfg.ref_num_waypoints,
            ctrl_scale=self.cfg.bezier_ctrl_scale,
        )
        self._ref_waypoints[env_ids] = wps
        self._ref_tangents[env_ids] = tans

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
