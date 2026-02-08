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
    from pxr import UsdPhysics
    
    root_prim = stage.GetPrimAtPath(pallet_prim_path)
    if not root_prim.IsValid():
        print(f"[警告] 找不到托盘 prim: {pallet_prim_path}")
        return
    
    # 遍历托盘及其所有子 prim
    prims_to_process = [root_prim]
    for prim in root_prim.GetDescendants():
        prims_to_process.append(prim)
    
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
    from pxr import UsdPhysics, PhysxSchema, UsdGeom
    
    root_prim = stage.GetPrimAtPath(pallet_prim_path)
    if not root_prim.IsValid():
        print(f"[警告] 找不到托盘 prim: {pallet_prim_path}")
        return
    
    # 遍历托盘及其所有子 prim
    prims_to_process = [root_prim]
    for prim in root_prim.GetDescendants():
        prims_to_process.append(prim)
    
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
            convex_api.GetMaxConvexHullsAttr().Set(32)   # 最大凸体数
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
        # separate left/right rotator IDs for order-independent steering
        self._left_rotator_id, _ = self.robot.find_joints(["left_rotator_joint"], preserve_order=True)
        self._right_rotator_id, _ = self.robot.find_joints(["right_rotator_joint"], preserve_order=True)
        self._lift_id, _ = self.robot.find_joints(["lift_joint"], preserve_order=True)
        self._lift_id = self._lift_id[0]

        # buffers
        self.actions = torch.zeros((self.num_envs, self.cfg.action_space), device=self.device)
        self._last_insert_depth = torch.zeros((self.num_envs,), device=self.device)
        self._fork_tip_z0 = torch.zeros((self.num_envs,), device=self.device)
        self._hold_counter = torch.zeros((self.num_envs,), dtype=torch.int32, device=self.device)
        
        # S1.0g: 增量奖励缓存变量
        self._is_first_step = torch.ones((self.num_envs,), dtype=torch.bool, device=self.device)
        self._last_E_align = torch.zeros((self.num_envs,), device=self.device)
        self._last_dist_front = torch.zeros((self.num_envs,), device=self.device)
        self._last_lift_pos = torch.zeros((self.num_envs,), device=self.device)

        # derived constants
        self._pallet_front_x = self.cfg.pallet_cfg.init_state.pos[0] + self.cfg.pallet_depth_m * 0.5
        self._insert_thresh = self.cfg.insert_fraction * self.cfg.pallet_depth_m
        # how many control steps to hold success
        ctrl_dt = self.cfg.sim.dt * self.cfg.decimation
        self._hold_steps = max(1, int(self.cfg.hold_time_s / ctrl_dt))

        # convenience references to data tensors
        self._joint_pos = self.robot.data.joint_pos
        self._joint_vel = self.robot.data.joint_vel

    def _setup_pallet_physics(self):
        """在环境克隆前，强制设置托盘物理属性
        
        必须在 clone_environments() 之前调用，确保模板环境的设置能被克隆继承。
        """
        from pxr import UsdPhysics, PhysxSchema, UsdGeom
        
        stage = self.sim.stage

        # 诊断：修改前的 USD 状态（只看 env_0）
        diag_pallet_path = self.cfg.pallet_cfg.prim_path.replace("env_.*", "env_0")
        self._log_pallet_usd(stage, diag_pallet_path, label="修改前")
        self._log_pallet_physx(label="修改前")
        
        # 只修改模板环境（env_0），让克隆继承
        pallet_path = self.cfg.pallet_cfg.prim_path.replace("env_.*", "env_0")
        root_prim = stage.GetPrimAtPath(pallet_path)
        if not root_prim.IsValid():
            print(f"[警告] 找不到托盘 prim: {pallet_path}")
            return

        # 遍历托盘及其所有子 prim
        prims_to_process = [root_prim]
        for prim in root_prim.GetDescendants():
            prims_to_process.append(prim)
            
        for prim in prims_to_process:
            # 1. 强制设置为动态刚体
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                rb_api = UsdPhysics.RigidBodyAPI(prim)
                rb_api.GetRigidBodyEnabledAttr().Set(True)
                rb_api.GetKinematicEnabledAttr().Set(False)
                print(f"[信息] 已设置 {prim.GetPath()} 为动态刚体")
            
            # 2. 设置凸分解碰撞体
            has_mesh = prim.IsA(UsdGeom.Mesh)
            has_collision = prim.HasAPI(UsdPhysics.CollisionAPI)
            
            if has_mesh or has_collision:
                mesh_collision_api = UsdPhysics.MeshCollisionAPI.Apply(prim)
                mesh_collision_api.GetApproximationAttr().Set("convexDecomposition")
                
                convex_api = PhysxSchema.PhysxConvexDecompositionCollisionAPI.Apply(prim)
                convex_api.GetMaxConvexHullsAttr().Set(32)
                convex_api.GetHullVertexLimitAttr().Set(64)
        
        print("[信息] 托盘物理属性已设置完成（模板环境）")
        # 诊断：修改后的 USD 状态（只看 env_0）
        self._log_pallet_usd(stage, diag_pallet_path, label="修改后")
        self._log_pallet_physx(label="修改后")

    def _log_pallet_usd(self, stage, pallet_path: str, label: str):
        """打印托盘 USD 物理属性诊断信息（仅用于排查问题）。"""
        from pxr import UsdPhysics, UsdGeom

        root_prim = stage.GetPrimAtPath(pallet_path)
        if not root_prim.IsValid():
            print(f"[诊断] {label} 找不到托盘 prim: {pallet_path}")
            return

        prims_to_process = [root_prim]
        for prim in root_prim.GetDescendants():
            prims_to_process.append(prim)

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
        # assets
        self.robot = Articulation(self.cfg.robot_cfg)
        self.pallet = RigidObject(self.cfg.pallet_cfg)

        # 在克隆之前修改模板环境（env_0）的托盘物理属性
        self._setup_pallet_physics()

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
        # left rotator needs opposite sign due to mirrored joint axis in USD
        steer_left = -steer
        steer_right = steer
        self.robot.set_joint_position_target(steer_left.unsqueeze(-1), joint_ids=self._left_rotator_id)
        self.robot.set_joint_position_target(steer_right.unsqueeze(-1), joint_ids=self._right_rotator_id)

        # lift: velocity target
        self.robot.set_joint_velocity_target(lift_v.unsqueeze(-1), joint_ids=[self._lift_id])

        # write to sim
        self.robot.write_data_to_sim()

    # ---------------------------
    # Observations / Rewards / Dones
    # ---------------------------
    def _compute_fork_tip(self) -> torch.Tensor:
        """Estimate fork tip position as the body with max projection along robot forward axis."""
        root_pos = self.robot.data.root_pos_w  # (N,3)
        root_quat = self.robot.data.root_quat_w  # (N,4)
        yaw = _quat_to_yaw(root_quat)
        # forward unit vector in world (x-forward in robot frame)
        fwd = torch.stack([torch.cos(yaw), torch.sin(yaw), torch.zeros_like(yaw)], dim=-1)  # (N,3)

        body_pos = self.robot.data.body_pos_w  # (N,B,3)
        rel = body_pos - root_pos.unsqueeze(1)  # (N,B,3)
        proj = (rel * fwd.unsqueeze(1)).sum(-1)  # (N,B)
        idx = torch.argmax(proj, dim=1)  # (N,)
        tip = body_pos[torch.arange(self.num_envs, device=self.device), idx]  # (N,3)
        return tip

    def _get_observations(self) -> dict[str, torch.Tensor]:
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
        tip = self._compute_fork_tip()
        insert_depth = torch.clamp(tip[:, 0] - self._pallet_front_x, min=0.0)
        insert_norm = (insert_depth / (self.cfg.pallet_depth_m + 1e-6)).unsqueeze(-1)

        obs = torch.cat(
            [
                d_xy_r,  # 2
                cos_dyaw.unsqueeze(-1), sin_dyaw.unsqueeze(-1),  # 2
                v_xy_r,  # 2
                yaw_rate,  # 1
                lift_pos, lift_vel,  # 2
                insert_norm,  # 1
                self.actions,  # 3
            ],
            dim=-1,
        )
        # Isaac Lab direct workflow expects a dict with at least the "policy" key.
        # For asymmetric actor-critic, add a "critic" key with privileged observations.
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        """S1.0g 距离自适应奖励函数

        用 smoothstep(dist_front) 生成 w_close/w_far，
        让奖励系数随距离自动从"远处探索(≈S0.4)"过渡到"近处收敛(≈S0.7)"。

        阶段1（w_ready 门控）：接近（对齐好了才给）+ 对齐改善(delta) + 绝对对齐惩罚 + 远距离前进速度奖励
        阶段2（w_ready 门控）：插入进度（严格正比于对齐质量）
        阶段3（w_lift 门控）：举升奖励（w_lift = w_lift_base * w_ready，已包含对齐门控）+ 距离自适应空举惩罚（只看插入深度）
        
        S1.0g 核心修复：修复 S1.0f 的举升奖励机制（移除 r_lift 中多余的 w_ready，让 pen_premature 只看插入深度），
        解决"永远学不会举升"的问题（避免 w_ready^2 抑制奖励和惩罚过重）
        """
        # ========== Step 0: 基础量计算 ==========
        tip = self._compute_fork_tip()
        insert_depth = torch.clamp(tip[:, 0] - self._pallet_front_x, min=0.0)
        insert_norm = insert_depth / (self.cfg.pallet_depth_m + 1e-6)

        # 距离：货叉到托盘前面（正值=在托盘前面）
        dist_front = self._pallet_front_x - tip[:, 0]
        dist_front_clamped = torch.clamp(dist_front, min=0.0)

        # 对齐误差
        root_pos = self.robot.data.root_pos_w
        pallet_pos = self.pallet.data.root_pos_w
        y_err = torch.abs(pallet_pos[:, 1] - root_pos[:, 1])

        yaw = _quat_to_yaw(self.robot.data.root_quat_w)
        pallet_yaw = _quat_to_yaw(self.pallet.data.root_quat_w)
        dyaw = (pallet_yaw - yaw + math.pi) % (2 * math.pi) - math.pi
        yaw_err_deg = torch.abs(dyaw) * (180.0 / math.pi)
        yaw_err_rad = torch.abs(dyaw)

        # ========== Step 1: 软门控权重 ==========
        E_align = y_err / self.cfg.lat_ready_m + yaw_err_deg / self.cfg.yaw_ready_deg

        # S1.0f: 恢复乘法门控 w_ready（回归 S0），必须同时满足横向和偏航要求
        w_lat = torch.clamp(1.0 - y_err / self.cfg.lat_ready_m, min=0.0, max=1.0)
        w_yaw = torch.clamp(1.0 - yaw_err_deg / self.cfg.yaw_ready_deg, min=0.0, max=1.0)
        w_ready = w_lat * w_yaw  # 乘法门控，任一维度达到阈值，w_ready = 0

        # S1.0e: w_lift 也添加 w_ready 门控，彻底切断斜怼+举升收益路径
        w_lift_base = torch.clamp(
            (insert_norm - self.cfg.insert_gate_norm) / self.cfg.insert_ramp_norm,
            min=0.0, max=1.0,
        )
        w_lift = w_lift_base * w_ready  # 添加对齐质量门控

        # ========== S1.0: 距离自适应权重 ==========
        w_close = smoothstep(
            (self.cfg.d_far - dist_front_clamped) / (self.cfg.d_far - self.cfg.d_close)
        )
        w_far = 1.0 - w_close

        # 动态系数插值（远≈S0.4, 近≈S0.7）
        k_approach = self.cfg.k_app_far * w_far + self.cfg.k_app_close * w_close
        k_align = self.cfg.k_align_far * w_far + self.cfg.k_align_close * w_close
        k_insert = self.cfg.k_ins_far * w_far + self.cfg.k_ins_close * w_close
        k_premature = self.cfg.k_pre_far * w_far + self.cfg.k_pre_close * w_close

        # ========== Step 2: 增量计算 ==========
        delta_E_align = self._last_E_align - E_align
        self._last_E_align = E_align.detach()

        delta_dist = self._last_dist_front - dist_front_clamped
        self._last_dist_front = dist_front_clamped.detach()

        progress = insert_depth - self._last_insert_depth
        self._last_insert_depth = insert_depth.detach()

        lift_delta = tip[:, 2] - self._fork_tip_z0
        delta_lift = lift_delta - self._last_lift_pos
        self._last_lift_pos = lift_delta.detach()

        # 机器人坐标系前进速度（用于 r_forward）
        root_lin_vel = self.robot.data.root_lin_vel_w
        v_xy_w = root_lin_vel[:, :2]
        R = _yaw_to_mat2(-yaw)
        v_xy_r = torch.einsum("nij,nj->ni", R, v_xy_w)
        v_xy_r_x = v_xy_r[:, 0]

        # ========== 阶段1: 接近 + 对齐 + 远距前进速度 ==========
        rew = torch.zeros((self.num_envs,), device=self.device)

        # S1.0f: 恢复 r_approach 的 w_ready 门控（回归 S0），对齐好了才给接近奖励
        r_approach = k_approach * w_ready * delta_dist
        r_approach = torch.where(self._is_first_step, torch.zeros_like(r_approach), r_approach)
        rew += r_approach

        r_align = k_align * delta_E_align
        r_align = torch.where(self._is_first_step, torch.zeros_like(r_align), r_align)
        rew += r_align

        # 远距离前进速度奖励（只在远处生效，近处自动消失）
        r_forward = self.cfg.k_forward * w_far * torch.clamp(v_xy_r_x, min=0.0)
        rew += r_forward

        # ========== S1.0c 绝对对齐惩罚（近处持续惩罚未对齐状态） ==========
        pen_align_abs = -self.cfg.k_align_abs * w_close * torch.clamp(E_align, 0.0, 2.0)
        rew += pen_align_abs

        # ========== 阶段2: 插入（S1.0c: w_ready 直接门控，对齐越好→插入奖励越大） ==========
        r_insert = k_insert * w_ready * torch.clamp(progress, min=0.0)
        rew += r_insert

        # ========== 阶段3: 举升（S1.0g: 修复双重门控 - w_lift 已包含 w_ready，不再重复） ==========
        r_lift = self.cfg.k_lift * w_lift * torch.clamp(delta_lift, min=0.0)
        r_lift = torch.where(self._is_first_step, torch.zeros_like(r_lift), r_lift)
        rew += r_lift

        # 空举惩罚（S1.0g: 只看插入深度，不掺对齐，避免"插入够深但对齐略差"时被重罚）
        pen_premature = -k_premature * (1.0 - w_lift_base) * torch.clamp(delta_lift, min=0.0)
        pen_premature = torch.where(self._is_first_step, torch.zeros_like(pen_premature), pen_premature)
        rew += pen_premature

        # ========== 常驻惩罚 ==========
        rew += self.cfg.rew_action_l2 * (self.actions ** 2).sum(dim=1)
        rew += self.cfg.rew_time_penalty

        dist_far_threshold = 2.0
        dist_excess = torch.clamp(dist_front_clamped - dist_far_threshold, min=0.0)
        pen_dist_far = -self.cfg.k_dist_far * dist_excess
        rew += pen_dist_far

        # ========== 成功判定与奖励 ==========
        inserted_enough = insert_depth >= self._insert_thresh
        aligned_enough = (y_err <= self.cfg.max_lateral_err_m) & (
            yaw_err_rad <= math.radians(self.cfg.max_yaw_err_deg)
        )
        lifted_enough = lift_delta >= self.cfg.lift_delta_m
        success_now = inserted_enough & aligned_enough & lifted_enough

        self._hold_counter = torch.where(
            success_now, self._hold_counter + 1, torch.zeros_like(self._hold_counter)
        )
        success = self._hold_counter >= self._hold_steps

        time_ratio = self.episode_length_buf.float() / (self.max_episode_length + 1e-6)
        time_bonus = self.cfg.rew_success_time * (1.0 - time_ratio)
        success_reward = torch.where(
            success,
            self.cfg.rew_success + time_bonus,
            torch.zeros_like(rew),
        )
        rew += success_reward

        # 清除首步标记
        self._is_first_step[:] = False

        # ========== S1.0c 日志输出 ==========
        if "log" not in self.extras:
            self.extras["log"] = {}

        # 核心诊断
        self.extras["log"]["s0/E_align"] = E_align.mean()
        self.extras["log"]["s0/w_ready"] = w_ready.mean()
        self.extras["log"]["s0/w_lift"] = w_lift.mean()
        self.extras["log"]["s0/w_close"] = w_close.mean()
        self.extras["log"]["s0/w_far"] = w_far.mean()

        # 奖励分量
        self.extras["log"]["s0/r_align"] = r_align.mean()
        self.extras["log"]["s0/r_approach"] = r_approach.mean()
        self.extras["log"]["s0/r_forward"] = r_forward.mean()
        self.extras["log"]["s0/r_insert"] = r_insert.mean()
        self.extras["log"]["s0/r_lift"] = r_lift.mean()
        self.extras["log"]["s0/pen_align_abs"] = pen_align_abs.mean()
        self.extras["log"]["s0/pen_premature"] = pen_premature.mean()
        self.extras["log"]["s0/pen_dist_far"] = pen_dist_far.mean()
        self.extras["log"]["s0/time_penalty"] = self.cfg.rew_time_penalty
        self.extras["log"]["s0/time_bonus"] = torch.where(
            success, time_bonus, torch.zeros_like(time_bonus)
        ).mean()
        self.extras["log"]["s0/success"] = success_reward.mean()

        # 有效系数（观察距离自适应是否生效）
        self.extras["log"]["s0/k_approach_eff"] = k_approach.mean()
        self.extras["log"]["s0/k_align_eff"] = k_align.mean()
        self.extras["log"]["s0/k_premature_eff"] = k_premature.mean()

        # 误差分布
        self.extras["log"]["err/lateral_mean"] = y_err.mean()
        self.extras["log"]["err/yaw_deg_mean"] = yaw_err_deg.mean()
        self.extras["log"]["err/insert_norm_mean"] = insert_norm.mean()
        self.extras["log"]["err/lift_delta_mean"] = lift_delta.mean()
        self.extras["log"]["err/dist_front_mean"] = dist_front_clamped.mean()

        # 阶段命中率
        self.extras["log"]["phase/frac_inserted"] = inserted_enough.float().mean()
        self.extras["log"]["phase/frac_aligned"] = aligned_enough.float().mean()
        self.extras["log"]["phase/frac_lifted"] = lifted_enough.float().mean()
        self.extras["log"]["phase/frac_success_now"] = success_now.float().mean()
        self.extras["log"]["phase/frac_success"] = success.float().mean()
        self.extras["log"]["phase/hold_counter_max"] = self._hold_counter.float().max()

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
        time_out_now = self.episode_length_buf >= self.max_episode_length - 1
        self.extras["log"]["term/frac_tipped"] = tipped.float().mean()
        self.extras["log"]["term/frac_timeout"] = time_out_now.float().mean()

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

        terminated = tipped | success
        return terminated, time_out

    # ---------------------------
    # Reset
    # ---------------------------
    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        # reset counters
        self._last_insert_depth[env_ids] = 0.0
        self._hold_counter[env_ids] = 0
        self._is_first_step[env_ids] = True

        # reset pallet (fixed pose; optional: you can randomize here)
        pallet_pos = torch.tensor(self.cfg.pallet_cfg.init_state.pos, device=self.device).repeat(len(env_ids), 1)
        pallet_quat = torch.tensor(self.cfg.pallet_cfg.init_state.rot, device=self.device).repeat(len(env_ids), 1)
        self._write_root_pose(self.pallet, pallet_pos, pallet_quat, env_ids)

        # randomize robot pose around pallet, facing +x
        x = sample_uniform(-2.5, -1.0, (len(env_ids), 1), device=self.device)
        y = sample_uniform(-0.6, 0.6, (len(env_ids), 1), device=self.device)
        z = torch.full((len(env_ids), 1), 0.03, device=self.device)
        yaw = sample_uniform(-0.25, 0.25, (len(env_ids), 1), device=self.device)

        pos = torch.cat([x, y, z], dim=1)
        # yaw quaternion (w,x,y,z)
        half = yaw * 0.5
        quat = torch.cat([torch.cos(half), torch.zeros_like(half), torch.zeros_like(half), torch.sin(half)], dim=1)

        self._write_root_pose(self.robot, pos, quat, env_ids)

        # reset velocities to zero
        zeros3 = torch.zeros((len(env_ids), 3), device=self.device)
        self._write_root_vel(self.robot, zeros3, zeros3, env_ids)

        # reset joints (lift down, wheels zero, steering zero)
        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        joint_vel = torch.zeros_like(joint_pos)
        self._write_joint_state(self.robot, joint_pos, joint_vel, env_ids)

        # one sim update to populate buffers
        self.scene.write_data_to_sim()
        self.sim.reset()
        self.scene.update(self.cfg.sim.dt)

        # baseline fork tip height
        tip = self._compute_fork_tip()
        self._fork_tip_z0[env_ids] = tip[:, 2][env_ids]

        # S1.0g: 初始化增量奖励缓存为当前状态（防止首步异常奖励）
        dist_front_reset = torch.clamp(self._pallet_front_x - x.squeeze(-1), min=0.0)
        self._last_dist_front[env_ids] = dist_front_reset

        y_err_reset = torch.abs(y.squeeze(-1))
        yaw_err_deg_reset = torch.abs(yaw.squeeze(-1)) * (180.0 / math.pi)
        E_align_reset = y_err_reset / self.cfg.lat_ready_m + yaw_err_deg_reset / self.cfg.yaw_ready_deg
        self._last_E_align[env_ids] = E_align_reset

        lift_delta_reset = tip[:, 2][env_ids] - self._fork_tip_z0[env_ids]
        self._last_lift_pos[env_ids] = lift_delta_reset

        # reset robot actuators
        self.robot.reset(env_ids)

    # ---------------------------
    # Compatibility helpers (API name differences across versions)
    # ---------------------------
    def _write_root_pose(self, asset, pos, quat, env_ids):
        if hasattr(asset, "write_root_pose_to_sim"):
            asset.write_root_pose_to_sim(pos, quat, env_ids)
        elif hasattr(asset, "write_root_state_to_sim"):
            # some versions use a single tensor for root_state
            root_state = torch.zeros((len(env_ids), 13), device=self.device)
            root_state[:, 0:3] = pos
            root_state[:, 3:7] = quat
            asset.write_root_state_to_sim(root_state, env_ids)
        else:
            raise AttributeError("Asset has no known root pose writer.")

    def _write_root_vel(self, asset, lin_vel, ang_vel, env_ids):
        if hasattr(asset, "write_root_velocity_to_sim"):
            asset.write_root_velocity_to_sim(lin_vel, ang_vel, env_ids)
        elif hasattr(asset, "write_root_state_to_sim"):
            # if only root_state is supported, caller should set full state; keep as no-op
            pass
        else:
            raise AttributeError("Asset has no known root velocity writer.")

    def _write_joint_state(self, articulation, joint_pos, joint_vel, env_ids):
        if hasattr(articulation, "write_joint_state_to_sim"):
            articulation.write_joint_state_to_sim(joint_pos, joint_vel, env_ids)
        elif hasattr(articulation, "write_joint_pos_to_sim") and hasattr(articulation, "write_joint_vel_to_sim"):
            articulation.write_joint_pos_to_sim(joint_pos, env_ids)
            articulation.write_joint_vel_to_sim(joint_vel, env_ids)
        else:
            raise AttributeError("Articulation has no known joint state writer.")
