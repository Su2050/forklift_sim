from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import torch

import re
from pxr import UsdPhysics, PhysxSchema

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




# Isaac Sim's pallet.usd is often shipped as a pure visual prop without physics APIs.
# IsaacLab's RigidObject wrapper requires a prim with USD RigidBodyAPI.
_PALLET_ROOT_RE = re.compile(r"^/World/envs/env_\d+/Pallet$")


def _force_pallet_rigid_body(
    stage,
    *,
    rigid_body_enabled: bool = True,
    kinematic_enabled: bool = True,
    disable_gravity: bool = True,
    max_depenetration_velocity: float = 1.0,
) -> int:
    """Apply minimal physics APIs to Pallet prims so RigidObject can initialize.

    Some asset USDs (e.g. pallet.usd) ship as visual props without USD physics APIs.
    IsaacLab's RigidObject requires at least one prim with USD RigidBodyAPI.

    Returns number of pallet root prims patched.
    """
    count = 0
    for prim in stage.Traverse():
        path = prim.GetPath().pathString
        if not _PALLET_ROOT_RE.match(path):
            continue

        # Apply USD physics APIs (idempotent)
        UsdPhysics.RigidBodyAPI.Apply(prim)
        PhysxSchema.PhysxRigidBodyAPI.Apply(prim)

        # Set attributes
        rb = UsdPhysics.RigidBodyAPI(prim)
        rb.CreateRigidBodyEnabledAttr().Set(bool(rigid_body_enabled))
        rb.CreateKinematicEnabledAttr().Set(bool(kinematic_enabled))

        prb = PhysxSchema.PhysxRigidBodyAPI(prim)
        prb.CreateDisableGravityAttr().Set(bool(disable_gravity))
        prb.CreateMaxDepenetrationVelocityAttr().Set(float(max_depenetration_velocity))

        count += 1
    return count

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
        self._lift_id, _ = self.robot.find_joints(["lift_joint"], preserve_order=True)
        self._lift_id = self._lift_id[0]

        # buffers
        self.actions = torch.zeros((self.num_envs, self.cfg.action_space), device=self.device)
        self._last_insert_depth = torch.zeros((self.num_envs,), device=self.device)
        self._fork_tip_z0 = torch.zeros((self.num_envs,), device=self.device)
        self._hold_counter = torch.zeros((self.num_envs,), dtype=torch.int32, device=self.device)

        # derived constants
        self._pallet_front_x = self.cfg.pallet_cfg.init_state.pos[0] + self.cfg.pallet_depth_m * 0.5
        self._insert_thresh = self.cfg.insert_fraction * self.cfg.pallet_depth_m
        # how many control steps to hold success
        ctrl_dt = self.cfg.sim.dt * self.cfg.decimation
        self._hold_steps = max(1, int(self.cfg.hold_time_s / ctrl_dt))

        # convenience references to data tensors
        self._joint_pos = self.robot.data.joint_pos
        self._joint_vel = self.robot.data.joint_vel

    # ---------------------------
    # Scene setup
    # ---------------------------
    def _setup_scene(self):
        # assets
        self.robot = Articulation(self.cfg.robot_cfg)
        self.pallet = RigidObject(self.cfg.pallet_cfg)

        # ground
        spawn_ground_plane(prim_path="/World/ground", cfg=self.cfg.ground_cfg)

        # clone envs
        self.scene.clone_environments(copy_from_source=False)

        # ensure pallet prims have RigidBodyAPI before sim.reset() initializes assets
_force_pallet_rigid_body(
    self.sim.stage,
    rigid_body_enabled=getattr(getattr(self.cfg.pallet_cfg.spawn, "rigid_props", None), "rigid_body_enabled", True),
    kinematic_enabled=getattr(getattr(self.cfg.pallet_cfg.spawn, "rigid_props", None), "kinematic_enabled", True),
    disable_gravity=getattr(getattr(self.cfg.pallet_cfg.spawn, "rigid_props", None), "disable_gravity", True),
    max_depenetration_velocity=getattr(getattr(self.cfg.pallet_cfg.spawn, "rigid_props", None), "max_depenetration_velocity", 1.0),
)

        # collision filtering (needed for CPU sim)
        if getattr(self.device, "type", str(self.device)) == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])

        # add to scene
        self.scene.articulations["robot"] = self.robot
        self.scene.rigid_objects["pallet"] = self.pallet

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

        # steering: position targets (symmetric)
        self.robot.set_joint_position_target(steer.unsqueeze(-1).repeat(1, len(self._rotator_ids)), joint_ids=self._rotator_ids)

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
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        # compute fork tip and insertion depth
        tip = self._compute_fork_tip()
        insert_depth = torch.clamp(tip[:, 0] - self._pallet_front_x, min=0.0)

        # progress is delta insertion depth
        progress = insert_depth - self._last_insert_depth
        self._last_insert_depth = insert_depth.detach()

        # alignment penalties: lateral error and yaw error
        root_pos = self.robot.data.root_pos_w
        pallet_pos = self.pallet.data.root_pos_w
        lateral_err = torch.abs(pallet_pos[:, 1] - root_pos[:, 1])

        yaw = _quat_to_yaw(self.robot.data.root_quat_w)
        pallet_yaw = _quat_to_yaw(self.pallet.data.root_quat_w)
        yaw_err = torch.abs((pallet_yaw - yaw + math.pi) % (2 * math.pi) - math.pi)

        # lift progress (delta z from baseline)
        lift_delta = tip[:, 2] - self._fork_tip_z0

        # success condition
        inserted_enough = insert_depth >= self._insert_thresh
        aligned_enough = (lateral_err <= self.cfg.max_lateral_err_m) & (yaw_err <= math.radians(self.cfg.max_yaw_err_deg))
        lifted_enough = lift_delta >= self.cfg.lift_delta_m
        success_now = inserted_enough & aligned_enough & lifted_enough

        # hold counter
        self._hold_counter = torch.where(success_now, self._hold_counter + 1, torch.zeros_like(self._hold_counter))
        success = self._hold_counter >= self._hold_steps

        # reward components
        rew = torch.zeros((self.num_envs,), device=self.device)
        rew += self.cfg.rew_progress * progress
        rew += self.cfg.rew_align * lateral_err
        rew += self.cfg.rew_yaw * yaw_err
        rew += self.cfg.rew_lift * torch.clamp(lift_delta, min=0.0)
        rew += self.cfg.rew_action_l2 * (self.actions**2).sum(dim=1)
        rew += torch.where(success, torch.full_like(rew, self.cfg.rew_success), torch.zeros_like(rew))

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
