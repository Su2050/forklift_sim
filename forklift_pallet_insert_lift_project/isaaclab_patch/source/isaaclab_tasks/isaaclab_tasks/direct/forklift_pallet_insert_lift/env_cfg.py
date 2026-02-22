"""Forklift Pallet Insert environment configuration (S2.0a).

S2.0a: paper-style R+/R- reward with Bezier reference trajectory.
Approach + insertion only (no lift phase).
"""

from __future__ import annotations

from isaaclab.utils import configclass
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR


@configclass
class ForkliftPalletInsertLiftEnvCfg(DirectRLEnvCfg):
    """S2.0a configuration: paper-style reward, Bezier trajectory, insertion only."""

    # ===== 环境基础参数 =====
    decimation = 4
    episode_length_s = 36.0

    action_space = 3            # [drive, steer, lift] (lift disabled in _apply_action)
    observation_space = 15      # 15D vector (same as S1.0N)
    state_space = 0

    # ===== 仿真参数 =====
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)

    # ===== 场景 =====
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=128,
        env_spacing=6.0,
        replicate_physics=True,
        clone_in_fabric=False,
    )

    # ===== 资产路径 =====
    forklift_usd_path: str = f"{ISAAC_NUCLEUS_DIR}/Robots/IsaacSim/ForkliftC/forklift_c.usd"
    pallet_usd_path: str = f"{ISAAC_NUCLEUS_DIR}/Props/Pallet/pallet.usd"

    # ===== 托盘几何 =====
    pallet_depth_m: float = 2.16   # 1.2m x 1.8 scale

    # ===== 动作范围 =====
    wheel_speed_rad_s: float = 20.0
    steer_angle_rad: float = 0.6
    lift_speed_m_s: float = 0.5

    # ===== S2.0a: 论文式 R+/R- 奖励 =====
    # R+ = a1/(r_d+e) + a2/(r_cd+e) + a3/(r_cpsi+e) + a4*r_g
    # R- = a5*r_p + a6*r_v + a7*r_a + a8*r_ini

    alpha_1: float = 1.0       # 距离接近 (1/r_d)
    alpha_2: float = 0.5       # 轨迹跟踪 (1/r_cd)
    alpha_3: float = 0.3       # 朝向对齐 (1/r_cpsi)
    alpha_4: float = 50.0      # 到达目标 (r_g, 离散)

    alpha_5: float = 5.0       # 托盘被撞动 (r_p)
    alpha_6: float = 1.0       # 超速惩罚 (r_v)
    alpha_7: float = 0.1       # 动作抖动 (r_a)
    alpha_8: float = 0.5       # 起步卡死 (r_ini)

    eps_d: float = 0.05        # 1/r_d epsilon (m)
    eps_cd: float = 0.02       # 1/r_cd epsilon (m)
    eps_cpsi: float = 0.05     # 1/r_cpsi epsilon (rad)

    pallet_vel_thresh: float = 0.01   # m/s
    speed_thresh: float = 0.07        # m/s (论文值，可能需调高到 0.3~0.5)
    idle_speed_thresh: float = 0.05   # m/s
    idle_dist_thresh: float = 0.3     # m

    # ===== 参考轨迹（Bezier）=====
    ref_num_waypoints: int = 64
    bezier_ctrl_scale: float = 0.4

    # ===== 成功判定（无 lift，即时判定无 hold counter）=====
    goal_insert_fraction: float = 0.40
    goal_max_lateral_m: float = 0.15
    goal_max_yaw_deg: float = 5.0

    rew_timeout: float = -10.0

    # ===== 观测辅助参数（_get_observations 使用）=====
    y_err_obs_scale: float = 0.8
    lift_pos_scale: float = 1.0

    # ===== 终止阈值 =====
    max_roll_pitch_rad: float = 0.45  # ~25 deg
    max_time_s: float = episode_length_s

    # ===== 叉车配置 =====
    robot_cfg: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/IsaacSim/ForkliftC/forklift_c.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                max_linear_velocity=20.0,
                max_angular_velocity=20.0,
                max_depenetration_velocity=5.0,
                enable_gyroscopic_forces=True,
            ),
            mass_props=sim_utils.MassPropertiesCfg(density=3000.0),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=0,
                sleep_threshold=0.005,
                stabilization_threshold=0.001,
            ),
            activate_contact_sensors=False,
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(-3.5, 0.0, 0.03),
            rot=(1.0, 0.0, 0.0, 0.0),
            joint_pos={
                "left_front_wheel_joint": 0.0,
                "right_front_wheel_joint": 0.0,
                "left_rotator_joint": 0.0,
                "right_rotator_joint": 0.0,
                "left_back_wheel_joint": 0.0,
                "right_back_wheel_joint": 0.0,
                "lift_joint": 0.0,
            },
        ),
        actuators={
            "front_wheels": ImplicitActuatorCfg(
                joint_names_expr=["left_front_wheel_joint", "right_front_wheel_joint"],
                velocity_limit=40.0,
                effort_limit=200.0,
                stiffness=0.0,
                damping=100.0,
            ),
            "back_wheels": ImplicitActuatorCfg(
                joint_names_expr=["left_back_wheel_joint", "right_back_wheel_joint"],
                velocity_limit=40.0,
                effort_limit=200.0,
                stiffness=0.0,
                damping=50.0,
            ),
            "rotators": ImplicitActuatorCfg(
                joint_names_expr=["left_rotator_joint", "right_rotator_joint"],
                velocity_limit=10.0,
                effort_limit=300.0,
                stiffness=4000.0,
                damping=400.0,
            ),
            "lift": ImplicitActuatorCfg(
                joint_names_expr=["lift_joint"],
                velocity_limit=1.0,
                effort_limit=50000.0,
                stiffness=200000.0,
                damping=10000.0,
            ),
        },
    )

    # ===== 托盘配置（动态刚体）=====
    pallet_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Pallet",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Pallet/pallet.usd",
            scale=(1.8, 1.8, 1.8),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                kinematic_enabled=False,
                disable_gravity=False,
                max_depenetration_velocity=1.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=45.0),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
                contact_offset=0.02,
                rest_offset=0.005,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.15),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    # ===== 地面 =====
    ground_cfg: GroundPlaneCfg = GroundPlaneCfg()
