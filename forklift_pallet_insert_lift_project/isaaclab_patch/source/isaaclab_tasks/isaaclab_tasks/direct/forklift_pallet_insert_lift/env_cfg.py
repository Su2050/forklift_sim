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
    """Configuration for the Forklift Pallet Insert+Lift environment (direct workflow).

    S1.0g: 修复 S1.0f 的举升奖励机制，解决"永远学不会举升"的问题。
    S1.0g 改动：
      - 修复 r_lift：移除多余的 w_ready，改为 r_lift = k_lift * w_lift * delta_lift（避免 w_ready^2 抑制奖励）
      - 修复 pen_premature：改用 w_lift_base，改为 pen_premature = -k_premature * (1.0 - w_lift_base) * delta_lift（只看插入深度，不掺对齐）
      - 保留 S1.0f 的改进：严格对齐阈值、乘法门控 w_ready、r_approach 门控、绝对对齐惩罚、距离自适应系数
    """

    # env
    decimation = 4
    episode_length_s = 12.0

    # actions: [drive, steer, lift]
    action_space = 3

    # observations: vector, see env._get_observations()
    observation_space = 14

    # no separate privileged state in this minimal patch
    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)

    # scene replication
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=128,
        env_spacing=6.0,
        replicate_physics=True,
        clone_in_fabric=True,
    )

    # assets
    forklift_usd_path: str = f"{ISAAC_NUCLEUS_DIR}/Robots/IsaacSim/ForkliftC/forklift_c.usd"
    pallet_usd_path: str = f"{ISAAC_NUCLEUS_DIR}/Props/Pallet/pallet.usd"

    # pallet geometry assumptions (Euro pallet default, scaled 4x)
    # 原始深度 1.2m * 4.0 = 4.8m
    pallet_depth_m: float = 4.8

    # KPI
    insert_fraction: float = 2.0 / 3.0
    lift_delta_m: float = 0.12
    hold_time_s: float = 1.0
    max_lateral_err_m: float = 0.03
    max_yaw_err_deg: float = 3.0

    # action limits (normalized actions in [-1, 1] are scaled by these)
    wheel_speed_rad_s: float = 20.0
    steer_angle_rad: float = 0.6
    lift_speed_m_s: float = 0.25

    # ---------- S1.0 奖励参数（距离自适应） ----------
    # 阶段门控阈值（S1.0f: 恢复严格阈值，回归 S0）
    lat_ready_m: float = 0.10  # S1.0f: 0.5→0.10，回归 S0 严格阈值
    yaw_ready_deg: float = 10.0  # S1.0f: 30.0→10.0，回归 S0 严格阈值
    d_safe_m: float = 0.7
    
    # 插入深度门控阈值
    insert_gate_norm: float = 0.60  # 插入深度达到60%后开始允许举升
    insert_ramp_norm: float = 0.10  # 线性区间宽度（60%→70%）

    # 距离阈值（用于距离自适应权重）
    d_far: float = 2.6   # 远端阈值
    d_close: float = 1.1  # 近端阈值

    # 距离自适应系数（远处值）
    k_app_far: float = 10.0
    k_align_far: float = 2.0
    k_ins_far: float = 8.0
    k_pre_far: float = 12.0
    k_dist_far: float = 0.3

    # 距离自适应系数（近处值）
    k_app_close: float = 8.0
    k_align_close: float = 10.0
    k_ins_close: float = 15.0
    k_pre_close: float = 5.0

    # 固定系数
    k_align_abs: float = 0.10  # S1.0d: 0.05→0.10，绝对对齐惩罚系数
    k_lift: float = 20.0  # 举升奖励系数
    k_forward: float = 0.02  # 远距离前进速度奖励系数

    # 常驻惩罚
    rew_action_l2: float = -0.01
    rew_time_penalty: float = -0.003

    # 成功奖励
    rew_success: float = 100.0
    rew_success_time: float = 30.0

    # termination thresholds
    max_roll_pitch_rad: float = 0.45  # ~25 deg
    max_time_s: float = episode_length_s

    # robot cfg (forklift_c joint naming as used in community examples)
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
            pos=(-6.0, 0.0, 0.03),  # 后移到 X=-6m，与放大后的托盘保持距离
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
            # rolling joints (velocity targets)
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
            # steering joints (position targets)
            "rotators": ImplicitActuatorCfg(
                joint_names_expr=["left_rotator_joint", "right_rotator_joint"],
                velocity_limit=10.0,
                effort_limit=300.0,
                stiffness=4000.0,
                damping=400.0,
            ),
            # lift joint (velocity targets)
            "lift": ImplicitActuatorCfg(
                joint_names_expr=["lift_joint"],
                velocity_limit=1.0,
                effort_limit=5000.0,   # 5000 N，确保能举起托盘
                stiffness=0.0,         # 速度控制模式，stiffness 必须为 0
                damping=1000.0,        # 阻尼控制速度响应
            ),
        },
    )

    # pallet cfg (dynamic rigid body for realistic physics interaction)
    # 修改说明：
    # 1. 从 kinematic 改为动态刚体，使托盘可以被叉车推动和举起
    # 2. 添加 scale=4.0 使托盘放大到与叉车货叉兼容的尺寸
    #    - 原始托盘插入孔宽度 ~228mm，货叉宽度 ~394mm
    #    - 放大 4x 后插入孔宽度 ~912mm，足够容纳货叉
    pallet_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Pallet",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Pallet/pallet.usd",
            scale=(4.0, 4.0, 4.0),  # 放大托盘使其与叉车货叉兼容
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                kinematic_enabled=False,  # 改为动态刚体，可被推动/举起
                disable_gravity=False,    # 恢复重力，托盘会落在地面上
                max_depenetration_velocity=1.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=30.0),  # 空托盘约 20-30 kg
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.30),  # 抬高到与货叉高度对齐（放大后托盘更高）
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    # ground
    ground_cfg: GroundPlaneCfg = GroundPlaneCfg()
