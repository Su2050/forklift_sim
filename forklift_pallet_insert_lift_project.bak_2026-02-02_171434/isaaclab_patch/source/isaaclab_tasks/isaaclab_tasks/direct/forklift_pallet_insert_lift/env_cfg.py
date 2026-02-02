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
    """Configuration for the Forklift Pallet Insert+Lift environment (direct workflow)."""

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

    # pallet geometry assumptions (Euro pallet default)
    pallet_depth_m: float = 1.2

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

    # reward scales
    rew_progress = 2.0
    rew_align = -1.0
    rew_yaw = -0.2
    rew_lift = 1.0
    rew_success = 10.0
    rew_action_l2 = -0.01

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
            pos=(-2.0, 0.0, 0.03),
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
                effort_limit=500.0,
                stiffness=2000.0,
                damping=200.0,
            ),
        },
    )

    # pallet cfg (kinematic fixed pallet for stable first learning)
    pallet_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Pallet",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Pallet/pallet.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                kinematic_enabled=True,
                disable_gravity=True,
                max_depenetration_velocity=1.0,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    # ground
    ground_cfg: GroundPlaneCfg = GroundPlaneCfg()
