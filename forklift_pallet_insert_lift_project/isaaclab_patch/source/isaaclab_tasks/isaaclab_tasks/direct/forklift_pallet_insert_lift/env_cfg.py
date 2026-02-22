"""Forklift Pallet Insert+Lift 环境配置（DirectRLEnvCfg）。

此文件集中定义仿真环境的所有可调参数：环境步长、场景复制、资产配置、
奖励系数、任务 KPI 等。训练时由 `gym.register()` 中的
`env_cfg_entry_point` 自动加载并实例化。
"""

from __future__ import annotations

import os
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

    S1.0h: 修复仿真环境 + 对齐学习闭环 + 奖励泵漏洞修复。
    S1.0h 环境修正：
      - 托盘缩放 4.0x → 1.8x（匹配 isaac_sim_asset_import.md 文档）
      - pallet_depth_m 4.8 → 2.16（1.2m × 1.8）
      - 托盘质量 30kg → 45kg，初始高度 0.30 → 0.15
      - 添加碰撞接触参数 collision_props（contactOffset/restOffset）
      - 修复 _pallet_front_x 符号 bug（+ → -，指向 pocket 开口而非远端面）
    S1.0h 奖励修正：
      - r_approach 软门控（0.2 + 0.8*w_ready），打破对齐-接近死锁
      - 对齐阈值随距离插值（远处松、近处紧）
      - yaw 目标分两段（远处对准指向方向，近处对准托盘朝向）
      - r_forward 绑定到朝向因子 w_yaw
      - r_insert / r_lift 势函数 shaping（带 gamma=0.99），修复奖励泵漏洞
      - reset 时正确初始化势能缓存
    """

    # ===== 环境基础参数 =====
    decimation = 4
    episode_length_s = 36.0  # was 12.0; ~1080 steps for expert to complete insertion+lift

    # actions: [drive, steer, lift]（驾驶、转向、举升）
    action_space = 3

    # observations: 向量观测，具体构成见 env._get_observations()
    # S1.0N: 13→15（新增 y_err_obs / yaw_err_obs）
    observation_space = 15

    # no separate privileged state in this minimal patch（不使用特权观测）
    state_space = 0

    # ===== 仿真参数 =====
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)

    # ===== 场景复制与并行环境 =====
    # clone_in_fabric=False: 修复 body_pos_w 全部等于 root_pos_w 的问题
    # 原因：Fabric clone 失败导致 body link 位置不追踪（见诊断报告）
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=128,
        env_spacing=6.0,
        replicate_physics=True,
        clone_in_fabric=False,
    )

    # ===== 资产路径 =====
    forklift_usd_path: str = f"{ISAAC_NUCLEUS_DIR}/Robots/IsaacSim/ForkliftC/forklift_c.usd"
    pallet_usd_path: str = f"{ISAAC_NUCLEUS_DIR}/Props/Pallet/pallet.usd"

    # ===== 托盘几何参数（欧标托盘，按 1.8x 缩放）=====
    # 原始深度 1.2m × 1.8 = 2.16m（参考 docs/learning_guiding/isaac_sim_asset_import.md）
    pallet_depth_m: float = 2.16

    # ===== KPI（成功判定指标）=====
    # S1.0k sanity check 发现：convexDecomposition 碰撞体阻止叉齿插入超过 ~1.03m，
    # 原 2/3 (1.44m) 阈值物理不可达。降低到 0.40 (0.864m)，留 15% 安全余量。
    # 详见 docs/diagnostic_reports/success_sanity_check_2026-02-10.md
    insert_fraction: float = 0.40
    lift_delta_m: float = 0.3      # S1.0T: 0.25→1.0, 防作弊计划降低门槛 1.0→0.3
    # S1.0M: hold_time_s 从 1.0 降到 0.33（hold_steps: 30→~10）。
    # sanity check A2 显示即使理论成功位姿，hold_counter 最高 4/30 即断，
    # 物理抖动使 30 步连续保持几乎不可能。课程阶段先出 success 再收紧。
    hold_time_s: float = 0.33
    # S1.0M: 放宽成功对齐阈值（课程学习起点）。
    # 当前 lateral≈0.25m（需 ≤0.03m，差 8 倍），yaw≈7.5°（需 ≤3.0°，差 2.5 倍）。
    # 先让 success 出现，后续逐步收紧。
    max_lateral_err_m: float = 0.15
    max_yaw_err_deg: float = 5.0        # StageB3 课程收紧: 8→5
    # S1.0N: hold counter 全维度 Schmitt trigger（抗物理抖动）
    hysteresis_ratio: float = 1.2       # 对齐 exit 阈值 = entry × 1.2
    insert_exit_epsilon: float = 0.02   # 插入深度 exit 容差（与 insert_depth 同单位）
    lift_exit_epsilon: float = 0.08     # S1.0T: 0.02→0.08 等比放大 (m)
    # S1.0O-C2: hold counter 衰减（越界不清零，改为 *= decay）
    hold_counter_decay: float = 0.8
    
    # ---- 防作弊与终局优化 ----
    max_insert_z_err: float = 0.4       # 最大允许的货叉与托盘高度差（防止隔空飞越作弊）
    rew_stay_still: float = 0.5         # 成功后保持静止的奖励

    # ===== 动作范围（[-1, 1] 的归一化动作会乘以下列缩放）=====
    wheel_speed_rad_s: float = 20.0
    steer_angle_rad: float = 0.6
    lift_speed_m_s: float = 0.5

    # ---------- S1.0k 奖励参数（三阶段势函数 + 严格几何） ----------
    # PPO 折扣因子（势函数 shaping 用）
    # S1.0L: 纯差分 shaping，避免 gamma<1 在慢变化势函数上形成负常数底噪。
    gamma: float = 1.0

    # Stage 1: 距离带 + 粗对齐
    # S1.0L: 距离参考点改为 base（root），插入深度仍使用 tip。
    stage_distance_ref: str = "base"
    d1_min: float = 2.0      # 距离带下界 (m)
    d1_max: float = 3.0      # 距离带上界 (m)
    e_band_scale: float = 0.5  # 距离带误差归一化尺度
    # S1.0M: 收紧对齐尺度，增大 alignment 在 E1/E2 中的权重。
    y_scale1: float = 0.15   # S1.0M: 0.25→0.15，lateral 在 E1 中权重 1.0→1.67
    yaw_scale1: float = 10.0  # S1.0M: 15→10，yaw 在 E1 中权重 0.5→0.75
    k_phi1: float = 6.0      # Stage1 势函数强度

    # Stage 2: 微调接近（从距离带推到口前）
    d2_scale: float = 1.0    # Stage2 前向距离尺度 (m)
    y_scale2: float = 0.08   # S1.0M: 0.12→0.08，lateral 在 E2 中权重 2.08→3.13
    yaw_scale2: float = 5.0  # S1.0M: 8→5，yaw 在 E2 中权重 0.94→1.50
    k_phi2: float = 10.0     # Stage2 势函数强度
    # S1.0M: 放宽 w_align2 门控。原 y_gate2=0.25 在 y≈0.25m 时 w_align2≈0.37（丢弃 63%）。
    y_gate2: float = 0.60    # S1.0M: 0.25→0.40，信号保留翻倍
    yaw_gate2: float = 20.0  # S1.0M: 15→20

    # Stage 3: 插入深化
    # S1.0L: 推迟 Stage3 接管，避免早期压制对齐 shaping。
    ins_start: float = 0.10  # 插入接管起始阈值 (归一化)
    ins_ramp: float = 0.15   # 插入接管缓坡宽度
    # S1.0L: 放宽 Stage3 对齐门控，避免 gate 过紧导致插入信号过弱。
    y_gate3: float = 0.18    # Stage3 严对齐门控 (m)
    yaw_gate3: float = 12.0  # Stage3 严对齐门控 (deg)
    k_ins: float = 18.0      # 插入势函数强度
    # S1.0L: 默认不再用 (1-w3) 压制 phi1/phi2，保留开关便于回滚。
    suppress_preinsert_phi_with_w3: bool = False

    # 举升
    # S1.0M: insert_gate_norm 从 0.60 降到 0.35。
    # 物理最大 insert_norm ≈ 0.477（convexDecomposition 碰撞限制），
    # 原 0.60 永远不可达 → w_lift_base 恒为 0，phi_lift 恒为 0，
    # pen_premature 惩罚一切举升 → success 永远不可能。
    insert_gate_norm: float = 0.35  # 允许举升的插入深度门槛
    insert_ramp_norm: float = 0.08  # 举升门控缓坡（0.35~0.43 打开）
    k_lift: float = 20.0     # 举升势函数强度
    k_pre: float = 5.0       # S1.0M: 10→5，降低空举惩罚避免打爆探索
    # S1.0O-A3: premature lift 惩罚分段温和化
    premature_hard_thresh: float = 0.05    # insert_norm < 此值时全额惩罚
    premature_soft_thresh: float = 0.20    # insert_norm >= 此值时惩罚 → 0
    # S1.0O-A3: lift 进度 delta 势函数
    k_lift_progress: float = 1.2           # lift delta shaping 权重
    sigma_lift: float = 0.15               # lift 误差尺度 (m)

    # 常驻惩罚
    rew_action_l2: float = -0.01
    rew_time_penalty: float = -0.003
    k_dist_cont: float = 0.02  # S1.0k: 0.03→0.02，避免压死探索

    # 成功奖励
    rew_success: float = 100.0
    rew_success_time: float = 30.0

    # 超时终局惩罚
    rew_timeout: float = -10.0

    # 里程碑奖励（一次性触发）
    rew_milestone_approach: float = 1.0
    rew_milestone_coarse_align: float = 2.0
    rew_milestone_insert_10: float = 5.0
    rew_milestone_insert_30: float = 10.0
    # S1.0M: 新增对齐里程碑，给对齐一条"强奖励通路"
    rew_milestone_fine_align: float = 5.0      # y<0.10m & yaw<5°
    rew_milestone_precise_align: float = 8.0   # y<0.05m & yaw<3°
    # S1.0N: gate_align 里程碑（entry 条件本身，绑定 approach flag 防早触发）
    rew_milestone_gate_align: float = 2.5      # y<0.15m & yaw<8° & approach 已触发

    # S1.0O-B1: 增大 sigma + k，让 lateral 0.2~0.4m 区间梯度更强（S1.0N: 0.1/0.15/8.0）
    k_hold_align: float = 0.3          # delta shaping 权重
    hold_align_sigma_y: float = 0.25   # 横向尺度 (m)
    hold_align_sigma_yaw: float = 8.0  # 偏航尺度 (deg) — A3B1C2_v2: 12→8 收紧 yaw 梯度

    # ---- S1.0Q: 死区惩罚 (A1) ----
    dead_zone_insert_thresh: float = 0.30   # 死区插入阈值（归一化）
    dead_zone_lat_thresh: float = 0.20      # 死区横向阈值 (m)
    k_dead_zone: float = 0.5               # 死区惩罚权重
    dead_zone_pen_clamp: float = 0.05      # 每步最大惩罚绝对值

    # ---- S1.0Q: 撤退鼓励 (A2/A2v2) ----
    k_retreat: float = 0.0                  # B0=0（不激活）; A2v2 改为 2.0
    retreat_reward_clamp: float = 0.1       # 每步最大撤退奖励
    retreat_window_size: int = 8            # A2v2: 撤退窗口长度（步数）

    # ---- S1.0Q: Milestone 死区衰减 ----
    # S1.0S: 锁定 A1 最佳配置为默认值（s1.0q 验证 milestone_dead_zone_scale=0.0 最优）
    milestone_dead_zone_scale: float = 0.0  # S1.0S default: A1 验证最优

    # ---- S1.0Q: 插入进度门控 (B1) ----
    # S1.0S: 锁定 B1a' 最佳配置为默认值（s1.0q 验证 ins_floor=0.2 最优）
    ins_floor: float = 0.2                  # S1.0S default: B1a' 验证最优
    ins_lat_gate_sigma: float = 1e6         # B0=无穷大（不生效）; B1b 改为 0.20

    # ---- S1.0Q: 横向精调 (C1) ----
    k_lat_fine: float = 0.0                 # B0=0（不激活）; C1 改为 0.8
    lat_fine_sigma: float = 0.15            # 横向高斯尺度 (m)
    lat_fine_ins_thresh: float = 0.05       # 激活门槛（归一化 insert_norm）

    # ---- S1.0Q: 观测分辨率 (C2) ----
    y_err_fine_scale: float = 0.20          # 精细横向归一化尺度 (m)
    yaw_err_fine_scale_deg: float = 8.0     # 精细偏航归一化尺度 (°)

    # ---- S1.0S Phase-0.5: 初始位姿鲁棒性 ----
    # 方案 B: y_err_obs 归一化尺度（原硬编码 0.5，|y|>0.5m 时观测饱和）
    # 扩大到 0.8 消除 Y=[-0.6,+0.6] 范围内的观测饱和
    y_err_obs_scale: float = 0.8            # 默认 0.5（原行为）; Phase-0.5 P1/P3 改为 0.8

    # 失败早停
    early_stop_d_xy_max: float = 3.0
    early_stop_d_xy_steps: int = 30
    early_stop_stall_phi_eps: float = 0.001
    early_stop_stall_steps: int = 60
    early_stop_stall_action_eps: float = 0.05
    rew_early_stop_fly: float = -2.0
    rew_early_stop_stall: float = -1.0

    # ---- S1.0Q Batch-3: Dead-zone stuck detector ----
    dz_stuck_ins_eps: float = 0.005         # insert_norm 变化阈值
    dz_stuck_lat_eps: float = 0.005         # y_err 变化阈值 (m)
    dz_stuck_steps: int = 99999             # 默认不激活; Exp4 改为 30
    rew_early_stop_dz_stuck: float = -2.0   # 卡死 penalty
    # ---- S1.0Q Batch-4: stuck detector 消融控制 ----
    dz_stuck_early_done: bool = True        # False = penOnly（只给 penalty 不终止）

    # ---- S1.0S Phase-2: 举升里程碑 ----
    rew_milestone_lift_10cm: float = 3.0    # 首次 lift_height >= 0.10m
    rew_milestone_lift_20cm: float = 5.0    # 首次 lift_height >= 0.20m

    # ---- S1.0T: 高举升里程碑 ----
    rew_milestone_lift_50cm: float = 6.0    # 首次 lift_height >= 0.50m
    rew_milestone_lift_75cm: float = 8.0    # 首次 lift_height >= 0.75m

    # ---- S1.0T: 观测归一化 ----
    lift_pos_scale: float = 1.0             # obs 中 lift_pos /= scale（防止高举升 OOD）

    # ---- S1.0S Phase-R: 远场大横偏修正奖励 ----
    k_far_lat: float = 0.0                  # 默认不激活; Phase-R 改为 2.0
    far_lat_y_thresh: float = 0.4           # y_err 激活阈值 (m)
    far_lat_ins_thresh: float = 0.1         # insert_norm 激活阈值（仅远场生效）

    # ---- S1.0S Phase-3: 全局进展停滞检测器 ----
    global_stall_phi_eps: float = 0.01      # phi_total 变化阈值
    global_stall_steps: int = 120         # 默认不激活; Phase-3 改为 120（约 4 秒）
    rew_global_stall: float = -1.5          # 全局停滞 penalty

    # ---- S1.0U: Fork-tip alignment gate ----
    # Adds tip_y_err (fork tip lateral error) to w_align2/w_align3 gates,
    # distance-weighted so far-field training is unaffected.
    fork_reach_m: float = 1.87              # root to fork tip forward distance (m)
    tip_y_gate2: float = 0.50              # Stage2 fork tip lateral gate sigma (m)
    tip_y_gate3: float = 0.20              # Stage3 fork tip lateral gate sigma (m)
    tip_y_weight_dist_thresh: float = 2.5  # tip_y_err starts blending in below this dist (m)

    # ---- S1.0U: Collision proximity penalty ----
    k_tip_proximity_pen: float = 1.5       # proximity penalty weight
    tip_prox_dist_thresh: float = 2.2      # only active when dist < this (m)
    tip_prox_lat_thresh: float = 0.15      # tip_y_err dead-zone (m), no penalty below

    # ---- S1.0U: Success/hold tip alignment consistency ----
    tip_align_entry_m: float = 0.12        # tip_y_err <= this to enter hold (near-field)
    tip_align_exit_m: float = 0.16         # Schmitt exit: tip_y_err > this breaks hold
    tip_align_near_dist: float = 2.2       # tip constraint only active below this dist

    # termination thresholds
    max_roll_pitch_rad: float = 0.45  # ~25 deg
    max_time_s: float = episode_length_s

    # ===== 叉车配置（forklift_c 关节命名）=====
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
            pos=(-3.5, 0.0, 0.03),  # S1.0h: 缩放 1.8x 后调整初始距离
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
            # lift joint (position control — logs32 验证的唯一可行 drive 方式)
            # Isaac Lab 的 ImplicitActuator 会将 stiffness/damping 写入 PhysX 关节驱动，
            # 覆盖 USD DriveAPI 原始值。
            # logs32 证明：stiffness=200000 + set_joint_position_target 可产生
            # force ≈ 200000*pos_error ≈ 3200N（足以克服重力），lift 稳定升至 0.23m。
            # 纯速度控制（stiffness=0）经 31 次实验全部失败。
            "lift": ImplicitActuatorCfg(
                joint_names_expr=["lift_joint"],
                velocity_limit=1.0,
                effort_limit=50000.0,  # 50kN，与 USD DriveAPI maxForce 一致
                stiffness=200000.0,    # 位置控制（logs32 验证值）
                damping=10000.0,       # 阻尼（logs32 验证值）
            ),
        },
    )

    # ===== 托盘配置（动态刚体）=====
    # S1.0h 修改说明（参考 docs/learning_guiding/isaac_sim_asset_import.md）：
    # 1. 从 kinematic 改为动态刚体，使托盘可以被叉车推动和举起
    # 2. scale=1.8 使托盘放大到与叉车货叉兼容的尺寸
    #    - 原始托盘插入孔宽度 ~228mm，货叉宽度 ~394mm
    #    - 放大 1.8x 后插入孔宽度 ~410mm，足够容纳货叉
    # 3. 添加 collision_props 设置接触参数（参考 collision_mesh_guide 第 6.1 节）
    # 注意：如果修改托盘缩放比例，需同步更新：
    # - pallet_depth_m（按深度比例缩放）
    # - init_state.pos[2]（初始高度）
    # - 以及 env.py 中基于 pallet_depth_m 的插入判定逻辑
    pallet_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Pallet",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{os.environ.get('ISAACLAB_PATH', '/home/uniubi/projects/forklift_sim/IsaacLab')}/../assets/pallet_com_shifted.usd",
            scale=(1.8, 1.8, 1.8),  # 托盘统一缩放（修改后需同步更新相关几何参数）
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                kinematic_enabled=False,  # 动态刚体，可被推动/举起
                disable_gravity=False,    # 受重力影响，落在地面上
                max_depenetration_velocity=1.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=45.0),  # S1.0h: 空托盘约 45 kg
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
                contact_offset=0.02,   # 碰撞检测提前量（防止微小穿透）
                rest_offset=0.005,     # 静止时最小间隙
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            # 初始高度需与缩放比例匹配，避免托盘悬空或穿地
            pos=(0.0, 0.0, 0.15),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    # ===== 地面 =====
    ground_cfg: GroundPlaneCfg = GroundPlaneCfg()
