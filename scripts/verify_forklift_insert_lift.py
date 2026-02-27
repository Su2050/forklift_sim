#!/usr/bin/env python3
"""
Isaac Sim叉车插入举升功能验证脚本

手动控制叉车完成完整的插入和举升流程，验证Isaac Sim物理仿真是否真正支持这个功能。
包括：
1. 环境初始化检查
2. 手动控制叉车移动
3. 精确对齐托盘
4. 推进插入
5. 举升托盘
6. 验证物理交互和碰撞检测
"""

import sys
from pathlib import Path

# 环境检查：确保在正确的Python环境中运行
try:
    import torch
except ImportError:
    print("=" * 80)
    print("错误：无法导入 torch 模块")
    print("=" * 80)
    print("\n原因：脚本需要通过 isaaclab.sh 运行，以使用IsaacLab的Python环境。")
    print("\n正确的运行方式：")
    print("  cd /home/uniubi/projects/forklift_sim/IsaacLab")
    print("  自动测试：./isaaclab.sh -p ../scripts/verify_forklift_insert_lift.py --headless")
    print("  手动控制：./isaaclab.sh -p ../scripts/verify_forklift_insert_lift.py --manual")
    print("\n详细说明请参考：docs/verify_forklift_insert_lift_usage.md")
    print("=" * 80)
    sys.exit(1)

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple
import time

# 添加 IsaacLab 路径
isaaclab_path = Path(__file__).resolve().parent.parent / "IsaacLab"
sys.path.insert(0, str(isaaclab_path / "source"))

# 首先初始化 Isaac Sim
from isaaclab.app import AppLauncher
import argparse

parser = argparse.ArgumentParser(description="验证叉车插入举升功能")
parser.add_argument("--manual", action="store_true", help="启用手动键盘控制模式")
parser.add_argument("--auto-align", action="store_true", help="手动模式下先自动对齐到理想位置")
parser.add_argument("--sanity-check", action="store_true", help="运行 success 判定 sanity check（A层逻辑+B层物理可达性）")
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()

# 启动 Isaac Sim
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# 在 Isaac Sim 初始化后导入
import carb

from isaaclab.devices import Se2Keyboard, Se2KeyboardCfg
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "../forklift_pallet_insert_lift_project/isaaclab_patch/source/isaaclab_tasks"))
from isaaclab_tasks.direct.forklift_pallet_insert_lift.env_cfg import ForkliftPalletInsertLiftEnvCfg
from isaaclab_tasks.direct.forklift_pallet_insert_lift.env import ForkliftPalletInsertLiftEnv
from pxr import UsdPhysics, PhysxSchema


def print_section(title: str):
    """打印分节标题"""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def print_info(label: str, value):
    """打印信息"""
    print(f"  {label}: {value}")


@dataclass
class TestResult:
    """测试结果"""
    name: str
    passed: bool
    details: str
    metrics: Dict[str, float] = None


class ForkliftKeyboard(Se2Keyboard):
    """叉车键盘控制器（基于Se2Keyboard）
    
    键位：
    - W/S: 前进/后退
    - A/D: 左转/右转
    - R/F: 货叉上升/下降（R=Raise, F=Fall，避免与Isaac Sim相机控制Q/E冲突）
    - SPACE: 停止
    """

    def __init__(self, cfg: Se2KeyboardCfg, lift_sensitivity: float = 0.5):
        super().__init__(cfg)
        self._lift_sensitivity = lift_sensitivity
        self._lift_command = 0.0
        self._steer_command = 0.0
        self._lift_up_pressed = False
        self._lift_down_pressed = False
        self._steer_left_pressed = False
        self._steer_right_pressed = False

    def advance(self) -> torch.Tensor:
        """返回 (drive, steer, lift)"""
        base_cmd = super().advance()
        drive = base_cmd[0].item()
        # 使用自己管理的转向命令，而不是 Se2Keyboard 的
        steer = float(self._steer_command)
        lift = float(self._lift_command)
        return torch.tensor([drive, steer, lift], dtype=torch.float32, device=self._sim_device)

    def reset(self):
        super().reset()
        self._lift_command = 0.0
        self._steer_command = 0.0
        self._lift_up_pressed = False
        self._lift_down_pressed = False
        self._steer_left_pressed = False
        self._steer_right_pressed = False

    def _create_key_bindings(self):
        super()._create_key_bindings()
        # 添加 W/S 映射到前进/后退
        self._INPUT_KEY_MAPPING.update(
            {
                "W": self._INPUT_KEY_MAPPING["UP"],
                "S": self._INPUT_KEY_MAPPING["DOWN"],
            }
        )

    def _on_keyboard_event(self, event, *args, **kwargs):
        # 处理升降键 R/F（避免与Isaac Sim相机控制Q/E冲突）
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if event.input.name == "R":
                self._lift_up_pressed = True
            elif event.input.name == "F":
                self._lift_down_pressed = True
            elif event.input.name == "A":
                self._steer_left_pressed = True
            elif event.input.name == "D":
                self._steer_right_pressed = True
            elif event.input.name == "SPACE":
                self.reset()
        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            if event.input.name == "R":
                self._lift_up_pressed = False
            elif event.input.name == "F":
                self._lift_down_pressed = False
            elif event.input.name == "A":
                self._steer_left_pressed = False
            elif event.input.name == "D":
                self._steer_right_pressed = False

        # 更新升降命令
        if self._lift_up_pressed and not self._lift_down_pressed:
            self._lift_command = self._lift_sensitivity
        elif self._lift_down_pressed and not self._lift_up_pressed:
            self._lift_command = -self._lift_sensitivity
        else:
            self._lift_command = 0.0
        
        # 更新转向命令（A=左转=正，D=右转=负）
        if self._steer_left_pressed and not self._steer_right_pressed:
            self._steer_command = 0.5  # 左转
        elif self._steer_right_pressed and not self._steer_left_pressed:
            self._steer_command = -0.5  # 右转
        else:
            self._steer_command = 0.0

        return super()._on_keyboard_event(event, *args, **kwargs)


class ForkliftVerification:
    """叉车验证类"""
    
    def __init__(self):
        self.env = None
        self.results: List[TestResult] = []
        self.cfg = None
        
    def initialize_environment(self, manual_mode: bool = False):
        """初始化环境
        
        Args:
            manual_mode: 是否为手动模式，手动模式下禁用自动重置
        """
        print_section("环境初始化")
        
        # 创建环境配置
        self.cfg = ForkliftPalletInsertLiftEnvCfg()
        self.cfg.scene.num_envs = 1
        
        # 手动模式下禁用自动重置（设置为1小时）
        if manual_mode:
            self.cfg.episode_length_s = 3600.0
            self.cfg.max_time_s = 3600.0
            print_info("模式", "手动模式（禁用自动重置）")
        
        print_info("环境数量", self.cfg.scene.num_envs)
        print_info("任务", "Isaac-Forklift-PalletInsertLift-Direct-v0")
        
        print("\n[INFO] 正在创建环境...")
        self.env = ForkliftPalletInsertLiftEnv(self.cfg)
        print("[INFO] 环境创建成功")
        
        # 重置环境
        self.env.reset()
        
        # 统一设置叉车到较远位置，确保与托盘完全分开
        # 无论手动模式还是自动测试模式，都从相同的初始位置开始
        print("\n[INFO] 设置叉车到初始位置（与托盘分开）...")
        init_pos = torch.tensor([[-6.0, 0.0, 0.1]], device=self.env.device)
        init_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=self.env.device)
        env_ids = torch.tensor([0], device=self.env.device)
        self.env._write_root_pose(self.env.robot, init_pos, init_quat, env_ids)
        zeros3 = torch.zeros((1, 3), device=self.env.device)
        self.env._write_root_vel(self.env.robot, zeros3, zeros3, env_ids)
        # 关节状态归零（与 set_robot_ideal_position 一致）
        joint_pos = self.env.robot.data.default_joint_pos[env_ids].clone()
        joint_vel = torch.zeros_like(joint_pos)
        self.env._write_joint_state(self.env.robot, joint_pos, joint_vel, env_ids)
        # 完整的物理同步三步
        self.env.scene.write_data_to_sim()
        self.env.sim.reset()
        self.env.scene.update(self.env.cfg.sim.dt)
        self.env.robot.reset(env_ids)
        print("[INFO] 叉车已设置到 X=-6.0m")
        
        return True
    
    def check_environment_init(self) -> TestResult:
        """检查环境初始化"""
        print_section("环境初始化检查")
        
        details = []
        passed = True
        
        # 1. 检查叉车是否加载
        if self.env.robot is None:
            details.append("❌ 叉车未加载")
            passed = False
        else:
            details.append("✅ 叉车已加载")
            print_info("叉车关节数", len(self.env.robot.joint_names))
        
        # 2. 检查托盘是否加载
        if self.env.pallet is None:
            details.append("❌ 托盘未加载")
            passed = False
        else:
            details.append("✅ 托盘已加载")
            pallet_pos = self.env.pallet.data.root_pos_w[0]
            print_info("托盘位置", f"({pallet_pos[0]:.3f}, {pallet_pos[1]:.3f}, {pallet_pos[2]:.3f})")
        
        # 2.5 诊断 body_pos_w（排查 _compute_fork_tip 返回 root 位置的问题）
        print("\n[DIAGNOSTIC] body_pos_w 诊断:")
        body_pos = self.env.robot.data.body_pos_w  # (N,B,3)
        root_pos_diag = self.env.robot.data.root_pos_w  # (N,3)
        print_info("body_pos_w shape", str(list(body_pos.shape)))
        print_info("root_pos_w", f"({root_pos_diag[0,0]:.4f}, {root_pos_diag[0,1]:.4f}, {root_pos_diag[0,2]:.4f})")
        # 打印每个 body 的名称和位置
        body_names = self.env.robot.body_names if hasattr(self.env.robot, 'body_names') else [f"body_{i}" for i in range(body_pos.shape[1])]
        print(f"  共 {len(body_names)} 个 body:")
        all_same_as_root = True
        for i, name in enumerate(body_names):
            bx, by, bz = body_pos[0, i, 0].item(), body_pos[0, i, 1].item(), body_pos[0, i, 2].item()
            rx, ry, rz = root_pos_diag[0, 0].item(), root_pos_diag[0, 1].item(), root_pos_diag[0, 2].item()
            diff = ((bx - rx)**2 + (by - ry)**2 + (bz - rz)**2)**0.5
            marker = " ← ROOT" if diff < 0.001 else f" (dist from root: {diff:.4f}m)"
            if diff >= 0.001:
                all_same_as_root = False
            print(f"    [{i:2d}] {name:30s}  pos=({bx:.4f}, {by:.4f}, {bz:.4f}){marker}")
        if all_same_as_root:
            print("  ⚠️  所有 body 位置均等于 root_pos_w —— _compute_fork_tip() 无法正常工作！")
            print("  ⚠️  可能原因: clone_in_fabric 问题 或 USD 中所有 link 在同一 rigid body 下")
        else:
            print("  ✅ body 位置存在差异，_compute_fork_tip() 原理上可正常工作")
        # 打印 _compute_fork_tip 的结果
        tip = self.env._compute_fork_tip()
        print_info("_compute_fork_tip() 结果", f"({tip[0,0]:.4f}, {tip[0,1]:.4f}, {tip[0,2]:.4f})")

        # 2.6 诊断 lift_joint 限制和驱动属性
        print("\n[DIAGNOSTIC] lift_joint 限制和驱动:")
        lift_id = self.env._lift_id
        if hasattr(self.env.robot.data, 'joint_pos_limits'):
            jlim = self.env.robot.data.joint_pos_limits
            print_info("joint_pos_limits shape", str(list(jlim.shape)))
            if len(jlim.shape) == 3 and jlim.shape[1] > lift_id:
                lo = jlim[0, lift_id, 0].item()
                hi = jlim[0, lift_id, 1].item()
                print_info(f"lift_joint [{lift_id}] 位置限制", f"[{lo:.6f}, {hi:.6f}]")
                if abs(hi - lo) < 1e-6:
                    print("  ⚠️  lift_joint 上下限相同 → 关节被锁死！这是 lift 不动的原因")
                elif lo == 0.0 and hi == 0.0:
                    print("  ⚠️  lift_joint 限制为 [0, 0] → 关节被锁死！")
        # 打印所有关节的限制
        if hasattr(self.env.robot.data, 'joint_pos_limits'):
            jlim = self.env.robot.data.joint_pos_limits
            print("  所有关节位置限制:")
            for i, name in enumerate(self.env.robot.joint_names):
                if i < jlim.shape[1]:
                    lo = jlim[0, i, 0].item()
                    hi = jlim[0, i, 1].item()
                    lock = " ← LOCKED" if abs(hi - lo) < 1e-6 else ""
                    print(f"    [{i:2d}] {name:30s}  [{lo:.6f}, {hi:.6f}]{lock}")
        # 检查 USD 中的 drive 属性
        try:
            from pxr import Usd, UsdPhysics
            stage = self.env.sim.stage
            robot_prim_path = self.cfg.robot_cfg.prim_path.replace("env_.*", "env_0")
            for prim in Usd.PrimRange(stage.GetPrimAtPath(robot_prim_path)):
                if "lift" in prim.GetName().lower() and prim.IsA(UsdPhysics.Joint) if hasattr(UsdPhysics, 'Joint') else False:
                    print(f"  USD lift joint prim: {prim.GetPath()}")
                # 检查 DriveAPI
                if UsdPhysics.DriveAPI.Get(prim, "linear") or UsdPhysics.DriveAPI.Get(prim, "angular"):
                    for drive_type in ["linear", "angular"]:
                        drive_api = UsdPhysics.DriveAPI.Get(prim, drive_type)
                        if drive_api:
                            name_str = prim.GetName()
                            if "lift" in name_str.lower():
                                print(f"  USD DriveAPI ({drive_type}) on {prim.GetPath()}:")
                                if drive_api.GetStiffnessAttr():
                                    print(f"    stiffness = {drive_api.GetStiffnessAttr().Get()}")
                                if drive_api.GetDampingAttr():
                                    print(f"    damping = {drive_api.GetDampingAttr().Get()}")
                                if drive_api.GetMaxForceAttr():
                                    print(f"    maxForce = {drive_api.GetMaxForceAttr().Get()}")
        except Exception as e:
            print(f"  USD drive 检查失败: {e}")

        # 3. 检查关节配置
        print("\n关节配置:")
        print_info("前轮关节IDs", self.env._front_wheel_ids)
        print_info("后轮关节IDs", self.env._back_wheel_ids)
        print_info("转向关节IDs", self.env._rotator_ids)
        print_info("升降关节ID", self.env._lift_id)
        
        # 4. 检查物理步数配置
        print("\n物理步数配置:")
        print_info("decimation", self.cfg.decimation)
        print_info("physics_dt", self.cfg.sim.dt)
        print_info("环境步长", f"{self.cfg.sim.dt * self.cfg.decimation:.6f}s")
        print_info("每环境步的物理步数", self.cfg.decimation)
        
        # 5. 检查升降关节配置
        print("\n升降关节配置:")
        lift_joint_name = self.env.robot.joint_names[self.env._lift_id]
        print_info("升降关节名称", lift_joint_name)
        
        # 检查升降关节的执行器配置
        if hasattr(self.env.robot, 'actuators'):
            lift_actuator = None
            for actuator in self.env.robot.actuators.values():
                # 使用 joint_indices 而不是 joint_ids
                joint_indices = getattr(actuator, 'joint_indices', None)
                if joint_indices is not None and self.env._lift_id in joint_indices:
                    lift_actuator = actuator
                    break
            
            if lift_actuator:
                print_info("升降执行器类型", type(lift_actuator).__name__)
                if hasattr(lift_actuator, 'effort_limit'):
                    print_info("effort_limit", lift_actuator.effort_limit)
                if hasattr(lift_actuator, 'velocity_limit'):
                    print_info("velocity_limit", lift_actuator.velocity_limit)
                if hasattr(lift_actuator, 'stiffness'):
                    print_info("stiffness", lift_actuator.stiffness)
                if hasattr(lift_actuator, 'damping'):
                    print_info("damping", lift_actuator.damping)
        
        # 检查升降关节的位置限制
        lift_joint_pos = self.env._joint_pos[0, self.env._lift_id].item()
        if hasattr(self.env.robot, 'data') and hasattr(self.env.robot.data, 'default_joint_pos'):
            default_lift_pos = self.env.robot.data.default_joint_pos[0, self.env._lift_id].item()
            print_info("当前升降位置", f"{lift_joint_pos:.4f}m")
            print_info("默认升降位置", f"{default_lift_pos:.4f}m")
        
        # 检查升降关节的关节限制
        if hasattr(self.env.robot, 'data') and hasattr(self.env.robot.data, 'joint_pos_limits'):
            pos_limits = self.env.robot.data.joint_pos_limits
            if pos_limits is not None and pos_limits.shape[0] > self.env._lift_id:
                lift_min = pos_limits[self.env._lift_id, 0].item()
                lift_max = pos_limits[self.env._lift_id, 1].item()
                print_info("升降位置限制", f"[{lift_min:.4f}, {lift_max:.4f}]m")
            elif pos_limits is not None:
                print_info("升降位置限制", f"数据形状不匹配: {pos_limits.shape}, lift_id={self.env._lift_id}")
        
        print_info("lift_speed_m_s", self.cfg.lift_speed_m_s)
        print_info("预期最大升降速度", f"{self.cfg.lift_speed_m_s:.4f} m/s")
        
        # 5. 检查轮子执行器配置
        print("\n轮子执行器配置:")
        if hasattr(self.env.robot, 'actuators'):
            front_wheel_actuator = None
            back_wheel_actuator = None
            
            for actuator in self.env.robot.actuators.values():
                joint_indices = getattr(actuator, 'joint_indices', None)
                if joint_indices is not None:
                    # 检查是否是前轮执行器
                    if any(idx in self.env._front_wheel_ids for idx in joint_indices):
                        front_wheel_actuator = actuator
                    # 检查是否是后轮执行器
                    if any(idx in self.env._back_wheel_ids for idx in joint_indices):
                        back_wheel_actuator = actuator
            
            if front_wheel_actuator:
                print_info("前轮执行器类型", type(front_wheel_actuator).__name__)
                if hasattr(front_wheel_actuator, 'effort_limit'):
                    print_info("前轮effort_limit", front_wheel_actuator.effort_limit)
                if hasattr(front_wheel_actuator, 'velocity_limit'):
                    print_info("前轮velocity_limit", front_wheel_actuator.velocity_limit)
                if hasattr(front_wheel_actuator, 'stiffness'):
                    print_info("前轮stiffness", front_wheel_actuator.stiffness)
                if hasattr(front_wheel_actuator, 'damping'):
                    print_info("前轮damping", front_wheel_actuator.damping)
            else:
                print_info("前轮执行器", "未找到")
            
            if back_wheel_actuator:
                print_info("后轮执行器类型", type(back_wheel_actuator).__name__)
                if hasattr(back_wheel_actuator, 'effort_limit'):
                    print_info("后轮effort_limit", back_wheel_actuator.effort_limit)
                if hasattr(back_wheel_actuator, 'velocity_limit'):
                    print_info("后轮velocity_limit", back_wheel_actuator.velocity_limit)
                if hasattr(back_wheel_actuator, 'stiffness'):
                    print_info("后轮stiffness", back_wheel_actuator.stiffness)
                if hasattr(back_wheel_actuator, 'damping'):
                    print_info("后轮damping", back_wheel_actuator.damping)
            else:
                print_info("后轮执行器", "未找到")
            
            # 对比配置
            if front_wheel_actuator and lift_actuator:
                front_effort = getattr(front_wheel_actuator, 'effort_limit', None)
                lift_effort = getattr(lift_actuator, 'effort_limit', None)
                if front_effort is not None and lift_effort is not None:
                    front_effort_val = front_effort[0, 0].item() if isinstance(front_effort, torch.Tensor) else front_effort
                    lift_effort_val = lift_effort[0, 0].item() if isinstance(lift_effort, torch.Tensor) else lift_effort
                    print_info("前轮vs升降effort_limit", f"前轮={front_effort_val:.1f}, 升降={lift_effort_val:.1f}, 比例={front_effort_val/lift_effort_val:.2f}")
        
        print_info("wheel_speed_rad_s", self.cfg.wheel_speed_rad_s)
        print_info("预期最大轮子速度", f"{self.cfg.wheel_speed_rad_s:.2f} rad/s")
        
        # 6. 检查叉车物理属性
        print("\n叉车物理属性检查:")
        try:
            # 检查叉车总质量
            if hasattr(self.env.robot, 'data') and hasattr(self.env.robot.data, 'root_mass'):
                root_mass = self.env.robot.data.root_mass[0].item()
                print_info("叉车总质量", f"{root_mass:.2f} kg")
            
            # 检查叉车位置和速度限制
            if hasattr(self.env.robot, 'data') and hasattr(self.env.robot.data, 'root_lin_vel_w'):
                current_vel = self.env.robot.data.root_lin_vel_w[0]
                vel_magnitude = torch.norm(current_vel[:2]).item()
                print_info("当前速度大小", f"{vel_magnitude:.4f} m/s")
            
            # 检查重力配置
            if hasattr(self.env.sim, 'physics_context'):
                gravity = self.env.sim.physics_context.get_gravity()
                print_info("重力配置", f"({gravity[0]:.2f}, {gravity[1]:.2f}, {gravity[2]:.2f}) m/s²")
            
            # 检查叉车初始位置
            initial_pos = self.env.robot.data.root_pos_w[0]
            print_info("初始位置", f"({initial_pos[0]:.4f}, {initial_pos[1]:.4f}, {initial_pos[2]:.4f})")
            if initial_pos[2] < 0.05:
                details.append(f"⚠️  叉车初始位置过低（z={initial_pos[2]:.4f}m），可能嵌入地面")
            
        except Exception as e:
            print_info("物理属性检查", f"错误: {e}")
        
        # 4. 检查托盘kinematic模式
        print("\n托盘物理属性检查:")
        stage = self.env.sim.stage
        pallet_prim = stage.GetPrimAtPath("/World/envs/env_0/Pallet")
        
        if pallet_prim.IsValid():
            rb_api = UsdPhysics.RigidBodyAPI(pallet_prim)
            if rb_api:
                rigid_body_enabled = rb_api.GetRigidBodyEnabledAttr().Get()
                kinematic_enabled = rb_api.GetKinematicEnabledAttr().Get()
                
                print_info("rigid_body_enabled", rigid_body_enabled)
                print_info("kinematic_enabled", kinematic_enabled)
                
                if kinematic_enabled:
                    details.append("✅ 托盘是kinematic模式（固定，无法被举升）")
                else:
                    details.append("⚠️  托盘不是kinematic模式（可以被举升）")
            else:
                details.append("⚠️  无法获取托盘RigidBodyAPI")
        else:
            details.append("❌ 无法找到托盘prim")
            passed = False
        
        # 5. 检查初始位置
        robot_pos = self.env.robot.data.root_pos_w[0]
        robot_quat = self.env.robot.data.root_quat_w[0]
        print("\n初始状态:")
        print_info("叉车位置", f"({robot_pos[0]:.3f}, {robot_pos[1]:.3f}, {robot_pos[2]:.3f})")
        print_info("托盘位置", f"({pallet_pos[0]:.3f}, {pallet_pos[1]:.3f}, {pallet_pos[2]:.3f})")
        
        # 计算相对位置
        rel_pos = pallet_pos - robot_pos
        print_info("相对位置", f"({rel_pos[0]:.3f}, {rel_pos[1]:.3f}, {rel_pos[2]:.3f})")
        print_info("距离", f"{torch.norm(rel_pos[:2]):.3f}m")
        
        return TestResult(
            name="环境初始化检查",
            passed=passed,
            details="\n".join(details),
            metrics={
                "robot_joints": len(self.env.robot.joint_names),
                "distance_to_pallet": float(torch.norm(rel_pos[:2])),
            }
        )
    
    def manual_control(self, drive: float, steer: float, lift: float, steps: int = 1):
        """手动控制叉车
        
        Args:
            drive: 驱动速度 (-1.0 到 1.0)
            steer: 转向角度 (-1.0 到 1.0)
            lift: 升降速度 (-1.0 到 1.0)
            steps: 执行步数
        """
        actions = torch.tensor([[drive, steer, lift]], device=self.env.device)
        
        for _ in range(steps):
            self.env.step(actions)

    def print_current_status(self):
        """打印当前状态信息"""
        metrics = self.get_insertion_metrics()
        lift_pos = self.env._joint_pos[0, self.env._lift_id].item()
        root_pos = self.env.robot.data.root_pos_w[0]
        root_quat = self.env.robot.data.root_quat_w[0]
        yaw = math.degrees(self._quat_to_yaw(root_quat).item())

        print("\n当前状态:")
        print_info("叉车位置", f"({root_pos[0]:.3f}, {root_pos[1]:.3f}, {root_pos[2]:.3f})")
        print_info("叉车朝向", f"{yaw:.2f}°")
        print_info("插入深度", f"{metrics['insert_depth']:.4f}m ({metrics['insert_norm']*100:.1f}%)")
        print_info("横向误差", f"{metrics['lateral_err']*100:.2f}cm")
        print_info("偏航误差", f"{metrics['yaw_err_deg']:.2f}°")
        print_info("升降位置", f"{lift_pos:.4f}m")
    
    def get_fork_tip_position(self) -> torch.Tensor:
        """获取货叉尖端位置"""
        return self.env._compute_fork_tip()[0]
    
    def get_insertion_metrics(self) -> Dict[str, float]:
        """获取插入相关指标"""
        tip = self.get_fork_tip_position()
        pallet_pos = self.env.pallet.data.root_pos_w[0]
        
        # 计算_pallet_front_x
        pallet_front_x = pallet_pos[0] - self.cfg.pallet_depth_m * 0.5
        
        # 计算插入深度
        dist_front = tip[0] - pallet_front_x
        insert_depth = max(float(dist_front), 0.0)
        insert_norm = insert_depth / (self.cfg.pallet_depth_m + 1e-6)
        
        # 计算对齐误差
        robot_pos = self.env.robot.data.root_pos_w[0]
        lateral_err = torch.abs(pallet_pos[1] - robot_pos[1]).item()
        
        # 计算偏航误差
        robot_yaw = self._quat_to_yaw(self.env.robot.data.root_quat_w[0])
        pallet_yaw = self._quat_to_yaw(self.env.pallet.data.root_quat_w[0])
        yaw_err = torch.abs((pallet_yaw - robot_yaw + math.pi) % (2 * math.pi) - math.pi).item()
        yaw_err_deg = math.degrees(yaw_err)
        
        return {
            "fork_tip_x": tip[0].item(),
            "fork_tip_y": tip[1].item(),
            "fork_tip_z": tip[2].item(),
            "pallet_pos_x": pallet_pos[0].item(),
            "pallet_front_x": pallet_front_x,
            "dist_front": dist_front,
            "insert_depth": insert_depth,
            "insert_norm": insert_norm,
            "lateral_err": lateral_err,
            "yaw_err_deg": yaw_err_deg,
        }
    
    def _quat_to_yaw(self, q: torch.Tensor) -> torch.Tensor:
        """从四元数提取偏航角"""
        w, x, y, z = q.unbind(-1)
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return torch.atan2(siny_cosp, cosy_cosp)
    
    def _yaw_to_quat(self, yaw: torch.Tensor) -> torch.Tensor:
        """从偏航角转换为四元数 (w, x, y, z)"""
        half = yaw * 0.5
        return torch.stack([
            torch.cos(half),
            torch.zeros_like(half),
            torch.zeros_like(half),
            torch.sin(half)
        ], dim=-1)
    
    def print_metrics(self, metrics: Dict[str, float]):
        """打印指标"""
        print("\n当前指标:")
        print_info("货叉尖端位置", f"({metrics['fork_tip_x']:.4f}, {metrics['fork_tip_y']:.4f}, {metrics['fork_tip_z']:.4f})")
        print_info("托盘位置x", f"{metrics['pallet_pos_x']:.4f}")
        print_info("托盘前部x", f"{metrics['pallet_front_x']:.4f}")
        print_info("距离前部", f"{metrics['dist_front']:.4f}m")
        print_info("插入深度", f"{metrics['insert_depth']:.4f}m ({metrics['insert_norm']*100:.2f}%)")
        print_info("横向误差", f"{metrics['lateral_err']*100:.2f}cm")
        print_info("偏航误差", f"{metrics['yaw_err_deg']:.2f}°")
    
    def set_robot_ideal_position(self, distance_from_front=0.5):
        """设置叉车到理想对齐位置：与托盘对齐，距离托盘前部适当距离
        
        Args:
            distance_from_front: 距离托盘前部的距离（米），默认0.5米
        """
        pallet_pos = self.env.pallet.data.root_pos_w[0]
        pallet_yaw = self._quat_to_yaw(self.env.pallet.data.root_quat_w[0])
        
        # 计算理想位置：托盘前部前方一定距离，横向对齐，偏航对齐
        pallet_front_x = pallet_pos[0] - self.cfg.pallet_depth_m * 0.5
        
        # 需要计算货叉尖端相对于root的偏移
        # 先获取当前货叉尖端位置，计算偏移
        current_tip = self.get_fork_tip_position()
        current_root = self.env.robot.data.root_pos_w[0]
        fork_offset_x = current_tip[0] - current_root[0]  # 货叉尖端相对于root的x偏移
        
        # 计算理想root位置：使得货叉尖端距离托盘前部为distance_from_front
        ideal_tip_x = pallet_front_x - distance_from_front
        ideal_root_x = ideal_tip_x - fork_offset_x
        
        # 设置位置和姿态（横向对齐，偏航对齐）
        ideal_pos = torch.tensor([ideal_root_x, pallet_pos[1], 0.1], device=self.env.device)
        ideal_quat = self._yaw_to_quat(pallet_yaw)
        
        # ---- 写入位姿 + 速度 + 关节状态 ----
        env_ids = torch.tensor([0], device=self.env.device)
        self.env._write_root_pose(self.env.robot, ideal_pos.unsqueeze(0), ideal_quat.unsqueeze(0), env_ids)

        zeros3 = torch.zeros((1, 3), device=self.env.device)
        self.env._write_root_vel(self.env.robot, zeros3, zeros3, env_ids)

        # FIX #2: 重置关节状态（轮子速度归零、lift 归零），防止残留速度导致卡死
        joint_pos = self.env.robot.data.default_joint_pos[env_ids].clone()
        joint_vel = torch.zeros_like(joint_pos)
        self.env._write_joint_state(self.env.robot, joint_pos, joint_vel, env_ids)

        # ---- 物理同步（完整三步，与 _reset_idx 一致） ----
        self.env.scene.write_data_to_sim()
        # FIX #1 CRITICAL: 必须调用 sim.reset() 让 PhysX 接受传送后的状态
        self.env.sim.reset()
        self.env.scene.update(self.env.cfg.sim.dt)
        # FIX #3: 重置执行器内部状态（PID 积分、力矩缓存等）
        self.env.robot.reset(env_ids)

        # ---- 更新环境缓存（匹配 env.py S1.0h 的约定） ----
        tip = self.env._compute_fork_tip()
        insert_depth = torch.clamp(tip[:, 0] - self.env._pallet_front_x, min=0.0)
        self.env._last_insert_depth[0] = insert_depth[0]

        # FIX #5: E_align（S1.0j 专属，S1.0k 已移除）
        if hasattr(self.env, '_last_E_align'):
            root_pos = self.env.robot.data.root_pos_w[0:1]
            pallet_pos_b = self.env.pallet.data.root_pos_w[0:1]
            y_err = torch.abs(pallet_pos_b[:, 1] - root_pos[:, 1])
            yaw_robot = self._quat_to_yaw(self.env.robot.data.root_quat_w[0:1])
            yaw_pallet_b = self._quat_to_yaw(self.env.pallet.data.root_quat_w[0:1])
            dyaw = (yaw_pallet_b - yaw_robot + math.pi) % (2 * math.pi) - math.pi
            yaw_err_deg = torch.abs(dyaw) * (180.0 / math.pi)
            lat_ready = self.cfg.lat_ready_close if hasattr(self.cfg, 'lat_ready_close') else 0.10
            yaw_ready = self.cfg.yaw_ready_close_deg if hasattr(self.cfg, 'yaw_ready_close_deg') else 10.0
            E_align = y_err / lat_ready + yaw_err_deg / yaw_ready
            self.env._last_E_align[0] = E_align[0]

        # FIX #4: dist_front（S1.0j 专属，S1.0k 已移除）
        if hasattr(self.env, '_last_dist_front'):
            dist_front = torch.clamp(self.env._pallet_front_x - tip[:, 0], min=0.0)
            self.env._last_dist_front[0] = dist_front[0]

        # FIX #8: lift_pos 使用 env.py 约定（lift_height = tip_z - fork_tip_z0）
        self.env._last_lift_pos[0] = tip[0, 2] - self.env._fork_tip_z0[0]

        # FIX #6: 设置首步标记，防止增量奖励异常
        self.env._is_first_step[0] = True

        # FIX #9: 重置 episode 计时器，防止跨测试累计导致 episode timeout
        self.env.episode_length_buf[0] = 0

        # FIX #7: 重置势函数缓存（兼容 S1.0j 和 S1.0k）
        if hasattr(self.env, '_last_Phi_insert'):
            self.env._last_Phi_insert[0] = 0.0
        if hasattr(self.env, '_last_Phi_lift'):
            self.env._last_Phi_lift[0] = 0.0
        if hasattr(self.env, '_last_phi_total'):
            self.env._last_phi_total[0] = 0.0

        # 重置 actions 缓存
        self.env.actions[0] = 0.0
        
        # 验证设置结果
        final_tip = self.get_fork_tip_position()
        final_dist = final_tip[0] - pallet_front_x
        print(f"\n设置理想位置完成:")
        print_info("目标距离托盘前部", f"{distance_from_front:.3f}m")
        print_info("实际距离托盘前部", f"{final_dist:.3f}m")
        print_info("更新后的_last_insert_depth", f"{self.env._last_insert_depth[0].item():.4f}m")
    
    def test_approach(self) -> TestResult:
        """测试接近托盘：叉车自动驾驶接近"""
        print_section("阶段1：接近托盘测试（自动驾驶接近）")
        
        details = []
        passed = True
        
        # 获取初始位置
        initial_pos = self.env.robot.data.root_pos_w[0].clone()
        initial_metrics = self.get_insertion_metrics()
        initial_distance = abs(initial_metrics['dist_front'])  # 使用绝对值
        
        print(f"初始距离托盘前部: {initial_distance:.3f}m")
        print(f"初始位置: ({initial_pos[0]:.3f}, {initial_pos[1]:.3f}, {initial_pos[2]:.3f})")
        
        # 叉车自动前进接近托盘
        print("\n叉车自动驾驶接近托盘...")
        target_distance = 0.5  # 目标：距离托盘前部0.5米
        max_steps = 600  # 最多 600 步（约 20 秒）
        
        for step in range(max_steps):
            metrics = self.get_insertion_metrics()
            current_dist = abs(metrics['dist_front'])
            
            if current_dist <= target_distance + 0.1:  # 到达目标距离（允许 10cm 误差）
                print(f"  步数 {step}: 到达目标距离 {current_dist:.3f}m")
                break
            
            # 简单的前进控制：全速前进，不转向
            self.manual_control(drive=1.0, steer=0.0, lift=0.0, steps=1)
            
            if step % 60 == 0:
                print(f"  步数 {step}: 距离托盘前部 {current_dist:.3f}m")
        
        # 检查驾驶后的位置
        final_pos = self.env.robot.data.root_pos_w[0]
        final_metrics = self.get_insertion_metrics()
        final_distance = abs(final_metrics['dist_front'])  # 使用绝对值
        
        displacement = final_pos - initial_pos
        distance_error = abs(final_distance - target_distance)
        
        print(f"\n设置后状态:")
        print_info("最终位置", f"({final_pos[0]:.3f}, {final_pos[1]:.3f}, {final_pos[2]:.3f})")
        print_info("位置变化", f"({displacement[0]:.3f}, {displacement[1]:.3f}, {displacement[2]:.3f})")
        print_info("实际距离托盘前部", f"{final_distance:.3f}m")
        print_info("目标距离托盘前部", f"{target_distance:.3f}m")
        print_info("距离误差", f"{distance_error:.3f}m")
        
        # 验证对齐状态
        lateral_err = final_metrics['lateral_err']
        yaw_err = final_metrics['yaw_err_deg']
        print_info("横向误差", f"{lateral_err*100:.2f}cm")
        print_info("偏航误差", f"{yaw_err:.2f}°")
        
        # 判断是否成功
        if distance_error < 0.1:  # 距离误差小于10cm
            details.append(f"✅ 成功设置理想位置（距离误差 {distance_error:.3f}m < 0.1m）")
        else:
            details.append(f"⚠️  位置设置有误差（距离误差 {distance_error:.3f}m >= 0.1m）")
            passed = False
        
        if lateral_err < 0.05:  # 横向误差小于5cm
            details.append(f"✅ 横向对齐良好（{lateral_err*100:.2f}cm < 5cm）")
        else:
            details.append(f"⚠️  横向对齐未达标（{lateral_err*100:.2f}cm >= 5cm）")
            passed = False
        
        if yaw_err < 5.0:  # 偏航误差小于5度
            details.append(f"✅ 偏航对齐良好（{yaw_err:.2f}° < 5°）")
        else:
            details.append(f"⚠️  偏航对齐未达标（{yaw_err:.2f}° >= 5°）")
            passed = False
        
        return TestResult(
            name="接近测试",
            passed=passed,
            details="\n".join(details),
            metrics={
                "initial_distance": initial_distance,
                "final_distance": final_distance,
                "target_distance": target_distance,
                "distance_error": distance_error,
                "lateral_err": lateral_err,
                "yaw_err": yaw_err,
            }
        )
    
    def test_alignment(self) -> TestResult:
        """测试对齐：验证设置后的对齐状态"""
        print_section("阶段2：对齐测试（验证理想对齐状态）")
        
        details = []
        passed = True
        
        # 确保已经设置了理想对齐位置（如果还没有设置）
        current_metrics = self.get_insertion_metrics()
        if current_metrics['lateral_err'] > 0.1 or current_metrics['yaw_err_deg'] > 10.0:
            print("检测到未对齐状态，先设置理想对齐位置...")
            self.set_robot_ideal_position(distance_from_front=0.5)
        
        # 获取对齐状态
        metrics = self.get_insertion_metrics()
        lateral_err = metrics['lateral_err']
        yaw_err = metrics['yaw_err_deg']
        
        print(f"\n对齐状态验证:")
        print_info("横向误差", f"{lateral_err*100:.2f}cm")
        print_info("偏航误差", f"{yaw_err:.2f}°")
        print_info("货叉尖端y", f"{metrics['fork_tip_y']:.4f}m")
        print_info("托盘位置y", f"{metrics['pallet_pos_x']:.4f}m")
        
        # 判断是否对齐成功
        lateral_ok = lateral_err < 0.05  # 5cm阈值
        yaw_ok = yaw_err < 5.0  # 5度阈值
        
        if lateral_ok:
            details.append(f"✅ 横向对齐成功（{lateral_err*100:.2f}cm < 5cm）")
        else:
            details.append(f"⚠️  横向对齐未达标（{lateral_err*100:.2f}cm >= 5cm）")
            passed = False
        
        if yaw_ok:
            details.append(f"✅ 偏航对齐成功（{yaw_err:.2f}° < 5°）")
        else:
            details.append(f"⚠️  偏航对齐未达标（{yaw_err:.2f}° >= 5°）")
            passed = False
        
        # 验证距离托盘前部的距离
        dist_front = metrics['dist_front']
        if abs(dist_front) < 0.6:  # 距离托盘前部应该在0.5米左右
            details.append(f"✅ 距离托盘前部合适（{abs(dist_front):.3f}m）")
        else:
            details.append(f"⚠️  距离托盘前部不合适（{abs(dist_front):.3f}m）")
        
        return TestResult(
            name="对齐测试",
            passed=passed,
            details="\n".join(details),
            metrics={
                "lateral_err": lateral_err,
                "yaw_err": yaw_err,
                "dist_front": dist_front,
            }
        )
    
    def test_insertion(self) -> TestResult:
        """测试插入：在理想对齐位置基础上推进插入"""
        print_section("阶段3：插入测试（从理想对齐位置推进插入）")
        
        details = []
        passed = True
        
        # 1. 先设置理想对齐位置
        print("设置理想对齐位置...")
        self.set_robot_ideal_position(distance_from_front=0.5)
        
        # 2. 验证对齐状态
        initial_metrics = self.get_insertion_metrics()
        initial_insert_depth = initial_metrics['insert_depth']
        initial_dist_front = initial_metrics['dist_front']
        
        print(f"\n初始状态:")
        print_info("插入深度", f"{initial_insert_depth:.4f}m")
        print_info("距离托盘前部", f"{initial_dist_front:.4f}m")
        print_info("横向误差", f"{initial_metrics['lateral_err']*100:.2f}cm")
        print_info("偏航误差", f"{initial_metrics['yaw_err_deg']:.2f}°")
        
        # 验证对齐状态
        if initial_metrics['lateral_err'] > 0.05 or initial_metrics['yaw_err_deg'] > 5.0:
            details.append(f"⚠️  对齐状态未达标，可能影响插入")
        
        # 3. 控制叉车向前推进，直到插入
        # 物理碰撞约 1.0m 时卡住（托盘内部几何限制），目标设为可达值
        target_insert_depth = min(0.8, self.env._insert_thresh)
        max_steps = 400
        
        print(f"\n控制叉车向前推进，目标插入深度: {target_insert_depth:.2f}m")
        
        # 验证速度目标传递：在第一步之前验证
        print("\n速度目标传递验证:")
        test_action = torch.tensor([[0.3, 0.0, 0.0]], device=self.env.device)
        self.env.step(test_action)
        
        # 读取执行器接收到的速度目标
        front_target_after_step = None
        back_target_after_step = None
        if hasattr(self.env.robot, 'actuators'):
            for actuator in self.env.robot.actuators.values():
                joint_indices = getattr(actuator, 'joint_indices', None)
                if joint_indices is not None:
                    front_matched = [idx for idx in joint_indices if idx in self.env._front_wheel_ids]
                    if front_matched and hasattr(actuator, 'data') and hasattr(actuator.data, 'joint_vel_target'):
                        front_target_after_step = actuator.data.joint_vel_target[0, 0].item()
                    back_matched = [idx for idx in joint_indices if idx in self.env._back_wheel_ids]
                    if back_matched and hasattr(actuator, 'data') and hasattr(actuator.data, 'joint_vel_target'):
                        back_target_after_step = actuator.data.joint_vel_target[0, 0].item()
        
        expected_target = 0.3 * self.cfg.wheel_speed_rad_s
        print_info("动作值", "0.3")
        print_info("预期速度目标", f"{expected_target:.4f} rad/s")
        if front_target_after_step is not None:
            print_info("执行器接收到的前轮目标", f"{front_target_after_step:.4f} rad/s")
            if abs(front_target_after_step - expected_target) < 0.01:
                print_info("速度目标传递", "✅ 正确")
            else:
                print_info("速度目标传递", f"⚠️  不匹配（差异={abs(front_target_after_step - expected_target):.4f} rad/s）")
        else:
            print_info("速度目标传递", "⚠️  无法读取执行器目标")
        
        # 重置环境状态（回到初始位置）
        self.set_robot_ideal_position(distance_from_front=0.5)
        
        for i in range(max_steps):
            metrics = self.get_insertion_metrics()
            
            if metrics['insert_depth'] >= target_insert_depth:
                print(f"  达到目标插入深度！步数: {i}")
                break
            
            # 如果距离前部还比较远，继续前进
            # dist_front < 0 表示货叉在托盘前部之前，需要前进
            # dist_front > 0 表示货叉已超过托盘前部，已插入
            abs_dist_front = abs(metrics['dist_front'])
            if metrics['dist_front'] < 0:  # 还未到达托盘前部
                if abs_dist_front > 0.1:
                    drive = 0.3  # 距离较远，快速前进
                else:
                    drive = 0.2  # 接近时慢速推进
            elif metrics['dist_front'] > 0:  # 已超过托盘前部，继续推进以增加插入深度
                drive = 0.2  # 已插入，中速推进
            else:  # dist_front == 0，刚好在托盘前部
                drive = 0.2  # 开始插入，中速推进
            
            # 保持对齐（不转向）
            steer = 0.0
            
            # 记录推进前的位置
            before_pos = self.env.robot.data.root_pos_w[0].clone()
            before_tip = self.get_fork_tip_position().clone()
            
            # 检查控制逻辑状态（在应用动作前）
            root_pos = self.env.robot.data.root_pos_w[0:1]
            pallet_pos = self.env.pallet.data.root_pos_w[0:1]
            lateral_err = torch.abs(pallet_pos[:, 1] - root_pos[:, 1])
            yaw_robot = self._quat_to_yaw(self.env.robot.data.root_quat_w[0:1])
            yaw_pallet = self._quat_to_yaw(self.env.pallet.data.root_quat_w[0:1])
            yaw_err = torch.abs((yaw_pallet - yaw_robot + math.pi) % (2 * math.pi) - math.pi)
            
            inserted_enough = self.env._last_insert_depth[0] >= self.env._insert_thresh
            aligned_enough = (lateral_err[0] <= self.cfg.max_lateral_err_m) & (yaw_err[0] <= math.radians(self.cfg.max_yaw_err_deg))
            lock_drive_steer = inserted_enough & aligned_enough
            
            # 计算实际应用的drive值
            drive_before_lock = drive
            drive_after_lock = 0.0 if lock_drive_steer else drive
            
            self.manual_control(drive=drive, steer=steer, lift=0.0, steps=1)
            
            # 记录推进后的位置
            after_pos = self.env.robot.data.root_pos_w[0]
            after_tip = self.get_fork_tip_position()
            pos_delta = after_pos - before_pos
            tip_delta = after_tip - before_tip
            
            if i % 20 == 0 or i < 5:  # 前5步也打印，便于调试
                # 计算实际设置的轮子速度目标
                drive_target_rad_s = drive_after_lock * self.cfg.wheel_speed_rad_s
                
                # 检查执行器接收到的速度目标
                front_wheel_target = None
                back_wheel_target = None
                front_wheel_targets = []
                back_wheel_targets = []
                
                if hasattr(self.env.robot, 'actuators'):
                    for actuator in self.env.robot.actuators.values():
                        joint_indices = getattr(actuator, 'joint_indices', None)
                        if joint_indices is not None:
                            # 检查前轮
                            front_matched = [idx for idx in joint_indices if idx in self.env._front_wheel_ids]
                            if front_matched:
                                if hasattr(actuator, 'data') and hasattr(actuator.data, 'joint_vel_target'):
                                    # 获取所有匹配关节的目标速度
                                    targets = actuator.data.joint_vel_target[0, :len(front_matched)]
                                    front_wheel_targets.extend([t.item() for t in targets])
                            
                            # 检查后轮
                            back_matched = [idx for idx in joint_indices if idx in self.env._back_wheel_ids]
                            if back_matched:
                                if hasattr(actuator, 'data') and hasattr(actuator.data, 'joint_vel_target'):
                                    # 获取所有匹配关节的目标速度
                                    targets = actuator.data.joint_vel_target[0, :len(back_matched)]
                                    back_wheel_targets.extend([t.item() for t in targets])
                
                # 计算平均值
                if front_wheel_targets:
                    front_wheel_target = sum(front_wheel_targets) / len(front_wheel_targets)
                if back_wheel_targets:
                    back_wheel_target = sum(back_wheel_targets) / len(back_wheel_targets)
                
                # 获取实际轮子速度
                front_wheel_vel = self.env._joint_vel[0, self.env._front_wheel_ids].mean().item()
                back_wheel_vel = self.env._joint_vel[0, self.env._back_wheel_ids].mean().item()
                
                print(f"  步数 {i}: dist_front={metrics['dist_front']:.4f}m, insert_depth={metrics['insert_depth']:.4f}m ({metrics['insert_norm']*100:.2f}%)")
                print(f"      位置变化: ({pos_delta[0]:.4f}, {pos_delta[1]:.4f}, {pos_delta[2]:.4f}), 货叉变化: ({tip_delta[0]:.4f}, {tip_delta[1]:.4f}, {tip_delta[2]:.4f})")
                print(f"      控制逻辑: drive_before={drive_before_lock:.2f}, drive_after={drive_after_lock:.2f}, lock={lock_drive_steer}")
                print(f"      速度设置: drive动作={drive_before_lock:.2f}, wheel_speed_rad_s={self.cfg.wheel_speed_rad_s:.2f}, 预期目标={drive_target_rad_s:.4f} rad/s")
                if front_wheel_target is not None or back_wheel_target is not None:
                    target_info = []
                    if front_wheel_target is not None:
                        target_info.append(f"前轮={front_wheel_target:.4f} rad/s")
                    if back_wheel_target is not None:
                        target_info.append(f"后轮={back_wheel_target:.4f} rad/s")
                    print(f"      执行器目标: {', '.join(target_info)}")
                else:
                    print(f"      执行器目标: 无法读取（可能执行器数据未更新）")
                print(f"      实际速度: 前轮={front_wheel_vel:.4f} rad/s, 后轮={back_wheel_vel:.4f} rad/s")
                if front_wheel_target is not None:
                    ratio_front = abs(front_wheel_vel / front_wheel_target) if abs(front_wheel_target) > 1e-6 else 0.0
                    print(f"      速度比: 前轮实际/目标={ratio_front:.2%}")
                if back_wheel_target is not None:
                    ratio_back = abs(back_wheel_vel / back_wheel_target) if abs(back_wheel_target) > 1e-6 else 0.0
                    print(f"      速度比: 后轮实际/目标={ratio_back:.2%}")
                print(f"      状态检查: inserted_enough={inserted_enough}, aligned_enough={aligned_enough.item()}")
                print(f"      阈值: _last_insert_depth={self.env._last_insert_depth[0].item():.4f}m, _insert_thresh={self.env._insert_thresh:.4f}m")
                print(f"      对齐: lateral_err={lateral_err[0].item()*100:.2f}cm, yaw_err={math.degrees(yaw_err[0].item()):.2f}°")
        
        # 4. 验证插入结果
        final_metrics = self.get_insertion_metrics()
        final_insert_depth = final_metrics['insert_depth']
        final_insert_norm = final_metrics['insert_norm']
        final_dist_front = final_metrics['dist_front']
        
        print(f"\n最终插入状态:")
        print_info("插入深度", f"{final_insert_depth:.4f}m")
        print_info("归一化插入深度", f"{final_insert_norm*100:.2f}%")
        print_info("距离托盘前部", f"{final_dist_front:.4f}m")
        
        # 检查物理插入是否发生
        print("\n物理插入检查:")
        final_tip = self.get_fork_tip_position()
        final_pallet_pos = self.env.pallet.data.root_pos_w[0]
        pallet_front_x = final_pallet_pos[0] - self.cfg.pallet_depth_m * 0.5
        pallet_back_x = final_pallet_pos[0] + self.cfg.pallet_depth_m * 0.5
        
        tip_inside = final_tip[0] > pallet_front_x and final_tip[0] < pallet_back_x
        
        print_info("托盘前部x", f"{pallet_front_x:.4f}")
        print_info("托盘后部x", f"{pallet_back_x:.4f}")
        print_info("货叉尖端x", f"{final_tip[0]:.4f}")
        print_info("货叉是否在托盘内部", tip_inside)
        
        # 检查叉车是否被卡住（检查速度和位置变化）
        print("\n叉车运动检查:")
        final_root_pos = self.env.robot.data.root_pos_w[0]
        final_root_vel = self.env.robot.data.root_lin_vel_w[0]
        print_info("叉车位置", f"({final_root_pos[0]:.4f}, {final_root_pos[1]:.4f}, {final_root_pos[2]:.4f})")
        print_info("叉车速度", f"({final_root_vel[0]:.4f}, {final_root_vel[1]:.4f}, {final_root_vel[2]:.4f}) m/s")
        
        # 检查轮子速度
        front_wheel_vel = self.env._joint_vel[0, self.env._front_wheel_ids].mean().item()
        back_wheel_vel = self.env._joint_vel[0, self.env._back_wheel_ids].mean().item()
        print_info("前轮平均速度", f"{front_wheel_vel:.4f} rad/s")
        print_info("后轮平均速度", f"{back_wheel_vel:.4f} rad/s")
        
        # 检查是否有异常的下沉（可能被卡住）
        if final_root_pos[2] < 0.05:
            details.append(f"⚠️  叉车位置过低（z={final_root_pos[2]:.4f}m），可能被卡住或下沉")
        
        # 检查速度是否接近0（可能被卡住）
        speed_magnitude = torch.norm(final_root_vel[:2]).item()
        if speed_magnitude < 0.001 and abs(front_wheel_vel) > 0.01:
            details.append(f"⚠️  轮子在转但叉车不动（轮速={front_wheel_vel:.4f} rad/s，但速度={speed_magnitude:.4f} m/s），可能打滑或被卡住")
        
        # 判断结果
        if final_insert_depth > 0.01:  # 1cm阈值
            details.append(f"✅ 插入深度计算有值（{final_insert_depth:.4f}m）")
        else:
            details.append(f"❌ 插入深度计算为0（这是训练日志中发现的问题）")
            passed = False
        
        if tip_inside:
            details.append(f"✅ 货叉物理上进入了托盘内部")
        else:
            details.append(f"⚠️  货叉未进入托盘内部（可能被碰撞检测阻止）")
            if final_insert_depth == 0:
                details.append("   这可能是插入深度计算为0的原因")
                passed = False
        
        if final_insert_depth >= target_insert_depth * 0.8:  # 达到目标的80%
            details.append(f"✅ 达到目标插入深度（{final_insert_depth:.4f}m >= {target_insert_depth*0.8:.4f}m）")
        else:
            details.append(f"⚠️  未达到目标插入深度（{final_insert_depth:.4f}m < {target_insert_depth*0.8:.4f}m）")
        
        return TestResult(
            name="插入测试",
            passed=passed,
            details="\n".join(details),
            metrics={
                "initial_insert_depth": initial_insert_depth,
                "final_insert_depth": final_insert_depth,
                "target_insert_depth": target_insert_depth,
                "tip_inside_pallet": tip_inside,
            }
        )
    
    def test_lift(self) -> TestResult:
        """测试举升：在插入状态下测试举升功能"""
        print_section("阶段4：举升测试（在插入状态下）")
        
        details = []
        passed = True
        
        # 重置 episode 计时器，防止插入测试步数累积导致 episode timeout
        self.env.episode_length_buf[0] = 0
        
        # 1. 确保已经插入托盘（物理碰撞限制最大约 1.0m，门槛设为可达值）
        min_insert_for_lift = min(0.5, self.env._insert_thresh * 0.5)
        current_metrics = self.get_insertion_metrics()
        if current_metrics['insert_depth'] < min_insert_for_lift:
            print(f"检测到插入不足（{current_metrics['insert_depth']:.3f}m < {min_insert_for_lift:.3f}m），先推进插入...")
            self.set_robot_ideal_position(distance_from_front=0.5)
            
            # 推进插入
            print("推进插入...")
            for i in range(400):
                metrics = self.get_insertion_metrics()
                if metrics['insert_depth'] >= min_insert_for_lift:
                    print(f"  达到插入深度，步数: {i}")
                    break
                
                # 使用修复后的推进逻辑
                abs_dist_front = abs(metrics['dist_front'])
                if metrics['dist_front'] < 0:  # 还未到达托盘前部
                    drive = 0.3 if abs_dist_front > 0.1 else 0.2
                elif metrics['dist_front'] > 0:  # 已超过托盘前部
                    drive = 0.2
                else:
                    drive = 0.2
                
                self.manual_control(drive=drive, steer=0.0, lift=0.0, steps=1)
                
                if i % 40 == 0:
                    print(f"  步数 {i}: dist_front={metrics['dist_front']:.4f}m, insert_depth={metrics['insert_depth']:.4f}m")
            
            current_metrics = self.get_insertion_metrics()
            print(f"插入深度: {current_metrics['insert_depth']:.4f}m")
        
        # 获取初始状态
        initial_lift_pos = self.env._joint_pos[0, self.env._lift_id].item()
        initial_pallet_pos = self.env.pallet.data.root_pos_w[0].clone()
        initial_fork_tip_z = self.get_fork_tip_position()[2].item()
        initial_insert_depth = current_metrics['insert_depth']
        
        print(f"\n初始状态（插入状态下）:")
        print_info("插入深度", f"{initial_insert_depth:.4f}m")
        print_info("升降关节位置", f"{initial_lift_pos:.4f}m")
        print_info("货叉尖端高度", f"{initial_fork_tip_z:.4f}m")
        print_info("托盘位置z", f"{initial_pallet_pos[2]:.4f}m")
        
        # 尝试举升
        print("\n尝试举升...")
        steps = 200  # 增加步数
        
        for i in range(steps):
            # 记录举升前的位置
            before_lift_pos = self.env._joint_pos[0, self.env._lift_id].item()
            before_fork_tip_z = self.get_fork_tip_position()[2].item()
            
            # 检查升降关节的当前状态
            lift_vel_before = self.env._joint_vel[0, self.env._lift_id].item()
            
            self.manual_control(drive=0.0, steer=0.0, lift=0.5, steps=1)
            
            # 记录举升后的位置
            after_lift_pos = self.env._joint_pos[0, self.env._lift_id].item()
            after_fork_tip_z = self.get_fork_tip_position()[2].item()
            lift_vel_after = self.env._joint_vel[0, self.env._lift_id].item()
            lift_delta_step = after_lift_pos - before_lift_pos
            fork_tip_delta_step = after_fork_tip_z - before_fork_tip_z
            
            if i % 40 == 0 or i < 5:  # 前5步也打印
                lift_pos = self.env._joint_pos[0, self.env._lift_id].item()
                fork_tip_z = self.get_fork_tip_position()[2].item()
                pallet_pos = self.env.pallet.data.root_pos_w[0]
                insert_metrics = self.get_insertion_metrics()
                print(f"  步数 {i}: lift_pos={lift_pos:.4f}m, fork_tip_z={fork_tip_z:.4f}m, pallet_z={pallet_pos[2]:.4f}m, insert_depth={insert_metrics['insert_depth']:.4f}m")
                print(f"      单步变化: lift_delta={lift_delta_step:.6f}m, fork_tip_delta={fork_tip_delta_step:.6f}m")
                print(f"      升降速度: before={lift_vel_before:.6f} rad/s, after={lift_vel_after:.6f} rad/s")
                print(f"      lift动作=0.5, lift_speed_m_s={self.cfg.lift_speed_m_s:.2f}, 预期速度={0.5 * self.cfg.lift_speed_m_s:.4f} m/s")
                
                # 检查升降关节的目标速度
                if hasattr(self.env.robot, 'actuators'):
                    for actuator in self.env.robot.actuators.values():
                        # 使用 joint_indices 而不是 joint_ids
                        joint_indices = getattr(actuator, 'joint_indices', None)
                        if joint_indices is not None and self.env._lift_id in joint_indices:
                            if hasattr(actuator, 'data') and hasattr(actuator.data, 'joint_vel_target'):
                                target_vel = actuator.data.joint_vel_target[0, 0].item()
                                print(f"      目标速度: {target_vel:.6f} m/s")
                            break
        
        # 检查最终状态
        final_lift_pos = self.env._joint_pos[0, self.env._lift_id].item()
        final_fork_tip_z = self.get_fork_tip_position()[2].item()
        final_pallet_pos = self.env.pallet.data.root_pos_w[0]
        final_metrics = self.get_insertion_metrics()
        final_insert_depth = final_metrics['insert_depth']
        
        lift_delta = final_lift_pos - initial_lift_pos
        fork_tip_delta = final_fork_tip_z - initial_fork_tip_z
        pallet_delta = final_pallet_pos[2] - initial_pallet_pos[2]
        
        print(f"\n最终状态:")
        print_info("插入深度", f"{final_insert_depth:.4f}m")
        print_info("升降关节位置", f"{final_lift_pos:.4f}m")
        print_info("货叉尖端高度", f"{final_fork_tip_z:.4f}m")
        print_info("托盘位置z", f"{final_pallet_pos[2]:.4f}m")
        print_info("升降变化", f"{lift_delta:.4f}m")
        print_info("货叉高度变化", f"{fork_tip_delta:.4f}m")
        print_info("托盘高度变化", f"{pallet_delta:.4f}m")
        
        # 验证升降关节是否工作
        if lift_delta > 0.01:
            details.append(f"✅ 升降关节正常工作（上升 {lift_delta:.4f}m）")
        else:
            details.append(f"❌ 升降关节未工作（变化 {lift_delta:.4f}m）")
            passed = False
        
        # 验证货叉高度是否增加
        if fork_tip_delta > 0.01:
            details.append(f"✅ 货叉高度增加（上升 {fork_tip_delta:.4f}m）")
        else:
            details.append(f"⚠️  货叉高度未明显增加（变化 {fork_tip_delta:.4f}m）")
        
        # 验证托盘是否跟随（如果是kinematic，应该不跟随）
        if abs(pallet_delta) < 0.001:
            details.append(f"✅ 托盘保持固定（kinematic模式，符合预期）")
        else:
            details.append(f"⚠️  托盘位置变化（{pallet_delta:.4f}m），可能不是kinematic模式")
        
        # 验证插入深度是否保持
        if abs(final_insert_depth - initial_insert_depth) < 0.05:
            details.append(f"✅ 插入深度保持稳定（变化 {abs(final_insert_depth - initial_insert_depth):.4f}m）")
        else:
            details.append(f"⚠️  插入深度变化较大（变化 {abs(final_insert_depth - initial_insert_depth):.4f}m）")
        
        return TestResult(
            name="举升测试",
            passed=passed,
            details="\n".join(details),
            metrics={
                "initial_lift_pos": initial_lift_pos,
                "final_lift_pos": final_lift_pos,
                "lift_delta": lift_delta,
                "pallet_delta": pallet_delta,
                "fork_tip_delta": fork_tip_delta,
            }
        )

    def run_manual_mode(self, auto_align: bool = False):
        """手动控制模式
        
        Args:
            auto_align: 是否在启动时自动对齐（已弃用，现在默认总是对齐）
        """
        print("=" * 80)
        print("Isaac Sim叉车手动控制模式")
        print("=" * 80)

        if not self.initialize_environment(manual_mode=True):
            print("❌ 环境初始化失败")
            return

        # 位置设置已在 initialize_environment() 中统一处理
        print("[INFO] 按 R 键可移动到托盘附近")

        keyboard_cfg = Se2KeyboardCfg(
            v_x_sensitivity=0.5,
            omega_z_sensitivity=0.8,
            sim_device=self.env.device,
        )
        keyboard = ForkliftKeyboard(keyboard_cfg, lift_sensitivity=0.5)

        print("\n键位说明:")
        print("  W/S: 前进/后退")
        print("  A/D: 左转/右转")
        print("  R/F: 货叉上升/下降（Raise/Fall）")
        print("  SPACE: 停止所有动作")
        print("  G: 重置到理想位置")
        print("  P: 打印当前状态")
        print("  ESC: 退出")

        keyboard.add_callback("G", lambda: self.set_robot_ideal_position(distance_from_front=0.5))
        keyboard.add_callback("P", self.print_current_status)
        keyboard.add_callback("ESCAPE", simulation_app.close)

        sim = self.env.sim
        frame_count = 0

        while simulation_app.is_running():
            if hasattr(sim, "is_stopped") and sim.is_stopped():
                break
            if hasattr(sim, "is_playing") and not sim.is_playing():
                sim.step()
                continue

            cmd = keyboard.advance()
            self.manual_control(drive=cmd[0].item(), steer=cmd[1].item(), lift=cmd[2].item(), steps=1)

            if frame_count % 30 == 0:
                self.print_current_status()
            frame_count += 1

        if self.env:
            self.env.close()
        simulation_app.close()
    
    def verify_fork_tip_computation(self) -> TestResult:
        """验证_compute_fork_tip()计算的准确性
        
        验证策略：独立复现运动学计算（root_pos + yaw旋转偏移 + lift），
        而非依赖 body_pos_w（Fabric clone 失败时所有 body 位置等于 root，不可靠）。
        """
        print_section("验证货叉尖端计算")
        
        details = []
        passed = True
        
        root_pos = self.env.robot.data.root_pos_w[0]
        root_quat = self.env.robot.data.root_quat_w[0]
        
        # ---- 1. 独立复现运动学计算 ----
        yaw = self._quat_to_yaw(root_quat)
        fwd_offset = self.env._fork_forward_offset
        z_base = self.env._fork_z_base
        lift_pos = self.env._joint_pos[0, self.env._lift_id].item()
        
        computed_tip_x = root_pos[0].item() + fwd_offset * math.cos(yaw.item())
        computed_tip_y = root_pos[1].item() + fwd_offset * math.sin(yaw.item())
        computed_tip_z = root_pos[2].item() + z_base + lift_pos
        computed_tip = torch.tensor([computed_tip_x, computed_tip_y, computed_tip_z],
                                     device=root_pos.device)
        
        # ---- 2. 环境的 _compute_fork_tip() 结果 ----
        env_tip = self.env._compute_fork_tip()[0]
        
        print("货叉尖端计算验证（运动学复现 vs env._compute_fork_tip）:")
        print_info("独立计算的tip位置", f"({computed_tip[0]:.4f}, {computed_tip[1]:.4f}, {computed_tip[2]:.4f})")
        print_info("环境计算的tip位置", f"({env_tip[0]:.4f}, {env_tip[1]:.4f}, {env_tip[2]:.4f})")
        
        diff = torch.norm(computed_tip - env_tip).item()
        print_info("差异", f"{diff:.6f}")
        
        if diff < 0.01:
            details.append(f"✅ _compute_fork_tip() 运动学一致（差异 {diff:.6f}）")
        else:
            details.append(f"❌ _compute_fork_tip() 运动学不一致（差异 {diff:.6f}）")
            passed = False
        
        # ---- 3. 偏移量合理性检查 ----
        print(f"\n运动学参数:")
        print_info("_fork_forward_offset", f"{fwd_offset:.4f}m")
        print_info("_fork_z_base", f"{z_base:.4f}m")
        print_info("lift_pos", f"{lift_pos:.4f}m")
        print_info("root_pos", f"({root_pos[0]:.4f}, {root_pos[1]:.4f}, {root_pos[2]:.4f})")
        print_info("yaw", f"{math.degrees(yaw.item()):.2f}°")
        
        if 0.5 < fwd_offset < 5.0:
            details.append(f"✅ 前向偏移合理（{fwd_offset:.4f}m）")
        else:
            details.append(f"⚠️  前向偏移可能不合理（{fwd_offset:.4f}m），预期 0.5~5.0m")
        
        # ---- 4. body_pos_w 数据质量诊断（仅信息输出，不影响通过/失败） ----
        body_pos = self.env.robot.data.body_pos_w[0]  # (B, 3)
        rel = body_pos - root_pos.unsqueeze(0)
        max_body_dist = torch.norm(rel, dim=-1).max().item()
        print(f"\n[INFO] body_pos_w 数据质量:")
        print_info("body 最大偏离 root 距离", f"{max_body_dist:.6f}m")
        if max_body_dist < 1e-3:
            print("  ⚠️  所有 body 位置等于 root（Fabric clone 已知问题，不影响 _compute_fork_tip）")
            details.append("ℹ️  body_pos_w 全零（Fabric clone 问题），运动学方法不受影响")
        else:
            details.append(f"✅ body_pos_w 数据正常（最大偏离 {max_body_dist:.4f}m）")
        
        return TestResult(
            name="货叉尖端计算验证",
            passed=passed,
            details="\n".join(details),
            metrics={
                "computed_tip_x": computed_tip[0].item(),
                "env_tip_x": env_tip[0].item(),
                "difference": diff,
                "fork_forward_offset": fwd_offset,
            }
        )
    
    def verify_pallet_front_x_computation(self) -> TestResult:
        """验证_pallet_front_x计算的准确性"""
        print_section("验证托盘前部x坐标计算")
        
        details = []
        passed = True
        
        pallet_pos = self.env.pallet.data.root_pos_w[0]
        pallet_depth = self.cfg.pallet_depth_m
        
        # 计算方式1：使用环境中的计算方式
        computed_front_x = pallet_pos[0] - pallet_depth * 0.5
        
        # 计算方式2：使用环境中的_pallet_front_x
        env_front_x = self.env._pallet_front_x
        
        print("托盘前部x坐标计算验证:")
        print_info("托盘位置x", f"{pallet_pos[0]:.4f}")
        print_info("托盘深度", f"{pallet_depth:.4f}")
        print_info("计算的前部x", f"{computed_front_x:.4f} (= {pallet_pos[0]:.4f} - {pallet_depth*0.5:.4f})")
        print_info("环境中的前部x", f"{env_front_x:.4f}")
        
        # 检查是否一致
        diff = abs(computed_front_x - env_front_x)
        print_info("差异", f"{diff:.6f}")
        
        if diff < 1e-5:
            details.append("✅ _pallet_front_x计算正确")
        else:
            details.append(f"❌ _pallet_front_x计算有误（差异 {diff:.6f}）")
            passed = False
        
        # 验证符号是否正确
        print(f"\n符号验证:")
        print_info("托盘中心x", f"{pallet_pos[0]:.4f}")
        print_info("托盘前部x", f"{computed_front_x:.4f}")
        print_info("托盘后部x", f"{pallet_pos[0] + pallet_depth*0.5:.4f}")
        
        if computed_front_x < pallet_pos[0]:
            details.append("✅ 前部x < 中心x（符号正确，假设x轴向前）")
        else:
            details.append("⚠️  前部x >= 中心x（可能需要检查坐标系）")
        
        return TestResult(
            name="托盘前部x坐标计算验证",
            passed=passed,
            details="\n".join(details),
            metrics={
                "pallet_pos_x": pallet_pos[0].item(),
                "computed_front_x": computed_front_x,
                "env_front_x": env_front_x,
                "difference": diff,
            }
        )
    
    def verify_insertion_depth_computation(self) -> TestResult:
        """验证插入深度计算的准确性"""
        print_section("验证插入深度计算")
        
        details = []
        passed = True
        
        tip = self.get_fork_tip_position()
        pallet_pos = self.env.pallet.data.root_pos_w[0]
        pallet_front_x = self.env._pallet_front_x
        
        # 计算dist_front和insert_depth
        dist_front = tip[0] - pallet_front_x
        insert_depth = max(float(dist_front), 0.0)
        insert_norm = insert_depth / (self.cfg.pallet_depth_m + 1e-6)
        
        print("插入深度计算验证:")
        print_info("货叉尖端x", f"{tip[0]:.4f}")
        print_info("托盘前部x", f"{pallet_front_x:.4f}")
        print_info("dist_front", f"{dist_front:.4f} (= {tip[0]:.4f} - {pallet_front_x:.4f})")
        print_info("insert_depth", f"{insert_depth:.4f} (= clamp({dist_front:.4f}, min=0.0))")
        print_info("insert_norm", f"{insert_norm*100:.2f}%")
        
        # 分析dist_front的符号
        print(f"\ndist_front符号分析:")
        if dist_front < 0:
            details.append(f"⚠️  dist_front < 0（{dist_front:.4f}），表示货叉还未到达托盘前部")
            details.append("   这是训练日志中发现的问题：dist_front_p50 = -2.39m")
            details.append("   如果dist_front < 0，insert_depth会被clamp为0")
            passed = False
        elif dist_front > 0:
            details.append(f"✅ dist_front > 0（{dist_front:.4f}），表示货叉已超过托盘前部")
            if insert_depth > 0:
                details.append(f"✅ insert_depth > 0（{insert_depth:.4f}m），计算正确")
            else:
                details.append(f"❌ insert_depth = 0，但dist_front > 0，计算有误")
                passed = False
        else:
            details.append(f"⚠️  dist_front = 0，货叉刚好在托盘前部")
        
        # 检查物理位置关系
        pallet_back_x = pallet_pos[0] + self.cfg.pallet_depth_m * 0.5
        print(f"\n物理位置关系:")
        print_info("托盘前部x", f"{pallet_front_x:.4f}")
        print_info("托盘中心x", f"{pallet_pos[0]:.4f}")
        print_info("托盘后部x", f"{pallet_back_x:.4f}")
        print_info("货叉尖端x", f"{tip[0]:.4f}")
        
        if pallet_front_x <= tip[0] <= pallet_back_x:
            details.append("✅ 货叉尖端在托盘内部（物理插入成功）")
        elif tip[0] < pallet_front_x:
            details.append("⚠️  货叉尖端在托盘前部之前（未插入）")
        else:
            details.append("⚠️  货叉尖端在托盘后部之后（可能穿透）")
        
        return TestResult(
            name="插入深度计算验证",
            passed=passed,
            details="\n".join(details),
            metrics={
                "dist_front": dist_front,
                "insert_depth": insert_depth,
                "insert_norm": insert_norm,
                "tip_inside": pallet_front_x <= tip[0] <= pallet_back_x,
            }
        )
    
    def generate_report(self):
        """生成测试报告"""
        print_section("测试报告")
        
        print("\n测试结果汇总:")
        print("-" * 80)
        
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results if r.passed)
        
        for i, result in enumerate(self.results, 1):
            status = "✅ 通过" if result.passed else "❌ 失败"
            print(f"\n{i}. {result.name}: {status}")
            print(f"   详情:")
            for line in result.details.split('\n'):
                print(f"     {line}")
            if result.metrics:
                print(f"   指标:")
                for key, value in result.metrics.items():
                    print(f"     {key}: {value}")
        
        print("\n" + "-" * 80)
        print(f"总计: {passed_tests}/{total_tests} 测试通过")
        
        if passed_tests == total_tests:
            print("✅ 所有测试通过！")
        else:
            print(f"⚠️  {total_tests - passed_tests} 个测试未通过")
        
        # 生成诊断建议
        print("\n" + "=" * 80)
        print("诊断建议")
        print("=" * 80)
        
        # 检查插入深度为0的问题
        insert_test = next((r for r in self.results if r.name == "插入测试"), None)
        if insert_test and not insert_test.passed:
            print("\n⚠️  插入测试未通过，可能的原因：")
            print("  1. dist_front < 0，货叉还未到达托盘前部")
            print("  2. 碰撞检测阻止了物理插入")
            print("  3. _pallet_front_x计算有误（符号错误）")
            print("  4. 坐标系转换问题")
        
        # 检查物理验证结果
        fork_tip_test = next((r for r in self.results if r.name == "货叉尖端计算验证"), None)
        if fork_tip_test and not fork_tip_test.passed:
            print("\n⚠️  货叉尖端计算验证未通过，需要检查_compute_fork_tip()实现")
        
        pallet_front_test = next((r for r in self.results if r.name == "托盘前部x坐标计算验证"), None)
        if pallet_front_test and not pallet_front_test.passed:
            print("\n⚠️  托盘前部x坐标计算验证未通过，需要检查_pallet_front_x计算")
    
    # ==================================================================
    # Success Sanity Check（A层逻辑 + B层物理可达性）
    # ==================================================================

    def teleport_to_success_pose(self, desired_insert: float = 1.5, lift_val: float = 0.15, lat_offset: float = 0.0):
        """将叉车传送到"理应成功"的完美位姿，同步所有内部缓存。

        Args:
            desired_insert: 目标插入深度 (m)，默认 1.5m（> 阈值 1.44m）
            lift_val: lift 关节位置 (m)，默认 0.15m（> 阈值 0.12m）
        """
        env = self.env
        cfg = self.cfg
        env_ids = torch.tensor([0], device=env.device)

        # ---- 1. 读取托盘状态 ----
        pallet_pos = env.pallet.data.root_pos_w[0]  # (3,)
        pallet_yaw = self._quat_to_yaw(env.pallet.data.root_quat_w[0])

        # ---- 2. 计算目标 tip 位置（沿托盘插入轴） ----
        # S1.0k: s_front = -0.5 * pallet_depth_m
        # insert_depth = clamp(s_tip - s_front, min=0)
        # 我们要 s_tip = s_front + desired_insert
        # s_tip = (tip_xy - pallet_xy) . u_in
        # 所以 tip 在世界坐标中：
        cos_yaw = math.cos(pallet_yaw.item())
        sin_yaw = math.sin(pallet_yaw.item())

        s_front = -0.5 * cfg.pallet_depth_m
        desired_s_tip = s_front + desired_insert

        # 增加 lateral 偏移 (沿 v_lat 方向)
        v_lat_x = -sin_yaw
        v_lat_y = cos_yaw

        # tip 世界坐标 = pallet_xy + desired_s_tip * u_in + lat_offset * v_lat
        desired_tip_x = pallet_pos[0].item() + desired_s_tip * cos_yaw + lat_offset * v_lat_x
        desired_tip_y = pallet_pos[1].item() + desired_s_tip * sin_yaw + lat_offset * v_lat_y

        # ---- 3. 从 tip 反推 root ----
        # tip = root + fork_forward_offset * [cos(yaw), sin(yaw)]
        fwd = env._fork_forward_offset
        root_x = desired_tip_x - fwd * cos_yaw
        root_y = desired_tip_y - fwd * sin_yaw
        root_z = 0.03  # 与 _reset_idx 一致

        # ---- 4. 写入位姿 ----
        pos = torch.tensor([[root_x, root_y, root_z]], device=env.device)
        quat = self._yaw_to_quat(pallet_yaw).unsqueeze(0)
        env._write_root_pose(env.robot, pos, quat, env_ids)

        zeros3 = torch.zeros((1, 3), device=env.device)
        env._write_root_vel(env.robot, zeros3, zeros3, env_ids)

        # ---- 5. 写入关节状态 (lift=lift_val) ----
        joint_pos = env.robot.data.default_joint_pos[env_ids].clone()
        joint_pos[0, env._lift_id] = lift_val
        joint_vel = torch.zeros_like(joint_pos)
        env._write_joint_state(env.robot, joint_pos, joint_vel, env_ids)
        
        # ---- 5.5 如果举升高度大于0.15，也要把托盘抬起来，否则会穿模爆炸 ----
        if lift_val > 0.15:
            new_pallet_pos = pallet_pos.clone()
            new_pallet_pos[2] = lift_val
            env._write_root_pose(env.pallet, new_pallet_pos.unsqueeze(0), env.pallet.data.root_quat_w, env_ids)
            env._write_root_vel(env.pallet, zeros3, zeros3, env_ids)

        # ---- 6. 物理同步 ----
        # 注意：不能调用 env.sim.reset()！
        # sim.reset() 是全局 PhysX 引擎重置，会将所有环境的位姿覆盖回 config 默认值，
        # 完全摧毁上面刚写入的传送位姿（详见 env.py _reset_idx 中的 S0.7 postmortem）。
        # 正确做法：write_data_to_sim → 单步物理推进 → scene.update 刷新数据
        env.scene.write_data_to_sim()
        env.sim.step(render=False)
        env.scene.update(env.cfg.sim.dt)

        # ---- 7. 刷新关节缓存（physx view → 内部 tensor） ----
        env._joint_pos[:] = env.robot.root_physx_view.get_dof_positions()
        env._joint_vel[:] = env.robot.root_physx_view.get_dof_velocities()

        # ---- 8. 同步所有内部缓存 ----
        env._lift_pos_target[0] = lift_val       # 关键：防止 _apply_action 拉回 0
        env._fork_tip_z0[0] = root_z             # lift_height 基准
        env._last_insert_depth[0] = 0.0
        env._last_lift_pos[0] = 0.0
        env._is_first_step[0] = True
        env._hold_counter[0] = 0
        env.episode_length_buf[0] = 0
        env.actions[0] = 0.0

        # 计算初始 phi_total 并写入缓存（与 _reset_idx 同理）
        y_err_r = torch.tensor([0.0], device=env.device)   # 完美对齐
        yaw_err_deg_r = torch.tensor([0.0], device=env.device)
        dist_front_r = torch.tensor([0.0], device=env.device)  # 已深插入，dist_front=0
        e_band_r = torch.tensor([cfg.d1_min], device=env.device)  # dist_front=0 < d1_min
        E1_r = (e_band_r / cfg.e_band_scale
                + y_err_r / cfg.y_scale1
                + yaw_err_deg_r / cfg.yaw_scale1)
        phi1_r = cfg.k_phi1 / (1.0 + E1_r)
        E2_r = (dist_front_r / cfg.d2_scale
                + y_err_r / cfg.y_scale2
                + yaw_err_deg_r / cfg.yaw_scale2)
        phi2_base_r = cfg.k_phi2 / (1.0 + E2_r)
        # w_band: dist_front=0 < d1_min → (d1_max - 0)/(d1_max - d1_min) > 1 → clamp → 1
        phi2_r = phi2_base_r * 1.0 * 1.0  # w_band=1, w_align2=1 (完美对齐)
        # insert_norm = desired_insert / pallet_depth
        ins_norm_r = desired_insert / (cfg.pallet_depth_m + 1e-6)
        # w3: smoothstep((ins_norm - ins_start) / ins_ramp) → 接近 1
        w3_raw = (ins_norm_r - cfg.ins_start) / cfg.ins_ramp
        w3_r = min(max(w3_raw, 0.0), 1.0)
        w3_r = w3_r * w3_r * (3.0 - 2.0 * w3_r)  # smoothstep
        phi_ins_r = cfg.k_ins * (0.2 + 0.8 * 1.0) * ins_norm_r * w3_r  # w_align3=1
        # phi_lift
        w_lift_base_raw = (ins_norm_r - cfg.insert_gate_norm) / cfg.insert_ramp_norm
        w_lift_base_r = min(max(w_lift_base_raw, 0.0), 1.0)
        w_lift_base_r = w_lift_base_r * w_lift_base_r * (3.0 - 2.0 * w_lift_base_r)
        # lift_height ≈ lift_val (因为 root_z = fork_tip_z0, fork_z_base ≈ 0)
        phi_lift_r = cfg.k_lift * w_lift_base_r * 1.0 * lift_val  # w_align3=1
        phi_total_r = (phi1_r + phi2_r) * (1.0 - w3_r) + phi_ins_r + phi_lift_r
        env._last_phi_total[0] = phi_total_r.item() if isinstance(phi_total_r, torch.Tensor) else phi_total_r

        # 验证传送结果
        tip = self.get_fork_tip_position()
        print(f"  [teleport] root=({root_x:.4f}, {root_y:.4f}, {root_z:.4f})")
        print(f"  [teleport] tip =({tip[0]:.4f}, {tip[1]:.4f}, {tip[2]:.4f})")
        print(f"  [teleport] pallet=({pallet_pos[0]:.4f}, {pallet_pos[1]:.4f}, {pallet_pos[2]:.4f})")
        print(f"  [teleport] lift_joint={lift_val:.4f}m, _lift_pos_target={env._lift_pos_target[0]:.4f}")
        print(f"  [teleport] _fork_tip_z0={env._fork_tip_z0[0]:.4f}, _last_phi_total={env._last_phi_total[0]:.4f}")

    def _read_success_components(self):
        """读取环境内部的 success 判定各分量（需在 step 之后调用）。

        Returns:
            dict with keys: insert_depth, y_err, yaw_err_deg, lift_height,
                            inserted_enough, aligned_enough, lifted_enough,
                            success_now, hold_counter, hold_steps
        """
        env = self.env
        cfg = self.cfg

        # 刷新关节数据（与 _get_rewards 一致）
        env._joint_pos[:] = env.robot.root_physx_view.get_dof_positions()
        env._joint_vel[:] = env.robot.root_physx_view.get_dof_velocities()

        root_pos = env.robot.data.root_pos_w
        pallet_pos = env.pallet.data.root_pos_w
        tip = env._compute_fork_tip()

        robot_yaw = self._quat_to_yaw(env.robot.data.root_quat_w[0])
        pallet_yaw = self._quat_to_yaw(env.pallet.data.root_quat_w[0])

        # S1.0k 中心线几何
        cp = torch.cos(pallet_yaw)
        sp = torch.sin(pallet_yaw)
        u_in = torch.stack([cp, sp])
        v_lat = torch.stack([-sp, cp])

        rel_robot = root_pos[0, :2] - pallet_pos[0, :2]
        y_err = torch.abs(torch.dot(rel_robot, v_lat)).item()

        yaw_err = torch.atan2(
            torch.sin(robot_yaw - pallet_yaw),
            torch.cos(robot_yaw - pallet_yaw),
        )
        yaw_err_deg = abs(yaw_err.item()) * (180.0 / math.pi)
        yaw_err_rad = abs(yaw_err.item())

        # 插入深度
        rel_tip = tip[0, :2] - pallet_pos[0, :2]
        s_tip = torch.dot(rel_tip, u_in).item()
        s_front = -0.5 * cfg.pallet_depth_m
        insert_depth = max(s_tip - s_front, 0.0)

        # 举升
        lift_height = (tip[0, 2] - env._fork_tip_z0[0]).item()

        # S1.0U: tip alignment flags
        stage_dist_front = tip[0, 0].item() - s_front
        is_near = stage_dist_front < cfg.tip_align_near_dist
        
        rel_tip_lat = tip[0, :2] - pallet_pos[0, :2]
        tip_y_signed = torch.dot(rel_tip_lat, v_lat).item()
        tip_y_err = abs(tip_y_signed)
        tip_align_ok = tip_y_err <= cfg.tip_align_entry_m

        # 判定
        inserted_enough = insert_depth >= env._insert_thresh
        aligned_enough = (y_err <= cfg.max_lateral_err_m) and (yaw_err_rad <= math.radians(cfg.max_yaw_err_deg))
        aligned_enough = aligned_enough and (not is_near or tip_align_ok)
        lifted_enough = lift_height >= cfg.lift_delta_m
        success_now = inserted_enough and aligned_enough and lifted_enough

        return {
            "insert_depth": insert_depth,
            "y_err": y_err,
            "yaw_err_deg": yaw_err_deg,
            "tip_y_err": tip_y_err,
            "lift_height": lift_height,
            "inserted_enough": inserted_enough,
            "aligned_enough": aligned_enough,
            "lifted_enough": lifted_enough,
            "success_now": success_now,
            "hold_counter": env._hold_counter[0].item(),
            "hold_steps": env._hold_steps,
            "insert_thresh": env._insert_thresh,
        }

    def run_sanity_check(self):
        """运行 success 判定 sanity check（A层逻辑 + B层物理可达性）"""
        print("=" * 80)
        print("  SUCCESS 判定 SANITY CHECK")
        print("=" * 80)

        # 初始化环境
        if not self.initialize_environment():
            print("[FATAL] 环境初始化失败")
            return

        cfg = self.cfg
        env = self.env

        # 打印关键阈值
        print_section("关键阈值")
        print_info("insert_thresh", f"{env._insert_thresh:.4f}m (insert_fraction={cfg.insert_fraction:.4f} * pallet_depth={cfg.pallet_depth_m:.4f})")
        print_info("max_lateral_err_m", f"{cfg.max_lateral_err_m:.4f}m")
        print_info("max_yaw_err_deg", f"{cfg.max_yaw_err_deg:.2f}deg")
        print_info("lift_delta_m", f"{cfg.lift_delta_m:.4f}m")
        print_info("hold_steps", f"{env._hold_steps} (hold_time_s={cfg.hold_time_s:.2f}s)")
        print_info("fork_forward_offset", f"{env._fork_forward_offset:.4f}m")
        print_info("fork_z_base", f"{env._fork_z_base:.4f}m")

        # 收集诊断结果
        diag = {}

        # ==================================================================
        # A1：传送到完美位姿，step 1 步
        # ==================================================================
        print_section("A1: 传送到完美位姿 → step 1 步")
        self.teleport_to_success_pose(desired_insert=1.5, lift_val=cfg.lift_delta_m + 0.05)

        # step 1 步（零动作）
        actions = torch.tensor([[0.0, 0.0, 0.0]], device=env.device)
        env.step(actions)

        comp = self._read_success_components()
        print("\n  [A1] step 1 后各分量:")
        print_info("insert_depth", f"{comp['insert_depth']:.4f}m (阈值={comp['insert_thresh']:.4f}m) → {'PASS' if comp['inserted_enough'] else 'FAIL'}")
        print_info("y_err", f"{comp['y_err']:.6f}m (阈值={cfg.max_lateral_err_m:.4f}m) → {'PASS' if comp['aligned_enough'] else 'FAIL'}")
        print_info("yaw_err_deg", f"{comp['yaw_err_deg']:.4f}deg (阈值={cfg.max_yaw_err_deg:.2f}deg)")
        print_info("lift_height", f"{comp['lift_height']:.4f}m (阈值={cfg.lift_delta_m:.4f}m) → {'PASS' if comp['lifted_enough'] else 'FAIL'}")
        print_info("success_now", f"{'TRUE' if comp['success_now'] else 'FALSE'}")
        print_info("hold_counter", f"{comp['hold_counter']:.0f}/{comp['hold_steps']}")

        diag["a1_success_now"] = comp["success_now"]
        diag["a1_inserted"] = comp["inserted_enough"]
        diag["a1_aligned"] = comp["aligned_enough"]
        diag["a1_lifted"] = comp["lifted_enough"]
        diag["a1_insert_depth"] = comp["insert_depth"]
        diag["a1_y_err"] = comp["y_err"]
        diag["a1_yaw_err_deg"] = comp["yaw_err_deg"]
        diag["a1_lift_height"] = comp["lift_height"]

        # ==================================================================
        # A2：连续 step 100 步，检查 hold_counter
        # ==================================================================
        print_section("A2: 连续 step 100 步（零动作）检查 hold_counter")

        max_hold = 0
        hold_broke_at = -1
        hold_broke_reason = ""
        a2_terminated = False
        
        # 记录生存曲线
        still_ok_history = []

        for step_i in range(100):
            env.step(actions)
            comp = self._read_success_components()
            hc = int(comp["hold_counter"])
            max_hold = max(max_hold, hc)
            still_ok_history.append(comp["success_now"])

            if step_i < 5 or step_i % 10 == 0 or hc == 0:
                print(f"  step {step_i+2:3d}: hold={hc:3d}, ok={'T' if comp['success_now'] else 'F'} "
                      f"| ins={comp['insert_depth']:.3f} (m={comp['insert_depth']-comp['insert_thresh']:.3f}) "
                      f"| y={comp['y_err']:.3f} (m={cfg.max_lateral_err_m-comp['y_err']:.3f}) "
                      f"| yaw={comp['yaw_err_deg']:.2f} (m={cfg.max_yaw_err_deg-comp['yaw_err_deg']:.2f}) "
                      f"| tip={comp['tip_y_err']:.3f} (m={cfg.tip_align_entry_m-comp['tip_y_err']:.3f}) "
                      f"| lift={comp['lift_height']:.3f} (m={comp['lift_height']-cfg.lift_delta_m:.3f})")

            # 检查 hold 中断
            if not comp["success_now"] and hold_broke_at < 0 and max_hold > 0:
                hold_broke_at = step_i + 2
                reasons = []
                if not comp["inserted_enough"]:
                    reasons.append(f"ins_drop(m={comp['insert_depth']-comp['insert_thresh']:.3f})")
                if not comp["aligned_enough"]:
                    reasons.append(f"align_drop(y_m={cfg.max_lateral_err_m-comp['y_err']:.3f}, yaw_m={cfg.max_yaw_err_deg-comp['yaw_err_deg']:.2f}, tip_m={cfg.tip_align_entry_m-comp['tip_y_err']:.3f})")
                if not comp["lifted_enough"]:
                    reasons.append(f"lift_drop(m={comp['lift_height']-cfg.lift_delta_m:.3f})")
                hold_broke_reason = "; ".join(reasons)

            # 检查 terminated
            terminated, _ = env._get_dones()
            if terminated[0].item():
                a2_terminated = True
                print(f"  → terminated=True at step {step_i+2}, hold_counter={hc}")
                break

        a2_success = max_hold >= env._hold_steps
        print(f"\n  [A2] hold_counter 最高: {max_hold}/{env._hold_steps}")
        print(f"  [A2] success (terminated): {'TRUE' if a2_terminated else 'FALSE'}")
        if hold_broke_at >= 0:
            print(f"  [A2] hold 中断于 step {hold_broke_at}: {hold_broke_reason}")
            
        print(f"  [A2] 生存曲线 (前20步): {''.join(['O' if ok else 'X' for ok in still_ok_history[:20]])}")

        diag["a2_max_hold"] = max_hold
        diag["a2_success"] = a2_success
        diag["a2_terminated"] = a2_terminated
        diag["a2_hold_broke_reason"] = hold_broke_reason

        # ==================================================================
        # A3：场景 B (对齐余量测试)
        # ==================================================================
        print_section("A3: 场景 B (完美位姿 + lateral 偏移 0.05m)")
        env.reset()
        
        self.teleport_to_success_pose(desired_insert=1.5, lift_val=cfg.lift_delta_m + 0.05, lat_offset=0.05)
        
        actions = torch.tensor([[0.0, 0.0, 0.0]], device=env.device)
        env.step(actions)
        
        max_hold_b = 0
        for step_i in range(100):
            env.step(actions)
            comp = self._read_success_components()
            max_hold_b = max(max_hold_b, int(comp["hold_counter"]))
            if terminated[0].item() if (terminated := env._get_dones()[0]) is not None else False: break
            
        print(f"  [A3] 场景 B hold_counter 最高: {max_hold_b}/{env._hold_steps}")

        # ==================================================================
        # A4：场景 C (lift 余量测试)
        # ==================================================================
        print_section(f"A4: 场景 C (完美位姿 + lift 目标 {cfg.lift_delta_m + 0.01:.3f}m 贴线)")
        env.reset()
        self.teleport_to_success_pose(desired_insert=1.5, lift_val=cfg.lift_delta_m + 0.01)
        
        env.step(actions)
        max_hold_c = 0
        for step_i in range(100):
            env.step(actions)
            comp = self._read_success_components()
            max_hold_c = max(max_hold_c, int(comp["hold_counter"]))
            if terminated[0].item() if (terminated := env._get_dones()[0]) is not None else False: break
            
        print(f"  [A4] 场景 C hold_counter 最高: {max_hold_c}/{env._hold_steps}")

        # ==================================================================
        # B1：物理插入深度可达性
        # ==================================================================
        print_section("B1: 物理插入深度可达性（从对齐位置前进）")

        # 重置环境
        env.reset()
        self.set_robot_ideal_position(distance_from_front=0.5)

        max_insert = 0.0
        stall_count = 0
        stall_threshold = 200
        b1_steps = 0

        for step_i in range(800):
            self.manual_control(drive=0.3, steer=0.0, lift=0.0, steps=1)
            metrics = self.get_insertion_metrics()
            cur_insert = metrics["insert_depth"]

            if cur_insert > max_insert + 0.001:
                max_insert = cur_insert
                stall_count = 0
            else:
                stall_count += 1

            b1_steps = step_i + 1

            if step_i % 100 == 0 or stall_count == stall_threshold:
                print(f"  step {step_i:4d}: insert_depth={cur_insert:.4f}m, max={max_insert:.4f}m, stall={stall_count}")

            if stall_count >= stall_threshold:
                # 输出卡住诊断
                front_vel = env._joint_vel[0, env._front_wheel_ids].mean().item()
                root_vel = env.robot.data.root_lin_vel_w[0]
                speed = torch.norm(root_vel[:2]).item()
                print(f"  [B1] 卡住！连续 {stall_threshold} 步无增长")
                print(f"       轮速={front_vel:.4f} rad/s, 车体速度={speed:.4f} m/s")
                if speed < 0.005 and abs(front_vel) > 0.1:
                    print(f"       → 轮子在转但车不动 → 碰撞阻挡")
                elif speed < 0.005:
                    print(f"       → 轮子和车都不动 → 可能被完全卡死")
                else:
                    print(f"       → 车在动但插入深度不增 → 可能滑移/偏航")
                break

        b1_reachable = max_insert >= env._insert_thresh
        print(f"\n  [B1] 最大插入深度: {max_insert:.4f}m (阈值={env._insert_thresh:.4f}m)")
        print(f"  [B1] 结论: {'可达' if b1_reachable else '不可达'}")
        if not b1_reachable:
            print(f"  [B1] 缺口: {env._insert_thresh - max_insert:.4f}m")

        diag["b1_max_insert"] = max_insert
        diag["b1_reachable"] = b1_reachable
        diag["b1_steps"] = b1_steps

        # ==================================================================
        # B2：物理举升高度可达性
        # ==================================================================
        print_section("B2: 物理举升高度可达性（在当前插入状态下举升）")

        # 重置 episode 计时器
        env.episode_length_buf[0] = 0

        max_lift = -999.0
        stall_count = 0
        stall_threshold = 60
        b2_steps = 0
        initial_lift_pos = env._joint_pos[0, env._lift_id].item()

        for step_i in range(400):
            self.manual_control(drive=0.0, steer=0.0, lift=1.0, steps=1)

            tip = self.get_fork_tip_position()
            cur_lift = (tip[2] - env._fork_tip_z0[0]).item()

            if cur_lift > max_lift + 0.001:
                max_lift = cur_lift
                stall_count = 0
            else:
                stall_count += 1

            b2_steps = step_i + 1

            if step_i % 60 == 0 or stall_count == stall_threshold:
                lift_joint = env._joint_pos[0, env._lift_id].item()
                print(f"  step {step_i:4d}: lift_height={cur_lift:.4f}m, max={max_lift:.4f}m, lift_joint={lift_joint:.4f}m")

            if stall_count >= stall_threshold:
                lift_joint = env._joint_pos[0, env._lift_id].item()
                print(f"  [B2] 卡住！连续 {stall_threshold} 步无增长")
                print(f"       lift_joint={lift_joint:.4f}m, _lift_pos_target={env._lift_pos_target[0]:.4f}m")
                break

        b2_reachable = max_lift >= cfg.lift_delta_m
        print(f"\n  [B2] 最大举升高度: {max_lift:.4f}m (阈值={cfg.lift_delta_m:.4f}m)")
        print(f"  [B2] 结论: {'可达' if b2_reachable else '不可达'}")

        diag["b2_max_lift"] = max_lift
        diag["b2_reachable"] = b2_reachable
        diag["b2_steps"] = b2_steps

        # ==================================================================
        # 综合诊断报告
        # ==================================================================
        print("\n")
        print("=" * 64)
        print("          SUCCESS SANITY CHECK 诊断报告")
        print("=" * 64)

        print("\n--- A 层：判定逻辑验证 ---")
        a1_ok = diag["a1_success_now"]
        print(f"[A1] 传送到完美位姿后 (step 1):")
        print(f"  inserted_enough: {'PASS' if diag['a1_inserted'] else 'FAIL'}  (insert_depth={diag['a1_insert_depth']:.4f}m, 阈值={env._insert_thresh:.4f}m)")
        print(f"  aligned_enough:  {'PASS' if diag['a1_aligned'] else 'FAIL'}  (y_err={diag['a1_y_err']:.6f}m, yaw_err={diag['a1_yaw_err_deg']:.4f}deg)")
        print(f"  lifted_enough:   {'PASS' if diag['a1_lifted'] else 'FAIL'}  (lift_height={diag['a1_lift_height']:.4f}m, 阈值={cfg.lift_delta_m:.4f}m)")
        print(f"  success_now:     {'PASS' if a1_ok else 'FAIL'}")

        print(f"\n[A2] 连续 hold 40 步:")
        print(f"  hold_counter 最高: {diag['a2_max_hold']}/{env._hold_steps}")
        print(f"  success (terminated): {'PASS' if diag['a2_terminated'] else 'FAIL'}")
        if diag["a2_hold_broke_reason"]:
            print(f"  hold 中断原因: {diag['a2_hold_broke_reason']}")

        print(f"\n--- B 层：物理可达性验证 ---")
        print(f"[B1] 物理最大插入深度: {diag['b1_max_insert']:.4f}m (阈值={env._insert_thresh:.4f}m)")
        print(f"  结论: {'可达' if diag['b1_reachable'] else '不可达'}")

        print(f"\n[B2] 物理最大举升高度: {diag['b2_max_lift']:.4f}m (阈值={cfg.lift_delta_m:.4f}m)")
        print(f"  结论: {'可达' if diag['b2_reachable'] else '不可达'}")

        print(f"\n{'=' * 64}")
        print(f"          综合诊断")
        print(f"{'=' * 64}")

        has_issue = False
        if not a1_ok:
            has_issue = True
            if not diag["a1_inserted"]:
                print("  [CRITICAL] A1 inserted_enough=FAIL → 插入深度计算/阈值/坐标系有 bug")
            if not diag["a1_aligned"]:
                print("  [CRITICAL] A1 aligned_enough=FAIL → 对齐误差计算/阈值有 bug")
            if not diag["a1_lifted"]:
                print("  [CRITICAL] A1 lifted_enough=FAIL → lift_height 计算或 _fork_tip_z0 有 bug")

        if not diag["a2_terminated"] and a1_ok:
            has_issue = True
            print("  [CRITICAL] A2 hold_counter 未累积到阈值 → 位姿漂移或 hold_counter 逻辑 bug")

        if not diag["b1_reachable"]:
            has_issue = True
            print(f"  [CRITICAL] B1 物理最大插入={diag['b1_max_insert']:.4f}m < 阈值 {env._insert_thresh:.4f}m")
            print(f"             → 碰撞配置阻止深插入，需修改碰撞体或降低 insert_fraction")

        if not diag["b2_reachable"]:
            has_issue = True
            print(f"  [WARNING]  B2 物理最大举升={diag['b2_max_lift']:.4f}m < 阈值 {cfg.lift_delta_m:.4f}m")
            print(f"             → lift 关节驱动/限位有问题")

        if a1_ok and diag["a2_terminated"] and not diag["b1_reachable"]:
            print(f"\n  >>> 训练 success=0 的根因：物理上插入深度不可达 <<<")
            print(f"  >>> 建议：降低 insert_fraction 或修改碰撞体 <<<")

        if a1_ok and diag["a2_terminated"] and diag["b1_reachable"] and diag["b2_reachable"]:
            print(f"\n  >>> 判定逻辑和物理可达性均正常，训练 success=0 是策略问题 <<<")
            print(f"  >>> 建议：调整奖励/课程学习/超参数 <<<")

        if not has_issue:
            print("  所有检查通过！")

        print("=" * 64)
        import sys
        sys.stdout.flush()

        # 关闭
        if env:
            env.close()
        simulation_app.close()

    def run_all_tests(self):
        """运行所有测试"""
        print("=" * 80)
        print("Isaac Sim叉车插入举升功能验证")
        print("=" * 80)
        
        # 初始化环境
        if not self.initialize_environment():
            print("❌ 环境初始化失败")
            return
        
        # 运行测试
        self.results.append(self.check_environment_init())
        
        # 物理验证（在插入测试前后进行）
        self.results.append(self.verify_fork_tip_computation())
        self.results.append(self.verify_pallet_front_x_computation())
        
        self.results.append(self.test_approach())
        self.results.append(self.test_alignment())
        self.results.append(self.test_insertion())
        
        # 在插入测试之后验证插入深度计算（此时货叉已进入托盘，dist_front > 0）
        self.results.append(self.verify_insertion_depth_computation())
        
        self.results.append(self.test_lift())
        
        # 生成报告
        self.generate_report()
        
        # 关闭环境
        if self.env:
            self.env.close()
        simulation_app.close()


def main():
    """主函数"""
    verifier = ForkliftVerification()
    if getattr(args_cli, "sanity_check", False):
        verifier.run_sanity_check()
    elif args_cli.manual:
        if getattr(args_cli, "headless", False):
            print("[错误] 手动模式需要可视化界面，请去掉 --headless 参数运行。")
            simulation_app.close()
            return
        verifier.run_manual_mode(auto_align=args_cli.auto_align)
    else:
        verifier.run_all_tests()


if __name__ == "__main__":
    main()
