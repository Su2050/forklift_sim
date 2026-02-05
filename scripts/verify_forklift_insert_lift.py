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
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()

# 启动 Isaac Sim
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# 在 Isaac Sim 初始化后导入
import carb

from isaaclab.devices import Se2Keyboard, Se2KeyboardCfg
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
        self.env.scene.write_data_to_sim()
        self.env.scene.update(self.env.cfg.sim.dt)
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
        insert_depth = torch.clamp(torch.tensor(dist_front), min=0.0).item()
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
        
        # 使用环境的_write_root_pose方法设置
        env_ids = torch.tensor([0], device=self.env.device)
        self.env._write_root_pose(self.env.robot, ideal_pos.unsqueeze(0), ideal_quat.unsqueeze(0), env_ids)
        
        # 重置速度为零
        zeros3 = torch.zeros((1, 3), device=self.env.device)
        self.env._write_root_vel(self.env.robot, zeros3, zeros3, env_ids)
        
        # 同步到sim并更新
        self.env.scene.write_data_to_sim()
        self.env.scene.update(self.env.cfg.sim.dt)
        
        # 重要：更新环境的缓存状态（_last_insert_depth等）
        # 重新计算插入深度并更新缓存
        tip = self.env._compute_fork_tip()
        insert_depth = torch.clamp(tip[:, 0] - self.env._pallet_front_x, min=0.0)
        self.env._last_insert_depth[0] = insert_depth[0]
        
        # 更新其他缓存
        root_pos = self.env.robot.data.root_pos_w[0:1]
        pallet_pos = self.env.pallet.data.root_pos_w[0:1]
        lateral_err = torch.abs(pallet_pos[:, 1] - root_pos[:, 1])
        yaw_robot = self._quat_to_yaw(self.env.robot.data.root_quat_w[0:1])
        yaw_pallet = self._quat_to_yaw(self.env.pallet.data.root_quat_w[0:1])
        yaw_err = torch.abs((yaw_pallet - yaw_robot + math.pi) % (2 * math.pi) - math.pi)
        E_align = lateral_err + yaw_err * 0.1
        self.env._last_E_align[0] = E_align[0]
        
        # 更新dist_front缓存
        dist_front = tip[:, 0] - self.env._pallet_front_x
        self.env._last_dist_front[0] = dist_front[0]
        
        # 更新lift位置缓存
        self.env._last_lift_pos[0] = self.env._joint_pos[0, self.env._lift_id]
        
        # 重置actions缓存
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
        target_insert_depth = 0.3  # 目标插入深度30cm
        max_steps = 200
        
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
                drive = 0.1  # 已插入，慢速推进
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
        
        # 1. 确保已经插入托盘
        current_metrics = self.get_insertion_metrics()
        if current_metrics['insert_depth'] < 0.1:
            print("检测到未插入状态，先设置理想对齐位置并推进插入...")
            self.set_robot_ideal_position(distance_from_front=0.5)
            
            # 推进插入
            print("推进插入...")
            for i in range(200):
                metrics = self.get_insertion_metrics()
                if metrics['insert_depth'] >= 0.2:  # 插入20cm
                    print(f"  达到插入深度，步数: {i}")
                    break
                
                # 使用修复后的推进逻辑
                abs_dist_front = abs(metrics['dist_front'])
                if metrics['dist_front'] < 0:  # 还未到达托盘前部
                    drive = 0.3 if abs_dist_front > 0.1 else 0.2
                elif metrics['dist_front'] > 0:  # 已超过托盘前部
                    drive = 0.1
                else:
                    drive = 0.2
                
                self.manual_control(drive=drive, steer=0.0, lift=0.0, steps=1)
                
                if i % 20 == 0:
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
        print("  Q/E: 升降上升/下降")
        print("  SPACE: 停止所有动作")
        print("  R: 重置到理想位置")
        print("  P: 打印当前状态")
        print("  ESC: 退出")

        keyboard.add_callback("R", lambda: self.set_robot_ideal_position(distance_from_front=0.5))
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
        """验证_compute_fork_tip()计算的准确性"""
        print_section("验证货叉尖端计算")
        
        details = []
        passed = True
        
        # 获取所有body位置
        root_pos = self.env.robot.data.root_pos_w[0]
        root_quat = self.env.robot.data.root_quat_w[0]
        body_pos = self.env.robot.data.body_pos_w[0]  # (B, 3)
        
        # 计算yaw和forward向量
        yaw = self._quat_to_yaw(root_quat)
        fwd = torch.stack([torch.cos(yaw), torch.sin(yaw), torch.zeros_like(yaw)])
        
        # 计算每个body的投影
        rel = body_pos - root_pos.unsqueeze(0)  # (B, 3)
        proj = (rel * fwd.unsqueeze(0)).sum(-1)  # (B,)
        idx = torch.argmax(proj)
        
        # 获取计算出的tip
        computed_tip = body_pos[idx]
        
        # 使用环境的_compute_fork_tip()方法
        env_tip = self.env._compute_fork_tip()[0]
        
        print("货叉尖端计算验证:")
        print_info("计算出的tip位置", f"({computed_tip[0]:.4f}, {computed_tip[1]:.4f}, {computed_tip[2]:.4f})")
        print_info("环境计算的tip位置", f"({env_tip[0]:.4f}, {env_tip[1]:.4f}, {env_tip[2]:.4f})")
        
        # 检查是否一致
        diff = torch.norm(computed_tip - env_tip).item()
        print_info("差异", f"{diff:.6f}")
        
        if diff < 1e-5:
            details.append("✅ _compute_fork_tip()计算正确")
        else:
            details.append(f"❌ _compute_fork_tip()计算有误（差异 {diff:.6f}）")
            passed = False
        
        # 打印投影最大的body信息
        print(f"\n投影最大的body:")
        print_info("Body索引", f"{idx.item()}")
        print_info("投影值", f"{proj[idx].item():.4f}")
        print_info("Body位置", f"({body_pos[idx][0]:.4f}, {body_pos[idx][1]:.4f}, {body_pos[idx][2]:.4f})")
        
        # 打印所有body的投影（前5个最大的）
        top5_proj, top5_idx = torch.topk(proj, min(5, len(proj)))
        print(f"\n投影前5的body:")
        for i, (proj_val, body_idx) in enumerate(zip(top5_proj, top5_idx)):
            body_name = self.env.robot.body_names[body_idx] if hasattr(self.env.robot, 'body_names') else f"body_{body_idx}"
            print(f"  {i+1}. {body_name}: 投影={proj_val:.4f}, 位置=({body_pos[body_idx][0]:.4f}, {body_pos[body_idx][1]:.4f}, {body_pos[body_idx][2]:.4f})")
        
        return TestResult(
            name="货叉尖端计算验证",
            passed=passed,
            details="\n".join(details),
            metrics={
                "computed_tip_x": computed_tip[0].item(),
                "env_tip_x": env_tip[0].item(),
                "difference": diff,
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
        insert_depth = torch.clamp(torch.tensor(dist_front), min=0.0).item()
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
        
        # 在插入测试前再次验证计算
        self.results.append(self.verify_insertion_depth_computation())
        
        self.results.append(self.test_insertion())
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
    if args_cli.manual:
        if getattr(args_cli, "headless", False):
            print("[错误] 手动模式需要可视化界面，请去掉 --headless 参数运行。")
            simulation_app.close()
            return
        verifier.run_manual_mode(auto_align=args_cli.auto_align)
    else:
        verifier.run_all_tests()


if __name__ == "__main__":
    main()
