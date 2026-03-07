"""叉车插盘举升策略 — 独立推理脚本。

本脚本从 model_1999.pt 中提取 Actor 网络权重与观测归一化参数,
构建一个零依赖（仅需 torch / numpy）的推理管线。

使用方式：
  1. 作为库导入：
       from infer import ForkliftPolicy
       policy = ForkliftPolicy("model_1999.pt")
       action = policy.infer(obs_15d)

  2. 命令行快速测试：
       python infer.py --model model_1999.pt --test
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Actor network (mirrors rsl_rl ActorCritic.actor)
# ---------------------------------------------------------------------------
class ActorMLP(nn.Module):
    """与训练时完全一致的 Actor 网络结构。

    结构: Linear(15,256) → ELU → Linear(256,256) → ELU →
          Linear(256,128) → ELU → Linear(128,3)
    """

    def __init__(self, obs_dim: int = 15, act_dim: int = 3,
                 hidden_dims: tuple[int, ...] = (256, 256, 128)):
        super().__init__()
        layers: list[nn.Module] = []
        prev = obs_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ELU())
            prev = h
        layers.append(nn.Linear(prev, act_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Running observation normalizer
# ---------------------------------------------------------------------------
class ObsNormalizer:
    """复现训练时的 running mean/std 归一化。"""

    def __init__(self, mean: np.ndarray, std: np.ndarray):
        self.mean = mean.astype(np.float32)
        self.std = np.clip(std.astype(np.float32), a_min=1e-6, a_max=None)

    def normalize(self, obs: np.ndarray) -> np.ndarray:
        return (obs - self.mean) / self.std


# ---------------------------------------------------------------------------
# 观测构建辅助函数
# ---------------------------------------------------------------------------
def build_obs(
    pallet_pos_robot: np.ndarray,
    pallet_yaw_robot: float,
    robot_vel_xy_robot: np.ndarray,
    yaw_rate: float,
    lift_pos: float,
    lift_vel: float,
    insert_depth_m: float,
    prev_actions: np.ndarray,
    y_signed_m: float,
    dyaw_rad: float,
    pallet_depth_m: float = 2.16,
    y_err_obs_scale: float = 0.8,
    lift_pos_scale: float = 1.0,
) -> np.ndarray:
    """从传感器原始数据构建 15 维观测向量。

    Args:
        pallet_pos_robot: (2,) 托盘在 robot frame 中的 [x, y] 位置 (m)
        pallet_yaw_robot: 托盘朝向与叉车朝向的偏航角差 (rad)
        robot_vel_xy_robot: (2,) 叉车在 robot frame 中的 [vx, vy] 速度 (m/s)
        yaw_rate: 叉车偏航角速度 (rad/s)
        lift_pos: 升降关节位置 (m)
        lift_vel: 升降关节速度 (m/s)
        insert_depth_m: 叉齿已插入托盘的深度 (m), ≥0
        prev_actions: (3,) 上一步动作 [drive, steer, lift], 范围 [-1, 1]
        y_signed_m: 叉车相对托盘中心线的横向偏差 (m, 带符号, 左正右负)
        dyaw_rad: 叉车朝向与托盘朝向的偏航差 (rad, 带符号)
        pallet_depth_m: 托盘深度 (m), 默认 2.16 (1.2m × 1.8 缩放)
        y_err_obs_scale: 横向误差归一化尺度, 默认 0.8
        lift_pos_scale: lift_pos 缩放因子, 默认 1.0

    Returns:
        obs: (15,) float32 观测向量
    """
    d_xy_r = pallet_pos_robot[:2].astype(np.float32)
    cos_dyaw = np.float32(math.cos(pallet_yaw_robot))
    sin_dyaw = np.float32(math.sin(pallet_yaw_robot))
    v_xy_r = robot_vel_xy_robot[:2].astype(np.float32)
    yr = np.float32(yaw_rate)
    lp = np.float32(lift_pos / lift_pos_scale)
    lv = np.float32(lift_vel)
    insert_norm = np.float32(np.clip(insert_depth_m / (pallet_depth_m + 1e-6), 0.0, 1.0))
    pa = prev_actions[:3].astype(np.float32)
    y_err_obs = np.float32(np.clip(y_signed_m / y_err_obs_scale, -1.0, 1.0))
    yaw_err_obs = np.float32(np.clip(dyaw_rad / math.radians(15.0), -1.0, 1.0))

    return np.array([
        d_xy_r[0], d_xy_r[1],          # 0-1
        cos_dyaw, sin_dyaw,             # 2-3
        v_xy_r[0], v_xy_r[1],          # 4-5
        yr,                             # 6
        lp, lv,                         # 7-8
        insert_norm,                    # 9
        pa[0], pa[1], pa[2],           # 10-12
        y_err_obs,                      # 13
        yaw_err_obs,                    # 14
    ], dtype=np.float32)


# ---------------------------------------------------------------------------
# 动作映射
# ---------------------------------------------------------------------------
ACTION_SCALES = {
    "wheel_speed_rad_s": 20.0,
    "steer_angle_rad": 0.6,
    "lift_speed_m_s": 0.5,
}


def map_action_to_physical(action: np.ndarray) -> dict:
    """将 [-1, 1] 归一化动作映射到物理指令。

    Args:
        action: (3,) 网络输出 [drive, steer, lift]

    Returns:
        dict with keys: drive_rad_s, steer_rad, lift_m_s
    """
    return {
        "drive_rad_s": float(action[0] * ACTION_SCALES["wheel_speed_rad_s"]),
        "steer_rad": float(action[1] * ACTION_SCALES["steer_angle_rad"]),
        "lift_m_s": float(action[2] * ACTION_SCALES["lift_speed_m_s"]),
    }


# ---------------------------------------------------------------------------
# 主策略类
# ---------------------------------------------------------------------------
class ForkliftPolicy:
    """叉车插盘举升策略的推理封装。

    Example:
        policy = ForkliftPolicy("model_1999.pt", device="cpu")
        obs = build_obs(...)
        action = policy.infer(obs)            # (3,) np.float32, [-1, 1]
        cmd = map_action_to_physical(action)  # dict: drive/steer/lift
    """

    def __init__(self, model_path: str, device: str = "cpu"):
        self.device = torch.device(device)

        ckpt = torch.load(model_path, map_location=self.device, weights_only=False)
        sd = ckpt["model_state_dict"]

        # --- 观测归一化参数 ---
        obs_mean = sd["actor_obs_normalizer._mean"].squeeze().cpu().numpy()
        obs_std = sd["actor_obs_normalizer._std"].squeeze().cpu().numpy()
        self.normalizer = ObsNormalizer(obs_mean, obs_std)

        # --- Actor 网络 ---
        self.actor = ActorMLP(obs_dim=15, act_dim=3, hidden_dims=(256, 256, 128))
        actor_sd = {}
        for k, v in sd.items():
            if k.startswith("actor.") and "normalizer" not in k:
                new_key = k.replace("actor.", "net.")
                actor_sd[new_key] = v
        self.actor.load_state_dict(actor_sd)
        self.actor.to(self.device)
        self.actor.eval()

        self._prev_actions = np.zeros(3, dtype=np.float32)

        print(f"[ForkliftPolicy] loaded from {model_path}")
        print(f"  obs normalizer count: {sd['actor_obs_normalizer.count'].item():.0f}")
        print(f"  action noise std: {torch.exp(sd['log_std']).cpu().numpy()}")

    @torch.no_grad()
    def infer(self, obs_15d: np.ndarray) -> np.ndarray:
        """单步推理。

        Args:
            obs_15d: (15,) 原始观测向量（未归一化）

        Returns:
            action: (3,) 动作, 已 clip 到 [-1, 1]
        """
        obs_norm = self.normalizer.normalize(obs_15d)
        obs_t = torch.from_numpy(obs_norm).unsqueeze(0).to(self.device)
        action_t = self.actor(obs_t).squeeze(0).cpu().numpy()
        action = np.clip(action_t, -1.0, 1.0)
        self._prev_actions = action.copy()
        return action

    @torch.no_grad()
    def infer_batch(self, obs_batch: np.ndarray) -> np.ndarray:
        """批量推理。

        Args:
            obs_batch: (N, 15) 原始观测向量（未归一化）

        Returns:
            actions: (N, 3) 动作, 已 clip 到 [-1, 1]
        """
        obs_norm = (obs_batch - self.normalizer.mean) / self.normalizer.std
        obs_t = torch.from_numpy(obs_norm.astype(np.float32)).to(self.device)
        actions_t = self.actor(obs_t).cpu().numpy()
        return np.clip(actions_t, -1.0, 1.0)

    @property
    def prev_actions(self) -> np.ndarray:
        """上一步动作（用于构建下一步 obs[10:13]）。"""
        return self._prev_actions.copy()


# ---------------------------------------------------------------------------
# 命令行测试
# ---------------------------------------------------------------------------
def _run_test(model_path: str):
    """用模拟数据测试推理管线。"""
    policy = ForkliftPolicy(model_path, device="cpu")

    print("\n" + "=" * 60)
    print("  推理管线测试")
    print("=" * 60)

    # 模拟场景：叉车在托盘左后方 3.5m，偏航 5°
    obs = build_obs(
        pallet_pos_robot=np.array([3.5, 0.2]),
        pallet_yaw_robot=math.radians(5.0),
        robot_vel_xy_robot=np.array([0.0, 0.0]),
        yaw_rate=0.0,
        lift_pos=0.0,
        lift_vel=0.0,
        insert_depth_m=0.0,
        prev_actions=np.array([0.0, 0.0, 0.0]),
        y_signed_m=0.2,
        dyaw_rad=math.radians(5.0),
    )

    print(f"\n输入观测 (15 dim):")
    labels = [
        "d_x_r", "d_y_r", "cos_dyaw", "sin_dyaw",
        "v_x_r", "v_y_r", "yaw_rate",
        "lift_pos", "lift_vel", "insert_norm",
        "act_drive", "act_steer", "act_lift",
        "y_err_obs", "yaw_err_obs",
    ]
    for i, (name, val) in enumerate(zip(labels, obs)):
        print(f"  [{i:2d}] {name:12s} = {val:+.4f}")

    # 推理
    t0 = time.perf_counter()
    n_iters = 1000
    for _ in range(n_iters):
        action = policy.infer(obs)
    dt = (time.perf_counter() - t0) / n_iters * 1000

    cmd = map_action_to_physical(action)

    print(f"\n输出动作 (3 dim):")
    print(f"  [0] drive  = {action[0]:+.4f}  →  {cmd['drive_rad_s']:+.2f} rad/s")
    print(f"  [1] steer  = {action[1]:+.4f}  →  {cmd['steer_rad']:+.2f} rad")
    print(f"  [2] lift   = {action[2]:+.4f}  →  {cmd['lift_m_s']:+.4f} m/s")

    print(f"\n推理延迟: {dt:.3f} ms / step（{n_iters} 次平均）")
    print(f"等效控制频率: {1000/dt:.0f} Hz")

    # 批量推理测试
    batch = np.tile(obs, (1024, 1))
    t0 = time.perf_counter()
    for _ in range(100):
        policy.infer_batch(batch)
    dt_batch = (time.perf_counter() - t0) / 100 * 1000
    print(f"批量推理 (1024 envs): {dt_batch:.2f} ms / batch")

    print("\n" + "=" * 60)
    print("  测试通过 ✓")
    print("=" * 60)


def _export_onnx(model_path: str, output_path: str):
    """导出为 ONNX 格式（含归一化层）。"""

    class PolicyWithNorm(nn.Module):
        def __init__(self, actor: ActorMLP, mean: torch.Tensor, std: torch.Tensor):
            super().__init__()
            self.actor = actor
            self.register_buffer("obs_mean", mean)
            self.register_buffer("obs_std", std)

        def forward(self, obs_raw: torch.Tensor) -> torch.Tensor:
            obs_norm = (obs_raw - self.obs_mean) / self.obs_std
            return torch.clamp(self.actor(obs_norm), -1.0, 1.0)

    policy = ForkliftPolicy(model_path, device="cpu")
    wrapped = PolicyWithNorm(
        policy.actor,
        torch.from_numpy(policy.normalizer.mean).unsqueeze(0),
        torch.from_numpy(policy.normalizer.std).unsqueeze(0),
    )
    wrapped.eval()

    dummy = torch.randn(1, 15)
    try:
        torch.onnx.export(
            wrapped, dummy, output_path,
            input_names=["obs_raw"],
            output_names=["action"],
            dynamic_axes={"obs_raw": {0: "batch"}, "action": {0: "batch"}},
            opset_version=17,
        )
        print(f"[ONNX] 已导出到 {output_path}")
    except ModuleNotFoundError as e:
        print(f"[ONNX] 导出失败: {e}")
        print("  请安装: pip install onnx onnxscript")
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="叉车插盘举升策略推理")
    parser.add_argument("--model", type=str, default="model_1999.pt",
                        help="模型文件路径")
    parser.add_argument("--test", action="store_true",
                        help="运行推理管线测试")
    parser.add_argument("--export-onnx", type=str, default=None,
                        help="导出 ONNX 模型到指定路径 (如 policy.onnx)")
    args = parser.parse_args()

    if args.export_onnx:
        _export_onnx(args.model, args.export_onnx)
    elif args.test:
        _run_test(args.model)
    else:
        parser.print_help()
