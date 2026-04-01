#!/usr/bin/env python3
"""Independent forklift camera evaluation tool."""

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher


REPO_ROOT = Path(__file__).resolve().parents[3]


parser = argparse.ArgumentParser(description="Evaluate forklift camera pose and record camera stream.")
parser.add_argument("--cam-name", type=str, required=True, help="Output directory name under outputs/camera_eval.")
parser.add_argument("--cam-x", type=float, required=True, help="Camera X offset in cm.")
parser.add_argument("--cam-y", type=float, required=True, help="Camera Y offset in cm.")
parser.add_argument("--cam-z", type=float, required=True, help="Camera Z offset in cm.")
parser.add_argument("--pitch-deg", type=float, required=True, help="Camera pitch in degrees.")
parser.add_argument("--yaw-deg", type=float, default=0.0, help="Camera yaw in degrees.")
parser.add_argument("--roll-deg", type=float, default=0.0, help="Camera roll in degrees.")
parser.add_argument("--mount-body", type=str, default="body", help="Mount body name.")
parser.add_argument("--hfov-deg", type=float, default=90.0, help="Horizontal field of view in degrees.")
parser.add_argument("--resolution", type=int, default=320, help="Square camera resolution.")
parser.add_argument("--steps", type=int, default=150, help="Number of simulation steps to run.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import cv2
import gymnasium as gym
import numpy as np
import torch
from PIL import Image
from pxr import Gf, Sdf, UsdGeom, UsdShade

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.direct.forklift_pallet_insert_lift.env_cfg import ForkliftPalletInsertLiftEnvCfg

def ensure_preview_material(stage, mat_path: str, color: tuple[float, float, float]):
    material = UsdShade.Material.Define(stage, mat_path)
    shader = UsdShade.Shader.Define(stage, f"{mat_path}/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.2)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return material


def spawn_marker_cube(stage, prim_path: str, pos_xyz: tuple[float, float, float], size: float,
                      color: tuple[float, float, float], mat_path: str):
    cube = UsdGeom.Cube.Define(stage, prim_path)
    cube.GetSizeAttr().Set(size)
    cube.ClearXformOpOrder()
    cube.AddTranslateOp().Set(Gf.Vec3d(*pos_xyz))
    cube.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    material = ensure_preview_material(stage, mat_path, color)
    UsdShade.MaterialBindingAPI(cube.GetPrim()).Bind(material)


def tensor_to_rgb_image(img_tensor: torch.Tensor) -> np.ndarray:
    img_np = img_tensor.detach().cpu().numpy()
    img_np = np.transpose(img_np, (1, 2, 0))
    if img_np.dtype != np.uint8:
        img_np = (img_np * 255.0).clip(0, 255).astype(np.uint8)
    return img_np


def main():
    print("[DEBUG] Starting main()")
    out_dir = REPO_ROOT / "outputs" / "camera_eval" / args_cli.cam_name
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = ForkliftPalletInsertLiftEnvCfg()
    cfg.scene.num_envs = 1
    cfg.use_camera = True
    cfg.use_asymmetric_critic = False
    cfg.camera_width = args_cli.resolution
    cfg.camera_height = args_cli.resolution
    cfg.camera_hfov_deg = args_cli.hfov_deg
    cfg.camera_mount_body = args_cli.mount_body
    cfg.camera_pos_local = (args_cli.cam_x, args_cli.cam_y, args_cli.cam_z)
    cfg.camera_rpy_local_deg = (args_cli.roll_deg, args_cli.pitch_deg, args_cli.yaw_deg)
    cfg.robot_cfg.init_state.pos = (-2.0, 0.0, 0.03)

    print("[DEBUG] Creating env...")
    env = gym.make("Isaac-Forklift-PalletInsertLift-Direct-v0", cfg=cfg)
    print("[DEBUG] Resetting env...")
    obs, _ = env.reset()
    print("[DEBUG] Env reset done.")

    stage = env.unwrapped.sim.stage
    spawn_marker_cube(stage, "/World/Debug/TipMarker", (-0.2, 0.0, 0.1), 0.2, (1.0, 0.0, 0.0), "/World/Debug/Materials/TipMarker")
    spawn_marker_cube(stage, "/World/Debug/PalletMarker", (0.0, 0.0, 0.3), 0.3, (0.0, 0.2, 1.0), "/World/Debug/Materials/PalletMarker")

    frames: list[np.ndarray] = []
    robot_positions: list[tuple[float, float, float]] = []
    mid_step = max(1, args_cli.steps // 2)

    print("[DEBUG] Starting step loop...")
    for step in range(args_cli.steps):
        action = torch.zeros((1, 3), device=env.unwrapped.device, dtype=torch.float32)
        if step <= 60:
            action[:, 0] = 1.0
        else:
            action[:, 2] = 1.0

        obs, _, terminated, truncated, _ = env.step(action)
        
        if "image" not in obs["policy"]:
            print(f"[DEBUG] Step {step}: 'image' not in obs['policy']. Keys: {obs['policy'].keys()}")
            continue
            
        img_np = tensor_to_rgb_image(obs["policy"]["image"][0])
        frames.append(img_np)

        robot_pos = env.unwrapped.robot.data.root_pos_w[0].detach().cpu().tolist()
        robot_positions.append((float(robot_pos[0]), float(robot_pos[1]), float(robot_pos[2])))

        if step == 0:
            Image.fromarray(img_np).save(out_dir / "frame_start.png")
        elif step == mid_step:
            Image.fromarray(img_np).save(out_dir / "frame_mid.png")
        elif step == args_cli.steps - 1:
            Image.fromarray(img_np).save(out_dir / "frame_end.png")

        if bool(terminated[0]) or bool(truncated[0]):
            print(f"[DEBUG] Step {step}: terminated or truncated")
            break

    print(f"[DEBUG] Loop done. Frames captured: {len(frames)}")
    env.close()

    if not frames:
        raise RuntimeError("No frames captured from camera evaluation.")

    height, width = frames[0].shape[:2]
    video_path = out_dir / "video.mp4"
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (width, height))
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()
    print(f"[DEBUG] Video saved to {video_path}")

    diffs = []
    for idx in range(1, min(10, len(frames))):
        diff = np.mean(np.abs(frames[idx].astype(np.float32) - frames[0].astype(np.float32)))
        diffs.append(float(diff))
    avg_diff = float(np.mean(diffs)) if diffs else 0.0

    with (out_dir / "metrics.txt").open("w", encoding="utf-8") as f:
        f.write(f"Background stability (avg diff 0-10): {avg_diff:.2f}\n")
        f.write(f"Frames captured: {len(frames)}\n")
        f.write(f"Camera pos (cm): {args_cli.cam_x}, {args_cli.cam_y}, {args_cli.cam_z}\n")
        f.write(f"Camera rpy (deg): {args_cli.roll_deg}, {args_cli.pitch_deg}, {args_cli.yaw_deg}\n")
        f.write(f"Camera hfov: {args_cli.hfov_deg}\n")
        f.write(f"Camera mount: {args_cli.mount_body}\n")
        f.write(f"First robot root pos: {robot_positions[0]}\n")
        f.write(f"Last robot root pos: {robot_positions[-1]}\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[DEBUG] Exception: {e}")
        import traceback
        traceback.print_exc()
    finally:
        simulation_app.close()
