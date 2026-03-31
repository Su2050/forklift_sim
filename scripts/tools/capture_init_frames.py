#!/usr/bin/env python3
"""Capture camera frame 0 for a grid of forklift initializations."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path

from isaaclab.app import AppLauncher


def _parse_csv_floats(raw: str | None, fallback: list[float]) -> list[float]:
    if raw is None or raw.strip() == "":
        return fallback
    values: list[float] = []
    for chunk in raw.split(","):
        text = chunk.strip()
        if not text:
            continue
        values.append(float(text))
    return values or fallback


parser = argparse.ArgumentParser(description="Capture frame 0 camera images for multiple initialization cases.")
parser.add_argument("--output-tag", type=str, default="init_frame_scan", help="Suffix used in the timestamped output folder name.")
parser.add_argument("--output-root", type=str, default="/home/uniubi/projects/forklift_sim/output", help="Root directory for timestamped output folders.")
parser.add_argument("--resolution", type=int, default=256, help="Camera width and height.")
parser.add_argument("--renders", type=int, default=8, help="Number of extra render passes after teleporting poses.")
parser.add_argument("--x-values", type=str, default="", help="Comma-separated root x values in meters.")
parser.add_argument("--y-values", type=str, default="", help="Comma-separated root y values in meters.")
parser.add_argument("--yaw-values", type=str, default="", help="Comma-separated root yaw values in degrees.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import numpy as np
import torch
from PIL import Image, ImageDraw

from isaaclab_tasks.direct.forklift_pallet_insert_lift.env import ForkliftPalletInsertLiftEnv
from isaaclab_tasks.direct.forklift_pallet_insert_lift.env_cfg import ForkliftPalletInsertLiftEnvCfg


REPO_ROOT = Path("/home/uniubi/projects/forklift_sim")


def _yaw_to_quat_wxyz(yaw_rad: torch.Tensor) -> torch.Tensor:
    half = yaw_rad * 0.5
    return torch.stack(
        [
            torch.cos(half),
            torch.zeros_like(half),
            torch.zeros_like(half),
            torch.sin(half),
        ],
        dim=-1,
    )


def _tensor_to_uint8_rgb(img_tensor: torch.Tensor) -> np.ndarray:
    img = torch.clamp(img_tensor.detach().cpu(), 0.0, 1.0)
    img = (img.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    return img


def _slugify_value(value: float) -> str:
    return f"{value:+.2f}".replace("+", "p").replace("-", "m").replace(".", "p")


def _build_cases(cfg: ForkliftPalletInsertLiftEnvCfg) -> list[dict[str, float | str]]:
    x_default = [cfg.stage1_init_x_min_m, 0.5 * (cfg.stage1_init_x_min_m + cfg.stage1_init_x_max_m), cfg.stage1_init_x_max_m]
    y_default = [cfg.stage1_init_y_min_m, 0.0, cfg.stage1_init_y_max_m]
    yaw_default = [cfg.stage1_init_yaw_deg_min, 0.0, cfg.stage1_init_yaw_deg_max]

    x_values = _parse_csv_floats(args_cli.x_values, x_default)
    y_values = _parse_csv_floats(args_cli.y_values, y_default)
    yaw_values = _parse_csv_floats(args_cli.yaw_values, yaw_default)

    cases: list[dict[str, float | str]] = []
    case_id = 0
    for x in x_values:
        for y in y_values:
            for yaw_deg in yaw_values:
                label = f"case_{case_id:02d}_x{_slugify_value(x)}_y{_slugify_value(y)}_yaw{_slugify_value(yaw_deg)}"
                cases.append(
                    {
                        "id": case_id,
                        "label": label,
                        "x_m": float(x),
                        "y_m": float(y),
                        "yaw_deg": float(yaw_deg),
                    }
                )
                case_id += 1
    return cases


def _make_contact_sheet(images: list[np.ndarray], captions: list[str], tile_size: int, out_path: Path) -> None:
    cols = 3
    rows = math.ceil(len(images) / cols)
    caption_h = 42
    sheet = Image.new("RGB", (cols * tile_size, rows * (tile_size + caption_h)), color=(18, 18, 18))
    draw = ImageDraw.Draw(sheet)

    for idx, (img_np, caption) in enumerate(zip(images, captions, strict=True)):
        row = idx // cols
        col = idx % cols
        x0 = col * tile_size
        y0 = row * (tile_size + caption_h)
        tile = Image.fromarray(img_np).resize((tile_size, tile_size))
        sheet.paste(tile, (x0, y0))
        draw.rectangle((x0, y0 + tile_size, x0 + tile_size, y0 + tile_size + caption_h), fill=(28, 28, 28))
        draw.text((x0 + 8, y0 + tile_size + 6), caption, fill=(235, 235, 235))

    sheet.save(out_path)


def main() -> None:
    cfg = ForkliftPalletInsertLiftEnvCfg()
    cfg.use_camera = True
    cfg.use_asymmetric_critic = False
    cfg.camera_width = int(args_cli.resolution)
    cfg.camera_height = int(args_cli.resolution)
    cfg.wait_for_textures = False

    cases = _build_cases(cfg)
    cfg.scene.num_envs = len(cases)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args_cli.output_root) / f"{timestamp}_{args_cli.output_tag}"
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    env = ForkliftPalletInsertLiftEnv(cfg)
    env.reset()

    env_ids = torch.arange(len(cases), device=env.device, dtype=torch.long)

    pallet_pos = torch.tensor(cfg.pallet_cfg.init_state.pos, device=env.device, dtype=torch.float32).repeat(len(cases), 1)
    pallet_quat = torch.tensor(cfg.pallet_cfg.init_state.rot, device=env.device, dtype=torch.float32).repeat(len(cases), 1)
    zeros3 = torch.zeros((len(cases), 3), device=env.device, dtype=torch.float32)

    robot_pos = torch.zeros((len(cases), 3), device=env.device, dtype=torch.float32)
    yaw_rad = torch.zeros((len(cases),), device=env.device, dtype=torch.float32)
    robot_pos[:, 2] = float(cfg.robot_cfg.init_state.pos[2])

    manifest_cases: list[dict[str, object]] = []
    captions: list[str] = []

    for idx, case in enumerate(cases):
        robot_pos[idx, 0] = float(case["x_m"])
        robot_pos[idx, 1] = float(case["y_m"])
        yaw_rad[idx] = math.radians(float(case["yaw_deg"]))
        captions.append(
            f"{case['label']}\nx={case['x_m']:.2f}, y={case['y_m']:.2f}, yaw={case['yaw_deg']:.1f}"
        )

    robot_quat = _yaw_to_quat_wxyz(yaw_rad)

    env._write_root_pose(env.pallet, pallet_pos, pallet_quat, env_ids)
    env._write_root_vel(env.pallet, zeros3, zeros3, env_ids)
    env._write_root_pose(env.robot, robot_pos, robot_quat, env_ids)
    env._write_root_vel(env.robot, zeros3, zeros3, env_ids)

    joint_pos = env.robot.data.default_joint_pos[env_ids].clone()
    joint_vel = torch.zeros_like(joint_pos)
    env._write_joint_state(env.robot, joint_pos, joint_vel, env_ids)
    env._lift_pos_target[env_ids] = 0.0

    env.scene.write_data_to_sim()
    env.sim.forward()
    for _ in range(max(1, int(args_cli.renders))):
        env.sim.render()

    rgb = env._get_camera_image()

    saved_images: list[np.ndarray] = []
    for idx, case in enumerate(cases):
        img_np = _tensor_to_uint8_rgb(rgb[idx])
        saved_images.append(img_np)
        image_name = f"{case['label']}.png"
        image_path = frames_dir / image_name
        Image.fromarray(img_np).save(image_path)
        manifest_cases.append(
            {
                "case_id": int(case["id"]),
                "label": str(case["label"]),
                "x_m": float(case["x_m"]),
                "y_m": float(case["y_m"]),
                "yaw_deg": float(case["yaw_deg"]),
                "frame_index": 0,
                "image_path": str(image_path),
            }
        )

    sheet_path = output_dir / "contact_sheet.png"
    _make_contact_sheet(saved_images, captions, tile_size=int(args_cli.resolution), out_path=sheet_path)

    manifest = {
        "timestamp": timestamp,
        "output_dir": str(output_dir),
        "frame_index": 0,
        "camera": {
            "mount_body": str(cfg.camera_mount_body),
            "resolution": int(args_cli.resolution),
            "hfov_deg": float(cfg.camera_hfov_deg),
            "pos_local_cm": list(cfg.camera_pos_local),
            "rpy_local_deg": list(cfg.camera_rpy_local_deg),
        },
        "stage1_defaults": {
            "x_range_m": [float(cfg.stage1_init_x_min_m), float(cfg.stage1_init_x_max_m)],
            "y_range_m": [float(cfg.stage1_init_y_min_m), float(cfg.stage1_init_y_max_m)],
            "yaw_range_deg": [float(cfg.stage1_init_yaw_deg_min), float(cfg.stage1_init_yaw_deg_max)],
        },
        "cases": manifest_cases,
        "contact_sheet": str(sheet_path),
    }

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    summary_lines = [
        f"output_dir={output_dir}",
        f"num_cases={len(cases)}",
        f"frame_index=0",
        f"resolution={args_cli.resolution}",
        f"contact_sheet={sheet_path}",
        f"manifest={manifest_path}",
    ]
    (output_dir / "summary.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print(f"[INFO] Saved {len(cases)} frame-0 images to {frames_dir}")
    print(f"[INFO] Contact sheet: {sheet_path}")
    print(f"[INFO] Manifest: {manifest_path}")

    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
