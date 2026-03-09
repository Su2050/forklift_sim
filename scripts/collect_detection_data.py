#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image


PROJECT_DIR = Path("/home/uniubi/projects/forklift_sim")
PATCH_TASKS_DIR = PROJECT_DIR / "forklift_pallet_insert_lift_project" / "isaaclab_patch" / "source" / "isaaclab_tasks"
DEPLOYMENT_DIR = PROJECT_DIR / "deployment"
if str(PATCH_TASKS_DIR) not in sys.path:
    sys.path.insert(0, str(PATCH_TASKS_DIR))
if str(DEPLOYMENT_DIR) not in sys.path:
    sys.path.insert(0, str(DEPLOYMENT_DIR))

from isaaclab.app import AppLauncher  # noqa: E402


PALLET_WIDTH_M = 0.8 * 1.8
PALLET_DEPTH_M = 1.2 * 1.8
HOLE_CENTER_Y_M = 0.32
HOLE_WIDTH_M = 0.34
HOLE_HEIGHT_M = 0.18
HOLE_FRONT_FACE_OFFSET_M = 0.04
HOLE_CENTER_Z_M = 0.00

STAGE_RATIOS = {
    "far": 0.22,
    "mid": 0.28,
    "near": 0.32,
    "insert": 0.18,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect pallet-hole detection data from IsaacLab")
    AppLauncher.add_app_launcher_args(parser)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--version", type=str, default="s1.0zd")
    parser.add_argument("--target-images", type=int, default=8000)
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--camera-width", type=int, default=64)
    parser.add_argument("--camera-height", type=int, default=64)
    parser.add_argument("--hfov-deg", type=float, default=90.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--control-mode", choices=["heuristic", "random", "teacher"], default="heuristic")
    parser.add_argument("--teacher-model", type=Path, default=None)
    parser.add_argument("--max-steps", type=int, default=40000)
    parser.add_argument("--min-box-size-px", type=float, default=0.5)
    parser.add_argument("--stage1-init-x-min-m", type=float, default=-3.8)
    parser.add_argument("--stage1-init-x-max-m", type=float, default=-3.0)
    parser.add_argument("--stage1-init-y-min-m", type=float, default=-0.08)
    parser.add_argument("--stage1-init-y-max-m", type=float, default=0.08)
    parser.add_argument("--stage1-init-yaw-deg-min", type=float, default=-4.0)
    parser.add_argument("--stage1-init-yaw-deg-max", type=float, default=4.0)
    return parser.parse_args()


def quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    return torch.stack(
        (
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ),
        dim=-1,
    )


def quat_conj(q: torch.Tensor) -> torch.Tensor:
    return torch.stack((q[..., 0], -q[..., 1], -q[..., 2], -q[..., 3]), dim=-1)


def quat_apply(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    zeros = torch.zeros_like(v[..., :1])
    v_quat = torch.cat((zeros, v), dim=-1)
    return quat_mul(quat_mul(q, v_quat), quat_conj(q))[..., 1:]


def quat_from_rpy_deg(roll: float, pitch: float, yaw: float, device: torch.device) -> torch.Tensor:
    rr = math.radians(roll) * 0.5
    pr = math.radians(pitch) * 0.5
    yr = math.radians(yaw) * 0.5
    cr, sr = math.cos(rr), math.sin(rr)
    cp, sp = math.cos(pr), math.sin(pr)
    cy, sy = math.cos(yr), math.sin(yr)
    return torch.tensor(
        [
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ],
        dtype=torch.float32,
        device=device,
    )


def project_points_world_to_image(
    points_world: torch.Tensor,
    cam_pos_world: torch.Tensor,
    cam_quat_world: torch.Tensor,
    width: int,
    height: int,
    hfov_deg: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    rel_world = points_world - cam_pos_world.unsqueeze(1)
    rel_cam = quat_apply(quat_conj(cam_quat_world).unsqueeze(1), rel_world)

    x_forward = rel_cam[..., 0]
    y_left = rel_cam[..., 1]
    z_up = rel_cam[..., 2]

    hfov_rad = math.radians(hfov_deg)
    fx = width / (2.0 * math.tan(hfov_rad / 2.0))
    fy = fx
    cx = width * 0.5
    cy = height * 0.5

    visible = x_forward > 1e-4
    u = cx - fx * (y_left / torch.clamp(x_forward, min=1e-4))
    v = cy - fy * (z_up / torch.clamp(x_forward, min=1e-4))
    pixels = torch.stack((u, v), dim=-1)
    return pixels, visible


def classify_stage(obs_15d: torch.Tensor) -> str:
    d_x = float(obs_15d[0].item())
    insert_norm = float(obs_15d[9].item())
    if insert_norm > 0.05:
        return "insert"
    if d_x > 2.0:
        return "far"
    if d_x > 1.0:
        return "mid"
    return "near"


def stage_targets(total_images: int) -> dict[str, int]:
    return {key: max(1, int(total_images * ratio)) for key, ratio in STAGE_RATIOS.items()}


def heuristic_actions(obs_15d: torch.Tensor) -> torch.Tensor:
    d_y = obs_15d[:, 1]
    dyaw = torch.atan2(obs_15d[:, 3], obs_15d[:, 2])
    insert_norm = obs_15d[:, 9]

    steer = torch.clamp(-2.0 * d_y - 1.5 * dyaw, -1.0, 1.0)
    drive = torch.where(insert_norm < 0.15, torch.full_like(insert_norm, 0.7), torch.full_like(insert_norm, 0.25))
    drive = torch.where(insert_norm > 0.5, torch.full_like(insert_norm, 0.05), drive)
    lift = torch.where(insert_norm > 0.55, torch.full_like(insert_norm, 0.2), torch.zeros_like(insert_norm))

    return torch.stack((drive, steer, lift), dim=-1)


def random_actions(num_envs: int, device: torch.device) -> torch.Tensor:
    return torch.rand((num_envs, 3), device=device) * 2.0 - 1.0


@dataclass
class CocoAccumulator:
    output_dir: Path
    width: int
    height: int
    target_images: int

    def __post_init__(self) -> None:
        self.images_dir = self.output_dir / "images"
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.images: list[dict[str, Any]] = []
        self.annotations: list[dict[str, Any]] = []
        self.image_counter = 1
        self.annotation_counter = 1
        self.stage_counts = {stage: 0 for stage in STAGE_RATIOS}
        self.stage_targets = stage_targets(self.target_images)

    def add_image(
        self,
        rgb_chw: torch.Tensor,
        episode_id: str,
        frame_id: int,
        stage: str,
        boxes_xyxy: list[tuple[float, float, float, float, int]],
    ) -> None:
        image_id = self.image_counter
        self.image_counter += 1

        img_hwc = (rgb_chw.permute(1, 2, 0).cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
        rel_path = Path("images") / episode_id / f"frame_{frame_id:06d}.jpg"
        abs_path = self.output_dir / rel_path
        abs_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(img_hwc).save(abs_path, quality=95)

        self.images.append(
            {
                "id": image_id,
                "file_name": rel_path.as_posix(),
                "width": self.width,
                "height": self.height,
                "episode_id": episode_id,
                "frame_id": frame_id,
                "stage": stage,
            }
        )

        for x1, y1, x2, y2, category_id in boxes_xyxy:
            w = max(0.0, x2 - x1)
            h = max(0.0, y2 - y1)
            self.annotations.append(
                {
                    "id": self.annotation_counter,
                    "image_id": image_id,
                    "category_id": category_id,
                    "bbox": [x1, y1, w, h],
                    "area": w * h,
                    "iscrowd": 0,
                }
            )
            self.annotation_counter += 1

        self.stage_counts[stage] += 1

    def enough_data(self) -> bool:
        if len(self.images) < self.target_images:
            return False
        return all(self.stage_counts[stage] >= self.stage_targets[stage] for stage in self.stage_counts)

    def export(self, seed: int) -> None:
        episode_ids = sorted({img["episode_id"] for img in self.images})
        rng = random.Random(seed)
        rng.shuffle(episode_ids)

        n_total = len(episode_ids)
        n_train = max(1, int(n_total * 0.8))
        n_val = max(1, int(n_total * 0.1))
        train_eps = set(episode_ids[:n_train])
        val_eps = set(episode_ids[n_train:n_train + n_val])
        test_eps = set(episode_ids[n_train + n_val:])

        def split_name(ep_id: str) -> str:
            if ep_id in train_eps:
                return "train"
            if ep_id in val_eps:
                return "val"
            return "test"

        split_to_images = {"train": [], "val": [], "test": []}
        split_to_annotations = {"train": [], "val": [], "test": []}
        image_to_split = {}
        for image in self.images:
            split = split_name(image["episode_id"])
            img_copy = dict(image)
            img_copy["split"] = split
            split_to_images[split].append(img_copy)
            image_to_split[image["id"]] = split

        for ann in self.annotations:
            split_to_annotations[image_to_split[ann["image_id"]]].append(dict(ann))

        categories = [
            {"id": 1, "name": "left_hole"},
            {"id": 2, "name": "right_hole"},
        ]

        for split in ("train", "val", "test"):
            payload = {
                "images": split_to_images[split],
                "annotations": split_to_annotations[split],
                "categories": categories,
            }
            with (self.output_dir / f"annotations_{split}.json").open("w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)

        summary = {
            "target_images": self.target_images,
            "actual_images": len(self.images),
            "stage_counts": self.stage_counts,
            "stage_targets": self.stage_targets,
            "episodes": len(episode_ids),
            "splits": {
                "train": len(train_eps),
                "val": len(val_eps),
                "test": len(test_eps),
            },
        }
        with (self.output_dir / "summary.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)


def build_hole_boxes_xyxy(
    env: ForkliftPalletInsertLiftEnv,
    env_ids: torch.Tensor,
    width: int,
    height: int,
    hfov_deg: float,
    min_box_size_px: float,
) -> tuple[list[list[tuple[float, float, float, float, int]]], dict[str, Any]]:
    device = env.device
    pallet_pos = env.pallet.data.root_pos_w[env_ids]
    pallet_quat = env.pallet.data.root_quat_w[env_ids]

    body_name = str(getattr(env.cfg, "camera_mount_body", "body"))
    try:
        body_ids, _ = env.robot.find_bodies([body_name], preserve_order=True)
        body_pos = env.robot.data.body_pos_w[env_ids][:, body_ids[0]]
        body_quat = env.robot.data.body_quat_w[env_ids][:, body_ids[0]]
    except Exception:
        body_pos = env.robot.data.root_pos_w[env_ids]
        body_quat = env.robot.data.root_quat_w[env_ids]

    local_offset = torch.tensor(env.cfg.camera_pos_local, dtype=torch.float32, device=device) / 100.0
    local_quat = quat_from_rpy_deg(*env.cfg.camera_rpy_local_deg, device=device)
    cam_pos = body_pos + quat_apply(body_quat, local_offset.unsqueeze(0).expand_as(body_pos))
    cam_quat = quat_mul(body_quat, local_quat.unsqueeze(0).expand_as(body_quat))

    front_x = -env.cfg.pallet_depth_m * 0.5 + HOLE_FRONT_FACE_OFFSET_M
    half_w = HOLE_WIDTH_M * 0.5
    half_h = HOLE_HEIGHT_M * 0.5

    hole_defs = [
        (-HOLE_CENTER_Y_M, 1),
        (HOLE_CENTER_Y_M, 2),
    ]

    all_boxes: list[list[tuple[float, float, float, float, int]]] = []
    debug_stats: dict[str, Any] = {
        "accepted": 0,
        "behind_camera": 0,
        "out_of_frame": 0,
        "too_small": 0,
        "per_hole": {},
    }
    for idx in range(len(env_ids)):
        image_boxes: list[tuple[float, float, float, float, int]] = []
        for y_center, category_id in hole_defs:
            local_corners = torch.tensor(
                [
                    [front_x, y_center - half_w, HOLE_CENTER_Z_M - half_h],
                    [front_x, y_center + half_w, HOLE_CENTER_Z_M - half_h],
                    [front_x, y_center + half_w, HOLE_CENTER_Z_M + half_h],
                    [front_x, y_center - half_w, HOLE_CENTER_Z_M + half_h],
                ],
                dtype=torch.float32,
                device=device,
            )
            world_corners = quat_apply(pallet_quat[idx].unsqueeze(0).expand(4, -1), local_corners) + pallet_pos[idx]
            pixels, visible = project_points_world_to_image(
                world_corners.unsqueeze(0),
                cam_pos[idx].unsqueeze(0),
                cam_quat[idx].unsqueeze(0),
                width,
                height,
                hfov_deg,
            )
            pixels = pixels[0]
            visible = visible[0]
            x_min = float(pixels[:, 0].min().item())
            y_min = float(pixels[:, 1].min().item())
            x_max = float(pixels[:, 0].max().item())
            y_max = float(pixels[:, 1].max().item())
            raw_w = x_max - x_min
            raw_h = y_max - y_min
            all_visible = bool(torch.all(visible))
            in_frame = x_max >= 0.0 and x_min <= (width - 1.0) and y_max >= 0.0 and y_min <= (height - 1.0)

            if idx == 0:
                debug_stats["per_hole"][category_id] = {
                    "all_visible": all_visible,
                    "in_frame": in_frame,
                    "raw_xyxy": [round(x_min, 2), round(y_min, 2), round(x_max, 2), round(y_max, 2)],
                    "raw_size": [round(raw_w, 2), round(raw_h, 2)],
                    "visible_corners": int(visible.sum().item()),
                }

            if not all_visible:
                debug_stats["behind_camera"] += 1
                continue
            if not in_frame:
                debug_stats["out_of_frame"] += 1
                continue

            x1 = float(torch.clamp(pixels[:, 0].min(), 0.0, width - 1.0).item())
            y1 = float(torch.clamp(pixels[:, 1].min(), 0.0, height - 1.0).item())
            x2 = float(torch.clamp(pixels[:, 0].max(), 0.0, width - 1.0).item())
            y2 = float(torch.clamp(pixels[:, 1].max(), 0.0, height - 1.0).item())
            if (x2 - x1) < min_box_size_px or (y2 - y1) < min_box_size_px:
                debug_stats["too_small"] += 1
                continue
            image_boxes.append((x1, y1, x2, y2, category_id))
            debug_stats["accepted"] += 1
        all_boxes.append(image_boxes)
    return all_boxes, debug_stats


def maybe_make_teacher(policy_mode: str, teacher_model: Path | None):
    if policy_mode != "teacher":
        return None
    if teacher_model is None:
        raise ValueError("--teacher-model is required when --control-mode=teacher")
    from infer import ForkliftPolicy  # lazy import
    return ForkliftPolicy(str(teacher_model), device="cpu")


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    from isaaclab_tasks.direct.forklift_pallet_insert_lift.env import ForkliftPalletInsertLiftEnv
    from isaaclab_tasks.direct.forklift_pallet_insert_lift.env_cfg import ForkliftPalletInsertLiftEnvCfg

    cfg = ForkliftPalletInsertLiftEnvCfg()
    cfg.seed = args.seed
    cfg.scene.num_envs = args.num_envs
    cfg.use_camera = True
    cfg.use_asymmetric_critic = True
    cfg.camera_width = args.camera_width
    cfg.camera_height = args.camera_height
    cfg.camera_hfov_deg = args.hfov_deg
    cfg.stage1_init_x_min_m = args.stage1_init_x_min_m
    cfg.stage1_init_x_max_m = args.stage1_init_x_max_m
    cfg.stage1_init_y_min_m = args.stage1_init_y_min_m
    cfg.stage1_init_y_max_m = args.stage1_init_y_max_m
    cfg.stage1_init_yaw_deg_min = args.stage1_init_yaw_deg_min
    cfg.stage1_init_yaw_deg_max = args.stage1_init_yaw_deg_max
    cfg.wait_for_textures = False

    env = ForkliftPalletInsertLiftEnv(cfg, render_mode=None)
    teacher = maybe_make_teacher(args.control_mode, args.teacher_model)

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    accumulator = CocoAccumulator(
        output_dir=output_dir,
        width=args.camera_width,
        height=args.camera_height,
        target_images=args.target_images,
    )
    print(
        f"[collect-data] start output_dir={output_dir} target_images={args.target_images} "
        f"num_envs={args.num_envs} control_mode={args.control_mode} "
        f"min_box_size_px={args.min_box_size_px} "
        f"init_x=[{args.stage1_init_x_min_m},{args.stage1_init_x_max_m}] "
        f"init_y=[{args.stage1_init_y_min_m},{args.stage1_init_y_max_m}] "
        f"init_yaw=[{args.stage1_init_yaw_deg_min},{args.stage1_init_yaw_deg_max}]",
        flush=True,
    )

    episode_counters = [0 for _ in range(args.num_envs)]
    frame_counters = [0 for _ in range(args.num_envs)]

    obs, _ = env.reset()

    step = 0
    while step < args.max_steps and not accumulator.enough_data():
        images = obs["image"]
        obs_critic = obs["critic"]
        env_ids = torch.arange(env.num_envs, device=env.device)
        hole_boxes, hole_debug = build_hole_boxes_xyxy(
            env=env,
            env_ids=env_ids,
            width=args.camera_width,
            height=args.camera_height,
            hfov_deg=args.hfov_deg,
            min_box_size_px=args.min_box_size_px,
        )

        for env_idx in range(env.num_envs):
            if len(hole_boxes[env_idx]) != 2:
                continue
            stage = classify_stage(obs_critic[env_idx])
            episode_id = f"env{env_idx:02d}_ep{episode_counters[env_idx]:04d}"
            accumulator.add_image(
                rgb_chw=images[env_idx].detach().cpu(),
                episode_id=episode_id,
                frame_id=frame_counters[env_idx],
                stage=stage,
                boxes_xyxy=hole_boxes[env_idx],
            )
            frame_counters[env_idx] += 1

        if args.control_mode == "teacher":
            actions_np = teacher.infer_batch(obs_critic.detach().cpu().numpy())
            actions = torch.from_numpy(actions_np).to(env.device)
        elif args.control_mode == "random":
            actions = random_actions(env.num_envs, env.device)
        else:
            actions = heuristic_actions(obs_critic)

        obs, _, terminated, truncated, _ = env.step(actions)
        dones = (terminated | truncated).detach().cpu().numpy()
        for env_idx, done in enumerate(dones):
            if done:
                episode_counters[env_idx] += 1
                frame_counters[env_idx] = 0
        step += 1
        if step == 1 or step % 50 == 0:
            print(
                f"[collect-data] step={step} images={len(accumulator.images)} "
                f"stage_counts={accumulator.stage_counts} "
                f"box_debug={hole_debug}",
                flush=True,
            )

    accumulator.export(seed=args.seed)
    print(
        f"[collect-data] done images={len(accumulator.images)} "
        f"stage_counts={accumulator.stage_counts}",
        flush=True,
    )
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
