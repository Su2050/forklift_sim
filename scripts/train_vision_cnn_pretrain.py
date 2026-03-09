#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


PROJECT_DIR = Path("/home/uniubi/projects/forklift_sim")
PATCH_TASKS_DIR = PROJECT_DIR / "forklift_pallet_insert_lift_project" / "isaaclab_patch" / "source" / "isaaclab_tasks"
if str(PATCH_TASKS_DIR) not in sys.path:
    sys.path.insert(0, str(PATCH_TASKS_DIR))

from isaaclab_tasks.direct.forklift_pallet_insert_lift.vision_backbone import (  # noqa: E402
    MobileNetVisionBackbone,
    save_backbone_checkpoint,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MobileNet backbone on pallet-hole detection data")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--version", type=str, default="s1.0zd")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--imagenet-init", action="store_true")
    return parser.parse_args()


@dataclass
class Sample:
    image_path: Path
    target: torch.Tensor


class CocoBoxDataset(Dataset):
    def __init__(self, data_dir: Path, split: str, image_size: int) -> None:
        annotation_path = data_dir / f"annotations_{split}.json"
        with annotation_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)

        images = {entry["id"]: entry for entry in payload["images"]}
        anns_by_image: dict[int, list[dict[str, Any]]] = {}
        for ann in payload["annotations"]:
            anns_by_image.setdefault(ann["image_id"], []).append(ann)

        self.samples: list[Sample] = []
        for image_id, image in images.items():
            anns = anns_by_image.get(image_id, [])
            if len(anns) != 2:
                continue
            anns = sorted(anns, key=lambda ann: ann["category_id"])
            target = []
            for ann in anns:
                x, y, w, h = ann["bbox"]
                cx = (x + w * 0.5) / image["width"]
                cy = (y + h * 0.5) / image["height"]
                target.extend([cx, cy, w / image["width"], h / image["height"]])
            self.samples.append(
                Sample(
                    image_path=data_dir / image["file_name"],
                    target=torch.tensor(target, dtype=torch.float32),
                )
            )

        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1, hue=0.03),
                transforms.ToTensor(),
            ]
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[idx]
        image = Image.open(sample.image_path).convert("RGB")
        return self.transform(image), sample.target


class HoleBoxRegressor(nn.Module):
    def __init__(self, imagenet_init: bool = False) -> None:
        super().__init__()
        self.backbone = MobileNetVisionBackbone(imagenet_init=imagenet_init)
        self.head = nn.Sequential(
            nn.Linear(576, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 8),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(image)
        pred = self.head(feat)
        return torch.sigmoid(pred)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, criterion: nn.Module) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_l1 = 0.0
    total_samples = 0
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device)
            preds = model(images)
            loss = criterion(preds, targets)
            total_loss += loss.item() * images.size(0)
            total_l1 += torch.abs(preds - targets).mean().item() * images.size(0)
            total_samples += images.size(0)
    return {
        "loss": total_loss / max(total_samples, 1),
        "l1": total_l1 / max(total_samples, 1),
    }


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = CocoBoxDataset(args.data_dir, "train", args.image_size)
    val_dataset = CocoBoxDataset(args.data_dir, "val", args.image_size)
    if not len(train_dataset):
        raise RuntimeError("Training dataset is empty; check collected annotations")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model = HoleBoxRegressor(imagenet_init=args.imagenet_init).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))
    criterion = nn.SmoothL1Loss(beta=0.02)

    best_val_loss = float("inf")
    history: list[dict[str, float]] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        seen = 0
        for images, targets in train_loader:
            images = images.to(device)
            targets = targets.to(device)

            preds = model(images)
            loss = criterion(preds, targets)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            seen += images.size(0)

        scheduler.step()
        train_loss = running_loss / max(seen, 1)
        val_metrics = evaluate(model, val_loader, device, criterion)
        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_l1": val_metrics["l1"],
        }
        history.append(record)
        print(json.dumps(record, ensure_ascii=False))

        last_ckpt = output_dir / "model_last.pt"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "history": history,
                "args": vars(args),
            },
            last_ckpt,
        )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_ckpt = output_dir / "model_best.pt"
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "history": history,
                    "args": vars(args),
                },
                best_ckpt,
            )
            save_backbone_checkpoint(
                model.backbone,
                output_dir / "backbone_best.pt",
                metadata={
                    "version": args.version,
                    "val_loss": best_val_loss,
                    "image_size": args.image_size,
                },
            )

    with (output_dir / "history.json").open("w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
