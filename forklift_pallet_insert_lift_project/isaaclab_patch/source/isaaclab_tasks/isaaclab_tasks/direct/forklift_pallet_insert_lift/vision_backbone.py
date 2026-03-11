from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torchvision import models as tv_models


class MobileNetVisionBackbone(nn.Module):
    """Shared MobileNetV3-Small visual backbone for RL and pretraining."""

    def __init__(self, imagenet_init: bool = False):
        super().__init__()
        weights = tv_models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if imagenet_init else None
        mobilenet = tv_models.mobilenet_v3_small(weights=weights)
        self.features = mobilenet.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        feat = self.features(image)
        feat = self.pool(feat)
        return feat.flatten(1)


class ResNet18VisionBackbone(nn.Module):
    """ResNet-18 visual backbone using standard ImageNet features (RRL paradigm)."""

    def __init__(self, imagenet_init: bool = True):
        super().__init__()
        weights = tv_models.ResNet18_Weights.IMAGENET1K_V1 if imagenet_init else None
        resnet = tv_models.resnet18(weights=weights)
        # Remove the final fully connected layer
        self.features = nn.Sequential(*list(resnet.children())[:-1])

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        # ResNet features output shape: (B, 512, 1, 1)
        feat = self.features(image)
        return feat.flatten(1)


def freeze_module(module: nn.Module, frozen: bool = True) -> None:
    for param in module.parameters():
        param.requires_grad = not frozen


def _extract_state_dict(payload: Any) -> dict[str, torch.Tensor]:
    if isinstance(payload, dict):
        if "backbone_state_dict" in payload:
            return payload["backbone_state_dict"]
        if "model_state_dict" in payload and isinstance(payload["model_state_dict"], dict):
            return payload["model_state_dict"]
        if all(isinstance(v, torch.Tensor) for v in payload.values()):
            return payload
    raise ValueError("Unsupported checkpoint format for backbone loading")


def load_backbone_checkpoint(backbone: nn.Module, checkpoint_path: str | Path) -> tuple[list[str], list[str]]:
    """Load a backbone checkpoint into ``backbone``.

    Supported formats:
    - plain backbone state_dict
    - checkpoint dict with ``backbone_state_dict``
    - full model ``model_state_dict`` whose keys contain one of:
      ``backbone.``, ``image_encoder.``, ``image_backbone.``
    """

    checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    raw_state = _extract_state_dict(checkpoint)

    prefixes = (
        "backbone.",
        "image_backbone.",
        "image_encoder.",
        "image_encoder.0.",
    )

    state_dict: dict[str, torch.Tensor] = {}
    for key, value in raw_state.items():
        matched = False
        for prefix in prefixes:
            if key.startswith(prefix):
                state_dict[key[len(prefix):]] = value
                matched = True
                break
        if not matched and key in backbone.state_dict():
            state_dict[key] = value

    if not state_dict:
        raise ValueError(f"No backbone weights found in checkpoint: {checkpoint_path}")

    incompatible = backbone.load_state_dict(state_dict, strict=False)
    return list(incompatible.missing_keys), list(incompatible.unexpected_keys)


def save_backbone_checkpoint(
    backbone: nn.Module,
    output_path: str | Path,
    metadata: dict[str, Any] | None = None,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "backbone_state_dict": backbone.state_dict(),
        "metadata": metadata or {},
    }
    torch.save(payload, output_path)
