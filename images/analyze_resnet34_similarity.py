#!/usr/bin/env python3
import argparse
import csv
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image
from torchvision.models import ResNet34_Weights, resnet34


if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DISTANCE_PATTERN = re.compile(r"([+-]?\d+(?:\.\d+)?)m")


@dataclass(frozen=True)
class ImageSample:
    path: Path
    distance_m: float
    label: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze feature similarity between distance-labeled images with ResNet34."
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=Path("insertion_seq_network_input"),
        help="Directory containing distance-labeled images.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis_outputs/resnet34_similarity"),
        help="Directory to store analysis outputs.",
    )
    parser.add_argument(
        "--weights",
        choices=("default", "none"),
        default="default",
        help="Use ImageNet pretrained weights (`default`) or random init (`none`).",
    )
    parser.add_argument(
        "--reference-distances",
        type=float,
        nargs="*",
        default=(-0.5, 0.0),
        help="Reference distances used for one-vs-all similarity plots.",
    )
    return parser.parse_args()


def load_samples(image_dir: Path) -> list[ImageSample]:
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    samples: list[ImageSample] = []
    for path in sorted(image_dir.glob("*")):
        if path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}:
            continue

        match = DISTANCE_PATTERN.search(path.name)
        if not match:
            raise ValueError(f"Could not parse distance from filename: {path.name}")

        distance_m = float(match.group(1))
        samples.append(
            ImageSample(
                path=path,
                distance_m=distance_m,
                label=f"{distance_m:+.1f}m",
            )
        )

    samples.sort(key=lambda sample: sample.distance_m)
    if not samples:
        raise ValueError(f"No supported images found in {image_dir}")
    return samples


def build_model(weight_mode: str) -> tuple[torch.nn.Module, object | None]:
    if weight_mode == "default":
        weights = ResNet34_Weights.DEFAULT
        model = resnet34(weights=weights)
    else:
        weights = None
        model = resnet34(weights=None)

    model.fc = torch.nn.Identity()
    model.eval()
    return model, weights


def normalize_rows(matrix: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.clip(norms, eps, None)


def cosine_similarity_matrix(matrix: np.ndarray) -> np.ndarray:
    matrix_norm = normalize_rows(matrix)
    return matrix_norm @ matrix_norm.T


def extract_features(
    samples: Iterable[ImageSample], model: torch.nn.Module, weights: object | None
) -> tuple[np.ndarray, np.ndarray]:
    if weights is not None:
        preprocess = weights.transforms()
    else:
        preprocess = ResNet34_Weights.DEFAULT.transforms()

    features = []
    pixels = []
    with torch.no_grad():
        for sample in samples:
            image = Image.open(sample.path).convert("RGB")
            pixels.append(np.asarray(image, dtype=np.float32).reshape(-1))
            tensor = preprocess(image).unsqueeze(0)
            feature = model(tensor).squeeze(0).cpu().numpy().astype(np.float32)
            features.append(feature)

    return np.stack(features), np.stack(pixels)


def write_adjacent_csv(
    samples: list[ImageSample],
    feature_sim: np.ndarray,
    pixel_sim: np.ndarray,
    feature_step_l2: np.ndarray,
    output_path: Path,
) -> None:
    fieldnames = [
        "from_label",
        "to_label",
        "from_distance_m",
        "to_distance_m",
        "delta_distance_m",
        "feature_cosine_similarity",
        "feature_l2_distance",
        "pixel_cosine_similarity",
    ]
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for idx in range(len(samples) - 1):
            writer.writerow(
                {
                    "from_label": samples[idx].label,
                    "to_label": samples[idx + 1].label,
                    "from_distance_m": samples[idx].distance_m,
                    "to_distance_m": samples[idx + 1].distance_m,
                    "delta_distance_m": samples[idx + 1].distance_m - samples[idx].distance_m,
                    "feature_cosine_similarity": float(feature_sim[idx, idx + 1]),
                    "feature_l2_distance": float(feature_step_l2[idx]),
                    "pixel_cosine_similarity": float(pixel_sim[idx, idx + 1]),
                }
            )


def render_heatmap(
    matrix: np.ndarray,
    labels: list[str],
    title: str,
    output_path: Path,
    cmap: str = "viridis",
) -> None:
    fig, ax = plt.subplots(figsize=(9, 7))
    image = ax.imshow(matrix, cmap=cmap, vmin=float(matrix.min()), vmax=float(matrix.max()))
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_title(title)
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def render_adjacent_plot(
    samples: list[ImageSample],
    feature_sim: np.ndarray,
    pixel_sim: np.ndarray,
    feature_step_l2: np.ndarray,
    output_path: Path,
) -> None:
    x_values = [sample.distance_m for sample in samples[:-1]]

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(x_values, np.diag(feature_sim, 1), marker="o", label="ResNet34 feature cosine")
    axes[0].plot(x_values, np.diag(pixel_sim, 1), marker="s", label="Pixel cosine")
    axes[0].set_ylabel("Cosine similarity")
    axes[0].set_title("Adjacent-step similarity (distance to next image)")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(x_values, feature_step_l2, marker="o", color="tab:red")
    axes[1].set_xlabel("Distance start (m)")
    axes[1].set_ylabel("Feature L2 distance")
    axes[1].set_title("Adjacent-step ResNet34 feature change magnitude")
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def render_reference_plot(
    samples: list[ImageSample],
    feature_sim: np.ndarray,
    reference_distances: Iterable[float],
    output_path: Path,
) -> dict[str, str]:
    labels = [sample.label for sample in samples]
    distances = [sample.distance_m for sample in samples]

    fig, ax = plt.subplots(figsize=(10, 5))
    reference_map: dict[str, str] = {}
    for ref_distance in reference_distances:
        best_idx = min(
            range(len(samples)),
            key=lambda idx: abs(samples[idx].distance_m - ref_distance),
        )
        reference_label = labels[best_idx]
        reference_map[f"{ref_distance:+.2f}"] = reference_label
        ax.plot(
            distances,
            feature_sim[best_idx],
            marker="o",
            label=f"Ref {reference_label}",
        )

    ax.set_xlabel("Distance (m)")
    ax.set_ylabel("Cosine similarity")
    ax.set_title("One-vs-all ResNet34 feature similarity")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return reference_map


def summarize(
    samples: list[ImageSample],
    feature_sim: np.ndarray,
    pixel_sim: np.ndarray,
    feature_step_l2: np.ndarray,
    reference_map: dict[str, str],
) -> dict:
    adjacent_feature_cos = np.diag(feature_sim, 1)
    adjacent_pixel_cos = np.diag(pixel_sim, 1)

    min_idx = int(np.argmin(adjacent_feature_cos))
    max_idx = int(np.argmax(adjacent_feature_cos))

    return {
        "num_images": len(samples),
        "distance_range_m": [samples[0].distance_m, samples[-1].distance_m],
        "reference_map": reference_map,
        "adjacent_feature_cosine": {
            "mean": float(adjacent_feature_cos.mean()),
            "min": float(adjacent_feature_cos[min_idx]),
            "min_pair": [samples[min_idx].label, samples[min_idx + 1].label],
            "max": float(adjacent_feature_cos[max_idx]),
            "max_pair": [samples[max_idx].label, samples[max_idx + 1].label],
        },
        "adjacent_pixel_cosine": {
            "mean": float(adjacent_pixel_cos.mean()),
            "min": float(adjacent_pixel_cos.min()),
            "max": float(adjacent_pixel_cos.max()),
        },
        "adjacent_feature_l2": {
            "mean": float(feature_step_l2.mean()),
            "min": float(feature_step_l2.min()),
            "max": float(feature_step_l2.max()),
        },
    }


def main() -> None:
    args = parse_args()
    samples = load_samples(args.image_dir)
    model, weights = build_model(args.weights)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    features, pixels = extract_features(samples, model, weights)

    feature_sim = cosine_similarity_matrix(features)
    pixel_sim = cosine_similarity_matrix(pixels)
    feature_step_l2 = np.linalg.norm(np.diff(features, axis=0), axis=1)

    labels = [sample.label for sample in samples]

    render_heatmap(
        feature_sim,
        labels,
        "ResNet34 Feature Cosine Similarity",
        args.output_dir / "feature_similarity_heatmap.png",
    )
    render_heatmap(
        pixel_sim,
        labels,
        "Pixel Cosine Similarity",
        args.output_dir / "pixel_similarity_heatmap.png",
        cmap="magma",
    )
    render_adjacent_plot(
        samples,
        feature_sim,
        pixel_sim,
        feature_step_l2,
        args.output_dir / "adjacent_similarity.png",
    )
    reference_map = render_reference_plot(
        samples,
        feature_sim,
        args.reference_distances,
        args.output_dir / "reference_similarity.png",
    )

    np.save(args.output_dir / "feature_vectors.npy", features)
    np.save(args.output_dir / "feature_similarity_matrix.npy", feature_sim)
    np.save(args.output_dir / "pixel_similarity_matrix.npy", pixel_sim)

    with (args.output_dir / "ordered_samples.json").open("w") as handle:
        json.dump(
            [
                {
                    "path": str(sample.path),
                    "distance_m": sample.distance_m,
                    "label": sample.label,
                }
                for sample in samples
            ],
            handle,
            indent=2,
        )

    write_adjacent_csv(
        samples,
        feature_sim,
        pixel_sim,
        feature_step_l2,
        args.output_dir / "adjacent_similarity.csv",
    )

    summary = summarize(samples, feature_sim, pixel_sim, feature_step_l2, reference_map)
    with (args.output_dir / "summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"Saved outputs to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
