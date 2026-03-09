#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


METRICS = [
    "episode/success_rate_ema",
    "episode/success_rate_total",
    "phase/frac_aligned",
    "phase/hold_counter_mean",
    "err/yaw_deg_near_success",
    "err/lateral_near_success",
    "Mean reward",
    "Mean episode length",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate compare report for scratch vs finetune RL runs")
    parser.add_argument("--scratch-log", type=Path, required=True)
    parser.add_argument("--finetune-log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--version", type=str, required=True)
    return parser.parse_args()


def parse_metrics(log_path: Path) -> dict[str, float | str]:
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    metrics: dict[str, float | str] = {}
    for name in METRICS:
        pattern = re.compile(rf"{re.escape(name)}:\s+(-?\d+(?:\.\d+)?)")
        matches = pattern.findall(text)
        if matches:
            metrics[name] = float(matches[-1])
    iter_matches = re.findall(r"Learning iteration\s+(\d+)/(\d+)", text)
    if iter_matches:
        last_iter = iter_matches[-1]
        metrics["last_iteration"] = f"{last_iter[0]}/{last_iter[1]}"
    return metrics


def main() -> None:
    args = parse_args()
    scratch = parse_metrics(args.scratch_log)
    finetune = parse_metrics(args.finetune_log)

    lines = [
        "# Vision CNN Scratch vs Pretrained Compare",
        "",
        f"- version: {args.version}",
        f"- scratch_log: `{args.scratch_log}`",
        f"- finetune_log: `{args.finetune_log}`",
        "",
        "## Metrics",
        "",
        "| metric | scratch | finetune |",
        "| --- | ---: | ---: |",
    ]

    for metric in ["last_iteration", *METRICS]:
        scratch_val = scratch.get(metric, "N/A")
        finetune_val = finetune.get(metric, "N/A")
        lines.append(f"| {metric} | {scratch_val} | {finetune_val} |")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (args.output.with_suffix(".json")).write_text(
        json.dumps({"scratch": scratch, "finetune": finetune}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
