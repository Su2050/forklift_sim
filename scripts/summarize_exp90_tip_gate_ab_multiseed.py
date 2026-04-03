#!/usr/bin/env python3
from __future__ import annotations

import argparse
import statistics
from pathlib import Path

from summarize_exp90_tip_gate_ab import METRICS, last_n_mean, parse_log


def mean_ignore_none(values: list[float | None]) -> float | None:
    clean = [v for v in values if v is not None]
    if not clean:
        return None
    return statistics.mean(clean)


def fmt(value: float | None, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def seed_label_from_path(path: Path) -> str:
    name = path.name
    for token in name.split("_"):
        if token.startswith("seed"):
            return token
    return name


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize multi-seed Exp9.0 tip-gate A/B runs")
    parser.add_argument("--strict-logs", nargs="+", required=True, type=Path)
    parser.add_argument("--relaxed-logs", nargs="+", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--last-n", type=int, default=20)
    args = parser.parse_args()

    if len(args.strict_logs) != len(args.relaxed_logs):
        raise SystemExit("strict log count must equal relaxed log count")

    strict_runs = [parse_log(path, "strict") for path in args.strict_logs]
    relaxed_runs = [parse_log(path, "relaxed0175") for path in args.relaxed_logs]

    lines: list[str] = []
    lines.append("# Exp9.0 Tip-Gate A/B Multiseed Result")
    lines.append("")
    lines.append(f"Seeds: `{', '.join(seed_label_from_path(path) for path in args.strict_logs)}`")
    lines.append("")
    lines.append("## 1. Per-Seed Quick View")
    lines.append("")
    lines.append("| Seed | strict hold | relaxed hold | strict success | relaxed success | strict band0175 | relaxed band0175 |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for strict_run, relaxed_run in zip(strict_runs, relaxed_runs):
        seed_label = seed_label_from_path(strict_run.path)
        lines.append(
            f"| {seed_label} | "
            f"{fmt(last_n_mean(strict_run.data['phase/frac_hold_entry'], args.last_n))} | "
            f"{fmt(last_n_mean(relaxed_run.data['phase/frac_hold_entry'], args.last_n))} | "
            f"{fmt(last_n_mean(strict_run.data['phase/frac_success'], args.last_n))} | "
            f"{fmt(last_n_mean(relaxed_run.data['phase/frac_success'], args.last_n))} | "
            f"{fmt(last_n_mean(strict_run.data['phase/frac_prehold_reachable_band_companion'], args.last_n))} | "
            f"{fmt(last_n_mean(relaxed_run.data['phase/frac_prehold_reachable_band_companion'], args.last_n))} |"
        )

    lines.append("")
    lines.append("## 2. Mean Last-N Across Seeds")
    lines.append("")
    lines.append(f"窗口：最后 `{args.last_n}` 个 iteration")
    lines.append("")
    lines.append("| Metric | strict mean | relaxed 0.175 mean | delta (relaxed-strict) |")
    lines.append("| --- | ---: | ---: | ---: |")
    for metric in METRICS:
        strict_mean = mean_ignore_none([last_n_mean(run.data[metric], args.last_n) for run in strict_runs])
        relaxed_mean = mean_ignore_none([last_n_mean(run.data[metric], args.last_n) for run in relaxed_runs])
        delta = None if strict_mean is None or relaxed_mean is None else relaxed_mean - strict_mean
        lines.append(f"| `{metric}` | {fmt(strict_mean)} | {fmt(relaxed_mean)} | {fmt(delta)} |")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n")
    print(f"Summary written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
