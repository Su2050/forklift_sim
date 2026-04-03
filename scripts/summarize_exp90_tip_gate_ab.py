#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
import statistics
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


ANSI_RE = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")
ITER_RE = re.compile(r"Learning iteration\s+(\d+)/(\d+)")

METRICS = [
    "phase/frac_inserted",
    "phase/frac_prehold_reachable_band",
    "phase/frac_prehold_reachable_band_companion",
    "diag/prehold_reachable_band_frac_of_inserted",
    "diag/prehold_reachable_band_companion_frac_of_inserted",
    "phase/frac_hold_entry",
    "phase/frac_success",
    "phase/frac_success_strict",
    "err/center_lateral_inserted_mean",
    "err/tip_lateral_inserted_mean",
    "err/yaw_deg_inserted_mean",
    "diag/max_hold_counter",
]


@dataclass
class ParsedLog:
    path: Path
    run_label: str
    last_iter: int
    max_iter: int
    data: dict[str, list[tuple[int, float]]]


def strip_ansi(text: str) -> str:
    return ANSI_RE.sub("", text)


def parse_log(path: Path, run_label: str) -> ParsedLog:
    text = strip_ansi(path.read_text())
    matches = list(ITER_RE.finditer(text))
    data: dict[str, list[tuple[int, float]]] = {metric: [] for metric in METRICS}
    last_iter = -1
    max_iter = -1
    for idx, match in enumerate(matches):
        it = int(match.group(1))
        max_iter = int(match.group(2))
        block_start = match.end()
        block_end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        block = text[block_start:block_end]
        last_iter = max(last_iter, it)
        for metric in METRICS:
            metric_match = re.search(re.escape(metric) + r":\s*([-\d.eE+]+)", block)
            if metric_match:
                data[metric].append((it, float(metric_match.group(1))))
    if last_iter < 0:
        raise RuntimeError(f"No iteration blocks parsed from {path}")
    return ParsedLog(path=path, run_label=run_label, last_iter=last_iter, max_iter=max_iter, data=data)


def last_n_mean(series: list[tuple[int, float]], n: int) -> float | None:
    if not series:
        return None
    return statistics.mean(v for _, v in series[-n:])


def peak(series: list[tuple[int, float]]) -> tuple[float | None, int | None]:
    if not series:
        return None, None
    it, value = max(series, key=lambda item: item[1])
    return value, it


def first_positive(series: list[tuple[int, float]], eps: float = 1e-12) -> int | None:
    for it, value in series:
        if value > eps:
            return it
    return None


def positive_count(series: list[tuple[int, float]], eps: float = 1e-12) -> int:
    return sum(1 for _, value in series if value > eps)


def iter_value_map(series: list[tuple[int, float]]) -> dict[int, float]:
    return {it: value for it, value in series}


def compare_gap(parsed: ParsedLog, a: str, b: str, last_n: int) -> float | None:
    map_a = iter_value_map(parsed.data[a])
    map_b = iter_value_map(parsed.data[b])
    common_its = sorted(set(map_a) & set(map_b))
    if not common_its:
        return None
    tail = common_its[-last_n:]
    if not tail:
        return None
    return statistics.mean(map_a[it] - map_b[it] for it in tail)


def count_gap_positive(parsed: ParsedLog, a: str, b: str, eps: float = 1e-12) -> int:
    map_a = iter_value_map(parsed.data[a])
    map_b = iter_value_map(parsed.data[b])
    common_its = sorted(set(map_a) & set(map_b))
    return sum(1 for it in common_its if (map_a[it] - map_b[it]) > eps)


def fmt(value: float | None, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def fmt_it(value: int | None) -> str:
    return "n/a" if value is None else str(value)


def build_report(strict_log: ParsedLog, relaxed_log: ParsedLog, last_n: int) -> str:
    strict_gap = compare_gap(
        strict_log,
        "phase/frac_prehold_reachable_band",
        "phase/frac_hold_entry",
        last_n,
    )
    relaxed_gap = compare_gap(
        relaxed_log,
        "phase/frac_prehold_reachable_band",
        "phase/frac_hold_entry",
        last_n,
    )

    lines: list[str] = []
    lines.append("# Exp9.0 Tip-Gate Short A/B Result")
    lines.append("")
    lines.append(f"日期：`{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}`")
    lines.append("")
    lines.append("## 1. 运行设置")
    lines.append("")
    lines.append("| Run | Log | Final iter |")
    lines.append("| --- | --- | ---: |")
    lines.append(f"| strict (`tip_entry=0.12`) | `{strict_log.path.name}` | `{strict_log.last_iter}/{strict_log.max_iter}` |")
    lines.append(f"| relaxed (`tip_entry=0.175`) | `{relaxed_log.path.name}` | `{relaxed_log.last_iter}/{relaxed_log.max_iter}` |")
    lines.append("")
    lines.append("## 2. Last-N Mean 对比")
    lines.append("")
    lines.append(f"窗口：最后 `{last_n}` 个 iteration")
    lines.append("")
    lines.append("| Metric | strict | relaxed 0.175 | delta (relaxed-strict) |")
    lines.append("| --- | ---: | ---: | ---: |")
    for metric in METRICS:
        strict_mean = last_n_mean(strict_log.data[metric], last_n)
        relaxed_mean = last_n_mean(relaxed_log.data[metric], last_n)
        delta = None
        if strict_mean is not None and relaxed_mean is not None:
            delta = relaxed_mean - strict_mean
        lines.append(
            f"| `{metric}` | {fmt(strict_mean)} | {fmt(relaxed_mean)} | {fmt(delta)} |"
        )
    lines.append("")
    lines.append("## 3. 首次命中与峰值")
    lines.append("")
    lines.append("| Metric | strict first>0 | relaxed first>0 | strict peak@iter | relaxed peak@iter |")
    lines.append("| --- | ---: | ---: | --- | --- |")
    for metric in [
        "phase/frac_prehold_reachable_band",
        "phase/frac_prehold_reachable_band_companion",
        "phase/frac_hold_entry",
        "phase/frac_success",
        "phase/frac_success_strict",
    ]:
        strict_peak_val, strict_peak_it = peak(strict_log.data[metric])
        relaxed_peak_val, relaxed_peak_it = peak(relaxed_log.data[metric])
        lines.append(
            f"| `{metric}` | {fmt_it(first_positive(strict_log.data[metric]))} | "
            f"{fmt_it(first_positive(relaxed_log.data[metric]))} | "
            f"{fmt(strict_peak_val)} @ {fmt_it(strict_peak_it)} | "
            f"{fmt(relaxed_peak_val)} @ {fmt_it(relaxed_peak_it)} |"
        )
    lines.append("")
    lines.append("## 4. 事件计数")
    lines.append("")
    lines.append("| Event | strict | relaxed 0.175 |")
    lines.append("| --- | ---: | ---: |")
    lines.append(
        f"| `phase/frac_prehold_reachable_band > 0` iterations | "
        f"{positive_count(strict_log.data['phase/frac_prehold_reachable_band'])} | "
        f"{positive_count(relaxed_log.data['phase/frac_prehold_reachable_band'])} |"
    )
    lines.append(
        f"| `phase/frac_prehold_reachable_band_companion > 0` iterations | "
        f"{positive_count(strict_log.data['phase/frac_prehold_reachable_band_companion'])} | "
        f"{positive_count(relaxed_log.data['phase/frac_prehold_reachable_band_companion'])} |"
    )
    lines.append(
        f"| `phase/frac_hold_entry > 0` iterations | "
        f"{positive_count(strict_log.data['phase/frac_hold_entry'])} | "
        f"{positive_count(relaxed_log.data['phase/frac_hold_entry'])} |"
    )
    lines.append(
        f"| `phase/frac_success > 0` iterations | "
        f"{positive_count(strict_log.data['phase/frac_success'])} | "
        f"{positive_count(relaxed_log.data['phase/frac_success'])} |"
    )
    lines.append(
        f"| `phase/frac_prehold_reachable_band > phase/frac_hold_entry` iterations | "
        f"{count_gap_positive(strict_log, 'phase/frac_prehold_reachable_band', 'phase/frac_hold_entry')} | "
        f"{count_gap_positive(relaxed_log, 'phase/frac_prehold_reachable_band', 'phase/frac_hold_entry')} |"
    )
    lines.append(
        f"| `phase/frac_prehold_reachable_band_companion > phase/frac_hold_entry` iterations | "
        f"{count_gap_positive(strict_log, 'phase/frac_prehold_reachable_band_companion', 'phase/frac_hold_entry')} | "
        f"{count_gap_positive(relaxed_log, 'phase/frac_prehold_reachable_band_companion', 'phase/frac_hold_entry')} |"
    )
    lines.append("")
    lines.append("## 5. 关键判读")
    lines.append("")
    lines.append(
        f"- strict 组 `phase/frac_prehold_reachable_band - phase/frac_hold_entry` 的 last-{last_n} 平均差为 `{fmt(strict_gap)}`。"
    )
    lines.append(
        f"- relaxed 组同一差值的 last-{last_n} 平均差为 `{fmt(relaxed_gap)}`。"
    )

    strict_hold = last_n_mean(strict_log.data["phase/frac_hold_entry"], last_n)
    relaxed_hold = last_n_mean(relaxed_log.data["phase/frac_hold_entry"], last_n)
    strict_success = last_n_mean(strict_log.data["phase/frac_success"], last_n)
    relaxed_success = last_n_mean(relaxed_log.data["phase/frac_success"], last_n)
    strict_band = last_n_mean(strict_log.data["phase/frac_prehold_reachable_band"], last_n)
    strict_band_companion = last_n_mean(strict_log.data["phase/frac_prehold_reachable_band_companion"], last_n)
    relaxed_band = last_n_mean(relaxed_log.data["phase/frac_prehold_reachable_band"], last_n)
    relaxed_band_companion = last_n_mean(relaxed_log.data["phase/frac_prehold_reachable_band_companion"], last_n)

    if (
        strict_gap is not None
        and strict_gap > 0.0
        and strict_hold is not None
        and relaxed_hold is not None
        and relaxed_hold > strict_hold
    ):
        lines.append(
            "- 这更像 strict 组里确实存在“到达 0.17 带但没进 hold”的堆积，而 relaxed gate 把其中一部分转化成了 hold。"
        )
    else:
        lines.append(
            "- 目前还没有看到非常强的“reachable band -> hold 转化”证据，可能需要更长一点的短跑或更多 seed。"
        )

    if (
        strict_success is not None
        and relaxed_success is not None
        and relaxed_success > strict_success
    ):
        lines.append("- relaxed 组的 `phase/frac_success` 更高，说明放宽 tip gate 至少在短跑阶段对成功闭环有正向作用。")
    else:
        lines.append("- relaxed 组暂时没有在 `phase/frac_success` 上形成明显优势，后续要重点看 hold 和 success_strict 的分叉。")

    lines.append("")
    lines.append("## 6. 快速结论")
    lines.append("")
    lines.append(
        f"- strict last-{last_n}: `band017={fmt(strict_band)}`, `band0175={fmt(strict_band_companion)}`, "
        f"`hold={fmt(strict_hold)}`, `success={fmt(strict_success)}`"
    )
    lines.append(
        f"- relaxed last-{last_n}: `band017={fmt(relaxed_band)}`, `band0175={fmt(relaxed_band_companion)}`, "
        f"`hold={fmt(relaxed_hold)}`, `success={fmt(relaxed_success)}`"
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize Exp9.0 tip-gate strict vs 0.175 short runs")
    parser.add_argument("--strict-log", required=True, type=Path)
    parser.add_argument("--relaxed-log", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--last-n", type=int, default=20)
    args = parser.parse_args()

    strict_log = parse_log(args.strict_log, "strict")
    relaxed_log = parse_log(args.relaxed_log, "relaxed0175")
    report = build_report(strict_log, relaxed_log, args.last_n)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report)
    print(f"Summary written to {args.output}")
    print("")
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
