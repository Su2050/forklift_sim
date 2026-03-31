"""Summarize mirrored single-point steering diagnostics for Exp8.3.

This script reads multiple `diagnose_exp83_single_point_steering.py` outputs,
groups mirrored `(y, yaw)` pairs, and reports whether the policy's signed raw
steer flips together with the environment steering target.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


SIGN_EPS = 0.02


@dataclass(frozen=True)
class ModeStats:
    success: bool
    ever_inserted: bool
    ever_hold_entry: bool
    ever_clean_insert_ready: bool
    mean_target_cmd: float
    mean_raw_steer: float
    mean_applied_steer: float
    steer_wrong_sign_frac: float
    episode_steps: int


@dataclass(frozen=True)
class CaseStats:
    label: str
    checkpoint: str
    x_root: float
    y_m: float
    yaw_deg: float
    normal: ModeStats
    zero_steer: ModeStats
    normal_csv_path: str
    zero_csv_path: str


def _sign_of(value: float) -> int:
    if value > SIGN_EPS:
        return 1
    if value < -SIGN_EPS:
        return -1
    return 0


def _mean_from_csv(csv_path: Path, key: str) -> float:
    values: list[float] = []
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if int(row["step"]) <= 0:
                continue
            values.append(float(row[key]))
    return float(sum(values) / len(values)) if values else 0.0


def _load_case(summary_path: Path) -> CaseStats:
    data = json.loads(summary_path.read_text(encoding="utf-8"))

    def _mode_stats(mode_key: str) -> tuple[ModeStats, str]:
        mode = data["modes"][mode_key]
        csv_path = Path(mode["csv_path"])
        mean_raw_steer = _mean_from_csv(csv_path, "raw_steer")
        mean_applied_steer = _mean_from_csv(csv_path, "applied_steer")
        stats = ModeStats(
            success=bool(mode["success"]),
            ever_inserted=bool(mode["ever_inserted"]),
            ever_hold_entry=bool(mode["ever_hold_entry"]),
            ever_clean_insert_ready=bool(mode["ever_clean_insert_ready"]),
            mean_target_cmd=float(mode["mean_steer_target_cmd"]),
            mean_raw_steer=mean_raw_steer,
            mean_applied_steer=mean_applied_steer,
            steer_wrong_sign_frac=float(mode["steer_wrong_sign_frac"]),
            episode_steps=int(mode["episode_steps"]),
        )
        return stats, str(csv_path)

    normal, normal_csv = _mode_stats("normal")
    zero_steer, zero_csv = _mode_stats("zero_steer")
    return CaseStats(
        label=str(data["label"]),
        checkpoint=str(data["checkpoint"]),
        x_root=float(data["x_root"]),
        y_m=float(data["y_m"]),
        yaw_deg=float(data["yaw_deg"]),
        normal=normal,
        zero_steer=zero_steer,
        normal_csv_path=normal_csv,
        zero_csv_path=zero_csv,
    )


def _iter_summary_paths(input_dir: Path, label_prefix: str) -> Iterable[Path]:
    for path in sorted(input_dir.glob(f"{label_prefix}*_summary.json")):
        yield path


def _case_key(case: CaseStats) -> tuple[float, float]:
    return (round(case.y_m, 6), round(case.yaw_deg, 6))


def _mirror_key(case: CaseStats) -> tuple[float, float]:
    return (round(abs(case.y_m), 6), round(abs(case.yaw_deg), 6))


def _format_sign(sign_value: int) -> str:
    if sign_value > 0:
        return "+"
    if sign_value < 0:
        return "-"
    return "0"


def _build_pair_report(a: CaseStats, b: CaseStats) -> dict[str, object]:
    a_target_sign = _sign_of(a.normal.mean_target_cmd)
    b_target_sign = _sign_of(b.normal.mean_target_cmd)
    a_raw_sign = _sign_of(a.normal.mean_raw_steer)
    b_raw_sign = _sign_of(b.normal.mean_raw_steer)

    return {
        "pair_key": {"abs_y_m": abs(a.y_m), "abs_yaw_deg": abs(a.yaw_deg)},
        "case_a": {
            "label": a.label,
            "y_m": a.y_m,
            "yaw_deg": a.yaw_deg,
            "normal_success": a.normal.success,
            "zero_success": a.zero_steer.success,
            "normal_mean_target_cmd": a.normal.mean_target_cmd,
            "normal_mean_raw_steer": a.normal.mean_raw_steer,
            "normal_mean_applied_steer": a.normal.mean_applied_steer,
            "normal_target_sign": _format_sign(a_target_sign),
            "normal_raw_sign": _format_sign(a_raw_sign),
            "normal_wrong_sign_frac": a.normal.steer_wrong_sign_frac,
        },
        "case_b": {
            "label": b.label,
            "y_m": b.y_m,
            "yaw_deg": b.yaw_deg,
            "normal_success": b.normal.success,
            "zero_success": b.zero_steer.success,
            "normal_mean_target_cmd": b.normal.mean_target_cmd,
            "normal_mean_raw_steer": b.normal.mean_raw_steer,
            "normal_mean_applied_steer": b.normal.mean_applied_steer,
            "normal_target_sign": _format_sign(b_target_sign),
            "normal_raw_sign": _format_sign(b_raw_sign),
            "normal_wrong_sign_frac": b.normal.steer_wrong_sign_frac,
        },
        "mirror_checks": {
            "target_flips_sign": a_target_sign == -b_target_sign and a_target_sign != 0,
            "raw_flips_sign": a_raw_sign == -b_raw_sign and a_raw_sign != 0,
            "raw_matches_target_both_cases": a_raw_sign == a_target_sign and b_raw_sign == b_target_sign,
            "normal_success_prefers_positive_target_case": (
                (a_target_sign > 0 and a.normal.success and not b.normal.success)
                or (b_target_sign > 0 and b.normal.success and not a.normal.success)
            ),
        },
    }


def _write_markdown(report_path: Path, label_prefix: str, pairs: list[dict[str, object]], singles: list[CaseStats]) -> None:
    lines: list[str] = []
    lines.append(f"# Mirrored Steering Audit: `{label_prefix}`")
    lines.append("")
    lines.append("## Overall")
    lines.append("")

    total_pairs = len(pairs)
    target_flip_pairs = sum(bool(p["mirror_checks"]["target_flips_sign"]) for p in pairs)
    raw_flip_pairs = sum(bool(p["mirror_checks"]["raw_flips_sign"]) for p in pairs)
    raw_match_pairs = sum(bool(p["mirror_checks"]["raw_matches_target_both_cases"]) for p in pairs)

    lines.append(f"- mirrored pairs analyzed: {total_pairs}")
    lines.append(f"- pairs where env target flips sign: {target_flip_pairs}/{total_pairs}")
    lines.append(f"- pairs where policy raw steer flips sign: {raw_flip_pairs}/{total_pairs}")
    lines.append(f"- pairs where raw steer matches target sign in both cases: {raw_match_pairs}/{total_pairs}")
    lines.append("")
    lines.append("## Pairs")
    lines.append("")

    for pair in pairs:
        key = pair["pair_key"]
        a = pair["case_a"]
        b = pair["case_b"]
        checks = pair["mirror_checks"]
        lines.append(
            f"- `|y|={key['abs_y_m']:.3f}, |yaw|={key['abs_yaw_deg']:.1f}`: "
            f"target flip={checks['target_flips_sign']}, raw flip={checks['raw_flips_sign']}, "
            f"raw matches both={checks['raw_matches_target_both_cases']}"
        )
        lines.append(
            f"  case A `(y={a['y_m']:+.3f}, yaw={a['yaw_deg']:+.1f})`: "
            f"target={a['normal_target_sign']} ({a['normal_mean_target_cmd']:+.3f}), "
            f"raw={a['normal_raw_sign']} ({a['normal_mean_raw_steer']:+.3f}), "
            f"normal_success={a['normal_success']}, zero_success={a['zero_success']}, "
            f"wrong_sign_frac={a['normal_wrong_sign_frac']:.3f}"
        )
        lines.append(
            f"  case B `(y={b['y_m']:+.3f}, yaw={b['yaw_deg']:+.1f})`: "
            f"target={b['normal_target_sign']} ({b['normal_mean_target_cmd']:+.3f}), "
            f"raw={b['normal_raw_sign']} ({b['normal_mean_raw_steer']:+.3f}), "
            f"normal_success={b['normal_success']}, zero_success={b['zero_success']}, "
            f"wrong_sign_frac={b['normal_wrong_sign_frac']:.3f}"
        )
        lines.append("")

    if singles:
        lines.append("## Single Cases")
        lines.append("")
        for case in sorted(singles, key=lambda c: (c.yaw_deg, c.y_m)):
            lines.append(
                f"- `{case.label}`: `(y={case.y_m:+.3f}, yaw={case.yaw_deg:+.1f})`, "
                f"target={case.normal.mean_target_cmd:+.3f}, raw={case.normal.mean_raw_steer:+.3f}, "
                f"normal_success={case.normal.success}, zero_success={case.zero_steer.success}"
            )
        lines.append("")

    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize mirrored single-point steering diagnostics.")
    parser.add_argument("--input_dir", type=Path, required=True)
    parser.add_argument("--label_prefix", type=str, required=True)
    parser.add_argument("--output_json", type=Path, required=True)
    parser.add_argument("--output_md", type=Path, required=True)
    args = parser.parse_args()

    cases = [_load_case(path) for path in _iter_summary_paths(args.input_dir, args.label_prefix)]
    if not cases:
        raise SystemExit(f"No summary files found for prefix {args.label_prefix!r} in {args.input_dir}")

    by_key = {_case_key(case): case for case in cases}
    pairs: list[dict[str, object]] = []
    used_keys: set[tuple[float, float]] = set()

    for case in sorted(cases, key=lambda c: (_mirror_key(c), c.yaw_deg, c.y_m)):
        key = _case_key(case)
        if key in used_keys:
            continue
        mirror = by_key.get((round(-case.y_m, 6), round(-case.yaw_deg, 6)))
        if mirror is not None and _case_key(mirror) not in used_keys and _case_key(mirror) != key:
            pairs.append(_build_pair_report(case, mirror))
            used_keys.add(key)
            used_keys.add(_case_key(mirror))

    singles = [case for case in cases if _case_key(case) not in used_keys]

    result = {
        "label_prefix": args.label_prefix,
        "input_dir": str(args.input_dir),
        "pair_count": len(pairs),
        "pairs": pairs,
        "single_cases": [
            {
                "label": case.label,
                "y_m": case.y_m,
                "yaw_deg": case.yaw_deg,
                "normal_mean_target_cmd": case.normal.mean_target_cmd,
                "normal_mean_raw_steer": case.normal.mean_raw_steer,
                "normal_success": case.normal.success,
                "zero_success": case.zero_steer.success,
            }
            for case in singles
        ],
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    _write_markdown(args.output_md, args.label_prefix, pairs, singles)
    print(f"[DONE] wrote {args.output_json}")
    print(f"[DONE] wrote {args.output_md}")


if __name__ == "__main__":
    main()
