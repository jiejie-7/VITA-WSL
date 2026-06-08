from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt


def moving_average(values: list[float], window: int) -> list[float]:
    if window <= 1 or len(values) <= 1:
        return values
    out: list[float] = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        chunk = values[start : i + 1]
        out.append(sum(chunk) / len(chunk))
    return out


def quantile(sorted_vals: list[float], q: float) -> float:
    if not sorted_vals:
        raise ValueError("empty values")
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    pos = (len(sorted_vals) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = pos - lo
    return sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac


def mean_std(values: list[float]) -> tuple[float, float]:
    mean = sum(values) / len(values)
    if len(values) <= 1:
        return mean, 0.0
    var = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return mean, var**0.5


def load_jsonl_metric(path: Path, metric: str, phase: str) -> list[tuple[int, float]]:
    rows: list[tuple[int, float]] = []
    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            entry_phase = entry.get("phase")
            if entry_phase is None:
                entry_phase = "eval" if any(k.startswith("eval_") for k in entry) else "train"
            if phase != "all" and entry_phase != phase:
                continue
            if metric not in entry:
                continue
            step = entry.get("total_env_steps", entry.get("step", idx))
            rows.append((int(step), float(entry[metric])))
    return rows


def load_summary_metric(path: Path, metric: str) -> list[tuple[int, float]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} is not a summary dict")
    suffix = f"/{metric}/{metric}"
    key = next((k for k in data if k.endswith(suffix)), None)
    if key is None:
        raise KeyError(f"No summary key ending with {suffix!r} in {path}")
    rows = sorted(data[key], key=lambda row: int(row[1]))
    return [(int(row[1]), float(row[2])) for row in rows]


def load_metric(path: Path, metric: str, phase: str) -> list[tuple[int, float]]:
    if path.name == "summary.json":
        return load_summary_metric(path, metric)
    return load_jsonl_metric(path, metric, phase)


def aggregate(
    files: Iterable[Path],
    metric: str,
    phase: str,
    band: str,
) -> tuple[list[int], list[float], list[float], list[float], list[int]]:
    by_step: dict[int, list[float]] = {}
    for path in files:
        rows = load_metric(path, metric, phase)
        if not rows:
            raise ValueError(f"No {metric!r} values found in {path}")
        for step, value in rows:
            by_step.setdefault(step, []).append(value)

    steps = sorted(step for step, vals in by_step.items() if vals)
    centers: list[float] = []
    lowers: list[float] = []
    uppers: list[float] = []
    counts: list[int] = []
    for step in steps:
        vals = by_step[step]
        counts.append(len(vals))
        if band == "iqr":
            vals_sorted = sorted(vals)
            centers.append(quantile(vals_sorted, 0.5))
            lowers.append(quantile(vals_sorted, 0.25))
            uppers.append(quantile(vals_sorted, 0.75))
        else:
            center, std = mean_std(vals)
            if band == "sem":
                std = std / (len(vals) ** 0.5)
            centers.append(center)
            lowers.append(center - std)
            uppers.append(center + std)
    return steps, centers, lowers, uppers, counts


def parse_groups(raw_groups: list[list[str]]) -> list[tuple[str, list[Path]]]:
    groups: list[tuple[str, list[Path]]] = []
    for group in raw_groups:
        if len(group) < 2:
            raise ValueError("--group requires LABEL followed by at least one file")
        label = group[0]
        files = [Path(p) for p in group[1:]]
        groups.append((label, files))
    return groups


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot multi-seed mean/std or median/IQR curves.")
    parser.add_argument("--group", nargs="+", action="append", required=True, metavar=("LABEL", "FILE"))
    parser.add_argument("--metric", default="eval_win_rate")
    parser.add_argument("--phase", default="eval", choices=["all", "train", "eval"])
    parser.add_argument("--band", default="std", choices=["std", "sem", "iqr"])
    parser.add_argument("--smooth", type=int, default=1)
    parser.add_argument("--title", default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--colors", nargs="*", default=None)
    parser.add_argument("--winrate-style", action="store_true")
    args = parser.parse_args()

    groups = parse_groups(args.group)
    if args.colors and len(args.colors) != len(groups):
        raise ValueError("--colors must match number of groups")

    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    ax.set_facecolor("#f9f9f9")
    ax.grid(True, alpha=0.35, color="#cccccc")

    for idx, (label, files) in enumerate(groups):
        steps, center, lower, upper, counts = aggregate(files, args.metric, args.phase, args.band)
        if min(counts) != max(counts):
            print(f"[WARN] {label}: seed count varies across steps, min={min(counts)}, max={max(counts)}")
        center = moving_average(center, args.smooth)
        lower = moving_average(lower, args.smooth)
        upper = moving_average(upper, args.smooth)

        x_vals = [s / 1e6 for s in steps]
        y_vals = center
        y_lower = lower
        y_upper = upper
        if args.winrate_style and "win_rate" in args.metric:
            y_vals = [v * 100.0 for v in y_vals]
            y_lower = [max(0.0, v * 100.0) for v in y_lower]
            y_upper = [min(100.0, v * 100.0) for v in y_upper]

        color = args.colors[idx] if args.colors else None
        line = ax.plot(x_vals, y_vals, linewidth=2.2, label=f"{label} (n={max(counts)})", color=color)[0]
        ax.fill_between(x_vals, y_lower, y_upper, alpha=0.18, color=line.get_color())

    ax.set_xlabel("T (mil)")
    if args.winrate_style and "win_rate" in args.metric:
        ax.set_ylabel("Test Win Rate%")
        ax.set_ylim(0, 100)
        ax.set_xlim(left=0)
    else:
        ax.set_ylabel(args.metric)
    ax.set_title(args.title or args.metric)
    ax.legend(frameon=True)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=300)
    print(args.output)


if __name__ == "__main__":
    main()
