from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def moving_average(values: list[float], window: int) -> list[float]:
    if window <= 1 or len(values) < window:
        return values
    out: list[float] = []
    acc = 0.0
    for i, v in enumerate(values):
        acc += v
        if i >= window:
            acc -= values[i - window]
        if i >= window - 1:
            out.append(acc / window)
    return [out[0]] * (len(values) - len(out)) + out


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot one MAGIC metric from a JSON-lines train.log.")
    parser.add_argument("--log", type=Path, required=True, help="Path to train.log")
    parser.add_argument("--metric", type=str, default="won_flag_rate", help="Metric key in train.log")
    parser.add_argument("--output", type=Path, required=True, help="Output PNG path")
    parser.add_argument("--label", type=str, default=None, help="Legend label")
    parser.add_argument("--smooth", type=int, default=20, help="Moving average window")
    parser.add_argument(
        "--phase",
        type=str,
        choices=["train", "eval", "all"],
        default="train",
        help="Which phase entries to use",
    )
    args = parser.parse_args()

    xs: list[float] = []
    ys: list[float] = []
    with args.log.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            phase = obj.get("phase", "train")
            if args.phase != "all" and phase != args.phase:
                continue
            if args.metric not in obj:
                continue
            if "total_env_steps" not in obj:
                continue
            xs.append(float(obj["total_env_steps"]) / 1e6)
            ys.append(float(obj[args.metric]))

    if not xs:
        raise ValueError(f"No data found for metric '{args.metric}' in {args.log}")

    ys = moving_average(ys, max(1, args.smooth))

    ylabel = args.metric
    if args.metric in {"won_flag_rate", "eval_win_rate", "incre_win_rate"}:
        ys = [v * 100.0 for v in ys]
        ylabel = "Success Rate %"

    args.output.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.plot(xs, ys, linewidth=2, label=args.label or args.metric)
    plt.xlabel("T (mil)")
    plt.ylabel(ylabel)
    plt.xlim(left=0)
    if ylabel == "Success Rate %":
        plt.ylim(0, 100)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.output, dpi=200)
    print(args.output)


if __name__ == "__main__":
    main()
