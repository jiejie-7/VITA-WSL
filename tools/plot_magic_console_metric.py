from __future__ import annotations

import argparse
import re
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
    parser = argparse.ArgumentParser(description="Plot MAGIC metrics from the original console log format.")
    parser.add_argument("--log", type=Path, required=True, help="Path to train_grf.log / train_*.log")
    parser.add_argument("--metric", type=str, default="Success", help="Console metric name, e.g. Success, Score-Reward, Steps-Taken")
    parser.add_argument("--output", type=Path, required=True, help="Output PNG path")
    parser.add_argument("--label", type=str, default="MAGIC", help="Legend label")
    parser.add_argument("--smooth", type=int, default=1, help="Moving average window over epochs")
    args = parser.parse_args()

    epochs: list[int] = []
    values: list[float] = []
    current_epoch: int | None = None

    metric_prefix = f"{args.metric}:"
    epoch_re = re.compile(r"^Epoch\s+(\d+)")

    for line in args.log.read_text(encoding="utf-8", errors="ignore").splitlines():
        stripped = line.strip()
        m = epoch_re.match(stripped)
        if m:
            current_epoch = int(m.group(1))
            continue
        if current_epoch is None:
            continue
        if stripped.startswith(metric_prefix):
            try:
                val = float(stripped.split()[-1])
            except Exception:
                continue
            epochs.append(current_epoch)
            values.append(val)

    if not values:
        raise ValueError(f"No metric '{args.metric}' found in {args.log}")

    values = moving_average(values, max(1, args.smooth))

    ylabel = args.metric
    if args.metric.lower() == "success":
        values = [v * 100.0 for v in values]
        ylabel = "Success Rate %"

    args.output.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, values, linewidth=2, label=args.label)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    if ylabel == "Success Rate %":
        plt.ylim(0, 100)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.output, dpi=200)
    print(args.output)


if __name__ == "__main__":
    main()
