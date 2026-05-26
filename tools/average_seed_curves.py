from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Average multiple training curves by step.")
    parser.add_argument("--log-files", type=Path, nargs="+", required=True)
    parser.add_argument("--metric", type=str, default="eval_win_rate")
    parser.add_argument("--phase", type=str, choices=["all", "train", "eval"], default="eval")
    parser.add_argument("--x-axis", type=str, choices=["step", "total_env_steps"], default="total_env_steps")
    parser.add_argument("--output-jsonl", type=Path, required=True)
    args = parser.parse_args()

    by_step: dict[int, list[float]] = defaultdict(list)
    for path in args.log_files:
        for idx, entry in enumerate(load_jsonl(path)):
            phase = entry.get("phase")
            if phase is None:
                phase = "eval" if any(k.startswith("eval_") for k in entry.keys()) else "train"
            if args.phase != "all" and phase != args.phase:
                continue
            value = entry.get(args.metric)
            if value is None:
                continue
            step_val = entry.get(args.x_axis)
            if step_val is None:
                step_val = entry.get("step", idx + 1)
            step_val = int(step_val)
            by_step[step_val].append(float(value))

    if not by_step:
        raise ValueError(f"No values found for metric {args.metric!r}")

    steps = sorted(by_step.keys())
    with args.output_jsonl.open("w", encoding="utf-8") as f:
        for i, step in enumerate(steps, start=1):
            vals = by_step[step]
            row = {
                "step": i,
                "phase": args.phase if args.phase != "all" else "eval",
                "total_env_steps": step,
                args.metric: sum(vals) / len(vals),
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(args.output_jsonl)


if __name__ == "__main__":
    main()
