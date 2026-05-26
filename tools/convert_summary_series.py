from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert one tensorboard summary series to JSONL.")
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--source-suffix", type=str, required=True, help="Suffix match for the TB key.")
    parser.add_argument("--output-key", type=str, required=True)
    parser.add_argument("--phase", type=str, choices=["train", "eval"], default="train")
    parser.add_argument("--output-jsonl", type=Path, required=True)
    args = parser.parse_args()

    with args.summary.open("r", encoding="utf-8") as f:
        data = json.load(f)

    source_key = next((k for k in data if k.endswith(args.source_suffix)), None)
    if source_key is None:
        raise KeyError(f"No summary key ends with {args.source_suffix!r}")

    rows = sorted(data[source_key], key=lambda r: int(r[1]))
    with args.output_jsonl.open("w", encoding="utf-8") as f:
        for i, row in enumerate(rows, start=1):
            step = int(row[1])
            value = float(row[2])
            out = {
                "step": i,
                "phase": args.phase,
                "total_env_steps": step,
                args.output_key: value,
            }
            f.write(json.dumps(out, ensure_ascii=False) + "\n")

    print(args.output_jsonl)


if __name__ == "__main__":
    main()
