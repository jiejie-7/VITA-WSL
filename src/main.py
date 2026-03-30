from __future__ import annotations

import argparse
from pathlib import Path

from src.algorithms.onpolicy_baseline import run_onpolicy_football, run_onpolicy_smac
from src.utils import load_config, set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MAPPO/VITA on supported on-policy environments.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    parser.add_argument("--log-dir", type=str, default="logs", help="Directory for logs.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed = int(cfg.get("seed", 42))
    set_seed(seed)

    algo_name = str(cfg.get("algorithm", "mappo")).lower()
    env_name = str(cfg.get("env", {}).get("name", "smac")).lower()

    run_dir = Path(args.log_dir) / str(algo_name)
    if env_name == "smac":
        run_onpolicy_smac(cfg, config_path=Path(args.config), run_dir=run_dir)
    elif env_name in {"football", "grf"}:
        run_onpolicy_football(cfg, config_path=Path(args.config), run_dir=run_dir)
    else:
        raise ValueError(f"Unsupported environment: {env_name!r}. Supported: smac, football/grf.")


if __name__ == "__main__":
    main()
