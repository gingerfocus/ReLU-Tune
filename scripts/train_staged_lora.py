#!/usr/bin/env python3
"""Entry point for staged ReLU-Tune LoRA training."""

import argparse
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description="Train a staged LoRA ReLU-Tune run")
    parser.add_argument(
        "--config",
        action="append",
        required=True,
        help="Config path. Pass multiple times to layer configs.",
    )
    parser.add_argument("--run-dir", default=None, help="Override run directory")
    args = parser.parse_args()

    from src.config import load_merged_config, resolve_run_dir
    from src.pipeline import run_staged_training

    config = load_merged_config(args.config)
    run_dir = resolve_run_dir(config, run_dir=args.run_dir)
    run_staged_training(config, run_dir)


if __name__ == "__main__":
    main()
