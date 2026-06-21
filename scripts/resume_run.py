#!/usr/bin/env python3
"""Resume a staged ReLU-Tune run from its config snapshot and run state."""

import argparse
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description="Resume a staged LoRA ReLU-Tune run")
    parser.add_argument("--run-dir", required=True, help="Path to the existing run directory")
    args = parser.parse_args()

    from src.pipeline import load_config_snapshot, run_staged_training

    run_dir = Path(args.run_dir)
    config = load_config_snapshot(run_dir)
    run_staged_training(config, run_dir)


if __name__ == "__main__":
    main()
