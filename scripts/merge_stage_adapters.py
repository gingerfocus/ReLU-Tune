#!/usr/bin/env python3
"""Merge completed stage adapters into one dense checkpoint."""

import argparse
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description="Merge completed stage adapters for a run")
    parser.add_argument("--run-dir", required=True, help="Run directory containing run_state.json")
    parser.add_argument("--output-dir", default=None, help="Override merged output directory")
    args = parser.parse_args()

    from src.data import load_tokenizer
    from src.modeling import merge_stage_adapters, prepare_model_with_activation, save_merged_model
    from src.pipeline import load_config_snapshot
    from src.state import load_run_state

    run_dir = Path(args.run_dir)
    state = load_run_state(run_dir / "run_state.json")
    config = load_config_snapshot(run_dir)

    model, layer_indices = prepare_model_with_activation(
        config["model_id"],
        config["activation"],
        output_dir=run_dir,
        use_bf16=config["training"]["bf16"],
    )
    model = merge_stage_adapters(model, state.completed_stages)

    tokenizer = load_tokenizer(config["model_id"])
    output_dir = Path(args.output_dir) if args.output_dir else run_dir / "final_merged"
    save_merged_model(
        model,
        tokenizer,
        output_dir,
        activation_type=config["activation"]["type"],
        layer_indices=layer_indices,
    )
    print(f"Merged model saved to {output_dir}")


if __name__ == "__main__":
    main()
