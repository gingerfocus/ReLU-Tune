#!/usr/bin/env python3
"""Measure prefill layerwise activation sparsity for a ReLU-Tune model."""

import argparse
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description="Measure ReLU-Tune prefill sparsity")
    parser.add_argument("--model-path", default=None, help="Path to merged model, adapter, or run directory")
    parser.add_argument("--base-model", default=None, help="Base HuggingFace model ID")
    parser.add_argument(
        "--activation",
        choices=["relu", "relu2"],
        default=None,
        help="Apply activation swap when measuring a base model",
    )
    parser.add_argument("--config", action="append", default=[], help="Optional config path(s)")
    parser.add_argument("--device", default=None, help="cuda or cpu (default: auto)")
    parser.add_argument("--float32", action="store_true", help="Load model in float32 instead of bf16")
    parser.add_argument("--num-samples", type=int, default=None, help="Override sample count")
    parser.add_argument("--batch-size", type=int, default=None, help="Override sparsity batch size")
    parser.add_argument("--block-size", type=int, default=None, help="Override token block size")
    parser.add_argument("--threshold", type=float, default=None, help="Override near-zero threshold")
    parser.add_argument("--output-path", default=None, help="Optional JSON output path")
    args = parser.parse_args()

    from src.config import load_merged_config
    from src.evaluation import get_default_device, load_model_for_evaluation, move_model_to_device
    from src.sparsity import measure_prefill_sparsity, save_prefill_sparsity

    config = load_merged_config(args.config) if args.config else {}
    sparsity_config = config.get("measure_sparsity", {})
    data_config = config.get("data", {})

    device = args.device or get_default_device()
    model, tokenizer, description, _metadata = load_model_for_evaluation(
        model_path=args.model_path,
        base_model=args.base_model,
        activation_type=args.activation,
        use_bf16=not args.float32,
    )
    model = move_model_to_device(model, device)

    payload = measure_prefill_sparsity(
        model=model,
        tokenizer=tokenizer,
        num_samples=(
            args.num_samples
            if args.num_samples is not None
            else sparsity_config.get("num_samples", 32)
        ),
        batch_size=(
            args.batch_size
            if args.batch_size is not None
            else sparsity_config.get("batch_size", 4)
        ),
        dataset_name=sparsity_config.get("dataset", "allenai/c4"),
        dataset_config=sparsity_config.get("dataset_config", "en"),
        dataset_split=sparsity_config.get("dataset_split", "train"),
        text_column=sparsity_config.get("text_column", "text"),
        block_size=(
            args.block_size
            if args.block_size is not None
            else data_config.get("block_size", 1024)
        ),
        threshold=args.threshold if args.threshold is not None else sparsity_config.get("threshold", 0.0),
        device=device,
    )
    payload["model"] = description

    if args.output_path:
        output_path = Path(args.output_path)
    elif args.model_path:
        path = Path(args.model_path)
        output_path = (path if path.is_dir() else path.parent) / "prefill_sparsity.json"
    else:
        output_path = Path("evaluation_results") / "prefill_sparsity.json"

    save_prefill_sparsity(output_path, payload)
    print(f"[Sparsity] Saved to {output_path}")
    print(f"[Sparsity] Average sparsity: {payload['summary']['average_sparsity']:.2f}%")


if __name__ == "__main__":
    main()
