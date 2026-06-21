#!/usr/bin/env python3
"""Evaluate a ReLU-Tune model on benchmarks and perplexity datasets."""

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description="Evaluate a ReLU-Tune model")
    parser.add_argument(
        "--model-path",
        default=None,
        help="Path to a merged model, adapter, or run directory",
    )
    parser.add_argument(
        "--base-model",
        default=None,
        help="Base Hugging Face model ID",
    )
    parser.add_argument(
        "--activation",
        choices=["relu", "relu2"],
        default=None,
        help="Apply an activation swap when evaluating a base model",
    )
    parser.add_argument(
        "--config",
        action="append",
        default=[],
        help="Optional config path(s) that provide evaluation defaults",
    )
    parser.add_argument("--device", default=None, help="cuda or cpu (default: auto)")
    parser.add_argument(
        "--float32",
        action="store_true",
        help="Load the model in float32 instead of bf16",
    )
    parser.add_argument(
        "--skip-benchmarks",
        action="store_true",
        help="Skip lm-eval benchmark evaluation",
    )
    parser.add_argument(
        "--skip-perplexity",
        action="store_true",
        help="Skip perplexity evaluation",
    )
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=None,
        help="Override the benchmark task list",
    )
    parser.add_argument(
        "--num-fewshot",
        type=int,
        default=None,
        help="Few-shot count for lm-eval",
    )
    parser.add_argument(
        "--benchmark-batch-size",
        default=None,
        help="lm-eval batch size",
    )
    parser.add_argument(
        "--perplexity-datasets",
        nargs="+",
        default=None,
        help="Override the perplexity dataset list",
    )
    parser.add_argument(
        "--perplexity-batch-size",
        type=int,
        default=None,
        help="Perplexity batch size",
    )
    parser.add_argument(
        "--perplexity-seq-len",
        type=int,
        default=None,
        help="Perplexity chunk length",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to save the evaluation JSON",
    )
    args = parser.parse_args()

    if not args.model_path and not args.base_model:
        parser.error("Either --model-path or --base-model must be provided")
    if args.base_model and args.model_path:
        parser.error("--base-model and --model-path are mutually exclusive")
    if args.activation and not args.base_model:
        parser.error("--activation is only supported with --base-model")

    from src.config import load_merged_config
    from src.evaluation import (
        compute_perplexity,
        describe_model,
        evaluate_benchmarks,
        get_default_device,
        load_model_for_evaluation,
        move_model_to_device,
        print_perplexity_results,
        print_results_table,
        save_evaluation_results,
    )
    from src.runtime import collect_runtime_metadata

    config = load_merged_config(args.config) if args.config else {}
    evaluation_config = config.get("evaluation", {})

    benchmarks = args.benchmarks or evaluation_config.get(
        "benchmarks",
        ["piqa", "openbookqa", "sciq", "winogrande", "hellaswag", "arc_easy", "boolq"],
    )
    num_fewshot = (
        args.num_fewshot
        if args.num_fewshot is not None
        else evaluation_config.get("num_fewshot", 0)
    )
    benchmark_batch_size = (
        args.benchmark_batch_size
        if args.benchmark_batch_size is not None
        else evaluation_config.get("benchmark_batch_size", "auto")
    )
    perplexity_datasets = args.perplexity_datasets or evaluation_config.get(
        "perplexity_datasets",
        ["wikitext2", "refinedweb"],
    )
    perplexity_batch_size = (
        args.perplexity_batch_size
        if args.perplexity_batch_size is not None
        else evaluation_config.get("perplexity_batch_size", 4)
    )
    perplexity_seq_len = (
        args.perplexity_seq_len
        if args.perplexity_seq_len is not None
        else evaluation_config.get("perplexity_seq_len", 1024)
    )

    perplexity_dataset_overrides = {}
    wikitext2_config = evaluation_config.get("wikitext2", {})
    if wikitext2_config:
        perplexity_dataset_overrides["wikitext2"] = wikitext2_config
    refinedweb_config = evaluation_config.get("refinedweb", {})
    if refinedweb_config:
        perplexity_dataset_overrides["refinedweb"] = refinedweb_config

    device = args.device or get_default_device()
    model, tokenizer, description, metadata = load_model_for_evaluation(
        model_path=args.model_path,
        base_model=args.base_model,
        activation_type=args.activation,
        use_bf16=not args.float32,
    )
    model = move_model_to_device(model, device)
    describe_model(model)

    benchmark_results = None
    if not args.skip_benchmarks:
        if not benchmarks:
            raise ValueError(
                "No benchmarks configured. Pass --benchmarks or provide config evaluation.benchmarks"
            )
        benchmark_results = evaluate_benchmarks(
            model,
            benchmarks=benchmarks,
            batch_size=benchmark_batch_size,
            num_fewshot=num_fewshot,
            device=device,
        )
        print_results_table(benchmark_results["summary"])

    perplexity_results = None
    if not args.skip_perplexity:
        perplexity_results = compute_perplexity(
            model,
            tokenizer,
            datasets=perplexity_datasets,
            seq_len=perplexity_seq_len,
            batch_size=perplexity_batch_size,
            dataset_overrides=perplexity_dataset_overrides,
            device=device,
        )
        print_perplexity_results(perplexity_results)

    if args.output_dir:
        output_dir = Path(args.output_dir)
    elif args.model_path:
        model_path = Path(args.model_path)
        output_dir = model_path / "eval" if model_path.is_dir() else model_path.parent / "eval"
    else:
        output_dir = Path("evaluation_results")

    result_path = save_evaluation_results(
        output_dir=output_dir,
        model_description=description,
        metadata=metadata,
        benchmark_results=benchmark_results,
        perplexity_results=perplexity_results,
        runtime_metadata=collect_runtime_metadata(
            extra={
                "device": device,
                "model_description": description,
            }
        ),
    )
    print(f"\n[Results] Saved to {result_path}")


if __name__ == "__main__":
    main()
