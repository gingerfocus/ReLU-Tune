#!/usr/bin/env python3
"""Layer-by-layer ReLU-fication sweep to identify the best candidate layers.

For each of the 16 layers in Llama 3.2 1B, this script:
  1. ReLU-fies only that single layer
  2. Applies LoRA only to that layer
  3. Runs a short training (default 30 steps)
  4. Collects train loss, validation loss, and optionally prefill sparsity
  5. Ranks all 16 layers by a combined score
  6. Outputs the top-K layers as a ready-to-use config snippet

Usage:
  python scripts/relu_layer_sweep.py \
      --config configs/train_llama32_1b.yaml \
      --steps-per-layer 30 \
      --top-k 8 \
      --measure-sparsity \
      --output results/layer_sweep.json
"""

import argparse
import faulthandler
import gc
import json
import subprocess
import sys
import time
from pathlib import Path

faulthandler.enable()

import torch
import yaml

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.activations import apply_activation_to_mlp_layers, get_activation_summary
from src.config import load_merged_config
from src.data import load_tokenizer, prepare_refinedweb_train_validation, slice_stage_dataset
from src.lora import apply_lora, get_lora_target_modules
from src.modeling import load_base_model
from src.runtime import enable_transformers_checkpoint_resume_compat, get_samples_per_step, set_seed
from src.sparsity import measure_prefill_sparsity

def _cleanup(*objects):
    for obj in objects:
        if obj is not None:
            del obj
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _extract_final_loss(log_history):
    for entry in reversed(log_history):
        if "loss" in entry:
            return entry["loss"]
    return None


def _extract_final_eval_loss(log_history):
    for entry in reversed(log_history):
        if "eval_loss" in entry:
            return entry["eval_loss"]
    return None


def run_single_layer(
    layer_index,
    config,
    tokenizer,
    train_dataset,
    eval_dataset,
    steps_per_layer,
    measure_sparsity,
    sparsity_config,
):
    model = load_base_model(config["model_id"], use_bf16=config["training"]["bf16"])
    model, _ = apply_activation_to_mlp_layers(
        model,
        activation_name=config["activation"]["type"],
        layer_indices=[layer_index],
    )

    target_modules = get_lora_target_modules(
        model,
        layer_indices=[layer_index],
        include_mlp=config["lora"].get("include_mlp", True),
        include_attention=config["lora"].get("include_attention", True),
    )
    model = apply_lora(
        model,
        target_modules=target_modules,
        rank=config["lora"]["r"],
        alpha=config["lora"]["alpha"],
        dropout=config["lora"].get("dropout", 0.0),
        layers_to_transform=[layer_index],
    )

    samples_per_step = get_samples_per_step(
        config["training"]["batch_size"],
        config["training"]["gradient_accumulation_steps"],
    )
    stage_dataset = slice_stage_dataset(
        train_dataset,
        start_index=layer_index * steps_per_layer * samples_per_step,
        length=steps_per_layer * samples_per_step,
    )

    from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments

    training_args = TrainingArguments(
        output_dir=f"/tmp/relu_sweep_layer_{layer_index}",
        max_steps=steps_per_layer,
        per_device_train_batch_size=config["training"]["batch_size"],
        gradient_accumulation_steps=config["training"]["gradient_accumulation_steps"],
        learning_rate=float(config["training"]["learning_rate"]),
        warmup_steps=config["training"]["warmup_steps"],
        weight_decay=float(config["training"]["weight_decay"]),
        logging_steps=config["training"]["logging_steps"],
        save_steps=steps_per_layer + 1,
        save_total_limit=0,
        bf16=config["training"]["bf16"],
        gradient_checkpointing=config["training"]["gradient_checkpointing"],
        lr_scheduler_type=config["training"]["lr_scheduler_type"],
        optim=config["training"]["optimizer"],
        torch_compile=config["training"].get("torch_compile", False),
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        logging_first_step=True,
        eval_strategy="steps",
        eval_steps=steps_per_layer,
        per_device_eval_batch_size=config.get("validation", {}).get("batch_size", config["training"]["batch_size"]),
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=stage_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    trainer.train()
    log_history = trainer.state.log_history

    result = {
        "layer": layer_index,
        "train_loss": _extract_final_loss(log_history),
        "eval_loss": _extract_final_eval_loss(log_history),
    }

    if measure_sparsity:
        sparsity_payload = measure_prefill_sparsity(
            model=trainer.model,
            tokenizer=tokenizer,
            num_samples=sparsity_config.get("num_samples", 16),
            batch_size=sparsity_config.get("batch_size", 4),
            dataset_name=sparsity_config.get("dataset", "allenai/c4"),
            dataset_config=sparsity_config.get("dataset_config", "en"),
            dataset_split=sparsity_config.get("dataset_split", "train"),
            text_column=sparsity_config.get("text_column", "text"),
            block_size=config["data"]["block_size"],
            threshold=sparsity_config.get("threshold", 0.0),
            device=training_args.device,
        )
        result["sparsity"] = sparsity_payload["results"].get(layer_index, 0.0)

    _cleanup(trainer, model)
    return result


def compute_ranks(results, measure_sparsity):
    layers = sorted(results, key=lambda r: r["layer"])
    n = len(layers)

    def rank_by(key, reverse=False):
        sorted_layers = sorted(layers, key=lambda r: r.get(key, float("inf")), reverse=reverse)
        ranks = {}
        for rank, entry in enumerate(sorted_layers, start=1):
            ranks[entry["layer"]] = rank
        return ranks

    train_ranks = rank_by("train_loss", reverse=False)
    eval_ranks = rank_by("eval_loss", reverse=False)

    combined = {}
    for entry in layers:
        li = entry["layer"]
        score = train_ranks[li] + eval_ranks[li]
        if measure_sparsity:
            sparsity_ranks = rank_by("sparsity", reverse=True)
            score += sparsity_ranks[li]
        combined[li] = score

    sorted_combined = sorted(combined.items(), key=lambda x: x[1])
    return sorted_combined, train_ranks, eval_ranks


def main():
    parser = argparse.ArgumentParser(description="Sweep ReLU-fication across all layers")
    parser.add_argument("--config", action="append", required=True, help="Base config path(s)")
    parser.add_argument("--steps-per-layer", type=int, default=30, help="Training steps per layer")
    parser.add_argument("--top-k", type=int, default=8, help="Number of layers to select")
    parser.add_argument("--measure-sparsity", action="store_true", help="Measure prefill sparsity per layer")
    parser.add_argument("--output", default=None, help="Path to save JSON results")
    parser.add_argument("--num-layers", type=int, default=16, help="Number of layers to sweep (default 16)")
    # Hidden/internal arguments for subprocess communication
    parser.add_argument("--single-layer", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--progress-file", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    # Determine progress file path
    if args.progress_file:
        progress_file = Path(args.progress_file)
    else:
        if args.output:
            out_path = Path(args.output)
            progress_file = out_path.with_name(out_path.stem + "_progress.json")
        else:
            progress_file = Path("layer_sweep_progress.json")

    # ------------------ WORKER MODE ------------------
    if args.single_layer is not None:
        enable_transformers_checkpoint_resume_compat()
        config = load_merged_config(args.config)
        set_seed(config.get("seed", 42))

        tokenizer = load_tokenizer(config["model_id"])
        validation_config = config.get("validation", {})
        train_dataset, eval_dataset = prepare_refinedweb_train_validation(
            model_id=config["model_id"],
            tokenizer=tokenizer,
            train_docs=config["data"]["train_docs"],
            validation_docs=validation_config.get("docs", 0),
            validation_skip_docs=validation_config.get("skip_docs", 0),
            block_size=config["data"]["block_size"],
            revision=config["data"]["revision"],
            seed=config.get("seed", 42),
            cache_root=config["data"]["cache_root"],
        )

        samples_per_step = get_samples_per_step(
            config["training"]["batch_size"],
            config["training"]["gradient_accumulation_steps"],
        )
        total_needed = args.num_layers * args.steps_per_layer * samples_per_step
        if len(train_dataset) < total_needed:
            raise ValueError(
                f"Train dataset too small: {len(train_dataset):,} < {total_needed:,} sequences needed"
            )

        sparsity_config = config.get("measure_sparsity", {})
        layer_index = args.single_layer

        print(f"\n--- [Worker] Running Layer {layer_index} ---")
        result = run_single_layer(
            layer_index=layer_index,
            config=config,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            steps_per_layer=args.steps_per_layer,
            measure_sparsity=args.measure_sparsity,
            sparsity_config=sparsity_config,
        )

        # Print results of the layer
        print(f"  Train loss: {result['train_loss']:.4f}" if result["train_loss"] is not None else "  Train loss: N/A")
        print(f"  Eval loss:  {result['eval_loss']:.4f}" if result["eval_loss"] is not None else "  Eval loss: N/A")
        if args.measure_sparsity:
            print(f"  Sparsity:   {result['sparsity']:.2f}%")

        # Load existing progress results
        results = []
        if progress_file.exists():
            try:
                results = json.loads(progress_file.read_text(encoding="utf-8"))
            except Exception:
                pass

        results = [r for r in results if r["layer"] != layer_index]
        results.append(result)
        results = sorted(results, key=lambda x: x["layer"])

        # Atomic write
        temp_file = progress_file.with_suffix(".tmp")
        temp_file.parent.mkdir(parents=True, exist_ok=True)
        temp_file.write_text(json.dumps(results, indent=2), encoding="utf-8")
        temp_file.replace(progress_file)

        sys.exit(0)

    # ------------------ MANAGER MODE ------------------
    enable_transformers_checkpoint_resume_compat()
    config = load_merged_config(args.config)
    set_seed(config.get("seed", 42))

    print("=" * 70)
    print("ReLU-Tune Layer Sweep (Manager Mode)")
    print("=" * 70)
    print(f"Model: {config['model_id']}")
    print(f"Activation: {config['activation']['type']}")
    print(f"Steps per layer: {args.steps_per_layer}")
    print(f"Top-K: {args.top_k}")
    print(f"Measure sparsity: {args.measure_sparsity}")
    print(f"Num layers: {args.num_layers}")
    print(f"Progress file: {progress_file}")
    print("=" * 70)

    # Filter out --single-layer and --progress-file arguments from sys.argv
    base_args = []
    skip_next = False
    for arg in sys.argv[1:]:
        if skip_next:
            skip_next = False
            continue
        if arg in ("--single-layer", "--progress-file"):
            skip_next = True
            continue
        if arg.startswith("--single-layer=") or arg.startswith("--progress-file="):
            continue
        base_args.append(arg)

    # Load existing progress
    results = []
    if progress_file.exists():
        try:
            results = json.loads(progress_file.read_text(encoding="utf-8"))
            print(f"Loaded {len(results)} existing result(s) from progress file.")
        except Exception:
            print("Failed to parse progress file, starting fresh.")

    completed_layers = {r["layer"] for r in results}

    max_retries = 10
    cooldown_seconds = 10
    for layer_index in range(args.num_layers):
        if layer_index in completed_layers:
            print(f"Layer {layer_index} already complete (skipped)")
            continue

        cmd = [
            sys.executable,
            sys.argv[0]
        ] + base_args + [
            "--single-layer", str(layer_index),
            "--progress-file", str(progress_file)
        ]

        success = False
        for attempt in range(1, max_retries + 1):
            print(f"\n[Manager] Spawning subprocess for Layer {layer_index} (Attempt {attempt}/{max_retries})...")
            try:
                # run subprocess and stream stdout/stderr directly
                subprocess.run(cmd, check=True)
                success = True
                break
            except subprocess.CalledProcessError as exc:
                print(f"[Manager] Subprocess for Layer {layer_index} failed on attempt {attempt}/{max_retries}: {exc}")
                if attempt < max_retries:
                    print(f"[Manager] Waiting {cooldown_seconds} seconds before retrying...")
                    time.sleep(cooldown_seconds)
                else:
                    print(f"[Manager] Subprocess for Layer {layer_index} failed after {max_retries} attempts. Aborting.")
                    raise exc

    # Load final results
    if progress_file.exists():
        results = json.loads(progress_file.read_text(encoding="utf-8"))
    else:
        raise FileNotFoundError(f"Progress file {progress_file} not found after run.")

    sorted_combined, train_ranks, eval_ranks = compute_ranks(results, args.measure_sparsity)
    top_k_layers = [layer for layer, _ in sorted_combined[: args.top_k]]

    print("\n" + "=" * 70)
    print("Layer Sweep Results")
    print("=" * 70)
    header = f"{'Layer':<6} {'Train Loss':<12} {'Eval Loss':<12} {'Train Rank':<11} {'Eval Rank':<10}"
    if args.measure_sparsity:
        header += f" {'Sparsity':<10} {'Combined':<9}"
    else:
        header += f" {'Combined':<9}"
    print(header)
    print("-" * len(header))

    for entry in sorted(results, key=lambda r: r["layer"]):
        li = entry["layer"]
        tl = f"{entry['train_loss']:.4f}" if entry["train_loss"] is not None else "N/A"
        el = f"{entry['eval_loss']:.4f}" if entry["eval_loss"] is not None else "N/A"
        tr = train_ranks[li]
        er = eval_ranks[li]
        combined_score = sorted_combined[[l for l, _ in sorted_combined].index(li)][1]
        selected = " *" if li in top_k_layers else ""
        row = f"{li:<6} {tl:<12} {el:<12} {tr:<11} {er:<10}"
        if args.measure_sparsity:
            sp = f"{entry['sparsity']:.2f}%"
            row += f" {sp:<10} {combined_score:<9}"
        else:
            row += f" {combined_score:<9}"
        row += selected
        print(row)

    print("-" * len(header))
    print(f"\nSelected top-{args.top_k} layers: {sorted(top_k_layers)}")

    print("\n" + "-" * 50)
    print("Config snippet for partial ReLU-fication:")
    print("-" * 50)
    snippet = {
        "activation": {
            "type": config["activation"]["type"],
            "layer_mode": "partial",
            "layer_indices": sorted(top_k_layers),
        }
    }
    print(yaml.safe_dump(snippet, sort_keys=False, default_flow_style=False))

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": {
                "model_id": config["model_id"],
                "activation_type": config["activation"]["type"],
                "steps_per_layer": args.steps_per_layer,
                "top_k": args.top_k,
                "measure_sparsity": args.measure_sparsity,
            },
            "results": results,
            "ranking": {
                "combined": [{"layer": li, "score": sc} for li, sc in sorted_combined],
                "train_loss_ranks": train_ranks,
                "eval_loss_ranks": eval_ranks,
            },
            "selected_layers": sorted(top_k_layers),
            "config_snippet": snippet,
        }
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nResults saved to {output_path}")

    # Clean up progress file on successful completion
    if progress_file.exists():
        try:
            progress_file.unlink()
        except Exception:
            pass

if __name__ == "__main__":
    main()
