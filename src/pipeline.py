import gc
import json
from pathlib import Path
import shutil

import torch
import yaml
from transformers import DataCollatorForLanguageModeling, Trainer, TrainerCallback, TrainingArguments

from .activations import get_activation_summary, get_model_layers, save_activation_config
from .config import load_yaml
from .data import load_tokenizer, prepare_refinedweb_train_validation, slice_stage_dataset
from .lora import apply_lora, get_lora_target_modules
from .modeling import merge_stage_adapters, prepare_model_with_activation, save_merged_model, save_stage_metadata
from .runtime import (
    enable_transformers_checkpoint_resume_compat,
    get_num_devices,
    get_samples_per_step,
    save_runtime_metadata,
    set_seed,
    setup_logging,
)
from .sparsity import measure_prefill_sparsity, save_prefill_sparsity
from .staged_training import get_run_paths, get_stage_bounds, get_stage_paths
from .state import (
    build_initial_run_state,
    complete_active_stage,
    load_run_state,
    mark_checkpoint,
    save_run_state,
)


class RunStateCheckpointCallback(TrainerCallback):
    def __init__(self, state, state_path):
        self._state = state
        self._state_path = state_path

    def on_save(self, args, state, control, **kwargs):
        checkpoint_path = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        mark_checkpoint(self._state, checkpoint_path)
        save_run_state(self._state, self._state_path)


def _save_config_snapshot(config, path):
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)


def _maybe_init_wandb(config, stage_number):
    wandb_config = config.get("logging", {}).get("wandb", {})
    if not wandb_config.get("enabled", False):
        return None

    import wandb

    run_name = f"{config['run_name']}-stage-{stage_number}"
    group = wandb_config.get("group") or config["run_name"]
    return wandb.init(
        project=wandb_config["project"],
        entity=wandb_config.get("entity"),
        group=group,
        name=run_name,
        reinit=True,
    )


def _save_stage_metrics(path, log_history):
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(log_history, handle, indent=2)


def load_config_snapshot(run_dir):
    return load_yaml(Path(run_dir) / "config_snapshot.yaml")


def _initialize_run_state(config, run_dir):
    run_paths = get_run_paths(run_dir)
    state_path = run_paths["state"]
    if state_path.exists():
        return load_run_state(state_path)

    total_steps = config["training"]["total_steps"]
    stage_size = config["training"]["stage_size"]
    get_stage_bounds(total_steps, stage_size)
    state = build_initial_run_state(
        base_model=config["model_id"],
        activation_type=config["activation"]["type"],
        stage_size=stage_size,
        total_steps=total_steps,
        lora_rank=config["lora"]["r"],
    )
    save_run_state(state, state_path)
    return state


def _build_stage_dataset(config, train_dataset, stage_start_step, stage_end_step):
    samples_per_step = get_samples_per_step(
        config["training"]["batch_size"],
        config["training"]["gradient_accumulation_steps"],
    )
    start_index = stage_start_step * samples_per_step
    length = (stage_end_step - stage_start_step) * samples_per_step
    return slice_stage_dataset(train_dataset, start_index=start_index, length=length)


def _build_training_arguments(config, output_dir, stage_steps):
    training = config["training"]
    validation = config.get("validation", {})
    validation_enabled = validation.get("enabled", False)
    report_to = "none"
    if config.get("logging", {}).get("wandb", {}).get("enabled", False):
        report_to = "wandb"

    training_kwargs = dict(
        output_dir=str(output_dir),
        max_steps=stage_steps,
        per_device_train_batch_size=training["batch_size"],
        gradient_accumulation_steps=training["gradient_accumulation_steps"],
        learning_rate=float(training["learning_rate"]),
        warmup_steps=training["warmup_steps"],
        weight_decay=float(training["weight_decay"]),
        logging_steps=training["logging_steps"],
        save_steps=training["save_steps"],
        save_strategy="steps",
        save_total_limit=training.get("save_total_limit", 1),
        bf16=training["bf16"],
        gradient_checkpointing=training["gradient_checkpointing"],
        lr_scheduler_type=training["lr_scheduler_type"],
        optim=training["optimizer"],
        torch_compile=training.get("torch_compile", False),
        report_to=report_to,
        remove_unused_columns=False,
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        logging_first_step=True,
    )
    if validation_enabled:
        training_kwargs.update(
            eval_strategy="steps",
            eval_steps=validation.get("eval_steps", training["save_steps"]),
            per_device_eval_batch_size=validation.get("batch_size", training["batch_size"]),
        )
    return TrainingArguments(**training_kwargs)


def _prepare_stage_model(config, run_dir, completed_stages):
    model, layer_indices = prepare_model_with_activation(
        config["model_id"],
        config["activation"],
        output_dir=run_dir,
        use_bf16=config["training"]["bf16"],
    )
    if completed_stages:
        model = merge_stage_adapters(model, completed_stages)

    target_modules = get_lora_target_modules(
        model,
        layer_indices=layer_indices,
        include_mlp=config["lora"].get("include_mlp", True),
        include_attention=config["lora"].get("include_attention", True),
    )
    model = apply_lora(
        model,
        target_modules=target_modules,
        rank=config["lora"]["r"],
        alpha=config["lora"]["alpha"],
        dropout=config["lora"].get("dropout", 0.0),
        layers_to_transform=layer_indices,
    )
    save_activation_config(run_dir, config["activation"]["type"], layer_indices)
    return model, layer_indices


def _cleanup_model_objects(*objects):
    for obj in objects:
        if obj is not None:
            del obj
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _delete_stage_checkpoints(stage_root):
    stage_root = Path(stage_root)
    for checkpoint_dir in stage_root.glob("checkpoint-*"):
        if checkpoint_dir.is_dir():
            shutil.rmtree(checkpoint_dir)


def run_staged_training(config, run_dir):
    enable_transformers_checkpoint_resume_compat()
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(run_dir)

    run_paths = get_run_paths(run_dir)
    run_paths["logs"].mkdir(parents=True, exist_ok=True)
    run_paths["eval"].mkdir(parents=True, exist_ok=True)

    config_snapshot = run_dir / "config_snapshot.yaml"
    if not config_snapshot.exists():
        _save_config_snapshot(config, config_snapshot)

    save_runtime_metadata(
        run_dir / "run_metadata.json",
        extra={
            "model_id": config["model_id"],
            "activation": config["activation"]["type"],
            "total_steps": config["training"]["total_steps"],
            "stage_size": config["training"]["stage_size"],
        },
    )

    set_seed(config.get("seed", 42))
    state = _initialize_run_state(config, run_dir)

    print("=" * 70)
    print("ReLU-Tune staged training")
    print("=" * 70)
    print(f"Model: {config['model_id']}")
    print(f"Activation: {config['activation']['type']}")
    print(f"Total steps: {config['training']['total_steps']}")
    print(f"Stage size: {config['training']['stage_size']}")
    print(f"Num devices: {get_num_devices()}")
    print("=" * 70)

    tokenizer = load_tokenizer(config["model_id"])
    validation_config = config.get("validation", {})
    validation_enabled = validation_config.get("enabled", False)
    train_dataset, eval_dataset = prepare_refinedweb_train_validation(
        model_id=config["model_id"],
        tokenizer=tokenizer,
        train_docs=config["data"]["train_docs"],
        validation_docs=validation_config.get("docs", 0) if validation_enabled else 0,
        validation_skip_docs=validation_config.get("skip_docs", 0),
        block_size=config["data"]["block_size"],
        revision=config["data"]["revision"],
        seed=config.get("seed", 42),
        cache_root=config["data"]["cache_root"],
    )

    required_sequences = config["training"]["total_steps"] * get_samples_per_step(
        config["training"]["batch_size"],
        config["training"]["gradient_accumulation_steps"],
    )
    if len(train_dataset) < required_sequences:
        raise ValueError(
            "Packed train dataset too small for requested run: "
            f"{len(train_dataset):,} < {required_sequences:,}"
        )

    while state.active_stage is not None:
        stage = state.active_stage
        stage_steps = stage.end_step - stage.start_step
        stage_paths = get_stage_paths(run_dir, stage.stage)
        stage_paths["root"].mkdir(parents=True, exist_ok=True)

        print(f"\n[Stage {stage.stage}] {stage.start_step} -> {stage.end_step}")

        stage_train_dataset = _build_stage_dataset(
            config,
            train_dataset,
            stage_start_step=stage.start_step,
            stage_end_step=stage.end_step,
        )
        print(f"[Stage {stage.stage}] Train sequences: {len(stage_train_dataset):,}")

        model, layer_indices = _prepare_stage_model(
            config,
            run_dir,
            state.completed_stages,
        )
        print(get_activation_summary(model))
        training_args = _build_training_arguments(
            config,
            output_dir=stage_paths["root"],
            stage_steps=stage_steps,
        )
        wandb_run = _maybe_init_wandb(config, stage.stage)
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=stage_train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            callbacks=[RunStateCheckpointCallback(state, run_paths["state"])],
        )

        sparsity_config = config.get("measure_sparsity", {})
        if sparsity_config.get("enabled", False):
            sparsity_payload = measure_prefill_sparsity(
                model=trainer.model,
                tokenizer=tokenizer,
                num_samples=sparsity_config.get("num_samples", 32),
                batch_size=sparsity_config.get("batch_size", 4),
                dataset_name=sparsity_config.get("dataset", "allenai/c4"),
                dataset_config=sparsity_config.get("dataset_config", "en"),
                dataset_split=sparsity_config.get("dataset_split", "train"),
                text_column=sparsity_config.get("text_column", "text"),
                block_size=config["data"]["block_size"],
                threshold=sparsity_config.get("threshold", 0.0),
                device=training_args.device,
            )
            sparsity_payload["stage"] = stage.stage
            save_prefill_sparsity(stage_paths["root"] / "prefill_sparsity.json", sparsity_payload)
            print(
                f"[Stage {stage.stage}] Prefill sparsity avg: "
                f"{sparsity_payload['summary']['average_sparsity']:.2f}%"
            )

        resume_from_checkpoint = None
        if stage.latest_checkpoint is not None and Path(stage.latest_checkpoint).exists():
            resume_from_checkpoint = stage.latest_checkpoint
            print(f"[Stage {stage.stage}] Resuming from {resume_from_checkpoint}")
        else:
            print(f"[Stage {stage.stage}] Starting fresh")

        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        trainer.save_model(stage_paths["adapter"])
        _save_stage_metrics(stage_paths["root"] / "metrics.json", trainer.state.log_history)
        save_stage_metadata(
            stage_paths["root"] / "stage_metadata.json",
            {
                "stage": stage.stage,
                "global_start_step": stage.start_step,
                "global_end_step": stage.end_step,
                "stage_steps": stage_steps,
                "activation_type": config["activation"]["type"],
                "activation_layers": layer_indices,
            },
        )
        complete_active_stage(state, stage_paths["adapter"])
        save_run_state(state, run_paths["state"])
        print(f"[Stage {stage.stage}] Saved adapter to {stage_paths['adapter']}")

        if wandb_run is not None:
            wandb_run.finish()
        _delete_stage_checkpoints(stage_paths["root"])
        _cleanup_model_objects(trainer, model)

    print("\n[Final] Merging completed stage adapters into one dense model")
    final_model, final_layer_indices = prepare_model_with_activation(
        config["model_id"],
        config["activation"],
        output_dir=run_dir,
        use_bf16=config["training"]["bf16"],
    )
    final_model = merge_stage_adapters(final_model, state.completed_stages)
    save_merged_model(
        final_model,
        tokenizer,
        run_dir / "final_merged",
        activation_type=config["activation"]["type"],
        layer_indices=final_layer_indices,
    )
    _cleanup_model_objects(final_model)
    print(f"[Final] Saved merged model to {run_dir / 'final_merged'}")
