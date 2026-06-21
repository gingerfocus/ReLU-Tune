import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM

from .activations import apply_activation_to_mlp_layers, get_model_layers, save_activation_config


def _resolve_dtype(use_bf16):
    return torch.bfloat16 if use_bf16 else torch.float32


def _resolve_activation_layer_indices(model, activation_config):
    layer_mode = activation_config.get("layer_mode", "all")
    num_layers = len(get_model_layers(model))
    if layer_mode == "all":
        return list(range(num_layers))
    if layer_mode == "partial":
        layer_indices = activation_config.get("layer_indices")
        if not layer_indices:
            raise ValueError("activation.layer_indices is required when layer_mode='partial'")
        return list(layer_indices)
    raise ValueError(f"Unsupported layer_mode: {layer_mode}")


def load_base_model(model_id, use_bf16=True):
    try:
        return AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=_resolve_dtype(use_bf16),
            attn_implementation="flash_attention_2",
        )
    except Exception as exc:
        print(
            f"[Model Load] flash_attention_2 unavailable for {model_id}; "
            f"falling back to eager attention. Original error: {exc}"
        )
        return AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=_resolve_dtype(use_bf16),
            attn_implementation="eager",
        )


def prepare_model_with_activation(model_id, activation_config, output_dir=None, use_bf16=True):
    model = load_base_model(model_id, use_bf16=use_bf16)
    layer_indices = _resolve_activation_layer_indices(model, activation_config)
    model, layer_indices = apply_activation_to_mlp_layers(
        model,
        activation_name=activation_config["type"],
        layer_indices=layer_indices,
    )
    if output_dir is not None:
        save_activation_config(output_dir, activation_config["type"], layer_indices)
    return model, layer_indices


def merge_adapter_into_model(model, adapter_path):
    peft_model = PeftModel.from_pretrained(model, adapter_path, is_trainable=False)
    return peft_model.merge_and_unload()


def merge_stage_adapters(model, completed_stages):
    current = model
    for stage in completed_stages:
        print(f"Merged adapter from {stage.adapter_path} into model")
        current = merge_adapter_into_model(current, stage.adapter_path)
    return current


def save_merged_model(model, tokenizer, output_dir, activation_type=None, layer_indices=None):
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    if activation_type is not None and layer_indices is not None:
        save_activation_config(path, activation_type, layer_indices)


def save_stage_metadata(output_dir, payload):
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    with (path / "stage_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
