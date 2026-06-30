from peft import LoraConfig, TaskType, get_peft_model

from .activations import get_model_layers


def get_lora_target_modules(model=None, layer_indices=None, include_mlp=True, include_attention=True):
    targets = []
    if include_mlp:
        targets.extend(["gate_proj", "up_proj", "down_proj"])
    if include_attention:
        targets.extend(["q_proj", "k_proj", "v_proj", "o_proj"])
    return targets


def apply_lora(model, rank, alpha, dropout, target_modules, layers_to_transform=None):
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=target_modules,
        layers_to_transform=layers_to_transform,
        bias="none",
    )
    return get_peft_model(model, config)
