from peft import LoraConfig, TaskType, get_peft_model

from .activations import get_model_layers


def get_lora_target_modules(model, layer_indices=None, include_mlp=True, include_attention=True):
    layers = get_model_layers(model)
    if layer_indices is None:
        layer_indices = list(range(len(layers)))

    targets = []
    for index in layer_indices:
        if include_mlp:
            targets.extend(
                [
                    f"model.layers.{index}.mlp.gate_proj",
                    f"model.layers.{index}.mlp.up_proj",
                    f"model.layers.{index}.mlp.down_proj",
                ]
            )
        if include_attention:
            targets.extend(
                [
                    f"model.layers.{index}.self_attn.q_proj",
                    f"model.layers.{index}.self_attn.k_proj",
                    f"model.layers.{index}.self_attn.v_proj",
                    f"model.layers.{index}.self_attn.o_proj",
                ]
            )
    return targets


def apply_lora(model, rank, alpha, dropout, target_modules):
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=target_modules,
        bias="none",
    )
    return get_peft_model(model, config)
