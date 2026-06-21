import json
from pathlib import Path

import torch
import torch.nn as nn


class ReLU2(nn.Module):
    """Squared ReLU activation."""

    def forward(self, x):
        y = torch.relu(x)
        return y * y


def build_activation(name):
    normalized = name.lower()
    if normalized == "relu":
        return nn.ReLU()
    if normalized in {"relu2", "relu^2"}:
        return ReLU2()
    raise ValueError(f"Unsupported activation type: {name}")


def get_model_layers(model):
    candidates = [
        lambda m: m.model.layers,
        lambda m: m.base_model.model.layers,
        lambda m: m.base_model.model.model.layers,
        lambda m: m.model.model.layers,
    ]
    for candidate in candidates:
        try:
            return candidate(model)
        except AttributeError:
            continue
    raise ValueError("Model structure not recognized: could not locate transformer layers")


def apply_activation_to_mlp_layers(model, activation_name, layer_indices=None):
    layers = get_model_layers(model)
    if layer_indices is None:
        layer_indices = list(range(len(layers)))

    applied = []
    for index in layer_indices:
        if index < 0 or index >= len(layers):
            continue
        mlp = getattr(layers[index], "mlp", None)
        if mlp is None or not hasattr(mlp, "act_fn"):
            continue
        mlp.act_fn = build_activation(activation_name)
        applied.append(index)

    return model, sorted(applied)


def save_activation_config(output_dir, activation_name, layer_indices):
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)

    with (path / "activation_config.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "type": activation_name,
                "layer_indices": list(layer_indices),
            },
            handle,
            indent=2,
        )


def load_activation_config(config_path):
    with open(config_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload["type"], payload["layer_indices"]


def get_activation_summary(model):
    layers = get_model_layers(model)
    lines = []
    lines.append("-" * 50)
    lines.append(f"{'Layer':<10} | {'Activation':<30}")
    lines.append("-" * 50)
    for index, layer in enumerate(layers):
        mlp = getattr(layer, "mlp", None)
        if mlp is None or not hasattr(mlp, "act_fn"):
            continue
        lines.append(f"{index:<10} | {mlp.act_fn.__class__.__name__:<30}")
    lines.append("-" * 50)
    return "\n".join(lines)
