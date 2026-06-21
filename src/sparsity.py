import json
from pathlib import Path

import torch
from datasets import load_dataset

from .activations import get_model_layers


def _get_activation_modules(model):
    modules = []
    for index, layer in enumerate(get_model_layers(model)):
        mlp = getattr(layer, "mlp", None)
        if mlp is None or not hasattr(mlp, "act_fn"):
            continue
        modules.append((index, mlp.act_fn))
    return modules


def _get_model_device(model, requested_device):
    requested_device = torch.device(requested_device)
    try:
        current_device = next(model.parameters()).device
    except StopIteration:
        current_device = requested_device
    if current_device != requested_device:
        model.to(requested_device)
        return requested_device
    return current_device


def measure_prefill_sparsity(
    model,
    tokenizer,
    num_samples=32,
    batch_size=4,
    dataset_name="allenai/c4",
    dataset_config="en",
    dataset_split="train",
    text_column="text",
    block_size=1024,
    threshold=0.0,
    device="cuda",
):
    device = _get_model_device(model, device)
    modules = _get_activation_modules(model)
    layer_stats = {index: {"zeros": 0, "total": 0} for index, _ in modules}
    handles = []

    def hook_fn(layer_index, _module, _inputs, output):
        output_tensor = output[0] if isinstance(output, tuple) else output
        flat = output_tensor.detach().flatten()
        if threshold > 0:
            num_zeros = (flat.abs() <= threshold).sum().item()
        else:
            num_zeros = (flat == 0).sum().item()
        layer_stats[layer_index]["zeros"] += num_zeros
        layer_stats[layer_index]["total"] += flat.numel()

    for layer_index, module in modules:
        handles.append(module.register_forward_hook(lambda m, i, o, idx=layer_index: hook_fn(idx, m, i, o)))

    dataset = load_dataset(
        dataset_name,
        dataset_config,
        split=dataset_split,
        streaming=True,
    ).take(num_samples)

    batches = []
    for sample in dataset:
        text = sample.get(text_column, "")
        tokens = tokenizer(
            text,
            truncation=True,
            max_length=block_size,
            padding="max_length",
            return_tensors="pt",
        )
        batches.append(tokens["input_ids"][0])

    was_training = model.training
    model.eval()
    with torch.no_grad():
        for start in range(0, len(batches), batch_size):
            input_ids = torch.stack(batches[start : start + batch_size]).to(device)
            model(input_ids=input_ids)
    if was_training:
        model.train()

    for handle in handles:
        handle.remove()

    results = {}
    for layer_index, stats in layer_stats.items():
        total = stats["total"]
        results[layer_index] = 0.0 if total == 0 else (stats["zeros"] / total) * 100.0

    values = list(results.values())
    return {
        "mode": "prefill",
        "dataset": {
            "name": dataset_name,
            "config": dataset_config,
            "split": dataset_split,
            "text_column": text_column,
        },
        "num_samples": num_samples,
        "batch_size": batch_size,
        "block_size": block_size,
        "threshold": threshold,
        "results": results,
        "summary": {
            "measured_layers": len(results),
            "average_sparsity": 0.0 if not values else sum(values) / len(values),
        },
    }


def save_prefill_sparsity(output_path, payload):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path
