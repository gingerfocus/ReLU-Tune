import json
from datetime import datetime
from pathlib import Path

import torch
from datasets import load_dataset

from .activations import (
    apply_activation_to_mlp_layers,
    get_activation_summary,
    load_activation_config,
)
from .data import load_tokenizer
from .modeling import load_base_model
from .runtime import collect_runtime_metadata

BENCHMARK_ALIASES = {
    "piqa": "piqa",
    "openbookqa": "openbookqa",
    "obqa": "openbookqa",
    "sciq": "sciq",
    "winogrande": "winogrande",
    "hellaswag": "hellaswag",
    "arc_easy": "arc_easy",
    "arc_challenge": "arc_challenge",
    "boolq": "boolq",
    "lambada": "lambada_openai",
    "lambada_openai": "lambada_openai",
}

BENCHMARK_METRICS = {
    "piqa": ("acc", "acc,none"),
    "openbookqa": ("acc_norm", "acc_norm,none"),
    "sciq": ("acc", "acc,none"),
    "winogrande": ("acc", "acc,none"),
    "hellaswag": ("acc_norm", "acc_norm,none"),
    "arc_easy": ("acc", "acc,none"),
    "arc_challenge": ("acc_norm", "acc_norm,none"),
    "boolq": ("acc", "acc,none"),
    "lambada_openai": ("acc", "acc,none"),
}

PERPLEXITY_DATASETS = {
    "wikitext2": {
        "path": "Salesforce/wikitext",
        "name": "wikitext-2-raw-v1",
        "split": "test",
        "text_column": "text",
        "streaming": False,
        "num_samples": -1,
    },
    "refinedweb": {
        "path": "tiiuae/falcon-refinedweb",
        "name": None,
        "split": "train",
        "text_column": "content",
        "streaming": True,
        "revision": "c735840",
        "skip_docs": 200_000,
        "num_samples": 500,
    },
}


def is_lora_adapter(path):
    return (Path(path) / "adapter_config.json").exists()


def _read_adapter_base_model(adapter_path):
    with open(Path(adapter_path) / "adapter_config.json", "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload["base_model_name_or_path"]


def _resolve_activation_config_path(model_path):
    path = Path(model_path)
    candidates = [
        path / "activation_config.json",
        path.parent / "activation_config.json",
        path.parent.parent / "activation_config.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _load_activation_payload(model_path):
    config_path = _resolve_activation_config_path(model_path)
    if config_path is None:
        return None
    activation_type, layer_indices = load_activation_config(config_path)
    return {
        "type": activation_type,
        "layer_indices": layer_indices,
    }


def _load_adapter_model(adapter_path, use_bf16):
    from peft import PeftModel

    base_model_id = _read_adapter_base_model(adapter_path)
    model = load_base_model(base_model_id, use_bf16=use_bf16)
    activation_payload = _load_activation_payload(adapter_path)
    if activation_payload is not None:
        model, _ = apply_activation_to_mlp_layers(
        model,
        activation_payload["type"],
        activation_payload["layer_indices"],
    )
    model = PeftModel.from_pretrained(model, adapter_path, is_trainable=False)
    tokenizer = load_tokenizer(base_model_id)
    return model, tokenizer, {
        "kind": "adapter",
        "base_model": base_model_id,
        "activation": activation_payload,
    }


def _load_full_model(model_path, use_bf16):
    model = load_base_model(model_path, use_bf16=use_bf16)
    activation_payload = _load_activation_payload(model_path)
    if activation_payload is not None:
        model, _ = apply_activation_to_mlp_layers(
            model,
            activation_payload["type"],
            activation_payload["layer_indices"],
        )
    tokenizer = load_tokenizer(model_path)
    return model, tokenizer, {
        "kind": "full",
        "base_model": model_path,
        "activation": activation_payload,
    }


def load_model_for_evaluation(model_path=None, base_model=None, activation_type=None, use_bf16=True):
    if model_path:
        path = Path(model_path)
        if (path / "run_state.json").exists():
            if (path / "final_merged").exists():
                path = path / "final_merged"
            else:
                raise ValueError(
                    f"{path} looks like a run directory but no final_merged export exists yet. "
                    f"Run merge_stage_adapters.py first or point evaluation at a specific adapter directory."
                )
        if is_lora_adapter(path):
            model, tokenizer, metadata = _load_adapter_model(path, use_bf16=use_bf16)
            description = f"Adapter: {path}"
        else:
            model, tokenizer, metadata = _load_full_model(path, use_bf16=use_bf16)
            description = f"Full: {path}"
        return model, tokenizer, description, metadata

    if not base_model:
        raise ValueError("Either model_path or base_model must be provided")

    model = load_base_model(base_model, use_bf16=use_bf16)
    tokenizer = load_tokenizer(base_model)
    activation_payload = None
    description = f"Base: {base_model}"
    if activation_type:
        model, layer_indices = apply_activation_to_mlp_layers(model, activation_type)
        activation_payload = {"type": activation_type, "layer_indices": layer_indices}
        description = f"{activation_type.upper()}: {base_model}"
    return model, tokenizer, description, {
        "kind": "base",
        "base_model": base_model,
        "activation": activation_payload,
    }


def get_default_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def move_model_to_device(model, device):
    return model.to(device)


def describe_model(model):
    print("")
    print("=" * 60)
    print("Model summary")
    print("=" * 60)
    print(get_activation_summary(model))
    print("=" * 60)


def evaluate_benchmarks(model, benchmarks, batch_size="auto", num_fewshot=0, device="cuda"):
    import lm_eval
    from lm_eval.models.huggingface import HFLM

    tasks = [BENCHMARK_ALIASES.get(name.lower(), name.lower()) for name in benchmarks]
    wrapper = HFLM(pretrained=model, batch_size=batch_size, device=device)
    raw = lm_eval.simple_evaluate(
        model=wrapper,
        tasks=tasks,
        num_fewshot=num_fewshot,
        batch_size=batch_size,
        device=device,
    )
    formatted = {}
    summary = {}
    for task in tasks:
        task_results = raw["results"].get(task, {})
        formatted[task] = task_results
        metric_name, metric_key = BENCHMARK_METRICS.get(task, ("acc", "acc,none"))
        value = task_results.get(metric_key, task_results.get(metric_name))
        summary[task] = value
    return {"results": formatted, "summary": summary, "raw": raw}


def _load_perplexity_texts(dataset_key, num_samples=None, dataset_overrides=None):
    config = dict(PERPLEXITY_DATASETS[dataset_key])
    if dataset_overrides:
        config.update(dataset_overrides.get(dataset_key, {}))
    if dataset_key == "refinedweb":
        dataset = load_dataset(
            config["path"],
            split=config["split"],
            streaming=True,
            revision=config["revision"],
        )
        texts = []
        for index, sample in enumerate(dataset.skip(config["skip_docs"])):
            if num_samples is not None and num_samples >= 0 and index >= num_samples:
                break
            texts.append(sample[config["text_column"]])
        return texts

    dataset = load_dataset(
        config["path"],
        config["name"],
        split=config["split"],
    )
    if num_samples is not None and num_samples >= 0:
        dataset = dataset.select(range(min(num_samples, len(dataset))))
    return [row[config["text_column"]] for row in dataset]


def _tokenize_and_chunk(texts, tokenizer, seq_len):
    token_ids = []
    for text in texts:
        if not text or not text.strip():
            continue
        token_ids.extend(tokenizer.encode(text, add_special_tokens=False))

    chunk_count = len(token_ids) // seq_len
    if chunk_count == 0:
        return []

    token_ids = token_ids[: chunk_count * seq_len]
    return [token_ids[index:index + seq_len] for index in range(0, len(token_ids), seq_len)]


def compute_perplexity(
    model,
    tokenizer,
    datasets=None,
    seq_len=1024,
    batch_size=4,
    dataset_overrides=None,
    device="cuda",
):
    datasets = datasets or ["wikitext2", "refinedweb"]
    model.eval()
    results = {}
    for dataset_key in datasets:
        dataset_config = (dataset_overrides or {}).get(dataset_key, {})
        dataset_num_samples = dataset_config.get(
            "num_samples",
            PERPLEXITY_DATASETS[dataset_key].get("num_samples"),
        )
        texts = _load_perplexity_texts(
            dataset_key,
            dataset_num_samples,
            dataset_overrides=dataset_overrides,
        )
        chunks = _tokenize_and_chunk(texts, tokenizer, seq_len)
        if not chunks:
            results[dataset_key] = {
                "perplexity": None,
                "avg_loss": None,
                "num_tokens": 0,
                "num_chunks": 0,
                "seq_len": seq_len,
            }
            continue

        total_loss = 0.0
        total_tokens = 0
        for start_index in range(0, len(chunks), batch_size):
            batch_chunks = chunks[start_index:start_index + batch_size]
            input_ids = torch.tensor(batch_chunks, device=device)
            with torch.no_grad():
                outputs = model(input_ids=input_ids, labels=input_ids)
            token_count = input_ids.shape[0] * (input_ids.shape[1] - 1)
            total_loss += outputs.loss.item() * token_count
            total_tokens += token_count

        average_loss = total_loss / total_tokens
        results[dataset_key] = {
            "perplexity": torch.exp(torch.tensor(average_loss)).item(),
            "avg_loss": average_loss,
            "num_tokens": total_tokens,
            "num_chunks": len(chunks),
            "seq_len": seq_len,
        }
    return results


def print_results_table(summary):
    lines = []
    lines.append("")
    lines.append("=" * 40)
    lines.append(f"{'Benchmark':<20} | {'Accuracy':>15}")
    lines.append("-" * 40)
    for benchmark, value in summary.items():
        if isinstance(value, float):
            lines.append(f"{benchmark:<20} | {value:>14.4f}")
        else:
            lines.append(f"{benchmark:<20} | {str(value):>15}")
    lines.append("=" * 40)
    table = "\n".join(lines)
    print(table)
    return table


def print_perplexity_results(results):
    lines = []
    lines.append("")
    lines.append("=" * 50)
    lines.append(f"{'Dataset':<20} | {'Perplexity':>12} | {'Avg Loss':>10}")
    lines.append("-" * 50)
    for dataset_key, payload in results.items():
        if payload["perplexity"] is None:
            lines.append(f"{dataset_key:<20} | {'n/a':>12} | {'n/a':>10}")
        else:
            lines.append(
                f"{dataset_key:<20} | {payload['perplexity']:>12.2f} | {payload['avg_loss']:>10.4f}"
            )
    lines.append("=" * 50)
    table = "\n".join(lines)
    print(table)
    return table


def _json_safe(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (torch.dtype, torch.device)):
        return str(value)
    if callable(value):
        name = getattr(value, "__name__", type(value).__name__)
        return f"<callable:{name}>"
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    return repr(value)


def save_evaluation_results(
    output_dir,
    model_description,
    metadata,
    benchmark_results,
    perplexity_results,
    runtime_metadata,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = output_dir / f"evaluation_{timestamp}.json"
    payload = {
        "model_description": model_description,
        "metadata": _json_safe(metadata),
        "benchmarks": _json_safe(benchmark_results),
        "perplexity": _json_safe(perplexity_results),
        "runtime": _json_safe(runtime_metadata),
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return path
