from copy import deepcopy
from pathlib import Path

import yaml


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def deep_merge(base, override):
    result = deepcopy(base)
    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def load_merged_config(config_paths):
    if not config_paths:
        raise ValueError("At least one config path is required")

    merged = None
    for config_path in config_paths:
        payload = load_yaml(config_path)
        merged = payload if merged is None else deep_merge(merged, payload)
    return merged


def resolve_run_dir(config, run_dir=None):
    if run_dir is not None:
        return Path(run_dir)
    output_root = config.get("output_root", "./runs")
    run_name = config.get("run_name")
    if not run_name:
        raise ValueError("Config must define run_name when run_dir is not provided")
    return Path(output_root) / run_name
