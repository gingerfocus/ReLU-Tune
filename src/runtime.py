import atexit
import json
import os
import platform
import sys
from importlib import metadata
from pathlib import Path

import torch


class Tee:
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for handle in self.files:
            handle.write(obj)
            handle.flush()

    def flush(self):
        for handle in self.files:
            handle.flush()

    def isatty(self):
        return getattr(self.files[0], "isatty", lambda: False)()

    def close(self):
        for handle in self.files[1:]:
            try:
                handle.close()
            except Exception:
                pass


def setup_logging(run_dir, filename="train.log"):
    log_dir = Path(run_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / filename
    if not isinstance(sys.stdout, Tee):
        handle = open(log_path, "a", buffering=1, encoding="utf-8")
        atexit.register(handle.close)
        sys.stdout = Tee(sys.stdout, handle)
        sys.stderr = Tee(sys.stderr, handle)
    return log_path


def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_num_devices():
    return max(1, torch.cuda.device_count())


def get_samples_per_step(batch_size, gradient_accumulation_steps):
    return batch_size * gradient_accumulation_steps * get_num_devices()


def _package_version(name):
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def collect_runtime_metadata(extra=None):
    payload = {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": {
            "torch": _package_version("torch"),
            "transformers": _package_version("transformers"),
            "peft": _package_version("peft"),
            "accelerate": _package_version("accelerate"),
            "datasets": _package_version("datasets"),
            "lm_eval": _package_version("lm_eval") or _package_version("lm-eval"),
            "flash_attn": _package_version("flash-attn"),
        },
        "torch": {
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "bf16_supported": torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False,
            "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        },
    }
    if torch.cuda.is_available():
        payload["torch"]["devices"] = [
            {
                "index": index,
                "name": torch.cuda.get_device_name(index),
            }
            for index in range(torch.cuda.device_count())
        ]
    if extra:
        payload["extra"] = extra
    return payload


def save_runtime_metadata(output_path, extra=None):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = collect_runtime_metadata(extra=extra)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path


_TRANSFORMERS_RESUME_COMPAT_LOGGED = False


def enable_transformers_checkpoint_resume_compat():
    global _TRANSFORMERS_RESUME_COMPAT_LOGGED

    if not _TRANSFORMERS_RESUME_COMPAT_LOGGED:
        print(
            "[Runtime] Enabling Transformers checkpoint resume compatibility. "
            "This relaxes safe deserialization checks so local training checkpoints "
            "can be resumed. Only resume checkpoints you trust."
        )
        _TRANSFORMERS_RESUME_COMPAT_LOGGED = True

    os.environ["TRANSFORMERS_ALLOW_UNSAFE_DESERIALIZATION"] = "1"
    import transformers.utils
    import transformers.utils.import_utils

    transformers.utils.import_utils.check_torch_load_is_safe = lambda: None
    if hasattr(transformers.utils, "check_torch_load_is_safe"):
        transformers.utils.check_torch_load_is_safe = lambda: None
