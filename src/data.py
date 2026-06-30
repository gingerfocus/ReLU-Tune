from pathlib import Path

from datasets import Dataset, load_dataset, load_from_disk
from transformers import AutoTokenizer


def load_tokenizer(model_id):
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def tokenize_and_pack(dataset, tokenizer, block_size):
    def tokenize_batch(examples):
        return tokenizer(examples["text"], max_length=131072, truncation=False)

    def group_texts(examples):
        concatenated = {key: sum(examples[key], []) for key in examples.keys()}
        total_length = len(concatenated["input_ids"])
        total_length = (total_length // block_size) * block_size
        result = {
            key: [tokens[i:i + block_size] for i in range(0, total_length, block_size)]
            for key, tokens in concatenated.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    remove_columns = dataset.column_names
    tokenized = dataset.map(
        tokenize_batch,
        batched=True,
        remove_columns=remove_columns,
    )
    packed = tokenized.map(group_texts, batched=True)
    return packed


def _cache_name(
    model_id,
    train_docs,
    validation_docs,
    validation_skip_docs,
    block_size,
    revision,
):
    model = model_id.replace("/", "_")
    rev = revision.replace("/", "_")
    return (
        f"{model}_refinedweb_train{train_docs}_val{validation_docs}"
        f"_skip{validation_skip_docs}_blk{block_size}_{rev}"
    )


def _materialize_refinedweb_region(revision, skip_docs, take_docs):
    stream = load_dataset(
        "tiiuae/falcon-refinedweb",
        split="train",
        streaming=True,
        revision=revision,
    )
    if skip_docs:
        stream = stream.skip(skip_docs)
    return Dataset.from_list(
        [{"text": sample["content"]} for sample in stream.take(take_docs)]
    )


def prepare_refinedweb_train_validation(
    model_id,
    tokenizer,
    train_docs,
    validation_docs,
    validation_skip_docs,
    block_size,
    revision,
    seed,
    cache_root,
):
    cache_root = Path(cache_root)
    cache_dir = cache_root / _cache_name(
        model_id,
        train_docs,
        validation_docs,
        validation_skip_docs,
        block_size,
        revision,
    )
    train_path = cache_dir / "train"
    validation_path = cache_dir / "validation"

    if train_path.exists() and (validation_docs == 0 or validation_path.exists()):
        train_dataset = load_from_disk(str(train_path))
        validation_dataset = (
            None if validation_docs == 0 else load_from_disk(str(validation_path))
        )
        return train_dataset, validation_dataset

    train_raw_dataset = _materialize_refinedweb_region(
        revision=revision,
        skip_docs=0,
        take_docs=train_docs,
    )
    train_dataset = tokenize_and_pack(
        train_raw_dataset,
        tokenizer,
        block_size=block_size,
    ).shuffle(seed=seed)

    validation_dataset = None
    if validation_docs > 0:
        validation_raw_dataset = _materialize_refinedweb_region(
            revision=revision,
            skip_docs=validation_skip_docs,
            take_docs=validation_docs,
        )
        validation_dataset = tokenize_and_pack(
            validation_raw_dataset,
            tokenizer,
            block_size=block_size,
        )

    cache_dir.mkdir(parents=True, exist_ok=True)
    train_dataset.save_to_disk(str(train_path))
    if validation_dataset is not None:
        validation_dataset.save_to_disk(str(validation_path))
    return train_dataset, validation_dataset


def slice_stage_dataset(train_dataset, start_index, length):
    end_index = min(start_index + length, len(train_dataset))
    if start_index >= end_index:
        raise ValueError(
            f"Requested empty training slice: start_index={start_index}, end_index={end_index}"
        )
    return train_dataset.select(range(start_index, end_index))
