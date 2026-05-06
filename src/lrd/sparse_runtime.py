import json
import os

import torch
from transformers import AutoConfig, AutoTokenizer

from upstream_runtime import ensure_scaling_retriever_importable

ensure_scaling_retriever_importable()

from scaling_retriever.dataset.data_collator import LlamaSparseCollectionCollator, T5SparseCollectionCollator
from scaling_retriever.dataset.dataset import CollectionDataset
from scaling_retriever.modeling.llm_encoder import BertSparse, LlamaBiSparse, T5Sparse


def resolve_device(device_arg: str | None) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def infer_model_type(model_name_or_path: str, lora_name_or_path: str | None):
    config_source = lora_name_or_path if lora_name_or_path else model_name_or_path
    if lora_name_or_path and os.path.isdir(lora_name_or_path):
        adapter_config_path = os.path.join(lora_name_or_path, "adapter_config.json")
        if os.path.exists(adapter_config_path):
            with open(adapter_config_path, "r", encoding="utf-8") as fin:
                adapter_config = json.load(fin)
            config_source = adapter_config["base_model_name_or_path"]
    return AutoConfig.from_pretrained(config_source).model_type


def infer_model_class(model_name_or_path: str, lora_name_or_path: str | None):
    model_type = infer_model_type(model_name_or_path, lora_name_or_path)
    if model_type == "llama":
        return LlamaBiSparse
    if model_type == "bert":
        return BertSparse
    if model_type == "t5":
        return T5Sparse
    raise ValueError(f"Unsupported sparse model_type={model_type!r}")


def load_sparse_model(model_name_or_path: str, lora_name_or_path: str | None, device: torch.device):
    model_cls = infer_model_class(model_name_or_path, lora_name_or_path)
    if lora_name_or_path:
        model = model_cls.load_from_lora(lora_name_or_path)
    else:
        model = model_cls.load(model_name_or_path)
    model.to(device)
    model.eval()
    return model


def load_tokenizer(model_name_or_path: str):
    return AutoTokenizer.from_pretrained(model_name_or_path)


def build_collection_collator(model_name_or_path: str, lora_name_or_path: str | None, tokenizer, max_length: int):
    model_type = infer_model_type(model_name_or_path, lora_name_or_path)
    if model_type == "t5":
        return T5SparseCollectionCollator(tokenizer=tokenizer, max_length=max_length)
    return LlamaSparseCollectionCollator(tokenizer=tokenizer, max_length=max_length)


def load_collection_dataset(corpus_path: str, data_source: str):
    return CollectionDataset(corpus_path, data_source=data_source)


def encode_query_batch(model, tokenizer, queries: list[str], query_max_length: int, device: torch.device):
    tokenized = tokenizer(
        queries,
        max_length=query_max_length,
        truncation=True,
        padding="longest",
        pad_to_multiple_of=8,
        return_tensors="pt",
    )
    tokenized = {key: value.to(device) for key, value in tokenized.items()}
    with torch.inference_mode():
        query_reps = model.query_encode(**tokenized)
    return query_reps, tokenized
