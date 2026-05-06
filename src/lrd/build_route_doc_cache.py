import argparse
import os
import sys

import scipy.sparse as sp
import torch
from torch.utils.data import DataLoader, Subset


REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from doc_cache import SparseDocCacheWriter, build_literal_mask
from sparse_runtime import (
    build_collection_collator,
    load_collection_dataset,
    load_sparse_model,
    load_tokenizer,
    resolve_device,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Build the literal/expansion document cache used by the LRD diagnostic.")
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--lora_name_or_path", default=None)
    parser.add_argument("--corpus_path", required=True)
    parser.add_argument("--corpus_format", choices=("msmarco", "wiki"), default="msmarco")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--doc_max_length", type=int, default=192)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--docs_per_part", type=int, default=50000)
    parser.add_argument("--max_docs", type=int, default=0)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def maybe_limit_dataset(dataset, max_docs: int):
    if int(max_docs) <= 0:
        return dataset
    limit = min(len(dataset), int(max_docs))
    return Subset(dataset, range(limit))


def main():
    args = parse_args()
    device = resolve_device(args.device)
    model = load_sparse_model(args.model_name_or_path, args.lora_name_or_path, device=device)
    tokenizer = load_tokenizer(args.model_name_or_path)
    dataset = load_collection_dataset(args.corpus_path, data_source=args.corpus_format)
    dataset = maybe_limit_dataset(dataset, args.max_docs)
    collator = build_collection_collator(
        args.model_name_or_path,
        args.lora_name_or_path,
        tokenizer=tokenizer,
        max_length=args.doc_max_length,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
    )
    writer = SparseDocCacheWriter(
        cache_dir=args.output_dir,
        rep_names=("literal", "expansion"),
        docs_per_part=args.docs_per_part,
        meta={
            "model_name_or_path": args.model_name_or_path,
            "lora_name_or_path": args.lora_name_or_path,
            "corpus_path": args.corpus_path,
            "corpus_format": args.corpus_format,
            "doc_max_length": int(args.doc_max_length),
        },
    )

    use_autocast = device.type == "cuda" and torch.cuda.is_bf16_supported()
    autocast_dtype = torch.bfloat16 if use_autocast else None

    with torch.inference_mode():
        for batch in loader:
            doc_ids = batch["ids"]
            inputs = {key: value.to(device) for key, value in batch.items() if key != "ids"}
            if use_autocast:
                with torch.amp.autocast("cuda", dtype=autocast_dtype):
                    doc_reps = model.doc_encode(**inputs)
            else:
                doc_reps = model.doc_encode(**inputs)
            literal_mask = build_literal_mask(
                inputs["input_ids"],
                inputs["attention_mask"],
                doc_reps.shape[-1],
            )
            literal_reps = (doc_reps * literal_mask).detach().float().cpu().numpy()
            expansion_reps = (doc_reps * (1.0 - literal_mask)).detach().float().cpu().numpy()
            writer.add_batch(
                doc_ids,
                {
                    "literal": sp.csr_matrix(literal_reps.astype("float16", copy=False)),
                    "expansion": sp.csr_matrix(expansion_reps.astype("float16", copy=False)),
                },
            )

    writer.finalize()


if __name__ == "__main__":
    main()
