import argparse
import json
import os
import sys
from collections import defaultdict

import torch


REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from doc_cache import SparseDocCache, csr_to_padded_tensors
from route_decomposition import decompose_sparse_matching
from sparse_runtime import encode_query_batch, load_sparse_model, load_tokenizer, resolve_device


def parse_args():
    parser = argparse.ArgumentParser(description="Run the LRD four-route diagnostic on a fixed candidate set.")
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--lora_name_or_path", default=None)
    parser.add_argument("--query_path", required=True, help="TSV with: qid<TAB>query")
    parser.add_argument(
        "--candidate_path",
        required=True,
        help="Either a TREC run file or JSONL with query_id/doc_ids.",
    )
    parser.add_argument("--candidate_format", choices=("trec", "jsonl"), default="trec")
    parser.add_argument("--doc_cache_dir", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--query_max_length", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--topk", type=int, default=100)
    parser.add_argument("--device", default=None)
    parser.add_argument("--ll", type=float, default=1.0)
    parser.add_argument("--le", type=float, default=1.0)
    parser.add_argument("--el", type=float, default=1.0)
    parser.add_argument("--ee", type=float, default=1.0)
    return parser.parse_args()


def load_queries(query_path: str) -> dict[str, str]:
    queries = {}
    with open(query_path, "r", encoding="utf-8") as fin:
        for line in fin:
            line = line.rstrip("\n")
            if not line:
                continue
            qid, query = line.split("\t", 1)
            queries[str(qid)] = query
    return queries


def load_candidates_trec(candidate_path: str, topk: int) -> dict[str, list[str]]:
    grouped = defaultdict(list)
    with open(candidate_path, "r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 6:
                raise ValueError(f"Invalid TREC run line: {line}")
            qid, _, doc_id, rank, *_rest = parts
            if int(rank) <= topk:
                grouped[str(qid)].append(str(doc_id))
    return dict(grouped)


def load_candidates_jsonl(candidate_path: str, topk: int) -> dict[str, list[str]]:
    grouped = {}
    with open(candidate_path, "r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            qid = str(record["query_id"])
            if "doc_ids" in record:
                doc_ids = [str(doc_id) for doc_id in record["doc_ids"][:topk]]
            elif "docs" in record:
                doc_ids = [str(item["doc_id"]) for item in record["docs"][:topk]]
            else:
                raise ValueError("JSONL candidate records must include doc_ids or docs")
            grouped[qid] = doc_ids
    return grouped


def load_candidates(candidate_path: str, candidate_format: str, topk: int) -> dict[str, list[str]]:
    if candidate_format == "trec":
        return load_candidates_trec(candidate_path, topk=topk)
    return load_candidates_jsonl(candidate_path, topk=topk)


def chunked(items: list[tuple[str, str]], batch_size: int):
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def main():
    args = parse_args()
    device = resolve_device(args.device)
    queries = load_queries(args.query_path)
    candidates = load_candidates(args.candidate_path, args.candidate_format, topk=args.topk)
    model = load_sparse_model(args.model_name_or_path, args.lora_name_or_path, device=device)
    tokenizer = load_tokenizer(args.model_name_or_path)
    doc_cache = SparseDocCache(args.doc_cache_dir, rep_names=("literal", "expansion"))
    route_weights = {
        "ll": float(args.ll),
        "le": float(args.le),
        "el": float(args.el),
        "ee": float(args.ee),
    }

    records = [(qid, query) for qid, query in queries.items() if qid in candidates and candidates[qid]]
    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)

    with open(args.output_path, "w", encoding="utf-8") as fout:
        for batch in chunked(records, batch_size=max(1, int(args.batch_size))):
            batch_qids = [qid for qid, _ in batch]
            batch_queries = [query for _, query in batch]
            query_reps, tokenized = encode_query_batch(
                model,
                tokenizer,
                batch_queries,
                query_max_length=args.query_max_length,
                device=device,
            )

            for batch_idx, qid in enumerate(batch_qids):
                doc_ids = candidates[qid]
                doc_matrices = doc_cache.fetch(doc_ids)
                literal_indices, literal_values, literal_mask = csr_to_padded_tensors(doc_matrices["literal"])
                expansion_indices, expansion_values, expansion_mask = csr_to_padded_tensors(doc_matrices["expansion"])
                bundle = decompose_sparse_matching(
                    query_reps=query_reps[batch_idx : batch_idx + 1],
                    query_input_ids=tokenized["input_ids"][batch_idx : batch_idx + 1],
                    query_attention_mask=tokenized["attention_mask"][batch_idx : batch_idx + 1],
                    doc_literal_indices=literal_indices,
                    doc_literal_values=literal_values,
                    doc_literal_mask=literal_mask,
                    doc_expansion_indices=expansion_indices,
                    doc_expansion_values=expansion_values,
                    doc_expansion_mask=expansion_mask,
                )
                weighted = bundle.weighted_score(route_weights).squeeze(0).tolist()
                ll_scores = bundle.ll.squeeze(0).tolist()
                le_scores = bundle.le.squeeze(0).tolist()
                el_scores = bundle.el.squeeze(0).tolist()
                ee_scores = bundle.ee.squeeze(0).tolist()
                docs = []
                for idx, doc_id in enumerate(doc_ids):
                    docs.append(
                        {
                            "doc_id": str(doc_id),
                            "weighted_score": float(weighted[idx]),
                            "ll": float(ll_scores[idx]),
                            "le": float(le_scores[idx]),
                            "el": float(el_scores[idx]),
                            "ee": float(ee_scores[idx]),
                        }
                    )
                docs.sort(key=lambda item: item["weighted_score"], reverse=True)
                fout.write(
                    json.dumps(
                        {
                            "query_id": qid,
                            "query": queries[qid],
                            "route_weights": route_weights,
                            "docs": docs,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )


if __name__ == "__main__":
    main()
