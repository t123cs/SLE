#!/usr/bin/env python3
"""Build hard-query text and soft trajectory evidence for single-index SDE."""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from src.sde import terrier_utils as terrier  # noqa: E402
from src.sde.single_index import trace  # noqa: E402


DATASETS = ("nfcorpus", "scidocs", "fiqa-2018", "arguana", "scifact")


def update_evidence(
    entry: list[Any],
    probability: float,
    template_bit: int,
) -> None:
    probability = min(max(probability, 0.0), 1.0)
    entry[0] += probability
    entry[1] += math.log1p(-min(probability, 1.0 - 1e-12))
    entry[2] = max(entry[2], probability)
    entry[3] += 1
    entry[4] |= template_bit


def validate_trace_coverage(
    doc_ids: list[str],
    prompt_ids: set[str],
    observed_keys: dict[str, set[tuple[str, int]]],
    expected_templates: int,
    sample_idx_max: int,
) -> None:
    if len(prompt_ids) != expected_templates:
        raise RuntimeError(
            "Trace template count mismatch: "
            f"expected {expected_templates}, found {len(prompt_ids)} "
            f"({', '.join(sorted(prompt_ids)) or 'none'})."
        )
    expected_keys = {
        (prompt_id, sample_idx)
        for prompt_id in prompt_ids
        for sample_idx in range(sample_idx_max + 1)
    }
    incomplete: list[str] = []
    for doc_id in doc_ids:
        if observed_keys.get(doc_id, set()) != expected_keys:
            incomplete.append(doc_id)
            if len(incomplete) >= 10:
                break
    if incomplete:
        raise RuntimeError(
            "Every corpus document must have the complete retained template/sample "
            f"set. First incomplete document IDs: {', '.join(incomplete)}"
        )


def build_dataset(
    args: argparse.Namespace,
    dataset: str,
    tokenizer: Any | None,
    tokeniser: Any,
    stemmer: Any,
    stopwords: set[str],
) -> Path:
    corpus_path = args.data_root / dataset / "test" / "collection.tsv"
    trace_path = (
        args.trace_root
        / dataset
        / "train_data_multitemplate_unfiltered.jsonl"
    )
    if not corpus_path.is_file() or not trace_path.is_file():
        raise FileNotFoundError(f"Missing corpus or trace input for {dataset}.")

    doc_ids, doc_texts = terrier.load_corpus(corpus_path)
    doc_ids = [str(doc_id) for doc_id in doc_ids]
    doc_index = {doc_id: index for index, doc_id in enumerate(doc_ids)}
    original_counts = [
        dict(
            Counter(
                trace.terrier_terms(
                    tokeniser,
                    stemmer,
                    stopwords,
                    text,
                    min_len=1,
                )
            )
        )
        for text in doc_texts
    ]

    hard_counts = [Counter() for _ in doc_ids]
    hard_text_parts: list[list[str]] = [[] for _ in doc_ids]
    evidence = {
        "boundary": [
            defaultdict(lambda: [0.0, 0.0, 0.0, 0, 0]) for _ in doc_ids
        ],
        "all": [
            defaultdict(lambda: [0.0, 0.0, 0.0, 0, 0]) for _ in doc_ids
        ],
    }
    prompt_bits: dict[str, int] = {}
    observed_keys: dict[str, set[tuple[str, int]]] = defaultdict(set)
    stats: Counter[str] = Counter()

    with trace_path.open(encoding="utf-8") as source:
        for line_number, raw in enumerate(source, start=1):
            if not raw.strip():
                continue
            row = json.loads(raw)
            if not trace.row_allowed_by_sample_idx(row, args.sample_idx_max):
                continue
            doc_id = str(row.get("doc_id", row.get("pos_id", ""))).strip()
            if doc_id not in doc_index:
                stats["trace_rows_without_corpus_document"] += 1
                continue
            prompt_id = str(row.get("prompt_id", "")).strip()
            if not prompt_id:
                raise RuntimeError(
                    f"Missing prompt_id at {trace_path}:{line_number}."
                )
            sample_idx = int(row.get("sample_idx", 0))
            row_key = (prompt_id, sample_idx)
            if row_key in observed_keys[doc_id]:
                raise RuntimeError(
                    "Duplicate retained trace row for "
                    f"doc={doc_id}, prompt={prompt_id}, sample={sample_idx}."
                )
            observed_keys[doc_id].add(row_key)

            index = doc_index[doc_id]
            query_text = trace.clean_generated_query(row.get("query_text", ""))
            if query_text:
                hard_text_parts[index].append(query_text)
                hard_counts[index].update(
                    trace.terrier_terms(
                        tokeniser,
                        stemmer,
                        stopwords,
                        query_text,
                        min_len=1,
                    )
                )

            if prompt_id not in prompt_bits:
                prompt_bits[prompt_id] = 1 << len(prompt_bits)
            template_bit = prompt_bits[prompt_id]
            safe_positions, mask_stats = trace.build_rank1_boundary_mask(
                row,
                tokenizer,
                tokeniser,
                stemmer,
                stopwords,
                min_len=args.boundary_min_len,
            )
            stats.update(mask_stats)

            try:
                decoded_steps, trace_format = trace.normalized_candidate_steps(
                    row,
                    tokenizer,
                )
            except ValueError as error:
                raise RuntimeError(
                    f"Invalid candidate trace at {trace_path}:{line_number}: {error}"
                ) from error
            if decoded_steps is None:
                raise RuntimeError(
                    "Single-index SDE requires decoded_candidates or indices/probs "
                    f"at {trace_path}:{line_number}."
                )
            stats[f"trace_rows_{trace_format}"] += 1

            for position, candidates in enumerate(decoded_steps):
                if not isinstance(candidates, list):
                    continue
                for _, candidate in trace.iter_soft_alternatives(
                    candidates,
                    args.candidate_topk,
                ):
                    text, probability = (
                        trace.decoded_candidate_text_and_probability(candidate)
                    )
                    if probability < args.event_min_probability:
                        continue
                    if trace.is_decoded_special_candidate(text):
                        stats["special_candidates_filtered"] += 1
                        continue
                    terms = trace.terrier_terms(
                        tokeniser,
                        stemmer,
                        stopwords,
                        text,
                        min_len=args.candidate_min_len,
                    )
                    for term in terms:
                        update_evidence(
                            evidence["all"][index][term],
                            probability,
                            template_bit,
                        )
                        stats["candidate_terms_all"] += 1
                        if position in safe_positions:
                            update_evidence(
                                evidence["boundary"][index][term],
                                probability,
                                template_bit,
                            )
                            stats["candidate_terms_boundary"] += 1
            stats["trace_rows"] += 1

    validate_trace_coverage(
        doc_ids,
        set(prompt_bits),
        observed_keys,
        args.expected_templates,
        args.sample_idx_max,
    )

    hard_count_rows = [dict(counts) for counts in hard_counts]
    materialized_evidence: dict[str, list[dict[str, tuple[Any, ...]]]] = {}
    candidate_df: dict[str, dict[str, int]] = {}
    for position_mode, per_doc in evidence.items():
        rows: list[dict[str, tuple[Any, ...]]] = []
        df: Counter[str] = Counter()
        for terms in per_doc:
            row = {term: tuple(values) for term, values in terms.items()}
            rows.append(row)
            df.update(row.keys())
        materialized_evidence[position_mode] = rows
        candidate_df[position_mode] = dict(df)

    base_df: Counter[str] = Counter()
    original_df: Counter[str] = Counter()
    for original, hard in zip(original_counts, hard_count_rows):
        original_df.update(original)
        base_df.update(set(original) | set(hard))

    payload = {
        "version": 1,
        "method": "sde_single_index",
        "dataset": dataset,
        "doc_ids": doc_ids,
        "original_counts": original_counts,
        "hard_counts": hard_count_rows,
        "hard_texts": ["\n".join(parts) for parts in hard_text_parts],
        "base_df": dict(base_df),
        "original_df": dict(original_df),
        "candidate_evidence": materialized_evidence,
        "candidate_df": candidate_df,
        "prompt_bits": prompt_bits,
        "config": {
            "sample_idx_max": args.sample_idx_max,
            "expected_templates": args.expected_templates,
            "candidate_topk": args.candidate_topk,
            "event_min_probability": args.event_min_probability,
            "candidate_min_len": args.candidate_min_len,
            "boundary_min_len": args.boundary_min_len,
            "hard_queries_per_doc": (
                args.expected_templates * (args.sample_idx_max + 1)
            ),
            "boundary_sequence": "rank1_trajectory",
        },
        "stats": dict(stats),
        "paths": {
            "corpus": str(corpus_path),
            "trace": str(trace_path),
        },
    }
    args.cache_root.mkdir(parents=True, exist_ok=True)
    output = args.cache_root / f"{dataset}.sde_single_index_cache.pkl"
    with output.open("wb") as destination:
        pickle.dump(payload, destination, protocol=pickle.HIGHEST_PROTOCOL)
    print(
        f"[cache] {dataset} docs={len(doc_ids)} prompts={len(prompt_bits)} "
        f"path={output}"
    )
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_root",
        type=Path,
        default=Path(os.environ.get("BEIR_DATA_ROOT", REPO_ROOT / "data/beir")),
    )
    parser.add_argument("--trace_root", type=Path, required=True)
    parser.add_argument("--cache_root", type=Path, required=True)
    model_path = os.environ.get("MODEL_PATH")
    parser.add_argument(
        "--model_path",
        type=Path,
        default=Path(model_path).expanduser() if model_path else None,
    )
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS))
    parser.add_argument("--sample_idx_max", type=int, default=0)
    parser.add_argument("--expected_templates", type=int, default=6)
    parser.add_argument("--candidate_topk", type=int, default=5)
    parser.add_argument("--event_min_probability", type=float, default=0.001)
    parser.add_argument("--candidate_min_len", type=int, default=2)
    parser.add_argument("--boundary_min_len", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.expected_templates <= 0:
        raise SystemExit("--expected_templates must be positive")
    if args.sample_idx_max < 0:
        raise SystemExit("--sample_idx_max must be non-negative")
    if args.candidate_topk < 2:
        raise SystemExit("--candidate_topk must be at least 2")
    if not 0 <= args.event_min_probability <= 1:
        raise SystemExit("--event_min_probability must be in [0, 1]")

    terrier.ensure_pyterrier()
    trace_paths = [
        args.trace_root
        / dataset
        / "train_data_multitemplate_unfiltered.jsonl"
        for dataset in args.datasets
    ]
    tokenizer = trace.load_candidate_tokenizer(trace_paths, args.model_path)
    tokeniser = terrier.pt.terrier.TerrierTokeniser.java_tokeniser(
        terrier.pt.terrier.TerrierTokeniser._to_obj("english")
    )
    stemmer = terrier.pt.terrier.TerrierStemmer.porter
    stopwords = terrier.load_terrier_stopwords()
    for dataset in args.datasets:
        build_dataset(
            args,
            dataset,
            tokenizer,
            tokeniser,
            stemmer,
            stopwords,
        )


if __name__ == "__main__":
    main()

