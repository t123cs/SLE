#!/usr/bin/env python3
"""Build and evaluate SDE in one BM25 index."""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from src.sde import terrier_utils as terrier  # noqa: E402


def idf_value(num_docs: int, document_frequency: int) -> float:
    document_frequency = min(max(int(document_frequency), 0), num_docs)
    return math.log1p(
        (num_docs - document_frequency + 0.5)
        / (document_frequency + 0.5)
    )


def noisy_or_probability(values: tuple[Any, ...]) -> float:
    return 1.0 - math.exp(float(values[1]))


def select_document_terms(
    cache: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[list[dict[str, int]], dict[str, Any]]:
    num_docs = len(cache["doc_ids"])
    evidence_rows = cache["candidate_evidence"][args.position_mode]
    candidate_df = cache["candidate_df"][args.position_mode]
    base_df = cache["base_df"]
    max_candidate_df = (
        num_docs
        if args.candidate_max_df_ratio >= 1
        else max(1, int(num_docs * args.candidate_max_df_ratio))
    )
    soft_counts: list[dict[str, int]] = []
    stats: Counter[str] = Counter()

    for original, hard, evidence in zip(
        cache["original_counts"],
        cache["hard_counts"],
        evidence_rows,
    ):
        hard_terms = set(hard)
        original_terms = set(original)
        ranked: list[tuple[float, float, float, str]] = []
        for term, values in evidence.items():
            if term in hard_terms:
                stats["candidates_in_hard_filtered"] += 1
                continue
            if args.exclude_original_terms and term in original_terms:
                stats["candidates_in_original_filtered"] += 1
                continue
            term_candidate_df = int(candidate_df.get(term, 0))
            if term_candidate_df > max_candidate_df:
                stats["candidate_high_df_filtered"] += 1
                continue
            probability = noisy_or_probability(values)
            if probability < args.min_aggregate_probability:
                stats["candidate_low_probability_filtered"] += 1
                continue
            combined_df = min(
                num_docs,
                int(base_df.get(term, 0)) + term_candidate_df,
            )
            idf = idf_value(num_docs, combined_df)
            ranked.append((probability * idf, probability, idf, term))

        ranked.sort(key=lambda row: (-row[0], -row[1], row[3]))
        selected = ranked[: args.max_soft_terms]
        soft = {term: 1 for _, _, _, term in selected} if args.mode == "sde" else {}
        soft_counts.append(soft)
        stats["soft_terms_selected"] += len(soft)
        stats["documents_with_soft_terms"] += bool(soft)

    stats["avg_soft_terms_per_doc"] = (
        stats["soft_terms_selected"] / num_docs if num_docs else 0.0
    )
    return soft_counts, dict(stats)


def render_term_text(counts: dict[str, int]) -> str:
    return " ".join(
        term
        for term, frequency in sorted(counts.items())
        for _ in range(int(frequency))
    )


def index_raw_text_corpus(
    index_dir: Path,
    document_ids: list[str],
    document_texts: list[str],
) -> Any:
    """Build the raw-text index used by the single-index evaluation."""
    output = index_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame({"docno": document_ids, "text": document_texts})
    indexer = terrier.pt.DFIndexer(str(output), overwrite=True)
    return indexer.index(frame["text"], frame["docno"])


def run(args: argparse.Namespace) -> dict[str, Any]:
    with args.cache.open("rb") as source:
        cache = pickle.load(source)
    if cache.get("method") != "sde_single_index":
        raise RuntimeError("Cache was not produced by the single-index SDE pipeline.")

    dataset = str(cache["dataset"])
    query_dir = args.data_root / dataset / args.query_split
    queries = terrier.load_queries(query_dir / "queries.jsonl")
    qrels = terrier.load_qrels(
        query_dir / f"qrels.{args.query_split}.tsv"
    )

    terrier.ensure_pyterrier()
    query_frame = terrier.prepare_query_df(queries, qrels)
    soft_counts, selection_stats = select_document_terms(cache, args)

    if args.out_dir.exists() and args.overwrite:
        shutil.rmtree(args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    index_dir = args.out_dir / "index"

    corpus_ids, corpus_texts = terrier.load_corpus(cache["paths"]["corpus"])
    corpus_ids = [str(doc_id) for doc_id in corpus_ids]
    if corpus_ids != cache["doc_ids"]:
        raise RuntimeError("Cache document order does not match the corpus.")

    indexed_texts: list[str] = []
    for document_text, hard_text, soft in zip(
        corpus_texts,
        cache["hard_texts"],
        soft_counts,
    ):
        soft_text = render_term_text(soft)
        indexed_texts.append(
            "\n".join(
                part
                for part in (document_text, hard_text, soft_text)
                if part
            )
        )

    index_ref = index_raw_text_corpus(index_dir, corpus_ids, indexed_texts)
    index_stats = terrier.read_index_stats(index_dir)
    retriever = terrier.build_retriever(
        index_ref,
        args.retrieval_topn,
        args.k1,
        args.b,
    )
    ranking = retriever.transform(query_frame)[
        ["qid", "docno", "score", "rank"]
    ].copy()
    metrics = terrier.compute_metrics_from_ranking(ranking, qrels)

    run_path = args.out_dir / "sde_single_index.run"
    results_path = args.out_dir / "sde_single_index_results.json"
    summary_path = args.out_dir / "sde_single_index_summary.tsv"
    run_tag = "sde_single_index" if args.mode == "sde" else "hard_single_index"
    terrier.write_trec_run(ranking, run_path, run_tag)

    config = {
        "mode": args.mode,
        "query_split": args.query_split,
        "position_mode": args.position_mode,
        "min_aggregate_probability": args.min_aggregate_probability,
        "candidate_max_df_ratio": args.candidate_max_df_ratio,
        "max_soft_terms": args.max_soft_terms,
        "exclude_original_terms": args.exclude_original_terms,
        "soft_term_frequency": 1,
        "score": "noisy_or_probability_times_combined_idf",
        "k1": args.k1,
        "b": args.b,
        "retrieval_topn": args.retrieval_topn,
    }
    result = {
        "dataset": dataset,
        "method": (
            "SDE, single index"
            if args.mode == "sde"
            else "Hard decoded, single index"
        ),
        "cache": str(args.cache),
        "config": config,
        "selection_stats": selection_stats,
        "index_stats": index_stats,
        "metrics": metrics,
        "run": str(run_path),
    }
    results_path.write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )
    pd.DataFrame([{**config, **metrics}]).to_csv(
        summary_path,
        sep="\t",
        index=False,
    )
    if args.cleanup_index:
        shutil.rmtree(index_dir, ignore_errors=True)

    print(
        f"[result] {dataset} mode={args.mode} "
        f"ndcg@10={metrics['ndcg_at_10']:.6f} "
        f"mrr@10={metrics['mrr_at_10']:.6f} "
        f"soft={selection_stats['avg_soft_terms_per_doc']:.2f}"
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument(
        "--data_root",
        type=Path,
        default=Path(os.environ.get("BEIR_DATA_ROOT", REPO_ROOT / "data/beir")),
    )
    parser.add_argument(
        "--query_split",
        choices=("train", "test"),
        default="test",
    )
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("hard", "sde"), default="sde")
    parser.add_argument(
        "--position_mode",
        choices=("boundary", "all"),
        default="all",
    )
    parser.add_argument(
        "--min_aggregate_probability",
        type=float,
        default=0.01,
    )
    parser.add_argument(
        "--candidate_max_df_ratio",
        type=float,
        default=0.05,
    )
    parser.add_argument("--max_soft_terms", type=int, default=16)
    parser.add_argument("--exclude_original_terms", action="store_true")
    parser.add_argument("--k1", type=float, default=0.9)
    parser.add_argument("--b", type=float, default=0.4)
    parser.add_argument("--retrieval_topn", type=int, default=1000)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--cleanup_index", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0 <= args.min_aggregate_probability <= 1:
        raise SystemExit("--min_aggregate_probability must be in [0, 1]")
    if not 0 <= args.candidate_max_df_ratio <= 1:
        raise SystemExit("--candidate_max_df_ratio must be in [0, 1]")
    if args.max_soft_terms < 0:
        raise SystemExit("--max_soft_terms must be non-negative")
    if args.k1 < 0 or not 0 <= args.b <= 1:
        raise SystemExit("BM25 requires k1 >= 0 and b in [0, 1]")
    if args.retrieval_topn <= 0:
        raise SystemExit("--retrieval_topn must be positive")
    run(args)


if __name__ == "__main__":
    main()
