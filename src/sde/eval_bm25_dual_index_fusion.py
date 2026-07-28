#!/usr/bin/env python3
"""Evaluate SDE with the factorized query-document auxiliary index."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Sequence

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.sde.factorized_qdoc import (  # noqa: E402
    build_factorized_qdoc_components as factor_builder,
    factorized_qdoc_index as factor,
    prepare_factorized_qdoc_components as component_prepare,
)
from src.sde.terrier_utils import (  # noqa: E402
    build_retriever,
    compute_metrics_from_ranking,
    ensure_pyterrier,
    index_corpus,
    load_corpus,
    load_qrels,
    load_queries,
    prepare_query_df,
    read_index_stats,
    write_trec_run,
)


DEFAULT_ALPHA = 0.5
DEFAULT_DOC_TOPN = 300
DEFAULT_QDOC_TOPN = 1000
DEFAULT_K1 = 0.9
DEFAULT_B = 0.4
DEFAULT_DECAY = 0.3


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def minmax_scores(scores: dict[Any, float]) -> dict[Any, float]:
    if not scores:
        return {}
    minimum = min(scores.values())
    maximum = max(scores.values())
    denominator = maximum - minimum
    if abs(denominator) < 1e-12:
        return {key: 0.0 for key in scores}
    return {
        key: (float(score) - minimum) / denominator
        for key, score in scores.items()
    }


def source_docnos(index: factor.FactorizedQDocIndex) -> list[str]:
    source_count = int(index.qdoc_source.max()) + 1
    values = [""] * source_count
    for document_id, qdocno in enumerate(index.qdoc_docnos):
        source_id = int(index.qdoc_source[document_id])
        if not values[source_id]:
            values[source_id] = factor.source_docno(qdocno)
    if any(not value for value in values):
        raise ValueError("Factorized index contains an unnamed source group")
    return values


def retrieve_original(
    retriever: Any,
    query_id: str,
    query: str,
) -> list[tuple[str, float]]:
    frame = pd.DataFrame([{"qid": query_id, "query": query}])
    result = retriever.transform(frame)
    if result is None or result.empty:
        return []
    return [
        (str(row.docno), float(row.score))
        for row in result[["docno", "score"]].itertuples(index=False)
    ]


def aggregate_qdoc_results(
    results: Sequence[tuple[int, str, float]],
    qdoc_source: Any,
    decay: float = DEFAULT_DECAY,
) -> dict[int, float]:
    scores: dict[int, float] = {}
    hit_counts: dict[int, int] = {}
    for document_id, _, score in results:
        source_id = int(qdoc_source[int(document_id)])
        position = hit_counts.get(source_id, 0)
        scores[source_id] = (
            scores.get(source_id, 0.0)
            + float(score) * (float(decay) ** position)
        )
        hit_counts[source_id] = position + 1
    return scores


def fuse_results(
    original_results: Sequence[tuple[str, float]],
    qdoc_results: Sequence[tuple[int, str, float]],
    qdoc_source: Any,
    source_names: Sequence[str],
    alpha: float,
    decay: float = DEFAULT_DECAY,
) -> list[tuple[str, float]]:
    original = minmax_scores(dict(original_results))
    auxiliary = minmax_scores(
        aggregate_qdoc_results(qdoc_results, qdoc_source, decay)
    )
    fused = {
        document_id: (1.0 - alpha) * score
        for document_id, score in original.items()
    }
    for source_id, score in auxiliary.items():
        document_id = source_names[source_id]
        fused[document_id] = fused.get(document_id, 0.0) + alpha * score
    return sorted(fused.items(), key=lambda item: (-item[1], item[0]))


def build_or_load_factorized_index(
    expansion_jsonl: Path,
    model_path: Path,
    component_path: Path,
    component_stats_path: Path,
    factor_index_dir: Path,
    overwrite: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if overwrite:
        component_path.unlink(missing_ok=True)
        component_stats_path.unlink(missing_ok=True)
        shutil.rmtree(factor_index_dir, ignore_errors=True)

    if not component_path.is_file() or not component_stats_path.is_file():
        component_stats = component_prepare.prepare_components(
            expansion_jsonl,
            model_path,
            component_path,
            component_stats_path,
        )
    else:
        component_stats = json.loads(
            component_stats_path.read_text(encoding="utf-8")
        )

    manifest_path = factor_index_dir / factor.MANIFEST_FILE
    if not manifest_path.is_file():
        factor_manifest = factor_builder.build_component_factorized_index(
            component_path,
            factor_index_dir,
        )
    else:
        factor_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return component_stats, factor_manifest


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    ensure_pyterrier()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    component_path = (
        args.component_path
        if args.component_path is not None
        else args.out_dir / "factorized_components.jsonl.gz"
    )
    component_stats_path = (
        args.component_stats_path
        if args.component_stats_path is not None
        else args.out_dir / "factorized_components.json"
    )
    factor_index_dir = (
        args.factor_index_dir
        if args.factor_index_dir is not None
        else args.out_dir / "factorized_index"
    )
    doc_index_dir = (
        args.doc_index_dir
        if args.doc_index_dir is not None
        else args.out_dir / "doc_index"
    )

    component_stats, factor_manifest = build_or_load_factorized_index(
        args.expansion_jsonl,
        args.model_path,
        component_path,
        component_stats_path,
        factor_index_dir,
        args.overwrite,
    )
    factor_index = factor.FactorizedQDocIndex(factor_index_dir)
    structure_checks = factor_index.structure_checks()
    if not all(structure_checks.values()):
        raise RuntimeError(
            f"Factorized index validation failed: {structure_checks}"
        )

    document_ids, document_texts = load_corpus(args.corpus_tsv)
    document_ref = index_corpus(
        doc_index_dir,
        document_ids,
        document_texts,
        reuse_existing=not args.overwrite,
    )
    document_retriever = build_retriever(
        document_ref,
        args.doc_topn,
        args.k1,
        args.b,
    )
    processor = factor.QueryProcessor()
    source_names = source_docnos(factor_index)
    source_name_set = set(source_names)
    if len(source_names) != len(source_name_set):
        raise ValueError("Factorized source groups contain duplicate document IDs")
    unknown_sources = source_name_set - set(document_ids)
    if unknown_sources:
        raise ValueError(
            "Factorized source groups contain documents outside the corpus"
        )
    original_docs_without_qdocs = len(set(document_ids) - source_name_set)

    queries = load_queries(args.queries_json)
    qrels = load_qrels(args.qrels_tsv)
    query_frame = prepare_query_df(queries, qrels)
    ranking_rows: list[dict[str, Any]] = []
    for row in query_frame.itertuples(index=False):
        query_id = str(row.qid)
        query = str(row.query)
        original_results = retrieve_original(
            document_retriever,
            query_id,
            query,
        )
        qdoc_results = factor_index.score_terms(
            processor.terms(query),
            args.qdoc_topn,
        )
        fused = fuse_results(
            original_results,
            qdoc_results,
            factor_index.qdoc_source,
            source_names,
            args.fusion_alpha,
            args.aggregation_decay,
        )
        ranking_rows.extend(
            {
                "qid": query_id,
                "docno": document_id,
                "score": score,
                "rank": rank,
            }
            for rank, (document_id, score) in enumerate(fused)
        )

    ranking = pd.DataFrame(
        ranking_rows,
        columns=["qid", "docno", "score", "rank"],
    )
    metrics = compute_metrics_from_ranking(ranking, qrels)
    run_path = args.out_dir / "dual_index_fusion_best.run"
    summary_path = args.out_dir / "dual_index_fusion_summary.tsv"
    result_path = args.out_dir / "dual_index_fusion_results.json"
    write_trec_run(ranking, run_path, "sde_factorized_qdoc")
    pd.DataFrame(
        [
            {
                "fusion_mode": "minmax",
                "score_norm": "minmax",
                "agg_mode": f"sum_decay_{args.aggregation_decay:g}",
                "alpha": args.fusion_alpha,
                "n_queries": metrics["n_queries"],
                "ndcg_at_10": metrics["ndcg_at_10"],
                "mrr_at_10": metrics["mrr_at_10"],
                "recall_at_100": metrics["recall_at_100"],
                "map": metrics["map"],
            }
        ]
    ).to_csv(summary_path, sep="\t", index=False)

    result = {
        "method": "sde_factorized_query_document",
        "n_docs": len(document_ids),
        "n_query_docs": factor_index.document_count,
        "original_docs_without_qdocs": original_docs_without_qdocs,
        "n_eval_queries": int(query_frame.shape[0]),
        "doc_index_dir": str(doc_index_dir),
        "factor_index_dir": str(factor_index_dir),
        "doc_index_stats": read_index_stats(doc_index_dir),
        "factor_index_manifest": factor_manifest,
        "component_stats": component_stats,
        "structure_checks": structure_checks,
        "config": {
            "sample_idx_max": 0,
            "query_doc_mode": "query_text_plus_soft",
            "candidate_positions": "all",
            "blacklisted_token_ids": [220, 128009],
            "drop_numeric_terms": True,
            "drop_short_continuation_terms": False,
            "min_token_len": 2,
            "remove_stopwords": True,
            "prob_threshold": 0.01,
            "soft_topk_per_step": 5,
            "max_soft_terms_per_doc": 256,
            "term_weight_mode": "repeat_by_score",
            "repeat_score_scale": 3.0,
            "repeat_max_times": 3,
            "doc_retrieval_topn": args.doc_topn,
            "qdoc_retrieval_topn": args.qdoc_topn,
            "aggregation": f"sum_decay_{args.aggregation_decay:g}",
            "score_normalization": "minmax",
            "fusion_alpha": args.fusion_alpha,
            "doc_k1": args.k1,
            "doc_b": args.b,
            "qdoc_k1": args.k1,
            "qdoc_b": args.b,
        },
        "best_result": metrics,
        "best_run": str(run_path),
        "summary_tsv": str(summary_path),
    }
    write_json(result_path, result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate SDE with its factorized query-document index."
    )
    parser.add_argument("--corpus_tsv", type=Path, required=True)
    parser.add_argument("--queries_json", type=Path, required=True)
    parser.add_argument("--qrels_tsv", type=Path, required=True)
    parser.add_argument("--expansion_jsonl", type=Path, required=True)
    parser.add_argument("--model_path", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--doc_index_dir", type=Path)
    parser.add_argument("--factor_index_dir", type=Path)
    parser.add_argument("--component_path", type=Path)
    parser.add_argument("--component_stats_path", type=Path)
    parser.add_argument("--fusion_alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument("--doc_topn", type=int, default=DEFAULT_DOC_TOPN)
    parser.add_argument("--qdoc_topn", type=int, default=DEFAULT_QDOC_TOPN)
    parser.add_argument("--aggregation_decay", type=float, default=DEFAULT_DECAY)
    parser.add_argument("--k1", type=float, default=DEFAULT_K1)
    parser.add_argument("--b", type=float, default=DEFAULT_B)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if not 0.0 <= args.fusion_alpha <= 1.0:
        parser.error("--fusion_alpha must be in [0, 1]")
    if not 0.0 <= args.aggregation_decay <= 1.0:
        parser.error("--aggregation_decay must be in [0, 1]")
    if args.doc_topn <= 0 or args.qdoc_topn <= 0:
        parser.error("retrieval depths must be positive")
    return args


def main() -> None:
    args = parse_args()
    result = evaluate(args)
    metrics = result["best_result"]
    print(
        "nDCG@10={ndcg:.9f} MRR@10={mrr:.9f} "
        "MAP={map_value:.9f} Recall@100={recall:.9f}".format(
            ndcg=metrics["ndcg_at_10"],
            mrr=metrics["mrr_at_10"],
            map_value=metrics["map"],
            recall=metrics["recall_at_100"],
        )
    )


if __name__ == "__main__":
    main()
