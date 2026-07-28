#!/usr/bin/env python3
"""Shared BEIR and PyTerrier utilities for the SDE evaluation package."""

from __future__ import annotations

import json
import math
import os
import re
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

try:
    import pyterrier as pt
except Exception:
    pt = None


STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "to", "of", "in", "on", "at", "by", "for",
    "with", "from", "as", "or", "and", "but", "not", "it", "its", "i",
    "we", "you", "he", "she", "they", "this", "that", "my", "your", "our",
    "their", "what", "which", "who", "if", "so", "up", "out", "can", "no",
    "more", "also", "about", "than", "then", "just", "into", "over", "after",
    "all", "when", "there", "me", "him", "her", "them",
}
DEFAULT_TERRIER_JAR = (
    Path.home()
    / ".cache"
    / "pyterrier"
    / "terrier-assemblies-5.11-jar-with-dependencies.jar"
)


def sanitize_query_for_pyterrier(text: str) -> str:
    return " ".join(re.findall(r"[A-Za-z0-9]+", str(text)))


def sanitize_document_for_pyterrier(text: str) -> str:
    # DFIndexer treats angle-bracketed scientific expressions as markup.
    return str(text).replace("<", " ").replace(">", " ")


def iter_terrier_jar_candidates(
    explicit_path: str | Path | None = None,
) -> Iterable[Path]:
    roots = [
        Path(__file__).resolve().parents[2] / "cache" / "pyterrier",
        Path.home() / ".pyterrier",
        Path.home() / ".cache" / "pyterrier",
        DEFAULT_TERRIER_JAR.parent,
    ]
    candidates = [
        explicit_path,
        os.environ.get("TERRIER_JAR"),
        os.environ.get("PYTERRIER_TERRIER_JAR"),
        DEFAULT_TERRIER_JAR,
    ]

    seen: set[str] = set()
    for value in candidates:
        if not value:
            continue
        path = Path(value).expanduser()
        key = str(path)
        if key not in seen:
            seen.add(key)
            yield path

    for root in roots:
        if not root.is_dir():
            continue
        for path in sorted(
            root.glob("**/terrier-assemblies-*-jar-with-dependencies.jar")
        ):
            key = str(path)
            if key not in seen:
                seen.add(key)
                yield path


def load_terrier_stopwords(
    jar_path: str | Path | None = None,
) -> set[str]:
    for candidate in iter_terrier_jar_candidates(jar_path):
        if not candidate.is_file():
            continue
        try:
            with zipfile.ZipFile(candidate) as jar:
                with jar.open("stopword-list.txt") as source:
                    stopwords = {
                        decoded
                        for raw in source
                        if (decoded := raw.decode("utf-8").strip())
                        and not decoded.startswith("#")
                    }
            return stopwords
        except Exception:
            continue
    return set(STOPWORDS)


def load_corpus(path: str | Path) -> tuple[list[str], list[str]]:
    document_ids: list[str] = []
    document_texts: list[str] = []
    with Path(path).open("r", encoding="utf-8") as source:
        for raw in source:
            fields = raw.rstrip("\n").split("\t", 1)
            if len(fields) < 2:
                continue
            document_ids.append(fields[0])
            document_texts.append(fields[1])
    return document_ids, document_texts


def load_queries(path: str | Path) -> dict[str, str]:
    queries: dict[str, str] = {}
    with Path(path).open("r", encoding="utf-8") as source:
        for raw in source:
            if not raw.strip():
                continue
            row = json.loads(raw)
            query_id = str(row.get("qid", row.get("_id")))
            text = row.get("question", row.get("text", ""))
            if text:
                queries[query_id] = str(text)
    return queries


def load_qrels(path: str | Path) -> dict[str, dict[str, int]]:
    qrels: dict[str, dict[str, int]] = defaultdict(dict)
    header: list[str] | None = None
    with Path(path).open("r", encoding="utf-8") as source:
        for raw in source:
            line = raw.strip()
            if not line:
                continue
            fields = line.split("\t")
            if len(fields) < 3:
                continue
            if header is None and fields[0].lower() in {
                "qid",
                "query_id",
                "query-id",
            }:
                header = [field.lower() for field in fields]
                continue
            if header and {"query-id", "corpus-id", "score"}.issubset(header):
                query_id = fields[header.index("query-id")]
                document_id = fields[header.index("corpus-id")]
                relevance = int(fields[header.index("score")])
            elif header and {"qid", "docno", "label"}.issubset(header):
                query_id = fields[header.index("qid")]
                document_id = fields[header.index("docno")]
                relevance = int(fields[header.index("label")])
            elif len(fields) >= 4:
                query_id = fields[0]
                document_id = fields[2]
                relevance = int(fields[3])
            else:
                query_id = fields[0]
                document_id = fields[1]
                relevance = int(fields[2])
            qrels[query_id][document_id] = relevance
    return dict(qrels)


def prepare_query_df(
    queries: dict[str, str],
    qrels: dict[str, dict[str, int]],
) -> pd.DataFrame:
    rows = []
    for query_id in qrels:
        if query_id not in queries:
            continue
        query = sanitize_query_for_pyterrier(queries[query_id]).strip()
        if query:
            rows.append({"qid": query_id, "query": query})
    return pd.DataFrame(rows)


def ensure_pyterrier() -> None:
    if pt is None:
        raise RuntimeError("PyTerrier is not available in the current environment.")
    if hasattr(pt, "java"):
        if not pt.java.started():
            pt.java.init()
    elif not pt.started():
        pt.init()


def maybe_load_indexref(index_dir: str | Path) -> Any | None:
    data_properties = Path(index_dir) / "data.properties"
    if data_properties.is_file():
        return pt.IndexRef.of(str(data_properties.resolve()))
    return None


def index_corpus(
    index_dir: str | Path,
    document_ids: Sequence[str],
    document_texts: Sequence[str],
    reuse_existing: bool = False,
) -> Any:
    existing = maybe_load_indexref(index_dir) if reuse_existing else None
    if existing is not None:
        print(f"Reusing existing index at {index_dir}")
        return existing

    output = Path(index_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(
        {
            "docno": document_ids,
            "text": [
                sanitize_document_for_pyterrier(text)
                for text in document_texts
            ],
        }
    )
    indexer = pt.DFIndexer(str(output), overwrite=True)
    return indexer.index(frame["text"], frame["docno"])


def read_index_stats(index_dir: str | Path) -> dict[str, Any]:
    directory = Path(index_dir)
    properties: dict[str, str] = {}
    properties_path = directory / "data.properties"
    if properties_path.is_file():
        with properties_path.open("r", encoding="utf-8") as source:
            for raw in source:
                line = raw.strip()
                if not line or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                properties[key.strip()] = value.strip()

    def safe_int(key: str) -> int:
        try:
            return int(properties.get(key, "0"))
        except (TypeError, ValueError):
            return 0

    size_bytes = sum(
        item.stat().st_size for item in directory.rglob("*") if item.is_file()
    )
    return {
        "index_dir": str(directory),
        "num_documents": safe_int("num.Documents"),
        "num_terms": safe_int("num.Terms"),
        "num_pointers": safe_int("num.Pointers"),
        "size_bytes": size_bytes,
    }


def build_retriever(
    index_ref: Any,
    topk: int,
    k1: float,
    b: float,
) -> Any:
    controls = {"bm25.k_1": str(k1), "bm25.b": str(b)}
    if hasattr(pt, "terrier") and hasattr(pt.terrier, "Retriever"):
        try:
            return pt.terrier.Retriever(
                index_ref,
                wmodel="BM25",
                num_results=topk,
                controls=controls,
            )
        except TypeError:
            return pt.terrier.Retriever(
                index_ref,
                wmodel="BM25",
                num_results=topk,
                properties=controls,
            )
    return pt.BatchRetrieve(
        index_ref,
        wmodel="BM25",
        num_results=topk,
        controls=controls,
    )


def compute_metrics_from_ranking(
    ranking: pd.DataFrame,
    qrels: dict[str, dict[str, int]],
) -> dict[str, float | int]:
    grouped: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for row in ranking.itertuples(index=False):
        grouped[str(row.qid)].append((str(row.docno), int(row.rank)))

    ndcgs: list[float] = []
    mrrs: list[float] = []
    recalls: list[float] = []
    average_precisions: list[float] = []
    for query_id, relevance_by_document in qrels.items():
        ranked = sorted(grouped.get(query_id, []), key=lambda item: item[1])
        documents = [document_id for document_id, _ in ranked]

        reciprocal_rank = 0.0
        for rank, document_id in enumerate(documents[:10], start=1):
            if relevance_by_document.get(document_id, 0) > 0:
                reciprocal_rank = 1.0 / rank
                break

        dcg = sum(
            (2 ** relevance_by_document.get(document_id, 0) - 1)
            / math.log2(rank + 1)
            for rank, document_id in enumerate(documents[:10], start=1)
            if relevance_by_document.get(document_id, 0) > 0
        )
        ideal = sorted(
            (
                relevance
                for relevance in relevance_by_document.values()
                if relevance > 0
            ),
            reverse=True,
        )[:10]
        idcg = sum(
            (2**relevance - 1) / math.log2(rank + 1)
            for rank, relevance in enumerate(ideal, start=1)
        )

        relevant_count = sum(
            relevance > 0 for relevance in relevance_by_document.values()
        )
        hits = sum(
            relevance_by_document.get(document_id, 0) > 0
            for document_id in documents[:100]
        )
        hit_count = 0
        precision_sum = 0.0
        for rank, document_id in enumerate(documents, start=1):
            if relevance_by_document.get(document_id, 0) > 0:
                hit_count += 1
                precision_sum += hit_count / rank

        ndcgs.append(dcg / idcg if idcg else 0.0)
        mrrs.append(reciprocal_rank)
        recalls.append(hits / relevant_count if relevant_count else 0.0)
        average_precisions.append(
            precision_sum / relevant_count if relevant_count else 0.0
        )

    return {
        "n_queries": len(ndcgs),
        "ndcg_at_10": float(np.mean(ndcgs)) if ndcgs else 0.0,
        "mrr_at_10": float(np.mean(mrrs)) if mrrs else 0.0,
        "recall_at_100": float(np.mean(recalls)) if recalls else 0.0,
        "map": (
            float(np.mean(average_precisions))
            if average_precisions
            else 0.0
        ),
    }


def write_trec_run(
    ranking: pd.DataFrame,
    output: str | Path,
    tag: str,
) -> None:
    path = Path(output)
    if ranking is None or ranking.empty:
        path.write_text("", encoding="utf-8")
        return
    ordered = ranking.sort_values(
        ["qid", "rank", "docno"],
        ascending=[True, True, True],
    )
    with path.open("w", encoding="utf-8") as destination:
        for row in ordered.itertuples(index=False):
            destination.write(
                f"{row.qid} Q0 {row.docno} {int(row.rank) + 1} "
                f"{float(row.score):.6f} {tag}\n"
            )
