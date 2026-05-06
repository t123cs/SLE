#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List

from common import DATASET_SPECS, clean_id_for_path, ensure_dir, read_jsonl, write_csv, write_json


SPECIAL_TERMS = {
    "id",
    "eot",
    "eom",
    "eos",
    "bos",
    "pad",
    "unk",
    "sep",
    "cls",
    "assistant",
    "user",
    "system",
    "end",
    "start",
}


def configure_java_env() -> Dict[str, str]:
    env = os.environ.copy()
    env["JAVA_HOME"] = env.get("JAVA_HOME") or str(Path(sys.executable).resolve().parent.parent)
    env["PATH"] = f"{Path(sys.executable).resolve().parent}{os.pathsep}{env.get('PATH', '')}"
    return env


def docno(dataset: str, doc_id: str) -> str:
    return f"{dataset}::{doc_id}"


def load_sample_docs(sample_path: Path) -> List[Dict[str, str]]:
    docs = []
    for row in read_jsonl(sample_path):
        dataset = str(row["dataset"])
        doc_id = str(row["doc_id"])
        docs.append(
            {
                "dataset": dataset,
                "doc_id": doc_id,
                "docno": docno(dataset, doc_id),
                "text": str(row.get("text") or "empty"),
            }
        )
    return docs


def load_d2qpp_text(output_root: Path, doc: Dict[str, str]) -> str:
    path = (
        output_root
        / "artifacts"
        / "d2qpp"
        / "full"
        / "expanded_docs"
        / doc["dataset"]
        / f"{clean_id_for_path(doc['doc_id'])}.txt"
    )
    if not path.is_file():
        raise FileNotFoundError(f"missing D2Q++ expanded doc: {path}")
    return path.read_text(encoding="utf-8") or "empty"


def load_sde_terms(output_root: Path, doc: Dict[str, str]) -> List[str]:
    path = (
        output_root
        / "artifacts"
        / "sde"
        / "expansion_entries"
        / doc["dataset"]
        / f"{clean_id_for_path(doc['doc_id'])}.txt"
    )
    if not path.is_file():
        raise FileNotFoundError(f"missing SDE expansion entry: {path}")
    return path.read_text(encoding="utf-8").split()


def compact_sde_terms(terms: Iterable[str], min_len: int, cap: int, unique: bool) -> List[str]:
    out: List[str] = []
    seen = set()
    for term in terms:
        normalized = str(term).lower()
        if normalized in SPECIAL_TERMS:
            continue
        if len(normalized) < min_len or normalized.isdigit():
            continue
        if not re.match(r"^[a-z0-9]+$", normalized):
            continue
        if normalized in seen:
            if unique:
                continue
        else:
            if len(seen) >= cap:
                continue
            seen.add(normalized)
        out.append(normalized)
    return out


def write_json_collection(path: Path, rows: Iterable[Dict[str, str]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as fout:
        for row in rows:
            fout.write(json.dumps({"id": row["docno"], "contents": row["text"]}, ensure_ascii=False) + "\n")


def build_collection_inputs(
    output_root: Path,
    sample_docs: List[Dict[str, str]],
    collection_root: Path,
    compact_min_len: int,
    compact_cap: int,
    compact_unique: bool,
) -> Dict[str, Path]:
    reset_root(collection_root)
    original_rows = [{"docno": doc["docno"], "text": doc["text"]} for doc in sample_docs]
    segmented_original_rows = [{"docno": f"{doc['docno']}.text", "text": doc["text"]} for doc in sample_docs]
    d2qpp_rows = [{"docno": doc["docno"], "text": load_d2qpp_text(output_root, doc)} for doc in sample_docs]
    aux_current_rows = []
    aux_compact_rows = []
    compact_bytes = 0
    current_bytes = 0
    for doc in sample_docs:
        terms = load_sde_terms(output_root, doc)
        current_text = " ".join(terms) or "empty"
        compact_text = " ".join(compact_sde_terms(terms, compact_min_len, compact_cap, compact_unique)) or "empty"
        current_bytes += len((current_text + "\n").encode("utf-8"))
        compact_bytes += len((compact_text + "\n").encode("utf-8"))
        aux_current_rows.append({"docno": f"{doc['docno']}.aux", "text": current_text})
        aux_compact_rows.append({"docno": f"{doc['docno']}.aux", "text": compact_text})

    paths = {
        "original": collection_root / "original" / "docs.jsonl",
        "sde_original_segment": collection_root / "sde_original_segment" / "docs.jsonl",
        "d2qpp_append": collection_root / "d2qpp_append" / "docs.jsonl",
        "sde_aux_current": collection_root / "sde_aux_current" / "docs.jsonl",
        "sde_aux_compact": collection_root / "sde_aux_compact" / "docs.jsonl",
        "sde_segmented_current": collection_root / "sde_segmented_current" / "docs.jsonl",
        "sde_segmented_compact": collection_root / "sde_segmented_compact" / "docs.jsonl",
    }
    write_json_collection(paths["original"], original_rows)
    write_json_collection(paths["sde_original_segment"], segmented_original_rows)
    write_json_collection(paths["d2qpp_append"], d2qpp_rows)
    write_json_collection(paths["sde_aux_current"], aux_current_rows)
    write_json_collection(paths["sde_aux_compact"], aux_compact_rows)
    write_json_collection(paths["sde_segmented_current"], [*segmented_original_rows, *aux_current_rows])
    write_json_collection(paths["sde_segmented_compact"], [*segmented_original_rows, *aux_compact_rows])
    write_json(
        collection_root / "expansion_text_sizes.json",
        {
            "sde_current_kb_per_doc": current_bytes / len(sample_docs) / 1024.0,
            "sde_compact_kb_per_doc": compact_bytes / len(sample_docs) / 1024.0,
        },
    )
    return paths


def reset_root(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    ensure_dir(path)


def run_index(input_dir: Path, index_dir: Path, threads: int, log_path: Path, append: bool = False) -> float:
    ensure_dir(index_dir.parent)
    if not append and index_dir.exists():
        shutil.rmtree(index_dir)
    cmd = [
        sys.executable,
        "-m",
        "pyserini.index.lucene",
        "-collection",
        "JsonCollection",
        "-generator",
        "DefaultLuceneDocumentGenerator",
        "-input",
        str(input_dir),
        "-index",
        str(index_dir),
        "-threads",
        str(threads),
        "-quiet",
    ]
    if append:
        cmd.append("-append")
    env = configure_java_env()
    start = time.perf_counter()
    ensure_dir(log_path.parent)
    with log_path.open("a", encoding="utf-8") as log_file:
        log_file.write(f"\n===== {' '.join(cmd)} =====\n")
        subprocess.run(cmd, check=True, env=env, stdout=log_file, stderr=subprocess.STDOUT)
    return time.perf_counter() - start


def dir_size_bytes(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def load_queries(beir_root: Path, max_queries_per_dataset: int) -> List[Dict[str, str]]:
    queries = []
    for spec in DATASET_SPECS:
        split_root = beir_root / spec.slug / spec.split
        qids_with_qrels = set()
        with (split_root / f"qrels.{spec.split}.tsv").open("r", encoding="utf-8") as fin:
            for raw in fin:
                parts = raw.strip().split()
                if parts:
                    qids_with_qrels.add(parts[0])
        dataset_queries = []
        with (split_root / "queries.jsonl").open("r", encoding="utf-8") as fin:
            for raw in fin:
                if not raw.strip():
                    continue
                obj = json.loads(raw)
                qid = str(obj.get("qid") or obj.get("_id") or obj.get("id") or "")
                if qid not in qids_with_qrels:
                    continue
                text = sanitize_query(obj.get("question") or obj.get("query") or obj.get("text") or "")
                dataset_queries.append({"dataset": spec.slug, "qid": f"{spec.slug}::{qid}", "query": text})
        if max_queries_per_dataset > 0:
            dataset_queries = dataset_queries[:max_queries_per_dataset]
        queries.extend(dataset_queries)
    return queries


def sanitize_query(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9]+", " ", str(text))
    text = re.sub(r"\s+", " ", text).strip()
    return text or "empty"


def percentile(values: List[float], pct: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * pct / 100.0
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] * (hi - pos) + ordered[hi] * (pos - lo)


def measure_search(
    index_dir: Path,
    queries: List[Dict[str, str]],
    hits: int,
    search_hits: int,
    repeats: int,
    warmup: int,
    strip_segment_id: bool,
) -> Dict[str, Any]:
    from pyserini.search.lucene import LuceneSearcher

    searcher = LuceneSearcher(str(index_dir))
    searcher.set_bm25(k1=0.9, b=0.4)
    for query in queries[: min(warmup, len(queries))]:
        searcher.search(query["query"], k=search_hits, strip_segment_id=strip_segment_id, remove_dups=True)
    latencies = []
    candidate_counts = []
    start_all = time.perf_counter()
    for _ in range(repeats):
        for query in queries:
            start = time.perf_counter()
            result = searcher.search(query["query"], k=search_hits, strip_segment_id=strip_segment_id, remove_dups=True)
            result = result[:hits]
            latencies.append((time.perf_counter() - start) * 1000.0)
            candidate_counts.append(len({hit.docid for hit in result}))
    total = time.perf_counter() - start_all
    searcher.close()
    return {
        "p50_ms": percentile(latencies, 50),
        "p95_ms": percentile(latencies, 95),
        "p99_ms": percentile(latencies, 99),
        "qps": len(latencies) / total if total > 0 else 0.0,
        "candidate_union": statistics.mean(candidate_counts) if candidate_counts else 0.0,
        "timed_queries": len(latencies),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure Lucene segmented-index query costs for SDE.")
    parser.add_argument("--beir_data_root", required=True)
    parser.add_argument("--sample_path", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--hits", type=int, default=1000)
    parser.add_argument("--segmented_fetch_multiplier", type=int, default=2)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup_queries", type=int, default=50)
    parser.add_argument("--max_queries_per_dataset", type=int, default=0)
    parser.add_argument("--compact_min_len", type=int, default=5)
    parser.add_argument("--compact_cap", type=int, default=24)
    parser.add_argument("--compact_unique", action="store_true")
    parser.add_argument("--reuse_indexes", action="store_true")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    system_root = ensure_dir(output_root / "system_costs" / "lucene_segmented")
    collection_root = system_root / "collections"
    index_root = ensure_dir(system_root / "indexes")
    tables_root = ensure_dir(output_root / "tables")
    raw_root = ensure_dir(output_root / "raw")
    index_log_path = raw_root / "lucene_segmented_indexing.log"
    index_log_path.write_text("", encoding="utf-8")

    sample_docs = load_sample_docs(Path(args.sample_path))
    collection_paths = build_collection_inputs(
        output_root,
        sample_docs,
        collection_root,
        compact_min_len=args.compact_min_len,
        compact_cap=args.compact_cap,
        compact_unique=args.compact_unique,
    )
    collection_dirs = {name: path.parent for name, path in collection_paths.items()}
    build_times: Dict[str, float] = {}

    index_paths = {
        "BM25": index_root / "bm25_original",
        "Doc2Query++ Full": index_root / "d2qpp_full_append",
        "SDE segmented current": index_root / "sde_segmented_current",
        "SDE segmented compact": index_root / "sde_segmented_compact",
    }
    if not args.reuse_indexes:
        for path in index_paths.values():
            if path.exists():
                shutil.rmtree(path)

    print("[lucene] building BM25 original index")
    build_times["BM25"] = (
        0.0
        if index_paths["BM25"].exists()
        else run_index(collection_dirs["original"], index_paths["BM25"], args.threads, index_log_path)
    )
    print("[lucene] building Doc2Query++ append index")
    build_times["Doc2Query++ Full"] = (
        0.0
        if index_paths["Doc2Query++ Full"].exists()
        else run_index(collection_dirs["d2qpp_append"], index_paths["Doc2Query++ Full"], args.threads, index_log_path)
    )
    print("[lucene] building SDE segmented current index")
    if not index_paths["SDE segmented current"].exists():
        build_times["SDE segmented current"] = run_index(
            collection_dirs["sde_segmented_current"],
            index_paths["SDE segmented current"],
            args.threads,
            index_log_path,
        )
    else:
        build_times["SDE segmented current"] = 0.0
    print("[lucene] building SDE segmented compact index")
    if not index_paths["SDE segmented compact"].exists():
        build_times["SDE segmented compact"] = run_index(
            collection_dirs["sde_segmented_compact"],
            index_paths["SDE segmented compact"],
            args.threads,
            index_log_path,
        )
    else:
        build_times["SDE segmented compact"] = 0.0

    queries = load_queries(Path(args.beir_data_root), args.max_queries_per_dataset)
    if not queries:
        raise SystemExit("[ERROR] no queries loaded")

    rows = []
    footprint_rows = []
    for method, index_dir in index_paths.items():
        print(f"[lucene] measuring latency method={method} queries={len(queries)} repeats={args.repeats}")
        search_hits = args.hits * args.segmented_fetch_multiplier if method.startswith("SDE segmented") else args.hits
        metrics = measure_search(
            index_dir,
            queries,
            args.hits,
            search_hits,
            args.repeats,
            args.warmup_queries,
            strip_segment_id=method.startswith("SDE segmented"),
        )
        rows.append(
            {
                "Method": method,
                "Indexing Form": {
                    "BM25": "Lucene original index",
                    "Doc2Query++ Full": "Lucene append index",
                    "SDE segmented current": "Lucene original segment + aux segment",
                    "SDE segmented compact": "Lucene original segment + compact aux segment",
                }[method],
                "P50 ms/q": f"{metrics['p50_ms']:.3f}",
                "P95 ms/q": f"{metrics['p95_ms']:.3f}",
                "P99 ms/q": f"{metrics['p99_ms']:.3f}",
                "QPS": f"{metrics['qps']:.2f}",
                "Candidate Union": f"{metrics['candidate_union']:.1f}",
                "Lucene Hits": str(search_hits),
                "Timed Queries": str(metrics["timed_queries"]),
            }
        )
        footprint_rows.append(
            {
                "Method": method,
                "Indexing Form": rows[-1]["Indexing Form"],
                "On-Disk Index MB": f"{dir_size_bytes(index_dir) / 1024.0 / 1024.0:.3f}",
                "Build sec": f"{build_times[method]:.3f}",
            }
        )

    write_csv(
        tables_root / "lucene_segmented_query_latency.csv",
        rows,
        [
            "Method",
            "Indexing Form",
            "P50 ms/q",
            "P95 ms/q",
            "P99 ms/q",
            "QPS",
            "Candidate Union",
            "Lucene Hits",
            "Timed Queries",
        ],
    )
    write_csv(
        tables_root / "lucene_segmented_index_footprint.csv",
        footprint_rows,
        ["Method", "Indexing Form", "On-Disk Index MB", "Build sec"],
    )
    write_json(
        raw_root / "lucene_segmented_metadata.json",
        {
            "build_times_sec": build_times,
            "indexes": {method: str(path) for method, path in index_paths.items()},
            "queries": len(queries),
            "hits": args.hits,
            "segmented_fetch_multiplier": args.segmented_fetch_multiplier,
            "repeats": args.repeats,
            "warmup_queries": args.warmup_queries,
            "index_log_path": str(index_log_path),
            "compact_min_len": args.compact_min_len,
            "compact_cap": args.compact_cap,
            "compact_unique": args.compact_unique,
        },
    )
    print(f"[lucene] wrote {tables_root / 'lucene_segmented_query_latency.csv'}")
    print(f"[lucene] wrote {tables_root / 'lucene_segmented_index_footprint.csv'}")


if __name__ == "__main__":
    main()
