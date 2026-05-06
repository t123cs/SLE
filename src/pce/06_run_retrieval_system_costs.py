#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from common import DATASET_SPECS, clean_id_for_path, ensure_dir, read_jsonl, write_csv, write_json


def ensure_java_on_path() -> None:
    python_bin_dir = Path(sys.executable).resolve().parent
    os.environ["PATH"] = f"{python_bin_dir}{os.pathsep}{os.environ.get('PATH', '')}"


def require_pyterrier():
    ensure_java_on_path()
    import pandas as pd
    import pyterrier as pt

    if hasattr(pt, "java"):
        if not pt.java.started():
            pt.java.init()
    elif not pt.started():
        pt.init()
    return pt, pd


def docno(dataset: str, doc_id: str) -> str:
    return f"{dataset}::{doc_id}"


def load_sample_docs(sample_path: Path) -> List[Dict[str, str]]:
    docs = []
    for row in read_jsonl(sample_path):
        docs.append(
            {
                "dataset": str(row["dataset"]),
                "doc_id": str(row["doc_id"]),
                "docno": docno(str(row["dataset"]), str(row["doc_id"])),
                "text": str(row.get("text") or "empty"),
            }
        )
    return docs


def load_d2qpp_append_docs(output_root: Path, sample_docs: List[Dict[str, str]]) -> List[Dict[str, str]]:
    root = output_root / "artifacts" / "d2qpp" / "full" / "expanded_docs"
    out = []
    missing = []
    for doc in sample_docs:
        path = root / doc["dataset"] / f"{clean_id_for_path(doc['doc_id'])}.txt"
        if not path.is_file():
            missing.append(str(path))
            continue
        out.append({**doc, "text": path.read_text(encoding="utf-8") or "empty"})
    if missing:
        raise FileNotFoundError(f"Missing D2Q++ expanded docs, first missing path: {missing[0]}")
    return out


def load_sde_aux_docs(output_root: Path, sample_docs: List[Dict[str, str]]) -> List[Dict[str, str]]:
    root = output_root / "artifacts" / "sde" / "expansion_entries"
    out = []
    missing = []
    for doc in sample_docs:
        path = root / doc["dataset"] / f"{clean_id_for_path(doc['doc_id'])}.txt"
        if not path.is_file():
            missing.append(str(path))
            continue
        text = path.read_text(encoding="utf-8").strip() or "empty"
        out.append({**doc, "text": text})
    if missing:
        raise FileNotFoundError(f"Missing SDE expansion entries, first missing path: {missing[0]}")
    return out


def parse_qrels(path: Path) -> set[str]:
    qids = set()
    with path.open("r", encoding="utf-8") as fin:
        for raw in fin:
            parts = raw.strip().split()
            if parts:
                qids.add(parts[0])
    return qids


def sanitize_query(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9]+", " ", str(text))
    text = re.sub(r"\s+", " ", text).strip()
    return text or "empty"


def load_queries(beir_data_root: Path, max_queries_per_dataset: int = 0) -> List[Dict[str, str]]:
    queries: List[Dict[str, str]] = []
    for spec in DATASET_SPECS:
        split_root = beir_data_root / spec.slug / spec.split
        qrels_path = split_root / f"qrels.{spec.split}.tsv"
        queries_path = split_root / "queries.jsonl"
        qids_with_qrels = parse_qrels(qrels_path)
        dataset_queries = []
        with queries_path.open("r", encoding="utf-8") as fin:
            for raw in fin:
                if not raw.strip():
                    continue
                obj = json.loads(raw)
                qid = str(obj.get("qid") or obj.get("_id") or obj.get("id") or "")
                if not qid or qid not in qids_with_qrels:
                    continue
                query = sanitize_query(obj.get("question") or obj.get("query") or obj.get("text") or "")
                if query:
                    dataset_queries.append({"dataset": spec.slug, "qid": f"{spec.slug}::{qid}", "query": query})
        if max_queries_per_dataset > 0:
            dataset_queries = dataset_queries[:max_queries_per_dataset]
        queries.extend(dataset_queries)
    return queries


def index_ref_for(pt: Any, index_dir: Path):
    data_properties = index_dir / "data.properties"
    if data_properties.is_file():
        return pt.IndexRef.of(str(data_properties))
    return None


def build_index(pt: Any, pd: Any, index_dir: Path, docs: List[Dict[str, str]], reuse_indexes: bool) -> Tuple[Any, float]:
    existing = index_ref_for(pt, index_dir) if reuse_indexes else None
    if existing is not None:
        return existing, 0.0
    if index_dir.exists():
        shutil.rmtree(index_dir)
    ensure_dir(index_dir)
    df = pd.DataFrame({"docno": [d["docno"] for d in docs], "text": [d["text"] for d in docs]})
    start = time.perf_counter()
    indexer = pt.DFIndexer(str(index_dir), overwrite=True)
    ref = indexer.index(df["text"], df["docno"])
    return ref, time.perf_counter() - start


def read_properties(index_dir: Path) -> Dict[str, str]:
    props = {}
    props_path = index_dir / "data.properties"
    if props_path.is_file():
        with props_path.open("r", encoding="utf-8") as fin:
            for raw in fin:
                line = raw.strip()
                if line and "=" in line:
                    key, value = line.split("=", 1)
                    props[key.strip()] = value.strip()
    return props


def safe_int(props: Dict[str, str], key: str) -> int:
    try:
        return int(props.get(key, "0"))
    except Exception:
        return 0


def dir_size_bytes(index_dir: Path) -> int:
    size = 0
    for path in index_dir.rglob("*"):
        if path.is_file():
            size += path.stat().st_size
    return size


def index_stats(index_dir: Path) -> Dict[str, Any]:
    props = read_properties(index_dir)
    docs = safe_int(props, "num.Documents")
    tokens = safe_int(props, "num.Tokens")
    return {
        "index_dir": str(index_dir),
        "size_bytes": dir_size_bytes(index_dir),
        "num_documents": docs,
        "num_terms": safe_int(props, "num.Terms"),
        "num_pointers": safe_int(props, "num.Pointers"),
        "num_tokens": tokens,
        "avg_doc_len": (tokens / docs) if docs else 0.0,
    }


def build_retriever(pt: Any, index_ref: Any, topk: int, k1: float, b: float):
    controls = {"bm25.k_1": str(k1), "bm25.b": str(b)}
    try:
        return pt.terrier.Retriever(index_ref, wmodel="BM25", num_results=topk, controls=controls)
    except TypeError:
        return pt.terrier.Retriever(index_ref, wmodel="BM25", num_results=topk, properties=controls)


def percentile(values: List[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * (pct / 100.0)
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return ordered[int(pos)]
    return ordered[lo] * (hi - pos) + ordered[hi] * (pos - lo)


def run_single_retrieval(retriever: Any, pd: Any, query: Dict[str, str]) -> Tuple[float, int]:
    frame = pd.DataFrame([{"qid": query["qid"], "query": query["query"]}])
    start = time.perf_counter()
    result = retriever.transform(frame)
    elapsed = time.perf_counter() - start
    candidate_count = int(result["docno"].nunique()) if "docno" in result else int(len(result))
    return elapsed, candidate_count


def run_sde_retrieval(text_retriever: Any, aux_retriever: Any, pd: Any, query: Dict[str, str]) -> Tuple[float, int]:
    frame = pd.DataFrame([{"qid": query["qid"], "query": query["query"]}])
    start = time.perf_counter()
    text_result = text_retriever.transform(frame)
    aux_result = aux_retriever.transform(frame)
    elapsed = time.perf_counter() - start
    candidates = set()
    if "docno" in text_result:
        candidates.update(str(x) for x in text_result["docno"].tolist())
    if "docno" in aux_result:
        candidates.update(str(x) for x in aux_result["docno"].tolist())
    return elapsed, len(candidates)


def summarize_latency(latencies_sec: List[float], candidate_counts: List[int], total_time_sec: float) -> Dict[str, float]:
    count = len(latencies_sec)
    return {
        "p50_ms": percentile([x * 1000.0 for x in latencies_sec], 50),
        "p95_ms": percentile([x * 1000.0 for x in latencies_sec], 95),
        "p99_ms": percentile([x * 1000.0 for x in latencies_sec], 99),
        "qps": (count / total_time_sec) if total_time_sec > 0 else 0.0,
        "candidate_union": statistics.mean(candidate_counts) if candidate_counts else 0.0,
        "queries": count,
    }


def measure_latency(
    pd: Any,
    queries: List[Dict[str, str]],
    methods: Dict[str, Dict[str, Any]],
    repeats: int,
    warmup_queries: int,
    raw_latency_path: Path,
) -> List[Dict[str, Any]]:
    raw_latency_path.write_text("", encoding="utf-8")
    summaries = []
    warmup = queries[: max(0, min(warmup_queries, len(queries)))]
    for method_name, cfg in methods.items():
        print(f"[latency] method={method_name} warmup_queries={len(warmup)} timed_queries={len(queries)} repeats={repeats}")
        for query in warmup:
            if cfg["kind"] == "dual":
                run_sde_retrieval(cfg["text_retriever"], cfg["aux_retriever"], pd, query)
            else:
                run_single_retrieval(cfg["retriever"], pd, query)

        latencies: List[float] = []
        candidate_counts: List[int] = []
        total_start = time.perf_counter()
        with raw_latency_path.open("a", encoding="utf-8") as fout:
            for repeat in range(repeats):
                for query in queries:
                    if cfg["kind"] == "dual":
                        elapsed, candidates = run_sde_retrieval(cfg["text_retriever"], cfg["aux_retriever"], pd, query)
                    else:
                        elapsed, candidates = run_single_retrieval(cfg["retriever"], pd, query)
                    latencies.append(elapsed)
                    candidate_counts.append(candidates)
                    fout.write(
                        json.dumps(
                            {
                                "method": method_name,
                                "repeat": repeat,
                                "qid": query["qid"],
                                "dataset": query["dataset"],
                                "latency_ms": elapsed * 1000.0,
                                "candidate_union": candidates,
                            },
                            ensure_ascii=False,
                            sort_keys=True,
                        )
                        + "\n"
                    )
        total_time = time.perf_counter() - total_start
        summaries.append({"method": method_name, **summarize_latency(latencies, candidate_counts, total_time)})
    return summaries


def mean_kb_from_doc_log(path: Path, key: str) -> float:
    rows = [row for row in read_jsonl(path) if row.get("status") == "ok"]
    if not rows:
        return 0.0
    return sum(float(row.get(key) or 0.0) for row in rows) / len(rows) / 1024.0


def format_float(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build retrieval indexes and measure CPU-only query-time costs.")
    parser.add_argument("--beir_data_root", required=True)
    parser.add_argument("--sample_path", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--topk", type=int, default=1000)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup_queries", type=int, default=50)
    parser.add_argument("--max_queries_per_dataset", type=int, default=0)
    parser.add_argument("--k1", type=float, default=0.9)
    parser.add_argument("--b", type=float, default=0.4)
    parser.add_argument("--reuse_indexes", action="store_true")
    parser.add_argument("--skip_latency", action="store_true")
    args = parser.parse_args()

    pt, pd = require_pyterrier()
    output_root = Path(args.output_root)
    system_root = ensure_dir(output_root / "system_costs")
    indexes_root = ensure_dir(system_root / "indexes")
    tables_root = ensure_dir(output_root / "tables")
    raw_root = ensure_dir(output_root / "raw")

    sample_docs = load_sample_docs(Path(args.sample_path))
    d2qpp_docs = load_d2qpp_append_docs(output_root, sample_docs)
    sde_aux_docs = load_sde_aux_docs(output_root, sample_docs)
    queries = load_queries(Path(args.beir_data_root), max_queries_per_dataset=args.max_queries_per_dataset)
    if not queries:
        raise SystemExit("[ERROR] no queries loaded for latency measurement")

    build_times: Dict[str, float] = {}
    print(f"[index] documents={len(sample_docs)} queries={len(queries)} topk={args.topk}")
    bm25_ref, build_times["bm25_original"] = build_index(
        pt, pd, indexes_root / "bm25_original", sample_docs, args.reuse_indexes
    )
    d2qpp_ref, build_times["d2qpp_full_append"] = build_index(
        pt, pd, indexes_root / "d2qpp_full_append", d2qpp_docs, args.reuse_indexes
    )
    sde_aux_ref, build_times["sde_aux"] = build_index(pt, pd, indexes_root / "sde_aux", sde_aux_docs, args.reuse_indexes)

    stats = {
        "bm25_original": index_stats(indexes_root / "bm25_original"),
        "d2qpp_full_append": index_stats(indexes_root / "d2qpp_full_append"),
        "sde_aux": index_stats(indexes_root / "sde_aux"),
    }
    write_json(raw_root / "retrieval_system_index_stats.json", {"build_times_sec": build_times, "index_stats": stats})

    bm25_size = max(float(stats["bm25_original"]["size_bytes"]), 1.0)
    d2qpp_expansion_kb = mean_kb_from_doc_log(raw_root / "d2qpp_full_doc_logs.jsonl", "decoded_text_bytes")
    sde_expansion_kb = mean_kb_from_doc_log(raw_root / "sde_trace_doc_logs.jsonl", "expansion_entry_bytes")
    sde_total = {
        "size_bytes": stats["bm25_original"]["size_bytes"] + stats["sde_aux"]["size_bytes"],
        "num_terms": stats["bm25_original"]["num_terms"] + stats["sde_aux"]["num_terms"],
        "num_pointers": stats["bm25_original"]["num_pointers"] + stats["sde_aux"]["num_pointers"],
        "num_tokens": stats["bm25_original"]["num_tokens"] + stats["sde_aux"]["num_tokens"],
        "num_documents": stats["bm25_original"]["num_documents"] + stats["sde_aux"]["num_documents"],
    }
    sde_total["avg_doc_len"] = (
        sde_total["num_tokens"] / sde_total["num_documents"] if sde_total["num_documents"] else 0.0
    )

    footprint_rows = []
    for label, form, st, expansion_kb in (
        ("BM25", "original index", stats["bm25_original"], 0.0),
        ("Doc2Query++ Full", "append index", stats["d2qpp_full_append"], d2qpp_expansion_kb),
        ("SDE text index", "original text index", stats["bm25_original"], 0.0),
        ("SDE aux index", "aux expansion index", stats["sde_aux"], sde_expansion_kb),
        ("SDE total", "text + aux index", sde_total, sde_expansion_kb),
    ):
        footprint_rows.append(
            {
                "Method": label,
                "Indexing Form": form,
                "On-Disk Index MB": format_float(float(st["size_bytes"]) / (1024.0 * 1024.0), 3),
                "Ratio vs BM25": f"{(float(st['size_bytes']) / bm25_size):.3f}x",
                "Vocab": str(int(st["num_terms"])),
                "Postings": str(int(st["num_pointers"])),
                "Avg Doc Len": format_float(float(st["avg_doc_len"]), 2),
                "Expansion Text KB/doc": format_float(expansion_kb, 3),
            }
        )
    write_csv(
        tables_root / "retrieval_index_footprint.csv",
        footprint_rows,
        [
            "Method",
            "Indexing Form",
            "On-Disk Index MB",
            "Ratio vs BM25",
            "Vocab",
            "Postings",
            "Avg Doc Len",
            "Expansion Text KB/doc",
        ],
    )

    rebuild_rows = [
        {
            "Method": "BM25",
            "Build Text/Main Index sec": format_float(build_times["bm25_original"]),
            "Build Aux/Expansion Index sec": "-",
            "Total Rebuild sec": format_float(build_times["bm25_original"]),
            "Refresh Expansion Only sec": "-",
        },
        {
            "Method": "Doc2Query++ Full",
            "Build Text/Main Index sec": format_float(build_times["d2qpp_full_append"]),
            "Build Aux/Expansion Index sec": "-",
            "Total Rebuild sec": format_float(build_times["d2qpp_full_append"]),
            "Refresh Expansion Only sec": "no",
        },
        {
            "Method": "SDE",
            "Build Text/Main Index sec": format_float(build_times["bm25_original"]),
            "Build Aux/Expansion Index sec": format_float(build_times["sde_aux"]),
            "Total Rebuild sec": format_float(build_times["bm25_original"] + build_times["sde_aux"]),
            "Refresh Expansion Only sec": format_float(build_times["sde_aux"]),
        },
    ]
    write_csv(
        tables_root / "retrieval_rebuild_cost.csv",
        rebuild_rows,
        [
            "Method",
            "Build Text/Main Index sec",
            "Build Aux/Expansion Index sec",
            "Total Rebuild sec",
            "Refresh Expansion Only sec",
        ],
    )

    if not args.skip_latency:
        bm25_retriever = build_retriever(pt, bm25_ref, args.topk, args.k1, args.b)
        d2qpp_retriever = build_retriever(pt, d2qpp_ref, args.topk, args.k1, args.b)
        sde_aux_retriever = build_retriever(pt, sde_aux_ref, args.topk, args.k1, args.b)
        latency_summaries = measure_latency(
            pd,
            queries,
            {
                "BM25": {"kind": "single", "retriever": bm25_retriever},
                "Doc2Query++ Full": {"kind": "single", "retriever": d2qpp_retriever},
                "SDE": {"kind": "dual", "text_retriever": bm25_retriever, "aux_retriever": sde_aux_retriever},
            },
            repeats=args.repeats,
            warmup_queries=args.warmup_queries,
            raw_latency_path=raw_root / "retrieval_latency_runs.jsonl",
        )
        latency_rows = []
        forms = {
            "BM25": "original index",
            "Doc2Query++ Full": "append index",
            "SDE": "text + aux index",
        }
        for row in latency_summaries:
            latency_rows.append(
                {
                    "Method": row["method"],
                    "Indexing Form": forms[row["method"]],
                    "P50 ms/q": format_float(row["p50_ms"], 3),
                    "P95 ms/q": format_float(row["p95_ms"], 3),
                    "P99 ms/q": format_float(row["p99_ms"], 3),
                    "QPS": format_float(row["qps"], 2),
                    "Candidate Union": format_float(row["candidate_union"], 1),
                    "Timed Queries": str(int(row["queries"])),
                }
            )
        write_csv(
            tables_root / "retrieval_query_latency.csv",
            latency_rows,
            [
                "Method",
                "Indexing Form",
                "P50 ms/q",
                "P95 ms/q",
                "P99 ms/q",
                "QPS",
                "Candidate Union",
                "Timed Queries",
            ],
        )

    print(f"[system-costs] wrote {tables_root / 'retrieval_index_footprint.csv'}")
    print(f"[system-costs] wrote {tables_root / 'retrieval_rebuild_cost.csv'}")
    if not args.skip_latency:
        print(f"[system-costs] wrote {tables_root / 'retrieval_query_latency.csv'}")


if __name__ == "__main__":
    main()
