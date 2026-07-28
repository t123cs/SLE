#!/usr/bin/env python3
"""Measure SDE auxiliary-index serving.

The script keeps the verified non-SDE rows from the formal benchmark, then
measures the factorized query-document index under the same single-core
latency and equal-two-core throughput protocol.
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import time
from pathlib import Path
from queue import Empty
from typing import Any, Sequence

import pandas as pd

import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from src.pce.system_efficiency import measure_system_efficiency_formal as formal  # noqa: E402
from src.sde.factorized_qdoc import factorized_qdoc_index as factor  # noqa: E402


DATASETS = ["nfcorpus", "scidocs", "fiqa-2018", "arguana", "scifact"]
FUSION_ALPHA = 0.5
SEQUENTIAL = "factorized_qdoc_sequential"
PARALLEL = "factorized_qdoc_process_parallel"
DISPLAY_NAMES = {
    SEQUENTIAL: "SDE, auxiliary index, sequential",
    PARALLEL: "SDE, auxiliary index, parallel",
}
ANCHOR_DISPLAY_NAMES = {
    "bm25": "BM25",
    "d2qpp_full": "D2Q++ Full",
    "d2qpp_qgen_only": "D2Q++ QGen-only",
}
DEFAULT_FORMAL_ROOT = REPO_ROOT / "results" / "system_efficiency_formal"
DEFAULT_FACTOR_ROOT = (
    REPO_ROOT
    / "results"
    / "factorized_qdoc_online_index"
    / "factor_indexes"
    / "repeat_0"
)
DEFAULT_OUT_ROOT = REPO_ROOT / "results" / "factorized_qdoc_online"


def now() -> float:
    return time.perf_counter()


def directory_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def factorized_index_size(path: Path) -> int:
    return sum(
        (path / name).stat().st_size
        for name in [
            factor.ARRAYS_FILE,
            factor.TERMS_FILE,
            factor.DOCNOS_FILE,
            factor.MANIFEST_FILE,
        ]
    )


def text_index_path(formal_root: Path, dataset: str) -> Path:
    manifest = json.loads(
        (formal_root / "manifest.json").read_text(encoding="utf-8")
    )
    repeat = int(manifest.get("build_repeats", 3)) - 1
    return formal_root / "indexes" / f"repeat_{repeat}" / dataset / "bm25"


def factor_index_path(root: Path, dataset: str) -> Path:
    return root / dataset


class FactorizedRuntime:
    def __init__(self, text_index: Path, factor_index: Path) -> None:
        self.text = formal.retriever_for(text_index, 300, 0.9, 0.4)
        self.factor = factor.FactorizedQDocIndex(factor_index)
        self.processor = factor.QueryProcessor()
        source_count = int(self.factor.qdoc_source.max()) + 1
        self.source_docnos = [""] * source_count
        for doc_id, qdocno in enumerate(self.factor.qdoc_docnos):
            source_id = int(self.factor.qdoc_source[doc_id])
            if not self.source_docnos[source_id]:
                self.source_docnos[source_id] = factor.source_docno(qdocno)


def minmax_scores(scores: dict[Any, float]) -> dict[Any, float]:
    if not scores:
        return {}
    low = min(scores.values())
    high = max(scores.values())
    denominator = high - low
    if abs(denominator) < 1e-12:
        return {key: 0.0 for key in scores}
    return {key: (score - low) / denominator for key, score in scores.items()}


def fuse_factorized(
    text_results: Sequence[tuple[str, float]],
    qdoc_results: Sequence[tuple[int, str, float]],
    qdoc_source: Any,
    source_docnos: Sequence[str],
) -> tuple[list[tuple[str, float]], dict[str, float]]:
    start = now()
    aggregated: dict[int, float] = {}
    hit_counts: dict[int, int] = {}
    for doc_id, _, score in qdoc_results:
        source_id = int(qdoc_source[int(doc_id)])
        position = hit_counts.get(source_id, 0)
        aggregated[source_id] = (
            aggregated.get(source_id, 0.0) + float(score) * (0.3**position)
        )
        hit_counts[source_id] = position + 1
    aggregation_sec = now() - start

    start = now()
    text_norm = minmax_scores(dict(text_results))
    query_norm = minmax_scores(aggregated)
    fused = {
        docno: (1.0 - FUSION_ALPHA) * score
        for docno, score in text_norm.items()
    }
    for source_id, score in query_norm.items():
        docno = source_docnos[source_id]
        fused[docno] = fused.get(docno, 0.0) + FUSION_ALPHA * score
    ranking = sorted(fused.items(), key=lambda item: (-item[1], item[0]))
    fusion_sec = now() - start
    return ranking, {
        "aggregation_sec": aggregation_sec,
        "fusion_sec": fusion_sec,
    }


def run_sequential(
    runtime: FactorizedRuntime,
    item: dict[str, str],
) -> dict[str, Any]:
    cpu_start = time.process_time()
    start = now()
    branch_start = now()
    text_results = formal.maybe_transform(runtime.text, item["qid"], item["query"])
    text_sec = now() - branch_start
    branch_start = now()
    query_terms = runtime.processor.terms(item["query"])
    qdoc_results = runtime.factor.score_terms(query_terms, 1000)
    qdoc_sec = now() - branch_start
    ranking, components = fuse_factorized(
        text_results,
        qdoc_results,
        runtime.factor.qdoc_source,
        runtime.source_docnos,
    )
    return {
        "e2e_sec": now() - start,
        "cpu_sec": time.process_time() - cpu_start,
        "text_sec": text_sec,
        "aux_sec": qdoc_sec,
        "candidate_union": len(ranking),
        **components,
    }


def summarize(
    task: dict[str, Any],
    rows: list[dict[str, Any]],
    elapsed: float,
) -> dict[str, Any]:
    stats = formal.summarize_ms([float(row["e2e_sec"]) for row in rows])
    summary: dict[str, Any] = {
        "dataset": task["dataset"],
        "method": task["method"],
        "display_name": DISPLAY_NAMES[task["method"]],
        "queries": int(task["query_count"]),
        "repeats": int(task["repeats"]),
        "requests": len(rows),
        "qps": len(rows) / elapsed,
        "e2e_mean_ms": stats["mean_ms"],
        "e2e_p50_ms": stats["p50_ms"],
        "e2e_p95_ms": stats["p95_ms"],
        "e2e_p99_ms": stats["p99_ms"],
        "core_ms_per_query": (
            sum(float(row["cpu_sec"]) for row in rows) * 1000.0 / len(rows)
        ),
    }
    for component in [
        "text_sec",
        "aux_sec",
        "aggregation_sec",
        "fusion_sec",
        "ipc_scheduler_sec",
    ]:
        values = [float(row[component]) for row in rows if component in row]
        if values:
            key = component.replace("_sec", "")
            summary[f"{key}_mean_ms"] = sum(values) * 1000.0 / len(values)
            summary[f"{key}_p95_ms"] = formal.percentile(
                [value * 1000.0 for value in values], 95
            )
    summary["effective_cpu_cores"] = (
        summary["core_ms_per_query"] / summary["e2e_mean_ms"]
    )
    return summary


def latency_sequential_worker(task: dict[str, Any], result_queue: Any) -> None:
    try:
        formal.set_affinity(task["cpu_ids"][:1])
        factor.sde.ensure_pyterrier()
        runtime = FactorizedRuntime(
            Path(task["text_index"]),
            Path(task["factor_index"]),
        )
        queries = formal.query_items(
            Path(task["data_root"]),
            task["dataset"],
            task["split"],
            task["max_queries"],
        )
        task["query_count"] = len(queries)
        for item in queries[: min(len(queries), task["warmup_queries"])]:
            run_sequential(runtime, item)
        warm_rss = formal.rss_mb()
        rows: list[dict[str, Any]] = []
        start_all = now()
        with formal.PeakRSS() as monitor:
            for repeat in range(task["repeats"]):
                for item in queries:
                    row = run_sequential(runtime, item)
                    row.update(
                        {
                            "dataset": task["dataset"],
                            "method": task["method"],
                            "repeat": repeat,
                            "qid": item["qid"],
                        }
                    )
                    rows.append(row)
        elapsed = now() - start_all
        summary = summarize(task, rows, elapsed)
        summary.update(
            {
                "cpu_budget": 1,
                "retrieval_workers": 1,
                "service_processes": 1,
                "warm_rss_mb": warm_rss,
                "peak_rss_mb": monitor.peak_mb,
                "parallelism": "single process; text then factorized qdoc",
            }
        )
        formal.write_jsonl(Path(task["raw_path"]), rows)
        result_queue.put({"ok": True, "summary": summary})
    except Exception as exc:
        result_queue.put({"ok": False, "error": repr(exc), "task": task})


def branch_worker(
    name: str,
    task: dict[str, Any],
    cpu_id: int,
    request_queue: Any,
    response_queue: Any,
) -> None:
    try:
        formal.set_affinity([cpu_id])
        factor.sde.ensure_pyterrier()
        if name == "text":
            retriever = formal.retriever_for(
                Path(task["text_index"]), 300, 0.9, 0.4
            )
            factor_index = None
            processor = None
        else:
            retriever = None
            factor_index = factor.FactorizedQDocIndex(Path(task["factor_index"]))
            processor = factor.QueryProcessor()
        response_queue.put(
            {
                "type": "ready",
                "worker": name,
                "rss_mb": formal.rss_mb(),
            }
        )
        while True:
            item = request_queue.get()
            if item is None:
                return
            cpu_start = time.process_time()
            start = now()
            if name == "text":
                results = formal.maybe_transform(
                    retriever, str(item["qid"]), str(item["query"])
                )
            else:
                results = factor_index.score_terms(
                    processor.terms(str(item["query"])), 1000
                )
            response_queue.put(
                {
                    "type": "result",
                    "worker": name,
                    "request_id": int(item["request_id"]),
                    "wall_sec": now() - start,
                    "cpu_sec": time.process_time() - cpu_start,
                    "results": results,
                    "rss_mb": formal.rss_mb(),
                    "peak_rss_mb": formal.max_rss_mb(),
                }
            )
    except Exception as exc:
        response_queue.put(
            {"type": "error", "worker": name, "error": repr(exc)}
        )


def collect(
    response_queue: Any,
    expected: set[str],
    request_id: int | None = None,
) -> dict[str, dict[str, Any]]:
    found: dict[str, dict[str, Any]] = {}
    deadline = time.time() + 300.0
    while set(found) != expected:
        message = response_queue.get(timeout=max(0.1, deadline - time.time()))
        if message.get("type") == "error":
            raise RuntimeError(message)
        if request_id is None and message.get("type") == "ready":
            found[str(message["worker"])] = message
        elif request_id is not None and message.get("type") == "result":
            if int(message["request_id"]) != request_id:
                raise RuntimeError(f"Unexpected request id: {message}")
            found[str(message["worker"])] = message
    return found


def latency_parallel_worker(task: dict[str, Any], result_queue: Any) -> None:
    cpu_ids = list(task["cpu_ids"][:2])
    ctx = mp.get_context("spawn")
    response_queue = ctx.Queue()
    queues = {"text": ctx.Queue(), "qdoc": ctx.Queue()}
    workers = [
        ctx.Process(
            target=branch_worker,
            args=(name, task, cpu_id, queues[name], response_queue),
        )
        for name, cpu_id in zip(["text", "qdoc"], cpu_ids)
    ]
    try:
        formal.set_affinity([cpu_ids[0]])
        for worker in workers:
            worker.start()
        ready = collect(response_queue, {"text", "qdoc"})
        factor_index = factor.FactorizedQDocIndex(Path(task["factor_index"]))
        source_count = int(factor_index.qdoc_source.max()) + 1
        source_docnos = [""] * source_count
        for doc_id, qdocno in enumerate(factor_index.qdoc_docnos):
            source_id = int(factor_index.qdoc_source[doc_id])
            if not source_docnos[source_id]:
                source_docnos[source_id] = factor.source_docno(qdocno)
        queries = formal.query_items(
            Path(task["data_root"]),
            task["dataset"],
            task["split"],
            task["max_queries"],
        )
        task["query_count"] = len(queries)
        request_id = 0
        worker_peaks = {
            name: float(message["rss_mb"]) for name, message in ready.items()
        }

        def run_one(item: dict[str, str]) -> dict[str, Any]:
            nonlocal request_id
            request_id += 1
            payload = {"request_id": request_id, **item}
            coordinator_cpu_start = time.process_time()
            start = now()
            queues["text"].put(payload)
            queues["qdoc"].put(payload)
            received = collect(
                response_queue, {"text", "qdoc"}, request_id=request_id
            )
            receive_sec = now() - start
            text_result = received["text"]
            qdoc_result = received["qdoc"]
            ranking, components = fuse_factorized(
                text_result["results"],
                qdoc_result["results"],
                factor_index.qdoc_source,
                source_docnos,
            )
            e2e_sec = now() - start
            for name, result in received.items():
                worker_peaks[name] = max(
                    worker_peaks[name],
                    float(result["rss_mb"]),
                    float(result["peak_rss_mb"]),
                )
            ipc_sec = max(
                0.0,
                receive_sec
                - max(
                    float(text_result["wall_sec"]),
                    float(qdoc_result["wall_sec"]),
                ),
            )
            coordinator_cpu = time.process_time() - coordinator_cpu_start
            return {
                "e2e_sec": e2e_sec,
                "cpu_sec": (
                    coordinator_cpu
                    + float(text_result["cpu_sec"])
                    + float(qdoc_result["cpu_sec"])
                ),
                "text_sec": float(text_result["wall_sec"]),
                "aux_sec": float(qdoc_result["wall_sec"]),
                "ipc_scheduler_sec": ipc_sec,
                "candidate_union": len(ranking),
                **components,
            }

        for item in queries[: min(len(queries), task["warmup_queries"])]:
            run_one(item)
        warm_rss = formal.rss_mb() + sum(
            float(message["rss_mb"]) for message in ready.values()
        )
        rows: list[dict[str, Any]] = []
        start_all = now()
        with formal.PeakRSS() as monitor:
            for repeat in range(task["repeats"]):
                for item in queries:
                    row = run_one(item)
                    row.update(
                        {
                            "dataset": task["dataset"],
                            "method": task["method"],
                            "repeat": repeat,
                            "qid": item["qid"],
                        }
                    )
                    rows.append(row)
        elapsed = now() - start_all
        summary = summarize(task, rows, elapsed)
        summary.update(
            {
                "cpu_budget": 2,
                "retrieval_workers": 2,
                "service_processes": 3,
                "warm_rss_mb": warm_rss,
                "peak_rss_mb": monitor.peak_mb + sum(worker_peaks.values()),
                "parallelism": "text and factorized qdoc process workers",
            }
        )
        formal.write_jsonl(Path(task["raw_path"]), rows)
        result_queue.put({"ok": True, "summary": summary})
    except Exception as exc:
        result_queue.put({"ok": False, "error": repr(exc), "task": task})
    finally:
        for queue in queues.values():
            queue.put(None)
        for worker in workers:
            worker.join(timeout=20)
            if worker.is_alive():
                worker.terminate()
                worker.join(timeout=5)


def pool_worker(
    name: str,
    task: dict[str, Any],
    cpu_id: int,
    request_queue: Any,
    response_queue: Any,
) -> None:
    try:
        formal.set_affinity([cpu_id])
        factor.sde.ensure_pyterrier()
        runtime = FactorizedRuntime(
            Path(task["text_index"]), Path(task["factor_index"])
        )
        response_queue.put(
            {"type": "ready", "worker": name, "rss_mb": formal.rss_mb()}
        )
        while True:
            item = request_queue.get()
            if item is None:
                return
            start = now()
            row = run_sequential(runtime, item)
            response_queue.put(
                {
                    "type": "result",
                    "worker": name,
                    "request_id": int(item["request_id"]),
                    "service_sec": now() - start,
                    "cpu_sec": row["cpu_sec"],
                    "candidate_union": row["candidate_union"],
                    "rss_mb": formal.rss_mb(),
                    "peak_rss_mb": formal.max_rss_mb(),
                }
            )
    except Exception as exc:
        response_queue.put(
            {"type": "error", "worker": name, "error": repr(exc)}
        )


def throughput_worker(task: dict[str, Any], result_queue: Any) -> None:
    cpu_ids = list(task["cpu_ids"][:2])
    ctx = mp.get_context("spawn")
    response_queue = ctx.Queue()
    queues = [ctx.Queue(), ctx.Queue()]
    names = ["worker0", "worker1"]
    workers = [
        ctx.Process(
            target=pool_worker,
            args=(name, task, cpu_id, queue, response_queue),
        )
        for name, cpu_id, queue in zip(names, cpu_ids, queues)
    ]
    try:
        formal.set_affinity([cpu_ids[0]])
        for worker in workers:
            worker.start()
        ready = collect(response_queue, set(names))
        queries = formal.query_items(
            Path(task["data_root"]),
            task["dataset"],
            task["split"],
            task["max_queries"],
        )
        request_id = 0
        warm_count = min(len(queries), task["warmup_queries"])
        for idx, item in enumerate(queries[:warm_count]):
            request_id += 1
            queues[idx % 2].put({"request_id": request_id, **item})
            message = response_queue.get(timeout=300)
            if message.get("type") == "error":
                raise RuntimeError(message)

        rows: list[dict[str, Any]] = []
        repeat_elapsed: list[float] = []
        worker_peaks = {
            name: float(message["rss_mb"]) for name, message in ready.items()
        }
        for repeat in range(task["repeats"]):
            pending: dict[int, tuple[float, dict[str, str]]] = {}
            next_index = 0
            repeat_start = now()
            for worker_index in range(min(2, len(queries))):
                request_id += 1
                item = queries[next_index]
                next_index += 1
                pending[request_id] = (now(), item)
                queues[worker_index].put({"request_id": request_id, **item})
            while pending:
                message = response_queue.get(timeout=300)
                if message.get("type") == "error":
                    raise RuntimeError(message)
                rid = int(message["request_id"])
                submitted, item = pending.pop(rid)
                worker_name = str(message["worker"])
                worker_index = names.index(worker_name)
                worker_peaks[worker_name] = max(
                    worker_peaks[worker_name],
                    float(message["rss_mb"]),
                    float(message["peak_rss_mb"]),
                )
                rows.append(
                    {
                        "dataset": task["dataset"],
                        "method": task["method"],
                        "repeat": repeat,
                        "qid": item["qid"],
                        "e2e_sec": now() - submitted,
                        "service_sec": float(message["service_sec"]),
                        "cpu_sec": float(message["cpu_sec"]),
                        "candidate_union": int(message["candidate_union"]),
                    }
                )
                if next_index < len(queries):
                    request_id += 1
                    item = queries[next_index]
                    next_index += 1
                    pending[request_id] = (now(), item)
                    queues[worker_index].put({"request_id": request_id, **item})
            repeat_elapsed.append(now() - repeat_start)
        stats = formal.summarize_ms([float(row["e2e_sec"]) for row in rows])
        summary = {
            "dataset": task["dataset"],
            "method": task["method"],
            "display_name": DISPLAY_NAMES[task["method"]],
            "queries": len(queries),
            "repeats": task["repeats"],
            "requests": len(rows),
            "fixed_core_budget": 2,
            "concurrency": 2,
            "worker_processes": 2,
            "qps": len(rows) / sum(repeat_elapsed),
            "e2e_mean_ms": stats["mean_ms"],
            "e2e_p50_ms": stats["p50_ms"],
            "e2e_p95_ms": stats["p95_ms"],
            "e2e_p99_ms": stats["p99_ms"],
            "core_ms_per_query": (
                sum(float(row["cpu_sec"]) for row in rows) * 1000.0 / len(rows)
            ),
            "worker_warm_rss_mb": sum(
                float(message["rss_mb"]) for message in ready.values()
            ),
            "peak_rss_mb": sum(worker_peaks.values()),
        }
        formal.write_jsonl(Path(task["raw_path"]), rows)
        result_queue.put({"ok": True, "summary": summary})
    except Exception as exc:
        result_queue.put({"ok": False, "error": repr(exc), "task": task})
    finally:
        for queue in queues:
            queue.put(None)
        for worker in workers:
            worker.join(timeout=20)
            if worker.is_alive():
                worker.terminate()
                worker.join(timeout=5)


def validation_worker(task: dict[str, Any], result_queue: Any) -> None:
    try:
        formal.set_affinity(task["cpu_ids"][:1])
        factor.sde.ensure_pyterrier()
        factorized_index = factor.FactorizedQDocIndex(
            Path(task["factor_index"])
        )
        structure_checks = factorized_index.structure_checks()
        processor = factor.QueryProcessor()
        queries = formal.query_items(
            Path(task["data_root"]),
            task["dataset"],
            task["split"],
            task["max_queries"],
        )
        nonfinite_score_queries = 0
        duplicate_result_queries = 0
        invalid_result_queries = 0
        for item in queries:
            results = factorized_index.score_terms(
                processor.terms(item["query"]), 1000
            )
            document_ids = [row[0] for row in results]
            document_names = [row[1] for row in results]
            if any(not math.isfinite(row[2]) for row in results):
                nonfinite_score_queries += 1
            if (
                len(document_ids) != len(set(document_ids))
                or len(document_names) != len(set(document_names))
            ):
                duplicate_result_queries += 1
            if any(
                document_id < 0
                or document_id >= factorized_index.document_count
                or factorized_index.qdoc_docnos[document_id] != document_name
                for document_id, document_name, _ in results
            ):
                invalid_result_queries += 1
        failed_checks = [
            name for name, passed in structure_checks.items() if not passed
        ]
        result_queue.put(
            {
                "ok": True,
                "summary": {
                    "dataset": task["dataset"],
                    "queries": len(queries),
                    "structure_checks_passed": not failed_checks,
                    "failed_structure_checks": json.dumps(failed_checks),
                    "nonfinite_score_queries": nonfinite_score_queries,
                    "duplicate_result_queries": duplicate_result_queries,
                    "invalid_result_queries": invalid_result_queries,
                },
            }
        )
    except Exception as exc:
        result_queue.put({"ok": False, "error": repr(exc), "task": task})


def make_task(
    args: argparse.Namespace,
    dataset: str,
    method: str,
    cpu_ids: Sequence[int],
) -> dict[str, Any]:
    return {
        "dataset": dataset,
        "method": method,
        "data_root": str(args.data_root),
        "split": "test",
        "repeats": args.repeats,
        "warmup_queries": args.warmup_queries,
        "max_queries": args.max_queries,
        "cpu_ids": list(cpu_ids),
        "text_index": str(text_index_path(args.formal_root, dataset)),
        "factor_index": str(factor_index_path(args.factor_root, dataset)),
    }


def save_frame(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, sep="\t", index=False)


def aggregate(args: argparse.Namespace) -> None:
    validation = pd.read_csv(args.out_root / "validation.tsv", sep="\t")
    latency = pd.read_csv(args.out_root / "latency_summary.tsv", sep="\t")
    throughput = pd.read_csv(args.out_root / "fixed2_throughput_summary.tsv", sep="\t")
    anchor_manifest = json.loads(
        (args.formal_root / "manifest.json").read_text(encoding="utf-8")
    )
    cpu_model = next(
        (
            line.split(":", 1)[1].strip()
            for line in anchor_manifest["lscpu"].splitlines()
            if line.startswith("Model name:")
        ),
        "unknown CPU",
    )
    footprint_rows = []
    for dataset in args.datasets:
        text_bytes = directory_size(text_index_path(args.formal_root, dataset))
        factor_bytes = factorized_index_size(
            factor_index_path(args.factor_root, dataset)
        )
        footprint_rows.append(
            {
                "dataset": dataset,
                "text_index_mb": text_bytes / (1024.0**2),
                "factorized_qdoc_mb": factor_bytes / (1024.0**2),
                "total_index_mb": (text_bytes + factor_bytes) / (1024.0**2),
            }
        )
    footprint = pd.DataFrame(footprint_rows)
    footprint.to_csv(args.out_root / "footprint.tsv", sep="\t", index=False)

    anchor = pd.read_csv(
        args.formal_root / "system_efficiency_main_table.tsv", sep="\t"
    )
    anchor = anchor[
        anchor["method"].isin(
            ["bm25", "d2qpp_full", "d2qpp_qgen_only"]
        )
    ].copy()
    table_rows = [
        {
            "method": row.method,
            "display_name": ANCHOR_DISPLAY_NAMES[row.method],
            "mean_ms": float(row.e2e_mean_ms),
            "p95_ms": float(row.e2e_p95_ms),
            "fixed2_qps": float(row.fixed2_qps),
            "index_mb": float(row.total_index_mb),
        }
        for row in anchor.itertuples(index=False)
    ]
    factor_index_mb = float(footprint["total_index_mb"].sum())
    factor_manifests = [
        json.loads(
            (
                factor_index_path(args.factor_root, dataset) / factor.MANIFEST_FILE
            ).read_text(encoding="utf-8")
        )
        for dataset in args.datasets
    ]
    factor_formats = {item["format"] for item in factor_manifests}
    factor_tokenization_modes = {
        item.get("tokenization_mode", "source Terrier direct index")
        for item in factor_manifests
    }
    for method in [SEQUENTIAL, PARALLEL]:
        method_latency = latency[latency["method"] == method]
        if method == SEQUENTIAL:
            fixed2_qps = float(
                throughput[throughput["method"] == method]["qps"].mean()
            )
        else:
            fixed2_qps = float(method_latency["qps"].mean())
        table_rows.append(
            {
                "method": method,
                "display_name": DISPLAY_NAMES[method],
                "mean_ms": float(method_latency["e2e_mean_ms"].mean()),
                "p95_ms": float(method_latency["e2e_p95_ms"].mean()),
                "fixed2_qps": fixed2_qps,
                "index_mb": factor_index_mb,
            }
        )
    table = pd.DataFrame(table_rows)
    table.to_csv(args.out_root / "table5_factorized_qdoc.tsv", sep="\t", index=False)

    component_closure: dict[str, dict[str, float | int]] = {}
    for method in [SEQUENTIAL, PARALLEL]:
        residuals_ms: list[float] = []
        for dataset in args.datasets:
            raw_path = args.out_root / "raw_latency" / f"{dataset}.{method}.jsonl"
            with raw_path.open("r", encoding="utf-8") as fin:
                for raw in fin:
                    row = json.loads(raw)
                    if method == SEQUENTIAL:
                        expected = (
                            float(row["text_sec"])
                            + float(row["aux_sec"])
                            + float(row["aggregation_sec"])
                            + float(row["fusion_sec"])
                        )
                    else:
                        expected = (
                            max(float(row["text_sec"]), float(row["aux_sec"]))
                            + float(row["ipc_scheduler_sec"])
                            + float(row["aggregation_sec"])
                            + float(row["fusion_sec"])
                        )
                    residuals_ms.append((float(row["e2e_sec"]) - expected) * 1000.0)
        absolute = [abs(value) for value in residuals_ms]
        component_closure[method] = {
            "requests": len(residuals_ms),
            "mean_residual_ms": sum(residuals_ms) / len(residuals_ms),
            "p95_absolute_residual_ms": formal.percentile(absolute, 95),
            "max_absolute_residual_ms": max(absolute),
        }
    request_counts_ok = bool(
        (
            latency["requests"]
            == latency["queries"] * latency["repeats"]
        ).all()
        and (
            throughput["requests"]
            == throughput["queries"] * throughput["repeats"]
        ).all()
    )
    anchor_validation = json.loads(
        (args.formal_root / "validation.json").read_text(encoding="utf-8")
    )
    factorized_validation_ok = bool(
        validation["structure_checks_passed"].astype(bool).all()
        and validation["nonfinite_score_queries"].sum() == 0
        and validation["duplicate_result_queries"].sum() == 0
        and validation["invalid_result_queries"].sum() == 0
    )
    hard_pass = bool(
        factorized_validation_ok
        and len(latency) == len(args.datasets) * 2
        and len(throughput) == len(args.datasets)
        and len(factor_formats) == 1
        and len(factor_tokenization_modes) == 1
        and request_counts_ok
        and bool(anchor_validation.get("hard_pass"))
        and all(
            values["max_absolute_residual_ms"] <= 0.2
            for values in component_closure.values()
        )
    )
    manifest = {
        "hard_pass": hard_pass,
        "checks": {
            "anchor_formal_hard_pass": bool(
                anchor_validation.get("hard_pass")
            ),
            "factorized_index_structure_valid": bool(
                validation["structure_checks_passed"].astype(bool).all()
            ),
            "factorized_query_scores_finite": bool(
                validation["nonfinite_score_queries"].sum() == 0
            ),
            "factorized_results_well_formed": bool(
                validation["duplicate_result_queries"].sum() == 0
                and validation["invalid_result_queries"].sum() == 0
            ),
            "request_counts_exact": request_counts_ok,
            "component_closure_within_0_2_ms": all(
                values["max_absolute_residual_ms"] <= 0.2
                for values in component_closure.values()
            ),
            "factor_formats_consistent": len(factor_formats) == 1,
            "factor_tokenization_modes_consistent": (
                len(factor_tokenization_modes) == 1
            ),
        },
        "protocol": {
            "datasets": args.datasets,
            "warmup_queries": args.warmup_queries,
            "repeats": args.repeats,
            "doc_topn": 300,
            "qdoc_topn": 1000,
            "k1": 0.9,
            "b": 0.4,
            "aggregation": "sum_decay_0.3",
            "normalization": "minmax",
            "alpha": FUSION_ALPHA,
            "sequential_latency_cores": 1,
            "parallel_latency_cores": 2,
            "fixed_throughput_cores": 2,
        },
        "paths": {
            "formal_anchor_root": str(args.formal_root),
            "factor_root": str(args.factor_root),
            "out_root": str(args.out_root),
        },
        "hardware": {
            "hostname": anchor_manifest.get("hostname"),
            "cpu_model": cpu_model,
            "anchor_created_utc": anchor_manifest.get("created_utc"),
            "lscpu": anchor_manifest.get("lscpu"),
        },
        "component_closure": component_closure,
        "factor_formats": sorted(factor_formats),
        "factor_tokenization_modes": sorted(factor_tokenization_modes),
        "table5": table_rows,
    }
    formal.write_json(args.out_root / "validation.json", manifest)
    lines = [
        "# SDE auxiliary-index serving benchmark",
        "",
        f"Hard pass: `{str(hard_pass).lower()}`",
        "",
        "| Method | Mean (ms) | p95 (ms) | 2-core QPS | Index (MB) |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in table.itertuples(index=False):
        lines.append(
            f"| {row.display_name} | {row.mean_ms:.3f} | {row.p95_ms:.3f} | "
            f"{row.fixed2_qps:.3f} | {row.index_mb:.3f} |"
        )
    lines.extend(
        [
            "",
            f"The first three rows are read from the supplied {cpu_model} "
            "formal baseline root. We report "
            f"{args.repeats} measured repeats after {args.warmup_queries} "
            "warm-up queries per collection. Both SDE auxiliary-index "
            "rows use the same query-document index contents, "
            "aggregation, min-max normalization, and "
            f"alpha={FUSION_ALPHA:.2f}.",
        ]
    )
    (args.out_root / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=formal.DEFAULT_DATA_ROOT)
    parser.add_argument("--formal-root", type=Path, default=DEFAULT_FORMAL_ROOT)
    parser.add_argument("--factor-root", type=Path, default=DEFAULT_FACTOR_ROOT)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--datasets", nargs="+", default=DATASETS)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup-queries", type=int, default=50)
    parser.add_argument("--max-queries", type=int, default=0)
    parser.add_argument("--skip-validation", action="store_true")
    parser.add_argument("--skip-latency", action="store_true")
    parser.add_argument("--skip-throughput", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return pd.read_csv(path, sep="\t").to_dict("records")


def main() -> None:
    args = parse_args()
    args.out_root.mkdir(parents=True, exist_ok=True)
    physical = formal.physical_cpu_ids()
    if len(physical) < 2:
        raise RuntimeError("Two physical CPUs are required")
    cpu_ids = physical[:2]
    ctx = mp.get_context("spawn")

    validation_path = args.out_root / "validation.tsv"
    validation_rows = load_rows(validation_path)
    completed = {str(row["dataset"]) for row in validation_rows}
    if not args.skip_validation:
        for dataset in args.datasets:
            if dataset in completed and not args.overwrite:
                continue
            print(f"[validate] {dataset}", flush=True)
            task = make_task(args, dataset, SEQUENTIAL, cpu_ids)
            result = formal.run_child(ctx, validation_worker, task)
            validation_rows = [
                row for row in validation_rows if str(row["dataset"]) != dataset
            ]
            validation_rows.append(result["summary"])
            save_frame(validation_path, validation_rows)
        if any(
            not bool(row["structure_checks_passed"])
            or int(row["nonfinite_score_queries"])
            or int(row["duplicate_result_queries"])
            or int(row["invalid_result_queries"])
            for row in validation_rows
        ):
            raise RuntimeError("Factorized qdoc integrity validation failed")

    latency_path = args.out_root / "latency_summary.tsv"
    latency_rows = load_rows(latency_path)
    completed_latency = {
        (str(row["dataset"]), str(row["method"])) for row in latency_rows
    }
    if not args.skip_latency:
        for dataset_index, dataset in enumerate(args.datasets):
            methods = [SEQUENTIAL, PARALLEL]
            if dataset_index % 2:
                methods.reverse()
            for method in methods:
                key = (dataset, method)
                if key in completed_latency and not args.overwrite:
                    continue
                print(f"[latency] {dataset} {method}", flush=True)
                task = make_task(args, dataset, method, cpu_ids)
                task["raw_path"] = str(
                    args.out_root / "raw_latency" / f"{dataset}.{method}.jsonl"
                )
                worker = (
                    latency_sequential_worker
                    if method == SEQUENTIAL
                    else latency_parallel_worker
                )
                result = formal.run_child(ctx, worker, task)
                latency_rows = [
                    row
                    for row in latency_rows
                    if (str(row["dataset"]), str(row["method"])) != key
                ]
                latency_rows.append(result["summary"])
                save_frame(latency_path, latency_rows)

    throughput_path = args.out_root / "fixed2_throughput_summary.tsv"
    throughput_rows = load_rows(throughput_path)
    completed_throughput = {str(row["dataset"]) for row in throughput_rows}
    if not args.skip_throughput:
        for dataset in args.datasets:
            if dataset in completed_throughput and not args.overwrite:
                continue
            print(f"[fixed2] {dataset} {SEQUENTIAL}", flush=True)
            task = make_task(args, dataset, SEQUENTIAL, cpu_ids)
            task["raw_path"] = str(
                args.out_root / "raw_fixed2" / f"{dataset}.{SEQUENTIAL}.jsonl"
            )
            result = formal.run_child(ctx, throughput_worker, task)
            throughput_rows = [
                row for row in throughput_rows if str(row["dataset"]) != dataset
            ]
            throughput_rows.append(result["summary"])
            save_frame(throughput_path, throughput_rows)

    if (
        validation_path.is_file()
        and latency_path.is_file()
        and throughput_path.is_file()
    ):
        aggregate(args)
        print(
            pd.read_csv(
                args.out_root / "table5_factorized_qdoc.tsv", sep="\t"
            ).to_string(index=False),
            flush=True,
        )
        print(f"[done] {args.out_root}", flush=True)


if __name__ == "__main__":
    main()
