#!/usr/bin/env python3
"""Measure the BM25 and Doc2Query++ anchors used by the SDE system tables."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import multiprocessing as mp
import os
import platform
import resource
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from queue import Empty
from typing import Any, Iterable, Sequence

import pandas as pd
import psutil


REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from src.pce.system_efficiency.d2q_queryline import (  # noqa: E402
    d2qpp_queryline_specs,
    fit_lines_to_budget,
    load_trace_query_lines,
    make_30_query_lines,
    unique_preserve,
)
from src.sde.terrier_utils import (  # noqa: E402
    build_retriever,
    ensure_pyterrier,
    index_corpus,
    load_corpus,
    load_qrels,
    load_queries,
    maybe_load_indexref,
    prepare_query_df,
    read_index_stats,
)


DEFAULT_DATA_ROOT = REPO_ROOT / "data" / "beir"
DEFAULT_TRACE_ROOT = REPO_ROOT / "data" / "traces"
DEFAULT_OUT_ROOT = REPO_ROOT / "results" / "system_efficiency_formal"
DEFAULT_DATASETS = [
    "nfcorpus",
    "scidocs",
    "fiqa-2018",
    "arguana",
    "scifact",
]
METHODS = ["bm25", "d2qpp_full", "d2qpp_qgen_only"]
BUILD_METHODS = list(METHODS)
LATENCY_METHODS = list(METHODS)
POOL_METHODS = list(METHODS)
DISPLAY_NAMES = {
    "bm25": "BM25",
    "d2qpp_full": "D2Q++ Full",
    "d2qpp_qgen_only": "D2Q++ QGen-only",
}


def now() -> float:
    return time.perf_counter()


def percentile(values: Sequence[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * pct / 100.0
    low = int(position)
    high = min(len(ordered) - 1, low + (0 if position == low else 1))
    if low == high:
        return ordered[low]
    weight = position - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def summarize_ms(values: Sequence[float]) -> dict[str, float]:
    milliseconds = [float(value) * 1000.0 for value in values]
    if not milliseconds:
        return {
            "mean_ms": 0.0,
            "p50_ms": 0.0,
            "p95_ms": 0.0,
            "p99_ms": 0.0,
        }
    return {
        "mean_ms": sum(milliseconds) / len(milliseconds),
        "p50_ms": percentile(milliseconds, 50),
        "p95_ms": percentile(milliseconds, 95),
        "p99_ms": percentile(milliseconds, 99),
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as destination:
        for row in rows:
            destination.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def capture(command: Sequence[str]) -> str:
    try:
        result = subprocess.run(
            command,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        return result.stdout.strip()
    except Exception as exc:
        return f"unavailable: {exc!r}"


def cpu_times_sec(process: psutil.Process) -> float:
    value = process.cpu_times()
    return float(value.user + value.system)


def rss_mb(process: psutil.Process | None = None) -> float:
    process = process or psutil.Process(os.getpid())
    return process.memory_info().rss / (1024.0**2)


def max_rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


class PeakRSS:
    def __init__(self, interval_sec: float = 0.01) -> None:
        self.interval_sec = interval_sec
        self.peak_mb = rss_mb()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        process = psutil.Process(os.getpid())
        while not self._stop.is_set():
            try:
                self.peak_mb = max(self.peak_mb, rss_mb(process))
            except psutil.Error:
                return
            self._stop.wait(self.interval_sec)

    def __enter__(self) -> "PeakRSS":
        self._thread.start()
        return self

    def __exit__(self, *_: Any) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)
        self.peak_mb = max(self.peak_mb, max_rss_mb())


def physical_cpu_ids() -> list[int]:
    allowed = set(os.sched_getaffinity(0))
    selected: list[int] = []
    seen: set[tuple[int, int]] = set()
    for line in capture(["lscpu", "-p=CPU,CORE,SOCKET"]).splitlines():
        if not line or line.startswith("#"):
            continue
        fields = line.split(",")
        if len(fields) < 3:
            continue
        cpu, core, socket = map(int, fields[:3])
        identity = (socket, core)
        if cpu in allowed and identity not in seen:
            selected.append(cpu)
            seen.add(identity)
    return selected or sorted(allowed)


def set_affinity(cpu_ids: Sequence[int]) -> None:
    if cpu_ids:
        os.sched_setaffinity(0, {int(cpu_id) for cpu_id in cpu_ids})


def dataset_paths(
    data_root: Path,
    trace_root: Path,
    dataset: str,
    split: str,
) -> dict[str, Path]:
    split_root = data_root / dataset / split
    return {
        "corpus_tsv": split_root / "collection.tsv",
        "queries_json": split_root / "queries.jsonl",
        "qrels_tsv": split_root / f"qrels.{split}.tsv",
        "trace_jsonl": (
            trace_root / dataset / "train_data_multitemplate_unfiltered.jsonl"
        ),
    }


def prepared_path(out_root: Path, dataset: str) -> Path:
    return out_root / "prepared" / dataset / "d2q_expansions.jsonl.gz"


def build_index_path(
    out_root: Path,
    repeat: int,
    dataset: str,
    method: str,
) -> Path:
    return out_root / "indexes" / f"repeat_{repeat}" / dataset / method


def query_items(
    data_root: Path,
    dataset: str,
    split: str,
    max_queries: int = 0,
) -> list[dict[str, str]]:
    paths = dataset_paths(data_root, Path("."), dataset, split)
    queries = load_queries(paths["queries_json"])
    qrels = load_qrels(paths["qrels_tsv"])
    frame = prepare_query_df(queries, qrels)
    if max_queries > 0:
        frame = frame.iloc[:max_queries]
    return [
        {"qid": str(row.qid), "query": str(row.query)}
        for row in frame.itertuples(index=False)
    ]


def prepare_dataset(
    args: argparse.Namespace,
    dataset: str,
) -> dict[str, Any]:
    output = prepared_path(args.out_root, dataset)
    stats_path = output.parent / "preparation.json"
    if output.is_file() and stats_path.is_file() and not args.overwrite:
        return json.loads(stats_path.read_text(encoding="utf-8"))

    paths = dataset_paths(args.data_root, args.trace_root, dataset, args.split)
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing {name} for {dataset}: {path}")

    document_ids, _ = load_corpus(paths["corpus_tsv"])
    started = now()
    with PeakRSS() as monitor:
        trace_lines, trace_stats = load_trace_query_lines(
            paths["trace_jsonl"],
            args.queryline_sample_idx_max,
        )
        specs = {
            item["key"]: item for item in d2qpp_queryline_specs(dataset)
        }
        byte_targets = {
            method: int(
                round(float(specs[method]["target_kb_per_doc"]) * 1024.0)
            )
            for method in ("d2qpp_full", "d2qpp_qgen_only")
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256()
        byte_totals = {method: 0 for method in byte_targets}
        missing_query_lines = 0
        unique_source_lines = 0
        with gzip.open(output, "wt", encoding="utf-8", compresslevel=1) as target:
            for document_id in document_ids:
                source_lines = trace_lines.get(document_id, [])
                if not source_lines:
                    missing_query_lines += 1
                unique_source_lines += len(unique_preserve(source_lines))
                lines = make_30_query_lines(source_lines, 30)
                row: dict[str, str] = {"docno": document_id}
                for method, target_bytes in byte_targets.items():
                    expansion, used_bytes = fit_lines_to_budget(
                        lines,
                        target_bytes,
                    )
                    row[method] = expansion
                    byte_totals[method] += used_bytes
                serialized = (
                    json.dumps(
                        row,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                    + "\n"
                )
                digest.update(serialized.encode("utf-8"))
                target.write(serialized)

    stats = {
        "dataset": dataset,
        "n_docs": len(document_ids),
        "preparation_wall_sec": now() - started,
        "preparation_peak_rss_mb": monitor.peak_mb,
        "prepared_file": str(output),
        "prepared_file_bytes": output.stat().st_size,
        "prepared_content_sha256": digest.hexdigest(),
        "d2qpp_queries_per_doc": 30,
        "d2qpp_queryline_sample_idx_max": args.queryline_sample_idx_max,
        "missing_queryline_source_docs": missing_query_lines,
        "avg_unique_source_query_lines_per_doc": (
            unique_source_lines / len(document_ids) if document_ids else 0.0
        ),
        "avg_d2qpp_full_kb_per_doc": (
            byte_totals["d2qpp_full"]
            / max(1, len(document_ids))
            / 1024.0
        ),
        "avg_d2qpp_qgen_only_kb_per_doc": (
            byte_totals["d2qpp_qgen_only"]
            / max(1, len(document_ids))
            / 1024.0
        ),
        "source_files": {
            name: {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for name, path in paths.items()
        },
        "queryline_trace_stats": trace_stats,
    }
    write_json(stats_path, stats)
    return stats


def load_prepared(
    path: Path,
    expected_document_ids: Sequence[str],
) -> dict[str, list[str]]:
    values = {
        "d2qpp_full": [],
        "d2qpp_qgen_only": [],
    }
    with gzip.open(path, "rt", encoding="utf-8") as source:
        for index, raw in enumerate(source):
            row = json.loads(raw)
            if (
                index >= len(expected_document_ids)
                or str(row["docno"]) != str(expected_document_ids[index])
            ):
                raise ValueError(
                    f"Prepared data order mismatch at row {index}: "
                    f"{row.get('docno')}"
                )
            for method in values:
                values[method].append(str(row.get(method, "")))
    if any(
        len(expansions) != len(expected_document_ids)
        for expansions in values.values()
    ):
        raise ValueError("Prepared data length does not match the corpus")
    return values


def build_worker(task: dict[str, Any], result_queue: Any) -> None:
    try:
        set_affinity(task["cpu_ids"])
        dataset = str(task["dataset"])
        method = str(task["method"])
        paths = dataset_paths(
            Path(task["data_root"]),
            Path(task["trace_root"]),
            dataset,
            str(task["split"]),
        )
        document_ids, document_texts = load_corpus(paths["corpus_tsv"])
        if method == "bm25":
            index_texts = document_texts
        elif method in {"d2qpp_full", "d2qpp_qgen_only"}:
            prepared = load_prepared(
                Path(task["prepared_path"]),
                document_ids,
            )
            index_texts = [
                f"{text} {expansion}" if expansion else text
                for text, expansion in zip(
                    document_texts,
                    prepared[method],
                )
            ]
        else:
            raise ValueError(f"Unknown build method: {method}")

        index_dir = Path(task["index_dir"])
        shutil.rmtree(index_dir, ignore_errors=True)
        index_dir.parent.mkdir(parents=True, exist_ok=True)
        ensure_pyterrier()
        process = psutil.Process(os.getpid())
        io_start = process.io_counters()
        cpu_start = cpu_times_sec(process)
        rss_start = rss_mb(process)
        started = now()
        with PeakRSS() as monitor:
            index_corpus(
                index_dir,
                document_ids,
                index_texts,
                reuse_existing=False,
            )
        io_end = process.io_counters()
        result_queue.put(
            {
                "ok": True,
                "dataset": dataset,
                "method": method,
                "display_name": DISPLAY_NAMES[method],
                "repeat": int(task["repeat"]),
                "n_input_docs": len(document_ids),
                "build_wall_sec": now() - started,
                "build_cpu_sec": cpu_times_sec(process) - cpu_start,
                "build_start_rss_mb": rss_start,
                "build_peak_rss_mb": monitor.peak_mb,
                "write_bytes": max(
                    0,
                    int(io_end.write_bytes - io_start.write_bytes),
                ),
                "cpu_ids": list(task["cpu_ids"]),
                **read_index_stats(index_dir),
            }
        )
    except Exception as exc:
        result_queue.put(
            {"ok": False, "error": repr(exc), "task": task}
        )


def run_child(
    context: Any,
    target: Any,
    task: dict[str, Any],
    timeout_sec: float = 7200.0,
) -> dict[str, Any]:
    queue = context.Queue()
    process = context.Process(target=target, args=(task, queue))
    process.start()
    try:
        result = queue.get(timeout=timeout_sec)
    except Empty as exc:
        process.terminate()
        process.join(timeout=10)
        raise TimeoutError(f"Worker timed out: {task}") from exc
    process.join(timeout=30)
    if process.is_alive():
        process.terminate()
        process.join(timeout=10)
    if not result.get("ok"):
        raise RuntimeError(f"Worker failed: {result}")
    return result


def maybe_transform(
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


def retriever_for(
    index_dir: Path,
    topn: int,
    k1: float,
    b: float,
) -> Any:
    index_ref = maybe_load_indexref(index_dir)
    if index_ref is None:
        raise FileNotFoundError(f"Missing index: {index_dir}")
    return build_retriever(index_ref, topn, k1, b)


def summarize_latency_rows(
    task: dict[str, Any],
    rows: Sequence[dict[str, Any]],
    elapsed: float,
) -> dict[str, Any]:
    durations = [float(row["e2e_sec"]) for row in rows]
    stats = summarize_ms(durations)
    output = {
        "dataset": task["dataset"],
        "method": task["method"],
        "display_name": DISPLAY_NAMES[task["method"]],
        "queries": int(task["query_count"]),
        "repeats": int(task["repeats"]),
        "requests": len(rows),
        "qps": len(rows) / elapsed if elapsed > 0 else 0.0,
        "e2e_mean_ms": stats["mean_ms"],
        "e2e_p50_ms": stats["p50_ms"],
        "e2e_p95_ms": stats["p95_ms"],
        "e2e_p99_ms": stats["p99_ms"],
        "core_ms_per_query": (
            sum(float(row["cpu_sec"]) for row in rows)
            * 1000.0
            / max(1, len(rows))
        ),
    }
    output["effective_cpu_cores"] = (
        output["core_ms_per_query"] / output["e2e_mean_ms"]
        if output["e2e_mean_ms"] > 0
        else 0.0
    )
    return output


def latency_measurement(
    task: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    set_affinity(task["cpu_ids"][:1])
    ensure_pyterrier()
    queries = query_items(
        Path(task["data_root"]),
        str(task["dataset"]),
        str(task["split"]),
        int(task["max_queries"]),
    )
    task["query_count"] = len(queries)
    retriever = retriever_for(
        Path(task["main_index"]),
        1000,
        0.9,
        0.4,
    )

    def run_one(item: dict[str, str]) -> dict[str, Any]:
        cpu_start = time.process_time()
        started = now()
        results = maybe_transform(
            retriever,
            item["qid"],
            item["query"],
        )
        return {
            "e2e_sec": now() - started,
            "cpu_sec": time.process_time() - cpu_start,
            "candidate_union": len(results),
        }

    for item in queries[: min(len(queries), int(task["warmup_queries"]))]:
        run_one(item)
    warm_rss = rss_mb()
    rows: list[dict[str, Any]] = []
    started = now()
    with PeakRSS() as monitor:
        for repeat in range(int(task["repeats"])):
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
    summary = summarize_latency_rows(task, rows, now() - started)
    summary.update(
        {
            "cpu_budget": 1,
            "retrieval_workers": 1,
            "service_processes": 1,
            "warm_rss_mb": warm_rss,
            "peak_rss_mb": monitor.peak_mb,
            "parallelism": "single process",
        }
    )
    return summary, rows


def latency_worker(task: dict[str, Any], result_queue: Any) -> None:
    try:
        summary, rows = latency_measurement(task)
        write_jsonl(Path(task["raw_path"]), rows)
        result_queue.put({"ok": True, "summary": summary})
    except Exception as exc:
        result_queue.put(
            {"ok": False, "error": repr(exc), "task": task}
        )


def pool_service_worker(
    worker_name: str,
    task: dict[str, Any],
    cpu_id: int,
    request_queue: Any,
    response_queue: Any,
) -> None:
    try:
        set_affinity([cpu_id])
        ensure_pyterrier()
        retriever = retriever_for(
            Path(task["main_index"]),
            1000,
            0.9,
            0.4,
        )
        response_queue.put(
            {
                "type": "ready",
                "worker": worker_name,
                "rss_mb": rss_mb(),
            }
        )
        while True:
            item = request_queue.get()
            if item is None:
                return
            cpu_start = time.process_time()
            started = now()
            candidates = len(
                maybe_transform(
                    retriever,
                    str(item["qid"]),
                    str(item["query"]),
                )
            )
            response_queue.put(
                {
                    "type": "result",
                    "worker": worker_name,
                    "request_id": int(item["request_id"]),
                    "service_sec": now() - started,
                    "cpu_sec": time.process_time() - cpu_start,
                    "candidate_union": candidates,
                    "rss_mb": rss_mb(),
                    "peak_rss_mb": max_rss_mb(),
                }
            )
    except Exception as exc:
        response_queue.put(
            {
                "type": "error",
                "worker": worker_name,
                "error": repr(exc),
            }
        )


def collect_ready(
    response_queue: Any,
    expected: set[str],
) -> dict[str, dict[str, Any]]:
    deadline = time.time() + 300.0
    found: dict[str, dict[str, Any]] = {}
    while set(found) != expected:
        remaining = deadline - time.time()
        if remaining <= 0:
            raise TimeoutError(
                f"Timed out waiting for workers: {expected}"
            )
        message = response_queue.get(timeout=remaining)
        if message.get("type") == "error":
            raise RuntimeError(message)
        if message.get("type") == "ready":
            found[str(message["worker"])] = message
    return found


def throughput_pool2(
    task: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    cpu_ids = list(task["cpu_ids"][:2])
    if len(cpu_ids) < 2:
        raise ValueError("Two physical CPUs are required")
    set_affinity([cpu_ids[0]])
    context = mp.get_context("spawn")
    response_queue = context.Queue()
    request_queues = [context.Queue(), context.Queue()]
    names = ["worker0", "worker1"]
    workers = [
        context.Process(
            target=pool_service_worker,
            args=(name, task, cpu_id, queue, response_queue),
        )
        for name, cpu_id, queue in zip(names, cpu_ids, request_queues)
    ]
    for worker in workers:
        worker.start()
    ready = collect_ready(response_queue, set(names))
    queries = query_items(
        Path(task["data_root"]),
        str(task["dataset"]),
        str(task["split"]),
        int(task["max_queries"]),
    )
    task["query_count"] = len(queries)
    request_id = 0
    worker_peaks = {
        name: float(value["rss_mb"]) for name, value in ready.items()
    }

    def receive_result() -> dict[str, Any]:
        message = response_queue.get(timeout=300)
        if message.get("type") == "error":
            raise RuntimeError(message)
        return message

    try:
        warm_count = min(len(queries), int(task["warmup_queries"]))
        for index, item in enumerate(queries[:warm_count]):
            request_id += 1
            request_queues[index % 2].put(
                {"request_id": request_id, **item}
            )
            receive_result()

        rows: list[dict[str, Any]] = []
        repeat_elapsed: list[float] = []
        for repeat in range(int(task["repeats"])):
            pending: dict[int, tuple[float, dict[str, str]]] = {}
            next_index = 0
            repeat_started = now()
            for worker_index in range(min(2, len(queries))):
                request_id += 1
                item = queries[next_index]
                next_index += 1
                pending[request_id] = (now(), item)
                request_queues[worker_index].put(
                    {"request_id": request_id, **item}
                )
            while pending:
                message = receive_result()
                received_id = int(message["request_id"])
                submitted, item = pending.pop(received_id)
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
                        "candidate_union": int(
                            message["candidate_union"]
                        ),
                    }
                )
                if next_index < len(queries):
                    request_id += 1
                    item = queries[next_index]
                    next_index += 1
                    pending[request_id] = (now(), item)
                    request_queues[worker_index].put(
                        {"request_id": request_id, **item}
                    )
            repeat_elapsed.append(now() - repeat_started)

        stats = summarize_ms(
            [float(row["e2e_sec"]) for row in rows]
        )
        summary = {
            "dataset": task["dataset"],
            "method": task["method"],
            "display_name": DISPLAY_NAMES[task["method"]],
            "queries": len(queries),
            "repeats": int(task["repeats"]),
            "requests": len(rows),
            "fixed_core_budget": 2,
            "concurrency": 2,
            "worker_processes": 2,
            "qps": (
                len(rows) / sum(repeat_elapsed)
                if sum(repeat_elapsed) > 0
                else 0.0
            ),
            "e2e_mean_ms": stats["mean_ms"],
            "e2e_p50_ms": stats["p50_ms"],
            "e2e_p95_ms": stats["p95_ms"],
            "e2e_p99_ms": stats["p99_ms"],
            "core_ms_per_query": (
                sum(float(row["cpu_sec"]) for row in rows)
                * 1000.0
                / max(1, len(rows))
            ),
            "worker_warm_rss_mb": sum(
                float(value["rss_mb"]) for value in ready.values()
            ),
            "peak_rss_mb": sum(worker_peaks.values()),
        }
        summary["effective_cpu_cores"] = (
            summary["core_ms_per_query"] / summary["e2e_mean_ms"]
            if summary["e2e_mean_ms"] > 0
            else 0.0
        )
        return summary, rows
    finally:
        for queue in request_queues:
            queue.put(None)
        for worker in workers:
            worker.join(timeout=20)
            if worker.is_alive():
                worker.terminate()
                worker.join(timeout=5)


def throughput_worker(task: dict[str, Any], result_queue: Any) -> None:
    try:
        summary, rows = throughput_pool2(task)
        write_jsonl(Path(task["raw_path"]), rows)
        result_queue.put({"ok": True, "summary": summary})
    except Exception as exc:
        result_queue.put(
            {"ok": False, "error": repr(exc), "task": task}
        )


def method_task(
    args: argparse.Namespace,
    dataset: str,
    method: str,
    repeat: int,
    serve_cpus: Sequence[int],
) -> dict[str, Any]:
    return {
        "dataset": dataset,
        "method": method,
        "data_root": str(args.data_root),
        "trace_root": str(args.trace_root),
        "split": args.split,
        "repeats": args.latency_repeats,
        "warmup_queries": args.warmup_queries,
        "max_queries": args.max_queries,
        "cpu_ids": list(serve_cpus),
        "main_index": str(
            build_index_path(args.out_root, repeat, dataset, method)
        ),
    }


def dataframe_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return pd.read_csv(path, sep="\t").to_dict("records")


def save_rows(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, sep="\t", index=False)


def markdown_table(
    frame: pd.DataFrame,
    columns: Sequence[str],
    labels: Sequence[str],
) -> str:
    lines = [
        "| " + " | ".join(labels) + " |",
        "|" + "|".join(["---"] + ["---:"] * (len(labels) - 1)) + "|",
    ]
    for row in frame.itertuples(index=False):
        values: list[str] = []
        for column in columns:
            value = getattr(row, column)
            if pd.isna(value):
                values.append("")
            elif isinstance(value, float):
                values.append(f"{value:.3f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def aggregate_report(args: argparse.Namespace) -> None:
    build = pd.read_csv(
        args.out_root / "build_measurements.tsv",
        sep="\t",
    )
    latency = pd.read_csv(
        args.out_root / "latency_summary.tsv",
        sep="\t",
    )
    throughput = pd.read_csv(
        args.out_root / "fixed2_throughput_summary.tsv",
        sep="\t",
    )
    build = build[build["method"].isin(METHODS)].copy()
    latency = latency[latency["method"].isin(METHODS)].copy()
    throughput = throughput[throughput["method"].isin(METHODS)].copy()

    build_group = build.groupby(
        ["dataset", "method", "display_name"],
        as_index=False,
    ).agg(
        build_median_sec=("build_wall_sec", "median"),
        build_min_sec=("build_wall_sec", "min"),
        build_max_sec=("build_wall_sec", "max"),
        build_cpu_median_sec=("build_cpu_sec", "median"),
        peak_rss_mb=("build_peak_rss_mb", "max"),
        write_mb=(
            "write_bytes",
            lambda values: float(pd.Series(values).median()) / (1024.0**2),
        ),
        index_mb=(
            "size_bytes",
            lambda values: float(pd.Series(values).median()) / (1024.0**2),
        ),
        index_docs=("num_documents", "max"),
        index_terms=("num_terms", "max"),
        index_pointers=("num_pointers", "max"),
    )
    build_group.to_csv(
        args.out_root / "report_build_and_footprint.tsv",
        sep="\t",
        index=False,
    )

    footprint_rows: list[dict[str, Any]] = []
    lifecycle_rows: list[dict[str, Any]] = []
    for dataset in args.datasets:
        subset = build_group[
            build_group["dataset"] == dataset
        ].set_index("method")
        bm25_mb = float(subset.loc["bm25", "index_mb"])
        for method in METHODS:
            row = subset.loc[method]
            index_mb = float(row["index_mb"])
            footprint_rows.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "display_name": DISPLAY_NAMES[method],
                    "original_index_mb": index_mb,
                    "expansion_index_mb": "",
                    "total_index_mb": index_mb,
                    "ratio_vs_bm25": index_mb / bm25_mb,
                }
            )
            lifecycle_rows.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "display_name": DISPLAY_NAMES[method],
                    "initial_build_sec": float(row["build_median_sec"]),
                    "full_expansion_refresh_sec": (
                        float("nan")
                        if method == "bm25"
                        else float(row["build_median_sec"])
                    ),
                    "build_peak_rss_mb": float(row["peak_rss_mb"]),
                    "write_mb": float(row["write_mb"]),
                    "index_mb": index_mb,
                    "refresh_scope": (
                        "not applicable"
                        if method == "bm25"
                        else "rebuild combined text and expansion index"
                    ),
                }
            )

    footprint = pd.DataFrame(footprint_rows)
    totals: list[dict[str, Any]] = []
    bm25_total = float(
        footprint[footprint["method"] == "bm25"]["total_index_mb"].sum()
    )
    for method, subset in footprint.groupby("method"):
        total = float(subset["total_index_mb"].sum())
        totals.append(
            {
                "dataset": "BEIR5 total",
                "method": method,
                "display_name": DISPLAY_NAMES[method],
                "original_index_mb": total,
                "expansion_index_mb": "",
                "total_index_mb": total,
                "ratio_vs_bm25": total / bm25_total,
            }
        )
    footprint = pd.concat(
        [footprint, pd.DataFrame(totals)],
        ignore_index=True,
    )
    footprint.to_csv(
        args.out_root / "report_index_footprint.tsv",
        sep="\t",
        index=False,
    )

    lifecycle = pd.DataFrame(lifecycle_rows)
    lifecycle.to_csv(
        args.out_root / "report_index_lifecycle.tsv",
        sep="\t",
        index=False,
    )
    latency.to_csv(
        args.out_root / "report_online_latency.tsv",
        sep="\t",
        index=False,
    )
    throughput.to_csv(
        args.out_root / "report_fixed2_throughput.tsv",
        sep="\t",
        index=False,
    )

    latency_macro = latency.groupby(
        ["method", "display_name"],
        as_index=False,
    ).agg(
        e2e_mean_ms=("e2e_mean_ms", "mean"),
        e2e_p50_ms=("e2e_p50_ms", "mean"),
        e2e_p95_ms=("e2e_p95_ms", "mean"),
        e2e_p99_ms=("e2e_p99_ms", "mean"),
        qps=("qps", "mean"),
        core_ms_per_query=("core_ms_per_query", "mean"),
        effective_cpu_cores=("effective_cpu_cores", "mean"),
        warm_rss_mb=("warm_rss_mb", "mean"),
        peak_rss_mb=("peak_rss_mb", "mean"),
        cpu_budget=("cpu_budget", "max"),
    )
    throughput_macro = throughput.groupby(
        ["method", "display_name"],
        as_index=False,
    ).agg(
        fixed2_qps=("qps", "mean"),
        fixed2_p95_ms=("e2e_p95_ms", "mean"),
        fixed2_core_ms=("core_ms_per_query", "mean"),
        fixed2_peak_rss_mb=("peak_rss_mb", "mean"),
    )
    total_footprint = footprint[
        footprint["dataset"] == "BEIR5 total"
    ][["method", "total_index_mb", "ratio_vs_bm25"]]
    final = (
        latency_macro.merge(total_footprint, on="method", how="left")
        .merge(
            throughput_macro,
            on=["method", "display_name"],
            how="left",
        )
    )
    final.to_csv(
        args.out_root / "system_efficiency_main_table.tsv",
        sep="\t",
        index=False,
    )

    lifecycle_total = lifecycle.groupby(
        ["method", "display_name"],
        as_index=False,
    ).agg(
        initial_build_sec=("initial_build_sec", "sum"),
        full_expansion_refresh_sec=(
            "full_expansion_refresh_sec",
            lambda values: pd.to_numeric(
                values,
                errors="coerce",
            ).sum(min_count=1),
        ),
        build_peak_rss_mb=("build_peak_rss_mb", "max"),
        write_mb=("write_mb", "sum"),
        index_mb=("index_mb", "sum"),
    )
    report = [
        "# Formal Baseline System-Efficiency Report",
        "",
        "No LLM generation is performed. Doc2Query++ rows use the "
        "deterministic 30-query-line simulation.",
        "",
        "## Online Serving",
        "",
        markdown_table(
            final,
            [
                "display_name",
                "e2e_mean_ms",
                "e2e_p95_ms",
                "fixed2_qps",
                "total_index_mb",
            ],
            ["Method", "Mean ms", "p95 ms", "2-core QPS", "Index MB"],
        ),
        "",
        "## Index Lifecycle",
        "",
        markdown_table(
            lifecycle_total,
            [
                "display_name",
                "initial_build_sec",
                "full_expansion_refresh_sec",
                "write_mb",
                "build_peak_rss_mb",
            ],
            [
                "Method",
                "Initial build s",
                "Full refresh s",
                "Initial write MB",
                "Peak RSS MB",
            ],
        ),
        "",
        "Online values are unweighted collection averages. Index sizes, "
        "build times, refresh times, and write volumes are collection sums. "
        "Peak RSS is the maximum single collection task.",
        "",
        f"Warm-up queries: {args.warmup_queries}; measured repeats: "
        f"{args.latency_repeats}; build repeats: {args.build_repeats}.",
        "",
    ]
    (args.out_root / "system_efficiency_report.md").write_text(
        "\n".join(report),
        encoding="utf-8",
    )


def count_lines(path: Path) -> int:
    with path.open("rb") as source:
        return sum(1 for _ in source)


def validate_outputs(args: argparse.Namespace) -> dict[str, Any]:
    build = pd.read_csv(
        args.out_root / "build_measurements.tsv",
        sep="\t",
    )
    latency = pd.read_csv(
        args.out_root / "latency_summary.tsv",
        sep="\t",
    )
    throughput = pd.read_csv(
        args.out_root / "fixed2_throughput_summary.tsv",
        sep="\t",
    )
    build = build[build["method"].isin(METHODS)].copy()
    latency = latency[latency["method"].isin(METHODS)].copy()
    throughput = throughput[throughput["method"].isin(METHODS)].copy()
    query_counts = {
        dataset: len(
            query_items(
                args.data_root,
                dataset,
                args.split,
                args.max_queries,
            )
        )
        for dataset in args.datasets
    }

    nondeterministic: dict[str, dict[str, int]] = {}
    grouped = build.groupby(["dataset", "method"])
    for column in [
        "size_bytes",
        "num_documents",
        "num_terms",
        "num_pointers",
    ]:
        counts = grouped[column].nunique()
        if (counts > 1).any():
            nondeterministic[column] = {
                f"{dataset}/{method}": int(value)
                for (dataset, method), value in counts[counts > 1].items()
            }

    request_errors: list[dict[str, Any]] = []
    raw_errors: list[dict[str, Any]] = []
    for frame_name, frame, raw_dir in [
        ("latency", latency, args.out_root / "raw_latency"),
        ("fixed2", throughput, args.out_root / "raw_fixed2"),
    ]:
        for dataset in args.datasets:
            expected_queries = query_counts[dataset]
            expected_requests = expected_queries * args.latency_repeats
            for method in METHODS:
                subset = frame[
                    (frame["dataset"] == dataset)
                    & (frame["method"] == method)
                ]
                if len(subset) != 1:
                    request_errors.append(
                        {
                            "frame": frame_name,
                            "dataset": dataset,
                            "method": method,
                            "rows": len(subset),
                        }
                    )
                    continue
                row = subset.iloc[0]
                if (
                    int(row["queries"]) != expected_queries
                    or int(row["requests"]) != expected_requests
                ):
                    request_errors.append(
                        {
                            "frame": frame_name,
                            "dataset": dataset,
                            "method": method,
                            "queries": int(row["queries"]),
                            "requests": int(row["requests"]),
                            "expected_queries": expected_queries,
                            "expected_requests": expected_requests,
                        }
                    )
                raw_path = raw_dir / f"{dataset}.{method}.jsonl"
                actual = count_lines(raw_path) if raw_path.is_file() else -1
                if actual != expected_requests:
                    raw_errors.append(
                        {
                            "path": str(raw_path),
                            "actual": actual,
                            "expected": expected_requests,
                        }
                    )

    nonpositive: dict[str, int] = {}
    for name, frame in [("latency", latency), ("fixed2", throughput)]:
        for column in [
            "qps",
            "e2e_mean_ms",
            "e2e_p50_ms",
            "e2e_p95_ms",
            "e2e_p99_ms",
            "core_ms_per_query",
        ]:
            invalid = pd.to_numeric(
                frame[column],
                errors="coerce",
            ) <= 0
            if invalid.any():
                nonpositive[f"{name}/{column}"] = int(invalid.sum())

    row_counts = {
        "build": {
            "actual": len(build),
            "expected": (
                len(args.datasets) * len(METHODS) * args.build_repeats
            ),
        },
        "latency": {
            "actual": len(latency),
            "expected": len(args.datasets) * len(METHODS),
        },
        "fixed2": {
            "actual": len(throughput),
            "expected": len(args.datasets) * len(METHODS),
        },
    }
    hard_pass = (
        all(
            item["actual"] == item["expected"]
            for item in row_counts.values()
        )
        and not nondeterministic
        and not request_errors
        and not raw_errors
        and not nonpositive
    )
    payload = {
        "hard_pass": hard_pass,
        "row_counts": row_counts,
        "query_counts": query_counts,
        "index_stats_deterministic_across_build_repeats": (
            not nondeterministic
        ),
        "nondeterministic_index_stats": nondeterministic,
        "request_count_errors": request_errors,
        "raw_count_errors": raw_errors,
        "nonpositive_metrics": nonpositive,
    }
    write_json(args.out_root / "validation.json", payload)
    if not hard_pass:
        raise RuntimeError(
            f"Formal baseline validation failed: {payload}"
        )
    return payload


def environment_manifest(
    args: argparse.Namespace,
    build_cpus: Sequence[int],
    serve_cpus: Sequence[int],
) -> dict[str, Any]:
    return {
        "created_utc": time.strftime(
            "%Y-%m-%dT%H:%M:%SZ",
            time.gmtime(),
        ),
        "no_new_llm_generation": True,
        "python": sys.version,
        "platform": platform.platform(),
        "hostname": platform.node(),
        "lscpu": capture(["lscpu"]),
        "memory": capture(["free", "-h"]),
        "storage": capture(["df", "-h", str(args.out_root.parent)]),
        "java": capture(["java", "-version"]),
        "pip_freeze": capture([sys.executable, "-m", "pip", "freeze"]),
        "git_commit": capture(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"]
        ),
        "git_status": capture(
            ["git", "-C", str(REPO_ROOT), "status", "--short"]
        ),
        "allowed_cpus": sorted(os.sched_getaffinity(0)),
        "build_cpu_ids": list(build_cpus),
        "serve_cpu_ids": list(serve_cpus),
        "build_repeats": args.build_repeats,
        "latency_repeats": args.latency_repeats,
        "warmup_queries": args.warmup_queries,
        "max_queries": args.max_queries,
        "retrieval_config": {"topn": 1000, "k1": 0.9, "b": 0.4},
        "cache_policy": (
            "warm serving after explicit warm-up; OS page cache retained"
        ),
        "datasets": args.datasets,
        "paths": {
            "data_root": str(args.data_root),
            "trace_root": str(args.trace_root),
            "out_root": str(args.out_root),
        },
        "d2qpp_simulation": {
            "queries_per_doc": 30,
            "queryline_sample_idx_max": args.queryline_sample_idx_max,
            "note": (
                "No-LLM query-line simulation matched to the measured "
                "collection-level storage budgets"
            ),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure BM25 and Doc2Query++ system-efficiency anchors."
    )
    parser.add_argument(
        "--data_root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
    )
    parser.add_argument(
        "--trace_root",
        type=Path,
        default=DEFAULT_TRACE_ROOT,
    )
    parser.add_argument(
        "--out_root",
        type=Path,
        default=DEFAULT_OUT_ROOT,
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=DEFAULT_DATASETS,
    )
    parser.add_argument("--split", default="test")
    parser.add_argument("--build_repeats", type=int, default=3)
    parser.add_argument("--latency_repeats", type=int, default=3)
    parser.add_argument("--warmup_queries", type=int, default=50)
    parser.add_argument(
        "--queryline_sample_idx_max",
        type=int,
        default=1,
    )
    parser.add_argument("--build_cpu_count", type=int, default=8)
    parser.add_argument("--max_queries", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip_prepare", action="store_true")
    parser.add_argument("--skip_build", action="store_true")
    parser.add_argument("--skip_latency", action="store_true")
    parser.add_argument("--skip_throughput", action="store_true")
    args = parser.parse_args()
    if args.build_repeats <= 0 or args.latency_repeats <= 0:
        parser.error("repeat counts must be positive")
    if args.warmup_queries < 0 or args.max_queries < 0:
        parser.error("query counts must be non-negative")
    return args


def main() -> None:
    args = parse_args()
    args.out_root.mkdir(parents=True, exist_ok=True)
    physical = physical_cpu_ids()
    if len(physical) < 2:
        raise RuntimeError(
            f"At least two physical CPUs are required, found {physical}"
        )
    build_cpus = physical[
        : max(1, min(args.build_cpu_count, len(physical)))
    ]
    serve_cpus = physical[:2]
    manifest_path = args.out_root / "manifest.json"
    existing_manifest: dict[str, Any] = {}
    if manifest_path.is_file() and not args.overwrite:
        existing_manifest = json.loads(
            manifest_path.read_text(encoding="utf-8")
        )
    manifest = environment_manifest(args, build_cpus, serve_cpus)
    if existing_manifest:
        manifest["created_utc"] = existing_manifest.get(
            "created_utc",
            manifest["created_utc"],
        )
        if "completion" in existing_manifest:
            manifest["completion"] = existing_manifest["completion"]
    write_json(manifest_path, manifest)

    preparation: list[dict[str, Any]] = []
    for dataset in args.datasets:
        stats_path = (
            prepared_path(args.out_root, dataset).parent
            / "preparation.json"
        )
        if args.skip_prepare:
            if not stats_path.is_file():
                raise FileNotFoundError(stats_path)
            stats = json.loads(stats_path.read_text(encoding="utf-8"))
        else:
            print(f"[prepare] {dataset}", flush=True)
            stats = prepare_dataset(args, dataset)
        preparation.append(stats)
        write_json(
            args.out_root / "preparation_summary.json",
            preparation,
        )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["workload"] = {
        row["dataset"]: {
            "documents": int(row["n_docs"]),
            "queries": len(
                query_items(
                    args.data_root,
                    row["dataset"],
                    args.split,
                    args.max_queries,
                )
            ),
        }
        for row in preparation
    }
    manifest["input_artifacts"] = {
        row["dataset"]: {
            "source_files": row["source_files"],
            "prepared_file": row["prepared_file"],
            "prepared_file_bytes": row["prepared_file_bytes"],
            "prepared_content_sha256": row["prepared_content_sha256"],
        }
        for row in preparation
    }
    write_json(manifest_path, manifest)

    context = mp.get_context("spawn")
    build_path = args.out_root / "build_measurements.tsv"
    build_rows = [
        row
        for row in dataframe_rows(build_path)
        if str(row["method"]) in METHODS
    ]
    completed_builds = {
        (
            str(row["dataset"]),
            str(row["method"]),
            int(row["repeat"]),
        )
        for row in build_rows
    }
    if not args.skip_build:
        for repeat in range(args.build_repeats):
            rotated = METHODS[repeat % len(METHODS) :] + METHODS[
                : repeat % len(METHODS)
            ]
            for dataset in args.datasets:
                for method in rotated:
                    key = (dataset, method, repeat)
                    index_dir = build_index_path(
                        args.out_root,
                        repeat,
                        dataset,
                        method,
                    )
                    if (
                        key in completed_builds
                        and (index_dir / "data.properties").is_file()
                        and not args.overwrite
                    ):
                        continue
                    print(
                        f"[build] repeat={repeat} dataset={dataset} "
                        f"method={method}",
                        flush=True,
                    )
                    result = run_child(
                        context,
                        build_worker,
                        {
                            "dataset": dataset,
                            "method": method,
                            "repeat": repeat,
                            "data_root": str(args.data_root),
                            "trace_root": str(args.trace_root),
                            "split": args.split,
                            "prepared_path": str(
                                prepared_path(args.out_root, dataset)
                            ),
                            "index_dir": str(index_dir),
                            "cpu_ids": list(build_cpus),
                        },
                    )
                    build_rows = [
                        row
                        for row in build_rows
                        if (
                            str(row["dataset"]),
                            str(row["method"]),
                            int(row["repeat"]),
                        )
                        != key
                    ]
                    build_rows.append(
                        {
                            name: value
                            for name, value in result.items()
                            if name != "ok"
                        }
                    )
                    save_rows(build_path, build_rows)

    canonical_repeat = args.build_repeats - 1
    latency_path = args.out_root / "latency_summary.tsv"
    latency_rows = [
        row
        for row in dataframe_rows(latency_path)
        if str(row["method"]) in METHODS
    ]
    completed_latency = {
        (str(row["dataset"]), str(row["method"]))
        for row in latency_rows
    }
    if not args.skip_latency:
        for dataset in args.datasets:
            for method in METHODS:
                key = (dataset, method)
                if key in completed_latency and not args.overwrite:
                    continue
                print(
                    f"[latency] dataset={dataset} method={method}",
                    flush=True,
                )
                task = method_task(
                    args,
                    dataset,
                    method,
                    canonical_repeat,
                    serve_cpus,
                )
                task["raw_path"] = str(
                    args.out_root
                    / "raw_latency"
                    / f"{dataset}.{method}.jsonl"
                )
                result = run_child(context, latency_worker, task)
                latency_rows = [
                    row
                    for row in latency_rows
                    if (str(row["dataset"]), str(row["method"])) != key
                ]
                latency_rows.append(result["summary"])
                save_rows(latency_path, latency_rows)

    throughput_path = (
        args.out_root / "fixed2_throughput_summary.tsv"
    )
    throughput_rows = [
        row
        for row in dataframe_rows(throughput_path)
        if str(row["method"]) in METHODS
    ]
    completed_throughput = {
        (str(row["dataset"]), str(row["method"]))
        for row in throughput_rows
    }
    if not args.skip_throughput:
        for dataset in args.datasets:
            for method in METHODS:
                key = (dataset, method)
                if key in completed_throughput and not args.overwrite:
                    continue
                print(
                    f"[fixed2] dataset={dataset} method={method}",
                    flush=True,
                )
                task = method_task(
                    args,
                    dataset,
                    method,
                    canonical_repeat,
                    serve_cpus,
                )
                task["raw_path"] = str(
                    args.out_root
                    / "raw_fixed2"
                    / f"{dataset}.{method}.jsonl"
                )
                result = run_child(context, throughput_worker, task)
                throughput_rows = [
                    row
                    for row in throughput_rows
                    if (str(row["dataset"]), str(row["method"])) != key
                ]
                throughput_rows.append(result["summary"])
                save_rows(throughput_path, throughput_rows)

    if (
        build_path.is_file()
        and latency_path.is_file()
        and throughput_path.is_file()
    ):
        aggregate_report(args)
        validation = validate_outputs(args)
        manifest = json.loads(
            manifest_path.read_text(encoding="utf-8")
        )
        manifest["completion"] = {
            "finished_utc": time.strftime(
                "%Y-%m-%dT%H:%M:%SZ",
                time.gmtime(),
            ),
            "validation_hard_pass": bool(validation["hard_pass"]),
            "validation_file": str(
                args.out_root / "validation.json"
            ),
        }
        write_json(manifest_path, manifest)
        print(
            f"[done] {args.out_root / 'system_efficiency_report.md'}",
            flush=True,
        )


if __name__ == "__main__":
    main()
