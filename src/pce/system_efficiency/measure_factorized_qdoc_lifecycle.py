#!/usr/bin/env python3
"""Measure SDE factorized-index construction and out-of-place refresh."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import shutil
import sys
import time
from pathlib import Path
from queue import Empty
from typing import Any

import pandas as pd
import psutil

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from src.pce.system_efficiency import measure_system_efficiency_formal as formal  # noqa: E402
from src.sde.factorized_qdoc import (  # noqa: E402
    build_factorized_qdoc_components as factor_builder,
    factorized_qdoc_index as factor,
    prepare_factorized_qdoc_components as component_prepare,
)


DATASETS = ["nfcorpus", "scidocs", "fiqa-2018", "arguana", "scifact"]
DEFAULT_FORMAL_ROOT = REPO_ROOT / "results" / "system_efficiency_formal"
DEFAULT_TRACE_ROOT = REPO_ROOT / "data" / "traces"
DEFAULT_MODEL_PATH = REPO_ROOT / "models" / "tokenizer"
DEFAULT_OUT_ROOT = REPO_ROOT / "results" / "factorized_qdoc_lifecycle"


def now() -> float:
    return time.perf_counter()


def component_path(root: Path, dataset: str) -> Path:
    return root / "prepared" / dataset / "qdoc_components.jsonl.gz"


def component_stats_path(root: Path, dataset: str) -> Path:
    return root / "prepared" / dataset / "component_stats.json"


def factor_output_path(root: Path, repeat: int, dataset: str) -> Path:
    return root / "factor_indexes" / f"repeat_{repeat}" / dataset


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def prepare_components(
    args: argparse.Namespace, dataset: str
) -> dict[str, Any]:
    output = component_path(args.prepared_root, dataset)
    stats_output = component_stats_path(args.prepared_root, dataset)
    if output.is_file() and stats_output.is_file() and not args.overwrite:
        return json.loads(stats_output.read_text(encoding="utf-8"))
    expansion = (
        args.trace_root / dataset / "train_data_multitemplate_unfiltered.jsonl"
    )
    if not expansion.is_file():
        raise FileNotFoundError(expansion)
    return component_prepare.prepare_components(
        expansion, args.model_path, output, stats_output
    )


def build_worker(task: dict[str, Any], result_queue: Any) -> None:
    try:
        formal.set_affinity(task["cpu_ids"])
        factor.sde.ensure_pyterrier()
        dataset = str(task["dataset"])
        factor_dir = Path(task["factor_output"])
        shutil.rmtree(factor_dir, ignore_errors=True)
        factor_dir.parent.mkdir(parents=True, exist_ok=True)

        process = psutil.Process(os.getpid())
        io_start = process.io_counters()
        cpu_start = formal.cpu_times_sec(process)
        rss_start = formal.rss_mb(process)
        started = now()
        with formal.PeakRSS() as monitor:
            manifest = factor_builder.build_component_factorized_index(
                Path(task["prepared_path"]), factor_dir
            )
        refresh_wall_sec = now() - started
        io_end = process.io_counters()

        factorized_index = factor.FactorizedQDocIndex(factor_dir)
        structure_checks = factorized_index.structure_checks()
        factor_checks_exact = all(
            bool(value) for value in manifest["checks"].values()
        )
        structure_checks_exact = all(structure_checks.values())
        if not factor_checks_exact or not structure_checks_exact:
            raise AssertionError(
                f"{dataset}: factorized checks failed: {structure_checks}"
            )

        result_queue.put(
            {
                "ok": True,
                "dataset": dataset,
                "repeat": int(task["repeat"]),
                "qdocs": int(manifest["document_count"]),
                "factor_build_sec": refresh_wall_sec,
                "refresh_wall_sec": refresh_wall_sec,
                "build_cpu_sec": formal.cpu_times_sec(process) - cpu_start,
                "build_start_rss_mb": rss_start,
                "build_peak_rss_mb": monitor.peak_mb,
                "write_bytes": max(
                    0, int(io_end.write_bytes - io_start.write_bytes)
                ),
                "factorized_bytes": factor.runtime_size(factor_dir),
                "logical_num_documents": int(manifest["document_count"]),
                "logical_num_terms": int(manifest["term_count"]),
                "logical_num_pointers": int(manifest["logical_pointer_count"]),
                "logical_num_tokens": int(manifest["logical_tokens"]),
                "factor_pointer_count": int(manifest["factor_pointer_count"]),
                "factor_shared_pointer_count": int(
                    manifest["shared_pointer_count"]
                ),
                "factor_delta_pointer_count": int(
                    manifest["delta_pointer_count"]
                ),
                "factor_checks_exact": factor_checks_exact,
                "structure_checks_exact": structure_checks_exact,
                "cpu_ids": json.dumps(task["cpu_ids"]),
            }
        )
    except Exception as exc:
        result_queue.put({"ok": False, "error": repr(exc), "task": task})


def run_child(
    task: dict[str, Any], timeout_sec: float = 7200.0
) -> dict[str, Any]:
    context = mp.get_context("spawn")
    queue = context.Queue()
    process = context.Process(target=build_worker, args=(task, queue))
    process.start()
    try:
        result = queue.get(timeout=timeout_sec)
    except Empty as exc:
        process.terminate()
        process.join(timeout=10)
        raise TimeoutError(task) from exc
    process.join(timeout=30)
    if process.is_alive():
        process.terminate()
        process.join(timeout=10)
    if not result.get("ok"):
        raise RuntimeError(result)
    return result


def save_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, sep="\t", index=False)


def load_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return pd.read_csv(path, sep="\t").to_dict("records")


def aggregate(args: argparse.Namespace) -> None:
    builds = pd.read_csv(args.out_root / "build_measurements.tsv", sep="\t")
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
    medians = (
        builds.groupby("dataset", as_index=False)
        .agg(
            refresh_wall_sec=("refresh_wall_sec", "median"),
            factor_build_sec=("factor_build_sec", "median"),
            write_bytes=("write_bytes", "median"),
            build_peak_rss_mb=("build_peak_rss_mb", "max"),
            factorized_bytes=("factorized_bytes", "max"),
            factor_pointer_count=("factor_pointer_count", "max"),
        )
    )
    anchor_lifecycle = pd.read_csv(
        args.formal_root / "report_index_lifecycle.tsv", sep="\t"
    )
    bm25 = anchor_lifecycle[anchor_lifecycle["method"] == "bm25"][
        ["dataset", "initial_build_sec", "build_peak_rss_mb", "write_mb"]
    ].rename(
        columns={
            "initial_build_sec": "bm25_build_sec",
            "build_peak_rss_mb": "bm25_peak_rss_mb",
            "write_mb": "bm25_write_mb",
        }
    )
    per_dataset = medians.merge(bm25, on="dataset", how="left")
    per_dataset["initial_build_sec"] = (
        per_dataset["bm25_build_sec"] + per_dataset["refresh_wall_sec"]
    )
    per_dataset["full_expansion_refresh_sec"] = per_dataset[
        "refresh_wall_sec"
    ]
    per_dataset["initial_write_mb"] = (
        per_dataset["bm25_write_mb"]
        + per_dataset["write_bytes"] / (1024.0**2)
    )
    per_dataset["initial_peak_rss_mb"] = per_dataset[
        ["bm25_peak_rss_mb", "build_peak_rss_mb"]
    ].max(axis=1)
    per_dataset.to_csv(
        args.out_root / "factorized_lifecycle_by_dataset.tsv",
        sep="\t",
        index=False,
    )

    anchor_methods = ["bm25", "d2qpp_full", "d2qpp_qgen_only"]
    table_rows: list[dict[str, Any]] = []
    for method in anchor_methods:
        subset = anchor_lifecycle[
            (anchor_lifecycle["method"] == method)
            & (anchor_lifecycle["dataset"].isin(args.datasets))
        ]
        table_rows.append(
            {
                "method": method,
                "display_name": str(subset.iloc[0]["display_name"]),
                "initial_build_sec": float(subset["initial_build_sec"].sum()),
                "full_expansion_refresh_sec": (
                    ""
                    if method == "bm25"
                    else float(subset["full_expansion_refresh_sec"].sum())
                ),
                "write_mb": float(subset["write_mb"].sum()),
                "build_peak_rss_mb": float(subset["build_peak_rss_mb"].max()),
            }
        )
    table_rows.append(
        {
            "method": "factorized_qdoc",
            "display_name": "SDE, auxiliary index",
            "initial_build_sec": float(per_dataset["initial_build_sec"].sum()),
            "full_expansion_refresh_sec": float(
                per_dataset["full_expansion_refresh_sec"].sum()
            ),
            "write_mb": float(per_dataset["initial_write_mb"].sum()),
            "build_peak_rss_mb": float(
                per_dataset["initial_peak_rss_mb"].max()
            ),
        }
    )
    table = pd.DataFrame(table_rows)
    table.to_csv(
        args.out_root / "table6_factorized_qdoc.tsv", sep="\t", index=False
    )

    deterministic_columns = [
        "qdocs",
        "logical_num_documents",
        "logical_num_terms",
        "logical_num_pointers",
        "logical_num_tokens",
        "factor_pointer_count",
        "factor_shared_pointer_count",
        "factor_delta_pointer_count",
    ]
    nondeterministic: dict[str, Any] = {}
    for column in deterministic_columns:
        counts = builds.groupby("dataset")[column].nunique()
        bad = counts[counts != 1]
        if not bad.empty:
            nondeterministic[column] = bad.to_dict()
    expected_rows = len(args.datasets) * args.build_repeats
    hard_pass = bool(
        not nondeterministic
        and builds["factor_checks_exact"].astype(bool).all()
        and builds["structure_checks_exact"].astype(bool).all()
        and len(builds) == expected_rows
    )
    packaging_size_jitter: dict[str, dict[str, int]] = {}
    for dataset, subset in builds.groupby("dataset"):
        low = int(subset["factorized_bytes"].min())
        high = int(subset["factorized_bytes"].max())
        packaging_size_jitter[str(dataset)] = {
            "min_bytes": low,
            "max_bytes": high,
            "span_bytes": high - low,
        }
    tokenization_modes = {
        json.loads(
            (
                factor_output_path(args.out_root, repeat, dataset)
                / factor.MANIFEST_FILE
            ).read_text(encoding="utf-8")
        )["tokenization_mode"]
        for repeat in range(args.build_repeats)
        for dataset in args.datasets
    }
    hard_pass = hard_pass and len(tokenization_modes) == 1
    validation = {
        "hard_pass": hard_pass,
        "expected_build_rows": expected_rows,
        "actual_build_rows": len(builds),
        "nondeterministic_index_stats": nondeterministic,
        "manifest_metadata_size_jitter": packaging_size_jitter,
        "tokenization_modes": sorted(tokenization_modes),
        "all_factor_checks_exact": bool(
            builds["factor_checks_exact"].astype(bool).all()
        ),
        "all_structure_checks_exact": bool(
            builds["structure_checks_exact"].astype(bool).all()
        ),
        "build_path": "SDE traces -> factorized components -> factorized index",
        "hardware": {
            "hostname": anchor_manifest.get("hostname"),
            "cpu_model": cpu_model,
            "anchor_created_utc": anchor_manifest.get("created_utc"),
            "lscpu": anchor_manifest.get("lscpu"),
        },
    }
    write_json(args.out_root / "validation.json", validation)

    lines = [
        "# Factorized qdoc index lifecycle",
        "",
        f"Hard pass: `{str(hard_pass).lower()}`",
        "",
        "| Method | Initial build s | 100% expansion refresh s | "
        "Bytes written MB | Build peak RSS MB |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in table.itertuples(index=False):
        refresh = (
            ""
            if row.full_expansion_refresh_sec == ""
            else f"{float(row.full_expansion_refresh_sec):.3f}"
        )
        lines.append(
            f"| {row.display_name} | {row.initial_build_sec:.3f} | {refresh} | "
            f"{row.write_mb:.3f} | {row.build_peak_rss_mb:.3f} |"
        )
    lines.extend(
        [
            "",
            "The SDE expansion route is built from structure-aware components in "
            "one process; Terrier term normalization and payload compression are "
            "inside the refresh timer.",
            "Trace preparation is outside the indexing timer, matching the formal "
            "baseline protocol.",
        ]
    )
    (args.out_root / "report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--formal-root", type=Path, default=DEFAULT_FORMAL_ROOT)
    parser.add_argument("--trace-root", type=Path, default=DEFAULT_TRACE_ROOT)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument(
        "--prepared-root",
        type=Path,
        help="Reuse prepared factorized components from another result root.",
    )
    parser.add_argument("--datasets", nargs="+", default=DATASETS)
    parser.add_argument("--build-repeats", type=int, default=3)
    parser.add_argument("--build-cpu-count", type=int, default=8)
    parser.add_argument("--skip-prepare", action="store_true")
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.formal_root = args.formal_root.resolve()
    args.trace_root = args.trace_root.resolve()
    args.model_path = args.model_path.resolve()
    args.out_root = args.out_root.resolve()
    args.prepared_root = (
        args.prepared_root.resolve()
        if args.prepared_root is not None
        else args.out_root
    )
    args.out_root.mkdir(parents=True, exist_ok=True)
    physical = formal.physical_cpu_ids()
    build_cpus = physical[: max(1, min(args.build_cpu_count, len(physical)))]

    if not args.skip_prepare:
        for dataset in args.datasets:
            print(f"[prepare] {dataset}", flush=True)
            stats = prepare_components(args, dataset)
            print(
                f"[prepare] {dataset} qdocs={stats['qdocs']} "
                f"seconds={stats['preparation_sec']:.3f}",
                flush=True,
            )

    measurements_path = args.out_root / "build_measurements.tsv"
    rows = load_rows(measurements_path)
    completed = {
        (str(row["dataset"]), int(row["repeat"])) for row in rows
    }
    if not args.skip_build:
        for repeat in range(args.build_repeats):
            datasets = (
                args.datasets[repeat % len(args.datasets) :]
                + args.datasets[: repeat % len(args.datasets)]
            )
            for dataset in datasets:
                key = (dataset, repeat)
                if key in completed and not args.overwrite:
                    continue
                print(f"[build] repeat={repeat} dataset={dataset}", flush=True)
                task = {
                    "dataset": dataset,
                    "repeat": repeat,
                    "cpu_ids": list(build_cpus),
                    "prepared_path": str(
                        component_path(args.prepared_root, dataset)
                    ),
                    "factor_output": str(
                        factor_output_path(args.out_root, repeat, dataset)
                    ),
                }
                result = run_child(task)
                rows = [
                    row
                    for row in rows
                    if (str(row["dataset"]), int(row["repeat"])) != key
                ]
                rows.append(
                    {key: value for key, value in result.items() if key != "ok"}
                )
                save_rows(measurements_path, rows)
    if measurements_path.is_file():
        aggregate(args)
        print(
            pd.read_csv(
                args.out_root / "table6_factorized_qdoc.tsv", sep="\t"
            ).to_string(index=False),
            flush=True,
        )
        print(f"[done] {args.out_root}", flush=True)


if __name__ == "__main__":
    main()
