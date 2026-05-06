#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

from common import DATASET_SPECS, read_jsonl, summarize, write_csv, write_json


METHOD_ORDER = ["d2qpp_full", "d2qpp_qgen_only", "sde_trace", "sde_decoded6"]
METHOD_LABELS = {
    "d2qpp_full": "D2Q++ Full",
    "d2qpp_qgen_only": "D2Q++ QGen-Only",
    "sde_trace": "SDE Trace",
    "sde_decoded6": "SDE Decoded-Only 6",
}
METHOD_SCOPES = {
    "d2qpp_full": "topic+keyword+qgen",
    "d2qpp_qgen_only": "final qgen only",
    "sde_trace": "qgen+retrieval+trace audit",
    "sde_decoded6": "qgen only",
}


def load_existing_jsonl(path: Path) -> List[Dict[str, Any]]:
    return read_jsonl(path) if path.is_file() else []


def load_doc_logs(output_root: Path) -> List[Dict[str, Any]]:
    raw = output_root / "raw"
    rows: List[Dict[str, Any]] = []
    for filename in (
        "d2qpp_full_doc_logs.jsonl",
        "d2qpp_qgen_only_doc_logs.jsonl",
        "sde_trace_doc_logs.jsonl",
        "sde_decoded6_doc_logs.jsonl",
    ):
        rows.extend(load_existing_jsonl(raw / filename))
    return rows


def load_stage_logs(output_root: Path) -> List[Dict[str, Any]]:
    raw = output_root / "raw"
    rows: List[Dict[str, Any]] = []
    for path in raw.glob("*stage_logs.jsonl"):
        rows.extend(load_existing_jsonl(path))
    return rows


def success_rows(rows: List[Dict[str, Any]], method: str, dataset: str = "") -> List[Dict[str, Any]]:
    out = [row for row in rows if row.get("method") == method and row.get("status") == "ok"]
    if dataset:
        out = [row for row in out if row.get("dataset") == dataset]
    return out


def stage_total_by_method(stage_rows: List[Dict[str, Any]]) -> Dict[str, float]:
    totals: Dict[str, float] = defaultdict(float)
    for row in stage_rows:
        if row.get("status") != "ok":
            continue
        method = str(row.get("method"))
        totals[method] += float(row.get("total_time_sec") or 0.0)
    return totals


def stage_total_by_method_dataset(stage_rows: List[Dict[str, Any]]) -> Dict[Tuple[str, str], float]:
    totals: Dict[Tuple[str, str], float] = defaultdict(float)
    for row in stage_rows:
        if row.get("status") != "ok":
            continue
        method = str(row.get("method"))
        dataset = str(row.get("dataset") or "")
        totals[(method, dataset)] += float(row.get("total_time_sec") or 0.0)
    return totals


def adjusted_times_from_stage_total(rows: List[Dict[str, Any]], stage_total: float) -> List[float]:
    vals = [float(row.get("wall_time_sec") or 0.0) for row in rows]
    if not vals or not stage_total:
        return vals
    stage_mean = stage_total / len(rows)
    logged_mean = sum(vals) / len(vals)
    overhead = max(0.0, stage_mean - logged_mean)
    return [value + overhead for value in vals]


def adjusted_times(rows: List[Dict[str, Any]], method: str, stage_totals: Dict[str, float]) -> List[float]:
    return adjusted_times_from_stage_total(rows, stage_totals.get(method, 0.0))


def mean_value(rows: List[Dict[str, Any]], key: str) -> float:
    vals = [float(row.get(key) or 0.0) for row in rows]
    return sum(vals) / len(vals) if vals else 0.0


def mean_kb(rows: List[Dict[str, Any]], key: str) -> float:
    return mean_value(rows, key) / 1024.0


def retrieval_kb_per_doc(method: str, rows: List[Dict[str, Any]]) -> float:
    if not rows:
        return 0.0
    if method == "sde_trace":
        vals = [float(row.get("expansion_entry_bytes") or 0.0) for row in rows]
    else:
        vals = [float(row.get("decoded_text_bytes") or 0.0) for row in rows]
    return (sum(vals) / len(vals)) / 1024.0


def total_with_trace_kb_per_doc(method: str, rows: List[Dict[str, Any]]) -> float:
    if not rows:
        return 0.0
    vals = []
    for row in rows:
        retrieval_bytes = (
            float(row.get("expansion_entry_bytes") or 0.0)
            if method == "sde_trace"
            else float(row.get("decoded_text_bytes") or 0.0)
        )
        vals.append(retrieval_bytes + float(row.get("trace_gzip_bytes") or 0.0))
    return (sum(vals) / len(vals)) / 1024.0


def make_main_table(doc_rows: List[Dict[str, Any]], stage_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    stage_totals = stage_total_by_method(stage_rows)
    rows = []
    method_stats: Dict[str, Dict[str, float]] = {}
    for method in METHOD_ORDER:
        ok = success_rows(doc_rows, method)
        if not ok:
            continue
        times = adjusted_times(ok, method, stage_totals)
        stats = summarize(times)
        method_stats[method] = stats
        rows.append(
            {
                "Method": METHOD_LABELS[method],
                "Scope": METHOD_SCOPES[method],
                "Gen/doc": f"{mean_value(ok, 'num_generations'):.2f}",
                "LLM calls/doc": f"{mean_value(ok, 'llm_calls'):.2f}",
                "sec/doc mean": f"{stats['mean']:.6f}",
                "sec/doc p50": f"{stats['p50']:.6f}",
                "sec/doc p95": f"{stats['p95']:.6f}",
                "output tok/doc": f"{mean_value(ok, 'output_tokens'):.2f}",
                "decoded KB/doc": f"{mean_kb(ok, 'decoded_text_bytes'):.6f}",
                "trace raw KB/doc": f"{mean_kb(ok, 'trace_raw_bytes'):.6f}",
                "trace gzip KB/doc": f"{mean_kb(ok, 'trace_gzip_bytes'):.6f}",
                "expansion KB/doc": f"{mean_kb(ok, 'expansion_entry_bytes'):.6f}",
                "retrieval KB/doc": f"{retrieval_kb_per_doc(method, ok):.6f}",
                "total with trace KB/doc": f"{total_with_trace_kb_per_doc(method, ok):.6f}",
                "relative time": "",
                "relative retrieval storage": "",
                "relative total-with-trace storage": "",
                "success docs": str(len(ok)),
                "failed docs": str(
                    sum(1 for row in doc_rows if row.get("method") == method and row.get("status") != "ok")
                ),
            }
        )

    baseline_time = method_stats.get("d2qpp_full", {}).get("mean")
    baseline_storage = None
    for row in rows:
        if row["Method"] == METHOD_LABELS.get("d2qpp_full"):
            baseline_storage = float(row["retrieval KB/doc"])
    if baseline_time is None and "d2qpp_qgen_only" in method_stats:
        baseline_time = method_stats["d2qpp_qgen_only"]["mean"]
    if baseline_storage is None:
        for row in rows:
            if row["Method"] == METHOD_LABELS.get("d2qpp_qgen_only"):
                baseline_storage = float(row["retrieval KB/doc"])
                break

    for row in rows:
        sec_mean = float(row["sec/doc mean"])
        storage = float(row["retrieval KB/doc"])
        total_with_trace = float(row["total with trace KB/doc"])
        row["relative time"] = f"{(sec_mean / baseline_time):.4f}x" if baseline_time else ""
        row["relative retrieval storage"] = f"{(storage / baseline_storage):.4f}x" if baseline_storage else ""
        row["relative total-with-trace storage"] = (
            f"{(total_with_trace / baseline_storage):.4f}x" if baseline_storage else ""
        )
    return rows


def make_per_dataset_table(doc_rows: List[Dict[str, Any]], stage_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    stage_totals = stage_total_by_method_dataset(stage_rows)
    rows = []
    for spec in DATASET_SPECS:
        dataset = spec.slug
        full = success_rows(doc_rows, "d2qpp_full", dataset)
        qgen = success_rows(doc_rows, "d2qpp_qgen_only", dataset)
        sde = success_rows(doc_rows, "sde_trace", dataset)
        full_mean = (
            summarize(adjusted_times_from_stage_total(full, stage_totals.get(("d2qpp_full", dataset), 0.0)))["mean"]
            if full
            else 0.0
        )
        qgen_mean = (
            summarize(adjusted_times_from_stage_total(qgen, stage_totals.get(("d2qpp_qgen_only", dataset), 0.0)))[
                "mean"
            ]
            if qgen
            else 0.0
        )
        sde_mean = (
            summarize(adjusted_times_from_stage_total(sde, stage_totals.get(("sde_trace", dataset), 0.0)))["mean"]
            if sde
            else 0.0
        )
        sde_trace_kb = (
            mean_value(sde, "trace_gzip_bytes") / 1024.0 if sde else 0.0
        )
        sde_retrieval_kb = (
            mean_value(sde, "expansion_entry_bytes") / 1024.0 if sde else 0.0
        )
        rows.append(
            {
                "Dataset": spec.display_name,
                "D2Q++ Full sec/doc": f"{full_mean:.6f}" if full else "",
                "D2Q++ QGen sec/doc": f"{qgen_mean:.6f}" if qgen else "",
                "SDE sec/doc": f"{sde_mean:.6f}" if sde else "",
                "SDE/D2Q++ Full time": f"{(sde_mean / full_mean):.6f}" if full_mean else "",
                "SDE/D2Q++ QGen time": f"{(sde_mean / qgen_mean):.6f}" if qgen_mean else "",
                "SDE retrieval KB/doc": f"{sde_retrieval_kb:.6f}" if sde else "",
                "SDE trace KB/doc": f"{sde_trace_kb:.6f}" if sde else "",
                "D2Q++ Full success docs": str(len(full)),
                "D2Q++ QGen success docs": str(len(qgen)),
                "SDE success docs": str(len(sde)),
            }
        )
    return rows


def make_stage_breakdown(stage_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    totals: Dict[str, float] = defaultdict(float)
    for row in stage_rows:
        if row.get("status") == "ok":
            totals[str(row.get("method"))] += float(row.get("total_time_sec") or 0.0)
    out = []
    for row in stage_rows:
        if row.get("status") != "ok":
            continue
        method = str(row.get("method"))
        total = float(row.get("total_time_sec") or 0.0)
        out.append(
            {
                "Method": METHOD_LABELS.get(method, method),
                "Stage": row.get("stage"),
                "total sec": f"{total:.6f}",
                "sec/doc": f"{float(row.get('sec_per_doc') or 0.0):.6f}",
                "percent of method time": f"{(100.0 * total / totals[method]):.4f}" if totals[method] else "",
                "dataset": row.get("dataset", ""),
            }
        )
    return out


def make_summary(doc_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {"methods": {}, "warnings": []}
    for method in sorted(set(str(row.get("method")) for row in doc_rows)):
        method_rows = [row for row in doc_rows if row.get("method") == method]
        ok_rows = [row for row in method_rows if row.get("status") == "ok"]
        summary["methods"][method] = {
            "total_rows": len(method_rows),
            "success_rows": len(ok_rows),
            "failed_rows": len(method_rows) - len(ok_rows),
        }
        for spec in DATASET_SPECS:
            ds_rows = [row for row in method_rows if row.get("dataset") == spec.slug]
            if not ds_rows:
                continue
            ok = sum(1 for row in ds_rows if row.get("status") == "ok")
            rate = ok / len(ds_rows)
            if rate < 0.95:
                summary["warnings"].append(
                    {
                        "method": method,
                        "dataset": spec.slug,
                        "success_rate": rate,
                        "success_rows": ok,
                        "total_rows": len(ds_rows),
                    }
                )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate PCE JSONL logs into paper tables.")
    parser.add_argument("--output_root", required=True)
    args = parser.parse_args()

    output_root = Path(args.output_root)
    tables_root = output_root / "tables"
    doc_rows = load_doc_logs(output_root)
    stage_rows = load_stage_logs(output_root)
    if not doc_rows:
        raise SystemExit(f"[ERROR] no doc logs found under {output_root / 'raw'}")

    main_rows = make_main_table(doc_rows, stage_rows)
    per_dataset_rows = make_per_dataset_table(doc_rows, stage_rows)
    stage_breakdown_rows = make_stage_breakdown(stage_rows)
    summary = make_summary(doc_rows)

    write_csv(
        tables_root / "main_cost_table.csv",
        main_rows,
        [
            "Method",
            "Scope",
            "Gen/doc",
            "LLM calls/doc",
            "sec/doc mean",
            "sec/doc p50",
            "sec/doc p95",
            "output tok/doc",
            "decoded KB/doc",
            "trace raw KB/doc",
            "trace gzip KB/doc",
            "expansion KB/doc",
            "retrieval KB/doc",
            "total with trace KB/doc",
            "relative time",
            "relative retrieval storage",
            "relative total-with-trace storage",
            "success docs",
            "failed docs",
        ],
    )
    write_csv(
        tables_root / "per_dataset_breakdown.csv",
        per_dataset_rows,
        [
            "Dataset",
            "D2Q++ Full sec/doc",
            "D2Q++ QGen sec/doc",
            "SDE sec/doc",
            "SDE/D2Q++ Full time",
            "SDE/D2Q++ QGen time",
            "SDE retrieval KB/doc",
            "SDE trace KB/doc",
            "D2Q++ Full success docs",
            "D2Q++ QGen success docs",
            "SDE success docs",
        ],
    )
    write_csv(
        tables_root / "stage_breakdown.csv",
        stage_breakdown_rows,
        ["Method", "Stage", "total sec", "sec/doc", "percent of method time", "dataset"],
    )
    write_json(tables_root / "aggregation_summary.json", summary)
    print(f"[aggregate] wrote {tables_root / 'main_cost_table.csv'}")
    print(f"[aggregate] wrote {tables_root / 'per_dataset_breakdown.csv'}")
    print(f"[aggregate] wrote {tables_root / 'stage_breakdown.csv'}")
    if summary["warnings"]:
        print(f"[aggregate] warnings={len(summary['warnings'])} see aggregation_summary.json")


if __name__ == "__main__":
    main()
