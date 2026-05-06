#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

from common import ensure_dir, write_csv, write_json


METHODS = ["BM25", "Doc2Query++ Full", "SDE segmented compact"]
DISPLAY_METHOD = {
    "BM25": "BM25",
    "Doc2Query++ Full": "Doc2Query++ Full",
    "SDE segmented compact": "SDE",
}


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fin:
        return list(csv.DictReader(fin))


def by_method(rows: Iterable[Dict[str, str]]) -> Dict[str, Dict[str, str]]:
    return {row["Method"]: row for row in rows}


def fnum(value: Any) -> float:
    if value in (None, "", "-"):
        return 0.0
    return float(str(value))


def load_lucene_index_stats(indexes: Dict[str, str], logical_docs: int) -> Dict[str, Dict[str, float]]:
    from pyserini.index.lucene import IndexReader

    out: Dict[str, Dict[str, float]] = {}
    for method, index_path in indexes.items():
        reader = IndexReader(index_path)
        stats = reader.stats()
        postings = sum(term.df for term in reader.terms())
        out[method] = {
            "physical_docs": float(stats["documents"]),
            "logical_docs": float(logical_docs),
            "vocab": float(stats["unique_terms"]),
            "postings": float(postings),
            "total_terms": float(stats["total_terms"]),
            "indexed_terms_per_logical_doc": float(stats["total_terms"]) / float(logical_docs),
            "indexed_terms_per_physical_doc": float(stats["total_terms"]) / float(stats["documents"]),
        }
    return out


def latex_escape(text: str) -> str:
    return (
        text.replace("&", r"\&")
        .replace("%", r"\%")
        .replace("_", r"\_")
        .replace("#", r"\#")
    )


def write_time_latex(path: Path, rows: List[Dict[str, str]]) -> None:
    lines = [
        r"\begin{table}[!t]",
        r"\centering",
        r"\caption{End-to-end offline cost and CPU-only query-time latency on the deterministic 10,000-document sample. Offline cost includes generation/filtering/persistence and Lucene index construction. Query latency is measured with top-1000 candidates; SDE retrieves 2000 segment hits and deduplicates them to top-1000 logical documents.}",
        r"\label{tab:rq3-e2e-time}",
        r"\resizebox{\columnwidth}{!}{%",
        r"\begin{tabular}{lrrrrrrr}",
        r"\toprule",
        r"Method & Gen/filter s/doc & Index s/doc & Offline s/doc & P50 ms/q & P95 ms/q & P99 ms/q & QPS \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            "{} & {} & {} & {} & {} & {} & {} & {} \\\\".format(
                latex_escape(row["Method"]),
                row["Gen/filter sec/doc"],
                row["Index build sec/doc"],
                row["Offline E2E sec/doc"],
                row["P50 ms/q"],
                row["P95 ms/q"],
                row["P99 ms/q"],
                row["QPS"],
            )
        )
    lines.extend([r"\bottomrule", r"\end{tabular}}", r"\end{table}", ""])
    ensure_dir(path.parent)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_footprint_latex(path: Path, rows: List[Dict[str, str]]) -> None:
    lines = [
        r"\begin{table}[!t]",
        r"\centering",
        r"\caption{Deployed Lucene index footprint on the deterministic 10,000-document sample. SDE uses one physical Lucene index containing original-text and auxiliary-expansion segment documents.}",
        r"\label{tab:rq3-index-footprint}",
        r"\resizebox{\columnwidth}{!}{%",
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Method & Index MB & Ratio vs BM25 & Vocab & Postings & Terms/doc & Expansion KB/doc \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            "{} & {} & {} & {} & {} & {} & {} \\\\".format(
                latex_escape(row["Method"]),
                row["On-Disk Index MB"],
                row["Ratio vs BM25"],
                row["Vocab"],
                row["Postings"],
                row["Indexed terms/logical doc"],
                row["Expansion KB/doc"],
            )
        )
    lines.extend([r"\bottomrule", r"\end{tabular}}", r"\end{table}", ""])
    ensure_dir(path.parent)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create final RQ3 end-to-end time and footprint tables from existing statistics.")
    parser.add_argument("--output_root", default="results/pce_full")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    tables_root = ensure_dir(output_root / "tables")

    main_cost = by_method(read_csv(tables_root / "main_cost_table.csv"))
    lucene_latency = by_method(read_csv(tables_root / "lucene_segmented_query_latency.csv"))
    lucene_footprint = by_method(read_csv(tables_root / "lucene_segmented_index_footprint.csv"))
    sample_docs = sum(1 for _ in (output_root / "sample_docs.jsonl").open("r", encoding="utf-8"))
    metadata = json.loads((output_root / "raw" / "lucene_segmented_metadata.json").read_text(encoding="utf-8"))
    expansion_sizes = json.loads(
        (output_root / "system_costs" / "lucene_segmented" / "collections" / "expansion_text_sizes.json").read_text(
            encoding="utf-8"
        )
    )

    index_stats = load_lucene_index_stats(metadata["indexes"], sample_docs)

    gen_filter_sec_doc = {
        "BM25": 0.0,
        "Doc2Query++ Full": fnum(main_cost["D2Q++ Full"]["sec/doc mean"]),
        "SDE segmented compact": fnum(main_cost["SDE Trace"]["sec/doc mean"]),
    }
    expansion_kb_doc = {
        "BM25": 0.0,
        "Doc2Query++ Full": fnum(main_cost["D2Q++ Full"]["retrieval KB/doc"]),
        "SDE segmented compact": fnum(expansion_sizes["sde_compact_kb_per_doc"]),
    }

    time_rows: List[Dict[str, str]] = []
    footprint_rows: List[Dict[str, str]] = []
    bm25_mb = fnum(lucene_footprint["BM25"]["On-Disk Index MB"])

    for method in METHODS:
        latency = lucene_latency[method]
        footprint = lucene_footprint[method]
        stats = index_stats[method]
        index_sec_doc = fnum(footprint["Build sec"]) / sample_docs
        offline_sec_doc = gen_filter_sec_doc[method] + index_sec_doc

        time_rows.append(
            {
                "Method": DISPLAY_METHOD[method],
                "Indexing Form": footprint["Indexing Form"],
                "Gen/filter sec/doc": f"{gen_filter_sec_doc[method]:.6f}",
                "Index build sec/doc": f"{index_sec_doc:.6f}",
                "Offline E2E sec/doc": f"{offline_sec_doc:.6f}",
                "P50 ms/q": latency["P50 ms/q"],
                "P95 ms/q": latency["P95 ms/q"],
                "P99 ms/q": latency["P99 ms/q"],
                "QPS": latency["QPS"],
                "Candidate Union": latency["Candidate Union"],
                "Lucene Hits": latency["Lucene Hits"],
            }
        )
        footprint_rows.append(
            {
                "Method": DISPLAY_METHOD[method],
                "Indexing Form": footprint["Indexing Form"],
                "Physical Docs": str(int(stats["physical_docs"])),
                "On-Disk Index MB": footprint["On-Disk Index MB"],
                "Ratio vs BM25": f"{fnum(footprint['On-Disk Index MB']) / bm25_mb:.3f}x",
                "Vocab": str(int(stats["vocab"])),
                "Postings": str(int(stats["postings"])),
                "Indexed terms/logical doc": f"{stats['indexed_terms_per_logical_doc']:.2f}",
                "Indexed terms/physical doc": f"{stats['indexed_terms_per_physical_doc']:.2f}",
                "Expansion KB/doc": f"{expansion_kb_doc[method]:.3f}",
            }
        )

    write_csv(
        tables_root / "rq3_end_to_end_time.csv",
        time_rows,
        [
            "Method",
            "Indexing Form",
            "Gen/filter sec/doc",
            "Index build sec/doc",
            "Offline E2E sec/doc",
            "P50 ms/q",
            "P95 ms/q",
            "P99 ms/q",
            "QPS",
            "Candidate Union",
            "Lucene Hits",
        ],
    )
    write_csv(
        tables_root / "rq3_deployed_footprint.csv",
        footprint_rows,
        [
            "Method",
            "Indexing Form",
            "Physical Docs",
            "On-Disk Index MB",
            "Ratio vs BM25",
            "Vocab",
            "Postings",
            "Indexed terms/logical doc",
            "Indexed terms/physical doc",
            "Expansion KB/doc",
        ],
    )
    write_time_latex(tables_root / "rq3_end_to_end_time.tex", time_rows)
    write_footprint_latex(tables_root / "rq3_deployed_footprint.tex", footprint_rows)
    write_json(
        output_root / "raw" / "rq3_table_metadata.json",
        {
            "sample_docs": sample_docs,
            "source_tables": [
                str(tables_root / "main_cost_table.csv"),
                str(tables_root / "lucene_segmented_query_latency.csv"),
                str(tables_root / "lucene_segmented_index_footprint.csv"),
            ],
            "notes": [
                "No new retrieval was run by this script.",
                "Offline E2E sec/doc is generation/filtering/persistence plus Lucene index build divided by sample docs.",
                "SDE segmented compact uses 2000 Lucene segment hits and deduplicates to top-1000 logical documents.",
            ],
        },
    )
    print(f"[rq3] wrote {tables_root / 'rq3_end_to_end_time.csv'}")
    print(f"[rq3] wrote {tables_root / 'rq3_deployed_footprint.csv'}")
    print(f"[rq3] wrote {tables_root / 'rq3_end_to_end_time.tex'}")
    print(f"[rq3] wrote {tables_root / 'rq3_deployed_footprint.tex'}")


if __name__ == "__main__":
    main()
