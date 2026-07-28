#!/usr/bin/env python3
"""Prepare SDE query and soft-term components for factorized indexing."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from src.sde.factorized_qdoc import sde_qdoc_terms as lexicalization  # noqa: E402


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def build_soft_terms(expansion: Path, model_path: Path) -> dict[str, list[str]]:
    return lexicalization.build_soft_query_terms(
        expansion,
        model_path,
        sample_idx_max=0,
        prob_threshold=0.01,
        soft_topk_per_step=5,
        max_soft_terms_per_doc=256,
        term_weight_mode="repeat_by_score",
        repeat_score_scale=3.0,
        repeat_max_times=3,
    )


def prepare_components(
    expansion: Path,
    model_path: Path,
    output: Path,
    stats_output: Path,
) -> dict[str, Any]:
    """Store shared document soft terms once and query-specific terms per qdoc."""
    started = time.perf_counter()
    document_soft_terms = build_soft_terms(expansion, model_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    seen_qdocs: set[tuple[str, tuple[str, ...]]] = set()
    seen_sources: set[str] = set()
    current_source: str | None = None
    current_soft: list[str] = []
    current_qdocs: list[dict[str, Any]] = []
    qdoc_count = 0
    source_count = 0
    query_token_count = 0
    soft_token_count = 0
    logical_qdoc_tokens = 0
    digest = hashlib.sha256()

    def flush_group(output_file: Any) -> None:
        nonlocal source_count, soft_token_count
        if current_source is None:
            return
        if current_source in seen_sources:
            raise ValueError(f"Non-contiguous source group: {current_source}")
        seen_sources.add(current_source)
        row = {
            "source_docno": current_source,
            "soft_terms": current_soft,
            "qdocs": current_qdocs,
        }
        encoded = json.dumps(
            row, ensure_ascii=False, separators=(",", ":")
        ).encode("utf-8")
        output_file.write(encoded + b"\n")
        digest.update(encoded + b"\n")
        source_count += 1
        soft_token_count += len(current_soft)

    with (
        expansion.open("r", encoding="utf-8") as source,
        gzip.open(output, "wb") as destination,
    ):
        for line_number, raw in enumerate(source, start=1):
            if not raw.strip():
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{expansion}:{line_number}: invalid JSON"
                ) from exc
            document_id = str(row.get("pos_id", "")).strip()
            if not document_id:
                continue
            if not lexicalization.row_passes_prompt_filter(row, set(), set()):
                continue
            if not lexicalization.row_passes_quality_filter(row, "none", 0.0):
                continue
            sample_idx = row.get("sample_idx")
            if sample_idx is not None and int(sample_idx) > 0:
                continue

            query_terms = lexicalization.normalize(
                str(row.get("query_text", "")).strip(),
                lowercase=True,
                remove_stopwords=True,
                min_token_len=2,
            )
            soft_terms = document_soft_terms.get(document_id, [])
            logical_terms = tuple(query_terms + soft_terms)
            if not logical_terms:
                continue
            key = (document_id, logical_terms)
            if key in seen_qdocs:
                continue
            seen_qdocs.add(key)

            if current_source is not None and document_id != current_source:
                flush_group(destination)
                current_qdocs = []
            if document_id != current_source:
                current_source = document_id
                current_soft = list(soft_terms)

            current_qdocs.append(
                {
                    "docno": f"{document_id}##{qdoc_count}",
                    "query_terms": query_terms,
                }
            )
            qdoc_count += 1
            query_token_count += len(query_terms)
            logical_qdoc_tokens += len(logical_terms)
        flush_group(destination)

    if not qdoc_count:
        raise ValueError(f"No query-document components were produced from {expansion}")

    stats = {
        "format": "factorized-qdoc-components-v1",
        "source_expansion": str(expansion.resolve()),
        "component_file": str(output.resolve()),
        "component_sha256": digest.hexdigest(),
        "component_bytes": output.stat().st_size,
        "qdocs": qdoc_count,
        "source_documents": source_count,
        "source_documents_with_soft_terms": len(document_soft_terms),
        "query_tokens_once": query_token_count,
        "soft_tokens_once": soft_token_count,
        "logical_qdoc_tokens": logical_qdoc_tokens,
        "preparation_sec": time.perf_counter() - started,
        "parameters": {
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
        },
    }
    write_json(stats_output, stats)
    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expansion", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stats-output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stats = prepare_components(
        args.expansion.resolve(),
        args.model_path.resolve(),
        args.output.resolve(),
        args.stats_output.resolve(),
    )
    print(json.dumps(stats, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
