#!/usr/bin/env python3
"""Build the SDE factorized query-document index from prepared components."""

from __future__ import annotations

import argparse
import gzip
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterator

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from src.sde.factorized_qdoc import factorized_qdoc_index as factor  # noqa: E402


RUNTIME_FILES = [
    factor.ARRAYS_FILE,
    factor.TERMS_FILE,
    factor.DOCNOS_FILE,
    factor.MANIFEST_FILE,
]


def runtime_size(path: Path) -> int:
    return sum(
        (path / name).stat().st_size
        for name in RUNTIME_FILES
        if (path / name).is_file()
    )


def indexing_terms(
    tokeniser: Any,
    stemmer: Any,
    stopwords: set[str],
    text: str,
) -> list[str]:
    """Apply the same Stopwords,PorterStemmer pipeline as Terrier."""
    terms: list[str] = []
    for value in tokeniser.getTokens(text or ""):
        token = str(value)
        if token in stopwords:
            continue
        stemmed = str(stemmer.stem(token))
        if stemmed:
            terms.append(stemmed)
    return terms


def flatten_postings(
    ids_by_term: list[list[int]],
    frequencies_by_term: list[list[int]],
    frequency_dtype: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    term_count = len(ids_by_term)
    indptr = np.zeros(term_count + 1, dtype=np.int64)
    for term_id, identifiers in enumerate(ids_by_term):
        indptr[term_id + 1] = indptr[term_id] + len(identifiers)
    identifiers = np.empty(int(indptr[-1]), dtype=np.int32)
    frequencies = np.empty(int(indptr[-1]), dtype=frequency_dtype)
    for term_id in range(term_count):
        start, end = int(indptr[term_id]), int(indptr[term_id + 1])
        identifiers[start:end] = ids_by_term[term_id]
        frequencies[start:end] = frequencies_by_term[term_id]
    return indptr, identifiers, frequencies


def component_groups(path: Path) -> Iterator[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as source:
        for line_number, raw in enumerate(source, start=1):
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
            if not isinstance(row.get("soft_terms"), list):
                raise ValueError(f"{path}:{line_number}: missing soft_terms")
            if not isinstance(row.get("qdocs"), list):
                raise ValueError(f"{path}:{line_number}: missing qdocs")
            yield row


def build_component_factorized_index(
    component_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    factor.sde.ensure_pyterrier()
    started = time.perf_counter()
    tokeniser = factor.sde.pt.terrier.TerrierTokeniser.java_tokeniser(
        factor.sde.pt.terrier.TerrierTokeniser._to_obj("english")
    )
    stemmer = factor.sde.pt.terrier.TerrierStemmer.porter
    stopwords = factor.sde.load_terrier_stopwords()

    output_dir.mkdir(parents=True, exist_ok=True)
    for name in RUNTIME_FILES:
        (output_dir / name).unlink(missing_ok=True)

    terms: list[str] = []
    term_to_id: dict[str, int] = {}
    raw_to_ids: dict[str, tuple[int, ...]] = {}
    document_frequency: list[int] = []
    shared_sources: list[list[int]] = []
    shared_frequencies: list[list[int]] = []
    delta_qdocs: list[list[int]] = []
    delta_frequencies: list[list[int]] = []

    qdoc_docnos: list[str] = []
    qdoc_lengths: list[int] = []
    qdoc_source: list[int] = []
    source_groups: list[list[int]] = []
    source_group_histogram: Counter[int] = Counter()
    seen_sources: set[str] = set()
    max_frequency = 0
    logical_tokens = 0
    reconstructed_tokens = 0
    input_query_tokens = 0
    input_soft_tokens = 0

    def term_id(term: str) -> int:
        existing = term_to_id.get(term)
        if existing is not None:
            return existing
        new_id = len(terms)
        term_to_id[term] = new_id
        terms.append(term)
        document_frequency.append(0)
        shared_sources.append([])
        shared_frequencies.append([])
        delta_qdocs.append([])
        delta_frequencies.append([])
        return new_id

    def raw_ids(token: Any) -> tuple[int, ...]:
        raw = str(token)
        cached = raw_to_ids.get(raw)
        if cached is not None:
            return cached
        if (
            not raw
            or not raw.isascii()
            or not raw.isalnum()
            or raw != raw.lower()
        ):
            raise ValueError(
                f"Component token must be lowercase ASCII alphanumeric: {raw!r}"
            )
        mapped = tuple(
            term_id(term)
            for term in indexing_terms(tokeniser, stemmer, stopwords, raw)
        )
        raw_to_ids[raw] = mapped
        return mapped

    def term_counter(raw_tokens: list[Any]) -> Counter[int]:
        return Counter(
            identifier
            for token in raw_tokens
            for identifier in raw_ids(token)
        )

    for group_row in component_groups(component_path):
        source = str(group_row.get("source_docno", "")).strip()
        if not source:
            raise ValueError("Component group is missing source_docno")
        if source in seen_sources:
            raise ValueError(f"Duplicate source group: {source}")
        seen_sources.add(source)

        qdoc_rows = group_row["qdocs"]
        if not qdoc_rows:
            raise ValueError(f"Source group has no qdocs: {source}")
        soft_terms = group_row["soft_terms"]
        input_soft_tokens += len(soft_terms)
        soft = term_counter(soft_terms)
        soft_length = sum(soft.values())

        query_rows: list[tuple[str, Counter[int]]] = []
        for qdoc_row in qdoc_rows:
            docno = str(qdoc_row.get("docno", "")).strip()
            raw_query_terms = qdoc_row.get("query_terms")
            if not docno or not isinstance(raw_query_terms, list):
                raise ValueError(f"Invalid qdoc component for source {source}")
            input_query_tokens += len(raw_query_terms)
            query_rows.append((docno, term_counter(raw_query_terms)))

        common_query = query_rows[0][1].copy()
        for _, query in query_rows[1:]:
            common_query &= query
        shared = soft + common_query
        source_id = len(source_groups)
        group = list(
            range(len(qdoc_docnos), len(qdoc_docnos) + len(query_rows))
        )
        source_groups.append(group)
        source_group_histogram[len(group)] += 1

        for identifier, frequency in shared.items():
            shared_sources[identifier].append(source_id)
            shared_frequencies[identifier].append(int(frequency))
            max_frequency = max(max_frequency, int(frequency))

        shared_length = sum(shared.values())
        for doc_id, (docno, query) in zip(group, query_rows):
            qdoc_docnos.append(docno)
            qdoc_source.append(source_id)
            length = soft_length + sum(query.values())
            qdoc_lengths.append(length)
            logical_tokens += length

            for identifier in soft.keys() | query.keys():
                document_frequency[identifier] += 1

            residual = query - common_query
            residual_length = 0
            for identifier, frequency in residual.items():
                delta_qdocs[identifier].append(doc_id)
                delta_frequencies[identifier].append(int(frequency))
                residual_length += int(frequency)
                max_frequency = max(max_frequency, int(frequency))
            rebuilt_length = shared_length + residual_length
            if rebuilt_length != length:
                raise AssertionError(
                    f"Length mismatch for {docno}: {rebuilt_length} != {length}"
                )
            reconstructed_tokens += rebuilt_length

    if not qdoc_docnos:
        raise ValueError(f"No qdocs in {component_path}")

    document_count = len(qdoc_docnos)
    source_counts = np.asarray(
        [len(group) for group in source_groups], dtype=np.int16
    )
    max_group_size = int(source_counts.max())
    source_qdocs = np.full(
        (len(source_groups), max_group_size), -1, dtype=np.int32
    )
    for source_id, group in enumerate(source_groups):
        source_qdocs[source_id, : len(group)] = group
    qdoc_source_array = np.asarray(qdoc_source, dtype=np.int32)

    reconstructed_df = np.zeros(len(terms), dtype=np.int32)
    for identifier in range(len(terms)):
        shared_source_set = set(shared_sources[identifier])
        shared_documents = sum(
            len(source_groups[source_id]) for source_id in shared_source_set
        )
        residual_only_documents = sum(
            qdoc_source[doc_id] not in shared_source_set
            for doc_id in delta_qdocs[identifier]
        )
        reconstructed_df[identifier] = shared_documents + residual_only_documents
    document_frequency_array = np.asarray(document_frequency, dtype=np.int32)
    if not np.array_equal(document_frequency_array, reconstructed_df):
        bad = np.flatnonzero(document_frequency_array != reconstructed_df)
        raise AssertionError(
            f"DF mismatch for {len(bad)} terms; first={int(bad[0])}"
        )
    if reconstructed_tokens != logical_tokens:
        raise AssertionError(
            f"Token mismatch: {reconstructed_tokens} != {logical_tokens}"
        )

    frequency_dtype = np.uint16 if max_frequency <= 65535 else np.int32
    shared_indptr, shared_source_ids, shared_tf = flatten_postings(
        shared_sources, shared_frequencies, frequency_dtype
    )
    delta_indptr, delta_doc_ids, delta_tf = flatten_postings(
        delta_qdocs, delta_frequencies, frequency_dtype
    )
    qdoc_lengths_array = np.asarray(qdoc_lengths, dtype=np.int32)

    np.savez_compressed(
        output_dir / factor.ARRAYS_FILE,
        shared_indptr=shared_indptr,
        shared_source_ids=shared_source_ids,
        shared_tf=shared_tf,
        delta_indptr=delta_indptr,
        delta_doc_ids=delta_doc_ids,
        delta_tf=delta_tf,
        source_qdocs=source_qdocs,
        source_counts=source_counts,
        qdoc_source=qdoc_source_array,
        qdoc_lengths=qdoc_lengths_array,
        original_df=document_frequency_array,
    )
    factor.write_gzip_json(output_dir / factor.TERMS_FILE, terms)
    factor.write_gzip_json(output_dir / factor.DOCNOS_FILE, qdoc_docnos)

    shared_pointer_count = int(shared_indptr[-1])
    delta_pointer_count = int(delta_indptr[-1])
    logical_pointer_count = int(document_frequency_array.sum())
    payload_bytes = sum(
        (output_dir / name).stat().st_size
        for name in [factor.ARRAYS_FILE, factor.TERMS_FILE, factor.DOCNOS_FILE]
    )
    manifest = {
        "format": "factorized-qdoc-v1",
        "source_components": str(component_path.resolve()),
        "document_count": document_count,
        "source_document_count": len(source_groups),
        "term_count": len(terms),
        "logical_tokens": logical_tokens,
        "average_document_length": logical_tokens / document_count,
        "logical_pointer_count": logical_pointer_count,
        "shared_pointer_count": shared_pointer_count,
        "delta_pointer_count": delta_pointer_count,
        "factor_pointer_count": shared_pointer_count + delta_pointer_count,
        "pointer_ratio": (
            (shared_pointer_count + delta_pointer_count) / logical_pointer_count
        ),
        "payload_bytes": payload_bytes,
        "max_group_size": max_group_size,
        "source_group_histogram": {
            str(size): count
            for size, count in sorted(source_group_histogram.items())
        },
        "frequency_dtype": np.dtype(frequency_dtype).name,
        "raw_token_count": len(raw_to_ids),
        "input_query_tokens_once": input_query_tokens,
        "input_soft_tokens_once": input_soft_tokens,
        "tokenization_mode": "structure-aware Terrier term normalization",
        "build_seconds": time.perf_counter() - started,
        "checks": {
            "all_document_lengths_exact": True,
            "collection_tokens_exact": True,
            "all_document_frequencies_exact": True,
        },
    }
    factor.write_json(output_dir / factor.MANIFEST_FILE, manifest)
    manifest["total_factorized_bytes"] = runtime_size(output_dir)
    factor.write_json(output_dir / factor.MANIFEST_FILE, manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--components", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build_component_factorized_index(
        args.components.resolve(), args.output_dir.resolve()
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
