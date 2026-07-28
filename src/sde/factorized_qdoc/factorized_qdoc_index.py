#!/usr/bin/env python3
"""Runtime reader and BM25 scorer for the SDE factorized qdoc index."""

from __future__ import annotations

import gzip
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from src.sde import terrier_utils as sde  # noqa: E402


ARRAYS_FILE = "factorized_postings.npz"
TERMS_FILE = "terms.json.gz"
DOCNOS_FILE = "qdoc_docnos.json.gz"
MANIFEST_FILE = "manifest.json"
RUNTIME_FILES = [ARRAYS_FILE, TERMS_FILE, DOCNOS_FILE, MANIFEST_FILE]


def directory_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def runtime_size(path: Path) -> int:
    return sum(
        (path / name).stat().st_size
        for name in RUNTIME_FILES
        if (path / name).is_file()
    )


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def write_gzip_json(path: Path, payload: Any) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as destination:
        json.dump(payload, destination, ensure_ascii=False, separators=(",", ":"))


def read_gzip_json(path: Path) -> Any:
    with gzip.open(path, "rt", encoding="utf-8") as source:
        return json.load(source)


def source_docno(qdocno: str) -> str:
    source, marker, _ = qdocno.rpartition("##")
    if not marker:
        raise ValueError(f"Unexpected query-document id: {qdocno}")
    return source


class FactorizedQDocIndex:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.manifest = json.loads(
            (path / MANIFEST_FILE).read_text(encoding="utf-8")
        )
        with np.load(path / ARRAYS_FILE) as arrays:
            for name in arrays.files:
                setattr(self, name, arrays[name])
        self.terms: list[str] = read_gzip_json(path / TERMS_FILE)
        self.qdoc_docnos: list[str] = read_gzip_json(path / DOCNOS_FILE)
        self.term_to_id = {
            term: term_id for term_id, term in enumerate(self.terms)
        }
        self.document_count = int(self.manifest["document_count"])
        self.average_length = float(self.manifest["average_document_length"])
        self.max_group_size = int(self.manifest["max_group_size"])
        self.k1, self.b, self.k3 = 0.9, 0.4, 8.0
        self._tf = np.zeros(self.document_count, dtype=np.int32)
        self._scores = np.zeros(self.document_count, dtype=np.float64)
        self._matched = np.zeros(self.document_count, dtype=np.bool_)

    def structure_checks(self) -> dict[str, bool]:
        """Check the self-contained factorized representation."""
        term_count = len(self.terms)
        source_count = len(self.source_counts)
        checks: dict[str, bool] = {
            "manifest_checks": bool(self.manifest.get("checks"))
            and all(bool(value) for value in self.manifest["checks"].values()),
            "document_metadata_shapes": (
                len(self.qdoc_docnos)
                == len(self.qdoc_lengths)
                == len(self.qdoc_source)
                == self.document_count
            ),
            "document_ids_unique": len(set(self.qdoc_docnos))
            == self.document_count,
            "term_metadata_shapes": (
                len(self.original_df) == term_count
                and len(self.shared_indptr) == term_count + 1
                and len(self.delta_indptr) == term_count + 1
            ),
            "shared_posting_shapes": (
                len(self.shared_source_ids) == len(self.shared_tf)
                and int(self.shared_indptr[-1]) == len(self.shared_source_ids)
                and bool(np.all(np.diff(self.shared_indptr) >= 0))
            ),
            "delta_posting_shapes": (
                len(self.delta_doc_ids) == len(self.delta_tf)
                and int(self.delta_indptr[-1]) == len(self.delta_doc_ids)
                and bool(np.all(np.diff(self.delta_indptr) >= 0))
            ),
            "source_table_shape": (
                self.source_qdocs.ndim == 2
                and self.source_qdocs.shape
                == (source_count, self.max_group_size)
            ),
            "positive_document_lengths": bool(np.all(self.qdoc_lengths > 0)),
            "document_frequencies_valid": bool(
                np.all(self.original_df >= 0)
                and np.all(self.original_df <= self.document_count)
            ),
            "source_ids_valid": bool(
                source_count > 0
                and np.all(self.qdoc_source >= 0)
                and np.all(self.qdoc_source < source_count)
                and np.all(self.shared_source_ids >= 0)
                and np.all(self.shared_source_ids < source_count)
            ),
            "delta_document_ids_valid": bool(
                np.all(self.delta_doc_ids >= 0)
                and np.all(self.delta_doc_ids < self.document_count)
            ),
            "average_length_valid": bool(
                math.isfinite(self.average_length) and self.average_length > 0.0
            ),
        }
        source_rows_valid = True
        source_names_valid = True
        for source_id, count_value in enumerate(self.source_counts):
            count = int(count_value)
            if count <= 0 or count > self.max_group_size:
                source_rows_valid = False
                break
            document_ids = self.source_qdocs[source_id, :count]
            padding = self.source_qdocs[source_id, count:]
            if (
                np.any(document_ids < 0)
                or np.any(document_ids >= self.document_count)
                or np.any(padding != -1)
                or np.any(self.qdoc_source[document_ids] != source_id)
            ):
                source_rows_valid = False
                break
            names = {
                source_docno(self.qdoc_docnos[int(document_id)])
                for document_id in document_ids
            }
            if len(names) != 1:
                source_names_valid = False
        checks["source_rows_valid"] = source_rows_valid
        checks["source_document_names_consistent"] = source_names_valid
        return checks

    def score_terms(
        self,
        query_terms: Iterable[str] | dict[str, float | list[float]],
        topk: int,
    ) -> list[tuple[int, str, float]]:
        self._scores.fill(0.0)
        self._matched.fill(False)
        if isinstance(query_terms, dict):
            query_weights = query_terms
        else:
            frequencies = Counter(query_terms)
            maximum = max(frequencies.values(), default=1)
            query_weights = {
                term: value / maximum for term, value in frequencies.items()
            }
        for term, query_frequency in query_weights.items():
            term_id = self.term_to_id.get(term)
            if term_id is None:
                continue
            self._tf.fill(0)
            shared_start = int(self.shared_indptr[term_id])
            shared_end = int(self.shared_indptr[term_id + 1])
            sources = self.shared_source_ids[shared_start:shared_end]
            if len(sources):
                source_docs = self.source_qdocs[sources].reshape(-1)
                source_tf = np.repeat(
                    self.shared_tf[shared_start:shared_end], self.max_group_size
                )
                valid = source_docs >= 0
                self._tf[source_docs[valid]] = source_tf[valid]
            delta_start = int(self.delta_indptr[term_id])
            delta_end = int(self.delta_indptr[term_id + 1])
            residual_docs = self.delta_doc_ids[delta_start:delta_end]
            if len(residual_docs):
                np.add.at(
                    self._tf,
                    residual_docs,
                    self.delta_tf[delta_start:delta_end],
                )
            matched = np.flatnonzero(self._tf)
            if not len(matched):
                continue
            self._matched[matched] = True
            term_frequency = self._tf[matched].astype(np.float64)
            document_length = self.qdoc_lengths[matched].astype(np.float64)
            document_frequency = int(self.original_df[term_id])
            inverse_document_frequency = math.log2(
                (self.document_count - document_frequency + 0.5)
                / (document_frequency + 0.5)
            )
            normalizer = self.k1 * (
                (1.0 - self.b)
                + self.b * document_length / self.average_length
            )
            operator_weights = (
                query_frequency
                if isinstance(query_frequency, (list, tuple))
                else [query_frequency]
            )
            query_weight = sum(
                (self.k3 + 1.0) * weight / (self.k3 + weight)
                for weight in operator_weights
            )
            self._scores[matched] += (
                inverse_document_frequency
                * (
                    (self.k1 + 1.0)
                    * term_frequency
                    / (normalizer + term_frequency)
                )
                * query_weight
            )
        candidates = np.flatnonzero(self._matched)
        if not len(candidates):
            return []
        order = np.lexsort((candidates, -self._scores[candidates]))
        selected = candidates[order[:topk]]
        return [
            (
                int(doc_id),
                self.qdoc_docnos[int(doc_id)],
                float(self._scores[doc_id]),
            )
            for doc_id in selected
        ]


class QueryProcessor:
    def __init__(self) -> None:
        sde.ensure_pyterrier()
        self.parser = sde.pt.terrier.J.TerrierQLParser()
        self.to_matching_terms = sde.pt.terrier.J.TerrierQLToMatchingQueryTerms()
        self.pipeline = sde.pt.terrier.J.ApplyTermPipeline()

    def terms(self, query: str) -> dict[str, list[float]]:
        request = sde.pt.terrier.J.Request()
        request.setQueryID("factorized-qdoc")
        request.setOriginalQuery(query or "")
        self.parser.process(None, request)
        self.to_matching_terms.process(None, request)
        self.pipeline.process(None, request)
        entries = [
            (str(entry.getKey().toString()), float(entry.getValue().getWeight()))
            for entry in request.getMatchingQueryTerms()
        ]
        maximum = max((weight for _, weight in entries), default=1.0)
        if maximum <= 0.0:
            return {}
        weights: dict[str, list[float]] = {}
        for term, weight in entries:
            weights.setdefault(term, []).append(weight / maximum)
        return weights
