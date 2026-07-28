"""Prepare deterministic, no-LLM Doc2Query++ query-line simulations."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence


D2QPP_FULL_KB_PER_DOC = {
    "nfcorpus": 2.097,
    "scidocs": 2.199,
    "fiqa-2018": 1.743,
    "arguana": 1.915,
    "scifact": 2.155,
}
D2QPP_FULL_SEC_PER_DOC = {
    "nfcorpus": 7.153,
    "scidocs": 6.541,
    "fiqa-2018": 5.702,
    "arguana": 6.018,
    "scifact": 7.608,
}
D2QPP_QGEN_ONLY_STORAGE_RATIO = 1.620 / 1.911
D2QPP_QGEN_ONLY_TIME_RATIO = 4.613 / 6.108

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "have",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "why",
    "with",
}


def clean_query_line(line: str) -> str:
    line = str(line or "").strip()
    line = re.sub(r"^[-*\u2022]\s*", "", line)
    line = re.sub(r"^\(?\d+[\).\]]\s*", "", line)
    line = line.strip(" \t\r\n\"'`")
    return re.sub(r"\s+", " ", line).strip()


def extract_query_lines(text: str) -> list[str]:
    candidates: list[str] = []
    for raw in str(text or "").replace("\r", "\n").split("\n"):
        line = clean_query_line(raw)
        if not line:
            continue
        lower = line.lower()
        if lower.startswith("here are") or lower in {
            "alternative search queries:",
            "generated queries:",
        }:
            continue
        if len(line) >= 3:
            candidates.append(line)
    if not candidates:
        line = clean_query_line(text)
        if line:
            candidates.append(line)
    return candidates


def unique_preserve(
    lines: Iterable[str],
    limit: int | None = None,
) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for line in lines:
        key = re.sub(r"\s+", " ", line.lower()).strip()
        if not key or key in seen:
            continue
        seen.add(key)
        output.append(line)
        if limit is not None and len(output) >= limit:
            break
    return output


def term_pool(lines: Sequence[str]) -> list[str]:
    terms: list[str] = []
    seen: set[str] = set()
    for line in lines:
        for token in re.findall(r"[A-Za-z0-9][A-Za-z0-9\-]{2,}", line):
            key = token.lower()
            if key in STOPWORDS or key in seen:
                continue
            seen.add(key)
            terms.append(token)
    return terms


def augment_query(
    base: str,
    other: str,
    terms: Sequence[str],
    offset: int,
) -> str:
    base_terms = {
        token.lower()
        for token in re.findall(r"[A-Za-z0-9][A-Za-z0-9\-]{2,}", base)
    }
    extras: list[str] = []
    for index in range(len(terms)):
        term = terms[(offset + index) % len(terms)] if terms else ""
        if term and term.lower() not in base_terms:
            extras.append(term)
        if len(extras) >= 3:
            break
    if extras:
        return f"{base} {' '.join(extras)}"
    if other and other.lower() != base.lower():
        return f"{base} {other}"
    return base


def make_30_query_lines(
    source_lines: Sequence[str],
    queries_per_doc: int,
) -> list[str]:
    base = unique_preserve(source_lines) or ["empty query"]
    terms = term_pool(base)
    output: list[str] = []
    index = 0
    while len(output) < queries_per_doc:
        primary = base[index % len(base)]
        if index < len(base):
            output.append(primary)
        else:
            secondary = base[(index * 7 + 3) % len(base)]
            output.append(augment_query(primary, secondary, terms, index))
        index += 1
    return output[:queries_per_doc]


def fit_lines_to_budget(
    lines: Sequence[str],
    target_bytes: int,
) -> tuple[str, int]:
    if target_bytes <= 0:
        return "", 0
    text = "\n".join(lines).strip()
    raw = text.encode("utf-8")
    if len(raw) > target_bytes:
        text = raw[:target_bytes].decode("utf-8", errors="ignore").strip()
        return text, len(text.encode("utf-8"))
    if len(raw) == target_bytes:
        return text, len(raw)

    terms = term_pool(lines) or ["query"]
    padded = list(lines)
    cursor = 0
    while len("\n".join(padded).encode("utf-8")) < target_bytes:
        line_index = cursor % len(padded)
        term = terms[cursor % len(terms)]
        candidate = list(padded)
        candidate[line_index] = f"{candidate[line_index]} {term}"
        if len("\n".join(candidate).encode("utf-8")) > target_bytes:
            break
        padded = candidate
        cursor += 1
        if cursor > target_bytes:
            break
    text = "\n".join(padded).strip()
    return text, len(text.encode("utf-8"))


def load_trace_query_lines(
    trace_path: Path,
    sample_idx_max: int | None,
) -> tuple[dict[str, list[str]], dict[str, Any]]:
    grouped: dict[str, list[str]] = defaultdict(list)
    rows = 0
    kept_rows = 0
    split_lines = 0
    prompts: dict[str, int] = defaultdict(int)
    samples: dict[str, int] = defaultdict(int)
    with trace_path.open("r", encoding="utf-8") as source:
        for raw in source:
            if not raw.strip():
                continue
            rows += 1
            row = json.loads(raw)
            sample_idx = row.get("sample_idx")
            samples[str(sample_idx)] += 1
            if sample_idx_max is not None:
                try:
                    if int(sample_idx) > sample_idx_max:
                        continue
                except (TypeError, ValueError):
                    continue
            document_id = str(
                row.get("pos_id") or row.get("doc_id") or row.get("docno") or ""
            )
            if not document_id:
                continue
            lines = extract_query_lines(row.get("query_text", ""))
            if not lines:
                continue
            kept_rows += 1
            split_lines += len(lines)
            prompts[str(row.get("prompt_id") or row.get("prompt_name") or "")] += 1
            grouped[document_id].extend(lines)
    return dict(grouped), {
        "trace_rows": rows,
        "kept_trace_rows": kept_rows,
        "docs_with_query_lines": len(grouped),
        "split_query_lines": split_lines,
        "sample_idx_max": sample_idx_max,
        "samples_seen": dict(samples),
        "prompts_kept": dict(prompts),
    }


def d2qpp_queryline_specs(dataset: str) -> list[dict[str, Any]]:
    full_kb = D2QPP_FULL_KB_PER_DOC[dataset]
    full_sec = D2QPP_FULL_SEC_PER_DOC[dataset]
    return [
        {
            "key": "d2qpp_full",
            "target_kb_per_doc": full_kb,
            "generation_sec_per_doc": full_sec,
            "budget": "30 queries",
        },
        {
            "key": "d2qpp_qgen_only",
            "target_kb_per_doc": full_kb * D2QPP_QGEN_ONLY_STORAGE_RATIO,
            "generation_sec_per_doc": (
                full_sec * D2QPP_QGEN_ONLY_TIME_RATIO
            ),
            "budget": "30 queries",
        },
    ]
