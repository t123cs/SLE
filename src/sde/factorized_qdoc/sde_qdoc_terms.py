#!/usr/bin/env python3
"""Convert stored SDE traces into document-level soft terms."""

from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer


STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "to", "of", "in", "on", "at", "by", "for",
    "with", "from", "as", "or", "and", "but", "not", "it", "its", "i",
    "we", "you", "he", "she", "they", "this", "that", "my", "your", "our",
    "their", "what", "which", "who", "if", "so", "up", "out", "can", "no",
    "more", "also", "about", "than", "then", "just", "into", "over", "after",
    "all", "when", "there", "me", "him", "her", "them",
}
BAD_DECODED_TERMS = {"s", "t", "re", "ve", "ll", "d", "m"}


def row_passes_prompt_filter(
    row: dict[str, Any],
    prompt_include_ids: set[str],
    prompt_exclude_ids: set[str],
) -> bool:
    prompt_id = str(row.get("prompt_id", "")).strip()
    if prompt_include_ids and prompt_id not in prompt_include_ids:
        return False
    if prompt_exclude_ids and prompt_id in prompt_exclude_ids:
        return False
    return True


def synthetic_query_quality_score(
    row: dict[str, Any], mode: str
) -> float | None:
    if mode == "none":
        return None
    if mode != "avg_top1_prob":
        raise ValueError(f"Unsupported synthetic query quality mode: {mode}")
    top_probabilities: list[float] = []
    for step in row.get("probs", []):
        if isinstance(step, list) and step:
            try:
                top_probabilities.append(float(step[0]))
            except (TypeError, ValueError):
                continue
    if not top_probabilities:
        return None
    return sum(top_probabilities) / len(top_probabilities)


def row_passes_quality_filter(
    row: dict[str, Any], quality_mode: str, quality_min: float
) -> bool:
    if quality_mode == "none":
        return True
    score = synthetic_query_quality_score(row, quality_mode)
    return score is not None and score >= quality_min


def normalize(
    text: str,
    lowercase: bool = True,
    remove_stopwords: bool = True,
    min_token_len: int = 2,
) -> list[str]:
    if lowercase:
        text = text.lower()
    terms = re.findall(r"[a-z0-9]+", text)
    return [
        term
        for term in terms
        if len(term) >= min_token_len
        and (not remove_stopwords or term not in STOPWORDS)
    ]


def decoded_token_to_terms(
    text: str,
    lowercase: bool,
    remove_stopwords: bool,
    min_token_len: int,
    drop_numeric_terms: bool,
    extra_bad_terms: set[str] | None = None,
) -> list[str]:
    blocked = BAD_DECODED_TERMS | (extra_bad_terms or set())
    return [
        term
        for term in normalize(text, lowercase, remove_stopwords, min_token_len)
        if term not in blocked
        and (not drop_numeric_terms or not term.isdigit())
        and not term.isdigit()
        and re.search(r"[a-z]", term)
    ]


def token_piece_has_word_start(piece: str) -> bool:
    return bool(piece) and piece.startswith(("Ġ", "▁"))


def should_drop_short_continuation_piece(
    piece: str, terms: list[str], continuation_min_len: int
) -> bool:
    if token_piece_has_word_start(piece) or not terms:
        return False
    alpha_terms = [term for term in terms if re.search(r"[a-z]", term)]
    return bool(alpha_terms) and all(
        len(term) < continuation_min_len for term in alpha_terms
    )


def build_soft_query_terms(
    expansion_jsonl: Path,
    tokenizer_path: Path,
    *,
    sample_idx_max: int | None = 0,
    prompt_include_ids: set[str] | None = None,
    prompt_exclude_ids: set[str] | None = None,
    synthetic_query_quality_mode: str = "none",
    synthetic_query_quality_min: float = 0.0,
    prob_threshold: float = 0.01,
    soft_topk_per_step: int | None = 5,
    soft_topp_per_step: float | None = None,
    max_soft_terms_per_doc: int = 256,
    lowercase: bool = True,
    remove_stopwords: bool = True,
    min_token_len: int = 2,
    drop_numeric_terms: bool = True,
    blacklisted_token_ids: set[int] | None = None,
    drop_short_continuation_terms: bool = False,
    continuation_min_len: int = 5,
    extra_bad_terms: set[str] | None = None,
    term_weight_mode: str = "repeat_by_score",
    repeat_score_scale: float = 3.0,
    repeat_max_times: int = 3,
) -> dict[str, list[str]]:
    """Decode candidate token IDs and aggregate weighted soft terms per document."""
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        trust_remote_code=True,
        local_files_only=True,
    )
    include_ids = prompt_include_ids or set()
    exclude_ids = prompt_exclude_ids or set()
    blacklisted_ids = blacklisted_token_ids or {220, 128009}
    document_scores: dict[str, dict[str, float]] = defaultdict(
        lambda: defaultdict(float)
    )

    with expansion_jsonl.open("r", encoding="utf-8") as source:
        for raw in source:
            if not raw.strip():
                continue
            row = json.loads(raw)
            document_id = str(row.get("pos_id", "")).strip()
            if not document_id:
                continue
            if not row_passes_prompt_filter(row, include_ids, exclude_ids):
                continue
            if not row_passes_quality_filter(
                row, synthetic_query_quality_mode, synthetic_query_quality_min
            ):
                continue
            sample_idx = row.get("sample_idx")
            if (
                sample_idx_max is not None
                and sample_idx is not None
                and int(sample_idx) > sample_idx_max
            ):
                continue

            for step_ids, step_probs in zip(
                row.get("indices", []), row.get("probs", [])
            ):
                if not isinstance(step_ids, list) or not isinstance(step_probs, list):
                    continue
                selected: list[tuple[Any, Any]] = []
                cumulative_probability = 0.0
                for token_id, probability in zip(step_ids, step_probs):
                    if (
                        soft_topk_per_step is not None
                        and len(selected) >= soft_topk_per_step
                    ):
                        break
                    selected.append((token_id, probability))
                    try:
                        cumulative_probability += float(probability)
                    except (TypeError, ValueError):
                        continue
                    if (
                        soft_topp_per_step is not None
                        and cumulative_probability >= soft_topp_per_step
                    ):
                        break

                for token_id, probability in selected:
                    try:
                        token_id = int(token_id)
                        probability = float(probability)
                    except (TypeError, ValueError):
                        continue
                    if token_id in blacklisted_ids or probability < prob_threshold:
                        continue
                    try:
                        raw_token = str(tokenizer.convert_ids_to_tokens(token_id))
                        token_text = tokenizer.decode([token_id])
                    except Exception:
                        continue
                    terms = decoded_token_to_terms(
                        token_text,
                        lowercase,
                        remove_stopwords,
                        min_token_len,
                        drop_numeric_terms,
                        extra_bad_terms,
                    )
                    if (
                        drop_short_continuation_terms
                        and should_drop_short_continuation_piece(
                            raw_token, terms, continuation_min_len
                        )
                    ):
                        continue
                    for term in terms:
                        document_scores[document_id][term] += probability

    document_terms: dict[str, list[str]] = {}
    for document_id, scores in document_scores.items():
        terms: list[str] = []
        for term, score in sorted(
            scores.items(), key=lambda item: item[1], reverse=True
        )[:max_soft_terms_per_doc]:
            repeats = 1
            if term_weight_mode == "repeat_by_score":
                repeats = max(
                    1,
                    min(
                        repeat_max_times,
                        int(math.ceil(score * repeat_score_scale)),
                    ),
                )
            elif term_weight_mode != "uniform":
                raise ValueError(f"Unsupported term weight mode: {term_weight_mode}")
            terms.extend([term] * repeats)
        document_terms[document_id] = terms
    return document_terms
