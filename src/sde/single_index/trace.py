#!/usr/bin/env python3
"""Trace parsing and Terrier-space term extraction for single-index SDE."""

from __future__ import annotations

import json
import math
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Iterator

try:
    from transformers import AutoConfig, AutoTokenizer
except ImportError:
    AutoConfig = None
    AutoTokenizer = None


WORD_RE = re.compile(r"[A-Za-z0-9]")
LIST_MARKER_RE = re.compile(r"(?m)^\s*(?:[-*]|\d+[.)])\s+")
DECODED_SPECIAL_TOKEN_RE = re.compile(
    r"(?:<\|.*?\|>|<｜｜.*?｜｜>|end[\s_▁-]*of[\s_▁-]*sentence)",
    re.IGNORECASE,
)


def clean_generated_query(text: Any) -> str:
    cleaned = str(text or "").strip().strip("\"'")
    cleaned = LIST_MARKER_RE.sub(" ", cleaned)
    return " ".join(cleaned.split())


def candidate_trace_format(path: Path) -> str:
    """Return the candidate representation used by the first non-empty row."""
    with path.open(encoding="utf-8") as source:
        for raw in source:
            if not raw.strip():
                continue
            row = json.loads(raw)
            if isinstance(row.get("decoded_candidates"), list):
                return "decoded_candidates"
            if isinstance(row.get("indices"), list) and isinstance(
                row.get("probs"), list
            ):
                return "token_ids"
            return "unsupported"
    return "empty"


def load_candidate_tokenizer(
    trace_paths: Iterable[Path],
    model_path: Path | None,
) -> Any | None:
    """Load the generation tokenizer only when token-ID traces require it."""
    token_id_paths = [
        path
        for path in trace_paths
        if path.is_file() and candidate_trace_format(path) == "token_ids"
    ]
    if not token_id_paths:
        return None
    if model_path is None:
        joined = ", ".join(str(path) for path in token_id_paths)
        raise RuntimeError(
            "Token-ID traces require --model_path with the generation tokenizer: "
            f"{joined}"
        )
    if AutoConfig is None or AutoTokenizer is None:
        raise RuntimeError(
            "Token-ID traces require the optional transformers dependency."
        )
    config = AutoConfig.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True,
    )
    if getattr(config, "model_type", None) != "llama":
        raise RuntimeError(
            f"Expected a Llama tokenizer at {model_path}, found "
            f"model_type={getattr(config, 'model_type', None)!r}."
        )
    return AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True,
    )


def terrier_terms(
    tokeniser: Any,
    stemmer: Any,
    stopwords: set[str],
    text: Any,
    min_len: int = 1,
) -> list[str]:
    terms: list[str] = []
    for raw_token in tokeniser.getTokens(str(text or "")):
        token = str(raw_token)
        if token in stopwords:
            continue
        stemmed = str(stemmer.stem(token))
        if stemmed and len(stemmed) >= min_len and stemmed not in stopwords:
            terms.append(stemmed)
    return terms


@lru_cache(maxsize=200000)
def decode_piece(tokenizer: Any, token_id: int) -> str:
    return str(
        tokenizer.decode(
            [int(token_id)],
            clean_up_tokenization_spaces=False,
        )
    )


def decoded_candidate_steps(row: dict[str, Any]) -> list[list[Any]] | None:
    steps = row.get("decoded_candidates")
    if not isinstance(steps, list):
        return None
    return [candidates if isinstance(candidates, list) else [] for candidates in steps]


def normalized_candidate_steps(
    row: dict[str, Any],
    tokenizer: Any | None,
) -> tuple[list[list[Any]] | None, str | None]:
    """Return text/probability candidates for decoded and token-ID traces."""
    decoded_steps = decoded_candidate_steps(row)
    if decoded_steps is not None:
        for position, candidates in enumerate(decoded_steps):
            for rank, candidate in enumerate(candidates, start=1):
                if not isinstance(candidate, dict):
                    raise ValueError(
                        f"decoded candidate {position}:{rank} must be an object"
                    )
                if not isinstance(candidate.get("chosen"), bool):
                    raise ValueError(
                        f"decoded candidate {position}:{rank} must contain a "
                        "boolean chosen field"
                    )
        return decoded_steps, "decoded_candidates"

    indices = row.get("indices")
    probabilities = row.get("probs")
    if not isinstance(indices, list) or not isinstance(probabilities, list):
        return None, None
    if len(indices) != len(probabilities):
        raise ValueError("token-ID indices/probs outer length mismatch")
    if not indices:
        return [], "token_ids_empty"
    if tokenizer is None:
        raise ValueError("token-ID trace requires its generation tokenizer")

    generated_ids = row.get("generated_token_ids")
    if not isinstance(generated_ids, list) or len(generated_ids) != len(indices):
        raise ValueError(
            "token-ID trace requires generated_token_ids aligned with "
            "indices/probs"
        )
    if any(
        not isinstance(token_id, int) or token_id < 0
        for token_id in generated_ids
    ):
        raise ValueError("generated_token_ids must contain non-negative integers")

    normalized: list[list[dict[str, Any]]] = []
    for position, (step_ids, step_probabilities) in enumerate(
        zip(indices, probabilities)
    ):
        if not isinstance(step_ids, list) or not isinstance(
            step_probabilities, list
        ):
            raise ValueError(f"candidate step {position} is not a list")
        if not step_ids or len(step_ids) != len(step_probabilities):
            raise ValueError(f"candidate step {position} has invalid width")

        chosen_id = generated_ids[position]
        candidates: list[dict[str, Any]] = []
        for raw_token_id, raw_probability in zip(step_ids, step_probabilities):
            if not isinstance(raw_token_id, int) or raw_token_id < 0:
                raise ValueError(
                    f"candidate step {position} has an invalid token ID"
                )
            try:
                probability = float(raw_probability)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"candidate step {position} has an invalid probability"
                ) from error
            if not math.isfinite(probability) or probability < 0:
                raise ValueError(
                    f"candidate step {position} has an invalid probability"
                )
            candidates.append(
                {
                    "token_id": raw_token_id,
                    "decoded_token": decode_piece(tokenizer, raw_token_id),
                    "prob": probability,
                    "chosen": chosen_id == raw_token_id,
                }
            )
        normalized.append(candidates)
    return normalized, "token_ids"


def decoded_candidate_text_and_probability(candidate: Any) -> tuple[str, float]:
    if not isinstance(candidate, dict):
        return str(candidate or ""), 0.0
    text = str(
        candidate.get("decoded_token", candidate.get("token", "")) or ""
    )
    probability = candidate.get("prob")
    if probability is None and candidate.get("logprob") is not None:
        try:
            probability = math.exp(float(candidate["logprob"]))
        except (TypeError, ValueError, OverflowError):
            probability = 0.0
    try:
        value = float(probability or 0.0)
    except (TypeError, ValueError):
        value = 0.0
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"Invalid decoded candidate probability: {probability!r}")
    return text, value


def iter_soft_alternatives(
    candidates: list[Any],
    candidate_topk: int,
) -> Iterator[tuple[int, Any]]:
    """Yield non-chosen alternatives from ranks 2 through candidate_topk."""
    for rank, candidate in enumerate(candidates[:candidate_topk], start=1):
        if rank == 1:
            continue
        if isinstance(candidate, dict) and candidate.get("chosen"):
            continue
        yield rank, candidate


def is_decoded_special_candidate(text: Any) -> bool:
    return bool(DECODED_SPECIAL_TOKEN_RE.search(str(text or "")))


def is_wordish(text: Any) -> bool:
    return bool(WORD_RE.search(str(text or "")))


def is_punctuation_piece(piece: str) -> bool:
    stripped = piece.strip()
    return bool(stripped) and not is_wordish(stripped)


def starts_new_span(piece: str, current_pieces: list[str]) -> bool:
    if not current_pieces:
        return True
    if piece.startswith((" ", "\n", "\t")):
        return True
    if is_punctuation_piece(piece):
        return True
    if is_punctuation_piece(current_pieces[-1]):
        return True
    return False


def rank1_trajectory_pieces(
    row: dict[str, Any],
    tokenizer: Any | None,
) -> list[str]:
    """Return rank-1 pieces used to define the released boundary mask."""
    decoded_steps = row.get("decoded_candidates")
    if isinstance(decoded_steps, list):
        pieces: list[str] = []
        for candidates in decoded_steps:
            if not isinstance(candidates, list) or not candidates:
                continue
            candidate = candidates[0]
            if isinstance(candidate, dict):
                piece = candidate.get(
                    "decoded_token",
                    candidate.get("token", ""),
                )
            else:
                piece = candidate
            pieces.append(str(piece or ""))
        return pieces
    if tokenizer is None:
        return []
    return [
        decode_piece(tokenizer, int(step_ids[0]))
        for step_ids in row.get("indices", [])
        if step_ids
    ]


def build_rank1_boundary_mask(
    row: dict[str, Any],
    tokenizer: Any | None,
    tokeniser: Any,
    stemmer: Any,
    stopwords: set[str],
    min_len: int,
) -> tuple[set[int], dict[str, int]]:
    """Keep lexical single-token spans and starts of multi-token spans."""
    safe_positions: set[int] = set()
    stats: dict[str, int] = {}
    current_positions: list[int] = []
    current_pieces: list[str] = []

    def increment(key: str, value: int = 1) -> None:
        stats[key] = stats.get(key, 0) + value

    def flush() -> None:
        nonlocal current_positions, current_pieces
        if not current_positions:
            return
        text = "".join(current_pieces).strip()
        terms = terrier_terms(
            tokeniser,
            stemmer,
            stopwords,
            text,
            min_len=min_len,
        )
        if terms:
            increment("spans_with_terms")
            if len(current_positions) == 1:
                safe_positions.add(current_positions[0])
                increment("single_token_spans_kept")
            else:
                increment("multi_token_spans_masked")
                increment(
                    "positions_masked_in_multi_token_spans",
                    len(current_positions),
                )
                safe_positions.add(current_positions[0])
                increment("multi_token_starts_kept")
        current_positions = []
        current_pieces = []

    for position, piece in enumerate(rank1_trajectory_pieces(row, tokenizer)):
        if starts_new_span(piece, current_pieces):
            flush()
        current_positions.append(position)
        current_pieces.append(piece)
    flush()
    return safe_positions, stats


def row_allowed_by_sample_idx(
    row: dict[str, Any],
    sample_idx_max: int | None,
) -> bool:
    if sample_idx_max is None:
        return True
    sample_idx = row.get("sample_idx")
    if sample_idx is None:
        return True
    return int(sample_idx) <= sample_idx_max
