#!/usr/bin/env python3
"""Shared helpers for the Practical Cost Evaluation scripts.

The benchmark is intended to run on a GPU server. This module deliberately uses
only the Python standard library for orchestration and talks to vLLM through its
OpenAI-compatible HTTP API, so the scripts can be launched from a stable bash
entrypoint without installing the OpenAI SDK.
"""

from __future__ import annotations

import csv
import gzip
import json
import math
import os
import platform
import random
import re
import shutil
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


SAMPLE_SEED = 20260428


@dataclass(frozen=True)
class DatasetSpec:
    slug: str
    display_name: str
    expected_docs: int
    sample_docs: int
    split: str = "test"
    aliases: Tuple[str, ...] = ()


DATASET_SPECS: Tuple[DatasetSpec, ...] = (
    DatasetSpec("nfcorpus", "NFCorpus", 3633, 500, aliases=("nf-corpus", "nf_corpus")),
    DatasetSpec("scidocs", "SCIDOCS", 25657, 2508, aliases=("sci-docs", "sci_docs")),
    DatasetSpec("fiqa-2018", "FiQA-2018", 57638, 5637, aliases=("fiqa", "fiqa2018", "fiqa_2018")),
    DatasetSpec("arguana", "ArguAna", 8674, 849),
    DatasetSpec("scifact", "SciFact", 5183, 506),
)

SDE_PROMPT_TEMPLATES: Tuple[Dict[str, str], ...] = (
    {
        "id": "main_query",
        "name": "Main Search Query",
        "instruction": (
            "Given the following document snippet, predict the Google search query "
            "that a user would type to find this content.\n"
            "Output only the search query.\n"
            "Constraints:\n"
            "- Length: 3 to 12 words.\n"
            "- Format: Natural, concise query.\n"
            "- Do not use meta-terms like 'document', 'summary', or 'snippet'.\n"
            "- Focus on the main specific object or event described.\n\n"
            "Snippet: {doc_text}"
        ),
    },
    {
        "id": "problem_finding",
        "name": "Problem-Oriented Query",
        "instruction": (
            "Read the snippet and write a realistic search query that someone with the same "
            "problem, confusion, or need would use.\n"
            "Output only the query.\n"
            "Constraints:\n"
            "- 4 to 14 words.\n"
            "- Sound like a real web search.\n"
            "- Prioritize the user's likely intent over document phrasing.\n\n"
            "Snippet: {doc_text}"
        ),
    },
    {
        "id": "keyword_heavy",
        "name": "Keyword-Heavy Query",
        "instruction": (
            "Convert the document snippet into a strong keyword-style search query for retrieval.\n"
            "Output only the query.\n"
            "Constraints:\n"
            "- 3 to 12 words.\n"
            "- Emphasize rare, identifying, content-bearing words.\n"
            "- Avoid filler words unless necessary.\n\n"
            "Snippet: {doc_text}"
        ),
    },
    {
        "id": "entity_event",
        "name": "Entity or Event Query",
        "instruction": (
            "Write a search query focusing on the main entity, event, case, product, law, or concept "
            "that best identifies this snippet.\n"
            "Output only the query.\n"
            "Constraints:\n"
            "- 3 to 10 words.\n"
            "- Make it precise and disambiguating.\n\n"
            "Snippet: {doc_text}"
        ),
    },
    {
        "id": "how_why",
        "name": "How or Why Query",
        "instruction": (
            "Imagine a user wants to understand or resolve what this snippet is about.\n"
            "Write a plausible 'how', 'why', 'can', or explanatory search query.\n"
            "Output only the query.\n"
            "Constraints:\n"
            "- 4 to 14 words.\n"
            "- Keep it natural and specific.\n\n"
            "Snippet: {doc_text}"
        ),
    },
    {
        "id": "alt_paraphrase",
        "name": "Alternate Paraphrase Query",
        "instruction": (
            "Write an alternative search query that could retrieve this snippet using different wording "
            "from the original text.\n"
            "Output only the query.\n"
            "Constraints:\n"
            "- 3 to 12 words.\n"
            "- Prefer paraphrases, synonyms, or reformulations.\n"
            "- Still keep the query realistic.\n\n"
            "Snippet: {doc_text}"
        ),
    },
)

STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
    "do", "does", "did", "will", "would", "could", "should", "may", "might", "to", "of", "in",
    "on", "at", "by", "for", "with", "from", "as", "or", "and", "but", "not", "it", "its", "i",
    "we", "you", "he", "she", "they", "this", "that", "my", "your", "our", "their", "what",
    "which", "who", "if", "so", "up", "out", "can", "no", "more", "also", "about", "than", "then",
    "just", "into", "over", "after", "all", "when", "there", "me", "him", "her", "them",
}

BAD_DECODED_TERMS = {"s", "t", "re", "ve", "ll", "d", "m"}


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def reset_dir(path: Path) -> Path:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as fout:
        fout.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def compact_json_bytes(payload: Any) -> bytes:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8") + b"\n"


def write_json(path: Path, payload: Any) -> int:
    ensure_dir(path.parent)
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    path.write_text(text, encoding="utf-8")
    return path.stat().st_size


def write_compact_json(path: Path, payload: Any) -> int:
    ensure_dir(path.parent)
    path.write_bytes(compact_json_bytes(payload))
    return path.stat().st_size


def write_text(path: Path, text: str) -> int:
    ensure_dir(path.parent)
    path.write_text(text, encoding="utf-8")
    return path.stat().st_size


def write_gzip_json(path: Path, payload: Any) -> int:
    ensure_dir(path.parent)
    data = (json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
    with gzip.open(path, "wb") as fout:
        fout.write(data)
    return path.stat().st_size


def write_compact_gzip_json(path: Path, payload: Any, compresslevel: int = 9) -> int:
    ensure_dir(path.parent)
    path.write_bytes(gzip.compress(compact_json_bytes(payload), compresslevel=compresslevel, mtime=0))
    return path.stat().st_size


def utf8_len(text: str) -> int:
    return len(str(text).encode("utf-8"))


def text_len_words(text: str) -> int:
    return len(re.findall(r"\S+", text or ""))


def clean_id_for_path(doc_id: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", str(doc_id).strip())
    return cleaned[:180] or "blank_id"


def normalize_text(text: str, lowercase: bool = True, remove_stopwords: bool = True, min_token_len: int = 2) -> List[str]:
    if lowercase:
        text = text.lower()
    toks = re.findall(r"[a-z0-9]+", text)
    out: List[str] = []
    for tok in toks:
        if len(tok) < min_token_len:
            continue
        if remove_stopwords and tok in STOPWORDS:
            continue
        out.append(tok)
    return out


def decoded_token_to_terms(
    text: str,
    lowercase: bool = True,
    remove_stopwords: bool = True,
    min_token_len: int = 2,
    drop_numeric_terms: bool = True,
    extra_bad_terms: Optional[Iterable[str]] = None,
) -> List[str]:
    blocked = set(BAD_DECODED_TERMS)
    if extra_bad_terms:
        blocked.update(str(x).lower() for x in extra_bad_terms)
    terms = normalize_text(text, lowercase=lowercase, remove_stopwords=remove_stopwords, min_token_len=min_token_len)
    out: List[str] = []
    for term in terms:
        if drop_numeric_terms and term.isdigit():
            continue
        if term in blocked:
            continue
        if not re.search(r"[a-z]", term):
            continue
        out.append(term)
    return out


def unique_preserve_order(values: Iterable[str], limit: Optional[int] = None) -> List[str]:
    out: List[str] = []
    seen = set()
    for value in values:
        if not value or value in seen:
            continue
        out.append(value)
        seen.add(value)
        if limit is not None and len(out) >= limit:
            break
    return out


def coerce_query_item(item: Any) -> str:
    if isinstance(item, str):
        return item.strip()
    if isinstance(item, dict):
        for key in ("query", "search query", "search_query", "q", "text"):
            value = item.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        if len(item) == 1:
            key, value = next(iter(item.items()))
            if isinstance(value, str) and value.strip():
                return value.strip()
            if isinstance(key, str) and key.strip():
                return key.strip()
    return str(item).strip()


def parse_queries(text: str, expected: Optional[int] = None) -> List[str]:
    raw = str(text or "").strip()
    if not raw:
        return []

    fence_match = re.fullmatch(r"```(?:json|JSON)?\s*(.*?)\s*```", raw, flags=re.DOTALL)
    if fence_match:
        raw = fence_match.group(1).strip()

    if not raw.startswith("["):
        array_match = re.search(r"(\[[\s\S]*\])", raw)
        if array_match:
            raw = array_match.group(1).strip()

    try:
        payload = json.loads(raw)
        if isinstance(payload, list):
            values = [coerce_query_item(x) for x in payload]
            values = [value for value in values if value]
            return values[:expected] if expected else values
        if isinstance(payload, dict):
            for key in ("queries", "query", "items"):
                value = payload.get(key)
                if isinstance(value, list):
                    values = [coerce_query_item(x) for x in value]
                    values = [item for item in values if item]
                    return values[:expected] if expected else values
    except Exception:
        pass

    quoted_items = re.findall(r'"([^"\n]{3,300})"', raw)
    if expected and len(quoted_items) >= expected:
        return quoted_items[:expected]

    lines = []
    for line in raw.splitlines():
        line = line.strip()
        line = re.sub(r"^\s*[-*]\s+", "", line)
        line = re.sub(r"^\s*\d+[\).:-]\s+", "", line)
        line = line.strip().strip("\"'")
        if line:
            lines.append(line)
    if not lines and raw:
        lines = [raw.strip().strip("\"'")]
    return lines[:expected] if expected else lines


def read_sample_docs(sample_path: Path, limit_docs: int = 0) -> List[Dict[str, Any]]:
    rows = read_jsonl(sample_path)
    if limit_docs > 0:
        return rows[:limit_docs]
    return rows


def group_samples_by_dataset(sample_path: Path, limit_docs: int = 0) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in read_sample_docs(sample_path, limit_docs=limit_docs):
        grouped.setdefault(str(row["dataset"]), []).append(row)
    return grouped


def stage_log_row(
    dataset: str,
    method: str,
    stage: str,
    num_docs: int,
    start_time: float,
    status: str = "ok",
    error: Optional[str] = None,
) -> Dict[str, Any]:
    ended = time.time()
    total_time = ended - start_time
    return {
        "dataset": dataset,
        "method": method,
        "stage": stage,
        "num_docs": int(num_docs),
        "total_time_sec": float(total_time),
        "sec_per_doc": float(total_time / num_docs) if num_docs else 0.0,
        "status": status,
        "error": error,
        "started_at": start_time,
        "ended_at": ended,
    }


class VLLMClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        timeout_sec: float = 600.0,
        max_retries: int = 2,
        retry_sleep_sec: float = 3.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_sec = timeout_sec
        self.max_retries = max_retries
        self.retry_sleep_sec = retry_sleep_sec

    def chat(
        self,
        prompt: str,
        *,
        temperature: float,
        max_tokens: int,
        top_p: Optional[float] = None,
        logprobs: bool = False,
        top_logprobs: Optional[int] = None,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": float(temperature),
            "max_tokens": int(max_tokens),
        }
        if top_p is not None:
            payload["top_p"] = float(top_p)
        if logprobs:
            payload["logprobs"] = True
            payload["top_logprobs"] = int(top_logprobs or 5)

        data = json.dumps(payload).encode("utf-8")
        url = f"{self.base_url}/chat/completions"
        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            request = urllib.request.Request(
                url,
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_sec) as response:
                    return json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                body = exc.read().decode("utf-8", errors="replace")
                last_error = RuntimeError(f"HTTP {exc.code} from vLLM: {body[:1000]}")
            except Exception as exc:
                last_error = exc
            if attempt < self.max_retries:
                time.sleep(self.retry_sleep_sec)
        raise RuntimeError(str(last_error))


def completion_text(response: Dict[str, Any]) -> str:
    choices = response.get("choices") or []
    if not choices:
        return ""
    message = choices[0].get("message") or {}
    return str(message.get("content") or "").strip()


def usage_tokens(response: Dict[str, Any]) -> Tuple[int, int]:
    usage = response.get("usage") or {}
    return int(usage.get("prompt_tokens") or 0), int(usage.get("completion_tokens") or 0)


def extract_chat_top_logprobs(response: Dict[str, Any]) -> List[Dict[str, Any]]:
    choices = response.get("choices") or []
    if not choices:
        return []
    logprob_obj = choices[0].get("logprobs") or {}
    content = logprob_obj.get("content") or []
    out: List[Dict[str, Any]] = []
    for idx, step in enumerate(content):
        generated_token = step.get("token")
        generated_logprob = step.get("logprob")
        candidates = []
        for cand in step.get("top_logprobs") or []:
            token = cand.get("token")
            logprob = cand.get("logprob")
            if token is None or logprob is None:
                continue
            logprob_f = float(logprob)
            candidates.append(
                {
                    "token": token,
                    "logprob": logprob_f,
                    "prob": float(math.exp(logprob_f)) if logprob_f > -100 else 0.0,
                    "bytes": cand.get("bytes"),
                }
            )
        if not candidates and generated_token is not None and generated_logprob is not None:
            logprob_f = float(generated_logprob)
            candidates.append(
                {
                    "token": generated_token,
                    "logprob": logprob_f,
                    "prob": float(math.exp(logprob_f)) if logprob_f > -100 else 0.0,
                    "bytes": step.get("bytes"),
                }
            )
        out.append(
            {
                "step": idx,
                "token": generated_token,
                "logprob": generated_logprob,
                "candidates": candidates,
            }
        )
    return out


def maybe_token_id(tokenizer: Any, token_text: str) -> Optional[int]:
    if tokenizer is None:
        return None
    try:
        ids = tokenizer.encode(token_text, add_special_tokens=False)
        if len(ids) == 1:
            return int(ids[0])
    except Exception:
        return None
    return None


def simple_keyword_candidates(text: str, limit: int = 20) -> List[str]:
    counts: Dict[str, int] = {}
    for tok in normalize_text(text):
        counts[tok] = counts.get(tok, 0) + 1
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return [term for term, _ in ranked[:limit]]


def percentile(values: Sequence[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(x) for x in values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * pct
    lo = int(math.floor(rank))
    hi = int(math.ceil(rank))
    if lo == hi:
        return ordered[lo]
    weight = rank - lo
    return ordered[lo] * (1.0 - weight) + ordered[hi] * weight


def summarize(values: Sequence[float]) -> Dict[str, float]:
    if not values:
        return {"mean": 0.0, "p50": 0.0, "p95": 0.0, "std": 0.0}
    vals = [float(x) for x in values]
    return {
        "mean": float(statistics.mean(vals)),
        "p50": float(percentile(vals, 0.50)),
        "p95": float(percentile(vals, 0.95)),
        "std": float(statistics.pstdev(vals)) if len(vals) > 1 else 0.0,
    }


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def collect_environment() -> Dict[str, Any]:
    def run_capture(cmd: List[str]) -> Optional[str]:
        try:
            return subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT, timeout=20).strip()
        except Exception:
            return None

    python_packages: Dict[str, Optional[str]] = {}
    for package_name in ("vllm", "transformers", "torch", "pyterrier", "bertopic", "keybert"):
        try:
            module = __import__(package_name)
            python_packages[package_name] = str(getattr(module, "__version__", "unknown"))
        except Exception:
            python_packages[package_name] = None

    return {
        "recorded_at": now_iso(),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": sys.version,
        "python_executable": sys.executable,
        "cuda_version": run_capture(["nvcc", "--version"]),
        "nvidia_smi": run_capture(["nvidia-smi"]),
        "cpu": run_capture(["bash", "-lc", "lscpu | head -40"]),
        "ram": run_capture(["bash", "-lc", "free -h"]),
        "packages": python_packages,
        "env": {
            key: os.environ.get(key)
            for key in (
                "BEIR_DATA_ROOT",
                "BENCHMARK_OUTPUT_ROOT",
                "LLAMA31_8B_INSTRUCT",
                "VLLM_BASE_URL",
                "VLLM_MODEL",
                "VLLM_TENSOR_PARALLEL_SIZE",
                "VLLM_MAX_MODEL_LEN",
                "VLLM_GPU_MEMORY_UTILIZATION",
                "WARMUP_DOCS",
            )
        },
        "paths": {
            "python_on_path": shutil.which("python"),
            "python3_on_path": shutil.which("python3"),
        },
    }


def deterministic_sample(rows: List[Dict[str, Any]], quota: int, seed: int, dataset: str) -> List[Dict[str, Any]]:
    rng = random.Random(f"{seed}:{dataset}")
    if len(rows) < quota:
        raise ValueError(f"dataset={dataset} has only {len(rows)} docs, cannot sample quota={quota}")
    indices = list(range(len(rows)))
    rng.shuffle(indices)
    chosen = sorted(indices[:quota])
    return [rows[idx] for idx in chosen]
