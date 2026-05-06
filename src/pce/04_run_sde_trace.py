#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from common import (
    SDE_PROMPT_TEMPLATES,
    VLLMClient,
    append_jsonl,
    clean_id_for_path,
    completion_text,
    decoded_token_to_terms,
    ensure_dir,
    extract_chat_top_logprobs,
    group_samples_by_dataset,
    maybe_token_id,
    reset_dir,
    usage_tokens,
    write_compact_gzip_json,
    write_compact_json,
    write_text,
)


def load_tokenizer(model_path: str) -> Optional[Any]:
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
    except Exception as exc:
        print(f"[warn] tokenizer unavailable for token-id recovery: {exc}")
        return None


def build_trace_steps(logprob_steps: List[Dict[str, Any]], tokenizer: Optional[Any]) -> List[Dict[str, Any]]:
    trace_steps: List[Dict[str, Any]] = []
    for step in logprob_steps:
        candidates = []
        for cand in step.get("candidates") or []:
            token = str(cand.get("token") or "")
            terms = decoded_token_to_terms(token)
            candidates.append(
                {
                    "token_id": maybe_token_id(tokenizer, token),
                    "token": token,
                    "decoded_token": token,
                    "logprob": cand.get("logprob"),
                    "prob": cand.get("prob"),
                    "normalized_terms": terms,
                    "normalized_term": terms[0] if terms else None,
                    "bytes": cand.get("bytes"),
                }
            )
        generated_token = str(step.get("token") or "")
        trace_steps.append(
            {
                "step": int(step.get("step", len(trace_steps))),
                "generated_token": generated_token,
                "generated_token_id": maybe_token_id(tokenizer, generated_token),
                "generated_logprob": step.get("logprob"),
                "candidates": candidates,
            }
        )
    return trace_steps


def build_expansion_terms(
    traces: List[Dict[str, Any]],
    max_soft_terms_per_doc: int,
    term_weight_mode: str,
    repeat_score_scale: float,
    repeat_max_times: int,
) -> List[str]:
    term_scores: Dict[str, float] = defaultdict(float)
    for trace in traces:
        for step in trace.get("trace_steps", []):
            for cand in step.get("candidates", []):
                prob = float(cand.get("prob") or 0.0)
                for term in cand.get("normalized_terms") or []:
                    term_scores[str(term)] += prob
    ranked = sorted(term_scores.items(), key=lambda item: (-item[1], item[0]))
    out_terms: List[str] = []
    seen = set()
    for term, score in ranked[:max_soft_terms_per_doc]:
        if term in seen:
            continue
        repeats = 1
        if term_weight_mode == "repeat_by_score":
            repeats = max(1, min(repeat_max_times, int(math.ceil(score * repeat_score_scale))))
        out_terms.extend([term] * repeats)
        seen.add(term)
    return out_terms


def round_float(value: Any, digits: int = 6) -> Optional[float]:
    if value is None:
        return None
    try:
        return round(float(value), digits)
    except Exception:
        return None


def compact_trace_payload(raw_trace_payload: Dict[str, Any]) -> Dict[str, Any]:
    compact_traces: List[Dict[str, Any]] = []
    for trace in raw_trace_payload.get("traces", []):
        compact_steps = []
        for step in trace.get("trace_steps", []):
            compact_candidates = []
            for cand in step.get("candidates", []):
                compact_candidates.append(
                    [
                        cand.get("token_id"),
                        cand.get("decoded_token"),
                        round_float(cand.get("logprob")),
                        cand.get("normalized_terms") or [],
                    ]
                )
            compact_steps.append(
                [
                    int(step.get("step", len(compact_steps))),
                    step.get("generated_token_id"),
                    round_float(step.get("generated_logprob")),
                    compact_candidates,
                ]
            )
        compact_traces.append(
            {
                "pid": trace.get("prompt_id"),
                "q": trace.get("query_text"),
                "pt": trace.get("prompt_tokens"),
                "ct": trace.get("completion_tokens"),
                "s": compact_steps,
            }
        )
    return {
        "v": 1,
        "dataset": raw_trace_payload.get("dataset"),
        "doc_id": raw_trace_payload.get("doc_id"),
        "method": raw_trace_payload.get("method"),
        "model": raw_trace_payload.get("model"),
        "cfg": {
            "temperature": raw_trace_payload.get("temperature"),
            "top_p": raw_trace_payload.get("top_p"),
            "max_tokens": raw_trace_payload.get("max_tokens"),
            "top_logprobs": raw_trace_payload.get("top_logprobs"),
        },
        "traces": compact_traces,
    }


def run_one_doc(
    client: VLLMClient,
    tokenizer: Optional[Any],
    doc: Dict[str, Any],
    temperature: float,
    top_p: float,
    max_tokens: int,
    top_logprobs: int,
    max_soft_terms_per_doc: int,
    term_weight_mode: str,
    repeat_score_scale: float,
    repeat_max_times: int,
) -> Dict[str, Any]:
    traces: List[Dict[str, Any]] = []
    prompt_tokens = 0
    output_tokens = 0
    llm_time_sec = 0.0
    filter_time_sec = 0.0
    start = time.time()
    for template in SDE_PROMPT_TEMPLATES:
        prompt = template["instruction"].format(doc_text=str(doc["text"])[:2000])
        call_start = time.time()
        response = client.chat(
            prompt,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            logprobs=True,
            top_logprobs=top_logprobs,
        )
        llm_time_sec += time.time() - call_start
        p_tokens, c_tokens = usage_tokens(response)
        prompt_tokens += p_tokens
        output_tokens += c_tokens
        filter_start = time.time()
        raw_logprob_steps = extract_chat_top_logprobs(response)
        trace_steps = build_trace_steps(raw_logprob_steps, tokenizer)
        filter_time_sec += time.time() - filter_start
        traces.append(
            {
                "prompt_id": template["id"],
                "prompt_name": template["name"],
                "query_text": completion_text(response).strip().strip("\"'"),
                "prompt_tokens": p_tokens,
                "completion_tokens": c_tokens,
                "trace_steps": trace_steps,
            }
        )
    filter_start = time.time()
    expansion_terms = build_expansion_terms(
        traces,
        max_soft_terms_per_doc=max_soft_terms_per_doc,
        term_weight_mode=term_weight_mode,
        repeat_score_scale=repeat_score_scale,
        repeat_max_times=repeat_max_times,
    )
    filter_time_sec += time.time() - filter_start
    return {
        "traces": traces,
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "llm_time_sec": llm_time_sec,
        "filter_time_sec": filter_time_sec,
        "wall_time_sec": time.time() - start,
        "expansion_terms": expansion_terms,
    }


def warmup(
    client: VLLMClient,
    tokenizer: Optional[Any],
    docs: List[Dict[str, Any]],
    temperature: float,
    top_p: float,
    max_tokens: int,
    top_logprobs: int,
    max_soft_terms_per_doc: int,
    term_weight_mode: str,
    repeat_score_scale: float,
    repeat_max_times: int,
) -> None:
    for doc in docs:
        run_one_doc(
            client,
            tokenizer,
            doc,
            temperature,
            top_p,
            max_tokens,
            top_logprobs,
            max_soft_terms_per_doc,
            term_weight_mode,
            repeat_score_scale,
            repeat_max_times,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SDE six-prompt generation with top-k trace persistence.")
    parser.add_argument("--sample_path", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--vllm_base_url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer_path", default="")
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--max_tokens", type=int, default=32)
    parser.add_argument("--top_logprobs", type=int, default=5)
    parser.add_argument("--max_soft_terms_per_doc", type=int, default=256)
    parser.add_argument("--term_weight_mode", choices=["unique", "repeat_by_score"], default="repeat_by_score")
    parser.add_argument("--repeat_score_scale", type=float, default=3.0)
    parser.add_argument("--repeat_max_times", type=int, default=3)
    parser.add_argument("--warmup_docs", type=int, default=20)
    parser.add_argument("--limit_docs", type=int, default=0)
    args = parser.parse_args()

    sample_path = Path(args.sample_path)
    output_root = Path(args.output_root)
    raw_log = output_root / "raw" / "sde_trace_doc_logs.jsonl"
    stage_log = output_root / "raw" / "sde_trace_stage_logs.jsonl"
    raw_log.parent.mkdir(parents=True, exist_ok=True)
    raw_log.write_text("", encoding="utf-8")
    stage_log.write_text("", encoding="utf-8")

    client = VLLMClient(args.vllm_base_url, args.model)
    tokenizer = load_tokenizer(args.tokenizer_path or args.model)
    grouped = group_samples_by_dataset(sample_path, limit_docs=args.limit_docs)
    all_docs = [doc for docs in grouped.values() for doc in docs]
    if args.warmup_docs > 0:
        print(f"[warmup] sde_trace docs={min(args.warmup_docs, len(all_docs))}")
        warmup(
            client,
            tokenizer,
            all_docs[: args.warmup_docs],
            args.temperature,
            args.top_p,
            args.max_tokens,
            args.top_logprobs,
            args.max_soft_terms_per_doc,
            args.term_weight_mode,
            args.repeat_score_scale,
            args.repeat_max_times,
        )

    decoded_root = reset_dir(output_root / "artifacts" / "sde" / "decoded_queries")
    traces_raw_root = reset_dir(output_root / "artifacts" / "sde" / "traces_raw")
    traces_gzip_root = reset_dir(output_root / "artifacts" / "sde" / "traces_gzip")
    expansion_root = reset_dir(output_root / "artifacts" / "sde" / "expansion_entries")

    processed = 0
    method_start = time.time()
    persist_decoded_time = 0.0
    persist_raw_time = 0.0
    persist_gzip_time = 0.0
    persist_entry_time = 0.0
    llm_generation_time = 0.0
    filtering_time = 0.0

    for dataset, docs in grouped.items():
        for doc in docs:
            doc_id = str(doc["doc_id"])
            try:
                result = run_one_doc(
                    client,
                    tokenizer,
                    doc,
                    args.temperature,
                    args.top_p,
                    args.max_tokens,
                    args.top_logprobs,
                    args.max_soft_terms_per_doc,
                    args.term_weight_mode,
                    args.repeat_score_scale,
                    args.repeat_max_times,
                )
                llm_generation_time += float(result["llm_time_sec"])
                filtering_time += float(result["filter_time_sec"])
                doc_stem = clean_id_for_path(doc_id)
                decoded_payload = {
                    "dataset": dataset,
                    "doc_id": doc_id,
                    "queries": [
                        {
                            "prompt_id": trace["prompt_id"],
                            "prompt_name": trace["prompt_name"],
                            "query_text": trace["query_text"],
                        }
                        for trace in result["traces"]
                    ],
                }
                raw_trace_payload = {
                    "dataset": dataset,
                    "doc_id": doc_id,
                    "method": "sde_trace",
                    "model": args.model,
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "max_tokens": args.max_tokens,
                    "top_logprobs": args.top_logprobs,
                    "traces": result["traces"],
                }
                compact_trace = compact_trace_payload(raw_trace_payload)

                t0 = time.time()
                decoded_bytes = write_compact_json(decoded_root / dataset / f"{doc_stem}.json", decoded_payload)
                persist_decoded_time += time.time() - t0
                t0 = time.time()
                trace_raw_bytes = write_compact_json(traces_raw_root / dataset / f"{doc_stem}.json", compact_trace)
                persist_raw_time += time.time() - t0
                t0 = time.time()
                trace_gzip_bytes = write_compact_gzip_json(
                    traces_gzip_root / dataset / f"{doc_stem}.json.gz",
                    compact_trace,
                )
                persist_gzip_time += time.time() - t0
                t0 = time.time()
                expansion_entry_bytes = write_text(
                    expansion_root / dataset / f"{doc_stem}.txt",
                    " ".join(result["expansion_terms"]) + "\n",
                )
                persist_entry_time += time.time() - t0

                status = "ok"
                error = None
                if len(result["traces"]) != 6:
                    status = "partial"
                    error = f"expected 6 traces, got {len(result['traces'])}"
                elif trace_gzip_bytes <= 0 or trace_raw_bytes <= 0 or result["output_tokens"] <= 0:
                    status = "partial"
                    error = "nonzero trace/output acceptance check failed"
                row = {
                    "dataset": dataset,
                    "doc_id": doc_id,
                    "method": "sde_trace",
                    "num_generations": len(result["traces"]),
                    "llm_calls": 6,
                    "input_chars": len(str(doc.get("text", ""))),
                    "input_tokens": result["prompt_tokens"],
                    "output_tokens": result["output_tokens"],
                    "llm_time_sec": result["llm_time_sec"],
                    "filter_time_sec": result["filter_time_sec"],
                    "wall_time_sec": result["wall_time_sec"],
                    "decoded_text_bytes": decoded_bytes,
                    "trace_raw_bytes": trace_raw_bytes,
                    "trace_gzip_bytes": trace_gzip_bytes,
                    "expansion_entry_bytes": expansion_entry_bytes,
                    "status": status,
                    "error": error,
                }
            except Exception as exc:
                row = {
                    "dataset": dataset,
                    "doc_id": doc_id,
                    "method": "sde_trace",
                    "num_generations": 0,
                    "llm_calls": 6,
                    "input_chars": len(str(doc.get("text", ""))),
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "llm_time_sec": 0.0,
                    "filter_time_sec": 0.0,
                    "wall_time_sec": 0.0,
                    "decoded_text_bytes": 0,
                    "trace_raw_bytes": 0,
                    "trace_gzip_bytes": 0,
                    "expansion_entry_bytes": 0,
                    "status": "error",
                    "error": str(exc),
                }
            append_jsonl(raw_log, row)
            processed += 1
            if processed % 100 == 0:
                print(f"[progress] sde_trace processed={processed}")

    total_time = time.time() - method_start
    num_docs = max(processed, 1)
    for stage, value in (
        ("sde_llm_generation", llm_generation_time),
        ("sde_trace_filtering", filtering_time),
    ):
        append_jsonl(
            stage_log,
            {
                "dataset": "ALL",
                "method": "sde_trace",
                "stage": stage,
                "num_docs": processed,
                "total_time_sec": value,
                "sec_per_doc": value / num_docs,
                "status": "ok",
                "error": None,
            },
        )
    for stage, value in (
        ("persist_decoded_queries", persist_decoded_time),
        ("persist_raw_trace", persist_raw_time),
        ("persist_compressed_trace", persist_gzip_time),
        ("persist_expansion_entries", persist_entry_time),
    ):
        append_jsonl(
            stage_log,
            {
                "dataset": "ALL",
                "method": "sde_trace",
                "stage": stage,
                "num_docs": processed,
                "total_time_sec": value,
                "sec_per_doc": value / num_docs,
                "status": "ok",
                "error": None,
            },
        )

    print(f"[done] sde_trace doc_log={raw_log} processed={processed} total_time_sec={total_time:.3f}")


if __name__ == "__main__":
    main()
