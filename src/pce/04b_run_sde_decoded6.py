#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Dict, List

from common import (
    SDE_PROMPT_TEMPLATES,
    VLLMClient,
    append_jsonl,
    clean_id_for_path,
    completion_text,
    ensure_dir,
    group_samples_by_dataset,
    reset_dir,
    usage_tokens,
    write_compact_json,
)


def run_one_doc(client: VLLMClient, doc: Dict[str, Any], temperature: float, top_p: float, max_tokens: int) -> Dict[str, Any]:
    queries: List[Dict[str, Any]] = []
    prompt_tokens = 0
    output_tokens = 0
    start = time.time()
    for template in SDE_PROMPT_TEMPLATES:
        prompt = template["instruction"].format(doc_text=str(doc["text"])[:2000])
        response = client.chat(prompt, temperature=temperature, top_p=top_p, max_tokens=max_tokens)
        p_tokens, c_tokens = usage_tokens(response)
        prompt_tokens += p_tokens
        output_tokens += c_tokens
        queries.append(
            {
                "prompt_id": template["id"],
                "prompt_name": template["name"],
                "query_text": completion_text(response).strip().strip("\"'"),
            }
        )
    return {
        "queries": queries,
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "wall_time_sec": time.time() - start,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run optional SDE decoded-only six-query control.")
    parser.add_argument("--sample_path", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--vllm_base_url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--max_tokens", type=int, default=32)
    parser.add_argument("--warmup_docs", type=int, default=20)
    parser.add_argument("--limit_docs", type=int, default=0)
    args = parser.parse_args()

    output_root = Path(args.output_root)
    raw_log = output_root / "raw" / "sde_decoded6_doc_logs.jsonl"
    raw_log.parent.mkdir(parents=True, exist_ok=True)
    raw_log.write_text("", encoding="utf-8")
    decoded_root = reset_dir(output_root / "artifacts" / "sde" / "decoded_only6")
    grouped = group_samples_by_dataset(Path(args.sample_path), limit_docs=args.limit_docs)
    client = VLLMClient(args.vllm_base_url, args.model)

    all_docs = [doc for docs in grouped.values() for doc in docs]
    if args.warmup_docs > 0:
        for doc in all_docs[: args.warmup_docs]:
            run_one_doc(client, doc, args.temperature, args.top_p, args.max_tokens)

    processed = 0
    for dataset, docs in grouped.items():
        for doc in docs:
            doc_id = str(doc["doc_id"])
            try:
                result = run_one_doc(client, doc, args.temperature, args.top_p, args.max_tokens)
                decoded_bytes = write_compact_json(
                    decoded_root / dataset / f"{clean_id_for_path(doc_id)}.json",
                    {"dataset": dataset, "doc_id": doc_id, "queries": result["queries"]},
                )
                row = {
                    "dataset": dataset,
                    "doc_id": doc_id,
                    "method": "sde_decoded6",
                    "num_generations": len(result["queries"]),
                    "llm_calls": 6,
                    "input_chars": len(str(doc.get("text", ""))),
                    "input_tokens": result["prompt_tokens"],
                    "output_tokens": result["output_tokens"],
                    "wall_time_sec": result["wall_time_sec"],
                    "decoded_text_bytes": decoded_bytes,
                    "trace_raw_bytes": 0,
                    "trace_gzip_bytes": 0,
                    "expansion_entry_bytes": decoded_bytes,
                    "status": "ok" if len(result["queries"]) == 6 else "partial",
                    "error": None if len(result["queries"]) == 6 else f"expected 6 queries, got {len(result['queries'])}",
                }
            except Exception as exc:
                row = {
                    "dataset": dataset,
                    "doc_id": doc_id,
                    "method": "sde_decoded6",
                    "num_generations": 0,
                    "llm_calls": 6,
                    "input_chars": len(str(doc.get("text", ""))),
                    "input_tokens": 0,
                    "output_tokens": 0,
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
                print(f"[progress] sde_decoded6 processed={processed}")
    print(f"[done] sde_decoded6 doc_log={raw_log} processed={processed}")


if __name__ == "__main__":
    main()
