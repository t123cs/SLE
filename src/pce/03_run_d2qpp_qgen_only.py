#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List

from common import (
    VLLMClient,
    append_jsonl,
    clean_id_for_path,
    completion_text,
    ensure_dir,
    group_samples_by_dataset,
    parse_queries,
    reset_dir,
    simple_keyword_candidates,
    usage_tokens,
    utf8_len,
    write_compact_json,
    write_text,
)


def load_full_metadata(output_root: Path) -> Dict[str, Dict[str, Dict[str, Any]]]:
    meta_by_dataset: Dict[str, Dict[str, Dict[str, Any]]] = {}
    meta_root = output_root / "artifacts" / "d2qpp" / "topics_keywords"
    for path in meta_root.glob("*_doc_metadata.jsonl"):
        dataset = path.name[: -len("_doc_metadata.jsonl")]
        meta_by_dataset.setdefault(dataset, {})
        with path.open("r", encoding="utf-8") as fin:
            for line in fin:
                if not line.strip():
                    continue
                row = json.loads(line)
                meta_by_dataset[dataset][str(row["doc_id"])] = row
    return meta_by_dataset


def qgen_prompt(doc: Dict[str, Any], metadata: Dict[str, Any], call_idx: int) -> str:
    keywords = metadata.get("selected_keywords") or metadata.get("keywords") or []
    topic_name = metadata.get("topic_name") or metadata.get("topic") or "general"
    if isinstance(keywords, str):
        keywords_text = keywords
    else:
        keywords_text = ", ".join(str(x) for x in keywords[:12])
    return (
        "You are generating diverse web search queries for document expansion.\n"
        "Use the topic and keywords to cover different information needs.\n"
        "Crucial Output Instruction:\n"
        "Return exactly one JSON array of exactly 3 strings.\n"
        "Each string must be one concise search query.\n"
        "Do not include explanations, labels, markdown, code fences, bullets, numbering, or text outside the JSON array.\n\n"
        f"Topic: {topic_name}\n"
        f"Keywords: {keywords_text}\n"
        f"Call index: {call_idx + 1} of 10\n"
        f"Document:\n{str(doc['text'])[:2500]}"
    )


def fallback_metadata(doc: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "topic_name": "document topic",
        "selected_keywords": simple_keyword_candidates(str(doc.get("text", "")), limit=10),
        "metadata_source": "simple_fallback",
    }


def generate_30_queries(
    client: VLLMClient,
    doc: Dict[str, Any],
    metadata: Dict[str, Any],
    temperature: float,
    max_tokens: int,
) -> Dict[str, Any]:
    queries: List[str] = []
    prompt_tokens = 0
    output_tokens = 0
    calls = []
    start = time.time()
    for call_idx in range(10):
        prompt = qgen_prompt(doc, metadata, call_idx)
        response = client.chat(prompt, temperature=temperature, max_tokens=max_tokens)
        p_tokens, c_tokens = usage_tokens(response)
        prompt_tokens += p_tokens
        output_tokens += c_tokens
        text = completion_text(response)
        parsed = parse_queries(text, expected=3)
        queries.extend(parsed[:3])
        calls.append(
            {
                "call_idx": call_idx,
                "prompt_tokens": p_tokens,
                "completion_tokens": c_tokens,
                "raw_text": text,
                "parsed_queries": parsed[:3],
            }
        )
    return {
        "queries": queries[:30],
        "calls": calls,
        "wall_time_sec": time.time() - start,
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
    }


def warmup(client: VLLMClient, docs: List[Dict[str, Any]], temperature: float, max_tokens: int) -> None:
    for idx, doc in enumerate(docs):
        metadata = fallback_metadata(doc)
        prompt = qgen_prompt(doc, metadata, idx % 10)
        client.chat(prompt, temperature=temperature, max_tokens=max_tokens)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Doc2Query++ final generation only benchmark.")
    parser.add_argument("--sample_path", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--vllm_base_url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--max_tokens", type=int, default=192)
    parser.add_argument("--warmup_docs", type=int, default=20)
    parser.add_argument("--allow_simple_fallback", type=int, default=1)
    parser.add_argument("--limit_docs", type=int, default=0)
    args = parser.parse_args()

    sample_path = Path(args.sample_path)
    output_root = Path(args.output_root)
    raw_log = output_root / "raw" / "d2qpp_qgen_only_doc_logs.jsonl"
    raw_log.parent.mkdir(parents=True, exist_ok=True)
    raw_log.write_text("", encoding="utf-8")

    client = VLLMClient(args.vllm_base_url, args.model)
    grouped = group_samples_by_dataset(sample_path, limit_docs=args.limit_docs)
    all_docs = [doc for docs in grouped.values() for doc in docs]
    if args.warmup_docs > 0:
        print(f"[warmup] d2qpp_qgen_only docs={min(args.warmup_docs, len(all_docs))}")
        warmup(client, all_docs[: args.warmup_docs], args.temperature, args.max_tokens)

    metadata_cache = load_full_metadata(output_root)
    generated_root = reset_dir(output_root / "artifacts" / "d2qpp" / "qgen_only" / "generated_queries")
    expanded_root = reset_dir(output_root / "artifacts" / "d2qpp" / "qgen_only" / "expanded_docs")

    processed = 0
    for dataset, docs in grouped.items():
        for doc in docs:
            doc_id = str(doc["doc_id"])
            metadata = metadata_cache.get(dataset, {}).get(doc_id)
            if metadata is None:
                if not args.allow_simple_fallback:
                    row = {
                        "dataset": dataset,
                        "doc_id": doc_id,
                        "method": "d2qpp_qgen_only",
                        "num_generations": 0,
                        "llm_calls": 0,
                        "input_chars": len(str(doc.get("text", ""))),
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "wall_time_sec": 0.0,
                        "decoded_text_bytes": 0,
                        "trace_raw_bytes": 0,
                        "trace_gzip_bytes": 0,
                        "expansion_entry_bytes": 0,
                        "status": "error",
                        "error": "missing D2Q++ full topics/keywords metadata",
                    }
                    append_jsonl(raw_log, row)
                    continue
                metadata = fallback_metadata(doc)

            try:
                result = generate_30_queries(client, doc, metadata, args.temperature, args.max_tokens)
                decoded_text = "\n".join(result["queries"]) + "\n"
                doc_stem = clean_id_for_path(doc_id)
                decoded_path = generated_root / dataset / f"{doc_stem}.json"
                expanded_path = expanded_root / dataset / f"{doc_stem}.txt"
                query_text_bytes = sum(utf8_len(query + "\n") for query in result["queries"])
                decoded_artifact_bytes = write_compact_json(
                    decoded_path,
                    {
                        "dataset": dataset,
                        "doc_id": doc_id,
                        "method": "d2qpp_qgen_only",
                        "metadata_source": metadata.get("metadata_source", "d2qpp_full_cache"),
                        "queries": result["queries"],
                        "calls": result["calls"],
                    },
                )
                expansion_entry_bytes = write_text(expanded_path, decoded_text)
                status = "ok" if len(result["queries"]) == 30 else "partial"
                error = None if status == "ok" else f"expected 30 queries, got {len(result['queries'])}"
                row = {
                    "dataset": dataset,
                    "doc_id": doc_id,
                    "method": "d2qpp_qgen_only",
                    "num_generations": len(result["queries"]),
                    "llm_calls": 10,
                    "input_chars": len(str(doc.get("text", ""))),
                    "input_tokens": result["prompt_tokens"],
                    "output_tokens": result["output_tokens"],
                    "wall_time_sec": result["wall_time_sec"],
                    "decoded_text_bytes": query_text_bytes,
                    "trace_raw_bytes": 0,
                    "trace_gzip_bytes": 0,
                    "expansion_entry_bytes": expansion_entry_bytes,
                    "debug_artifact_bytes": decoded_artifact_bytes,
                    "status": status,
                    "error": error,
                }
            except Exception as exc:
                row = {
                    "dataset": dataset,
                    "doc_id": doc_id,
                    "method": "d2qpp_qgen_only",
                    "num_generations": 0,
                    "llm_calls": 10,
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
                print(f"[progress] d2qpp_qgen_only processed={processed}")

    print(f"[done] d2qpp_qgen_only doc_log={raw_log} processed={processed}")


if __name__ == "__main__":
    main()
