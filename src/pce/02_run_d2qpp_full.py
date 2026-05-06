#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
    stage_log_row,
    unique_preserve_order,
    usage_tokens,
    write_text,
)


DEFAULT_EMBEDDING_MODEL_PATH = "/path/to/workspace/models/cache/all-MiniLM-L6-v2"


def split_sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+", str(text or "").strip())
    return [part.strip() for part in parts if part.strip()]


def load_fewshot_examples(path: str) -> Dict[str, List[Dict[str, Any]]]:
    if not path:
        return {"topic_name": [], "multi_query": []}
    with Path(path).open("r", encoding="utf-8") as fin:
        payload = json.load(fin)
    if not isinstance(payload, dict):
        raise ValueError("few-shot examples file must contain a JSON object")
    return {
        "topic_name": list(payload.get("topic_name", [])),
        "multi_query": list(payload.get("multi_query", [])),
    }


def topic_name_prompt(
    topic_id: int,
    keywords: List[str],
    representative_texts: List[str],
    examples: List[Dict[str, Any]],
) -> str:
    lines = [
        "You will extract a short topic label from given documents and keywords.",
    ]
    if examples:
        lines.append("Here are four examples of topics you created before:")
        lines.append("")
        for idx, example in enumerate(examples[:4], start=1):
            lines.append(f"Example {idx}")
            lines.append("Sample texts from this topic:")
            for text in example.get("sample_texts", [])[:3]:
                lines.append(f"* {str(text)[:500]}")
            lines.append(f"Keywords: {', '.join(str(x) for x in example.get('keywords', [])[:20])}")
            lines.append(f"Topic: {example.get('topic', '')}")
            lines.append("")
    lines.extend(
        [
            "Your Task",
            "Sample texts from this topic:",
        ]
    )
    for text in representative_texts[:3]:
        lines.append(f"* {text[:500]}")
    lines.extend(
        [
            f"Keywords: {', '.join(keywords[:20])}",
            "Crucial Output Instruction:",
            "You MUST generate a single line as your response.",
            "This line MUST start EXACTLY with 'topic: ' (including the space after the colon).",
            "Following 'topic: ', provide ONLY the concise topic label.",
            "Do NOT add any other text, explanations, numbering, markdown, or any content before or after this single line.",
            "Topic: ",
        ]
    )
    return "\n".join(lines)


def keyword_select_prompt(doc: Dict[str, Any], topic_name: str, keywords: List[str], final_keyword_num: int) -> str:
    return (
        "You will receive a document along with a set of candidate keywords. "
        "Your task is to select the keywords that best align with the core theme of the document. "
        "Exclude keywords that are too broad or less relevant. "
        f"You may list up to {final_keyword_num} keywords, using only the keywords in the candidate keyword set:\n"
        f"Document: {str(doc['text'])[:3000]}\n"
        f"Candidate keyword set: {', '.join(keywords[:80])}\n"
        "Final Keywords: "
    )


def parse_selected_keywords(text: str, candidate_keywords: List[str], final_keyword_num: int) -> List[str]:
    values = parse_queries(text, expected=final_keyword_num)
    if len(values) == 1 and ("," in values[0] or ";" in values[0]):
        values = [part.strip() for part in re.split(r"[,;]", values[0]) if part.strip()]
    canonical_by_lower = {str(value).lower(): str(value) for value in candidate_keywords}
    selected: List[str] = []
    for value in values:
        canonical = canonical_by_lower.get(str(value).strip().lower())
        if canonical:
            selected.append(canonical)
    return unique_preserve_order(selected, limit=final_keyword_num)


def multi_query_prompt(
    doc: Dict[str, Any],
    topic_name: str,
    selected_keywords: List[str],
    call_idx: int,
    examples: List[Dict[str, Any]],
) -> str:
    lines = [
        "You are an expert assistant in crafting search queries for a given passage that cover specified topics and make use of given keywords.",
    ]
    if examples:
        lines.append("The following are some examples:")
        lines.append("")
        for idx, example in enumerate(examples[:6], start=1):
            lines.append(f"Example {idx}")
            lines.append(f"Article: {str(example.get('article', ''))[:900]}")
            lines.append(f"Topics: {', '.join(str(x) for x in example.get('topics', [])[:10])}")
            lines.append(f"Keywords: {', '.join(str(x) for x in example.get('keywords', [])[:20])}")
            lines.append("Generated Queries:")
            for query in example.get("queries", [])[:3]:
                lines.append(f"* {query}")
            lines.append("")
    lines.extend(
        [
            "Your Task:",
            "Now generate 3 relevant queries for this passage that collectively cover specified topics by using given keywords:",
            f"Passage: <document {str(doc['text'])[:3000]}>",
            f"Topics: <topics {topic_name}>",
            f"Keywords: <keywords {', '.join(selected_keywords[:10])}>",
            f"Generation batch: {call_idx + 1} of 10",
            "Crucial Output Instruction:",
            "Return exactly one JSON array of exactly 3 strings.",
            "Each string must be one concise search query.",
            "Do not include explanations, labels, markdown, bullets, numbering, or text outside the JSON array.",
            "Queries:",
        ]
    )
    return "\n".join(lines)


def require_or_fallback(import_name: str, fallback_allowed: bool) -> Any:
    try:
        return __import__(import_name)
    except Exception as exc:
        if fallback_allowed:
            print(f"[warn] dependency {import_name} unavailable, using fallback: {exc}")
            return None
        raise RuntimeError(f"Required dependency missing for D2Q++ full: {import_name}: {exc}")


def load_embedding_model(embedding_model_path: str, fallback_allowed: bool) -> Optional[Any]:
    if not embedding_model_path:
        if fallback_allowed:
            print("[warn] no embedding_model_path configured, using fallback keyword/topic logic")
            return None
        raise RuntimeError("D2Q++ full requires a local embedding_model_path to avoid implicit model downloads")
    try:
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(embedding_model_path, local_files_only=True)
    except Exception as exc:
        if fallback_allowed:
            print(f"[warn] embedding model unavailable, using fallback keyword/topic logic: {exc}")
            return None
        raise RuntimeError(f"Could not load embedding model at {embedding_model_path}: {exc}")


def fallback_topics(
    docs: List[Dict[str, Any]],
    topic_keyword_num: int,
    representative_text_num: int,
) -> Tuple[List[int], Dict[int, Dict[str, Any]]]:
    topics = [idx % 20 for idx, _ in enumerate(docs)]
    topic_info: Dict[int, Dict[str, Any]] = {}
    for topic_id in sorted(set(topics)):
        topic_docs = [docs[idx] for idx, topic in enumerate(topics) if topic == topic_id]
        keywords = simple_keyword_candidates(" ".join(str(doc["text"]) for doc in topic_docs), limit=topic_keyword_num)
        topic_info[topic_id] = {
            "topic_id": topic_id,
            "keywords": keywords,
            "representative_texts": [str(doc["text"])[:500] for doc in topic_docs[:representative_text_num]],
            "source": "fallback_round_robin",
        }
    return topics, topic_info


def run_bertopic(
    docs: List[Dict[str, Any]],
    fallback_allowed: bool,
    embedding_model: Optional[Any],
    topic_keyword_num: int,
    representative_text_num: int,
    min_topic_size: int,
) -> Tuple[List[int], Dict[int, Dict[str, Any]]]:
    module = require_or_fallback("bertopic", fallback_allowed)
    if module is None or embedding_model is None:
        return fallback_topics(docs, topic_keyword_num, representative_text_num)
    if len(docs) < 4:
        if not fallback_allowed:
            print("[warn] too few docs for BERTopic; using deterministic small-sample fallback")
        return fallback_topics(docs, topic_keyword_num, representative_text_num)

    from bertopic import BERTopic
    from hdbscan import HDBSCAN
    from umap import UMAP

    texts = [str(doc["text"]) for doc in docs]
    n_neighbors = max(2, min(15, len(docs) - 1))
    n_components = max(2, min(5, len(docs) - 2))
    effective_min_topic_size = max(2, min(min_topic_size, max(2, len(docs) // 5)))
    umap_model = UMAP(
        n_neighbors=n_neighbors,
        n_components=n_components,
        min_dist=0.0,
        metric="cosine",
        random_state=42,
    )
    hdbscan_model = HDBSCAN(
        min_cluster_size=effective_min_topic_size,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=False,
    )
    model = BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        top_n_words=topic_keyword_num,
        calculate_probabilities=False,
        verbose=False,
    )
    topics, _ = model.fit_transform(texts)
    topic_info: Dict[int, Dict[str, Any]] = {}
    for topic_id in sorted(set(int(x) for x in topics)):
        words = model.get_topic(topic_id) or []
        keywords = [str(word) for word, _ in words[:topic_keyword_num]]
        topic_docs = [docs[idx] for idx, topic in enumerate(topics) if int(topic) == topic_id]
        topic_info[topic_id] = {
            "topic_id": topic_id,
            "keywords": keywords,
            "representative_texts": [str(doc["text"])[:500] for doc in topic_docs[:representative_text_num]],
            "source": "bertopic",
        }
    return [int(x) for x in topics], topic_info


def run_keybert(
    docs: List[Dict[str, Any]],
    fallback_allowed: bool,
    diversity: float,
    embedding_model: Optional[Any],
    top_n: int,
) -> Dict[str, List[str]]:
    module = require_or_fallback("keybert", fallback_allowed)
    if module is None or embedding_model is None:
        return {str(doc["doc_id"]): simple_keyword_candidates(str(doc["text"]), limit=top_n) for doc in docs}

    from keybert import KeyBERT

    model = KeyBERT(model=embedding_model)
    out: Dict[str, List[str]] = {}
    for doc in docs:
        keywords = model.extract_keywords(
            str(doc["text"]),
            keyphrase_ngram_range=(1, 3),
            stop_words="english",
            top_n=top_n,
            use_mmr=True,
            diversity=diversity,
        )
        out[str(doc["doc_id"])] = [str(term) for term, _ in keywords]
    return out


def llm_topic_names(
    client: VLLMClient,
    topic_info: Dict[int, Dict[str, Any]],
    temperature: float,
    max_tokens: int,
    examples: List[Dict[str, Any]],
) -> Dict[int, str]:
    names: Dict[int, str] = {}
    for topic_id, info in topic_info.items():
        prompt = topic_name_prompt(topic_id, info.get("keywords", []), info.get("representative_texts", []), examples)
        response = client.chat(prompt, temperature=temperature, max_tokens=max_tokens)
        raw = completion_text(response).strip().strip("\"'")
        if raw.lower().startswith("topic:"):
            raw = raw.split(":", 1)[1].strip()
        names[topic_id] = raw or f"topic_{topic_id}"
    return names


def select_keywords(
    client: VLLMClient,
    docs: List[Dict[str, Any]],
    topics: List[int],
    topic_names: Dict[int, str],
    topic_info: Dict[int, Dict[str, Any]],
    doc_keywords: Dict[str, List[str]],
    temperature: float,
    max_tokens: int,
    final_keyword_num: int,
) -> Dict[str, List[str]]:
    selected: Dict[str, List[str]] = {}
    for idx, doc in enumerate(docs):
        doc_id = str(doc["doc_id"])
        topic_name = topic_names.get(topics[idx], f"topic_{topics[idx]}")
        candidate_keywords = unique_preserve_order(
            list(topic_info.get(topics[idx], {}).get("keywords", [])) + doc_keywords.get(doc_id, [])
        )
        prompt = keyword_select_prompt(doc, topic_name, candidate_keywords, final_keyword_num)
        response = client.chat(prompt, temperature=temperature, max_tokens=max_tokens)
        values = parse_selected_keywords(completion_text(response), candidate_keywords, final_keyword_num)
        if len(values) < final_keyword_num:
            values = unique_preserve_order(values + candidate_keywords, limit=final_keyword_num)
        selected[doc_id] = values[:final_keyword_num]
    return selected


def generate_queries(
    client: VLLMClient,
    docs: List[Dict[str, Any]],
    topics: List[int],
    topic_names: Dict[int, str],
    selected_keywords: Dict[str, List[str]],
    temperature: float,
    max_tokens: int,
    examples: List[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    results: Dict[str, Dict[str, Any]] = {}
    for idx, doc in enumerate(docs):
        doc_id = str(doc["doc_id"])
        topic_name = topic_names.get(topics[idx], f"topic_{topics[idx]}")
        queries: List[str] = []
        prompt_tokens = 0
        output_tokens = 0
        calls = []
        doc_start = time.time()
        for call_idx in range(10):
            response = client.chat(
                multi_query_prompt(doc, topic_name, selected_keywords.get(doc_id, []), call_idx, examples),
                temperature=temperature,
                max_tokens=max_tokens,
            )
            p_tokens, c_tokens = usage_tokens(response)
            prompt_tokens += p_tokens
            output_tokens += c_tokens
            raw_text = completion_text(response)
            parsed = parse_queries(raw_text, expected=3)
            queries.extend(parsed[:3])
            calls.append(
                {
                    "call_idx": call_idx,
                    "prompt_tokens": p_tokens,
                    "completion_tokens": c_tokens,
                    "raw_text": raw_text,
                    "parsed_queries": parsed[:3],
                }
            )
        results[doc_id] = {
            "queries": queries[:30],
            "calls": calls,
            "wall_time_sec": time.time() - doc_start,
            "prompt_tokens": prompt_tokens,
            "output_tokens": output_tokens,
        }
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Doc2Query++ full microbenchmark pipeline.")
    parser.add_argument("--sample_path", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--vllm_base_url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--topic_temperature", type=float, default=0.0)
    parser.add_argument("--keyword_temperature", type=float, default=0.0)
    parser.add_argument("--qgen_temperature", type=float, default=0.8)
    parser.add_argument("--topic_max_tokens", type=int, default=32)
    parser.add_argument("--keyword_max_tokens", type=int, default=128)
    parser.add_argument("--qgen_max_tokens", type=int, default=192)
    parser.add_argument("--embedding_model_path", default=DEFAULT_EMBEDDING_MODEL_PATH)
    parser.add_argument("--topic_keyword_num", type=int, default=10)
    parser.add_argument("--representative_text_num", type=int, default=3)
    parser.add_argument("--keybert_top_n", type=int, default=20)
    parser.add_argument("--final_keyword_num", type=int, default=10)
    parser.add_argument("--bertopic_min_topic_size", type=int, default=10)
    parser.add_argument("--keybert_diversity", type=float, default=0.3)
    parser.add_argument("--allow_fallback_dependencies", type=int, default=0)
    parser.add_argument("--fewshot_examples_json", default="")
    parser.add_argument("--limit_docs", type=int, default=0)
    args = parser.parse_args()

    sample_path = Path(args.sample_path)
    output_root = Path(args.output_root)
    stage_log = output_root / "raw" / "d2qpp_full_stage_logs.jsonl"
    doc_log = output_root / "raw" / "d2qpp_full_doc_logs.jsonl"
    ensure_dir(stage_log.parent)
    stage_log.write_text("", encoding="utf-8")
    doc_log.write_text("", encoding="utf-8")

    client = VLLMClient(args.vllm_base_url, args.model)
    fewshot_examples = load_fewshot_examples(args.fewshot_examples_json)
    grouped = group_samples_by_dataset(sample_path, limit_docs=args.limit_docs)
    selected_doc_total = sum(len(docs) for docs in grouped.values())
    metadata_root = reset_dir(output_root / "artifacts" / "d2qpp" / "topics_keywords")
    generated_root = reset_dir(output_root / "artifacts" / "d2qpp" / "full" / "generated_queries")
    expanded_root = reset_dir(output_root / "artifacts" / "d2qpp" / "full" / "expanded_docs")

    processed_total = 0
    fallback_allowed = bool(args.allow_fallback_dependencies)
    embedding_model = load_embedding_model(args.embedding_model_path, fallback_allowed=fallback_allowed)
    print(
        f"[config] d2qpp_full selected_docs={selected_doc_total} "
        f"embedding_model_path={args.embedding_model_path}"
    )
    for dataset, docs in grouped.items():
        print(f"[dataset] d2qpp_full dataset={dataset} docs={len(docs)}")

        start = time.time()
        sentences_by_doc = {str(doc["doc_id"]): split_sentences(str(doc["text"])) for doc in docs}
        append_jsonl(stage_log, stage_log_row(dataset, "d2qpp_full", "sentence_split", len(docs), start))

        try:
            start = time.time()
            topics, topic_info = run_bertopic(
                docs,
                fallback_allowed=fallback_allowed,
                embedding_model=embedding_model,
                topic_keyword_num=args.topic_keyword_num,
                representative_text_num=args.representative_text_num,
                min_topic_size=args.bertopic_min_topic_size,
            )
            append_jsonl(stage_log, stage_log_row(dataset, "d2qpp_full", "bertopic_fit_transform", len(docs), start))

            start = time.time()
            topic_names = llm_topic_names(
                client,
                topic_info,
                args.topic_temperature,
                args.topic_max_tokens,
                fewshot_examples["topic_name"],
            )
            append_jsonl(stage_log, stage_log_row(dataset, "d2qpp_full", "topic_name_llm", len(docs), start))

            start = time.time()
            doc_keywords = run_keybert(
                docs,
                fallback_allowed=fallback_allowed,
                diversity=args.keybert_diversity,
                embedding_model=embedding_model,
                top_n=args.keybert_top_n,
            )
            append_jsonl(stage_log, stage_log_row(dataset, "d2qpp_full", "keybert_keywords", len(docs), start))

            start = time.time()
            selected_keywords = select_keywords(
                client,
                docs,
                topics,
                topic_names,
                topic_info,
                doc_keywords,
                args.keyword_temperature,
                args.keyword_max_tokens,
                args.final_keyword_num,
            )
            append_jsonl(stage_log, stage_log_row(dataset, "d2qpp_full", "keyword_select_llm", len(docs), start))

            start = time.time()
            qgen_results = generate_queries(
                client,
                docs,
                topics,
                topic_names,
                selected_keywords,
                args.qgen_temperature,
                args.qgen_max_tokens,
                fewshot_examples["multi_query"],
            )
            append_jsonl(stage_log, stage_log_row(dataset, "d2qpp_full", "multi_query_llm_30", len(docs), start))

            start = time.time()
            meta_path = metadata_root / f"{dataset}_doc_metadata.jsonl"
            gen_path = generated_root / f"{dataset}.jsonl"
            expanded_dataset_dir = ensure_dir(expanded_root / dataset)
            meta_path.write_text("", encoding="utf-8")
            gen_path.write_text("", encoding="utf-8")
            for idx, doc in enumerate(docs):
                doc_id = str(doc["doc_id"])
                queries = qgen_results[doc_id]["queries"]
                metadata = {
                    "dataset": dataset,
                    "doc_id": doc_id,
                    "topic": topics[idx],
                    "topic_name": topic_names.get(topics[idx], f"topic_{topics[idx]}"),
                    "topic_keywords": topic_info.get(topics[idx], {}).get("keywords", []),
                    "keywords": doc_keywords.get(doc_id, []),
                    "selected_keywords": selected_keywords.get(doc_id, []),
                    "sentence_count": len(sentences_by_doc.get(doc_id, [])),
                    "metadata_source": "d2qpp_full",
                }
                append_jsonl(meta_path, metadata)
                append_jsonl(
                    gen_path,
                    {
                        "dataset": dataset,
                        "doc_id": doc_id,
                        "queries": queries,
                        "calls": qgen_results[doc_id]["calls"],
                    },
                )
                expanded_text = str(doc["text"]).rstrip() + "\n" + "\n".join(queries) + "\n"
                expanded_bytes = write_text(expanded_dataset_dir / f"{clean_id_for_path(doc_id)}.txt", expanded_text)
                status = "ok" if len(queries) == 30 else "partial"
                append_jsonl(
                    doc_log,
                    {
                        "dataset": dataset,
                        "doc_id": doc_id,
                        "method": "d2qpp_full",
                        "num_generations": len(queries),
                        "llm_calls": 10,
                        "input_chars": len(str(doc.get("text", ""))),
                        "input_tokens": qgen_results[doc_id]["prompt_tokens"],
                        "output_tokens": qgen_results[doc_id]["output_tokens"],
                        "wall_time_sec": qgen_results[doc_id]["wall_time_sec"],
                        "decoded_text_bytes": sum(len((query + "\n").encode("utf-8")) for query in queries),
                        "trace_raw_bytes": 0,
                        "trace_gzip_bytes": 0,
                        "expansion_entry_bytes": expanded_bytes,
                        "status": status,
                        "error": None if status == "ok" else f"expected 30 queries, got {len(queries)}",
                    },
                )
            append_jsonl(stage_log, stage_log_row(dataset, "d2qpp_full", "persist_expanded_docs", len(docs), start))
        except Exception as exc:
            append_jsonl(
                stage_log,
                {
                    "dataset": dataset,
                    "method": "d2qpp_full",
                    "stage": "pipeline_error",
                    "num_docs": len(docs),
                    "total_time_sec": 0.0,
                    "sec_per_doc": 0.0,
                    "status": "error",
                    "error": str(exc),
                },
            )
            raise

        processed_total += len(docs)
        print(f"[progress] d2qpp_full processed_total={processed_total}")

    print(f"[done] d2qpp_full stage_log={stage_log} doc_log={doc_log} processed={processed_total}")


if __name__ == "__main__":
    main()
