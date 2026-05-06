#!/usr/bin/env python3
"""Core SDE evaluation with a compact document-side expansion index.

This script implements the main SDE path:

1. Read retained generation traces for each document.
2. Accumulate document-side lexical scores from token-level distributions.
3. Render one compact expansion document per original document.
4. Run BM25 on the original corpus and on the expansion corpus, then fuse.
"""

import argparse
import json
import math
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from transformers import AutoTokenizer

try:
    import pyterrier as pt
except Exception:
    pt = None


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

DEFAULT_BLACKLISTED_MODEL_IDS = {220, 128009}
BAD_DECODED_TERMS = {"s", "t", "re", "ve", "ll", "d", "m"}


def parse_token_id_blacklist(arg):
    if not arg:
        return set(DEFAULT_BLACKLISTED_MODEL_IDS)
    out = set()
    for item in str(arg).split(","):
        item = item.strip()
        if item:
            out.add(int(item))
    return out


def normalize_terms(text, min_token_len=2):
    tokens = re.findall(r"[a-z0-9]+", str(text).lower())
    out = []
    for token in tokens:
        if len(token) < min_token_len:
            continue
        if token in STOPWORDS:
            continue
        if token.isdigit():
            continue
        out.append(token)
    return out


def meaningful_token(token):
    if len(token) < 2:
        return False
    if token.isdigit():
        return False
    if not re.search(r"[a-z]", token):
        return False
    return True


def decoded_token_to_terms(text, min_token_len=2):
    terms = normalize_terms(text, min_token_len=min_token_len)
    out = []
    for term in terms:
        if term in BAD_DECODED_TERMS:
            continue
        if not meaningful_token(term):
            continue
        out.append(term)
    return out


def token_piece_has_word_start(piece):
    if not piece:
        return False
    return piece.startswith(("Ġ", "▁"))


def should_drop_short_continuation_piece(piece, terms, continuation_min_len=5):
    if token_piece_has_word_start(piece):
        return False
    if not terms:
        return False
    alpha_terms = [term for term in terms if re.search(r"[a-z]", term)]
    if not alpha_terms:
        return False
    return all(len(term) < continuation_min_len for term in alpha_terms)


def unique_preserve_order(items):
    out = []
    seen = set()
    for item in items:
        if item and item not in seen:
            out.append(item)
            seen.add(item)
    return out


def fragment_support_for_term(term, soft_scores, min_fragment_len):
    support = 0.0
    covered = [False] * len(term)
    fragments = []

    exact_score = float(soft_scores.get(term, 0.0))
    if exact_score > 0.0:
        support += exact_score
        covered = [True] * len(term)
        fragments.append(term)

    for fragment, score in soft_scores.items():
        if fragment == term:
            continue
        if len(fragment) < min_fragment_len:
            continue
        if fragment in STOPWORDS or fragment in BAD_DECODED_TERMS:
            continue
        if not meaningful_token(fragment):
            continue
        start = term.find(fragment)
        if start < 0:
            continue
        support += float(score)
        fragments.append(fragment)
        for idx in range(start, min(len(term), start + len(fragment))):
            covered[idx] = True

    coverage = (sum(1 for flag in covered if flag) / len(term)) if term else 0.0
    return support, coverage, fragments


def recover_full_word_scores(
    generated_terms,
    soft_scores,
    mode,
    min_generated_count,
    min_soft_support,
    min_soft_coverage,
    min_fragment_len,
    generated_term_bonus,
):
    if mode == "none":
        return {}, set(), {"selected_generated_terms": 0, "suppressed_fragments": 0}

    counts = Counter(generated_terms)
    recovered_scores = {}
    suppressed_fragments = set()

    for term in unique_preserve_order(generated_terms):
        support, coverage, fragments = fragment_support_for_term(
            term,
            soft_scores,
            min_fragment_len,
        )
        selected = mode == "all"
        if mode == "supported":
            selected = (
                counts[term] >= min_generated_count
                or (support >= min_soft_support and coverage >= min_soft_coverage)
            )
        if not selected:
            continue

        recovered_scores[term] = support + (counts[term] * generated_term_bonus)
        for fragment in fragments:
            if fragment != term:
                suppressed_fragments.add(fragment)

    stats = {
        "selected_generated_terms": len(recovered_scores),
        "suppressed_fragments": len(suppressed_fragments),
    }
    return recovered_scores, suppressed_fragments, stats


def sanitize_query_for_pyterrier(text):
    return " ".join(re.findall(r"[A-Za-z0-9]+", str(text)))


def load_corpus(path):
    doc_ids = []
    doc_texts = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t", 1)
            if len(parts) < 2:
                continue
            doc_ids.append(parts[0])
            doc_texts.append(parts[1])
    return doc_ids, doc_texts


def load_queries(path):
    queries = {}
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            qid = str(row.get("qid", row.get("_id")))
            text = row.get("question", row.get("text", ""))
            if text:
                queries[qid] = text
    return queries


def load_qrels(path):
    qrels = defaultdict(dict)
    with open(path, "r", encoding="utf-8") as handle:
        first = True
        for line in handle:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            if first and parts[0].lower() in {"qid", "query_id"}:
                first = False
                continue
            first = False
            qid = parts[0]
            did = parts[2]
            rel = int(parts[3]) if len(parts) >= 4 else 1
            qrels[qid][did] = rel
    return dict(qrels)


def prepare_query_df(queries, qrels):
    rows = []
    for qid in qrels.keys():
        if qid not in queries:
            continue
        query = sanitize_query_for_pyterrier(queries[qid]).strip()
        if query:
            rows.append({"qid": qid, "query": query})
    return pd.DataFrame(rows)


def ensure_pyterrier():
    if pt is None:
        raise RuntimeError("PyTerrier is not available in the current environment.")
    if hasattr(pt, "java"):
        if not pt.java.started():
            pt.java.init()
    else:
        if not pt.started():
            pt.init()


def maybe_load_indexref(index_dir):
    data_properties = Path(index_dir) / "data.properties"
    if data_properties.exists():
        return pt.IndexRef.of(str(data_properties))
    return None


def index_corpus(index_dir, doc_ids, doc_texts, reuse_existing):
    existing = maybe_load_indexref(index_dir) if reuse_existing else None
    if existing is not None:
        print(f"Reusing existing index at {index_dir}")
        return existing

    Path(index_dir).mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame({"docno": doc_ids, "text": doc_texts})
    indexer = pt.DFIndexer(index_dir, overwrite=True)
    return indexer.index(frame["text"], frame["docno"])


def read_index_stats(index_dir):
    index_dir = Path(index_dir)
    props = {}
    props_path = index_dir / "data.properties"
    if props_path.is_file():
        with open(props_path, "r", encoding="utf-8") as handle:
            for raw in handle:
                line = raw.strip()
                if not line or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                props[key.strip()] = value.strip()

    def safe_int(key):
        try:
            return int(props.get(key, "0"))
        except Exception:
            return 0

    size_bytes = 0
    if index_dir.is_dir():
        for path in index_dir.rglob("*"):
            if path.is_file():
                size_bytes += path.stat().st_size

    return {
        "index_dir": str(index_dir),
        "num_documents": safe_int("num.Documents"),
        "num_terms": safe_int("num.Terms"),
        "num_pointers": safe_int("num.Pointers"),
        "size_bytes": size_bytes,
    }


def build_retriever(index_ref, topk, k1, b):
    if hasattr(pt, "terrier") and hasattr(pt.terrier, "Retriever"):
        try:
            return pt.terrier.Retriever(
                index_ref,
                wmodel="BM25",
                num_results=topk,
                controls={"bm25.k_1": str(k1), "bm25.b": str(b)},
            )
        except TypeError:
            return pt.terrier.Retriever(
                index_ref,
                wmodel="BM25",
                num_results=topk,
                properties={"bm25.k_1": str(k1), "bm25.b": str(b)},
            )
    return pt.BatchRetrieve(
        index_ref,
        wmodel="BM25",
        num_results=topk,
        controls={"bm25.k_1": str(k1), "bm25.b": str(b)},
    )


def normalize_scores(frame, score_col, mode):
    if frame.empty:
        return frame.copy()
    out = frame.copy()
    if mode == "direct":
        out["norm_score"] = out[score_col].astype(float)
        return out[["qid", "docno", "norm_score"]]
    if mode == "minmax":
        stats = out.groupby("qid")[score_col].agg(["min", "max"]).reset_index()
        out = out.merge(stats, on="qid", how="left")
        denom = out["max"] - out["min"]
        out["norm_score"] = np.where(
            denom.abs() < 1e-12,
            0.0,
            (out[score_col] - out["min"]) / denom,
        )
        return out[["qid", "docno", "norm_score"]]
    if mode == "zscore":
        stats = out.groupby("qid")[score_col].agg(["mean", "std"]).reset_index()
        out = out.merge(stats, on="qid", how="left")
        out["norm_score"] = np.where(
            out["std"].fillna(0.0).abs() < 1e-12,
            0.0,
            (out[score_col] - out["mean"]) / out["std"],
        )
        return out[["qid", "docno", "norm_score"]]
    raise ValueError(f"Unknown normalization mode: {mode}")


def fuse_result_frames(doc_results, expansion_results, score_normalization, fusion_alpha):
    doc_part = normalize_scores(
        doc_results[["qid", "docno", "score"]], "score", score_normalization
    ).rename(columns={"norm_score": "doc_score"})
    expansion_part = normalize_scores(
        expansion_results[["qid", "docno", "score"]], "score", score_normalization
    ).rename(columns={"norm_score": "expansion_score"})
    fused = doc_part.merge(expansion_part, on=["qid", "docno"], how="outer")
    fused["doc_score"] = fused["doc_score"].fillna(0.0)
    fused["expansion_score"] = fused["expansion_score"].fillna(0.0)
    fused["score"] = (
        ((1.0 - fusion_alpha) * fused["doc_score"])
        + (fusion_alpha * fused["expansion_score"])
    )
    fused = fused.sort_values(["qid", "score", "docno"], ascending=[True, False, True])
    fused["rank"] = fused.groupby("qid").cumcount()
    return fused[["qid", "docno", "score", "rank"]]


def compute_metrics_from_ranking(ranking_df, qrels):
    grouped = defaultdict(list)
    for row in ranking_df.itertuples(index=False):
        grouped[str(row.qid)].append((str(row.docno), int(row.rank)))

    ndcgs = []
    mrrs = []
    recalls = []
    aps = []
    for qid, rel_map in qrels.items():
        ranked = sorted(grouped.get(qid, []), key=lambda item: item[1])
        docs = [docno for docno, _ in ranked]

        mrr = 0.0
        for idx, docno in enumerate(docs[:10], start=1):
            if rel_map.get(docno, 0) > 0:
                mrr = 1.0 / idx
                break

        dcg = 0.0
        for idx, docno in enumerate(docs[:10], start=1):
            rel = rel_map.get(docno, 0)
            if rel > 0:
                dcg += (2 ** rel - 1) / math.log2(idx + 1)
        ideal_rels = sorted(
            [rel for rel in rel_map.values() if rel > 0],
            reverse=True,
        )[:10]
        idcg = 0.0
        for idx, rel in enumerate(ideal_rels, start=1):
            idcg += (2 ** rel - 1) / math.log2(idx + 1)
        ndcg = (dcg / idcg) if idcg > 0 else 0.0

        total_rel = sum(1 for rel in rel_map.values() if rel > 0)
        recall = 0.0
        if total_rel > 0:
            hits = sum(1 for docno in docs[:100] if rel_map.get(docno, 0) > 0)
            recall = hits / total_rel

        ap = 0.0
        if total_rel > 0:
            hit_count = 0
            precision_sum = 0.0
            for idx, docno in enumerate(docs, start=1):
                if rel_map.get(docno, 0) > 0:
                    hit_count += 1
                    precision_sum += hit_count / idx
            ap = precision_sum / total_rel

        ndcgs.append(ndcg)
        mrrs.append(mrr)
        recalls.append(recall)
        aps.append(ap)

    return {
        "n_queries": len(ndcgs),
        "ndcg_at_10": float(np.mean(ndcgs)) if ndcgs else 0.0,
        "mrr_at_10": float(np.mean(mrrs)) if mrrs else 0.0,
        "recall_at_100": float(np.mean(recalls)) if recalls else 0.0,
        "map": float(np.mean(aps)) if aps else 0.0,
    }


def write_trec_run(ranking_df, out_path, tag):
    if ranking_df is None or ranking_df.empty:
        Path(out_path).write_text("", encoding="utf-8")
        return
    out = ranking_df.sort_values(["qid", "rank", "docno"], ascending=[True, True, True])
    with open(out_path, "w", encoding="utf-8") as handle:
        for row in out.itertuples(index=False):
            handle.write(
                f"{str(row.qid)} Q0 {str(row.docno)} {int(row.rank) + 1} "
                f"{float(row.score):.6f} {tag}\n"
            )


def write_summary_tsv(path, score_normalization, fusion_alpha, metrics):
    row = {
        "score_normalization": score_normalization,
        "fusion_alpha": fusion_alpha,
        "ndcg@10": metrics["ndcg_at_10"],
        "mrr@10": metrics["mrr_at_10"],
        "map": metrics["map"],
        "recall@100": metrics["recall_at_100"],
    }
    pd.DataFrame([row]).to_csv(path, sep="\t", index=False)


def iter_selected_trace_rows(expansion_jsonl, sample_idx_max):
    with open(expansion_jsonl, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            doc_id = str(row.get("doc_id", row.get("pos_id", ""))).strip()
            if not doc_id:
                continue
            if sample_idx_max is not None:
                sample_idx = row.get("sample_idx")
                if sample_idx is not None and int(sample_idx) > sample_idx_max:
                    continue
            yield doc_id, row


def build_document_expansion_texts(
    expansion_jsonl,
    model_path,
    sample_idx_max,
    generated_query_term_mode,
    generated_query_min_count,
    generated_query_min_soft_support,
    generated_query_min_soft_coverage,
    generated_query_score_bonus,
    recovery_fragment_min_len,
    suppress_recovered_fragments,
    per_step_topk,
    prob_threshold,
    max_soft_terms_per_doc,
    term_weight_mode,
    repeat_score_scale,
    repeat_max_times,
    blacklisted_model_ids,
):
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True,
    )
    doc_term_scores = defaultdict(lambda: defaultdict(float))
    doc_generated_terms = defaultdict(list)

    for doc_id, row in iter_selected_trace_rows(expansion_jsonl, sample_idx_max):
        if generated_query_term_mode != "none":
            doc_generated_terms[doc_id].extend(normalize_terms(row.get("query_text", "")))

        indices = row.get("indices", [])
        probs = row.get("probs", [])
        for step_ids, step_probs in zip(indices, probs):
            if not isinstance(step_ids, list) or not isinstance(step_probs, list):
                continue
            for token_id, prob in list(zip(step_ids, step_probs))[:per_step_topk]:
                try:
                    token_id = int(token_id)
                    prob = float(prob)
                except Exception:
                    continue
                if token_id in blacklisted_model_ids:
                    continue
                if prob < prob_threshold:
                    continue

                try:
                    token_piece = tokenizer.convert_ids_to_tokens(token_id)
                except Exception:
                    token_piece = ""
                try:
                    decoded_text = tokenizer.decode([token_id])
                except Exception:
                    continue

                terms = decoded_token_to_terms(decoded_text)
                if should_drop_short_continuation_piece(token_piece, terms):
                    continue
                for term in terms:
                    doc_term_scores[doc_id][term] += prob

    doc_to_expansion_text = {}
    docs_with_expansion = 0
    total_generated_terms = 0
    total_selected_generated_terms = 0
    total_suppressed_fragments = 0
    total_unique_terms = 0
    total_rendered_terms = 0
    max_generated_terms = 0
    max_selected_generated_terms = 0
    max_unique_terms = 0
    max_rendered_terms = 0

    all_doc_ids = sorted(set(doc_term_scores.keys()) | set(doc_generated_terms.keys()))
    for doc_id in all_doc_ids:
        score_map = doc_term_scores.get(doc_id, {})
        generated_terms = doc_generated_terms.get(doc_id, [])
        recovered_scores, suppressed_fragments, recovery_stats = recover_full_word_scores(
            generated_terms,
            score_map,
            mode=generated_query_term_mode,
            min_generated_count=generated_query_min_count,
            min_soft_support=generated_query_min_soft_support,
            min_soft_coverage=generated_query_min_soft_coverage,
            min_fragment_len=recovery_fragment_min_len,
            generated_term_bonus=generated_query_score_bonus,
        )

        merged_scores = defaultdict(float)
        for term, score in score_map.items():
            if suppress_recovered_fragments and term in suppressed_fragments and term not in recovered_scores:
                continue
            merged_scores[term] += float(score)
        for term, score in recovered_scores.items():
            merged_scores[term] += float(score)

        ranked_terms = sorted(merged_scores.items(), key=lambda item: (-item[1], item[0]))
        retained = ranked_terms[:max_soft_terms_per_doc]
        rendered_terms = []
        for term, score in retained:
            repeats = 1
            if term_weight_mode == "repeat_by_score":
                repeats = max(
                    1,
                    min(repeat_max_times, int(math.ceil(score * repeat_score_scale))),
                )
            rendered_terms.extend([term] * repeats)
        if not rendered_terms:
            continue

        doc_to_expansion_text[doc_id] = " ".join(rendered_terms)
        docs_with_expansion += 1
        unique_generated_terms = len(unique_preserve_order(generated_terms))
        total_generated_terms += unique_generated_terms
        total_selected_generated_terms += recovery_stats["selected_generated_terms"]
        total_suppressed_fragments += recovery_stats["suppressed_fragments"]
        total_unique_terms += len(retained)
        total_rendered_terms += len(rendered_terms)
        max_generated_terms = max(max_generated_terms, unique_generated_terms)
        max_selected_generated_terms = max(
            max_selected_generated_terms,
            recovery_stats["selected_generated_terms"],
        )
        max_unique_terms = max(max_unique_terms, len(retained))
        max_rendered_terms = max(max_rendered_terms, len(rendered_terms))

    stats = {
        "docs_with_expansion": docs_with_expansion,
        "generated_query_term_mode": generated_query_term_mode,
        "avg_generated_terms_per_expanded_doc": (
            total_generated_terms / docs_with_expansion if docs_with_expansion else 0.0
        ),
        "avg_selected_generated_terms_per_expanded_doc": (
            total_selected_generated_terms / docs_with_expansion if docs_with_expansion else 0.0
        ),
        "avg_suppressed_fragments_per_expanded_doc": (
            total_suppressed_fragments / docs_with_expansion if docs_with_expansion else 0.0
        ),
        "avg_unique_terms_per_expanded_doc": (
            total_unique_terms / docs_with_expansion if docs_with_expansion else 0.0
        ),
        "avg_rendered_terms_per_expanded_doc": (
            total_rendered_terms / docs_with_expansion if docs_with_expansion else 0.0
        ),
        "max_generated_terms_per_expanded_doc": max_generated_terms,
        "max_selected_generated_terms_per_expanded_doc": max_selected_generated_terms,
        "max_unique_terms_per_expanded_doc": max_unique_terms,
        "max_rendered_terms_per_expanded_doc": max_rendered_terms,
        "sample_idx_max": sample_idx_max,
        "generated_query_min_count": generated_query_min_count,
        "generated_query_min_soft_support": generated_query_min_soft_support,
        "generated_query_min_soft_coverage": generated_query_min_soft_coverage,
        "generated_query_score_bonus": generated_query_score_bonus,
        "recovery_fragment_min_len": recovery_fragment_min_len,
        "suppress_recovered_fragments": suppress_recovered_fragments,
        "per_step_topk": per_step_topk,
        "prob_threshold": prob_threshold,
        "max_soft_terms_per_doc": max_soft_terms_per_doc,
        "term_weight_mode": term_weight_mode,
        "repeat_score_scale": repeat_score_scale,
        "repeat_max_times": repeat_max_times,
        "blacklisted_model_ids": sorted(blacklisted_model_ids),
    }
    return doc_to_expansion_text, stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus_tsv", required=True)
    parser.add_argument("--queries_json", required=True)
    parser.add_argument("--qrels_tsv", required=True)
    parser.add_argument("--expansion_jsonl", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--llm_path", dest="model_path", help=argparse.SUPPRESS)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--doc_index_dir", default=None)
    parser.add_argument("--expansion_index_dir", default=None)
    parser.add_argument("--reuse_expansion_index", action="store_true")
    parser.add_argument("--cleanup_expansion_index_dir", action="store_true")
    parser.add_argument("--sample_idx_max", type=int, default=0)
    parser.add_argument("--doc_retrieval_topn", type=int, default=300)
    parser.add_argument("--expansion_retrieval_topn", type=int, default=1000)
    parser.add_argument("--fusion_alpha", type=float, default=0.5)
    parser.add_argument(
        "--score_normalization",
        default="direct",
        choices=["direct", "minmax", "zscore"],
    )
    parser.add_argument("--doc_k1", type=float, default=0.9)
    parser.add_argument("--paper_k1", dest="doc_k1", type=float, help=argparse.SUPPRESS)
    parser.add_argument("--doc_b", type=float, default=0.4)
    parser.add_argument("--paper_b", dest="doc_b", type=float, help=argparse.SUPPRESS)
    parser.add_argument("--expansion_k1", type=float, default=0.9)
    parser.add_argument("--expansion_b", type=float, default=0.4)
    parser.add_argument("--per_step_topk", type=int, default=5)
    parser.add_argument(
        "--soft_topk_per_step",
        dest="per_step_topk",
        type=int,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--prob_threshold", type=float, default=0.01)
    parser.add_argument("--max_soft_terms_per_doc", type=int, default=256)
    parser.add_argument(
        "--term_weight_mode",
        default="repeat_by_score",
        choices=["unique", "repeat_by_score"],
    )
    parser.add_argument("--repeat_score_scale", type=float, default=3.0)
    parser.add_argument("--repeat_max_times", type=int, default=3)
    parser.add_argument("--model_token_id_blacklist", default="220,128009")
    parser.add_argument(
        "--generated_query_term_mode",
        default="supported",
        choices=["none", "supported", "all"],
        help=(
            "How generated query_text terms are used for full-word recovery. "
            "'supported' keeps generated terms only when they repeat across prompts "
            "or are supported by soft token fragments."
        ),
    )
    parser.add_argument("--generated_query_min_count", type=int, default=2)
    parser.add_argument("--generated_query_min_soft_support", type=float, default=0.25)
    parser.add_argument("--generated_query_min_soft_coverage", type=float, default=0.25)
    parser.add_argument("--generated_query_score_bonus", type=float, default=1.0)
    parser.add_argument("--recovery_fragment_min_len", type=int, default=2)
    parser.add_argument("--keep_recovered_fragments", action="store_true")
    parser.add_argument(
        "--exclude_generated_query_terms",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args()
    if args.exclude_generated_query_terms:
        args.generated_query_term_mode = "none"

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ensure_pyterrier()

    blacklisted_model_ids = parse_token_id_blacklist(args.model_token_id_blacklist)

    doc_ids, doc_texts = load_corpus(args.corpus_tsv)
    queries = load_queries(args.queries_json)
    qrels = load_qrels(args.qrels_tsv)
    query_df = prepare_query_df(queries, qrels)

    doc_to_expansion_text, expansion_doc_stats = build_document_expansion_texts(
        expansion_jsonl=args.expansion_jsonl,
        model_path=args.model_path,
        sample_idx_max=args.sample_idx_max,
        generated_query_term_mode=args.generated_query_term_mode,
        generated_query_min_count=args.generated_query_min_count,
        generated_query_min_soft_support=args.generated_query_min_soft_support,
        generated_query_min_soft_coverage=args.generated_query_min_soft_coverage,
        generated_query_score_bonus=args.generated_query_score_bonus,
        recovery_fragment_min_len=args.recovery_fragment_min_len,
        suppress_recovered_fragments=not args.keep_recovered_fragments,
        per_step_topk=args.per_step_topk,
        prob_threshold=args.prob_threshold,
        max_soft_terms_per_doc=args.max_soft_terms_per_doc,
        term_weight_mode=args.term_weight_mode,
        repeat_score_scale=args.repeat_score_scale,
        repeat_max_times=args.repeat_max_times,
        blacklisted_model_ids=blacklisted_model_ids,
    )

    expansion_doc_ids = []
    expansion_doc_texts = []
    for doc_id in doc_ids:
        expansion_text = doc_to_expansion_text.get(doc_id)
        if expansion_text:
            expansion_doc_ids.append(doc_id)
            expansion_doc_texts.append(expansion_text)

    doc_index_dir = args.doc_index_dir or str(out_dir / "doc_index")
    expansion_index_dir = args.expansion_index_dir or str(out_dir / "expansion_index")

    doc_index_ref = index_corpus(
        doc_index_dir,
        doc_ids,
        doc_texts,
        reuse_existing=True,
    )
    doc_index_stats = read_index_stats(doc_index_dir)

    expansion_index_stats = {
        "index_dir": str(expansion_index_dir),
        "num_documents": 0,
        "num_terms": 0,
        "num_pointers": 0,
        "size_bytes": 0,
    }
    if expansion_doc_ids:
        expansion_index_ref = index_corpus(
            expansion_index_dir,
            expansion_doc_ids,
            expansion_doc_texts,
            reuse_existing=args.reuse_expansion_index,
        )
        expansion_index_stats = read_index_stats(expansion_index_dir)
        expansion_retriever = build_retriever(
            expansion_index_ref,
            args.expansion_retrieval_topn,
            args.expansion_k1,
            args.expansion_b,
        )
        expansion_results = expansion_retriever.transform(query_df)[
            ["qid", "docno", "score", "rank"]
        ].copy()
    else:
        expansion_results = pd.DataFrame(columns=["qid", "docno", "score", "rank"])

    doc_retriever = build_retriever(
        doc_index_ref,
        args.doc_retrieval_topn,
        args.doc_k1,
        args.doc_b,
    )
    doc_results = doc_retriever.transform(query_df)[["qid", "docno", "score", "rank"]].copy()

    fused_results = fuse_result_frames(
        doc_results=doc_results,
        expansion_results=expansion_results,
        score_normalization=args.score_normalization,
        fusion_alpha=args.fusion_alpha,
    )
    metrics = compute_metrics_from_ranking(fused_results, qrels)

    result_path = out_dir / "dual_index_fusion_results.json"
    summary_path = out_dir / "dual_index_fusion_summary.tsv"
    run_path = out_dir / "dual_index_fusion_best.run"
    expansion_stats_path = out_dir / "expansion_index_stats.json"

    write_trec_run(fused_results, run_path, tag="dual_index")
    write_summary_tsv(
        summary_path,
        args.score_normalization,
        args.fusion_alpha,
        metrics,
    )

    result_payload = {
        "n_docs": len(doc_ids),
        "n_expansion_docs": len(expansion_doc_ids),
        "n_eval_queries": int(query_df.shape[0]),
        "doc_index_dir": doc_index_dir,
        "expansion_index_dir": expansion_index_dir,
        "doc_index_stats": doc_index_stats,
        "expansion_index_stats": expansion_index_stats,
        "expansion_doc_stats": expansion_doc_stats,
        "config": {
            "model_path": args.model_path,
            "sample_idx_max": args.sample_idx_max,
            "generated_query_term_mode": args.generated_query_term_mode,
            "generated_query_min_count": args.generated_query_min_count,
            "generated_query_min_soft_support": args.generated_query_min_soft_support,
            "generated_query_min_soft_coverage": args.generated_query_min_soft_coverage,
            "generated_query_score_bonus": args.generated_query_score_bonus,
            "recovery_fragment_min_len": args.recovery_fragment_min_len,
            "suppress_recovered_fragments": not args.keep_recovered_fragments,
            "doc_retrieval_topn": args.doc_retrieval_topn,
            "expansion_retrieval_topn": args.expansion_retrieval_topn,
            "fusion_alpha": args.fusion_alpha,
            "score_normalization": args.score_normalization,
            "doc_k1": args.doc_k1,
            "doc_b": args.doc_b,
            "expansion_k1": args.expansion_k1,
            "expansion_b": args.expansion_b,
            "per_step_topk": args.per_step_topk,
            "prob_threshold": args.prob_threshold,
            "max_soft_terms_per_doc": args.max_soft_terms_per_doc,
            "term_weight_mode": args.term_weight_mode,
            "repeat_score_scale": args.repeat_score_scale,
            "repeat_max_times": args.repeat_max_times,
            "model_token_id_blacklist": sorted(blacklisted_model_ids),
        },
        "best_result": metrics,
        "best_run": str(run_path),
        "summary_tsv": str(summary_path),
    }
    result_path.write_text(
        json.dumps(result_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    expansion_stats_path.write_text(
        json.dumps(expansion_index_stats, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("----------------------------------------------------------------------")
    print("[SDE Dual-Index Summary]")
    print(
        f"eval_queries={int(query_df.shape[0])} "
        f"expansion_docs={len(expansion_doc_ids)} "
        f"score_normalization={args.score_normalization} "
        f"fusion_alpha={args.fusion_alpha:.2f}"
    )
    print(
        f"doc_k1={args.doc_k1} doc_b={args.doc_b} "
        f"expansion_k1={args.expansion_k1} expansion_b={args.expansion_b}"
    )
    print(
        f"per_step_topk={args.per_step_topk} prob_threshold={args.prob_threshold:.4f} "
        f"max_soft_terms_per_doc={args.max_soft_terms_per_doc} "
        f"term_weight_mode={args.term_weight_mode} "
        f"generated_query_term_mode={args.generated_query_term_mode}"
    )
    print(
        f"ndcg@10={metrics['ndcg_at_10']:.6f} "
        f"mrr@10={metrics['mrr_at_10']:.6f} "
        f"map={metrics['map']:.6f} "
        f"recall@100={metrics['recall_at_100']:.6f}"
    )
    print(f"[Result JSON] {result_path}")
    print(f"[Summary TSV] {summary_path}")
    print(f"[Best Run   ] {run_path}")
    print("----------------------------------------------------------------------")

    if args.cleanup_expansion_index_dir:
        expansion_index_path = Path(expansion_index_dir)
        if expansion_index_path.is_dir() and expansion_index_path != (out_dir / "expansion_index"):
            shutil.rmtree(expansion_index_path, ignore_errors=True)
            print(f"[Cleanup] removed temporary expansion index at {expansion_index_dir}")


if __name__ == "__main__":
    main()
