#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

: "${CORPUS_TSV:=/path/to/collection.tsv}"
: "${QUERIES_JSON:=/path/to/queries.json}"
: "${QRELS_TSV:=/path/to/qrels.test.tsv}"
: "${EXPANSION_JSONL:=results/sde/generated_queries.jsonl}"
: "${LLM_PATH:=/path/to/Llama-3.1-8B-Instruct}"
: "${OUTPUT_DIR:=results/sde/dual_index_eval}"

mkdir -p "${OUTPUT_DIR}"

"${PYTHON_BIN}" "${REPO_ROOT}/src/sde/eval_bm25_dual_index_fusion.py" \
  --corpus_tsv "${CORPUS_TSV}" \
  --queries_json "${QUERIES_JSON}" \
  --qrels_tsv "${QRELS_TSV}" \
  --expansion_jsonl "${EXPANSION_JSONL}" \
  --llm_path "${LLM_PATH}" \
  --out_dir "${OUTPUT_DIR}"
