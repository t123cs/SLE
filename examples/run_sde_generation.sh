#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_PATH="${LLAMA31_8B_INSTRUCT:-${LLM_PATH:-/path/to/Llama-3.1-8B-Instruct}}"

: "${COLLECTION_TSV:=/path/to/collection.tsv}"
: "${OUTPUT_JSONL:=results/sde/generated_traces.full.jsonl}"

mkdir -p "$(dirname "${OUTPUT_JSONL}")"

"${PYTHON_BIN}" "${REPO_ROOT}/src/sde/prepare_expectation_data_multitemplate_unfiltered.py" \
  --llama_path "${MODEL_PATH}" \
  --corpus_path "${COLLECTION_TSV}" \
  --output_path "${OUTPUT_JSONL}"
