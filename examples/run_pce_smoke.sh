#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
BENCHMARK_ROOT="${BENCHMARK_OUTPUT_ROOT:-${REPO_ROOT}/results/pce_smoke}"
BEIR_DATA_ROOT="${BEIR_DATA_ROOT:-/path/to/prepared/beir}"
LIMIT_DOCS="${LIMIT_DOCS:-50}"
WARMUP_DOCS="${WARMUP_DOCS:-1}"
VLLM_BASE_URL="${VLLM_BASE_URL:-http://127.0.0.1:8000/v1}"
VLLM_MODEL="${VLLM_MODEL:-/path/to/Llama-3.1-8B-Instruct}"

mkdir -p "${BENCHMARK_ROOT}"

"${PYTHON_BIN}" "${REPO_ROOT}/src/pce/00_record_environment.py" \
  --output_root "${BENCHMARK_ROOT}"

"${PYTHON_BIN}" "${REPO_ROOT}/src/pce/01_make_sample.py" \
  --beir_data_root "${BEIR_DATA_ROOT}" \
  --output_root "${BENCHMARK_ROOT}"

"${PYTHON_BIN}" "${REPO_ROOT}/src/pce/03_run_d2qpp_qgen_only.py" \
  --output_root "${BENCHMARK_ROOT}" \
  --sample_path "${BENCHMARK_ROOT}/sample_docs.jsonl" \
  --warmup_docs "${WARMUP_DOCS}" \
  --limit_docs "${LIMIT_DOCS}" \
  --vllm_base_url "${VLLM_BASE_URL}" \
  --model "${VLLM_MODEL}"

"${PYTHON_BIN}" "${REPO_ROOT}/src/pce/04_run_sde_trace.py" \
  --output_root "${BENCHMARK_ROOT}" \
  --sample_path "${BENCHMARK_ROOT}/sample_docs.jsonl" \
  --warmup_docs "${WARMUP_DOCS}" \
  --limit_docs "${LIMIT_DOCS}" \
  --vllm_base_url "${VLLM_BASE_URL}" \
  --model "${VLLM_MODEL}"

"${PYTHON_BIN}" "${REPO_ROOT}/src/pce/05_aggregate_costs.py" \
  --output_root "${BENCHMARK_ROOT}"
