#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/data/beir}"
TRACE_ROOT="${TRACE_ROOT:-${REPO_ROOT}/data/traces}"
MODEL_PATH="${MODEL_PATH:-${REPO_ROOT}/models/tokenizer}"

FORMAL_ROOT="${FORMAL_ROOT:-${REPO_ROOT}/results/system_efficiency_formal}"
COMPONENT_ROOT="${COMPONENT_ROOT:-${REPO_ROOT}/results/factorized_qdoc_components}"
ONLINE_INDEX_ROOT="${ONLINE_INDEX_ROOT:-${REPO_ROOT}/results/factorized_qdoc_online_index}"
LIFECYCLE_ROOT="${LIFECYCLE_ROOT:-${REPO_ROOT}/results/factorized_qdoc_lifecycle}"
ONLINE_ROOT="${ONLINE_ROOT:-${REPO_ROOT}/results/factorized_qdoc_online}"

MODE="${1:-all}"

export TERRIER_VERSION="${TERRIER_VERSION:-5.11}"
export TERRIER_HELPER_VERSION="${TERRIER_HELPER_VERSION:-0.0.8}"

run_baselines() {
  "${PYTHON_BIN}" "${SCRIPT_DIR}/measure_system_efficiency_formal.py" \
    --data_root "${DATA_ROOT}" \
    --trace_root "${TRACE_ROOT}" \
    --out_root "${FORMAL_ROOT}"
}

prepare_factorized_inputs() {
  "${PYTHON_BIN}" \
    "${SCRIPT_DIR}/measure_factorized_qdoc_lifecycle.py" \
    --formal-root "${FORMAL_ROOT}" \
    --trace-root "${TRACE_ROOT}" \
    --model-path "${MODEL_PATH}" \
    --out-root "${COMPONENT_ROOT}" \
    --skip-build
}

build_online_index() {
  "${PYTHON_BIN}" \
    "${SCRIPT_DIR}/measure_factorized_qdoc_lifecycle.py" \
    --formal-root "${FORMAL_ROOT}" \
    --trace-root "${TRACE_ROOT}" \
    --model-path "${MODEL_PATH}" \
    --out-root "${ONLINE_INDEX_ROOT}" \
    --prepared-root "${COMPONENT_ROOT}" \
    --build-repeats 1 \
    --build-cpu-count 8 \
    --skip-prepare
}

measure_lifecycle() {
  "${PYTHON_BIN}" \
    "${SCRIPT_DIR}/measure_factorized_qdoc_lifecycle.py" \
    --formal-root "${FORMAL_ROOT}" \
    --trace-root "${TRACE_ROOT}" \
    --model-path "${MODEL_PATH}" \
    --out-root "${LIFECYCLE_ROOT}" \
    --prepared-root "${COMPONENT_ROOT}" \
    --build-repeats 3 \
    --build-cpu-count 8
}

measure_online() {
  "${PYTHON_BIN}" \
    "${SCRIPT_DIR}/benchmark_factorized_querydoc_formal.py" \
    --data-root "${DATA_ROOT}" \
    --formal-root "${FORMAL_ROOT}" \
    --factor-root "${ONLINE_INDEX_ROOT}/factor_indexes/repeat_0" \
    --out-root "${ONLINE_ROOT}" \
    --repeats 3 \
    --warmup-queries 50
}

case "${MODE}" in
  baselines)
    run_baselines
    ;;
  prepare-factorized)
    prepare_factorized_inputs
    ;;
  online-index)
    build_online_index
    ;;
  lifecycle)
    measure_lifecycle
    ;;
  online)
    measure_online
    ;;
  all)
    run_baselines
    prepare_factorized_inputs
    build_online_index
    measure_lifecycle
    measure_online
    ;;
  *)
    printf 'usage: %s {baselines|prepare-factorized|online-index|lifecycle|online|all}\n' \
      "$0" >&2
    exit 2
    ;;
esac
