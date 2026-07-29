#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG_PATH="${SDE_SINGLE_INDEX_CONFIG:-${REPO_ROOT}/configs/sde/sde_single_index_beir5.json}"
DRY_RUN="${DRY_RUN:-0}"

usage() {
  cat <<EOF
Usage: $(basename "$0") [--config PATH] [--dry-run]

The JSON config defines the trace, term-selection, BM25, and per-dataset
settings. BEIR_DATA_ROOT and TRACE_ROOT identify the prepared inputs.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      if [[ $# -lt 2 ]]; then
        echo "--config requires a path" >&2
        exit 2
      fi
      CONFIG_PATH="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ "${DRY_RUN}" != "0" && "${DRY_RUN}" != "1" ]]; then
  echo "DRY_RUN must be 0 or 1, got: ${DRY_RUN}" >&2
  exit 2
fi

: "${BEIR_DATA_ROOT:?Set BEIR_DATA_ROOT to the prepared BEIR root}"
: "${TRACE_ROOT:?Set TRACE_ROOT to the six-query SDE trace root}"

OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/results/sde_single_index_beir5}"
CACHE_ROOT="${CACHE_ROOT:-${OUTPUT_ROOT}/cache}"
QUERY_SPLIT="${QUERY_SPLIT:-test}"
BUILD_CACHE="${BUILD_CACHE:-1}"
RUN_HARD_CONTROL="${RUN_HARD_CONTROL:-1}"

if ! CONFIG_SHELL="$(
  "${PYTHON_BIN}" "${REPO_ROOT}/src/sde/single_index/config.py" \
    --config "${CONFIG_PATH}" \
    --format shell
)"; then
  exit 2
fi
# config.py validates each value and shell-quotes every rendered word.
eval "${CONFIG_SHELL}"

run_command() {
  if [[ "${DRY_RUN}" == "1" ]]; then
    printf 'DRY-RUN'
    printf ' %q' "$@"
    printf '\n'
  else
    "$@"
  fi
}

if [[ "${DRY_RUN}" != "1" ]]; then
  mkdir -p "${CACHE_ROOT}" "${OUTPUT_ROOT}/eval"
fi

if [[ "${BUILD_CACHE}" == "1" ]]; then
  model_args=()
  if [[ -n "${MODEL_PATH:-}" ]]; then
    model_args+=(--model_path "${MODEL_PATH}")
  fi
  run_command "${PYTHON_BIN}" \
    "${REPO_ROOT}/src/sde/single_index/prepare_cache.py" \
    --data_root "${BEIR_DATA_ROOT}" \
    --trace_root "${TRACE_ROOT}" \
    --cache_root "${CACHE_ROOT}" \
    --datasets "${CONFIG_DATASETS[@]}" \
    "${CONFIG_TRACE_ARGS[@]}" \
    "${model_args[@]}"
fi

run_eval() {
  local dataset="$1"
  local mode="$2"
  local position_mode="$3"
  local max_soft_terms="$4"
  local exclude_original="$5"
  local out_dir="${OUTPUT_ROOT}/eval/${mode}/${dataset}"
  local novelty_args=()
  if [[ "${exclude_original}" == "1" ]]; then
    novelty_args+=(--exclude_original_terms)
  fi

  run_command "${PYTHON_BIN}" \
    "${REPO_ROOT}/src/sde/single_index/evaluate.py" \
    --cache "${CACHE_ROOT}/${dataset}.sde_single_index_cache.pkl" \
    --data_root "${BEIR_DATA_ROOT}" \
    --query_split "${QUERY_SPLIT}" \
    --out_dir "${out_dir}" \
    --mode "${mode}" \
    --position_mode "${position_mode}" \
    --max_soft_terms "${max_soft_terms}" \
    "${CONFIG_RETRIEVAL_ARGS[@]}" \
    --overwrite \
    --cleanup_index \
    "${novelty_args[@]}"
}

for index in "${!CONFIG_DATASETS[@]}"; do
  dataset="${CONFIG_DATASETS[${index}]}"
  position_mode="${CONFIG_POSITION_MODES[${index}]}"
  max_soft_terms="${CONFIG_MAX_SOFT_TERMS[${index}]}"
  exclude_original="${CONFIG_EXCLUDE_ORIGINAL[${index}]}"

  if [[ "${RUN_HARD_CONTROL}" == "1" ]]; then
    run_eval \
      "${dataset}" \
      hard \
      "${position_mode}" \
      "${max_soft_terms}" \
      "${exclude_original}"
  fi
  run_eval \
    "${dataset}" \
    sde \
    "${position_mode}" \
    "${max_soft_terms}" \
    "${exclude_original}"

  if [[ "${RUN_HARD_CONTROL}" == "1" ]]; then
    run_command "${PYTHON_BIN}" \
      "${REPO_ROOT}/src/sde/single_index/paired_significance.py" \
      --qrels \
      "${BEIR_DATA_ROOT}/${dataset}/${QUERY_SPLIT}/qrels.${QUERY_SPLIT}.tsv" \
      --run_a \
      "${OUTPUT_ROOT}/eval/sde/${dataset}/sde_single_index.run" \
      --run_b \
      "${OUTPUT_ROOT}/eval/hard/${dataset}/sde_single_index.run" \
      --label_a sde_single_index \
      --label_b hard_single_index \
      --out \
      "${OUTPUT_ROOT}/eval/sde/${dataset}/paired_vs_hard.json" \
      --bootstrap_reps 10000 \
      --randomization_reps 10000 \
      --seed 20260713
  fi
done

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "Single-index SDE dry run complete: config=${CONFIG_PATH}"
else
  echo "Single-index SDE evaluation complete: ${OUTPUT_ROOT}"
fi
