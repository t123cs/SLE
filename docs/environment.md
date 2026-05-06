# Environment

The code was developed in local research environments, but public use should be
configured with explicit paths.

## Python

Install component dependencies as needed:

```bash
pip install -r requirements/sde.txt
pip install -r requirements/lrd.txt
pip install -r requirements/pce.txt
```

For LRD, also clone the upstream `scaling-retriever` repository and set
`SCALING_RETRIEVER_ROOT` to its local path before running the scripts under
`src/lrd/`.

## System Dependencies

- Java runtime for PyTerrier-based indexing and BM25 evaluation.
- CUDA and a GPU for local LLM generation.
- vLLM for server-backed generation in PCE.
- Optional BERTopic/KeyBERT stack for the full Doc2Query++ benchmark path.

## Common Variables

```bash
export PYTHON_BIN=python
export BEIR_DATA_ROOT=/path/to/prepared/beir
export MSMARCO_ROOT=/path/to/msmarco-full
export LLAMA31_8B_INSTRUCT=/path/to/Llama-3.1-8B-Instruct
export D2QPP_EMBEDDING_MODEL_PATH=/path/to/all-MiniLM-L6-v2
export VLLM_BASE_URL=http://127.0.0.1:8000/v1
export SCALING_RETRIEVER_ROOT=/path/to/scaling-retriever
```
