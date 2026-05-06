# SLE

**SLE** (Soft Lexical Expansion) is a research codebase for soft lexical
expansion and the small diagnostics/evaluation utilities used around it.
It contains three components:

- **SDE** (Soft Document Expansion): document-side synthetic query trace
  generation, compact lexical evidence construction, and dual-index BM25
  fusion.
- **LRD** (LSR Route Diagnostic): four-route diagnostic decomposition for
  trained learned sparse retrievers.
- **PCE** (Practical Cost Evaluation): scripts for measuring practical
  generation, storage, indexing, and query-time costs.

## Repository Layout

```text
.
├── src/
│   ├── sde/      # Soft Document Expansion generation, traces, and BM25 fusion
│   ├── lrd/      # LSR Route Diagnostic code
│   └── pce/      # Practical Cost Evaluation pipeline
├── configs/      # Reusable config templates
├── examples/     # Minimal runnable examples
├── requirements/ # Component-specific dependency lists
└── docs/         # Environment and data notes
```

## Code Included

- `src/sde/`: the SDE trace generator and the compact document-side expansion
  evaluator. The evaluator builds one expansion document per original document
  and fuses it with the original BM25 index; its default fusion weight is
  `fusion_alpha=0.5`.
- `src/lrd/`: the LRD route decomposition core plus minimal runtime wrappers
  for a trained sparse retriever.
- `src/pce/`: the PCE sampling, generation-cost, storage-cost, indexing-cost,
  and aggregation stages.
- `examples/`: small path templates for running the public entry points.

## Not Included

- Large BEIR/TREC prepared datasets.
- Generated hypotheses or decoding traces from full experiments.
- Server-specific batch runners.
- Model checkpoints.

## Quick Start

Install only the dependencies for the component you plan to run:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements/sde.txt
pip install -r requirements/lrd.txt
pip install -r requirements/pce.txt
```

Some workflows need Java/PyTerrier, vLLM, Faiss, BERTopic, KeyBERT, local
model directories, and prepared BEIR/MS MARCO data.
See [docs/environment.md](docs/environment.md) and [docs/data.md](docs/data.md).

LRD also depends on the upstream `scaling-retriever` repository. Clone it
separately and set `SCALING_RETRIEVER_ROOT=/path/to/scaling-retriever` before
running the LRD scripts under `src/lrd/`.

## Examples

SDE generation:

```bash
bash examples/run_sde_generation.sh
```

SDE dual-index evaluation:

```bash
bash examples/run_sde_dual_index_eval.sh
```

PCE smoke pipeline:

```bash
bash examples/run_pce_smoke.sh
```

Each example is a template: set the environment variables at the top or export
them before running.

## License

This repository is released under the MIT License. See [LICENSE](LICENSE).
