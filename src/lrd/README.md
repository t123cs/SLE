# LRD: LSR Route Diagnostic

This directory contains the lightweight diagnostic code used in Section 4 for four-route sparse matching decomposition. LRD is not presented here as a new training method or a full retriever framework. The public surface is limited to the analysis core and the small amount of scaffolding needed to run it on a trained sparse model.

Dependency:

- Clone the upstream `scaling-retriever` repository and set
  `SCALING_RETRIEVER_ROOT=/path/to/scaling-retriever`.
- Install its environment according to the upstream repository before running
  the scripts here.

Primary files:

- `route_decomposition.py`: the core `LL / LE / EL / EE` decomposition and
  weighted route scoring logic.
- `build_route_doc_cache.py`: builds the document-side `literal` and
  `expansion` sparse cache from a trained sparse model.
- `run_route_diagnostic.py`: runs the route decomposition on a fixed candidate
  set and writes per-document route scores.
- `doc_cache.py`: cache reader/writer utilities for document-side literal and
  expansion representations.
- `sparse_runtime.py`: thin model-loading and batching helpers for the
  diagnostic scripts on top of the upstream `scaling_retriever` package.
- `upstream_runtime.py`: resolves the external `scaling_retriever` dependency
  through `SCALING_RETRIEVER_ROOT`.
