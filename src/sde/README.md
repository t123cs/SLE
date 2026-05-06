# SDE: Soft Document Expansion

Soft Document Expansion (SDE) keeps the sparse retriever fixed and changes the
document-side lexical evidence supplied to BM25. The public path here generates
synthetic query traces, extracts compact lexical evidence from token-level
distributions, and evaluates a dual-index BM25 fusion setup.

Primary files:

- `prepare_expectation_data_multitemplate_unfiltered.py`: generates
  multi-template synthetic query traces. The output is the full trace form:
  each row stores `query_text`, top-k token ids/probabilities, prompt metadata,
  and generation metadata. For downstream use, it is enough to persist the
  extracted expansion evidence rather than the full trace.
- `eval_bm25_dual_index_fusion.py`: accumulates token-level probability mass
  into document-side lexical scores, uses `query_text` only as a selective
  full-word recovery signal, renders one compact expansion document per
  original document, then runs BM25 over the original corpus and the expansion
  corpus before fusing the scores. The default full-word recovery mode is
  `generated_query_term_mode=supported`, and the default fusion weight is
  `fusion_alpha=0.5`.

Use the templates under `examples/` for the recommended public entry points.
