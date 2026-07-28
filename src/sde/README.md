# SDE: Soft Document Expansion

Soft Document Expansion (SDE) keeps the sparse retriever fixed and supplies
trajectory-derived lexical evidence through a factorized query-document
auxiliary index. Retrieval uses the original BM25 index and the SDE auxiliary
index, then combines their normalized source-document scores.

## Files

- `prepare_expectation_data_multitemplate_unfiltered.py` generates
  multi-template synthetic-query traces with token IDs, probabilities, prompt
  metadata, and generation metadata.
- `eval_bm25_dual_index_fusion.py` builds or loads both retrieval routes,
  evaluates judged queries, and writes metrics and a TREC run.
- `terrier_utils.py` provides shared PyTerrier indexing, retrieval, data
  loading, metric, and run-writing utilities.
- `factorized_qdoc/sde_qdoc_terms.py` converts stored trajectories into
  weighted document-level soft terms.
- `factorized_qdoc/prepare_factorized_qdoc_components.py` groups shared soft
  terms and query-specific terms by source document.
- `factorized_qdoc/build_factorized_qdoc_components.py` builds the factorized
  auxiliary index.
- `factorized_qdoc/factorized_qdoc_index.py` loads and scores that index with
  BM25.

## Factorized Auxiliary Index

For source document `d` and synthetic query document `j`, the logical term
multiset is:

```text
L[d,j] = document_soft_terms[d] + synthetic_query_terms[d,j]
```

Terms shared by every synthetic query document of the same source are stored
once as source-level postings. Query-specific residual terms are stored as
query-document postings. The index also stores the source-to-query-document
mapping, logical document lengths, document frequencies, and collection
statistics required by BM25.

At query time, shared and residual term frequencies reconstruct `L[d,j]`
before scoring. The representation therefore preserves the logical
query-document term frequencies and BM25 statistics while reducing repeated
postings for source-shared terms.

## SDE Configuration

The evaluator uses:

```text
trace samples                 sample_idx=0 from each of six templates
query-document text           normalized query terms plus document soft terms
candidate positions           all decoding positions
candidate depth               top 5 per decoding step
probability threshold         0.01
soft-term budget              256 unique terms per source document
term weighting                repeat_by_score, scale 3, at most 3 repeats
original retrieval depth      300
auxiliary retrieval depth     1000
query-document aggregation    sum_decay_0.3
route normalization           minmax
fusion alpha                  0.5; SciFact effectiveness uses 0.25
original/auxiliary BM25       k1=0.9, b=0.4
```

Effectiveness evaluation uses alpha 0.25 for SciFact and alpha 0.5 for
NFCorpus, SCIDOCS, FiQA-2018, and ArguAna. The controlled system-efficiency
benchmark uses alpha 0.5 for every collection.

## Inputs

The evaluator consumes:

```text
data/beir/<dataset>/test/collection.tsv
data/beir/<dataset>/test/queries.jsonl
data/beir/<dataset>/test/qrels.test.tsv
data/traces/<dataset>/train_data_multitemplate_unfiltered.jsonl
models/tokenizer/
```

The tokenizer decodes candidate token IDs already present in the traces.

## Evaluation

Run from the repository root. SciFact uses alpha 0.25:

```bash
python src/sde/eval_bm25_dual_index_fusion.py \
  --corpus_tsv data/beir/scifact/test/collection.tsv \
  --queries_json data/beir/scifact/test/queries.jsonl \
  --qrels_tsv data/beir/scifact/test/qrels.test.tsv \
  --expansion_jsonl data/traces/scifact/train_data_multitemplate_unfiltered.jsonl \
  --model_path models/tokenizer \
  --fusion_alpha 0.25 \
  --out_dir results/scifact_sde
```

`--doc_index_dir`, `--component_path`, `--component_stats_path`, and
`--factor_index_dir` reuse existing artifacts. The evaluator writes:

```text
dual_index_fusion_results.json
dual_index_fusion_summary.tsv
dual_index_fusion_best.run
```

Full-collection online-serving and index-lifecycle measurements are documented
in `src/pce/README.md`.
