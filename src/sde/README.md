# SDE: Soft Document Expansion

Soft Document Expansion (SDE) converts generation trajectories into lexical
evidence for sparse retrieval. This release provides two index deployments:

- **SDE, auxiliary index** keeps the original BM25 index and stores SDE evidence
  in a factorized query-document auxiliary index.
- **SDE, single index** appends generated queries and sparse trajectory terms to
  each source document before building one BM25 index.

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
- `single_index/prepare_cache.py` extracts the six generated queries and
  trajectory evidence used by single-index SDE.
- `single_index/evaluate.py` builds and evaluates single-index SDE and its hard
  query control.
- `single_index/config.py` validates the released BEIR-5 configuration.
- `single_index/paired_significance.py` performs paired bootstrap and sign-flip
  comparisons between two TREC runs.

## Factorized Auxiliary Index

For source document `d` and synthetic query document `j`, the logical term
multiset is:

```text
L[d,j] = document_soft_terms[d] + synthetic_query_terms[d,j]
```

Terms shared by every synthetic query document of the same source are stored
once as source-level postings. Query-specific terms are stored as
query-document postings. The index also stores the source-to-query-document
mapping, logical document lengths, document frequencies, and collection
statistics required by BM25.

At query time, shared and query-specific term frequencies reconstruct `L[d,j]`
before scoring. The representation therefore preserves the logical
query-document term frequencies and BM25 statistics while reducing repeated
postings for source-shared terms.

### Auxiliary-Index Configuration

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

## Single-Index SDE

For each source document, single-index SDE retains one generated query from
each of six templates and selects a sparse set of trajectory-derived soft
terms:

```text
d_single = d + q_1 + ... + q_6 + t_1 + ... + t_m
```

The included trace generator defaults to one randomly sampled query per
template, for six generated queries per source document.

The six query strings are retained in full after whitespace and outer-format
cleanup. Each selected soft term is appended exactly once. Terms already
present in the generated queries are excluded from the soft set. The SciFact
configuration also excludes terms present in the source document.

At each decoding position, the selector reads the top five stored candidates,
excludes rank 1, and excludes the chosen candidate at any rank. Soft evidence
therefore comes from the non-chosen candidates among ranks 2 through 5.
Candidate events require probability at least `0.001`.

Corpus text, generated queries, and candidates use Terrier English
tokenization, stopword removal, and Porter stemming. Candidate terms require a
minimum length of 2; the boundary-span lexical check uses a minimum length of
3.

For candidate term `t` with retained event probabilities `p_i`, the
document-level probability is:

```text
P_d(t) = 1 - product_i(1 - p_i)
```

Candidate document frequency is limited to 5% of the collection. The ranking
score uses combined document frequency:

```text
df_combined(t) = min(N, df_base(t) + df_candidate(t))
IDF(t) = log(1 + (N - df_combined(t) + 0.5) / (df_combined(t) + 0.5))
score_d(t) = P_d(t) * IDF(t)
```

Terms with `P_d(t) < 0.01` are removed. The remaining terms are sorted by
`score_d(t)` and truncated to the dataset-specific budget.

The `boundary` position mode reconstructs lexical spans from the rank-1
trajectory pieces. It keeps single-token lexical spans and the first position
of each multi-token lexical span.

### Released BEIR-5 Configuration

| Dataset | Selection source | Positions | Maximum soft terms | Exclude source terms |
|---|---|---|---:|---|
| NFCorpus | NFCorpus train | all | 16 | no |
| SCIDOCS | FiQA-2018 train transfer | all | 16 | no |
| FiQA-2018 | FiQA-2018 train | all | 16 | no |
| ArguAna | NFCorpus train transfer | all | 16 | no |
| SciFact | SciFact train | boundary | 4 | yes |

All single-index runs use BM25 with `k1=0.9`, `b=0.4`, and retrieval depth
1000. The complete executable configuration is stored in
`configs/sde/sde_single_index_beir5.json`.

NFCorpus, FiQA-2018, and SciFact parameters were selected on their train
queries using:

```text
delta_nDCG@10_vs_hard - 0.5 * bootstrap_standard_error
```

SCIDOCS receives the FiQA-2018 configuration, and ArguAna receives the
NFCorpus configuration. Test queries are used only for final evaluation.

## Inputs

The evaluator consumes:

```text
data/beir/<dataset>/test/collection.tsv
data/beir/<dataset>/test/queries.jsonl
data/beir/<dataset>/test/qrels.test.tsv
data/traces/<dataset>/train_data_multitemplate_unfiltered.jsonl
models/tokenizer/
```

Single-index SDE accepts decoded candidate objects or token-ID candidates.
Token-ID traces use the generation tokenizer supplied through `MODEL_PATH`.
Decoded candidate traces require no tokenizer. Decoded candidate objects carry
a boolean `chosen` field; token-ID traces carry position-aligned
`generated_token_ids`. The included trace generator writes the latter field.

## Evaluation

Run the auxiliary-index evaluator from the repository root. SciFact uses alpha
0.25:

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

Run the released single-index configuration across the five collections:

```bash
BEIR_DATA_ROOT=data/beir \
TRACE_ROOT=data/traces \
MODEL_PATH=models/tokenizer \
bash examples/run_sde_single_index_beir5.sh
```

`MODEL_PATH` may be omitted for decoded candidate traces. Set
`RUN_HARD_CONTROL=0` to skip the hard query control, `BUILD_CACHE=0` to reuse
prepared caches, or `QUERY_SPLIT=train` for train-query evaluation. Inspect the
resolved commands without creating outputs:

```bash
BEIR_DATA_ROOT=data/beir \
TRACE_ROOT=data/traces \
bash examples/run_sde_single_index_beir5.sh --dry-run
```

Each single-index run writes:

```text
sde_single_index_results.json
sde_single_index_summary.tsv
sde_single_index.run
```

Full-collection online-serving and index-lifecycle measurements are documented
in `src/pce/README.md`.
