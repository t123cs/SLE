# Data

This repository does not ship datasets. The experiments use two kinds of data: public retrieval benchmark data and generated derivative data.

## Public Benchmark Data

SDE and PCE use BEIR-style retrieval datasets. The main document-expansion experiments use:

- `nfcorpus`
- `scidocs`
- `fiqa-2018`
- `arguana`
- `scifact`

These datasets come from the BEIR benchmark. See the
[BEIR paper](https://openreview.net/forum?id=wCu6T5xFjeJ) and
[BEIR dataset list](https://github.com/beir-cellar/beir/wiki/Datasets-available).

LRD uses MS MARCO and TREC Deep Learning evaluation data:

- MS MARCO document/passage ranking data from the
  [official MS MARCO ranking datasets page](https://microsoft.github.io/msmarco/Datasets.html).
- TREC DL 2019/2020 topics and qrels from NIST:
  [2019](https://trec.nist.gov/data/deep2019.html),
  [2020](https://trec.nist.gov/data/deep2020.html).

## Generated Derivative Data

Generated hypotheses, synthetic queries, decoding traces, benchmark samples, raw generation logs, indexes, and result tables are derived from the public benchmark data plus local model inference. 
They are not included in the repository.

The main SDE generator is:

```text
src/sde/prepare_expectation_data_multitemplate_unfiltered.py
```

This script writes the full trace version of the generated data. 
For practical use, it is sufficient to persist only the extracted expansion evidence used by the downstream retrieval/indexing pipeline.

The PCE pipeline writes generated benchmark artifacts under a local output directory such as `results/pce_smoke/`.

## Expected Local Layout

For SDE and PCE, use BEIR-style prepared files:

```text
<BEIR_DATA_ROOT>/<dataset>/<split>/collection.tsv
<BEIR_DATA_ROOT>/<dataset>/<split>/queries.jsonl
<BEIR_DATA_ROOT>/<dataset>/<split>/qrels.<split>.tsv
```

The PCE sampler also accepts:

```text
<BEIR_DATA_ROOT>/<dataset>/collection.tsv
<BEIR_DATA_ROOT>/<dataset>/corpus.jsonl
```

For LRD, keep MS MARCO and TREC DL files outside the repository and pass their
locations through local config files or environment variables such as
`MSMARCO_ROOT`.
