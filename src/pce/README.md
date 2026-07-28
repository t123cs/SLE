# PCE: Practical Cost Evaluation

This directory contains the Practical Cost Evaluation (PCE) workflows used to
measure SLE/SDE and Doc2Query++ generation, retrieval, and index-lifecycle
costs.

## Generation-Cost Pipeline

The original numbered stages remain available:

```text
00_record_environment.py
01_make_sample.py
02_run_d2qpp_full.py
03_run_d2qpp_qgen_only.py
04_run_sde_trace.py
04b_run_sde_decoded6.py
05_aggregate_costs.py
06_run_retrieval_system_costs.py
07_run_lucene_segmented_costs.py
08_make_rq3_tables.py
```

Use `examples/run_pce_smoke.sh` as the small generation-cost entry point. The
Doc2Query++ code approximates its efficiency path for cost comparison;
retrieval effectiveness remains associated with the corresponding original
implementation.

## Full-Collection System Efficiency

`system_efficiency/` measures the online-serving and index-lifecycle results
for:

- BM25;
- Doc2Query++ Full;
- Doc2Query++ QGen-only;
- SDE with its auxiliary index, served sequentially;
- SDE with its auxiliary index, served in parallel.

The files are:

- `system_efficiency/measure_system_efficiency_formal.py`: prepares data,
  builds the BM25 and Doc2Query++ indexes, and measures anchor latency,
  throughput, footprint, write volume, and memory.
- `system_efficiency/d2q_queryline.py`: prepares deterministic Doc2Query++
  query-line simulations from stored traces.
- `system_efficiency/measure_factorized_qdoc_lifecycle.py`: measures SDE
  auxiliary-index construction and complete out-of-place refresh.
- `system_efficiency/benchmark_factorized_querydoc_formal.py`: measures
  sequential and process-parallel SDE serving.
- `system_efficiency/run_efficiency.sh`: runs the workflow in stages.

The SDE evaluator and factorized auxiliary-index implementation live under
`src/sde/` and are shared by effectiveness and efficiency evaluation.

## Environment

The formal system measurements ran on two Intel Xeon Platinum 8375C sockets at
2.90 GHz, with 32 physical cores per socket, 64 physical cores, 128 logical
CPUs, 503 GiB of RAM, and no swap. The software environment used Ubuntu
22.04.5 LTS, Linux 5.15.0-78-generic, Python 3.12.3, PyTerrier 1.0.4, Terrier
5.11 with helper 0.0.8, OpenJDK 21.0.11, NumPy 2.2.6, Pandas 2.3.3, psutil
7.0.0, and Transformers 4.56.2.

Install the Python dependencies in an isolated environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r src/pce/system_efficiency/requirements.txt
```

Java 21 must be available on `PATH`. PyTerrier resolves Terrier artifacts on
first initialization when they are absent from its local cache.

## Inputs

The default repository layout is:

```text
data/
  beir/<dataset>/test/{collection.tsv,queries.jsonl,qrels.test.tsv}
  traces/<dataset>/train_data_multitemplate_unfiltered.jsonl
models/tokenizer/
```

`<dataset>` is one of `nfcorpus`, `scidocs`, `fiqa-2018`, `arguana`, and
`scifact`. The workflow consumes existing six-template SDE traces and performs
no new LLM generation.

Set explicit roots when the data follows another layout:

```bash
export DATA_ROOT=/path/to/beir
export TRACE_ROOT=/path/to/sde_traces
export MODEL_PATH=/path/to/local/tokenizer
```

## Running the System Benchmark

Run from the repository root:

```bash
bash src/pce/system_efficiency/run_efficiency.sh all
```

Each stage can also run independently:

```bash
bash src/pce/system_efficiency/run_efficiency.sh baselines
bash src/pce/system_efficiency/run_efficiency.sh prepare-factorized
bash src/pce/system_efficiency/run_efficiency.sh online-index
bash src/pce/system_efficiency/run_efficiency.sh lifecycle
bash src/pce/system_efficiency/run_efficiency.sh online
```

Completed artifacts are reused unless overwrite is explicitly requested from
the corresponding Python entry point.

## Measurement Protocol

The default full-collection protocol uses:

- all judged test queries from all five collections;
- 50 warm-up queries per collection;
- 3 complete-query measured repeats;
- 1 fixed physical core for sequential latency;
- 2 fixed physical cores for parallel latency;
- 2 physical cores for every throughput measurement;
- up to 8 fixed physical cores for each index-build task;
- 3 index-construction repeats;
- a warm serving state with the OS page cache retained after query warm-up.

Mean latency, p95 latency, and two-core QPS are unweighted averages of the five
collection-level measurements. Index footprint is the sum of the five
collection-specific indexes.

The controlled system-efficiency configuration is:

```text
original retrieval depth       300
auxiliary qdoc retrieval depth 1000
qdoc aggregation               sum_decay_0.3
score normalization            minmax
fusion alpha                   0.5
original score weight          0.5
auxiliary score weight         0.5
original/qdoc BM25 k1          0.9
original/qdoc BM25 b           0.4
```

Sequential serving executes the original route, factorized auxiliary route,
aggregation, normalization, and fusion in one process. Parallel serving runs
the two retrieval routes in separate processes pinned to two physical cores
and performs aggregation and fusion in the coordinator.

## Index Lifecycle

`Full refresh` is a complete out-of-place refresh of the expansion data. SDE
retains the original BM25 index and builds a new factorized auxiliary-index
version while the deployed version remains available.

`Initial write MB` records writes from initial construction. Build and refresh
times and write volumes are summed across collections. `Peak RSS MB` is the
maximum peak of a single collection-specific task. Index sizes use 2^20 bytes
per reported MB. The structure-aware auxiliary refresh uses one factorization
process.

Principal outputs are:

```text
results/system_efficiency_formal/system_efficiency_main_table.tsv
results/system_efficiency_formal/report_index_lifecycle.tsv
results/factorized_qdoc_online/table5_factorized_qdoc.tsv
results/factorized_qdoc_lifecycle/table6_factorized_qdoc.tsv
results/factorized_qdoc_online/validation.json
```
