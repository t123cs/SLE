# PCE: Practical Cost Evaluation

This directory contains the Python stages for Practical Cost Evaluation (PCE):
the SLE/SDE vs Doc2Query++ cost pipeline.

The D2Q++ code provided here is only intended to approximate the efficiency path of the original method for cost comparison; it does not guarantee reproduction of the original retrieval effectiveness.

Pipeline stages:

- `00_record_environment.py`
- `01_make_sample.py`
- `02_run_d2qpp_full.py`
- `03_run_d2qpp_qgen_only.py`
- `04_run_sde_trace.py`
- `04b_run_sde_decoded6.py`
- `05_aggregate_costs.py`
- `06_run_retrieval_system_costs.py`
- `07_run_lucene_segmented_costs.py`
- `08_make_rq3_tables.py`

Use `examples/run_pce_smoke.sh` as the small public entry point.
