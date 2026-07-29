#!/usr/bin/env python3
"""Paired per-query comparison for two TREC runs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


METRICS = ("ndcg_at_10", "mrr_at_10", "map", "recall_at_100")


def load_qrels(path: Path) -> dict[str, dict[str, int]]:
    qrels: dict[str, dict[str, int]] = defaultdict(dict)
    with path.open(encoding="utf-8") as source:
        for line_number, raw in enumerate(source):
            fields = raw.rstrip("\n").split("\t")
            if (
                line_number == 0
                and fields
                and fields[0].lower() in {"query-id", "qid"}
            ):
                continue
            if len(fields) >= 4:
                query_id, document_id, relevance = (
                    fields[0],
                    fields[2],
                    fields[3],
                )
            elif len(fields) >= 3:
                query_id, document_id, relevance = fields[:3]
            else:
                continue
            qrels[str(query_id)][str(document_id)] = int(relevance)
    return dict(qrels)


def load_run(path: Path) -> dict[str, list[str]]:
    rows: dict[str, list[tuple[int, str]]] = defaultdict(list)
    with path.open(encoding="utf-8") as source:
        for raw in source:
            fields = raw.split()
            if len(fields) < 6:
                continue
            query_id = str(fields[0])
            document_id = str(fields[2])
            rank = int(fields[3])
            rows[query_id].append((rank, document_id))
    return {
        query_id: [document_id for _, document_id in sorted(items)]
        for query_id, items in rows.items()
    }


def metrics_for_query(
    ranked_documents: list[str],
    relevance: dict[str, int],
) -> dict[str, float]:
    gains = [relevance.get(document_id, 0) for document_id in ranked_documents]
    dcg = sum(
        (2**gain - 1) / math.log2(rank + 1)
        for rank, gain in enumerate(gains[:10], start=1)
        if gain > 0
    )
    ideal = sorted(
        (gain for gain in relevance.values() if gain > 0),
        reverse=True,
    )[:10]
    idcg = sum(
        (2**gain - 1) / math.log2(rank + 1)
        for rank, gain in enumerate(ideal, start=1)
    )
    first_relevant = next(
        (
            rank
            for rank, gain in enumerate(gains[:10], start=1)
            if gain > 0
        ),
        None,
    )
    total_relevant = sum(gain > 0 for gain in relevance.values())
    hit_count = 0
    precision_sum = 0.0
    for rank, gain in enumerate(gains, start=1):
        if gain <= 0:
            continue
        hit_count += 1
        precision_sum += hit_count / rank
    return {
        "ndcg_at_10": dcg / idcg if idcg else 0.0,
        "mrr_at_10": 1.0 / first_relevant if first_relevant else 0.0,
        "map": precision_sum / total_relevant if total_relevant else 0.0,
        "recall_at_100": (
            sum(gain > 0 for gain in gains[:100]) / total_relevant
            if total_relevant
            else 0.0
        ),
    }


def bootstrap_interval(
    deltas: np.ndarray,
    random: np.random.Generator,
    repetitions: int,
) -> list[float]:
    means = np.empty(repetitions, dtype=np.float64)
    for start in range(0, repetitions, 1000):
        batch_size = min(1000, repetitions - start)
        samples = random.integers(
            0,
            len(deltas),
            size=(batch_size, len(deltas)),
        )
        means[start : start + batch_size] = deltas[samples].mean(axis=1)
    return [float(value) for value in np.quantile(means, [0.025, 0.975])]


def sign_flip_pvalue(
    deltas: np.ndarray,
    random: np.random.Generator,
    repetitions: int,
) -> float:
    observed = abs(float(deltas.mean()))
    exceedances = 0
    for start in range(0, repetitions, 1000):
        batch_size = min(1000, repetitions - start)
        signs = (
            random.integers(
                0,
                2,
                size=(batch_size, len(deltas)),
                dtype=np.int8,
            )
            * 2
            - 1
        )
        permuted = np.abs((signs * deltas).mean(axis=1))
        exceedances += int(
            np.count_nonzero(permuted >= observed - 1e-15)
        )
    return (exceedances + 1) / (repetitions + 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qrels", type=Path, required=True)
    parser.add_argument("--run_a", type=Path, required=True)
    parser.add_argument("--run_b", type=Path, required=True)
    parser.add_argument("--label_a", default="A")
    parser.add_argument("--label_b", default="B")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--per_query_out", type=Path)
    parser.add_argument("--bootstrap_reps", type=int, default=10000)
    parser.add_argument("--randomization_reps", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260713)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    qrels = load_qrels(args.qrels)
    runs = {
        args.label_a: load_run(args.run_a),
        args.label_b: load_run(args.run_b),
    }
    query_ids = sorted(qrels)
    per_query = {
        label: [
            metrics_for_query(run.get(query_id, []), qrels[query_id])
            for query_id in query_ids
        ]
        for label, run in runs.items()
    }
    random = np.random.default_rng(args.seed)
    result = {
        "labels": {"a": args.label_a, "b": args.label_b},
        "queries": len(query_ids),
        "seed": args.seed,
        "bootstrap_reps": args.bootstrap_reps,
        "randomization_reps": args.randomization_reps,
        "metrics": {},
    }
    for metric in METRICS:
        values_a = np.array(
            [row[metric] for row in per_query[args.label_a]],
            dtype=np.float64,
        )
        values_b = np.array(
            [row[metric] for row in per_query[args.label_b]],
            dtype=np.float64,
        )
        deltas = values_a - values_b
        result["metrics"][metric] = {
            "mean_a": float(values_a.mean()),
            "mean_b": float(values_b.mean()),
            "mean_delta_a_minus_b": float(deltas.mean()),
            "wins_a": int(np.count_nonzero(deltas > 1e-12)),
            "wins_b": int(np.count_nonzero(deltas < -1e-12)),
            "ties": int(np.count_nonzero(np.abs(deltas) <= 1e-12)),
            "paired_bootstrap_95_ci": bootstrap_interval(
                deltas,
                random,
                args.bootstrap_reps,
            ),
            "sign_flip_two_sided_p": sign_flip_pvalue(
                deltas,
                random,
                args.randomization_reps,
            ),
        }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )
    if args.per_query_out:
        args.per_query_out.parent.mkdir(parents=True, exist_ok=True)
        rows = []
        for index, query_id in enumerate(query_ids):
            row: dict[str, str | float] = {"qid": query_id}
            for metric in METRICS:
                value_a = per_query[args.label_a][index][metric]
                value_b = per_query[args.label_b][index][metric]
                row[f"{args.label_a}_{metric}"] = value_a
                row[f"{args.label_b}_{metric}"] = value_b
                row[f"delta_{metric}"] = value_a - value_b
            rows.append(row)
        with args.per_query_out.open(
            "w",
            encoding="utf-8",
            newline="",
        ) as destination:
            writer = csv.DictWriter(
                destination,
                fieldnames=rows[0].keys(),
                delimiter="\t",
            )
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

