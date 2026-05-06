#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from common import (
    DATASET_SPECS,
    SAMPLE_SEED,
    append_jsonl,
    deterministic_sample,
    ensure_dir,
    text_len_words,
    utf8_len,
    write_json,
)


def candidate_dataset_dirs(root: Path, slug: str, aliases: Iterable[str]) -> List[Path]:
    names = [slug]
    for alias in aliases:
        if alias not in names:
            names.append(alias)
    if slug == "fiqa-2018":
        names.extend(["fiqa", "mteb___fiqa"])
    dirs = []
    for name in names:
        candidate = root / name
        if candidate not in dirs:
            dirs.append(candidate)
    return dirs


def find_collection_file(root: Path, slug: str, split: str, aliases: Iterable[str]) -> Tuple[Path, str]:
    checked: List[str] = []
    for ds_dir in candidate_dataset_dirs(root, slug, aliases):
        candidates = [
            ds_dir / split / "collection.tsv",
            ds_dir / "collection.tsv",
            ds_dir / "corpus.jsonl",
            ds_dir / "corpus.json",
        ]
        for path in candidates:
            checked.append(str(path))
            if path.is_file():
                if path.name == "collection.tsv":
                    return path, "collection_tsv"
                return path, "beir_corpus_jsonl"
    raise FileNotFoundError(
        f"Could not find collection for dataset={slug} under {root}. Checked:\n"
        + "\n".join(f"  - {path}" for path in checked)
    )


def load_collection_tsv(path: Path, dataset: str) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    with path.open("r", encoding="utf-8") as fin:
        for line in fin:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t", 1)
            if len(parts) < 2:
                continue
            doc_id, text = parts[0], parts[1]
            rows.append(
                {
                    "dataset": dataset,
                    "doc_id": str(doc_id),
                    "text": str(text),
                    "text_len_chars": len(str(text)),
                    "text_len_words": text_len_words(str(text)),
                }
            )
    return rows


def load_beir_corpus(path: Path, dataset: str) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    with path.open("r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            doc_id = str(obj.get("_id") or obj.get("id") or obj.get("doc_id") or "")
            if not doc_id:
                continue
            title = str(obj.get("title") or "").strip()
            body = str(obj.get("text") or obj.get("contents") or "").strip()
            text = f"{title}\n{body}".strip() if title else body
            rows.append(
                {
                    "dataset": dataset,
                    "doc_id": doc_id,
                    "text": text,
                    "text_len_chars": len(text),
                    "text_len_words": text_len_words(text),
                }
            )
    return rows


def material_difference(actual: int, expected: int, tolerance: float) -> bool:
    if actual == expected:
        return False
    return abs(actual - expected) > max(1, int(round(expected * tolerance)))


def round_robin_rows(groups: List[List[Dict[str, object]]]) -> Iterable[Dict[str, object]]:
    max_len = max((len(group) for group in groups), default=0)
    for offset in range(max_len):
        for group in groups:
            if offset < len(group):
                yield group[offset]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the stratified 10K PCE sample.")
    parser.add_argument("--beir_data_root", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--seed", type=int, default=SAMPLE_SEED)
    parser.add_argument(
        "--count_tolerance",
        type=float,
        default=0.0,
        help="Relative corpus-size tolerance. Default is strict exact count matching.",
    )
    args = parser.parse_args()

    beir_root = Path(args.beir_data_root)
    output_root = Path(args.output_root)
    sample_path = output_root / "sample_docs.jsonl"
    manifest_path = output_root / "sample_manifest.json"
    ensure_dir(output_root)
    sample_path.write_text("", encoding="utf-8")

    manifest = {
        "seed": args.seed,
        "beir_data_root": str(beir_root),
        "sample_path": str(sample_path),
        "datasets": [],
    }
    seen = set()
    total = 0
    total_utf8_text_bytes = 0
    sampled_groups: List[List[Dict[str, object]]] = []

    for spec in DATASET_SPECS:
        collection_path, loader_kind = find_collection_file(beir_root, spec.slug, spec.split, spec.aliases)
        if loader_kind == "collection_tsv":
            rows = load_collection_tsv(collection_path, spec.slug)
        else:
            rows = load_beir_corpus(collection_path, spec.slug)

        actual = len(rows)
        if material_difference(actual, spec.expected_docs, args.count_tolerance):
            raise SystemExit(
                f"[ERROR] dataset={spec.slug} corpus size mismatch: "
                f"expected={spec.expected_docs} actual={actual} path={collection_path}"
            )

        sampled = deterministic_sample(rows, spec.sample_docs, args.seed, spec.slug)
        for row in sampled:
            key = (row["dataset"], row["doc_id"])
            if key in seen:
                raise SystemExit(f"[ERROR] duplicate sampled doc: {key}")
            seen.add(key)
            total_utf8_text_bytes += utf8_len(str(row["text"]))
        sampled_groups.append(sampled)
        total += len(sampled)
        manifest["datasets"].append(
            {
                "dataset": spec.slug,
                "display_name": spec.display_name,
                "collection_path": str(collection_path),
                "loader_kind": loader_kind,
                "expected_docs": spec.expected_docs,
                "actual_docs": actual,
                "sample_docs": len(sampled),
            }
        )
        print(
            f"[sample] dataset={spec.slug} actual={actual} quota={len(sampled)} "
            f"path={collection_path}"
        )

    if total != 10000:
        raise SystemExit(f"[ERROR] total sampled rows mismatch: expected=10000 actual={total}")
    for row in round_robin_rows(sampled_groups):
        append_jsonl(sample_path, row)
    manifest["total_sample_docs"] = total
    manifest["total_utf8_text_bytes"] = total_utf8_text_bytes
    manifest["sample_write_order"] = "round_robin_by_dataset"
    write_json(manifest_path, manifest)
    print(f"[sample] wrote {sample_path} rows={total}")
    print("[sample] write_order=round_robin_by_dataset")
    print(f"[sample] wrote {manifest_path}")


if __name__ == "__main__":
    main()
