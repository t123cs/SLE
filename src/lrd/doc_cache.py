import json
import os
import pickle
import re
from collections import OrderedDict, defaultdict

import numpy as np
import scipy.sparse as sp
import torch


def build_literal_mask(input_ids: torch.Tensor, attention_mask: torch.Tensor, vocab_size: int) -> torch.Tensor:
    batch_size = int(input_ids.shape[0])
    mask = torch.zeros((batch_size, int(vocab_size)), dtype=torch.float32, device=input_ids.device)
    active = attention_mask.to(dtype=mask.dtype)
    mask.scatter_(1, input_ids, active)
    return mask


def csr_to_padded_tensors(matrix):
    indptr = matrix.indptr
    nnz_per_row = np.diff(indptr)
    max_nnz = int(nnz_per_row.max()) if len(nnz_per_row) > 0 else 0
    row_count = matrix.shape[0]

    indices_np = np.zeros((row_count, max_nnz), dtype=np.int64)
    values_np = np.zeros((row_count, max_nnz), dtype=np.float16)
    mask_np = np.zeros((row_count, max_nnz), dtype=np.bool_)

    total_nnz = int(nnz_per_row.sum())
    if total_nnz > 0 and max_nnz > 0:
        row_ids = np.repeat(np.arange(row_count, dtype=np.int64), nnz_per_row)
        row_offsets = np.repeat(indptr[:-1], nnz_per_row)
        col_ids = np.arange(total_nnz, dtype=np.int64) - row_offsets
        indices_np[row_ids, col_ids] = matrix.indices.astype(np.int64, copy=False)
        values_np[row_ids, col_ids] = matrix.data.astype(np.float16, copy=False)
        mask_np[row_ids, col_ids] = True

    return (
        torch.from_numpy(indices_np),
        torch.from_numpy(values_np),
        torch.from_numpy(mask_np),
    )


class SparseDocCacheWriter:
    def __init__(self, cache_dir, rep_names=("literal", "expansion"), docs_per_part=50000, meta=None):
        self.cache_dir = cache_dir
        self.rep_names = tuple(rep_names)
        self.docs_per_part = max(1, int(docs_per_part))
        self.meta = dict(meta or {})
        self.part_index = 0
        self.num_written_docs = 0
        self._pending_ids = []
        self._pending_matrices = {rep_name: [] for rep_name in self.rep_names}
        os.makedirs(self.cache_dir, exist_ok=True)

    def add_batch(self, doc_ids, rep_matrix_map):
        if len(doc_ids) == 0:
            return
        self._pending_ids.extend([str(doc_id) for doc_id in doc_ids])
        for rep_name in self.rep_names:
            self._pending_matrices[rep_name].append(rep_matrix_map[rep_name].tocsr())
        self._flush_ready_parts()

    def finalize(self):
        self._flush_ready_parts(force=True)
        manifest = {
            "num_parts": self.part_index,
            "docs_per_part": self.docs_per_part,
            "num_docs": self.num_written_docs,
            "rep_names": list(self.rep_names),
        }
        meta = dict(self.meta)
        meta.update(
            {
                "cache_type": "sparse_doc_cache",
                "rep_names": list(self.rep_names),
                "docs_per_part": self.docs_per_part,
                "num_docs": self.num_written_docs,
            }
        )
        with open(os.path.join(self.cache_dir, "meta.json"), "w", encoding="utf-8") as fout:
            json.dump(meta, fout, indent=2)
        with open(os.path.join(self.cache_dir, "corpus_shard_00_manifest.json"), "w", encoding="utf-8") as fout:
            json.dump(manifest, fout, indent=2)

    def _flush_ready_parts(self, force=False):
        while len(self._pending_ids) >= self.docs_per_part or (force and self._pending_ids):
            stacked = self._stack_pending()
            part_size = len(self._pending_ids) if force else self.docs_per_part
            part_ids = self._pending_ids[:part_size]
            part_matrices = {
                rep_name: stacked[rep_name][:part_size].tocsr()
                for rep_name in self.rep_names
            }
            self._write_part(part_ids, part_matrices)
            self._pending_ids = self._pending_ids[part_size:]
            if len(self._pending_ids) > 0:
                self._pending_matrices = {
                    rep_name: [stacked[rep_name][part_size:].tocsr()]
                    for rep_name in self.rep_names
                }
            else:
                self._pending_matrices = {rep_name: [] for rep_name in self.rep_names}

    def _stack_pending(self):
        stacked = {}
        for rep_name in self.rep_names:
            matrices = self._pending_matrices[rep_name]
            if len(matrices) == 1:
                stacked[rep_name] = matrices[0]
            else:
                stacked[rep_name] = sp.vstack(matrices, format="csr")
        return stacked

    def _write_part(self, part_ids, part_matrices):
        base = os.path.join(self.cache_dir, f"corpus_shard_00_part_{self.part_index:05d}")
        with open(f"{base}_ids.json", "w", encoding="utf-8") as fout:
            json.dump(part_ids, fout)
        with open(f"{base}_bundle.pkl", "wb") as fout:
            pickle.dump(
                {
                    "ids": [str(doc_id) for doc_id in part_ids],
                    "matrices": {rep_name: part_matrices[rep_name] for rep_name in self.rep_names},
                },
                fout,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        self.num_written_docs += len(part_ids)
        self.part_index += 1


class SparseDocCache:
    def __init__(self, cache_dir, rep_names=("literal", "expansion"), max_loaded_parts=8):
        self.cache_dir = cache_dir
        self.rep_names = tuple(rep_names)
        self.max_loaded_parts = max_loaded_parts
        self.pid_to_loc = {}
        self._part_cache = OrderedDict()
        self._build_pid_index()

    def _build_pid_index(self):
        manifest_pattern = re.compile(r"corpus_shard_(\d+)_manifest\.json$")
        shard_indices = []
        for name in os.listdir(self.cache_dir):
            match = manifest_pattern.match(name)
            if match:
                shard_indices.append(int(match.group(1)))

        if not shard_indices:
            raise FileNotFoundError(f"no corpus shard manifests found under {self.cache_dir}")

        for shard_index in sorted(shard_indices):
            manifest_path = os.path.join(self.cache_dir, f"corpus_shard_{shard_index:02d}_manifest.json")
            with open(manifest_path, "r", encoding="utf-8") as fin:
                manifest = json.load(fin)
            num_parts = int(manifest.get("num_parts", 0))
            for part_index in range(num_parts):
                ids_path = os.path.join(
                    self.cache_dir,
                    f"corpus_shard_{shard_index:02d}_part_{part_index:05d}_ids.json",
                )
                with open(ids_path, "r", encoding="utf-8") as fin:
                    part_ids = json.load(fin)
                for row_idx, pid in enumerate(part_ids):
                    self.pid_to_loc[str(pid)] = (shard_index, part_index, row_idx)

    def _bundle_path(self, shard_index, part_index):
        return os.path.join(
            self.cache_dir,
            f"corpus_shard_{shard_index:02d}_part_{part_index:05d}_bundle.pkl",
        )

    def _base_path(self, shard_index, part_index):
        return os.path.join(
            self.cache_dir,
            f"corpus_shard_{shard_index:02d}_part_{part_index:05d}",
        )

    def _load_part(self, shard_index, part_index):
        cache_key = (shard_index, part_index)
        if cache_key in self._part_cache:
            payload = self._part_cache.pop(cache_key)
            self._part_cache[cache_key] = payload
            return payload

        bundle_path = self._bundle_path(shard_index, part_index)
        if os.path.exists(bundle_path):
            with open(bundle_path, "rb") as fin:
                payload = pickle.load(fin)
        else:
            base_path = self._base_path(shard_index, part_index)
            with open(f"{base_path}_ids.json", "r", encoding="utf-8") as fin:
                ids = json.load(fin)
            payload = {
                "ids": [str(pid) for pid in ids],
                "matrices": {
                    rep_name: sp.load_npz(f"{base_path}_{rep_name}.npz").tocsr()
                    for rep_name in self.rep_names
                },
            }

        self._part_cache[cache_key] = payload
        while len(self._part_cache) > self.max_loaded_parts:
            self._part_cache.popitem(last=False)
        return payload

    def fetch(self, doc_ids):
        part_to_requests = defaultdict(list)
        for output_idx, doc_id in enumerate(doc_ids):
            doc_key = str(doc_id)
            if doc_key not in self.pid_to_loc:
                raise KeyError(f"doc id {doc_key} not found in cache index {self.cache_dir}")
            shard_index, part_index, row_idx = self.pid_to_loc[doc_key]
            part_to_requests[(shard_index, part_index)].append((output_idx, row_idx))

        part_output_orders = []
        gathered_blocks = {rep_name: [] for rep_name in self.rep_names}
        for (shard_index, part_index), requests in part_to_requests.items():
            payload = self._load_part(shard_index, part_index)
            output_indices = np.fromiter((output_idx for output_idx, _ in requests), dtype=np.int64, count=len(requests))
            rows = np.fromiter((row_idx for _, row_idx in requests), dtype=np.int64, count=len(requests))
            part_output_orders.append(output_indices)
            for rep_name in self.rep_names:
                gathered_blocks[rep_name].append(payload["matrices"][rep_name][rows])

        stacked_output_order = np.concatenate(part_output_orders)
        restore_order = np.empty(len(doc_ids), dtype=np.int64)
        restore_order[stacked_output_order] = np.arange(len(doc_ids), dtype=np.int64)

        outputs = {}
        for rep_name, blocks in gathered_blocks.items():
            if len(blocks) == 1:
                stacked = blocks[0]
            else:
                stacked = sp.vstack(blocks, format="csr")
            outputs[rep_name] = stacked[restore_order]
        return outputs
