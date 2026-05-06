from dataclasses import dataclass
from typing import Mapping

import torch


# Paper order from Section 4: LL, LE, EL, EE.
ROUTE_NAMES = ("ll", "le", "el", "ee")


def build_literal_mask(input_ids: torch.Tensor, attention_mask: torch.Tensor, vocab_size: int) -> torch.Tensor:
    batch_size = int(input_ids.shape[0])
    mask = torch.zeros((batch_size, int(vocab_size)), dtype=torch.float32, device=input_ids.device)
    active = attention_mask.to(dtype=mask.dtype)
    mask.scatter_(1, input_ids, active)
    return mask


def split_query_literal_and_expansion(
    query_reps: torch.Tensor,
    literal_mask: torch.Tensor,
    expansion_alpha: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    literal_mask = literal_mask.to(device=query_reps.device, dtype=query_reps.dtype)
    query_literal = query_reps * literal_mask
    query_expansion = query_reps * (1.0 - literal_mask)
    if expansion_alpha is not None:
        query_expansion = query_expansion * expansion_alpha.to(
            device=query_reps.device,
            dtype=query_reps.dtype,
        ).unsqueeze(1)
    return query_literal, query_expansion


def score_dense_queries_with_sparse_docs(
    query_reps: torch.Tensor,
    doc_indices: torch.Tensor,
    doc_values: torch.Tensor,
    doc_mask: torch.Tensor,
) -> torch.Tensor:
    doc_indices = doc_indices.to(device=query_reps.device)
    doc_values = doc_values.to(device=query_reps.device, dtype=query_reps.dtype)
    doc_mask = doc_mask.to(device=query_reps.device, dtype=query_reps.dtype)
    gathered = query_reps[:, doc_indices]
    return (gathered * doc_values.unsqueeze(0) * doc_mask.unsqueeze(0)).sum(dim=-1)


def score_grouped_dense_queries_with_sparse_docs(
    query_reps: torch.Tensor,
    doc_indices: torch.Tensor,
    doc_values: torch.Tensor,
    doc_mask: torch.Tensor,
) -> torch.Tensor:
    doc_indices = doc_indices.to(device=query_reps.device)
    doc_values = doc_values.to(device=query_reps.device, dtype=query_reps.dtype)
    doc_mask = doc_mask.to(device=query_reps.device, dtype=query_reps.dtype)

    grouped_doc_indices = doc_indices.view(query_reps.size(0), -1, doc_indices.size(-1))
    grouped_doc_values = doc_values.view(query_reps.size(0), -1, doc_values.size(-1))
    grouped_doc_mask = doc_mask.view(query_reps.size(0), -1, doc_mask.size(-1))
    expanded_queries = query_reps.unsqueeze(1).expand(-1, grouped_doc_indices.size(1), -1)
    gathered = torch.gather(expanded_queries, 2, grouped_doc_indices)
    return (gathered * grouped_doc_values * grouped_doc_mask).sum(dim=-1)


@dataclass
class RouteScoreBundle:
    query_literal: torch.Tensor
    query_expansion: torch.Tensor
    ll: torch.Tensor
    le: torch.Tensor
    el: torch.Tensor
    ee: torch.Tensor

    def as_dict(self) -> dict[str, torch.Tensor]:
        return {
            "query_literal": self.query_literal,
            "query_expansion": self.query_expansion,
            "ll": self.ll,
            "le": self.le,
            "el": self.el,
            "ee": self.ee,
        }

    def weighted_score(self, route_weights: Mapping[str, float] | None = None) -> torch.Tensor:
        weights = normalize_route_weights(route_weights, self.ll)
        return (
            weights["ll"] * self.ll
            + weights["le"] * self.le
            + weights["el"] * self.el
            + weights["ee"] * self.ee
        )


def normalize_route_weights(
    route_weights: Mapping[str, float] | None,
    reference_tensor: torch.Tensor,
) -> dict[str, torch.Tensor]:
    route_weights = route_weights or {}
    return {
        name: reference_tensor.new_tensor(float(route_weights.get(name, 1.0)))
        for name in ROUTE_NAMES
    }


def decompose_sparse_matching(
    query_reps: torch.Tensor,
    query_input_ids: torch.Tensor,
    query_attention_mask: torch.Tensor,
    doc_literal_indices: torch.Tensor,
    doc_literal_values: torch.Tensor,
    doc_literal_mask: torch.Tensor,
    doc_expansion_indices: torch.Tensor,
    doc_expansion_values: torch.Tensor,
    doc_expansion_mask: torch.Tensor,
    grouped_docs: bool = False,
    expansion_alpha: torch.Tensor | None = None,
) -> RouteScoreBundle:
    literal_mask = build_literal_mask(
        query_input_ids,
        query_attention_mask,
        query_reps.shape[-1],
    )
    query_literal, query_expansion = split_query_literal_and_expansion(
        query_reps,
        literal_mask,
        expansion_alpha=expansion_alpha,
    )

    scorer = score_grouped_dense_queries_with_sparse_docs if grouped_docs else score_dense_queries_with_sparse_docs

    ll_scores = scorer(
        query_literal,
        doc_literal_indices,
        doc_literal_values,
        doc_literal_mask,
    )
    le_scores = scorer(
        query_literal,
        doc_expansion_indices,
        doc_expansion_values,
        doc_expansion_mask,
    )
    el_scores = scorer(
        query_expansion,
        doc_literal_indices,
        doc_literal_values,
        doc_literal_mask,
    )
    ee_scores = scorer(
        query_expansion,
        doc_expansion_indices,
        doc_expansion_values,
        doc_expansion_mask,
    )
    return RouteScoreBundle(
        query_literal=query_literal,
        query_expansion=query_expansion,
        ll=ll_scores,
        le=le_scores,
        el=el_scores,
        ee=ee_scores,
    )
