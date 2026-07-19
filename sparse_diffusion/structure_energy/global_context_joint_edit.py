from __future__ import annotations

from typing import Optional, Sequence

import torch
from torch import nn


def decode_queries_in_chunks(
    model: nn.Module,
    *,
    encoded: dict,
    query_edge_attr: torch.Tensor,
    query_edge_index: torch.Tensor,
    batch: torch.Tensor,
    chunk_size: int,
    query_edge_family_ids: Optional[torch.Tensor] = None,
    query_edge_stage_ids: Optional[torch.Tensor] = None,
    chunk_order: Optional[Sequence[int]] = None,
    edge_input_residual_scale: Optional[float] = None,
) -> torch.Tensor:
    """Decode a fixed query set in bounded-memory chunks.

    Every chunk shares the same immutable ``encoded`` context. Results are
    restored to canonical query order, so execution order cannot change the
    returned logits. This function deliberately does not update node states.
    """

    query_count = int(query_edge_index.shape[1])
    if query_edge_attr.shape[0] != query_count:
        raise ValueError("query edge attributes do not match query edge count")
    if chunk_size <= 0:
        raise ValueError("query chunk size must be positive")
    if query_count == 0:
        return query_edge_attr.new_empty((0, 0))

    spans = [
        (start, min(start + int(chunk_size), query_count))
        for start in range(0, query_count, int(chunk_size))
    ]
    order = list(range(len(spans))) if chunk_order is None else [int(i) for i in chunk_order]
    if sorted(order) != list(range(len(spans))):
        raise ValueError("chunk_order must be a permutation of all chunk ids")

    logits_parts = []
    index_parts = []
    for chunk_id in order:
        start, end = spans[chunk_id]
        edge_ids = torch.arange(start, end, device=query_edge_index.device)
        logits = model.decode_queries(
            encoded=encoded,
            query_edge_attr=query_edge_attr[start:end],
            query_edge_index=query_edge_index[:, start:end],
            batch=batch,
            query_edge_family_ids=(
                None
                if query_edge_family_ids is None
                else query_edge_family_ids[start:end]
            ),
            query_edge_stage_ids=(
                None
                if query_edge_stage_ids is None
                else query_edge_stage_ids[start:end]
            ),
            edge_input_residual_scale=edge_input_residual_scale,
        )
        logits_parts.append(logits)
        index_parts.append(edge_ids)

    decoded = torch.cat(logits_parts, dim=0)
    decoded_ids = torch.cat(index_parts, dim=0)
    restore = torch.argsort(decoded_ids)
    return decoded[restore]


def decode_edit_edge_slots_in_chunks(
    model: nn.Module,
    *,
    encoded: dict,
    candidate_endpoints: torch.Tensor,
    edge_slot_attr: torch.Tensor,
    candidate_family_ids: torch.Tensor,
    batch: torch.Tensor,
    chunk_size: int,
    chunk_order: Optional[Sequence[int]] = None,
) -> torch.Tensor:
    """Lazily encode the four edge slots of every joint edit proposal.

    Missing/new edges are ordinary decoder-only queries: callers encode their
    current state as the NULL class in ``edge_slot_attr``. They never enter
    ``encode_context`` or alter message passing. Removed edges can carry their
    current subtype in the same tensor.
    """

    if candidate_endpoints.ndim != 2 or candidate_endpoints.shape[-1] != 8:
        raise ValueError("candidate_endpoints must have shape [C,8]")
    candidate_count = int(candidate_endpoints.shape[0])
    if edge_slot_attr.shape[:2] != (candidate_count, 4):
        raise ValueError("edge_slot_attr must have shape [C,4,D]")
    if candidate_family_ids.shape != (candidate_count,):
        raise ValueError("candidate_family_ids must have shape [C]")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    endpoints = candidate_endpoints.reshape(candidate_count, 4, 2)
    valid = (endpoints[..., 0] >= 0) & (endpoints[..., 1] >= 0)
    flat_positions = torch.nonzero(valid.reshape(-1), as_tuple=False).flatten()
    repr_dim = int(getattr(model, "query_edge_repr_dim", 0))
    if flat_positions.numel() == 0:
        return edge_slot_attr.new_zeros((candidate_count, 4, repr_dim))

    flat_endpoints = endpoints.reshape(-1, 2)[flat_positions]
    flat_attr = edge_slot_attr.reshape(-1, edge_slot_attr.shape[-1])[flat_positions]
    flat_family = candidate_family_ids[:, None].expand(-1, 4).reshape(-1)[
        flat_positions
    ]
    spans = [
        (start, min(start + int(chunk_size), int(flat_positions.numel())))
        for start in range(0, int(flat_positions.numel()), int(chunk_size))
    ]
    order = list(range(len(spans))) if chunk_order is None else [int(i) for i in chunk_order]
    if sorted(order) != list(range(len(spans))):
        raise ValueError("chunk_order must be a permutation of all slot chunks")

    values = []
    value_positions = []
    for chunk_id in order:
        start, end = spans[chunk_id]
        pairs = flat_endpoints[start:end]
        query_index = pairs.t().contiguous()
        model.decode_queries(
            encoded=encoded,
            query_edge_attr=flat_attr[start:end],
            query_edge_index=query_index,
            batch=batch,
            query_edge_family_ids=flat_family[start:end],
        )
        query_repr = getattr(model, "last_query_edge_repr", None)
        if query_repr is None or query_repr.shape[0] != end - start:
            raise RuntimeError("query decoder did not expose last_query_edge_repr")
        if repr_dim == 0:
            repr_dim = int(query_repr.shape[-1])
        elif int(query_repr.shape[-1]) != repr_dim:
            raise RuntimeError("query representation width changed between chunks")
        values.append(query_repr)
        value_positions.append(flat_positions[start:end])

    decoded = torch.cat(values, dim=0)
    decoded_positions = torch.cat(value_positions, dim=0)
    output = decoded.new_zeros((candidate_count * 4, repr_dim))
    output[decoded_positions] = decoded
    return output.reshape(candidate_count, 4, repr_dim)


class GlobalContextJointEditResidual(nn.Module):
    """Zero-initialized context residual around a legacy listwise edit scorer.

    The four endpoint slots are ordered as two removed edges followed by two
    added edges. Pooling removed and added slots separately preserves the
    semantics of relocation and switch operations while remaining invariant
    to the order of edges inside either set. ``edge_slot_dim`` is retained for
    historical edge-query ablations; the paper method uses the node-only
    specialization below.
    """

    def __init__(
        self,
        legacy_scorer: nn.Module,
        *,
        node_hidden_dim: int,
        candidate_feature_dim: int,
        family_count: int = 0,
        feature_mask: Optional[Sequence[float]] = None,
        context_mode: str = "current",
        edge_slot_dim: int = 0,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        context_mode = str(context_mode).strip().lower()
        if context_mode not in {"none", "current", "counterfactual", "both"}:
            raise ValueError(f"unsupported edit context mode {context_mode!r}")
        self.legacy_scorer = legacy_scorer
        self.node_hidden_dim = int(node_hidden_dim)
        self.candidate_feature_dim = int(candidate_feature_dim)
        self.family_count = int(family_count)
        self.context_mode = context_mode
        self.edge_slot_dim = int(edge_slot_dim)
        if feature_mask is None:
            mask = torch.ones(self.candidate_feature_dim, dtype=torch.float32)
        else:
            mask = torch.as_tensor(tuple(feature_mask), dtype=torch.float32)
            if mask.shape != (self.candidate_feature_dim,):
                raise ValueError("candidate feature mask has the wrong width")
        self.register_buffer("feature_mask", mask, persistent=False)

        pair_width = 4 * self.node_hidden_dim
        self.endpoint_pair = nn.Sequential(
            nn.Linear(pair_width, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        slot_width = hidden_dim + self.edge_slot_dim
        residual_width = self.candidate_feature_dim + self.node_hidden_dim
        if self.context_mode in {"current", "both"}:
            residual_width += 2 * slot_width
        if self.context_mode in {"counterfactual", "both"}:
            if self.family_count <= 0:
                raise ValueError("counterfactual context needs a positive family count")
            self.relation_context = nn.Embedding(self.family_count, node_hidden_dim)
            self.left_message = nn.Linear(2 * node_hidden_dim, node_hidden_dim)
            self.right_message = nn.Linear(2 * node_hidden_dim, node_hidden_dim)
            self.virtual_update = nn.Sequential(
                nn.Linear(2 * node_hidden_dim, node_hidden_dim),
                nn.SiLU(),
                nn.Linear(node_hidden_dim, node_hidden_dim),
            )
            self.virtual_norm = nn.LayerNorm(node_hidden_dim)
            residual_width += 4 * self.node_hidden_dim
        self.residual = nn.Sequential(
            nn.Linear(residual_width, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        final = self.residual[-1]
        assert isinstance(final, nn.Linear)
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    def base_score(self, candidate_features: torch.Tensor) -> torch.Tensor:
        return self.legacy_scorer(candidate_features * self.feature_mask)

    def _counterfactual_repr(
        self,
        node_hidden: torch.Tensor,
        candidate_features: torch.Tensor,
        candidate_endpoints: torch.Tensor,
    ) -> torch.Tensor:
        """Approximate endpoint states after an edit without re-encoding the graph."""

        batch_size, candidate_count, _ = candidate_endpoints.shape
        endpoints = candidate_endpoints.reshape(batch_size, candidate_count, 4, 2)
        valid_edges = (endpoints[..., 0] >= 0) & (endpoints[..., 1] >= 0)
        batch_ids = torch.arange(batch_size, device=node_hidden.device)[:, None, None]
        left_ids = endpoints[..., 0].clamp(min=0)
        right_ids = endpoints[..., 1].clamp(min=0)
        left = node_hidden[batch_ids, left_ids]
        right = node_hidden[batch_ids, right_ids]

        family_start = 5
        family_logits = candidate_features[
            ..., family_start : family_start + self.family_count
        ]
        family_ids = family_logits.argmax(dim=-1)
        relation = self.relation_context(family_ids)[:, :, None, :].expand(
            -1, -1, 4, -1
        )
        to_left = self.left_message(torch.cat((right, relation), dim=-1))
        to_right = self.right_message(torch.cat((left, relation), dim=-1))
        signs = candidate_features.new_tensor((-1.0, -1.0, 1.0, 1.0))
        signs = signs[None, None, :, None]
        to_left = to_left * signs * valid_edges.unsqueeze(-1)
        to_right = to_right * signs * valid_edges.unsqueeze(-1)

        flat_ids = endpoints.reshape(batch_size, candidate_count, 8)
        flat_valid = flat_ids >= 0
        flat_safe = flat_ids.clamp(min=0)
        current = node_hidden[
            torch.arange(batch_size, device=node_hidden.device)[:, None, None],
            flat_safe,
        ]
        contributions = torch.stack((to_left, to_right), dim=4).reshape(
            batch_size, candidate_count, 8, self.node_hidden_dim
        )
        same_node = (flat_safe[..., :, None] == flat_safe[..., None, :]) & (
            flat_valid[..., :, None] & flat_valid[..., None, :]
        )
        message_delta = torch.einsum(
            "bcij,bcjh->bcih", same_node.to(contributions.dtype), contributions
        )
        update = self.virtual_update(torch.cat((current, message_delta), dim=-1))
        after = self.virtual_norm(current + update)
        delta = after - current

        positions = torch.arange(8, device=node_hidden.device)
        prior = positions[None, :] < positions[:, None]
        duplicate_before = (same_node & prior[None, None]).any(dim=-1)
        unique = flat_valid & ~duplicate_before
        denominator = unique.sum(dim=-1, keepdim=True).clamp(min=1).to(current.dtype)

        def pool(value: torch.Tensor) -> torch.Tensor:
            return (value * unique.unsqueeze(-1)).sum(dim=2) / denominator

        return torch.cat(
            (pool(current), pool(after), pool(delta), pool(delta.abs())), dim=-1
        )

    def _slot_repr(
        self,
        node_hidden: torch.Tensor,
        candidate_endpoints: torch.Tensor,
        edge_slot_hidden: Optional[torch.Tensor],
    ) -> torch.Tensor:
        batch_size, candidate_count, _ = candidate_endpoints.shape
        slots = []
        batch_ids = torch.arange(batch_size, device=node_hidden.device)[:, None]
        for slot_id in range(4):
            u = candidate_endpoints[..., 2 * slot_id]
            v = candidate_endpoints[..., 2 * slot_id + 1]
            valid = (u >= 0) & (v >= 0)
            hu = node_hidden[batch_ids, u.clamp(min=0)]
            hv = node_hidden[batch_ids, v.clamp(min=0)]
            pair = torch.cat((hu, hv, torch.abs(hu - hv), hu * hv), dim=-1)
            value = self.endpoint_pair(pair) * valid.unsqueeze(-1)
            if self.edge_slot_dim > 0:
                if edge_slot_hidden is None:
                    edge_value = value.new_zeros(
                        (batch_size, candidate_count, self.edge_slot_dim)
                    )
                else:
                    expected = (batch_size, candidate_count, 4, self.edge_slot_dim)
                    if tuple(edge_slot_hidden.shape) != expected:
                        raise ValueError(
                            f"edge_slot_hidden has shape {tuple(edge_slot_hidden.shape)}, "
                            f"expected {expected}"
                        )
                    edge_value = edge_slot_hidden[..., slot_id, :]
                value = torch.cat((value, edge_value * valid.unsqueeze(-1)), dim=-1)
            slots.append(value)
        return torch.stack(slots, dim=2)

    def forward(
        self,
        candidate_features: torch.Tensor,
        *,
        node_hidden: torch.Tensor,
        candidate_endpoints: torch.Tensor,
        edge_slot_hidden: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if candidate_features.ndim != 3:
            raise ValueError("candidate_features must have shape [B,C,D]")
        if candidate_endpoints.shape[:2] != candidate_features.shape[:2]:
            raise ValueError("candidate endpoints do not match candidate features")
        if candidate_endpoints.shape[-1] != 8:
            raise ValueError("candidate endpoints must contain four edge slots")
        if candidate_features.shape[-1] != self.candidate_feature_dim:
            raise ValueError("candidate feature width does not match residual config")

        masked_features = candidate_features * self.feature_mask
        legacy_logits = self.base_score(candidate_features)
        if self.context_mode == "none":
            return legacy_logits
        graph_context = node_hidden.mean(dim=1)[:, None, :].expand(
            -1, candidate_features.shape[1], -1
        )
        parts = [masked_features]
        if self.context_mode in {"current", "both"}:
            slot_repr = self._slot_repr(
                node_hidden, candidate_endpoints, edge_slot_hidden
            )
            parts.extend(
                (slot_repr[:, :, :2].sum(dim=2), slot_repr[:, :, 2:].sum(dim=2))
            )
        parts.append(graph_context)
        if self.context_mode in {"counterfactual", "both"}:
            parts.append(
                self._counterfactual_repr(
                    node_hidden, candidate_features, candidate_endpoints
                )
            )
        residual_input = torch.cat(parts, dim=-1)
        return legacy_logits + self.residual(residual_input).squeeze(-1)


class NodeContextJointEditResidual(GlobalContextJointEditResidual):
    """Joint edit scorer that consumes node context without edge hidden state.

    Existing typed edges are used only by the graph encoder for relation-aware
    message passing. Candidate removals and additions are represented on demand
    from their endpoint node states and compact edit features, so NULL pairs do
    not require persistent representations. The zero-initialized residual keeps
    the exact legacy listwise scorer in the function class.
    """

    uses_persistent_edge_state = False

    def __init__(
        self,
        legacy_scorer: nn.Module,
        *,
        node_hidden_dim: int,
        candidate_feature_dim: int,
        family_count: int = 0,
        feature_mask: Optional[Sequence[float]] = None,
        context_mode: str = "current",
        hidden_dim: int = 128,
    ) -> None:
        super().__init__(
            legacy_scorer,
            node_hidden_dim=node_hidden_dim,
            candidate_feature_dim=candidate_feature_dim,
            family_count=family_count,
            feature_mask=feature_mask,
            context_mode=context_mode,
            edge_slot_dim=0,
            hidden_dim=hidden_dim,
        )

    def forward(
        self,
        candidate_features: torch.Tensor,
        *,
        node_hidden: torch.Tensor,
        candidate_endpoints: torch.Tensor,
    ) -> torch.Tensor:
        return super().forward(
            candidate_features,
            node_hidden=node_hidden,
            candidate_endpoints=candidate_endpoints,
            edge_slot_hidden=None,
        )
