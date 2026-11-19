from __future__ import annotations

import itertools
from typing import Mapping, Sequence

import numpy as np
import torch

from .features import (
    FeatureSelection,
    _canonical_graphlet_signature,
    _degree_bucket,
)
from .graph import TypedGraph
from .switch_chain import SwitchProposal


_SUPPORTED_PREFIXES = (
    "triangle/",
    "root_triangle/",
    "root_triangle_total/",
)


class BatchedClosedTriangleKernel:
    """Exact batched feature deltas for degree-preserving 2-switches.

    Candidate generation and the committed graph mutation remain on the CPU.
    This kernel mirrors only the binary adjacency and computes all closed
    three-node statistics for a candidate list in one tensor program. It is
    exact for the supported feature prefixes and refuses every other feature.
    """

    def __init__(
        self,
        graph: TypedGraph,
        selection: FeatureSelection,
        device: str | torch.device,
    ) -> None:
        unsupported = [
            key
            for key in selection.keys
            if not key.startswith(_SUPPORTED_PREFIXES)
        ]
        if unsupported:
            raise ValueError(
                "batched triangle kernel does not support features: "
                + ", ".join(unsupported[:3])
            )
        self.device = torch.device(device)
        self.selection = selection
        self.feature_count = len(selection.keys)
        self.family_to_id = {
            name: index for index, name in enumerate(sorted(graph.family_specs))
        }
        self.type_count = len(graph.type_names)
        self.bucket_names = tuple(
            sorted({_degree_bucket(int(value)) for value in graph.degrees()})
        )
        self.bucket_to_id = {
            name: index for index, name in enumerate(self.bucket_names)
        }
        self.node_types = torch.as_tensor(
            graph.node_types, dtype=torch.long, device=self.device
        )
        self.node_buckets = torch.as_tensor(
            [self.bucket_to_id[_degree_bucket(int(value))] for value in graph.degrees()],
            dtype=torch.long,
            device=self.device,
        )
        adjacency = torch.zeros(
            (graph.num_nodes, graph.num_nodes),
            dtype=torch.bool,
            device=self.device,
        )
        if graph.edge_family:
            edges = torch.as_tensor(
                list(graph.edge_family), dtype=torch.long, device=self.device
            )
            adjacency[edges[:, 0], edges[:, 1]] = True
            adjacency[edges[:, 1], edges[:, 0]] = True
        self.adjacency = adjacency
        self.scale = torch.as_tensor(
            selection.scale, dtype=torch.float64, device=self.device
        )
        self._build_feature_lookups(graph)

    @staticmethod
    def supports(selection: FeatureSelection) -> bool:
        return all(key.startswith(_SUPPORTED_PREFIXES) for key in selection.keys)

    def _build_feature_lookups(self, graph: TypedGraph) -> None:
        key_to_index = {key: index for index, key in enumerate(self.selection.keys)}
        shape = (self.type_count,) * 3
        triangle = torch.full(shape, -1, dtype=torch.long, device=self.device)
        root_total = torch.full(
            (self.type_count, len(self.bucket_names)),
            -1,
            dtype=torch.long,
            device=self.device,
        )
        root_detail = torch.full(
            shape + (self.type_count, len(self.bucket_names)),
            -1,
            dtype=torch.long,
            device=self.device,
        )
        family_by_domain = {
            tuple(sorted((int(spec.src_type), int(spec.dst_type)))): family
            for family, spec in graph.family_specs.items()
        }
        for type_ids in itertools.product(range(self.type_count), repeat=3):
            pair_domains = ((0, 1), (0, 2), (1, 2))
            families = []
            for left, right in pair_domains:
                family = family_by_domain.get(
                    tuple(sorted((type_ids[left], type_ids[right])))
                )
                if family is None:
                    break
                families.append(family)
            if len(families) != 3:
                continue
            signature = _canonical_graphlet_signature(
                tuple(graph.type_names[index] for index in type_ids),
                tuple(families),
            )
            triangle[type_ids] = key_to_index.get(f"triangle/{signature}", -1)
            for root_type in set(type_ids):
                root_name = graph.type_names[root_type]
                for bucket_name, bucket_id in self.bucket_to_id.items():
                    key = (
                        f"root_triangle/{signature}/root={root_name}/"
                        f"degree={bucket_name}"
                    )
                    root_detail[type_ids + (root_type, bucket_id)] = key_to_index.get(
                        key, -1
                    )
        for type_id, type_name in enumerate(graph.type_names):
            for bucket_name, bucket_id in self.bucket_to_id.items():
                key = f"root_triangle_total/type={type_name}/degree={bucket_name}"
                root_total[type_id, bucket_id] = key_to_index.get(key, -1)
        self.triangle_lookup = triangle
        self.root_total_lookup = root_total
        self.root_detail_lookup = root_detail

    def _scatter(
        self,
        raw: torch.Tensor,
        candidate: torch.Tensor,
        feature: torch.Tensor,
        value: torch.Tensor,
    ) -> None:
        valid = (feature >= 0) & (value != 0)
        if not bool(valid.any()):
            return
        flat = raw.view(-1)
        flat_index = candidate[valid] * self.feature_count + feature[valid]
        flat.scatter_add_(0, flat_index, value[valid].to(raw.dtype))

    def _add_triangle_events(
        self,
        raw: torch.Tensor,
        candidate: torch.Tensor,
        nodes: torch.Tensor,
        value: torch.Tensor,
    ) -> None:
        if candidate.numel() == 0:
            return
        types = self.node_types[nodes]
        triangle_feature = self.triangle_lookup[
            types[:, 0], types[:, 1], types[:, 2]
        ]
        self._scatter(raw, candidate, triangle_feature, value)
        for root_position in range(3):
            root = nodes[:, root_position]
            root_type = self.node_types[root]
            root_bucket = self.node_buckets[root]
            total_feature = self.root_total_lookup[root_type, root_bucket]
            detail_feature = self.root_detail_lookup[
                types[:, 0],
                types[:, 1],
                types[:, 2],
                root_type,
                root_bucket,
            ]
            self._scatter(raw, candidate, total_feature, value)
            self._scatter(raw, candidate, detail_feature, value)

    def delta_matrix(self, proposals: Sequence[SwitchProposal]) -> torch.Tensor:
        if not proposals:
            raise ValueError("candidate proposal list is empty")
        if any(len(item.old_edges) != 2 or len(item.new_edges) != 2 for item in proposals):
            raise ValueError("batched triangle kernel requires 2-switch proposals")
        candidate_count = len(proposals)
        changed = torch.as_tensor(
            [list(item.old_edges) + list(item.new_edges) for item in proposals],
            dtype=torch.long,
            device=self.device,
        )
        signs = torch.tensor(
            [-1, -1, 1, 1], dtype=torch.int64, device=self.device
        )
        raw = torch.zeros(
            (candidate_count, self.feature_count),
            dtype=torch.int64,
            device=self.device,
        )
        endpoints = changed[:, :2, :].reshape(candidate_count, 4)
        sorted_endpoints = torch.sort(endpoints, dim=1).values
        if bool((sorted_endpoints[:, 1:] == sorted_endpoints[:, :-1]).any()):
            # Every legal homogeneous or bipartite 2-switch has four distinct
            # endpoints. Refuse malformed/custom proposals instead of silently
            # producing an approximate delta.
            raise ValueError("2-switch candidates must have four distinct endpoints")

        for slot in range(4):
            left = changed[:, slot, 0]
            right = changed[:, slot, 1]
            common = self.adjacency[left] & self.adjacency[right]
            common.scatter_(1, endpoints, False)
            event = common.nonzero(as_tuple=False)
            if event.numel() == 0:
                continue
            candidate = event[:, 0]
            third = event[:, 1]
            nodes = torch.stack((left[candidate], right[candidate], third), dim=1)
            value = signs[slot].expand(candidate.shape[0])
            self._add_triangle_events(raw, candidate, nodes, value)

        local = endpoints
        before = self.adjacency[local[:, :, None], local[:, None, :]].clone()
        after = before.clone()
        for slot in range(4):
            edge = changed[:, slot]
            matches_left = local == edge[:, 0, None]
            matches_right = local == edge[:, 1, None]
            left_pos = matches_left.to(torch.long).argmax(dim=1)
            right_pos = matches_right.to(torch.long).argmax(dim=1)
            value = bool(slot >= 2)
            rows = torch.arange(candidate_count, device=self.device)
            after[rows, left_pos, right_pos] = value
            after[rows, right_pos, left_pos] = value
        for positions in itertools.combinations(range(4), 3):
            a, b, c = positions
            old_closed = before[:, a, b] & before[:, a, c] & before[:, b, c]
            new_closed = after[:, a, b] & after[:, a, c] & after[:, b, c]
            value = new_closed.to(torch.int64) - old_closed.to(torch.int64)
            valid = value != 0
            if not bool(valid.any()):
                continue
            candidate = torch.arange(candidate_count, device=self.device)[valid]
            nodes = local[valid][:, [a, b, c]]
            self._add_triangle_events(raw, candidate, nodes, value[valid])
        return raw.to(torch.float64) / self.scale

    def commit(self, proposal: SwitchProposal) -> None:
        for left, right in proposal.old_edges:
            self.adjacency[int(left), int(right)] = False
            self.adjacency[int(right), int(left)] = False
        for left, right in proposal.new_edges:
            self.adjacency[int(left), int(right)] = True
            self.adjacency[int(right), int(left)] = True


def tensor_candidate_features(
    delta: torch.Tensor,
    current: torch.Tensor,
    family_ids: torch.Tensor,
    family_count: int,
    time_fraction: float,
    conditioning_mode: str,
    target: torch.Tensor | None = None,
) -> torch.Tensor:
    conditioning_mode = str(conditioning_mode).strip().lower()
    if conditioning_mode == "target_residual":
        if target is None:
            raise ValueError("target_residual conditioning requires a target")
        context = target - current
        interaction = delta * torch.sign(context)[None]
    elif conditioning_mode == "state_only":
        context = current
        interaction = delta * current[None]
    else:
        raise ValueError(f"unsupported switch conditioning mode {conditioning_mode!r}")
    context_batch = context[None].expand(delta.shape[0], -1)
    family = torch.nn.functional.one_hot(
        family_ids, num_classes=int(family_count)
    ).to(delta.dtype)
    scalar = torch.stack(
        (
            torch.full(
                (delta.shape[0],),
                float(time_fraction),
                dtype=delta.dtype,
                device=delta.device,
            ),
            (delta * context_batch).mean(dim=1),
            torch.linalg.vector_norm(delta, dim=1),
        ),
        dim=1,
    )
    return torch.cat((delta, context_batch, interaction, family, scalar), dim=1)
