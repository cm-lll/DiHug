from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F

from .graph import TypedGraph
from .mixed_edit_diffusion import (
    SCORER_ACTIONS,
    EditProposal,
    MixedEditSamplingResult,
    mixed_edit_feature_width,
)
from .gpu_edit_triton import fused_typed_closure


NOOP_ID = SCORER_ACTIONS.index("noop")
RELOCATION_ID = SCORER_ACTIONS.index("relocation")
SWITCH_ID = SCORER_ACTIONS.index("switch")


@dataclass
class GpuEditBatch:
    action: torch.Tensor
    family: torch.Tensor
    old_edges: torch.Tensor
    new_edges: torch.Tensor
    old_indices: torch.Tensor
    old_valid: torch.Tensor
    new_valid: torch.Tensor

    def index_select(self, indices: torch.Tensor) -> "GpuEditBatch":
        return GpuEditBatch(
            action=self.action[indices],
            family=self.family[indices],
            old_edges=self.old_edges[indices],
            new_edges=self.new_edges[indices],
            old_indices=self.old_indices[indices],
            old_valid=self.old_valid[indices],
            new_valid=self.new_valid[indices],
        )


class GpuTypedGraphState:
    """GPU-resident mutable graph for fixed-family-count edit diffusion.

    The hot sampling loop keeps adjacency, family edge tables, degree state,
    candidate construction, compact-v2 features, scoring, and commits on one
    CUDA device.  CPU conversion is explicit and intended only for sparse
    metric checkpoints and final serialization.
    """

    def __init__(
        self,
        graph: TypedGraph,
        target_rows: np.ndarray,
        device: str | torch.device,
        *,
        use_triton: bool = True,
    ) -> None:
        self.device = torch.device(device)
        self.use_triton = bool(use_triton)
        if self.device.type != "cuda":
            raise ValueError("GPU graph state requires a CUDA device")
        graph.validate_disjoint_family_domains()
        self.num_nodes = int(graph.num_nodes)
        self.type_names = tuple(graph.type_names)
        self.family_names = tuple(sorted(graph.family_specs))
        self.family_count = len(self.family_names)
        self.type_count = len(self.type_names)
        self.node_types = torch.as_tensor(
            graph.node_types, dtype=torch.long, device=self.device
        )
        self.node_type_one_hot = F.one_hot(
            self.node_types, num_classes=self.type_count
        ).to(torch.float32)
        self.target_rows = torch.as_tensor(
            np.asarray(target_rows), dtype=torch.float32, device=self.device
        )
        if self.target_rows.shape != (self.num_nodes, self.family_count):
            raise ValueError("role target shape disagrees with graph schema")

        counts = [len(graph.family_edges[name]) for name in self.family_names]
        self.family_edge_count_values = tuple(int(value) for value in counts)
        self.family_edge_counts = torch.tensor(
            counts, dtype=torch.long, device=self.device
        )
        self.max_family_edges = max(counts, default=0)
        self.edge_tables = torch.full(
            (self.family_count, self.max_family_edges, 2),
            -1,
            dtype=torch.long,
            device=self.device,
        )
        self.family_degree = torch.zeros(
            (self.family_count, self.num_nodes),
            dtype=torch.long,
            device=self.device,
        )
        self.adjacency = torch.zeros(
            (self.num_nodes, self.num_nodes),
            dtype=torch.bool,
            device=self.device,
        )
        self.family_src_nodes: list[torch.Tensor] = []
        self.family_dst_nodes: list[torch.Tensor] = []
        self.family_homogeneous: list[bool] = []
        capacities = []
        family_by_type = torch.full(
            (self.type_count, self.type_count),
            -1,
            dtype=torch.long,
            device=self.device,
        )
        for family_id, family in enumerate(self.family_names):
            spec = graph.family_specs[family]
            src = torch.as_tensor(
                spec.src_nodes, dtype=torch.long, device=self.device
            )
            dst = torch.as_tensor(
                spec.dst_nodes, dtype=torch.long, device=self.device
            )
            self.family_src_nodes.append(src)
            self.family_dst_nodes.append(dst)
            self.family_homogeneous.append(not spec.bipartite)
            capacity = (
                len(spec.src_nodes) * len(spec.dst_nodes)
                if spec.bipartite
                else len(spec.src_nodes) * (len(spec.src_nodes) - 1) // 2
            )
            capacities.append(capacity)
            family_by_type[int(spec.src_type), int(spec.dst_type)] = family_id
            family_by_type[int(spec.dst_type), int(spec.src_type)] = family_id
            if counts[family_id]:
                edges = torch.as_tensor(
                    graph.family_edges[family],
                    dtype=torch.long,
                    device=self.device,
                )
                self.edge_tables[family_id, : counts[family_id]] = edges
                self.adjacency[edges[:, 0], edges[:, 1]] = True
                self.adjacency[edges[:, 1], edges[:, 0]] = True
                ones = torch.ones(edges.numel(), dtype=torch.long, device=self.device)
                self.family_degree[family_id].scatter_add_(
                    0, edges.reshape(-1), ones
                )
        self.family_capacities = torch.tensor(
            capacities, dtype=torch.float32, device=self.device
        )
        max_endpoint_nodes = max(
            max(item.numel() for item in self.family_src_nodes),
            max(item.numel() for item in self.family_dst_nodes),
        )
        self.src_node_table = torch.zeros(
            (self.family_count, max_endpoint_nodes),
            dtype=torch.long,
            device=self.device,
        )
        self.dst_node_table = self.src_node_table.clone()
        self.src_node_counts = torch.zeros(
            self.family_count, dtype=torch.long, device=self.device
        )
        self.dst_node_counts = self.src_node_counts.clone()
        for family_id, (src, dst) in enumerate(
            zip(self.family_src_nodes, self.family_dst_nodes)
        ):
            self.src_node_table[family_id, : src.numel()] = src
            self.dst_node_table[family_id, : dst.numel()] = dst
            self.src_node_counts[family_id] = src.numel()
            self.dst_node_counts[family_id] = dst.numel()
        self.family_homogeneous_tensor = torch.tensor(
            self.family_homogeneous, dtype=torch.bool, device=self.device
        )
        self.family_src_types = torch.tensor(
            [
                int(graph.family_specs[family].src_type)
                for family in self.family_names
            ],
            dtype=torch.long,
            device=self.device,
        )
        self.family_by_type = family_by_type
        # A single slot map is sufficient because family candidate domains are
        # disjoint. It makes wedge-seeded proposal construction independent of
        # a scan over every family edge table.
        self.edge_slots = torch.full(
            (self.num_nodes, self.num_nodes),
            -1,
            dtype=torch.long,
            device=self.device,
        )
        for family_id, count in enumerate(counts):
            if count:
                edges = self.edge_tables[family_id, :count]
                slots = torch.arange(count, dtype=torch.long, device=self.device)
                self.edge_slots[edges[:, 0], edges[:, 1]] = slots
                self.edge_slots[edges[:, 1], edges[:, 0]] = slots
        self.total_degree = self.family_degree.sum(dim=0)
        self.degree_scale = max(float(np.log1p(self.num_nodes)), 1.0)
        self.common_scale = max(float(np.sqrt(self.num_nodes)), 1.0)

    @staticmethod
    def _canonical(edges: torch.Tensor) -> torch.Tensor:
        return torch.sort(edges, dim=-1).values

    def _sample_family(self, count: int, generator: torch.Generator) -> torch.Tensor:
        available = self.family_edge_counts > 0
        weights = available.to(torch.float32)
        return torch.multinomial(weights, int(count), replacement=True, generator=generator)

    def _random_endpoint_edges(
        self, family: torch.Tensor, generator: torch.Generator
    ) -> torch.Tensor:
        src_index = torch.floor(
            torch.rand(family.shape[0], device=self.device, generator=generator)
            * self.src_node_counts[family].to(torch.float32)
        ).to(torch.long)
        dst_index = torch.floor(
            torch.rand(family.shape[0], device=self.device, generator=generator)
            * self.dst_node_counts[family].to(torch.float32)
        ).to(torch.long)
        edges = torch.stack(
            (
                self.src_node_table[family, src_index],
                self.dst_node_table[family, dst_index],
            ),
            dim=1,
        )
        homogeneous = self.family_homogeneous_tensor[family]
        edges[homogeneous] = self._canonical(edges[homogeneous])
        return edges

    def _random_relocation_pool(
        self, count: int, generator: torch.Generator
    ) -> tuple[GpuEditBatch, torch.Tensor]:
        family = self._sample_family(count, generator)
        edge_count = self.family_edge_counts[family]
        old_index = torch.floor(
            torch.rand(count, device=self.device, generator=generator)
            * edge_count.to(torch.float32)
        ).to(torch.long)
        old_edge = self.edge_tables[family, old_index]
        new_edge = self._random_endpoint_edges(family, generator)
        exists = self.adjacency[new_edge[:, 0], new_edge[:, 1]]
        valid = (
            (new_edge[:, 0] != new_edge[:, 1])
            & ~exists
            & (new_edge != old_edge).any(dim=1)
        )
        batch = self._single_edge_batch(
            RELOCATION_ID, family, old_edge, new_edge, old_index
        )
        return batch, valid

    def _targeted_relocation_pool(
        self, count: int, generator: torch.Generator
    ) -> tuple[GpuEditBatch, torch.Tensor]:
        residual = self.target_rows.transpose(0, 1) - self.family_degree.to(
            torch.float32
        )
        family_mass = residual.clamp_min(0.0).sum(dim=1)
        fallback_family = (self.family_edge_counts > 0).to(torch.float32)
        family_weights = torch.where(
            family_mass.sum() > 0, family_mass, fallback_family
        )
        family = torch.multinomial(
            family_weights, int(count), replacement=True, generator=generator
        )
        safe_edges = self.edge_tables.clamp_min(0)
        old_weights = (
            (-residual).clamp_min(0.0).gather(1, safe_edges[:, :, 0])
            + (-residual).clamp_min(0.0).gather(1, safe_edges[:, :, 1])
        )
        edge_positions = torch.arange(
            self.max_family_edges, device=self.device
        )[None]
        old_weights *= edge_positions < self.family_edge_counts[:, None]
        selected_old_weights = old_weights[family]
        empty_old = selected_old_weights.sum(dim=1) <= 0
        selected_old_weights = torch.where(
            empty_old[:, None],
            (edge_positions < self.family_edge_counts[family, None]).to(torch.float32),
            selected_old_weights,
        )
        old_index = torch.multinomial(
            selected_old_weights, 1, replacement=True, generator=generator
        ).squeeze(1)
        old_edge = self.edge_tables[family, old_index]

        src_weights = residual.clamp_min(0.0)[family].gather(
            1, self.src_node_table[family]
        )
        dst_weights = residual.clamp_min(0.0)[family].gather(
            1, self.dst_node_table[family]
        )
        positions = torch.arange(self.src_node_table.shape[1], device=self.device)[None]
        src_weights *= positions < self.src_node_counts[family, None]
        dst_weights *= positions < self.dst_node_counts[family, None]
        src_has_mass = src_weights.sum(dim=1) > 0
        dst_has_mass = dst_weights.sum(dim=1) > 0
        safe_src_weights = torch.where(
            src_has_mass[:, None], src_weights, F.one_hot(torch.zeros_like(family), src_weights.shape[1]).to(torch.float32)
        )
        safe_dst_weights = torch.where(
            dst_has_mass[:, None], dst_weights, F.one_hot(torch.zeros_like(family), dst_weights.shape[1]).to(torch.float32)
        )
        src_index = torch.multinomial(
            safe_src_weights, 1, replacement=True, generator=generator
        ).squeeze(1)
        dst_index = torch.multinomial(
            safe_dst_weights, 1, replacement=True, generator=generator
        ).squeeze(1)
        sampled_src = self.src_node_table[family, src_index]
        sampled_dst = self.dst_node_table[family, dst_index]
        new_edge = torch.stack(
            (
                torch.where(src_has_mass, sampled_src, old_edge[:, 0]),
                torch.where(dst_has_mass, sampled_dst, old_edge[:, 1]),
            ),
            dim=1,
        )
        homogeneous = self.family_homogeneous_tensor[family]
        new_edge[homogeneous] = self._canonical(new_edge[homogeneous])
        valid = (
            (new_edge[:, 0] != new_edge[:, 1])
            & ~self.adjacency[new_edge[:, 0], new_edge[:, 1]]
            & (new_edge != old_edge).any(dim=1)
            & ~empty_old
        )
        return (
            self._single_edge_batch(
                RELOCATION_ID, family, old_edge, new_edge, old_index
            ),
            valid,
        )

    def _random_switch_pool(
        self, count: int, generator: torch.Generator
    ) -> tuple[GpuEditBatch, torch.Tensor]:
        family = self._sample_family(count, generator)
        edge_count = self.family_edge_counts[family]
        first = torch.floor(
            torch.rand(count, device=self.device, generator=generator)
            * edge_count.to(torch.float32)
        ).to(torch.long)
        second = torch.floor(
            torch.rand(count, device=self.device, generator=generator)
            * edge_count.to(torch.float32)
        ).to(torch.long)
        old_edges = torch.stack(
            (self.edge_tables[family, first], self.edge_tables[family, second]), dim=1
        )
        # Homogeneous edge rows are canonical, so randomize their orientation
        # before crossing to cover both legal switch complements.
        same_type = (
            self.node_types[old_edges[:, 0, 0]]
            == self.node_types[old_edges[:, 0, 1]]
        )
        flips = (
            torch.rand((count, 2), device=self.device, generator=generator) < 0.5
        ) & same_type[:, None]
        oriented = old_edges.clone()
        oriented[flips] = oriented[flips].flip(-1)
        new_edges = torch.stack(
            (
                torch.stack((oriented[:, 0, 0], oriented[:, 1, 1]), dim=1),
                torch.stack((oriented[:, 1, 0], oriented[:, 0, 1]), dim=1),
            ),
            dim=1,
        )
        homogeneous = self.node_types[new_edges[:, 0, 0]] == self.node_types[
            new_edges[:, 0, 1]
        ]
        new_edges[homogeneous] = self._canonical(new_edges[homogeneous])
        endpoints = old_edges.reshape(count, 4)
        four_distinct = torch.sort(endpoints, dim=1).values.diff(dim=1).ne(0).all(dim=1)
        new_exists = self.adjacency[new_edges[:, :, 0], new_edges[:, :, 1]].any(dim=1)
        new_distinct = (new_edges[:, 0] != new_edges[:, 1]).any(dim=1)
        valid = (
            (first != second)
            & four_distinct
            & ~new_exists
            & new_distinct
            & (new_edges[:, :, 0] != new_edges[:, :, 1]).all(dim=1)
        )
        batch = GpuEditBatch(
            action=torch.full(
                (count,), SWITCH_ID, dtype=torch.long, device=self.device
            ),
            family=family,
            old_edges=old_edges,
            new_edges=new_edges,
            old_indices=torch.stack((first, second), dim=1),
            old_valid=torch.ones((count, 2), dtype=torch.bool, device=self.device),
            new_valid=torch.ones((count, 2), dtype=torch.bool, device=self.device),
        )
        return batch, valid

    def _random_mixed_pool(
        self, count: int, generator: torch.Generator
    ) -> tuple[GpuEditBatch, torch.Tensor]:
        relocation, relocation_valid = self._random_relocation_pool(
            count, generator
        )
        switch, switch_valid = self._random_switch_pool(count, generator)
        choose_switch = (
            torch.rand(count, device=self.device, generator=generator) < 0.5
        )
        batch = GpuEditBatch(
            **{
                field: torch.where(
                    choose_switch.reshape(
                        (count,)
                        + (1,) * (getattr(relocation, field).ndim - 1)
                    ),
                    getattr(switch, field),
                    getattr(relocation, field),
                )
                for field in GpuEditBatch.__dataclass_fields__
            }
        )
        valid = torch.where(choose_switch, switch_valid, relocation_valid)
        return batch, valid

    def _orient_family_edges(
        self, family: torch.Tensor, edges: torch.Tensor
    ) -> torch.Tensor:
        oriented = edges.clone()
        homogeneous = self.family_homogeneous_tensor[family]
        if homogeneous.any():
            oriented[homogeneous] = self._canonical(oriented[homogeneous])
        bipartite = ~homogeneous
        if bipartite.any():
            rows = oriented[bipartite]
            families = family[bipartite]
            flip = self.node_types[rows[:, 0]] != self.family_src_types[families]
            rows[flip] = rows[flip].flip(1)
            oriented[bipartite] = rows
        return oriented

    def _sample_masked_nodes(
        self, mask: torch.Tensor, generator: torch.Generator
    ) -> tuple[torch.Tensor, torch.Tensor]:
        valid = mask.any(dim=1)
        fallback = F.one_hot(
            torch.zeros(mask.shape[0], dtype=torch.long, device=self.device),
            num_classes=mask.shape[1],
        ).to(torch.float32)
        weights = torch.where(valid[:, None], mask.to(torch.float32), fallback)
        sampled = torch.multinomial(
            weights, 1, replacement=True, generator=generator
        ).squeeze(1)
        return sampled, valid

    def _wedge_seeded_switch_pool(
        self, count: int, generator: torch.Generator
    ) -> tuple[GpuEditBatch, torch.Tensor]:
        """Vectorized counterpart of ``propose_wedge_seeded_switch``.

        The implementation samples open typed wedges directly and never ranks
        a random switch pool by closure gain. Degree-biased wedge sampling is
        intentionally excluded from the parity mode because the frozen paper
        protocol uses zero degree bias.
        """

        center_weights = (self.total_degree >= 2).to(torch.float32)
        center_available = center_weights.sum() > 0
        center_weights = torch.where(
            center_available,
            center_weights,
            F.one_hot(
                torch.zeros((), dtype=torch.long, device=self.device),
                num_classes=self.num_nodes,
            ).to(torch.float32),
        )
        centers = torch.multinomial(
            center_weights,
            int(count),
            replacement=True,
            generator=generator,
        )
        neighbor_mask = self.adjacency[centers]
        left, left_valid = self._sample_masked_nodes(neighbor_mask, generator)
        right_mask = neighbor_mask.clone()
        right_mask[torch.arange(count, device=self.device), left] = False
        right, right_valid = self._sample_masked_nodes(right_mask, generator)

        family = self.family_by_type[self.node_types[left], self.node_types[right]]
        safe_family = family.clamp_min(0)
        desired = self._orient_family_edges(
            safe_family, torch.stack((left, right), dim=1)
        )
        source = desired[:, 0]
        target = desired[:, 1]
        node_ids = torch.arange(self.num_nodes, device=self.device)[None]
        source_domains = self.family_by_type[
            self.node_types[source, None], self.node_types[None, :]
        ]
        target_domains = self.family_by_type[
            self.node_types[target, None], self.node_types[None, :]
        ]
        source_mask = self.adjacency[source] & (source_domains == safe_family[:, None])
        target_mask = self.adjacency[target] & (target_domains == safe_family[:, None])
        source_mask &= node_ids != centers[:, None]
        target_mask &= node_ids != centers[:, None]
        source_other, source_valid = self._sample_masked_nodes(source_mask, generator)
        target_other, target_valid = self._sample_masked_nodes(target_mask, generator)

        first_old = self._orient_family_edges(
            safe_family, torch.stack((source, source_other), dim=1)
        )
        second_old = self._orient_family_edges(
            safe_family, torch.stack((target_other, target), dim=1)
        )
        complement = self._orient_family_edges(
            safe_family, torch.stack((target_other, source_other), dim=1)
        )
        old_edges = torch.stack((first_old, second_old), dim=1)
        new_edges = torch.stack((desired, complement), dim=1)
        old_indices = torch.stack(
            (
                self.edge_slots[first_old[:, 0], first_old[:, 1]],
                self.edge_slots[second_old[:, 0], second_old[:, 1]],
            ),
            dim=1,
        )
        endpoints = torch.stack(
            (source, target, source_other, target_other), dim=1
        )
        four_distinct = (
            torch.sort(endpoints, dim=1).values.diff(dim=1).ne(0).all(dim=1)
        )
        valid = (
            center_available
            & left_valid
            & right_valid
            & source_valid
            & target_valid
            & (family >= 0)
            & ~self.adjacency[left, right]
            & four_distinct
            & (old_indices >= 0).all(dim=1)
            & ~self.adjacency[new_edges[:, :, 0], new_edges[:, :, 1]].any(dim=1)
            & (new_edges[:, 0] != new_edges[:, 1]).any(dim=1)
            & (new_edges[:, :, 0] != new_edges[:, :, 1]).all(dim=1)
        )
        batch = GpuEditBatch(
            action=torch.full(
                (count,), SWITCH_ID, dtype=torch.long, device=self.device
            ),
            family=safe_family,
            old_edges=old_edges,
            new_edges=new_edges,
            old_indices=old_indices,
            old_valid=torch.ones((count, 2), dtype=torch.bool, device=self.device),
            new_valid=torch.ones((count, 2), dtype=torch.bool, device=self.device),
        )
        return batch, valid

    def _single_edge_batch(
        self,
        action: int,
        family: torch.Tensor,
        old_edge: torch.Tensor,
        new_edge: torch.Tensor,
        old_index: torch.Tensor,
    ) -> GpuEditBatch:
        count = family.shape[0]
        pad = torch.full((count, 1, 2), -1, dtype=torch.long, device=self.device)
        index_pad = torch.full((count, 1), -1, dtype=torch.long, device=self.device)
        one = torch.ones((count, 1), dtype=torch.bool, device=self.device)
        zero = torch.zeros((count, 1), dtype=torch.bool, device=self.device)
        return GpuEditBatch(
            action=torch.full((count,), action, dtype=torch.long, device=self.device),
            family=family,
            old_edges=torch.cat((old_edge[:, None], pad), dim=1),
            new_edges=torch.cat((new_edge[:, None], pad), dim=1),
            old_indices=torch.cat((old_index[:, None], index_pad), dim=1),
            old_valid=torch.cat((one, zero), dim=1),
            new_valid=torch.cat((one, zero), dim=1),
        )

    def _noop(self, count: int = 1) -> GpuEditBatch:
        edges = torch.full((count, 2, 2), -1, dtype=torch.long, device=self.device)
        return GpuEditBatch(
            action=torch.full((count,), NOOP_ID, dtype=torch.long, device=self.device),
            family=torch.zeros(count, dtype=torch.long, device=self.device),
            old_edges=edges,
            new_edges=edges.clone(),
            old_indices=torch.full((count, 2), -1, dtype=torch.long, device=self.device),
            old_valid=torch.zeros((count, 2), dtype=torch.bool, device=self.device),
            new_valid=torch.zeros((count, 2), dtype=torch.bool, device=self.device),
        )

    @staticmethod
    def _concat(items: Sequence[GpuEditBatch]) -> GpuEditBatch:
        return GpuEditBatch(
            **{
                field: torch.cat([getattr(item, field) for item in items], dim=0)
                for field in GpuEditBatch.__dataclass_fields__
            }
        )

    @staticmethod
    def blend(
        role: GpuEditBatch, refinement: GpuEditBatch, role_phase: torch.Tensor
    ) -> GpuEditBatch:
        if role.action.shape != refinement.action.shape:
            raise ValueError("candidate batches must have equal shapes")
        return GpuEditBatch(
            **{
                field: torch.where(
                    role_phase.reshape((1,) + (1,) * (getattr(role, field).ndim - 1)),
                    getattr(role, field),
                    getattr(refinement, field),
                )
                for field in GpuEditBatch.__dataclass_fields__
            }
        )

    def _pick_valid(
        self,
        batch: GpuEditBatch,
        valid: torch.Tensor,
        count: int,
        generator: torch.Generator,
        priority: torch.Tensor | None = None,
    ) -> GpuEditBatch:
        random = torch.rand(valid.shape[0], device=self.device, generator=generator)
        score = random if priority is None else priority.to(torch.float32)
        score = score + valid.to(torch.float32) * 1.0e6
        indices = torch.topk(score, k=int(count), largest=True).indices
        selected = batch.index_select(indices)
        selected_valid = valid[indices]
        # No host read: invalid tail entries are converted to noops by masks.
        selected.action = torch.where(
            selected_valid,
            selected.action,
            torch.full_like(selected.action, NOOP_ID),
        )
        selected.family = torch.where(
            selected_valid, selected.family, torch.zeros_like(selected.family)
        )
        selected.old_valid &= selected_valid[:, None]
        selected.new_valid &= selected_valid[:, None]
        return selected

    def _edit_keys(self, batch: GpuEditBatch) -> torch.Tensor:
        def normalized_slots(edges: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
            canonical = self._canonical(edges.clamp_min(0))
            canonical = torch.where(valid[:, :, None], canonical, -torch.ones_like(canonical))
            radix = self.num_nodes + 1
            order_key = (canonical[:, :, 0] + 1) * radix + canonical[:, :, 1] + 1
            order = order_key.argsort(dim=1)
            return canonical.gather(1, order[:, :, None].expand(-1, -1, 2))

        old_edges = normalized_slots(batch.old_edges, batch.old_valid)
        new_edges = normalized_slots(batch.new_edges, batch.new_valid)
        return torch.cat(
            (
                batch.action[:, None],
                batch.family[:, None],
                old_edges.reshape(old_edges.shape[0], -1),
                new_edges.reshape(new_edges.shape[0], -1),
            ),
            dim=1,
        )

    def _ordered_pick_valid(
        self,
        batch: GpuEditBatch,
        valid: torch.Tensor,
        count: int,
        *,
        exclude: GpuEditBatch | None = None,
    ) -> GpuEditBatch:
        """Select first legal unique proposals without utility-based screening."""

        if int(count) <= 0:
            return self._noop(0)
        keys = self._edit_keys(batch)
        prefix = 0
        if exclude is not None and exclude.action.numel() > 0:
            exclude_keys = self._edit_keys(exclude)
            keys = torch.cat((exclude_keys, keys), dim=0)
            prefix = exclude_keys.shape[0]
        _, inverse = torch.unique(keys, dim=0, sorted=True, return_inverse=True)
        positions = torch.arange(keys.shape[0], dtype=torch.long, device=self.device)
        first = torch.full(
            (keys.shape[0],),
            keys.shape[0],
            dtype=torch.long,
            device=self.device,
        )
        first.scatter_reduce_(0, inverse, positions, reduce="amin", include_self=True)
        unique = first[inverse] == positions
        unique = unique[prefix:]
        eligible = valid & unique
        pool_positions = torch.arange(
            batch.action.shape[0], dtype=torch.long, device=self.device
        )
        ranked = torch.where(
            eligible,
            pool_positions,
            pool_positions + batch.action.shape[0],
        )
        keep = min(int(count), ranked.shape[0])
        selected_indices = torch.topk(ranked, k=keep, largest=False).indices
        selected = batch.index_select(selected_indices)
        selected_valid = eligible[selected_indices]
        if keep < int(count):
            selected = self._concat((selected, self._noop(int(count) - keep)))
            selected_valid = torch.cat(
                (
                    selected_valid,
                    torch.zeros(
                        int(count) - keep, dtype=torch.bool, device=self.device
                    ),
                )
            )
        selected.action = torch.where(
            selected_valid,
            selected.action,
            torch.full_like(selected.action, NOOP_ID),
        )
        selected.family = torch.where(
            selected_valid, selected.family, torch.zeros_like(selected.family)
        )
        selected.old_valid &= selected_valid[:, None]
        selected.new_valid &= selected_valid[:, None]
        return selected

    def canonical_ordered_candidates(
        self,
        count: int,
        generator: torch.Generator,
        *,
        role_phase: bool,
        targeted_fraction: float,
        closure_fraction: float,
        attempt_factor: int = 16,
    ) -> GpuEditBatch:
        """GPU proposal kernel matching canonical ordered-rejection semantics."""

        if int(count) < 2:
            raise ValueError("candidate count must be at least two")
        requested = int(count) - 1
        pool_size = max(256, requested * int(attempt_factor))
        pieces = [self._noop()]
        if role_phase:
            targeted = min(
                requested, int(round(requested * float(targeted_fraction)))
            )
            random_count = requested - targeted
            if targeted:
                pool, valid = self._targeted_relocation_pool(pool_size, generator)
                chosen = self._ordered_pick_valid(
                    pool, valid, targeted, exclude=self._concat(pieces)
                )
                pieces.append(chosen)
            if random_count:
                pool, valid = self._random_mixed_pool(pool_size, generator)
                chosen = self._ordered_pick_valid(
                    pool, valid, random_count, exclude=self._concat(pieces)
                )
                pieces.append(chosen)
            return self._concat(pieces)

        closure = min(
            requested, int(round(requested * float(closure_fraction)))
        )
        random_count = requested - closure
        if closure:
            pool, valid = self._wedge_seeded_switch_pool(pool_size, generator)
            chosen = self._ordered_pick_valid(
                pool, valid, closure, exclude=self._concat(pieces)
            )
            pieces.append(chosen)
        if random_count:
            pool, valid = self._random_switch_pool(pool_size, generator)
            chosen = self._ordered_pick_valid(
                pool, valid, random_count, exclude=self._concat(pieces)
            )
            pieces.append(chosen)
        return self._concat(pieces)

    def _stream_pick(
        self,
        *,
        kind: str,
        total_pool_size: int,
        chunk_size: int,
        count: int,
        generator: torch.Generator,
        priority: str,
    ) -> GpuEditBatch:
        """Keep a fixed-size GPU shortlist while streaming a large proposal pool."""

        if int(total_pool_size) < int(count):
            raise ValueError("proposal pool must be at least the shortlist size")
        if int(chunk_size) < int(count):
            raise ValueError("proposal chunk must be at least the shortlist size")
        best_batch = None
        best_valid = None
        best_score = None
        generated = 0
        while generated < int(total_pool_size):
            current = min(int(chunk_size), int(total_pool_size) - generated)
            if kind == "relocation":
                batch, valid = self._random_relocation_pool(current, generator)
            elif kind == "targeted_relocation":
                batch, valid = self._targeted_relocation_pool(current, generator)
            elif kind == "switch":
                batch, valid = self._random_switch_pool(current, generator)
            elif kind == "mixed":
                relocation_count = current // 2
                switch_count = current - relocation_count
                relocation, relocation_valid = self._random_relocation_pool(
                    relocation_count, generator
                )
                switch, switch_valid = self._random_switch_pool(
                    switch_count, generator
                )
                batch = self._concat((relocation, switch))
                valid = torch.cat((relocation_valid, switch_valid))
            else:
                raise ValueError(f"unknown GPU proposal stream {kind!r}")
            if priority == "random":
                score = torch.rand(current, device=self.device, generator=generator)
            elif priority == "role_gain":
                score = self.degree_gains(batch)
                valid &= score > 0
            elif priority == "closure":
                score = self._closure_components(batch)[0].sum(dim=1)
                # Random tie-breaking matters when a sparse state has many
                # zero-delta switches.
                score = score + 1.0e-5 * torch.rand(
                    current, device=self.device, generator=generator
                )
            else:
                raise ValueError(f"unknown GPU proposal priority {priority!r}")
            score = score.masked_fill(~valid, -torch.inf)
            if best_batch is not None:
                batch = self._concat((best_batch, batch))
                valid = torch.cat((best_valid, valid))
                score = torch.cat((best_score, score))
            keep = min(int(count), score.shape[0])
            indices = torch.topk(score, k=keep, largest=True).indices
            best_batch = batch.index_select(indices)
            best_valid = valid[indices]
            best_score = score[indices]
            generated += current
        assert best_batch is not None and best_valid is not None
        best_batch.action = torch.where(
            best_valid,
            best_batch.action,
            torch.full_like(best_batch.action, NOOP_ID),
        )
        best_batch.old_valid &= best_valid[:, None]
        best_batch.new_valid &= best_valid[:, None]
        return best_batch

    def random_candidates(
        self,
        count: int,
        generator: torch.Generator,
        *,
        role_phase: bool,
        targeted_fraction: float,
        closure_fraction: float,
        proposal_pool_size: int = 0,
        proposal_chunk_size: int = 1024,
        candidate_mode: str = "legacy_pool",
    ) -> GpuEditBatch:
        if int(count) < 2:
            raise ValueError("candidate count must be at least two")
        candidate_mode = str(candidate_mode).strip().lower()
        if candidate_mode == "canonical_ordered":
            return self.canonical_ordered_candidates(
                count,
                generator,
                role_phase=role_phase,
                targeted_fraction=targeted_fraction,
                closure_fraction=closure_fraction,
            )
        if candidate_mode != "legacy_pool":
            raise ValueError(f"unknown GPU candidate mode {candidate_mode!r}")
        requested = int(count) - 1
        pool_size = (
            max(256, requested * 32)
            if int(proposal_pool_size) <= 0
            else int(proposal_pool_size)
        )
        if int(proposal_pool_size) > 0:
            if role_phase:
                targeted = min(
                    requested, int(round(requested * float(targeted_fraction)))
                )
                random_count = requested - targeted
                pieces = [self._noop()]
                if targeted:
                    pieces.append(
                        self._stream_pick(
                            kind="targeted_relocation",
                            total_pool_size=pool_size,
                            chunk_size=proposal_chunk_size,
                            count=targeted,
                            generator=generator,
                            priority="role_gain",
                        )
                    )
                if random_count:
                    pieces.append(
                        self._stream_pick(
                            kind="mixed",
                            total_pool_size=pool_size,
                            chunk_size=proposal_chunk_size,
                            count=random_count,
                            generator=generator,
                            priority="random",
                        )
                    )
                return self._concat(pieces)
            closure = min(
                requested, int(round(requested * float(closure_fraction)))
            )
            random_count = requested - closure
            pieces = [self._noop()]
            if closure:
                pieces.append(
                    self._stream_pick(
                        kind="switch",
                        total_pool_size=pool_size,
                        chunk_size=proposal_chunk_size,
                        count=closure,
                        generator=generator,
                        priority="closure",
                    )
                )
            if random_count:
                pieces.append(
                    self._stream_pick(
                        kind="switch",
                        total_pool_size=pool_size,
                        chunk_size=proposal_chunk_size,
                        count=random_count,
                        generator=generator,
                        priority="random",
                    )
                )
            return self._concat(pieces)
        relocations, relocation_valid = self._random_relocation_pool(
            pool_size, generator
        )
        switches, switch_valid = self._random_switch_pool(pool_size, generator)
        if role_phase:
            targeted = min(requested, int(round(requested * float(targeted_fraction))))
            random_count = requested - targeted
            pieces = [self._noop()]
            if targeted:
                targeted_pool, targeted_valid = self._targeted_relocation_pool(
                    pool_size, generator
                )
                gains = self.degree_gains(targeted_pool)
                pieces.append(
                    self._pick_valid(
                        targeted_pool,
                        targeted_valid & (gains > 0),
                        targeted,
                        generator,
                        priority=gains,
                    )
                )
            if random_count:
                mixed = self._concat((relocations, switches))
                mixed_valid = torch.cat((relocation_valid, switch_valid))
                pieces.append(
                    self._pick_valid(mixed, mixed_valid, random_count, generator)
                )
            return self._concat(pieces)
        closure = min(requested, int(round(requested * float(closure_fraction))))
        random_count = requested - closure
        pieces = [self._noop()]
        if closure:
            closure_priority = self._closure_components(switches)[0].sum(dim=1)
            pieces.append(
                self._pick_valid(
                    switches,
                    switch_valid,
                    closure,
                    generator,
                    priority=closure_priority,
                )
            )
        if random_count:
            pieces.append(
                self._pick_valid(
                    switches, switch_valid, random_count, generator
                )
            )
        return self._concat(pieces)

    def _neighbors_after(
        self, nodes: torch.Tensor, batch: GpuEditBatch
    ) -> torch.Tensor:
        rows = self.adjacency[nodes].clone()
        candidate = torch.arange(nodes.shape[0], device=self.device)
        for slot in range(2):
            edge = batch.old_edges[:, slot]
            valid = batch.old_valid[:, slot]
            left_match = valid & (edge[:, 0] == nodes)
            right_match = valid & (edge[:, 1] == nodes)
            rows[candidate[left_match], edge[left_match, 1]] = False
            rows[candidate[right_match], edge[right_match, 0]] = False
        for slot in range(2):
            edge = batch.new_edges[:, slot]
            valid = batch.new_valid[:, slot]
            left_match = valid & (edge[:, 0] == nodes)
            right_match = valid & (edge[:, 1] == nodes)
            rows[candidate[left_match], edge[left_match, 1]] = True
            rows[candidate[right_match], edge[right_match, 0]] = True
        return rows

    def _slot_common(
        self, batch: GpuEditBatch, *, new: bool, slot: int
    ) -> torch.Tensor:
        edges = batch.new_edges[:, slot] if new else batch.old_edges[:, slot]
        valid = batch.new_valid[:, slot] if new else batch.old_valid[:, slot]
        safe = edges.clamp_min(0)
        if new:
            common = self._neighbors_after(safe[:, 0], batch) & self._neighbors_after(
                safe[:, 1], batch
            )
        else:
            common = self.adjacency[safe[:, 0]] & self.adjacency[safe[:, 1]]
        common &= valid[:, None]
        return common

    def _closure_components(
        self, batch: GpuEditBatch
    ) -> tuple[torch.Tensor, torch.Tensor]:
        slot_counts, normalized = self._closure_slots(batch)
        typed = -slot_counts[:, 0] - slot_counts[:, 1]
        typed += slot_counts[:, 2] + slot_counts[:, 3]
        return typed, normalized

    def _closure_slots(
        self, batch: GpuEditBatch
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.use_triton:
            return fused_typed_closure(
                self.adjacency,
                self.node_types,
                self.total_degree,
                batch.action,
                batch.old_edges,
                batch.new_edges,
                batch.old_valid,
                batch.new_valid,
                self.type_count,
            )
        return self._closure_slots_eager(batch)

    def _closure_slots_eager(
        self, batch: GpuEditBatch
    ) -> tuple[torch.Tensor, torch.Tensor]:
        count = batch.action.shape[0]
        slot_counts = torch.zeros(
            (count, 4, self.type_count),
            dtype=torch.float32,
            device=self.device,
        )
        triangle_delta = torch.zeros(
            (count, self.num_nodes), dtype=torch.float32, device=self.device
        )
        candidate = torch.arange(count, device=self.device)
        output_slot = 0
        for new, sign in ((False, -1.0), (True, 1.0)):
            edges = batch.new_edges if new else batch.old_edges
            valid_slots = batch.new_valid if new else batch.old_valid
            for slot in range(2):
                common = self._slot_common(batch, new=new, slot=slot)
                slot_counts[:, output_slot] = (
                    common.to(torch.float32) @ self.node_type_one_hot
                )
                counts = common.sum(dim=1).to(torch.float32) * valid_slots[:, slot]
                safe = edges[:, slot].clamp_min(0)
                triangle_delta[candidate, safe[:, 0]] += sign * counts
                triangle_delta[candidate, safe[:, 1]] += sign * counts
                triangle_delta += sign * common.to(torch.float32)
                output_slot += 1
        denominator = self.total_degree.to(torch.float32)
        denominator = denominator * (denominator - 1.0) / 2.0
        normalized = torch.where(
            denominator[None] > 0,
            triangle_delta / denominator.clamp_min(1.0)[None],
            torch.zeros_like(triangle_delta),
        ).sum(dim=1)
        degree_preserving = self._endpoint_deltas(batch).eq(0).all(dim=1)
        return slot_counts, normalized * degree_preserving

    def _closure_components_eager(
        self, batch: GpuEditBatch
    ) -> tuple[torch.Tensor, torch.Tensor]:
        slot_counts, normalized = self._closure_slots_eager(batch)
        typed = -slot_counts[:, 0] - slot_counts[:, 1]
        typed += slot_counts[:, 2] + slot_counts[:, 3]
        return typed, normalized

    def _endpoint_deltas(self, batch: GpuEditBatch) -> torch.Tensor:
        delta = torch.zeros(
            (batch.action.shape[0], self.num_nodes),
            dtype=torch.long,
            device=self.device,
        )
        for edges, valid, sign in (
            (batch.old_edges, batch.old_valid, -1),
            (batch.new_edges, batch.new_valid, 1),
        ):
            for slot in range(2):
                safe = edges[:, slot].clamp_min(0)
                value = valid[:, slot].to(torch.long) * sign
                delta.scatter_add_(1, safe[:, :1], value[:, None])
                delta.scatter_add_(1, safe[:, 1:], value[:, None])
        return delta

    def degree_gains(self, batch: GpuEditBatch) -> torch.Tensor:
        # Degree rows are integer-valued. Keep their global squared-error
        # ledger in float64 so large graphs do not accumulate float32 drift
        # across many ordered edits. Neural scoring casts this gain back to the
        # scorer dtype after the exact ledger update.
        delta = self._endpoint_deltas(batch).to(torch.float64)
        current = self.family_degree[batch.family].to(torch.float64)
        target = self.target_rows.transpose(0, 1)[batch.family].to(torch.float64)
        before = current - target
        return (before.square() - (before + delta).square()).sum(dim=1)

    def compact_features(
        self, batch: GpuEditBatch, time_fraction: float, model_input_dim: int
    ) -> torch.Tensor:
        count = batch.action.shape[0]
        action = F.one_hot(batch.action, num_classes=len(SCORER_ACTIONS)).to(torch.float32)
        family = F.one_hot(batch.family, num_classes=self.family_count).to(torch.float32)
        family *= (batch.action != NOOP_ID)[:, None]
        endpoint_rows = []
        slot_counts, normalized_closure = self._closure_slots(batch)
        common_by_slot = slot_counts.sum(dim=2)
        for new in (False, True):
            edges = batch.new_edges if new else batch.old_edges
            valid_slots = batch.new_valid if new else batch.old_valid
            for slot in range(2):
                safe = edges[:, slot].clamp_min(0)
                valid = valid_slots[:, slot].to(torch.float32)
                family_degree = self.family_degree[batch.family]
                values = torch.stack(
                    (
                        torch.log1p(self.total_degree[safe[:, 0]].to(torch.float32)),
                        torch.log1p(self.total_degree[safe[:, 1]].to(torch.float32)),
                        torch.log1p(family_degree.gather(1, safe[:, :1]).squeeze(1).to(torch.float32)),
                        torch.log1p(family_degree.gather(1, safe[:, 1:]).squeeze(1).to(torch.float32)),
                    ),
                    dim=1,
                )
                endpoint_rows.append(values / self.degree_scale * valid[:, None])
        endpoint = torch.stack(endpoint_rows, dim=1).reshape(count, 16)
        common = common_by_slot / self.common_scale
        density = self.family_edge_counts[batch.family].to(torch.float32) / self.family_capacities[
            batch.family
        ].clamp_min(1.0)
        scalar = torch.stack(
            (
                torch.full((count,), float(time_fraction), device=self.device),
                density * (batch.action != NOOP_ID),
                batch.old_valid.sum(dim=1).to(torch.float32),
                batch.new_valid.sum(dim=1).to(torch.float32),
            ),
            dim=1,
        )
        local = torch.cat((action, family, endpoint, common, scalar), dim=1)
        if int(model_input_dim) == local.shape[1]:
            return local
        expected = mixed_edit_feature_width(
            "typed_closure_compact_v2", self.family_count, self.type_count
        )
        if int(model_input_dim) != expected:
            raise ValueError(
                "GPU backend supports legacy_local_v1 and typed_closure_compact_v2; "
                f"checkpoint input width is {model_input_dim}"
            )
        typed = -slot_counts[:, 0] - slot_counts[:, 1]
        typed += slot_counts[:, 2] + slot_counts[:, 3]
        typed = typed / self.common_scale
        closure = torch.cat(
            (
                typed,
                typed.sum(dim=1, keepdim=True),
                normalized_closure[:, None],
            ),
            dim=1,
        )
        return torch.cat((local, closure), dim=1)

    def commit(self, batch: GpuEditBatch, selected: torch.Tensor) -> None:
        old_edges = batch.old_edges[selected]
        new_edges = batch.new_edges[selected]
        old_valid = batch.old_valid[selected]
        new_valid = batch.new_valid[selected]
        family = batch.family[selected]
        indices = batch.old_indices[selected]
        for slot in range(2):
            valid = old_valid[slot]
            edge = old_edges[slot].clamp_min(0)
            old_value = self.adjacency[edge[0], edge[1]]
            value = torch.where(valid, torch.zeros_like(old_value), old_value)
            self.adjacency[edge[0], edge[1]] = value
            self.adjacency[edge[1], edge[0]] = value
            previous_slot = self.edge_slots[edge[0], edge[1]]
            slot_value = torch.where(
                valid,
                torch.full_like(previous_slot, -1),
                previous_slot,
            )
            self.edge_slots[edge[0], edge[1]] = slot_value
            self.edge_slots[edge[1], edge[0]] = slot_value
            delta = valid.to(torch.long).expand(2).neg()
            self.total_degree.scatter_add_(0, edge, delta)
            self.family_degree[family].scatter_add_(0, edge, delta)
        for slot in range(2):
            valid = new_valid[slot]
            edge = new_edges[slot].clamp_min(0)
            old_value = self.adjacency[edge[0], edge[1]]
            value = torch.where(valid, torch.ones_like(old_value), old_value)
            self.adjacency[edge[0], edge[1]] = value
            self.adjacency[edge[1], edge[0]] = value
            delta = valid.to(torch.long).expand(2)
            self.total_degree.scatter_add_(0, edge, delta)
            self.family_degree[family].scatter_add_(0, edge, delta)
            edge_index = indices[slot].clamp_min(0)
            previous = self.edge_tables[family, edge_index]
            self.edge_tables[family, edge_index] = torch.where(
                valid, edge, previous
            )
            previous_slot = self.edge_slots[edge[0], edge[1]]
            slot_value = torch.where(valid, edge_index, previous_slot)
            self.edge_slots[edge[0], edge[1]] = slot_value
            self.edge_slots[edge[1], edge[0]] = slot_value

    def relation_degree_error(self) -> torch.Tensor:
        current = self.family_degree.transpose(0, 1).to(torch.float64)
        target = self.target_rows.to(torch.float64)
        return (current - target).square().sum()

    def context_node_features(self, time_fraction: float) -> torch.Tensor:
        columns = [
            torch.log1p(self.total_degree.to(torch.float32)) / self.degree_scale
        ]
        columns.extend(
            torch.log1p(self.family_degree[index].to(torch.float32))
            / self.degree_scale
            for index in range(self.family_count)
        )
        columns.extend(
            (
                (self.total_degree == 0).to(torch.float32),
                torch.full(
                    (self.num_nodes,),
                    float(time_fraction),
                    dtype=torch.float32,
                    device=self.device,
                ),
            )
        )
        return torch.stack(columns, dim=1)

    def context_edge_tensors(self) -> tuple[torch.Tensor, torch.Tensor]:
        edges = []
        relations = []
        for family_id in range(self.family_count):
            count = self.family_edge_count_values[family_id]
            table = self.edge_tables[family_id, :count]
            edges.append(torch.cat((table, table.flip(1)), dim=0))
            relations.append(
                torch.full(
                    (2 * int(count),),
                    family_id,
                    dtype=torch.long,
                    device=self.device,
                )
            )
        directed = torch.cat(edges, dim=0)
        return directed.transpose(0, 1), torch.cat(relations)

    def undirected_edges(self) -> torch.Tensor:
        return torch.cat(
            [
                self.edge_tables[index, : self.family_edge_count_values[index]]
                for index in range(self.family_count)
            ],
            dim=0,
        )

    def gpu_graph_metrics(self, edge_chunk_size: int = 2048) -> dict[str, float]:
        """Compute the paper trace metrics from the resident CUDA state."""

        edges = self.undirected_edges()
        triangles_per_node = torch.zeros(
            self.num_nodes, dtype=torch.float64, device=self.device
        )
        common_total = torch.zeros((), dtype=torch.float64, device=self.device)
        for start in range(0, edges.shape[0], int(edge_chunk_size)):
            chunk = edges[start : start + int(edge_chunk_size)]
            common = self.adjacency[chunk[:, 0]] & self.adjacency[chunk[:, 1]]
            common_float = common.to(torch.float64)
            common_total += common_float.sum()
            # Every triangle has exactly one edge opposite each participating
            # node, so summing common-neighbor indicators gives exact local
            # triangle counts without endpoint correction.
            triangles_per_node += common_float.sum(dim=0)
        triangles = common_total / 3.0
        degree = self.total_degree.to(torch.float64)
        denominator = degree * (degree - 1.0) / 2.0
        local_clustering = torch.where(
            denominator > 0,
            triangles_per_node / denominator.clamp_min(1.0),
            torch.zeros_like(degree),
        )
        wedges = denominator.sum()
        transitivity = torch.where(
            wedges > 0,
            3.0 * triangles / wedges.clamp_min(1.0),
            torch.zeros_like(wedges),
        )

        endpoint_degree = degree[edges]
        left = torch.cat((endpoint_degree[:, 0], endpoint_degree[:, 1]))
        right = torch.cat((endpoint_degree[:, 1], endpoint_degree[:, 0]))
        left_centered = left - left.mean()
        right_centered = right - right.mean()
        covariance = (left_centered * right_centered).sum()
        scale = torch.sqrt(
            left_centered.square().sum() * right_centered.square().sum()
        )
        assortativity = torch.where(
            scale > 0, covariance / scale.clamp_min(1.0e-12), torch.zeros_like(scale)
        )

        # GPU label propagation is only called at sparse trace points.  The
        # one-scalar convergence check does not reintroduce hot-loop traffic.
        labels = torch.arange(self.num_nodes, dtype=torch.long, device=self.device)
        for _ in range(self.num_nodes):
            old = labels
            low = torch.minimum(labels[edges[:, 0]], labels[edges[:, 1]])
            updated = labels.clone()
            updated.scatter_reduce_(0, edges[:, 0], low, reduce="amin", include_self=True)
            updated.scatter_reduce_(0, edges[:, 1], low, reduce="amin", include_self=True)
            updated = updated[updated]
            labels = updated
            if bool(torch.equal(labels, old)):
                break
        component_size = torch.bincount(labels, minlength=self.num_nodes)
        lcc = component_size.max().to(torch.float64)
        packed = torch.stack(
            (
                torch.tensor(float(edges.shape[0]), dtype=torch.float64, device=self.device),
                triangles,
                local_clustering.mean(),
                transitivity,
                lcc,
                assortativity,
                degree.square().sum(),
                degree.max(),
            )
        ).cpu().tolist()
        names = (
            "edges",
            "triangles",
            "clustering",
            "transitivity",
            "lcc",
            "assortativity",
            "sum_degree_sq",
            "max_degree",
        )
        return {name: float(value) for name, value in zip(names, packed)}

    @staticmethod
    def candidate_endpoints(batch: GpuEditBatch) -> torch.Tensor:
        return torch.cat((batch.old_edges, batch.new_edges), dim=1).reshape(
            batch.action.shape[0], 8
        )

    def to_graph(self, template: TypedGraph) -> TypedGraph:
        family_edges = {}
        for family_id, family in enumerate(self.family_names):
            count = int(self.family_edge_counts[family_id].cpu())
            family_edges[family] = [
                tuple(map(int, edge))
                for edge in self.edge_tables[family_id, :count].cpu().tolist()
            ]
        return TypedGraph(
            template.node_types.copy(),
            template.type_names,
            template.family_specs,
            family_edges,
        )


def gpu_features_for_proposals(
    graph: TypedGraph,
    proposals: Sequence[EditProposal],
    target_rows: np.ndarray,
    time_fraction: float,
    device: str | torch.device,
    model_input_dim: int,
) -> torch.Tensor:
    """Test/diagnostic bridge for exact CPU-vs-GPU feature parity."""

    state = GpuTypedGraphState(graph, target_rows, device)
    family_to_id = {name: index for index, name in enumerate(state.family_names)}
    actions = torch.tensor(
        [SCORER_ACTIONS.index(item.action) for item in proposals],
        dtype=torch.long,
        device=state.device,
    )
    families = torch.tensor(
        [family_to_id[item.family] for item in proposals],
        dtype=torch.long,
        device=state.device,
    )
    old_edges = torch.full(
        (len(proposals), 2, 2), -1, dtype=torch.long, device=state.device
    )
    new_edges = old_edges.clone()
    old_indices = torch.full(
        (len(proposals), 2), -1, dtype=torch.long, device=state.device
    )
    old_valid = torch.zeros((len(proposals), 2), dtype=torch.bool, device=state.device)
    new_valid = old_valid.clone()
    for row, proposal in enumerate(proposals):
        edges = graph.family_edges[proposal.family]
        for slot, edge in enumerate(proposal.old_edges):
            old_edges[row, slot] = torch.tensor(edge, device=state.device)
            old_indices[row, slot] = edges.index(tuple(edge))
            old_valid[row, slot] = True
        for slot, edge in enumerate(proposal.new_edges):
            new_edges[row, slot] = torch.tensor(edge, device=state.device)
            new_valid[row, slot] = True
    batch = GpuEditBatch(
        actions, families, old_edges, new_edges, old_indices, old_valid, new_valid
    )
    return state.compact_features(batch, time_fraction, model_input_dim)


def sample_gpu_role_guided_edit_diffusion(
    initial_graph: TypedGraph,
    model: torch.nn.Module,
    target_rows: np.ndarray,
    *,
    steps: int,
    candidates_per_step: int,
    seed: int,
    device: str | torch.device,
    metrics_every: int,
    role_weight: float,
    gain_scale: float,
    schedule_power: float,
    schedule_floor: float,
    role_error_per_node_threshold: float,
    model_time_horizon_steps: int,
    model_time_start_step: int,
    closure_candidate_fraction: float,
    targeted_role_candidate_fraction: float,
    auto_role_balance: float = 0.0,
    role_calibration_batches: int = 32,
    context_model: torch.nn.Module | None = None,
    context_refresh_steps: int = 20,
    proposal_pool_size: int = 0,
    proposal_chunk_size: int = 1024,
    candidate_mode: str = "legacy_pool",
    metric_edge_chunk_size: int = 2048,
    phase_check_interval: int = 32,
    use_triton: bool = True,
) -> MixedEditSamplingResult:
    """Run fixed-count role-guided edit diffusion without hot-loop host traffic."""

    if int(steps) <= 0 or int(model_time_horizon_steps) <= 0:
        raise ValueError("steps and model time horizon must be positive")
    if int(phase_check_interval) <= 0:
        raise ValueError("GPU phase check interval must be positive")
    state = GpuTypedGraphState(
        initial_graph, target_rows, device, use_triton=use_triton
    )
    model = model.to(state.device).eval()
    if context_model is not None:
        if int(context_refresh_steps) <= 0:
            raise ValueError("context refresh interval must be positive")
        context_model = context_model.to(state.device).eval()
    model_input_dim = int(model.input_dim)
    generator = torch.Generator(device=state.device)
    generator.manual_seed(int(seed))

    effective_role_weight = float(role_weight)
    with torch.no_grad():
        if float(auto_role_balance) > 0.0:
            logits = []
            gains = []
            for _ in range(int(role_calibration_batches)):
                candidates = state.random_candidates(
                    candidates_per_step,
                    generator,
                    role_phase=True,
                    targeted_fraction=0.0,
                    closure_fraction=0.0,
                    proposal_pool_size=0,
                    candidate_mode=candidate_mode,
                )
                matrix = state.compact_features(candidates, 1.0, model_input_dim)
                logits.append(model(matrix[None]).squeeze(0))
                gains.append(state.degree_gains(candidates).abs())
            flat_logits = torch.cat(logits)
            positive_gains = torch.cat(gains)
            positive_gains = positive_gains[positive_gains > 0]
            if positive_gains.numel() == 0:
                raise ValueError("automatic role calibration found no nonzero gains")
            logit_scale = flat_logits.std().clamp_min(1.0e-6)
            gain_value = torch.quantile(positive_gains, 0.9).clamp_min(1.0e-6)
            # This one scalar read happens before the sampling loop.
            effective_role_weight = float(
                (
                    float(auto_role_balance)
                    * logit_scale
                    * float(gain_scale)
                    / gain_value
                ).cpu()
            )

        role_error = state.relation_degree_error()
        threshold = float(role_error_per_node_threshold) * state.num_nodes
        role_phase = bool((role_error > threshold).cpu())
        initial_metrics = state.gpu_graph_metrics(metric_edge_chunk_size)
        trace = [
            {
                "step": 0.0,
                "relation_degree_error": float(role_error.cpu()),
                "relation_degree_error_per_node": float(role_error.cpu())
                / state.num_nodes,
                "phase": "role_alignment" if role_phase else "switch_refinement",
                "role_phase_transition_step": (
                    None if role_phase else 0
                ),
                "effective_role_weight": effective_role_weight,
                "sampling_backend": "gpu_resident",
                **initial_metrics,
            }
        ]
        action_counts = torch.zeros(
            len(SCORER_ACTIONS), dtype=torch.long, device=state.device
        )
        applied = torch.zeros((), dtype=torch.long, device=state.device)
        transition_step = -1 if role_phase else 0
        context_hidden = None
        context_role_conditions = None
        if context_model is not None and getattr(
            context_model, "role_condition_dim", 0
        ) > 0:
            context_role_conditions = context_model.normalize_role_rows(
                torch.from_numpy(np.asarray(target_rows))[None]
                .float()
                .to(state.device),
                state.node_types[None],
            )

        for completed in range(1, int(steps) + 1):
            time_fraction = max(
                0.0,
                1.0
                - (int(model_time_start_step) + completed - 1)
                / float(model_time_horizon_steps),
            )
            candidates = state.random_candidates(
                candidates_per_step,
                generator,
                role_phase=role_phase,
                targeted_fraction=(
                    targeted_role_candidate_fraction if role_phase else 0.0
                ),
                closure_fraction=(
                    0.0 if role_phase else closure_candidate_fraction
                ),
                proposal_pool_size=proposal_pool_size,
                proposal_chunk_size=proposal_chunk_size,
                candidate_mode=candidate_mode,
            )
            features = state.compact_features(
                candidates, time_fraction, model_input_dim
            )
            structure_logits = model(features[None]).squeeze(0)
            if context_model is not None and not role_phase:
                if context_hidden is None or (
                    (completed - 1) % int(context_refresh_steps) == 0
                ):
                    edge_index, edge_types = state.context_edge_tensors()
                    context_hidden = context_model.encode(
                        state.context_node_features(time_fraction)[None],
                        state.node_types[None],
                        edge_index,
                        edge_types,
                        role_conditions=context_role_conditions,
                    )
                context_logits = context_model.score_encoded(
                    context_hidden,
                    features[None],
                    state.candidate_endpoints(candidates)[None],
                ).squeeze(0)
                structure_logits = context_logits
            degree_gains = state.degree_gains(candidates)
            annealed = float(time_fraction) ** float(schedule_power)
            multiplier = float(schedule_floor) + (1.0 - float(schedule_floor)) * annealed
            scores = structure_logits + (
                float(effective_role_weight)
                * multiplier
                * degree_gains.to(structure_logits.dtype)
                / float(gain_scale)
            )
            selected = scores.argmax()
            selected_action = candidates.action[selected]
            state.commit(candidates, selected)
            role_error = (role_error - degree_gains[selected]).clamp_min(0.0)
            if role_phase and (
                completed % int(phase_check_interval) == 0
                or completed == int(steps)
            ):
                if float(role_error.cpu()) <= threshold:
                    role_phase = False
                    transition_step = completed
            action_counts.scatter_add_(
                0, selected_action[None], torch.ones(1, dtype=torch.long, device=state.device)
            )
            applied += (selected_action != NOOP_ID).to(torch.long)

            if completed == int(steps) or (
                int(metrics_every) > 0 and completed % int(metrics_every) == 0
            ):
                error_value = float(role_error.cpu())
                trace.append(
                    {
                        "step": float(completed),
                        "relation_degree_error": error_value,
                        "relation_degree_error_per_node": error_value / state.num_nodes,
                        "phase": (
                            "role_alignment"
                            if role_phase
                            else "switch_refinement"
                        ),
                        "role_phase_transition_step": transition_step,
                        "effective_role_weight": effective_role_weight,
                        "sampling_backend": "gpu_resident",
                        **state.gpu_graph_metrics(metric_edge_chunk_size),
                    }
                )

    final_graph = state.to_graph(initial_graph)
    final_graph.validate()
    recomputed_role_error = state.relation_degree_error()
    if not torch.isclose(role_error, recomputed_role_error, rtol=0.0, atol=1.0e-4):
        raise RuntimeError("GPU incremental relation-degree error drifted from state")
    counts = action_counts.cpu().tolist()
    return MixedEditSamplingResult(
        graph=final_graph,
        trace=trace,
        action_counts={name: int(counts[index]) for index, name in enumerate(SCORER_ACTIONS)},
        proposed=int(steps) * int(candidates_per_step),
        applied=int(applied.cpu()),
    )


def _encode_batched_context(
    states: Sequence[GpuTypedGraphState],
    context_model: torch.nn.Module,
    role_conditions: torch.Tensor | None,
    time_fraction: float,
) -> torch.Tensor:
    """Encode independent chains as one block-diagonal sparse HGT batch."""

    if not states:
        raise ValueError("batched context needs at least one graph state")
    node_count = states[0].num_nodes
    if any(state.num_nodes != node_count for state in states):
        raise ValueError("batched context requires equal node counts")
    node_features = torch.stack(
        [state.context_node_features(time_fraction) for state in states]
    )
    node_types = torch.stack([state.node_types for state in states])
    edge_parts = []
    type_parts = []
    for batch_id, state in enumerate(states):
        edge_index, edge_types = state.context_edge_tensors()
        edge_parts.append(edge_index + batch_id * node_count)
        type_parts.append(edge_types)
    return context_model.encode(
        node_features,
        node_types,
        torch.cat(edge_parts, dim=1),
        torch.cat(type_parts),
        role_conditions=role_conditions,
    )


def sample_gpu_role_guided_edit_diffusion_batched(
    initial_graphs: Sequence[TypedGraph],
    model: torch.nn.Module,
    target_rows: Sequence[np.ndarray],
    *,
    steps: int,
    candidates_per_step: int,
    seeds: Sequence[int],
    device: str | torch.device = "cuda",
    metrics_every: int = 100,
    role_weight: float = 0.1,
    gain_scale: float = 10.0,
    schedule_power: float = 1.0,
    schedule_floor: float = 0.0,
    role_error_per_node_threshold: float = 0.0,
    model_time_horizon_steps: int = 1,
    model_time_start_step: int = 0,
    closure_candidate_fraction: float = 0.0,
    targeted_role_candidate_fraction: float = 0.0,
    auto_role_balance: float = 0.0,
    role_calibration_batches: int = 32,
    context_model: torch.nn.Module | None = None,
    context_refresh_steps: int = 20,
    proposal_pool_size: int = 0,
    proposal_chunk_size: int = 1024,
    candidate_mode: str = "legacy_pool",
    metric_edge_chunk_size: int = 2048,
    phase_check_interval: int = 32,
    use_triton: bool = True,
) -> list[MixedEditSamplingResult]:
    """Run independent edit chains with shared batched MLP/HGT forwards.

    Candidate construction, random streams, selection, and mutation remain
    chain-local. Only dense neural operations are batched, so no graph can send
    messages to another graph and every chain retains the single-chain
    transition semantics.
    """

    chain_count = len(initial_graphs)
    if chain_count == 0:
        raise ValueError("batched sampling needs at least one chain")
    if len(target_rows) != chain_count or len(seeds) != chain_count:
        raise ValueError("graphs, role rows, and seeds must have equal length")
    if int(steps) <= 0 or int(model_time_horizon_steps) <= 0:
        raise ValueError("steps and model time horizon must be positive")
    if int(phase_check_interval) <= 0:
        raise ValueError("GPU phase check interval must be positive")
    if context_model is not None and int(context_refresh_steps) <= 0:
        raise ValueError("context refresh interval must be positive")

    reference = initial_graphs[0]
    for graph in initial_graphs[1:]:
        if (
            graph.num_nodes != reference.num_nodes
            or graph.type_names != reference.type_names
            or tuple(sorted(graph.family_specs))
            != tuple(sorted(reference.family_specs))
            or not np.array_equal(graph.node_types, reference.node_types)
        ):
            raise ValueError(
                "batched chains must share node count, schema, and node types"
            )

    states = [
        GpuTypedGraphState(graph, rows, device, use_triton=use_triton)
        for graph, rows in zip(initial_graphs, target_rows)
    ]
    model = model.to(states[0].device).eval()
    if context_model is not None:
        context_model = context_model.to(states[0].device).eval()
    model_input_dim = int(model.input_dim)
    generators = []
    for seed in seeds:
        generator = torch.Generator(device=states[0].device)
        generator.manual_seed(int(seed))
        generators.append(generator)

    effective_role_weights = []
    with torch.no_grad():
        for state, generator in zip(states, generators):
            effective = float(role_weight)
            if float(auto_role_balance) > 0.0:
                logits = []
                gains = []
                for _ in range(int(role_calibration_batches)):
                    candidates = state.random_candidates(
                        candidates_per_step,
                        generator,
                        role_phase=True,
                        targeted_fraction=0.0,
                        closure_fraction=0.0,
                        proposal_pool_size=0,
                        candidate_mode=candidate_mode,
                    )
                    matrix = state.compact_features(
                        candidates, 1.0, model_input_dim
                    )
                    logits.append(model(matrix[None]).squeeze(0))
                    gains.append(state.degree_gains(candidates).abs())
                flat_logits = torch.cat(logits)
                positive_gains = torch.cat(gains)
                positive_gains = positive_gains[positive_gains > 0]
                if positive_gains.numel() == 0:
                    raise ValueError(
                        "automatic role calibration found no nonzero gains"
                    )
                effective = float(
                    (
                        float(auto_role_balance)
                        * flat_logits.std().clamp_min(1.0e-6)
                        * float(gain_scale)
                        / torch.quantile(positive_gains, 0.9).clamp_min(1.0e-6)
                    ).cpu()
                )
            effective_role_weights.append(effective)

        role_errors = [state.relation_degree_error() for state in states]
        thresholds = [
            float(role_error_per_node_threshold) * state.num_nodes
            for state in states
        ]
        role_phases = [
            bool((error > threshold).cpu())
            for error, threshold in zip(role_errors, thresholds)
        ]
        transition_steps = [-1 if phase else 0 for phase in role_phases]
        traces = []
        for state, error, phase, transition, effective in zip(
            states,
            role_errors,
            role_phases,
            transition_steps,
            effective_role_weights,
        ):
            error_value = float(error.cpu())
            traces.append(
                [
                    {
                        "step": 0.0,
                        "relation_degree_error": error_value,
                        "relation_degree_error_per_node": error_value
                        / state.num_nodes,
                        "phase": (
                            "role_alignment" if phase else "switch_refinement"
                        ),
                        "role_phase_transition_step": (
                            None if phase else transition
                        ),
                        "effective_role_weight": effective,
                        "sampling_backend": "gpu_resident_batched",
                        **state.gpu_graph_metrics(metric_edge_chunk_size),
                    }
                ]
            )

        action_counts = torch.zeros(
            (chain_count, len(SCORER_ACTIONS)),
            dtype=torch.long,
            device=states[0].device,
        )
        applied = torch.zeros(
            chain_count, dtype=torch.long, device=states[0].device
        )
        context_hidden = None
        context_role_conditions = None
        if context_model is not None and getattr(
            context_model, "role_condition_dim", 0
        ) > 0:
            context_role_conditions = context_model.normalize_role_rows(
                torch.stack([state.target_rows for state in states]),
                torch.stack([state.node_types for state in states]),
            )

        for completed in range(1, int(steps) + 1):
            time_fraction = max(
                0.0,
                1.0
                - (int(model_time_start_step) + completed - 1)
                / float(model_time_horizon_steps),
            )
            candidate_batches = [
                state.random_candidates(
                    candidates_per_step,
                    generator,
                    role_phase=phase,
                    targeted_fraction=(
                        targeted_role_candidate_fraction if phase else 0.0
                    ),
                    closure_fraction=(
                        0.0 if phase else closure_candidate_fraction
                    ),
                    proposal_pool_size=proposal_pool_size,
                    proposal_chunk_size=proposal_chunk_size,
                    candidate_mode=candidate_mode,
                )
                for state, generator, phase in zip(
                    states, generators, role_phases
                )
            ]
            features = torch.stack(
                [
                    state.compact_features(
                        candidates, time_fraction, model_input_dim
                    )
                    for state, candidates in zip(states, candidate_batches)
                ]
            )
            structure_logits = model(features)
            if context_model is not None and not all(role_phases):
                if context_hidden is None or (
                    (completed - 1) % int(context_refresh_steps) == 0
                ):
                    context_hidden = _encode_batched_context(
                        states,
                        context_model,
                        context_role_conditions,
                        time_fraction,
                    )
                endpoints = torch.stack(
                    [
                        state.candidate_endpoints(candidates)
                        for state, candidates in zip(states, candidate_batches)
                    ]
                )
                context_logits = context_model.score_encoded(
                    context_hidden, features, endpoints
                )
                phase_mask = torch.tensor(
                    role_phases, dtype=torch.bool, device=states[0].device
                )[:, None]
                structure_logits = torch.where(
                    phase_mask, structure_logits, context_logits
                )

            degree_gains = torch.stack(
                [
                    state.degree_gains(candidates)
                    for state, candidates in zip(states, candidate_batches)
                ]
            )
            annealed = float(time_fraction) ** float(schedule_power)
            multiplier = float(schedule_floor) + (
                1.0 - float(schedule_floor)
            ) * annealed
            weights = torch.tensor(
                effective_role_weights,
                dtype=structure_logits.dtype,
                device=states[0].device,
            )[:, None]
            scores = structure_logits + (
                weights
                * multiplier
                * degree_gains.to(structure_logits.dtype)
                / float(gain_scale)
            )
            selected = scores.argmax(dim=1)

            for chain, (state, candidates) in enumerate(
                zip(states, candidate_batches)
            ):
                selected_index = selected[chain]
                selected_action = candidates.action[selected_index]
                state.commit(candidates, selected_index)
                role_errors[chain] = (
                    role_errors[chain] - degree_gains[chain, selected_index]
                ).clamp_min(0.0)
                if role_phases[chain] and (
                    completed % int(phase_check_interval) == 0
                    or completed == int(steps)
                ):
                    if float(role_errors[chain].cpu()) <= thresholds[chain]:
                        role_phases[chain] = False
                        transition_steps[chain] = completed
                action_counts[chain].scatter_add_(
                    0,
                    selected_action[None],
                    torch.ones(1, dtype=torch.long, device=state.device),
                )
                applied[chain] += (selected_action != NOOP_ID).to(torch.long)

            if completed == int(steps) or (
                int(metrics_every) > 0 and completed % int(metrics_every) == 0
            ):
                for chain, state in enumerate(states):
                    error_value = float(role_errors[chain].cpu())
                    traces[chain].append(
                        {
                            "step": float(completed),
                            "relation_degree_error": error_value,
                            "relation_degree_error_per_node": error_value
                            / state.num_nodes,
                            "phase": (
                                "role_alignment"
                                if role_phases[chain]
                                else "switch_refinement"
                            ),
                            "role_phase_transition_step": transition_steps[chain],
                            "effective_role_weight": effective_role_weights[chain],
                            "sampling_backend": "gpu_resident_batched",
                            **state.gpu_graph_metrics(metric_edge_chunk_size),
                        }
                    )

    results = []
    for chain, (state, initial_graph) in enumerate(zip(states, initial_graphs)):
        final_graph = state.to_graph(initial_graph)
        final_graph.validate()
        recomputed = state.relation_degree_error()
        if not torch.isclose(
            role_errors[chain], recomputed, rtol=0.0, atol=1.0e-4
        ):
            raise RuntimeError(
                "batched GPU relation-degree error drifted from state"
            )
        counts = action_counts[chain].cpu().tolist()
        results.append(
            MixedEditSamplingResult(
                graph=final_graph,
                trace=traces[chain],
                action_counts={
                    name: int(counts[index])
                    for index, name in enumerate(SCORER_ACTIONS)
                },
                proposed=int(steps) * int(candidates_per_step),
                applied=int(applied[chain].cpu()),
            )
        )
    return results
