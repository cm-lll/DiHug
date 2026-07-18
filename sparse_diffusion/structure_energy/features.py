from __future__ import annotations

import itertools
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from typing import Mapping, MutableMapping, Optional, Sequence, Set, Tuple

import numpy as np

from .graph import Edge, TypedGraph


FeatureCounter = Counter[str]
Triple = Tuple[int, int, int]
GWESP_DECAY = 0.5


@lru_cache(maxsize=4096)
def _canonical_graphlet_signature(
    types: Tuple[str, str, str],
    families: Tuple[str, str, str],
) -> str:
    pair_family = {
        (0, 1): families[0],
        (0, 2): families[1],
        (1, 2): families[2],
    }
    encodings = []
    for order in itertools.permutations(range(3)):
        ordered_types = tuple(types[index] for index in order)
        ordered_families = tuple(
            pair_family[tuple(sorted((order[left], order[right])))]
            for left, right in ((0, 1), (0, 2), (1, 2))
        )
        encodings.append(ordered_types + ordered_families)
    return "|".join(min(encodings))


def _degree_bucket(value: int) -> str:
    value = int(value)
    if value <= 3:
        return str(value)
    if value <= 5:
        return "4-5"
    if value <= 9:
        return "6-9"
    if value <= 19:
        return "10-19"
    return "20+"


@dataclass(frozen=True)
class FeatureSelection:
    keys: Tuple[str, ...]
    null_mean: np.ndarray
    scale: np.ndarray

    def vector(self, values: Mapping[str, int | float]) -> np.ndarray:
        raw = np.fromiter(
            (float(values.get(key, 0.0)) for key in self.keys),
            dtype=np.float64,
            count=len(self.keys),
        )
        return (raw - self.null_mean) / self.scale

    def delta_vector(self, values: Mapping[str, int | float]) -> np.ndarray:
        return np.fromiter(
            (
                float(values.get(key, 0.0)) / float(scale)
                for key, scale in zip(self.keys, self.scale)
            ),
            dtype=np.float64,
            count=len(self.keys),
        )

    def to_json_dict(self) -> dict:
        return {
            "keys": list(self.keys),
            "null_mean": self.null_mean.tolist(),
            "scale": self.scale.tolist(),
        }


class GraphletFeatureExtractor:
    """Exact typed three-node and degree-mixing sufficient statistics."""

    def __init__(
        self, graph: TypedGraph, tracked_keys: Optional[Sequence[str]] = None
    ) -> None:
        self.node_types = graph.node_types.copy()
        self.type_names = tuple(graph.type_names)
        # Relation degrees are invariant under every legal family switch.
        self.fixed_total_degree = graph.degrees()
        self.tracked_keys = None if tracked_keys is None else frozenset(tracked_keys)
        domains = {
            tuple(sorted((int(spec.src_type), int(spec.dst_type))))
            for spec in graph.family_specs.values()
        }
        self.family_can_change_triangle = {}
        for family, spec in graph.family_specs.items():
            left_type = int(spec.src_type)
            right_type = int(spec.dst_type)
            self.family_can_change_triangle[family] = any(
                tuple(sorted((left_type, third_type))) in domains
                and tuple(sorted((right_type, third_type))) in domains
                for third_type in range(len(graph.type_names))
            )

    def _filter(self, values: FeatureCounter) -> FeatureCounter:
        if self.tracked_keys is None:
            return values
        return Counter(
            {
                key: value
                for key, value in values.items()
                if key in self.tracked_keys and value
            }
        )

    def _type_name(self, node: int) -> str:
        return self.type_names[int(self.node_types[int(node)])]

    def _graphlet_signature(self, graph: TypedGraph, nodes: Triple) -> str:
        nodes = tuple(int(node) for node in nodes)
        types = tuple(self._type_name(node) for node in nodes)
        families = tuple(
            graph.family_of(nodes[left], nodes[right]) or "-"
            for left, right in ((0, 1), (0, 2), (1, 2))
        )
        return _canonical_graphlet_signature(types, families)

    def _wedge_key(
        self, graph: TypedGraph, center: int, left: int, right: int
    ) -> str:
        descriptors = sorted(
            (
                f"{self._type_name(left)}@{graph.family_of(center, left)}",
                f"{self._type_name(right)}@{graph.family_of(center, right)}",
            )
        )
        return (
            f"wedge/center={self._type_name(center)}/"
            f"ends={descriptors[0]}::{descriptors[1]}"
        )

    def triple_contribution(self, graph: TypedGraph, nodes: Triple) -> FeatureCounter:
        u, v, w = sorted(int(node) for node in nodes)
        present = (
            graph.has_edge(u, v),
            graph.has_edge(u, w),
            graph.has_edge(v, w),
        )
        edge_count = int(sum(present))
        if edge_count not in (2, 3):
            return Counter()
        if edge_count == 2:
            degree_in_triple = {
                u: int(present[0]) + int(present[1]),
                v: int(present[0]) + int(present[2]),
                w: int(present[1]) + int(present[2]),
            }
            center = next(node for node, value in degree_in_triple.items() if value == 2)
            endpoints = [node for node in (u, v, w) if node != center]
            return Counter(
                {self._wedge_key(graph, center, endpoints[0], endpoints[1]): 1}
            )

        signature = self._graphlet_signature(graph, (u, v, w))
        result: FeatureCounter = Counter({f"triangle/{signature}": 1})
        for root in (u, v, w):
            root_type = self._type_name(root)
            bucket = _degree_bucket(self.fixed_total_degree[root])
            result[
                f"root_triangle_total/type={root_type}/degree={bucket}"
            ] += 1
            result[
                f"root_triangle/{signature}/root={root_type}/degree={bucket}"
            ] += 1
        return result

    def edge_mixing_contribution(
        self, graph: TypedGraph, family: str, edge: Edge
    ) -> FeatureCounter:
        u, v = graph.orient_edge(family, edge)
        left_bucket = _degree_bucket(self.fixed_total_degree[u])
        right_bucket = _degree_bucket(self.fixed_total_degree[v])
        spec = graph.family_specs[family]
        if not spec.bipartite and right_bucket < left_bucket:
            left_bucket, right_bucket = right_bucket, left_bucket
        return Counter({f"mix/{family}/{left_bucket}:{right_bucket}": 1})

    def edgewise_shared_partner_contribution(
        self, graph: TypedGraph, edge: Edge
    ) -> FeatureCounter:
        """Typed curved-ERGM closure statistic for one existing edge.

        The first shared partner contributes one unit and later partners have
        geometrically diminishing returns. This distinguishes distributed
        closure from piling many triangles onto the same hub edge.
        """

        u, v = sorted((int(edge[0]), int(edge[1])))
        if not graph.has_edge(u, v):
            return Counter()
        family = graph.family_of(u, v)
        if family is None:
            return Counter()
        left, right = graph.orient_edge(family, (u, v))
        left_bucket = _degree_bucket(self.fixed_total_degree[left])
        right_bucket = _degree_bucket(self.fixed_total_degree[right])
        spec = graph.family_specs[family]
        if not spec.bipartite and right_bucket < left_bucket:
            left_bucket, right_bucket = right_bucket, left_bucket
        shared_by_type = Counter(
            self._type_name(node) for node in graph.adjacency[u] & graph.adjacency[v]
        )
        result: FeatureCounter = Counter()
        for shared_type, count in shared_by_type.items():
            if count <= 0:
                continue
            value = (1.0 - GWESP_DECAY ** int(count)) / (1.0 - GWESP_DECAY)
            key = (
                f"gwesp/family={family}/degree={left_bucket}:{right_bucket}/"
                f"shared={shared_type}"
            )
            result[key] += float(value)
        return result

    def extract(self, graph: TypedGraph) -> FeatureCounter:
        values: FeatureCounter = Counter()
        for family, edges in graph.family_edges.items():
            for edge in edges:
                values.update(self.edge_mixing_contribution(graph, family, edge))
                values.update(self.edgewise_shared_partner_contribution(graph, edge))

        # Count all typed neighbor pairs combinatorially, then subtract the
        # three closed wedges contributed by every triangle. This avoids
        # enumerating O(sum_v degree(v)^2) pairs around high-degree nodes.
        for center, neighbors in enumerate(graph.adjacency):
            groups: Counter[tuple[str, str]] = Counter(
                (
                    self._type_name(other),
                    str(graph.family_of(center, other)),
                )
                for other in neighbors
            )
            descriptors = sorted(groups)
            for left_pos, left_desc in enumerate(descriptors):
                left_count = groups[left_desc]
                for right_desc in descriptors[left_pos:]:
                    right_count = groups[right_desc]
                    if left_desc == right_desc:
                        count = left_count * (left_count - 1) // 2
                    else:
                        count = left_count * right_count
                    if count <= 0:
                        continue
                    key = (
                        f"wedge/center={self._type_name(center)}/"
                        f"ends={left_desc[0]}@{left_desc[1]}::"
                        f"{right_desc[0]}@{right_desc[1]}"
                    )
                    values[key] += int(count)

        for u in range(graph.num_nodes):
            for v in graph.adjacency[u]:
                if v <= u:
                    continue
                for w in graph.adjacency[u] & graph.adjacency[v]:
                    if w <= v:
                        continue
                    triangle = (u, v, w)
                    for center, left, right in (
                        (u, v, w),
                        (v, u, w),
                        (w, u, v),
                    ):
                        wedge_key = self._wedge_key(graph, center, left, right)
                        values[wedge_key] -= 1
                        if values[wedge_key] == 0:
                            del values[wedge_key]
                    values.update(self.triple_contribution(graph, triangle))
        return self._filter(values)

    @staticmethod
    def _affected_triples(
        graph: TypedGraph, changed_edges: Sequence[Edge]
    ) -> Set[Triple]:
        endpoints = {int(node) for edge in changed_edges for node in edge}
        affected: Set[Triple] = set()
        for u_raw, v_raw in changed_edges:
            u, v = int(u_raw), int(v_raw)
            third_nodes = set(graph.adjacency[u]) | set(graph.adjacency[v]) | endpoints
            third_nodes.discard(u)
            third_nodes.discard(v)
            for third in third_nodes:
                affected.add(tuple(sorted((u, v, int(third)))))
        return affected

    @staticmethod
    def _affected_closed_triples(
        graph: TypedGraph, changed_edges: Sequence[Edge]
    ) -> Set[Triple]:
        """Triangles affected by an atomic multi-edge replacement.

        Away from the changed endpoints, toggling ``(u, v)`` can only change
        triples whose third node is already adjacent to both endpoints. New
        interactions among changed edges are confined to triples formed by the
        changed endpoints themselves. This avoids scanning hub-neighbor unions
        when no open-wedge feature is tracked.
        """

        endpoints = {int(node) for edge in changed_edges for node in edge}
        affected: Set[Triple] = set()
        for u_raw, v_raw in changed_edges:
            u, v = int(u_raw), int(v_raw)
            for third in graph.adjacency[u] & graph.adjacency[v]:
                affected.add(tuple(sorted((u, v, int(third)))))
        for triple in itertools.combinations(sorted(endpoints), 3):
            affected.add(tuple(int(node) for node in triple))
        return affected

    @staticmethod
    def _gwesp_affected_edges(
        graph: TypedGraph, changed_edges: Sequence[Edge]
    ) -> Set[Edge]:
        """Edges whose shared-partner count can change under ``changed_edges``.

        Toggling ``(u, v)`` can only alter its own GWESP contribution and the
        contributions of ``(u, w)``/``(v, w)`` for a common neighbor ``w``.
        Scanning every edge incident to either endpoint is exact but needlessly
        expensive around IMDB hubs.
        """

        affected: Set[Edge] = set()
        for u_raw, v_raw in changed_edges:
            u, v = int(u_raw), int(v_raw)
            affected.add(tuple(sorted((u, v))))
            for other in graph.adjacency[u] & graph.adjacency[v]:
                affected.add(tuple(sorted((u, int(other)))))
                affected.add(tuple(sorted((v, int(other)))))
        return affected

    def switch_delta(
        self,
        graph: TypedGraph,
        family: str,
        indices: Sequence[int],
        old_edges: Sequence[Edge],
        new_edges: Sequence[Edge],
    ) -> FeatureCounter:
        tracks_global_closure = self.tracked_keys is None or any(
            key.startswith("gwesp/") for key in self.tracked_keys
        )
        tracks_open_wedges = self.tracked_keys is None or any(
            key.startswith("wedge/") for key in self.tracked_keys
        )
        if (
            self.tracked_keys is not None
            and not tracks_global_closure
            and not any(family in key for key in self.tracked_keys)
        ):
            graph.replace_edges(family, indices, old_edges, new_edges)
            return Counter()
        before: FeatureCounter = Counter()
        affected: Set[Triple] = set()
        if self.family_can_change_triangle[family]:
            changed = old_edges + new_edges
            affected = (
                self._affected_triples(graph, changed)
                if tracks_open_wedges
                else self._affected_closed_triples(graph, changed)
            )
            for triple in affected:
                before.update(self.triple_contribution(graph, triple))
        for edge in old_edges:
            before.update(self.edge_mixing_contribution(graph, family, edge))
        if tracks_global_closure:
            for edge in self._gwesp_affected_edges(graph, old_edges + new_edges):
                before.update(self.edgewise_shared_partner_contribution(graph, edge))

        graph.replace_edges(family, indices, old_edges, new_edges)
        after: FeatureCounter = Counter()
        for triple in affected:
            after.update(self.triple_contribution(graph, triple))
        for edge in new_edges:
            after.update(self.edge_mixing_contribution(graph, family, edge))
        if tracks_global_closure:
            for edge in self._gwesp_affected_edges(graph, old_edges + new_edges):
                after.update(self.edgewise_shared_partner_contribution(graph, edge))

        delta = after.copy()
        delta.subtract(before)
        return self._filter(
            Counter({key: value for key, value in delta.items() if value})
        )

    @staticmethod
    def apply_delta(
        target: MutableMapping[str, int | float],
        delta: Mapping[str, int | float],
    ) -> None:
        for key, value in delta.items():
            updated = float(target.get(key, 0.0)) + float(value)
            if abs(updated) > 1.0e-12:
                target[key] = updated
            elif key in target:
                del target[key]


def select_features(
    real: Mapping[str, int | float],
    null_samples: Sequence[Mapping[str, int | float]],
    max_features: int = 64,
    min_support: float = 2.0,
    drop_fixed_degree_wedges: bool = True,
    reserve_prefix_counts: Optional[Mapping[str, int]] = None,
) -> FeatureSelection:
    if not null_samples:
        raise ValueError("at least one null sample is required")
    keys = sorted(
        set(real).union(*(set(sample) for sample in null_samples))
    )
    matrix = np.asarray(
        [[float(sample.get(key, 0.0)) for key in keys] for sample in null_samples],
        dtype=np.float64,
    )
    real_vector = np.asarray([float(real.get(key, 0.0)) for key in keys])
    mean = matrix.mean(axis=0)
    std = matrix.std(axis=0, ddof=1) if matrix.shape[0] > 1 else np.zeros(len(keys))
    scale = np.maximum(std, 1.0)
    score = np.abs(real_vector - mean) / scale
    supported = np.maximum(real_vector, mean) >= float(min_support)
    candidates = np.flatnonzero(supported)
    if drop_fixed_degree_wedges:
        candidates = np.asarray(
            [index for index in candidates if not keys[index].startswith("wedge/")],
            dtype=np.int64,
        )
    if candidates.size == 0:
        raise ValueError("no graphlet features passed the support threshold")
    budget = max(int(max_features), 1)
    order = candidates[np.argsort(score[candidates])[::-1]]
    selected: list[int] = []
    selected_set: set[int] = set()
    for prefix, raw_count in (reserve_prefix_counts or {}).items():
        prefix_order = [
            int(index)
            for index in order
            if keys[int(index)].startswith(str(prefix))
            and int(index) not in selected_set
        ]
        count = int(raw_count)
        if count >= 0:
            prefix_order = prefix_order[:count]
        for index in prefix_order:
            if len(selected) >= budget:
                break
            selected.append(index)
            selected_set.add(index)
    for raw_index in order:
        index = int(raw_index)
        if len(selected) >= budget:
            break
        if index not in selected_set:
            selected.append(index)
            selected_set.add(index)
    keep = np.asarray(selected, dtype=np.int64)
    selected_keys = tuple(keys[index] for index in keep)
    return FeatureSelection(
        keys=selected_keys,
        null_mean=mean[keep],
        scale=scale[keep],
    )
