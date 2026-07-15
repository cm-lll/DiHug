from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np
import torch


Edge = Tuple[int, int]


@dataclass(frozen=True)
class FamilySpec:
    name: str
    src_type: int
    dst_type: int
    src_nodes: Tuple[int, ...]
    dst_nodes: Tuple[int, ...]

    @property
    def bipartite(self) -> bool:
        return self.src_type != self.dst_type


class TypedGraph:
    """Mutable undirected heterogeneous graph with disjoint family domains."""

    def __init__(
        self,
        node_types: np.ndarray,
        type_names: Sequence[str],
        family_specs: Mapping[str, FamilySpec],
        family_edges: Mapping[str, Iterable[Edge]],
    ) -> None:
        self.node_types = np.asarray(node_types, dtype=np.int64)
        self.type_names = tuple(str(name) for name in type_names)
        self.family_specs = dict(family_specs)
        self.num_nodes = int(self.node_types.shape[0])
        self.family_edges: Dict[str, List[Edge]] = {
            name: [] for name in self.family_specs
        }
        self.family_edge_sets: Dict[str, Set[Edge]] = {
            name: set() for name in self.family_specs
        }
        self.adjacency: List[Set[int]] = [set() for _ in range(self.num_nodes)]
        self.edge_family: Dict[Edge, str] = {}

        for family, edges in family_edges.items():
            if family not in self.family_specs:
                raise KeyError(f"unknown family {family!r}")
            for edge in edges:
                self.add_edge(family, edge)

    @staticmethod
    def canonical_edge(u: int, v: int) -> Edge:
        u, v = int(u), int(v)
        return (u, v) if u < v else (v, u)

    def orient_edge(self, family: str, edge: Edge) -> Edge:
        spec = self.family_specs[family]
        u, v = int(edge[0]), int(edge[1])
        if u == v:
            raise ValueError("self loops are not legal")
        tu, tv = int(self.node_types[u]), int(self.node_types[v])
        if spec.bipartite:
            if tu == spec.src_type and tv == spec.dst_type:
                return u, v
            if tv == spec.src_type and tu == spec.dst_type:
                return v, u
            raise ValueError(
                f"edge {(u, v)} has endpoint types {(tu, tv)}, "
                f"incompatible with family {family}"
            )
        if tu != spec.src_type or tv != spec.src_type:
            raise ValueError(f"edge {(u, v)} is outside family {family}")
        return self.canonical_edge(u, v)

    def has_edge(self, u: int, v: int) -> bool:
        if int(u) == int(v):
            return False
        return self.canonical_edge(u, v) in self.edge_family

    def family_of(self, u: int, v: int) -> Optional[str]:
        if int(u) == int(v):
            return None
        return self.edge_family.get(self.canonical_edge(u, v))

    def add_edge(self, family: str, edge: Edge, index: Optional[int] = None) -> Edge:
        oriented = self.orient_edge(family, edge)
        canonical = self.canonical_edge(*oriented)
        if canonical in self.edge_family:
            raise ValueError(
                f"duplicate pair {canonical}: existing={self.edge_family[canonical]}, "
                f"new={family}"
            )
        if oriented in self.family_edge_sets[family]:
            raise ValueError(f"duplicate family edge {family}:{oriented}")
        if index is None:
            self.family_edges[family].append(oriented)
        else:
            self.family_edges[family].insert(int(index), oriented)
        self.family_edge_sets[family].add(oriented)
        self.edge_family[canonical] = family
        u, v = canonical
        self.adjacency[u].add(v)
        self.adjacency[v].add(u)
        return oriented

    def remove_edge(self, family: str, edge: Edge, index: Optional[int] = None) -> Edge:
        oriented = self.orient_edge(family, edge)
        canonical = self.canonical_edge(*oriented)
        if self.edge_family.get(canonical) != family:
            raise KeyError(f"missing family edge {family}:{oriented}")
        if index is None:
            self.family_edges[family].remove(oriented)
        else:
            found = self.family_edges[family][int(index)]
            if found != oriented:
                raise ValueError(
                    f"family edge index mismatch: expected {oriented}, found {found}"
                )
            self.family_edges[family].pop(int(index))
        self.family_edge_sets[family].remove(oriented)
        del self.edge_family[canonical]
        u, v = canonical
        self.adjacency[u].remove(v)
        self.adjacency[v].remove(u)
        return oriented

    def replace_edges(
        self,
        family: str,
        indices: Sequence[int],
        old_edges: Sequence[Edge],
        new_edges: Sequence[Edge],
    ) -> None:
        if not (len(indices) == len(old_edges) == len(new_edges)):
            raise ValueError("edge replacement batch lengths disagree")
        if len(set(int(index) for index in indices)) != len(indices):
            raise ValueError("edge replacement indices are not unique")
        for index, edge in sorted(
            zip(indices, old_edges), key=lambda item: item[0], reverse=True
        ):
            self.remove_edge(family, edge, index=int(index))
        # Reinsert at the original positions so proposal probabilities remain
        # independent of graph history.
        for index, edge in sorted(zip(indices, new_edges), key=lambda item: item[0]):
            self.add_edge(family, edge, index=int(index))

    def clone(self) -> "TypedGraph":
        return TypedGraph(
            self.node_types.copy(),
            self.type_names,
            self.family_specs,
            {name: list(edges) for name, edges in self.family_edges.items()},
        )

    @property
    def num_edges(self) -> int:
        return len(self.edge_family)

    def degrees(self) -> np.ndarray:
        return np.fromiter(
            (len(neighbors) for neighbors in self.adjacency),
            dtype=np.int64,
            count=self.num_nodes,
        )

    def family_degrees(self, family: str) -> np.ndarray:
        degree = np.zeros(self.num_nodes, dtype=np.int64)
        for u, v in self.family_edges[family]:
            degree[u] += 1
            degree[v] += 1
        return degree

    def all_family_degrees(self) -> Dict[str, np.ndarray]:
        return {name: self.family_degrees(name) for name in self.family_specs}

    def validate(self) -> None:
        seen: Set[Edge] = set()
        for family, edges in self.family_edges.items():
            if len(edges) != len(self.family_edge_sets[family]):
                raise AssertionError(f"duplicate edges in family {family}")
            for edge in edges:
                oriented = self.orient_edge(family, edge)
                if oriented != edge:
                    raise AssertionError(f"noncanonical family edge {family}:{edge}")
                canonical = self.canonical_edge(*edge)
                if canonical in seen:
                    raise AssertionError(f"pair assigned to multiple families: {canonical}")
                seen.add(canonical)
                if self.edge_family.get(canonical) != family:
                    raise AssertionError(f"edge-family index mismatch for {canonical}")
        if seen != set(self.edge_family):
            raise AssertionError("edge-family index contains stale entries")
        for node, neighbors in enumerate(self.adjacency):
            if node in neighbors:
                raise AssertionError(f"self loop at node {node}")
            for other in neighbors:
                if node not in self.adjacency[other]:
                    raise AssertionError(f"asymmetric adjacency {(node, other)}")
                if self.canonical_edge(node, other) not in seen:
                    raise AssertionError(f"adjacency edge absent from family index")

    def validate_disjoint_family_domains(self) -> None:
        """Require one relation family per unordered endpoint-type domain."""

        domains: Dict[Tuple[int, int], str] = {}
        for family, spec in self.family_specs.items():
            domain = tuple(sorted((int(spec.src_type), int(spec.dst_type))))
            previous = domains.get(domain)
            if previous is not None:
                raise ValueError(
                    "overlapping family candidate domains are not supported: "
                    f"{previous} and {family}"
                )
            domains[domain] = family

    def to_json_dict(self) -> dict:
        return {
            "num_nodes": self.num_nodes,
            "type_names": list(self.type_names),
            "node_types": self.node_types.tolist(),
            "families": {
                name: {
                    "src_type": int(spec.src_type),
                    "dst_type": int(spec.dst_type),
                    "edges": [[int(u), int(v)] for u, v in self.family_edges[name]],
                }
                for name, spec in self.family_specs.items()
            },
        }

    @classmethod
    def from_json_dict(cls, payload: Mapping) -> "TypedGraph":
        node_types = np.asarray(payload["node_types"], dtype=np.int64)
        type_names = tuple(payload["type_names"])
        nodes_by_type = {
            type_id: tuple(np.flatnonzero(node_types == type_id).tolist())
            for type_id in range(len(type_names))
        }
        specs = {}
        edges = {}
        for family, family_payload in payload["families"].items():
            src_type = int(family_payload["src_type"])
            dst_type = int(family_payload["dst_type"])
            specs[family] = FamilySpec(
                family,
                src_type,
                dst_type,
                nodes_by_type[src_type],
                nodes_by_type[dst_type],
            )
            edges[family] = [
                (int(edge[0]), int(edge[1]))
                for edge in family_payload.get("edges", [])
            ]
        graph = cls(node_types, type_names, specs, edges)
        graph.validate()
        return graph


def load_typed_graph(dataset_dir: str | Path, active_only: bool = True) -> TypedGraph:
    """Load one canonical undirected graph from the raw subgraph directory."""

    dataset_dir = Path(dataset_dir)
    nodes_raw = torch.load(dataset_dir / "nodes.pt", map_location="cpu", weights_only=False)
    edges_raw = torch.load(dataset_dir / "edges.pt", map_location="cpu", weights_only=False)
    with (dataset_dir / "meta.json").open("r", encoding="utf-8") as handle:
        meta = json.load(handle)

    type_names = tuple(meta.get("node_types") or nodes_raw.keys())
    type_to_id = {name: index for index, name in enumerate(type_names)}
    raw_ids: List[int] = []
    raw_to_dense: Dict[int, int] = {}
    ids_by_type: Dict[str, List[int]] = {}
    node_types: List[int] = []
    for type_name in type_names:
        ids = [int(value) for value in nodes_raw[type_name]["ids"]]
        dense_ids = []
        for raw_id in ids:
            if raw_id in raw_to_dense:
                raise ValueError(f"duplicate raw node id {raw_id}")
            dense_id = len(raw_ids)
            raw_ids.append(raw_id)
            raw_to_dense[raw_id] = dense_id
            dense_ids.append(dense_id)
            node_types.append(type_to_id[type_name])
        ids_by_type[type_name] = dense_ids

    family_specs: Dict[str, FamilySpec] = {}
    family_edges: Dict[str, List[Edge]] = {}
    endpoint_meta = meta.get("fam_endpoints", {})
    raw_families = edges_raw.get("families", {})
    for family, endpoint in endpoint_meta.items():
        payload = raw_families.get(family)
        edge_count = 0 if payload is None else int(payload["src_local"].numel())
        if active_only and edge_count == 0:
            continue
        src_name = str(endpoint["src_type"])
        dst_name = str(endpoint["dst_type"])
        spec = FamilySpec(
            name=str(family),
            src_type=type_to_id[src_name],
            dst_type=type_to_id[dst_name],
            src_nodes=tuple(ids_by_type[src_name]),
            dst_nodes=tuple(ids_by_type[dst_name]),
        )
        family_specs[family] = spec
        family_edges[family] = []
        if payload is None:
            continue
        src_local = payload["src_local"].reshape(-1).tolist()
        dst_local = payload["dst_local"].reshape(-1).tolist()
        labels = payload.get("y")
        if labels is None:
            labels = torch.ones(len(src_local), dtype=torch.long)
        labels = labels.reshape(-1).tolist()
        for src_index, dst_index, label in zip(src_local, dst_local, labels):
            if int(label) <= 0:
                continue
            family_edges[family].append(
                (
                    ids_by_type[src_name][int(src_index)],
                    ids_by_type[dst_name][int(dst_index)],
                )
            )

    graph = TypedGraph(
        np.asarray(node_types, dtype=np.int64),
        type_names,
        family_specs,
        family_edges,
    )
    graph.validate()
    return graph


def uniform_edge_view(graph: TypedGraph, edge_count: int, seed: int) -> TypedGraph:
    """Return an exact-size uniform edge view while preserving the full schema.

    The operation only observes ``graph`` itself.  It is useful for evaluating a
    learned full-graph generator under an edge-held-out protocol: first sample a
    full graph from the learned prior, then expose the requested number of its
    edges without consulting the held-out target adjacency or target families.
    """

    edge_count = int(edge_count)
    if edge_count < 0 or edge_count > graph.num_edges:
        raise ValueError(
            f"edge view size {edge_count} is outside [0, {graph.num_edges}]"
        )
    entries = [
        (family, edge)
        for family in sorted(graph.family_edges)
        for edge in graph.family_edges[family]
    ]
    rng = np.random.default_rng(int(seed))
    selected = set(
        rng.choice(len(entries), size=edge_count, replace=False).astype(int).tolist()
    )
    family_edges = {family: [] for family in graph.family_specs}
    for index, (family, edge) in enumerate(entries):
        if index in selected:
            family_edges[family].append(edge)
    view = TypedGraph(
        graph.node_types.copy(),
        graph.type_names,
        graph.family_specs,
        family_edges,
    )
    view.validate()
    return view
