from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Tuple

import numpy as np

from .features import FeatureSelection
from .graph import FamilySpec, TypedGraph
from .roles import RolePlan, realize_role_plan
from .switch_chain import LinearGraphletEnergy


@dataclass(frozen=True)
class StructureEnergyCheckpoint:
    """Identity-free structural state learned from one training graph.

    The checkpoint contains relation-degree rows as multisets within each node
    type and a frozen graphlet energy. It intentionally has no edge list or
    original node identifier.
    """

    type_names: Tuple[str, ...]
    type_counts: Tuple[int, ...]
    family_names: Tuple[str, ...]
    family_endpoints: Mapping[str, Tuple[int, int]]
    role_rows_by_type: Mapping[int, Tuple[Tuple[int, ...], ...]]
    selection: FeatureSelection
    weights: np.ndarray

    @classmethod
    def from_training_graph(
        cls, graph: TypedGraph, energy: LinearGraphletEnergy
    ) -> "StructureEnergyCheckpoint":
        graph.validate_disjoint_family_domains()
        family_names = tuple(sorted(graph.family_specs))
        degree = graph.all_family_degrees()
        rows: Dict[int, Tuple[Tuple[int, ...], ...]] = {}
        counts = []
        for type_id in range(len(graph.type_names)):
            nodes = np.flatnonzero(graph.node_types == type_id).tolist()
            counts.append(len(nodes))
            # Sorting removes the last accidental dependence on source IDs.
            rows[type_id] = tuple(
                sorted(
                    tuple(int(degree[family][node]) for family in family_names)
                    for node in nodes
                )
            )
        return cls(
            type_names=tuple(graph.type_names),
            type_counts=tuple(counts),
            family_names=family_names,
            family_endpoints={
                family: (
                    int(graph.family_specs[family].src_type),
                    int(graph.family_specs[family].dst_type),
                )
                for family in family_names
            },
            role_rows_by_type=rows,
            selection=energy.selection,
            weights=energy.weights.copy(),
        )

    def schema_graph(self) -> TypedGraph:
        offsets = np.cumsum((0,) + self.type_counts)
        node_types = np.concatenate(
            [
                np.full(count, type_id, dtype=np.int64)
                for type_id, count in enumerate(self.type_counts)
            ]
        )
        nodes_by_type = {
            type_id: tuple(range(int(offsets[type_id]), int(offsets[type_id + 1])))
            for type_id in range(len(self.type_names))
        }
        specs = {}
        for family in self.family_names:
            src_type, dst_type = self.family_endpoints[family]
            specs[family] = FamilySpec(
                family,
                int(src_type),
                int(dst_type),
                nodes_by_type[int(src_type)],
                nodes_by_type[int(dst_type)],
            )
        return TypedGraph(
            node_types,
            self.type_names,
            specs,
            {family: [] for family in self.family_names},
        )

    def sample_role_plan(self, seed: int) -> tuple[TypedGraph, RolePlan]:
        schema = self.schema_graph()
        rng = np.random.default_rng(int(seed))
        family_degrees = {
            family: np.zeros(schema.num_nodes, dtype=np.int64)
            for family in self.family_names
        }
        for type_id in range(len(self.type_names)):
            nodes = np.flatnonzero(schema.node_types == type_id)
            rows = self.role_rows_by_type[int(type_id)]
            if len(rows) != len(nodes):
                raise ValueError(f"role row count mismatch for type {type_id}")
            order = rng.permutation(len(rows))
            for node, row_index in zip(nodes.tolist(), order.tolist()):
                row = rows[row_index]
                for family_index, family in enumerate(self.family_names):
                    family_degrees[family][node] = int(row[family_index])
        plan = RolePlan(
            permutation=np.arange(schema.num_nodes, dtype=np.int64),
            family_degrees=family_degrees,
        )
        plan.validate(schema)
        return schema, plan

    def realize(self, seed: int) -> TypedGraph:
        schema, plan = self.sample_role_plan(seed)
        return realize_role_plan(schema, plan)

    def energy(self) -> LinearGraphletEnergy:
        return LinearGraphletEnergy(self.selection, self.weights.copy())

    def to_json_dict(self) -> dict:
        return {
            "format": "dihug-structure-energy-v1",
            "type_names": list(self.type_names),
            "type_counts": list(self.type_counts),
            "family_names": list(self.family_names),
            "family_endpoints": {
                family: list(self.family_endpoints[family])
                for family in self.family_names
            },
            "role_rows_by_type": {
                str(type_id): [list(row) for row in self.role_rows_by_type[type_id]]
                for type_id in range(len(self.type_names))
            },
            "selection": self.selection.to_json_dict(),
            "weights": self.weights.tolist(),
        }

    @classmethod
    def from_json_dict(cls, payload: Mapping) -> "StructureEnergyCheckpoint":
        if payload.get("format") != "dihug-structure-energy-v1":
            raise ValueError("unsupported structure-energy checkpoint format")
        selection_payload = payload["selection"]
        selection = FeatureSelection(
            keys=tuple(selection_payload["keys"]),
            null_mean=np.asarray(selection_payload["null_mean"], dtype=np.float64),
            scale=np.asarray(selection_payload["scale"], dtype=np.float64),
        )
        return cls(
            type_names=tuple(payload["type_names"]),
            type_counts=tuple(int(value) for value in payload["type_counts"]),
            family_names=tuple(payload["family_names"]),
            family_endpoints={
                str(family): (int(value[0]), int(value[1]))
                for family, value in payload["family_endpoints"].items()
            },
            role_rows_by_type={
                int(type_id): tuple(tuple(int(value) for value in row) for row in rows)
                for type_id, rows in payload["role_rows_by_type"].items()
            },
            selection=selection,
            weights=np.asarray(payload["weights"], dtype=np.float64),
        )

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(self.to_json_dict(), handle, indent=2, sort_keys=True)

    @classmethod
    def load(cls, path: str | Path) -> "StructureEnergyCheckpoint":
        with Path(path).open("r", encoding="utf-8") as handle:
            return cls.from_json_dict(json.load(handle))
