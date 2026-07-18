from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import torch
from torch import nn


@dataclass(frozen=True)
class SubtypeSchema:
    node_values: Mapping[str, tuple[int, ...]]
    edge_values: Mapping[str, tuple[int, ...]]

    @property
    def prediction_enabled(self) -> bool:
        return any(len(values) > 1 for values in self.node_values.values()) or any(
            len(values) > 1 for values in self.edge_values.values()
        )

    def to_json_dict(self) -> dict:
        return {
            "format": "dihug-subtype-schema-v1",
            "node_values": {key: list(value) for key, value in self.node_values.items()},
            "edge_values": {key: list(value) for key, value in self.edge_values.items()},
            "prediction_enabled": self.prediction_enabled,
        }


def load_subtype_schema(dataset_dir: str | Path) -> SubtypeSchema:
    dataset_dir = Path(dataset_dir)
    nodes = torch.load(dataset_dir / "nodes.pt", map_location="cpu", weights_only=False)
    edges = torch.load(dataset_dir / "edges.pt", map_location="cpu", weights_only=False)
    node_values = {}
    for type_name, payload in nodes.items():
        values = payload.get("subtype")
        node_values[str(type_name)] = tuple(
            sorted(int(value) for value in torch.unique(values).tolist())
        ) if values is not None else ()
    edge_values = {}
    for family, payload in edges.get("families", {}).items():
        values = payload.get("y")
        positive = values[values > 0] if values is not None else torch.ones(1)
        edge_values[str(family)] = tuple(
            sorted(int(value) for value in torch.unique(positive).tolist())
        )
    return SubtypeSchema(node_values=node_values, edge_values=edge_values)


class ConditionalSubtypeHead(nn.Module):
    """Optional subtype decoder reusing final global-HGT node states.

    Heads with one or zero observed classes are omitted. This keeps current
    single-subtype datasets exact while allowing future datasets to train CE
    without changing the topology diffusion state space.
    """

    def __init__(
        self,
        hidden_dim: int,
        node_type_names: Sequence[str],
        family_names: Sequence[str],
        schema: SubtypeSchema,
    ) -> None:
        super().__init__()
        self.node_heads = nn.ModuleDict(
            {
                name: nn.Linear(hidden_dim, len(schema.node_values.get(name, ())))
                for name in node_type_names
                if len(schema.node_values.get(name, ())) > 1
            }
        )
        self.edge_heads = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.Linear(4 * hidden_dim, hidden_dim),
                    nn.SiLU(),
                    nn.Linear(hidden_dim, len(schema.edge_values.get(name, ()))),
                )
                for name in family_names
                if len(schema.edge_values.get(name, ())) > 1
            }
        )

    def node_logits(self, type_name: str, hidden: torch.Tensor) -> torch.Tensor | None:
        return None if type_name not in self.node_heads else self.node_heads[type_name](hidden)

    def edge_logits(
        self, family: str, left: torch.Tensor, right: torch.Tensor
    ) -> torch.Tensor | None:
        if family not in self.edge_heads:
            return None
        pair = torch.cat((left, right, torch.abs(left - right), left * right), dim=-1)
        return self.edge_heads[family](pair)
