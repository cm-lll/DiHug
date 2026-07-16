from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping

import numpy as np

from .graph import TypedGraph
from .mixed_edit_diffusion import (
    MarginalGraphPrior,
    domain_edge_from_rank,
    family_domain_size,
)


@dataclass(frozen=True)
class FamilyCountCondition:
    """Learned family-count condition for constrained edit diffusion.

    ``fixed`` is the conditional p(G | N, M) protocol. ``binomial`` samples M
    from the learned family marginals before constructing a uniform graph in
    Omega(M). Neither mode stores node identities, role rows, or clean edges.
    """

    marginal: MarginalGraphPrior
    observed_counts: Mapping[str, int]

    @classmethod
    def from_training_graph(cls, graph: TypedGraph) -> "FamilyCountCondition":
        marginal = MarginalGraphPrior.from_training_graph(graph)
        counts = {
            family: int(len(graph.family_edges[family]))
            for family in marginal.family_names
        }
        return cls(marginal=marginal, observed_counts=counts)

    def sample_counts(self, seed: int, mode: str = "binomial") -> Dict[str, int]:
        mode = str(mode).strip().lower()
        if mode == "fixed":
            return {name: int(self.observed_counts[name]) for name in self.marginal.family_names}
        if mode != "binomial":
            raise ValueError("family count mode must be fixed or binomial")
        schema = self.marginal.schema_graph()
        rng = np.random.default_rng(int(seed))
        return {
            family: int(
                rng.binomial(
                    family_domain_size(schema.family_specs[family]),
                    float(self.marginal.family_probabilities[family]),
                )
            )
            for family in self.marginal.family_names
        }

    def sample_graph(self, seed: int, mode: str = "binomial") -> TypedGraph:
        graph = self.marginal.schema_graph()
        counts = self.sample_counts(seed, mode=mode)
        rng = np.random.default_rng(int(seed) + 104729)
        for family in self.marginal.family_names:
            capacity = family_domain_size(graph.family_specs[family])
            count = int(counts[family])
            if count < 0 or count > capacity:
                raise ValueError(f"invalid count for {family}: {count}/{capacity}")
            if count == 0:
                continue
            ranks = rng.choice(capacity, size=count, replace=False)
            for rank in np.asarray(ranks, dtype=np.int64).tolist():
                graph.add_edge(
                    family,
                    domain_edge_from_rank(graph.family_specs[family], int(rank)),
                )
        graph.validate()
        return graph

    def to_json_dict(self) -> dict:
        return {
            "format": "dihug-family-count-condition-v1",
            "marginal": self.marginal.to_json_dict(),
            "observed_counts": {
                name: int(self.observed_counts[name])
                for name in self.marginal.family_names
            },
            "inference_contract": {
                "fixed": "conditional_on_training_family_counts",
                "binomial": "sampled_from_learned_family_marginals",
                "clean_adjacency": False,
                "node_role_rows": False,
            },
        }


def family_edge_counts(graph: TypedGraph) -> Dict[str, int]:
    return {
        family: int(len(graph.family_edges[family]))
        for family in sorted(graph.family_specs)
    }
