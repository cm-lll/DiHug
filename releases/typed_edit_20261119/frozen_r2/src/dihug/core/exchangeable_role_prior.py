from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Sequence, Tuple

import numpy as np

from .graph import TypedGraph
from .mixed_edit_diffusion import MarginalGraphPrior, family_domain_size
from .roles import RolePlan, realize_role_plan
from .switch_diffusion import randomize_role_realization


@dataclass(frozen=True)
class TypeRoleMixture:
    family_names: Tuple[str, ...]
    weights: np.ndarray
    means: np.ndarray
    covariances: np.ndarray
    discrete: bool = False

    def to_json_dict(self) -> dict:
        return {
            "family_names": list(self.family_names),
            "weights": self.weights.tolist(),
            "means": self.means.tolist(),
            "covariances": self.covariances.tolist(),
            "discrete": bool(self.discrete),
        }

    @classmethod
    def from_json_dict(cls, payload: Mapping) -> "TypeRoleMixture":
        return cls(
            family_names=tuple(payload["family_names"]),
            weights=np.asarray(payload["weights"], dtype=np.float64),
            means=np.asarray(payload["means"], dtype=np.float64),
            covariances=np.asarray(payload["covariances"], dtype=np.float64),
            discrete=bool(payload.get("discrete", False)),
        )


def _fit_log_degree_mixture(
    values: np.ndarray, family_names: Sequence[str], components: int, seed: int
) -> TypeRoleMixture:
    matrix = np.log1p(np.asarray(values, dtype=np.float64))
    count, width = matrix.shape
    component_count = max(1, min(int(components), count))
    rng = np.random.default_rng(int(seed))
    if component_count == 1:
        assignments = np.zeros(count, dtype=np.int64)
    else:
        # Quantile-spread initialization is deterministic up to a type-local seed.
        projection = matrix @ rng.normal(size=width)
        order = np.argsort(projection)
        centers = matrix[order[np.linspace(0, count - 1, component_count).astype(int)]]
        assignments = np.zeros(count, dtype=np.int64)
        for _ in range(30):
            distances = np.square(matrix[:, None, :] - centers[None, :, :]).sum(axis=2)
            updated = distances.argmin(axis=1)
            if np.array_equal(updated, assignments) and _ > 0:
                break
            assignments = updated
            for component in range(component_count):
                selected = matrix[assignments == component]
                if selected.size:
                    centers[component] = selected.mean(axis=0)

    global_covariance = (
        np.cov(matrix, rowvar=False)
        if count > 1
        else np.eye(width, dtype=np.float64) * 1.0e-3
    )
    global_covariance = np.atleast_2d(global_covariance)
    if global_covariance.shape != (width, width):
        global_covariance = np.eye(width, dtype=np.float64) * float(global_covariance.item())
    means = []
    covariances = []
    weights = []
    for component in range(component_count):
        selected = matrix[assignments == component]
        if selected.shape[0] == 0:
            continue
        mean = selected.mean(axis=0)
        if selected.shape[0] > 1:
            covariance = np.atleast_2d(np.cov(selected, rowvar=False))
            if covariance.shape != (width, width):
                covariance = np.eye(width) * float(covariance.item())
        else:
            covariance = global_covariance.copy()
        covariance = 0.75 * covariance + 0.25 * global_covariance
        covariance = covariance + np.eye(width) * 1.0e-3
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        covariance = (eigenvectors * np.maximum(eigenvalues, 1.0e-4)) @ eigenvectors.T
        means.append(mean)
        covariances.append(covariance)
        weights.append(selected.shape[0] / count)
    normalized = np.asarray(weights, dtype=np.float64)
    normalized /= normalized.sum()
    return TypeRoleMixture(
        family_names=tuple(family_names),
        weights=normalized,
        means=np.stack(means),
        covariances=np.stack(covariances),
    )


def _fit_categorical_role_codebook(
    values: np.ndarray, family_names: Sequence[str], max_prototypes: int
) -> TypeRoleMixture:
    rows = np.asarray(values, dtype=np.int64)
    prototypes, counts = np.unique(rows, axis=0, return_counts=True)
    # A non-positive budget means the empirical type-level role distribution.
    # It stores anonymous prototype frequencies, never node-to-role assignments.
    if int(max_prototypes) > 0 and len(prototypes) > int(max_prototypes):
        # Merge only the rarest tail rows; common structural roles stay exact.
        order = np.argsort(counts)[::-1]
        keep = order[: max(int(max_prototypes) - 1, 1)]
        tail = order[max(int(max_prototypes) - 1, 1) :]
        tail_count = counts[tail].sum()
        tail_mean = np.rint(
            np.average(prototypes[tail], axis=0, weights=counts[tail])
        ).astype(np.int64)
        prototypes = np.concatenate((prototypes[keep], tail_mean[None]), axis=0)
        counts = np.concatenate((counts[keep], np.asarray([tail_count])))
    weights = counts.astype(np.float64) / counts.sum()
    width = prototypes.shape[1]
    return TypeRoleMixture(
        family_names=tuple(family_names),
        weights=weights,
        means=np.log1p(prototypes.astype(np.float64)),
        covariances=np.zeros((len(prototypes), width, width), dtype=np.float64),
        discrete=True,
    )


def _sample_type_propensities(
    mixture: TypeRoleMixture,
    count: int,
    rng: np.random.Generator,
    sampling: str = "iid",
) -> np.ndarray:
    sampling = str(sampling).strip().lower()
    if sampling == "iid":
        components = rng.choice(len(mixture.weights), size=int(count), p=mixture.weights)
    elif sampling == "systematic":
        # Variance-reduced categorical sampling.  The random phase gives an
        # unbiased empirical draw, while the final shuffle keeps nodes within
        # each type exchangeable and prevents prototype ordering from leaking
        # into anonymous node IDs.
        positions = (rng.random() + np.arange(int(count), dtype=np.float64)) / max(
            int(count), 1
        )
        cumulative = np.cumsum(np.asarray(mixture.weights, dtype=np.float64))
        cumulative[-1] = 1.0
        components = np.searchsorted(cumulative, positions, side="right")
        rng.shuffle(components)
    else:
        raise ValueError("role component sampling must be iid or systematic")
    values = np.zeros((int(count), len(mixture.family_names)), dtype=np.float64)
    for component in range(len(mixture.weights)):
        selected = np.flatnonzero(components == component)
        if selected.size == 0:
            continue
        if mixture.discrete:
            sampled = np.repeat(mixture.means[component][None], selected.size, axis=0)
        else:
            sampled = rng.multivariate_normal(
                mixture.means[component], mixture.covariances[component], size=selected.size
            )
        values[selected] = np.maximum(np.expm1(sampled), 0.0)
    return values if mixture.discrete else values + 0.05


def _temper_role_correlations(
    values: np.ndarray,
    strength: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Weaken cross-family role dependence without changing any marginal column."""
    strength = float(strength)
    if not 0.0 <= strength <= 1.0:
        raise ValueError("role correlation strength must be in [0, 1]")
    values = np.asarray(values)
    if strength == 1.0 or values.shape[0] < 2 or values.shape[1] < 2:
        return values.copy()

    tempered = values.copy()
    shuffled_count = int(round((1.0 - strength) * values.shape[0]))
    if shuffled_count < 2:
        return tempered
    for column in range(values.shape[1]):
        selected = (
            np.arange(values.shape[0], dtype=np.int64)
            if shuffled_count == values.shape[0]
            else rng.choice(values.shape[0], size=shuffled_count, replace=False)
        )
        tempered[selected, column] = tempered[rng.permutation(selected), column]
    return tempered


def _project_bounded_stub_sum(
    propensities: np.ndarray,
    total: int,
    cap: int,
    rng: np.random.Generator,
) -> np.ndarray:
    count = len(propensities)
    total = int(total)
    cap = int(cap)
    if total < 0 or total > count * cap:
        raise ValueError("stub target is outside the endpoint capacity")
    if total == 0:
        return np.zeros(count, dtype=np.int64)
    values = np.maximum(np.rint(propensities), 0).astype(np.int64).clip(max=cap)
    difference = total - int(values.sum())
    while difference > 0:
        available = np.flatnonzero(values < cap)
        if available.size == 0:
            raise RuntimeError("no endpoint capacity remains while adding stubs")
        weights = np.asarray(propensities[available], dtype=np.float64) + 0.25
        weights *= (cap - values[available]) / max(cap, 1)
        weights /= weights.sum()
        selected = int(rng.choice(available, p=weights))
        values[selected] += 1
        difference -= 1
    while difference < 0:
        available = np.flatnonzero(values > 0)
        weights = values[available].astype(np.float64)
        weights /= weights.sum()
        selected = int(rng.choice(available, p=weights))
        values[selected] -= 1
        difference += 1
    return values


@dataclass(frozen=True)
class ExchangeableJointRolePrior:
    """Type-exchangeable joint relation-degree prior without node-level rows."""

    marginal: MarginalGraphPrior
    type_mixtures: Mapping[int, TypeRoleMixture]
    components: int
    role_model: str = "categorical"
    role_component_sampling: str = "systematic"
    role_correlation_strength: float = 1.0
    randomization_steps_per_edge: float = 2.0

    @classmethod
    def from_training_graph(
        cls,
        graph: TypedGraph,
        components: int = 64,
        seed: int = 0,
        randomization_steps_per_edge: float = 2.0,
        role_model: str = "categorical",
        role_component_sampling: str = "systematic",
        role_correlation_strength: float = 1.0,
    ) -> "ExchangeableJointRolePrior":
        role_component_sampling = str(role_component_sampling).strip().lower()
        if role_component_sampling not in {"iid", "systematic"}:
            raise ValueError("role component sampling must be iid or systematic")
        if not 0.0 <= float(role_correlation_strength) <= 1.0:
            raise ValueError("role correlation strength must be in [0, 1]")
        marginal = MarginalGraphPrior.from_training_graph(graph)
        family_degrees = graph.all_family_degrees()
        mixtures = {}
        for type_id in range(len(graph.type_names)):
            nodes = np.flatnonzero(graph.node_types == type_id)
            incident = tuple(
                family
                for family in marginal.family_names
                if type_id
                in (
                    graph.family_specs[family].src_type,
                    graph.family_specs[family].dst_type,
                )
            )
            values = np.stack(
                [family_degrees[family][nodes] for family in incident], axis=1
            )
            if str(role_model) == "categorical":
                mixtures[type_id] = _fit_categorical_role_codebook(
                    values,
                    incident,
                    max_prototypes=(
                        0 if int(components) <= 0 else max(int(components), 2)
                    ),
                )
            elif str(role_model) == "log_gaussian":
                mixtures[type_id] = _fit_log_degree_mixture(
                    values, incident, components, int(seed) + 7919 * type_id
                )
            else:
                raise ValueError("role model must be categorical or log_gaussian")
        return cls(
            marginal=marginal,
            type_mixtures=mixtures,
            components=int(components),
            role_model=str(role_model),
            role_component_sampling=role_component_sampling,
            role_correlation_strength=float(role_correlation_strength),
            randomization_steps_per_edge=float(randomization_steps_per_edge),
        )

    def sample(self, seed: int, max_attempts: int = 128) -> TypedGraph:
        rng = np.random.default_rng(int(seed))
        schema = self.marginal.schema_graph()
        family_counts = {
            family: int(
                rng.binomial(
                    family_domain_size(schema.family_specs[family]),
                    self.marginal.family_probabilities[family],
                )
            )
            for family in self.marginal.family_names
        }
        for attempt in range(int(max_attempts)):
            propensities: Dict[int, Dict[str, np.ndarray]] = {}
            for type_id, mixture in self.type_mixtures.items():
                nodes = np.flatnonzero(schema.node_types == type_id)
                sampled = _sample_type_propensities(
                    mixture,
                    len(nodes),
                    rng,
                    sampling=self.role_component_sampling,
                )
                sampled = _temper_role_correlations(
                    sampled,
                    strength=self.role_correlation_strength,
                    rng=rng,
                )
                propensities[type_id] = {
                    family: sampled[:, index]
                    for index, family in enumerate(mixture.family_names)
                }
            family_degrees = {
                family: np.zeros(schema.num_nodes, dtype=np.int64)
                for family in self.marginal.family_names
            }
            for family in self.marginal.family_names:
                spec = schema.family_specs[family]
                edge_count = family_counts[family]
                if spec.bipartite:
                    left = np.asarray(spec.src_nodes, dtype=np.int64)
                    right = np.asarray(spec.dst_nodes, dtype=np.int64)
                    family_degrees[family][left] = _project_bounded_stub_sum(
                        propensities[spec.src_type][family], edge_count, len(right), rng
                    )
                    family_degrees[family][right] = _project_bounded_stub_sum(
                        propensities[spec.dst_type][family], edge_count, len(left), rng
                    )
                else:
                    nodes = np.asarray(spec.src_nodes, dtype=np.int64)
                    family_degrees[family][nodes] = _project_bounded_stub_sum(
                        propensities[spec.src_type][family],
                        2 * edge_count,
                        max(len(nodes) - 1, 0),
                        rng,
                    )
            plan = RolePlan(
                permutation=np.arange(schema.num_nodes, dtype=np.int64),
                family_degrees=family_degrees,
            )
            try:
                realized = realize_role_plan(schema, plan)
            except (ValueError, AssertionError):
                continue
            steps = int(round(self.randomization_steps_per_edge * realized.num_edges))
            return randomize_role_realization(
                realized, steps=steps, seed=int(seed) + 104729 + attempt
            )
        raise RuntimeError(
            f"failed to realize exchangeable role sample after {max_attempts} attempts"
        )

    def to_json_dict(self) -> dict:
        return {
            "format": "dihug-exchangeable-joint-role-prior-v1",
            "marginal": self.marginal.to_json_dict(),
            "components": int(self.components),
            "role_model": self.role_model,
            "role_component_sampling": self.role_component_sampling,
            "role_correlation_strength": float(self.role_correlation_strength),
            "randomization_steps_per_edge": float(self.randomization_steps_per_edge),
            "type_mixtures": {
                str(type_id): mixture.to_json_dict()
                for type_id, mixture in self.type_mixtures.items()
            },
            "inference_contract": {
                "stores_training_adjacency": False,
                "stores_per_node_role_rows": False,
                "samples_family_edge_counts": True,
                "type_exchangeable": True,
                "role_component_sampling": self.role_component_sampling,
                "role_correlation_strength": float(self.role_correlation_strength),
            },
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_json_dict(), indent=2, sort_keys=True), encoding="utf-8"
        )

    @classmethod
    def from_json_dict(cls, payload: Mapping) -> "ExchangeableJointRolePrior":
        if payload.get("format") != "dihug-exchangeable-joint-role-prior-v1":
            raise ValueError("unsupported exchangeable role prior")
        return cls(
            marginal=MarginalGraphPrior.from_json_dict(payload["marginal"]),
            type_mixtures={
                int(type_id): TypeRoleMixture.from_json_dict(value)
                for type_id, value in payload["type_mixtures"].items()
            },
            components=int(payload["components"]),
            role_model=str(payload.get("role_model", "log_gaussian")),
            role_component_sampling=str(
                payload.get("role_component_sampling", "iid")
            ),
            role_correlation_strength=float(
                payload.get("role_correlation_strength", 1.0)
            ),
            randomization_steps_per_edge=float(payload["randomization_steps_per_edge"]),
        )

    @classmethod
    def load(cls, path: str | Path) -> "ExchangeableJointRolePrior":
        return cls.from_json_dict(json.loads(Path(path).read_text(encoding="utf-8")))
