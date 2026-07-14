from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Mapping

import numpy as np

from .checkpoint import StructureEnergyCheckpoint
from .diagnostics import summarize_chain_metrics
from .features import GraphletFeatureExtractor, select_features
from .fit import fit_moment_matching_energy, fit_pairwise_nce
from .graph import TypedGraph, load_typed_graph
from .roles import realize_role_plan, sample_role_plan, validate_role_realization
from .switch_chain import ChainResult, ChainSample, graph_metrics, run_switch_chain


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(value), handle, ensure_ascii=True, indent=2, sort_keys=True)


def _chain_payload(result: ChainResult) -> dict:
    return {
        "attempted": result.attempted,
        "valid": result.valid,
        "accepted": result.accepted,
        "valid_rate": result.valid_rate,
        "acceptance_rate": result.acceptance_rate,
        "samples": [
            {
                "step": sample.step,
                "energy": sample.energy,
                "features": sample.features,
                "metrics": sample.metrics,
            }
            for sample in result.samples
        ],
    }


def _relative_error(value: float, target: float) -> float:
    return abs(float(value) - float(target)) / max(abs(float(target)), 1.0e-12)


def _edge_jaccard(left: TypedGraph, right: TypedGraph) -> float:
    left_edges = set(left.edge_family)
    right_edges = set(right.edge_family)
    union = left_edges | right_edges
    return 1.0 if not union else float(len(left_edges & right_edges) / len(union))


def _mixing_pass(summary: Mapping[str, Mapping[str, float]], config) -> bool:
    for name in ("triangles", "clustering"):
        values = summary.get(name, {})
        rhat = float(values.get("split_r_hat", float("nan")))
        ess = float(values.get("ess_min", 0.0))
        drift_z = float(values.get("half_drift_z", float("inf")))
        if not math.isfinite(rhat) or rhat > float(config.rhat_threshold):
            return False
        if ess < float(config.min_ess_per_chain):
            return False
        if not math.isfinite(drift_z) or drift_z > float(
            config.chain_drift_z_threshold
        ):
            return False
    return True


@dataclass(frozen=True)
class GateConfig:
    dataset_dir: Path
    output_dir: Path
    seed: int = 0
    null_chains: int = 4
    null_steps: int = 50_000
    null_burn_in: int = 10_000
    energy_chains: int = 4
    energy_steps: int = 100_000
    energy_burn_in: int = 20_000
    thin: int = 500
    verify_every: int = 0
    max_features: int = 64
    min_feature_support: float = 2.0
    nce_l2: float = 0.1
    nce_max_weight: float = 8.0
    moment_rounds: int = 24
    moment_steps_per_round: int = 5_000
    moment_samples_per_round: int = 4
    moment_learning_rate: float = 0.2
    moment_l2: float = 0.01
    moment_tolerance: float = 0.10
    structure_tolerance: float = 0.10
    rhat_threshold: float = 1.10
    min_ess_per_chain: float = 20.0
    curveball_probability: float = 0.0
    chain_drift_z_threshold: float = 2.0
    chain_max_extensions: int = 4
    chain_extension_steps: int = 100_000
    diagnostic_window: int = 64
    moment_max_batches_per_round: int = 16
    moment_min_samples_per_chain: int = 16
    moment_min_ess_per_chain: float = 8.0
    moment_rhat_threshold: float = 1.10
    moment_drift_z_threshold: float = 2.0
    moment_warmup_steps_per_round: int = 0
    moment_step_decay: float = 0.60
    moment_update_norm_clip: float = 5.0
    moment_diagnostic_quantile: float = 0.90


def _tail_metric_chains(
    results: List[ChainResult], window: int
) -> List[List[Mapping[str, float]]]:
    return [
        [sample.metrics for sample in result.samples[-max(int(window), 1) :]]
        for result in results
    ]


def _merge_chain_results(previous: ChainResult, continuation: ChainResult) -> ChainResult:
    offset = int(previous.attempted)
    continued_samples = [
        ChainSample(
            step=offset + int(sample.step),
            energy=float(sample.energy),
            features=dict(sample.features),
            metrics=dict(sample.metrics),
        )
        for sample in continuation.samples
    ]
    return ChainResult(
        graph=continuation.graph,
        samples=list(previous.samples) + continued_samples,
        attempted=int(previous.attempted) + int(continuation.attempted),
        valid=int(previous.valid) + int(continuation.valid),
        accepted=int(previous.accepted) + int(continuation.accepted),
    )


def _extend_until_mixed(
    results: List[ChainResult],
    config,
    seed: int,
    reference: Mapping[str, float] | None,
    energy=None,
) -> tuple[List[ChainResult], dict, int]:
    """Continue frozen-kernel chains until diagnostics pass or budget expires."""

    extensions = 0
    while True:
        summary = summarize_chain_metrics(
            _tail_metric_chains(results, config.diagnostic_window),
            reference=reference,
        )
        if _mixing_pass(summary, config):
            return results, summary, extensions
        if extensions >= int(config.chain_max_extensions):
            return results, summary, extensions

        updated = []
        for chain_id, result in enumerate(results):
            continuation = run_switch_chain(
                result.graph,
                GraphletFeatureExtractor(
                    result.graph,
                    tracked_keys=None if energy is None else energy.selection.keys,
                ),
                steps=int(config.chain_extension_steps),
                burn_in=max(int(config.thin), 1),
                thin=max(int(config.thin), 1),
                seed=(
                    int(seed)
                    + 10_000_019 * (extensions + 1)
                    + 1009 * chain_id
                ),
                energy=energy,
                verify_every=int(config.verify_every),
                curveball_probability=float(config.curveball_probability),
            )
            updated.append(_merge_chain_results(result, continuation))
        results = updated
        extensions += 1
        print(
            f"[STRUCTURE-ENERGY] adaptive_extension={extensions} "
            f"samples_per_chain={min(len(item.samples) for item in results)}",
            flush=True,
        )


def _run_chains(
    source: TypedGraph,
    count: int,
    steps: int,
    burn_in: int,
    thin: int,
    seed: int,
    verify_every: int,
    curveball_probability: float,
    energy=None,
) -> List[ChainResult]:
    results = []
    for chain_id in range(int(count)):
        chain_seed = int(seed) + 1009 * chain_id
        plan = sample_role_plan(source, chain_seed)
        initial = realize_role_plan(source, plan)
        extractor = GraphletFeatureExtractor(
            initial,
            tracked_keys=None if energy is None else energy.selection.keys,
        )
        print(
            f"[STRUCTURE-ENERGY] chain={chain_id} seed={chain_seed} "
            f"steps={steps} burn_in={burn_in} energy={energy is not None}",
            flush=True,
        )
        result = run_switch_chain(
            initial,
            extractor,
            steps=steps,
            burn_in=burn_in,
            thin=thin,
            seed=chain_seed + 17,
            energy=energy,
            verify_every=verify_every,
            curveball_probability=curveball_probability,
        )
        validate_role_realization(result.graph, plan)
        print(
            f"[STRUCTURE-ENERGY] chain={chain_id} "
            f"valid={result.valid_rate:.3f} accept={result.acceptance_rate:.3f} "
            f"samples={len(result.samples)}",
            flush=True,
        )
        results.append(result)
    return results


def _run_checkpoint_chains(
    checkpoint: StructureEnergyCheckpoint,
    count: int,
    steps: int,
    burn_in: int,
    thin: int,
    seed: int,
    verify_every: int,
    curveball_probability: float,
) -> List[ChainResult]:
    """Generate using only checkpoint state; no training graph is accepted."""

    results = []
    energy = checkpoint.energy()
    for chain_id in range(int(count)):
        chain_seed = int(seed) + 1009 * chain_id
        schema, plan = checkpoint.sample_role_plan(chain_seed)
        initial = realize_role_plan(schema, plan)
        print(
            f"[CHECKPOINT-SAMPLER] chain={chain_id} seed={chain_seed} "
            f"steps={steps} burn_in={burn_in}",
            flush=True,
        )
        result = run_switch_chain(
            initial,
            GraphletFeatureExtractor(
                initial, tracked_keys=checkpoint.selection.keys
            ),
            steps=steps,
            burn_in=burn_in,
            thin=thin,
            seed=chain_seed + 17,
            energy=energy,
            verify_every=verify_every,
            curveball_probability=curveball_probability,
        )
        validate_role_realization(result.graph, plan)
        print(
            f"[CHECKPOINT-SAMPLER] chain={chain_id} "
            f"valid={result.valid_rate:.3f} accept={result.acceptance_rate:.3f} "
            f"samples={len(result.samples)}",
            flush=True,
        )
        results.append(result)
    return results


def run_gates(config: GateConfig) -> dict:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    source = load_typed_graph(config.dataset_dir, active_only=True)
    source.validate_disjoint_family_domains()
    real_extractor = GraphletFeatureExtractor(source)
    real_features = real_extractor.extract(source)
    real_metrics = graph_metrics(source)

    active_families = {
        name: len(edges) for name, edges in source.family_edges.items()
    }
    print(
        f"[GATE-0] nodes={source.num_nodes} edges={source.num_edges} "
        f"families={active_families}",
        flush=True,
    )
    role_plan = sample_role_plan(source, config.seed)
    role_realization = realize_role_plan(source, role_plan)
    validate_role_realization(role_realization, role_plan)
    role_metrics = graph_metrics(role_realization)
    _write_json(config.output_dir / "gate0_role_graph.json", role_realization.to_json_dict())

    null_results = _run_chains(
        source,
        count=config.null_chains,
        steps=config.null_steps,
        burn_in=config.null_burn_in,
        thin=config.thin,
        seed=config.seed + 10_000,
        verify_every=config.verify_every,
        curveball_probability=config.curveball_probability,
        energy=None,
    )
    null_results, null_summary, null_extensions = _extend_until_mixed(
        null_results,
        config,
        seed=config.seed + 11_000,
        reference=None,
        energy=None,
    )
    null_samples_by_chain = [
        [
            sample.features
            for sample in result.samples[-max(int(config.diagnostic_window), 1) :]
        ]
        for result in null_results
    ]
    null_samples = [sample for chain in null_samples_by_chain for sample in chain]
    if len(null_samples) < 4:
        raise RuntimeError(
            f"null chains produced only {len(null_samples)} samples; need at least 4"
        )
    for chain_id, result in enumerate(null_results):
        _write_json(
            config.output_dir / f"gate1_null_chain_{chain_id}.json",
            _chain_payload(result),
        )
        _write_json(
            config.output_dir / f"gate1_null_graph_{chain_id}.json",
            result.graph.to_json_dict(),
        )

    if len(null_samples_by_chain) >= 2 and len(null_samples_by_chain[-1]) > 0:
        discovery_samples = [
            sample for chain in null_samples_by_chain[:-1] for sample in chain
        ]
        heldout_samples = list(null_samples_by_chain[-1])
    else:
        discovery_samples = null_samples
        heldout_samples = None
    if len(discovery_samples) < 2:
        raise RuntimeError("too few independent null samples for feature discovery")

    selection = select_features(
        real_features,
        discovery_samples,
        max_features=config.max_features,
        min_support=config.min_feature_support,
    )
    nce_energy, fit_report = fit_pairwise_nce(
        real_features,
        discovery_samples,
        selection,
        heldout_samples=heldout_samples,
        l2=config.nce_l2,
        max_weight=config.nce_max_weight,
        seed=config.seed,
    )
    _write_json(
        config.output_dir / "gate2_discriminator.json", nce_energy.to_json_dict()
    )
    _write_json(config.output_dir / "gate2_fit_report.json", fit_report.to_json_dict())

    energy, moment_report, _ = fit_moment_matching_energy(
        real_features,
        selection,
        [result.graph for result in null_results],
        rounds=config.moment_rounds,
        steps_per_round=config.moment_steps_per_round,
        samples_per_round=config.moment_samples_per_round,
        learning_rate=config.moment_learning_rate,
        l2=config.moment_l2,
        max_weight=config.nce_max_weight,
        tolerance=config.moment_tolerance,
        seed=config.seed + 15_000,
        initial_weights=nce_energy.weights,
        verify_every=config.verify_every,
        curveball_probability=config.curveball_probability,
        max_batches_per_round=config.moment_max_batches_per_round,
        min_samples_per_chain=config.moment_min_samples_per_chain,
        min_ess_per_chain=config.moment_min_ess_per_chain,
        rhat_threshold=config.moment_rhat_threshold,
        drift_z_threshold=config.moment_drift_z_threshold,
        warmup_steps_per_round=config.moment_warmup_steps_per_round,
        step_decay=config.moment_step_decay,
        update_norm_clip=config.moment_update_norm_clip,
        diagnostic_quantile=config.moment_diagnostic_quantile,
    )
    _write_json(config.output_dir / "gate3_energy_model.json", energy.to_json_dict())
    _write_json(
        config.output_dir / "gate3_moment_fit_report.json",
        moment_report.to_json_dict(),
    )
    checkpoint = StructureEnergyCheckpoint.from_training_graph(source, energy)
    checkpoint.save(config.output_dir / "structure_energy_checkpoint.json")

    energy_results = _run_checkpoint_chains(
        checkpoint,
        count=config.energy_chains,
        steps=config.energy_steps,
        burn_in=config.energy_burn_in,
        thin=config.thin,
        seed=config.seed + 20_000,
        verify_every=config.verify_every,
        curveball_probability=config.curveball_probability,
    )
    energy_results, energy_summary, energy_extensions = _extend_until_mixed(
        energy_results,
        config,
        seed=config.seed + 21_000,
        reference=real_metrics,
        energy=energy,
    )
    for chain_id, result in enumerate(energy_results):
        _write_json(
            config.output_dir / f"gate3_energy_chain_{chain_id}.json",
            _chain_payload(result),
        )
        _write_json(
            config.output_dir / f"gate3_energy_graph_{chain_id}.json",
            result.graph.to_json_dict(),
        )

    generated_pairwise_jaccard = [
        _edge_jaccard(energy_results[left].graph, energy_results[right].graph)
        for left in range(len(energy_results))
        for right in range(left + 1, len(energy_results))
    ]
    generated_source_jaccard = [
        _edge_jaccard(result.graph, source) for result in energy_results
    ]
    triangle_error = _relative_error(
        energy_summary["triangles"]["mean"], real_metrics["triangles"]
    )
    clustering_error = _relative_error(
        energy_summary["clustering"]["mean"], real_metrics["clustering"]
    )
    gate_report = {
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in config.__dict__.items()
        },
        "graph": {
            "nodes": source.num_nodes,
            "active_families": active_families,
            "real_metrics": real_metrics,
            "role_realization_metrics": role_metrics,
        },
        "gate0": {
            "passed": True,
            "degree_realization_exact": True,
        },
        "gate1": {
            "passed": bool(
                all(result.samples for result in null_results)
                and _mixing_pass(null_summary, config)
            ),
            "summary": null_summary,
            "valid_rates": [result.valid_rate for result in null_results],
            "acceptance_rates": [result.acceptance_rate for result in null_results],
            "adaptive_extensions": int(null_extensions),
        },
        "gate2": {
            "passed": bool(fit_report.success and fit_report.heldout_pair_accuracy > 0.5),
            "selected_features": list(selection.keys),
            "fit": fit_report.to_json_dict(),
        },
        "gate3": {
            "passed": bool(
                triangle_error <= config.structure_tolerance
                and clustering_error <= config.structure_tolerance
                and _mixing_pass(energy_summary, config)
            ),
            "summary": energy_summary,
            "moment_fit": moment_report.to_json_dict(),
            "triangle_relative_error": triangle_error,
            "clustering_relative_error": clustering_error,
            "valid_rates": [result.valid_rate for result in energy_results],
            "acceptance_rates": [result.acceptance_rate for result in energy_results],
            "adaptive_extensions": int(energy_extensions),
            "pairwise_edge_jaccard_mean": float(np.mean(generated_pairwise_jaccard))
            if generated_pairwise_jaccard
            else 0.0,
            "source_edge_jaccard_mean": float(np.mean(generated_source_jaccard)),
        },
    }
    _write_json(config.output_dir / "gate_report.json", gate_report)
    print(
        "[GATE-REPORT] "
        f"gate0={int(gate_report['gate0']['passed'])} "
        f"gate1={int(gate_report['gate1']['passed'])} "
        f"gate2={int(gate_report['gate2']['passed'])} "
        f"gate3={int(gate_report['gate3']['passed'])} "
        f"tri_err={triangle_error:.4f} cluster_err={clustering_error:.4f}",
        flush=True,
    )
    return gate_report
