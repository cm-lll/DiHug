#!/usr/bin/env python3
"""Benchmark exact multi-chain batching for the frozen sparse-HGT context.

This gate does not alter graph transitions. It compares independently encoded
chains with one block-diagonal sparse-HGT batch and requires hidden states and
candidate logits to agree before reporting speedups.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch

from sparse_diffusion.structure_energy.cardinality_conditioned_edit import (
    FamilyCountCondition,
)
from sparse_diffusion.structure_energy.gpu_role_guided_edit import GpuTypedGraphState
from sparse_diffusion.structure_energy.mixed_edit_diffusion import MarginalGraphPrior
from sparse_diffusion.structure_energy.role_guided_edit_diffusion import (
    ExchangeableRoleDegreePrior,
)
from sparse_diffusion.structure_energy.rooted_context_edit import (
    load_rooted_context_checkpoint,
)


def _time_cuda(fn, repeats: int) -> float:
    torch.cuda.synchronize()
    started = time.perf_counter()
    for _ in range(int(repeats)):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - started) / float(repeats)


def _context_inputs(states, model, time_fraction: float):
    node_features = torch.stack(
        [state.context_node_features(time_fraction) for state in states]
    )
    node_types = torch.stack([state.node_types for state in states])
    edge_parts = []
    type_parts = []
    node_count = states[0].num_nodes
    for batch_id, state in enumerate(states):
        edge_index, edge_types = state.context_edge_tensors()
        edge_parts.append(edge_index + batch_id * node_count)
        type_parts.append(edge_types)
    edge_index = torch.cat(edge_parts, dim=1)
    edge_types = torch.cat(type_parts)
    role_conditions = None
    if model.role_condition_dim > 0:
        raw = torch.stack([state.target_rows for state in states])
        role_conditions = model.normalize_role_rows(raw, node_types)
    return node_features, node_types, edge_index, edge_types, role_conditions


def _sequential_encode(states, model, time_fraction: float):
    outputs = []
    for state in states:
        edge_index, edge_types = state.context_edge_tensors()
        role_conditions = None
        if model.role_condition_dim > 0:
            role_conditions = model.normalize_role_rows(
                state.target_rows[None], state.node_types[None]
            )
        outputs.append(
            model.encode(
                state.context_node_features(time_fraction)[None],
                state.node_types[None],
                edge_index,
                edge_types,
                role_conditions=role_conditions,
            )
        )
    return torch.cat(outputs, dim=0)


def _batched_encode(states, model, time_fraction: float):
    inputs = _context_inputs(states, model, time_fraction)
    return model.encode(*inputs[:4], role_conditions=inputs[4])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--context-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--chains", default="1,2,4,8")
    parser.add_argument("--candidates", default="16,32,64")
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1729)
    args = parser.parse_args()

    if not torch.cuda.is_available() or torch.device(args.device).type != "cuda":
        raise RuntimeError("batched context benchmark requires CUDA")
    model, payload = load_rooted_context_checkpoint(
        args.context_checkpoint, legacy_scorer=None, device=args.device
    )
    model.eval()
    embedded = payload["legacy_checkpoint_payload"]
    marginal = MarginalGraphPrior.from_json_dict(embedded["prior"])
    role_prior = ExchangeableRoleDegreePrior.from_json_dict(
        payload["exchangeable_role_prior"]
    )
    counts = role_prior.family_counts(marginal.family_endpoints)
    condition = FamilyCountCondition(marginal=marginal, observed_counts=counts)
    max_chains = max(int(value) for value in args.chains.split(","))
    states = []
    generators = []
    for chain in range(max_chains):
        graph = condition.sample_graph(args.seed + 71 + chain, mode="binomial")
        roles = role_prior.sample_assignment_for_graph(
            graph, args.seed + 91_919 + chain
        )
        states.append(GpuTypedGraphState(graph, roles, args.device, use_triton=True))
        generator = torch.Generator(device=torch.device(args.device))
        generator.manual_seed(args.seed + 1_000_003 * chain)
        generators.append(generator)

    rows = []
    with torch.no_grad():
        for chain_count in [int(value) for value in args.chains.split(",")]:
            subset = states[:chain_count]
            for _ in range(args.warmup):
                _sequential_encode(subset, model, 0.5)
                _batched_encode(subset, model, 0.5)
            sequential_hidden = _sequential_encode(subset, model, 0.5)
            batched_hidden = _batched_encode(subset, model, 0.5)
            hidden_error = float((sequential_hidden - batched_hidden).abs().max().cpu())
            if hidden_error > 1.0e-5:
                raise RuntimeError(
                    f"batched HGT hidden mismatch: chains={chain_count} error={hidden_error}"
                )
            sequential_encode_seconds = _time_cuda(
                lambda: _sequential_encode(subset, model, 0.5), args.repeats
            )
            batched_encode_seconds = _time_cuda(
                lambda: _batched_encode(subset, model, 0.5), args.repeats
            )

            for candidate_count in [
                int(value) for value in args.candidates.split(",")
            ]:
                batches = [
                    state.random_candidates(
                        candidate_count,
                        generators[index],
                        role_phase=False,
                        targeted_fraction=0.0,
                        closure_fraction=1.0,
                        proposal_pool_size=0,
                    )
                    for index, state in enumerate(subset)
                ]
                features = torch.stack(
                    [
                        state.compact_features(
                            batch, 0.5, int(model.joint.legacy_scorer.input_dim)
                        )
                        for state, batch in zip(subset, batches)
                    ]
                )
                endpoints = torch.stack(
                    [state.candidate_endpoints(batch) for state, batch in zip(subset, batches)]
                )
                sequential_logits = torch.cat(
                    [
                        model.score_encoded(
                            sequential_hidden[index : index + 1],
                            features[index : index + 1],
                            endpoints[index : index + 1],
                        )
                        for index in range(chain_count)
                    ],
                    dim=0,
                )
                batched_logits = model.score_encoded(
                    batched_hidden, features, endpoints
                )
                logit_error = float(
                    (sequential_logits - batched_logits).abs().max().cpu()
                )
                if logit_error > 1.0e-5:
                    raise RuntimeError(
                        "batched candidate logit mismatch: "
                        f"chains={chain_count} candidates={candidate_count} "
                        f"error={logit_error}"
                    )

                sequential_total_seconds = _time_cuda(
                    lambda: torch.cat(
                        [
                            model.score_encoded(
                                _sequential_encode([state], model, 0.5),
                                features[index : index + 1],
                                endpoints[index : index + 1],
                            )
                            for index, state in enumerate(subset)
                        ],
                        dim=0,
                    ),
                    args.repeats,
                )
                batched_total_seconds = _time_cuda(
                    lambda: model.score_encoded(
                        _batched_encode(subset, model, 0.5), features, endpoints
                    ),
                    args.repeats,
                )
                rows.append(
                    {
                        "chains": chain_count,
                        "candidates": candidate_count,
                        "hidden_max_abs_error": hidden_error,
                        "logit_max_abs_error": logit_error,
                        "sequential_encode_ms": sequential_encode_seconds * 1e3,
                        "batched_encode_ms": batched_encode_seconds * 1e3,
                        "encode_speedup": sequential_encode_seconds
                        / batched_encode_seconds,
                        "sequential_encode_score_ms": sequential_total_seconds * 1e3,
                        "batched_encode_score_ms": batched_total_seconds * 1e3,
                        "end_to_end_speedup": sequential_total_seconds
                        / batched_total_seconds,
                        "peak_reserved_mb": torch.cuda.max_memory_reserved() / 2**20,
                    }
                )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    args.output.with_suffix(".json").write_text(
        json.dumps({"rows": rows}, indent=2, sort_keys=True)
    )
    print(json.dumps(rows, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
