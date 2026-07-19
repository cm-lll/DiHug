#!/usr/bin/env python3
"""Compare canonical CPU and ordered GPU edit proposal distributions."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from sparse_diffusion.structure_energy.cardinality_conditioned_edit import (
    FamilyCountCondition,
)
from sparse_diffusion.structure_energy.gpu_role_guided_edit import (
    GpuEditBatch,
    GpuTypedGraphState,
)
from sparse_diffusion.structure_energy.mixed_edit_diffusion import (
    SCORER_ACTIONS,
    EditProposal,
    MarginalGraphPrior,
    typed_closure_delta,
)
from sparse_diffusion.structure_energy.role_guided_edit_diffusion import (
    ExchangeableRoleDegreePrior,
    proposal_relation_degree_gain,
    sample_role_alignment_candidates,
    sample_switch_refinement_candidates,
)
from sparse_diffusion.structure_energy.rooted_context_edit import (
    load_rooted_context_checkpoint,
)


def _gpu_proposals(
    batch: GpuEditBatch, family_names: tuple[str, ...]
) -> list[EditProposal]:
    action = batch.action.cpu().tolist()
    family = batch.family.cpu().tolist()
    old_edges = batch.old_edges.cpu().tolist()
    new_edges = batch.new_edges.cpu().tolist()
    old_valid = batch.old_valid.cpu().tolist()
    new_valid = batch.new_valid.cpu().tolist()
    proposals = []
    for row in range(len(action)):
        action_name = SCORER_ACTIONS[int(action[row])]
        proposals.append(
            EditProposal(
                action_name,
                family_names[int(family[row])],
                tuple(
                    tuple(map(int, old_edges[row][slot]))
                    for slot in range(2)
                    if old_valid[row][slot]
                ),
                tuple(
                    tuple(map(int, new_edges[row][slot]))
                    for slot in range(2)
                    if new_valid[row][slot]
                ),
            )
        )
    return proposals


def _proposal_key(proposal: EditProposal) -> tuple:
    return (
        proposal.action,
        "" if proposal.action == "noop" else proposal.family,
        tuple(sorted(proposal.old_edges)),
        tuple(sorted(proposal.new_edges)),
    )


def _summarize(
    graph,
    proposals: list[EditProposal],
    target_rows: np.ndarray,
    family_names: tuple[str, ...],
    candidates_per_batch: int,
) -> dict:
    family_to_id = {name: index for index, name in enumerate(family_names)}
    action_counts = Counter(proposal.action for proposal in proposals)
    family_counts = Counter(
        proposal.family for proposal in proposals if proposal.action != "noop"
    )
    gains = np.asarray(
        [
            proposal_relation_degree_gain(
                graph, proposal, target_rows, family_to_id
            )
            for proposal in proposals
        ],
        dtype=np.float64,
    )
    closure = np.asarray(
        [float(typed_closure_delta(graph, proposal).sum()) for proposal in proposals],
        dtype=np.float64,
    )
    batches = [
        proposals[offset : offset + int(candidates_per_batch)]
        for offset in range(0, len(proposals), int(candidates_per_batch))
    ]
    return {
        "actions": action_counts,
        "families": family_counts,
        "gains": gains,
        "closure": closure,
        "unique_fraction": float(
            np.mean(
                [
                    len({_proposal_key(proposal) for proposal in batch})
                    / max(len(batch), 1)
                    for batch in batches
                ]
            )
        ),
    }


def _total_variation(left: Counter, right: Counter) -> float:
    keys = set(left) | set(right)
    left_total = max(sum(left.values()), 1)
    right_total = max(sum(right.values()), 1)
    return 0.5 * sum(
        abs(left[key] / left_total - right[key] / right_total) for key in keys
    )


def _standardized_mean_difference(left: np.ndarray, right: np.ndarray) -> float:
    variance = 0.5 * (float(left.var()) + float(right.var()))
    return abs(float(left.mean()) - float(right.mean())) / max(variance**0.5, 1.0e-8)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--context-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batches", type=int, default=128)
    parser.add_argument("--candidates", type=int, default=16)
    args = parser.parse_args()

    context, payload = load_rooted_context_checkpoint(
        args.context_checkpoint, legacy_scorer=None, device=args.device
    )
    embedded = payload["legacy_checkpoint_payload"]
    marginal = MarginalGraphPrior.from_json_dict(embedded["prior"])
    role_prior = ExchangeableRoleDegreePrior.from_json_dict(
        payload["exchangeable_role_prior"]
    )
    condition = FamilyCountCondition(
        marginal=marginal,
        observed_counts=role_prior.family_counts(marginal.family_endpoints),
    )
    graph = condition.sample_graph(args.seed + 71, mode="binomial")
    target_rows = role_prior.sample_assignment_for_graph(
        graph, args.seed + 91_919
    )
    state = GpuTypedGraphState(graph, target_rows, args.device, use_triton=True)
    cpu_rng = np.random.default_rng(args.seed + 7_000_001)
    gpu_rng = torch.Generator(device=torch.device(args.device))
    gpu_rng.manual_seed(args.seed + 7_000_003)
    family_names = tuple(sorted(graph.family_specs))
    reports = {}

    for phase in ("role", "closure"):
        cpu_proposals = []
        gpu_proposals = []
        cpu_started = time.perf_counter()
        for _ in range(args.batches):
            if phase == "role":
                candidates = sample_role_alignment_candidates(
                    graph,
                    target_rows,
                    family_names,
                    cpu_rng,
                    args.candidates,
                    targeted_fraction=0.5,
                )
            else:
                candidates = sample_switch_refinement_candidates(
                    graph,
                    cpu_rng,
                    args.candidates,
                    closure_fraction=1.0,
                )
            cpu_proposals.extend(candidates)
        cpu_seconds = time.perf_counter() - cpu_started

        torch.cuda.synchronize()
        gpu_started = time.perf_counter()
        gpu_batches = []
        for _ in range(args.batches):
            gpu_batches.append(
                state.canonical_ordered_candidates(
                    args.candidates,
                    gpu_rng,
                    role_phase=phase == "role",
                    targeted_fraction=0.5,
                    closure_fraction=1.0,
                )
            )
        torch.cuda.synchronize()
        gpu_seconds = time.perf_counter() - gpu_started
        for batch in gpu_batches:
            gpu_proposals.extend(_gpu_proposals(batch, family_names))

        cpu = _summarize(
            graph, cpu_proposals, target_rows, family_names, args.candidates
        )
        gpu = _summarize(
            graph, gpu_proposals, target_rows, family_names, args.candidates
        )
        noop_cpu = cpu["actions"].get("noop", 0) / args.batches
        noop_gpu = gpu["actions"].get("noop", 0) / args.batches
        report = {
            "cpu_batches_per_second": args.batches / cpu_seconds,
            "gpu_batches_per_second": args.batches / gpu_seconds,
            "speedup": cpu_seconds / gpu_seconds,
            "action_total_variation": _total_variation(cpu["actions"], gpu["actions"]),
            "family_total_variation": _total_variation(cpu["families"], gpu["families"]),
            "degree_gain_smd": _standardized_mean_difference(
                cpu["gains"], gpu["gains"]
            ),
            "closure_delta_smd": _standardized_mean_difference(
                cpu["closure"], gpu["closure"]
            ),
            "cpu_noops_per_batch": noop_cpu,
            "gpu_noops_per_batch": noop_gpu,
            "cpu_unique_fraction": cpu["unique_fraction"],
            "gpu_unique_fraction": gpu["unique_fraction"],
            "cpu_degree_gain_mean": float(cpu["gains"].mean()),
            "gpu_degree_gain_mean": float(gpu["gains"].mean()),
            "cpu_closure_delta_mean": float(cpu["closure"].mean()),
            "gpu_closure_delta_mean": float(gpu["closure"].mean()),
        }
        report["distribution_gate"] = bool(
            report["action_total_variation"] <= 0.05
            and report["family_total_variation"] <= 0.10
            and report["degree_gain_smd"] <= 0.25
            and report["closure_delta_smd"] <= 0.25
            and noop_cpu == 1.0
            and noop_gpu == 1.0
            and cpu["unique_fraction"] == 1.0
            and gpu["unique_fraction"] == 1.0
        )
        reports[phase] = report

    payload = {
        "candidates": args.candidates,
        "batches": args.batches,
        "phases": reports,
        "all_distribution_gates": all(
            report["distribution_gate"] for report in reports.values()
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
