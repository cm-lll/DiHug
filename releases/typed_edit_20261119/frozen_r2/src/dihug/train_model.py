#!/usr/bin/env python3
"""Run fair mixed edit-switch diffusion gates, training, and sampling."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from dihug.core.graph import load_typed_graph
from dihug.core.exchangeable_role_prior import (
    ExchangeableJointRolePrior,
)
from dihug.core.factorized_edit_denoiser import (
    build_factorized_edit_dataset,
    evaluate_factorized_fixed_t,
    load_factorized_edit_checkpoint,
    sample_factorized_edit_diffusion,
    save_factorized_edit_checkpoint,
    train_factorized_edit_denoiser,
)
from dihug.core.mixed_edit_diffusion import (
    EDIT_ACTIONS,
    FixedCountKernelSchedule,
    MIXED_EDIT_FEATURE_PROFILES,
    MarginalGraphPrior,
    MixedKernelSchedule,
    build_mixed_edit_diffusion_dataset,
    edge_edit_distance,
    evaluate_closed_form_policy,
    evaluate_fixed_t_denoising,
    load_mixed_edit_checkpoint,
    mixed_edit_feature_width,
    mixed_edit_time_feature_index,
    permute_typed_graph_within_types,
    run_edit_distance_oracle,
    sample_mixed_edit_diffusion,
    save_mixed_edit_checkpoint,
)
from dihug.core.role_guided_edit_diffusion import (
    sample_role_alignment_candidates,
    sample_switch_refinement_candidates,
)
from dihug.core.switch_chain import graph_metrics
from dihug.core.switch_diffusion import train_switch_denoiser


KERNELS = {
    "bd": ("birth", "death"),
    "bdr": ("birth", "death", "relocation"),
    "bdrs": EDIT_ACTIONS,
    "rs": ("relocation", "switch"),
    "switch": ("switch",),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("data/imdb"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("oracle", "train", "sample"), required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument(
        "--init-checkpoint",
        type=Path,
        help="Optional edit-scorer pretraining checkpoint for strict fine-tuning.",
    )
    parser.add_argument(
        "--denoiser", choices=("factorized", "listwise"), default="factorized"
    )
    parser.add_argument("--kernel", choices=tuple(KERNELS), default="bdrs")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--trajectories", type=int, default=16)
    parser.add_argument("--forward-steps", type=int, default=1000)
    parser.add_argument("--record-every", type=int, default=10)
    parser.add_argument(
        "--forward-mode",
        choices=("micro", "closed_form", "fixed_count"),
        default="micro",
    )
    parser.add_argument(
        "--fixed-count-schedule",
        choices=("legacy", "monotone"),
        default="legacy",
        help=(
            "Forward operation clock for fixed-count diffusion. 'legacy' "
            "preserves published checkpoints; 'monotone' increases relocation "
            "toward high noise and switch toward the clean endpoint."
        ),
    )
    parser.add_argument("--fixed-count-schedule-power", type=float, default=2.0)
    parser.add_argument("--fixed-count-schedule-floor", type=float, default=0.05)
    parser.add_argument(
        "--forward-family-power",
        type=float,
        default=0.0,
        help=(
            "Fixed-count family clock exponent: 0 is uniform over families, "
            "1 is approximately uniform per existing edge."
        ),
    )
    parser.add_argument("--forward-family-pseudocount", type=float, default=1.0)
    parser.add_argument(
        "--closed-form-noise-fractions", default="0.05,0.2,0.5,0.8,1.0"
    )
    parser.add_argument(
        "--anonymous-role-condition",
        action="store_true",
        help="Condition on an iid type-exchangeable role latent; edges stay Bernoulli.",
    )
    parser.add_argument("--train-candidates", type=int, default=32)
    parser.add_argument(
        "--training-candidate-policy",
        choices=("random_injected", "guided_injected", "guided_free"),
        default="random_injected",
        help=(
            "Training-list construction. guided_free uses deployment-style "
            "guided support without forcing the recorded inverse into the list."
        ),
    )
    parser.add_argument("--training-guided-role-fraction", type=float, default=0.5)
    parser.add_argument(
        "--training-guided-closure-fraction", type=float, default=1.0
    )
    parser.add_argument(
        "--training-guided-phase-cutoff",
        type=float,
        default=0.5,
        help="Use role-guided lists at or above this forward-noise fraction.",
    )
    parser.add_argument(
        "--training-guided-closure-hub-bias-power", type=float, default=0.0
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--sample-steps", type=int, default=20000)
    parser.add_argument("--sample-candidates", type=int, default=64)
    parser.add_argument("--sample-temperature", type=float, default=0.0)
    parser.add_argument(
        "--factorized-decode",
        choices=("hierarchical", "joint_leaf"),
        default="hierarchical",
    )
    parser.add_argument("--metrics-every", type=int, default=1000)
    parser.add_argument("--fixed-t-trials", type=int, default=64)
    parser.add_argument(
        "--dataset-cache",
        type=Path,
        help="Optional shared NPZ cache for listwise candidate tensors.",
    )
    parser.add_argument("--balanced-action-aux-weight", type=float, default=0.1)
    parser.add_argument(
        "--feature-profile",
        choices=MIXED_EDIT_FEATURE_PROFILES,
        default="typed_closure_full_v2",
    )
    parser.add_argument(
        "--training-objective",
        choices=("inverse_ce", "utility_kl", "hybrid", "scheduled_hybrid"),
        default="inverse_ce",
    )
    parser.add_argument("--inverse-loss-weight", type=float, default=1.0)
    parser.add_argument("--utility-loss-weight", type=float, default=1.0)
    parser.add_argument("--utility-temperature", type=float, default=1.0)
    parser.add_argument("--utility-schedule-power", type=float, default=1.0)
    parser.add_argument(
        "--disable-time-conditioning",
        action="store_true",
        help=(
            "Strict ablation: zero the diffusion-time feature in the dataset "
            "and the corresponding first-layer scorer weights."
        ),
    )
    parser.add_argument(
        "--role-corruption-fraction",
        type=float,
        default=0.0,
        help="Fraction of training trajectories bridged to weak anonymous role priors.",
    )
    parser.add_argument(
        "--role-correlation-strengths",
        default="0,0.25,0.5,0.75,1",
        help="Comma-separated rho schedule for role-corruption trajectories.",
    )
    parser.add_argument("--role-prior-components", type=int, default=64)
    parser.add_argument("--role-prior-randomization-steps-per-edge", type=float, default=10.0)
    parser.add_argument(
        "--type-orbit-augmentation",
        action="store_true",
        help="Use an independent within-type node permutation as each clean target.",
    )
    parser.add_argument(
        "--clean-noop-examples-per-trajectory",
        type=int,
        default=0,
        help="Number of clean-orbit noop examples added per trajectory.",
    )
    return parser.parse_args()


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "oracle":
        source = load_typed_graph(args.dataset_dir, active_only=True)
        clean_metrics = graph_metrics(source)
        prior = MarginalGraphPrior.from_training_graph(source)
        result = run_edit_distance_oracle(
            prior,
            source,
            steps=args.sample_steps,
            candidates_per_step=args.sample_candidates,
            seed=args.seed,
            actions=KERNELS[args.kernel],
            metrics_every=args.metrics_every,
        )
        payload = {
            "format": "dihug-mixed-edit-oracle-gate-v1",
            "diagnostic_only": True,
            "oracle_objective": "clean_edge_edit_distance",
            "kernel": args.kernel,
            "seed": args.seed,
            "prior": prior.to_json_dict(),
            "clean_metrics": clean_metrics,
            "initial": result.trace[0],
            "final": {
                "edge_edit_distance": edge_edit_distance(result.graph, source),
                **graph_metrics(result.graph),
            },
            "trace": list(result.trace),
            "action_counts": dict(result.action_counts),
            "proposed": result.proposed,
            "applied": result.applied,
        }
        write_json(args.output_dir / "oracle_report.json", payload)
        print(
            "[MIXED-EDIT-ORACLE] "
            f"kernel={args.kernel} "
            f"distance={result.trace[0]['edge_edit_distance']:.0f}->"
            f"{payload['final']['edge_edit_distance']:.0f} "
            f"tri={result.trace[0]['triangles']:.0f}->"
            f"{payload['final']['triangles']:.0f}",
            flush=True,
        )
        return

    if args.mode == "train":
        source = load_typed_graph(args.dataset_dir, active_only=True)
        prior = MarginalGraphPrior.from_training_graph(source)
        if not 0.0 <= float(args.role_corruption_fraction) <= 1.0:
            raise ValueError("--role-corruption-fraction must be in [0, 1]")
        if args.role_corruption_fraction > 0.0 and args.denoiser != "listwise":
            raise ValueError("role corruption currently supports only listwise denoisers")
        if args.role_corruption_fraction > 0.0 and args.kernel != "bdrs":
            raise ValueError("role corruption requires the reversible bdrs edit kernel")
        if args.role_corruption_fraction > 0.0 and args.dataset_cache is not None:
            raise ValueError("role corruption cannot reuse an unlabelled dataset cache")
        if (
            args.training_candidate_policy == "guided_free"
            and args.training_objective not in {"utility_kl", "hybrid", "scheduled_hybrid"}
        ):
            raise ValueError(
                "guided_free candidates require a utility-aware training objective"
            )
        if (
            args.type_orbit_augmentation
            and args.forward_mode == "micro"
            and args.role_corruption_fraction != 1.0
        ):
            raise ValueError(
                "type-orbit augmentation requires role-corruption-fraction=1 so "
                "the training terminal distribution matches the sampling prior"
            )
        if args.forward_mode == "closed_form" and args.denoiser != "listwise":
            raise ValueError("closed-form forward noise currently requires listwise")
        if args.forward_mode == "closed_form" and args.training_objective != "utility_kl":
            raise ValueError("closed-form free candidates require utility_kl training")
        if args.anonymous_role_condition and args.forward_mode != "closed_form":
            raise ValueError("anonymous role conditioning requires closed-form noise")
        if args.anonymous_role_condition and args.feature_profile != "typed_closure_role_v3":
            raise ValueError(
                "anonymous role conditioning requires typed_closure_role_v3"
            )
        if args.forward_mode == "fixed_count":
            if args.kernel != "rs":
                raise ValueError("fixed-count diffusion requires --kernel rs")
            allowed_fixed_objectives = {"inverse_ce", "scheduled_hybrid"}
            if args.training_candidate_policy == "guided_free":
                allowed_fixed_objectives.update({"utility_kl", "hybrid"})
            if args.training_objective not in allowed_fixed_objectives:
                raise ValueError(
                    "fixed-count diffusion objective is incompatible with the "
                    "selected training candidate policy"
                )
            if args.role_corruption_fraction != 0.0:
                raise ValueError(
                    "fixed-count diffusion cannot use birth/death role corruption"
                )
        elif args.fixed_count_schedule != "legacy":
            raise ValueError("--fixed-count-schedule applies only to fixed_count mode")
        elif args.forward_family_power != 0.0:
            raise ValueError("--forward-family-power applies only to fixed_count mode")

        forward_schedule = (
            FixedCountKernelSchedule(
                relocation_floor=args.fixed_count_schedule_floor,
                switch_floor=args.fixed_count_schedule_floor,
                power=args.fixed_count_schedule_power,
            )
            if args.fixed_count_schedule == "monotone"
            else MixedKernelSchedule()
        )

        clean_targets = {}
        if args.type_orbit_augmentation:
            clean_targets = {
                trajectory_id: permute_typed_graph_within_types(
                    source,
                    args.seed + 20_000_033 + trajectory_id,
                )
                for trajectory_id in range(int(args.trajectories))
            }

        role_corruption_targets = {}
        role_corruption_metadata = {
            "enabled": False,
            "fraction": 0.0,
            "strengths": [],
            "trajectory_ids": [],
            "forward_mode": args.forward_mode,
            "closed_form_noise_fractions": [
                float(value.strip())
                for value in args.closed_form_noise_fractions.split(",")
                if value.strip()
            ],
            "terminal_matches_sampling_prior": args.forward_mode
            in {"closed_form", "fixed_count"},
            "anonymous_role_condition": False,
        }
        training_candidate_diagnostics = {}
        diffusion_contract = {
            "format": "dihug-edit-diffusion-contract-v1",
            "forward_mode": args.forward_mode,
            "forward_actions": list(KERNELS[args.kernel]),
            "reverse_actions": list(KERNELS[args.kernel]),
            "reverse_actions_match_forward": True,
            "objective": (
                "sampled_inverse_edit_surrogate"
                if args.training_objective == "inverse_ce"
                else "time_weighted_x0_denoising_surrogate"
                if args.training_objective == "scheduled_hybrid"
                else args.training_objective
            ),
            "exact_full_elbo": False,
            "candidate_normalization": "sampled_legal_edit_set",
            "candidate_positive_injection_training_only": (
                args.training_candidate_policy != "guided_free"
            ),
            "training_candidate_policy": args.training_candidate_policy,
            "decoder": "categorical_or_zero_temperature_map",
            "forward_schedule": args.fixed_count_schedule,
            "forward_schedule_power": float(args.fixed_count_schedule_power),
            "forward_schedule_floor": float(args.fixed_count_schedule_floor),
            "forward_family_power": float(args.forward_family_power),
            "forward_family_pseudocount": float(
                args.forward_family_pseudocount
            ),
        }
        initial_state_dict = None
        if args.init_checkpoint is not None:
            initial_model, _, initial_metadata = load_mixed_edit_checkpoint(
                args.init_checkpoint, device="cpu"
            )
            if initial_metadata["feature_profile"] != args.feature_profile:
                raise ValueError(
                    "initial checkpoint feature profile does not match training"
                )
            initial_state_dict = initial_model.state_dict()
            diffusion_contract["pretraining"] = {
                "checkpoint": str(args.init_checkpoint),
                "sha256": hashlib.sha256(args.init_checkpoint.read_bytes()).hexdigest(),
                "role": "parameter_initialization_only",
            }
        if args.forward_mode == "fixed_count":
            diffusion_contract.update(
                state_space="schema_valid_graphs_conditioned_on_family_counts",
                terminal_prior="uniform_on_omega_M",
                terminal_prior_stationary=True,
                terminal_exact_sampler=True,
                family_counts_are_outer_latent=True,
                forward_kernel="symmetric_relocation_typed_2switch",
            )
        diffusion_contract["time_conditioning"] = not bool(
            args.disable_time_conditioning
        )
        latent_role_prior = None
        if args.anonymous_role_condition:
            latent_role_prior = ExchangeableJointRolePrior.from_training_graph(
                source,
                components=args.role_prior_components,
                seed=args.seed,
                randomization_steps_per_edge=0.0,
                role_model="categorical",
                role_component_sampling="iid",
            )
            role_corruption_metadata.update(
                anonymous_role_condition=True,
                anonymous_role_prior=latent_role_prior.to_json_dict(),
                role_component_sampling="iid",
                role_initializes_edges=False,
            )
        if args.role_corruption_fraction > 0.0:
            strengths = tuple(
                float(value.strip())
                for value in args.role_correlation_strengths.split(",")
                if value.strip()
            )
            if not strengths or any(not 0.0 <= value <= 1.0 for value in strengths):
                raise ValueError("role correlation strengths must all be in [0, 1]")
            role_count = min(
                int(args.trajectories),
                max(1, int(round(args.trajectories * args.role_corruption_fraction))),
            )
            role_ids = np.rint(
                np.linspace(0, int(args.trajectories) - 1, role_count)
            ).astype(np.int64)
            if len(set(role_ids.tolist())) != role_count:
                raise AssertionError("role corruption trajectory allocation is not unique")
            base_role_prior = ExchangeableJointRolePrior.from_training_graph(
                source,
                components=args.role_prior_components,
                seed=args.seed,
                randomization_steps_per_edge=(
                    args.role_prior_randomization_steps_per_edge
                ),
                role_model="categorical",
                role_component_sampling="systematic",
            )
            assigned_strengths = []
            for index, trajectory_id in enumerate(role_ids.tolist()):
                strength = strengths[index % len(strengths)]
                assigned_strengths.append(strength)
                role_corruption_targets[int(trajectory_id)] = replace(
                    base_role_prior,
                    role_correlation_strength=float(strength),
                ).sample(args.seed + 10_000_019 + int(trajectory_id))
            role_corruption_metadata = {
                "enabled": True,
                "fraction": float(args.role_corruption_fraction),
                "strength_schedule": list(strengths),
                "strengths": assigned_strengths,
                "trajectory_ids": role_ids.tolist(),
                "components": int(args.role_prior_components),
                "randomization_steps_per_edge": float(
                    args.role_prior_randomization_steps_per_edge
                ),
                "target_contains_training_adjacency": False,
                "corruption_bridge": "exact_birth_death_relocation",
            }
        if args.dataset_cache is not None and args.denoiser != "listwise":
            raise ValueError("--dataset-cache currently supports only listwise denoisers")
        if args.dataset_cache is not None and args.dataset_cache.exists():
            cached = np.load(args.dataset_cache)
            from dihug.core.switch_diffusion import (
                SwitchDiffusionDataset,
            )

            dataset = SwitchDiffusionDataset(
                candidate_features=cached["candidate_features"],
                candidate_improvements=cached["candidate_improvements"],
                targets=cached["targets"],
                trajectory_ids=cached["trajectory_ids"],
            )
            expected_width = mixed_edit_feature_width(
                args.feature_profile,
                family_count=len(source.family_specs),
                type_count=len(source.type_names),
            )
            if int(dataset.candidate_features.shape[-1]) != expected_width:
                raise ValueError(
                    f"cached feature width {dataset.candidate_features.shape[-1]} "
                    f"does not match {args.feature_profile} width {expected_width}"
                )
            if "feature_profile" in cached.files:
                cached_profile = str(cached["feature_profile"].item())
                if cached_profile != args.feature_profile:
                    raise ValueError(
                        f"cached feature profile {cached_profile!r} does not match "
                        f"requested profile {args.feature_profile!r}"
                    )
            if "kernel" in cached.files:
                cached_kernel = str(cached["kernel"].item())
                if cached_kernel != args.kernel:
                    raise ValueError(
                        f"cached kernel {cached_kernel!r} does not match requested "
                        f"kernel {args.kernel!r}"
                    )
            for key, requested in (
                ("fixed_count_schedule", args.fixed_count_schedule),
                ("forward_family_power", float(args.forward_family_power)),
                ("training_candidate_policy", args.training_candidate_policy),
            ):
                if key not in cached.files:
                    if requested not in {"legacy", "random_injected", 0.0}:
                        raise ValueError(
                            f"cached dataset predates non-default {key}"
                        )
                    continue
                cached_value = cached[key].item()
                if cached_value != requested:
                    raise ValueError(
                        f"cached {key} {cached_value!r} does not match "
                        f"requested {requested!r}"
                    )
        elif args.denoiser == "factorized":
            dataset = build_factorized_edit_dataset(
                source,
                trajectories=args.trajectories,
                forward_steps=args.forward_steps,
                record_every=args.record_every,
                candidates_per_example=args.train_candidates,
                seed=args.seed,
                actions=KERNELS[args.kernel],
            )
        else:
            candidate_list_builder = None
            candidate_positive_injection = True
            if args.training_candidate_policy != "random_injected":
                if args.forward_mode != "fixed_count" or args.kernel != "rs":
                    raise ValueError(
                        "guided training candidates require fixed-count rs diffusion"
                    )
                if not 0.0 <= args.training_guided_phase_cutoff <= 1.0:
                    raise ValueError(
                        "training guided phase cutoff must lie in [0,1]"
                    )
                family_names = tuple(sorted(source.family_specs))

                def candidate_list_builder(
                    graph,
                    clean_graph,
                    rng,
                    count,
                    time_fraction,
                    actions,
                ):
                    if float(time_fraction) >= float(
                        args.training_guided_phase_cutoff
                    ):
                        target_rows = np.stack(
                            [
                                clean_graph.family_degrees(family)
                                for family in family_names
                            ],
                            axis=1,
                        )
                        return sample_role_alignment_candidates(
                            graph,
                            target_rows,
                            family_names,
                            rng,
                            count,
                            targeted_fraction=args.training_guided_role_fraction,
                            actions=actions,
                        )
                    return sample_switch_refinement_candidates(
                        graph,
                        rng,
                        count,
                        closure_fraction=args.training_guided_closure_fraction,
                        hub_bias_power=(
                            args.training_guided_closure_hub_bias_power
                        ),
                    )

                candidate_positive_injection = (
                    args.training_candidate_policy == "guided_injected"
                )
            dataset = build_mixed_edit_diffusion_dataset(
                source,
                trajectories=args.trajectories,
                forward_steps=args.forward_steps,
                record_every=args.record_every,
                candidates_per_example=args.train_candidates,
                seed=args.seed,
                actions=KERNELS[args.kernel],
                feature_profile=args.feature_profile,
                role_corruption_targets=role_corruption_targets,
                clean_targets=clean_targets,
                clean_noop_examples_per_trajectory=(
                    args.clean_noop_examples_per_trajectory
                ),
                forward_mode=args.forward_mode,
                closed_form_noise_fractions=tuple(
                    float(value.strip())
                    for value in args.closed_form_noise_fractions.split(",")
                    if value.strip()
                ),
                use_role_condition=args.anonymous_role_condition,
                schedule=forward_schedule,
                family_clock_power=args.forward_family_power,
                family_clock_pseudocount=args.forward_family_pseudocount,
                candidate_list_builder=candidate_list_builder,
                candidate_positive_injection=candidate_positive_injection,
                candidate_diagnostics=training_candidate_diagnostics,
            )
        inverse_queries = training_candidate_diagnostics.get("inverse_queries", 0.0)
        if inverse_queries > 0.0:
            training_candidate_diagnostics["natural_inverse_coverage"] = (
                training_candidate_diagnostics.get("natural_inverse_hits", 0.0)
                / inverse_queries
            )
        diffusion_contract["training_candidate_diagnostics"] = dict(
            training_candidate_diagnostics
        )
        if args.disable_time_conditioning:
            if args.denoiser != "listwise":
                raise ValueError("time-conditioning ablation requires listwise denoiser")
            time_index = mixed_edit_time_feature_index(len(source.family_specs))
            ablated_features = np.asarray(dataset.candidate_features).copy()
            ablated_features[..., time_index] = 0.0
            dataset = replace(dataset, candidate_features=ablated_features)
            if initial_state_dict is not None:
                initial_state_dict = {
                    key: value.clone() for key, value in initial_state_dict.items()
                }
                first_weight = initial_state_dict.get("network.0.weight")
                if first_weight is None:
                    raise KeyError("cannot locate first scorer layer for time ablation")
                first_weight[:, time_index] = 0.0
        dataset_path = args.output_dir / "mixed_edit_dataset.npz"
        np.savez_compressed(
            dataset_path,
            candidate_features=dataset.candidate_features,
            candidate_improvements=dataset.candidate_improvements,
            targets=dataset.targets,
            trajectory_ids=dataset.trajectory_ids,
            feature_profile=np.asarray(args.feature_profile),
            kernel=np.asarray(args.kernel),
            role_corruption=np.asarray(json.dumps(role_corruption_metadata)),
            type_orbit_augmentation=np.asarray(args.type_orbit_augmentation),
            forward_mode=np.asarray(args.forward_mode),
            anonymous_role_condition=np.asarray(args.anonymous_role_condition),
            fixed_count_schedule=np.asarray(args.fixed_count_schedule),
            forward_family_power=np.asarray(float(args.forward_family_power)),
            forward_family_pseudocount=np.asarray(
                float(args.forward_family_pseudocount)
            ),
            training_candidate_policy=np.asarray(args.training_candidate_policy),
            clean_noop_examples_per_trajectory=np.asarray(
                args.clean_noop_examples_per_trajectory
            ),
        )
        if args.dataset_cache is not None and not args.dataset_cache.exists():
            args.dataset_cache.parent.mkdir(parents=True, exist_ok=True)
            args.dataset_cache.write_bytes(dataset_path.read_bytes())
        if args.denoiser == "factorized":
            model, train_report = train_factorized_edit_denoiser(
                dataset,
                type_count=len(source.type_names),
                family_count=len(source.family_specs),
                hidden_dim=args.hidden_dim,
                layers=args.layers,
                dropout=args.dropout,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                balanced_action_aux_weight=args.balanced_action_aux_weight,
                device=args.device,
                seed=args.seed,
            )
            checkpoint = args.output_dir / "factorized_edit_diffusion.pt"
            save_factorized_edit_checkpoint(
                checkpoint,
                model,
                prior,
                hidden_dim=args.hidden_dim,
                layers=args.layers,
                dropout=args.dropout,
                actions=KERNELS[args.kernel],
                report=train_report,
            )
            fixed_t = (
                evaluate_factorized_fixed_t(
                    source,
                    model,
                    time_fractions=(0.1, 0.5, 0.9),
                    trials_per_time=args.fixed_t_trials,
                    forward_steps=args.forward_steps,
                    candidates_per_example=args.train_candidates,
                    seed=args.seed + 991,
                    actions=KERNELS[args.kernel],
                    device=args.device,
                )
                if args.fixed_t_trials > 0
                else {}
            )
            serialized_train_report = dict(train_report)
        else:
            model, train_report = train_switch_denoiser(
                dataset,
                hidden_dim=args.hidden_dim,
                layers=args.layers,
                dropout=args.dropout,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                training_objective=args.training_objective,
                inverse_loss_weight=args.inverse_loss_weight,
                utility_loss_weight=args.utility_loss_weight,
                utility_temperature=args.utility_temperature,
                utility_schedule_power=args.utility_schedule_power,
                time_feature_index=mixed_edit_time_feature_index(
                    len(source.family_specs)
                ),
                initial_state_dict=initial_state_dict,
                device=args.device,
                seed=args.seed,
            )
            if args.disable_time_conditioning:
                time_index = mixed_edit_time_feature_index(len(source.family_specs))
                first_weight = model.state_dict()["network.0.weight"]
                if not torch.count_nonzero(first_weight[:, time_index]).eq(0):
                    raise AssertionError("time-ablation scorer retained nonzero time weights")
            checkpoint = args.output_dir / "mixed_edit_diffusion.pt"
            save_mixed_edit_checkpoint(
                checkpoint,
                model,
                prior,
                hidden_dim=args.hidden_dim,
                layers=args.layers,
                dropout=args.dropout,
                report=train_report,
                actions=KERNELS[args.kernel],
                feature_profile=args.feature_profile,
                training_noise=role_corruption_metadata,
                diffusion_contract=diffusion_contract,
            )
            if args.fixed_t_trials <= 0:
                fixed_t = {}
            elif args.forward_mode == "closed_form":
                fixed_t = evaluate_closed_form_policy(
                    source,
                    model,
                    time_fractions=(0.1, 0.5, 0.9, 1.0),
                    trials_per_time=args.fixed_t_trials,
                    candidates_per_example=args.train_candidates,
                    seed=args.seed + 991,
                    device=args.device,
                    actions=KERNELS[args.kernel],
                    feature_profile=args.feature_profile,
                    use_role_condition=args.anonymous_role_condition,
                )
            else:
                fixed_t = evaluate_fixed_t_denoising(
                    source,
                    model,
                    time_fractions=(0.1, 0.5, 0.9),
                    trials_per_time=args.fixed_t_trials,
                    forward_steps=args.forward_steps,
                    candidates_per_example=args.train_candidates,
                    seed=args.seed + 991,
                    device=args.device,
                    actions=KERNELS[args.kernel],
                )
            serialized_train_report = train_report.to_json_dict()
        write_json(
            args.output_dir / "train_report.json",
            {
                "format": "dihug-mixed-edit-train-v1",
                "denoiser": args.denoiser,
                "training_objective": args.training_objective,
                "feature_profile": args.feature_profile,
                "train": serialized_train_report,
                "fixed_t": fixed_t,
                "prior": prior.to_json_dict(),
                "diffusion_contract": diffusion_contract,
                "training_noise": role_corruption_metadata,
            },
        )
        print(
            "[MIXED-EDIT-TRAIN] "
            f"denoiser={args.denoiser} "
            f"examples={len(dataset.targets)} "
            f"top1={serialized_train_report.get('validation_candidate_top1', serialized_train_report.get('validation_top1', float('nan'))):.4f} "
            f"fixed_t={json.dumps(fixed_t, sort_keys=True)}",
            flush=True,
        )
        return

    if args.checkpoint is None:
        raise ValueError("--checkpoint is required in sample mode")
    if args.denoiser == "factorized":
        model, prior, checkpoint_metadata = load_factorized_edit_checkpoint(
            args.checkpoint, device=args.device
        )
    else:
        model, prior, checkpoint_metadata = load_mixed_edit_checkpoint(
            args.checkpoint, device=args.device
        )
    checkpoint_actions = tuple(checkpoint_metadata["actions"])
    if checkpoint_actions != tuple(KERNELS[args.kernel]):
        raise ValueError(
            f"checkpoint kernel {checkpoint_actions} does not match --kernel "
            f"{KERNELS[args.kernel]}"
        )
    sampler = (
        sample_factorized_edit_diffusion
        if args.denoiser == "factorized"
        else sample_mixed_edit_diffusion
    )
    sample_kwargs = {
        "steps": args.sample_steps,
        "candidates_per_step": args.sample_candidates,
        "seed": args.seed,
        "device": args.device,
        "temperature": args.sample_temperature,
        "metrics_every": args.metrics_every,
        "actions": KERNELS[args.kernel],
    }
    if args.denoiser == "factorized":
        sample_kwargs["decode_mode"] = args.factorized_decode
    else:
        feature_profile = checkpoint_metadata["feature_profile"]
        sample_kwargs["feature_profile"] = feature_profile
        if feature_profile == "typed_closure_role_v3":
            role_payload = checkpoint_metadata["training_noise"].get(
                "anonymous_role_prior"
            )
            if role_payload is None:
                raise RuntimeError("role-conditioned checkpoint lacks its role prior")
            role_prior = ExchangeableJointRolePrior.from_json_dict(role_payload)
            role_graph = role_prior.sample(args.seed + 80_000_113)
            sample_kwargs["role_condition"] = role_graph.all_family_degrees()
    result = sampler(prior, model, **sample_kwargs)
    payload = {
        "format": "dihug-mixed-edit-sampling-v1",
        "denoiser": args.denoiser,
        "seed": args.seed,
        "initial": result.trace[0],
        "final": graph_metrics(result.graph),
        "trace": list(result.trace),
        "action_counts": dict(result.action_counts),
        "proposed": result.proposed,
        "applied": result.applied,
        "prior": prior.to_json_dict(),
        "train_report": checkpoint_metadata["train_report"],
        "sampling_config": {
            "kernel": args.kernel,
            "steps": args.sample_steps,
            "candidates": args.sample_candidates,
            "temperature": args.sample_temperature,
            "factorized_decode": (
                args.factorized_decode if args.denoiser == "factorized" else None
            ),
        },
    }
    write_json(args.output_dir / "sampling_report.json", payload)
    write_json(args.output_dir / "generated_graph.json", result.graph.to_json_dict())
    print(
        "[MIXED-EDIT-SAMPLE] "
        f"tri={payload['initial']['triangles']:.0f}->{payload['final']['triangles']:.0f} "
        f"clustering={payload['initial']['clustering']:.6f}->"
        f"{payload['final']['clustering']:.6f} "
        f"edges={payload['initial']['edges']:.0f}->{payload['final']['edges']:.0f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
