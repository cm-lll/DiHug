#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from sparse_diffusion.structure_energy.graph import load_typed_graph
from sparse_diffusion.structure_energy.mixed_edit_diffusion import load_mixed_edit_checkpoint
from sparse_diffusion.structure_energy.rooted_context_edit import (
    build_rooted_context_dataset,
    save_rooted_context_checkpoint,
    schema_supports_triangles,
    train_rooted_context_residual,
)
from sparse_diffusion.structure_energy.role_guided_edit_diffusion import (
    ExchangeableRoleDegreePrior,
)
from sparse_diffusion.structure_energy.subtype_decoder import load_subtype_schema


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--legacy-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--trajectories", type=int, default=4)
    parser.add_argument("--examples-per-trajectory", type=int, default=24)
    parser.add_argument("--switches-per-example", type=int, default=128)
    parser.add_argument("--candidates", type=int, default=16)
    parser.add_argument("--closure-fraction", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--alignment-checkpoint", type=Path)
    parser.add_argument(
        "--state-protocol",
        choices=("legacy_switch_chain", "closed_form_marginal"),
        default="legacy_switch_chain",
    )
    parser.add_argument("--noise-levels", default="0.25,0.5,0.75,1.0")
    parser.add_argument("--targeted-role-candidate-fraction", type=float, default=0.0)
    parser.add_argument("--terminal-noop-examples-per-trajectory", type=int, default=0)
    parser.add_argument("--stop-loss-weight", type=float, default=0.25)
    parser.add_argument("--use-role-condition", action="store_true")
    parser.add_argument("--freeze-context-encoder", action="store_true")
    return parser.parse_args()


def alignment_model_shape(path: Path) -> dict[str, int]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format") != "dihug-role-conditioned-latent-alignment-v1":
        raise ValueError("unsupported latent-alignment checkpoint")
    state = payload.get("model_state", {})
    hidden_dim, role_dim = state["role_input.0.weight"].shape
    layer_ids = {
        int(key.split(".")[1]) for key in state if key.startswith("layers.")
    }
    heads = int(state["layers.0.relation_pri"].shape[1])
    return {
        "hidden_dim": int(hidden_dim),
        "role_condition_dim": int(role_dim),
        "layers": len(layer_ids),
        "heads": heads,
    }


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    source = load_typed_graph(args.dataset_dir, active_only=True)
    subtype_schema = load_subtype_schema(args.dataset_dir)
    legacy, _, metadata = load_mixed_edit_checkpoint(args.legacy_checkpoint, device="cpu")
    legacy.eval()
    profile = str(metadata.get("feature_profile", "typed_closure_compact_v2"))
    noise_levels = tuple(
        float(value) for value in str(args.noise_levels).split(",") if value.strip()
    )
    if not schema_supports_triangles(source):
        args.output_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "format": "dihug-rooted-context-residual-gate-v1",
            "dataset": str(args.dataset_dir),
            "legacy_checkpoint": str(args.legacy_checkpoint),
            "feature_profile": profile,
            "nodes": source.num_nodes,
            "edges": source.num_edges,
            "applicable": False,
            "reason": "typed schema has no triangle-capable family triple",
            "base_selected_utility": 0.0,
            "context_selected_utility": 0.0,
            "utility_gain": 0.0,
            "non_regression": 1.0,
        }
        (args.output_dir / "gate_report.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
        )
        print("[ROOTED-CONTEXT-GATE] " + json.dumps(payload, sort_keys=True), flush=True)
        return
    dataset = build_rooted_context_dataset(
        source,
        legacy,
        trajectories=args.trajectories,
        examples_per_trajectory=args.examples_per_trajectory,
        switches_per_example=args.switches_per_example,
        candidates_per_example=args.candidates,
        closure_fraction=args.closure_fraction,
        seed=args.seed,
        feature_profile=profile,
        state_protocol=args.state_protocol,
        noise_levels=noise_levels,
        targeted_role_candidate_fraction=args.targeted_role_candidate_fraction,
        terminal_noop_examples_per_trajectory=(
            args.terminal_noop_examples_per_trajectory
        ),
    )
    model_shape = {
        "hidden_dim": args.hidden_dim,
        "layers": args.layers,
        "heads": args.heads,
        "role_condition_dim": (
            len(source.family_specs) if args.use_role_condition else 0
        ),
    }
    if args.alignment_checkpoint is not None:
        model_shape = alignment_model_shape(args.alignment_checkpoint)
    model, report = train_rooted_context_residual(
        dataset,
        legacy,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=args.device,
        seed=args.seed,
        alignment_checkpoint=args.alignment_checkpoint,
        freeze_context_encoder=args.freeze_context_encoder,
        stop_loss_weight=args.stop_loss_weight,
        **model_shape,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_rooted_context_checkpoint(
        args.output_dir / "rooted_context_residual.pt",
        model,
        report,
        args.legacy_checkpoint,
        role_prior=ExchangeableRoleDegreePrior.from_training_graph(
            source
        ).to_json_dict(),
        subtype_schema=subtype_schema.to_json_dict(),
    )
    payload = {
        "format": "dihug-rooted-context-residual-gate-v1",
        "dataset": str(args.dataset_dir),
        "legacy_checkpoint": str(args.legacy_checkpoint),
        "feature_profile": profile,
        "nodes": source.num_nodes,
        "edges": source.num_edges,
        "subtype_prediction_enabled": subtype_schema.prediction_enabled,
        "subtype_schema": subtype_schema.to_json_dict(),
        "state_protocol": args.state_protocol,
        "alignment_checkpoint": (
            None
            if args.alignment_checkpoint is None
            else str(args.alignment_checkpoint)
        ),
        "noise_levels": list(noise_levels),
        "use_role_condition": bool(args.use_role_condition),
        "terminal_noop_examples_per_trajectory": (
            args.terminal_noop_examples_per_trajectory
        ),
        **report,
    }
    (args.output_dir / "gate_report.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    print("[ROOTED-CONTEXT-GATE] " + json.dumps(payload, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
    schema_supports_triangles,
