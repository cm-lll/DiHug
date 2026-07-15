from collections import Counter
import math
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from sparse_diffusion.structure_energy.checkpoint import StructureEnergyCheckpoint
from sparse_diffusion.structure_energy.diagnostics import (
    half_drift_z_score,
    r_hat,
    relative_half_drift,
    split_r_hat,
)
from sparse_diffusion.structure_energy.features import (
    FeatureSelection,
    GraphletFeatureExtractor,
    select_features,
)
from sparse_diffusion.structure_energy.fit import (
    _continuation_schedule,
    _feature_covariance,
    _natural_gradient_direction,
    _trust_region_update,
    fit_moment_matching_energy,
    fit_pairwise_nce,
)
from sparse_diffusion.structure_energy.graph import FamilySpec, TypedGraph
from sparse_diffusion.structure_energy.roles import (
    permuted_witness,
    realize_role_plan,
    sample_role_plan,
    validate_role_realization,
)
from sparse_diffusion.structure_energy.relocation_diffusion import (
    EdgeCountCondition,
    build_relocation_diffusion_dataset,
    load_relocation_checkpoint,
    sample_relocation_diffusion,
    sample_valid_relocation,
    save_relocation_checkpoint,
)
from sparse_diffusion.structure_energy.mixed_edit_diffusion import (
    EditProposal,
    MarginalGraphPrior,
    MixedKernelSchedule,
    apply_edit,
    build_mixed_edit_diffusion_dataset,
    edge_edit_distance,
    inverse_edit,
    marginal_log_probability,
    mixed_transition_probability,
    run_edit_distance_oracle,
    sample_mixed_forward_transition,
    sample_valid_edit,
)
from sparse_diffusion.structure_energy.factorized_edit_denoiser import (
    FactorizedEditDenoiser,
    build_factorized_edit_dataset,
    load_factorized_edit_checkpoint,
    sample_factorized_edit_candidates,
    sample_factorized_edit_diffusion,
    save_factorized_edit_checkpoint,
    train_factorized_edit_denoiser,
)
from sparse_diffusion.structure_energy.switch_chain import (
    LinearGraphletEnergy,
    graph_metrics,
    propose_curveball,
    propose_switch,
    run_parallel_tempering,
    run_switch_chain,
)
from sparse_diffusion.structure_energy.switch_diffusion import (
    SwitchDenoiserMLP,
    apply_switch,
    build_switch_diffusion_dataset,
    fit_learned_switch_condition,
    inverse_switch,
    load_switch_checkpoint,
    randomize_role_realization,
    proposal_delta,
    sample_switch_diffusion,
    sample_switch_candidates,
    sample_valid_switch,
    save_switch_checkpoint,
    switch_diffusion_time_fraction,
    train_switch_denoiser,
)
from sparse_diffusion.structure_energy.switch_accelerator import (
    BatchedClosedTriangleKernel,
)


def _toy_graph() -> TypedGraph:
    node_types = np.asarray([0] * 6 + [1] * 4, dtype=np.int64)
    specs = {
        "AA": FamilySpec("AA", 0, 0, tuple(range(6)), tuple(range(6))),
        "AB": FamilySpec("AB", 0, 1, tuple(range(6)), tuple(range(6, 10))),
    }
    return TypedGraph(
        node_types,
        ("A", "B"),
        specs,
        {
            "AA": [(0, 1), (2, 3), (4, 5)],
            "AB": [
                (0, 6),
                (0, 7),
                (1, 6),
                (1, 8),
                (2, 7),
                (2, 9),
                (3, 8),
                (3, 9),
                (4, 6),
                (4, 9),
                (5, 7),
                (5, 8),
            ],
        },
    )


def test_anonymous_role_plan_has_constructive_feasibility_witness():
    source = _toy_graph()
    plan = sample_role_plan(source, seed=17)

    realized = realize_role_plan(source, plan)
    witness = permuted_witness(source, plan)

    validate_role_realization(realized, plan)
    validate_role_realization(witness, plan)
    assert realized.num_edges == source.num_edges
    assert witness.num_edges == source.num_edges
    for family in source.family_specs:
        np.testing.assert_array_equal(
            realized.family_degrees(family), plan.family_degrees[family]
        )


def test_incremental_graphlet_delta_matches_full_recomputation():
    graph = _toy_graph()
    extractor = GraphletFeatureExtractor(graph)
    before = extractor.extract(graph)
    degrees_before = graph.all_family_degrees()
    rng = np.random.default_rng(23)

    proposal = None
    for _ in range(100):
        proposal = propose_switch(graph, "AB", rng)
        if proposal is not None:
            break
    assert proposal is not None
    delta = extractor.switch_delta(
        graph,
        proposal.family,
        proposal.indices,
        proposal.old_edges,
        proposal.new_edges,
    )

    incremental = Counter(before)
    extractor.apply_delta(incremental, delta)
    assert incremental == extractor.extract(graph)
    for family, expected in degrees_before.items():
        np.testing.assert_array_equal(graph.family_degrees(family), expected)
    graph.validate()


def test_closed_triangle_delta_fast_path_matches_full_recomputation():
    source = _toy_graph()
    graph = source.clone()
    full_extractor = GraphletFeatureExtractor(graph)
    all_values = full_extractor.extract(graph)
    tracked = tuple(
        key
        for key in all_values
        if key.startswith("triangle/") or key.startswith("root_triangle/")
    )
    extractor = GraphletFeatureExtractor(graph, tracked_keys=tracked)
    before = extractor.extract(graph)
    rng = np.random.default_rng(401)
    proposal = None
    for _ in range(100):
        proposal = sample_valid_switch(graph, rng)
        if proposal is not None:
            break
    assert proposal is not None

    delta = extractor.switch_delta(
        graph,
        proposal.family,
        proposal.indices,
        proposal.old_edges,
        proposal.new_edges,
    )
    incremental = Counter(before)
    extractor.apply_delta(incremental, delta)

    assert incremental == extractor.extract(graph)
    graph.validate()


def test_batched_triangle_kernel_matches_exact_python_deltas():
    graph = _toy_graph()
    all_values = GraphletFeatureExtractor(graph).extract(graph)
    keys = tuple(
        sorted(
            key
            for key in all_values
            if key.startswith(
                ("triangle/", "root_triangle/", "root_triangle_total/")
            )
        )
    )
    selection = FeatureSelection(
        keys=keys,
        null_mean=np.zeros(len(keys), dtype=np.float64),
        scale=np.asarray(
            [1.0 + index / 10.0 for index in range(len(keys))],
            dtype=np.float64,
        ),
    )
    extractor = GraphletFeatureExtractor(graph, tracked_keys=selection.keys)
    kernel = BatchedClosedTriangleKernel(graph, selection, "cpu")
    rng = np.random.default_rng(402)
    candidates, _ = sample_switch_candidates(graph, rng, 8)

    expected = np.stack(
        [
            selection.delta_vector(proposal_delta(graph, extractor, proposal))
            for proposal in candidates
        ]
    )
    actual = kernel.delta_matrix(candidates).cpu().numpy()
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1.0e-12)

    apply_switch(graph, candidates[0])
    kernel.commit(candidates[0])
    extractor = GraphletFeatureExtractor(graph, tracked_keys=selection.keys)
    candidates, _ = sample_switch_candidates(graph, rng, 8)
    expected = np.stack(
        [
            selection.delta_vector(proposal_delta(graph, extractor, proposal))
            for proposal in candidates
        ]
    )
    actual = kernel.delta_matrix(candidates).cpu().numpy()
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1.0e-12)


def test_hierarchical_feature_selection_reserves_global_closure_groups():
    real = {
        "triangle/a": 10,
        "triangle/b": 8,
        "root_triangle_total/type=A/degree=2": 18,
        "root_triangle/detail/high_z": 100,
        "root_triangle/detail/next": 90,
    }
    null_samples = [
        {
            "triangle/a": 2 + offset,
            "triangle/b": 2,
            "root_triangle_total/type=A/degree=2": 4 + offset,
            "root_triangle/detail/high_z": 0,
            "root_triangle/detail/next": 0,
        }
        for offset in (0, 1, 0, 1)
    ]
    selection = select_features(
        real,
        null_samples,
        max_features=4,
        reserve_prefix_counts={
            "triangle/": -1,
            "root_triangle_total/": -1,
        },
    )

    assert "triangle/a" in selection.keys
    assert "triangle/b" in selection.keys
    assert "root_triangle_total/type=A/degree=2" in selection.keys
    assert len(selection.keys) == 4


def test_curveball_trade_is_exact_and_preserves_bipartite_degrees():
    graph = _toy_graph()
    extractor = GraphletFeatureExtractor(graph)
    before = extractor.extract(graph)
    degrees_before = graph.all_family_degrees()
    rng = np.random.default_rng(101)
    proposal = None
    for _ in range(200):
        proposal = propose_curveball(graph, "AB", rng)
        if proposal is not None:
            break
    assert proposal is not None

    delta = extractor.switch_delta(
        graph,
        proposal.family,
        proposal.indices,
        proposal.old_edges,
        proposal.new_edges,
    )
    incremental = Counter(before)
    extractor.apply_delta(incremental, delta)

    assert incremental == extractor.extract(graph)
    for family, expected in degrees_before.items():
        np.testing.assert_array_equal(graph.family_degrees(family), expected)
    graph.validate()


def test_uniform_switch_chain_preserves_every_family_degree():
    source = _toy_graph()
    plan = sample_role_plan(source, seed=5)
    initial = realize_role_plan(source, plan)

    result = run_switch_chain(
        initial,
        GraphletFeatureExtractor(initial),
        steps=500,
        burn_in=100,
        thin=100,
        seed=9,
        verify_every=25,
    )

    validate_role_realization(result.graph, plan)
    assert result.valid > 0
    assert result.accepted == result.valid
    assert all(sample.metrics["edges"] == source.num_edges for sample in result.samples)


def test_switch_diffusion_forward_action_has_exact_legal_inverse():
    source = _toy_graph()
    plan = sample_role_plan(source, seed=301)
    graph = permuted_witness(source, plan)
    before = {
        family: tuple(edges) for family, edges in graph.family_edges.items()
    }
    rng = np.random.default_rng(302)
    proposal = sample_valid_switch(graph, rng)

    assert proposal is not None
    apply_switch(graph, proposal)
    apply_switch(graph, inverse_switch(proposal))

    assert before == {
        family: tuple(edges) for family, edges in graph.family_edges.items()
    }
    validate_role_realization(graph, plan)


def test_edge_count_condition_contains_no_role_or_motif_targets():
    source = _toy_graph()
    condition = EdgeCountCondition.from_training_graph(source)
    payload = condition.to_json_dict()

    assert payload["family_edge_counts"] == {"AA": 3, "AB": 12}
    assert "role_rows_by_type" not in payload
    assert "selection" not in payload
    assert "target_vector" not in payload
    initial = condition.sample_uniform(seed=901)
    assert initial.num_edges == source.num_edges
    for family in source.family_specs:
        assert len(initial.family_edges[family]) == len(source.family_edges[family])


def test_relocation_has_exact_inverse_and_can_change_degrees():
    graph = _toy_graph()
    before_edges = {
        family: tuple(edges) for family, edges in graph.family_edges.items()
    }
    before_degree = graph.degrees().copy()
    rng = np.random.default_rng(902)
    proposal = sample_valid_relocation(graph, rng)

    assert proposal is not None
    apply_switch(graph, proposal)
    assert not np.array_equal(graph.degrees(), before_degree)
    apply_switch(graph, inverse_switch(proposal))
    assert before_edges == {
        family: tuple(edges) for family, edges in graph.family_edges.items()
    }
    np.testing.assert_array_equal(graph.degrees(), before_degree)
    graph.validate()


def test_relocation_diffusion_tiny_end_to_end_preserves_only_family_counts():
    source = _toy_graph()
    dataset = build_relocation_diffusion_dataset(
        source,
        trajectories=3,
        forward_steps=20,
        record_every=5,
        candidates_per_example=4,
        seed=903,
    )
    assert dataset.candidate_features.shape[:2] == (12, 4)
    assert np.isfinite(dataset.candidate_features).all()
    model, report = train_switch_denoiser(
        dataset,
        hidden_dim=16,
        layers=1,
        dropout=0.0,
        epochs=2,
        batch_size=4,
        learning_rate=1.0e-2,
        training_objective="inverse_ce",
        inverse_loss_weight=1.0,
        utility_loss_weight=0.0,
        device="cpu",
        seed=904,
    )
    condition = EdgeCountCondition.from_training_graph(source)
    result = sample_relocation_diffusion(
        condition,
        model,
        steps=10,
        candidates_per_step=4,
        seed=905,
        device="cpu",
        temperature=0.0,
        metrics_every=5,
    )
    assert result.applied == 10
    for family in source.family_specs:
        assert len(result.graph.family_edges[family]) == len(source.family_edges[family])
    result.graph.validate()

    with TemporaryDirectory() as directory:
        checkpoint = Path(directory) / "relocation.pt"
        save_relocation_checkpoint(
            checkpoint,
            model,
            condition,
            hidden_dim=16,
            layers=1,
            dropout=0.0,
            report=report,
        )
        loaded_model, loaded_condition, _ = load_relocation_checkpoint(checkpoint)
        assert isinstance(loaded_model, SwitchDenoiserMLP)
        assert loaded_condition.family_edge_counts == condition.family_edge_counts


def test_marginal_prior_samples_edge_counts_instead_of_copying_them():
    source = _toy_graph()
    prior = MarginalGraphPrior.from_training_graph(source)
    payload = prior.to_json_dict()

    assert "family_edge_counts" not in payload
    assert payload["inference_contract"]["exact_family_edge_counts"] is False
    assert payload["inference_contract"]["relation_degree_rows"] is False
    counts = {
        tuple(len(prior.sample(seed).family_edges[name]) for name in prior.family_names)
        for seed in range(16)
    }
    assert len(counts) > 1


def test_mixed_kernel_birth_death_satisfies_detailed_balance():
    graph = _toy_graph()
    prior = MarginalGraphPrior.from_training_graph(graph)
    schedule = MixedKernelSchedule()
    family = "AA"
    missing = next(
        (u, v)
        for u in graph.family_specs[family].src_nodes
        for v in graph.family_specs[family].src_nodes
        if u < v and not graph.has_edge(u, v)
    )
    birth = EditProposal("birth", family, (), (missing,))
    left = math.exp(marginal_log_probability(graph, prior))
    left *= mixed_transition_probability(graph, prior, birth, 0.7, schedule)

    changed = graph.clone()
    apply_edit(changed, birth)
    death = inverse_edit(birth)
    right = math.exp(marginal_log_probability(changed, prior))
    right *= mixed_transition_probability(changed, prior, death, 0.7, schedule)
    np.testing.assert_allclose(left, right, rtol=1.0e-12, atol=0.0)


def test_every_mixed_edit_has_a_legal_exact_inverse():
    source = _toy_graph()
    prior = MarginalGraphPrior.from_training_graph(source)
    rng = np.random.default_rng(906)
    schedule = MixedKernelSchedule()
    seen = set()
    for step in range(500):
        proposal = sample_mixed_forward_transition(
            source, prior, rng, (step + 1) / 500.0, schedule
        )
        if proposal is None:
            continue
        before = {
            family: set(edges) for family, edges in source.family_edge_sets.items()
        }
        apply_edit(source, proposal)
        apply_edit(source, inverse_edit(proposal))
        assert before == {
            family: set(edges) for family, edges in source.family_edge_sets.items()
        }
        source.validate()
        seen.add(proposal.action)
    assert seen == {"noop", "birth", "death", "relocation", "switch"}


def test_edit_distance_oracle_uses_marginal_prior_and_improves_coverage():
    clean = _toy_graph()
    prior = MarginalGraphPrior.from_training_graph(clean)
    initial = prior.sample(seed=978)
    initial_distance = edge_edit_distance(initial, clean)
    result = run_edit_distance_oracle(
        prior,
        clean,
        steps=300,
        candidates_per_step=32,
        seed=907,
        actions=("birth", "death", "relocation", "switch"),
        metrics_every=100,
    )
    assert result.trace[0]["edge_edit_distance"] == initial_distance
    assert edge_edit_distance(result.graph, clean) < initial_distance
    result.graph.validate()


def test_mixed_edit_dataset_contains_known_inverse_without_target_metrics():
    source = _toy_graph()
    dataset = build_mixed_edit_diffusion_dataset(
        source,
        trajectories=3,
        forward_steps=40,
        record_every=4,
        candidates_per_example=8,
        seed=908,
    )
    assert dataset.candidate_features.ndim == 3
    assert dataset.candidate_features.shape[1] == 8
    assert np.isfinite(dataset.candidate_features).all()
    assert np.all(dataset.candidate_improvements == 0.0)
    assert np.all(dataset.targets >= 0)
    assert np.all(dataset.targets < 8)


def test_factorized_candidates_cover_positive_endpoint_group():
    graph = _toy_graph()
    rng = np.random.default_rng(909)
    positive = sample_valid_edit(
        graph, rng, actions=("relocation",), family="AB"
    )
    assert positive is not None
    candidates, target = sample_factorized_edit_candidates(
        graph,
        rng,
        count=16,
        actions=("birth", "death", "relocation", "switch"),
        positive=positive,
    )
    same_group = [
        proposal
        for proposal in candidates
        if proposal.action == positive.action and proposal.family == positive.family
    ]
    assert candidates[target] == positive
    assert len(same_group) >= 4
    assert sum(proposal.action == "noop" for proposal in candidates) == 1


def test_factorized_edit_denoiser_tiny_end_to_end_is_target_free():
    source = _toy_graph()
    dataset = build_factorized_edit_dataset(
        source,
        trajectories=3,
        forward_steps=30,
        record_every=3,
        candidates_per_example=8,
        seed=910,
    )
    model, report = train_factorized_edit_denoiser(
        dataset,
        type_count=len(source.type_names),
        family_count=len(source.family_specs),
        hidden_dim=16,
        layers=1,
        dropout=0.0,
        epochs=2,
        batch_size=4,
        learning_rate=1.0e-2,
        device="cpu",
        seed=911,
    )
    assert isinstance(model, FactorizedEditDenoiser)
    assert set(report["validation_action_recall"]) == {
        "noop",
        "birth",
        "death",
        "relocation",
        "switch",
    }
    prior = MarginalGraphPrior.from_training_graph(source)
    result = sample_factorized_edit_diffusion(
        prior,
        model,
        steps=10,
        candidates_per_step=8,
        seed=912,
        device="cpu",
        metrics_every=5,
    )
    result.graph.validate()
    with TemporaryDirectory() as directory:
        checkpoint = Path(directory) / "factorized.pt"
        save_factorized_edit_checkpoint(
            checkpoint,
            model,
            prior,
            hidden_dim=16,
            layers=1,
            dropout=0.0,
            actions=("birth", "death", "relocation", "switch"),
            report=report,
        )
        loaded, loaded_prior, metadata = load_factorized_edit_checkpoint(checkpoint)
        assert isinstance(loaded, FactorizedEditDenoiser)
        assert loaded_prior.to_json_dict() == prior.to_json_dict()
        assert "training_adjacency" not in metadata


def test_switch_diffusion_macro_schedule_has_exact_discrete_time_layers():
    assert switch_diffusion_time_fraction(1, 4, 2, 2) == 0.5
    assert switch_diffusion_time_fraction(2, 4, 2, 2) == 0.5
    assert switch_diffusion_time_fraction(3, 4, 2, 2) == 1.0
    assert switch_diffusion_time_fraction(4, 4, 2, 2) == 1.0

    source = _toy_graph()
    values = GraphletFeatureExtractor(source).extract(source)
    keys = tuple(sorted(values)[: min(8, len(values))])
    selection = FeatureSelection(
        keys=keys,
        null_mean=np.zeros(len(keys), dtype=np.float64),
        scale=np.ones(len(keys), dtype=np.float64),
    )
    dataset, _ = build_switch_diffusion_dataset(
        source,
        selection,
        trajectories=1,
        forward_steps=4,
        record_every=1,
        candidates_per_example=4,
        seed=808,
        conditioning_mode="state_only",
        diffusion_steps=2,
        switches_per_diffusion_step=2,
    )
    np.testing.assert_allclose(
        dataset.candidate_features[:, 0, -3],
        np.asarray([0.5, 0.5, 1.0, 1.0]),
    )


def test_switch_diffusion_tiny_end_to_end_preserves_role_degrees():
    source = _toy_graph()
    values = GraphletFeatureExtractor(source).extract(source)
    keys = tuple(sorted(values)[: min(8, len(values))])
    selection = FeatureSelection(
        keys=keys,
        null_mean=np.zeros(len(keys), dtype=np.float64),
        scale=np.ones(len(keys), dtype=np.float64),
    )
    dataset, target = build_switch_diffusion_dataset(
        source,
        selection,
        trajectories=3,
        forward_steps=20,
        record_every=5,
        candidates_per_example=4,
        seed=303,
    )

    assert dataset.candidate_features.shape[:2] == (12, 4)
    assert dataset.candidate_improvements.shape == (12, 4)
    assert np.isfinite(dataset.candidate_features).all()
    model, report = train_switch_denoiser(
        dataset,
        hidden_dim=16,
        layers=1,
        dropout=0.0,
        epochs=3,
        batch_size=4,
        learning_rate=1.0e-2,
        device="cpu",
        seed=304,
    )
    assert isinstance(model, SwitchDenoiserMLP)
    assert np.isfinite(report.validation_loss)

    plan = sample_role_plan(source, seed=305)
    initial = realize_role_plan(source, plan)
    initial = randomize_role_realization(initial, steps=20, seed=306)
    expected = initial.all_family_degrees()
    result = sample_switch_diffusion(
        initial,
        model,
        selection,
        target,
        steps=10,
        candidates_per_step=4,
        seed=307,
        device="cpu",
        temperature=0.0,
        metrics_every=5,
        selection_mode="model",
    )

    assert result.applied == 10
    for family, degree in expected.items():
        np.testing.assert_array_equal(result.graph.family_degrees(family), degree)
    assert result.proposed == 40
    assert result.selection_mode == "model"
    assert 0.0 <= result.oracle_agreement <= 1.0
    assert all(sample["edges"] == source.num_edges for sample in result.trace)


def test_target_free_switch_checkpoint_contains_no_clean_target():
    source = _toy_graph()
    values = GraphletFeatureExtractor(source).extract(source)
    keys = tuple(sorted(key for key in values if key.startswith("triangle/")))
    selection = FeatureSelection(
        keys=keys,
        null_mean=np.zeros(len(keys), dtype=np.float64),
        scale=np.ones(len(keys), dtype=np.float64),
    )
    structure = StructureEnergyCheckpoint.from_training_graph(
        source,
        LinearGraphletEnergy(selection, np.zeros(len(keys), dtype=np.float64)),
    )
    dataset, _ = build_switch_diffusion_dataset(
        source,
        selection,
        trajectories=2,
        forward_steps=20,
        record_every=2,
        candidates_per_example=4,
        seed=71,
        conditioning_mode="state_only",
    )
    model, report = train_switch_denoiser(
        dataset,
        hidden_dim=16,
        layers=1,
        dropout=0.0,
        epochs=1,
        batch_size=4,
        training_objective="scheduled_hybrid",
        inverse_loss_weight=1.0,
        utility_loss_weight=1.0,
        utility_temperature=0.5,
        utility_schedule_power=2.0,
        device="cpu",
        seed=72,
    )
    assert report.training_objective == "scheduled_hybrid"
    assert report.utility_schedule_power == 2.0
    assert np.isfinite(report.validation_utility_cross_entropy)
    assert 0.0 <= report.validation_model_oracle_top1 <= 1.0
    with TemporaryDirectory() as directory:
        path = Path(directory) / "target_free.pt"
        save_switch_checkpoint(
            path,
            model,
            structure,
            None,
            hidden_dim=16,
            layers=1,
            dropout=0.0,
            report=report,
            conditioning_mode="state_only",
        )
        import torch

        payload = torch.load(path, map_location="cpu", weights_only=False)
        assert "target_vector" not in payload
        assert "candidate_improvements" not in payload
        assert payload["inference_contract"] == {
            "clean_graphlet_target": False,
            "training_adjacency": False,
            "anonymous_relation_degree_rows": True,
            "node_type_counts": True,
            "family_schema": True,
            "learned_graphlet_condition": False,
        }
        _, restored, target, _, mode = load_switch_checkpoint(path, "cpu")

    assert target is None
    assert mode == "state_only"
    assert restored.selection.keys == selection.keys


def test_learned_switch_condition_recovers_training_utility_without_target_input():
    source = _toy_graph()
    values = GraphletFeatureExtractor(source).extract(source)
    keys = tuple(sorted(key for key in values if key.startswith("triangle/")))
    selection = FeatureSelection(
        keys=keys,
        null_mean=np.zeros(len(keys), dtype=np.float64),
        scale=np.ones(len(keys), dtype=np.float64),
    )
    dataset, _ = build_switch_diffusion_dataset(
        source,
        selection,
        trajectories=3,
        forward_steps=20,
        record_every=2,
        candidates_per_example=4,
        seed=73,
        conditioning_mode="state_only",
    )
    estimate, rank, rmse = fit_learned_switch_condition(
        dataset.candidate_features,
        dataset.candidate_improvements,
        len(keys),
    )
    assert estimate.shape == (len(keys),)
    assert rank > 0
    assert rmse < 1.0e-5

    model, report = train_switch_denoiser(
        dataset,
        hidden_dim=16,
        layers=1,
        dropout=0.0,
        epochs=1,
        batch_size=8,
        training_objective="utility_kl",
        utility_temperature=1.0,
        learned_condition_feature_count=len(keys),
        device="cpu",
        seed=74,
    )
    assert model.learned_target is not None
    assert report.learned_condition_rank > 0
    assert report.learned_condition_rmse < 1.0e-5


def test_learned_condition_noop_is_an_exact_self_transition():
    graph = _toy_graph()
    values = GraphletFeatureExtractor(graph).extract(graph)
    keys = tuple(sorted(key for key in values if key.startswith("triangle/")))
    selection = FeatureSelection(
        keys=keys,
        null_mean=np.zeros(len(keys), dtype=np.float64),
        scale=np.ones(len(keys), dtype=np.float64),
    )
    family_count = len(graph.family_specs)
    input_dim = 3 * len(keys) + family_count + 3
    model = SwitchDenoiserMLP(
        input_dim=input_dim,
        hidden_dim=8,
        layers=1,
        dropout=0.0,
        learned_condition_feature_count=len(keys),
    )
    model.set_learned_target(selection.vector(values))
    before = {
        family: tuple(edges) for family, edges in graph.family_edges.items()
    }
    result = sample_switch_diffusion(
        graph,
        model,
        selection,
        None,
        steps=5,
        candidates_per_step=4,
        seed=75,
        device="cpu",
        temperature=0.0,
        metrics_every=1,
        selection_mode="model",
        conditioning_mode="state_only",
        switch_kernel="python",
        learned_condition_noop=True,
        learned_condition_noop_patience=2,
    )
    assert result.applied == 0
    assert result.proposed == 8
    assert before == {
        family: tuple(edges) for family, edges in result.graph.family_edges.items()
    }


def test_parallel_tempering_preserves_roles_and_exact_exchange_kernel():
    source = _toy_graph()
    plan = sample_role_plan(source, seed=13)
    initial = realize_role_plan(source, plan)
    values = GraphletFeatureExtractor(source).extract(source)
    key = next(key for key in values if key.startswith("triangle/"))
    selection = FeatureSelection(
        keys=(key,),
        null_mean=np.asarray([0.0]),
        scale=np.asarray([1.0]),
    )
    result = run_parallel_tempering(
        initial,
        LinearGraphletEnergy(selection, np.asarray([0.0])),
        temperatures=(0.0, 0.5, 1.0),
        rounds=4,
        local_steps=50,
        initialization_steps=50,
        seed=21,
        verify_every=25,
    )

    validate_role_realization(result.graph, plan)
    assert len(result.samples) == 4
    assert result.exchange_attempted == 4
    assert result.exchange_accepted == 4


def test_pairwise_nce_recovers_graphlet_enrichment_direction():
    selection = FeatureSelection(
        keys=("triangle/AAB",),
        null_mean=np.asarray([1.0]),
        scale=np.asarray([1.0]),
    )
    null_samples = [
        {"triangle/AAB": value} for value in (0, 1, 2, 1, 0, 2, 1, 0)
    ]

    energy, report = fit_pairwise_nce(
        {"triangle/AAB": 8},
        null_samples,
        selection,
        l2=0.1,
        seed=3,
    )

    assert report.success
    assert report.heldout_pair_accuracy == 1.0
    assert energy.weights[0] > 0.0
    assert energy.value({"triangle/AAB": 8}) > energy.value(
        {"triangle/AAB": 1}
    )


def test_fixed_degree_feature_selection_drops_redundant_open_wedges():
    real = {"triangle/AAB": 8, "wedge/center=A/ends=A::B": 2}
    null = [
        {"triangle/AAB": value, "wedge/center=A/ends=A::B": 10 - value}
        for value in (0, 1, 2, 1, 0, 2, 1, 0)
    ]

    selection = select_features(real, null, max_features=2, min_support=1)

    assert selection.keys == ("triangle/AAB",)


def test_split_rhat_detects_shared_drift_hidden_from_ordinary_rhat():
    chains = [
        np.linspace(0.0, 10.0, 40) + offset
        for offset in (0.0, 0.02, -0.02, 0.01)
    ]

    assert r_hat(chains) < 1.05
    assert split_r_hat(chains) > 1.10
    assert relative_half_drift(chains, reference=10.0) > 0.40
    assert half_drift_z_score(chains) > 2.0


def test_rhat_rejects_zero_variance_chains_with_different_means():
    chains = [[value] * 8 for value in (0.0, 0.0, 1.0, 1.0)]

    assert np.isinf(r_hat(chains))
    assert np.isinf(split_r_hat(chains))


def test_moment_fit_skips_update_when_expectation_diagnostics_are_impossible():
    source = _toy_graph()
    values = GraphletFeatureExtractor(source).extract(source)
    key = next(key for key in values if key.startswith("triangle/"))
    selection = FeatureSelection(
        keys=(key,),
        null_mean=np.asarray([0.0]),
        scale=np.asarray([1.0]),
    )

    energy, report, _ = fit_moment_matching_energy(
        values,
        selection,
        [source, source],
        rounds=2,
        steps_per_round=20,
        samples_per_round=4,
        max_batches_per_round=1,
        min_samples_per_chain=8,
        seed=7,
    )

    assert report.successful_updates == 0
    assert not any(item.expectation_converged for item in report.trajectory)
    np.testing.assert_allclose(energy.weights, np.zeros(1))


def test_moment_fit_accumulates_samples_while_weights_are_frozen():
    source = _toy_graph()
    values = GraphletFeatureExtractor(source).extract(source)
    key = next(key for key in values if key.startswith("triangle/"))
    selection = FeatureSelection(
        keys=(key,),
        null_mean=np.asarray([0.0]),
        scale=np.asarray([1.0]),
    )

    _, report, _ = fit_moment_matching_energy(
        values,
        selection,
        [source, source],
        rounds=2,
        steps_per_round=20,
        samples_per_round=4,
        max_batches_per_round=1,
        min_samples_per_chain=7,
        max_retained_samples_per_chain=16,
        seed=71,
    )

    first, second = report.trajectory
    assert first.samples_per_chain < 7
    assert second.samples_carried_in == first.samples_per_chain
    assert second.samples_per_chain > first.samples_per_chain
    assert first.weight_version == second.weight_version == 0


def test_natural_gradient_update_respects_local_kl_and_ascent_direction():
    samples = np.asarray(
        [
            [-2.0, -1.8],
            [-1.0, -0.9],
            [0.0, 0.0],
            [1.0, 0.9],
            [2.0, 1.8],
            [0.2, -0.2],
        ],
        dtype=np.float64,
    )
    gradient = np.asarray([1.0, -1.0], dtype=np.float64)
    covariance, rank = _feature_covariance(samples, shrinkage=0.1)
    direction, condition = _natural_gradient_direction(
        gradient, covariance, damping=0.05, l2=0.01
    )
    update, approximate_kl = _trust_region_update(
        direction,
        step_size=1.0,
        covariance=covariance,
        max_norm=2.0,
        kl_radius=0.05,
    )

    assert rank == 2
    assert np.isfinite(condition)
    assert float(update @ gradient) > 0.0
    assert np.linalg.norm(update) <= 2.0 + 1.0e-12
    assert approximate_kl <= 0.05 + 1.0e-12


def test_continuation_schedule_is_geometric_and_reaches_real_target():
    levels = _continuation_schedule(6, 0.05)

    assert levels.shape == (6,)
    assert levels[0] == 0.05
    assert levels[-1] == 1.0
    assert np.all(np.diff(levels) > 0.0)
    np.testing.assert_allclose(levels[1:] / levels[:-1], levels[1] / levels[0])


def test_moment_continuation_scales_the_initial_kl_radius():
    source = _toy_graph()
    values = GraphletFeatureExtractor(source).extract(source)
    key = next(key for key in values if key.startswith("triangle/"))
    selection = FeatureSelection(
        keys=(key,),
        null_mean=np.asarray([0.0]),
        scale=np.asarray([1.0]),
    )

    _, report, _ = fit_moment_matching_energy(
        values,
        selection,
        [source, source, source, source],
        rounds=2,
        steps_per_round=40,
        samples_per_round=4,
        max_batches_per_round=4,
        min_samples_per_chain=8,
        min_ess_per_chain=1.0,
        rhat_threshold=10.0,
        drift_z_threshold=10.0,
        learning_rate=1.0,
        update_norm_clip=2.0,
        update_method="natural",
        kl_trust_radius=0.12,
        continuation_stages=6,
        continuation_start_fraction=0.05,
        continuation_stage_tolerance=0.01,
        seed=82,
    )

    first = report.trajectory[0]
    assert first.continuation_stage == 0
    assert first.continuation_stages == 6
    assert first.continuation_fraction == 0.05
    assert first.approximate_kl <= 0.02 + 1.0e-12
    assert report.reached_continuation_fraction == 0.05


def test_moment_fit_natural_update_reports_bounded_kl():
    source = _toy_graph()
    values = GraphletFeatureExtractor(source).extract(source)
    key = next(key for key in values if key.startswith("triangle/"))
    selection = FeatureSelection(
        keys=(key,),
        null_mean=np.asarray([0.0]),
        scale=np.asarray([1.0]),
    )

    energy, report, _ = fit_moment_matching_energy(
        values,
        selection,
        [source, source, source, source],
        rounds=2,
        steps_per_round=40,
        samples_per_round=4,
        max_batches_per_round=4,
        min_samples_per_chain=8,
        min_ess_per_chain=1.0,
        rhat_threshold=10.0,
        drift_z_threshold=10.0,
        learning_rate=1.0,
        update_norm_clip=2.0,
        update_method="natural",
        natural_gradient_damping=0.05,
        covariance_shrinkage=0.1,
        kl_trust_radius=0.01,
        seed=81,
    )

    first = report.trajectory[0]
    assert report.successful_updates == 1
    assert first.update_method == "natural"
    assert first.actual_update_norm > 0.0
    assert first.approximate_kl <= 0.01 + 1.0e-12
    assert np.isfinite(first.fisher_condition)
    assert first.covariance_rank >= 0
    assert not np.allclose(energy.weights, np.zeros(1))


def test_moment_fit_updates_only_after_frozen_expectation_passes():
    source = _toy_graph()
    values = GraphletFeatureExtractor(source).extract(source)
    key = next(key for key in values if key.startswith("triangle/"))
    selection = FeatureSelection(
        keys=(key,),
        null_mean=np.asarray([0.0]),
        scale=np.asarray([1.0]),
    )

    energy, report, _ = fit_moment_matching_energy(
        values,
        selection,
        [source, source, source, source],
        rounds=2,
        steps_per_round=40,
        samples_per_round=4,
        max_batches_per_round=4,
        min_samples_per_chain=8,
        min_ess_per_chain=1.0,
        rhat_threshold=10.0,
        drift_z_threshold=10.0,
        learning_rate=1.0,
        update_norm_clip=0.25,
        seed=8,
    )

    assert report.successful_updates == 1
    assert all(item.expectation_converged for item in report.trajectory)
    assert all(item.samples_carried_in == 0 for item in report.trajectory)
    assert all(item.actual_update_norm <= 0.25 for item in report.trajectory)
    assert np.linalg.norm(energy.weights) <= 0.25 + 1.0e-12
    assert not np.allclose(energy.weights, np.zeros(1))


def test_graph_metrics_count_typed_triangles_without_changing_degree_scope():
    metrics = graph_metrics(_toy_graph())

    assert metrics["edges"] == 15
    assert metrics["triangles"] == 2
    assert metrics["lcc"] == 10


def test_overlapping_family_domains_are_rejected_by_product_state_model():
    graph = _toy_graph()
    specs = dict(graph.family_specs)
    specs["AA_SECOND"] = FamilySpec(
        "AA_SECOND", 0, 0, tuple(range(6)), tuple(range(6))
    )
    overlapping = TypedGraph(
        graph.node_types,
        graph.type_names,
        specs,
        {
            "AA": graph.family_edges["AA"],
            "AB": graph.family_edges["AB"],
            "AA_SECOND": [],
        },
    )

    try:
        overlapping.validate_disjoint_family_domains()
    except ValueError as exc:
        assert "overlapping family candidate domains" in str(exc)
    else:
        raise AssertionError("overlapping family domains were accepted")


def test_checkpoint_sampler_needs_no_training_adjacency():
    source = _toy_graph()
    extractor = GraphletFeatureExtractor(source)
    values = extractor.extract(source)
    keys = tuple(sorted(values))
    selection = FeatureSelection(
        keys=keys,
        null_mean=np.zeros(len(keys), dtype=np.float64),
        scale=np.ones(len(keys), dtype=np.float64),
    )
    checkpoint = StructureEnergyCheckpoint.from_training_graph(
        source,
        LinearGraphletEnergy(selection, np.zeros(len(keys), dtype=np.float64)),
    )
    with TemporaryDirectory() as directory:
        path = Path(directory) / "structure.json"
        checkpoint.save(path)
        payload = path.read_text(encoding="utf-8")
        assert '"edges"' not in payload
        restored = StructureEnergyCheckpoint.load(path)
        generated = restored.realize(seed=31)

    assert generated.num_edges == source.num_edges
    assert generated.type_names == source.type_names
    generated.validate()
    round_trip = TypedGraph.from_json_dict(generated.to_json_dict())
    assert round_trip.edge_family == generated.edge_family


def test_forward_coupling_diagnostic_runs_exact_marginal_kernels():
    from sparse_diffusion.structure_energy.markov_diagnostics import (
        run_forward_coupling_diagnostic,
    )

    report = run_forward_coupling_diagnostic(
        _toy_graph(),
        steps=20,
        checkpoints=(0, 5, 10, 20),
        replicas=2,
        seed=91,
    )

    assert report["format"] == "dihug-forward-coupling-diagnostic-v1"
    assert [row["step"] for row in report["trajectory"]] == [0, 5, 10, 20]
    assert np.isfinite(report["edge_disagreement_contraction"])
    assert all(
        0.0 <= row["edge_disagreement_mean"] <= 1.0
        for row in report["trajectory"]
    )


def test_state_alias_audit_exposes_degree_preserving_summary_collisions():
    from sparse_diffusion.structure_energy.markov_diagnostics import (
        audit_state_summary_aliasing,
    )

    report = audit_state_summary_aliasing(_toy_graph(), trials=8, seed=92)

    assert report["format"] == "dihug-state-summary-aliasing-v1"
    assert report["accepted_trials"] == 8
    assert report["summary_collision_rate"] == 1.0
    assert 0.0 <= report["motif_change_rate"] <= 1.0


def test_hierarchical_factorized_decode_does_not_compare_split_leaf_modes():
    import torch

    from sparse_diffusion.structure_energy.factorized_edit_denoiser import (
        select_factorized_candidate,
    )

    # Noop is the largest individual leaf, while switch is the largest action.
    # Hierarchical MAP must choose the switch action before ranking endpoints.
    action_logits = torch.tensor([2.0, -2.0, -2.0, -2.0, 3.0])
    family_logits = torch.zeros((5, 2))
    family_logits[4] = torch.tensor([2.0, 1.0])
    endpoint_logits = torch.tensor([4.0, 1.0, 3.0, 2.0])
    action_ids = torch.tensor([0, 4, 4, 4])
    family_ids = torch.tensor([0, 0, 0, 1])
    generator = torch.Generator().manual_seed(7)

    selected = select_factorized_candidate(
        action_logits,
        family_logits,
        endpoint_logits,
        action_ids,
        family_ids,
        temperature=0.0,
        generator=generator,
    )

    assert selected == 2


def test_free_proposal_coverage_does_not_inject_the_clean_inverse():
    from sparse_diffusion.structure_energy.markov_diagnostics import (
        run_free_proposal_coverage_diagnostic,
    )

    report = run_free_proposal_coverage_diagnostic(
        _toy_graph(),
        time_fractions=(0.25, 0.75),
        trials_per_time=3,
        forward_steps=12,
        candidate_counts=(4, 8),
        seed=93,
    )

    assert report["format"] == "dihug-free-proposal-coverage-v1"
    assert report["positive_injection"] is False
    assert report["candidate_counts"] == [4, 8]
    for time_report in report["results"].values():
        assert time_report["4"]["trials"] == 3
        assert time_report["8"]["trials"] == 3
        assert (
            time_report["8"]["improving_candidate_rate"]
            >= time_report["4"]["improving_candidate_rate"]
        )


def test_hgt_edit_dataset_keeps_graph_state_and_candidate_endpoints_aligned():
    from sparse_diffusion.structure_energy.hgt_edit_denoiser import (
        build_hgt_edit_dataset,
    )

    dataset = build_hgt_edit_dataset(
        _toy_graph(),
        trajectories=2,
        forward_steps=4,
        record_every=2,
        candidates_per_example=4,
        seed=94,
    )

    assert dataset.candidate_features.shape[:2] == (4, 4)
    assert dataset.candidate_endpoints.shape == (4, 4, 8)
    assert dataset.node_features.shape[:2] == (4, 10)
    assert len(dataset.edge_indices) == 4
    assert np.all((dataset.candidate_endpoints == -1) | (dataset.candidate_endpoints < 10))


def test_hgt_free_x0_supervision_uses_non_injected_candidate_utilities():
    from sparse_diffusion.structure_energy.hgt_edit_denoiser import (
        build_hgt_edit_dataset,
    )

    dataset = build_hgt_edit_dataset(
        _toy_graph(),
        trajectories=2,
        forward_steps=8,
        record_every=2,
        candidates_per_example=8,
        seed=941,
        supervision="free_x0",
    )

    assert dataset.candidate_improvements.shape == (8, 8)
    np.testing.assert_array_equal(
        dataset.targets, dataset.candidate_improvements.argmax(axis=1)
    )
    assert np.isfinite(dataset.candidate_improvements).all()


def test_hgt_listwise_scores_are_equivariant_to_same_type_node_permutation():
    import torch

    from sparse_diffusion.structure_energy.hgt_edit_denoiser import (
        HGTListwiseEditDenoiser,
        candidate_endpoint_matrix,
        graph_edge_tensors,
        graph_node_features,
    )
    from sparse_diffusion.structure_energy.mixed_edit_diffusion import (
        mixed_edit_candidate_matrix,
        sample_edit_candidates,
    )

    graph = _toy_graph()
    rng = np.random.default_rng(95)
    candidates, _ = sample_edit_candidates(graph, rng, 4)
    permutation = np.arange(graph.num_nodes)
    permutation[[0, 1]] = permutation[[1, 0]]
    permuted_edges = {
        family: [(int(permutation[u]), int(permutation[v])) for u, v in edges]
        for family, edges in graph.family_edges.items()
    }
    permuted = TypedGraph(
        graph.node_types.copy(), graph.type_names, graph.family_specs, permuted_edges
    )
    permuted_candidates = [
        EditProposal(
            proposal.action,
            proposal.family,
            tuple((int(permutation[u]), int(permutation[v])) for u, v in proposal.old_edges),
            tuple((int(permutation[u]), int(permutation[v])) for u, v in proposal.new_edges),
        )
        for proposal in candidates
    ]
    model = HGTListwiseEditDenoiser(
        node_feature_dim=graph_node_features(graph, 0.5).shape[1],
        candidate_feature_dim=mixed_edit_candidate_matrix(graph, candidates, 0.5).shape[1],
        type_count=2,
        family_count=2,
        hidden_dim=16,
        heads=4,
        hgt_layers=1,
        scorer_layers=1,
        dropout=0.0,
        endpoint_dim=8,
    ).eval()

    def score(current, proposals):
        edges, relations = graph_edge_tensors(current)
        return model(
            torch.from_numpy(graph_node_features(current, 0.5))[None],
            torch.from_numpy(current.node_types)[None],
            torch.from_numpy(edges),
            torch.from_numpy(relations),
            torch.from_numpy(mixed_edit_candidate_matrix(current, proposals, 0.5))[None],
            torch.from_numpy(candidate_endpoint_matrix(proposals))[None],
        )

    with torch.no_grad():
        original_scores = score(graph, candidates)
        permuted_scores = score(permuted, permuted_candidates)
    torch.testing.assert_close(original_scores, permuted_scores, rtol=1e-5, atol=1e-5)
