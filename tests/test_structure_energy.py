from collections import Counter
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
    inverse_switch,
    randomize_role_realization,
    sample_switch_diffusion,
    sample_valid_switch,
    train_switch_denoiser,
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
