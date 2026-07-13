from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from sparse_diffusion.structure_energy.checkpoint import StructureEnergyCheckpoint
from sparse_diffusion.structure_energy.features import (
    FeatureSelection,
    GraphletFeatureExtractor,
)
from sparse_diffusion.structure_energy.fit import fit_pairwise_nce
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
