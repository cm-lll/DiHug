import torch

from sparse_diffusion.structure_energy.subtype_decoder import (
    ConditionalSubtypeHead,
    SubtypeSchema,
    load_subtype_schema,
)


def test_current_imdb_subtypes_are_detected_as_single_class():
    schema = load_subtype_schema("data/IMDB_movie_subgraphs/subgraph_000")
    assert not schema.prediction_enabled
    assert all(len(values) == 1 for values in schema.node_values.values())
    assert all(len(values) == 1 for values in schema.edge_values.values())


def test_conditional_subtype_head_only_builds_nontrivial_heads():
    schema = SubtypeSchema(
        node_values={"A": (0, 1), "B": (0,)},
        edge_values={"A-B": (1, 2, 3), "A-A": (1,)},
    )
    head = ConditionalSubtypeHead(8, ("A", "B"), ("A-B", "A-A"), schema)
    hidden = torch.randn(4, 8)
    assert head.node_logits("A", hidden).shape == (4, 2)
    assert head.node_logits("B", hidden) is None
    assert head.edge_logits("A-B", hidden, hidden).shape == (4, 3)
    assert head.edge_logits("A-A", hidden, hidden) is None
