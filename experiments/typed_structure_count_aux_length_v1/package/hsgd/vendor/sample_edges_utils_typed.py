"""Typed adaptation of SparseDiff get_computational_graph, frozen d9c4b20.

Source: sparse_diffusion/diffusion/sample_edges_utils.py:67-126,
https://github.com/qym7/SparseDiff (MIT; see vendor/LICENSE).
Reuse concat, mirrored query edges, coalesce(min/max), query-mask logic.
Adapt only integer state/family metadata and original upper-query order mapping.
No homogeneous one-hot summation or merging of inconsistent relation labels.
"""
import torch
from torch_geometric.utils import coalesce


def get_computational_graph(triu_query_edge_index, query_families,
                            clean_edge_index, clean_families, n_nodes):
    # Inputs already checked: old edges are directed; query edges canonical.
    query_edge_index = torch.cat([triu_query_edge_index, triu_query_edge_index.flip(0)], dim=1)
    query_family = torch.cat([query_families, query_families])
    comp_edge_index = torch.cat([clean_edge_index, query_edge_index], dim=1)
    metadata = torch.stack([
        torch.cat([clean_families + 1, torch.zeros_like(query_family)]),
        torch.cat([clean_families, query_family])], dim=-1)
    # Exact upstream min/max reductions; state min=0 identifies queries,
    # state max preserves actual presence when a query overlaps an old edge.
    _, minima = coalesce(comp_edge_index, metadata, num_nodes=n_nodes, reduce="min")
    edge_index, maxima = coalesce(comp_edge_index, metadata, num_nodes=n_nodes, reduce="max")
    if not torch.equal(minima[:, 1], maxima[:, 1]):
        raise ValueError("coalesce cannot merge different relation families")
    query_mask = minima[:, 0] == 0
    keys = edge_index[0] * n_nodes + edge_index[1]
    query_keys = triu_query_edge_index[0] * n_nodes + triu_query_edge_index[1]
    query_index = torch.searchsorted(keys, query_keys)
    if len(query_index):
        if (query_index >= len(keys)).any() or not torch.equal(keys[query_index], query_keys):
            raise ValueError("query mapping incomplete")
        if not query_mask[query_index].all():
            raise ValueError("query mask/mapping disagreement")
    return edge_index, maxima[:, 0], maxima[:, 1], query_mask, query_index
