"""Essential typed adapter over SparseDiff's original sparse Transformer.

Types are observed, never sampled. Relation families condition input and output
support; no type Q/K modulation, HGT matrices, spectral/motif features or FiLM.
Queries participate in message passing, as in SparseDiff. Hence this is a
query-conditioned reference implementation, not a query-free production encoder.
"""
import numpy as np
import itertools
import torch
from torch import nn
from hsgd.holders import PlaceHolder
from hsgd.vendor.conv_transformer_model import GraphTransformerConv


class TypedSparseDenoiser(nn.Module):
    def __init__(self, n_types, n_families, layers=3, width=128, edge_width=64, heads=4, dropout=0.0,
                 family_pairs=None, prior=None, global_width=32,
                 ff_node=None, ff_edge=None, ff_global=None, cn_hidden=0):
        super().__init__()
        if width % heads or edge_width % heads:
            raise ValueError("hidden dimensions must divide heads")
        self.n_types, self.n_families = n_types, n_families
        self.family_pairs = tuple(family_pairs or itertools.combinations_with_replacement(range(n_types), 2))
        if len(self.family_pairs) != n_families or len(set(self.family_pairs)) != n_families:
            raise ValueError("one relation state per unique type pair required")
        if any(a > b or a < 0 or b >= n_types for a, b in self.family_pairs):
            raise ValueError("invalid schema")
        prior = np.full(n_families, .5) if prior is None else np.asarray(prior, dtype=float)
        if prior.shape != (n_families,) or not np.isfinite(prior).all() or ((prior < 0) | (prior > 1)).any():
            raise ValueError("invalid exact family prior")
        self.register_buffer("prior", torch.as_tensor(prior, dtype=torch.float64))
        dims = PlaceHolder(X=n_types, E=2 * n_families + 1, y=1, charge=0)
        self.backbone = GraphTransformerConv(
            n_layers=layers, input_dims=dims,
            hidden_dims=dict(dx=width, de=edge_width, dy=global_width, n_head=heads,
                             dim_ffX=2 * width if ff_node is None else ff_node,
                             dim_ffE=2 * edge_width if ff_edge is None else ff_edge),
            output_dims=PlaceHolder(X=n_types, E=n_families + 1, y=0, charge=0),
            dropout=dropout, sn_hidden_dim=0)
        # Source-pinned GraphTransformerConv does not pass dim_ffy through to
        # XEyTransformerLayer (default 2048). Make requested dimensions explicit
        # in this adapter; vendor bytes remain unchanged. Default preserves v1.
        if ff_global is not None:
            for layer in self.backbone.tf_layers:
                if layer.last_layer:
                    layer.lin_y1 = nn.Linear(global_width, ff_global)
                    layer.lin_y2 = nn.Linear(ff_global, global_width)
        if cn_hidden not in (0,32):
            raise ValueError("frozen experiment supports baseline0 or CN32 only")
        self.common_neighbor = None
        if cn_hidden:
            from hsgd.common_neighbor import CommonNeighborReadout
            self.common_neighbor = CommonNeighborReadout(width,n_types,n_families,cn_hidden)

    def forward(self, types, existing_edges, existing_families, query_edges, query_families, time):
        """Single graph; existing/query arrays have each physical edge once.

        Build sorted directed computational union; query edge state is 0 if absent.
        Append family one-hot to each state vector (including no-edge queries) so
        family information does not disappear in sparse zero edge attributes.
        Backbone output then softmaxes ONLY none and this slot's family relation.
        """
        if self.common_neighbor is not None:
            return self.forward_upstream(types,existing_edges,existing_families,query_edges,query_families,time)
        device = next(self.parameters()).device
        ty = torch.as_tensor(types, dtype=torch.long, device=device)
        n = len(ty)
        if n < 2 or (ty < 0).any() or (ty >= self.n_types).any():
            raise ValueError("invalid fixed type inventory")
        old = np.asarray(existing_edges, dtype=np.int64).reshape(2, -1)
        qe = np.asarray(query_edges, dtype=np.int64).reshape(2, -1)
        of, qf = np.asarray(existing_families, dtype=np.int64), np.asarray(query_families, dtype=np.int64)
        if len(of) != old.shape[1] or len(qf) != qe.shape[1] or ((of < 0) | (of >= self.n_families)).any() or ((qf < 0) | (qf >= self.n_families)).any():
            raise ValueError("invalid family metadata")
        if ((old < 0) | (old >= n)).any() or ((qe < 0) | (qe >= n)).any() or (old[0] >= old[1]).any() or (qe[0] >= qe[1]).any():
            raise ValueError("canonical upper-triangle edges required")
        types_np = ty.detach().cpu().numpy()
        for edges, family in [(old, of), (qe, qf)]:
            observed = np.sort(types_np[edges], axis=0).T
            expected = np.asarray(self.family_pairs, dtype=np.int64)[family]
            if not np.array_equal(observed, expected):
                raise ValueError("relation family incompatible with endpoint types")
        if len(of) and (self.prior[torch.as_tensor(of, device=device)] == 0).any():
            raise ValueError("existing edge outside exact prior support")
        state = {}
        for (a, b), f in zip(old.T, of):
            if (int(a), int(b)) in state:
                raise ValueError("duplicate existing physical edge")
            state[int(a), int(b)] = (int(f), int(f) + 1)
        queries, query_set = [], set()
        for (a, b), f in zip(qe.T, qf):
            key = int(a), int(b)
            if key in query_set:
                raise ValueError("duplicate query")
            queries.append(key)
            query_set.add(key)
            if key in state and state[key][0] != f:
                raise ValueError("family mismatch for query")
            state.setdefault(key, (int(f), 0))
        directed = sorted([(a, b, f, c) for (u, v), (f, c) in state.items() for a, b in [(u, v), (v, u)]])
        if not directed:
            return torch.empty(0, device=device)
        arr = torch.tensor(directed, dtype=torch.long, device=device)
        edge_index = arr[:, :2].T
        attrs = torch.cat([
            torch.nn.functional.one_hot(arr[:, 3], self.n_families + 1).float(),
            torch.nn.functional.one_hot(arr[:, 2], self.n_families).float(),
        ], dim=-1)
        x = torch.nn.functional.one_hot(ty, self.n_types).float()
        # Time is the sole global feature, matched in training and inference.
        pred = self.backbone(x, attrs, edge_index, torch.tensor([[float(time)]], device=device),
                             torch.zeros(n, dtype=torch.long, device=device))
        lookup = {(a, b): k for k, (a, b, _, _) in enumerate(directed)}
        idx = torch.tensor([lookup[p] for p in queries], dtype=torch.long, device=device)
        family = torch.as_tensor(qf + 1, device=device)
        logits = torch.stack([pred.edge_attr[idx, 0], pred.edge_attr[idx, family]], dim=-1)
        p0 = logits.softmax(-1)[:, 1]
        pi = self.prior[torch.as_tensor(qf, device=device)]
        return torch.where(pi == 0, torch.zeros_like(p0), torch.where(pi == 1, torch.ones_like(p0), p0))

    def forward_upstream(self, types, existing_edges, existing_families,
                         query_edges, query_families, time):
        """Explicit equivalent path; original forward retained as v4 baseline."""
        from hsgd.computational import FrozenTypedContext
        token = ("one-forward", 0)
        context = FrozenTypedContext(self, types, existing_edges, existing_families, token)
        try:
            return context.predict(query_edges, query_families, time, token)
        finally:
            context.close()
