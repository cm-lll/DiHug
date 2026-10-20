"""Frozen-context upstream computational graph path; old model.forward retained.

No implicit cache keyed by mutable arrays. One explicit context per frozen Xt;
caller supplies an exact state-version token, closes it before committing Xt-1.
Tensor version guards prevent mutated cached metadata from being reused.
"""
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn.functional as F
from hsgd.vendor.sample_edges_utils_typed import get_computational_graph


@dataclass
class GraphInputs:
    edges: torch.Tensor
    states: torch.Tensor
    families: torch.Tensor
    query_mask: torch.Tensor
    query_index: torch.Tensor
    query_families: torch.Tensor
    attrs: torch.Tensor


def check_edges(types, pairs, edges, families, n_families, label):
    edges = np.asarray(edges, dtype=np.int64).reshape(2, -1)
    families = np.asarray(families, dtype=np.int64)
    n = len(types)
    if families.ndim != 1 or len(families) != edges.shape[1] or ((families < 0) | (families >= n_families)).any():
        raise ValueError("invalid family metadata")
    if ((edges < 0) | (edges >= n)).any() or (edges[0] >= edges[1]).any():
        raise ValueError("canonical upper-triangle edges required")
    if not np.array_equal(np.sort(types[edges], axis=0).T, pairs[families]):
        raise ValueError("relation family incompatible with endpoint types")
    keys = edges[0] * n + edges[1]
    if len(np.unique(keys)) != len(keys):
        raise ValueError("duplicate " + label)
    return edges.copy(), families.copy()


class FrozenTypedContext:
    def __init__(self, model, types, edges, families, state_version):
        if state_version is None:
            raise ValueError("explicit frozen-state version required")
        self.model, self.state_version, self.closed = model, state_version, False
        self.device = next(model.parameters()).device
        self.types = np.asarray(types, dtype=np.int64).copy()
        if self.types.ndim != 1 or len(self.types) < 2 or (self.types < 0).any() or (self.types >= model.n_types).any():
            raise ValueError("invalid fixed type inventory")
        self.pairs = np.asarray(model.family_pairs, dtype=np.int64)
        self.prior = model.prior.detach().cpu().numpy().copy()
        self.old, self.of = check_edges(self.types, self.pairs, edges, families, model.n_families, "existing physical edge")
        if (self.prior[self.of] == 0).any():
            raise ValueError("existing edge outside exact prior support")
        for arr in (self.types, self.pairs, self.prior, self.old, self.of):
            arr.flags.writeable = False
        # Prepare once per Xt. No repeated node H2D or old-edge extraction.
        ty = torch.tensor(self.types, dtype=torch.long, device=self.device)
        self.x = F.one_hot(ty, model.n_types).float()
        self.batch = torch.zeros(len(ty), dtype=torch.long, device=self.device)
        old = torch.tensor(self.old, dtype=torch.long, device=self.device)
        of = torch.tensor(self.of, dtype=torch.long, device=self.device)
        self.directed = torch.cat([old, old.flip(0)], dim=1)
        self.directed_families = torch.cat([of, of])
        # Complete frozen Xt, physical edges counted once; never query edges.
        sizes=np.bincount(self.types,minlength=model.n_types)
        capacities=np.array([sizes[a]*(sizes[a]-1)//2 if a==b else sizes[a]*sizes[b] for a,b in self.pairs],dtype=np.float64)
        counts=np.bincount(self.of,minlength=model.n_families).astype(np.float64)
        density=np.divide(counts,capacities,out=np.zeros_like(counts),where=capacities>0)
        log_count=np.divide(np.log1p(counts),np.log1p(capacities),out=np.zeros_like(counts),where=capacities>0)
        self.global_statistics=torch.tensor(np.r_[log_count,density][None],dtype=torch.float32,device=self.device)
        self._guards = [(v, v._version) for v in (self.x, self.batch, self.directed, self.directed_families, model.prior)]
        self.cn_nodes,self.cn_time,self.cn_wedges=None,None,None
        self._cn_parameters=[(p,p._version) for p in model.parameters()]
        self._time=None
        self._guards.append((self.global_statistics,self.global_statistics._version))
        if model.common_neighbor is not None:
            from hsgd.common_neighbor import Wedges
            self.cn_wedges=Wedges(self)

    def check(self, state_version):
        if self.closed or state_version != self.state_version:
            raise ValueError("stale/closed frozen context")
        if next(self.model.parameters()).device != self.device or any(v._version != version for v, version in self._guards):
            raise ValueError("cached metadata or model device changed")
        if any(p._version!=version for p,version in self._cn_parameters):
            raise ValueError("model parameters changed inside frozen CN context")

    def prepare_query(self, edges, families, state_version):
        self.check(state_version)
        return check_edges(self.types, self.pairs, edges, families, self.model.n_families, "query")

    def transfer_query(self, prepared, state_version):
        self.check(state_version)
        qe, qf = prepared
        return torch.as_tensor(qe, dtype=torch.long, device=self.device), torch.as_tensor(qf, dtype=torch.long, device=self.device)

    def assemble(self, query, state_version):
        self.check(state_version)
        qe, qf = query
        ei, states, fam, mask, idx = get_computational_graph(qe, qf, self.directed, self.directed_families, len(self.types))
        attrs = torch.cat([F.one_hot(states, self.model.n_families + 1).float(),
                           F.one_hot(fam, self.model.n_families).float()], dim=-1)
        return GraphInputs(ei, states, fam, mask, idx, qf, attrs)

    def run_network(self, graph, time, state_version):
        self.check(state_version)
        if not np.isfinite(time) or not 0<=time<=1:raise ValueError('invalid normalized diffusion time')
        if self._time is None:self._time=float(time)
        elif self._time!=float(time):raise ValueError('frozen context used at a different time')
        if not graph.edges.numel():
            logits = torch.empty((0, 2), device=self.device)
            return logits, torch.empty(0, device=self.device)
        logits = self.model.joint_predict(self,graph,time)
        p0 = logits.softmax(-1)[:, 1]
        pi = self.model.prior[graph.query_families]
        p0 = torch.where(pi == 0, torch.zeros_like(p0), torch.where(pi == 1, torch.ones_like(p0), p0))
        return logits, p0

    def predict(self, edges, families, time, state_version, details=False):
        graph = self.assemble(self.transfer_query(self.prepare_query(edges, families, state_version), state_version), state_version)
        logits, p0 = self.run_network(graph, time, state_version)
        return (graph, logits, p0) if details else p0

    def close(self):
        self.closed = True
        self.node_residual=None
        self.cn_nodes=None
        self.cn_wedges=None


def legacy_arrays(context, query_edges, query_families, state_version):
    """Exact old dictionary/sorted-list builder retained for direct inspection."""
    qe, qf = context.prepare_query(query_edges, query_families, state_version)
    state = {}
    for (a, b), f in zip(context.old.T, context.of):
        state[int(a), int(b)] = (int(f), int(f) + 1)
    queries = []
    for (a, b), f in zip(qe.T, qf):
        key = int(a), int(b)
        queries.append(key)
        if key in state and state[key][0] != f:
            raise ValueError("family mismatch for query")
        state.setdefault(key, (int(f), 0))
    directed = sorted([(a,b,f,c) for (u,v),(f,c) in state.items() for a,b in [(u,v),(v,u)]])
    arr = np.asarray(directed, dtype=np.int64).reshape(-1,4)
    lookup = {(a,b):k for k,(a,b,_,_) in enumerate(directed)}
    idx = np.asarray([lookup[key] for key in queries], dtype=np.int64)
    return arr, idx, qf


def legacy_transfer(context, arrays, include_query_mask=True):
    arr, idx, qf = [torch.as_tensor(v, device=context.device) for v in arrays]
    states, fam = arr[:,3], arr[:,2]
    attrs = torch.cat([F.one_hot(states, context.model.n_families+1).float(),
                       F.one_hot(fam, context.model.n_families).float()], dim=-1)
    mask = None
    if include_query_mask:
        keys = arr[:,0]*len(context.types)+arr[:,1]
        selected = keys[idx]
        reverse = arr[idx,1]*len(context.types)+arr[idx,0]
        mask = torch.isin(keys, torch.cat([selected,reverse]))
    return GraphInputs(arr[:,:2].T, states, fam, mask, idx, qf, attrs)
