"""Graph partitions used by sparse block-query diffusion.

Training and sampling pass a local, undirected graph. The returned blocks are
local node ids, so callers can use the same condensed pair indexing as the
uniform query sampler.
"""

from __future__ import annotations

import math
from collections import deque
from typing import List, Optional

import torch
from torch import Tensor


def triu_condensed_index(u: int, v: int, n: int) -> int:
    """Map an upper-triangular pair ``u < v`` to a condensed index."""
    if u > v:
        u, v = v, u
    i, j = u, v
    return n * (n - 1) // 2 - (n - i) * (n - i - 1) // 2 + j - i - 1


def _edge_index_to_adj_list(edge_index: Tensor, n: int) -> list[list[int]]:
    adj: list[list[int]] = [[] for _ in range(n)]
    if edge_index.numel() == 0:
        return adj
    ei = edge_index.detach().cpu()
    for a, b in zip(ei[0].tolist(), ei[1].tolist()):
        if a == b:
            continue
        if 0 <= a < n and 0 <= b < n:
            adj[a].append(b)
            adj[b].append(a)
    for i in range(n):
        adj[i] = sorted(set(adj[i]))
    return adj


def _edge_pairs_to_adj_list(edge_index_local: list[list[int]], n: int) -> list[list[int]]:
    adj: list[list[int]] = [[] for _ in range(n)]
    if not edge_index_local or len(edge_index_local) < 2:
        return adj
    for a, b in zip(edge_index_local[0], edge_index_local[1]):
        if a == b:
            continue
        if 0 <= a < n and 0 <= b < n:
            adj[a].append(b)
            adj[b].append(a)
    for i in range(n):
        adj[i] = sorted(set(adj[i]))
    return adj


def _enumerate_components(adj: list[list[int]], n: int) -> list[list[int]]:
    visited = [False] * n
    out: list[list[int]] = []
    for start in range(n):
        if visited[start]:
            continue
        stack = [start]
        visited[start] = True
        cur: list[int] = []
        while stack:
            u = stack.pop()
            cur.append(u)
            for v in adj[u]:
                if not visited[v]:
                    visited[v] = True
                    stack.append(v)
        out.append(sorted(cur))
    out.sort(key=lambda comp: comp[0])
    return out


def _induced_adj(adj: list[list[int]], comp: list[int]) -> tuple[list[list[int]], dict[int, int]]:
    idx_of = {old: j for j, old in enumerate(comp)}
    adj_c = [[] for _ in range(len(comp))]
    for old in comp:
        io = idx_of[old]
        for nb in adj[old]:
            if nb in idx_of:
                adj_c[io].append(idx_of[nb])
    for i in range(len(comp)):
        adj_c[i] = sorted(set(adj_c[i]))
    return adj_c, idx_of


def _min_degree_seed(adj: list[list[int]], n: int, generator: Optional[torch.Generator]) -> int:
    degrees = [len(adj[i]) for i in range(n)]
    mn = min(degrees)
    cand = [i for i in range(n) if degrees[i] == mn]
    if generator is None:
        j = torch.randint(len(cand), (1,)).item()
    else:
        j = torch.randint(len(cand), (1,), generator=generator).item()
    return cand[int(j)]


def _multi_source_dist(adj: list[list[int]], seeds: list[int], n: int) -> list[int]:
    inf = 10**9
    dist = [inf] * n
    q: deque[int] = deque()
    for seed in seeds:
        dist[seed] = 0
        q.append(seed)
    while q:
        u = q.popleft()
        for v in adj[u]:
            if dist[v] > dist[u] + 1:
                dist[v] = dist[u] + 1
                q.append(v)
    return dist


def _farthest_additional_seed(
    adj: list[list[int]],
    seeds: list[int],
    n: int,
    generator: Optional[torch.Generator],
) -> Optional[int]:
    dist = _multi_source_dist(adj, seeds, n)
    best = -1
    cand: list[int] = []
    seed_set = set(seeds)
    for i in range(n):
        if i in seed_set:
            continue
        d = dist[i]
        if d > best:
            best = d
            cand = [i]
        elif d == best:
            cand.append(i)
    if not cand:
        return None
    if generator is None:
        j = torch.randint(len(cand), (1,)).item()
    else:
        j = torch.randint(len(cand), (1,), generator=generator).item()
    return cand[int(j)]


def _voronoi_blocks(adj: list[list[int]], seeds: list[int], n: int) -> List[List[int]]:
    assign = [-1] * n
    q: deque[tuple[int, int]] = deque()
    for sid, seed in enumerate(seeds):
        assign[seed] = sid
        q.append((seed, sid))
    while q:
        u, sid = q.popleft()
        for v in adj[u]:
            if assign[v] < 0:
                assign[v] = sid
                q.append((v, sid))
    ks = max(1, len(seeds))
    for i in range(n):
        if assign[i] < 0:
            assign[i] = i % ks
    buckets: list[list[int]] = [[] for _ in range(len(seeds))]
    for i in range(n):
        buckets[assign[i]].append(i)
    return [sorted(block) for block in buckets if block]


def _strip_blocks(nc: int, m_target: int) -> List[List[int]]:
    order = list(range(nc))
    return [order[i : i + m_target] for i in range(0, nc, m_target)]


def _blocks_single_connected_component(
    adj_c: list[list[int]],
    nc: int,
    n_full: int,
    rho: float,
    generator: Optional[torch.Generator],
) -> List[List[int]]:
    if nc <= 0:
        return []
    if nc == 1:
        return [[0]]

    m_target = max(2, min(n_full, int(math.ceil(rho * n_full))))
    if nc <= m_target:
        return [list(range(nc))]

    total_edges = sum(len(x) for x in adj_c) // 2
    if total_edges == 0:
        return _strip_blocks(nc, m_target)

    k = max(1, min(nc, int(math.ceil(nc / m_target))))
    seeds: list[int] = [_min_degree_seed(adj_c, nc, generator)]
    while len(seeds) < k:
        next_seed = _farthest_additional_seed(adj_c, seeds, nc, generator)
        if next_seed is None or next_seed in seeds:
            break
        seeds.append(next_seed)
    return _voronoi_blocks(adj_c, seeds, nc)


def connected_blocks_from_graph(
    edge_index_local: Tensor,
    n: int,
    rho: float,
    *,
    generator: Optional[torch.Generator] = None,
) -> List[List[int]]:
    """Disjoint connected-BFS/Voronoi blocks over local node ids."""
    if n <= 0:
        return []
    rho = min(max(float(rho), 1e-8), 1.0)
    if n == 1:
        return [[0]]

    adj = _edge_index_to_adj_list(edge_index_local, n)
    return connected_blocks_from_adj_list(adj, n, rho, generator=generator)


def connected_blocks_from_adj_list(
    adj: list[list[int]],
    n: int,
    rho: float,
    *,
    generator: Optional[torch.Generator] = None,
) -> List[List[int]]:
    """Disjoint connected-BFS/Voronoi blocks from an adjacency list."""
    if n <= 0:
        return []
    rho = min(max(float(rho), 1e-8), 1.0)
    if n == 1:
        return [[0]]

    components = _enumerate_components(adj, n)
    all_blocks: List[List[int]] = []
    for comp in components:
        adj_c, _ = _induced_adj(adj, comp)
        inner = _blocks_single_connected_component(adj_c, len(comp), n, rho, generator)
        inv = [comp[j] for j in range(len(comp))]
        for block in inner:
            all_blocks.append(sorted(inv[j] for j in block))
    return all_blocks


def metis_blocks_from_graph(edge_index_local: Tensor, n: int, rho: float) -> List[List[int]]:
    """Partition local node ids with METIS, falling back only for trivial graphs."""
    if n <= 0:
        return []
    if n == 1:
        return [[0]]
    rho = min(max(float(rho), 1e-8), 1.0)
    target_size = max(2, min(n, int(math.ceil(rho * n))))
    num_parts = max(1, min(n, int(math.ceil(n / target_size))))
    if num_parts <= 1:
        return [list(range(n))]

    try:
        import pymetis  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "METIS block partitioning requires the optional dependency 'pymetis'. "
            "Install it or set model.block_partition_mode='connected_bfs'."
        ) from exc

    adj = _edge_index_to_adj_list(edge_index_local, n)
    _edgecuts, membership = pymetis.part_graph(num_parts, adjacency=adj)
    buckets: list[list[int]] = [[] for _ in range(num_parts)]
    for node_idx, part_idx in enumerate(membership):
        buckets[int(part_idx)].append(node_idx)
    blocks = [sorted(block) for block in buckets if block]
    return blocks or [list(range(n))]


def metis_blocks_from_adj_list(adj: list[list[int]], n: int, rho: float) -> List[List[int]]:
    """Partition an adjacency list with METIS."""
    if n <= 0:
        return []
    if n == 1:
        return [[0]]
    rho = min(max(float(rho), 1e-8), 1.0)
    target_size = max(2, min(n, int(math.ceil(rho * n))))
    num_parts = max(1, min(n, int(math.ceil(n / target_size))))
    if num_parts <= 1:
        return [list(range(n))]

    try:
        import pymetis  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "METIS block partitioning requires the optional dependency 'pymetis'. "
            "Install it or set model.block_partition_mode='connected_bfs'."
        ) from exc

    _edgecuts, membership = pymetis.part_graph(num_parts, adjacency=adj)
    buckets: list[list[int]] = [[] for _ in range(num_parts)]
    for node_idx, part_idx in enumerate(membership):
        buckets[int(part_idx)].append(node_idx)
    blocks = [sorted(block) for block in buckets if block]
    return blocks or [list(range(n))]


def _weighted_edges_to_adj(
    edge_index_local: Tensor,
    edge_family_local: Tensor,
    n: int,
    relation_balance_power: float = 0.5,
) -> tuple[list[list[int]], list[list[int]]]:
    """Build integer-weight adjacency with relation-frequency normalization.

    Dense relation families receive smaller per-edge weights, so partitioning is
    structure-first without letting one frequent family dominate all cuts.
    """
    adj_map: list[dict[int, float]] = [dict() for _ in range(n)]
    if edge_index_local.numel() == 0:
        return [[] for _ in range(n)], [[] for _ in range(n)]

    ei = edge_index_local.detach().cpu().long()
    fam = edge_family_local.detach().cpu().long().reshape(-1)
    if fam.numel() != ei.shape[1]:
        fam = torch.zeros(ei.shape[1], dtype=torch.long)
    fam_counts = torch.bincount(fam.clamp_min(0), minlength=int(fam.max().item()) + 1 if fam.numel() else 1).float()
    power = max(0.0, float(relation_balance_power))

    for idx, (a, b) in enumerate(zip(ei[0].tolist(), ei[1].tolist())):
        if a == b or not (0 <= a < n and 0 <= b < n):
            continue
        f = int(fam[idx].item()) if idx < fam.numel() else 0
        count = float(fam_counts[f].item()) if 0 <= f < fam_counts.numel() else 1.0
        w = 1.0 / max(count, 1.0) ** power
        adj_map[a][b] = adj_map[a].get(b, 0.0) + w
        adj_map[b][a] = adj_map[b].get(a, 0.0) + w

    positive = [w for m in adj_map for w in m.values() if w > 0]
    if not positive:
        return [[] for _ in range(n)], [[] for _ in range(n)]
    scale = 1000.0 / max(positive)
    adjacency: list[list[int]] = []
    eweights: list[list[int]] = []
    for m in adj_map:
        items = sorted(m.items())
        adjacency.append([int(v) for v, _ in items])
        eweights.append([max(1, int(round(float(w) * scale))) for _, w in items])
    return adjacency, eweights


def _recursive_weighted_kl_blocks(
    adjacency: list[list[int]],
    eweights: list[list[int]],
    n: int,
    num_parts: int,
) -> List[List[int]]:
    """Dependency-free-ish fallback using NetworkX weighted bisection."""
    try:
        import networkx as nx  # type: ignore
    except ImportError:
        return connected_blocks_from_adj_list(adjacency, n, 1.0 / max(1, num_parts))

    graph = nx.Graph()
    graph.add_nodes_from(range(n))
    for u, (nbrs, ws) in enumerate(zip(adjacency, eweights)):
        for v, w in zip(nbrs, ws):
            if u < v:
                graph.add_edge(u, v, weight=float(w))

    blocks: List[List[int]] = [list(range(n))]
    while len(blocks) < num_parts:
        split_idx = max(range(len(blocks)), key=lambda i: len(blocks[i]))
        block = blocks.pop(split_idx)
        if len(block) <= 2:
            blocks.append(block)
            break
        sub = graph.subgraph(block).copy()
        if sub.number_of_edges() == 0:
            mid = len(block) // 2
            left, right = block[:mid], block[mid:]
        else:
            mid = len(block) // 2
            init = (set(block[:mid]), set(block[mid:]))
            try:
                left_set, right_set = nx.algorithms.community.kernighan_lin_bisection(
                    sub, partition=init, weight="weight", max_iter=10
                )
                left, right = sorted(left_set), sorted(right_set)
            except Exception:
                left, right = block[:mid], block[mid:]
        if not left or not right:
            blocks.append(block)
            break
        blocks.extend([left, right])
    return [sorted(b) for b in blocks if b]



def _refine_blocks_by_degree_load(
    blocks: List[List[int]],
    edge_index_local: Tensor,
    n: int,
    node_type_local: Optional[Tensor] = None,
    max_iter: int = 200,
) -> List[List[int]]:
    """Balance hub/degree mass by same-type swaps after METIS.

    METIS balances node counts and minimizes cuts, which can cluster hubs into one
    block. For block-query training we want each block to carry similar structural
    load. This refinement only swaps nodes with the same node type, so per-block
    type-count templates stay unchanged.
    """
    if not blocks or len(blocks) <= 1 or n <= 1 or edge_index_local.numel() == 0:
        return blocks
    max_iter = max(0, int(max_iter))
    if max_iter <= 0:
        return blocks

    deg = torch.zeros(n, dtype=torch.float32)
    ei = edge_index_local.detach().cpu().long()
    for u, v in zip(ei[0].tolist(), ei[1].tolist()):
        if u == v or not (0 <= int(u) < n and 0 <= int(v) < n):
            continue
        deg[int(u)] += 1.0
        deg[int(v)] += 1.0
    if float(deg.sum().item()) <= 0.0:
        return blocks

    if node_type_local is None:
        types = torch.zeros(n, dtype=torch.long)
    else:
        types = node_type_local.detach().cpu().long().reshape(-1)
        if types.numel() != n:
            types = torch.zeros(n, dtype=torch.long)

    refined = [list(map(int, b)) for b in blocks if b]
    if len(refined) <= 1:
        return refined
    loads = [float(deg[torch.tensor(b, dtype=torch.long)].sum().item()) if b else 0.0 for b in refined]

    # Keep candidate search bounded. The highest-degree nodes in the overloaded
    # block and lowest-degree nodes in the underloaded block matter most.
    cand_limit = 64
    eps = max(1.0, float(deg.mean().item()) * 0.05)
    for _ in range(max_iter):
        hi = max(range(len(refined)), key=lambda i: loads[i])
        lo = min(range(len(refined)), key=lambda i: loads[i])
        current_gap = float(loads[hi] - loads[lo])
        if current_gap <= eps:
            break
        hi_nodes = refined[hi]
        lo_nodes = refined[lo]
        if not hi_nodes or not lo_nodes:
            break
        hi_by_type: dict[int, list[int]] = {}
        lo_by_type: dict[int, list[int]] = {}
        for u in hi_nodes:
            hi_by_type.setdefault(int(types[u].item()), []).append(int(u))
        for v in lo_nodes:
            lo_by_type.setdefault(int(types[v].item()), []).append(int(v))

        best = None
        best_gap = current_gap
        for t, hs in hi_by_type.items():
            ls = lo_by_type.get(t)
            if not ls:
                continue
            hs_sorted = sorted(hs, key=lambda x: float(deg[x].item()), reverse=True)[:cand_limit]
            ls_sorted = sorted(ls, key=lambda x: float(deg[x].item()))[:cand_limit]
            for u in hs_sorted:
                du = float(deg[u].item())
                for v in ls_sorted:
                    dv = float(deg[v].item())
                    if du <= dv:
                        continue
                    new_hi = loads[hi] - du + dv
                    new_lo = loads[lo] - dv + du
                    new_gap = abs(new_hi - new_lo)
                    if new_gap + 1e-6 < best_gap:
                        best_gap = new_gap
                        best = (u, v, du, dv, new_hi, new_lo)
        if best is None:
            break
        u, v, du, dv, new_hi, new_lo = best
        refined[hi][refined[hi].index(u)] = v
        refined[lo][refined[lo].index(v)] = u
        refined[hi].sort()
        refined[lo].sort()
        loads[hi] = float(new_hi)
        loads[lo] = float(new_lo)
    return [sorted(b) for b in refined if b]

def hetero_metis_blocks_from_graph(
    edge_index_local: Tensor,
    edge_family_local: Tensor,
    n: int,
    rho: float,
    *,
    relation_balance_power: float = 0.5,
    node_type_local: Optional[Tensor] = None,
    refine_degree_balance: bool = False,
    refine_max_iter: int = 200,
) -> List[List[int]]:
    """Hard, disjoint blocks for heterogeneous graphs.

    The objective is structure-first: minimize relation-normalized weighted cut
    while keeping block size near ``rho * n``. Relation normalization prevents a
    dense family from dominating the partition.
    """
    if n <= 0:
        return []
    if n == 1:
        return [[0]]
    rho = min(max(float(rho), 1e-8), 1.0)
    target_size = max(2, min(n, int(math.ceil(rho * n))))
    num_parts = max(1, min(n, int(math.ceil(n / target_size))))
    if num_parts <= 1:
        return [list(range(n))]

    adjacency, eweights = _weighted_edges_to_adj(
        edge_index_local, edge_family_local, n, relation_balance_power=relation_balance_power
    )
    if sum(len(x) for x in adjacency) == 0:
        return _strip_blocks(n, target_size)

    try:
        import pymetis  # type: ignore
        xadj = [0]
        adjncy: list[int] = []
        ewgt: list[int] = []
        for nbrs, ws in zip(adjacency, eweights):
            adjncy.extend(nbrs)
            ewgt.extend(ws)
            xadj.append(len(adjncy))
        _edgecuts, membership = pymetis.part_graph(num_parts, xadj=xadj, adjncy=adjncy, eweights=ewgt)
        buckets: list[list[int]] = [[] for _ in range(num_parts)]
        for node_idx, part_idx in enumerate(membership):
            buckets[int(part_idx)].append(node_idx)
        blocks = [sorted(block) for block in buckets if block]
    except Exception:
        blocks = _recursive_weighted_kl_blocks(adjacency, eweights, n, num_parts)
    blocks = blocks or [list(range(n))]
    if bool(refine_degree_balance):
        blocks = _refine_blocks_by_degree_load(
            blocks,
            edge_index_local,
            n,
            node_type_local=node_type_local,
            max_iter=int(refine_max_iter),
        )
    return blocks or [list(range(n))]


def partition_blocks_from_graph(
    edge_index_local: Tensor,
    n: int,
    rho: float,
    *,
    mode: str = "connected_bfs",
    generator: Optional[torch.Generator] = None,
) -> List[List[int]]:
    """Common partition backend used by training and sampling."""
    if mode == "connected_bfs":
        return connected_blocks_from_graph(edge_index_local, n, rho, generator=generator)
    if mode == "metis":
        return metis_blocks_from_graph(edge_index_local, n, rho)
    raise ValueError(
        f"Unknown block partition mode {mode!r}; expected 'connected_bfs' or 'metis'."
    )


def partition_blocks_from_edge_pairs(
    edge_index_local: list[list[int]],
    n: int,
    rho: float,
    *,
    mode: str = "connected_bfs",
    generator: Optional[torch.Generator] = None,
) -> List[List[int]]:
    """Common partition backend for multiprocessing workers using Python edge lists."""
    adj = _edge_pairs_to_adj_list(edge_index_local, n)
    if mode == "connected_bfs":
        return connected_blocks_from_adj_list(adj, n, rho, generator=generator)
    if mode == "metis":
        return metis_blocks_from_adj_list(adj, n, rho)
    raise ValueError(
        f"Unknown block partition mode {mode!r}; expected 'connected_bfs' or 'metis'."
    )


def block_condensed_indices(block: list[int], n: int) -> list[int]:
    """Return all condensed edge-pair ids induced by a local node block."""
    block_nodes = sorted(int(x) for x in block)
    return [
        int(triu_condensed_index(block_nodes[i], block_nodes[j], n))
        for i in range(len(block_nodes))
        for j in range(i + 1, len(block_nodes))
    ]


def partition_blocks_worker_chunk(
    chunk: list[tuple[int, list[list[int]], int, float, str, int]]
) -> list[tuple[int, list[list[int]]]]:
    """Partition graphs and precompute block-local condensed pair ids."""
    out: list[tuple[int, list[list[int]]]] = []
    for graph_idx, edge_index_local, n, rho, mode, seed in chunk:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))
        blocks = partition_blocks_from_edge_pairs(
            edge_index_local,
            int(n),
            float(rho),
            mode=mode,
            generator=generator,
        )
        block_indices = [block_condensed_indices(block, int(n)) for block in blocks]
        out.append((int(graph_idx), block_indices))
    return out
