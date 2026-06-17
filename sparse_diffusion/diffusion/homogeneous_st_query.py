"""
Homogeneous-graph query sampling via spanning-tree partition (training-time, uses clean graph).

- Builds a BFS spanning tree on the largest connected component, then **Tree K-cut**:
  iteratively balanced tree cuts until K connected blocks (K>=1).
- K = ceil(1/edge_fraction), clamped by `homogeneous_tree_k_max` and n.

Optional: expand each block by 1-hop neighbors on the **clean** graph (`homogeneous_block_1hop_expand`).

Sampling / inference: training uses clean `data.edge_index` only for partitioning.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Dict, List, Optional, Set, Tuple

import torch
from torch import Tensor

def _undirected_adjacency(edge_index: Tensor, num_nodes: int) -> List[List[int]]:
    ei = edge_index.detach().cpu()
    adj: List[List[int]] = [[] for _ in range(num_nodes)]
    for i in range(ei.shape[1]):
        u, v = int(ei[0, i]), int(ei[1, i])
        if u == v or u < 0 or v < 0 or u >= num_nodes or v >= num_nodes:
            continue
        adj[u].append(v)
        adj[v].append(u)
    return adj


def _largest_cc_nodes(adj: List[List[int]], num_nodes: int) -> List[int]:
    seen = [False] * num_nodes
    best: List[int] = []
    for s in range(num_nodes):
        if seen[s]:
            continue
        comp: List[int] = []
        stack = [s]
        seen[s] = True
        while stack:
            u = stack.pop()
            comp.append(u)
            for v in adj[u]:
                if not seen[v]:
                    seen[v] = True
                    stack.append(v)
        if len(comp) > len(best):
            best = comp
    return best if best else [0]


def _bfs_take_k(
    adj: List[List[int]],
    allowed: Tensor,
    start: int,
    k: int,
) -> Tensor:
    """BFS on CPU adjacency, restricted to allowed nodes. Returns (<=k,) long tensor."""
    if k <= 0:
        return torch.zeros(0, dtype=torch.long)
    n = int(allowed.numel())
    if n == 0:
        return torch.zeros(0, dtype=torch.long)
    if start < 0 or start >= n or not bool(allowed[start].item()):
        start = int(torch.where(allowed)[0][0].item())
    seen = torch.zeros(n, dtype=torch.bool)
    out: List[int] = []
    q: deque = deque([start])
    seen[start] = True
    while q and len(out) < k:
        u = q.popleft()
        if not bool(allowed[u].item()):
            continue
        out.append(u)
        for v in adj[u]:
            if 0 <= v < n and (not bool(seen[v].item())) and bool(allowed[v].item()):
                seen[v] = True
                q.append(v)
    if not out:
        return torch.where(allowed)[0][:k].long()
    return torch.tensor(out, dtype=torch.long)


def one_hop_expand_node_set_clean(
    adj: List[List[int]], node_ids: Tensor, num_nodes: int
) -> Tensor:
    """
    Union of node_ids and their 1-hop neighbors on the **clean** undirected adjacency.
    Local node indices in [0, num_nodes). Deduplicated by construction (set).
    """
    s = {int(x) for x in node_ids.detach().cpu().tolist()}
    for u in list(s):
        if 0 <= u < num_nodes:
            for v in adj[u]:
                if 0 <= v < num_nodes:
                    s.add(v)
    if not s:
        return torch.zeros(0, dtype=torch.long)
    return torch.tensor(sorted(s), dtype=torch.long)


def _subtree_sizes_parent_tree(parent: List[int]) -> Tuple[List[int], List[List[int]]]:
    n = len(parent)
    children: List[List[int]] = [[] for _ in range(n)]
    root = 0
    for v in range(n):
        p = parent[v]
        if p < 0:
            continue
        if p == v:
            root = v
            continue
        children[p].append(v)

    size = [1] * n
    order: List[int] = []

    def dfs(u: int) -> None:
        for c in children[u]:
            dfs(c)
        order.append(u)

    dfs(root)
    for u in order:
        s = 1
        for c in children[u]:
            s += size[c]
        size[u] = s
    return size, children


def _build_bfs_tree_on_cc(
    edge_index: Tensor, num_nodes: int
) -> Tuple[List[int], List[int], List[List[int]]]:
    """
    Largest CC, BFS spanning tree on local indices.
    Returns (cc_global, parent_loc length local_n, children_loc).
    """
    adj = _undirected_adjacency(edge_index, num_nodes)
    cc = _largest_cc_nodes(adj, num_nodes)
    if len(cc) < 2:
        return cc, [0], [[]]

    idx_map = {g: i for i, g in enumerate(cc)}
    local_n = len(cc)
    loc_adj: List[List[int]] = [[] for _ in range(local_n)]
    for g in cc:
        for v in adj[g]:
            if v in idx_map:
                loc_adj[idx_map[g]].append(idx_map[v])

    root_local = 0
    parent_loc = [-1] * local_n
    parent_loc[root_local] = root_local
    q: deque = deque([root_local])
    while q:
        u = q.popleft()
        for v in loc_adj[u]:
            if parent_loc[v] == -1:
                parent_loc[v] = u
                q.append(v)
    for v in range(local_n):
        if parent_loc[v] == -1:
            parent_loc[v] = root_local

    children_loc: List[List[int]] = [[] for _ in range(local_n)]
    for v in range(local_n):
        p = parent_loc[v]
        if p >= 0 and p != v:
            children_loc[p].append(v)

    return cc, parent_loc, children_loc


def _count_subtree_side_in_S(
    children_loc: List[List[int]],
    S: Set[int],
    c: int,
    parent_c: int,
) -> int:
    """Count nodes in S reachable from c without going back to parent_c (tree subtree at c)."""
    if c not in S:
        return 0
    cnt = 1
    for ch in children_loc[c]:
        if ch != parent_c and ch in S:
            cnt += _count_subtree_side_in_S(children_loc, S, ch, c)
    return cnt


def _collect_tree_side_from_edge(
    children_loc: List[List[int]],
    S: Set[int],
    c: int,
    p: int,
) -> Set[int]:
    """Component containing c after deleting tree edge (p,c), intersected with S."""
    out: Set[int] = {c}

    def dfs(u: int, par: int) -> None:
        for ch in children_loc[u]:
            if ch != par and ch in S:
                out.add(ch)
                dfs(ch, u)

    dfs(c, p)
    return out


def _balanced_two_way_split_tree_set(
    parent_loc: List[int],
    children_loc: List[List[int]],
    S: Set[int],
) -> Tuple[Set[int], Set[int]]:
    """
    Split connected node set S (local tree indices) into two non-empty parts
    by removing one tree edge with both ends in S, minimizing ||A|-|B||.
    """
    if len(S) < 2:
        return set(S), set()

    best_diff = len(S) + 1
    best_L: Optional[Set[int]] = None
    best_R: Optional[Set[int]] = None

    for c in S:
        p = parent_loc[c]
        if p == c or p not in S:
            continue
        sz_c_side = _count_subtree_side_in_S(children_loc, S, c, p)
        sz_other = len(S) - sz_c_side
        d = abs(sz_c_side - sz_other)
        if d < best_diff:
            best_diff = d
            L_side = _collect_tree_side_from_edge(children_loc, S, c, p)
            R_side = S - L_side
            if len(L_side) > 0 and len(R_side) > 0:
                best_L, best_R = L_side, R_side

    if best_L is None or best_R is None:
        nodes = sorted(S)
        mid = max(1, len(nodes) // 2)
        return set(nodes[:mid]), set(nodes[mid:])

    return best_L, best_R


def _iterative_k_partition_tree_sets(
    parent_loc: List[int],
    children_loc: List[List[int]],
    S_full: Set[int],
    K: int,
) -> List[Set[int]]:
    """K-1 times: split the largest current component by one balanced tree cut → up to K sets."""
    K = max(1, min(K, len(S_full)))
    if K == 1:
        return [set(S_full)]
    components: List[Set[int]] = [set(S_full)]
    for _ in range(K - 1):
        if len(components) >= K:
            break
        idx = max(range(len(components)), key=lambda i: len(components[i]))
        C = components[idx]
        if len(C) < 2:
            break
        L, R = _balanced_two_way_split_tree_set(parent_loc, children_loc, C)
        if len(L) == 0 or len(R) == 0:
            break
        components = [components[i] for i in range(len(components)) if i != idx] + [L, R]
    return components


def tree_k_partition_masks(
    edge_index: Tensor,
    num_nodes: int,
    K: int,
) -> Tuple[List[Tensor], int]:
    """
    Returns (list of K_eff boolean masks shape (num_nodes,), K_eff).
    Masks are disjoint on the largest CC; nodes outside CC are merged into mask 0.
    K_eff may be less than requested K if some tree splits fail.
    """
    K_in = max(1, int(K))
    sub_ei = edge_index.cpu()
    cc, parent_loc, children_loc = _build_bfs_tree_on_cc(sub_ei, num_nodes)
    local_n = len(cc)
    if local_n < 2:
        m = torch.zeros(num_nodes, dtype=torch.bool)
        if cc:
            m[cc[0]] = True
        return [m], 1

    K_req = min(K_in, local_n)
    S_full: Set[int] = set(range(local_n))
    parts = _iterative_k_partition_tree_sets(parent_loc, children_loc, S_full, K_req)
    K_eff = len(parts)

    masks: List[Tensor] = []
    assigned = torch.zeros(num_nodes, dtype=torch.bool)
    for part in parts:
        m = torch.zeros(num_nodes, dtype=torch.bool)
        for li in part:
            if 0 <= li < local_n:
                g = cc[li]
                m[g] = True
                assigned[g] = True
        masks.append(m)
    outside = ~assigned
    if outside.any() and len(masks) > 0:
        masks[0] = masks[0] | outside
    return masks, K_eff


def resolve_tree_k(edge_fraction: float, k_max: int, num_nodes: int) -> int:
    K = int(math.ceil(1.0 / float(edge_fraction)))
    K = max(1, min(K, k_max, max(1, num_nodes)))
    return K


def balanced_two_split_from_bfs_tree(
    edge_index: Tensor, num_nodes: int,
) -> Tuple[Tensor, Tensor]:
    """Backward-compat: 2-way split via tree_k_partition_masks(K=2)."""
    masks, _ = tree_k_partition_masks(edge_index, num_nodes, 2)
    if len(masks) >= 2:
        return masks[0], masks[1]
    return masks[0], ~masks[0]


def sample_query_edges_st_block_full(
    num_nodes_per_graph: Tensor,
    edge_index: Tensor,
    batch: Tensor,
    edge_fraction: float,
    device: torch.device,
    ptr: Tensor,
    max_block_nodes: int = 800,
    generator: Optional[torch.Generator] = None,
    expand_1hop_clean: bool = False,
    log_block_stats: bool = True,
    tree_k_max: int = 64,
) -> Tuple[Tensor, Tensor, Optional[Dict[str, float]]]:
    """
    Tree K-cut (K from edge_fraction via resolve_tree_k); pick one block;
    BFS-connected subset of size min(max_block_nodes, |block|); full upper-triangle
    query inside subset (+ optional 1-hop). Subset size does not use edge_fraction.
    """
    assert num_nodes_per_graph.dtype == torch.long
    bs = int(num_nodes_per_graph.shape[0])
    ef = float(edge_fraction)
    assert 0.0 < ef <= 1.0
    max_block_nodes = int(max(2, max_block_nodes))

    all_edges: List[Tensor] = []
    all_batch: List[Tensor] = []
    stats_accum: Optional[Dict[str, float]] = None
    n_graphs_done = 0

    for b in range(bs):
        n = int(num_nodes_per_graph[b].item())
        if n < 2:
            continue
        lo = int(ptr[b].item())
        sub_ei = edge_index[:, (batch[edge_index[0]] == b)]
        sub_ei = (sub_ei - lo).cpu()
        adj = _undirected_adjacency(sub_ei, n)

        K = resolve_tree_k(ef, tree_k_max, n)
        masks, K_eff = tree_k_partition_masks(sub_ei, n, K)
        K_use = len(masks)
        if generator is not None:
            pick = int(torch.randint(0, K_use, (1,), generator=generator).item())
        else:
            pick = int(torch.randint(0, K_use, (1,)).item())
        block_mask = masks[pick]
        block_n = int(block_mask.sum().item())
        # 资源由树块 + max_block_nodes 控制，不再用 edge_fraction 缩子集
        s_target = max(2, min(max_block_nodes, max(1, block_n)))

        chosen_local = _bfs_take_k(adj, block_mask, start=0, k=s_target)
        if int(chosen_local.numel()) < 2:
            chosen_local = torch.arange(min(n, s_target), dtype=torch.long)

        n_before_1hop = int(chosen_local.numel())
        if expand_1hop_clean and n_before_1hop > 0:
            chosen_local = one_hop_expand_node_set_clean(adj, chosen_local, n)

        m = int(chosen_local.numel())
        pairs = torch.triu_indices(m, m, offset=1)
        if pairs.numel() == 0:
            continue

        chosen_local = chosen_local.to(device)
        ge0 = chosen_local[pairs[0]] + lo
        ge1 = chosen_local[pairs[1]] + lo
        ge = torch.stack([ge0, ge1], dim=0)

        all_edges.append(ge)
        all_batch.append(torch.full((ge.shape[1],), b, dtype=torch.long, device=device))

        if log_block_stats:
            if stats_accum is None:
                stats_accum = {}
            stats_accum["train/homo_block/n_nodes_graph"] = (
                stats_accum.get("train/homo_block/n_nodes_graph", 0.0) + float(n)
            )
            stats_accum["train/homo_block/tree_k"] = stats_accum.get("train/homo_block/tree_k", 0.0) + float(
                K_eff
            )
            stats_accum["train/homo_block/block_pick"] = stats_accum.get("train/homo_block/block_pick", 0.0) + float(
                pick
            )
            if expand_1hop_clean:
                n_after_1hop = m
                stats_accum["train/homo_block/n_nodes_core_before_1hop"] = (
                    stats_accum.get("train/homo_block/n_nodes_core_before_1hop", 0.0)
                    + float(n_before_1hop)
                )
                stats_accum["train/homo_block/n_nodes_after_1hop"] = (
                    stats_accum.get("train/homo_block/n_nodes_after_1hop", 0.0) + float(n_after_1hop)
                )
                stats_accum["train/homo_block/ratio_nodes_after_to_n"] = (
                    stats_accum.get("train/homo_block/ratio_nodes_after_to_n", 0.0)
                    + float(n_after_1hop) / float(max(n, 1))
                )
            n_graphs_done += 1

    if not all_edges:
        raise RuntimeError("st_block_full produced no query edges; use baseline for this batch.")

    edge_index_out = torch.cat(all_edges, dim=1)
    batch_out = torch.cat(all_batch, dim=0)
    out_stats: Optional[Dict[str, float]] = None
    if stats_accum is not None and n_graphs_done > 0:
        out_stats = {k: v / float(n_graphs_done) for k, v in stats_accum.items()}
        out_stats["train/homo_block/n_query_edges"] = float(edge_index_out.shape[1])
    return edge_index_out, batch_out, out_stats
