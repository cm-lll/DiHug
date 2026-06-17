import torch
import torch.nn.functional as F
from torch_geometric.utils import coalesce

import utils


def condensed_to_matrix_index(condensed_index, num_nodes):
    """From https://stackoverflow.com/questions/5323818/condensed-matrix-function-to-find-pairs.
    condensed_index: (E)
    num_nodes: (bs)
    """
    b = 1 - (2 * num_nodes)
    i = torch.div(
        (-b - torch.sqrt(b**2 - 8 * condensed_index)), 2, rounding_mode="floor"
    )
    j = condensed_index + torch.div(i * (b + i + 2), 2, rounding_mode="floor") + 1
    return torch.vstack((i.long(), j.long()))


def matrix_to_condensed_index(matrix_index, num_nodes):
    """From https://stackoverflow.com/questions/5323818/condensed-matrix-function-to-find-pairs.
    matrix_index: (2, E)
    num_nodes: (bs).
    """
    n = num_nodes
    i = matrix_index[0]
    j = matrix_index[1]
    index = n * (n - 1) / 2 - (n - i) * (n - i - 1) / 2 + j - i - 1
    return index
        

def matrix_to_condensed_index_batch(matrix_index, num_nodes, edge_batch):
    """From https://stackoverflow.com/questions/5323818/condensed-matrix-function-to-find-pairs.
    matrix_index: (2, E)
    num_nodes: (bs).
    """
    n = num_nodes[edge_batch]
    i = matrix_index[0]
    j = matrix_index[1]
    index = n * (n - 1) / 2 - (n - i) * (n - i - 1) / 2 + j - i - 1
    return index


def condensed_to_matrix_index_batch(condensed_index, num_nodes, edge_batch, ptr):
    """From https://stackoverflow.com/questions/5323818/condensed-matrix-function-to-find-pairs.
    condensed_index: (E) example: [0, 1, 0, 2] where [0, 1] are edges for graph0 and [0,2] edges for graph 1
    num_nodes: (bs)
    edge_batch: (E): tells to which graph each edge belongs
    ptr: (bs+1): contains the offset for the number of nodes in each graph.
    """
    bb = -2 * num_nodes[edge_batch] + 1

    # Edge ptr adds an offset of n (n-1) / 2 to each edge index
    ptr_condensed_index = condensed_index
    ii = torch.div(
        (-bb - torch.sqrt(bb**2 - 8 * ptr_condensed_index)), 2, rounding_mode="floor"
    )
    jj = (
        ptr_condensed_index
        + torch.div(ii * (bb + ii + 2), 2, rounding_mode="floor")
        + 1
    )
    n_per_edge = num_nodes[edge_batch].long()
    # 防止数值/边界导致负或越界索引（避免下游 get_computational_graph 报 negative index）
    max_idx = n_per_edge - 1
    zeros = torch.zeros_like(max_idx)
    ii = ii.long()
    jj = jj.long()
    ii = torch.maximum(zeros, torch.minimum(ii, max_idx))
    jj = torch.maximum(zeros, torch.minimum(jj, max_idx))
    return torch.vstack((ii, jj)) + ptr[edge_batch]


def get_computational_graph(
    triu_query_edge_index,
    clean_edge_index,
    clean_edge_attr,
    triu=True,
    heterogeneous=False,
    for_message_passing=True,
    total_num_nodes=None,
):
    """
    构建「用于消息传递的计算图」= 噪声边(clean) + 查询边(query)，合并去重。

    含义：clean = 加噪后的图上已有边（模型看到的当前图结构）；query = 本轮要预测的边。
    两者都放进同一张图，模型才能沿边做 MP，并在 query 边上输出预测用于算损失。
    详见 docs/VALIDATION_MP_GRAPH.md。

    :param triu_query_edge_index: 本轮采样的查询边 (2, E_q)，上三角或有向
    :param clean_edge_index/clean_edge_attr: 加噪后的边（即 sparse_noisy_data 的 edge_index_t / edge_attr_t）
    :param heterogeneous: 是否为异质图
    :param for_message_passing: True 时为 MP 添加反向边；False 时（采样）保持有向
    :param total_num_nodes: 可选，总节点数；提供时会对 edge_index 做越界检查
    :return: query_mask（哪些边来自 query）, comp_edge_index, comp_edge_attr（合并去重后的图）
    """
    # ---- Strict validation: fail fast instead of silent fallback ----
    if clean_edge_index.dim() != 2 or clean_edge_index.shape[0] != 2:
        raise RuntimeError(
            f"clean_edge_index must be shape (2, E), got {tuple(clean_edge_index.shape)}"
        )
    if triu_query_edge_index.dim() != 2 or triu_query_edge_index.shape[0] != 2:
        raise RuntimeError(
            f"triu_query_edge_index must be shape (2, E_q), got {tuple(triu_query_edge_index.shape)}"
        )
    if clean_edge_attr.dim() != 2:
        raise RuntimeError(
            f"clean_edge_attr must be 2D one-hot tensor (E, de), got {tuple(clean_edge_attr.shape)}"
        )
    if clean_edge_attr.shape[0] != clean_edge_index.shape[1]:
        raise RuntimeError(
            f"clean edge count mismatch: clean_edge_index has {clean_edge_index.shape[1]} edges, "
            f"but clean_edge_attr has {clean_edge_attr.shape[0]} rows"
        )
    if clean_edge_attr.shape[1] <= 0:
        raise RuntimeError(
            f"invalid clean_edge_attr width de={clean_edge_attr.shape[1]} (must be > 0)"
        )
    if clean_edge_index.numel() > 0 and (clean_edge_index < 0).any():
        bad_min = int(clean_edge_index.min().item())
        raise RuntimeError(f"clean_edge_index contains negative index: min={bad_min}")
    if triu_query_edge_index.numel() > 0 and (triu_query_edge_index < 0).any():
        bad_min = int(triu_query_edge_index.min().item())
        raise RuntimeError(f"triu_query_edge_index contains negative index: min={bad_min}")
    
    # Adapter layer (SparseDiff state -> pyHGT MP graph):
    # Diffusion/sampling state keeps canonical single directed edges per undirected slot.
    # For message passing only, duplicate (u,v)/(v,u) via to_undirected so both endpoints
    # receive neighbor messages without storing rev_* families in the diffusion state.
    # Sampling (for_message_passing=False) keeps the canonical directed graph unchanged.
    if clean_edge_index.shape[1] > 0:
        if not heterogeneous or for_message_passing:
            # 同质图：总是使用to_undirected
            # 异质图：仅在消息传递时使用to_undirected（支持双向信息流通）
            clean_edge_index, clean_edge_attr = utils.to_undirected(clean_edge_index, clean_edge_attr)
        # 异质图且非消息传递（采样阶段）：保持有向边结构，不转换
    if clean_edge_attr.dim() != 2 or clean_edge_attr.shape[0] != clean_edge_index.shape[1]:
        raise RuntimeError(
            f"after clean to_undirected: edge/attr mismatch, "
            f"edge_index={tuple(clean_edge_index.shape)}, edge_attr={tuple(clean_edge_attr.shape)}"
        )
    
    de = clean_edge_attr.shape[-1]
    device = triu_query_edge_index.device

    # create default query edge attr
    default_query_edge_attr = torch.zeros((triu_query_edge_index.shape[1], de)).to(
        device
    )
    default_query_edge_attr[:, 0] = 1

    # if query_edge_attr is None, use default query edge attr
    # 异质图：在消息传递时需要双向信息流通，但在采样时保持有向边结构
    if triu:
        if not heterogeneous or for_message_passing:
            # 同质图：总是使用to_undirected
            # 异质图：仅在消息传递时使用to_undirected（支持双向信息流通）
            query_edge_index, default_query_edge_attr = utils.to_undirected(
                triu_query_edge_index, default_query_edge_attr
            )
        else:
            # 异质图且非消息传递（采样阶段）：保持有向边结构
            query_edge_index, default_query_edge_attr = triu_query_edge_index, default_query_edge_attr
    else:
        query_edge_index, default_query_edge_attr = triu_query_edge_index, default_query_edge_attr
    if default_query_edge_attr.dim() != 2 or default_query_edge_attr.shape[0] != query_edge_index.shape[1]:
        raise RuntimeError(
            f"query edge/attr mismatch, edge_index={tuple(query_edge_index.shape)}, "
            f"edge_attr={tuple(default_query_edge_attr.shape)}"
        )

    # get the computational graph: positive edges + random edges
    comp_edge_index = torch.hstack([clean_edge_index, query_edge_index])
    if comp_edge_index.numel() == 0:
        raise RuntimeError(
            "computational graph has zero edges (clean + query are both empty). "
            "This is unexpected for validation/training."
        )
    if total_num_nodes is not None and comp_edge_index.numel() > 0:
        max_idx = int(comp_edge_index.max().item())
        if max_idx >= int(total_num_nodes):
            raise RuntimeError(
                f"comp_edge_index out of range: max={max_idx} >= total_num_nodes={int(total_num_nodes)}. "
                f"clean_edge_index.shape={tuple(clean_edge_index.shape)}, "
                f"query_edge_index.shape={tuple(query_edge_index.shape)}"
            )

    comp_attr_stacked = torch.vstack([clean_edge_attr, default_query_edge_attr])
    if comp_attr_stacked.shape[0] != comp_edge_index.shape[1]:
        raise RuntimeError(
            f"comp edge/attr mismatch before coalesce: edges={comp_edge_index.shape[1]}, "
            f"attr_rows={comp_attr_stacked.shape[0]}"
        )
    default_comp_edge_attr = torch.argmax(comp_attr_stacked, -1).long()

    # reduce repeated edges and get the mask
    assert comp_edge_index.dtype == torch.long
    _, min_default_edge_attr = coalesce(
        comp_edge_index, default_comp_edge_attr, reduce="min"
    )

    max_comp_edge_index, max_default_edge_attr = coalesce(
        comp_edge_index, default_comp_edge_attr, reduce="max"
    )
    if max_comp_edge_index.numel() > 0 and (max_comp_edge_index < 0).any():
        bad_min = int(max_comp_edge_index.min().item())
        raise RuntimeError(f"coalesce produced negative edge index: min={bad_min}")
    if max_default_edge_attr.numel() > 0:
        min_attr = int(max_default_edge_attr.min().item())
        max_attr = int(max_default_edge_attr.max().item())
        if min_attr < 0 or max_attr >= de:
            raise RuntimeError(
                f"coalesce produced edge attr out of range: min={min_attr}, max={max_attr}, de={de}. "
                f"clean_edge_attr.shape={tuple(clean_edge_attr.shape)}, "
                f"default_query_edge_attr.shape={tuple(default_query_edge_attr.shape)}, "
                f"comp_edge_index.shape={tuple(comp_edge_index.shape)}"
            )
    query_mask = min_default_edge_attr == 0
    comp_edge_attr = F.one_hot(max_default_edge_attr.long(), num_classes=de).float()

    return query_mask, max_comp_edge_index, comp_edge_attr


def check_symmetry(edge_index):
    cond1 = edge_index[0].sort()[0].equal(edge_index[1].sort()[0])
    cond2 = (edge_index[0] < edge_index[1]).sum() == (
        edge_index[1] < edge_index[0]
    ).sum()
    print((edge_index[0] < edge_index[1]).sum(), (edge_index[1] < edge_index[0]).sum())
    return cond1 and cond2


def mask_query_graph_from_comp_graph(
    triu_query_edge_index, edge_index, edge_attr, num_classes, heterogeneous=False, for_message_passing=True
):
    # 异质图：在消息传递时需要双向信息流通，但在采样时保持有向边结构
    if heterogeneous and not for_message_passing:
        # 异质图且非消息传递（采样阶段）：保持有向边结构
        query_edge_index = triu_query_edge_index
    else:
        # 同质图：总是使用to_undirected
        # 异质图且消息传递：使用to_undirected支持双向信息流通
        query_edge_index = utils.to_undirected(triu_query_edge_index)
    # import pdb; pdb.set_trace()

    all_edge_index = torch.hstack([edge_index, query_edge_index])
    all_edge_attr = torch.hstack(
        [
            torch.argmax(edge_attr, -1),
            torch.zeros(query_edge_index.shape[1]).to(edge_index.device),
        ]
    )

    assert all_edge_index.dtype == torch.long
    _, min_edge_attr = coalesce(all_edge_index, all_edge_attr, reduce="min")

    max_edge_index, max_edge_attr = coalesce(
        all_edge_index, all_edge_attr, reduce="max"
    )

    return (
        min_edge_attr == 0,
        F.one_hot(max_edge_attr.long(), num_classes=num_classes),
        max_edge_index,
    )


def sample_non_existing_edge_attr(query_edges_dist_batch, num_edges_to_sample):
    device = query_edges_dist_batch.device
    max_edges_to_sample = int(num_edges_to_sample.max())

    if max_edges_to_sample == 0:
        return torch.tensor([]).to(device)

    query_mask = (
        torch.ones((len(num_edges_to_sample), max_edges_to_sample))
        .cumsum(-1)
        .to(device)
    )
    query_mask[
        query_mask > num_edges_to_sample.unsqueeze(-1).repeat(1, max_edges_to_sample)
    ] = 0
    query_mask[query_mask > 0] = 1
    query_edge_attr = (
        torch.multinomial(query_edges_dist_batch, max_edges_to_sample, replacement=True)
        + 1
    )
    query_edge_attr = query_edge_attr.flatten()[query_mask.flatten().bool()]

    return query_edge_attr
