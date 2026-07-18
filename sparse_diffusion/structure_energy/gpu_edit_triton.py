from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _edge_status_after(
    adjacency,
    left,
    right,
    candidate,
    old_edges,
    new_edges,
    old_valid,
    new_valid,
    node_count: tl.constexpr,
):
    value = tl.load(adjacency + left * node_count + right)
    for edit_slot in tl.static_range(0, 2):
        base = candidate * 4 + edit_slot * 2
        a = tl.load(old_edges + base)
        b = tl.load(old_edges + base + 1)
        valid = tl.load(old_valid + candidate * 2 + edit_slot)
        match = valid & (((a == left) & (b == right)) | ((b == left) & (a == right)))
        value = value & ~match
    for edit_slot in tl.static_range(0, 2):
        base = candidate * 4 + edit_slot * 2
        a = tl.load(new_edges + base)
        b = tl.load(new_edges + base + 1)
        valid = tl.load(new_valid + candidate * 2 + edit_slot)
        match = valid & (((a == left) & (b == right)) | ((b == left) & (a == right)))
        value = value | match
    return value


@triton.jit
def _typed_closure_kernel(
    adjacency,
    node_types,
    degrees,
    actions,
    old_edges,
    new_edges,
    old_valid,
    new_valid,
    slot_counts,
    normalized,
    node_count: tl.constexpr,
    type_count: tl.constexpr,
    BLOCK: tl.constexpr,
):
    candidate = tl.program_id(0)
    slot = tl.program_id(1)
    block = tl.program_id(2)
    node = block * BLOCK + tl.arange(0, BLOCK)
    node_mask = node < node_count
    safe_node = tl.where(node_mask, node, 0)
    is_new = slot >= 2
    edit_slot = slot % 2
    edge_base = candidate * 4 + edit_slot * 2
    old_slot_valid = tl.load(old_valid + candidate * 2 + edit_slot)
    new_slot_valid = tl.load(new_valid + candidate * 2 + edit_slot)
    valid = tl.where(is_new, new_slot_valid, old_slot_valid)
    old_left = tl.load(old_edges + edge_base)
    old_right = tl.load(old_edges + edge_base + 1)
    new_left = tl.load(new_edges + edge_base)
    new_right = tl.load(new_edges + edge_base + 1)
    left = tl.where(is_new, new_left, old_left)
    right = tl.where(is_new, new_right, old_right)
    left = tl.where(valid, left, 0)
    right = tl.where(valid, right, 0)

    left_status = tl.load(
        adjacency + left * node_count + node, mask=node_mask, other=0
    )
    right_status = tl.load(
        adjacency + right * node_count + node, mask=node_mask, other=0
    )
    left_after = _edge_status_after(
        adjacency,
        left,
        safe_node,
        candidate,
        old_edges,
        new_edges,
        old_valid,
        new_valid,
        node_count,
    )
    right_after = _edge_status_after(
        adjacency,
        right,
        safe_node,
        candidate,
        old_edges,
        new_edges,
        old_valid,
        new_valid,
        node_count,
    )
    left_status = tl.where(is_new, left_after, left_status)
    right_status = tl.where(is_new, right_after, right_status)
    common = node_mask & valid & left_status & right_status
    types = tl.load(node_types + safe_node, mask=node_mask, other=0)
    for type_id in tl.static_range(0, type_count):
        value = tl.sum((common & (types == type_id)).to(tl.float32), axis=0)
        tl.atomic_add(
            slot_counts + candidate * 4 * type_count + slot * type_count + type_id,
            value,
        )

    action = tl.load(actions + candidate)
    is_switch = action == 4
    degree_node = tl.load(degrees + node, mask=node_mask, other=0).to(tl.float32)
    degree_left = tl.load(degrees + left).to(tl.float32)
    degree_right = tl.load(degrees + right).to(tl.float32)
    denom_node = degree_node * (degree_node - 1.0) * 0.5
    denom_left = degree_left * (degree_left - 1.0) * 0.5
    denom_right = degree_right * (degree_right - 1.0) * 0.5
    inv_node = tl.where(denom_node > 0, 1.0 / denom_node, 0.0)
    inv_left = tl.where(denom_left > 0, 1.0 / denom_left, 0.0)
    inv_right = tl.where(denom_right > 0, 1.0 / denom_right, 0.0)
    sign = tl.where(is_new, 1.0, -1.0)
    contribution = tl.sum(
        common.to(tl.float32)
        * sign
        * (inv_node + inv_left + inv_right)
        * is_switch,
        axis=0,
    )
    tl.atomic_add(normalized + candidate, contribution)


def fused_typed_closure(
    adjacency: torch.Tensor,
    node_types: torch.Tensor,
    degrees: torch.Tensor,
    actions: torch.Tensor,
    old_edges: torch.Tensor,
    new_edges: torch.Tensor,
    old_valid: torch.Tensor,
    new_valid: torch.Tensor,
    type_count: int,
    block_size: int = 256,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not adjacency.is_cuda:
        raise ValueError("fused typed closure requires CUDA tensors")
    candidate_count = int(actions.shape[0])
    node_count = int(adjacency.shape[0])
    slot_counts = torch.zeros(
        (candidate_count, 4, int(type_count)),
        dtype=torch.float32,
        device=adjacency.device,
    )
    normalized = torch.zeros(
        candidate_count, dtype=torch.float32, device=adjacency.device
    )
    grid = (
        candidate_count,
        4,
        triton.cdiv(node_count, int(block_size)),
    )
    _typed_closure_kernel[grid](
        adjacency,
        node_types,
        degrees,
        actions,
        old_edges,
        new_edges,
        old_valid,
        new_valid,
        slot_counts,
        normalized,
        node_count=node_count,
        type_count=int(type_count),
        BLOCK=int(block_size),
    )
    return slot_counts, normalized
