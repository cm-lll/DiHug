#!/usr/bin/env python3
"""Smoke test: fused HGTConv message vs legacy loop (same weights, same inputs)."""
from __future__ import annotations

import argparse
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "sparse_diffusion"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import torch

from sparse_diffusion.models.pyhgt_denoiser import HGTConv


def _random_pubmed_like_graph(
    num_nodes: int,
    num_edges: int,
    num_types: int,
    num_relations: int,
    dx: int,
    device: torch.device,
) -> tuple[torch.Tensor, ...]:
    gen = torch.Generator(device=device)
    gen.manual_seed(42)
    node_type = torch.randint(0, num_types, (num_nodes,), device=device, generator=gen)
    src = torch.randint(0, num_nodes, (num_edges,), device=device, generator=gen)
    dst = torch.randint(0, num_nodes, (num_edges,), device=device, generator=gen)
    edge_index = torch.stack([src, dst], dim=0)
    edge_type = torch.randint(0, num_relations, (num_edges,), device=device, generator=gen)
    edge_family = torch.randint(0, 16, (num_edges,), device=device, generator=gen)
    edge_subtype = torch.zeros(num_edges, dtype=torch.long, device=device)
    node_inp = torch.randn(num_nodes, dx, device=device, generator=gen)
    return node_inp, node_type, edge_index, edge_type, edge_subtype, edge_family


def _run_conv(conv: HGTConv, use_legacy: bool, *args) -> torch.Tensor:
    orig_message = conv.message
    try:
        if use_legacy:
            conv.message = conv.message_legacy  # type: ignore[method-assign]
        else:
            conv.message = orig_message
        return conv(*args)
    finally:
        conv.message = orig_message


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--nodes", type=int, default=1565)
    parser.add_argument("--edges", type=int, default=327776)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--rtol", type=float, default=1e-4)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dx = 128
    num_types = 4
    num_relations = 16
    n_heads = 8

    conv = HGTConv(
        dx,
        dx,
        num_types,
        num_relations,
        n_heads,
        dropout=0.0,
        use_norm=True,
        use_edge_phi_fusion=True,
        num_edge_families=16,
        max_edge_phi=1,
        use_dual_softmax=True,
    ).to(device)
    conv.eval()

    node_inp, node_type, edge_index, edge_type, edge_subtype, edge_family = _random_pubmed_like_graph(
        args.nodes, args.edges, num_types, num_relations, dx, device
    )

    with torch.no_grad():
        out_fused = _run_conv(conv, False, node_inp, node_type, edge_index, edge_type, edge_subtype, edge_family)
        out_legacy = _run_conv(conv, True, node_inp, node_type, edge_index, edge_type, edge_subtype, edge_family)
        _run_conv(conv, False, node_inp, node_type, edge_index, edge_type, edge_subtype, edge_family)

    max_diff = (out_fused - out_legacy).abs().max().item()
    mean_diff = (out_fused - out_legacy).abs().mean().item()
    ok = torch.allclose(out_fused, out_legacy, atol=args.atol, rtol=args.rtol)
    print(f"device={device} nodes={args.nodes} edges={args.edges}")
    print(f"max_abs_diff={max_diff:.6e} mean_abs_diff={mean_diff:.6e} allclose={ok}")

    if device.type == "cuda":
        torch.cuda.synchronize(device)

        def bench(legacy: bool) -> float:
            for _ in range(2):
                _run_conv(conv, legacy, node_inp, node_type, edge_index, edge_type, edge_subtype, edge_family)
            torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            for _ in range(args.iters):
                _run_conv(conv, legacy, node_inp, node_type, edge_index, edge_type, edge_subtype, edge_family)
            torch.cuda.synchronize(device)
            return (time.perf_counter() - t0) / args.iters

        t_legacy = bench(True)
        t_fused = bench(False)
        speedup = t_legacy / t_fused if t_fused > 0 else float("inf")
        print(f"latency_ms legacy={t_legacy * 1e3:.1f} fused={t_fused * 1e3:.1f} speedup={speedup:.2f}x")

    if not ok:
        raise SystemExit(1)
    print("OK")


if __name__ == "__main__":
    main()
