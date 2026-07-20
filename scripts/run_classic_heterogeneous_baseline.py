#!/usr/bin/env python3
"""Reproducible typed adaptations of classic graph generators.

The runner deliberately shares the same public conditions across methods:
node types are fixed, relation counts are sampled from empirical Binomial
marginals, and every generated edge must belong to a legal schema family.
NetGAN uses the implementation vendored by HG2NP.  The other three models are
compact PyTorch implementations of VGAE, GraphRNN-S, and GraphVAE suitable for
the single-network protocol used in this repository.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import types
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

from sparse_diffusion.structure_energy.graph import TypedGraph, load_typed_graph
from sparse_diffusion.structure_energy.heterogeneous_metrics import (
    aggregate_comparisons,
    compare_heterogeneous_graphs,
)


Pair = Tuple[int, int]


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def family_pairs(graph: TypedGraph, family: str) -> np.ndarray:
    spec = graph.family_specs[family]
    if spec.bipartite:
        return np.asarray(
            [(u, v) for u in spec.src_nodes for v in spec.dst_nodes],
            dtype=np.int64,
        )
    nodes = np.asarray(spec.src_nodes, dtype=np.int64)
    left, right = np.triu_indices(len(nodes), k=1)
    return np.column_stack((nodes[left], nodes[right])).astype(np.int64)


def empirical_family_count(graph: TypedGraph, family: str, rng: np.random.Generator) -> int:
    capacity = len(family_pairs(graph, family))
    probability = len(graph.family_edges[family]) / max(capacity, 1)
    return int(rng.binomial(capacity, probability))


def edge_index(graph: TypedGraph, device: torch.device) -> torch.Tensor:
    entries = []
    for edges in graph.family_edges.values():
        for u, v in edges:
            entries.extend(((u, v), (v, u)))
    return torch.tensor(entries, dtype=torch.long, device=device).t().contiguous()


def select_family_edges(
    graph: TypedGraph,
    scores: Dict[str, np.ndarray],
    rng: np.random.Generator,
) -> TypedGraph:
    selected: Dict[str, List[Pair]] = {name: [] for name in graph.family_specs}
    for family in sorted(graph.family_specs):
        pairs = family_pairs(graph, family)
        count = min(empirical_family_count(graph, family, rng), len(pairs))
        if count == 0:
            continue
        values = np.asarray(scores[family], dtype=np.float64).reshape(-1)
        if len(values) != len(pairs):
            raise ValueError(f"score length mismatch for {family}")
        # Gumbel top-k is sampling without replacement from positive weights.
        logits = values - np.max(values)
        gumbel = -np.log(-np.log(rng.uniform(1e-9, 1 - 1e-9, len(values))))
        indices = np.argpartition(logits + gumbel, -count)[-count:]
        selected[family] = [tuple(map(int, pairs[i])) for i in indices]
    generated = TypedGraph(
        graph.node_types.copy(), graph.type_names, graph.family_specs, selected
    )
    generated.validate()
    return generated


class TypedVGAE(nn.Module):
    def __init__(self, num_types: int, num_families: int, hidden: int, latent: int):
        super().__init__()
        self.type_embedding = nn.Embedding(num_types, hidden)
        self.conv1 = GCNConv(hidden, hidden)
        self.conv_mu = GCNConv(hidden, latent)
        self.conv_logstd = GCNConv(hidden, latent)
        self.relations = nn.Parameter(torch.empty(num_families, latent, latent))
        nn.init.xavier_uniform_(self.relations)

    def encode(self, node_types: torch.Tensor, edges: torch.Tensor):
        hidden = F.relu(self.conv1(self.type_embedding(node_types), edges))
        return self.conv_mu(hidden, edges), self.conv_logstd(hidden, edges).clamp(-5, 3)

    def score(self, z: torch.Tensor, family_id: int, pairs: torch.Tensor) -> torch.Tensor:
        left = z[pairs[:, 0]] @ self.relations[family_id]
        return (left * z[pairs[:, 1]]).sum(-1)


def sample_negative_pairs(
    graph: TypedGraph, family: str, count: int, rng: np.random.Generator
) -> np.ndarray:
    pairs = family_pairs(graph, family)
    positive = {TypedGraph.canonical_edge(*edge) for edge in graph.family_edges[family]}
    mask = np.fromiter(
        (TypedGraph.canonical_edge(*pair) not in positive for pair in pairs),
        dtype=bool,
        count=len(pairs),
    )
    candidates = pairs[mask]
    if len(candidates) <= count:
        return candidates
    return candidates[rng.choice(len(candidates), size=count, replace=False)]


def train_vgae(graph: TypedGraph, args, device: torch.device):
    model = TypedVGAE(
        len(graph.type_names), len(graph.family_specs), args.hidden, args.latent
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    types = torch.as_tensor(graph.node_types, dtype=torch.long, device=device)
    edges = edge_index(graph, device)
    families = sorted(graph.family_specs)
    rng = np.random.default_rng(args.seed)
    for epoch in range(args.epochs):
        model.train()
        mu, logstd = model.encode(types, edges)
        z = mu + torch.randn_like(mu) * logstd.exp()
        losses = []
        for family_id, family in enumerate(families):
            positive = np.asarray(graph.family_edges[family], dtype=np.int64)
            negative = sample_negative_pairs(graph, family, len(positive), rng)
            pairs = np.row_stack((positive, negative))
            labels = torch.cat(
                (torch.ones(len(positive)), torch.zeros(len(negative)))
            ).to(device)
            logits = model.score(
                z, family_id, torch.as_tensor(pairs, dtype=torch.long, device=device)
            )
            losses.append(F.binary_cross_entropy_with_logits(logits, labels))
        kl = -0.5 * torch.mean(1 + 2 * logstd - mu.square() - (2 * logstd).exp())
        loss = torch.stack(losses).mean() + args.kl_weight * kl
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch == 0 or (epoch + 1) % args.log_every == 0:
            print(f"[VGAE] epoch={epoch + 1} loss={loss.item():.6f}", flush=True)
    return model


@torch.no_grad()
def vgae_scores(model: TypedVGAE, graph: TypedGraph, device: torch.device, chunk: int):
    model.eval()
    types = torch.as_tensor(graph.node_types, dtype=torch.long, device=device)
    mu, logstd = model.encode(types, edge_index(graph, device))
    z = mu + torch.randn_like(mu) * logstd.exp()
    scores = {}
    for family_id, family in enumerate(sorted(graph.family_specs)):
        pairs = family_pairs(graph, family)
        chunks = []
        for start in range(0, len(pairs), chunk):
            batch = torch.as_tensor(
                pairs[start : start + chunk], dtype=torch.long, device=device
            )
            chunks.append(model.score(z, family_id, batch).cpu())
        scores[family] = torch.cat(chunks).numpy()
    return scores


class DenseGraphVAE(nn.Module):
    def __init__(self, dimension: int, hidden: int, latent: int):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(dimension, hidden), nn.ReLU())
        self.mu = nn.Linear(hidden, latent)
        self.logstd = nn.Linear(hidden, latent)
        self.decoder = nn.Sequential(nn.Linear(latent, hidden), nn.ReLU(), nn.Linear(hidden, dimension))

    def forward(self, x: torch.Tensor):
        hidden = self.encoder(x)
        mu, logstd = self.mu(hidden), self.logstd(hidden).clamp(-5, 3)
        z = mu + torch.randn_like(mu) * logstd.exp()
        return self.decoder(z), mu, logstd


def legal_vector(graph: TypedGraph):
    families = sorted(graph.family_specs)
    pairs = {family: family_pairs(graph, family) for family in families}
    truth = []
    for family in families:
        positive = {TypedGraph.canonical_edge(*edge) for edge in graph.family_edges[family]}
        truth.extend(
            float(TypedGraph.canonical_edge(*pair) in positive) for pair in pairs[family]
        )
    return families, pairs, np.asarray(truth, dtype=np.float32)


def train_graphvae(graph: TypedGraph, args, device: torch.device):
    families, pairs, truth = legal_vector(graph)
    if len(truth) > args.graphvae_max_candidates:
        raise ValueError(
            f"GraphVAE-small candidate domain {len(truth):,} exceeds "
            f"--graphvae-max-candidates={args.graphvae_max_candidates:,}"
        )
    model = DenseGraphVAE(len(truth), args.hidden, args.latent).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    target = torch.as_tensor(truth, device=device).unsqueeze(0)
    positive = max(float(target.sum()), 1.0)
    pos_weight = torch.tensor((target.numel() - positive) / positive, device=device)
    for epoch in range(args.epochs):
        logits, mu, logstd = model(target)
        recon = F.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight)
        kl = -0.5 * torch.mean(1 + 2 * logstd - mu.square() - (2 * logstd).exp())
        loss = recon + args.kl_weight * kl
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch == 0 or (epoch + 1) % args.log_every == 0:
            print(f"[GraphVAE] epoch={epoch + 1} loss={loss.item():.6f}", flush=True)
    return model, families, pairs, target


@torch.no_grad()
def graphvae_scores(model, families, pairs, target, device):
    model.eval()
    logits, _, _ = model(target)
    values = logits[0].cpu().numpy()
    scores, offset = {}, 0
    for family in families:
        scores[family] = values[offset : offset + len(pairs[family])]
        offset += len(pairs[family])
    return scores


class TypedGraphRNN(nn.Module):
    """GraphRNN-S adaptation conditioned on the fixed node-type sequence."""

    def __init__(self, width: int, num_types: int, hidden: int):
        super().__init__()
        self.width = width
        self.gru = nn.GRU(width + num_types, hidden, batch_first=True)
        self.output = nn.Linear(hidden, width)

    def forward(self, previous_rows: torch.Tensor, type_onehot: torch.Tensor):
        hidden, _ = self.gru(torch.cat((previous_rows, type_onehot), dim=-1))
        return self.output(hidden)


def graph_rows(graph: TypedGraph, width: int):
    rows = np.zeros((graph.num_nodes, width), dtype=np.float32)
    for i in range(graph.num_nodes):
        for distance in range(1, min(width, i) + 1):
            rows[i, distance - 1] = float(graph.has_edge(i, i - distance))
    previous = np.zeros_like(rows)
    previous[1:] = rows[:-1]
    return previous, rows


def train_graphrnn(graph: TypedGraph, args, device: torch.device):
    width = min(args.max_prev_node, graph.num_nodes - 1)
    model = TypedGraphRNN(width, len(graph.type_names), args.hidden).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    previous, rows = graph_rows(graph, width)
    previous = torch.as_tensor(previous, device=device).unsqueeze(0)
    target = torch.as_tensor(rows, device=device).unsqueeze(0)
    types = F.one_hot(
        torch.as_tensor(graph.node_types, dtype=torch.long, device=device),
        num_classes=len(graph.type_names),
    ).float().unsqueeze(0)
    valid = torch.zeros_like(target)
    for i in range(graph.num_nodes):
        valid[:, i, : min(width, i)] = 1
    positive = max(float((target * valid).sum()), 1.0)
    pos_weight = torch.tensor((valid.sum().item() - positive) / positive, device=device)
    for epoch in range(args.epochs):
        logits = model(previous, types)
        raw = F.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight, reduction="none")
        loss = (raw * valid).sum() / valid.sum().clamp_min(1)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch == 0 or (epoch + 1) % args.log_every == 0:
            print(f"[GraphRNN] epoch={epoch + 1} loss={loss.item():.6f}", flush=True)
    return model, width


@torch.no_grad()
def graphrnn_scores(model, graph: TypedGraph, width: int, device: torch.device):
    model.eval()
    rows = torch.zeros(1, graph.num_nodes, width, device=device)
    previous = torch.zeros_like(rows)
    types = F.one_hot(
        torch.as_tensor(graph.node_types, dtype=torch.long, device=device),
        num_classes=len(graph.type_names),
    ).float().unsqueeze(0)
    hidden = None
    probabilities = np.full((graph.num_nodes, graph.num_nodes), -30.0, dtype=np.float32)
    for i in range(graph.num_nodes):
        step_input = torch.cat((previous[:, i : i + 1], types[:, i : i + 1]), dim=-1)
        encoded, hidden = model.gru(step_input, hidden)
        logits = model.output(encoded)[0, 0]
        for distance in range(1, min(width, i) + 1):
            j = i - distance
            probabilities[i, j] = probabilities[j, i] = float(logits[distance - 1])
        if i + 1 < graph.num_nodes:
            sampled = torch.bernoulli(torch.sigmoid(logits))
            previous[:, i + 1] = sampled
    return {
        family: np.asarray([probabilities[u, v] for u, v in family_pairs(graph, family)])
        for family in graph.family_specs
    }


def train_netgan(graph: TypedGraph, args, device: torch.device):
    root = Path(args.hg2np_root).resolve()
    sys.path.insert(0, str(root / "NetGAN"))
    try:
        import numba  # noqa: F401
    except ImportError:
        # HG2NP only uses numba as an optional decorator around its random-walk
        # routine.  Falling back to plain Python preserves transition semantics.
        def jit(*jit_args, **jit_kwargs):
            if jit_args and callable(jit_args[0]) and len(jit_args) == 1:
                return jit_args[0]
            return lambda function: function

        sys.modules["numba"] = types.SimpleNamespace(jit=jit)
    from Code.training import Trainer  # type: ignore

    matrix = sp.csr_matrix(
        (
            np.ones(2 * graph.num_edges),
            (
                [u for edges in graph.family_edges.values() for u, v in edges]
                + [v for edges in graph.family_edges.values() for u, v in edges],
                [v for edges in graph.family_edges.values() for u, v in edges]
                + [u for edges in graph.family_edges.values() for u, v in edges],
            ),
        ),
        shape=(graph.num_nodes, graph.num_nodes),
    )
    trainer = Trainer(
        matrix,
        graph.num_nodes,
        max_iterations=args.iterations,
        rw_len=args.walk_length,
        batch_size=args.batch_size,
        seed=args.seed,
        val_share=0.1,
        test_share=0.05,
        set_ops=False,
    )
    # Avoid expensive intermediate graph assembly during short reproducibility runs.
    trainer.train(
        create_graph_every=max(args.iterations, 1),
        plot_graph_every=0,
        num_samples_graph=args.walk_samples,
        stopping_criterion="val",
        max_patience=1,
    )
    transition = trainer.create_transition_matrix(args.walk_samples)
    dense = np.log1p(transition.toarray())
    return {
        family: np.asarray([dense[u, v] for u, v in family_pairs(graph, family)])
        for family in graph.family_specs
    }, trainer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=("netgan", "vgae", "graphrnn", "graphvae"), required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--latent", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--kl-weight", type=float, default=1e-3)
    parser.add_argument("--chunk-size", type=int, default=65536)
    parser.add_argument("--max-prev-node", type=int, default=256)
    parser.add_argument("--graphvae-max-candidates", type=int, default=50000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--walk-length", type=int, default=12)
    parser.add_argument("--walk-samples", type=int, default=50000)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument(
        "--hg2np-root",
        default="/data2/lyh/gnn_project/DiHug/5090server/HG2NP",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_all(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    graph = load_typed_graph(args.dataset_dir, active_only=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    protocol = {
        "method": args.method,
        "adaptation": "fixed node types; Binomial family counts; schema-valid edges",
        "dataset_dir": str(args.dataset_dir.resolve()),
        "num_nodes": graph.num_nodes,
        "num_edges": graph.num_edges,
        "seed": args.seed,
        "device": str(device),
    }
    if args.method == "vgae":
        model = train_vgae(graph, args, device)
        score_fn = lambda: vgae_scores(model, graph, device, args.chunk_size)
        torch.save(model.state_dict(), args.output_dir / "model.pt")
    elif args.method == "graphvae":
        model, families, pairs, target = train_graphvae(graph, args, device)
        score_fn = lambda: graphvae_scores(model, families, pairs, target, device)
        torch.save(model.state_dict(), args.output_dir / "model.pt")
        protocol["scope"] = "small-graph only"
    elif args.method == "graphrnn":
        model, width = train_graphrnn(graph, args, device)
        score_fn = lambda: graphrnn_scores(model, graph, width, device)
        torch.save(model.state_dict(), args.output_dir / "model.pt")
        protocol["variant"] = "GraphRNN-S typed adaptation"
        protocol["max_prev_node"] = width
    else:
        scores, trainer = train_netgan(graph, args, device)
        score_fn = lambda: scores
        torch.save(
            {"generator": trainer.generator.state_dict(), "discriminator": trainer.discriminator.state_dict()},
            args.output_dir / "model.pt",
        )
        protocol["implementation"] = "HG2NP/NetGAN/Code"

    rows = []
    for sample in range(args.samples):
        sample_rng = np.random.default_rng(args.seed * 100003 + sample)
        generated = select_family_edges(graph, score_fn(), sample_rng)
        path = args.output_dir / f"sample_{sample:03d}.json"
        path.write_text(json.dumps(generated.to_json_dict()), encoding="utf-8")
        row = compare_heterogeneous_graphs(graph, generated)
        row.update({"sample": sample, "path": str(path)})
        rows.append(row)
    report = {
        "format": "dihug-classic-heterogeneous-baseline-v1",
        "protocol": protocol,
        "rows": rows,
        "aggregate": aggregate_comparisons(rows),
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(report["aggregate"], indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
