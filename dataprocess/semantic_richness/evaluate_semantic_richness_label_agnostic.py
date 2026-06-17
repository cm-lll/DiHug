# evaluate_semantic_richness_label_agnostic.py (v4)
#
# Purpose: show that coherent subtypes carry structure beyond frequency-matched permutations.
# Default: compare ONLY structured vs randomized (permuted) on the same topology:
#   structured.json  : node.subtype + edge.subtype aligned with real co-occurrence
#   randomized.json  : same subtype marginals but permuted -> breaks label–structure map
# Optional third file raw.json (only node.type / edge.type) via --versions raw,structured,randomized
#
# Metrics are label-agnostic and give RAW non-zero scores, while rewarding coherent subtypes:
#   - SchemaRichness: log(#node_types + #edge_types) at coarse level (unchanged under subtype permute)
#   - HierarchyDepth: 1 (types only) / 2 (+node subtypes) / 3 (+edge subtypes)
#   - MI(label, degree | type): within-type correlation between label and degree
#   - ConnSigDiv: divergence of neighbor-LABEL signatures across labels
#   - EdgePred(edge-label | type,label): predictability of edge-label given source label and type
#   - TripleSparsity: sparsity of (label_src, edge_label, label_dst) space
#
# Run:
#   python evaluate_semantic_richness_label_agnostic.py --dir <DIR> --out out
#   python ... --dir <DIR> --out out --versions raw,structured,randomized   # legacy 3-way

import argparse, json, math, os
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_graph(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def entropy_from_counts(counts: Counter) -> float:
    tot = sum(counts.values())
    if tot == 0:
        return 0.0
    H = 0.0
    for c in counts.values():
        p = c / tot
        if p > 0:
            H -= p * math.log(p + 1e-12)
    return H


def mutual_information(x, y) -> float:
    joint = Counter(zip(x, y))
    px = Counter(x)
    py = Counter(y)
    n = len(x)
    mi = 0.0
    for (a, b), c in joint.items():
        p_ab = c / n
        p_a = px[a] / n
        p_b = py[b] / n
        mi += p_ab * math.log((p_ab + 1e-12) / (p_a * p_b + 1e-12))
    return mi


def node_label(n: dict) -> str:
    st = n.get("subtype")
    return st if st is not None else n["type"]


def edge_label(e: dict) -> str:
    st = e.get("subtype")
    return st if st is not None else e["type"]


def hierarchy_depth(nodes, edges) -> int:
    d = 1
    if any(n.get("subtype") is not None for n in nodes):
        d += 1
    if any(e.get("subtype") is not None for e in edges):
        d += 1
    return d


def schema_richness(nodes, edges) -> float:
    T = len({n["type"] for n in nodes})
    R = len({e["type"] for e in edges})
    return float(math.log(T + R + 1e-12))


def mi_label_degree_within_type(nodes, edges) -> float:
    out_deg = Counter(e["src"] for e in edges)
    by_type = defaultdict(list)
    for n in nodes:
        by_type[n["type"]].append(n)

    total = len(nodes)
    mi_sum = 0.0
    for _, ns in by_type.items():
        labs = [node_label(n) for n in ns]
        deg_bins = []
        for n in ns:
            d = out_deg.get(n["id"], 0)
            deg_bins.append(d if d <= 10 else 11)
        mi_sum += mutual_information(labs, deg_bins) * (len(ns) / total)
    return float(mi_sum)


def js_divergence(p, q) -> float:
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    p = p / (p.sum() + 1e-12)
    q = q / (q.sum() + 1e-12)
    m = 0.5 * (p + q)

    def kl(a, b):
        a = np.clip(a, 1e-12, None)
        b = np.clip(b, 1e-12, None)
        return float(np.sum(a * np.log(a / b)))

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def connectivity_signature_divergence(nodes, edges) -> float:
    labels = sorted({node_label(n) for n in nodes})
    idx = {l: i for i, l in enumerate(labels)}

    out_nei = defaultdict(list)
    for e in edges:
        out_nei[e["src"]].append(e["dst"])

    id2lab = {n["id"]: node_label(n) for n in nodes}

    group = defaultdict(list)
    for n in nodes:
        group[node_label(n)].append(n["id"])

    sig = {}
    for lab, vids in group.items():
        acc = np.zeros(len(labels), dtype=float)
        cnt = 0
        for v in vids:
            neigh = out_nei.get(v, [])
            if not neigh:
                continue
            h = np.zeros(len(labels), dtype=float)
            for u in neigh:
                h[idx[id2lab[u]]] += 1
            h /= (h.sum() + 1e-12)
            acc += h
            cnt += 1
        sig[lab] = (acc / cnt) if cnt > 0 else (np.ones(len(labels)) / len(labels))

    labs = list(sig.keys())
    if len(labs) < 2:
        return 0.0

    dists = []
    for i in range(len(labs)):
        for j in range(i + 1, len(labs)):
            dists.append(js_divergence(sig[labs[i]], sig[labs[j]]))
    return float(np.mean(dists)) if dists else 0.0


def edge_label_predictability_within_type(nodes, edges) -> float:
    rel_labels = sorted({edge_label(e) for e in edges})
    if not rel_labels:
        return 0.0
    denom = math.log(len(rel_labels) + 1e-12)

    id2type = {n["id"]: n["type"] for n in nodes}
    id2lab = {n["id"]: node_label(n) for n in nodes}

    counts = defaultdict(Counter)
    weight = Counter()
    for e in edges:
        key = (id2type[e["src"]], id2lab[e["src"]])
        counts[key][edge_label(e)] += 1
        weight[key] += 1

    total = sum(weight.values())
    avg_norm_entropy = 0.0
    for key, w in weight.items():
        H = entropy_from_counts(counts[key])
        avg_norm_entropy += (w / total) * (H / denom if denom > 0 else 0.0)

    return float(max(0.0, 1.0 - avg_norm_entropy))


def typed_triple_sparsity(nodes, edges) -> float:
    labels = sorted({node_label(n) for n in nodes})
    rels = sorted({edge_label(e) for e in edges})
    if not labels or not rels:
        return 0.0

    id2lab = {n["id"]: node_label(n) for n in nodes}
    obs = set()
    for e in edges:
        obs.add((id2lab[e["src"]], edge_label(e), id2lab[e["dst"]]))

    total = len(labels) * len(rels) * len(labels)
    return float(max(0.0, 1.0 - len(obs) / total))


ALL_AXES = [
    "SchemaRichness",
    "HierarchyDepth",
    "MI_Label_Degree_WithinType",
    "ConnSignatureDiv",
    "EdgeLabelPredictability_WithinType",
    "TypedTripleSparsity",
]

# For structured vs permuted with same coarse types and same subtype *presence*, Schema + Depth are
# identical across versions — they carry no discriminative signal. Radar + normalized CSV then use only:
DISCRIMINATIVE_AXES = [
    "MI_Label_Degree_WithinType",
    "ConnSignatureDiv",
    "EdgeLabelPredictability_WithinType",
    "TypedTripleSparsity",
]

RADAR_LABELS_FULL = [
    "Schema",
    "Depth",
    "MI(label,deg|type)",
    "ConnSigDiv",
    "EdgePred(edge-label|type)",
    "TripleSparsity",
]

RADAR_LABELS_DISC = [
    "MI(label,deg|type)",
    "ConnSigDiv",
    "EdgePred(edge-label|type)",
    "TripleSparsity",
]


def compute_metrics(path) -> dict:
    g = load_graph(path)
    nodes, edges = g["nodes"], g["edges"]
    return {
        "SchemaRichness": schema_richness(nodes, edges),
        "HierarchyDepth": hierarchy_depth(nodes, edges),
        "MI_Label_Degree_WithinType": mi_label_degree_within_type(nodes, edges),
        "ConnSignatureDiv": connectivity_signature_divergence(nodes, edges),
        "EdgeLabelPredictability_WithinType": edge_label_predictability_within_type(nodes, edges),
        "TypedTripleSparsity": typed_triple_sparsity(nodes, edges),
    }


def select_plot_axes(metrics: dict) -> tuple[list, list]:
    """Return (dataframe column names, polar tick labels)."""
    keys = list(metrics.keys())
    ref = keys[0]
    schema_depth_constant = all(
        abs(metrics[k]["SchemaRichness"] - metrics[ref]["SchemaRichness"]) < 1e-9
        and metrics[k]["HierarchyDepth"] == metrics[ref]["HierarchyDepth"]
        for k in keys
    )
    if schema_depth_constant:
        return DISCRIMINATIVE_AXES, RADAR_LABELS_DISC
    return ALL_AXES, RADAR_LABELS_FULL


def normalize(metrics: dict, axes: list) -> pd.DataFrame:
    maxv = {a: max(metrics[k][a] for k in metrics) or 1.0 for a in axes}

    norm = {}
    for k in metrics:
        row = {}
        for a in axes:
            if a == "HierarchyDepth":
                row[a] = metrics[k]["HierarchyDepth"] / 3.0
            else:
                row[a] = metrics[k][a] / maxv[a]
        norm[k] = row

    return pd.DataFrame(norm).T[axes]


def draw_radar(df: pd.DataFrame, out_png: str, version_order: list, radar_labels: list):
    axes = list(df.columns)
    values = {k: df.loc[k, axes].values.tolist() for k in df.index}

    N = len(radar_labels)
    ang = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    ang += ang[:1]

    plt.figure(figsize=(7, 7))
    ax = plt.subplot(111, polar=True)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(ang[:-1]), radar_labels)
    ax.set_ylim(0, 1.05)

    for name in version_order:
        if name not in values:
            continue
        v = values[name] + values[name][:1]
        ax.plot(ang, v, linewidth=2, label=name)
        ax.fill(ang, v, alpha=0.15)

    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1))
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="Directory containing <stem>.json for each --versions stem")
    ap.add_argument("--out", default="out", help="Output directory")
    ap.add_argument(
        "--versions",
        default="structured,randomized",
        help="Comma-separated JSON stems to load, plot, and use for per-axis max normalization "
        "(default: structured,randomized). Use raw,structured,randomized for three-way.",
    )
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    versions = [v.strip() for v in args.versions.split(",") if v.strip()]
    if len(versions) < 2:
        raise SystemExit("--versions must list at least two stems")

    metrics = {}
    for v in versions:
        path = os.path.join(args.dir, f"{v}.json")
        if not os.path.isfile(path):
            raise SystemExit(f"missing {path}")
        metrics[v] = compute_metrics(path)

    plot_axes, radar_labels = select_plot_axes(metrics)
    df = normalize(metrics, plot_axes)
    df = df.reindex([v for v in versions if v in df.index])
    df.to_csv(os.path.join(args.out, "metrics_normalized.csv"))

    with open(os.path.join(args.out, "metrics_raw.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    draw_radar(df, os.path.join(args.out, "radar.png"), version_order=versions, radar_labels=radar_labels)

    print("Wrote:", os.path.join(args.out, "metrics_raw.json"))
    print("Wrote:", os.path.join(args.out, "metrics_normalized.csv"))
    print("Wrote:", os.path.join(args.out, "radar.png"))
    print("versions:", versions)
    print("plot_axes:", plot_axes)


if __name__ == "__main__":
    main()
