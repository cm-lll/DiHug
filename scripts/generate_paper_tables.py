#!/usr/bin/env python3
"""Generate unified paper tables from frozen final results."""
import json, pathlib, csv, statistics, sys, hashlib, datetime, os

ROOT = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else pathlib.Path("results")
OUT = pathlib.Path("paper_tables")
OUT.mkdir(exist_ok=True)
COMMIT = sys.argv[2] if len(sys.argv) > 2 else "unknown"

def load_json(p):
    try: return json.loads(pathlib.Path(p).read_text())
    except: return None

main_candidates = []
if os.environ.get("MAIN_AGGREGATE"):
    main_candidates.append(pathlib.Path(os.environ["MAIN_AGGREGATE"]))
main_candidates.extend([
    ROOT / "paper_final_context_binomial_xiaob/aggregate.json",
    ROOT / "paper_final_context_xiaob/aggregate.json",
])
main_path = next((p for p in main_candidates if p.is_file()), None)
main_agg = load_json(main_path) if main_path is not None else None
cross_path = ROOT / "paper_cross_dataset_binomial_0717/aggregate.json"
cross_agg = load_json(cross_path)
family_count_mode = (
    str(main_agg.get("family_count_mode", "fixed")) if main_agg else "unknown"
)
generation_scope = (
    str(main_agg.get("generation_scope"))
    if main_agg and main_agg.get("generation_scope")
    else (
        "conditional-fixed-family-count"
        if family_count_mode == "fixed"
        else "unconditional-family-count"
        if family_count_mode == "binomial"
        else "unknown"
    )
)

# ═══════════════════════════════════════════
# Manifest
# ═══════════════════════════════════════════
manifest = {
    "commit": COMMIT,
    "generated": datetime.datetime.now().isoformat(),
    "datasets": {
        "imdb": {"nodes": 1653, "edges": 4079, "triangles": 655, "clustering": 0.4689},
        "pubmed": {"nodes": 1565, "edges": 12284, "triangles": 61137, "clustering": 0.335},
        "dblp": {"nodes": 11240, "edges": 36645, "triangles": 0, "clustering": 0},
        "syn100": {"nodes": 100, "edges": 400, "triangles": 211, "clustering": 0.1713353720410996},
        "syn200": {"nodes": 200, "edges": 900, "triangles": 261, "clustering": 0.08390400161414092},
        "syn500": {"nodes": 500, "edges": 2494, "triangles": 391, "clustering": 0.04438343405904855},
    },
    "protocol": {
        "family_count_mode": family_count_mode,
        "generation_scope": generation_scope,
        "denoiser": "learned-listwise-relocation-switch",
        "optional_residual": "frozen-rooted-context",
        "source_aggregate": str(main_path) if main_path is not None else None,
        "cross_dataset_aggregate": str(cross_path) if cross_agg else None,
    },
}
with open(OUT / "manifest.json", "w") as f:
    json.dump(manifest, f, indent=2)

# ═══════════════════════════════════════════
# Main fullgraph table
# ═══════════════════════════════════════════
rows = []
if main_agg:
    for g in main_agg.get("aggregates", []):
        ds = g["dataset"]
        real = manifest["datasets"].get(ds, {})
        rows.append({
            "dataset": ds,
            "method": g["mode"],
            "family_count_mode": family_count_mode,
            "generation_scope": generation_scope,
            "n": g["n"],
            "edges_mean": g["edges_mean"],
            "lcc_mean": g["lcc_mean"],
            "triangles_mean": g["triangles_mean"],
            "triangles_std": g["triangles_std"],
            "triangles_err": abs(g["triangles_mean"]-real.get("triangles",0))/max(real.get("triangles",1),1),
            "clustering_mean": g["clustering_mean"],
            "clustering_std": g["clustering_std"],
            "clustering_err": abs(g["clustering_mean"]-real.get("clustering",0.001))/max(real.get("clustering",0.001),0.001),
            "transitivity_mean": g["transitivity_mean"],
            "assortativity_mean": g["assortativity_mean"],
            "max_degree_mean": g["max_degree_mean"],
            "sum_degree_sq_mean": g["sum_degree_sq_mean"],
        })

if rows:
    with open(OUT / "main_fullgraph.tsv", "w") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), delimiter='\t')
        w.writeheader(); w.writerows(rows)

# Cross-dataset generalization table. Keep every frozen mode in the artifact;
# the paper can select legacy when legacy/context are exactly identical.
cross_rows = []
if cross_agg:
    for group in cross_agg.get("aggregates", []):
        row = {
            "dataset": group["dataset"],
            "method": group["mode"],
            "family_count_mode": "binomial",
            "n": group["n"],
        }
        for metric in (
            "edges", "lcc", "triangles", "clustering", "transitivity",
            "assortativity", "sum_degree_sq", "max_degree",
        ):
            real = float(group[f"real_{metric}"])
            final = float(group[f"final_{metric}_mean"])
            row[f"real_{metric}"] = real
            row[f"final_{metric}_mean"] = final
            row[f"final_{metric}_std"] = float(group[f"final_{metric}_std"])
            row[f"{metric}_relative_error"] = (
                abs(final - real) / abs(real) if abs(real) > 1.0e-12
                else abs(final - real)
            )
        cross_rows.append(row)

if cross_rows:
    with open(OUT / "cross_dataset.tsv", "w") as f:
        w = csv.DictWriter(f, fieldnames=list(cross_rows[0]), delimiter='\t')
        w.writeheader(); w.writerows(cross_rows)

# ═══════════════════════════════════════════
# Metric definitions
# ═══════════════════════════════════════════
metrics = {
    "Edges": "Total edge count",
    "LCC": "Largest connected component size",
    "TC": "Triangle count",
    "Clustering": "Average local clustering coefficient",
    "Transitivity": "3 × triangles / wedges",
    "Power-law": "Power-law coefficient of degree distribution",
    "Assortativity": "Degree assortativity coefficient",
    "Degree_MMD": "Maximum mean discrepancy of degree distributions",
    "EO": "Edge overlap with real graph (Jaccard)",
    "Uniqueness": "Fraction of generated edges not in training graph",
}
with open(OUT / "metric_definitions.json", "w") as f:
    json.dump(metrics, f, indent=2)

print(f"[PAPER-TABLES] wrote {len(rows)} rows to {OUT}/")
print(f"[PAPER-TABLES] wrote {len(cross_rows)} cross-dataset rows")
for r in rows:
    print(f"  {r['dataset']}/{r['method']}: tri={r['triangles_mean']:.0f}±{r['triangles_std']:.0f} err={r['triangles_err']:.3%} clust={r['clustering_mean']:.4f} err={r['clustering_err']:.3%}")
