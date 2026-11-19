"""Read-only CPU audit of archived edge flips; no model or fitting code.

Post-hoc explanatory metrics, NOT a new loss or an independent test set.
Uses the old stratified edge panel unchanged, plus separately labelled cases.
"""
import os
for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[name] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import argparse
import csv
import hashlib
import itertools
import json
import resource
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy.sparse.csgraph import connected_components

torch.set_num_threads(1)
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research_branches/hard_structure_rebar_cpu_20261108"))
sys.path.insert(0, str(ROOT / "research_branches/joint_transport_cpu_20261108"))
from objective import JointLocalScore
from transport import JointTransport

OLD = ROOT / "results/hard_structure_response_20261108T182154"
EDGE = ROOT / "results/hard_structure_edge_direction_cpu_20261108"
TRANSPORT = ROOT / "results/joint_transport_cpu_20261108"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def table(path, rows):
    if rows:
        with path.open("w") as f:
            w = csv.DictWriter(f, list(rows[0]))
            w.writeheader()
            w.writerows(rows)


def empirical_tv(x, y):
    a, b = Counter(map(tuple, x)), Counter(map(tuple, y))
    return sum(abs(a[k] - b[k]) for k in a.keys() | b.keys()) / (2 * len(x))


class AllocationMetrics:
    def __init__(self, core):
        self.core = core
        with torch.no_grad():
            self.ref = core.features(core.reference).numpy()
        self.nodes = {t: ix.numpy() for t, ix in core.nodes.items()}
        self.masks = {t: core.masks[t].numpy()[core.r:] for t in self.nodes}
        self.triples = list(itertools.combinations_with_replacement(core.type_values, 3))
        self.triple_index = {v: i for i, v in enumerate(self.triples)}
        self.ref_composition = self.triangle_counts(self.ref)

    def triangle_counts(self, raw):
        counts = np.zeros(len(self.triples))
        for t, ix in self.nodes.items():
            for j, pair in enumerate(self.core.triangle_pairs):
                counts[self.triple_index[tuple(sorted((t, *pair)))]] += raw[ix, self.core.r + j].sum() / 3
        assert np.max(abs(counts - np.round(counts))) < 1e-9
        return counts

    def evaluate(self, raw):
        parts = []
        for t, ix in self.nodes.items():
            x, y = raw[ix], self.ref[ix]
            tri_x, tri_y = x[:, self.core.r:], y[:, self.core.r:]
            a, b = tri_x.sum(1), tri_y.sum(1)
            mask = self.masks[t]
            parts.append(dict(node_type=int(t), nodes=len(ix),
                node_triangle_W1=float(np.mean(abs(np.sort(a) - np.sort(b)))),
                typed_node_triangle_W1=float(np.mean(abs(np.sort(tri_x[:, mask], axis=0) - np.sort(tri_y[:, mask], axis=0)))) if mask.any() else 0.,
                joint_role_TV=empirical_tv(x, y),
                max_node_triangle_abs_error=float(abs(a.max() - b.max()))))
        result = {k: float(np.mean([p[k] for p in parts])) for k in (
            "node_triangle_W1", "typed_node_triangle_W1", "joint_role_TV", "max_node_triangle_abs_error")}
        c, r = self.triangle_counts(raw), self.ref_composition
        result["triangle_composition_TV"] = float(abs(c / c.sum() - r / r.sum()).sum() / 2) if c.sum() and r.sum() else None
        result["triangle_abs_error"] = float(abs(c.sum() - r.sum()))
        return result, parts


def mmd_parts(core, raw):
    rows = []
    for t, indices in core.nodes.items():
        with torch.no_grad():
            x = torch.from_numpy(raw[indices.numpy()][:, core.masks[t].numpy()])
            x = torch.log1p(x) / core.scales[t]
            xx = float(core.kernel(x, x).mean())
            xy = float(core.kernel(x, core.targets[t]).mean())
            yy = float(core.target_self[t])
        rows.append(dict(node_type=int(t), within=xx, cross=-2 * xy, target=yy, total=xx - 2 * xy + yy))
    return rows


def connectivity(a):
    n, lab = connected_components(a, directed=False)
    return dict(lcc=int(np.bincount(lab).max()), components=int(n), isolates=int((a.sum(1) == 0).sum()))


def independent_features(types, q, f, b, r, pairs):
    n = len(types)
    raw = np.zeros((n, r + len(pairs)))
    a = np.zeros((n, n), dtype=bool)
    for e, (i, j) in enumerate(q.T):
        if b[e]:
            a[i, j] = a[j, i] = True
            raw[i, f[e]] += 1
            raw[j, f[e]] += 1
    lookup = {v: r + k for k, v in enumerate(pairs)}
    for i, j, k in itertools.combinations(range(n), 3):
        if a[i, j] and a[i, k] and a[j, k]:
            for center, u, v in ((i, j, k), (j, i, k), (k, i, j)):
                raw[center, lookup[tuple(sorted((types[u], types[v])))]] += 1
    return raw


def main(out):
    out.mkdir(parents=True, exist_ok=False)
    start = time.time()
    bound = {}
    def bind(path):
        bound[str(path)] = sha(path)
        return path
    def read(path):
        return json.loads(bind(path).read_text())
    def write(name, obj):
        (out / name).write_text(json.dumps(obj, indent=2, allow_nan=False))
    protocol = dict(scope="posthoc role allocation audit, no candidate fitting or new score selection",
        source_edges=str(TRANSPORT / "selected_edges.csv"), selected_edges=21826,
        graphs=96, methods=["MMD", "JOINT_W1"], synthetic_graphs=64,
        all_historical_cases_separate=True, score_tolerance=1e-12,
        metric_tolerance=1e-12, elapsed_limit_seconds=600, rss_limit_MiB=2048,
        GPU=0, model_calls=0, optimizer_updates=0, new_panel_samples=0,
        metrics="type-equal raw-count triangle W1, active typed triangle marginal W1, joint degree+triangle TV, maximum participation error, composition TV; connectivity on historic cases only",
        warning="new metrics are retrospective explanations; no significance or new training gate")
    write("protocol.json", protocol)
    checks = 0
    # Exhaustive four-node feature/flip and identity/relabel checks.
    types = np.array([0, 0, 1, 1]); q = np.array(list(itertools.combinations(range(4), 2))).T
    f = types[q[0]] + types[q[1]]; y = np.array([1, 1, 0, 1, 0, 1])
    core = JointLocalScore(types, q, f, y, 3); metric = AllocationMetrics(core)
    for code in range(64):
        bits = ((code >> np.arange(6)) & 1).astype(float)
        transport = JointTransport(core, bits)
        assert np.array_equal(transport.raw, independent_features(types, q, f, bits, 3, core.triangle_pairs))
        before, _ = metric.evaluate(transport.raw)
        relabel = transport.raw[[1, 0, 3, 2]]
        assert metric.evaluate(relabel)[0] == before
        for e in range(6):
            flipped, _ = transport.flipped_raw(e)
            nb = bits.copy(); nb[e] = 1 - nb[e]
            assert np.array_equal(flipped, independent_features(types, q, f, nb, 3, core.triangle_pairs))
            checks += 1
    zero, _ = metric.evaluate(metric.ref)
    assert all(v == 0 for v in zero.values())
    write("tests.json", dict(graphs=64, flip_checks=checks, identity=True, within_type_relabel=True))
    selected = list(csv.DictReader(bind(TRANSPORT / "selected_edges.csv").open()))
    cases = list(csv.DictReader(bind(TRANSPORT / "old_case_retest.csv").open()))
    graph_rows = list(csv.DictReader(bind(EDGE / "graph_summary.csv").open()))
    tasks = read(ROOT / "results/output_boundary_replay_20261108T081654/tasks.json")
    groups, casegroups = defaultdict(list), defaultdict(list)
    keyfields = ("condition", "arm", "t", "draw")
    for row in selected: groups[tuple(row[k] for k in keyfields)].append(row)
    for row in cases: casegroups[tuple(row[k] for k in keyfields)].append(row)
    assert len(selected) == 21826 and len(groups) == 96
    datasets = {}
    for ds in ("IMDB", "PubMed"):
        cfg = next(x for x in tasks if x['dataset'] == ds)['row']['config']['datasets'][ds]
        p = ROOT / 'research_branches/typed_structure_early_cn_expansion_review/experiments/typed_structure_early_cn_expansion_v1/package' / cfg['file']
        assert sha(bind(p)) == cfg['sha256']
        with np.load(p) as z: types = z['types'].copy()
        fixed = torch.load(bind(OLD / f'{ds}_fixed_inputs.pt'), map_location='cpu')['inputs']
        datasets[ds] = (types, {str(t): inp for t, seed, inp in fixed})
    result, case_result, details, kernel_rows, bases = [], [], [], [], []
    realchecks = 0
    for g in graph_rows:
        if time.time() - start > 600 or resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 > 2048:
            raise RuntimeError('CPU audit resource limit')
        key = tuple(g[k] for k in keyfields); ds = key[0].split('_')[0]
        types, inputs = datasets[ds]; inp = inputs[key[2]]
        q, f, y = map(np.asarray, (inp['query'], inp['f'], inp['labels']))
        core = JointLocalScore(types, q, f, y, int(f.max()) + 1)
        metric = AllocationMetrics(core)
        flipfile = EDGE / f'{key[0]}_{key[1]}_t{key[2]}_draw{key[3]}_flips.npz'
        with np.load(bind(flipfile)) as z:
            bits = (1 - z['flip_sign']) / 2
        transport = JointTransport(core, bits)
        base, baseparts = metric.evaluate(transport.raw)
        ids = dict(zip(keyfields, key)); ids['dataset'] = ds
        assert abs(base['triangle_abs_error'] - abs(float(g['triangles']) - float(g['reference_triangles']))) < 1e-9
        bases.append(dict(**ids, **base))
        selected_slots = {int(row['slot']) for row in groups[key]}
        case_slots = {int(row['slot']) for row in casegroups[key]}
        cache = {}
        for slot in selected_slots | case_slots:
            raw, changed = transport.flipped_raw(slot)
            value, parts = metric.evaluate(raw)
            changes = {k + '_flip': (None if base[k] is None or value[k] is None else value[k] - base[k]) for k in base}
            cache[slot] = changes
        for row in groups[key]:
            delta = cache[int(row['slot'])]
            assert abs(delta['triangle_abs_error_flip'] - float(row['triangle_abs_error_flip'])) < 1e-8
            result.append(dict(**row, dataset=ds, **{k:v for k,v in delta.items() if k != 'triangle_abs_error_flip'}))
        # Same pre-existing examples: diagnostic attribution, not prevalence estimates.
        old_kernel = mmd_parts(core, transport.raw)
        assert abs(np.mean([v['total'] for v in old_kernel]) - float(g['score'])) < 1e-11
        ref_a = np.zeros_like(transport.a); ref_a[q[0], q[1]] = y; ref_a[q[1], q[0]] = y
        ref_conn, old_conn = connectivity(ref_a), connectivity(transport.a)
        for row in casegroups[key]:
            slot = int(row['slot']); raw, _ = transport.flipped_raw(slot)
            value, parts = metric.evaluate(raw); new_kernel = mmd_parts(core, raw)
            actual_delta = np.mean([v['total'] - u['total'] for u, v in zip(old_kernel, new_kernel)])
            assert abs(actual_delta - float(row['score_flip'])) < 1e-11
            a = transport.a.copy(); i, j = q[:, slot]; a[i, j] = a[j, i] = not a[i, j]
            conn = connectivity(a)
            conn_delta = {k + '_abs_error_flip': abs(conn[k] - ref_conn[k]) - abs(old_conn[k] - ref_conn[k]) for k in conn}
            case_result.append(dict(**row, dataset=ds, **cache[slot], **conn_delta))
            for bp, vp, bk, vk in zip(baseparts, parts, old_kernel, new_kernel):
                details.append(dict(**ids, kind=row['kind'], slot=slot, node_type=bp['node_type'], nodes=bp['nodes'],
                    **{k + '_before': bp[k] for k in bp if k not in ('node_type', 'nodes')},
                    **{k + '_after': vp[k] for k in vp if k not in ('node_type', 'nodes')}))
                kernel_rows.append(dict(**ids, kind=row['kind'], slot=slot, node_type=bk['node_type'],
                    within_delta=vk['within']-bk['within'], cross_delta=vk['cross']-bk['cross'], total_delta=vk['total']-bk['total']))
        # Full raw feature recomputation for two fixed slot indices per graph.
        for slot in (0, len(bits)-1):
            raw, _ = transport.flipped_raw(slot); nb = bits.copy(); nb[slot] = 1-nb[slot]
            with torch.no_grad(): expected = core.features(torch.from_numpy(nb)).numpy()
            assert np.array_equal(raw, expected)
            realchecks += 1
        print(json.dumps(dict(graph=len(bases), total=96, seconds=time.time()-start)), flush=True)
    table(out / 'selected_edge_metrics.csv', result)
    table(out / 'historical_cases.csv', case_result)
    table(out / 'case_type_metrics.csv', details)
    table(out / 'case_kernel_decomposition.csv', kernel_rows)
    table(out / 'graph_metrics.csv', bases)
    write('asset_binding.json', bound)
    assert all(sha(p) == h for p, h in bound.items())
    assert not torch.cuda.is_initialized()
    write('receipt.json', dict(status='complete', graphs=len(bases), selected_edges=len(result),
        historical_case_rows=len(case_result), real_full_feature_checks=realchecks, assets_unchanged=True,
        seconds=time.time()-start, peak_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        GPU=0, model_calls=0, optimizer_updates=0, new_panel_samples=0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    main(args.output)
