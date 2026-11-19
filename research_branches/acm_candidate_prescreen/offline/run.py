"""Hold out whole archived paths for budgeted ACM candidate reranking."""

import csv
import hashlib
import json
import os
from collections import defaultdict
from itertools import combinations
from math import comb
from pathlib import Path
from time import monotonic

import numpy as np
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits


SOURCE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get('DIHUG_ACM_ARCHIVE_ROOT', SOURCE.parents[2] / 'results')).resolve()
OUTPUT_ROOT = Path(os.environ.get('DIHUG_ACM_OUTPUT_ROOT', SOURCE.parent / 'runs')).resolve()
OUT = OUTPUT_ROOT / 'offline'
ARCHIVE = ROOT / 'acm_frozen_participation_replay_20261118'
REFERENCE = ROOT / 'typed_edit_complexity_20261116/data/acm/graph.json'
PANELS = ('pilot', 'bank4')
BUDGETS = (1, 2, 4, 8)
WITNESSES = 4
SEED = 20261119
EPS = 1e-12


def archive_path(saved):
    """Resolve a historical manifest path relative to the supplied archive root."""
    parts = Path(saved).parts
    if 'results' not in parts:
        raise ValueError(f'Expected a historical path under results/: {saved}')
    relative = parts[parts.index('results') + 1:]
    if not relative or '..' in relative:
        raise ValueError(f'Invalid historical archive path: {saved}')
    return ROOT.joinpath(*relative)


def archive_hash(files, current):
    for saved, digest in files.items():
        if 'results' in Path(saved).parts and archive_path(saved) == current:
            return digest
    raise KeyError(f'No hash recorded for {current.relative_to(ROOT)}')


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def table(name, rows):
    with (OUT / name).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def edge(value):
    u, v = value
    return (min(u, v), max(u, v))


def action_key(values, i):
    return (int(values['family'][i]),
            tuple(sorted(edge(v) for v in values['old_edges'][i])),
            tuple(sorted(edge(v) for v in values['new_edges'][i])))


class ArchivedState:
    """Read the current neighborhood from the original recorded edits only."""

    def __init__(self, sets, node_count):
        self.edges = [s.copy() for s in sets]
        self.neighbors = [[] for _ in range(node_count)]
        self.positions = [{} for _ in range(node_count)]
        for group in self.edges:
            for u, v in sorted(group):
                self.add(u, v)
                self.add(v, u)

    def add(self, u, v):
        assert v not in self.positions[u]
        self.positions[u][v] = len(self.neighbors[u])
        self.neighbors[u].append(v)

    def remove(self, u, v):
        index = self.positions[u].pop(v)
        last = self.neighbors[u].pop()
        if last != v:
            self.neighbors[u][index] = last
            self.positions[u][last] = index

    def observe(self, values):
        family, old, new = action_key(values, 0)
        assert values['action'] == [4] and sorted(x for e in old for x in e) == sorted(x for e in new for x in e)
        for u, v in old:
            self.edges[family].remove((u, v))
            self.remove(u, v)
            self.remove(v, u)
        for u, v in new:
            self.edges[family].add((u, v))
            self.add(u, v)
            self.add(v, u)

    def witness_sample(self, node, rng):
        neighbors = self.neighbors[node]
        if len(neighbors) <= WITNESSES:
            return neighbors
        return [neighbors[i] for i in rng.choice(len(neighbors), WITNESSES, replace=False)]


class Schema:
    def __init__(self, reference, columns, target):
        self.types = np.asarray(reference['node_types'])
        self.n_types = len(reference['type_names'])
        self.target = target
        self.families = sorted(reference['families'], key=lambda row: row['name'])
        self.sets = [{edge(e) for e in row['edges']} for row in self.families]
        self.family_count = len(self.families)
        self.degree = np.zeros((len(self.types), self.family_count), dtype=int)
        for i, group in enumerate(self.sets):
            for u, v in group:
                self.degree[u, i] += 1
                self.degree[v, i] += 1
        self.total = self.degree.sum(axis=1)
        tri_cols = [name for kind, name in columns if kind == 'typed_triangles']
        self.triangle_count = len(tri_cols)
        self.triangle_positions = {tuple(map(int, name.split(','))): i for i, name in enumerate(tri_cols)}
        n, k = self.n_types, self.triangle_count
        self.witness_to_motif = np.full((n, n, n), -1, dtype=int)
        for a in range(n):
            for b in range(n):
                self.witness_to_motif[a, b] = [self.triangle_positions.get(tuple(sorted((a, b, c))), -1) for c in range(n)]
        self.support = np.zeros((self.family_count, len(target)), dtype=bool)
        self.half_denominator = np.repeat([max(1, target[:k].sum()), max(1, target[k:].sum())], [k, len(target) - k])
        for j, family in enumerate(self.families):
            motif_ids = self.witness_to_motif[family['src_type'], family['dst_type']]
            self.support[j, motif_ids[motif_ids >= 0]] = True
            self.support[j, k:] = [name == family['name'] for _, name in columns[k:]]
        self.relation_onehot = np.eye(self.family_count)

    def closure_sketch(self, state, endpoints, query_edges, samples):
        result = np.zeros(self.triangle_count)
        for u, v in query_edges:
            positions = self.witness_to_motif[self.types[u], self.types[v]]
            for start, other in ((u, v), (v, u)):
                take = samples[start]
                weight = self.total[start] / (2 * max(1, len(take)))
                possible = state.positions[other]
                for w in take:
                    motif = positions[self.types[w]]
                    if motif >= 0 and w not in endpoints and w in possible:
                        result[motif] += weight
        return result

    def proposal_features(self, state, key, samples, residual, urgency):
        family, old, new = key
        endpoints = tuple(sorted({u for pair in old for u in pair}))
        assert len(endpoints) == 4
        removed = self.closure_sketch(state, endpoints, old, samples)
        added = self.closure_sketch(state, endpoints, new, samples)
        difference = added - removed
        k = self.triangle_count
        before = residual[:k]
        proxy_by_type = (np.abs(before) - np.abs(before + difference)) / self.half_denominator[:k]
        degrees = np.log1p(self.degree[list(endpoints)])
        totals = np.sort(np.log1p(self.total[list(endpoints)]))
        internal = sum(v in state.positions[u] for u, v in combinations(endpoints, 2))
        vector = np.concatenate((self.relation_onehot[family], np.sort(degrees, axis=0).ravel(), totals,
                                 [internal, urgency[family]], residual / self.half_denominator,
                                 removed, added, proxy_by_type, [proxy_by_type.sum()]))
        return vector, float(proxy_by_type.sum())

    def gain(self, residual, delta):
        return ((np.abs(residual) - np.abs(residual + delta)) / self.half_denominator).sum(axis=-1) / 2


def prepare(panel, reference, rng):
    folder = ARCHIVE / panel
    inv, manifest = (read(folder / name) for name in ('inventory.json', 'manifest.json'))
    assert sha(folder / 'inventory.json') == manifest['inventory']
    assert sha(folder / 'target.npz') == manifest['target_sha']
    assert not inv['excluded_rows'] and not inv['repeats_within_panel']
    encoded = archive_path(manifest['folder'])
    for name in ('data.npz', 'receipt.json'):
        assert sha(encoded / name) == manifest['assets'][name]
    data_hashes = read(encoded / 'receipt.json')['inputs']
    raw = archive_path(inv['raw'])
    assert sha(raw / 'acceptance.json') == manifest['raw_acceptance']
    assert sha(REFERENCE) == archive_hash(read(raw / 'manifest.json')['files'], REFERENCE)
    with np.load(encoded / 'data.npz') as data:
        groups, counts, deltas, labels, refcounts = [data[name] for name in ('groups', 'counts', 'delta', 'y', 'refcounts')]
    with np.load(folder / 'target.npz') as archive:
        actions, allowed = archive['actions'], archive['allowed']
    schema = Schema(reference, manifest['columns'], refcounts)
    rows = []
    duplicates = injected = 0
    for trajectory in sorted((raw / 'trajectories').iterdir()):
        expected = data_hashes[trajectory.name][1]
        assert sha(trajectory / 'record.json') == expected
        path = read(trajectory / 'record.json')
        registered_steps = {r['step']: r for r in path['records']}
        state = ArchivedState(schema.sets, len(schema.types))
        for step in path['path']:
            state.observe(step['edit'])
            step_number = step['step']
            if step_number not in registered_steps:
                continue
            record = registered_steps[step_number]
            index = len(rows)
            assert groups[index] == trajectory.name and labels[index] == record['label']
            values = record['candidates']
            assert np.array_equal(actions[index], values['action'])
            blocked = action_key(values, labels[index])
            unique = {}
            for slot, code in enumerate(values['action']):
                if code == 0:
                    continue
                key = action_key(values, slot)
                if key == blocked:
                    injected += 1
                    continue
                if key in unique:
                    prior = unique[key]
                    assert allowed[index, slot] == allowed[index, prior]
                    assert np.array_equal(deltas[index, slot], deltas[index, prior])
                    duplicates += 1
                    continue
                unique[key] = slot
            assert len(unique) >= max(BUDGETS), (panel, trajectory.name, step_number, len(unique))
            nodes = {v for _, old, _ in unique for pair in old for v in pair}
            samples = {u: state.witness_sample(u, rng) for u in sorted(nodes)}
            residual = counts[index] - refcounts
            urgency = schema.support @ (np.abs(residual) / schema.half_denominator)
            features, proxy, types = [], [], []
            slots = list(unique.values())
            for key in unique:
                vector, guess = schema.proposal_features(state, key, samples, residual, urgency)
                features.append(vector)
                proxy.append(guess)
                types.append(key[0])
            true = schema.gain(residual, deltas[index, slots])
            rows.append(dict(panel=panel, path=trajectory.name, step=step_number, connected=record['components'] == 1,
                             slots=slots, x=np.asarray(features), proxy=np.asarray(proxy), types=np.asarray(types),
                             urgency=urgency, y=np.where(allowed[index, slots], true, 0),
                             allowed=allowed[index, slots], tie=rng.permutation(len(slots))))
        end_families = sorted(path['final_graph']['families'], key=lambda item: item['name'])
        assert all(group == {edge(e) for e in end['edges']} for group, end in zip(state.edges, end_families))
    assert len(rows) == len(groups) == len(inv['states']) == 1664
    assert len(inv['paths']) == 19
    assert all(np.isfinite(r['x']).all() and np.isfinite(r['y']).all() for r in rows)
    return rows, inv, dict(rows=len(rows), paths=len(inv['paths']), duplicate_slots_removed=duplicates,
                           injected_slots_removed=injected, fewest_candidates=min(len(r['y']) for r in rows),
                           most_candidates=max(len(r['y']) for r in rows),
                           mean_candidates=float(np.mean([len(r['y']) for r in rows])),
                           connected=sum(r['connected'] for r in rows), features=rows[0]['x'].shape[1],
                           archive_manifest_sha256=sha(folder / 'manifest.json'))


def rank(score, tie):
    return np.lexsort((tie, -score))


def quota_order(proxy_order, relation, priorities):
    queues = defaultdict(list)
    for index in proxy_order:
        queues[relation[index]].append(index)
    served = dict.fromkeys(queues, 0)
    output = []
    for _ in range(max(BUDGETS)):
        f = max(queues, key=lambda key: priorities[key] / (1 + served[key]))
        output.append(queues[f].pop(0))
        served[f] += 1
        if not queues[f]:
            del queues[f]
    return np.asarray(output)


def random_outcome(gain, k):
    sorted_gain = np.sort(gain)
    n = len(gain)
    denominator = comb(n, k)
    best = sum(sorted_gain[i] * comb(i, k - 1) for i in range(k - 1, n)) / denominator
    found = 1 - comb(int((gain <= EPS).sum()), k) / denominator
    return float(best), found


def evaluate(row, prediction):
    proxy_order = rank(row['proxy'], row['tie'])
    ordering = {
        'by_type': quota_order(proxy_order, row['types'], np.ones_like(row['urgency'])),
        'by_residual': quota_order(proxy_order, row['types'], row['urgency'].clip(min=1e-8)),
        'proxy': proxy_order,
        'learned': rank(prediction, row['tie']),
    }
    gain = row['y'].clip(min=0)
    oracle = float(gain.max())
    results = []
    for k in BUDGETS:
        for method in ('random', *ordering):
            out = dict(panel=row['panel'], path=row['path'], step=row['step'], connected=row['connected'],
                       budget=k, method=method, candidates=len(gain), oracle=oracle)
            if method == 'random':
                value, hit = random_outcome(gain, k)
                chosen_slots = ''
            else:
                selection = ordering[method][:k]
                assert len(set(selection)) == k
                value = float(gain[selection].max())
                hit = float(value > EPS)
                chosen_slots = ' '.join(str(row['slots'][i]) for i in selection)
            assert 0 <= value <= oracle + EPS
            results.append(dict(out, gain=value, hit=hit, gap=max(0, oracle - value), shortlisted_slots=chosen_slots))
    return results


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    reference = read(REFERENCE)
    rng = np.random.default_rng(SEED)
    start = monotonic()
    panels, fingerprints, checks = {}, {}, {}
    for name in PANELS:
        panels[name], fingerprints[name], checks[name] = prepare(name, reference, rng)
    left, right = (fingerprints[name] for name in PANELS)
    assert set(left['paths'].values()).isdisjoint(right['paths'].values())
    assert set(left['states']).isdisjoint(right['states'])
    dump('input_checks.json', dict(panels=checks, shared_state_fingerprints=0, shared_trajectory_fingerprints=0,
                                  reference_sha256=sha(REFERENCE), source_sha256=sha(Path(__file__)),
                                  protocol_sha256=sha(SOURCE / 'PROTOCOL_ZH.md'), prepare_seconds=monotonic() - start))
    outcome, fits = [], []
    for source, test in (PANELS, tuple(reversed(PANELS))):
        train, held = panels[source], panels[test]
        counts = {traj: sum(r['path'] == traj for r in train) for traj in {r['path'] for r in train}}
        x = np.concatenate([r['x'] for r in train])
        y = np.concatenate([r['y'] for r in train]) * 1000
        sample_weight = np.concatenate([np.full(len(r['y']), 1 / (len(r['y']) * counts[r['path']])) for r in train])
        model = HistGradientBoostingRegressor(max_iter=120, max_leaf_nodes=15, learning_rate=0.08,
                                              min_samples_leaf=40, l2_regularization=0.1, early_stopping=False,
                                              random_state=SEED)
        tic = monotonic()
        with threadpool_limits(limits=2):
            model.fit(x, y, sample_weight=sample_weight / sample_weight.mean())
            prediction = model.predict(np.concatenate([r['x'] for r in held])) / 1000
        assert np.isfinite(prediction).all()
        fits.append(dict(training=source, held_out=test, train_states=len(train), held_out_states=len(held),
                         training_candidates=len(x), settings=model.get_params(), seconds=monotonic()-tic))
        offset = 0
        for row in held:
            stop = offset + len(row['x'])
            outcome.extend(evaluate(row, prediction[offset:stop]))
            offset = stop
        assert offset == len(prediction)
    table('per_state.csv', outcome)
    dump('run.json', dict(sk_learn=sklearn.__version__, fits=fits, new_graphs=0, new_candidates=0,
                          cpu_only=True, elapsed_seconds=monotonic()-start, outcome_rows=len(outcome)))


if __name__ == '__main__':
    main()
