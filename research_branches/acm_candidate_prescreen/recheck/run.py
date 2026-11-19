"""Replay saved connected ACM suffixes for two fixed proposal budgets."""

import csv
import gzip
import json
import os
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from hashlib import sha256
from multiprocessing import get_context
from pathlib import Path
from time import monotonic

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits


SOURCE = Path(__file__).resolve().parent
sys.path.insert(0, str(SOURCE.parent))
from offline import run as prior

OUT = prior.OUTPUT_ROOT / 'recheck'
PREVIOUS = prior.OUT
BASE = prior.ROOT / 'typed_edit_autonomous_20261118'
BUDGETS = (2, 4)
START_TIME = monotonic()
MODELS = {}
FORBIDDEN_STATES = set()
REFERENCE_SETS = []
REFERENCE_COUNTS = None
COLUMNS = None


def dump(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def rebuild_models(reference):
    checks = prior.read(PREVIOUS / 'input_checks.json')
    assert prior.sha(prior.SOURCE / 'run.py') == checks['source_sha256']
    assert prior.sha(prior.SOURCE / 'PROTOCOL_ZH.md') == checks['protocol_sha256']
    params = {fit['training']: fit['settings'] for fit in prior.read(PREVIOUS / 'run.json')['fits']}
    previous = pd.read_csv(PREVIOUS / 'per_state.csv')
    heldout = previous[(previous.method == 'learned') & (previous.budget == 4)].set_index(['panel', 'path', 'step'])
    rng = np.random.default_rng(prior.SEED)
    data = {}
    for panel in prior.PANELS:
        rows, index, bound = prior.prepare(panel, reference, rng)
        assert bound['archive_manifest_sha256'] == checks['panels'][panel]['archive_manifest_sha256']
        data[panel] = rows
        FORBIDDEN_STATES.update(index['states'])
    verified = {}
    for source, test in (prior.PANELS, prior.PANELS[::-1]):
        training, held = data[source], data[test]
        by_path = Counter(row['path'] for row in training)
        weight = np.concatenate([np.full(len(r['y']), 1 / (len(r['y']) * by_path[r['path']])) for r in training])
        model = HistGradientBoostingRegressor(**params[source])
        with threadpool_limits(limits=2):
            model.fit(np.concatenate([r['x'] for r in training]), np.concatenate([r['y'] for r in training]) * 1000,
                      sample_weight=weight / weight.mean())
            scored = model.predict(np.concatenate([r['x'] for r in held]))
        cursor = 0
        for row in held:
            step = len(row['x'])
            choice = prior.rank(scored[cursor:cursor+step] / 1000, row['tie'])[:4]
            slots = ' '.join(str(row['slots'][i]) for i in choice)
            assert slots == heldout.loc[test, row['path'], row['step']].shortlisted_slots
            cursor += step
        assert cursor == len(scored)
        MODELS[source] = model
        verified[source] = len(held)
    return verified


def jobs():
    libraries = [
        (BASE / 'acm_solver_repair/deploy', BASE / 'additional_f_pilot/prefix/acm', False),
        (BASE / 'additional_confirmation_v1/deploy/acm', BASE / 'additional_confirmation_v1/prefix/acm', True),
    ]
    result = []
    for deploy, prefix, repeated in libraries:
        accepted, starts = prior.read(deploy / 'acceptance.json'), prior.read(prefix / 'acceptance.json')
        assert accepted['passed'] and starts['passed']
        manifest = prior.read(deploy / 'manifest.json')
        for row in manifest['jobs']:
            start = f"{row['seed']}_{row['producer']}"
            arm = f"{row['arm']}_{row['repeat']}" if repeated else row['arm']
            start_graph = prefix / 'outputs' / start / 'graph.json'
            folder = deploy / 'outputs' / start / arm
            assert prior.sha(start_graph) == prior.archive_hash(starts['outputs'], start_graph)
            assert prior.sha(folder / 'graph2048.json') == prior.archive_hash(accepted['outputs'], folder / 'graph2048.json')
            assert prior.read(folder / 'report2048.json')['connected']
            result.append((start, arm, start_graph, folder))
    assert len(result) == 24 and len({r[0] for r in result}) == 7
    return result


@lru_cache(maxsize=262144)
def edge_fingerprint(family, u, v):
    return int.from_bytes(sha256(f'{family},{u},{v}'.encode()).digest(), 'big')


def initial_fingerprint(schema):
    value = 0
    for family, edges, reference in zip(schema.families, schema.sets, REFERENCE_SETS):
        for u, v in edges ^ reference:
            value ^= edge_fingerprint(family['name'], u, v)
    return value


def key_as_record(key):
    family, old, new = key
    return dict(action=[4], family=[family], old_edges=[old], new_edges=[new])


def connected_after(state, key):
    _, old, new = key
    removed = dict(old + tuple((v, u) for u, v in old))
    added = dict(new + tuple((v, u) for u, v in new))
    # From a connected graph, each fragment after removing old edges has one
    # of their endpoints. It suffices to find all four in the edited graph.
    start = min(removed, key=lambda u: len(state.neighbors[u]))
    targets = set(removed) - {start}
    queue, seen = [start], {start}
    for u in queue:
        missing = removed.get(u)
        successors = added.get(u)
        if successors is not None and successors not in seen:
            seen.add(successors)
            targets.discard(successors)
            queue.append(successors)
        for v in state.neighbors[u]:
            if v != missing and v not in seen:
                seen.add(v)
                targets.discard(v)
                queue.append(v)
        if not targets:
            return True
    return False


def full_connectivity(state, key=None):
    neighbors = [set(values) for values in state.neighbors]
    if key is not None:
        _, old, new = key
        for u, v in old:
            neighbors[u].remove(v)
            neighbors[v].remove(u)
        for u, v in new:
            neighbors[u].add(v)
            neighbors[v].add(u)
    queue, seen = [0], {0}
    for u in queue:
        fresh = neighbors[u] - seen
        seen.update(fresh)
        queue.extend(fresh)
    return len(seen) == len(neighbors)


def proposals(values, state):
    unique = {}
    for i, action in enumerate(values['action']):
        if action == 0:
            continue
        assert action == 4
        key = prior.action_key(values, i)
        unique.setdefault(key, i)
        _, old, new = key
        assert all(v in state.positions[u] for u, v in old)
        assert all(u != v and v not in state.positions[u] for u, v in new)
    assert len(unique) >= max(BUDGETS)
    return unique


def evaluate(job):
    start, arm, start_graph, folder = job
    rng = np.random.default_rng([prior.SEED, int(start.split('_')[0]), int(start.endswith('_D'))])
    schema = prior.Schema(prior.read(start_graph), COLUMNS, REFERENCE_COUNTS)
    with np.load(folder / 'features.npz') as data:
        counts, deltas = data['counts'], data['delta']
    trace = prior.read(folder / 'trace.json')
    assert len(trace) == len(counts) == len(deltas) == 2048
    assert np.array_equal(deltas[:, 0], np.zeros_like(counts))
    state = prior.ArchivedState(schema.sets, len(schema.types))
    fingerprint = initial_fingerprint(schema)
    fingerprints = set()
    rows, actual_edits = [], []
    for i, step in enumerate(trace):
        values = step['candidates']
        assert step['step'] == i
        fp = f'{fingerprint:064x}'
        assert fp not in FORBIDDEN_STATES
        fingerprints.add(fp)
        unique = proposals(values, state)
        keys, slots = list(unique), list(unique.values())
        nodes = sorted({u for _, old, _ in keys for pair in old for u in pair})
        witnesses = {u: state.witness_sample(u, rng) for u in nodes}
        residual = counts[i] - REFERENCE_COUNTS
        urgency = schema.support @ (np.abs(residual) / schema.half_denominator)
        features, proxy = zip(*(schema.proposal_features(state, key, witnesses, residual, urgency) for key in keys))
        tie = rng.permutation(len(keys))
        relation = np.array([key[0] for key in keys])
        rules = prior.quota_order(prior.rank(np.array(proxy), tie), relation, urgency.clip(min=1e-8))[:max(BUDGETS)]
        rows.append((keys, slots, tie, rules, np.asarray(features)))
        actual = step['effective']
        if i + 1 < len(trace):
            assert np.array_equal(counts[i] + deltas[i, actual], counts[i+1])
        if values['action'][actual] != 0:
            edit = prior.action_key(values, actual)
            f, old, new = edit
            for u, v in old + new:
                fingerprint ^= edge_fingerprint(schema.families[f]['name'], u, v)
            state.observe(key_as_record(edit))
            actual_edits.append(edit)
        else:
            actual_edits.append(None)
    end = sorted(prior.read(folder / 'graph2048.json')['families'], key=lambda row: row['name'])
    assert all(group == {prior.edge(e) for e in record['edges']} for group, record in zip(state.edges, end))
    flat = np.concatenate([row[4] for row in rows])
    with threadpool_limits(limits=1):
        scores = {name: model.predict(flat) / 1000 for name, model in MODELS.items()}
    state = prior.ArchivedState(schema.sets, len(schema.types))
    bests, detail = [], []
    offset = audit_count = 0
    for i, (keys, slots, tie, rule, features) in enumerate(rows):
        selection = {'residual': rule}
        for name, values in scores.items():
            selection[name] = prior.rank(values[offset:offset+len(features)], tie)[:max(BUDGETS)]
        offset += len(features)
        gains, checks = {}, {}
        for indices in selection.values():
            for j in indices:
                if j in gains:
                    continue
                # Only shortlisted actions access the archived exact increment.
                true_gain = float(schema.gain(counts[i] - REFERENCE_COUNTS, deltas[i, slots[j]]))
                if true_gain > prior.EPS or i % 128 == 0:
                    checks[j] = connected_after(state, keys[j])
                gains[j] = max(0, true_gain) if checks.get(j, False) else 0.0
        if i % 128 == 0:
            assert full_connectivity(state)
            for j, allowed in checks.items():
                assert allowed == full_connectivity(state, keys[j]), (start, arm, i, j)
                audit_count += 1
        record = dict(start=start, archive_arm=arm, step=i, eligible=len(keys))
        for k in BUDGETS:
            for name, indices in selection.items():
                subset = list(indices[:k])
                assert len(set(subset)) == k
                chosen = max(subset, key=lambda j: gains[j])
                record[f'{name}_{k}_slots'] = ' '.join(str(slots[j]) for j in subset)
                record[f'{name}_{k}_gain'] = gains[chosen]
        bests.append(record)
        if actual_edits[i] is not None:
            state.observe(key_as_record(actual_edits[i]))
    assert offset == len(flat) and full_connectivity(state)
    detail = dict(start=start, arm=arm, states=len(trace), unique_path_states=len(fingerprints),
                  min_eligible=min(len(row[0]) for row in rows), audited_shortlisted_connectivity=audit_count,
                  checked_start_sha256=prior.sha(start_graph), checked_end_sha256=prior.sha(folder / 'graph2048.json'),
                  trace_sha256=prior.sha(folder / 'trace.json'), counts_sha256=prior.sha(folder / 'features.npz'))
    return bests, detail


def main():
    global COLUMNS, REFERENCE_SETS, REFERENCE_COUNTS
    OUT.mkdir(parents=True, exist_ok=True)
    reference = prior.read(prior.REFERENCE)
    reference_families = sorted(reference['families'], key=lambda row: row['name'])
    REFERENCE_SETS = [{prior.edge(e) for e in family['edges']} for family in reference_families]
    manifest = prior.read(prior.ARCHIVE/'bank4/manifest.json')
    COLUMNS = manifest['columns']
    with np.load(prior.archive_path(manifest['folder'])/'data.npz') as data:
        REFERENCE_COUNTS = data['refcounts']
    available = jobs()
    reproduced = rebuild_models(reference)
    dump('training_reproduction.json', dict(reproduced_heldout_top4=reproduced, fits_on_evaluation_states=0,
                                            training_state_fingerprints=len(FORBIDDEN_STATES),
                                            inherited_source_sha256=prior.sha(prior.SOURCE/'run.py')))
    workers = min(int(os.environ.get('ACM_OFFLINE_WORKERS', '6')), 6)
    details = []
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context('fork')) as pool, gzip.open(OUT/'per_state.csv.gz', 'wt', newline='') as stream:
        writer = None
        for values, receipt in pool.map(evaluate, available):
            if writer is None:
                writer = csv.DictWriter(stream, fieldnames=list(values[0]))
                writer.writeheader()
            writer.writerows(values)
            details.append(receipt)
            print(receipt['start'], receipt['arm'], receipt['states'], receipt['audited_shortlisted_connectivity'], flush=True)
    dump('verification.json', dict(passed=True, trajectories=details, states=sum(r['states'] for r in details),
                                   training_state_overlap=0, generated_graphs=0, newly_sampled_candidates=0,
                                   source_sha256=prior.sha(Path(__file__)), protocol_sha256=prior.sha(SOURCE/'PROTOCOL_ZH.md'),
                                   elapsed_seconds=monotonic()-START_TIME))


if __name__ == '__main__':
    main()
