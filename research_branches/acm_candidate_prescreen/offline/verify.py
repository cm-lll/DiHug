"""Check saved shortlists and gains against original hard-graph count archives."""

import hashlib
import json
from math import comb, isclose
import numpy as np
import pandas as pd

from run import ARCHIVE, OUT, SOURCE, archive_path



def read(path):
    return json.loads(path.read_text())


def replacements(candidates, i):
    sides = []
    for side in ('old_edges', 'new_edges'):
        sides.append(frozenset(frozenset(e) for e in candidates[side][i]))
    return candidates['family'][i], *sides


def main():
    checks = read(OUT / 'input_checks.json')
    assert all(hashlib.sha256((SOURCE / name).read_bytes()).hexdigest() == checks[key]
               for name, key in (('run.py', 'source_sha256'), ('PROTOCOL_ZH.md', 'protocol_sha256')))
    saved = pd.read_csv(OUT / 'per_state.csv', keep_default_na=False)
    lookup = {name: list(group.itertuples(index=False)) for name, group in saved.groupby(['panel', 'path', 'step'])}
    verified = []
    for panel in ('pilot', 'bank4'):
        dest = ARCHIVE / panel
        raw = archive_path(read(dest / 'inventory.json')['raw'])
        encoded = archive_path(read(dest / 'manifest.json')['folder'])
        with np.load(encoded / 'data.npz') as data, np.load(dest / 'target.npz') as target:
            counts, changes, ref = data['counts'], data['delta'], data['refcounts']
            groups, allowed = data['groups'], target['allowed']
        tri_count = sum(a == 'typed_triangles' for a, _ in read(dest / 'manifest.json')['columns'])
        tri, cycles = slice(None, tri_count), slice(tri_count, None)
        def error(row):
            return sum(np.abs(row[part] - ref[part]).sum() / max(1, ref[part].sum()) for part in (tri, cycles)) / 2
        index = 0
        for folder in sorted((raw / 'trajectories').iterdir()):
            for record in read(folder / 'record.json')['records']:
                assert groups[index] == folder.name
                key = (panel, folder.name, record['step'])
                candidates = record['candidates']
                injected = replacements(candidates, record['label'])
                valid = {}
                for slot, action in enumerate(candidates['action']):
                    if action == 0:
                        continue
                    edit = replacements(candidates, slot)
                    if edit != injected:
                        valid.setdefault(edit, slot)
                available = set(valid.values())
                before = error(counts[index])
                gains = {slot: max(0, before - error(counts[index] + changes[index, slot])) if allowed[index, slot] else 0
                         for slot in available}
                for row in lookup.pop(key):
                    assert row.candidates == len(available)
                    assert row.connected == (record['components'] == 1)
                    assert isclose(row.oracle, max(gains.values()), abs_tol=1e-12)
                    if row.method == 'random':
                        levels, multiplicity = np.unique(list(gains.values()), return_counts=True)
                        through = np.cumsum(multiplicity)
                        denominator = comb(len(available), row.budget)
                        p_through = np.array([comb(int(x), row.budget) / denominator for x in through])
                        value = float(levels @ np.diff(p_through, prepend=0))
                        hit = 1 - p_through[levels <= 1e-12].max(initial=0)
                    else:
                        slots = list(map(int, row.shortlisted_slots.split()))
                        assert len(slots) == len(set(slots)) == row.budget
                        assert set(slots) <= available
                        value = max(gains[i] for i in slots)
                        hit = float(value > 1e-12)
                    assert isclose(row.gain, value, rel_tol=0, abs_tol=1e-12)
                    assert isclose(row.hit, hit, rel_tol=0, abs_tol=1e-12)
                index += 1
        assert index == len(groups) == 1664
        verified.append(panel)
    assert not lookup
    receipt = dict(accepted=True, verified_panels=verified, verified_states=3328, verified_rows=len(saved),
                   checks=['exact gains independently recomputed from recorded count changes',
                           'connectivity rejection after shortlisting', 'equal candidate counts',
                           'injected inverses and duplicate edits excluded', 'random expectation verified'])
    (OUT / 'verification.json').write_text(json.dumps(receipt, indent=2) + '\n')


if __name__ == '__main__':
    main()
