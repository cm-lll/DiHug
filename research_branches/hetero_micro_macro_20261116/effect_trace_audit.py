"""CPU-only post-run audit of the fixed complete generation panel."""
import hashlib
import json
import sys
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def main(out):
    p = read(out / 'protocol.json')
    rows = []
    for task in p['tasks']:
        folder = out / task['model'] / task['arm'] / task['readout'] / f"{task['arm']}_seed{task['sample_seed']}"
        row = dict(model=task['model'], arm=task['arm'], sample_seed=task['sample_seed'])
        if not (folder / 'summary.json').exists():
            rows.append(dict(**row, status='missing_complete_chain'))
            continue
        parent = Path(task['paired_parent'])
        identity = read(folder / 'identity.json')
        old_identity = read(parent / 'identity.json')
        steps = [json.loads(s) for s in (folder / 'steps.jsonl').read_text().splitlines()]
        old_steps = [json.loads(s) for s in (parent / 'steps.jsonl').read_text().splitlines()]
        summary = read(folder / 'summary.json')
        assert len(steps) == len(old_steps) == 1000
        assert identity['updates'] == 128 and identity['readout'] == 'ema'
        assert identity['initial_hash'] == old_identity['initial_hash']
        assert digest(folder / 'graph.npz') == summary['graph_sha256']
        assert digest(task['checkpoint']) == identity['checkpoint_sha256']
        for index, (a, b) in enumerate(zip(steps, old_steps)):
            assert a['time'] == b['time'] == 1000-index
            assert a['uniform_hash'] == b['uniform_hash']
            assert a['query_hash'] == b['query_hash']
            assert a['covered_slots'] == b['covered_slots'] and a['exact_coverage']
            assert a['paired_parent_contract']
        rows.append(dict(**row, status='passed', paired_steps=len(steps), graph_sha256=summary['graph_sha256']))
    result = dict(status='passed' if len(rows)==32 and all(x['status']=='passed' for x in rows) else 'partial',
                  complete_chains=sum(x['status']=='passed' for x in rows),
                  paired_steps=sum(x.get('paired_steps', 0) for x in rows), rows=rows,
                  scope='Input/query/uniform/identity and graph integrity, not bitwise CUDA recomputation or quality.')
    (out / 'final_trace_audit.json').write_text(json.dumps(result, indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!='rows'}))


if __name__ == '__main__':
    main(Path(sys.argv[1]))
