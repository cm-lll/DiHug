"""Read-only CPU audit and budget; never imports a denoiser or launches workers."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '1'
import csv
import hashlib
import json
import statistics
import time
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
SHORT = ROOT / 'results/analytic_joint_short_20261113'
PRIOR = ROOT / 'results/fullchain_reliability_run_20261113T133731'
OUT = ROOT / 'results/analytic_joint_generation_plan_20261113'

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(2**20), b''):
            h.update(chunk)
    return h.hexdigest()

def digest(x):
    h = hashlib.sha256()
    def add(v):
        if torch.is_tensor(v):
            h.update(str((v.dtype, tuple(v.shape))).encode())
            h.update(v.detach().cpu().contiguous().numpy().tobytes())
        elif isinstance(v, np.ndarray):
            h.update(str((v.dtype, v.shape)).encode()); h.update(v.tobytes())
        elif isinstance(v, dict):
            for k in sorted(v, key=str):
                h.update(repr(k).encode()); add(v[k])
        elif isinstance(v, (tuple, list)):
            for a in v: add(a)
        else:
            h.update(repr(v).encode())
    add(x)
    return h.hexdigest()

def main():
    start = time.time(); torch.set_num_threads(1)
    OUT.mkdir(exist_ok=False)
    bindings = {}
    def bind(path, expected=None):
        path = Path(path).resolve(); value = sha(path)
        if expected is not None: assert value == expected, str(path)
        bindings[str(path)] = value
        return path
    def read(path): return json.loads(bind(path).read_text())
    def write(name, data):
        (OUT / name).write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False))
    bind(__file__)
    short = read(SHORT / 'protocol.json'); previous = read(PRIOR / 'protocol.json')
    for path, h in short['bindings'].items(): bind(path, h)
    for path, h in previous['source_bindings'].items(): bind(path, h)
    parents = []; checkpoints = []; tasks = []; reused = []; costs = []
    for row in previous['models']:
        job = next(j for j in short['jobs'] if j['id'] == row['id'])
        assert job['parent']['config'] == row['config']
        assert job['initial'] == row['initial_complete']
        parent = torch.load(bind(row['readout']), map_location='cpu', weights_only=False)
        initial = torch.load(bind(row['initial_complete']), map_location='cpu', weights_only=False)
        assert parent['updates'] == 3000 and parent['readout'] == 'ema_0.99'
        assert initial['updates'] == 0 and not initial['optimizer']['state']
        assert digest(parent['model']) == digest(initial['model']) == row['model_hash']
        data = np.load(bind(row['data'])); types = data['types']; schema = data['schema']
        result = read(SHORT / row['id'] / 'result.json')
        # Ordinal choice fixed before any graph quality is read; no seed replacement.
        samples = sorted(row['reused'], key=lambda r:r['index'])[:4]
        assert [r['index'] for r in samples] == [0, 1, 2, 3]
        for sample in samples:
            folder = Path(sample['path']); ident = read(folder / 'identity.json')
            summary = read(folder / 'summary.json'); graph = bind(folder / 'graph.npz', summary['graph_sha256'])
            assert summary['status'] == 'complete' and summary['steps'] == 1000
            assert ident['updates'] == 0 and ident['model_hash'] == row['model_hash']
            assert ident['sample_seed'] == sample['sample_seed'] and ident['T'] == 1000 and ident['R'] == 1
            assert ident['train_seed'] == row['train_seed'] and ident['dataset'] == row['dataset']
            g = np.load(graph)
            assert np.array_equal(g['types'], types) and np.array_equal(g['schema'], schema)
            log = bind(folder / 'steps.jsonl'); lines = [json.loads(x) for x in log.read_text().splitlines()]
            assert len(lines) == 1000
            assert [x['time'] for x in lines] == list(range(1000, 0, -1))
            assert all(x['exact_coverage'] and x['blocks'] == 1 and x['uniform_hash'] and x['query_hash'] for x in lines)
            evaluation = bind(PRIOR / 'evaluation' / f'{row["id"]}_{sample["index"]:02d}.json')
            ev = json.loads(evaluation.read_text()); assert ev['graph_sha256'] == sha(graph)
            reused.append(dict(model=row['id'], dataset=row['dataset'], train_seed=row['train_seed'],
                sample_seed=sample['sample_seed'], ordinal=sample['index'], path=str(folder), graph=str(graph),
                evaluation=str(evaluation), initial_hash=ident['initial_hash'], steps=str(log)))
        for arm in ('bias', 'risk'):
            cp = bind(SHORT / row['id'] / f'{arm}_step16_complete.pt', result['arms'][arm]['evaluations']['16']['checkpoint_sha256'])
            state = torch.load(cp, map_location='cpu', weights_only=False)
            assert state['updates'] == state['ema']['updates'] == 16 and state['ema']['decay'] == .99
            assert state['binding'] == initial['binding']
            assert digest(state['rng']) == digest(initial['rng'])
            assert state['model'].keys() == initial['model'].keys()
            assert state['ema']['shadow'].keys() == initial['ema']['shadow'].keys()
            assert all(int(v['step']) == 16 for v in state['optimizer']['state'].values())
            for k, v in state['model'].items():
                assert v.shape == initial['model'][k].shape and v.dtype == initial['model'][k].dtype
                assert torch.isfinite(v).all()
            for k, v in state['ema']['shadow'].items():
                assert v.shape == state['model'][k].shape and v.dtype == state['model'][k].dtype
                assert torch.isfinite(v).all()
            for readout in ('raw', 'ema'):
                effective = dict(state['model'])
                if readout == 'ema': effective.update(state['ema']['shadow'])
                mh = digest(effective)
                checkpoints.append(dict(model=row['id'], arm=arm, readout=readout, checkpoint=str(cp),
                    checkpoint_sha256=sha(cp), model_hash=mh, updates=16, EMA_updates=16,
                    buffers='raw checkpoint buffers retained; only EMA shadow parameter keys replaced',
                    parent_model_hash=row['model_hash'], short_protocol_sha256=sha(SHORT/'protocol.json')))
                for sample in samples:
                    tasks.append(dict(model=row['id'], dataset=row['dataset'], train_seed=row['train_seed'],
                        arm=arm, readout=readout, checkpoint=str(cp), model_hash=mh,
                        sample_seed=sample['sample_seed'], paired_parent=str(Path(sample['path'])),
                        T=1000, R=1, feedback_updates=16))
        secs = [read(p)['seconds'] for p in sorted((PRIOR/row['id']).glob('chain_*/summary.json'))]
        assert len(secs) == 10
        costs.append(dict(model=row['id'], dataset=row['dataset'], historical_chains=10,
            seconds_mean=statistics.mean(secs), seconds_min=min(secs), seconds_max=max(secs),
            new_chains=16, projected_body_GPU_hours=16*statistics.mean(secs)/3600))
        parents.append(dict(id=row['id'], dataset=row['dataset'], train_seed=row['train_seed'],
            data=row['data'], config=row['config'], parent_readout=row['readout'],
            parent_model_hash=row['model_hash'], sampling_seeds=[s['sample_seed'] for s in samples]))
    body = sum(c['projected_body_GPU_hours'] for c in costs)
    protocol = dict(status='prepared_not_executed', execution_authorized=False,
        parents=parents, checkpoints=checkpoints, tasks=tasks, reused=reused,
        counts=dict(new_chains=64, reused_graphs=16, unique_final_graphs=80, new_training=0,
            new_readouts=16, physical_complete_checkpoints=8, samples_per_readout=4),
        sampling='Original R1/T1000 marginal kernel, same initial/noise/query streams within each parent; compare 1000 step hashes to reused chain; no guidance or graph filtering.',
        seed_rule='First four historical frozen-parent trajectories by manifest ordinal, not quality. Seeds differ across training models; all five readouts within one model share them.',
        readout_rule='Fixed feedback step16, raw/EMA separate; parent EMA3000 shared once, no best-checkpoint selection.',
        evidence_boundary='Historical developmental sample seeds, not untouched final confirmation. 16 feedback updates after EMA3000 with new Adam, six fixed times per update; not from-scratch training or IID continuation. CUDA recomputation bitwise NOT certified.',
        gates=['All bound hashes unchanged, current GPU availability', 'Strict effective-state readout hash including buffers',
            'First fixed chain10 steps each dataset included in panel: coverage, source/RNG identity, save/load, cost',
            'Continue only if all remaining64 chains fit remaining wall and GPU allocation limits; no effect threshold'],
        budgets=dict(history=costs, projected_body_GPU_hours=body, overhead_loading_saving_GPU_hours=.3,
            uncertainty_fraction=.25, projected_GPU_hours_with_margin=1.25*body+.3,
            requested_GPU_hours=13, requested_wall_minutes=170, science_stop_minutes=140,
            delivery_reserve_minutes=30, max_devices=8,
            clock='relative to approved execution wiring start; no active deadline or reserved GPU now'),
        scheduling='8 ready workers, two per parent, each one fixed readout and2arms×4chains; release as soon as finished. If fewer cards available, recompute full queue before start.',
        evaluation=['whole graph primary; LCC separate', 'joint degree, per-relation degree W1, low degree/hub',
            'edges, triangle counts/composition, MP2/MP3 counts/composition, components/connectivity',
            'all six within-model graph overlap pairs, reference overlap, duplicate checks',
            'per sample then model mean/sample SD; two training seeds per dataset, paired differences; no significance claim'],
        stopping='wall/GPU-hour limit first; checkpoint partial chains, no shortened chains, resampling, extra training, weight/formula/EMA changes',
        unresolved=['new generation adapter not yet implemented or GPU tested',
            'first-chain calibration must check changed graph density/runtime',
            'sampling distributions and diversity not determined by conditional proxy improvements'])
    write('protocol.json', protocol); write('asset_bindings.json', bindings)
    with (OUT/'task_manifest.csv').open('w') as f:
        w=csv.DictWriter(f, fieldnames=list(tasks[0]));w.writeheader();w.writerows(tasks)
    write('audit.json', dict(status='passed_CPU_asset_checks', seconds=time.time()-start,
        bindings=len(bindings), complete_checkpoints=8, readouts=16, reused_graphs=16,
        model_forwards=0, model_backwards=0, GPU_hours=0, new_chains=0,
        result='Preparation only; runtime integration and budget approval pending.'))
    print(json.dumps(dict(output=str(OUT), body_GPU_hours=body, with_margin=1.25*body+.3, seconds=time.time()-start)))

if __name__ == '__main__': main()
