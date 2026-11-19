"""CPU fixed-state audit of the frozen original typed-edit implementation."""
import os
import sys
import json
import time
import hashlib
import tempfile
import inspect
import textwrap
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'research_branches/typed_edit_complexity_20261116'))
import common as b
from dihug.core.gpu_role_guided_edit import GpuTypedGraphState, GpuEditBatch
import dihug.core.gpu_role_guided_edit as original_core
from dihug.core.mixed_edit_diffusion import load_mixed_edit_checkpoint

OUT = ROOT/'results/typed_edit_alignment_audit_20261118/cpu_v2'
EXT = ROOT/'results/typed_edit_external_adaptation_20261117'
PROP = ROOT/'results/typed_edit_proposal_audit_20261117'
EPOCH = ROOT/'results/typed_edit_epoch_gate_20261117'
GEN = ROOT/'results/typed_edit_epoch_generation_20261117'
STATES = {'prefix': GEN/'prefix/initial.json',
          'M810_selected_endpoint': GEN/'outputs/suffix_M_810_e20/graph.json',
          'N_endpoint': GEN/'outputs/suffix_N/graph.json'}


# Audit-local CPU adaptation: change only the device guard, never the frozen module.
# Tensor arithmetic and candidate construction are inherited unchanged; Triton stays off.
_constructor = textwrap.dedent(inspect.getsource(GpuTypedGraphState.__init__))
_guard = '    if self.device.type != "cuda":\n        raise ValueError("GPU graph state requires a CUDA device")\n'
assert _constructor.count(_guard) == 1
_namespace = dict(vars(original_core))
exec(compile('from __future__ import annotations\n'+_constructor.replace(_guard, ''),
             '<audit_only_cpu_constructor>', 'exec'), _namespace)
class CpuAuditState(GpuTypedGraphState):
    __init__ = _namespace['__init__']

GpuTypedGraphState = CpuAuditState


def put(name, value):
    (OUT/name).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def spec(ds, seed):
    data = (b.RUN/'data/freebase' if ds == 'freebase' else
            ROOT/'results/typed_edit_revision_20261116/delicious/data')
    cp = (b.RUN/'fits/freebase/seed810/M/model.pt' if ds == 'freebase' else
          data.parent/'fits/seed710/M/model.pt')
    folder = EXT/f'{ds}_reachability_18000/outputs/M_{seed}'
    return data, cp, folder


def input_files():
    files = list(b.FROZEN.rglob('*.py')) + [Path(b.__file__), Path(__file__),
        Path(__file__).with_name('PROTOCOL.md'), PROP/'records.json', PROP/'acceptance.json',
        b.RUN/'cache/pubmed/data.npz', GEN/'prefix/target.json', *STATES.values()]
    for seed in [810, 811]:
        q = EPOCH/f'fits/seed{seed}/M'
        files += [q/'selection.json', q/f'epoch{b.read(q/"selection.json")["epoch"]}.pt']
    for ds in ['freebase', 'delicious']:
        for seed in [100, 101]:
            data, cp, folder = spec(ds, seed)
            report = b.read(folder/'report.json')
            files += [cp, folder/'report.json', data/f'starts/seed{seed}/initial.json',
                      data/f'starts/seed{seed}/target.json']
            for s in report['snapshots'].values():
                assert b.sha(s['path']) == s['sha256']
                files.append(Path(s['path']))
    return sorted(set(files))


def verify():
    for path, h in b.read(OUT/'manifest.json')['files'].items():
        assert b.sha(path) == h, path


def serialize(batch):
    return {k: v.tolist() for k, v in vars(batch).items()}


def deserialize(values):
    return GpuEditBatch(**{k: torch.tensor(v, dtype=torch.bool if k.endswith('_valid') else torch.long)
                           for k, v in values.items()})


def independent_gains(graph, target, batch):
    """Recompute each changed family degree from edge endpoint integer deltas."""
    arrays = serialize(batch)
    degrees = np.column_stack([graph.family_degrees(f) for f in sorted(graph.family_specs)])
    result = []
    for i, f in enumerate(arrays['family']):
        change = np.zeros(graph.num_nodes, dtype=np.int64)
        for field, valid, sign in [('old_edges', 'old_valid', -1), ('new_edges', 'new_valid', 1)]:
            for edge, yes in zip(arrays[field][i], arrays[valid][i]):
                if yes:
                    for v in edge:
                        change[v] += sign
        residual = degrees[:, f] - target[:, f]
        result.append(int(np.dot(residual, residual) - np.dot(residual+change, residual+change)))
    return np.array(result)


def stage1_draw(ds, seed, step, graph, target, model, effective, draw_seed):
    state = GpuTypedGraphState(graph, target, 'cpu', use_triton=False)
    before = b.graph_payload(state.to_graph(graph))
    batch = state.random_candidates(16, torch.Generator().manual_seed(draw_seed),
        role_phase=True, targeted_fraction=.5, closure_fraction=0., candidate_mode='legacy_pool')
    x = state.compact_features(batch, 1-step/20000, model.input_dim)
    logits = model(x[None]).squeeze(0).numpy()
    gain = state.degree_gains(batch).numpy()
    exact = independent_gains(graph, target, batch)
    assert np.array_equal(gain, exact)
    assert np.isfinite(x.numpy()).all() and np.isfinite(logits).all()
    chosen = int(np.argmax(logits + effective*gain/10))
    fixed = int(np.argmax(logits + .05*gain/10))
    maximum = gain.max()
    # Least weight at which a maximum-gain candidate can dominate all lower-gain candidates.
    best = int(np.argmax(np.where(gain == maximum, logits, -np.inf)))
    lower = gain < maximum
    critical = max(0., float(np.max((logits[lower]-logits[best])*10/(maximum-gain[lower])))) if lower.any() else 0.
    assert before == b.graph_payload(state.to_graph(graph))
    return dict(dataset=ds, start_seed=seed, step=step, draw_seed=draw_seed,
        effective_weight=effective, time=1-step/20000, candidates=serialize(batch),
        logits=logits.tolist(), degree_gains=gain.tolist(), selected=chosen,
        selected_gain=float(gain[chosen]), maximum_gain=float(maximum),
        fixed_weight_selected=fixed, fixed_weight_gain=float(gain[fixed]),
        critical_weight_for_max_gain=critical, score_sd=float(logits.std()),
        guidance_sd=float((effective*gain/10).std()),
        conflicts_with_positive=bool(gain[chosen]<0 and maximum>0),
        misses_positive=bool(gain[chosen]<=0 and maximum>0))


def stage2_features(record, states):
    graph, state = states[record['state']]
    result = {}
    for policy, data in record['policies'].items():
        batch = deserialize(data['candidates'])
        x = state.compact_features(batch, record['time_fraction'], 45)
        delta = state._closure_components(batch)[0].sum(1).numpy()
        assert np.array_equal(delta, data['triangle_deltas'])
        assert np.all(state.degree_gains(batch).numpy() == 0)
        assert np.all(independent_gains(graph, state.target_rows.numpy(), batch) == 0)
        result[policy] = x
    return result


def mean(rows, key):
    return float(np.mean([r[key] for r in rows])) if rows else None


@torch.no_grad()
def execute():
    b.setup()
    assert tempfile.gettempdir() == str(ROOT/'.runtime_tmp')
    assert not torch.cuda.is_initialized()
    OUT.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    files = input_files()
    put('manifest.json', dict(files={str(f): b.sha(f) for f in files},
        proposal_seeds=list(range(2026111800, 2026111808)), time_interventions=[.1, .5, .8],
        cpu_only=True, training_updates=0, graph_commits=0))
    models = {}
    for ds in ['freebase', 'delicious']:
        models[ds] = load_mixed_edit_checkpoint(spec(ds, 100)[1], 'cpu')[0].eval()
    pmodels = {}
    for seed in [810, 811]:
        folder = EPOCH/f'fits/seed{seed}/M'
        epoch = b.read(folder/'selection.json')['epoch']
        model = b.model(45, 'M', seed, 'cpu')
        model.load_state_dict(torch.load(folder/f'epoch{epoch}.pt', map_location='cpu', weights_only=False)['state_dict'])
        pmodels[seed] = model.eval()
    records = b.read(PROP/'records.json')
    target = np.array(b.read(GEN/'prefix/target.json'))
    states = {}
    for name, path in STATES.items():
        graph = b.restore_graph(b.read(path))
        states[name] = graph, GpuTypedGraphState(graph, target, 'cpu', use_triton=False)
    # Engineering gate: representative expensive states, then bound all planned records.
    gate_times = []
    for ds in ['freebase', 'delicious']:
        data, _, folder = spec(ds, 100)
        report = b.read(folder/'report.json')
        graph = b.restore_graph(b.read(folder/'step10000.json'))
        target = np.array(b.read(data/'starts/seed100/target.json'))
        tic = time.monotonic()
        stage1_draw(ds, 100, 10000, graph, target, models[ds],
                    report['trace'][0]['effective_role_weight'], 2026111899)
        gate_times.append(time.monotonic()-tic)
    tic = time.monotonic()
    feats = stage2_features(records[0], states)
    error = 0.
    for policy, x in feats.items():
        for seed, model in pmodels.items():
            old = records[0]['policies'][policy]['selections'][f'M_{seed}']['logits']
            error = max(error, float(np.max(abs(model(x[None]).numpy()[0]-old))))
    t2 = time.monotonic()-tic
    bound = 2*(max(gate_times)*128 + t2*96*4) + 30
    passed = error <= 2e-5 and bound <= 300
    put('gate.json', dict(passed=passed, original_logit_max_error=error,
        stage1_batch_seconds=gate_times, stage2_record_seconds=t2,
        conservative_seconds=bound, hard_timeout_seconds=600,
        engineering_model_forward_calls=6, independent_degree_gain_checks=64))
    assert passed, b.read(OUT/'gate.json')
    rows = []
    traces = []
    skipped = []
    for ds in ['freebase', 'delicious']:
        for seed in [100, 101]:
            data, _, folder = spec(ds, seed)
            report = b.read(folder/'report.json')
            target = np.array(b.read(data/f'starts/seed{seed}/target.json'))
            weight = report['trace'][0]['effective_role_weight']
            assert all(r['effective_role_weight'] == weight for r in report['trace'])
            for r in report['trace']:
                traces.append(dict(dataset=ds, start_seed=seed, **r))
            for step in [0, 10000, 14000, 18000]:
                if step and report['snapshots'][str(step)]['transition_step'] >= 0:
                    skipped.append(dict(dataset=ds, start_seed=seed, step=step, reason='already_stage2'))
                    continue
                path = data/f'starts/seed{seed}/initial.json' if step == 0 else folder/f'step{step}.json'
                graph = b.restore_graph(b.read(path))
                residual = np.square(np.column_stack([graph.family_degrees(f) for f in sorted(graph.family_specs)])-target).sum()/graph.num_nodes
                trace = next(t for t in report['trace'] if t['step'] == step)
                assert abs(residual-trace['relation_degree_error_per_node']) < 1e-8
                for draw_seed in range(2026111800, 2026111808):
                    rows.append(stage1_draw(ds, seed, step, graph, target, models[ds], weight, draw_seed))
                print(ds, seed, step, round(time.monotonic()-started, 2), flush=True)
    put('stage1_records.json', rows)
    put('historical_traces.json', traces)
    put('stage1_skipped.json', skipped)
    selections = []
    feature_rows = []
    original_error = 0.
    for rid, record in enumerate(records):
        feats = stage2_features(record, states)
        for policy, x in feats.items():
            feature_rows.append(x.numpy())
            item = record['policies'][policy]
            for seed, model in pmodels.items():
                for variant, t in [('original', record['time_fraction']), ('t0.1', .1), ('t0.5', .5), ('t0.8', .8)]:
                    changed = x.clone()
                    changed[:, 35] = t
                    logits = model(changed[None]).numpy()[0]
                    if variant == 'original':
                        old = item['selections'][f'M_{seed}']['logits']
                        original_error = max(original_error, float(np.max(abs(logits-old))))
                        assert np.argmax(logits) == item['selections'][f'M_{seed}']['index']
                    k = int(logits.argmax())
                    neg = np.array(item['triangle_deltas']) < 0
                    no = item['candidates']['action'].index(0)
                    selections.append(dict(record=rid, state=record['state'], policy=policy,
                        seed=seed, variant=variant, time=t, logits=logits.tolist(), selected=k,
                        noop=k==no, triangle_delta=item['triangle_deltas'][k],
                        best_negative_minus_noop=float(logits[neg].max()-logits[no]) if neg.any() else None))
    assert original_error <= 2e-5, original_error
    put('stage2_selections.json', selections)
    np.savez_compressed(OUT/'stage2_features.npz', x=np.stack(feature_rows))
    train = np.load(b.RUN/'cache/pubmed/data.npz')
    tx = train['x'][train['ids']<30]
    # Use actual training switch candidates, not only positive labels; report marginal support, not density.
    ts = tx[tx[:, :, :5].argmax(2) == 4]
    dist = []
    for state_name in STATES:
        for policy in ['0', '1']:
            xx = np.stack([feature_rows[2*i+int(policy=='0')] for i,r in enumerate(records) if r['state']==state_name])
            # Explicit key order in source is ['1','0']; assert instead of silently assuming.
            assert all(list(r['policies']) == ['1', '0'] for r in records)
            xs = xx[xx[:, :, :5].argmax(2) == 4]
            for j in list(range(15,35))+list(range(39,45)):
                lo, hi = float(ts[:,j].min()), float(ts[:,j].max())
                dist.append(dict(state=state_name, policy=policy, coordinate=j, train_min=lo, train_max=hi,
                    deploy_min=float(xs[:,j].min()), deploy_max=float(xs[:,j].max()),
                    outside_train_range=float(((xs[:,j]<lo-1e-5)|(xs[:,j]>hi+1e-5)).mean())))
    put('feature_support.json', dist)
    one = []
    for ds in ['freebase', 'delicious']:
        for seed in [100,101]:
            for step in [0,10000,14000,18000]:
                rr = [r for r in rows if (r['dataset'],r['start_seed'],r['step']) == (ds,seed,step)]
                if rr:
                    one.append(dict(dataset=ds,start_seed=seed,step=step,batches=len(rr),
                        negative_choices=sum(r['selected_gain']<0 for r in rr),
                        conflicts_with_positive=sum(r['conflicts_with_positive'] for r in rr),
                        misses_positive=sum(r['misses_positive'] for r in rr),
                        mean_selected_gain=mean(rr,'selected_gain'),mean_max_gain=mean(rr,'maximum_gain'),
                        mean_fixed_weight_gain=mean(rr,'fixed_weight_gain'),
                        effective_weight=rr[0]['effective_weight'],
                        critical_weight_mean=mean(rr,'critical_weight_for_max_gain'),
                        model_sd_mean=mean(rr,'score_sd'),guidance_sd_mean=mean(rr,'guidance_sd')))
    two = []
    for state_name in STATES:
        for policy in ['0','1']:
            for seed in [810,811]:
                for variant in ['original','t0.1','t0.5','t0.8']:
                    rr = [r for r in selections if (r['state'],r['policy'],r['seed'],r['variant']) == (state_name,policy,seed,variant)]
                    two.append(dict(state=state_name,policy=policy,seed=seed,variant=variant,n=len(rr),
                        noop=sum(r['noop'] for r in rr),negative=sum(r['triangle_delta']<0 for r in rr),
                        mean_delta=mean(rr,'triangle_delta')))
    put('summary.json', dict(stage1=one,stage2=two,stage1_skipped=skipped,
        limitations=['fixed-state development diagnostics, not trajectories or quality effects',
        'CPU proposal streams differ from historical CUDA streams',
        'marginal range checks do not establish full distribution coverage',
        'no actual per-step historical logits were saved'],
        original_stage2_logit_max_error=original_error))
    verify()
    put('completion.json', dict(completed=True,seconds=time.monotonic()-started,
        scientific_model_forward_calls=len(rows)+len(selections),engineering_model_forward_calls=6,
        stage1_batches=len(rows),stage2_selections=len(selections),independent_training_fits=0,
        new_generated_graphs=0,graph_commits=0,gpu_calls=0,
        original_logit_max_error=original_error,inputs_unchanged=True))


if __name__ == '__main__':
    try:
        execute()
    except BaseException as exc:
        if OUT.exists():
            put('failure.json',dict(type=type(exc).__name__,message=str(exc)))
        raise
