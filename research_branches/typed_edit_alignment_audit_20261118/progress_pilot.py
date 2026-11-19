"""One bounded first-stage inference-policy intervention; no training."""
import json
import time
import sys
from pathlib import Path
import numpy as np
import torch
import audit as a
import metrics

b = a.b
OUT = a.ROOT/'results/typed_edit_alignment_audit_20261118/progress_filter_pilot_128'
ARMS = ['original_M','filtered_M','filtered_N']


def put(name, value):
    path = OUT/name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def verify():
    for path, sha in b.read(OUT/'manifest.json')['files'].items():
        assert b.sha(path)==sha,path


def inputs(ds, seed):
    data,cp,folder = a.spec(ds,seed)
    graph = b.restore_graph(b.read(folder/'step10000.json'))
    target = np.array(b.read(data/f'starts/seed{seed}/target.json'))
    weight = b.read(folder/'report.json')['trace'][0]['effective_role_weight']
    return graph,target,weight,cp


@torch.no_grad()
def chain(ds,seed,arm,model,steps):
    graph,target,weight,_ = inputs(ds,seed)
    state = a.CpuAuditState(graph,target,'cpu',use_triton=False)
    initial_error = float(state.relation_degree_error())
    error = initial_error
    moves = []
    start = time.monotonic()
    arrived = False
    calls = 0
    for k in range(steps):
        draw_seed = 202611180000+1000*seed+k
        batch = state.random_candidates(16,torch.Generator().manual_seed(draw_seed),
            role_phase=True,targeted_fraction=.5,closure_fraction=0.,candidate_mode='legacy_pool')
        gain = state.degree_gains(batch)
        if arm != 'filtered_N':
            x = state.compact_features(batch,1-(10000+k)/20000,model.input_dim)
            logits = model(x[None]).squeeze(0)
            calls += 1
            scores = logits + weight*gain.to(logits.dtype)/10
        else:
            logits = torch.zeros(16)
            scores = .05*gain.to(torch.float32)/10 + 1e-6*torch.rand(16,generator=torch.Generator().manual_seed(draw_seed+9999999))
        allowed = torch.ones(16,dtype=torch.bool)
        if arm != 'original_M':
            allowed = (gain>0) if bool((gain>0).any()) else batch.action==0
        selected = int(scores.masked_fill(~allowed,-torch.inf).argmax())
        chosen_gain = float(gain[selected])
        if arm != 'original_M':
            assert chosen_gain >= 0
            if bool((gain>0).any()):
                assert chosen_gain > 0
        state.commit(batch,torch.tensor(selected))
        recomputed = float(state.relation_degree_error())
        assert recomputed == error-chosen_gain
        error = recomputed
        values = a.serialize(batch)
        moves.append(dict(step=k+1,draw_seed=draw_seed,gains=gain.tolist(),logits=logits.tolist(),
            scores=scores.tolist(),allowed=allowed.tolist(),selected=selected,
            selected_gain=chosen_gain,error_per_node=error/graph.num_nodes,
            candidate={key:val[selected] for key,val in values.items()},
            candidates_sha256=a.hashlib.sha256(json.dumps(values,sort_keys=True).encode()).hexdigest()))
        if (k+1)%32==0 or k+1==steps:
            if error/graph.num_nodes <= 5.5:
                arrived=True
                break
    result = state.to_graph(graph)
    result.validate()
    assert np.array_equal(result.node_types,graph.node_types)
    for f in graph.family_specs:
        assert len(result.family_edges[f])==len(graph.family_edges[f])
    assert calls==0 if arm=='filtered_N' else calls==len(moves)
    return result,dict(dataset=ds,start_seed=seed,arm=arm,steps=len(moves),arrived=arrived,
        effective_weight=.05 if arm=='filtered_N' else weight,
        initial_error_per_node=initial_error/graph.num_nodes,
        final_error_per_node=error/graph.num_nodes,model_calls=calls,
        moves=moves,wall_seconds=time.monotonic()-start)


def main():
    b.setup()
    assert b.read(a.OUT/'acceptance.json')['passed']
    OUT.mkdir(parents=True,exist_ok=False)
    start=time.monotonic()
    files=a.input_files()+[Path(__file__),Path(metrics.__file__),Path(__file__).with_name('PROGRESS_FILTER_PILOT.md'),a.OUT/'acceptance.json']
    for ds in ['freebase','delicious']:
        files.append(a.spec(ds,100)[0]/'graph.json')
    jobs=[dict(dataset=ds,start_seed=seed,arm=arm,steps=128) for ds in ['freebase','delicious'] for seed in [100,101] for arm in ARMS]
    put('jobs.json',dict(jobs=jobs,training=0,maximum_steps=1536,hard_seconds=600,
        threshold=5.5,check_every=32,time_horizon=20000,time_offset=10000,development=True))
    files.append(OUT/'jobs.json')
    put('manifest.json',dict(files={str(p):b.sha(p) for p in files}))
    models={ds:a.load_mixed_edit_checkpoint(a.spec(ds,100)[1],'cpu')[0].eval() for ds in ['freebase','delicious']}
    refs={ds:b.restore_graph(b.read(a.spec(ds,100)[0]/'graph.json')) for ds in models}
    rc={ds:metrics.counts(g) for ds,g in refs.items()}
    gate=[]
    metric_times=[]
    initial_metrics={}
    for ds in models:
        for seed in [100,101]:
            tic=time.monotonic()
            initial_metrics[(ds,seed)]=metrics.compare(inputs(ds,seed)[0],refs[ds],rc[ds])
            metric_times.append(time.monotonic()-tic)
        first=[]
        for arm in ARMS:
            _, r=chain(ds,100,arm,models[ds],8)
            first.append(r['moves'][0]['candidates_sha256'])
            gate.append(dict(dataset=ds,arm=arm,seconds=r['wall_seconds'],steps=r['steps'],model_calls=r['model_calls']))
        assert len(set(first))==1
    bound=2*(sum(r['seconds']/r['steps']*128*2 for r in gate)+max(metric_times)*12)+30
    put('gate.json',dict(passed=bound<=480,conservative_seconds=bound,engineering=gate,
        engineering_moves=48,engineering_model_calls=32,metric_seconds=metric_times,
        same_first_candidates=True,ledger_matches_after_each_commit=True))
    assert bound<=480,bound
    results=[]
    for job in jobs:
        ds,seed,arm=job['dataset'],job['start_seed'],job['arm']
        graph,r=chain(ds,seed,arm,models[ds],job['steps'])
        key=f'{ds}_{seed}_{arm}'
        put(Path('outputs')/key/'graph.json',b.graph_payload(graph))
        put(Path('outputs')/key/'trace.json',r)
        z=metrics.compare(graph,refs[ds],rc[ds])
        z.update({k:v for k,v in r.items() if k!='moves'})
        z['initial_metrics']=initial_metrics[(ds,seed)]
        z['graph_sha256']=b.sha(OUT/'outputs'/key/'graph.json')
        z['output_key']=key
        results.append(z)
        put('per_output.json',results)
        print(ds,seed,arm,'steps',r['steps'],'degree',r['final_error_per_node'],
            'primary',z['primary_error'],'seconds',round(time.monotonic()-start,2),flush=True)
    checks=[]
    for ds in models:
        for seed in [100,101]:
            rr={r['arm']:r for r in results if r['dataset']==ds and r['start_seed']==seed}
            o,m,n=[rr[k] for k in ARMS]
            zero=m['initial_metrics']
            checks.append(dict(dataset=ds,start_seed=seed,
                residual_improves=m['final_error_per_node']<m['initial_error_per_node'],
                residual_not_worse_than_original=m['final_error_per_node']<=o['final_error_per_node'],
                structure_guard=m['primary_error']<=zero['primary_error']*1.05,
                lcc_guard=m['lcc_fraction']>=zero['lcc_fraction']-.01,
                connectivity_guard=not zero['connected'] or m['connected'],
                learned_beats_matched_N=m['primary_error']<n['primary_error']))
    screen=all(all(r[k] for k in ['residual_improves','residual_not_worse_than_original','structure_guard','lcc_guard','connectivity_guard']) for r in checks)
    verify()
    put('decision.json',dict(checks=checks,development_screen_passed=screen,
        candidate_for_independent_increment=screen and all(r['learned_beats_matched_N'] for r in checks),
        automatic_followup=False,stage2_increment_proven=False))
    put('completion.json',dict(completed=True,wall_seconds=time.monotonic()-start,scientific_short_suffixes=len(results),
        scientific_steps=sum(r['steps'] for r in results),scientific_model_calls=sum(r['model_calls'] for r in results),
        new_training_fits=0,gpu_calls=0,engineering_moves=48,engineering_model_calls=32,
        historical_assets_unchanged=True))


if __name__=='__main__':
    try:
        main()
    except BaseException as exc:
        if OUT.exists():
            put('failure.json',dict(type=type(exc).__name__,message=str(exc)))
        raise
