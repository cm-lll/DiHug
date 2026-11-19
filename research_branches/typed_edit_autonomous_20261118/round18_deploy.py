"""Independent IMDB scoring and a matched, untrained structural-prior control."""
import round18_train as train
import round16_deploy as old
from potential import Potential,ComponentState,encode
from candidate_consequences import Consequences
from round01 import t,b,np,torch,time,Path,ROOT,SRC,json,csv
OUT=ROOT/'results/typed_edit_autonomous_20261118/r18_deploy';r5=old.prev.r5;cache=old.prev.cache
FIELDS=['primary_error','triangle_vector_relative_l1','cycle_vector_relative_l1','triangles','connected','components','lcc_fraction']
SPECS={};IMDB_ARMS=['S_812','S_813','S_814','N','NS','Q']
JOBS=[dict(dataset='pubmed',start_id=s,arm='Q') for s in [106,107]]+[dict(dataset='imdb',start_id=s,arm=a) for s in [106,107] for a in IMDB_ARMS]


def put(name,v):
    p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')


def spec(ds):
    if ds not in SPECS:
        graph=b.graph(ds);rc=t.metrics.counts(graph)
        if ds=='imdb':columns=b.read(train.cache.OUT/'schema.json')['columns'];norm=b.read(train.cache.OUT/'normalization.json')
        else:columns=b.read(cache.OUT/'schema.json')['columns'];norm=b.read(old.train.r11.OUT/'normalization.json')
        SPECS[ds]=(graph,rc,columns,norm)
    return SPECS[ds]


def model(ds,arm):
    if arm in ['N','NS']:return None
    ref,rc,columns,norm=spec(ds);m=Potential(ref.num_nodes,columns,norm['mean'],norm['sd'])
    if arm=='Q':
        with torch.no_grad():m.weights[m.k:].fill_(1.)
    else:
        seed=int(arm.split('_')[-1]);folder=train.OUT if ds=='imdb' else old.train.OUT
        m.load_state_dict(torch.load(folder/f'fits/{seed}/checkpoint.pt',map_location='cpu',weights_only=False)['state_dict'])
    return m.eval()


def value(ds,graph):
    ref,rc,_,_=spec(ds);v=t.metrics.compare(graph,ref,rc);return {k:v[k] for k in FIELDS}


def base_seed(job):return (2026111818000 if job['dataset']=='imdb' else 2026111816000)+100000*job['start_id']


def verify():
    train.verify();old.verify()
    for p,h in b.read(OUT/'manifest.json')['files'].items():assert b.sha(p)==h,p


@torch.no_grad()
def chain(job,steps):
    tic=time.monotonic();ds=job['dataset'];arm=job['arm'];sid=job['start_id'];folder=b.RUN/f'prefixes/{ds}/seed{sid}';binding=b.read(folder/'binding.json')
    assert b.sha(folder/'initial.json')==binding['initial_sha256'] and b.sha(folder/'target.json')==binding['target_sha256']
    initial=b.restore_graph(b.read(folder/'initial.json'));target=np.array(b.read(folder/'target.json'));_,_,columns,_=spec(ds)
    tracker=ComponentState(initial,columns);state=t.a.CpuAuditState(initial,target,'cpu',use_triton=False);degrees=state.family_degree.clone();m=model(ds,arm)
    init_seconds=time.monotonic()-tic;loop_seconds=0.;metric_seconds=0.;trace=[];raws=[];deltas=[];scores=[];snapshots={}
    for step in range(steps):
        start=time.monotonic();seed=base_seed(job)+step;batch=state.random_candidates(16,torch.Generator().manual_seed(seed),role_phase=False,targeted_fraction=0.,closure_fraction=0.,candidate_mode='legacy_pool')
        values=t.a.serialize(batch);raw=tracker.counts.copy();dd=tracker.differences(t.proposals(initial,values));raws.append(raw);deltas.append(dd)
        if m is None:
            available=np.arange(16) if arm=='N' else np.flatnonzero(np.array(values['action'])==4)
            chosen=int(available[int(torch.randint(0,len(available),(1,),generator=torch.Generator().manual_seed(seed+5000000)))]) if len(available) else 0;q=None
        else:q=m(torch.from_numpy(encode(raw,dd,initial.num_nodes,columns))[None]).numpy()[0];assert np.isfinite(q).all();chosen=int(q.argmax())
        before=tracker.n_components;reject=tracker.guarded(values,chosen);after=tracker.n_components;effective=0 if reject else chosen
        state.commit(batch,torch.tensor(effective));assert torch.equal(degrees,state.family_degree) and after<=before
        trace.append(dict(step=step,proposal_seed=seed,time_fraction=1-(binding['transition_step']+step)/32000,candidates=values,chosen=chosen,effective=effective,
            connectivity_rejected=reject,components_before=before,components_after=after));scores.append(q);loop_seconds+=time.monotonic()-start
        if step+1 in {128,steps}:
            start=time.monotonic();graph=state.to_graph(initial);graph.validate()
            for f in tracker.names:assert tracker.edges[f]=={tuple(sorted(e)) for e in graph.family_edges[f]} and np.array_equal(graph.family_degrees(f),initial.family_degrees(f))
            _,actual=cache.context(graph,columns);assert np.array_equal(actual['raw'],tracker.counts) and actual['components']==tracker.n_components
            snapshots[step+1]=dict(graph=b.graph_payload(graph),metric=value(ds,graph));metric_seconds+=time.monotonic()-start
    return trace,np.stack(raws),np.stack(deltas),None if m is None else np.stack(scores),snapshots,dict(init_seconds=init_seconds,loop_seconds=loop_seconds,metric_seconds=metric_seconds)


def components(nb):
    unseen=set(range(len(nb)));count=0
    while unseen:
        count+=1;stack=[unseen.pop()]
        while stack:
            u=stack.pop()
            for v in nb[u]:
                if v in unseen:unseen.remove(v);stack.append(v)
    return count


def main():
    b.setup();train.verify();old.verify();assert b.read(train.OUT/'acceptance.json')['passed'];OUT.mkdir(parents=True,exist_ok=False)
    files=[Path(__file__),SRC/'ROUND18.md',SRC/'potential.py',SRC/'candidate_consequences.py',train.OUT/'acceptance.json',old.OUT/'acceptance.json',old.OUT/'summary.json']
    for seed in train.SEEDS:files.append(train.OUT/f'fits/{seed}/checkpoint.pt')
    for ds in ['imdb','pubmed']:
        files.append(b.RUN/f'data/{ds}/graph.json')
        for sid in [106,107]:files.extend(b.RUN/f'prefixes/{ds}/seed{sid}/{name}' for name in ['initial.json','target.json','binding.json'])
    files.extend([old.train.OUT/'fits/812/checkpoint.pt',old.OUT/'outputs/start106/S_812/trace.json',old.OUT/'outputs/start106/S_812/features.npz',old.OUT/'outputs/start106/N/trace.json',old.train.r11.OUT/'normalization.json'])
    put('manifest.json',dict(files={str(p):b.sha(p) for p in files},jobs=JOBS,new_graphs=14,scientific_opportunities=7168,learned_contexts=3072,fixed_Q_contexts=2048,new_training_updates=0))
    maxerr=0.
    for arm in ['S_812','N']:
        trace,raw,dd,q,snaps,rp=chain(dict(dataset='pubmed',start_id=106,arm=arm),8);prior=b.read(old.OUT/f'outputs/start106/{arm}/trace.json')
        assert [{k:v for k,v in row.items() if not k.startswith('components_')} for row in trace]==prior[:8]
        if q is not None:maxerr=max(maxerr,float(abs(q-np.load(old.OUT/f'outputs/start106/{arm}/features.npz')['logits'][:8]).max()));assert maxerr<=2e-5
        put(f'engineering/parity_{arm}.json',rp)
    cost=30.;zero={}
    for job in JOBS:
        trace,raw,dd,q,snaps,rp=chain(job,8);key=f"{job['dataset']}_{job['start_id']}_{job['arm']}";put(f'engineering/{key}.json',rp)
        cost+=2*(rp['init_seconds']+64*rp['loop_seconds']+2*rp['metric_seconds'])
        zkey=f"{job['dataset']}_{job['start_id']}"
        if zkey not in zero:zero[zkey]=value(job['dataset'],b.restore_graph(b.read(b.RUN/f"prefixes/{job['dataset']}/seed{job['start_id']}/initial.json")))
    put('zero.json',zero);put('engineering.json',dict(passed=cost<=1200,cost_bound_seconds=cost,engineering_opportunities=128,pubmed_prior_logit_error=maxerr))
    assert cost<=1200;verify();start=time.monotonic();rows=[]
    for job in JOBS:
        assert time.monotonic()-start<1200
        trace,raw,dd,q,snaps,rp=chain(job,512);folder=Path('outputs')/job['dataset']/f"start{job['start_id']}"/job['arm'];put(folder/'trace.json',trace)
        np.savez_compressed(OUT/folder/'features.npz',counts=raw,delta=dd,logits=np.empty((0,16),dtype=np.float32) if q is None else q);put(folder/'timing.json',rp)
        for step,snapshot in snaps.items():
            put(folder/f'graph{step}.json',snapshot['graph']);row=dict(**job,steps=step,**snapshot['metric'],noop_choices=sum(v['candidates']['action'][v['chosen']]==0 for v in trace[:step]),
                connectivity_rejections=sum(v['connectivity_rejected'] for v in trace[:step]),effective_switches=sum(v['candidates']['action'][v['effective']]==4 for v in trace[:step]))
            rows.append(row);put(folder/f'report{step}.json',row)
        print('IMDB/Q generated',job,round(time.monotonic()-start,2),flush=True)
    put('completion.json',dict(completed=True,new_graphs=14,snapshot_evaluations=28,scientific_opportunities=7168,learned_contexts=3072,fixed_Q_contexts=2048,seconds=time.monotonic()-start))
    start=time.monotonic();replayed=0;fullchecks=0;logit_error=0.
    for job in JOBS:
        ds=job['dataset'];sid=job['start_id'];arm=job['arm'];folder=OUT/'outputs'/ds/f'start{sid}'/arm;trace=b.read(folder/'trace.json');saved=np.load(folder/'features.npz')
        initial=b.restore_graph(b.read(b.RUN/f'prefixes/{ds}/seed{sid}/initial.json'));edges,nb=r5.topology(initial);names=sorted(edges);_,_,columns,_=spec(ds);m=model(ds,arm)
        if m is not None:
            xx=np.stack([encode(raw,dd,initial.num_nodes,columns) for raw,dd in zip(saved['counts'],saved['delta'])]);_,q,_=b.c.eval_model(m,xx,np.zeros(512,dtype=np.int64),'cpu')
            logit_error=max(logit_error,float(abs(q-saved['logits']).max()));assert logit_error<=2e-5
        for i,record in enumerate(trace):
            assert time.monotonic()-start<1500
            values=record['candidates'];raw=saved['counts'][i];dd=saved['delta'][i];before=components(nb);assert before==record['components_before']
            if i%32==0:
                graph=r5.graph_from(initial,edges);_,actual=cache.context(graph,columns);assert np.array_equal(actual['raw'],raw)
                independent=Consequences(graph,columns,actual['raw']).differences(t.proposals(graph,values));assert np.array_equal(independent,dd);fullchecks+=1
            chosen=record['chosen']
            if m is not None:assert chosen==int(saved['logits'][i].argmax())
            else:
                available=np.arange(16) if arm=='N' else np.flatnonzero(np.array(values['action'])==4)
                expected=int(available[int(torch.randint(0,len(available),(1,),generator=torch.Generator().manual_seed(record['proposal_seed']+5000000)))]) if len(available) else 0
                assert chosen==expected
            r5.edit(edges,nb,names,values,chosen);after=components(nb);reject=after>before;assert reject==record['connectivity_rejected']
            if reject:r5.edit(edges,nb,names,values,chosen,reverse=True);after=components(nb)
            effective=0 if reject else chosen;assert record['effective']==effective and record['components_after']==after and after<=before
            if i+1<len(trace):assert np.array_equal(raw+dd[effective],saved['counts'][i+1])
            if i+1 in [128,512]:
                graph=r5.graph_from(initial,edges);expected=b.restore_graph(b.read(folder/f'graph{i+1}.json'))
                for f in names:assert edges[f]=={tuple(sorted(e)) for e in expected.family_edges[f]} and np.array_equal(graph.family_degrees(f),initial.family_degrees(f))
                v=value(ds,graph);row=b.read(folder/f'report{i+1}.json')
                for k,vv in v.items():assert row[k]==vv
            replayed+=1
        print('IMDB/Q accepted',job,round(time.monotonic()-start,2),flush=True)
    imdb=[];pubmed=[];prior=b.read(old.OUT/'summary.json')['rows']
    for sid in [106,107]:
        rr={v['arm']:v for v in rows if v['dataset']=='imdb' and v['start_id']==sid and v['steps']==512}
        for seed in train.SEEDS:
            e=rr[f'S_{seed}']['primary_error'];imdb.append(dict(start_id=sid,seed=seed,error=e,N=rr['N']['primary_error'],NS=rr['NS']['primary_error'],Q=rr['Q']['primary_error'],
                structural_passed=e<min(zero[f'imdb_{sid}']['primary_error'],rr['N']['primary_error'],rr['NS']['primary_error']),supervision_vs_Q_passed=e<rr['Q']['primary_error']))
        q=next(v['primary_error'] for v in rows if v['dataset']=='pubmed' and v['start_id']==sid and v['steps']==512)
        for seed in train.SEEDS:
            e=next(v['primary_error'] for v in prior if v['start_id']==sid and v['arm']==f'S_{seed}' and v['steps']==512);pubmed.append(dict(start_id=sid,seed=seed,S=e,Q=q,passed=e<q))
    put('summary.json',dict(rows=rows,zero=zero,imdb_decisions=imdb,pubmed_Q_decisions=pubmed,
        imdb_structural_passed=all(v['structural_passed'] for v in imdb),imdb_supervision_vs_Q_passed=all(v['supervision_vs_Q_passed'] for v in imdb),
        pubmed_supervision_vs_Q_passed=all(v['passed'] for v in pubmed),imdb_all_outputs_whole_connected=all(v['connected'] for v in rows if v['dataset']=='imdb' and v['steps']==512),new_training_updates=0))
    with (OUT/'per_output.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    per_fit=[]
    for step in [128,512]:
        for arm in IMDB_ARMS:
            rr=[v for v in rows if v['dataset']=='imdb' and v['steps']==step and v['arm']==arm]
            per_fit.append(dict(arm=arm,steps=step,outputs=2,coefficient_fits=1 if arm.startswith('S_') else 0,
                primary_error_mean=float(np.mean([v['primary_error'] for v in rr])),across_start_sample_sd=float(np.std([v['primary_error'] for v in rr],ddof=1))))
    agg=[]
    for step in [128,512]:
        vals=[v['primary_error_mean'] for v in per_fit if v['steps']==step and v['arm'].startswith('S_')]
        agg.append(dict(steps=step,optimization_repeats=3,outputs_per_repeat=2,mean=float(np.mean(vals)),fit_sample_sd=float(np.std(vals,ddof=1))))
    put('aggregation.json',dict(imdb_per_arm=per_fit,imdb_fit_aggregate=agg));verify()
    put('acceptance.json',dict(passed=True,replayed_opportunities=replayed,sampled_full_state_recounts=fullchecks,snapshot_metric_recomputations=28,logit_error=logit_error,seconds=time.monotonic()-start))
    print('IMDB/Q decisions',imdb,pubmed,flush=True)


if __name__=='__main__':
    try:main()
    except BaseException as exc:put('failure.json',dict(type=type(exc).__name__,message=str(exc)));raise
