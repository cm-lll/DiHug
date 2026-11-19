"""Frozen P/Q/random suffixes and explicit complete-policy accounting."""
import round22_prefix as prefix
import round20_deploy as learned
from potential import ComponentState,encode
from candidate_consequences import Consequences
from round01 import t,b,np,torch,time,Path,ROOT,SRC,json,csv
p=learned.p;r5=p.r5;cache=p.cache;OUT=ROOT/'results/typed_edit_autonomous_20261118/r23_complete'
ARMS=['P_812','P_813','P_814','Q','N','NS']


def put(name,v):
    path=OUT/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')


def inputs(job):
    sid=job['start_id'];producer=job['producer'];folder=prefix.OUT/f'outputs/{sid}_{producer}'
    return b.restore_graph(b.read(folder/'graph.json')),np.array(b.read(prefix.OUT/f'inputs/{sid}/target.json')),b.read(folder/'report.json')


def verify():
    prefix.verify();learned.verify()
    for path,h in b.read(OUT/'manifest.json')['files'].items():assert b.sha(path)==h,path


@torch.no_grad()
def chain(job,steps):
    tic=time.monotonic();initial,target,pr=inputs(job);assert pr['arrived'] and pr['connected'];arm=job['arm'];_,_,columns,_=p.spec('imdb')
    tracker=ComponentState(initial,columns);assert tracker.n_components==1;state=t.a.CpuAuditState(initial,target,'cpu',use_triton=False);degree=state.family_degree.clone();m=learned.model('imdb',arm)
    init_seconds=time.monotonic()-tic;loop_seconds=0.;metric_seconds=0.;trace=[];raws=[];deltas=[];scores=[];snaps={}
    for step in range(steps):
        tic=time.monotonic();seed=2026111823000+100000*job['start_id']+step
        batch=state.random_candidates(16,torch.Generator().manual_seed(seed),role_phase=False,targeted_fraction=0.,closure_fraction=0.,candidate_mode='legacy_pool')
        vals=t.a.serialize(batch);raw=tracker.counts.copy();dd=tracker.differences(t.proposals(initial,vals));raws.append(raw);deltas.append(dd)
        if m is None:
            available=np.arange(16) if arm=='N' else np.flatnonzero(np.array(vals['action'])==4);chosen=int(available[int(torch.randint(0,len(available),(1,),generator=torch.Generator().manual_seed(seed+5000000)))]) if len(available) else 0;q=None
        else:q=m(torch.from_numpy(encode(raw,dd,initial.num_nodes,columns))[None]).numpy()[0];assert np.isfinite(q).all();chosen=int(q.argmax())
        reject=tracker.guarded(vals,chosen);effective=0 if reject else chosen;state.commit(batch,torch.tensor(effective));assert torch.equal(degree,state.family_degree) and tracker.n_components==1
        trace.append(dict(step=step,proposal_seed=seed,time_fraction=1-(pr['steps']+step)/20000,candidates=vals,chosen=chosen,effective=effective,connectivity_rejected=reject));scores.append(q);loop_seconds+=time.monotonic()-tic
        if step+1 in [128,steps]:
            tic=time.monotonic();graph=state.to_graph(initial);graph.validate()
            for f in tracker.names:assert tracker.edges[f]=={tuple(sorted(e)) for e in graph.family_edges[f]} and np.array_equal(graph.family_degrees(f),initial.family_degrees(f))
            _,fresh=cache.context(graph,columns);assert np.array_equal(fresh['raw'],tracker.counts);snaps[step+1]=dict(graph=b.graph_payload(graph),metrics=prefix.stats(graph,target));metric_seconds+=time.monotonic()-tic
    return trace,np.stack(raws),np.stack(deltas),None if m is None else np.stack(scores),snaps,dict(init_seconds=init_seconds,loop_seconds=loop_seconds,metric_seconds=metric_seconds)


def main():
    b.setup();prefix.verify();learned.verify();assert b.read(prefix.OUT/'acceptance.json')['passed'];ps=b.read(prefix.OUT/'summary.json');assert ps['C_arrival_and_coverage_passed'] and ps['C_whole_connected'];OUT.mkdir(parents=True,exist_ok=False)
    jobs=[dict(start_id=s,producer='C',arm=a) for s in prefix.SEEDS for a in ARMS];failures=[]
    for sid in prefix.SEEDS:
        report=b.read(prefix.OUT/f'outputs/{sid}_D/report.json')
        if report['arrived'] and report['connected']:jobs.append(dict(start_id=sid,producer='D',arm='N'))
        else:failures.append(dict(start_id=sid,producer='D',reason='first_stage_not_eligible',report=report))
    files=[Path(__file__),SRC/'ROUND23.md',prefix.OUT/'acceptance.json',prefix.OUT/'summary.json',learned.OUT/'acceptance.json']
    files.extend(learned.train.OUT/f'fits/imdb/{s}/checkpoint.pt' for s in learned.train.SEEDS)
    files.extend(prefix.OUT/f'outputs/{s}_{a}/{n}' for s in prefix.SEEDS for a in ['C','D'] for n in ['graph.json','report.json'])
    files.extend(prefix.OUT/f'inputs/{s}/target.json' for s in prefix.SEEDS)
    put('manifest.json',dict(files={str(f):b.sha(f) for f in files},jobs=jobs,failed_full_requests=failures,new_training_updates=0,first_stage_fit='original M810 for C; none for D',stage_two_coefficient_fits=3))
    original=p.model;p.model=learned.model
    tr,raw,dd,q,_,_=p.chain(dict(dataset='imdb',start_id=106,arm='P_812'),8);p.model=original
    old=learned.OUT/'outputs/imdb/start106/P_812';assert tr==b.read(old/'trace.json')[:8];parity=float(abs(q-np.load(old/'features.npz')['logits'][:8]).max());assert parity<=2e-5
    cost=30.;zero={}
    for job in jobs:
        _,_,_,_,_,rp=chain(job,8);cost+=2*(rp['init_seconds']+64*rp['loop_seconds']+2*rp['metric_seconds']);put(f"engineering/{job['start_id']}_{job['producer']}_{job['arm']}.json",rp)
        key=f"{job['start_id']}_{job['producer']}"
        if key not in zero:g,target,pr=inputs(job);zero[key]=prefix.stats(g,target)
    put('zero.json',zero);put('engineering.json',dict(passed=cost<=1200,cost_bound_seconds=cost,engineering_opportunities=8+8*len(jobs),prior_logit_error=parity));assert cost<=1200;verify();tic=time.monotonic();rows=[]
    for job in jobs:
        assert time.monotonic()-tic<1200
        trace,raw,dd,q,snaps,rp=chain(job,512);folder=Path('outputs')/f"{job['start_id']}_{job['producer']}"/job['arm'];put(folder/'trace.json',trace);put(folder/'timing.json',rp)
        np.savez_compressed(OUT/folder/'features.npz',counts=raw,delta=dd,logits=np.empty((0,16),dtype=np.float32) if q is None else q);_,_,pr=inputs(job)
        for step,v in snaps.items():
            put(folder/f'graph{step}.json',v['graph']);row=dict(**job,steps=step,**v['metrics'],first_stage_steps=pr['steps'],total_attempts=pr['steps']+step,first_stage_model_calls=pr['model_calls'],stage_two_learned_calls=step if job['arm'].startswith('P_') else 0,
                noop_choices=sum(x['candidates']['action'][x['chosen']]==0 for x in trace[:step]),connectivity_rejections=sum(x['connectivity_rejected'] for x in trace[:step]),effective_switches=sum(x['candidates']['action'][x['effective']]==4 for x in trace[:step]),
                first_stage_seconds_reused=pr['seconds'],stage_two_seconds_for_512=rp['init_seconds']+rp['loop_seconds']+rp['metric_seconds'])
            rows.append(row);put(folder/f'report{step}.json',row)
        print('complete generated',job,round(time.monotonic()-tic,2),flush=True)
    put('completion.json',dict(completed=True,new_graphs=len(jobs),scientific_opportunities=512*len(jobs),snapshot_evaluations=2*len(jobs),new_training_updates=0,seconds=time.monotonic()-tic));tic=time.monotonic();replayed=0;full=0;error=0.
    for job in jobs:
        folder=OUT/'outputs'/f"{job['start_id']}_{job['producer']}"/job['arm'];trace=b.read(folder/'trace.json');saved=np.load(folder/'features.npz');initial,target,pr=inputs(job);edges,nb=r5.topology(initial);names=sorted(edges);_,_,columns,_=p.spec('imdb');m=learned.model('imdb',job['arm'])
        if m is not None:
            xx=np.stack([encode(raw,dd,initial.num_nodes,columns) for raw,dd in zip(saved['counts'],saved['delta'])]);_,q,_=b.c.eval_model(m,xx,np.zeros(512,dtype=np.int64),'cpu');error=max(error,float(abs(q-saved['logits']).max()));assert error<=2e-5
        for i,v in enumerate(trace):
            assert time.monotonic()-tic<1500
            vals=v['candidates'];raw=saved['counts'][i];dd=saved['delta'][i]
            if i%32==0:
                g=r5.graph_from(initial,edges);_,actual=cache.context(g,columns);assert np.array_equal(actual['raw'],raw);fresh=Consequences(g,columns,actual['raw']).differences(t.proposals(g,vals));assert np.array_equal(fresh,dd);full+=1
            chosen=v['chosen']
            if m is not None:assert chosen==int(saved['logits'][i].argmax())
            else:
                av=np.arange(16) if job['arm']=='N' else np.flatnonzero(np.array(vals['action'])==4);expected=int(av[int(torch.randint(0,len(av),(1,),generator=torch.Generator().manual_seed(v['proposal_seed']+5000000)))]) if len(av) else 0;assert chosen==expected
            r5.edit(edges,nb,names,vals,chosen);reject=not r5.connected(nb);assert reject==v['connectivity_rejected']
            if reject:r5.edit(edges,nb,names,vals,chosen,True)
            effective=0 if reject else chosen;assert effective==v['effective'] and r5.connected(nb)
            if i+1<len(trace):assert np.array_equal(raw+dd[effective],saved['counts'][i+1])
            if i+1 in [128,512]:
                g=r5.graph_from(initial,edges);stored=b.restore_graph(b.read(folder/f'graph{i+1}.json'))
                for f in names:assert edges[f]=={tuple(sorted(e)) for e in stored.family_edges[f]} and np.array_equal(g.family_degrees(f),initial.family_degrees(f))
                row=b.read(folder/f'report{i+1}.json')
                for k,value in prefix.stats(g,target).items():assert row[k]==value
            replayed+=1
        print('complete accepted',job,round(time.monotonic()-tic,2),flush=True)
    decisions=[]
    for sid in prefix.SEEDS:
        rr={v['arm']:v for v in rows if v['start_id']==sid and v['producer']=='C' and v['steps']==512}
        for seed in learned.train.SEEDS:
            e=rr[f'P_{seed}']['primary_error'];decisions.append(dict(start_id=sid,seed=seed,P=e,Q=rr['Q']['primary_error'],N=rr['N']['primary_error'],NS=rr['NS']['primary_error'],zero=zero[f'{sid}_C']['primary_error'],structural_passed=e<min(zero[f'{sid}_C']['primary_error'],rr['N']['primary_error'],rr['NS']['primary_error']),supervision_vs_Q_passed=e<rr['Q']['primary_error']))
    perfit=[float(np.mean([v['P'] for v in decisions if v['seed']==s])) for s in learned.train.SEEDS]
    put('summary.json',dict(rows=rows,decisions=decisions,zero=zero,failed_full_requests=failures,structural_passed=all(v['structural_passed'] for v in decisions),supervision_vs_Q_passed=all(v['supervision_vs_Q_passed'] for v in decisions),all_whole_connected=all(v['connected'] for v in rows),per_fit_means=perfit,fit_mean=float(np.mean(perfit)),fit_sample_sd=float(np.std(perfit,ddof=1)),first_stage_fits=1,coefficient_fits=3,development_not_untouched_confirmation=True))
    with (OUT/'per_output.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    verify();put('acceptance.json',dict(passed=True,replayed_opportunities=replayed,sampled_full_state_recounts=full,snapshot_metric_recomputations=2*len(jobs),logit_error=error,seconds=time.monotonic()-tic));print('complete decisions',decisions,flush=True)


if __name__=='__main__':
    try:main()
    except BaseException as exc:put('failure.json',dict(type=type(exc).__name__,message=str(exc)));raise
