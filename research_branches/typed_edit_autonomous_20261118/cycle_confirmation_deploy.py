"""Paired coverage confirmation on new starts; base/coverage/Q share normalization."""
import cycle_confirmation_prefix_gpu as newprefix
import confirmation_common as contract
import confirmation_fit as fit
import confirmation_prefix_v2 as prefix
from potential import ComponentState,encode
from candidate_consequences import Consequences
from round01 import t,b,np,torch,time,Path,ROOT,SRC,json,csv
BASE=ROOT/'results/typed_edit_autonomous_20261118';OUT=BASE/'cycle_repair_confirmation/deploy'
DATASETS=['imdb'];SEEDS=[14001,14002];r5=prefix.r5;p=prefix.p
put=lambda name,v:contract.put(OUT,name,v)
verify=lambda:contract.verify(OUT)

def inputs(job):
    ds=job['dataset'];sid=job['seed'];folder=newprefix.OUT/f"outputs/{sid}_{job['producer']}"
    return b.restore_graph(b.read(folder/'graph.json')),np.array(b.read(newprefix.OUT/f'inputs/{sid}/target.json')),b.read(folder/'report.json')

def model(job):
    if job['arm'] in ['N','NS']:return None
    m=fit.model(job['dataset'],job['repeat'])
    if job['arm']=='F':
        cp=BASE/f"cycle_repair_confirmation/fit/repeat{job['repeat']}/checkpoint.pt" if job['variant']=='coverage' else fit.OUT/f"fits/{job['dataset']}/{job['repeat']}/checkpoint.pt"
        m.load_state_dict(torch.load(cp,map_location='cpu',weights_only=False)['state_dict'])
    else:
        m=m.float()
        with torch.no_grad():m.weights[:m.k].zero_();m.weights[m.k:].fill_(1.)
    return m.eval()

def folder_for(job):return Path('outputs')/job['dataset']/f"{job['seed']}_{job['producer']}"/f"{job['arm']}_{job['repeat']}_{job['variant']}"

def choose_random(vals,arm,seed):
    av=np.arange(16) if arm=='N' else np.flatnonzero(np.array(vals['action'])==4)
    return int(av[int(torch.randint(0,len(av),(1,),generator=torch.Generator().manual_seed(seed+5000000)))]) if len(av) else 0

@torch.no_grad()
def chain(job,steps,deadline=None):
    tic=time.monotonic();ds=job['dataset'];sid=job['seed'];arm=job['arm'];g,target,pr=inputs(job)
    assert pr['arrived'];cols=b.read(fit.CACHE/ds/'schema.json')['columns'];tracker=ComponentState(g,cols);assert tracker.n_components==1
    state=t.a.CpuAuditState(g,target,'cpu',use_triton=False);degree=state.family_degree.clone();m=model(job)
    init=time.monotonic()-tic;loop=0.;evaluation=0.;trace=[];raws=[];deltas=[];scores=[];snaps={}
    for step in range(steps):
        if deadline is not None:assert time.monotonic()<deadline
        tic=time.monotonic();seed=202611186400000+DATASETS.index(ds)*1000000000+sid*10000+step
        batch=state.random_candidates(16,torch.Generator().manual_seed(seed),role_phase=False,targeted_fraction=0.,closure_fraction=.5 if ds=='imdb' else 0.,candidate_mode='legacy_pool')
        vals=t.a.serialize(batch);raw=tracker.counts.copy();dd=tracker.differences(t.proposals(g,vals))
        if m is None:chosen=choose_random(vals,arm,seed);q=None
        else:
            xx=torch.from_numpy(encode(raw,dd,g.num_nodes,cols))[None];q=m(xx.double() if arm=='F' else xx).numpy()[0];assert np.isfinite(q).all();chosen=int(q.argmax())
        rejected=tracker.guarded(vals,chosen);effective=0 if rejected else chosen;state.commit(batch,torch.tensor(effective));assert torch.equal(degree,state.family_degree) and tracker.n_components==1
        trace.append(dict(step=step,proposal_seed=seed,candidates=vals,chosen=chosen,effective=effective,connectivity_rejected=rejected));raws.append(raw);deltas.append(dd);scores.append(q);loop+=time.monotonic()-tic
        if step+1 in [512,1024,2048,steps]:
            tic=time.monotonic();graph=state.to_graph(g);graph.validate()
            for f in tracker.names:assert tracker.edges[f]=={tuple(sorted(e)) for e in graph.family_edges[f]} and np.array_equal(graph.family_degrees(f),g.family_degrees(f))
            _,fresh=p.cache.context(graph,cols);assert np.array_equal(fresh['raw'],tracker.counts)
            snaps[step+1]=dict(graph=b.graph_payload(graph),metrics=prefix.stats(ds,graph,target));evaluation+=time.monotonic()-tic
    return trace,np.stack(raws),np.stack(deltas),None if m is None else np.stack(scores),snaps,dict(init_seconds=init,loop_seconds=loop,metric_seconds=evaluation)

def main():
    b.setup();fit.prior.verify();fit.source_check(fit.CACHE);fit.source_check(fit.OUT);fit.source_check(newprefix.OUT)
    covparents=[BASE/f'cycle_repair_confirmation/fit/repeat{r}' for r in [1,2,3]]
    for parent in covparents:
        fit.source_check(parent);assert b.read(parent/'summary.json')['passed']
    assert b.read(fit.OUT/'summary.json')['passed'];OUT.mkdir(parents=True,exist_ok=False)
    jobs=[];unavailable=[];zero={}
    for ds in DATASETS:
        for sid in SEEDS:
            for producer in ['C','D']:
                job=dict(dataset=ds,seed=sid,producer=producer);g,target,pr=inputs(job)
                zero[f'{ds}_{sid}_{producer}']=prefix.stats(ds,g,target)
                if not pr['arrived']:unavailable.append(dict(**job,reason='first_stage_did_not_arrive',report=pr));continue
                specs=([('F',r,v) for v in ['base','coverage'] for r in [1,2,3]]+[('Q',r,'fixed') for r in [1,2,3]]+[('N',0,'random'),('NS',0,'random')]) if producer=='C' else [('N',0,'random')]
                jobs.extend(dict(**job,arm=arm,repeat=repeat,variant=variant) for arm,repeat,variant in specs)
    files=[Path(__file__),SRC/'CYCLE_REPAIR_CONFIRMATION.md',SRC/'CYCLE_REPAIR_GPU_PREFIX.md',fit.OUT/'summary.json',newprefix.OUT/'summary.json']
    for parent in [fit.CACHE,fit.OUT,newprefix.OUT]+covparents:
        files.extend([parent/'manifest.json',parent/'acceptance.json']);files.extend(Path(f) for f in b.read(parent/'acceptance.json').get('outputs',{}))
    contract.bind(OUT,files,jobs=jobs,unavailable=unavailable,maximum_suffix_chains=24,fixed_opportunities=2048,new_training_updates=0)
    put('zero.json',zero);put('unavailable.json',unavailable);cost=30.;engineering=[];candidate_first={}
    for job in jobs:
        tr,_,_,_,_,rp=chain(job,8);key=f"{job['dataset']}_{job['seed']}_{job['producer']}"
        if key in candidate_first:assert candidate_first[key]==tr[0]['candidates']
        else:candidate_first[key]=tr[0]['candidates']
        cost+=2*(rp['init_seconds']+256*rp['loop_seconds']+3*rp['metric_seconds']);engineering.append(dict(**job,**rp))
    put('engineering.json',dict(passed=cost<=4800,cost_bound_seconds=cost,engineering_opportunities=8*len(jobs),rows=engineering));assert cost<=4800;verify();tic=time.monotonic();rows=[]
    for job in jobs:
        tr,raw,dd,q,snaps,rp=chain(job,2048,deadline=tic+4800);folder=folder_for(job)
        put(folder/'trace.json',tr);put(folder/'timing.json',rp);np.savez_compressed(OUT/folder/'features.npz',counts=raw,delta=dd,logits=np.empty((0,16)) if q is None else q)
        _,_,pr=inputs(job)
        for step,v in snaps.items():
            put(folder/f'graph{step}.json',v['graph']);row=dict(**job,steps=step,**v['metrics'],first_stage_steps=pr['steps'],total_attempts=pr['steps']+step,effective_switches=sum(x['candidates']['action'][x['effective']]==4 for x in tr[:step]),connectivity_rejections=sum(x['connectivity_rejected'] for x in tr[:step]))
            rows.append(row);put(folder/f'report{step}.json',row)
        print('confirmation suffix generated',job,rows[-1]['primary_error'],round(time.monotonic()-tic,1),flush=True)
    put('completion.json',dict(completed=True,new_suffix_chains=len(jobs),scientific_opportunities=2048*len(jobs),new_training_updates=0,seconds=time.monotonic()-tic));tic=time.monotonic();errors={};full=0;replayed=0
    for job in jobs:
        ds=job['dataset'];folder=OUT/folder_for(job);tr=b.read(folder/'trace.json');z=np.load(folder/'features.npz');g,target,_=inputs(job);edges,nb=r5.topology(g);names=sorted(edges);cols=b.read(fit.CACHE/ds/'schema.json')['columns'];m=model(job)
        if m is not None:
            with torch.no_grad():
                xx=torch.from_numpy(np.stack([encode(raw,dd,g.num_nodes,cols) for raw,dd in zip(z['counts'],z['delta'])]));q=m(xx.double() if job['arm']=='F' else xx).numpy()
            err=float(abs(q-z['logits']).max());assert err<=(1e-9 if job['arm']=='F' else 2e-5);errors[str(folder_for(job))]=err
        for i,v in enumerate(tr):
            assert time.monotonic()-tic<3600
            vals=v['candidates'];raw=z['counts'][i];dd=z['delta'][i]
            if i%32==0:
                graph=r5.graph_from(g,edges);_,fresh=p.cache.context(graph,cols);assert np.array_equal(fresh['raw'],raw);assert np.array_equal(Consequences(graph,cols,fresh['raw']).differences(t.proposals(graph,vals)),dd);full+=1
            chosen=v['chosen'];expected=choose_random(vals,job['arm'],v['proposal_seed']) if m is None else int(q[i].argmax());assert chosen==expected
            r5.edit(edges,nb,names,vals,chosen);rejected=not r5.connected(nb);assert rejected==v['connectivity_rejected']
            if rejected:r5.edit(edges,nb,names,vals,chosen,True)
            effective=0 if rejected else chosen;assert effective==v['effective'] and r5.connected(nb)
            if i+1<len(tr):assert np.array_equal(raw+dd[effective],z['counts'][i+1])
            if i+1 in [512,1024,2048]:
                graph=r5.graph_from(g,edges);stored=b.restore_graph(b.read(folder/f'graph{i+1}.json'))
                for f in names:assert edges[f]=={tuple(sorted(e)) for e in stored.family_edges[f]} and np.array_equal(graph.family_degrees(f),g.family_degrees(f))
                report=b.read(folder/f'report{i+1}.json')
                for k,value in prefix.stats(ds,graph,target).items():assert report[k]==value
            replayed+=1
        print('confirmation suffix accepted',job,round(time.monotonic()-tic,1),flush=True)
    decisions=[];fitmeans=[]
    for sid in SEEDS:
        rr={(v['arm'],v['repeat'],v['variant']):v for v in rows if v['seed']==sid and v['producer']=='C' and v['steps']==2048}
        for repeat in [1,2,3]:
            if ('F',repeat,'coverage') not in rr:
                decisions.append(dict(dataset='imdb',seed=sid,repeat=repeat,passed=False,reason='unavailable_prefix'));continue
            cov=rr['F',repeat,'coverage']['primary_error'];base=rr['F',repeat,'base']['primary_error'];q=rr['Q',repeat,'fixed']['primary_error']
            n=rr['N',0,'random']['primary_error'];ns=rr['NS',0,'random']['primary_error'];zz=zero[f'imdb_{sid}_C']['primary_error']
            decisions.append(dict(dataset='imdb',seed=sid,repeat=repeat,F_coverage=cov,F_base=base,Q=q,N=n,NS=ns,zero=zz,
                                  passed=cov<min(base,n,ns,zz),better_than_Q=cov<q))
    aggregate=[]
    for arm,variant in [('F','base'),('F','coverage'),('Q','fixed')]:
        values=[]
        for repeat in [1,2,3]:
            selected=[v['primary_error'] for v in rows if v['producer']=='C' and v['arm']==arm and v['variant']==variant and v['repeat']==repeat and v['steps']==2048]
            if len(selected)==2:
                value=float(np.mean(selected));values.append(value)
                fitmeans.append(dict(dataset='imdb',arm=arm,variant=variant,repeat=repeat,starts=2,primary_error=value))
        aggregate.append(dict(dataset='imdb',arm=arm,variant=variant,complete_repeats=len(values),
                              mean=float(np.mean(values)) if values else None,
                              sample_sd=float(np.std(values,ddof=1)) if len(values)>1 else None,
                              unit='paired bank mean over two fresh starts; old bank and MLP shared'))
    passed=len(decisions)==6 and all(v['passed'] for v in decisions)
    put('summary.json',dict(passed=passed,decisions=decisions,fit_means=fitmeans,aggregate=aggregate,rows=rows,
                           unavailable_prefixes=unavailable,Q_not_part_of_core_gate=True,not_cross_network_generalization=True,
                           base_banks_reused=True,augmentation_banks_new=True,new_original_MLP_fits=0))
    if rows:
        fields=list(dict.fromkeys(k for row in rows for k in row))
        with (OUT/'per_output.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    verify();outputs={str(f):b.sha(f) for f in OUT.glob('outputs/*/*/*/graph*.json')}
    put('acceptance.json',dict(passed=True,scientific_core_gate_passed=passed,replayed_opportunities=replayed,full_state_recounts=full,snapshot_metric_recomputations=3*len(jobs),logit_errors=errors,outputs=outputs,seconds=time.monotonic()-tic))

if __name__=='__main__':
    try:main()
    except BaseException as exc:put('failure.json',dict(type=type(exc).__name__,message=str(exc)));raise
