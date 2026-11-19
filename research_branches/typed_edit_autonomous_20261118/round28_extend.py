"""Matched fixed-budget continuation of all four second-stage policies."""
import round27_finalize as accepted
import round27_deploy as prev
from potential import ComponentState,encode
from candidate_consequences import Consequences
from round01 import t,b,np,torch,time,Path,ROOT,SRC,json,csv
OUT=ROOT/'results/typed_edit_autonomous_20261118/r28_extend'
JOBS=[dict(**job,arm=arm) for job in prev.JOBS for arm in ['F','Q','N','NS']]

def put(name,v):
    f=OUT/name;f.parent.mkdir(parents=True,exist_ok=True);f.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')

def source(job):
    ds=job['dataset'];sid=job['start_id'];arm=job['arm']
    if arm=='F':return prev.OUT/'outputs'/ds/str(sid)
    if ds=='imdb':return prev.imdb.OUT/f'outputs/{sid}_C/{arm}'
    if arm=='Q':return prev.p.OUT/f'outputs/pubmed/start{sid}/Q'
    return prev.p.old.OUT/f'outputs/start{sid}/{arm}'

def model(job):return prev.model(job['dataset']) if job['arm']=='F' else prev.p.model(job['dataset'],job['arm'])

def verify():
    accepted.verify()
    for f,h in b.read(OUT/'manifest.json')['files'].items():assert b.sha(f)==h,f

@torch.no_grad()
def chain(job,steps,offset=512):
    tic=time.monotonic();ds=job['dataset'];sid=job['start_id'];arm=job['arm'];original,target,pr=prev.inputs(job)
    g=b.restore_graph(b.read(source(job)/'graph512.json')) if offset else original
    g.validate()
    for f in g.family_edges:assert np.array_equal(g.family_degrees(f),original.family_degrees(f))
    _,_,columns,_=prev.p.spec(ds);tracker=ComponentState(g,columns);assert tracker.n_components==1
    state=t.a.CpuAuditState(g,target,'cpu',use_triton=False);degree=state.family_degree.clone();m=model(job)
    init=time.monotonic()-tic;loop=0.;evaluation=0.;trace=[];raws=[];deltas=[];scores=[];snaps={}
    for local_step in range(steps):
        tic=time.monotonic();step=offset+local_step;seed=(2026111823000 if ds=='imdb' else 2026111816000)+100000*sid+step
        batch=state.random_candidates(16,torch.Generator().manual_seed(seed),role_phase=False,targeted_fraction=0.,closure_fraction=.5 if ds=='imdb' else 0.,candidate_mode='legacy_pool')
        vals=t.a.serialize(batch);raw=tracker.counts.copy();dd=tracker.differences(t.proposals(g,vals))
        if m is None:
            available=np.arange(16) if arm=='N' else np.flatnonzero(np.array(vals['action'])==4)
            chosen=int(available[int(torch.randint(0,len(available),(1,),generator=torch.Generator().manual_seed(seed+5000000)))]) if len(available) else 0;q=None
        else:
            x=torch.from_numpy(encode(raw,dd,g.num_nodes,columns))[None];x=x.double() if arm=='F' else x;q=m(x).numpy()[0];assert np.isfinite(q).all();chosen=int(q.argmax())
        reject=tracker.guarded(vals,chosen);effective=0 if reject else chosen;state.commit(batch,torch.tensor(effective));assert torch.equal(degree,state.family_degree) and tracker.n_components==1
        trace.append(dict(step=step,proposal_seed=seed,candidates=vals,chosen=chosen,effective=effective,connectivity_rejected=reject));raws.append(raw);deltas.append(dd);scores.append(q);loop+=time.monotonic()-tic
        if step+1 in [1024,2048,offset+steps]:
            tic=time.monotonic();graph=state.to_graph(g);graph.validate()
            for f in tracker.names:assert tracker.edges[f]=={tuple(sorted(e)) for e in graph.family_edges[f]} and np.array_equal(graph.family_degrees(f),original.family_degrees(f))
            _,fresh=prev.cache.context(graph,columns);assert np.array_equal(fresh['raw'],tracker.counts)
            snaps[step+1]=dict(graph=b.graph_payload(graph),metrics=prev.metrics(ds,graph,target));evaluation+=time.monotonic()-tic
    return trace,np.stack(raws),np.stack(deltas),None if m is None else np.stack(scores),snaps,dict(init_seconds=init,loop_seconds=loop,metric_seconds=evaluation)

def main():
    b.setup();accepted.verify();assert b.read(accepted.OUT/'acceptance.json')['passed'];OUT.mkdir(parents=True,exist_ok=False)
    files=[Path(__file__),SRC/'ROUND28.md',accepted.OUT/'acceptance.json',accepted.OUT/'summary.json']
    for job in JOBS:
        files.extend(source(job)/name for name in ['graph512.json','report512.json','trace.json','features.npz'])
    put('manifest.json',dict(files={str(f):b.sha(f) for f in files},jobs=JOBS,new_training_updates=0,continued_chains=16,new_scientific_opportunities=24576,primary_endpoint=2048))
    cost=30.;parity={};zero=b.read(prev.OUT/'zero.json')
    for job in JOBS:
        tr,_,_,q,_,_=chain(job,8,0);wanted=b.read(source(job)/'trace.json')[:8]
        for a,c in zip(tr,wanted):
            for k in a:assert a[k]==c[k],(job,k)
        error=0. if q is None else float(abs(q-np.load(source(job)/'features.npz')['logits'][:8]).max())
        assert error<=(1e-9 if job['arm']=='F' else 2e-5)
        _,_,_,_,_,rp=chain(job,8);cost+=2*(rp['init_seconds']+192*rp['loop_seconds']+2*rp['metric_seconds']);key=f"{job['dataset']}_{job['start_id']}_{job['arm']}";parity[key]=error;put(f'engineering/{key}.json',rp)
        original,target,_=prev.inputs(job);g=b.restore_graph(b.read(source(job)/'graph512.json'));expected=b.read(source(job)/'report512.json')
        for k,v in prev.metrics(job['dataset'],g,target).items():
            if k in expected:assert expected[k]==v
    put('engineering.json',dict(passed=cost<=1800,cost_bound_seconds=cost,prior_score_errors=parity,engineering_opportunities=256));assert cost<=1800;verify();tic=time.monotonic();rows=[]
    for job in JOBS:
        assert time.monotonic()-tic<1800
        tr,raw,dd,q,snaps,rp=chain(job,1536);folder=Path('outputs')/job['dataset']/str(job['start_id'])/job['arm'];put(folder/'trace.json',tr);put(folder/'timing.json',rp)
        np.savez_compressed(OUT/folder/'features.npz',counts=raw,delta=dd,logits=np.empty((0,16)) if q is None else q);_,_,pr=prev.inputs(job);old_trace=b.read(source(job)/'trace.json')
        for step,v in snaps.items():
            put(folder/f'graph{step}.json',v['graph']);combined=old_trace+tr[:step-512]
            row=dict(**job,steps=step,**v['metrics'],first_stage_steps=pr['steps'],total_attempts=pr['steps']+step,effective_switches=sum(x['candidates']['action'][x['effective']]==4 for x in combined),connectivity_rejections=sum(x['connectivity_rejected'] for x in combined));rows.append(row);put(folder/f'report{step}.json',row)
        print('extended',job,rows[-1]['primary_error'],round(time.monotonic()-tic,1),flush=True)
    put('completion.json',dict(completed=True,new_graphs=0,continued_chains=16,new_endpoint_outputs=16,scientific_opportunities=24576,new_training_updates=0,seconds=time.monotonic()-tic));tic=time.monotonic();errors={};full=0;replayed=0
    for job in JOBS:
        ds=job['dataset'];sid=job['start_id'];arm=job['arm'];folder=OUT/'outputs'/ds/str(sid)/arm;tr=b.read(folder/'trace.json');z=np.load(folder/'features.npz');original,target,_=prev.inputs(job);g=b.restore_graph(b.read(source(job)/'graph512.json'));edges,nb=prev.r5.topology(g);names=sorted(edges);_,_,cols,_=prev.p.spec(ds);m=model(job)
        if m is not None:
            with torch.no_grad():
                xx=torch.from_numpy(np.stack([encode(raw,dd,g.num_nodes,cols) for raw,dd in zip(z['counts'],z['delta'])]));q=m(xx.double() if arm=='F' else xx).numpy()
            err=float(abs(q-z['logits']).max());assert err<=(1e-9 if arm=='F' else 2e-5);errors[f'{ds}_{sid}_{arm}']=err
        for i,v in enumerate(tr):
            assert time.monotonic()-tic<1800
            vals=v['candidates'];raw=z['counts'][i];dd=z['delta'][i]
            if i%32==0:
                graph=prev.r5.graph_from(g,edges);_,fresh=prev.cache.context(graph,cols);assert np.array_equal(fresh['raw'],raw);assert np.array_equal(Consequences(graph,cols,fresh['raw']).differences(t.proposals(graph,vals)),dd);full+=1
            chosen=v['chosen']
            if m is not None:assert chosen==int(q[i].argmax())
            else:
                av=np.arange(16) if arm=='N' else np.flatnonzero(np.array(vals['action'])==4);expected=int(av[int(torch.randint(0,len(av),(1,),generator=torch.Generator().manual_seed(v['proposal_seed']+5000000)))]) if len(av) else 0;assert chosen==expected
            prev.r5.edit(edges,nb,names,vals,chosen);reject=not prev.r5.connected(nb);assert reject==v['connectivity_rejected']
            if reject:prev.r5.edit(edges,nb,names,vals,chosen,True)
            effective=0 if reject else chosen;assert effective==v['effective'] and prev.r5.connected(nb)
            if i+1<len(tr):assert np.array_equal(raw+dd[effective],z['counts'][i+1])
            step=v['step']+1
            if step in [1024,2048]:
                graph=prev.r5.graph_from(g,edges);stored=b.restore_graph(b.read(folder/f'graph{step}.json'))
                for f in names:assert edges[f]=={tuple(sorted(e)) for e in stored.family_edges[f]} and np.array_equal(graph.family_degrees(f),original.family_degrees(f))
                report=b.read(folder/f'report{step}.json')
                for key,value in prev.metrics(ds,graph,target).items():assert report[key]==value
            replayed+=1
        print('extension replayed',job,round(time.monotonic()-tic,1),flush=True)
    decisions=[]
    for job in prev.JOBS:
        ds=job['dataset'];sid=job['start_id'];rr={r['arm']:r for r in rows if r['dataset']==ds and r['start_id']==sid and r['steps']==2048};e=rr['F']['primary_error'];z=zero[f'{ds}_{sid}']['primary_error']
        decisions.append(dict(**job,F=e,Q=rr['Q']['primary_error'],N=rr['N']['primary_error'],NS=rr['NS']['primary_error'],zero=z,structural_passed=e<min(z,rr['N']['primary_error'],rr['NS']['primary_error']),supervision_vs_Q_passed=e<rr['Q']['primary_error']))
    put('summary.json',dict(rows=rows,decisions=decisions,structural_passed=all(v['structural_passed'] for v in decisions),supervision_vs_Q_passed=all(v['supervision_vs_Q_passed'] for v in decisions),training_seed_variance_not_estimated=True))
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with (OUT/'per_output.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    verify();put('acceptance.json',dict(passed=True,replayed_opportunities=replayed,sampled_full_state_recounts=full,snapshot_metric_recomputations=32,logit_errors=errors,seconds=time.monotonic()-tic));print('extension decisions',decisions,flush=True)

if __name__=='__main__':
    try:main()
    except BaseException as exc:put('failure.json',dict(type=type(exc).__name__,message=str(exc)));raise
