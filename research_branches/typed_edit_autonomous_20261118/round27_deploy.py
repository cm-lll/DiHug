"""Four fixed chains for deterministic inverse-CE potentials."""
import round26_nocache as fit
import round24_complete as imdb
import round20_deploy as old
from potential import ComponentState,encode
from candidate_consequences import Consequences
from round01 import t,b,np,torch,time,Path,ROOT,SRC,json,csv
p=old.p;r5=p.r5;cache=p.cache
OUT=ROOT/'results/typed_edit_autonomous_20261118/r27_deploy'
JOBS=[dict(dataset=ds,start_id=s) for ds,ss in [('pubmed',[106,107]),('imdb',[108,109])] for s in ss]

def put(name,value):
    f=OUT/name;f.parent.mkdir(parents=True,exist_ok=True);f.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')

def inputs(job):
    ds=job['dataset'];sid=job['start_id']
    if ds=='imdb':return imdb.inputs(dict(start_id=sid,producer='C'))
    folder=b.RUN/f'prefixes/{ds}/seed{sid}';binding=b.read(folder/'binding.json')
    assert b.sha(folder/'initial.json')==binding['initial_sha256'] and b.sha(folder/'target.json')==binding['target_sha256']
    return b.restore_graph(b.read(folder/'initial.json')),np.array(b.read(folder/'target.json')),dict(steps=binding['transition_step'])

def model(ds):
    m=fit.model(ds);m.load_state_dict(torch.load(fit.OUT/f'fits/{ds}/checkpoint.pt',map_location='cpu',weights_only=False)['state_dict']);return m.eval()

def metrics(ds,g,target):return imdb.prefix.stats(g,target) if ds=='imdb' else p.value(ds,g)

def verify():
    fit.verify()
    for f,h in b.read(OUT/'manifest.json')['files'].items():assert b.sha(f)==h,f

@torch.no_grad()
def chain(job,steps):
    tic=time.monotonic();ds=job['dataset'];sid=job['start_id'];g,target,pr=inputs(job);_,_,columns,_=p.spec(ds)
    tracker=ComponentState(g,columns);assert tracker.n_components==1;state=t.a.CpuAuditState(g,target,'cpu',use_triton=False);degree=state.family_degree.clone();m=model(ds)
    init=time.monotonic()-tic;loop=0.;evaluation=0.;trace=[];raws=[];deltas=[];scores=[];snaps={}
    for step in range(steps):
        tic=time.monotonic();seed=(2026111823000 if ds=='imdb' else 2026111816000)+100000*sid+step
        batch=state.random_candidates(16,torch.Generator().manual_seed(seed),role_phase=False,targeted_fraction=0.,closure_fraction=.5 if ds=='imdb' else 0.,candidate_mode='legacy_pool')
        vals=t.a.serialize(batch);raw=tracker.counts.copy();dd=tracker.differences(t.proposals(g,vals));q=m(torch.from_numpy(encode(raw,dd,g.num_nodes,columns)).double()[None]).numpy()[0]
        assert np.isfinite(q).all();chosen=int(q.argmax());reject=tracker.guarded(vals,chosen);effective=0 if reject else chosen;state.commit(batch,torch.tensor(effective))
        assert torch.equal(degree,state.family_degree) and tracker.n_components==1
        trace.append(dict(step=step,proposal_seed=seed,candidates=vals,chosen=chosen,effective=effective,connectivity_rejected=reject));raws.append(raw);deltas.append(dd);scores.append(q);loop+=time.monotonic()-tic
        if step+1 in [128,steps]:
            tic=time.monotonic();graph=state.to_graph(g);graph.validate()
            for f in tracker.names:assert tracker.edges[f]=={tuple(sorted(e)) for e in graph.family_edges[f]} and np.array_equal(graph.family_degrees(f),g.family_degrees(f))
            _,fresh=cache.context(graph,columns);assert np.array_equal(fresh['raw'],tracker.counts)
            snaps[step+1]=dict(graph=b.graph_payload(graph),metrics=metrics(ds,graph,target));evaluation+=time.monotonic()-tic
    return trace,np.stack(raws),np.stack(deltas),np.stack(scores),snaps,dict(init_seconds=init,loop_seconds=loop,metric_seconds=evaluation)

def main():
    b.setup();fit.verify();assert b.read(fit.OUT/'acceptance.json')['passed'] and b.read(fit.OUT/'summary.json')['passed'];OUT.mkdir(parents=True,exist_ok=False)
    files=[Path(__file__),SRC/'ROUND27.md',fit.OUT/'acceptance.json',fit.OUT/'summary.json',imdb.OUT/'summary.json',imdb.OUT/'acceptance.json',old.OUT/'summary.json',old.OUT/'acceptance.json']
    for ds in ['pubmed','imdb']:files.append(fit.OUT/f'fits/{ds}/checkpoint.pt')
    put('manifest.json',dict(files={str(f):b.sha(f) for f in files},jobs=JOBS,new_training_updates=0,new_graphs=4,scientific_opportunities=2048,deterministic_fits_per_reference=1,development_not_untouched_confirmation=True))
    parity=0.
    for ds in ['pubmed','imdb']:
        if ds=='imdb':tr,_,_,q,_,_=imdb.chain(dict(start_id=108,producer='C',arm='P_812'),8);folder=imdb.OUT/'outputs/108_C/P_812'
        else:
            original=p.model;p.model=old.model
            try:tr,_,_,q,_,_=p.chain(dict(dataset=ds,start_id=106,arm='P_812'),8)
            finally:p.model=original
            folder=old.OUT/'outputs/pubmed/start106/P_812'
        assert tr==b.read(folder/'trace.json')[:8];parity=max(parity,float(abs(q-np.load(folder/'features.npz')['logits'][:8]).max()));assert parity<=2e-5
    cost=20.;zero={}
    for job in JOBS:
        tr,_,_,_,_,rp=chain(job,8);ds=job['dataset'];sid=job['start_id'];folder=imdb.OUT/f'outputs/{sid}_C/Q' if ds=='imdb' else p.OUT/f'outputs/pubmed/start{sid}/Q'
        assert tr[0]['candidates']==b.read(folder/'trace.json')[0]['candidates'];cost+=2*(rp['init_seconds']+64*rp['loop_seconds']+2*rp['metric_seconds'])
        g,target,_=inputs(job);zero[f'{ds}_{sid}']=metrics(ds,g,target);put(f'engineering/{ds}_{sid}.json',rp)
    put('engineering.json',dict(passed=cost<=600,cost_bound_seconds=cost,old_logit_error=parity,engineering_opportunities=48));assert cost<=600;put('zero.json',zero);verify();tic=time.monotonic();rows=[]
    for job in JOBS:
        assert time.monotonic()-tic<600
        tr,raw,dd,q,snaps,rp=chain(job,512);folder=Path('outputs')/job['dataset']/str(job['start_id']);put(folder/'trace.json',tr);put(folder/'timing.json',rp);np.savez_compressed(OUT/folder/'features.npz',counts=raw,delta=dd,logits=q)
        _,_,pr=inputs(job)
        for step,v in snaps.items():
            put(folder/f'graph{step}.json',v['graph']);row=dict(**job,steps=step,**v['metrics'],first_stage_steps=pr['steps'],total_attempts=pr['steps']+step,effective_switches=sum(x['candidates']['action'][x['effective']]==4 for x in tr[:step]),connectivity_rejections=sum(x['connectivity_rejected'] for x in tr[:step]));rows.append(row);put(folder/f'report{step}.json',row)
        print('F generated',job,rows[-1]['primary_error'],flush=True)
    put('completion.json',dict(completed=True,new_graphs=4,scientific_opportunities=2048,new_training_updates=0,seconds=time.monotonic()-tic));tic=time.monotonic();error=0.;full=0;replayed=0
    for job in JOBS:
        ds=job['dataset'];sid=job['start_id'];folder=OUT/'outputs'/ds/str(sid);tr=b.read(folder/'trace.json');z=np.load(folder/'features.npz');g,target,_=inputs(job);edges,nb=r5.topology(g);names=sorted(edges);_,_,cols,_=p.spec(ds);m=model(ds)
        with torch.no_grad():
            xx=np.stack([encode(raw,dd,g.num_nodes,cols) for raw,dd in zip(z['counts'],z['delta'])]);q=m(torch.from_numpy(xx).double()).numpy()
        error=max(error,float(abs(q-z['logits']).max()));assert error<=1e-9
        for i,v in enumerate(tr):
            assert time.monotonic()-tic<900
            vals=v['candidates'];raw=z['counts'][i];dd=z['delta'][i]
            if i%32==0:
                graph=r5.graph_from(g,edges);_,fresh=cache.context(graph,cols);assert np.array_equal(fresh['raw'],raw);independent=Consequences(graph,cols,fresh['raw']).differences(t.proposals(graph,vals));assert np.array_equal(independent,dd);full+=1
            chosen=v['chosen'];assert chosen==int(q[i].argmax());r5.edit(edges,nb,names,vals,chosen);reject=not r5.connected(nb);assert reject==v['connectivity_rejected']
            if reject:r5.edit(edges,nb,names,vals,chosen,True)
            effective=0 if reject else chosen;assert effective==v['effective'] and r5.connected(nb)
            if i+1<len(tr):assert np.array_equal(raw+dd[effective],z['counts'][i+1])
            if i+1 in [128,512]:
                graph=r5.graph_from(g,edges);stored=b.restore_graph(b.read(folder/f'graph{i+1}.json'))
                for f in names:assert edges[f]=={tuple(sorted(e)) for e in stored.family_edges[f]} and np.array_equal(graph.family_degrees(f),g.family_degrees(f))
                report=b.read(folder/f'report{i+1}.json')
                for key,value in metrics(ds,graph,target).items():assert report[key]==value
            replayed+=1
    decisions=[]
    for row in rows:
        if row['steps']!=512:continue
        ds=row['dataset'];sid=row['start_id'];source=b.read(imdb.OUT/'summary.json') if ds=='imdb' else b.read(old.OUT/'summary.json');previous=[v for v in source['decisions'] if v['start_id']==sid and v.get('dataset',ds)==ds];assert len(previous)==3
        c=previous[0];e=row['primary_error'];decisions.append(dict(dataset=ds,start_id=sid,F=e,Q=c['Q'],N=c['N'],NS=c['NS'],zero=c['zero'],all_P=[v['P'] for v in previous],structural_passed=e<min(c['zero'],c['N'],c['NS']),supervision_vs_Q_passed=e<c['Q'],better_than_all_P=all(e<v['P'] for v in previous)))
    put('summary.json',dict(rows=rows,decisions=decisions,structural_passed=all(v['structural_passed'] for v in decisions),supervision_vs_Q_passed=all(v['supervision_vs_Q_passed'] for v in decisions),training_seed_variance_not_estimated=True))
    with (OUT/'per_output.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    verify();put('acceptance.json',dict(passed=True,replayed_opportunities=replayed,sampled_full_state_recounts=full,snapshot_metric_recomputations=8,logit_error=error,seconds=time.monotonic()-tic));print('F decisions',decisions,flush=True)

if __name__=='__main__':
    try:main()
    except BaseException as exc:put('failure.json',dict(type=type(exc).__name__,message=str(exc)));raise
