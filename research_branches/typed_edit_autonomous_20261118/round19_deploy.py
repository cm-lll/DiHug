"""Uniform-replay scorer on the unchanged R16/R18 shared starts and streams."""
import round19_train as train
from round01 import t,b,np,torch,time,Path,ROOT,SRC,json,csv
from potential import encode
from candidate_consequences import Consequences
p=train.prior;r5=p.r5;cache=p.cache;OUT=ROOT/'results/typed_edit_autonomous_20261118/r19_deploy'
JOBS=[dict(dataset=ds,start_id=sid,arm=f'U_{seed}') for ds in train.DATASETS for sid in [106,107] for seed in train.SEEDS]
original_model=p.model


def put(name,v):
    path=OUT/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')


def model(ds,arm):
    if not arm.startswith('U_'):return original_model(ds,arm)
    m=train.model(ds);m.load_state_dict(torch.load(train.OUT/f"fits/{ds}/{arm[2:]}/checkpoint.pt",map_location='cpu',weights_only=False)['state_dict']);return m.eval()


def verify():
    train.verify()
    for path,h in b.read(OUT/'manifest.json')['files'].items():assert b.sha(path)==h,path


def main():
    b.setup();train.verify();assert b.read(train.OUT/'acceptance.json')['passed'];OUT.mkdir(parents=True,exist_ok=False)
    files=[Path(__file__),SRC/'ROUND19.md',train.OUT/'acceptance.json',p.OUT/'acceptance.json',p.OUT/'summary.json',p.old.OUT/'summary.json']
    files.extend(train.OUT/f'fits/{ds}/{s}/checkpoint.pt' for ds in train.DATASETS for s in train.SEEDS)
    put('manifest.json',dict(files={str(f):b.sha(f) for f in files},jobs=JOBS,new_graphs=12,scientific_opportunities=6144,new_training_updates=0))
    p.model=model;error=0.;cost=30.
    for ds in train.DATASETS:
        trace,raw,dd,q,_,rp=p.chain(dict(dataset=ds,start_id=106,arm='S_812'),8)
        folder=p.OUT/'outputs/imdb/start106/S_812' if ds=='imdb' else p.old.OUT/'outputs/start106/S_812'
        old=b.read(folder/'trace.json')[:8]
        if ds=='pubmed':trace=[{k:v for k,v in row.items() if not k.startswith('components_')} for row in trace]
        assert trace==old;error=max(error,float(abs(q-np.load(folder/'features.npz')['logits'][:8]).max()));assert error<=2e-5
    for job in JOBS:
        _,_,_,_,_,rp=p.chain(job,8);cost+=2*(rp['init_seconds']+64*rp['loop_seconds']+2*rp['metric_seconds'])
        put(f"engineering/{job['dataset']}_{job['start_id']}_{job['arm']}.json",rp)
    put('engineering.json',dict(passed=cost<=1200,cost_bound_seconds=cost,engineering_opportunities=112,previous_score_error=error));assert cost<=1200
    rows=[];verify();tic=time.monotonic()
    for job in JOBS:
        assert time.monotonic()-tic<1200
        trace,raw,dd,q,snaps,rp=p.chain(job,512);folder=Path('outputs')/job['dataset']/f"start{job['start_id']}"/job['arm'];put(folder/'trace.json',trace)
        np.savez_compressed(OUT/folder/'features.npz',counts=raw,delta=dd,logits=q);put(folder/'timing.json',rp)
        for step,v in snaps.items():
            put(folder/f'graph{step}.json',v['graph']);row=dict(**job,steps=step,**v['metric'],noop_choices=sum(x['candidates']['action'][x['chosen']]==0 for x in trace[:step]),connectivity_rejections=sum(x['connectivity_rejected'] for x in trace[:step]),effective_switches=sum(x['candidates']['action'][x['effective']]==4 for x in trace[:step]))
            rows.append(row);put(folder/f'report{step}.json',row)
        print('uniform generated',job,round(time.monotonic()-tic,2),flush=True)
    put('completion.json',dict(completed=True,new_graphs=12,scientific_opportunities=6144,snapshot_evaluations=24,new_training_updates=0,seconds=time.monotonic()-tic))
    tic=time.monotonic();replayed=0;fullchecks=0;error=0.
    for job in JOBS:
        ds=job['dataset'];sid=job['start_id'];arm=job['arm'];folder=OUT/'outputs'/ds/f'start{sid}'/arm
        trace=b.read(folder/'trace.json');saved=np.load(folder/'features.npz');initial=b.restore_graph(b.read(b.RUN/f'prefixes/{ds}/seed{sid}/initial.json'))
        edges,nb=r5.topology(initial);names=sorted(edges);_,_,columns,_=p.spec(ds);m=model(ds,arm)
        xx=np.stack([encode(raw,dd,initial.num_nodes,columns) for raw,dd in zip(saved['counts'],saved['delta'])]);_,q,_=b.c.eval_model(m,xx,np.zeros(512,dtype=np.int64),'cpu')
        error=max(error,float(abs(q-saved['logits']).max()));assert error<=2e-5
        for i,record in enumerate(trace):
            assert time.monotonic()-tic<1500
            values=record['candidates'];raw=saved['counts'][i];dd=saved['delta'][i];before=p.components(nb);assert before==record['components_before']
            if i%32==0:
                graph=r5.graph_from(initial,edges);_,actual=cache.context(graph,columns);assert np.array_equal(actual['raw'],raw)
                independent=Consequences(graph,columns,actual['raw']).differences(t.proposals(graph,values));assert np.array_equal(independent,dd);fullchecks+=1
            chosen=record['chosen'];assert chosen==int(saved['logits'][i].argmax());r5.edit(edges,nb,names,values,chosen);after=p.components(nb)
            reject=after>before;assert reject==record['connectivity_rejected']
            if reject:r5.edit(edges,nb,names,values,chosen,reverse=True);after=p.components(nb)
            effective=0 if reject else chosen;assert record['effective']==effective and record['components_after']==after and after<=before
            if i+1<len(trace):assert np.array_equal(raw+dd[effective],saved['counts'][i+1])
            if i+1 in [128,512]:
                graph=r5.graph_from(initial,edges);expected=b.restore_graph(b.read(folder/f'graph{i+1}.json'))
                for f in names:assert edges[f]=={tuple(sorted(e)) for e in expected.family_edges[f]} and np.array_equal(graph.family_degrees(f),initial.family_degrees(f))
                row=b.read(folder/f'report{i+1}.json')
                for key,val in p.value(ds,graph).items():assert row[key]==val
            replayed+=1
        print('uniform accepted',job,round(time.monotonic()-tic,2),flush=True)
    previous=b.read(p.OUT/'summary.json');zero=previous['zero'];controls=previous['rows']+[dict(dataset='pubmed',**v) for v in b.read(p.old.OUT/'summary.json')['rows']];decisions=[]
    for row in rows:
        if row['steps']!=512:continue
        ds=row['dataset'];sid=row['start_id'];seed=int(row['arm'][2:]);cc={v['arm']:v for v in controls if v['dataset']==ds and v['start_id']==sid and v['steps']==512}
        e=row['primary_error'];decisions.append(dict(dataset=ds,start_id=sid,seed=seed,U=e,S=cc[f'S_{seed}']['primary_error'],N=cc['N']['primary_error'],NS=cc['NS']['primary_error'],Q=cc['Q']['primary_error'],zero=zero[f'{ds}_{sid}']['primary_error'],structural_passed=e<min(zero[f'{ds}_{sid}']['primary_error'],cc['N']['primary_error'],cc['NS']['primary_error']),supervision_vs_Q_passed=e<cc['Q']['primary_error'],better_than_S=e<cc[f'S_{seed}']['primary_error']))
    agg=[]
    for ds in train.DATASETS:
        for step in [128,512]:
            perfit=[float(np.mean([v['primary_error'] for v in rows if v['dataset']==ds and v['steps']==step and v['arm']==f'U_{s}'])) for s in train.SEEDS]
            agg.append(dict(dataset=ds,steps=step,optimization_repeats=3,outputs_per_repeat=2,per_fit_mean=perfit,mean=float(np.mean(perfit)),fit_sample_sd=float(np.std(perfit,ddof=1))))
    put('summary.json',dict(rows=rows,decisions=decisions,aggregation=agg,structural_passed=all(v['structural_passed'] for v in decisions),supervision_vs_Q_passed=all(v['supervision_vs_Q_passed'] for v in decisions)))
    with (OUT/'per_output.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    verify();put('acceptance.json',dict(passed=True,replayed_opportunities=replayed,sampled_full_state_recounts=fullchecks,snapshot_metric_recomputations=24,logit_error=error,seconds=time.monotonic()-tic));print('uniform decisions',decisions,flush=True)


if __name__=='__main__':
    try:main()
    except BaseException as exc:put('failure.json',dict(type=type(exc).__name__,message=str(exc)));raise
