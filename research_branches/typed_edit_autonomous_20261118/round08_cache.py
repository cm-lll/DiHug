"""Reconstruct candidate consequences on exactly the existing cached records."""
import round06_cache_fast as c
from candidate_consequences import Consequences
from round01 import t,b,np,torch,time,Path,ROOT,SRC,json
OUT=ROOT/'results/typed_edit_autonomous_20261118/r08_cache'
mixed=c.mixed


def put(name,v):
    p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')


def verify():
    c.verify()
    for p,h in b.read(OUT/'manifest.json')['files'].items():assert b.sha(p)==h,p


def delta(g,ps,columns,raw,check=False):
    tracker=Consequences(g,columns,raw);d=tracker.differences(ps)
    for f in tracker.names:assert tracker.edges[f]=={tuple(sorted(e)) for e in g.family_edges[f]}
    if check:
        _,old=c.context(g,columns);assert old['raw']==raw
        k=next((k for k,p in enumerate(ps) if p.action!='noop'),0)
        tracker.proposal(ps[k]);after=c.r5.graph_from(g,tracker.edges);_,new=c.context(after,columns)
        assert np.array_equal(np.array(new['raw'])-np.array(raw),d[k])
        tracker.proposal(ps[k],True);assert np.array_equal(tracker.counts,raw)
    return (d/tracker.scale).astype(np.float32)


def pair(kind,j,columns,engineering=False):
    graph=b.graph('pubmed');start=time.monotonic();rows=[]
    if kind=='train':
        terminal=b.restore_graph(b.read(b.RUN/f'cache/pubmed/pair{j}/bridge_terminal.json'))
        seed=2026111600+10000019*j;source=np.load(b.RUN/'cache/pubmed/data.npz')
    else:
        prior=b.MarginalGraphPrior.from_training_graph(graph);role=b.ExchangeableRoleDegreePrior.from_training_graph(graph)
        terminal=b.FamilyCountCondition(prior,role.family_counts(prior.family_endpoints)).sample_graph(720000000+2*j,mode='binomial')
        seed=710000000+100003*j;source=np.load(t.EPOCH/'validation.npz')
    raw=b.read(c.OUT/f'mixed/{kind}_{j}/counts.json');original=mixed.mixed_edit_candidate_matrix
    def capture(g,ps,*args,**kwargs):
        x=original(g,ps,*args,**kwargs);i=len(rows)
        rows.append(delta(g,ps,columns,raw[i]['raw'],check=engineering and i%20==0));return x
    mixed.mixed_edit_candidate_matrix=capture
    try:
        d=mixed.build_mixed_edit_diffusion_dataset(graph,trajectories=2,forward_steps=1000,record_every=10,
            candidates_per_example=16,seed=seed,actions=('birth','death','relocation','switch'),
            feature_profile='typed_closure_compact_v2',role_corruption_targets={0:terminal})
    finally:mixed.mixed_edit_candidate_matrix=original
    ids=d.trajectory_ids+2*j;select=(source['ids']//2)==j
    assert len(rows)==200 and np.array_equal(d.candidate_features,source['x'][select])
    assert np.array_equal(d.targets,source['y'][select]) and np.array_equal(ids,source['ids'][select])
    result=dict(delta=np.stack(rows),ids=ids)
    folder=OUT/('engineering' if engineering else 'mixed')/f'{kind}_{j}';folder.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(folder/'delta.npz',**result)
    report=dict(kind=kind,pair=j,rows=200,old_features_labels_ids_exact=True,builder_seed=seed,
        seconds=time.monotonic()-start,checked_full_difference_rows=10 if engineering else 0)
    put(folder.relative_to(OUT)/'report.json',report);return result,report


def corruption(kind,tid,columns,engineering=False):
    initial=b.graph('pubmed');edges,nb=c.r5.topology(initial);names=sorted(edges)
    source=t.OUT/f'cache/{kind}_{tid}';old=np.load(source/'data.npz');record=b.read(source/'record.json');rows=[]
    raw=b.read(c.OUT/f'corruption/{kind}_{tid}/counts.json')
    for item in record['path']:
        c.r5.edit(edges,nb,names,item['edit'],0)
        if item['step']%4:continue
        g=c.r5.graph_from(initial,edges);i=len(rows);ps=t.proposals(g,record['records'][i]['candidates'])
        cpu=mixed.mixed_edit_candidate_matrix(g,ps,item['step']/128,'typed_closure_compact_v2')
        assert np.max(abs(cpu-old['x'][i]))<=1e-5
        rows.append(delta(g,ps,columns,raw[i]['raw'],check=engineering and i%8==0))
    assert len(rows)==32
    folder=OUT/('engineering' if engineering else 'corruption')/f'{kind}_{tid}';folder.mkdir(parents=True,exist_ok=False)
    result=dict(delta=np.stack(rows),ids=old['ids']);np.savez_compressed(folder/'delta.npz',**result);return result


def main():
    b.setup();c.verify();assert b.read(c.OUT/'cache_acceptance.json')['passed'];OUT.mkdir(parents=True,exist_ok=False)
    files=[Path(__file__),SRC/'ROUND08.md',SRC/'candidate_consequences.py',SRC/'current_state.py',c.OUT/'cache_acceptance.json',c.OUT/'schema.json']
    files.extend(c.OUT.rglob('counts.json'))
    put('manifest.json',dict(files={str(p):b.sha(p) for p in files},unique_records=5048,new_stochastic_trajectories=0))
    columns=b.read(c.OUT/'schema.json')['columns'];_,report=pair('train',0,columns,True)
    start=time.monotonic();corruption('B',4,columns,True);newtime=time.monotonic()-start
    cost=2*(23*report['seconds']+14*newtime)+30
    put('engineering.json',dict(passed=cost<=1800,cost_bound_seconds=cost,mixed_seconds=report['seconds'],corruption_seconds=newtime,
        engineering_rows=232,full_difference_checks=14))
    assert cost<=1800;verify();start=time.monotonic();reports=[]
    for kind,ids,name in [('train',range(15),'old_train'),('assessment',range(8,16),'old_assessment')]:
        blocks=[]
        for j in ids:
            assert time.monotonic()-start<1800
            q,rp=pair(kind,j,columns);blocks.append(q);reports.append(rp)
            print('delta pair',kind,j,round(time.monotonic()-start,2),flush=True)
        np.savez_compressed(OUT/f'{name}.npz',delta=np.concatenate([q['delta'] for q in blocks]),ids=np.concatenate([q['ids'] for q in blocks]))
    train_keys=[('U' if tid%2 else 'B',tid) for tid in range(8)]
    val_keys=[('U',tid) for tid in range(8,12)]+[('B',8),('B',10)]
    for name,kk in [('new_B_train',train_keys),('new_assessment',val_keys)]:
        blocks=[]
        for kind,tid in kk:
            assert time.monotonic()-start<1800
            blocks.append(corruption(kind,tid,columns));print('delta corruption',kind,tid,round(time.monotonic()-start,2),flush=True)
        np.savez_compressed(OUT/f'{name}.npz',delta=np.concatenate([q['delta'] for q in blocks]),ids=np.concatenate([q['ids'] for q in blocks]))
    old,news,splits=t.data()
    for name,z in [('old_train',old),('old_assessment',splits['old_assessment']),('new_B_train',news['B']),('new_assessment',splits['new_assessment'])]:
        q=np.load(OUT/f'{name}.npz');assert np.array_equal(q['ids'],z['ids']) and q['delta'].shape==(len(z['y']),16,len(columns))
        assert np.isfinite(q['delta']).all() and (q['delta'][z['x'][:,:,0]>.5]==0).all()
    verify();put('acceptance.json',dict(passed=True,unique_records=5048,new_stochastic_trajectories=0,old_feature_label_id_exact=True,
        reports=reports,seconds=time.monotonic()-start,outputs={str(p):b.sha(p) for p in OUT.rglob('*.npz')}))
    print('candidate consequence cache accepted',round(time.monotonic()-start,2),flush=True)


if __name__=='__main__':
    try:main()
    except BaseException as exc:put('failure.json',dict(type=type(exc).__name__,message=str(exc)));raise
