"""Reconstruct existing graph states for target-free whole-graph context features."""
import round05 as r5
from round01 import t,b,np,torch,time,Path,ROOT,SRC,json
import dihug.core.mixed_edit_diffusion as mixed
metrics=t.metrics
OUT=ROOT/'results/typed_edit_autonomous_20261118/r06_fast'


def put(name,v):
    p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')


def keys(graph):
    c=metrics.counts(graph)
    return [(family,key) for family in ['typed_triangles','cross_relation_four_cycles'] for key in sorted(c[family])]


from round06_fast_counts import context


def verify():
    r5.verify()
    for p,h in b.read(OUT/'cache_manifest.json')['files'].items():assert b.sha(p)==h,p


def pair(kind,j,columns,engineering=False):
    graph=b.graph('pubmed');start=time.monotonic();rows=[];records=[]
    if kind=='train':
        terminal=b.restore_graph(b.read(b.RUN/f'cache/pubmed/pair{j}/bridge_terminal.json'))
        seed=2026111600+10000019*j;source=np.load(b.RUN/'cache/pubmed/data.npz')
    else:
        prior=b.MarginalGraphPrior.from_training_graph(graph);role=b.ExchangeableRoleDegreePrior.from_training_graph(graph)
        terminal=b.FamilyCountCondition(prior,role.family_counts(prior.family_endpoints)).sample_graph(720000000+2*j,mode='binomial')
        seed=710000000+100003*j;source=np.load(t.EPOCH/'validation.npz')
    original=mixed.mixed_edit_candidate_matrix
    def capture(g,*args,**kwargs):
        x=original(g,*args,**kwargs);v,c=context(g,columns);rows.append(v);records.append(c);return x
    mixed.mixed_edit_candidate_matrix=capture
    try:
        d=mixed.build_mixed_edit_diffusion_dataset(graph,trajectories=2,forward_steps=1000,record_every=10,
            candidates_per_example=16,seed=seed,actions=('birth','death','relocation','switch'),
            feature_profile='typed_closure_compact_v2',role_corruption_targets={0:terminal})
    finally:mixed.mixed_edit_candidate_matrix=original
    ids=d.trajectory_ids+2*j;select=(source['ids']//2)==j
    assert len(rows)==200 and np.array_equal(d.candidate_features,source['x'][select])
    assert np.array_equal(d.targets,source['y'][select]) and np.array_equal(ids,source['ids'][select])
    result=dict(context=np.stack(rows),ids=ids)
    folder=OUT/('engineering' if engineering else 'mixed')/f'{kind}_{j}';folder.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(folder/'context.npz',**result);put(folder.relative_to(OUT)/'counts.json',records)
    report=dict(kind=kind,pair=j,rows=200,old_features_labels_ids_exact=True,builder_seed=seed,
        seconds=time.monotonic()-start,disconnected_records=sum(x['components']!=1 for x in records),
        min_lcc_fraction=min(x['lcc_fraction'] for x in records),source_sha256=b.sha(folder/'context.npz'))
    put(folder.relative_to(OUT)/'report.json',report)
    return result,report


def corruption(kind,tid,columns):
    initial=b.graph('pubmed');edges,nb=r5.topology(initial);names=sorted(edges)
    source=t.OUT/f'cache/{kind}_{tid}';old=np.load(source/'data.npz');record=b.read(source/'record.json');rows=[];counts=[]
    for item in record['path']:
        r5.edit(edges,nb,names,item['edit'],0)
        if item['step']%4:continue
        g=r5.graph_from(initial,edges);rec=record['records'][len(rows)]
        cpu=mixed.mixed_edit_candidate_matrix(g,t.proposals(g,rec['candidates']),item['step']/128,'typed_closure_compact_v2')
        assert np.max(abs(cpu-old['x'][len(rows)]))<=1e-5
        v,c=context(g,columns);rows.append(v);counts.append(c)
    assert len(rows)==32
    final=b.restore_graph(record['final_graph'])
    for f in names:assert edges[f]=={tuple(sorted(e)) for e in final.family_edges[f]}
    folder=OUT/f'corruption/{kind}_{tid}';folder.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(folder/'context.npz',context=np.stack(rows),ids=old['ids'])
    put(folder.relative_to(OUT)/'counts.json',counts)
    return dict(context=np.stack(rows),ids=old['ids'])


def main():
    b.setup();r5.verify();assert b.read(r5.OUT/'acceptance.json')['passed'];OUT.mkdir(parents=True,exist_ok=False)
    files=[Path(__file__),SRC/'ROUND06.md',SRC/'ROUND06_CACHE_AMENDMENT.md',SRC/'round06_cache.py',SRC/'round06_fast_counts.py',r5.OUT/'summary.json',r5.OUT/'acceptance.json',
        Path(mixed.__file__),Path(metrics.__file__),b.RUN/'cache/pubmed/data.npz',t.EPOCH/'validation.npz']
    for j in range(15):files.append(b.RUN/f'cache/pubmed/pair{j}/bridge_terminal.json')
    put('cache_manifest.json',dict(files={str(p):b.sha(p) for p in files},training_rows=3000,assessment_rows=1600,
        new_B_rows=256,new_assessment_rows=192,new_random_trajectories=0))
    graph=b.graph('pubmed');columns=keys(graph);put('schema.json',dict(columns=columns,width=len(columns),normalization='log1p(current_count)/log1p(N)'))
    put('metric_fixtures.json',metrics.fixtures());_,report=pair('train',0,columns,True)
    assert b.read(OUT/'engineering/train_0/counts.json')==b.read(ROOT/'results/typed_edit_autonomous_20261118/r06/engineering/train_0/counts.json')
    worst=b.restore_graph(b.read(t.OUT/'cache/B_4/record.json')['final_graph']);baseline=metrics.counts(worst)
    fast,details=context(worst,columns);assert details['raw']==[baseline[a][k] for a,k in columns]
    assert details['components']==baseline['components'] and details['lcc_fraction']==baseline['lcc_fraction']
    start=time.monotonic();context(worst,columns);single=time.monotonic()-start
    cost=2*(23*report['seconds']+448*max(.1,2*single))+30
    put('cache_engineering.json',dict(passed=cost<=1200,cost_bound_seconds=cost,mixed_pair_seconds=report['seconds'],
        full_context_seconds=single,engineering_reconstructed_records=200,new_stochastic_trajectories=0))
    assert cost<=1200;verify();start=time.monotonic();reports=[]
    for kind,ids,name in [('train',range(15),'old_train'),('assessment',range(8,16),'old_assessment')]:
        blocks=[]
        for j in ids:
            assert time.monotonic()-start<1200
            q,rp=pair(kind,j,columns);blocks.append(q);reports.append(rp)
            print('context pair',kind,j,round(time.monotonic()-start,2),flush=True)
        np.savez_compressed(OUT/f'{name}.npz',context=np.concatenate([q['context'] for q in blocks]),ids=np.concatenate([q['ids'] for q in blocks]))
    train_keys=[('U' if tid%2 else 'B',tid) for tid in range(8)]
    val_keys=[('U',tid) for tid in range(8,12)]+[('B',8),('B',10)]
    for name,kk in [('new_B_train',train_keys),('new_assessment',val_keys)]:
        blocks=[]
        for kind,tid in kk:
            assert time.monotonic()-start<1200
            blocks.append(corruption(kind,tid,columns));print('corruption context',kind,tid,round(time.monotonic()-start,2),flush=True)
        np.savez_compressed(OUT/f'{name}.npz',context=np.concatenate([q['context'] for q in blocks]),ids=np.concatenate([q['ids'] for q in blocks]))
    old,news,splits=t.data()
    for name,z in [('old_train',old),('old_assessment',splits['old_assessment']),('new_B_train',news['B']),('new_assessment',splits['new_assessment'])]:
        q=np.load(OUT/f'{name}.npz');assert np.array_equal(q['ids'],z['ids']) and q['context'].shape==(len(z['y']),len(columns))
        assert np.isfinite(q['context']).all()
    verify();put('cache_acceptance.json',dict(passed=True,unique_record_contexts=5048,new_random_trajectories=0,
        old_feature_label_id_exact=True,corruption_feature_error_max_tolerance=1e-5,metric_fixtures=18,
        reports=reports,seconds=time.monotonic()-start,context_width=len(columns),
        outputs={str(p):b.sha(p) for p in OUT.rglob('*') if p.is_file() and p.name!='cache_acceptance.json'}))
    print('global context cache accepted',len(columns),round(time.monotonic()-start,2),flush=True)


if __name__=='__main__':
    try:main()
    except BaseException as exc:put('cache_failure.json',dict(type=type(exc).__name__,message=str(exc)));raise
