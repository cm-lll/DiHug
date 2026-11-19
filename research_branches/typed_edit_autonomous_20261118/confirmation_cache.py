"""Three fresh corruption banks per reference for frozen F confirmation."""
import confirmation_bank as producer
import confirmation_common as contract
import round30_deploy as prior
from candidate_consequences import Consequences
from potential import encode
from round01 import t,b,np,torch,time,Path,ROOT,SRC,json
OUT=ROOT/'results/typed_edit_autonomous_20261118/confirmation_v1/cache'
DATASETS=['imdb','pubmed'];REPEATS=[1,2,3]
KEYS=[('short','B' if i%2==0 else 'U',i) for i in range(8)]+[('deep',k,i) for i in [100,102] for k in ['U','B']]+[('typed','H',i) for i in range(4)]
put=lambda name,v:contract.put(OUT,name,v)
verify=lambda:contract.verify(OUT)

def main():
    b.setup();prior.verify();assert b.read(prior.OUT/'acceptance.json')['passed'] and not b.read(prior.OUT/'summary.json')['replace_F_for_confirmation'];OUT.mkdir(parents=True,exist_ok=False)
    files=[Path(__file__),SRC/'CONFIRMATION_V1.md',Path(producer.base.__file__),prior.OUT/'acceptance.json',prior.OUT/'summary.json']
    files.extend(b.RUN/f'data/{ds}/graph.json' for ds in DATASETS);files.extend(b.RUN/f'cache/{ds}/data.npz' for ds in DATASETS)
    contract.bind(OUT,files,datasets=DATASETS,repeats=REPEATS,scientific_corruption_steps=104448,new_supervised_records=7680,new_training_updates=0)
    columns={};cost=30.;eng=[]
    for ds in DATASETS:
        g=b.graph(ds);cc=t.metrics.counts(g);cols=[(a,k) for a in ['typed_triangles','cross_relation_four_cycles'] for k in sorted(cc[a])];columns[ds]=cols
        assert len(g.type_names)==4;put(f'{ds}/schema.json',dict(dataset=ds,n=g.num_nodes,columns=cols,k=len(cols)))
        timers={}
        for bank,kind,tid in [('short','U',0),('short','B',0),('deep','U',100),('deep','B',100)]+[('typed','H',i) for i in range(4)]:
            z,rp=producer.trajectory(ds,999,bank,kind,tid,cols,32);timers[(bank,kind,tid)]=rp['seconds']/32;eng.append(dict(dataset=ds,bank=bank,kind=kind,tid=tid,seconds=rp['seconds'],steps=32,feature_error=rp['max_local_feature_error']))
        estimate=1024*max(timers[('short','U',0)],timers[('short','B',0)])+8192*max(timers[('deep','U',100)],timers[('deep','B',100)])+2048*sum(timers[('typed','H',i)] for i in range(4));cost+=2*3*estimate
    put('engineering.json',dict(passed=cost<=4800,cost_bound_seconds=cost,engineering_corruption_steps=512,rows=eng));assert cost<=4800;verify();tic=time.monotonic()
    for ds in DATASETS:
        for repeat in REPEATS:
            blocks=[]
            for bank,kind,tid in KEYS:
                assert time.monotonic()-tic<4800
                z,rp=producer.trajectory(ds,repeat,bank,kind,tid,columns[ds],deadline=tic+4800);folder=Path(ds)/f'repeat{repeat}'/'trajectories'/f'{bank}_{kind}_{tid}';put(folder/'record.json',rp);np.savez_compressed(OUT/folder/'data.npz',**z);blocks.append(z)
            merged={k:np.concatenate([v[k] for v in blocks]) for k in blocks[0]};merged['trajectory_key']=np.concatenate([np.full(len(z['y']),f'{bank}/{kind}/{tid}') for z,(bank,kind,tid) in zip(blocks,KEYS)])
            assert len(merged['y'])==1280;folder=Path(ds)/f'repeat{repeat}';np.savez_compressed(OUT/folder/'new_train.npz',**merged)
            put(folder/'normalization.json',dict(mean=merged['counts'].mean(0).tolist(),sd=np.maximum(merged['counts'].std(0),1).tolist(),rows=1280,training_only=True));print('confirmation bank generated',ds,repeat,round(time.monotonic()-tic,1),flush=True)
    put('completion.json',dict(completed=True,new_corruption_banks=6,new_corruption_trajectories=96,new_forward_steps=104448,new_records=7680,new_training_updates=0,seconds=time.monotonic()-tic));verify();tic=time.monotonic();steps=0;records=0;reports=[]
    for ds in DATASETS:
        ref=b.graph(ds);cols=columns[ds]
        for repeat in REPEATS:
            blocks=[];disconnected=0
            for bank,kind,tid in KEYS:
                folder=OUT/ds/f'repeat{repeat}'/'trajectories'/f'{bank}_{kind}_{tid}';rp=b.read(folder/'record.json');z=np.load(folder/'data.npz');edges,nb=producer.c.r5.topology(ref);names=sorted(edges);row=0
                for v in rp['path']:
                    assert time.monotonic()-tic<2400
                    producer.c.r5.edit(edges,nb,names,v['edit'],0);steps+=1
                    if v['step']%rp['record_every']:continue
                    rec=rp['records'][row];g=producer.c.r5.graph_from(ref,edges);_,raw=producer.c.context(g,cols);assert raw['raw']==rec['current_raw'] and raw['components']==rec['components'];disconnected+=raw['components']!=1
                    dd=Consequences(g,cols,raw['raw']).differences(t.proposals(g,rec['candidates']));assert np.array_equal(dd,z['delta'][row]) and np.array_equal(encode(np.array(raw['raw']),dd,g.num_nodes,cols),z['x'][row]);assert rec['label']==z['y'][row]
                    wanted=producer.base.key(rec['candidates'],rec['label']);assert sum(producer.base.key(rec['candidates'],i)==wanted for i in range(16))==1
                    producer.c.r5.edit(edges,nb,names,rec['candidates'],rec['label']);producer.c.r5.edit(edges,nb,names,v['edit'],0)
                    for f in names:assert edges[f]=={tuple(sorted(e)) for e in g.family_edges[f]}
                    row+=1;records+=1
                final=b.restore_graph(rp['final_graph'])
                for f in names:assert edges[f]=={tuple(sorted(e)) for e in final.family_edges[f]} and np.array_equal(final.family_degrees(f),ref.family_degrees(f))
                blocks.append({k:z[k] for k in z.files})
            folder=OUT/ds/f'repeat{repeat}';saved=np.load(folder/'new_train.npz');norm=b.read(folder/'normalization.json')
            for k in blocks[0]:assert np.array_equal(saved[k],np.concatenate([v[k] for v in blocks]))
            assert np.array_equal(saved['counts'].mean(0),norm['mean']) and np.array_equal(np.maximum(saved['counts'].std(0),1),norm['sd']);delta=saved['target_triangle_delta'];reports.append(dict(dataset=ds,repeat=repeat,records=len(delta),negative=int((delta<0).sum()),positive=int((delta>0).sum()),zero=int((delta==0).sum()),disconnected_records=disconnected));print('confirmation bank accepted',ds,repeat,round(time.monotonic()-tic,1),flush=True)
    assert steps==104448 and records==7680;put('summary.json',dict(banks=reports,independent_corruption_repeats_per_reference=3,shared_old_bank=True,not_cross_network_generalization=True));verify()
    outputs={str(p):b.sha(p) for p in OUT.rglob('*') if p.is_file() and p.name in ['new_train.npz','normalization.json','schema.json']}
    put('acceptance.json',dict(passed=True,replayed_corruption_steps=steps,independent_record_recounts=records,outputs=outputs,new_training_updates=0,seconds=time.monotonic()-tic))

if __name__=='__main__':
    try:main()
    except BaseException as exc:put('failure.json',dict(type=type(exc).__name__,message=str(exc)));raise
