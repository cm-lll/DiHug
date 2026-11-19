"""One all-type bank per extra reference, gated on accepted main confirmation."""
import argparse
import additional_confirmation_bank as producer
import confirmation_common as contract
from candidate_consequences import Consequences
from potential import encode
from round01 import t,b,np,torch,time,Path,ROOT,SRC
OUT=None;REPEAT=None

def main(ds,repeat):
    global OUT,REPEAT
    assert repeat in [2,3,4];REPEAT=repeat
    b.setup();base=ROOT/'results/typed_edit_autonomous_20261118'
    main_confirmation=base/'confirmation_v1/deploy'
    assert b.read(main_confirmation/'acceptance.json')['passed']
    assert b.read(main_confirmation/'summary.json')['passed'],'main frozen refinement gate not passed'
    for f,h in b.read(main_confirmation/'manifest.json')['files'].items():assert b.sha(f)==h,f
    engineering=base/f'additional_f_engineering_finalized/{ds}'
    assert b.read(engineering/'acceptance.json')['passed'] and b.read(engineering/'summary.json')['cache_cost_gate_passed']
    for f,h in b.read(engineering/'manifest.json')['files'].items():assert b.sha(f)==h,f
    pilot=base/({'acm':'acm_guidance4','freebase':'freebase_legal_selection'}[ds])/'deploy'
    from additional_fit import source_check
    source_check(pilot);assert b.read(pilot/'summary.json')['passed']
    OUT=base/f'additional_confirmation_v1/cache/{ds}/{repeat}';OUT.mkdir(parents=True,exist_ok=False)
    ref=b.graph(ds);cc=t.metrics.counts(ref)
    cols=[(a,k) for a in ['typed_triangles','cross_relation_four_cycles'] for k in sorted(cc[a])]
    keys=[('short','B' if i%2==0 else 'U',i) for i in range(8)]
    keys += [('deep',k,i) for i in [100,102] for k in ['U','B']]
    keys += [('typed','H',i) for i in range(len(ref.type_names))]
    expected_steps=9216+2048*len(ref.type_names);expected_rows=768+128*len(ref.type_names)
    files=[Path(__file__),SRC/'ADDITIONAL_CONFIRMATION_V1.md',pilot/'acceptance.json',pilot/'summary.json',
           main_confirmation/'manifest.json',main_confirmation/'acceptance.json',main_confirmation/'summary.json',
           engineering/'manifest.json',engineering/'acceptance.json',engineering/'summary.json',
           b.RUN/f'data/{ds}/graph.json',b.RUN/f'cache/{ds}/data.npz']
    contract.bind(OUT,files,dataset=ds,repeat=repeat,steps=expected_steps,records=expected_rows,new_model_fits=0)
    contract.put(OUT,'schema.json',dict(dataset=ds,n=ref.num_nodes,columns=cols,k=len(cols)))
    cost=20.;eng=[]
    for name,kind,tid in keys:
        _,rp=producer.trajectory(ds,999,name,kind,tid,cols,steps=32,deadline=time.monotonic()+120)
        total=128 if name=='short' else 2048
        cost+=2*total/32*rp['seconds'];eng.append(dict(bank=name,kind=kind,tid=tid,seconds=rp['seconds']))
    contract.put(OUT,'engineering.json',dict(passed=cost<=4800,cost_bound_seconds=cost,engineering_opportunities=32*len(keys),rows=eng))
    assert cost<=4800;contract.verify(OUT)
    tic=time.monotonic();blocks=[]
    for name,kind,tid in keys:
        z,rp=producer.trajectory(ds,repeat,name,kind,tid,cols,deadline=tic+4800)
        folder=OUT/f'trajectories/{name}_{kind}_{tid}';contract.put(folder,'record.json',rp)
        np.savez_compressed(folder/'data.npz',**z);blocks.append(z)
        print('additional bank trajectory',ds,name,kind,tid,round(time.monotonic()-tic,1),flush=True)
    merged={k:np.concatenate([v[k] for v in blocks]) for k in blocks[0]}
    assert len(merged['y'])==expected_rows
    np.savez_compressed(OUT/'new_train.npz',**merged)
    contract.put(OUT,'normalization.json',dict(mean=merged['counts'].mean(0).tolist(),sd=np.maximum(merged['counts'].std(0),1).tolist(),training_only=True,rows=expected_rows))
    contract.put(OUT,'completion.json',dict(completed=True,corruption_steps=expected_steps,records=expected_rows,new_model_fits=0,seconds=time.monotonic()-tic))
    contract.verify(OUT);tic=time.monotonic();steps=0;records=0;disconnected=0;blocks=[]
    for name,kind,tid in keys:
        folder=OUT/f'trajectories/{name}_{kind}_{tid}';rp=b.read(folder/'record.json');z=np.load(folder/'data.npz')
        edges,nb=producer.c.r5.topology(ref);names=sorted(edges);row=0
        for v in rp['path']:
            assert time.monotonic()-tic<4800
            producer.c.r5.edit(edges,nb,names,v['edit'],0);steps+=1
            if v['step']%rp['record_every']:continue
            rec=rp['records'][row];g=producer.c.r5.graph_from(ref,edges);_,raw=producer.c.context(g,cols)
            assert raw['raw']==rec['current_raw'] and raw['components']==rec['components']
            disconnected+=raw['components']!=1
            dd=Consequences(g,cols,raw['raw']).differences(t.proposals(g,rec['candidates']))
            assert np.array_equal(dd,z['delta'][row]) and np.array_equal(encode(np.array(raw['raw']),dd,g.num_nodes,cols),z['x'][row])
            assert rec['label']==z['y'][row]
            wanted=producer.base.key(rec['candidates'],rec['label'])
            assert sum(producer.base.key(rec['candidates'],i)==wanted for i in range(16))==1
            producer.c.r5.edit(edges,nb,names,rec['candidates'],rec['label'])
            producer.c.r5.edit(edges,nb,names,v['edit'],0)
            for f in names:assert edges[f]=={tuple(sorted(e)) for e in g.family_edges[f]}
            row+=1;records+=1
        final=b.restore_graph(rp['final_graph'])
        for f in names:assert edges[f]=={tuple(sorted(e)) for e in final.family_edges[f]} and np.array_equal(final.family_degrees(f),ref.family_degrees(f))
        blocks.append({k:z[k] for k in z.files})
    saved=np.load(OUT/'new_train.npz');norm=b.read(OUT/'normalization.json')
    for k in blocks[0]:assert np.array_equal(saved[k],np.concatenate([v[k] for v in blocks]))
    assert np.array_equal(saved['counts'].mean(0),norm['mean'])
    assert np.array_equal(np.maximum(saved['counts'].std(0),1),norm['sd'])
    assert steps==expected_steps and records==expected_rows
    contract.put(OUT,'summary.json',dict(dataset=ds,bank_repeat=repeat,corruption_steps=steps,records=records,disconnected_records=disconnected,covered_types=len(ref.type_names),new_model_fits=0))
    contract.verify(OUT)
    outputs={str(OUT/name):b.sha(OUT/name) for name in ['schema.json','normalization.json','new_train.npz']}
    contract.put(OUT,'acceptance.json',dict(passed=True,replayed_steps=steps,independent_record_recounts=records,outputs=outputs,seconds=time.monotonic()-tic))
    print('Additional bank accepted',ds,records,flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('dataset',choices=['acm','freebase']);parser.add_argument('repeat',type=int,choices=[2,3,4]);args=parser.parse_args()
    try:main(args.dataset,args.repeat)
    except BaseException as exc:
        if OUT is not None:contract.put(OUT,'failure.json',dict(type=type(exc).__name__,message=str(exc)))
        raise
