"""Longer fixed forward corruptions, with exact inverse labels and unchanged features."""
import round08_cache as dc
import round08_train as r8
from candidate_consequences import Consequences
from round01 import t,b,np,torch,time,Path,ROOT,SRC,json
c=dc.c;oldcache=t.a
import importlib.util
spec=importlib.util.spec_from_file_location('coverage_original_cache',ROOT/'research_branches/typed_edit_corruption_coverage_20261118/cache.py')
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
OUT=ROOT/'results/typed_edit_autonomous_20261118/r09_cache'


def put(name,v):
    p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')


def verify():
    r8.verify()
    for p,h in b.read(OUT/'manifest.json')['files'].items():assert b.sha(p)==h,p


@torch.no_grad()
def trajectory(kind,tid,steps=2048):
    ref=b.graph('pubmed');columns=b.read(c.OUT/'schema.json')['columns']
    target=np.column_stack([ref.family_degrees(f) for f in sorted(ref.family_specs)])
    state=t.a.CpuAuditState(ref,target,'cpu',use_triton=False);degree=state.family_degree.clone();tracker=Consequences(ref,columns)
    rows=[];ys=[];records=[];path=[];start=time.monotonic();maxerr=0.
    for step in range(1,steps+1):
        seed=2026111810000+tid*10000+step
        batch=state.random_candidates(16,torch.Generator().manual_seed(seed),role_phase=False,targeted_fraction=0.,closure_fraction=float(kind=='B'),candidate_mode='legacy_pool')
        choice=int(torch.randint(1,16,(1,),generator=torch.Generator().manual_seed(seed+5000000)))
        chosen=base.take(batch,slice(choice,choice+1));values=t.a.serialize(chosen);before=tracker.counts.copy()
        tracker.apply(values,0);state.commit(chosen,torch.tensor(0));assert torch.equal(state.family_degree,degree)
        path.append(dict(step=step,edit=values))
        if step%16:continue
        inv=base.inverse(chosen);iv=t.a.serialize(inv);tracker.apply(iv,0);assert np.array_equal(tracker.counts,before);tracker.apply(values,0)
        batch=state.random_candidates(16,torch.Generator().manual_seed(seed+9000000),role_phase=False,targeted_fraction=0.,closure_fraction=0.,candidate_mode='legacy_pool')
        values=t.a.serialize(batch);wanted=base.key(iv,0);hits=[i for i in range(16) if base.key(values,i)==wanted];assert len(hits)<=1
        label=hits[0] if hits else 15
        if not hits:
            for name,value in vars(batch).items():value[15]=getattr(inv,name)[0]
        values=t.a.serialize(batch);assert sum(base.key(values,i)==wanted for i in range(16))==1
        tf=step/2048;local=state.compact_features(batch,tf,45).numpy();graph=state.to_graph(ref);ps=t.proposals(graph,values)
        cpu=c.mixed.mixed_edit_candidate_matrix(graph,ps,tf,'typed_closure_compact_v2')
        maxerr=max(maxerr,float(abs(local-cpu).max()));assert maxerr<=1e-5
        current,raw=c.context(graph,columns);assert np.array_equal(current,tracker.features()) and np.array_equal(raw['raw'],tracker.counts)
        delta=tracker.normalized(ps);x=np.concatenate([local,np.tile(current,(16,1)),delta],-1)
        assert np.isfinite(x).all();rows.append(x);ys.append(label)
        records.append(dict(step=step,label=label,candidates=values,time_fraction=tf,current_raw=raw['raw'],components=raw['components'],lcc_fraction=raw['lcc_fraction']))
    final=state.to_graph(ref);final.validate()
    for f in tracker.names:
        assert tracker.edges[f]=={tuple(sorted(e)) for e in final.family_edges[f]}
        assert np.array_equal(ref.family_degrees(f),final.family_degrees(f))
    z=dict(x=np.stack(rows),y=np.array(ys,dtype=np.int64),ids=np.full(len(ys),tid,dtype=np.int64))
    return z,dict(kind=kind,tid=tid,steps=steps,records=records,path=path,final_graph=b.graph_payload(final),seconds=time.monotonic()-start,max_local_feature_error=maxerr)


def main():
    b.setup();r8.verify();OUT.mkdir(parents=True,exist_ok=False)
    files=[Path(__file__),SRC/'ROUND09.md',SRC/'candidate_consequences.py',Path(base.__file__),ROOT/'results/typed_edit_autonomous_20261118/coverage_audit/summary.json']
    put('manifest.json',dict(files={str(p):b.sha(p) for p in files},scientific_forward_steps=8192,unique_records=512,keys=[['U',100],['B',100],['U',102],['B',102]]))
    times=[]
    for kind in ['U','B']:
        z,rp=trajectory(kind,998,32);put(f'engineering/{kind}.json',rp);times.append(rp['seconds'])
    bound=2*8192*max(times)/32+30;put('engineering.json',dict(passed=bound<=900,cost_bound_seconds=bound,engineering_steps=64,engineering_records=4))
    assert bound<=900;verify();start=time.monotonic();banks=[]
    for tid in [100,102]:
        for kind in ['U','B']:
            assert time.monotonic()-start<900
            z,rp=trajectory(kind,tid);banks.append(z);put(f'cache/{kind}_{tid}/record.json',rp)
            np.savez_compressed(OUT/f'cache/{kind}_{tid}/data.npz',**z)
            print('long corruption',kind,tid,round(time.monotonic()-start,2),flush=True)
    new={k:np.concatenate([z[k] for z in banks]) for k in ['x','y','ids']};np.savez_compressed(OUT/'new512.npz',**new)
    old,short,splits=r8.data();merged={k:np.concatenate([short[k],new[k]]) for k in ['x','y','ids']};np.savez_compressed(OUT/'merged768.npz',**merged)
    put('completion.json',dict(completed=True,scientific_forward_steps=8192,new_records=512,seconds=time.monotonic()-start,new_surrogate_chains=0))
    verify();start=time.monotonic();checked=0;disconnected=0;columns=b.read(c.OUT/'schema.json')['columns'];ref=b.graph('pubmed')
    for tid in [100,102]:
        for kind in ['U','B']:
            rp=b.read(OUT/f'cache/{kind}_{tid}/record.json');z=np.load(OUT/f'cache/{kind}_{tid}/data.npz');edges,nb=c.r5.topology(ref);names=sorted(edges);row=0
            for p in rp['path']:
                c.r5.edit(edges,nb,names,p['edit'],0);checked+=1
                if p['step']%16:continue
                record=rp['records'][row];g=c.r5.graph_from(ref,edges);ctx,raw=c.context(g,columns);assert raw['raw']==record['current_raw']
                assert raw['components']==record['components'];disconnected+=raw['components']!=1
                ps=t.proposals(g,record['candidates']);local=c.mixed.mixed_edit_candidate_matrix(g,ps,record['time_fraction'],'typed_closure_compact_v2')
                d=dc.delta(g,ps,columns,raw['raw'],check=row%32==0);x=np.concatenate([local,np.tile(ctx,(16,1)),d],-1)
                assert abs(x-z['x'][row]).max()<=1e-5 and record['label']==z['y'][row]
                label=record['label'];iv=record['candidates'];c.r5.edit(edges,nb,names,iv,label)
                c.r5.edit(edges,nb,names,p['edit'],0)
                for f in names:assert edges[f]=={tuple(sorted(e)) for e in g.family_edges[f]}
                row+=1
            final=b.restore_graph(rp['final_graph'])
            for f in names:assert edges[f]=={tuple(sorted(e)) for e in final.family_edges[f]} and np.array_equal(final.family_degrees(f),ref.family_degrees(f))
            print('accepted long corruption',kind,tid,round(time.monotonic()-start,2),flush=True)
    initial=b.restore_graph(b.read(t.a.GEN/'prefix/initial.json'));point,_=c.context(initial,columns)
    ctx=merged['x'][:,0,45:71];fixedsd=np.maximum(short['x'][:,0,45:71].std(0),.01)
    nearest=float(np.sqrt(np.mean(((ctx-point)/fixedsd)**2,axis=1)).min())
    outside=int(((point<ctx.min(0)-1e-7)|(point>ctx.max(0)+1e-7)).sum())
    dd=new['x'][np.arange(len(new['y'])),new['y'],43];neg=int((dd<0).sum());pos=int((dd>0).sum())
    previous=b.read(ROOT/'results/typed_edit_autonomous_20261118/coverage_audit/summary.json')['states']['new_B_train']['nearest_normalized_rms']
    checks=dict(negative_at_least_32=neg>=32,positive_at_least_32=pos>=32,outside_at_most_12=outside<=12,nearest_improved_20pct=nearest<=previous*.8)
    put('coverage_gate.json',dict(passed=all(checks.values()),checks=checks,negative=neg,positive=pos,outside_observed_range=outside,nearest_fixed_scale_rms=nearest,previous_rms=previous))
    verify();put('acceptance.json',dict(passed=True,replayed_forward_steps=checked,full_record_recounts=512,disconnected_records=disconnected,
        seconds=time.monotonic()-start,outputs={str(p):b.sha(p) for p in OUT.rglob('*.npz')}))
    print('coverage gate',checks,'nearest',nearest,'outside',outside,flush=True)


if __name__=='__main__':
    try:main()
    except BaseException as exc:put('failure.json',dict(type=type(exc).__name__,message=str(exc)));raise
