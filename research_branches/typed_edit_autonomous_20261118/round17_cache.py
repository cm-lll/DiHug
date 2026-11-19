"""IMDB reconstruction and pre-specified inverse-edit corruption banks."""
import round09_cache as original
import round08_cache as dc
from potential import encode,scales
from candidate_consequences import Consequences
from round01 import t,b,np,torch,time,Path,ROOT,SRC,json
c=dc.c;base=original.base;mixed=c.mixed
OUT=ROOT/'results/typed_edit_autonomous_20261118/r17_cache'
DS='imdb';PROFILE='typed_closure_compact_v2';ACTIONS=['noop','birth','death','relocation','switch']


def put(name,v):
    p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')


def verify():
    t.verify_train()
    for p,h in b.read(OUT/'manifest.json')['files'].items():assert b.sha(p)==h,p


def pack(xs,ys,ids,raws,deltas,actions,columns):
    y=np.asarray(ys,dtype=np.int64);dd=np.asarray(deltas,dtype=np.int64);aa=np.asarray(actions,dtype=np.int64);tri=sum(a=='typed_triangles' for a,k in columns)
    return dict(x=np.asarray(xs,dtype=np.float32),y=y,ids=np.asarray(ids,dtype=np.int64),counts=np.asarray(raws,dtype=np.int64),delta=dd,
        target_action=aa[np.arange(len(y)),y],target_triangle_delta=dd[np.arange(len(y)),y,:tri].sum(-1))


def pair(j,columns,engineering=False):
    graph=b.graph(DS);source=np.load(b.RUN/f'cache/{DS}/data.npz');terminal=b.restore_graph(b.read(b.RUN/f'cache/{DS}/pair{j}/bridge_terminal.json'))
    xs=[];raws=[];deltas=[];actions=[];components=[];original_matrix=mixed.mixed_edit_candidate_matrix;start=time.monotonic()
    def capture(g,ps,*args,**kwargs):
        x=original_matrix(g,ps,*args,**kwargs);_,raw=c.context(g,columns);tracker=Consequences(g,columns,raw['raw']);dd=tracker.differences(ps)
        for f in tracker.names:assert tracker.edges[f]=={tuple(sorted(e)) for e in g.family_edges[f]}
        if engineering and len(xs)%20==0:
            k=next((i for i,p in enumerate(ps) if p.action!='noop'),0);tracker.proposal(ps[k]);_,actual=c.context(c.r5.graph_from(g,tracker.edges),columns)
            assert np.array_equal(np.array(actual['raw'])-np.array(raw['raw']),dd[k]);tracker.proposal(ps[k],True)
        xs.append(encode(np.array(raw['raw']),dd,g.num_nodes,columns));raws.append(raw['raw']);deltas.append(dd);actions.append([ACTIONS.index(p.action) for p in ps]);components.append(raw['components']);return x
    mixed.mixed_edit_candidate_matrix=capture
    try:
        d=mixed.build_mixed_edit_diffusion_dataset(graph,trajectories=2,forward_steps=1000,record_every=10,candidates_per_example=16,
            seed=2026111600+10000019*j,actions=('birth','death','relocation','switch'),feature_profile=PROFILE,role_corruption_targets={0:terminal})
    finally:mixed.mixed_edit_candidate_matrix=original_matrix
    select=(source['ids']//2)==j;ids=d.trajectory_ids+2*j
    assert len(xs)==200 and np.array_equal(d.candidate_features,source['x'][select]) and np.array_equal(d.targets,source['y'][select]) and np.array_equal(ids,source['ids'][select])
    z=pack(xs,d.targets,ids,raws,deltas,actions,columns);folder=OUT/('engineering' if engineering else 'mixed')/f'pair{j}';folder.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(folder/'data.npz',**z);report=dict(rows=200,old_features_labels_ids_exact=True,disconnected_records=sum(k!=1 for k in components),seconds=time.monotonic()-start)
    put(folder.relative_to(OUT)/'report.json',report);return z,report


@torch.no_grad()
def trajectory(bank,kind,tid,columns,steps=None):
    horizon=128 if bank=='short' else 2048;steps=horizon if steps is None else steps;every=4 if bank=='short' else 16
    ref=b.graph(DS);target=np.column_stack([ref.family_degrees(f) for f in sorted(ref.family_specs)]);state=t.a.CpuAuditState(ref,target,'cpu',use_triton=False)
    degrees=state.family_degree.clone();tracker=Consequences(ref,columns);width=np.load(b.RUN/f'cache/{DS}/data.npz')['x'].shape[-1]
    seed_base={'short':2026111800000,'deep':2026111810000,'typed':2026111820000}[bank]
    xs=[];ys=[];raws=[];deltas=[];actions=[];records=[];path=[];start=time.monotonic();maxerr=0.
    for step in range(1,steps+1):
        seed=seed_base+tid*10000+step;generator=torch.Generator().manual_seed(seed)
        if kind=='H':
            pool,valid=state._random_switch_pool(480,generator);close=bool(torch.randint(0,2,(1,),generator=torch.Generator().manual_seed(seed+3000000)))
            incident=torch.tensor([tid in (ref.family_specs[f].src_type,ref.family_specs[f].dst_type) for f in tracker.names]);mask=(~incident if close else incident)[pool.family]
            priority=state._closure_components(pool)[0].sum(1) if close else None
            batch=state._pick_valid(pool,valid&mask,15,generator,priority=priority);choice=int(torch.randint(0,15,(1,),generator=torch.Generator().manual_seed(seed+5000000)))
        else:
            batch=state.random_candidates(16,generator,role_phase=False,targeted_fraction=0.,closure_fraction=float(kind=='B'),candidate_mode='legacy_pool')
            choice=int(torch.randint(1,16,(1,),generator=torch.Generator().manual_seed(seed+5000000)))
        chosen=base.take(batch,slice(choice,choice+1));values=t.a.serialize(chosen)
        if step%every==0:before_edges=state.edge_tables.clone();before_adj=state.adjacency.clone()
        tracker.apply(values,0);state.commit(chosen,torch.tensor(0));assert torch.equal(degrees,state.family_degree);path.append(dict(step=step,edit=values))
        if step%every:continue
        inv=base.inverse(chosen);after_edges=state.edge_tables.clone();after_adj=state.adjacency.clone()
        state.commit(inv,torch.tensor(0));assert torch.equal(before_edges,state.edge_tables) and torch.equal(before_adj,state.adjacency)
        state.commit(chosen,torch.tensor(0));assert torch.equal(after_edges,state.edge_tables) and torch.equal(after_adj,state.adjacency)
        batch=state.random_candidates(16,torch.Generator().manual_seed(seed+9000000),role_phase=False,targeted_fraction=0.,closure_fraction=0.,candidate_mode='legacy_pool')
        wanted=base.key(t.a.serialize(inv),0);values=t.a.serialize(batch);hits=[i for i in range(16) if base.key(values,i)==wanted];assert len(hits)<=1;label=hits[0] if hits else 15
        if not hits:
            for name,value in vars(batch).items():value[15]=getattr(inv,name)[0]
        values=t.a.serialize(batch);assert sum(base.key(values,i)==wanted for i in range(16))==1
        graph=state.to_graph(ref);ps=t.proposals(graph,values);local=state.compact_features(batch,step/horizon,width).numpy();cpu=mixed.mixed_edit_candidate_matrix(graph,ps,step/horizon,PROFILE)
        maxerr=max(maxerr,float(abs(local-cpu).max()));assert maxerr<=1e-5
        _,actual=c.context(graph,columns);assert np.array_equal(actual['raw'],tracker.counts);raw=tracker.counts.copy();dd=tracker.differences(ps)
        xs.append(encode(raw,dd,ref.num_nodes,columns));ys.append(label);raws.append(raw);deltas.append(dd);actions.append(values['action'])
        records.append(dict(step=step,label=label,candidates=values,current_raw=raw.tolist(),components=actual['components'],lcc_fraction=actual['lcc_fraction'],time_fraction=step/horizon))
    final=state.to_graph(ref);final.validate()
    for f in tracker.names:assert tracker.edges[f]=={tuple(sorted(e)) for e in final.family_edges[f]} and np.array_equal(final.family_degrees(f),ref.family_degrees(f))
    z=pack(xs,ys,np.full(len(ys),tid,dtype=np.int64),raws,deltas,actions,columns)
    return z,dict(bank=bank,kind=kind,tid=tid,steps=steps,record_every=every,records=records,path=path,final_graph=b.graph_payload(final),seconds=time.monotonic()-start,max_local_feature_error=maxerr)


def main():
    b.setup();t.verify_train();eng=ROOT/'results/typed_edit_autonomous_20261118/r17_engineering/acceptance.json';assert b.read(eng)['passed'];OUT.mkdir(parents=True,exist_ok=False)
    graph=b.graph(DS);counts=t.metrics.counts(graph);columns=[(a,k) for a in ['typed_triangles','cross_relation_four_cycles'] for k in sorted(counts[a])]
    put('schema.json',dict(dataset=DS,n=graph.num_nodes,columns=columns,k=len(columns),input_width=2*len(columns),coefficient_count=2*len(columns)))
    files=[Path(__file__),SRC/'ROUND17.md',SRC/'potential.py',SRC/'candidate_consequences.py',eng,b.RUN/f'data/{DS}/graph.json',b.RUN/f'cache/{DS}/data.npz']
    files.extend(b.RUN/f'cache/{DS}/pair{j}/bridge_terminal.json' for j in range(15));put('manifest.json',dict(files={str(p):b.sha(p) for p in files},old_records=3000,new_records=1280,new_corruption_trajectories=16,new_forward_steps=17408))
    _,pr=pair(0,columns,True);times={}
    for bank,kind,tid in [('short','U',998),('short','B',998),('deep','B',998),('typed','H',3)]:
        z,rp=trajectory(bank,kind,tid,columns,32);put(f'engineering/{bank}_{kind}_{tid}.json',rp);times[bank+'_'+kind]=rp['seconds']/32
    cost=2*(15*pr['seconds']+1024*max(times['short_U'],times['short_B'])+8192*times['deep_B']+8192*times['typed_H'])+30
    put('engineering.json',dict(passed=cost<=1800,cost_bound_seconds=cost,old_reconstructed_engineering_records=200,new_engineering_forward_steps=128))
    assert cost<=1800;verify();start=time.monotonic();oldblocks=[]
    for j in range(15):
        assert time.monotonic()-start<1800
        z,rp=pair(j,columns);oldblocks.append(z);print('IMDB old pair',j,round(time.monotonic()-start,2),flush=True)
    old={k:np.concatenate([z[k] for z in oldblocks]) for k in oldblocks[0]};np.savez_compressed(OUT/'old_train.npz',**old)
    keys=[('short','B' if i%2==0 else 'U',i) for i in range(8)]+[('deep',k,i) for i in [100,102] for k in ['U','B']]+[('typed','H',i) for i in range(4)]
    blocks=[]
    for bank,kind,tid in keys:
        assert time.monotonic()-start<1800
        z,rp=trajectory(bank,kind,tid,columns);blocks.append(z);folder=Path('cache')/f'{bank}_{kind}_{tid}'
        put(folder/'record.json',rp);np.savez_compressed(OUT/folder/'data.npz',**z);print('IMDB corruption',bank,kind,tid,round(time.monotonic()-start,2),flush=True)
    new={k:np.concatenate([z[k] for z in blocks]) for k in blocks[0]};new['trajectory_key']=np.concatenate([np.full(len(z['y']),f'{bank}/{kind}/{tid}') for z,(bank,kind,tid) in zip(blocks,keys)])
    assert len(new['y'])==1280 and len(old['y'])==3000;np.savez_compressed(OUT/'new_train.npz',**new)
    put('normalization.json',dict(mean=new['counts'].mean(0).tolist(),sd=np.maximum(new['counts'].std(0),1).tolist(),rows=1280,training_only=True))
    put('completion.json',dict(completed=True,old_reconstructed_records=3000,new_records=1280,new_corruption_trajectories=16,new_forward_steps=17408,seconds=time.monotonic()-start))
    verify();start=time.monotonic();replayed=0;record_checks=0;disconnected=0
    for bank,kind,tid in keys:
        folder=OUT/f'cache/{bank}_{kind}_{tid}';rp=b.read(folder/'record.json');z=np.load(folder/'data.npz');edges,nb=c.r5.topology(graph);names=sorted(edges);row=0
        for p in rp['path']:
            c.r5.edit(edges,nb,names,p['edit'],0);replayed+=1
            if p['step']%rp['record_every']:continue
            rec=rp['records'][row];g=c.r5.graph_from(graph,edges);_,raw=c.context(g,columns);assert raw['raw']==rec['current_raw'] and raw['components']==rec['components'];disconnected+=raw['components']!=1
            dd=Consequences(g,columns,raw['raw']).differences(t.proposals(g,rec['candidates']));assert np.array_equal(dd,z['delta'][row]) and np.array_equal(encode(np.array(raw['raw']),dd,g.num_nodes,columns),z['x'][row])
            assert rec['label']==z['y'][row];c.r5.edit(edges,nb,names,rec['candidates'],rec['label']);c.r5.edit(edges,nb,names,p['edit'],0)
            for f in names:assert edges[f]=={tuple(sorted(e)) for e in g.family_edges[f]}
            row+=1;record_checks+=1
        final=b.restore_graph(rp['final_graph'])
        for f in names:assert edges[f]=={tuple(sorted(e)) for e in final.family_edges[f]} and np.array_equal(final.family_degrees(f),graph.family_degrees(f))
        print('accepted IMDB corruption',bank,kind,tid,round(time.monotonic()-start,2),flush=True)
    neg=int((new['target_triangle_delta']<0).sum());pos=int((new['target_triangle_delta']>0).sum());noop=int((old['target_action']==0).sum())
    assert neg>=16 and pos>=16 and noop>=32
    verify();put('acceptance.json',dict(passed=True,new_replayed_steps=replayed,new_record_recounts=record_checks,new_disconnected_records=disconnected,new_negative=neg,new_positive=pos,old_noop_labels=noop,
        seconds=time.monotonic()-start,outputs={str(p):b.sha(p) for p in OUT.rglob('*.npz')}));print('IMDB cache accepted',neg,pos,disconnected,flush=True)


if __name__=='__main__':
    try:main()
    except BaseException as exc:put('failure.json',dict(type=type(exc).__name__,message=str(exc)));raise
