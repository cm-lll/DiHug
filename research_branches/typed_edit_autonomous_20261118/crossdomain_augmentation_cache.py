"""One frozen direction, paired targeted/ordinary inverse supervision banks."""
import argparse
import confirmation_bank as original
import confirmation_common as contract
import crossdomain_augmentation_diagnosis as diagnosis
from candidate_consequences import Consequences
from potential import encode
from round01 import t,b,np,torch,time,Path,ROOT,SRC

BASE=ROOT/'results/typed_edit_crossdomain_augmentation_20261118'

def trajectory(ds,kind,bank,tid,steps,selection,cols,deadline,engineering=False):
    ref=b.graph(ds);names=sorted(ref.family_specs);family=names.index(selection['key']);idx=selection['index'];direction=selection['forward_direction']
    target=np.column_stack([ref.family_degrees(f) for f in names]);state=t.a.CpuAuditState(ref,target,'cpu',use_triton=False)
    degree=state.family_degree.clone();tracker=Consequences(ref,cols);reference=int(tracker.counts[idx]);path=[];records=[];xs=[];ys=[];raws=[];deltas=[];actions=[]
    width=np.load(b.RUN/f'cache/{ds}/data.npz')['x'].shape[-1];fe=0.;audit=0.;start=time.monotonic()
    for step in range(1,steps+1):
        assert time.monotonic()<deadline
        seed=202611189100000+['pubmed','acm'].index(ds)*100000000000+bank*1000000000+tid*10000+step
        gen=torch.Generator().manual_seed(seed);pool,valid=state._random_switch_pool(480,gen)
        batch=state._pick_valid(pool,valid&(pool.family==family),15,gen)
        graph=state.to_graph(ref);vals=t.a.serialize(batch);dd=tracker.differences(t.proposals(graph,vals));value=direction*dd[:,idx]
        eligible=np.flatnonzero(np.array(vals['action'])==4)
        if kind=='targeted' and value.max()>0:choice=int(value.argmax())
        else:choice=int(eligible[int(torch.randint(0,len(eligible),(1,),generator=torch.Generator().manual_seed(seed+5000000)))]) if len(eligible) else 0
        chosen=original.base.take(batch,slice(choice,choice+1));v=t.a.serialize(chosen)
        if engineering and step%4==0:
            tic=time.monotonic();edges,nb=original.c.r5.topology(graph)
            for j in range(15):
                original.c.r5.edit(edges,nb,names,vals,j);_,fresh=original.c.context(original.c.r5.graph_from(graph,edges),cols)
                assert np.array_equal(fresh['raw']-tracker.counts,dd[j]);original.c.r5.edit(edges,nb,names,vals,j,True)
            audit+=time.monotonic()-tic
        before=state.edge_tables.clone();adj=state.adjacency.clone();tracker.apply(v,0);state.commit(chosen,torch.tensor(0));assert torch.equal(degree,state.family_degree)
        inv=original.base.inverse(chosen)
        if step%4==0:
            after=state.edge_tables.clone();adjafter=state.adjacency.clone();state.commit(inv,torch.tensor(0))
            assert torch.equal(before,state.edge_tables) and torch.equal(adj,state.adjacency)
            state.commit(chosen,torch.tensor(0));assert torch.equal(after,state.edge_tables) and torch.equal(adjafter,state.adjacency)
        path.append(dict(step=step,seed=seed,edit=v,selected_delta=int(dd[choice,idx]),count=int(tracker.counts[idx])))
        if step%4:continue
        graph=state.to_graph(ref);_,actual=original.c.context(graph,cols);assert np.array_equal(actual['raw'],tracker.counts)
        batch=state.random_candidates(16,torch.Generator().manual_seed(seed+9000000),role_phase=False,targeted_fraction=0.,closure_fraction=0.,candidate_mode='legacy_pool')
        vals=t.a.serialize(batch);wanted=original.base.key(t.a.serialize(inv),0);hits=[i for i in range(16) if original.base.key(vals,i)==wanted];assert len(hits)<=1;label=hits[0] if hits else 15
        if not hits:
            for name,field in vars(batch).items():field[15]=getattr(inv,name)[0]
        vals=t.a.serialize(batch);assert sum(original.base.key(vals,i)==wanted for i in range(16))==1
        ps=t.proposals(graph,vals);local=state.compact_features(batch,step/128,width).numpy();cpu=original.mixed.mixed_edit_candidate_matrix(graph,ps,step/128,original.PROFILE)
        fe=max(fe,float(abs(local-cpu).max()));assert fe<=1e-5
        dd=tracker.differences(ps);covered=direction*(tracker.counts[idx]-reference)>0 and direction*dd[label,idx]<0
        xs.append(encode(tracker.counts,dd,ref.num_nodes,cols));ys.append(label);raws.append(tracker.counts.copy());deltas.append(dd);actions.append(vals['action'])
        records.append(dict(step=step,label=label,candidates=vals,counts=tracker.counts.tolist(),covered=bool(covered),components=actual['components']))
    graph=state.to_graph(ref);graph.validate()
    for f in names:assert np.array_equal(graph.family_degrees(f),ref.family_degrees(f))
    z=original.pack(xs,ys,np.full(len(ys),2000+tid,dtype=np.int64),raws,deltas,actions,cols)
    return z,dict(dataset=ds,kind=kind,bank=bank,trajectory=tid,steps=steps,path=path,records=records,covered=sum(v['covered'] for v in records),feature_error=fe,seconds=time.monotonic()-start,extra_engineering_audit_seconds=audit,final_graph=b.graph_payload(graph))

def audit_trajectory(ds,z,report,cols,deadline):
    ref=b.graph(ds);edges,nb=original.c.r5.topology(ref);names=sorted(edges);selected=0
    rec={r['step']:r for r in report['records']}
    for entry in report['path']:
        assert time.monotonic()<deadline
        step=entry['step'];before={f:set(es) for f,es in edges.items()} if step in rec else None
        original.c.r5.edit(edges,nb,names,entry['edit'],0)
        if step not in rec:continue
        r=rec[step];i=selected;g=original.c.r5.graph_from(ref,edges);_,actual=original.c.context(g,cols)
        assert np.array_equal(actual['raw'],z['counts'][i]) and np.array_equal(actual['raw'],r['counts'])
        for f in names:assert np.array_equal(g.family_degrees(f),ref.family_degrees(f))
        candidates=r['candidates'];label=r['label'];dd=z['delta'][i]
        tocheck=range(16) if step%32==0 else [label]
        for j in tocheck:
            original.c.r5.edit(edges,nb,names,candidates,j);_,new=original.c.context(original.c.r5.graph_from(ref,edges),cols)
            assert np.array_equal(new['raw']-actual['raw'],dd[j]);original.c.r5.edit(edges,nb,names,candidates,j,True)
        original.c.r5.edit(edges,nb,names,candidates,label)
        # The true inverse restores exactly the pre-forward topology.
        assert edges==before
        original.c.r5.edit(edges,nb,names,candidates,label,True)
        for f in names:assert edges[f]=={tuple(sorted(e)) for e in g.family_edges[f]}
        selected+=1
    final=b.restore_graph(report['final_graph'])
    for f in names:assert edges[f]=={tuple(sorted(e)) for e in final.family_edges[f]}
    assert selected==len(z['y']);return dict(passed=True,steps=len(report['path']),records=selected)

def main(ds,phase,bank):
    b.setup();selected=b.read(diagnosis.OUT/'summary.json')['selected'][ds];assert selected['positive_gap'];a=diagnosis.assets(ds,1 if ds=='pubmed' else 2);cols=b.read(a['schema'])['columns']
    out=BASE/f'{phase}/{ds}/bank{bank}';out.mkdir(parents=True,exist_ok=False)
    files=[Path(__file__),SRC/'CROSSDOMAIN_AUGMENTATION_VALIDATION.md',diagnosis.OUT/'summary.json',diagnosis.OUT/'manifest.json',diagnosis.OUT/'acceptance.json',a['schema'],b.RUN/f'data/{ds}/graph.json',b.RUN/f'cache/{ds}/data.npz']
    if phase=='cache':
        eng=BASE/f'engineering/{ds}/bank999';assert b.read(eng/'summary.json')['passed'];files += [eng/'manifest.json',eng/'acceptance.json',eng/'summary.json']
    contract.bind(out,files,dataset=ds,bank=bank,selection=selected,new_complete_surrogates=0,new_fits=0)
    if phase=='engineering':
        rows=[];cost=20.;tic=time.monotonic()
        for kind in ['targeted','ordinary']:
            z,r=trajectory(ds,kind,bank,0,32,selected,cols,tic+600,True);np.savez_compressed(out/f'{kind}.npz',**z);contract.put(out,f'{kind}.json',r)
            # Scientific cache production omits the extra full candidate engineering audit.
            cost+=2*32*(r['seconds']-r['extra_engineering_audit_seconds']);rows.append({k:r[k] for k in ['kind','covered','seconds','extra_engineering_audit_seconds','feature_error']})
        passed=rows[0]['covered']>=4 and cost<=1800
        contract.put(out,'summary.json',dict(passed=passed,rows=rows,cost_bound_seconds=cost,selection=selected));contract.verify(out)
        contract.put(out,'acceptance.json',dict(passed=True,engineering_gate_passed=passed));print(ds,phase,passed,cost,rows,flush=True);return
    tic=time.monotonic();reports=[]
    for kind in ['targeted','ordinary']:
        blocks=[];rr=[]
        for tid in range(8):
            z,r=trajectory(ds,kind,bank,tid,128,selected,cols,tic+1800);folder=out/f'{kind}/trajectories/{tid}';contract.put(folder,'record.json',r);np.savez_compressed(folder/'data.npz',**z)
            blocks.append(z);rr.append(r);print(ds,kind,tid,r['covered'],r['seconds'],flush=True)
        merged={k:np.concatenate([z[k] for z in blocks]) for k in blocks[0]};np.savez_compressed(out/f'{kind}/augmentation.npz',**merged)
        reports.append(dict(kind=kind,records=len(merged['y']),covered=sum(v['covered'] for v in rr)))
    contract.put(out,'completion.json',dict(completed=True,records=512,forward_steps=2048,seconds=time.monotonic()-tic));tic=time.monotonic();checks=[]
    for kind in ['targeted','ordinary']:
        for tid in range(8):
            f=out/f'{kind}/trajectories/{tid}';z=np.load(f/'data.npz');r=b.read(f/'record.json');check=audit_trajectory(ds,z,r,cols,tic+1800)
            checks.append(dict(kind=kind,trajectory=tid,**check));contract.put(out,f'checks/{kind}_{tid}.json',checks[-1])
    passed=reports[0]['covered']>=128;contract.verify(out)
    contract.put(out,'summary.json',dict(passed=passed,rows=reports,selection=selected,normalization_changed=False))
    contract.put(out,'acceptance.json',dict(passed=True,coverage_gate_passed=passed,checks=checks,seconds=time.monotonic()-tic,outputs={str(out/f'{k}/augmentation.npz'):b.sha(out/f'{k}/augmentation.npz') for k in ['targeted','ordinary']}));print(ds,'cache accepted',passed,reports,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('dataset',choices=['pubmed','acm']);p.add_argument('phase',choices=['engineering','cache']);p.add_argument('--bank',type=int,required=True);args=p.parse_args()
    try:main(args.dataset,args.phase,args.bank)
    except BaseException as exc:
        out=BASE/f'{args.phase}/{args.dataset}/bank{args.bank}'
        if out.exists():contract.put(out,'failure.json',dict(type=type(exc).__name__,message=str(exc)))
        raise
