"""Fresh augmentation banks for paired new-start coverage confirmation."""
import argparse
import confirmation_fit as frozen
import confirmation_bank as original
import confirmation_common as contract
from candidate_consequences import Consequences
from potential import encode
from round01 import t,b,np,torch,time,Path,ROOT,SRC
BASE=ROOT/'results/typed_edit_autonomous_20261118';OUT=None

def main(repeat):
    global OUT
    assert repeat in [2,3,4]
    OUT=BASE/f'cycle_repair_confirmation/cache/repeat{repeat}'
    b.setup();parent=BASE/'cycle_repair_engineering_timed'
    frozen.source_check(BASE/'cycle_repair_pilot/deploy')
    assert b.read(BASE/'cycle_repair_pilot/deploy/summary.json')['passed']
    cost=b.read(parent/'summary.json')['cost_bound_seconds'];assert cost<=600 and 3*cost<=1800
    assert b.read(parent/'acceptance.json')['passed'] and b.read(parent/'summary.json')['passed']
    for f,h in b.read(parent/'manifest.json')['files'].items():assert b.sha(f)==h,f
    OUT.mkdir(parents=True,exist_ok=False);ref=b.graph('imdb');cache=BASE/'confirmation_v1/cache_accounted_v2/imdb'
    cols=b.read(cache/'schema.json')['columns'];index=cols.index(['cross_relation_four_cycles','link_Movie_Genre'])
    names=sorted(ref.family_specs);family=names.index('link_Movie_Genre');width=np.load(b.RUN/'cache/imdb/data.npz')['x'].shape[-1]
    files=[Path(__file__),SRC/'CYCLE_REPAIR_CONFIRMATION.md',BASE/'cycle_repair_pilot/deploy/manifest.json',BASE/'cycle_repair_pilot/deploy/acceptance.json',BASE/'cycle_repair_pilot/deploy/summary.json',parent/'manifest.json',parent/'acceptance.json',parent/'summary.json',b.RUN/'data/imdb/graph.json',b.RUN/'cache/imdb/data.npz',cache/'schema.json']
    contract.bind(OUT,files,augmentation_repeat=repeat,reused_engineering_cost_seconds=cost,new_fits=0,new_complete_surrogates=0,forward_steps=1024,records=256)
    tic=time.monotonic();blocks=[];reports=[];feature_error=0.;positive=0
    for tid in range(8):
        target=np.column_stack([ref.family_degrees(f) for f in names]);state=t.a.CpuAuditState(ref,target,'cpu',use_triton=False)
        degree=state.family_degree.clone();tracker=Consequences(ref,cols);reference=int(tracker.counts[index])
        path=[];records=[];xs=[];ys=[];raws=[];deltas=[];actions=[]
        for step in range(1,129):
            assert time.monotonic()-tic<600
            seed=202611185100000+repeat*1000000000+tid*10000+step;generator=torch.Generator().manual_seed(seed)
            pool,valid=state._random_switch_pool(480,generator);batch=state._pick_valid(pool,valid&(pool.family==family),15,generator)
            graph=state.to_graph(ref);values=t.a.serialize(batch);dd=tracker.differences(t.proposals(graph,values))
            if step%4==0:
                edges,nb=original.c.r5.topology(graph)
                for j in range(15):
                    original.c.r5.edit(edges,nb,names,values,j);_,raw=original.c.context(original.c.r5.graph_from(graph,edges),cols)
                    assert raw['raw'][index]-tracker.counts[index]==dd[j,index]
                    original.c.r5.edit(edges,nb,names,values,j,True)
            if dd[:,index].max()>0:choice=int(dd[:,index].argmax())
            else:
                available=np.flatnonzero(np.array(values['action'])==4)
                choice=int(available[int(torch.randint(0,len(available),(1,),generator=torch.Generator().manual_seed(seed+5000000)))]) if len(available) else 0
            chosen=original.base.take(batch,slice(choice,choice+1));v=t.a.serialize(chosen)
            before=state.edge_tables.clone();adj=state.adjacency.clone()
            tracker.apply(v,0);state.commit(chosen,torch.tensor(0));assert torch.equal(degree,state.family_degree)
            inverse=original.base.inverse(chosen);after=state.edge_tables.clone();after_adj=state.adjacency.clone()
            state.commit(inverse,torch.tensor(0));assert torch.equal(before,state.edge_tables) and torch.equal(adj,state.adjacency)
            state.commit(chosen,torch.tensor(0));assert torch.equal(after,state.edge_tables) and torch.equal(after_adj,state.adjacency)
            graph=state.to_graph(ref);_,raw=original.c.context(graph,cols);assert np.array_equal(raw['raw'],tracker.counts)
            path.append(dict(step=step,seed=seed,edit=v,cycle_count=int(tracker.counts[index])))
            if step%4:continue
            candidates=state.random_candidates(16,torch.Generator().manual_seed(seed+9000000),role_phase=False,targeted_fraction=0.,closure_fraction=0.,candidate_mode='legacy_pool')
            wanted=original.base.key(t.a.serialize(inverse),0);vals=t.a.serialize(candidates);hits=[i for i in range(16) if original.base.key(vals,i)==wanted]
            assert len(hits)<=1;label=hits[0] if hits else 15
            if not hits:
                for name,value in vars(candidates).items():value[15]=getattr(inverse,name)[0]
            vals=t.a.serialize(candidates);assert sum(original.base.key(vals,i)==wanted for i in range(16))==1
            ps=t.proposals(graph,vals);local=state.compact_features(candidates,step/128,width).numpy()
            cpu=original.mixed.mixed_edit_candidate_matrix(graph,ps,step/128,original.PROFILE)
            feature_error=max(feature_error,float(abs(local-cpu).max()));assert feature_error<=1e-5
            dd=tracker.differences(ps);covered=tracker.counts[index]>reference and dd[label,index]<0;positive+=int(covered)
            # Recompute inverse-list deltas from an independent freshly constructed state.
            assert np.array_equal(dd,Consequences(graph,cols,raw['raw']).differences(ps))
            xs.append(encode(tracker.counts,dd,ref.num_nodes,cols));ys.append(label);raws.append(tracker.counts.copy());deltas.append(dd);actions.append(vals['action'])
            records.append(dict(step=step,candidates=vals,label=label,counts=tracker.counts.tolist(),above_and_inverse_negative=bool(covered),components=raw['components']))
        z=original.pack(xs,ys,np.full(len(ys),2000+tid,dtype=np.int64),raws,deltas,actions,cols)
        folder=OUT/f'trajectories/{tid}';contract.put(folder,'record.json',dict(trajectory=tid,path=path,records=records,final_graph=b.graph_payload(graph)))
        np.savez_compressed(folder/'data.npz',**z);blocks.append(z)
        reports.append(dict(trajectory=tid,records=len(ys),above_and_inverse_negative=sum(r['above_and_inverse_negative'] for r in records),final_four_cycles=int(tracker.counts[index])))
        print('Four-cycle bank trajectory',tid,reports[-1],flush=True)
    merged={k:np.concatenate([z[k] for z in blocks]) for k in blocks[0]};assert len(merged['y'])==256
    np.savez_compressed(OUT/'augmentation.npz',**merged);contract.put(OUT,'schema.json',dict(dataset='imdb',n=ref.num_nodes,columns=cols,k=len(cols)))
    contract.verify(OUT);passed=positive>=128
    contract.put(OUT,'summary.json',dict(passed=passed,augmentation_repeat=repeat,records=256,forward_steps=1024,above_and_inverse_negative=positive,feature_error=feature_error,trajectories=reports,new_fits=0,new_complete_surrogates=0,seconds=time.monotonic()-tic))
    contract.put(OUT,'acceptance.json',dict(passed=True,coverage_gate_passed=passed,checked_forward_candidates=256*15,outputs={str(OUT/'augmentation.npz'):b.sha(OUT/'augmentation.npz'),str(OUT/'schema.json'):b.sha(OUT/'schema.json')},new_fits=0,new_complete_surrogates=0))
    print('Four-cycle augmentation accepted',passed,positive,flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument("repeat",type=int,choices=[2,3,4]);args=parser.parse_args()
    try:main(args.repeat)
    except BaseException as exc:
        if OUT is not None and OUT.exists():contract.put(OUT,'failure.json',dict(type=type(exc).__name__,message=str(exc)))
        raise
