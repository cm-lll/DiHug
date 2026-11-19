"""Prepared independent-corruption producer; no execution on import.

Uses the accepted R17 construction with explicitly separated repeat seeds.
Activation, sample count, fitting rule, and confirmation starts require a frozen
confirmation protocol after R30. This module alone starts no experiments.
"""
import round17_cache as legacy
from potential import encode
from candidate_consequences import Consequences
from round01 import t,b,np,torch,time
c=legacy.c;base=legacy.base;mixed=legacy.mixed;pack=legacy.pack
PROFILE=legacy.PROFILE


@torch.no_grad()
def trajectory(ds,repeat,bank,kind,tid,columns,steps=None,deadline=None):
    horizon=128 if bank=='short' else 2048;steps=horizon if steps is None else steps;every=4 if bank=='short' else 16
    ref=b.graph(ds);target=np.column_stack([ref.family_degrees(f) for f in sorted(ref.family_specs)]);state=t.a.CpuAuditState(ref,target,'cpu',use_triton=False)
    degrees=state.family_degree.clone();tracker=Consequences(ref,columns);width=np.load(b.RUN/f'cache/{ds}/data.npz')['x'].shape[-1]
    assert repeat in [1,2,3,999]
    seed_base=202611183100000+{'imdb':0,'pubmed':100000000000}[ds]+repeat*1000000000+{'short':0,'deep':100000000,'typed':200000000}[bank]
    xs=[];ys=[];raws=[];deltas=[];actions=[];records=[];path=[];start=time.monotonic();maxerr=0.
    for step in range(1,steps+1):
        if deadline is not None:assert time.monotonic()<deadline
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
    return z,dict(dataset=ds,repeat=repeat,bank=bank,kind=kind,tid=tid,steps=steps,record_every=every,records=records,path=path,final_graph=b.graph_payload(final),seconds=time.monotonic()-start,max_local_feature_error=maxerr)
