"""Same-pool highest component-admissible score; bounded first-stage selection repair."""
import confirmation_prefix_v2 as core
from round01 import t,b,np,torch,time
p=core.p;r5=core.r5;inputs=core.inputs;stats=core.stats
changes=core.changes;independent=core.independent;merging=core.merging;edit=core.edit
prefer_positive_role_coverage=core.prefer_positive_role_coverage

@torch.no_grad()
def chain(job,arm,limit,m,device,engineering=False,evaluate=True,deadline=None):
    ds=job['dataset'];seed=job['seed'];base=job['base_seed'];horizon=20000 if ds=='imdb' else 32000;tic=time.monotonic()
    initial,target=inputs(job);state=t.a.CpuAuditState(initial,target,device,use_triton=False);edges,nb=r5.topology(initial);names=sorted(edges)
    gen=torch.Generator(device=device).manual_seed(base+99999);weight=.05;cal={};calls=0
    if arm!='D':
        logits=[];gains=[]
        for _ in range(32):
            batch=state.random_candidates(16,gen,role_phase=True,targeted_fraction=0.,closure_fraction=0.,candidate_mode='legacy_pool')
            logits.append(m(state.compact_features(batch,1.,m.input_dim)[None])[0].cpu());gains.append(state.degree_gains(batch).abs().cpu());calls+=1
        qq=torch.cat(logits);gg=torch.cat(gains);gg=gg[gg>0];std=qq.std().clamp_min(1e-6);q90=torch.quantile(gg,.9).clamp_min(1e-6);weight=float(.5*std*10/q90)
        cal=dict(logit_std=float(std),positive_gain_q90=float(q90),calibration_calls=32)
    trace=[];xx=[];qs=[];arrived=False;snapshot=None;loop_start=time.monotonic();initial_components=p.components(nb)
    for step in range(limit):
        if deadline is not None:assert time.monotonic()<deadline
        draw=base+step;tf=1-step/horizon
        batch=state.random_candidates(16,torch.Generator(device=device).manual_seed(draw),role_phase=True,targeted_fraction=.5,closure_fraction=0.,candidate_mode='legacy_pool')
        vals=t.a.serialize(batch);gain=state.degree_gains(batch).cpu();coverage=state.positive_role_coverage_gains(batch).cpu();x=state.compact_features(batch,tf,m.input_dim)
        if arm=='D':q=torch.zeros(16);score=.05*gain.to(torch.float32)/10+1e-6*torch.rand(16,generator=torch.Generator().manual_seed(draw+5000000))
        else:q=m(x[None])[0].cpu();calls+=1;score=q+weight*gain.to(q.dtype)/10
        if arm!='O':score=prefer_positive_role_coverage(score,gain,coverage)
        merges=merging(edges,nb,names,vals)
        if engineering:assert np.array_equal(merges,merging(edges,nb,names,vals,True))
        if (merges>0).any():score=score+(torch.tensor(merges)>0).to(score.dtype)*((score.max()-score.min()).abs()+1)
        unconstrained=int(score.argmax());before=p.components(nb);checked=[]
        for candidate in sorted(range(16),key=lambda j:(-float(score[j]),j)):
            edit(edges,nb,names,vals,candidate);after=p.components(nb);checked.append(candidate)
            if after<=before:
                chosen=candidate;break
            edit(edges,nb,names,vals,candidate,True)
        else:raise AssertionError('noop must be component-admissible')
        reject=False
        if engineering:
            edit(edges,nb,names,vals,chosen,True);legal=[]
            for j in range(16):
                edit(edges,nb,names,vals,j);legal.append(p.components(nb)<=before);edit(edges,nb,names,vals,j,True)
            wanted=min((j for j in range(16) if legal[j]),key=lambda j:(-float(score[j]),j));assert chosen==wanted
            edit(edges,nb,names,vals,chosen)
        effective=chosen;state.commit(batch,torch.tensor(effective,device=device));err=float(state.relation_degree_error());zeros=int(state.positive_role_zero_count())
        if engineering:
            before_graph=r5.graph_from(initial,edges);deg=np.column_stack([before_graph.family_degrees(f) for f in names])
            for u,d in changes(vals,effective).items():deg[u,vals['family'][effective]]-=d
            gg,cg=independent(deg,target,vals);assert np.array_equal(gg,gain.numpy()) and np.array_equal(cg,coverage.numpy())
            assert err==float(((state.family_degree.cpu().numpy().T-target)**2).sum())
        trace.append(dict(step=step+1,unconstrained_chosen=unconstrained,legality_checked_indices=checked,proposal_seed=draw,time_fraction=tf,candidates=vals,degree_gains=gain.tolist(),coverage_gains=coverage.tolist(),merge_gains=merges.tolist(),chosen=chosen,effective=effective,rejected=reject,components_before=before,components_after=after,degree_error=err,positive_target_zeros=zeros));xx.append(x.cpu().numpy());qs.append(q.numpy())
        if step+1==1024:
            checkpoint=state.to_graph(initial);snapshot=dict(graph=b.graph_payload(checkpoint),metrics=stats(ds,checkpoint,target))
        if (step+1)%32==0 or step+1==limit:
            if err/initial.num_nodes<=5.5 and (zeros==0 and after==1):arrived=True;break
    loop_seconds=time.monotonic()-loop_start;graph=state.to_graph(initial);graph.validate()
    for f in names:assert edges[f]=={tuple(sorted(e)) for e in graph.family_edges[f]} and len(graph.family_edges[f])==len(initial.family_edges[f])
    assert calls==0 if arm=='D' else calls==32+len(trace)
    report=dict(dataset=ds,seed=seed,arm=arm,steps=len(trace),arrived=arrived,effective_role_weight=weight,calibration=cal,model_calls=calls,initial_components=initial_components,checkpoint_1024=snapshot,seconds=time.monotonic()-tic,loop_seconds=loop_seconds,**(stats(ds,graph,target) if evaluate else dict(components=p.components(nb),degree_error_per_node=err/initial.num_nodes,positive_target_zeros=zeros)))
    return graph,trace,np.stack(xx),np.stack(qs),report

