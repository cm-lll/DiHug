"""Component-merging priority in IMDB first-stage development."""
import round18_deploy as p
from round01 import t,b,np,torch,time,Path,ROOT,SRC,json
from dihug.core.role_guided_edit_diffusion import prefer_positive_role_coverage
from dihug.core.mixed_edit_diffusion import mixed_edit_candidate_matrix
OUT=ROOT/'results/typed_edit_autonomous_20261118/r22_prefix';ARMS=['C','D'];SEEDS=[108,109]
CP=b.RUN/'fits/imdb/seed810/M/model.pt';r5=p.r5


def put(name,v):
    f=OUT/name;f.parent.mkdir(parents=True,exist_ok=True);f.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')


def verify():
    p.verify()
    for f,h in b.read(OUT/'manifest.json')['files'].items():assert b.sha(f)==h,f


def inputs(seed):
    folder=OUT/f'inputs/{seed}';return b.restore_graph(b.read(folder/'initial.json')),np.array(b.read(folder/'target.json'))


def changes(values,i):
    delta={}
    for field,valid,sign in [('old_edges','old_valid',-1),('new_edges','new_valid',1)]:
        for edge,yes in zip(values[field][i],values[valid][i]):
            if yes:
                for u in edge:delta[u]=delta.get(u,0)+sign
    return delta


def independent(degrees,target,values):
    gains=[];cover=[];tot=degrees.sum(1);positive=target.sum(1)>0
    for i,f in enumerate(values['family']):
        delta=changes(values,i);gains.append(sum((degrees[u,f]-target[u,f])**2-(degrees[u,f]+d-target[u,f])**2 for u,d in delta.items()))
        cover.append(sum(int(positive[u] and tot[u]==0)-int(positive[u] and tot[u]+d==0) for u,d in delta.items()))
    return np.array(gains),np.array(cover)


def edit(edges,nb,names,values,i,reverse=False):
    if values['action'][i]==0:return
    assert values['action'][i]==3;f=names[values['family'][i]]
    fields=[('old_edges','old_valid',-1),('new_edges','new_valid',1)]
    if reverse:fields=[('new_edges','new_valid',-1),('old_edges','old_valid',1)]
    for field,valid,sign in fields:
        for uv,yes in zip(values[field][i],values[valid][i]):
            if not yes:continue
            u,v=sorted(uv);assert u!=v
            if sign<0:assert (u,v) in edges[f];edges[f].remove((u,v));nb[u].remove(v);nb[v].remove(u)
            else:assert (u,v) not in edges[f] and v not in nb[u];edges[f].add((u,v));nb[u].add(v);nb[v].add(u)


def merging(edges,nb,names,values,exhaustive=False):
    labels=np.full(len(nb),-1,dtype=np.int64);component=0
    for root in range(len(nb)):
        if labels[root]>=0:continue
        labels[root]=component;stack=[root]
        while stack:
            u=stack.pop()
            for v in nb[u]:
                if labels[v]<0:labels[v]=component;stack.append(v)
        component+=1
    result=np.zeros(len(values['action']),dtype=np.int64)
    for i in range(len(result)):
        cross=any(yes and labels[uv[0]]!=labels[uv[1]] for uv,yes in zip(values['new_edges'][i],values['new_valid'][i]))
        if cross or exhaustive:
            edit(edges,nb,names,values,i);gain=component-p.components(nb);edit(edges,nb,names,values,i,True)
            if not cross:assert gain<=0
            result[i]=max(gain,0)
    return result


def stats(g,target):
    v=p.value('imdb',g);deg=np.column_stack([g.family_degrees(f) for f in sorted(g.family_specs)])
    return dict(**v,isolates=int((deg.sum(1)==0).sum()),positive_target_zeros=int(((target.sum(1)>0)&(deg.sum(1)==0)).sum()),target_isolates=int((target.sum(1)==0).sum()),degree_error_per_node=float(((deg-target)**2).sum()/g.num_nodes))


@torch.no_grad()
def chain(seed,arm,limit,m,engineering=False):
    initial,target=inputs(seed);state=t.a.CpuAuditState(initial,target,'cpu',use_triton=False);edges,nb=r5.topology(initial);names=sorted(edges)
    gen=torch.Generator().manual_seed(2026111821000+100000*seed+99999);weight=.05;cal={};tic=time.monotonic();calls=0
    if arm!='D':
        logits=[];gains=[]
        for _ in range(32):
            batch=state.random_candidates(16,gen,role_phase=True,targeted_fraction=0.,closure_fraction=0.,candidate_mode='legacy_pool')
            logits.append(m(state.compact_features(batch,1.,m.input_dim)[None])[0]);gains.append(state.degree_gains(batch).abs());calls+=1
        qq=torch.cat(logits);gg=torch.cat(gains);gg=gg[gg>0];std=qq.std().clamp_min(1e-6);q90=torch.quantile(gg,.9).clamp_min(1e-6);weight=float(.5*std*10/q90)
        cal=dict(logit_std=float(std),positive_gain_q90=float(q90),calibration_calls=32)
    trace=[];xx=[];qs=[];arrived=False;snapshot=None;loop_start=time.monotonic();initial_components=p.components(nb)
    for step in range(limit):
        draw=2026111821000+100000*seed+step;tf=1-step/20000
        batch=state.random_candidates(16,torch.Generator().manual_seed(draw),role_phase=True,targeted_fraction=.5,closure_fraction=0.,candidate_mode='legacy_pool')
        vals=t.a.serialize(batch);gain=state.degree_gains(batch);coverage=state.positive_role_coverage_gains(batch);x=state.compact_features(batch,tf,40)
        if arm=='D':q=torch.zeros(16);score=.05*gain.to(torch.float32)/10+1e-6*torch.rand(16,generator=torch.Generator().manual_seed(draw+5000000))
        else:q=m(x[None])[0];calls+=1;score=q+weight*gain.to(q.dtype)/10
        if arm!='O':score=prefer_positive_role_coverage(score,gain,coverage)
        merges=merging(edges,nb,names,vals)
        if engineering:assert np.array_equal(merges,merging(edges,nb,names,vals,True))
        if (merges>0).any():score=score+(torch.tensor(merges)>0).to(score.dtype)*((score.max()-score.min()).abs()+1)
        chosen=int(score.argmax());before=p.components(nb);edit(edges,nb,names,vals,chosen);after=p.components(nb);reject=arm!='O' and after>before
        if reject:edit(edges,nb,names,vals,chosen,True);after=p.components(nb)
        effective=0 if reject else chosen;state.commit(batch,torch.tensor(effective));err=float(state.relation_degree_error());zeros=int(state.positive_role_zero_count())
        if engineering:
            before_graph=r5.graph_from(initial,edges);deg=np.column_stack([before_graph.family_degrees(f) for f in names])
            for u,d in changes(vals,effective).items():deg[u,vals['family'][effective]]-=d
            gg,cg=independent(deg,target,vals);assert np.array_equal(gg,gain.numpy()) and np.array_equal(cg,coverage.numpy())
            assert err==float(((state.family_degree.numpy().T-target)**2).sum())
        trace.append(dict(step=step+1,proposal_seed=draw,time_fraction=tf,candidates=vals,degree_gains=gain.tolist(),coverage_gains=coverage.tolist(),merge_gains=merges.tolist(),chosen=chosen,effective=effective,rejected=reject,components_before=before,components_after=after,degree_error=err,positive_target_zeros=zeros));xx.append(x.numpy());qs.append(q.numpy())
        if step+1==1024:
            checkpoint=state.to_graph(initial);snapshot=dict(graph=b.graph_payload(checkpoint),metrics=stats(checkpoint,target))
        if (step+1)%32==0 or step+1==limit:
            if err/initial.num_nodes<=5.5 and (zeros==0 and after==1):arrived=True;break
    loop_seconds=time.monotonic()-loop_start;graph=state.to_graph(initial);graph.validate()
    for f in names:assert edges[f]=={tuple(sorted(e)) for e in graph.family_edges[f]} and len(graph.family_edges[f])==len(initial.family_edges[f])
    assert calls==0 if arm=='D' else calls==32+len(trace)
    report=dict(seed=seed,arm=arm,steps=len(trace),arrived=arrived,effective_role_weight=weight,calibration=cal,model_calls=calls,initial_components=initial_components,checkpoint_1024=snapshot,seconds=time.monotonic()-tic,loop_seconds=loop_seconds,**stats(graph,target))
    return graph,trace,np.stack(xx),np.stack(qs),report


def main():
    b.setup();p.verify();assert b.read(ROOT/'results/typed_edit_autonomous_20261118/r21_prefix_1024/acceptance.json')['passed'];OUT.mkdir(parents=True,exist_ok=False);ref=b.graph('imdb');prior=b.MarginalGraphPrior.from_training_graph(ref);role=b.ExchangeableRoleDegreePrior.from_training_graph(ref)
    for seed in SEEDS:
        graph=b.FamilyCountCondition(prior,role.family_counts(prior.family_endpoints)).sample_graph(seed+71,mode='binomial');target=role.sample_assignment_for_graph(graph,seed+91919)
        put(f'inputs/{seed}/initial.json',b.graph_payload(graph));put(f'inputs/{seed}/target.json',target.tolist());put(f'inputs/{seed}/report.json',stats(graph,target))
    files=[Path(__file__),SRC/'ROUND21.md',SRC/'ROUND22.md',SRC/'round21_prefix_1024.py',ROOT/'results/typed_edit_autonomous_20261118/r21_prefix_1024/acceptance.json',CP,b.RUN/'data/imdb/graph.json',Path(t.a.__file__)]
    files.extend(OUT/f'inputs/{s}/{name}' for s in SEEDS for name in ['initial.json','target.json'])
    put('manifest.json',dict(files={str(f):b.sha(f) for f in files},seeds=SEEDS,arms=ARMS,maximum_scientific_opportunities=6144,new_training_updates=0))
    m=t.a.load_mixed_edit_checkpoint(CP,'cpu')[0].eval();cost=30.;eng=[]
    for seed in SEEDS:
        for arm in ARMS:
            chain(seed,arm,16,m,True)
            _,_,_,_,rp=chain(seed,arm,32,m,False);eng.append(rp);cost+=2*(rp['seconds']+47*rp['loop_seconds'])
    put('engineering.json',dict(passed=cost<=1200,cost_bound_seconds=cost,engineering_opportunities=192,rows=eng,independent_candidate_gains_and_coverage=True));assert cost<=1200;verify();tic=time.monotonic();rows=[]
    for seed in SEEDS:
        for arm in ARMS:
            assert time.monotonic()-tic<1200
            graph,trace,x,q,rp=chain(seed,arm,1536,m);folder=Path('outputs')/f'{seed}_{arm}';put(folder/'graph.json',b.graph_payload(graph));put(folder/'trace.json',trace);put(folder/'report.json',rp)
            np.savez_compressed(OUT/folder/'features.npz',x=x,logits=q);rows.append(rp);print('prefix generated',seed,arm,rp['steps'],rp['arrived'],rp['components'],rp['isolates'],round(time.monotonic()-tic,2),flush=True)
    put('completion.json',dict(completed=True,new_graphs=6,scientific_opportunities=sum(v['steps'] for v in rows),scientific_model_calls=sum(v['model_calls'] for v in rows),new_training_updates=0,seconds=time.monotonic()-tic));tic=time.monotonic();replayed=0;featerr=0.;logerr=0.
    for row in rows:
        seed=row['seed'];arm=row['arm'];folder=OUT/'outputs'/f'{seed}_{arm}';initial,target=inputs(seed);edges,nb=r5.topology(initial);names=sorted(edges);degrees=np.column_stack([initial.family_degrees(f) for f in names]);trace=b.read(folder/'trace.json');saved=np.load(folder/'features.npz')
        if arm!='D':
            _,q,_=b.c.eval_model(m,saved['x'],np.zeros(len(trace),dtype=np.int64),'cpu');logerr=max(logerr,float(abs(q-saved['logits']).max()));assert logerr<=2e-5
        else:assert not saved['logits'].any() and row['model_calls']==0
        for i,v in enumerate(trace):
            assert time.monotonic()-tic<1200
            vals=v['candidates'];gg,cg=independent(degrees,target,vals);assert np.array_equal(gg,v['degree_gains']) and np.array_equal(cg,v['coverage_gains']);gain=torch.tensor(gg);cov=torch.tensor(cg)
            if i%32==0:
                g=r5.graph_from(initial,edges);fresh=mixed_edit_candidate_matrix(g,t.proposals(g,vals),v['time_fraction'],'typed_closure_compact_v2');featerr=max(featerr,float(abs(fresh-saved['x'][i]).max()));assert featerr<=1e-5
            if arm=='D':score=.05*gain.to(torch.float32)/10+1e-6*torch.rand(16,generator=torch.Generator().manual_seed(v['proposal_seed']+5000000))
            else:score=torch.from_numpy(saved['logits'][i])+row['effective_role_weight']*gain.to(torch.float32)/10
            if arm!='O':score=prefer_positive_role_coverage(score,gain,cov)
            merges=merging(edges,nb,names,vals);assert np.array_equal(merges,v['merge_gains'])
            if (merges>0).any():score=score+(torch.tensor(merges)>0).to(score.dtype)*((score.max()-score.min()).abs()+1)
            assert int(score.argmax())==v['chosen'];before=p.components(nb);assert before==v['components_before'];edit(edges,nb,names,vals,v['chosen']);after=p.components(nb);reject=arm!='O' and after>before;assert reject==v['rejected']
            if reject:edit(edges,nb,names,vals,v['chosen'],True);after=p.components(nb)
            effective=0 if reject else v['chosen'];assert effective==v['effective'] and after==v['components_after']
            for u,d in changes(vals,effective).items():degrees[u,vals['family'][effective]]+=d
            assert float(((degrees-target)**2).sum())==v['degree_error'];zeros=int(((target.sum(1)>0)&(degrees.sum(1)==0)).sum());assert zeros==v['positive_target_zeros'];replayed+=1
            if i+1==1024:
                checkpoint=r5.graph_from(initial,edges);snap=row['checkpoint_1024'];stored=b.restore_graph(snap['graph'])
                for f in names:assert edges[f]=={tuple(sorted(e)) for e in stored.family_edges[f]}
                assert stats(checkpoint,target)==snap['metrics']
        final=r5.graph_from(initial,edges);expected=b.restore_graph(b.read(folder/'graph.json'))
        for f in names:assert edges[f]=={tuple(sorted(e)) for e in expected.family_edges[f]} and len(edges[f])==len(initial.family_edges[f])
        for k,v in stats(final,target).items():assert row[k]==v
        assert row['arrived']==(row['degree_error_per_node']<=5.5 and (row['positive_target_zeros']==0 and row['components']==1))
        print('prefix accepted',seed,arm,round(time.monotonic()-tic,2),flush=True)
    cc=[v for v in rows if v['arm']=='C'];put('summary.json',dict(rows=rows,C_arrival_and_coverage_passed=all(v['arrived'] and v['positive_target_zeros']==0 and v['components']<=v['initial_components'] for v in cc),C_whole_connected=all(v['connected'] for v in cc),complete_two_stage=False))
    verify();put('acceptance.json',dict(passed=True,replayed_opportunities=replayed,feature_error=featerr,logit_error=logerr,seconds=time.monotonic()-tic))


if __name__=='__main__':
    try:main()
    except BaseException as exc:put('failure.json',dict(type=type(exc).__name__,message=str(exc)));raise
