"""Counterfactual single edits on fixed saved deployment states, not generation chains."""
import round01 as r
import round02 as r2
import round03 as r3
from round01 import t,b,np,torch,time,Path,ROOT,SRC,json,csv
from dihug.core.mixed_edit_diffusion import mixed_edit_candidate_matrix
a=t.a
metrics=t.metrics
OUT=ROOT/'results/typed_edit_autonomous_20261118/r04_cost600'
PROP=ROOT/'results/typed_edit_proposal_audit_20261117'


def put(name,v):
    p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')


def inputs():
    files=[Path(__file__),SRC/'ROUND04_COST_AMENDMENT.md',SRC/'ROUND04.md',SRC/'round04.py',PROP/'records.json',PROP/'acceptance.json',*a.STATES.values()]
    for seed in t.SEEDS:
        files += [r2.OUT/f'fits/{seed}/context800/checkpoint.pt',r3.OUT/f'fits/{seed}/contextC800/checkpoint.pt']
    return files


def verify():
    r3.verify()
    for p,h in b.read(OUT/'manifest.json')['files'].items():assert b.sha(p)==h,p


def apply(graph,values,index):
    names=sorted(graph.family_specs);edges={f:set(tuple(sorted(e)) for e in es) for f,es in graph.family_edges.items()}
    if values['action'][index]!=0:
        assert values['action'][index]==4;f=names[values['family'][index]]
        for pair,yes in zip(values['old_edges'][index],values['old_valid'][index]):
            if yes:
                edge=tuple(sorted(pair));assert edge in edges[f];edges[f].remove(edge)
        for pair,yes in zip(values['new_edges'][index],values['new_valid'][index]):
            if yes:
                edge=tuple(sorted(pair));assert edge not in edges[f];edges[f].add(edge)
    result=b.TypedGraph(graph.node_types.copy(),graph.type_names,graph.family_specs,{f:sorted(es) for f,es in edges.items()})
    result.validate()
    for f in names:assert np.array_equal(result.family_degrees(f),graph.family_degrees(f))
    return result


def compare_values(graph,reference,rc):
    result=metrics.compare(graph,reference,rc)
    return {k:result[k] for k in ['primary_error','triangle_vector_relative_l1','cycle_vector_relative_l1','triangles','connected','components','lcc_fraction']}


@torch.no_grad()
def assess(record,models,states,reference,rc):
    name=record['state'];graph,state,zero=states[name];policy=record['policies']['0'];values=policy['candidates']
    batch=a.deserialize(values);x=state.compact_features(batch,record['time_fraction'],45)
    cpu=mixed_edit_candidate_matrix(graph,t.proposals(graph,values),record['time_fraction'],'typed_closure_compact_v2')
    feature_error=float(abs(cpu-x.numpy()).max());assert feature_error<=1e-5
    selected={};logit_error=0.
    for key,m in models.items():
        q=m(x[None]).numpy()[0];selected[key]=dict(index=int(q.argmax()),logits=q.tolist())
        if key.startswith('parent_'):
            prior=policy['selections']['M_'+key.split('_')[-1]]['logits']
            logit_error=max(logit_error,float(abs(q-np.array(prior)).max()));assert logit_error<=2e-5
    selected['N']=dict(index=policy['selections']['N']['index'],logits=None)
    outcomes={}
    for index in {v['index'] for v in selected.values()}:
        raw=compare_values(apply(graph,values,index),reference,rc)
        reject=bool(zero['connected'] and not raw['connected'])
        outcomes[str(index)]=dict(raw=raw,connectivity_rejected=reject,guarded=zero if reject else raw)
    rows=[]
    for key,choice in selected.items():
        q=outcomes[str(choice['index'])];after=q['guarded']
        rows.append(dict(state=name,draw_seed=record['draw_seed'],model=key,index=choice['index'],
            action=values['action'][choice['index']],connectivity_rejected=q['connectivity_rejected'],
            delta_error=after['primary_error']-zero['primary_error'],
            raw_delta_error=q['raw']['primary_error']-zero['primary_error'],
            delta_triangles=after['triangles']-zero['triangles'],connected=after['connected'],lcc_fraction=after['lcc_fraction']))
    return dict(state=name,draw_seed=record['draw_seed'],time_fraction=record['time_fraction'],candidates=values,
        selections=selected,outcomes=outcomes,feature_error=feature_error,parent_logit_error=logit_error),rows


def main():
    b.setup();r3.verify();assert b.read(r3.OUT/'acceptance.json')['passed'];OUT.mkdir(parents=True,exist_ok=False)
    put('manifest.json',dict(files={str(p):b.sha(p) for p in inputs()},scientific_forwards=576,one_edit_readouts=672))
    reference=b.graph('pubmed');rc=metrics.counts(reference);target=np.array(b.read(a.GEN/'prefix/target.json'))
    states={}
    for key,path in a.STATES.items():
        graph=b.restore_graph(b.read(path));states[key]=(graph,a.CpuAuditState(graph,target,'cpu',use_triton=False),compare_values(graph,reference,rc))
    put('zero.json',{k:v[2] for k,v in states.items()});models={}
    for seed in t.SEEDS:
        models[f'parent_{seed}']=t.model(seed).eval()
        for key,source in [('contextB',r2.OUT/f'fits/{seed}/context800/checkpoint.pt'),('contextC',r3.OUT/f'fits/{seed}/contextC800/checkpoint.pt')]:
            m=r.ContextModel(seed);m.load_state_dict(torch.load(source,map_location='cpu',weights_only=False)['state_dict']);models[f'{key}_{seed}']=m.eval()
    records=b.read(PROP/'records.json');assert len(records)==96;times=[]
    for name in states:
        record=next(x for x in records if x['state']==name);start=time.monotonic();result,_=assess(record,models,states,reference,rc)
        times.append(time.monotonic()-start);put(f'engineering/{name}.json',result)
    cost=2*96*max(times)+30;put('engineering.json',dict(passed=cost<=600,cost_bound_seconds=cost,engineering_forwards=18,record_seconds=times))
    assert cost<=600;verify();start=time.monotonic();results=[];flat=[]
    for i,record in enumerate(records):
        assert time.monotonic()-start<600
        q,rr=assess(record,models,states,reference,rc);results.append(q);flat.extend(rr)
        if (i+1)%32==0:print('assessed',i+1,round(time.monotonic()-start,2),flush=True)
    put('records.json',results)
    with (OUT/'per_draw.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
    aggregate=[]
    for state in states:
        for model in list(models)+['N']:
            rr=[x for x in flat if x['state']==state and x['model']==model];assert len(rr)==32
            aggregate.append(dict(state=state,model=model,draws=32,mean_delta_error=float(np.mean([x['delta_error'] for x in rr])),
                draw_sample_sd=float(np.std([x['delta_error'] for x in rr],ddof=1)),
                mean_delta_triangles=float(np.mean([x['delta_triangles'] for x in rr])),
                noop_choices=sum(x['action']==0 for x in rr),connectivity_rejections=sum(x['connectivity_rejected'] for x in rr),
                paired_mean_delta_vs_N=float(np.mean([x['delta_error']-next(n['delta_error'] for n in flat if n['state']==state and n['draw_seed']==x['draw_seed'] and n['model']=='N') for x in rr]))))
    signals=[]
    for kind in ['contextB','contextC']:
        rr=[v for v in aggregate if v['state']=='prefix' and v['model'].startswith(kind+'_')]
        signals.append(dict(kind=kind,local_signal=all(v['mean_delta_error']<0 and v['paired_mean_delta_vs_N']<0 for v in rr)))
    put('summary.json',dict(per_state_model=aggregate,signals=signals,local_only=True,classification_gates_not_overridden=True,
        scientific_forwards=576,one_edit_readouts=672,seconds=time.monotonic()-start,surrogate_chains=0))
    start=time.monotonic();count=0
    for q in results:
        graph,_,zero=states[q['state']]
        for index,expected in q['outcomes'].items():
            fresh=compare_values(apply(graph,q['candidates'],int(index)),reference,rc)
            assert fresh==expected['raw'];assert expected['guarded']==(zero if expected['connectivity_rejected'] else fresh);count+=1
    verify();put('acceptance.json',dict(passed=True,independent_selected_outcome_recomputations=count,seconds=time.monotonic()-start,
        all_relation_degrees_exact=True,surrogate_chains=0))
    print('local signals',signals,flush=True)


if __name__=='__main__':
    try:main()
    except BaseException as exc:put('failure.json',dict(type=type(exc).__name__,message=str(exc)));raise
