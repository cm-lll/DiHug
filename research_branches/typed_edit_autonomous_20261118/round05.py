"""Seven fixed shared-prefix short chains; sparse edit consequences independently replayed."""
import round01 as r
import round03 as r3
import round04_cost600 as r4
from round01 import t,b,np,torch,time,Path,ROOT,SRC,json,csv
from dihug.core.mixed_edit_diffusion import mixed_edit_candidate_matrix
a=t.a;metrics=t.metrics
OUT=ROOT/'results/typed_edit_autonomous_20261118/r05'
ARMS=['parent_810','parent_811','C800_810','C800_811','contextC_810','contextC_811','N']
BASE_SEED=2026111805000


def put(name,v):
    p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')


def model(arm):
    if arm=='N':return None
    seed=int(arm.split('_')[-1])
    if arm.startswith('parent'):return t.model(seed).eval()
    context=arm.startswith('contextC');m=r.ContextModel(seed) if context else t.model(seed)
    cp=r3.OUT/f'fits/{seed}/{"contextC800" if context else "C800"}/checkpoint.pt'
    m.load_state_dict(torch.load(cp,map_location='cpu',weights_only=False)['state_dict']);return m.eval()


def verify():
    r4.verify()
    for p,h in b.read(OUT/'manifest.json')['files'].items():assert b.sha(p)==h,p


def topology(graph):
    edges={f:{tuple(sorted(e)) for e in es} for f,es in graph.family_edges.items()}
    nb=[set() for _ in range(graph.num_nodes)]
    for es in edges.values():
        for u,v in es:nb[u].add(v);nb[v].add(u)
    return edges,nb


def connected(nb):
    seen={0};stack=[0]
    while stack:
        u=stack.pop()
        for v in nb[u]:
            if v not in seen:seen.add(v);stack.append(v)
    return len(seen)==len(nb)


def edit(edges,nb,names,values,index,reverse=False):
    if values['action'][index]==0:return
    assert values['action'][index]==4;f=names[values['family'][index]];change={}
    fields=[('old_edges','old_valid',-1),('new_edges','new_valid',1)]
    if reverse:fields=[('new_edges','new_valid',-1),('old_edges','old_valid',1)]
    for field,valid,sign in fields:
        for uv,yes in zip(values[field][index],values[valid][index]):
            if not yes:continue
            u,v=sorted(uv);assert u!=v
            if sign<0:
                assert (u,v) in edges[f];edges[f].remove((u,v));nb[u].remove(v);nb[v].remove(u)
            else:
                assert (u,v) not in edges[f];edges[f].add((u,v));nb[u].add(v);nb[v].add(u)
            change[u]=change.get(u,0)+sign;change[v]=change.get(v,0)+sign
    assert all(x==0 for x in change.values())


def graph_from(initial,edges):return b.TypedGraph(initial.node_types.copy(),initial.type_names,initial.family_specs,{f:sorted(es) for f,es in edges.items()})


@torch.no_grad()
def chain(arm,steps,engineering=False):
    initial=b.restore_graph(b.read(a.GEN/'prefix/initial.json'));target=np.array(b.read(a.GEN/'prefix/target.json'))
    offset=b.read(a.GEN/'prefix/binding.json')['transition_step'];assert offset==6464
    state=a.CpuAuditState(initial,target,'cpu',use_triton=False);m=model(arm);edges,nb=topology(initial);names=sorted(edges)
    degrees=state.family_degree.clone();assert connected(nb);records=[];features=[];scores=[];maxerr=0.;start=time.monotonic()
    for step in range(steps):
        batch=state.random_candidates(16,torch.Generator().manual_seed(BASE_SEED+step),role_phase=False,
            targeted_fraction=0.,closure_fraction=0.,candidate_mode='legacy_pool')
        values=a.serialize(batch);tf=1-(offset+step)/32000;x=state.compact_features(batch,tf,45)
        if engineering:
            cpu=mixed_edit_candidate_matrix(graph_from(initial,edges),t.proposals(initial,values),tf,'typed_closure_compact_v2')
            maxerr=max(maxerr,float(abs(x.numpy()-cpu).max()));assert maxerr<=1e-5
        if m is None:q=None;chosen=int(torch.randint(0,16,(1,),generator=torch.Generator().manual_seed(BASE_SEED+step+5000000)))
        else:q=m(x[None]).numpy()[0];assert np.isfinite(q).all();chosen=int(q.argmax())
        edit(edges,nb,names,values,chosen);reject=not connected(nb)
        if reject:edit(edges,nb,names,values,chosen,reverse=True)
        effective=0 if reject else chosen;assert values['action'][0]==0
        state.commit(batch,torch.tensor(effective));assert torch.equal(state.family_degree,degrees)
        records.append(dict(step=step,proposal_seed=BASE_SEED+step,time_fraction=tf,candidates=values,
            chosen=chosen,effective=effective,connectivity_rejected=reject))
        features.append(x.numpy());scores.append(q)
    final=state.to_graph(initial);final.validate()
    for f in names:
        assert edges[f]=={tuple(sorted(e)) for e in final.family_edges[f]}
        assert np.array_equal(final.family_degrees(f),initial.family_degrees(f))
    assert connected(nb)
    return final,records,np.stack(features),None if m is None else np.stack(scores),dict(seconds=time.monotonic()-start,feature_error=maxerr)


def main():
    b.setup();r4.verify();assert b.read(r4.OUT/'acceptance.json')['passed']
    assert next(x for x in b.read(r4.OUT/'summary.json')['signals'] if x['kind']=='contextC')['local_signal']
    OUT.mkdir(parents=True,exist_ok=False)
    files=[Path(__file__),SRC/'ROUND05.md',r4.OUT/'summary.json',r4.OUT/'acceptance.json',
        a.GEN/'prefix/initial.json',a.GEN/'prefix/target.json',a.GEN/'prefix/binding.json']
    for seed in t.SEEDS:
        for arm in ['C800','contextC800']:files.append(r3.OUT/f'fits/{seed}/{arm}/checkpoint.pt')
    put('manifest.json',dict(files={str(p):b.sha(p) for p in files},arms=ARMS,steps=128,proposal_seed=BASE_SEED))
    reference=b.graph('pubmed');rc=metrics.counts(reference);initial=b.restore_graph(b.read(a.GEN/'prefix/initial.json'))
    zero=r4.compare_values(initial,reference,rc);put('zero.json',zero);times=[]
    for arm in ARMS:
        final,trace,x,q,report=chain(arm,8,True);times.append(report['seconds'])
        put(f'engineering/{arm}.json',dict(**report,trace=trace))
    cost=2*16*sum(times)+30;put('engineering.json',dict(passed=cost<=300,cost_bound_seconds=cost,engineering_opportunities=56,engineering_model_forwards=48,times=times))
    assert cost<=300;verify();start=time.monotonic();rows=[]
    for arm in ARMS:
        assert time.monotonic()-start<300
        final,trace,x,q,report=chain(arm,128)
        put(f'outputs/{arm}/graph.json',b.graph_payload(final));put(f'outputs/{arm}/trace.json',trace)
        np.savez_compressed(OUT/f'outputs/{arm}/features.npz',x=x,logits=np.empty((0,)) if q is None else q)
        result=r4.compare_values(final,reference,rc)
        row=dict(arm=arm,**result,delta_error=result['primary_error']-zero['primary_error'],
            noop_choices=sum(v['candidates']['action'][v['chosen']]==0 for v in trace),
            connectivity_rejections=sum(v['connectivity_rejected'] for v in trace),seconds=report['seconds'])
        rows.append(row);put(f'outputs/{arm}/report.json',row)
        print('generated',arm,round(time.monotonic()-start,2),flush=True)
    put('completion.json',dict(completed=True,scientific_opportunities=896,scientific_model_forwards=768,new_graphs=7,seconds=time.monotonic()-start))
    start=time.monotonic();feature_error=0.;logit_error=0.;replayed=0
    for arm in ARMS:
        trace=b.read(OUT/f'outputs/{arm}/trace.json');saved=np.load(OUT/f'outputs/{arm}/features.npz');edges,nb=topology(initial);names=sorted(edges)
        for i,record in enumerate(trace):
            graph=graph_from(initial,edges);v=record['candidates']
            cpu=mixed_edit_candidate_matrix(graph,t.proposals(graph,v),record['time_fraction'],'typed_closure_compact_v2')
            feature_error=max(feature_error,float(abs(cpu-saved['x'][i]).max()));assert feature_error<=1e-5
            chosen=record['chosen']
            if arm=='N':assert chosen==int(torch.randint(0,16,(1,),generator=torch.Generator().manual_seed(BASE_SEED+i+5000000)))
            else:assert chosen==int(saved['logits'][i].argmax())
            edit(edges,nb,names,v,chosen);reject=not connected(nb);assert reject==record['connectivity_rejected']
            if reject:edit(edges,nb,names,v,chosen,reverse=True)
            assert record['effective']==(0 if reject else chosen) and connected(nb);replayed+=1
        final=graph_from(initial,edges);expected=b.restore_graph(b.read(OUT/f'outputs/{arm}/graph.json'))
        for f in names:assert set(final.family_edges[f])==set(expected.family_edges[f])
        actual=r4.compare_values(final,reference,rc);row=next(x for x in rows if x['arm']==arm)
        for key,value in actual.items():assert value==row[key]
        if arm!='N':
            m=model(arm);_,q,_=b.c.eval_model(m,saved['x'],np.zeros(128,dtype=np.int64),'cpu')
            logit_error=max(logit_error,float(abs(q-saved['logits']).max()));assert logit_error<=2e-5
        print('accepted',arm,round(time.monotonic()-start,2),flush=True)
    n=next(x['primary_error'] for x in rows if x['arm']=='N')
    signals=[dict(seed=seed,passed=next(x['primary_error'] for x in rows if x['arm']==f'contextC_{seed}')<min(n,zero['primary_error'])) for seed in t.SEEDS]
    put('summary.json',dict(rows=rows,zero=zero,decisions=signals,short_chain_signal_passed=all(x['passed'] for x in signals),development_only=True,
        independent_from_scratch_fits=0,classification_gates_remain_failed=True))
    with (OUT/'per_chain.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    verify();put('acceptance.json',dict(passed=True,replayed_opportunities=replayed,feature_max_error=feature_error,logit_max_error=logit_error,
        all_relation_degrees_and_connectivity_preserved=True,seconds=time.monotonic()-start))
    print('short chain signal',signals,flush=True)


if __name__=='__main__':
    try:main()
    except BaseException as exc:put('failure.json',dict(type=type(exc).__name__,message=str(exc)));raise
