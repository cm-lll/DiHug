"""Training-aligned standalone potential scorer, shared-start development chains."""
import round14 as train
import round13_deploy as prior
from itertools import product
import round08_cache as dc
import round07 as r7
from candidate_consequences import Consequences
from round01 import t,b,np,torch,time,Path,ROOT,SRC,json,csv
r5=r7.r5;cache=r7.cache;a=t.a;metrics=t.metrics
OUT=ROOT/'results/typed_edit_autonomous_20261118/r14_deploy'


def put(name,v):
    p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')


def model(seed,variant):
    m=train.Model();m.load_state_dict(torch.load(train.OUT/f'fits/{seed}/checkpoint.pt',map_location='cpu',weights_only=False)['state_dict']);return m.eval()


def verify():
    train.verify();prior.verify()
    for p,h in b.read(OUT/'manifest.json')['files'].items():assert b.sha(p)==h,p


@torch.no_grad()
def chain(seed,variant,steps,engineering=False):
    initial=b.restore_graph(b.read(a.GEN/'prefix/initial.json'));target=np.array(b.read(a.GEN/'prefix/target.json'))
    columns=b.read(cache.OUT/'schema.json')['columns'];tracker=Consequences(initial,columns)
    state=a.CpuAuditState(initial,target,'cpu',use_triton=False);degrees=state.family_degree.clone();m=model(seed,variant)
    records=[];features=[];scores=[];error=0.;start=time.monotonic()
    for step in range(steps):
        batch=state.random_candidates(16,torch.Generator().manual_seed(r5.BASE_SEED+step),role_phase=False,
            targeted_fraction=0.,closure_fraction=0.,candidate_mode='legacy_pool')
        values=a.serialize(batch);tf=1-(6464+step)/32000;x=state.compact_features(batch,tf,45)
        ps=t.proposals(initial,values);d=tracker.normalized(ps);ctx=tracker.features()
        inp=np.concatenate([x.numpy(),np.tile(ctx,(16,1)),d,np.tile(tracker.counts,(16,1)).astype(np.float32)],-1)
        q=m(torch.from_numpy(inp)[None]).numpy()[0];assert np.isfinite(q).all();chosen=int(q.argmax())
        if engineering:
            graph=r5.graph_from(initial,tracker.edges);cpu=r7.mixed_edit_candidate_matrix(graph,ps,tf,'typed_closure_compact_v2')
            error=max(error,float(abs(cpu-x.numpy()).max()));assert error<=1e-5
            current,counts=cache.context(graph,columns);assert np.array_equal(current,ctx) and np.array_equal(counts['raw'],tracker.counts)
            before=tracker.counts.copy();tracker.proposal(ps[chosen]);_,after=cache.context(r5.graph_from(initial,tracker.edges),columns)
            assert np.array_equal((np.array(after['raw'])-before)/tracker.scale,d[chosen].astype(np.float64)) or np.allclose((np.array(after['raw'])-before)/tracker.scale,d[chosen],atol=1e-7,rtol=1e-7)
            tracker.proposal(ps[chosen],True)
        reject=tracker.guarded(values,chosen);effective=0 if reject else chosen
        state.commit(batch,torch.tensor(effective));assert torch.equal(state.family_degree,degrees) and tracker.connected()
        records.append(dict(step=step,proposal_seed=r5.BASE_SEED+step,time_fraction=tf,candidates=values,
            chosen=chosen,effective=effective,connectivity_rejected=reject))
        features.append(inp);scores.append(q)
    final=state.to_graph(initial);final.validate()
    for f in tracker.names:
        assert tracker.edges[f]=={tuple(sorted(e)) for e in final.family_edges[f]}
        assert np.array_equal(final.family_degrees(f),initial.family_degrees(f))
    return final,records,np.stack(features),np.stack(scores),dict(seconds=time.monotonic()-start,local_feature_error=error)


def main():
    b.setup();train.verify();assert b.read(train.OUT/'acceptance.json')['passed'] and b.read(train.r11.OUT/'raw_precision.json')['passed'];OUT.mkdir(parents=True,exist_ok=False)
    files=[Path(__file__),SRC/'ROUND14.md',SRC/'candidate_consequences.py',SRC/'current_state.py',train.OUT/'summary.json',train.OUT/'acceptance.json',
        prior.OUT/'summary.json',prior.OUT/'acceptance.json',train.r11.OUT/'raw_precision.json']
    for seed in t.SEEDS:files.append(train.OUT/f'fits/{seed}/checkpoint.pt')
    put('manifest.json',dict(files={str(p):b.sha(p) for p in files},new_opportunities=256,new_output_graphs=2,reused_outputs=19))
    times=[]
    for seed,variant in product(t.SEEDS,['S']):
        final,trace,x,q,report=chain(seed,variant,8,True);times.append(report['seconds']);put(f'engineering/{variant}_{seed}.json',dict(**report,trace=trace))
    cost=2*16*sum(times)+30;put('engineering.json',dict(passed=cost<=300,cost_bound_seconds=cost,engineering_opportunities=16,engineering_model_forwards=16))
    assert cost<=300;verify();start=time.monotonic();reference=b.graph('pubmed');rc=metrics.counts(reference)
    initial=b.restore_graph(b.read(a.GEN/'prefix/initial.json'));zero=b.read(r5.OUT/'zero.json');rows=[]
    for seed,variant in product(t.SEEDS,['S']):
        assert time.monotonic()-start<300
        arm=f'{variant}_{seed}';final,trace,x,q,report=chain(seed,variant,128)
        put(f'outputs/{arm}/graph.json',b.graph_payload(final));put(f'outputs/{arm}/trace.json',trace)
        np.savez_compressed(OUT/f'outputs/{arm}/features.npz',x=x,logits=q)
        v=r5.r4.compare_values(final,reference,rc);row=dict(arm=arm,**v,delta_error=v['primary_error']-zero['primary_error'],
            noop_choices=sum(v['candidates']['action'][v['chosen']]==0 for v in trace),
            connectivity_rejections=sum(v['connectivity_rejected'] for v in trace),seconds=report['seconds'])
        rows.append(row);put(f'outputs/{arm}/report.json',row);print('generated',arm,round(time.monotonic()-start,2),flush=True)
    put('completion.json',dict(completed=True,scientific_opportunities=256,scientific_forwards=256,new_graphs=2,seconds=time.monotonic()-start))
    start=time.monotonic();columns=b.read(cache.OUT/'schema.json')['columns'];feature_error=0.;logit_error=0.;replayed=0
    for seed,variant in product(t.SEEDS,['S']):
        arm=f'{variant}_{seed}';trace=b.read(OUT/f'outputs/{arm}/trace.json');saved=np.load(OUT/f'outputs/{arm}/features.npz')
        edges,nb=r5.topology(initial);names=sorted(edges)
        for i,record in enumerate(trace):
            graph=r5.graph_from(initial,edges);ctx,raw=cache.context(graph,columns);ps=t.proposals(graph,record['candidates'])
            d=dc.delta(graph,ps,columns,raw['raw'],check=i%16==0)
            cpu=r7.mixed_edit_candidate_matrix(graph,ps,record['time_fraction'],'typed_closure_compact_v2')
            inp=np.concatenate([cpu,np.tile(ctx,(16,1)),d,np.tile(np.array(raw['raw'],dtype=np.float32),(16,1))],-1)
            feature_error=max(feature_error,float(abs(inp-saved['x'][i]).max()));assert feature_error<=1e-5
            chosen=record['chosen'];assert chosen==int(saved['logits'][i].argmax())
            if i%16==0:
                r5.edit(edges,nb,names,record['candidates'],chosen)
                _,af=cache.context(r5.graph_from(initial,edges),columns)
                scale=np.array([np.sqrt(graph.num_nodes) if a=='typed_triangles' else graph.num_nodes for a,k in columns])
                assert np.allclose((np.array(af['raw'])-np.array(raw['raw']))/scale,d[chosen],atol=1e-7,rtol=1e-7)
                r5.edit(edges,nb,names,record['candidates'],chosen,reverse=True)
            r5.edit(edges,nb,names,record['candidates'],chosen);reject=not r5.connected(nb);assert reject==record['connectivity_rejected']
            if reject:r5.edit(edges,nb,names,record['candidates'],chosen,reverse=True)
            assert record['effective']==(0 if reject else chosen) and r5.connected(nb);replayed+=1
        graph=r5.graph_from(initial,edges);expected=b.restore_graph(b.read(OUT/f'outputs/{arm}/graph.json'))
        for f in names:assert set(graph.family_edges[f])==set(expected.family_edges[f])
        value=r5.r4.compare_values(graph,reference,rc);row=next(v for v in rows if v['arm']==arm)
        for k,v in value.items():assert v==row[k]
        _,q,_=b.c.eval_model(model(seed,variant),saved['x'],np.zeros(128,dtype=np.int64),'cpu')
        logit_error=max(logit_error,float(abs(q-saved['logits']).max()));assert logit_error<=2e-5
        print('accepted',arm,round(time.monotonic()-start,2),flush=True)
    previous=b.read(prior.OUT/'summary.json');old=previous['reused_rows']+previous['new_rows'];n=next(v['primary_error'] for v in old if v['arm']=='N')
    decisions=[dict(seed=int(v['arm'].split('_')[-1]),passed=v['primary_error']<min(n,zero['primary_error'])) for v in rows if v['arm'].startswith('S_')]
    put('summary.json',dict(new_rows=rows,reused_rows=old,zero=zero,decisions=decisions,short_chain_signal_passed=all(v['passed'] for v in decisions),development_only=True))
    with (OUT/'per_chain.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(old+rows)
    verify();put('acceptance.json',dict(passed=True,replayed_opportunities=replayed,current_context_full_recounts=256,
        local_feature_max_error=feature_error,logit_max_error=logit_error,seconds=time.monotonic()-start))
    print('short chain signal',decisions,flush=True)


if __name__=='__main__':
    try:main()
    except BaseException as exc:put('failure.json',dict(type=type(exc).__name__,message=str(exc)));raise
