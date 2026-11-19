"""New optimization-order fits on two pre-specified additional shared starts."""
import round14_deploy as prev
import round16_train as train
from round01 import t,b,np,torch,time,Path,ROOT,SRC,json,csv
from candidate_consequences import Consequences
OUT=ROOT/'results/typed_edit_autonomous_20261118/r16_deploy'
ARMS=['S_812','S_813','S_814','N','NS'];STARTS=[106,107];BASE=2026111816000


def put(name,v):
    p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')


def model(arm):
    if not arm.startswith('S_'):return None
    seed=int(arm.split('_')[-1]);m=train.Model();m.load_state_dict(torch.load(train.OUT/f'fits/{seed}/checkpoint.pt',map_location='cpu',weights_only=False)['state_dict']);return m.eval()


def verify():
    train.verify()
    for p,h in b.read(OUT/'manifest.json')['files'].items():assert b.sha(p)==h,p


def input_matrix(counts,delta,n):
    x=np.zeros((16,123),dtype=np.float32);scale=np.array([np.sqrt(n)]*20+[n]*6)
    x[:,71:97]=(delta/scale).astype(np.float32);x[:,97:]=counts.astype(np.float32);return x


@torch.no_grad()
def chain(arm,steps,base,start_id):
    start=time.monotonic();folder=b.RUN/f'prefixes/pubmed/seed{start_id}';binding=b.read(folder/'binding.json')
    assert b.sha(folder/'initial.json')==binding['initial_sha256'] and b.sha(folder/'target.json')==binding['target_sha256']
    initial=b.restore_graph(b.read(folder/'initial.json'));target=np.array(b.read(folder/'target.json'));offset=binding['transition_step']
    cols=b.read(prev.cache.OUT/'schema.json')['columns'];tracker=Consequences(initial,cols);state=t.a.CpuAuditState(initial,target,'cpu',use_triton=False)
    degrees=state.family_degree.clone();assert tracker.connected();m=model(arm);init_seconds=time.monotonic()-start;start=time.monotonic()
    records=[];counts=[];deltas=[];scores=[];snapshots={};loop_seconds=0.;metric_seconds=0.
    for step in range(steps):
        tick=time.monotonic();seed=base+step
        batch=state.random_candidates(16,torch.Generator().manual_seed(seed),role_phase=False,targeted_fraction=0.,closure_fraction=0.,candidate_mode='legacy_pool')
        values=t.a.serialize(batch);raw=tracker.counts.copy();delta=tracker.differences(t.proposals(initial,values));counts.append(raw);deltas.append(delta)
        if m is None:
            available=np.arange(16) if arm=='N' else np.flatnonzero(np.array(values['action'])==4)
            chosen=int(available[int(torch.randint(0,len(available),(1,),generator=torch.Generator().manual_seed(seed+5000000)))]) if len(available) else 0;q=None
        else:
            q=m(torch.from_numpy(input_matrix(raw,delta,initial.num_nodes))[None]).numpy()[0];assert np.isfinite(q).all();chosen=int(q.argmax())
        reject=tracker.guarded(values,chosen);effective=0 if reject else chosen
        state.commit(batch,torch.tensor(effective));assert torch.equal(degrees,state.family_degree) and tracker.connected()
        records.append(dict(step=step,proposal_seed=seed,time_fraction=1-(offset+step)/32000,candidates=values,chosen=chosen,effective=effective,connectivity_rejected=reject));scores.append(q)
        loop_seconds+=time.monotonic()-tick
        if step+1 in {128,steps}:
            tick=time.monotonic();graph=state.to_graph(initial);graph.validate()
            for f in tracker.names:
                assert tracker.edges[f]=={tuple(sorted(e)) for e in graph.family_edges[f]}
                assert np.array_equal(graph.family_degrees(f),initial.family_degrees(f))
            _,actual=prev.cache.context(graph,cols);assert np.array_equal(actual['raw'],tracker.counts)
            reference=b.graph('pubmed');metric=prev.r5.r4.compare_values(graph,reference,t.metrics.counts(reference))
            snapshots[step+1]=dict(graph=b.graph_payload(graph),metric=metric);metric_seconds+=time.monotonic()-tick
    return records,np.stack(counts),np.stack(deltas),None if m is None else np.stack(scores),snapshots,dict(init_seconds=init_seconds,loop_seconds=loop_seconds,metric_seconds=metric_seconds,seconds=time.monotonic()-start+init_seconds)


def main():
    b.setup();train.verify();assert b.read(train.OUT/'acceptance.json')['passed'];OUT.mkdir(parents=True,exist_ok=False)
    files=[Path(__file__),SRC/'ROUND16.md',SRC/'candidate_consequences.py',train.OUT/'summary.json',train.OUT/'acceptance.json']
    for seed in train.SEEDS:files.append(train.OUT/f'fits/{seed}/checkpoint.pt')
    for sid in STARTS:files.extend([b.RUN/f'prefixes/pubmed/seed{sid}/{name}' for name in ['initial.json','target.json','binding.json']])
    put('manifest.json',dict(files={str(p):b.sha(p) for p in files},arms=ARMS,start_ids=STARTS,steps=512,scientific_opportunities=5120,model_scored_contexts=3072,new_training_updates=0))
    cost=30.;zero={}
    for sid in STARTS:
        timing=[]
        initial=b.restore_graph(b.read(b.RUN/f'prefixes/pubmed/seed{sid}/initial.json'));reference=b.graph('pubmed')
        zero[str(sid)]=prev.r5.r4.compare_values(initial,reference,t.metrics.counts(reference))['primary_error']
        for arm in ARMS:
            trace,raw,delta,q,snaps,rp=chain(arm,8,BASE+100000*sid,sid);timing.append(rp);put(f'engineering/{sid}_{arm}.json',rp)
        cost+=2*sum(v['init_seconds']+64*v['loop_seconds']+2*v['metric_seconds'] for v in timing)
    put('zero.json',zero)
    put('engineering.json',dict(passed=cost<=900,cost_bound_seconds=cost,engineering_opportunities=80))
    assert cost<=900;verify();start=time.monotonic();rows=[]
    for stream in STARTS:
        for arm in ARMS:
            assert time.monotonic()-start<900
            trace,raw,delta,q,snaps,rp=chain(arm,512,BASE+100000*stream,stream);folder=Path('outputs')/f'start{stream}'/arm
            put(folder/'trace.json',trace);np.savez_compressed(OUT/folder/'features.npz',counts=raw,delta=delta,logits=np.empty((0,16),dtype=np.float32) if q is None else q)
            put(folder/'timing.json',rp)
            for step,snapshot in snaps.items():
                put(folder/f'graph{step}.json',snapshot['graph']);row=dict(start_id=stream,arm=arm,steps=step,**snapshot['metric'],
                    noop_choices=sum(v['candidates']['action'][v['chosen']]==0 for v in trace[:step]),
                    connectivity_rejections=sum(v['connectivity_rejected'] for v in trace[:step]),
                    effective_switches=sum(v['candidates']['action'][v['effective']]==4 for v in trace[:step]))
                rows.append(row);put(folder/f'report{step}.json',row)
            print('additional start completed',stream,arm,round(time.monotonic()-start,2),flush=True)
    put('completion.json',dict(completed=True,new_graphs=10,snapshot_evaluations=20,scientific_opportunities=5120,model_scored_contexts=3072,seconds=time.monotonic()-start))
    verify();start=time.monotonic();cols=b.read(prev.cache.OUT/'schema.json')['columns'];maxerr=0.;fullchecks=0;replayed=0
    for stream in STARTS:
        initial=b.restore_graph(b.read(b.RUN/f'prefixes/pubmed/seed{stream}/initial.json'))
        for arm in ARMS:
            folder=OUT/f'outputs/start{stream}/{arm}';trace=b.read(folder/'trace.json');saved=np.load(folder/'features.npz');edges,nb=prev.r5.topology(initial);names=sorted(edges);m=model(arm)
            xx=np.stack([input_matrix(raw,delta,initial.num_nodes) for raw,delta in zip(saved['counts'],saved['delta'])])
            if m is not None:
                _,q,_=b.c.eval_model(m,xx,np.zeros(512,dtype=np.int64),'cpu');maxerr=max(maxerr,float(abs(q-saved['logits']).max()));assert maxerr<=2e-5
            for i,record in enumerate(trace):
                assert time.monotonic()-start<1200
                values=record['candidates'];raw=saved['counts'][i];delta=saved['delta'][i]
                if i%32==0:
                    graph=prev.r5.graph_from(initial,edges);_,actual=prev.cache.context(graph,cols);assert np.array_equal(raw,actual['raw'])
                    independent=Consequences(graph,cols,actual['raw']).differences(t.proposals(graph,values));assert np.array_equal(independent,delta);fullchecks+=1
                chosen=record['chosen']
                if m is not None:assert chosen==int(saved['logits'][i].argmax())
                else:
                    available=np.arange(16) if arm=='N' else np.flatnonzero(np.array(values['action'])==4)
                    expected=int(available[int(torch.randint(0,len(available),(1,),generator=torch.Generator().manual_seed(record['proposal_seed']+5000000)))]) if len(available) else 0
                    assert chosen==expected
                prev.r5.edit(edges,nb,names,values,chosen);reject=not prev.r5.connected(nb);assert reject==record['connectivity_rejected']
                if reject:prev.r5.edit(edges,nb,names,values,chosen,reverse=True)
                effective=0 if reject else chosen;assert record['effective']==effective and prev.r5.connected(nb)
                if i+1<len(trace):assert np.array_equal(raw+delta[effective],saved['counts'][i+1])
                if i+1 in [128,512]:
                    graph=prev.r5.graph_from(initial,edges);expected=b.restore_graph(b.read(folder/f'graph{i+1}.json'))
                    for f in names:assert edges[f]=={tuple(sorted(e)) for e in expected.family_edges[f]} and np.array_equal(graph.family_degrees(f),initial.family_degrees(f))
                    reference=b.graph('pubmed');v=prev.r5.r4.compare_values(graph,reference,t.metrics.counts(reference));row=b.read(folder/f'report{i+1}.json')
                    for k,value in v.items():assert value==row[k]
                replayed+=1
            print('additional start accepted',stream,arm,round(time.monotonic()-start,2),flush=True)
    zero=b.read(OUT/'zero.json');decisions=[]
    for stream in STARTS:
        rr={v['arm']:v for v in rows if v['start_id']==stream and v['steps']==512}
        for arm in ['S_812','S_813','S_814']:
            decisions.append(dict(start_id=stream,arm=arm,error=rr[arm]['primary_error'],N=rr['N']['primary_error'],NS=rr['NS']['primary_error'],passed=rr[arm]['primary_error']<min(zero[str(stream)],rr['N']['primary_error'],rr['NS']['primary_error'])))
    put('summary.json',dict(rows=rows,zero=zero,decisions=decisions,new_fit_other_start_passed=all(v['passed'] for v in decisions),new_training_updates=0,confirmation_scope='new optimization repeats and other previously generated starts on one reference'))
    with (OUT/'per_output.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    aggregates=[]
    for step in [128,512]:
        for arm in ARMS:
            rr=[v for v in rows if v['steps']==step and v['arm']==arm];aggregates.append(dict(arm=arm,steps=step,outputs=2,training_fits=0 if arm in ['N','NS'] else 1,
                primary_error_mean=float(np.mean([v['primary_error'] for v in rr])),across_start_sample_sd=float(np.std([v['primary_error'] for v in rr],ddof=1))))
    fit_summary=[]
    for step in [128,512]:
        vals=[v['primary_error_mean'] for v in aggregates if v['steps']==step and v['arm'].startswith('S_')]
        fit_summary.append(dict(steps=step,optimization_repeats=3,graphs_per_repeat=2,mean=float(np.mean(vals)),fit_sample_sd=float(np.std(vals,ddof=1))))
    put('aggregation.json',dict(per_arm=aggregates,S_fit_aggregate=fit_summary))
    verify();put('acceptance.json',dict(passed=True,replayed_opportunities=replayed,sampled_full_state_recounts=fullchecks,snapshot_metric_recomputations=20,logit_error=maxerr,seconds=time.monotonic()-start))
    print('new-fit/start decisions',decisions,flush=True)


if __name__=='__main__':
    try:main()
    except BaseException as exc:put('failure.json',dict(type=type(exc).__name__,message=str(exc)));raise
