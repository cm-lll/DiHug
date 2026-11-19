"""Frozen uniform-proposal suffix pilot on two separately fitted additional domains."""
import argparse
import confirmation_common as contract
import confirmation_deploy as legacy
import additional_confirmation_fit as fit
import confirmation_prefix_v2 as prefix
from potential import ComponentState,encode
from candidate_consequences import Consequences
from round01 import t,b,np,torch,time,Path,ROOT,SRC,csv
BASE=ROOT/'results/typed_edit_autonomous_20261118'
OUT=None;DATASETS=['acm','freebase'];SEEDS=[15001,15002]
r5=prefix.r5;p=prefix.p;choose_random=legacy.choose_random

def inputs(job):
    ds=job['dataset'];sid=job['seed']
    folder=BASE/f"additional_confirmation_v1/prefix/{ds}/outputs/{sid}_{job['producer']}"
    target=BASE/f"additional_confirmation_v1/prefix/{ds}/inputs/{sid}/target.json"
    return b.restore_graph(b.read(folder/'graph.json')),np.array(b.read(target)),b.read(folder/'report.json')

def model(job):
    if job['arm'] in ['N','NS']:return None
    ds=job['dataset'];fit.REPEAT=job['repeat'];m=fit.model(ds)
    if job['arm']=='F':
        m.load_state_dict(torch.load(BASE/f"additional_confirmation_v1/fit/{ds}/{job['repeat']}/checkpoint.pt",map_location='cpu',weights_only=False)['state_dict'])
    else:
        m=m.float()
        with torch.no_grad():m.weights[:m.k].zero_();m.weights[m.k:].fill_(1.)
    return m.eval()

def folder_for(job):
    return Path('outputs')/f"{job['seed']}_{job['producer']}"/f"{job['arm']}_{job['repeat']}"

def main(ds):
    global OUT
    b.setup();parents=[BASE/f'additional_confirmation_v1/{phase}/{ds}/{rep}' for phase in ['cache','fit'] for rep in [2,3,4]]+[BASE/f'additional_confirmation_v1/prefix/{ds}']
    for parent in parents:fit.source_check(parent)
    for rep in [2,3,4]:assert b.read(BASE/f'additional_confirmation_v1/fit/{ds}/{rep}/summary.json')['passed']
    OUT=BASE/f'additional_confirmation_v1/deploy/{ds}';OUT.mkdir(parents=True,exist_ok=False)
    jobs=[];zero={};unavailable=[]
    for sid in SEEDS:
        for producer in ['C','D']:
            job=dict(dataset=ds,seed=sid,producer=producer)
            g,target,pr=inputs(job);zero[f'{sid}_{producer}']=prefix.stats(ds,g,target)
            if not pr['arrived']:
                unavailable.append(dict(**job,reason='first_stage_did_not_arrive',report=pr));continue
            if producer=='C':
                jobs.extend(dict(**job,arm=arm,repeat=rep) for arm in ['F','Q'] for rep in [2,3,4])
                jobs.extend(dict(**job,arm=arm,repeat=0) for arm in ['N','NS'])
            else:jobs.append(dict(**job,arm='N',repeat=0))
    files=[Path(__file__),SRC/'ADDITIONAL_CONFIRMATION_V1.md']
    for parent in parents:
        files += [parent/'manifest.json',parent/'acceptance.json',parent/'summary.json']
        files += [Path(f) for f in b.read(parent/'acceptance.json')['outputs']]
    contract.bind(OUT,files,dataset=ds,jobs=jobs,unavailable=unavailable,new_training_updates=0,maximum_suffix_chains=18)
    contract.put(OUT,'zero.json',zero);contract.put(OUT,'unavailable.json',unavailable)
    cost=30.;eng=[];first={}
    for job in jobs:
        tr,_,_,_,_,rp=chain(job,8);key=f"{job['seed']}_{job['producer']}"
        if key in first:assert first[key]==tr[0]['candidates']
        else:first[key]=tr[0]['candidates']
        cost+=2*(rp['init_seconds']+256*rp['loop_seconds']+3*rp['metric_seconds']);eng.append(dict(**job,**rp))
    contract.put(OUT,'engineering.json',dict(passed=cost<=4800,cost_bound_seconds=cost,opportunities=8*len(jobs),rows=eng))
    assert cost<=4800;contract.verify(OUT);tic=time.monotonic();rows=[]
    for job in jobs:
        tr,raw,dd,q,snaps,rp=chain(job,2048,deadline=tic+4800);folder=OUT/folder_for(job)
        contract.put(folder,'trace.json',tr);contract.put(folder,'timing.json',rp)
        np.savez_compressed(folder/'features.npz',counts=raw,delta=dd,logits=np.empty((0,16)) if q is None else q)
        _,_,pr=inputs(job)
        for step,v in snaps.items():
            contract.put(folder,f'graph{step}.json',v['graph'])
            row=dict(**job,steps=step,**v['metrics'],first_stage_steps=pr['steps'],total_attempts=pr['steps']+step,effective_switches=sum(v['candidates']['action'][v['effective']]==4 for v in tr[:step]),connectivity_rejections=sum(v['connectivity_rejected'] for v in tr[:step]))
            contract.put(folder,f'report{step}.json',row);rows.append(row)
        print('Additional suffix generated',job,rows[-1]['primary_error'],flush=True)
    contract.put(OUT,'completion.json',dict(completed=True,new_suffix_chains=len(jobs),scientific_opportunities=2048*len(jobs),new_training_updates=0,seconds=time.monotonic()-tic))
    tic=time.monotonic();errors={};replayed=0;full=0
    for job in jobs:
        folder=OUT/folder_for(job);tr=b.read(folder/'trace.json');z=np.load(folder/'features.npz')
        g,target,_=inputs(job);edges,nb=r5.topology(g);names=sorted(edges)
        cols=b.read(parents[0]/'schema.json')['columns'];m=model(job)
        if m is not None:
            with torch.no_grad():
                xx=torch.from_numpy(np.stack([encode(raw,dd,g.num_nodes,cols) for raw,dd in zip(z['counts'],z['delta'])]));q=m(xx.double() if job['arm']=='F' else xx).numpy()
            err=float(abs(q-z['logits']).max());assert err<=(1e-9 if job['arm']=='F' else 2e-5);errors[str(folder_for(job))]=err
        for i,v in enumerate(tr):
            assert time.monotonic()-tic<3600
            vals=v['candidates'];raw=z['counts'][i];dd=z['delta'][i]
            if i%32==0:
                graph=r5.graph_from(g,edges);_,fresh=p.cache.context(graph,cols)
                assert np.array_equal(fresh['raw'],raw)
                assert np.array_equal(Consequences(graph,cols,fresh['raw']).differences(t.proposals(graph,vals)),dd);full+=1
            chosen=v['chosen'];expected=choose_random(vals,job['arm'],v['proposal_seed']) if m is None else int(q[i].argmax());assert chosen==expected
            r5.edit(edges,nb,names,vals,chosen);reject=not r5.connected(nb);assert reject==v['connectivity_rejected']
            if reject:r5.edit(edges,nb,names,vals,chosen,True)
            effective=0 if reject else chosen;assert effective==v['effective'] and r5.connected(nb)
            if i+1<len(tr):assert np.array_equal(raw+dd[effective],z['counts'][i+1])
            if i+1 in [512,1024,2048]:
                graph=r5.graph_from(g,edges);stored=b.restore_graph(b.read(folder/f'graph{i+1}.json'))
                for f in names:assert edges[f]=={tuple(sorted(e)) for e in stored.family_edges[f]} and np.array_equal(graph.family_degrees(f),g.family_degrees(f))
                report=b.read(folder/f'report{i+1}.json')
                for k,value in prefix.stats(ds,graph,target).items():assert report[k]==value
            replayed+=1
        print('Additional suffix accepted',job,flush=True)
    decisions=[]
    for sid in SEEDS:
        rr={(v['arm'],v['repeat']):v for v in rows if v['seed']==sid and v['producer']=='C' and v['steps']==2048}
        for rep in [2,3,4]:
            if ('F',rep) not in rr:decisions.append(dict(seed=sid,repeat=rep,passed=False,reason='unavailable_prefix'));continue
            values={arm:rr[arm,rep if arm in ['F','Q'] else 0]['primary_error'] for arm in ['F','Q','N','NS']};zz=zero[f'{sid}_C']['primary_error']
            decisions.append(dict(seed=sid,repeat=rep,**values,zero=zz,passed=values['F']<min(zz,values['N'],values['NS']),better_than_Q=values['F']<values['Q']))
    passed=len(decisions)==6 and all(r['passed'] for r in decisions)
    contract.put(OUT,'summary.json',dict(dataset=ds,passed=passed,decisions=decisions,rows=rows,unavailable_prefixes=unavailable,coefficient_fits=3,repeat_unit='fresh corruption bank; old examples and original MLP shared',independent_confirmation=True))
    if rows:
        fields=list(dict.fromkeys(k for row in rows for k in row))
        with (OUT/'per_output.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    contract.verify(OUT);outputs={str(f):b.sha(f) for f in OUT.glob('outputs/*/*/graph*.json')}
    contract.put(OUT,'acceptance.json',dict(passed=True,scientific_pilot_gate_passed=passed,replayed_opportunities=replayed,full_state_recounts=full,logit_errors=errors,outputs=outputs,seconds=time.monotonic()-tic))

@torch.no_grad()
def chain(job,steps,deadline=None):
    tic=time.monotonic();ds=job['dataset'];sid=job['seed'];arm=job['arm'];g,target,pr=inputs(job)
    assert pr['arrived'];cols=b.read(BASE/f'additional_confirmation_v1/cache/{ds}/2/schema.json')['columns'];tracker=ComponentState(g,cols);assert tracker.n_components==1
    state=t.a.CpuAuditState(g,target,'cpu',use_triton=False);degree=state.family_degree.clone();m=model(job)
    init=time.monotonic()-tic;loop=0.;evaluation=0.;trace=[];raws=[];deltas=[];scores=[];snaps={}
    for step in range(steps):
        if deadline is not None:assert time.monotonic()<deadline
        tic=time.monotonic();seed=202611187400000+DATASETS.index(ds)*1000000000+sid*10000+step
        batch=state.random_candidates(16,torch.Generator().manual_seed(seed),role_phase=False,targeted_fraction=0.,closure_fraction=0.,candidate_mode='legacy_pool')
        vals=t.a.serialize(batch);raw=tracker.counts.copy();dd=tracker.differences(t.proposals(g,vals))
        if m is None:chosen=choose_random(vals,arm,seed);q=None
        else:
            xx=torch.from_numpy(encode(raw,dd,g.num_nodes,cols))[None];q=m(xx.double() if arm=='F' else xx).numpy()[0];assert np.isfinite(q).all();chosen=int(q.argmax())
        rejected=tracker.guarded(vals,chosen);effective=0 if rejected else chosen;state.commit(batch,torch.tensor(effective));assert torch.equal(degree,state.family_degree) and tracker.n_components==1
        trace.append(dict(step=step,proposal_seed=seed,candidates=vals,chosen=chosen,effective=effective,connectivity_rejected=rejected));raws.append(raw);deltas.append(dd);scores.append(q);loop+=time.monotonic()-tic
        if step+1 in [512,1024,2048,steps]:
            tic=time.monotonic();graph=state.to_graph(g);graph.validate()
            for f in tracker.names:assert tracker.edges[f]=={tuple(sorted(e)) for e in graph.family_edges[f]} and np.array_equal(graph.family_degrees(f),g.family_degrees(f))
            _,fresh=p.cache.context(graph,cols);assert np.array_equal(fresh['raw'],tracker.counts)
            snaps[step+1]=dict(graph=b.graph_payload(graph),metrics=prefix.stats(ds,graph,target));evaluation+=time.monotonic()-tic
    return trace,np.stack(raws),np.stack(deltas),None if m is None else np.stack(scores),snaps,dict(init_seconds=init,loop_seconds=loop,metric_seconds=evaluation)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('dataset',choices=['acm','freebase']);args=parser.parse_args()
    try:main(args.dataset)
    except BaseException as exc:
        if OUT is not None:contract.put(OUT,'failure.json',dict(type=type(exc).__name__,message=str(exc)))
        raise

