"""Fresh paired starts with targeted/ordinary/base/Q/random suffix selection."""
import argparse,concurrent.futures,multiprocessing
import confirmation_deploy as oldpub
import additional_confirmation_deploy as oldacm
import crossdomain_augmentation_fit as repair
import crossdomain_augmentation_diagnosis as diagnosis
import confirmation_common as contract
from potential import ComponentState,encode
from candidate_consequences import Consequences
from round01 import t,b,np,torch,time,Path,ROOT,SRC
BASE=repair.BASE

def inputs(job):
    if job.get('legacy'):return (oldpub if job['dataset']=='pubmed' else oldacm).inputs(job)
    parent=BASE/f"confirmation/prefix/{job['dataset']}";sid=job['seed'];f=parent/f"outputs/{sid}_{job['producer']}"
    return b.restore_graph(b.read(f/'graph.json')),np.array(b.read(parent/f'inputs/{sid}/target.json')),b.read(f/'report.json')

def model(job):
    ds=job['dataset'];arm=job['arm'];repeat=job['repeat']
    if arm in ['N','NS']:return None
    m=repair.model(ds,repeat)
    if arm in ['T','U']:
        cp=BASE/f"confirmation/fit/{ds}/bank{job['bank']}/{'targeted' if arm=='T' else 'ordinary'}/checkpoint.pt"
        m.load_state_dict(torch.load(cp,map_location='cpu',weights_only=False)['state_dict'])
    elif arm=='Q':
        m=m.float()
        with torch.no_grad():m.weights[:m.k].zero_();m.weights[m.k:].fill_(1.)
    else:assert arm=='B'
    return m.eval()

def output(job):return BASE/f"confirmation/deploy/{job['dataset']}/outputs/{job['seed']}_{job['producer']}/{job['arm']}_{job['bank']}"

@torch.no_grad()
def chain(job,steps,deadline=None):
    start=time.monotonic();g,target,pr=inputs(job);assert pr['arrived'];ds=job['dataset'];a=diagnosis.assets(ds,1 if ds=='pubmed' else 2);cols=b.read(a['schema'])['columns']
    tracker=ComponentState(g,cols);assert tracker.n_components==1;state=t.a.CpuAuditState(g,target,'cpu',use_triton=False);degree=state.family_degree.clone();m=model(job)
    init=time.monotonic()-start;loop=0.;evaluation=0.;trace=[];raws=[];deltas=[];scores=[];snaps={}
    for step in range(steps):
        if deadline is not None:assert time.monotonic()<deadline
        tic=time.monotonic();seed=job.get('proposal_base',202611189400000+['pubmed','acm'].index(ds)*100000000000+job['seed']*10000)+step
        batch=state.random_candidates(16,torch.Generator().manual_seed(seed),role_phase=False,targeted_fraction=0.,closure_fraction=0.,candidate_mode='legacy_pool')
        vals=t.a.serialize(batch);raw=tracker.counts.copy();dd=tracker.differences(t.proposals(g,vals))
        if m is None:chosen=oldpub.choose_random(vals,job['arm'],seed);q=None
        else:
            xx=torch.from_numpy(encode(raw,dd,g.num_nodes,cols))[None];q=m(xx.float() if job['arm']=='Q' else xx.double()).numpy()[0];assert np.isfinite(q).all();chosen=int(q.argmax())
        reject=tracker.guarded(vals,chosen);effective=0 if reject else chosen;state.commit(batch,torch.tensor(effective));assert torch.equal(degree,state.family_degree) and tracker.n_components==1
        trace.append(dict(step=step,proposal_seed=seed,candidates=vals,chosen=chosen,effective=effective,connectivity_rejected=reject));raws.append(raw);deltas.append(dd);scores.append(q);loop+=time.monotonic()-tic
        if step+1 in [512,1024,2048,steps]:
            tic=time.monotonic();graph=state.to_graph(g);graph.validate()
            for f in tracker.names:assert tracker.edges[f]=={tuple(sorted(e)) for e in graph.family_edges[f]} and np.array_equal(graph.family_degrees(f),g.family_degrees(f))
            _,fresh=oldpub.p.cache.context(graph,cols);assert np.array_equal(fresh['raw'],tracker.counts)
            snaps[step+1]=dict(graph=b.graph_payload(graph),metrics=oldpub.prefix.stats(ds,graph,target));evaluation+=time.monotonic()-tic
    return trace,np.stack(raws),np.stack(deltas),None if m is None else np.stack(scores),snaps,dict(init_seconds=init,loop_seconds=loop,metric_seconds=evaluation)

def run_job(job):
    torch.set_num_threads(1);tic=time.monotonic();tr,raw,dd,q,snaps,rp=chain(job,2048,deadline=tic+4800);folder=output(job);contract.put(folder,'trace.json',tr);contract.put(folder,'timing.json',rp)
    np.savez_compressed(folder/'features.npz',counts=raw,delta=dd,logits=np.empty((0,16)) if q is None else q);_,_,pr=inputs(job);rows=[]
    for step,snap in snaps.items():
        contract.put(folder,f'graph{step}.json',snap['graph']);row=dict(**job,steps=step,**snap['metrics'],first_stage_steps=pr['steps'],total_attempts=pr['steps']+step,effective_switches=sum(v['candidates']['action'][v['effective']]==4 for v in tr[:step]),connectivity_rejections=sum(v['connectivity_rejected'] for v in tr[:step]));contract.put(folder,f'report{step}.json',row);rows.append(row)
    scientific=time.monotonic()-tic;contract.put(folder,'completion.json',dict(completed=True,scientific_seconds=scientific,opportunities=2048));print('New suffix generated',job,rows[-1]['primary_error'],flush=True)
    tic=time.monotonic();g,target,_=inputs(job);edges,nb=oldpub.r5.topology(g);names=sorted(edges);a=diagnosis.assets(job['dataset'],1 if job['dataset']=='pubmed' else 2);cols=b.read(a['schema'])['columns'];m=model(job);error=0.
    if m is not None:
        with torch.no_grad():
            xx=torch.from_numpy(np.stack([encode(r,d,g.num_nodes,cols) for r,d in zip(raw,dd)]));qq=m(xx.float() if job['arm']=='Q' else xx.double()).numpy()
        error=float(abs(qq-q).max());assert error<=(2e-5 if job['arm']=='Q' else 1e-9)
    full=0
    for i,v in enumerate(tr):
        assert time.monotonic()-tic<3600
        vals=v['candidates'];counts=raw[i];delta=dd[i];seed=202611189400000+['pubmed','acm'].index(job['dataset'])*100000000000+job['seed']*10000+i;assert v['proposal_seed']==seed
        if i%32==0:
            graph=oldpub.r5.graph_from(g,edges);_,fresh=oldpub.p.cache.context(graph,cols);assert np.array_equal(fresh['raw'],counts)
            assert np.array_equal(Consequences(graph,cols,fresh['raw']).differences(t.proposals(graph,vals)),delta);full+=1
        expected=oldpub.choose_random(vals,job['arm'],seed) if m is None else int(qq[i].argmax());assert expected==v['chosen']
        oldpub.r5.edit(edges,nb,names,vals,expected);reject=not oldpub.r5.connected(nb);assert reject==v['connectivity_rejected']
        if reject:oldpub.r5.edit(edges,nb,names,vals,expected,True)
        effective=0 if reject else expected;assert effective==v['effective'] and oldpub.r5.connected(nb)
        if i+1<len(tr):assert np.array_equal(counts+delta[effective],raw[i+1])
        if i+1 in [512,1024,2048]:
            graph=oldpub.r5.graph_from(g,edges);stored=b.restore_graph(b.read(folder/f'graph{i+1}.json'))
            for f in names:assert edges[f]=={tuple(sorted(e)) for e in stored.family_edges[f]} and np.array_equal(graph.family_degrees(f),g.family_degrees(f))
            report=b.read(folder/f'report{i+1}.json')
            for k,value in oldpub.prefix.stats(job['dataset'],graph,target).items():assert report[k]==value
    result=dict(job=job,rows=rows,scientific_seconds=scientific,audit_seconds=time.monotonic()-tic,replayed_opportunities=len(tr),full_state_recounts=full,score_error=error)
    contract.put(folder,'acceptance.json',dict(passed=True,**{k:v for k,v in result.items() if k not in ['rows','job']}));print('New suffix accepted',job,flush=True);return result

def main(ds):
    b.setup();out=BASE/f'confirmation/deploy/{ds}';out.mkdir(parents=True,exist_ok=False);prefix=BASE/f'confirmation/prefix/{ds}';repair.original.source_check(prefix)
    banks=[2,3,4];files=[Path(__file__),SRC/'CROSSDOMAIN_AUGMENTATION_CONFIRMATION.md',prefix/'manifest.json',prefix/'summary.json',prefix/'acceptance.json']
    for bank in banks:
        f=BASE/f'confirmation/fit/{ds}/bank{bank}';repair.original.source_check(f);assert b.read(f/'summary.json')['passed'];files += [f/'manifest.json',f/'acceptance.json',f/'summary.json']+[f/k/'checkpoint.pt' for k in ['targeted','ordinary']]
        repeat=bank-1 if ds=='pubmed' else bank;a=diagnosis.assets(ds,repeat)
        files += [a['checkpoint'],a['schema'],a['cache']/'normalization.json']
    files += [Path(p) for p in b.read(prefix/'acceptance.json')['outputs']]
    parity=diagnosis.assets(ds,1 if ds=='pubmed' else 2)
    files += [parity['folder'](sid,'F')/'trace.json' for sid in parity['seeds']]
    jobs=[];zero={};missing=[]
    for sid in [17001,17002]:
        for producer in ['C','D']:
            job=dict(dataset=ds,seed=sid,producer=producer);g,target,pr=inputs(job);zero[f'{sid}_{producer}']=oldpub.prefix.stats(ds,g,target)
            if not pr['arrived']:missing.append(dict(**job,reason='prefix_not_arrived'));continue
            if producer=='C':
                for bank in banks:
                    repeat=bank-1 if ds=='pubmed' else bank
                    jobs += [dict(**job,arm=arm,bank=bank,repeat=repeat) for arm in ['T','U','B','Q']]
                jobs += [dict(**job,arm=arm,bank=0,repeat=0) for arm in ['N','NS']]
            else:jobs.append(dict(**job,arm='N',bank=0,repeat=0))
    contract.bind(out,files,dataset=ds,jobs=jobs,maximum_chains=30,maximum_workers=2,unavailable=missing,old_data_and_first_stage_MLP_shared=True);contract.put(out,'zero.json',zero)
    # Reproduce old base paths before using the new explicit random streams.
    old=oldpub if ds=='pubmed' else oldacm;a=diagnosis.assets(ds,1 if ds=='pubmed' else 2)
    for sid in a['seeds']:
        legacy=dict(dataset=ds,seed=sid,producer='C',arm='B',bank=2,repeat=1 if ds=='pubmed' else 2,legacy=True,proposal_base=(202611183400000+1000000000 if ds=='pubmed' else 202611187400000)+sid*10000)
        tr=chain(legacy,8)[0];assert tr==b.read(a['folder'](sid,'F')/'trace.json')[:8]
    cost=30.;eng=[]
    for job in jobs:
        _,_,_,_,_,rp=chain(job,8);cost+=2*(rp['init_seconds']+256*rp['loop_seconds']+3*rp['metric_seconds']);eng.append(dict(**job,**rp))
    contract.put(out,'engineering.json',dict(passed=cost<=4800,cost_bound_seconds=cost,rows=eng,old_base_first_eight_steps_exact=True));assert cost<=4800;contract.verify(out);start=time.monotonic()
    results=[]
    with concurrent.futures.ProcessPoolExecutor(max_workers=2,mp_context=multiprocessing.get_context('spawn')) as pool:
        pending=[pool.submit(run_job,job) for job in jobs]
        for future in concurrent.futures.as_completed(pending,timeout=8500):results.append(future.result())
    scientific=sum(r['scientific_seconds'] for r in results);audit=sum(r['audit_seconds'] for r in results);assert scientific<=4800 and audit<=3600
    rows=[row for result in results for row in result['rows']];decisions=[]
    for sid in [17001,17002]:
        rr={(r['arm'],r['bank']):r for r in rows if r['seed']==sid and r['producer']=='C' and r['steps']==2048}
        for bank in banks:
            if ('T',bank) not in rr:decisions.append(dict(seed=sid,bank=bank,passed=False,reason='prefix_unavailable'));continue
            vals={arm:rr[arm,bank if arm in ['T','U','B','Q'] else 0]['primary_error'] for arm in ['T','U','B','Q','N','NS']};zz=zero[f'{sid}_C']['primary_error']
            decisions.append(dict(seed=sid,bank=bank,**vals,zero=zz,passed=vals['T']<min(vals['B'],vals['N'],vals['NS'],zz),better_than_Q=vals['T']<vals['Q'],better_than_ordinary=vals['T']<vals['U']))
    contract.verify(out);contract.put(out,'summary.json',dict(dataset=ds,passed=all(d['passed'] for d in decisions),decisions=decisions,rows=rows,unavailable=missing,scientific_seconds=scientific,audit_seconds=audit,wall_seconds=time.monotonic()-start,independent_new_augmentation_and_starts=True,independent_whole_system_retraining=False))
    contract.put(out,'acceptance.json',dict(passed=True,replayed_opportunities=sum(r['replayed_opportunities'] for r in results),full_state_recounts=sum(r['full_state_recounts'] for r in results),snapshot_readouts=len(rows),score_errors=[r['score_error'] for r in results],scientific_seconds=scientific,audit_seconds=audit,outputs={str(p):b.sha(p) for p in out.glob('outputs/*/*/graph*.json')}));print(ds,'confirmation closed',decisions,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('dataset',choices=['pubmed','acm']);a=p.parse_args()
    try:main(a.dataset)
    except BaseException as exc:
        out=BASE/f'confirmation/deploy/{a.dataset}'
        if out.exists():contract.put(out,'failure.json',dict(type=type(exc).__name__,message=str(exc)))
        raise
