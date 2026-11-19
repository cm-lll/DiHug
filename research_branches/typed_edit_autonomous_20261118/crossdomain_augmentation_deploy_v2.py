"""Paired pilot suffixes with bytecode-identical frozen generation policies."""
import argparse,types
import confirmation_deploy as pubmed
import additional_confirmation_deploy as acm
import crossdomain_augmentation_fit as repair
import crossdomain_augmentation_diagnosis as diagnosis
import confirmation_common as contract
from candidate_consequences import Consequences
from potential import encode
from round01 import b,np,torch,time,Path,ROOT,SRC

BASE=repair.BASE

def main(ds,repeat,bank):
    b.setup();a=diagnosis.assets(ds,repeat);fit=BASE/f'fit_v2/{ds}/bank{bank}';repair.original.source_check(fit);repair.original.source_check(a['deploy'])
    assert b.read(fit/'summary.json')['passed'];out=BASE/f'deploy/{ds}/bank{bank}';out.mkdir(parents=True,exist_ok=False)
    original=pubmed if ds=='pubmed' else acm;cols=b.read(a['schema'])['columns'];jobs=[dict(dataset=ds,seed=sid,producer='C',arm='F',repeat=repeat,augmentation=kind) for sid in a['seeds'] for kind in ['targeted','ordinary']]
    def model(job):
        m=repair.model(ds,repeat);m.load_state_dict(torch.load(fit/f"{job['augmentation']}/checkpoint.pt",map_location='cpu',weights_only=False)['state_dict']);return m.eval()
    scope=dict(original.chain.__wrapped__.__globals__,model=model)
    chain=torch.no_grad()(types.FunctionType(original.chain.__wrapped__.__code__,scope,'crossdomain_augmented_chain',original.chain.__wrapped__.__defaults__))
    def folder(job):return out/f"outputs/{job['seed']}/{job['augmentation']}"
    files=[Path(__file__),SRC/'CROSSDOMAIN_AUGMENTATION_VALIDATION.md',a['deploy']/'manifest.json',a['deploy']/'acceptance.json',a['deploy']/'summary.json',fit/'manifest.json',fit/'acceptance.json',fit/'summary.json']
    files += [fit/f'{kind}/checkpoint.pt' for kind in ['targeted','ordinary']]
    for sid in a['seeds']:
        for arm in ['F','Q','N','NS']:files += [a['folder'](sid,arm)/'report2048.json',a['folder'](sid,arm)/'trace.json']
    contract.bind(out,files,jobs=jobs,new_suffixes=4,new_first_stages=0,new_fits=0,same_chain_bytecode=True,development_only=True)
    cost=20.;eng=[]
    for job in jobs:
        oldtrace=original.chain(job,8)[0];assert oldtrace==b.read(a['folder'](job['seed'],'F')/'trace.json')[:8]
        tr,_,_,_,_,rp=chain(job,8);assert tr[0]['candidates']==oldtrace[0]['candidates']
        cost+=2*(rp['init_seconds']+256*rp['loop_seconds']+3*rp['metric_seconds']);eng.append(dict(**job,**rp))
    contract.put(out,'engineering.json',dict(passed=cost<=1800,cost_bound_seconds=cost,rows=eng,base_first_eight_steps_exact=True));assert cost<=1800;contract.verify(out);tic=time.monotonic();rows=[]
    for job in jobs:
        tr,raw,dd,q,snaps,rp=chain(job,2048,deadline=tic+1800);f=folder(job);contract.put(f,'trace.json',tr);contract.put(f,'timing.json',rp);np.savez_compressed(f/'features.npz',counts=raw,delta=dd,logits=q)
        _,_,pr=original.inputs(job)
        for step,snap in snaps.items():
            contract.put(f,f'graph{step}.json',snap['graph']);row=dict(**job,steps=step,**snap['metrics'],first_stage_steps=pr['steps'],total_attempts=pr['steps']+step)
            contract.put(f,f'report{step}.json',row);rows.append(row)
        print(ds,job['seed'],job['augmentation'],'generated',rows[-1]['primary_error'],flush=True)
    contract.put(out,'completion.json',dict(completed=True,new_suffixes=4,opportunities=8192,seconds=time.monotonic()-tic));tic=time.monotonic();full=0;checked=0;errors={}
    for job in jobs:
        f=folder(job);tr=b.read(f/'trace.json');z=np.load(f/'features.npz');g,target,_=original.inputs(job);edges,nb=original.r5.topology(g);names=sorted(edges);m=model(job)
        with torch.no_grad():
            xx=torch.from_numpy(np.stack([encode(raw,dd,g.num_nodes,cols) for raw,dd in zip(z['counts'],z['delta'])])).double();q=m(xx).numpy()
        err=float(abs(q-z['logits']).max());assert err<=1e-9;errors[str(f)]=err
        for i,v in enumerate(tr):
            assert time.monotonic()-tic<1800
            vals=v['candidates'];raw=z['counts'][i];dd=z['delta'][i]
            expected_seed=(202611183400000+1000000000 if ds=='pubmed' else 202611187400000)+job['seed']*10000+i
            assert v['proposal_seed']==expected_seed
            if i%32==0:
                graph=original.r5.graph_from(g,edges);_,fresh=original.p.cache.context(graph,cols);assert np.array_equal(fresh['raw'],raw)
                assert np.array_equal(Consequences(graph,cols,fresh['raw']).differences(original.t.proposals(graph,vals)),dd);full+=1
            chosen=v['chosen'];assert chosen==int(q[i].argmax());original.r5.edit(edges,nb,names,vals,chosen);reject=not original.r5.connected(nb);assert reject==v['connectivity_rejected']
            if reject:original.r5.edit(edges,nb,names,vals,chosen,True)
            effective=0 if reject else chosen;assert v['effective']==effective and original.r5.connected(nb)
            if i+1<len(tr):assert np.array_equal(raw+dd[effective],z['counts'][i+1])
            if i+1 in [512,1024,2048]:
                graph=original.r5.graph_from(g,edges);stored=b.restore_graph(b.read(f/f'graph{i+1}.json'))
                for fam in names:assert edges[fam]=={tuple(sorted(e)) for e in stored.family_edges[fam]} and np.array_equal(graph.family_degrees(fam),g.family_degrees(fam))
                report=b.read(f/f'report{i+1}.json')
                for key,value in original.prefix.stats(ds,graph,target).items():assert report[key]==value
            checked+=1
        print(ds,job['seed'],job['augmentation'],'accepted',flush=True)
    decisions=[]
    for sid in a['seeds']:
        reports={arm:b.read(a['folder'](sid,arm)/'report2048.json') for arm in ['F','Q','N','NS']};g,target,_=original.inputs(dict(dataset=ds,seed=sid,producer='C'))
        zero=original.prefix.stats(ds,g,target)['primary_error'];rr={kind:next(r for r in rows if r['seed']==sid and r['augmentation']==kind and r['steps']==2048)['primary_error'] for kind in ['targeted','ordinary']}
        base={arm:r['primary_error'] for arm,r in reports.items()};passed=rr['targeted']<min(base['F'],base['N'],base['NS'],zero)
        decisions.append(dict(seed=sid,targeted=rr['targeted'],ordinary=rr['ordinary'],base=base['F'],Q=base['Q'],N=base['N'],NS=base['NS'],zero=zero,passed=passed,better_than_Q=rr['targeted']<base['Q'],better_than_ordinary=rr['targeted']<rr['ordinary']))
    contract.verify(out);contract.put(out,'summary.json',dict(dataset=ds,passed=all(d['passed'] for d in decisions),decisions=decisions,rows=rows,development_only=True,normalization_changed=False,Q_changed=False))
    contract.put(out,'acceptance.json',dict(passed=True,replayed_opportunities=checked,full_state_recounts=full,logit_errors=errors,seconds=time.monotonic()-tic,outputs={str(p):b.sha(p) for p in out.glob('outputs/*/*/graph*.json')}));print(ds,'pilot complete',decisions,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('dataset',choices=['pubmed','acm']);p.add_argument('--repeat',type=int,required=True);p.add_argument('--bank',type=int,required=True);args=p.parse_args()
    try:main(args.dataset,args.repeat,args.bank)
    except BaseException as exc:
        out=BASE/f'deploy/{args.dataset}/bank{args.bank}'
        if out.exists():contract.put(out,'failure.json',dict(type=type(exc).__name__,message=str(exc)))
        raise

