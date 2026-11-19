"""Fresh IMDB C/D starts on CPU, with an unchanged typed-edit first-stage policy."""
import confirmation_prefix_v2 as core
import confirmation_fit as frozen
import confirmation_common as contract
import prefix_replay
from round01 import t,b,np,torch,time,Path,ROOT,SRC
BASE=ROOT/'results/typed_edit_autonomous_20261118';OUT=BASE/'cycle_repair_confirmation/prefix'

def main():
    b.setup();frozen.source_check(BASE/'cycle_repair_pilot/deploy')
    assert b.read(BASE/'cycle_repair_pilot/deploy/summary.json')['passed']
    OUT.mkdir(parents=True,exist_ok=False);ref=b.graph('imdb')
    cp=b.RUN/'fits/imdb/seed810/M/model.pt';model=t.a.load_mixed_edit_checkpoint(cp,'cpu')[0].eval()
    jobs=[]
    for sid in [14001,14002]:
        prior=b.MarginalGraphPrior.from_training_graph(ref);role=b.ExchangeableRoleDegreePrior.from_training_graph(ref)
        graph=b.FamilyCountCondition(prior,role.family_counts(prior.family_endpoints)).sample_graph(sid+71,mode='binomial')
        target=role.sample_assignment_for_graph(graph,sid+91919);folder=OUT/f'inputs/{sid}'
        assert np.array_equal(graph.node_types,ref.node_types)
        contract.put(folder,'initial.json',b.graph_payload(graph));contract.put(folder,'target.json',target.tolist())
        jobs.append(dict(dataset='imdb',seed=sid,input_folder=str(folder),base_seed=202611186200000+sid*10000))
    files=[Path(__file__),SRC/'CYCLE_REPAIR_CONFIRMATION.md',cp,b.RUN/'data/imdb/graph.json',BASE/'cycle_repair_pilot/deploy/acceptance.json',BASE/'cycle_repair_pilot/deploy/summary.json']
    files += list(OUT.glob('inputs/*/*.json'))
    contract.bind(OUT,files,jobs=jobs,arms=['C','D'],cap=2048,device='cpu',new_model_fits=0,
                  count_mode='binomial',maximum_scientific_opportunities=8192)
    cost=20.;engineering=[];fe=0.;le=0.
    for job in jobs:
        for arm in ['C','D']:
            _,tr,x,q,_=core.chain(job,arm,16,model,'cpu',True,False)
            initial,target=core.inputs(job);edges,nb=core.r5.topology(initial);names=sorted(edges)
            for i,v in enumerate(tr):
                g=core.r5.graph_from(initial,edges)
                xx=core.mixed_edit_candidate_matrix(g,t.proposals(g,v['candidates']),v['time_fraction'],'typed_closure_compact_v2')
                fe=max(fe,float(abs(xx-x[i]).max()));assert fe<=1e-5
                if arm=='C':
                    with torch.no_grad():qq=model(torch.from_numpy(xx)[None])[0].numpy()
                    le=max(le,float(abs(qq-q[i]).max()));assert le<=2e-5
                core.edit(edges,nb,names,v['candidates'],v['effective'])
            _,_,_,_,rp=core.chain(job,arm,32,model,'cpu',False,False)
            cost+=2*(rp['seconds']+(2048/32-1)*rp['loop_seconds']);engineering.append(rp)
    contract.put(OUT,'engineering.json',dict(passed=cost<=1200,cost_bound_seconds=cost,
                                            opportunities=192,feature_error=fe,logit_error=le,rows=engineering))
    assert cost<=1200;contract.verify(OUT);tic=time.monotonic();rows=[]
    for job in jobs:
        for arm in ['C','D']:
            g,tr,x,q,rp=core.chain(job,arm,2048,model,'cpu',deadline=tic+1200)
            folder=OUT/f'outputs/{job["seed"]}_{arm}'
            contract.put(folder,'graph.json',b.graph_payload(g));contract.put(folder,'report.json',rp);contract.put(folder,'trace.json',tr)
            np.savez_compressed(folder/'features.npz',x=x,logits=q);rows.append(rp)
            print('Coverage confirmation prefix',job['seed'],arm,rp['steps'],rp['arrived'],flush=True)
    contract.put(OUT,'completion.json',dict(completed=True,first_stage_outputs=4,
        scientific_opportunities=sum(r['steps'] for r in rows),seconds=time.monotonic()-tic,new_model_fits=0))
    tic=time.monotonic();checks=[]
    for job in jobs:
        for arm in ['C','D']:
            check=prefix_replay.check(OUT/f'outputs/{job["seed"]}_{arm}',job,core.stats,tic+1200)
            checks.append(dict(seed=job['seed'],arm=arm,**check));contract.put(OUT,f'checks/{job["seed"]}_{arm}.json',checks[-1])
    contract.verify(OUT)
    contract.put(OUT,'summary.json',dict(rows=rows,all_C_arrived=all(r['arrived'] for r in rows if r['arm']=='C'),
        all_D_arrived=all(r['arrived'] for r in rows if r['arm']=='D'),count_mode='binomial',execution_backend='cpu'))
    files=list(OUT.glob('inputs/*/*.json'))+[p for p in OUT.glob('outputs/*/*') if p.name in ['graph.json','report.json']]
    contract.put(OUT,'acceptance.json',dict(passed=True,checks=checks,
        replayed_opportunities=sum(r['replayed_opportunities'] for r in checks),outputs={str(p):b.sha(p) for p in files},seconds=time.monotonic()-tic))

if __name__=='__main__':
    try:main()
    except BaseException as exc:
        if OUT.exists():contract.put(OUT,'failure.json',dict(type=type(exc).__name__,message=str(exc)))
        raise
