"""Fixed-count extra-domain C/D pilot, gated on accepted primary confirmation."""
import argparse,os,subprocess
import acm_guidance4_adapter
import freebase_legal_adapter
from target_condition_audit import feasible
import prefix_replay
import freebase_legal_replay
import confirmation_common as contract
from additional_fit import source_check
from round01 import t,b,np,torch,time,Path,ROOT,SRC
BASE=ROOT/'results/typed_edit_autonomous_20261118';OUT=None

def main(ds):
    global OUT,adapter,replay
    adapter=acm_guidance4_adapter if ds=='acm' else freebase_legal_adapter
    replay=prefix_replay if ds=='acm' else freebase_legal_replay
    b.setup();confirmation=BASE/({'acm':'acm_guidance4','freebase':'freebase_legal_selection'}[ds])/'deploy'
    source_check(confirmation);assert b.read(confirmation/'summary.json')['passed']
    engineering=BASE/f'additional_f_engineering_finalized/{ds}';source_check(engineering)
    assert torch.cuda.is_available()
    assert os.environ.get('CUDA_VISIBLE_DEVICES') in [None,'','0,1,2,3,4,5,6,7']
    query=subprocess.run(['nvidia-smi','--query-gpu=index,memory.free,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True).stdout
    inventory=[tuple(int(x.strip()) for x in line.split(',')) for line in query.strip().splitlines()]
    available=[(free,i) for i,free,util in inventory if free>=2048 and util<=5];assert available
    _,card=max(available);device=f'cuda:{card}';torch.cuda.set_device(card);torch.cuda.reset_peak_memory_stats()
    OUT=BASE/f'additional_confirmation_v1/prefix/{ds}';OUT.mkdir(parents=True,exist_ok=False)
    cp=b.RUN/f'fits/{ds}/seed810/M/model.pt';parent=OUT;ref=b.graph(ds)
    role=b.ExchangeableRoleDegreePrior.from_training_graph(ref);prior=b.MarginalGraphPrior.from_training_graph(ref)
    condition=b.FamilyCountCondition(prior,role.family_counts(prior.family_endpoints));jobs=[]
    for sid in [15001,15002]:
        initial=condition.sample_graph(sid+71,mode='fixed');target=role.sample_assignment_for_graph(initial,sid+91919)
        assert np.array_equal(target,role.sample_assignment(initial.node_types,sid+91919)) and all(feasible(ref,target).values())
        folder=OUT/f'inputs/{sid}';contract.put(folder,'initial.json',b.graph_payload(initial));contract.put(folder,'target.json',target.tolist())
        jobs.append(dict(dataset=ds,seed=sid,input_folder=str(folder),base_seed=202611187200000+['acm','freebase'].index(ds)*1000000000+sid*10000))
    files=[Path(__file__),SRC/'ADDITIONAL_CONFIRMATION_V1.md',cp,b.RUN/f'data/{ds}/graph.json',confirmation/'manifest.json',confirmation/'acceptance.json',confirmation/'summary.json',engineering/'manifest.json',engineering/'acceptance.json']
    files+=list(parent.glob('inputs/*/*.json'))
    contract.bind(OUT,files,dataset=ds,jobs=jobs,arms=['C','D'],count_mode='fixed',cap=8192,device=device,gpu_inventory=inventory,new_model_fits=0)
    model=t.a.load_mixed_edit_checkpoint(cp,device)[0].eval();cpu_model=t.a.load_mixed_edit_checkpoint(cp,'cpu')[0].eval()
    cost=30.;eng=[];fe=0.;le=0.
    for job in jobs:
        for arm in ['C','D']:
            _,tr,x,q,rp=adapter.chain(job,arm,16,model,device,True,False)
            initial,target=adapter.inputs(job);edges,nb=adapter.r5.topology(initial);names=sorted(edges)
            for i,v in enumerate(tr):
                g=adapter.r5.graph_from(initial,edges)
                fresh=adapter.core.mixed_edit_candidate_matrix(g,t.proposals(g,v['candidates']),v['time_fraction'],'typed_closure_compact_v2')
                fe=max(fe,float(abs(fresh-x[i]).max()));assert fe<=1e-5
                if arm=='C':
                    with torch.no_grad():qq=cpu_model(torch.from_numpy(fresh)[None])[0].numpy()
                    le=max(le,float(abs(qq-q[i]).max()));assert le<=2e-5
                adapter.edit(edges,nb,names,v['candidates'],v['effective'])
            _,_,_,_,rp=adapter.chain(job,arm,32,model,device,False,False)
            cost+=2*(rp['seconds']+(8192/32-1)*rp['loop_seconds']);eng.append(rp)
    contract.put(OUT,'engineering.json',dict(passed=cost<=3600,cost_bound_seconds=cost,opportunities=192,feature_error=fe,logit_error=le,rows=eng))
    assert cost<=3600;contract.verify(OUT);tic=time.monotonic();rows=[]
    for job in jobs:
        for arm in ['C','D']:
            graph,tr,x,q,rp=adapter.chain(job,arm,8192,model,device,deadline=tic+3600)
            folder=OUT/f'outputs/{job["seed"]}_{arm}'
            contract.put(folder,'graph.json',b.graph_payload(graph));contract.put(folder,'report.json',rp);contract.put(folder,'trace.json',tr)
            np.savez_compressed(folder/'features.npz',x=x,logits=q);rows.append(rp)
            print('Additional prefix generated',ds,job['seed'],arm,rp['steps'],rp['arrived'],rp['components'],flush=True)
    peak=torch.cuda.max_memory_allocated();assert peak<=2*1024**3
    contract.put(OUT,'completion.json',dict(completed=True,first_stage_outputs=4,scientific_opportunities=sum(r['steps'] for r in rows),peak_allocated_bytes=peak,seconds=time.monotonic()-tic,new_model_fits=0))
    del model;torch.cuda.empty_cache();tic=time.monotonic();checks=[]
    for job in jobs:
        for arm in ['C','D']:
            rp=replay.check(OUT/f'outputs/{job["seed"]}_{arm}',job,adapter.stats,min(tic+2400,time.monotonic()+600))
            checks.append(dict(seed=job['seed'],arm=arm,**rp))
            contract.put(OUT,f'checks/{job["seed"]}_{arm}.json',checks[-1])
    contract.verify(OUT)
    contract.put(OUT,'summary.json',dict(dataset=ds,rows=rows,all_C_arrived=all(r['arrived'] for r in rows if r['arm']=='C'),all_D_arrived=all(r['arrived'] for r in rows if r['arm']=='D'),count_mode='fixed',complete_two_stage=False))
    outputs={str(p):b.sha(p) for p in OUT.glob('outputs/*/*') if p.name in ['graph.json','report.json']}
    outputs.update({str(p):b.sha(p) for p in parent.glob('inputs/*/*.json')})
    contract.put(OUT,'acceptance.json',dict(passed=True,checks=checks,replayed_opportunities=sum(r['replayed_opportunities'] for r in checks),outputs=outputs,seconds=time.monotonic()-tic))
    print('Additional prefixes accepted',ds,flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('dataset',choices=['acm','freebase']);args=parser.parse_args()
    try:main(args.dataset)
    except BaseException as exc:
        if OUT is not None:contract.put(OUT,'failure.json',dict(type=type(exc).__name__,message=str(exc)))
        raise
