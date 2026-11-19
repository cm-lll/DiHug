"""Fresh C/D starts for an accepted cross-reference augmentation pilot."""
import argparse,subprocess
import confirmation_prefix_v2 as pubmed
import acm_guidance4_adapter as acm
import prefix_replay
import confirmation_common as contract
import crossdomain_augmentation_diagnosis as diagnosis
from round01 import t,b,np,torch,time,Path,ROOT,SRC

BASE=ROOT/'results/typed_edit_crossdomain_augmentation_20261118'

def main(ds,card):
    b.setup();pilot=BASE/f'deploy/{ds}/bank1';diagnosis.fit.source_check(pilot);assert b.read(pilot/'summary.json')['passed']
    assert torch.cuda.is_available();inventory=subprocess.run(['nvidia-smi','--query-gpu=index,memory.free,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True).stdout
    available=[tuple(int(x.strip()) for x in row.split(',')) for row in inventory.strip().splitlines()]
    assert any(i==card and free>=2048 and util<=10 for i,free,util in available),('selected_card_not_idle',card)
    device=f'cuda:{card}';torch.cuda.set_device(card);torch.cuda.reset_peak_memory_stats();adapter=pubmed if ds=='pubmed' else acm
    out=BASE/f'confirmation/prefix/{ds}';out.mkdir(parents=True,exist_ok=False);ref=b.graph(ds);role=b.ExchangeableRoleDegreePrior.from_training_graph(ref);prior=b.MarginalGraphPrior.from_training_graph(ref)
    condition=b.FamilyCountCondition(prior,role.family_counts(prior.family_endpoints));jobs=[]
    for sid in [17001,17002]:
        g=condition.sample_graph(sid+71,mode='binomial' if ds=='pubmed' else 'fixed');target=role.sample_assignment_for_graph(g,sid+91919);folder=out/f'inputs/{sid}'
        contract.put(folder,'initial.json',b.graph_payload(g));contract.put(folder,'target.json',target.tolist())
        jobs.append(dict(dataset=ds,seed=sid,input_folder=str(folder),base_seed=202611189300000+['pubmed','acm'].index(ds)*100000000000+sid*10000))
    cp=b.RUN/f'fits/{ds}/seed810/M/model.pt';files=[Path(__file__),SRC/'CROSSDOMAIN_AUGMENTATION_CONFIRMATION.md',pilot/'manifest.json',pilot/'acceptance.json',pilot/'summary.json',cp,b.RUN/f'data/{ds}/graph.json']+list(out.glob('inputs/*/*.json'))
    contract.bind(out,files,dataset=ds,jobs=jobs,device=device,gpu_inventory=available,cap=8192,new_model_fits=0,count_mode='binomial' if ds=='pubmed' else 'fixed')
    model=t.a.load_mixed_edit_checkpoint(cp,device)[0].eval();cpu=t.a.load_mixed_edit_checkpoint(cp,'cpu')[0].eval();cost=30.;eng=[];fe=0.;le=0.
    for job in jobs:
        for arm in ['C','D']:
            _,tr,x,q,_=adapter.chain(job,arm,16,model,device,True,False);initial,target=adapter.inputs(job);edges,nb=adapter.r5.topology(initial);names=sorted(edges)
            for i,v in enumerate(tr):
                graph=adapter.r5.graph_from(initial,edges);fresh=pubmed.mixed_edit_candidate_matrix(graph,t.proposals(graph,v['candidates']),v['time_fraction'],'typed_closure_compact_v2')
                fe=max(fe,float(abs(fresh-x[i]).max()));assert fe<=1e-5
                if arm=='C':
                    with torch.no_grad():qq=cpu(torch.from_numpy(fresh)[None])[0].numpy()
                    le=max(le,float(abs(qq-q[i]).max()));assert le<=2e-5
                pubmed.edit(edges,nb,names,v['candidates'],v['effective'])
            _,_,_,_,rp=adapter.chain(job,arm,32,model,device,False,False);cost+=2*(rp['seconds']+(8192/32-1)*rp['loop_seconds']);eng.append(rp)
    contract.put(out,'engineering.json',dict(passed=cost<=3600,cost_bound_seconds=cost,feature_error=fe,logit_error=le,rows=eng));assert cost<=3600;contract.verify(out);tic=time.monotonic();rows=[]
    for job in jobs:
        for arm in ['C','D']:
            g,tr,x,q,rp=adapter.chain(job,arm,8192,model,device,deadline=tic+3600);f=out/f"outputs/{job['seed']}_{arm}"
            contract.put(f,'graph.json',b.graph_payload(g));contract.put(f,'report.json',rp);contract.put(f,'trace.json',tr);np.savez_compressed(f/'features.npz',x=x,logits=q);rows.append(rp)
            print(ds,job['seed'],arm,'new prefix',rp['steps'],rp['arrived'],flush=True)
    peak=torch.cuda.max_memory_allocated();assert peak<=2*1024**3;contract.put(out,'completion.json',dict(completed=True,new_first_stage_outputs=4,scientific_opportunities=sum(r['steps'] for r in rows),seconds=time.monotonic()-tic,peak_allocated_bytes=peak,new_fits=0))
    del model,cpu;torch.cuda.empty_cache();tic=time.monotonic();checks=[]
    for job in jobs:
        for arm in ['C','D']:
            check=prefix_replay.check(out/f"outputs/{job['seed']}_{arm}",job,adapter.stats,min(tic+2400,time.monotonic()+600));checks.append(dict(seed=job['seed'],arm=arm,**check));contract.put(out,f"checks/{job['seed']}_{arm}.json",checks[-1])
    contract.verify(out);contract.put(out,'summary.json',dict(dataset=ds,rows=rows,all_C_arrived=all(r['arrived'] for r in rows if r['arm']=='C'),all_D_arrived=all(r['arrived'] for r in rows if r['arm']=='D')))
    outputs={str(p):b.sha(p) for p in out.glob('outputs/*/*') if p.name in ['graph.json','report.json']};outputs.update({str(p):b.sha(p) for p in out.glob('inputs/*/*.json')})
    contract.put(out,'acceptance.json',dict(passed=True,checks=checks,replayed_opportunities=sum(r['replayed_opportunities'] for r in checks),seconds=time.monotonic()-tic,outputs=outputs));print(ds,'new prefixes accepted',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('dataset',choices=['pubmed','acm']);p.add_argument('--card',type=int,required=True);a=p.parse_args()
    try:main(a.dataset,a.card)
    except BaseException as exc:
        out=BASE/f'confirmation/prefix/{a.dataset}'
        if out.exists():contract.put(out,'failure.json',dict(type=type(exc).__name__,message=str(exc)))
        raise
