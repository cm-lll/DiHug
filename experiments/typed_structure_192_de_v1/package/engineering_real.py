"""Eight-cell real-dictionary engineering, maximum THREE steps per cell.
No full chain, no scientific initialization or automatic formal continuation.
"""
import argparse,pathlib,json,time,signal,copy,resource,gc,os,traceback
import numpy as np
import torch
from prepare import save,sha
from experiment_contract import initialization,read_dictionary,reset_training_streams,source_binding
from normal2 import normal2,pair_inputs,sync
from probe import rng_get,rng_set,equal,input_hash,compare_diagnostics,metric,flatten
from hsgd.bw_preflight.common import cpu_tree,environment,schedule
from hsgd.computational import FrozenTypedContext
from hsgd.kernel import MarginalKernel
from restore_checks import restoration_passes

def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',choices=['IMDB','PubMed'],required=True);p.add_argument('--dictionaries',required=True);p.add_argument('--output',required=True);p.add_argument('--ledger',required=True);a=p.parse_args()
    out=pathlib.Path(a.output);out.mkdir(exist_ok=False);ledger=json.loads(pathlib.Path(a.ledger).read_text());deadline=min(ledger['preparation_deadline_epoch']-180,1793126889);result=dict(status='running',cells={},formal_updates=0,complete_chains=0);start=time.time()
    def check():
        if time.time()>=deadline:raise TimeoutError('delivery reserve')
        if (out/'STOP').exists():raise InterruptedError('STOP')
    def stop(s,f):raise InterruptedError('signal '+str(s))
    for s in [signal.SIGINT,signal.SIGTERM,signal.SIGHUP,signal.SIGALRM]:signal.signal(s,stop)
    signal.alarm(max(1,int(deadline-time.time())))
    try:
        check();torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True,warn_only=False);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        if not torch.cuda.is_available() or torch.cuda.device_count()!=1 or '2080 Ti' not in torch.cuda.get_device_name(0):raise RuntimeError('one RTX2080Ti required')
        save(out/'environment.json',environment('cuda'));save(out/'source.json',dict(zip(['files','sha256'],source_binding())));basis,receipt=read_dictionary(a.dictionaries,a.dataset)
        seed={'IMDB':2026102811,'PubMed':2026102812}[a.dataset]
        for arm in 'OABC':
            check();folder=out/arm;folder.mkdir();label=a.dataset+'_'+arm;row=dict(passed=False,actual_steps=0);result['cells'][label]=row
            m,case,cfg,proof=initialization(a.dataset,arm,seed,basis if arm=='C' else None);cfg=dict(cfg,seed=seed);m=m.cuda();rng,rngproof=reset_training_streams(seed,a.dataset,'cuda');initial=cpu_tree(m.state_dict());initial_rng=rng_get(rng,'cuda')
            binding=dict(source=source_binding()[1],config=cfg,dataset=a.dataset,arm=arm,basis_sha256=receipt['basis_sha256'])
            save(folder/'initialization.json',proof);save(folder/'rng.json',rngproof);torch.save(dict(model=initial,rng=initial_rng,updates=0,binding=binding,purpose='ENGINEERING ONLY NEVER SCIENTIFIC INITIAL'),folder/'initial.pt');row['initial_sha256']=sha(folder/'initial.pt')
            opt=torch.optim.AdamW(m.parameters(),**{k:v for k,v in cfg['optimizer'].items() if k!='type'});assert {id(p) for g in opt.param_groups for p in g['params']}=={id(p) for p in m.parameters()}
            paired=pair_inputs(case,cfg,50,np.random.default_rng(2026102801),500);row['input_sha256']=input_hash(paired);torch.cuda.reset_peak_memory_stats();d1={};d2={}
            normal2(m,opt,case,paired,1000,'cuda',False,d1);rng_set(initial_rng,rng,'cuda');normal2(m,opt,case,paired,1000,'cuda',False,d2);cmp=compare_diagnostics(d1,d2);save(folder/'same_gpu.json',cmp);row['same_gpu_passed']=cmp['passes'];del d1,d2
            if not cmp['passes']:raise RuntimeError('same GPU full normal2 gate failed; no updates')
            # Low/mid/high fresh Xt contexts, frozen dictionary identity and complete query.
            m.eval();timings=[]
            for t in [1,500,1000]:
                check();inp=pair_inputs(case,cfg,100+t,np.random.default_rng(2026102801+t),t)[0];sync('cuda');begin=time.perf_counter();ctx=FrozenTypedContext(m,case['space'].types,inp['edges'],inp['edge_families'],('engineering',t))
                try:
                    with torch.no_grad():prob=ctx.predict(inp['query'],inp['f'],t/1000,('engineering',t)).cpu().numpy().astype(np.float64)
                    enc=getattr(ctx,'encoding_receipt',None);assert len(prob)==int(np.sum(case['space'].counts[case['active']]))
                    if arm!='A':assert ctx.cn_wedges is None and m.common_neighbor is None
                    if arm=='C':assert np.isfinite(ctx.structural_roles.cpu().numpy()).all() and (ctx.structural_roles>=0).all()
                finally:ctx.close()
                sync('cuda');forward=time.perf_counter()-begin;state=np.zeros(case['space'].total,np.uint8);ff,ll=case['space'].encode(inp['edges']);state[case['space'].offsets[ff]+ll]=1
                begin=time.perf_counter();q=MarginalKernel(case['prior'],schedule(1000)).reverse(prob,state[inp['ids']],inp['f'],t-1,t);candidate=state.copy();candidate[inp['ids']]=(np.random.default_rng(123).random(len(q))<q);assert np.isin(candidate,[0,1]).all();posterior_commit=time.perf_counter()-begin
                timings.append(dict(t=t,slots=len(prob),forwards=1,network_seconds=forward,posterior_and_disposable_commit_seconds=posterior_commit,total_seconds=forward+posterior_commit,encoding=enc,engineering_state_not_scientific_graph=True))
            row['full_step']=timings
            m.load_state_dict(initial);opt=torch.optim.AdamW(m.parameters(),**{k:v for k,v in cfg['optimizer'].items() if k!='type'});rng_set(initial_rng,rng,'cuda');hist=[];continuous=None;lastdiag=None
            for i,pos in enumerate([1,2,2]):
                check()
                if i==2:
                    assert sha(folder/'after1.pt')==row['checkpoint_sha256'];ck=torch.load(folder/'after1.pt',map_location='cpu');assert ck['updates']==1 and ck['binding']==binding;del m,opt;gc.collect();m,_,_,_=initialization(a.dataset,arm,seed,basis if arm=='C' else None);m=m.cuda();m.load_state_dict(ck['model']);opt=torch.optim.AdamW(m.parameters(),**{k:v for k,v in cfg['optimizer'].items() if k!='type'});opt.load_state_dict(ck['optimizer']);rng_set(ck['rng'],rng,'cuda')
                    row['load_exact']=equal(cpu_tree(m.state_dict()),ck['model']) and equal(cpu_tree(opt.state_dict()),ck['optimizer']) and equal(rng_get(rng,'cuda'),ck['rng']);assert row['load_exact']
                inp=pair_inputs(case,cfg,pos,rng,500);ih=input_hash(inp)
                if i==2:assert ih==hist[1]['input_sha256']
                row['actual_steps']+=1;save(out/'summary.json',result) # reserved before step; never retry/replenish
                dg={};rec=normal2(m,opt,case,inp,1000,'cuda',True,dg);rec.update(input_sha256=ih,restored=i==2);hist.append(rec)
                if i==0:
                    begin=time.perf_counter();torch.save(dict(model=cpu_tree(m.state_dict()),optimizer=cpu_tree(opt.state_dict()),rng=rng_get(rng,'cuda'),updates=1,binding=binding),folder/'after1.pt');row['save_seconds']=time.perf_counter()-begin;row['checkpoint_sha256']=sha(folder/'after1.pt')
                if i==1:continuous=dict(model=cpu_tree(m.state_dict()),optimizer=cpu_tree(opt.state_dict()),rng=rng_get(rng,'cuda'));lastdiag=dg
                if i==2:
                    comp=compare_diagnostics(lastdiag,dg);comp['parameters']={k:metric(v,m.state_dict()[k]) for k,v in continuous['model'].items()};current=cpu_tree(opt.state_dict());comp['optimizer_exact']=equal(continuous['optimizer'],current);comp['optimizer']={k:metric(v,flatten(current)[k]) for k,v in flatten(continuous['optimizer']).items()};comp['rng_exact']=equal(continuous['rng'],rng_get(rng,'cuda'));comp['passes']=comp['passes'] and comp['optimizer_exact'] and restoration_passes(comp);save(folder/'restore.json',comp);assert comp['passes'];row['restore_passed']=True
            row.update(passed=True,updates=hist,normal2_seconds=max(x['normal2_seconds'] for x in hist),sampling_step_seconds=max(x['total_seconds'] for x in timings),validation_save_seconds=None,peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,parameters=proof['parameters'])
            save(out/'summary.json',result);del m,opt,initial,continuous,lastdiag,dg;gc.collect();torch.cuda.empty_cache();print(label,'PASS',row['actual_steps'],flush=True)
        result['status']='engineering_completed_no_formal'
    except BaseException as exc:result.update(status='failed_or_incomplete',error=repr(exc),traceback=traceback.format_exc())
    finally:
        result.update(seconds=time.time()-start,ended_epoch=time.time());save(out/'summary.json',result)
    return 0 if result['status']=='engineering_completed_no_formal' else 2
if __name__=='__main__':raise SystemExit(main())
