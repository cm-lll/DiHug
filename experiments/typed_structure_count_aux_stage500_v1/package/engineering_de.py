"""Four combination cells, at most TWO disposable normal2 updates each.

No scientific training entry, no reverse chain, no sampled state/graph.
Run one dataset per selected GPU; inherited OABC launchers are not DE launchers.
"""
import argparse, pathlib, json, time, signal, resource, gc, traceback, os
import numpy as np
import torch
from prepare import save, sha
from experiment_contract import initialization, read_dictionary, reset_training_streams, source_binding
from normal2 import normal2, pair_inputs, sync
from probe import rng_get, rng_set, equal, input_hash
from hsgd.bw_preflight.common import cpu_tree, environment, schedule, check_prob_device
from hsgd.computational import FrozenTypedContext
from hsgd.kernel import MarginalKernel

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',choices=['IMDB','PubMed'],required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--dictionaries',required=True)
    args=parser.parse_args();out=pathlib.Path(args.output);out.mkdir(parents=True,exist_ok=False)
    started=time.time();result=dict(status='running',cells={},formal_updates=0,complete_chains=0,new_graphs=0,
        start_epoch=started,limit_seconds=900,CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES'),
        scope='engineering weights ONLY; exactly <=2 steps per dataset-arm; no scientific initialization')
    def check():
        if time.time()>=started+900:raise TimeoutError('bounded engineering only')
        if (out/'STOP').exists():raise InterruptedError('STOP')
    def interrupted(sig,frame):raise InterruptedError('signal '+str(sig))
    for s in [signal.SIGINT,signal.SIGTERM,signal.SIGHUP,signal.SIGALRM]:signal.signal(s,interrupted)
    signal.alarm(900)
    try:
        torch.set_num_threads(1);torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True,warn_only=False)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        if not torch.cuda.is_available() or torch.cuda.device_count()!=1 or '2080 Ti' not in torch.cuda.get_device_name(0):
            raise RuntimeError('exactly one visible RTX2080Ti required')
        save(out/'environment.json',environment('cuda'))
        save(out/'source.json',dict(zip(['files','sha256'],source_binding())))
        basis,receipt=read_dictionary(args.dictionaries,args.dataset)
        seed={'IMDB':2026102811,'PubMed':2026102812}[args.dataset]
        for arm in 'DE':
            check();folder=out/arm;folder.mkdir();row=dict(passed=False,reserved_steps=0,completed_steps=0)
            result['cells'][args.dataset+'_'+arm]=row
            m,case,cfg,proof=initialization(args.dataset,arm,seed,basis if arm=='E' else None)
            cfg=dict(cfg,seed=seed);m=m.cuda();rng,rngproof=reset_training_streams(seed,args.dataset,'cuda')
            initial=cpu_tree(m.state_dict());initial_rng=rng_get(rng,'cuda')
            binding=dict(source_sha256=source_binding()[1],dictionary_sha256=receipt['basis_sha256'],dataset=args.dataset,arm=arm,config=cfg)
            save(folder/'initialization.json',proof);save(folder/'rng.json',rngproof)
            torch.save(dict(model=initial,rng=initial_rng,updates=0,binding=binding,purpose='DISPOSABLE ENGINEERING ONLY'),folder/'initial.pt')
            row['initial_sha256']=sha(folder/'initial.pt');row['parameters']=proof['parameters']
            opt=torch.optim.AdamW(m.parameters(),**{k:v for k,v in cfg['optimizer'].items() if k!='type'})
            assert {id(p) for g in opt.param_groups for p in g['params']}=={id(p) for p in m.parameters()}
            paired=pair_inputs(case,cfg,50,np.random.default_rng(2026102801),500)
            row['input_sha256']=input_hash(paired)
            for j,inp in enumerate(paired):np.savez_compressed(folder/('input_'+str(j)+'.npz'),**inp)
            torch.cuda.reset_peak_memory_stats()
            row['warmup_no_step']=normal2(m,opt,case,paired,1000,'cuda',False)
            grad_names=['refex_projection.weight','common_neighbor.phi.0.weight']+(['role_projection.weight'] if arm=='E' else [])
            row['gradients_at_initialization']={}
            for name in grad_names:
                g=dict(m.named_parameters())[name].grad
                assert g is not None and torch.isfinite(g).all() and g.abs().max()>0,name
                row['gradients_at_initialization'][name]=dict(max_abs=float(g.abs().max()),norm=float(g.norm()))
            row['optimizer_membership_exact']=True;row['updates']=[]
            for i in range(2):
                check();row['reserved_steps']+=1;save(out/'summary.json',result)
                rec=normal2(m,opt,case,paired,1000,'cuda',True);row['completed_steps']+=1;row['updates'].append(rec)
                if i==0:
                    payload=dict(model=cpu_tree(m.state_dict()),optimizer=cpu_tree(opt.state_dict()),rng=rng_get(rng,'cuda'),updates=1,binding=binding)
                    begin=time.perf_counter();torch.save(payload,folder/'after1.pt');row['save_seconds']=time.perf_counter()-begin
                    row['checkpoint_sha256']=sha(folder/'after1.pt')
                    del m,opt;gc.collect()
                    m,_,_,_=initialization(args.dataset,arm,seed,basis if arm=='E' else None);m=m.cuda()
                    ck=torch.load(folder/'after1.pt',map_location='cpu',weights_only=False)
                    assert sha(folder/'after1.pt')==row['checkpoint_sha256'] and ck['updates']==1 and ck['binding']==binding
                    m.load_state_dict(ck['model'],strict=True)
                    opt=torch.optim.AdamW(m.parameters(),**{k:v for k,v in cfg['optimizer'].items() if k!='type'})
                    opt.load_state_dict(ck['optimizer']);rng_set(ck['rng'],rng,'cuda')
                    row['model_load_exact']=equal(cpu_tree(m.state_dict()),ck['model'])
                    row['optimizer_load_exact']=equal(cpu_tree(opt.state_dict()),ck['optimizer'])
                    row['rng_load_exact']=equal(rng_get(rng,'cuda'),ck['rng'])
                    assert row['model_load_exact'] and row['optimizer_load_exact'] and row['rng_load_exact']
                    del payload,ck
            # Post-update inputs expose nonzero learned projections. Not quality evaluation.
            m.eval();row['full_query_components']=[]
            for t in [1,500,1000]:
                check();inp=pair_inputs(case,cfg,100+t,np.random.default_rng(2026102801+t),t)[0]
                expected=np.flatnonzero(np.repeat(case['active'],case['space'].counts))
                assert np.array_equal(np.sort(inp['ids']),expected)
                counters=(m.feature_seconds,m.encoding_seconds,m.transfer_seconds)
                sync('cuda');begin=time.perf_counter()
                ctx=FrozenTypedContext(m,case['space'].types,inp['edges'],inp['edge_families'],('probe',t))
                try:
                    with torch.no_grad():
                        p=ctx.predict(inp['query'],inp['f'],t/1000,('probe',t));check_prob_device(p,inp['f'],case,m.prior)
                        prob=p.cpu().numpy().astype(np.float64)
                    assert ctx.cn_wedges is not None and m.common_neighbor is not None
                    assert m.feature_calls>0 and hasattr(ctx,'structural_features')
                    np.testing.assert_array_equal(ctx.old,inp['edges'])
                    if arm=='E':
                        assert torch.isfinite(ctx.structural_roles).all() and (ctx.structural_roles>=0).all()
                        assert np.array_equal(m._basis_cpu,basis['basis']) and m.role_basis.grad is None
                    encoded=getattr(ctx,'encoding_receipt',None);wedges=ctx.cn_wedges.raw_wedges
                finally:ctx.close()
                sync('cuda');forward=time.perf_counter()-begin
                state=np.zeros(case['space'].total,np.uint8);f,l=case['space'].encode(inp['edges']);state[case['space'].offsets[f]+l]=1
                begin=time.perf_counter();q=MarginalKernel(case['prior'],schedule(1000)).reverse(prob,state[inp['ids']],inp['f'],t-1,t)
                posterior=time.perf_counter()-begin;assert np.isfinite(q).all() and ((q>=0)&(q<=1)).all()
                if t==1:np.testing.assert_allclose(q,prob,atol=2e-12,rtol=2e-12)
                row['full_query_components'].append(dict(t=t,slots=len(prob),forwards=1,input_sha256=input_hash([inp]),
                    network_context_and_features_seconds=forward,posterior_seconds=posterior,total_seconds=forward+posterior,
                    refex_seconds=m.feature_seconds-counters[0],NNLS_seconds=m.encoding_seconds-counters[1],
                    structural_transfer_seconds=m.transfer_seconds-counters[2],CN_wedges=wedges,encoding=encoded,
                    sampling_and_commit_executed=False,quality_evidence=False))
            row.update(passed=True,normal2_seconds=max(x['normal2_seconds'] for x in row['updates']),
                sampling_component_seconds=max(x['total_seconds'] for x in row['full_query_components']),
                peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,
                peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
            save(out/'summary.json',result);print(args.dataset,arm,'PASS',row['completed_steps'],flush=True)
            del m,opt,initial,ctx;gc.collect();torch.cuda.empty_cache()
        result['status']='passed_preparation_only'
    except BaseException as exc:result.update(status='failed_or_incomplete',error=repr(exc),traceback=traceback.format_exc())
    finally:
        signal.alarm(0);result.update(ended_epoch=time.time(),seconds=time.time()-started)
        save(out/'summary.json',result)
    return 0 if result['status']=='passed_preparation_only' else 2

if __name__=='__main__':raise SystemExit(main())
