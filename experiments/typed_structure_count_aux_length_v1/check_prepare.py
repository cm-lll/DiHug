"""At most four temporary optimizer calls per dataset. No sampling chain."""
import os,time,pathlib,argparse,json,traceback,resource,importlib.util
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from model_io import configure,fresh,ROOT
import torch,numpy as np
from runtime_contract import save,sha,source_binding
from probe import rng_get,rng_set,equal,flatten,metric,input_hash
from normal2 import normal2,pair_inputs,sync
from ema import ParameterEMA,cpu_copy
from integration import complete_normal2,checkpoint_payload,restore_payload,ema_validation
from stage import isolated_readout
from worker import validation
from count_aux import count_loss,case_contract,LAMBDA
from hsgd.computational import FrozenTypedContext

def compare(a,b):
    x,y=flatten(a),flatten(b)
    rows={k:metric(v,y[k]) for k,v in x.items()}
    return dict(exact=equal(a,b),passes=x.keys()==y.keys() and all(v['passes'] for v in rows.values()),tensors=rows)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--dataset',required=True);ap.add_argument('--output',required=True);ap.add_argument('--deadline',type=float,required=True);args=ap.parse_args()
    out=pathlib.Path(args.output);out.mkdir(parents=True,exist_ok=False);start=time.time()
    rec=dict(status='running',optimizer_step_calls=0,formal_updates=0,generated_graphs=0,dataset=args.dataset)
    def check():
        if time.time()>args.deadline:raise TimeoutError('engineering deadline')
    try:
        check();rec['environment']=configure();seed={'IMDB':2026102911,'PubMed':2026102912}[args.dataset]
        m,opt,case,cfg,rng,proof=fresh(args.dataset,'CE_COUNT',seed);ema=ParameterEMA(m)
        binding=dict(engineering_only=True,dataset=args.dataset,source_sha256=source_binding()[1])
        initial=checkpoint_payload(m,opt,ema,0,rng_get(rng,'cuda'),binding)
        torch.save(initial,out/'initial.pt');save(out/'initial_binding.json',dict(proof=proof,sha256=sha(out/'initial.pt')))
        rec['parameters']=proof['parameters'];rec['paired_initial_sha256']=proof['tensor_sha256']
        fixed=pair_inputs(case,cfg,1,np.random.default_rng(2026103001),500)
        rec['fixed_input_sha256']=input_hash(fixed)
        # lambda=0 uses the exact old CE function on the same real inputs; no steps.
        oldpath=ROOT/'reference_source/normal2.py'
        spec=importlib.util.spec_from_file_location('old_normal2',oldpath);old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
        d0={};d1={};random=rng_get(rng,'cuda')
        old.normal2(m,opt,case,fixed,1000,'cuda',False,d0);rng_set(random,rng,'cuda')
        normal2(m,opt,case,fixed,1000,'cuda',False,d1,lambda_count=0.)
        rec['lambda_zero_old_gradient']=compare(d0['gradients'],d1['gradients'])
        rec['lambda_zero_old_loss']=metric(d0['loss'],d1['loss'])
        if not rec['lambda_zero_old_gradient']['passes'] or not rec['lambda_zero_old_loss']['passes']:raise AssertionError('lambda0 differs')
        del d0,d1
        # One fixed-input scale report (separate CE/weighted auxiliary gradients at logits).
        inp=fixed[0];token=('scale',0);ctx=FrozenTypedContext(m,case['space'].types,inp['edges'],inp['edge_families'],token)
        try:
            from hsgd.losses import supported_binary_ce
            graph,logits,p=ctx.predict(inp['query'],inp['f'],.5,token,details=True)
            ce=supported_binary_ce(logits,torch.as_tensor(inp['labels'],device='cuda'),m.prior[graph.query_families]).sum()/18336
            aux,counts=count_loss(p,graph.query_families,*case_contract(case,inp))
            gc=torch.autograd.grad(ce,logits,retain_graph=True)[0];ga=torch.autograd.grad(LAMBDA*aux,logits)[0]
            rec['loss_scale']=dict(ce=float(ce),auxiliary=float(aux),weighted_auxiliary=LAMBDA*float(aux),
                ce_logits_grad_l2=float(gc.norm()),weighted_aux_logits_grad_l2=float(ga.norm()),
                ce_logits_grad_max=float(gc.abs().max()),weighted_aux_logits_grad_max=float(ga.abs().max()),
                count_gradient_nonzero=bool(torch.isfinite(ga).all() and ga.abs().max()>0))
            if not rec['loss_scale']['count_gradient_nonzero']:raise AssertionError('aux gradient absent/nonfinite')
        finally:ctx.close()
        del graph,logits,p,ce,aux,gc,ga,ctx
        # One CE control update; then three auxiliary calls: 1, 2, restored 2.
        histories=[];continuous=None
        for call,(arm,position) in enumerate([('CE',1),('CE_COUNT',1),('CE_COUNT',2),('CE_COUNT',2)]):
            check()
            if call in (0,1,3):
                del m,opt,ema
                m,opt,case,cfg,rng,proof2=fresh(args.dataset,arm,seed);ema=ParameterEMA(m)
                if proof2['tensor_sha256']!=proof['tensor_sha256']:raise AssertionError('paired initial differs')
                if call==3:
                    payload=torch.load(out/'after1.pt',map_location='cpu')
                    restore_payload(payload,m,opt,ema,binding,lambda state:rng_set(state,rng,'cuda'))
                    rec['load_exact']={k:equal(v,payload[k]) for k,v in [('model',m.state_dict()),('optimizer',opt.state_dict()),('ema',ema.state_dict()),('rng',rng_get(rng,'cuda'))]}
                    if not all(rec['load_exact'].values()):raise AssertionError('load mismatch')
                else:
                    torch.save(checkpoint_payload(m,opt,ema,0,rng_get(rng,'cuda'),binding),out/f'initial_{arm}.pt')
            if call==2:
                begin=time.perf_counter()
                rec['validation']=dict(raw=validation(m,case,cfg,seed,rng),ema=ema_validation(validation,m,ema,case,cfg,seed,rng))
                # Actual EMA load + full-query inference, without sampling/commit.
                with isolated_readout(m,opt,ema,rng,'cuda'):
                    state=ema.readout_state_dict(m);torch.save(dict(model=state,updates=ema.updates,readout='ema_0.99'),out/'engineering_readout.pt')
                    saved=torch.load(out/'engineering_readout.pt',map_location='cpu');m.load_state_dict(saved['model']);m.eval()
                    if not equal(m.state_dict(),state):raise AssertionError('EMA file selection')
                    with torch.no_grad():
                        ctx=FrozenTypedContext(m,case['space'].types,fixed[0]['edges'],fixed[0]['edge_families'],'isolated')
                        try:ctx.predict(fixed[0]['query'],fixed[0]['f'],.5,'isolated')
                        finally:ctx.close()
                    # Exercise all RNG restoration, independently of sampler internals.
                    rng.random();torch.rand(2,device='cuda')
                rec['intermediate_isolation_seconds']=time.perf_counter()-begin
            inputs=pair_inputs(case,cfg,position,rng);ih=input_hash(inputs)
            if call==1 and ih!=histories[0]['input_sha256']:raise AssertionError('CE/aux input pairing')
            rec['optimizer_step_calls']+=1;save(out/'summary.json',rec)
            begin=time.perf_counter();row=complete_normal2(normal2,m,opt,case,inputs,1000,'cuda',ema,position-1);sync('cuda')
            row.update(wall_seconds=time.perf_counter()-begin,arm=arm,input_sha256=ih)
            histories.append(row)
            payload=checkpoint_payload(m,opt,ema,position,rng_get(rng,'cuda'),binding)
            if call==1:
                begin=time.perf_counter();torch.save(payload,out/'after1.pt');rec['save_seconds']=time.perf_counter()-begin
            if call==2:continuous=payload;second_hash=ih;gradients=cpu_copy({k:q.grad for k,q in m.named_parameters()})
            if call==3:
                rec['continuation']={k:compare(continuous[k],payload[k]) for k in ('model','optimizer','ema')}
                rec['continuation']['gradient']=compare(gradients,{k:q.grad for k,q in m.named_parameters()})
                rec['continuation_discrete_exact']=ih==second_hash and equal(continuous['rng'],payload['rng']) and payload['updates']==2
                if not rec['continuation_discrete_exact'] or not all(x['passes'] for x in rec['continuation'].values()):raise AssertionError('continuation mismatch')
        rec['updates']=histories;rec['status']='passed'
    except BaseException as e:rec.update(status='failed',error=repr(e),traceback=traceback.format_exc())
    finally:
        rec.update(seconds=time.time()-start,finished_epoch=time.time(),peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
        if torch.cuda.is_initialized():rec.update(peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20)
        save(out/'summary.json',rec)
    return 0 if rec['status']=='passed' else 2
if __name__=='__main__':raise SystemExit(main())
