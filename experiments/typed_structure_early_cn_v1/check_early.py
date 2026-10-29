"""Bounded disposable engineering only: three optimizer calls per dataset."""
import os,time,pathlib,json,traceback,resource,copy,argparse
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from model_io import fresh,configure,sealed,tensor_hash,ROOT
import torch,numpy as np
from model_variants import StructuralDenoiser
from experiment_contract import read_dictionary
from runtime_contract import dictionary_expected,source_binding,sha,save
from probe import rng_get,rng_set,equal,metric,flatten,input_hash
from normal2 import pair_inputs,normal2,sync
from integration import complete_normal2,checkpoint_payload,restore_payload,ema_validation
from ema import ParameterEMA,cpu_copy
from hsgd.computational import FrozenTypedContext
from hsgd.kernel import MarginalKernel
from hsgd.bw_preflight.common import schedule

def compare_tree(a,b):
    aa,bb=flatten(a),flatten(b)
    rows={k:metric(v,bb[k]) for k,v in aa.items()}
    return dict(exact=equal(a,b),passes=set(aa)==set(bb) and all(r['passes'] for r in rows.values()),tensors=rows)

def prediction(m,case,inp,mask=False,reverse=False):
    token=('engineering',id(inp));c=FrozenTypedContext(m,case['space'].types,inp['edges'],inp['edge_families'],token)
    try:
        if mask or reverse:
            g=c.assemble(c.transfer_query(c.prepare_query(inp['query'],inp['f'],token),token),token)
            if reverse:
                keys=g.edges[0]*len(c.types)+g.edges[1]
                target=g.edges[1,g.query_index]*len(c.types)+g.edges[0,g.query_index]
                g.query_index=torch.searchsorted(keys,target)
            logits=m.joint_predict(c,g,inp['t']/1000,mask_cn=mask)
            return logits.detach().cpu(),None
        g,l,p=c.predict(inp['query'],inp['f'],inp['t']/1000,token,details=True)
        return l.detach().cpu(),p.detach().cpu()
    finally:c.close()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--dataset',required=True,choices=['IMDB','PubMed']);ap.add_argument('--output',required=True);a=ap.parse_args()
    out=pathlib.Path(a.output);out.mkdir(parents=True,exist_ok=False)
    started=time.time();r=dict(dataset=a.dataset,status='running',optimizer_step_calls=0,formal_updates=0,generated_graphs=0)
    def check():
        if time.time()>=1793244900:raise TimeoutError('engineering deadline 11:35 Asia/Shanghai')
    try:
        check();r['environment']=configure();seed={'IMDB':2026102911,'PubMed':2026102912}[a.dataset]
        m,opt,case,cfg,rng,proof=fresh(a.dataset,'E',seed);ema=ParameterEMA(m)
        binding=dict(source_sha256=source_binding()[1],dataset=a.dataset,config=cfg,proof=proof,engineering_only=True)
        initial=checkpoint_payload(m,opt,ema,0,rng_get(rng,'cuda'),binding)
        torch.save(initial,out/'initial.pt');save(out/'initial_binding.json',dict(binding=binding,sha256=sha(out/'initial.pt')))
        r['parameters']=proof['construction']['parameters'];r['parameter_delta']=proof['construction']['parameter_delta'];r['initial_rng_exact']=proof['original_training_rng_exact']
        basis,_=read_dictionary(ROOT/'dictionaries',a.dataset,dictionary_expected(a.dataset))
        old=StructuralDenoiser(case,cfg,'E',basis).cuda();old.load_state_dict(sealed(a.dataset)[0]['model']);old.eval();m.eval()
        fixed=pair_inputs(case,cfg,1,np.random.default_rng(2026102901),500)
        r['fixed_input_sha256']=input_hash(fixed)
        np.savez_compressed(out/'fixed_inputs.npz',**{str(i)+'_'+k:v for i,x in enumerate(fixed) for k,v in x.items() if isinstance(v,np.ndarray)})
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            before=rng_get(rng,'cuda');st=time.perf_counter();a0,_=prediction(old,case,fixed[0],True);sync('cuda');r['old_off_forward_seconds']=time.perf_counter()-st
            st=time.perf_counter();b0,_=prediction(m,case,fixed[0]);sync('cuda');r['early_zero_forward_seconds']=time.perf_counter()-st
            r['zero_projection_vs_old_off']=metric(b0,a0)
            r['forward_rng_unchanged']=equal(before,rng_get(rng,'cuda'))
            if not r['zero_projection_vs_old_off']['passes'] or not r['forward_rng_unchanged']:raise AssertionError('zero projection or random operator audit')
        del old
        rng_set(initial['rng'],rng,'cuda')
        records=[];continuous=None;second_input=None
        for call,position in enumerate([1,2,2]):
            check()
            if call==2:
                m,opt,case,cfg,rng,_=fresh(a.dataset,'E',seed);ema=ParameterEMA(m)
                p=torch.load(out/'after1.pt',map_location='cpu')
                restore_payload(p,m,opt,ema,binding,lambda s:rng_set(s,rng,'cuda'))
                r['loaded_exact']=dict(model=equal(m.state_dict(),p['model']),optimizer=equal(opt.state_dict(),p['optimizer']),ema=equal(ema.state_dict(),p['ema']),rng=equal(rng_get(rng,'cuda'),p['rng']))
                if not all(r['loaded_exact'].values()):raise AssertionError('load not exact')
            inputs=pair_inputs(case,cfg,position,rng);ih=input_hash(inputs)
            # Account before step: even an exception cannot hide a possible call.
            r['optimizer_step_calls']+=1;save(out/'summary.json',r)
            st=time.perf_counter();rec=complete_normal2(normal2,m,opt,case,inputs,1000,'cuda',ema,position-1);sync('cuda')
            rec.update(wall_seconds=time.perf_counter()-st,input_sha256=ih,projection_grad_norm=float(m.cn_edge_projection.weight.grad.norm()),
                cn_encoder_grad_norm=float(sum(p.grad.norm() for p in m.common_neighbor.parameters() if p.grad is not None)))
            records.append(rec)
            payload=checkpoint_payload(m,opt,ema,position,rng_get(rng,'cuda'),binding)
            if call==0:
                st=time.perf_counter();torch.save(payload,out/'after1.pt');r['checkpoint_save_seconds']=time.perf_counter()-st
                if rec['projection_grad_norm']<=0:raise AssertionError('projection gradient missing')
            if call==1:continuous=payload;second_input=ih;gradients=cpu_copy({k:p.grad for k,p in m.named_parameters()})
            if call==2:
                r['continuation']={k:compare_tree(continuous[k],payload[k]) for k in ('model','optimizer','ema')}
                r['continuation']['gradients']=compare_tree(gradients,{k:p.grad for k,p in m.named_parameters()})
                r['continuation_discrete_exact']=ih==second_input and equal(continuous['rng'],payload['rng']) and payload['updates']==2
                if not all(x['passes'] for x in r['continuation'].values()) or not r['continuation_discrete_exact']:raise AssertionError('continuation tolerance')
        r['updates']=records
        if records[1]['cn_encoder_grad_norm']<=0:raise AssertionError('CN learned upstream after projection update absent')
        m.eval();rng_before=rng_get(rng,'cuda');r['inference']=[]
        with torch.no_grad():
            for t in (1,500,1000):
                check();inp=pair_inputs(case,cfg,1,np.random.default_rng(2026102901),t)[0]
                begin=time.perf_counter();logits,p=prediction(m,case,inp);sync('cuda');sec=time.perf_counter()-begin
                keys=inp['edges'][0]*192+inp['edges'][1];query=inp['query'][0]*192+inp['query'][1]
                xt=np.isin(query,keys).astype(np.uint8);begin=time.perf_counter()
                q=MarginalKernel(case['prior'],schedule(1000)).reverse(p.numpy().astype(np.float64),xt,inp['f'],t-1,t)
                r['inference'].append(dict(t=t,full_slots=len(inp['ids']),seconds=sec,posterior_seconds=time.perf_counter()-begin,finite=bool(torch.isfinite(logits).all() and torch.isfinite(p).all() and np.isfinite(q).all()),no_sampling_or_commit=True))
            # Endpoint reversal with nonzero learned projection, same pair order.
            inp=copy.deepcopy(fixed[0]);l,_=prediction(m,case,inp)
            lr,_=prediction(m,case,inp,reverse=True);r['endpoint_reversal']=metric(lr,l)
            # Type-preserving arbitrary relabeling, graph/query rows correspond.
            perm=np.arange(192);pr=np.random.default_rng(2026102903)
            for typ in range(4):
                ids=np.flatnonzero(case['space'].types==typ);perm[ids]=pr.permutation(ids)
            inp=copy.deepcopy(fixed[0]);inp['edges']=np.sort(perm[inp['edges']],axis=0);inp['query']=np.sort(perm[inp['query']],axis=0)
            lp,_=prediction(m,case,inp);r['node_relabeling']=metric(lp,l)
        raw=cpu_copy(m.state_dict());mode=m.training
        ema_validation(lambda model: prediction(model,case,fixed[0]),m,ema)
        r['ema_switch_restores']=equal(raw,m.state_dict()) and m.training==mode and equal(rng_before,rng_get(rng,'cuda'))
        if not r['ema_switch_restores'] or not r['endpoint_reversal']['passes'] or not r['node_relabeling']['passes']:raise AssertionError('symmetry/switch')
        r['status']='passed'
    except BaseException as e:r.update(status='failed',error=repr(e),traceback=traceback.format_exc())
    finally:
        r.update(seconds=time.time()-started,finished_epoch=time.time(),rss_peak_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
        if torch.cuda.is_initialized():r.update(peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20)
        save(out/'summary.json',r)
    return 0 if r['status']=='passed' else 2
if __name__=='__main__':raise SystemExit(main())
