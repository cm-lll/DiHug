"""Exactly three temporary normal2 updates per dataset; no generation."""
import os,time,pathlib,argparse,traceback,resource,functools
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from model_io import configure,fresh
import torch
from runtime_contract import save,sha,source_binding
from probe import rng_get,rng_set,equal,flatten,metric,input_hash
from normal2 import normal2,pair_inputs,sync
from ema import ParameterEMA,cpu_copy
from integration import complete_normal2,checkpoint_payload,restore_payload,ema_validation
from worker import validation,atomic_checkpoint

def compare(a,b):
    x,y=flatten(a),flatten(b)
    if x.keys()!=y.keys():return dict(exact=False,passes=False,reason='keys')
    rows={k:metric(v,y[k]) for k,v in x.items()}
    return dict(exact=equal(a,b),passes=all(v['passes'] for v in rows.values()),tensors=rows)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--dataset',required=True);ap.add_argument('--output',required=True);ap.add_argument('--deadline',type=float,required=True);a=ap.parse_args()
    out=pathlib.Path(a.output);out.mkdir(parents=True,exist_ok=False);start=time.time()
    rec=dict(status='running',optimizer_step_calls=0,formal_updates=0,generated_graphs=0,dataset=a.dataset)
    def check():
        if time.time()>=a.deadline:raise TimeoutError('engineering deadline')
    try:
        check();rec['environment']=configure();seed={'IMDB':2026102911,'PubMed':2026102912}[a.dataset]
        m,opt,case,cfg,rng,proof=fresh(a.dataset,'CE_COUNT',seed);ema=ParameterEMA(m)
        binding=dict(engineering_only=True,dataset=a.dataset,source_sha256=source_binding()[1])
        initial=checkpoint_payload(m,opt,ema,0,rng_get(rng,'cuda'),binding)
        atomic_checkpoint(out/'initial.pt',initial);save(out/'initial_binding.json',dict(proof=proof,sha256=sha(out/'initial.pt')))
        rec['parameters']=proof['parameters']
        inputs=pair_inputs(case,cfg,1,rng);random=rng_get(rng,'cuda');rec['input_sha256']=input_hash(inputs)
        d0={};d1={}
        plain=normal2(m,opt,case,inputs,1000,'cuda',step=False,diagnostics=d0,observe=False)
        after_plain=rng_get(rng,'cuda');rng_set(random,rng,'cuda')
        observed=normal2(m,opt,case,inputs,1000,'cuda',step=False,diagnostics=d1,observe=True)
        rec['observation_gradient']=compare(d0['gradients'],d1['gradients'])
        rec['observation_loss']=metric(d0['loss'],d1['loss'])
        rec['observation_rng_exact']=equal(after_plain,rng_get(rng,'cuda'))
        rec['observation_does_not_change_state']=equal(m.state_dict(),initial['model']) and equal(opt.state_dict(),initial['optimizer'])
        rec['observation_example']=observed['gradient_observation']
        if not all([rec['observation_gradient']['passes'],rec['observation_loss']['passes'],rec['observation_rng_exact'],rec['observation_does_not_change_state']]):raise AssertionError('observation changes computation')
        del d0,d1
        rng_set(random,rng,'cuda');history=[]
        for call in range(3):
            check();position=1 if call==0 else 2
            if call==1:
                before=checkpoint_payload(m,opt,ema,1,rng_get(rng,'cuda'),binding);mode=m.training;vbegin=time.perf_counter()
                rec['validation']={'raw':validation(m,case,cfg,seed,rng),'ema':ema_validation(validation,m,ema,case,cfg,seed,rng)}
                after=checkpoint_payload(m,opt,ema,1,rng_get(rng,'cuda'),binding)
                rec['validation_isolation_exact']=equal(before,after) and m.training==mode
                rec['validation_seconds']=time.perf_counter()-vbegin
                if not rec['validation_isolation_exact']:raise AssertionError('validation changed state/RNG')
                del before,after
                inputs=pair_inputs(case,cfg,2,rng)
            if call==2:
                del m,opt,ema
                m,opt,case,cfg,rng,proof2=fresh(a.dataset,'CE_COUNT',seed);ema=ParameterEMA(m)
                payload=torch.load(out/'after1.pt',map_location='cpu')
                count=restore_payload(payload,m,opt,ema,binding,lambda state:rng_set(state,rng,'cuda'))
                rec['load_exact']={k:equal(v,payload[k]) for k,v in [('model',m.state_dict()),('optimizer',opt.state_dict()),('ema',ema.state_dict()),('rng',rng_get(rng,'cuda'))]}
                rec['load_count_exact']=count==1
                if not all(rec['load_exact'].values()) or count!=1:raise AssertionError('load mismatch')
                inputs=pair_inputs(case,cfg,2,rng)
            ih=input_hash(inputs);rec['optimizer_step_calls']+=1;save(out/'summary.json',rec)
            begin=time.perf_counter()
            row=complete_normal2(functools.partial(normal2,observe=call!=2),m,opt,case,inputs,1000,'cuda',ema,position-1)
            row.update(wall_seconds=time.perf_counter()-begin,input_sha256=ih);history.append(row)
            payload=checkpoint_payload(m,opt,ema,position,rng_get(rng,'cuda'),binding)
            if call==0:
                begin=time.perf_counter();atomic_checkpoint(out/'after1.pt',payload);rec['save_seconds']=time.perf_counter()-begin
            if call==1:continuous=payload;second_hash=ih;grads=cpu_copy({n:p.grad for n,p in m.named_parameters()})
            if call==2:
                rec['continuation']={k:compare(continuous[k],payload[k]) for k in ('model','optimizer','ema')}
                rec['continuation']['gradients']=compare(grads,{n:p.grad for n,p in m.named_parameters()})
                rec['continuation_discrete_exact']=ih==second_hash and equal(continuous['rng'],payload['rng']) and payload['updates']==2
                if not rec['continuation_discrete_exact'] or not all(x['passes'] for x in rec['continuation'].values()):raise AssertionError('continuation mismatch')
                atomic_checkpoint(out/'restored_after2.pt',payload)
        rec['updates']=history;rec['status']='passed'
    except BaseException as exc:rec.update(status='failed',error=repr(exc),traceback=traceback.format_exc())
    finally:
        rec.update(seconds=time.time()-start,finished_epoch=time.time(),peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
        if torch.cuda.is_initialized():rec.update(peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20)
        save(out/'summary.json',rec)
    return 0 if rec['status']=='passed' else 2
if __name__=='__main__':raise SystemExit(main())

