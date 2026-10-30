"""At most two engineering normal2 steps per cell; no chain.
Compare raw one update with restored-initial + EMA one update.
"""
import argparse,time,pathlib,sys,signal,traceback,resource
from runtime_contract import ROOT,save,source_binding,read
sys.path.insert(0,str(ROOT/'package'))
import torch
from model_io import configure,fresh
from ema import ParameterEMA,cpu_copy
from integration import complete_normal2,checkpoint_payload,restore_payload,ema_validation
from worker import validation,atomic_checkpoint
from normal2 import normal2,pair_inputs,sync
from probe import rng_get,rng_set,equal,input_hash,flatten,metric
def comparison(a,b):
    x,y=flatten(a),flatten(b);rows={k:metric(v,y[k]) for k,v in x.items()}
    return dict(exact=equal(a,b),passes=x.keys()==y.keys() and all(v['passes'] for v in rows.values()),tensors=rows)
def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);p.add_argument('--arm',required=True);p.add_argument('--output',required=True);p.add_argument('--deadline',type=float,required=True);a=p.parse_args()
    out=pathlib.Path(a.output);out.mkdir(parents=True,exist_ok=False);start=time.time();calls=0
    r=dict(status='incomplete',dataset=a.dataset,arm=a.arm,formal_updates=0,new_graphs=0,source_sha256=source_binding()[1])
    def check():
        if time.time()>=a.deadline or (out/'STOP').exists():raise TimeoutError('preparation deadline/STOP')
    def stop(*args):raise InterruptedError('engineering signal/deadline')
    for sig in (signal.SIGALRM,signal.SIGTERM,signal.SIGINT):signal.signal(sig,stop)
    signal.alarm(max(1,int(a.deadline-time.time())))
    try:
        check();r['environment']=configure()
        seed=read(ROOT/'protocol.json')['train_seeds'][a.dataset]
        m,o,c,cfg,rng,proof=fresh(a.dataset,a.arm,seed);ema=ParameterEMA(m)
        initial=checkpoint_payload(m,o,ema,0,rng_get(rng,'cuda'),dict(purpose='engineering only',proof=proof))
        atomic_checkpoint(out/'initial.pt',initial);save(out/'initialization.json',proof)
        real=o.step
        def counted(*args,**kwargs):
            nonlocal calls
            check()
            if calls>=2:raise RuntimeError('engineering cap')
            calls+=1;save(out/'updates.json',dict(optimizer_step_calls=calls,maximum=2));return real(*args,**kwargs)
        o.step=counted
        inp=pair_inputs(c,cfg,1,rng);ih=input_hash(inp)
        raw_rec=normal2(m,o,c,inp,1000,'cuda',True)
        expected=cpu_copy(dict(model=m.state_dict(),optimizer=o.state_dict(),grad={n:p.grad for n,p in m.named_parameters()},rng=rng_get(rng,'cuda')))
        restore_payload(initial,m,o,ema,initial['binding'],lambda s:rng_set(s,rng,'cuda'))
        inp=pair_inputs(c,cfg,1,rng);r['input_exact']=input_hash(inp)==ih
        sync('cuda');begin=time.perf_counter();rec=complete_normal2(normal2,m,o,c,inp,1000,'cuda',ema,0);sync('cuda');wall=time.perf_counter()-begin
        r['transparency']={k:comparison(v,expected[k]) for k,v in dict(model=m.state_dict(),optimizer=o.state_dict(),grad={n:p.grad for n,p in m.named_parameters()}).items()}
        r['rng_exact']=equal(rng_get(rng,'cuda'),expected['rng'])
        r['path']=dict(cn=m.common_neighbor is not None,refex=hasattr(m,'refex_projection'),roles=hasattr(m,'role_projection'),feature_calls=m.feature_calls)
        r['projection_gradients']={n:dict(finite=bool(torch.isfinite(p.grad).all()),norm=float(p.grad.norm())) for n,p in m.named_parameters() if any(k in n for k in ('refex_projection','role_projection','branch.degree_out')) and p.grad is not None}
        before=checkpoint_payload(m,o,ema,1,rng_get(rng,'cuda'),initial['binding'])
        begin=time.perf_counter();atomic_checkpoint(out/'after_engineering.pt',before);r['save_seconds']=time.perf_counter()-begin
        r['validation_raw']=validation(m,c,cfg,seed,rng)
        r['validation_ema']=ema_validation(validation,m,ema,c,cfg,seed,rng)
        r['validation_restored']=equal(before,checkpoint_payload(m,o,ema,1,rng_get(rng,'cuda'),initial['binding']))
        payload=torch.load(out/'after_engineering.pt',map_location='cpu')
        restore_payload(payload,m,o,ema,initial['binding'],lambda s:rng_set(s,rng,'cuda'))
        r['loaded_exact']=equal(payload,checkpoint_payload(m,o,ema,1,rng_get(rng,'cuda'),initial['binding']))
        r['cost']=dict(raw_normal2_seconds=raw_rec['normal2_seconds'],normal2_with_ema_seconds=wall,normal2_seconds=rec['normal2_seconds'],ema_overhead_seconds=wall-rec['normal2_seconds'],note='one-step repeats, not stable throughput')
        expected_path=dict(cn=a.arm in ('A','D','E'),refex=a.arm in ('C','D','E'),roles=a.arm in ('C','E'))
        passed=all(r[k] for k in ('input_exact','rng_exact','validation_restored','loaded_exact')) and all(x['passes'] for x in r['transparency'].values()) and all(r['path'][k]==v for k,v in expected_path.items())
        r['status']='passed' if passed else 'failed'
    except BaseException as e:r.update(error=repr(e),traceback=traceback.format_exc())
    finally:
        signal.alarm(0);r.update(optimizer_step_calls=calls,seconds=time.time()-start,finished_epoch=time.time(),peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
        if torch.cuda.is_initialized():r.update(peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20)
        save(out/'result.json',r)
    return 0 if r['status']=='passed' else 2
if __name__=='__main__':raise SystemExit(main())
