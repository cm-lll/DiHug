"""One GPU, exactly five temporary normal2 optimizer calls; no chain sampling.

2 raw +2 EMA +1 repeated second update after checkpoint restore. All initial
bytes, losses, random inputs and scientific functions are the formal ones.
"""
import argparse,pathlib,time,json,signal,traceback,resource,sys,os,gc
from runtime_contract import ROOT,source_binding,save,sha,output_check
sys.path.insert(0,str(ROOT/'package'))
import torch
from model_io import configure,fresh
from normal2 import normal2,pair_inputs,sync
from probe import equal,rng_get,rng_set,input_hash,flatten,metric
from hsgd.bw_preflight.common import cpu_tree
from worker import validation,atomic_checkpoint
from ema import ParameterEMA
from integration import complete_normal2,checkpoint_payload,restore_payload,ema_validation

def compare(a,b):
    aa=flatten(a);bb=flatten(b)
    if aa.keys()!=bb.keys():raise AssertionError('tensor keys differ')
    rows={k:metric(v,bb[k]) for k,v in aa.items()}
    return dict(exact=equal(a,b),passes=all(r['passes'] for r in rows.values()),tensors=rows)
def main():
    started=time.time();p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
    out=output_check(a.output,[ROOT]);out.mkdir(parents=True)
    count=[0];deadline=started+900
    result=dict(status='running',engineering_only=True,formal_training=0,new_graphs=0,source_sha256=source_binding()[1],started_epoch=started,cap_seconds=900)
    def interrupt(s,f):raise InterruptedError('preflight signal/deadline')
    for s in [signal.SIGTERM,signal.SIGINT,signal.SIGHUP,signal.SIGALRM]:signal.signal(s,interrupt)
    signal.alarm(max(1,int(deadline-time.time())))
    def check():
        if time.time()>=deadline or (out/'STOP').exists():raise InterruptedError('preflight budget/STOP')
    def build():
        check();m,o,c,cfg,r,proof=fresh(2026102812)
        original=o.step
        def counted(*args,**kwargs):
            check()
            if count[0]>=5:raise RuntimeError('engineering optimizer cap')
            count[0]+=1;save(out/'update_count.json',dict(optimizer_step_calls=count[0],maximum=5))
            return original(*args,**kwargs)
        o.step=counted
        return m,o,c,cfg,r,proof
    try:
        result['environment']=configure();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
        model,opt,case,cfg,rng,proof=build();result['initial']=proof
        atomic_checkpoint(out/'engineering_initial.pt',dict(model=cpu_tree(model.state_dict()),rng=rng_get(rng,'cuda'),updates=0,purpose='engineering only'))
        raw=[]
        for u in (1,2):
            inp=pair_inputs(case,cfg,u,rng);rec=normal2(model,opt,case,inp,1000,'cuda',True)
            raw.append(dict(input_hash=input_hash(inp),rng=rng_get(rng,'cuda'),model=cpu_tree(model.state_dict()),optimizer=cpu_tree(opt.state_dict()),gradients={n:cpu_tree(p.grad) for n,p in model.named_parameters()},record=rec))
        del model,opt;gc.collect();torch.cuda.empty_cache()
        model,opt,case,cfg,rng,proof=build();ema=ParameterEMA(model);enhanced=[];binding=dict(source_sha256=result['source_sha256'],purpose='isolated engineering',seed=2026102812)
        for u in (1,2):
            check();inp=pair_inputs(case,cfg,u,rng);sync('cuda');begin=time.perf_counter()
            rec=complete_normal2(normal2,model,opt,case,inp,1000,'cuda',ema,u-1);sync('cuda');wall=time.perf_counter()-begin
            row=dict(update=u,input_exact=input_hash(inp)==raw[u-1]['input_hash'],rng_exact=equal(rng_get(rng,'cuda'),raw[u-1]['rng']),
                     model=compare(model.state_dict(),raw[u-1]['model']),optimizer=compare(opt.state_dict(),raw[u-1]['optimizer']),
                     gradients=compare({n:p.grad for n,p in model.named_parameters()},raw[u-1]['gradients']),
                     original_normal2_seconds=raw[u-1]['record']['normal2_seconds'],normal2_with_ema_seconds=wall,
                     unchanged_normal2_component_seconds=rec['normal2_seconds'],ema_wrapper_seconds=wall-rec['normal2_seconds'])
            enhanced.append(row)
            if u==1:
                begin=time.perf_counter();atomic_checkpoint(out/'engineering_u1.pt',checkpoint_payload(model,opt,ema,1,rng_get(rng,'cuda'),binding));result['checkpoint_save_seconds']=time.perf_counter()-begin
        expected=checkpoint_payload(model,opt,ema,2,rng_get(rng,'cuda'),binding)
        before=(cpu_tree(model.state_dict()),cpu_tree(opt.state_dict()),rng_get(rng,'cuda'),model.training)
        check();begin=time.perf_counter();vraw=validation(model,case,cfg,2026102812,rng);vema=ema_validation(validation,model,ema,case,cfg,2026102812,rng)
        result['validation']=dict(raw=vraw,ema=vema,total_seconds=time.perf_counter()-begin,
                                  raw_model_exact=equal(before[0],model.state_dict()),optimizer_exact=equal(before[1],opt.state_dict()),
                                  rng_exact=equal(before[2],rng_get(rng,'cuda')),mode_exact=before[3]==model.training)
        del model,opt,ema;gc.collect();torch.cuda.empty_cache()
        model,opt,case,cfg,rng,proof=build();ema=ParameterEMA(model)
        loaded=torch.load(out/'engineering_u1.pt',map_location='cpu');begin=time.perf_counter()
        u=restore_payload(loaded,model,opt,ema,binding,lambda s:rng_set(s,rng,'cuda'))
        result['restore_seconds']=time.perf_counter()-begin
        result['loaded_exact']=dict(model=equal(loaded['model'],model.state_dict()),optimizer=equal(loaded['optimizer'],opt.state_dict()),
                                    ema=equal(loaded['ema'],ema.state_dict()),rng=equal(loaded['rng'],rng_get(rng,'cuda')),updates=u==1)
        inp=pair_inputs(case,cfg,2,rng);result['restored_input_exact']=input_hash(inp)==raw[1]['input_hash']
        complete_normal2(normal2,model,opt,case,inp,1000,'cuda',ema,1)
        actual=checkpoint_payload(model,opt,ema,2,rng_get(rng,'cuda'),binding)
        result['continuation']={k:compare(actual[k],expected[k]) for k in ['model','optimizer','ema']};result['continuation_rng_exact']=equal(actual['rng'],expected['rng'])
        atomic_checkpoint(out/'engineering_resumed_u2.pt',actual)
        result['paired_updates']=enhanced
        passed=all(x['input_exact'] and x['rng_exact'] and all(x[k]['passes'] for k in ['model','optimizer','gradients']) for x in enhanced)
        passed=passed and all(result['loaded_exact'].values()) and all(result['validation'][k] for k in ['raw_model_exact','optimizer_exact','rng_exact','mode_exact'])
        passed=passed and result['restored_input_exact'] and result['continuation_rng_exact'] and all(x['passes'] for x in result['continuation'].values())
        result['status']='passed' if passed else 'failed';result['compute_and_evidence_seconds']=time.perf_counter()-start
    except BaseException as exc:result.update(status='failed_or_incomplete',error=repr(exc),traceback=traceback.format_exc())
    finally:
        signal.alarm(0);result.update(optimizer_step_calls=count[0],finished_epoch=time.time(),total_seconds=time.time()-started,peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
        if torch.cuda.is_initialized():result.update(peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20)
        save(out/'preflight.json',result)
    return 0 if result['status']=='passed' else 2
if __name__=='__main__':raise SystemExit(main())
