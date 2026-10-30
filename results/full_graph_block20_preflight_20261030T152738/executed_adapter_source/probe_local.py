"""Bounded engineering only: <=3 normal2 updates and ONE reverse step per A."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('MKL_NUM_THREADS','1')
import argparse,copy,gc,hashlib,json,pathlib,random,resource,signal,sys,time,traceback,warnings
import block20 as B
import numpy as np
import torch
from ema import ParameterEMA
from hsgd.bw_preflight.common import cpu_tree,environment,rng_save,rng_restore,schedule,query_inventory,plan_for,seed_all
from hsgd.runtime import reverse_step
from hsgd.kernel import MarginalKernel
from probe import equal,flatten,metric

def save(p,x):
    p=pathlib.Path(p);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(x,indent=2,allow_nan=False));tmp.replace(p)
def atomic_torch(p,x):
    p=pathlib.Path(p);tmp=p.with_suffix('.tmp');torch.save(x,tmp);os.replace(tmp,p)
def mem():return dict(allocated_mib=torch.cuda.max_memory_allocated()/2**20,reserved_mib=torch.cuda.max_memory_reserved()/2**20,cpu_peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
def state_hash(s):return {k:hashlib.sha256(v.detach().cpu().contiguous().numpy().tobytes()).hexdigest() for k,v in s.items()}
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--dataset',choices=list(B.DATA),required=True);ap.add_argument('--output',required=True);ap.add_argument('--ledger',required=True);a=ap.parse_args()
    out=pathlib.Path(a.output);out.mkdir(parents=True,exist_ok=False);ledger=json.loads(pathlib.Path(a.ledger).read_text());stop=__import__('datetime').datetime.fromisoformat(ledger['science_stop_Asia_Shanghai']).timestamp()
    result=dict(dataset=a.dataset,arm='A',loss='CE',status='running',actual_optimizer_steps=0,completed_updates=0,full_T1000_chains=0,single_reverse_steps=0)
    def event(stage,idx=None,info=None):
        if time.time()>=stop:raise TimeoutError('ticket science stop; delivery reserve')
        result['last_phase']=dict(stage=stage,input_index=idx,info=info or {},epoch=time.time());save(out/'status.json',result);print(stage,idx,flush=True)
    signal.signal(signal.SIGALRM,lambda s,f:(_ for _ in ()).throw(TimeoutError('absolute science deadline')));signal.alarm(max(1,int(stop-time.time())))
    begin=time.perf_counter()
    with warnings.catch_warnings(record=True) as warns:
      warnings.simplefilter('always')
      try:
        torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True,warn_only=False);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        if not torch.cuda.is_available() or torch.cuda.device_count()!=1:raise RuntimeError('exactly one selected GPU required')
        save(out/'environment.json',environment('cuda'));case,cfg=B.load_case(a.dataset);save(out/'resolved_config.json',cfg)
        inv=query_inventory(case,plan_for(case,[cfg['seed'],0,21],20));save(out/'query_inventory.json',inv)
        seed_all(cfg['seed']);model=B.make_model(case,cfg,'A');initial=cpu_tree(model.state_dict());save(out/'initial_tensor_hashes.json',state_hash(initial));atomic_torch(out/'initial_engineering.pt',dict(model=initial,config=cfg,updates=0,engineering_only=True))
        model=model.to('cuda');opt=torch.optim.AdamW(model.parameters(),**{k:v for k,v in cfg['optimizer'].items() if k!='type'});ema=ParameterEMA(model,.99);rng=np.random.default_rng(cfg['seed']+71);torch.cuda.reset_peak_memory_stats()
        result['parameters']=sum(p.numel() for p in model.parameters());result['updates']=[]
        save(out/'binding.json',dict(data_sha256=B.DATA[a.dataset][2],config=cfg,source_parent=json.loads((B.ROOT/'source_parent.json').read_text()),query_inventory=inv,engineering_only=True))
        for seq in (1,2,3):
            event('prepare_update',seq)
            if seq==3:
                ck=torch.load(out/'after1.pt',map_location='cpu');event('restore')
                del opt,ema,model;gc.collect();torch.cuda.empty_cache()
                model=B.make_model(case,cfg,'A').to('cuda');model.load_state_dict(ck['model'],strict=True)
                opt=torch.optim.AdamW(model.parameters(),**{k:v for k,v in cfg['optimizer'].items() if k!='type'});opt.load_state_dict(ck['optimizer']);ema=ParameterEMA(model,.99);ema.load_state_dict(ck['ema'],model,1)
                rng_restore(ck['rng'],rng,'cuda')
                loaded=dict(model=equal(cpu_tree(model.state_dict()),ck['model']),optimizer=equal(cpu_tree(opt.state_dict()),ck['optimizer']),ema=equal(ema.state_dict(),ck['ema']),rng=equal(rng_save(rng,'cuda'),ck['rng']),updates=ck['updates']==1)
                save(out/'restore_load.json',loaded)
                if not all(loaded.values()):raise ValueError('incomplete checkpoint load')
                result['completed_updates']=1
            inp=B.pair_inputs(case,cfg,result['completed_updates'],rng,force_t=500)
            ih=hashlib.sha256()
            for x in inp:
                for k in ('ids','f','query','labels','edges','edge_families'):ih.update(np.ascontiguousarray(x[k]).tobytes())
            ih=ih.hexdigest()
            event('normal2_start',seq,dict(input_hash=ih))
            # Count an actual step only after the complete update returns.
            rec=B.normal2(model,opt,case,inp,cfg['T'],'cuda',phase=event)
            result['actual_optimizer_steps']+=1;result['completed_updates']+=1
            stamp=time.perf_counter();ema.update(model,result['completed_updates']);torch.cuda.synchronize();rec['EMA_seconds']=time.perf_counter()-stamp;rec['input_hash']=ih;rec['sequence']=seq;rec['memory']=mem();result['updates'].append(rec)
            if seq==1:
                payload=dict(model=cpu_tree(model.state_dict()),optimizer=cpu_tree(opt.state_dict()),ema=ema.state_dict(),rng=rng_save(rng,'cuda'),updates=1,config=cfg,data_sha256=B.DATA[a.dataset][2],engineering_only=True)
                stamp=time.perf_counter();atomic_torch(out/'after1.pt',payload);result['save_seconds']=time.perf_counter()-stamp;result['checkpoint_bytes']=(out/'after1.pt').stat().st_size
            if seq==2:
                continuous=dict(model=cpu_tree(model.state_dict()),optimizer=cpu_tree(opt.state_dict()),ema=ema.state_dict(),rng=rng_save(rng,'cuda'),input_hash=ih)
            if seq==3:
                actual=dict(model=cpu_tree(model.state_dict()),optimizer=cpu_tree(opt.state_dict()),ema=ema.state_dict(),rng=rng_save(rng,'cuda'),input_hash=ih)
                cmp={}
                for category in ('model','optimizer','ema'):
                    left,right=flatten(actual[category]),flatten(continuous[category]);cmp[category]=dict(exact=equal(actual[category],continuous[category]),tensors={k:metric(v,right[k]) for k,v in left.items()})
                    cmp[category]['within_original_tolerance']=all(v['passes'] for v in cmp[category]['tensors'].values())
                cmp.update(input_exact=ih==continuous['input_hash'],rng_exact=equal(actual['rng'],continuous['rng']))
                save(out/'restore_continue.json',cmp)
                result['restore_continuation_passed']=all(v['within_original_tolerance'] for k,v in cmp.items() if isinstance(v,dict)) and cmp['input_exact'] and cmp['rng_exact']
            save(out/'status.json',result)
        # One ordinary t=500 state; NOT a terminal-generated or full-chain state.
        event('single_reverse_step_start');model.eval();model.load_state_dict(ema.readout_state_dict(model),strict=True)
        state=np.zeros(case['space'].total,dtype=np.uint8);f,loc=case['space'].encode(inp[0]['edges']);state[case['space'].offsets[f]+loc]=1
        before=state.copy();h=B.digest(state);kernel=MarginalKernel(case['prior'],schedule(cfg['T'],cfg['schedule_offset']));events=[]
        def step_event(kind,**fields):
            events.append(dict(kind=kind,**fields));event(kind,None,fields)
        torch.cuda.reset_peak_memory_stats();value,stats=reverse_step(model,case,state,kernel,500,20,cfg['seed']+91,np.random.default_rng(cfg['seed']+93),'cuda',event=step_event,check_time=lambda:event('reverse_deadline_check'),trace=False)
        if not np.array_equal(before,state):raise AssertionError('frozen Xt modified before commit')
        if stats['blocks']!=20 or stats['covered_slots']!=int(np.sum(np.where(case['active'],case['space'].counts,0))) or not stats['exact_coverage']:raise ValueError('coverage')
        for family,prior in enumerate(case['prior']):
            if prior in (0.,1.) and not np.all(value[case['space'].offsets[family]:case['space'].offsets[family+1]]==int(prior)):raise ValueError('posterior fixed support')
        atomic_torch(out/'single_step_state.pt',dict(input_state=state,result_state=value,input_sha256=h,t=500,result_t=499,source='ordinary reference forward noise; temporary engineering EMA',complete_blocks=20,engineering_only=True))
        result.update(single_reverse_steps=1,sampling=stats,sampling_memory=mem(),frozen_Xt_unchanged=True,status='passed',single_step_state_commit='atomic only after full 20-block coverage')
      except BaseException as exc:
        result.update(status='failed',error=repr(exc),traceback=traceback.format_exc(),oom=isinstance(exc,torch.cuda.OutOfMemoryError),memory=mem() if torch.cuda.is_initialized() else None)
        (out/'failure.txt').write_text(result['traceback']);print(result['traceback'],flush=True)
      finally:
        signal.alarm(0);result['warnings']=[dict(category=w.category.__name__,message=str(w.message),file=w.filename,line=w.lineno) for w in warns];result['wall_seconds']=time.perf_counter()-begin;save(out/'status.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('updates','traceback','warnings')},indent=2),flush=True)

if __name__=='__main__':main()
