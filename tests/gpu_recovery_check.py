"""Bounded CUDA engineering checks; no scientific data/weights or full chain.

Run with exactly one visible idle NVIDIA GPU and CUBLAS_WORKSPACE_CONFIG set
before Python starts. Production strict determinism/TF32 settings are retained.
"""
import argparse
import copy
import gc
import hashlib
import importlib.util
import json
import os
import pathlib
import random
import statistics
import subprocess
import sys
import time
import traceback
from unittest import mock

ROOT=pathlib.Path(__file__).resolve().parents[1]
SCI=ROOT/'research_branches/typed_structure_early_cn_expansion_review/experiments/typed_structure_early_cn_expansion_v1'
sys.path[:0]=[str(ROOT/'experiments/o_cn_matched_execute_v1'),str(SCI/'package'),str(SCI)]
import numpy as np
import torch
import integration
from boundary_transactions import TrainingBoundary,SamplingBoundary,rng_capture,rng_restore
from ema import ParameterEMA,cpu_copy
from model_variants import StructuralDenoiser
from probe import equal,input_hash
from hsgd.bw_preflight import common
from hsgd.computational import FrozenTypedContext
from hsgd.kernel import MarginalKernel
from hsgd.runtime import reverse_step
from observers import RecordedGenerator

SOURCE='175373c83202b6c8b80aa570e0092c4652031ce6'
ORIGINAL='2f2cb300c1c7efc3dd6a28d462d5cb7584e88919'

def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def reset():
    random.seed(4381);np.random.seed(4381);torch.manual_seed(4381);torch.cuda.manual_seed_all(4381)
    return np.random.default_rng(4389)

def fixture():
    ty=np.repeat(np.arange(4),3);schema=[(a,b) for a in range(4) for b in range(a,4)]
    edges=np.array([(3*a,3*a+1 if a==b else 3*b) for a,b in schema],dtype=np.int64).T
    case=common.make_case(ty,edges,tuple(schema))
    cfg=json.loads((SCI/'package/config.json').read_text());cfg.update(seed=912,inputs_per_update=1,rounds=1)
    ks=[8,8,8,6];h=np.random.default_rng(903).uniform(.1,1.,(sum(ks),630));h/=np.linalg.norm(h,axis=1,keepdims=True)
    return case,cfg,dict(basis=h,offsets=np.r_[0,np.cumsum(ks)])

def objects(arm):
    rng=reset();case,cfg,basis=fixture()
    model=StructuralDenoiser(case,cfg,arm,basis if arm=='E' else None).cuda()
    opt=torch.optim.AdamW(model.parameters(),**{k:v for k,v in cfg['optimizer'].items() if k!='type'})
    ema=ParameterEMA(model);tx=TrainingBoundary(model,opt,ema,rng,'cuda',0)
    return case,cfg,model,opt,ema,rng,tx

def payload(tx):return tx.checkpoint(dict(engineering_only=True,source=SOURCE))

def compare(a,b):
    return {k:equal(a[k],b[k]) for k in ('model','optimizer','ema','rng','updates')}

def difference(a,b):
    if torch.is_tensor(a):return float((a.double()-b.double()).abs().max()) if a.numel() else 0.
    if isinstance(a,dict):return max((difference(v,b[k]) for k,v in a.items()),default=0.)
    if isinstance(a,(list,tuple)):return max((difference(x,y) for x,y in zip(a,b)),default=0.)
    return 0.

def probe(model,case,cfg,fail=False):
    inp=common.train_input(case,1000,1,cfg['seed'],9001,np.random.default_rng(937),force_t=500)
    old=model.training;model.eval();token='gpu-validation'
    ctx=FrozenTypedContext(model,case['space'].types,inp['edges'],inp['edge_families'],token)
    try:
        with torch.no_grad():_,_,p=ctx.predict(inp['query'],inp['f'],.5,token,details=True)
        # Explicitly exercise every global RNG inventory, including CUDA.
        random.random();np.random.random();torch.rand(3);torch.rand(3,device='cuda')
        if fail:raise ValueError('injected ordinary validation failure')
        return float(p.sum())
    finally:ctx.close();model.train(old)

def draw_all(rng):
    rng.random(7);random.random();np.random.random();torch.rand(7);torch.rand(7,device='cuda')

def restore(z,model,opt,ema,rng):
    # AdamW.load_state_dict may retain CPU step-tensor references. Each replay
    # must load its own snapshot copy, as an independent torch.load would do.
    z=cpu_copy(z)
    u=integration.restore_payload(z,model,opt,ema,z['binding'],lambda r:rng_restore(r,rng,'cuda'))
    return TrainingBoundary(model,opt,ema,rng,'cuda',u)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',required=True);a=ap.parse_args()
    out=pathlib.Path(a.output);out.mkdir(parents=True,exist_ok=True);start=time.time()
    def save(name,obj):(out/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')
    result=dict(status='running',source=SOURCE,records=[],benchmark=[],failures=[],GPU_used=True,
                scope='Real O/A/D/E eight-layer layouts; synthetic 12-node/66-slot graph; synthetic E basis; engineering updates and one-step reverse checks only')
    def record(name,**kw):result['records'].append(dict(name=name,**kw));save('results.json',result)
    try:
        # A later source-only test commit may contain this harness. Bind every
        # production Python file to 175373c instead of requiring the test commit
        # itself to equal the scientific commit.
        from check_runtime_bindings import verify_sources
        verify_sources()
        assert torch.cuda.is_available() and torch.cuda.device_count()==1
        torch.set_num_threads(1);torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True,warn_only=False)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        env=common.environment('cuda');env.update(CUBLAS_WORKSPACE_CONFIG=os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
            CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES'),host=os.uname().nodename,strict_determinism=torch.are_deterministic_algorithms_enabled())
        save('environment.json',env)
        # Check real execution modules, without invoking their main()/approval.
        modules={}
        for name in ('o_cn_matched_execute_v1','late_final_single_input_v1','d_late_ard_runtime_v1'):
            f=ROOT/'experiments'/name/('generation_worker.py' if name=='d_late_ard_runtime_v1' else 'worker_run.py')
            modules[name]=dict(path=str(f),sha256=hashlib.sha256(f.read_bytes()).hexdigest())
        for m in (integration,sys.modules['boundary_transactions'],sys.modules['model_variants']):
            f=pathlib.Path(m.__file__);modules[m.__name__]=dict(path=str(f),sha256=hashlib.sha256(f.read_bytes()).hexdigest())
        save('actual_sources.json',modules)
        update=load(ROOT/'experiments/o_cn_matched_execute_v1/single_input.py','cuda_update')
        rel='experiments/o_cn_matched_execute_v1/single_input.py'
        legacy=out/'temporary/legacy_single_input.py';legacy.parent.mkdir(exist_ok=True)
        legacy.write_bytes(subprocess.check_output(['git','show',ORIGINAL+':'+rel],cwd=ROOT))
        oldupdate=load(legacy,'legacy_cuda_update')
        oldpath=out/'temporary/legacy_integration.py'
        rel=str((SCI/'integration.py').relative_to(ROOT));oldpath.write_bytes(subprocess.check_output(['git','show',ORIGINAL+':'+rel],cwd=ROOT))
        oldintegration=load(oldpath,'legacy_cuda_integration')

        for arm in ('O','A','D','E'):
            case,cfg,model,opt,ema,rng,tx=objects(arm)
            save('synthetic_config.json',dict(config=cfg,nodes=12,legal_slots=66,synthetic_dictionary=True,role_K=[8,8,8,6]))
            initial=payload(tx)
            def step(owner,position):
                with owner.step(position):
                    inp=update.one_input(case,cfg,position,rng)
                    rec=update.update(model,opt,case,inp,1000,'cuda',ema,owner.completed,transaction=owner)
                return input_hash([inp]),rec
            ih,_=step(tx,1);after1=payload(tx)
            tx=restore(initial,model,opt,ema,rng)
            inp=oldupdate.one_input(case,cfg,1,rng)
            oldupdate.update(model,opt,case,inp,1000,'cuda',ema,0)
            old=integration.checkpoint_payload(model,opt,ema,1,rng_capture(rng,'cuda'),initial['binding'])
            same=compare(after1,old);assert all(same.values()),same
            record('normal_training_matches_pre_fix',arm=arm,exact=same,input_exact=ih==input_hash([inp]),parameters=sum(p.numel() for p in model.parameters()))
            tx=restore(after1,model,opt,ema,rng)
            expected_input,_=step(tx,2);expected=payload(tx)
            # Pre-optimizer failure at three actual sites; only the uncommitted work is retried.
            for fault in ('after_input','forward','gradient'):
                tx=restore(after1,model,opt,ema,rng);before=payload(tx);hook=None
                try:
                    with tx.step(2):
                        inp=update.one_input(case,cfg,2,rng)
                        if fault=='after_input':draw_all(rng);raise ValueError('injected after input')
                        if fault=='forward':hook=model.joint_head.register_forward_pre_hook(lambda *a:(_ for _ in ()).throw(ValueError('injected forward')))
                        if fault=='gradient':hook=model.joint_head[2].bias.register_hook(lambda g:g*float('nan'))
                        update.update(model,opt,case,inp,1000,'cuda',ema,1,transaction=tx)
                except (ValueError,FloatingPointError) as exc:message=repr(exc)
                else:raise AssertionError('fault not injected')
                finally:
                    if hook is not None:hook.remove()
                rollback=compare(before,payload(tx));assert all(rollback.values()),rollback
                # Genuine disk save/reload before retry; full CUDA RNG is restored.
                file=out/'temporary/resume.pt';torch.save(payload(tx),file)
                tx=restore(torch.load(file,map_location='cpu'),model,opt,ema,rng)
                retried,_=step(tx,2);replay=compare(expected,payload(tx));assert all(replay.values()) and retried==expected_input,(arm,fault,replay)
                record('pre_update_failure_resume',arm=arm,fault=fault,injected_error=message,rollback=rollback,replay=replay,input_exact=True)
            tx=restore(after1,model,opt,ema,rng)
            for ordinary in (False,True):
                before=payload(tx)
                try:
                    with tx.validation():
                        probe(model,case,cfg)
                        integration.ema_validation(lambda m:probe(m,case,cfg,ordinary),model,ema,boundary=tx)
                except ValueError:
                    assert ordinary
                same=compare(before,payload(tx));assert all(same.values()),same
                record('validation_full_state_isolation',arm=arm,ordinary_failure=ordinary,exact=same)

            if arm in ('A','E'):
                # One frozen reverse step t500->499, no complete generated graph.
                model.eval();kernel=MarginalKernel(case['prior'],common.schedule(1000))
                initial_state=(np.random.default_rng(99).random(66)<np.repeat(case['prior'],case['space'].counts)).astype(np.uint8)
                srng=RecordedGenerator(902,66);boundary=SamplingBoundary(initial_state,500,srng,'cuda');saved=boundary.checkpoint({})
                def reverse(s,t):return reverse_step(model,case,s,kernel,t,1,902,srng,'cuda',trace=False)
                def check(s,rec):assert rec['exact_coverage'] and rec['blocks']==1 and rec['covered_slots']==66
                baseline_state,rec=reverse(initial_state,500);check(baseline_state,rec);baseline_rng=rng_capture(srng,'cuda')
                boundary=SamplingBoundary(initial_state,500,srng,'cuda');boundary.restore(saved,{})
                boundary.advance(reverse,check);normal=boundary.checkpoint({})
                assert np.array_equal(normal['state'],baseline_state) and equal(normal['rng'],baseline_rng)
                boundary=SamplingBoundary(initial_state,500,srng,'cuda');boundary.restore(saved,{})
                def fail_check(s,rec):
                    check(s,rec);draw_all(srng);raise ValueError('after CUDA prediction/uniform, before graph commit')
                try:boundary.advance(reverse,fail_check)
                except ValueError:pass
                rolled=boundary.checkpoint({});assert equal(saved,rolled) and equal(saved['rng'],rng_capture(srng,'cuda'))
                file=out/'temporary/chain.pt';torch.save(rolled,file);boundary.restore(torch.load(file,map_location='cpu'),{})
                boundary.advance(reverse,check);assert equal(normal,boundary.checkpoint({}))
                record('sampling_commit_failure_resume',arm=arm,input_t=500,committed_t=499,graph_exact=True,full_RNG_exact=True,normal_matches_unwrapped=True,complete_chain=False)

                # Paired microbenchmark, same saved input and same raw/AdamW/EMA/RNG.
                for kind in ('training','validation','sampling'):
                    measurements={'baseline':[],'transaction':[]}
                    for repeat in range(5):  # first pair warmup, four retained
                        for version in ('baseline','transaction') if repeat%2==0 else ('transaction','baseline'):
                            tx=restore(after1,model,opt,ema,rng);model.train(kind=='training')
                            if kind=='sampling':
                                boundary=SamplingBoundary(initial_state,500,srng,'cuda');boundary.restore(saved,{})
                            gc.collect();torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();allocated=torch.cuda.memory_allocated();tick=time.perf_counter()
                            if kind=='training':
                                if version=='transaction':step(tx,2)
                                else:
                                    inp=oldupdate.one_input(case,cfg,2,rng);oldupdate.update(model,opt,case,inp,1000,'cuda',ema,1)
                            elif kind=='validation':
                                if version=='transaction':
                                    with tx.validation():probe(model,case,cfg);integration.ema_validation(lambda m:probe(m,case,cfg),model,ema,boundary=tx)
                                else:probe(model,case,cfg);oldintegration.ema_validation(lambda m:probe(m,case,cfg),model,ema)
                            elif version=='transaction':boundary.advance(reverse,check)
                            else:
                                v,r=reverse(initial_state,500);check(v,r)
                            torch.cuda.synchronize();row=dict(repeat=repeat,seconds=time.perf_counter()-tick,peak_allocated_MiB=torch.cuda.max_memory_allocated()/2**20,incremental_peak_MiB=(torch.cuda.max_memory_allocated()-allocated)/2**20,peak_reserved_MiB=torch.cuda.max_memory_reserved()/2**20,warmup=repeat==0)
                            measurements[version].append(row)
                    med={v:statistics.median(x['seconds'] for x in rows if not x['warmup']) for v,rows in measurements.items()}
                    result['benchmark'].append(dict(arm=arm,component=kind,measurements=measurements,median_seconds=med,delta_seconds=med['transaction']-med['baseline'],ratio=med['transaction']/med['baseline'],scope='four paired repeats after one warmup pair; synchronized; no checkpoint disk IO in timed region; validation includes raw+EMA probes'))
                    save('results.json',result)

            # A partial raw restore must poison both owner and direct serializer.
            tx=restore(after1,model,opt,ema,rng);original=model.load_state_dict;calls=[0]
            def partial(sd,strict=True):
                calls[0]+=1
                if calls[0]==2:
                    k='joint_head.2.bias';torch.nn.Module.load_state_dict(model,{k:sd[k]},strict=False)
                    raise RuntimeError('injected partial raw restore')
                return original(sd,strict=strict)
            try:
                with mock.patch.object(model,'load_state_dict',partial):
                    with tx.validation():integration.ema_validation(lambda m:probe(m,case,cfg),model,ema,boundary=tx)
            except RuntimeError:pass
            else:raise AssertionError('partial restore did not fail')
            assert not tx.resumable and not ema.at_update_boundary
            rejected=[]
            for fn in (lambda:payload(tx),lambda:integration.checkpoint_payload(model,opt,ema,1,rng_capture(rng,'cuda'),{})):
                try:fn()
                except RuntimeError:rejected.append(True)
                else:raise AssertionError('mixed state certified')
            record('partial_ema_raw_restore_refused',arm=arm,owner_and_direct_serializer_rejected=rejected,recovery=tx.last_failure)
            del model,opt,ema,tx,initial,after1,expected,old;gc.collect();torch.cuda.empty_cache()
        result['status']='passed'
    except BaseException as exc:
        result['status']='failed';result['failures'].append(dict(error=repr(exc),traceback=traceback.format_exc()))
        raise
    finally:
        result.update(seconds=time.time()-start,finished=time.strftime('%Y-%m-%dT%H:%M:%S%z'),CUDA_initialized=torch.cuda.is_initialized())
        if torch.cuda.is_initialized():torch.cuda.synchronize()
        save('results.json',result)

if __name__=='__main__':main()
