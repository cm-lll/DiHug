"""Bounded engineering only: two datasets, fourteen temporary updates, no chain.

Reuses real single-input update, EMA validation and transactional reverse_step.
Does not call/modify any formal launcher or accept a training-length argument.
Private asset root and output must be passed explicitly; no default locations.
"""
import argparse
import copy
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
SCI = ROOT/'research_branches/typed_structure_early_cn_expansion_review/experiments/typed_structure_early_cn_expansion_v1'
sys.path[:0] = [str(ROOT),str(ROOT/'experiments/o_cn_matched_execute_v1'),str(SCI/'package'),str(SCI)]
import numpy as np
import torch
from hsgd.bw_preflight import common
from hsgd.computational import FrozenTypedContext
from hsgd.runtime import reverse_step
from hsgd.kernel import MarginalKernel
from model_variants import StructuralDenoiser
from single_input import one_input, update
from ema import ParameterEMA
from boundary_transactions import TrainingBoundary, SamplingBoundary, rng_capture, rng_restore
from integration import restore_payload, ema_validation
from experiments.structure_input_v1.model import StructureInputA
from experiments.structure_input_v1 import model as candidate_module


def digest(value):
    h=hashlib.sha256()
    def visit(v):
        if torch.is_tensor(v):
            h.update(str((v.shape,v.dtype)).encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
        elif isinstance(v,np.ndarray):
            h.update(str((v.shape,v.dtype)).encode());h.update(v.tobytes())
        elif isinstance(v,dict):
            for k in sorted(v):h.update(str(k).encode());visit(v[k])
        elif isinstance(v,(tuple,list)):
            for item in v:visit(item)
        else:h.update(repr(v).encode())
    visit(value);return h.hexdigest()


def save(path,obj):
    path.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')


def compare_state(left,right,atol,rtol):
    """Integer/RNG metadata exact; float continuation uses existing cfg tolerance."""
    rows=[]
    def visit(a,b,path):
        if torch.is_tensor(a):
            if a.shape!=b.shape or a.dtype!=b.dtype:raise AssertionError(path+' shape/dtype')
            exact=torch.equal(a,b)
            ok=exact or (a.is_floating_point() and torch.allclose(a,b,atol=atol,rtol=rtol))
            rows.append(dict(path=path,exact=exact,within_tolerance=bool(ok),
                             max_abs=float((a-b).abs().max()) if a.numel() else 0.))
        elif isinstance(a,np.ndarray):
            assert np.array_equal(a,b),path
        elif isinstance(a,dict):
            assert a.keys()==b.keys(),path
            for k in a:visit(a[k],b[k],path+'/'+str(k))
        elif isinstance(a,(list,tuple)):
            assert len(a)==len(b),path
            for k,(x,y) in enumerate(zip(a,b)):visit(x,y,path+'/'+str(k))
        else:assert a==b,path
    for key in ('model','optimizer','ema','rng','updates'):visit(left[key],right[key],key)
    return dict(bitwise_equal=all(x['exact'] for x in rows),
                within_existing_tolerance=all(x['within_tolerance'] for x in rows),
                atol=atol,rtol=rtol,largest_differences=sorted(rows,key=lambda x:x['max_abs'],reverse=True)[:10])


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--assets-root',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True);ap.add_argument('--stop-epoch',type=float,required=True)
    a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    start=time.time();deadline=min(a.stop_epoch,start+14*60)
    report=dict(status='running',start_epoch=start,hard_stop_epoch=deadline,
                GPU_used=True,formal_training=False,complete_generated_graphs=0,records=[])
    def check():
        if time.time()>=deadline:raise TimeoutError('engineering budget expired')
    def emit(row):
        report['records'].append(row);save(a.output/'result.json',report)
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.set_float32_matmul_precision('highest')
    if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; no fallback')
    report['environment']=common.environment('cuda')
    report['source_hashes']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted((ROOT/'experiments/structure_input_v1').glob('*.py'))}
    report['production_entry_note']='isolated candidate construction; original production launcher remains A/O only'
    registry=json.loads((ROOT/'review/structure_input_v1/ASSET_BINDINGS.json').read_text())['assets']
    tasks=json.loads((a.assets_root/'results/cn_multiseed_20261104T042546/tasks.json').read_text())
    extract_original=candidate_module.extract
    phase={};expected_edges=None
    def observed_extract(types,edges,families,pairs,n_types):
        begin=time.perf_counter();value=extract_original(types,edges,families,pairs,n_types)
        secs=time.perf_counter()-begin
        if expected_edges is not None and not np.array_equal(np.asarray(edges,dtype=np.int64).reshape(-1,2).T,expected_edges):
            raise AssertionError('feature extractor did not receive the expected frozen Xt')
        emit(dict(kind='feature',**phase,seconds=secs,nodes=len(types),physical_edges=len(edges),
                  frozen_Xt_verified=expected_edges is not None,
                  components=len(value['component_sizes']),largest_component=max(value['component_sizes']),
                  triangles=sum(map(sum,value['triangle_counts']))//3,
                  component_fraction_std=float(np.std(np.asarray(value['node_features'])[:,-5]))))
        return value
    candidate_module.extract=observed_extract
    try:
        for dataset,seed in [('IMDB',2026110411),('PubMed',2026110421)]:
            check();phase=dict(dataset=dataset,stage='load')
            row=next(x for x in tasks if x['dataset']==dataset and x['train_seed']==seed and x['arm']=='A')
            cfg=row['config'];entry=cfg['datasets'][dataset]
            data_path=a.assets_root/SCI.relative_to(ROOT)/'package'/entry['file']
            assert hashlib.sha256(data_path.read_bytes()).hexdigest()==entry['sha256']
            with np.load(data_path,allow_pickle=False) as z:
                case=common.make_case(z['types'],z['edges'].T,tuple(map(tuple,z['schema'])))
                assert np.array_equal(case['space'].encode(z['edges'].T)[0]+1,z['relations'])
            asset=next(x for x in registry if x['kind']=='checkpoint_bytes_only' and x['dataset']==dataset and x['seed']==seed and x['update']==0)
            parent_path=a.assets_root/'results'/asset['artifact_id']
            assert hashlib.sha256(parent_path.read_bytes()).hexdigest()==asset['actual_sha256']
            parent=torch.load(parent_path,map_location='cpu')
            assert parent['updates']==0 and not parent['optimizer']['state'] and parent['ema']['updates']==0
            assert parent['binding']['dataset']==dataset and parent['binding']['train_seed']==seed
            assert parent['binding']['config']==cfg and parent['binding']['arm']=='A'
            common.seed_all(seed)
            ordinary=StructuralDenoiser(case,cfg,'A');ordinary.load_state_dict(parent['model'],strict=True)
            candidate=StructureInputA.from_initial_A(ordinary,case,cfg,completed_updates=0)
            base_state=copy.deepcopy(ordinary.state_dict());cand_state=copy.deepcopy(candidate.state_dict())
            emit(dict(kind='binding',dataset=dataset,train_seed=seed,data_sha256=entry['sha256'],
                      parent_sha256=asset['actual_sha256'],type_names=entry['type_names'],
                      schema=[list(p) for p in case['space'].pairs],slots=case['space'].total,
                      node_count=len(case['space'].types),reference_edges=case['clean'].shape[1]))
            del ordinary,candidate;gc.collect()
            input_hashes={};initial_logits={}
            for arm,state in [('A',base_state),('candidate',cand_state)]:
                check();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
                model=(StructuralDenoiser(case,cfg,'A') if arm=='A' else StructureInputA(case,cfg)).cuda()
                model.load_state_dict(state,strict=True)
                rng=np.random.default_rng(1);rng_restore(parent['rng'],rng,'cuda')
                opt=torch.optim.AdamW(model.parameters(),**{k:v for k,v in cfg['optimizer'].items() if k!='type'})
                ema=ParameterEMA(model);tx=TrainingBoundary(model,opt,ema,rng,'cuda',0)
                binding=dict(engineering=True,dataset=dataset,arm=arm)
                def snapshot():return tx.checkpoint(binding)
                def scientific_hash():
                    p=snapshot();return digest({k:p[k] for k in ('model','optimizer','ema','rng','updates')})
                def perform(position,t):
                    nonlocal expected_edges,phase
                    check();phase=dict(dataset=dataset,arm=arm,stage='update',position=position,t=t)
                    torch.cuda.synchronize();begin=time.perf_counter()
                    with tx.step(position):
                        inp=one_input(case,cfg,position,rng,force_t=t);expected_edges=inp['edges']
                        if position==1:
                            saved=rng_capture(rng,'cuda');model.eval()
                            with torch.no_grad():
                                ctx=FrozenTypedContext(model,case['space'].types,inp['edges'],inp['edge_families'],'parity')
                                try:_,logits,_=ctx.predict(inp['query'],inp['f'],t/1000,'parity',details=True)
                                finally:ctx.close()
                            initial_logits[arm]=logits.cpu();rng_restore(saved,rng,'cuda');model.train()
                            if arm=='candidate':
                                x,y=initial_logits['A'],initial_logits['candidate']
                                tol=cfg['tolerances']
                                emit(dict(kind='zero_initial_parity',dataset=dataset,bitwise_equal=torch.equal(x,y),
                                          max_abs=float((x-y).abs().max()),atol=tol['fp32_atol'],rtol=tol['fp32_rtol']))
                                assert torch.allclose(x,y,atol=tol['fp32_atol'],rtol=tol['fp32_rtol'])
                        result=update(model,opt,case,inp,1000,'cuda',ema,tx.completed,transaction=tx)
                    torch.cuda.synchronize();outer=time.perf_counter()-begin
                    ih=digest({k:inp[k] for k in ['t','ids','f','query','labels','edges','edge_families']})
                    if arm=='A':input_hashes[position]=ih
                    else:assert input_hashes[position]==ih
                    emit(dict(kind='update',**phase,**result,outer_seconds=outer,input_sha256=ih,
                              peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                              peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20))
                    if arm=='candidate':
                        for layer in (model.structure_node_projection,model.structure_pair_projection):
                            assert layer.weight.grad is not None and torch.isfinite(layer.weight.grad).all()
                    return inp
                first=perform(1,25)
                saved_path=a.output/(dataset+'_'+arm+'_engineering_step1.pt')
                tick=time.perf_counter();torch.save(snapshot(),saved_path)
                emit(dict(kind='save',dataset=dataset,arm=arm,seconds=time.perf_counter()-tick))
                perform(2,500);expected=snapshot()
                if arm=='candidate':
                    p=torch.load(saved_path,map_location='cpu')
                    count=restore_payload(p,model,opt,ema,binding,lambda s:rng_restore(s,rng,'cuda'))
                    tx=TrainingBoundary(model,opt,ema,rng,'cuda',count)
                    assert all(digest(snapshot()[k])==digest(p[k]) for k in ('model','optimizer','ema','rng','updates'))
                    before=scientific_hash()
                    try:
                        with tx.step(2):
                            one_input(case,cfg,2,rng,force_t=500)
                            raise RuntimeError('injected after input sampling')
                    except RuntimeError as exc:assert str(exc)=='injected after input sampling'
                    assert scientific_hash()==before and tx.resumable
                    perform(2,500)
                    compared=compare_state(expected,snapshot(),cfg['tolerances']['fp32_atol'],cfg['tolerances']['fp32_rtol'])
                    emit(dict(kind='training_restore',dataset=dataset,exact_load=True,input_failure_rollback=True,**compared))
                    assert compared['within_existing_tolerance']
                    del p
                last=perform(3,950)
                before=scientific_hash();expected_edges=last['edges'];phase=dict(dataset=dataset,arm=arm,stage='EMA')
                def evaluate(m):
                    m.eval()
                    with torch.no_grad():
                        return m.forward_upstream(case['space'].types,last['edges'],last['edge_families'],last['query'],last['f'],.95)
                with tx.validation():ema_validation(evaluate,model,ema,boundary=tx)
                assert scientific_hash()==before
                emit(dict(kind='EMA_validation',dataset=dataset,arm=arm,full_state_unchanged=True))
                # Same candidate factory for training and EMA sampling. Build
                # a separate readout under saved RNG, as no formal worker is run.
                with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                    sample_model=(StructuralDenoiser(case,cfg,'A') if arm=='A' else StructureInputA(case,cfg)).cuda()
                sample_model.load_state_dict(ema.readout_state_dict(model),strict=True);sample_model.eval()
                kernel=MarginalKernel(case['prior'],common.schedule(1000))
                sampling_rng=np.random.default_rng(2026110601)
                for t in (25,500,950):
                    check();phase=dict(dataset=dataset,arm=arm,stage='reverse_step',t=t)
                    inp=one_input(case,cfg,t,np.random.default_rng(2026110600+t),force_t=t)
                    expected_edges=inp['edges'];f,loc=case['space'].encode(expected_edges)
                    frozen=np.zeros(case['space'].total,dtype=np.uint8);frozen[case['space'].offsets[f]+loc]=1
                    chain=SamplingBoundary(frozen,t,sampling_rng,'cuda');start_chain=chain.checkpoint(binding)
                    def next_step(x,at):return reverse_step(sample_model,case,x,kernel,at,1,2026102821,sampling_rng,'cuda',check_time=check)
                    def validate(x,rec):assert rec['exact_coverage'] and rec['blocks']==1
                    rec=chain.advance(next_step,validate)
                    done=chain.checkpoint(binding)
                    emit(dict(kind='reverse_step',**phase,seconds=rec['step_seconds'],
                              coverage=rec['covered_slots'],exact_coverage=rec['exact_coverage'],
                              peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20))
                    if arm=='candidate' and t==500:
                        chain=SamplingBoundary(frozen,t,sampling_rng,'cuda');chain.restore(start_chain,binding)
                        def fail(x,rec):raise RuntimeError('injected after draw and before commit')
                        try:chain.advance(next_step,fail)
                        except RuntimeError as exc:assert str(exc)=='injected after draw and before commit'
                        assert digest(chain.checkpoint(binding))==digest(start_chain)
                        stop_path=a.output/(dataset+'_engineering_stopped_step.pt');torch.save(chain.checkpoint(binding),stop_path)
                        chain=SamplingBoundary(frozen,t,sampling_rng,'cuda');chain.restore(torch.load(stop_path,map_location='cpu'),binding)
                        chain.advance(next_step,validate)
                        assert digest(chain.checkpoint(binding))==digest(done)
                        emit(dict(kind='sampling_restore',dataset=dataset,exact_state_time_RNG=True))
                # Independent sampling must leave raw/AdamW/EMA/counter and
                # the complete TRAINING RNG unchanged, without repairing it.
                assert scientific_hash()==before
                emit(dict(kind='sampling_isolation',dataset=dataset,arm=arm,full_training_state_unchanged=True))
                emit(dict(kind='new_projection',dataset=dataset,arm=arm,
                          node_norm=float(model.structure_node_projection.weight.norm()) if arm=='candidate' else None,
                          pair_norm=float(model.structure_pair_projection.weight.norm()) if arm=='candidate' else None))
                del sample_model,model,opt,ema,tx,first,last;gc.collect();torch.cuda.empty_cache()
            del base_state,cand_state,parent;gc.collect()
        report['status']='passed'
    except BaseException as exc:
        report['status']='failed';report['error']=repr(exc);report['traceback']=traceback.format_exc()
        raise
    finally:
        candidate_module.extract=extract_original
        report['end_epoch']=time.time();report['process_seconds']=time.time()-start
        save(a.output/'result.json',report)


if __name__=='__main__':main()
