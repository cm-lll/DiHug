"""Bounded engineering measurements ONLY. No formal train/full-chain entry."""
import json
import hashlib
import resource
import time
from pathlib import Path
import numpy as np
import torch
from hsgd.bw_preflight import common as c
from hsgd.fusion_checks import fixture,make,mechanism,diagnostics
from hsgd.kernel import MarginalKernel
from hsgd.runtime import reverse_step


def inventory(case):
    degree=np.bincount(case['clean'].ravel(),minlength=len(case['space'].types))
    return dict(nodes=len(degree),physical_edges=case['clean'].shape[1],type_counts=np.bincount(case['space'].types).tolist(),
        family_edges=case['counts'].tolist(),all_slots=case['space'].total,
        active_slots=int(case['space'].counts[case['active']].sum()),Xt_reference_wedges=int((degree*(degree-1)//2).sum()))


def emit(out,name,data):
    c.json_save(out/name,data)
    print(json.dumps(dict(event=name,status=data.get('status'),arm=data.get('arm')),ensure_ascii=False),flush=True)


def run(kind,out,device):
    start,cpu=time.perf_counter(),time.process_time();out=Path(out);out.mkdir(parents=True,exist_ok=False)
    c.verify_manifest();torch.set_num_threads(1);torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(False,warn_only=False)
    env=c.environment(device);cfg=c.config()
    emit(out,'environment.json',dict(status='recorded',environment=env,code=c.code_hashes(),config=cfg,
        scope='engineering only; no formal training/full-chain generation',initialization_included=True))
    if kind=='plan':
        case=c.imdb_case(c.PACKAGE/'data/IMDB_movie_reference.npz',cfg)
        plan=c.plan_for(case,[1603,30,61],20)
        result=dict(status='planned_only',reference=inventory(case),query=c.query_inventory(case,plan),
            future_device_check=dict(device='one BW, user manual confirmation',outer_seconds=480,
                max_Xt_wedges=2000000,max_matched_paths_per_block=2000000,
                one_update_each_arm=True,one_frozen_20_block_engineering_step_each_arm=True,
                diagnostics_t=[1,30,1000],full_chains=0,old_CN_weights_imported=False),
            GPU_use=0,model_forwards=0,training_updates=0)
        emit(out,'summary.json',result);return result
    full=kind=='device-check'
    if full and (device!='cuda' or env.get('device_name')!='BW' or not env.get('hip')):
        raise RuntimeError('full-size device check requires confirmed BW HIP device; no CPU/NVIDIA fallback')
    case=c.imdb_case(c.PACKAGE/'data/IMDB_movie_reference.npz',cfg) if full else fixture()[0]
    rounds=20 if full else 3
    inputs=[c.train_input(case,1000,rounds,2026101407,j,np.random.default_rng(160300+j),force_t=t)
            for j,t in enumerate([1,10,30,500])]
    validations=[c.train_input(case,1000,rounds,2026101407,100+j,np.random.default_rng(160400+j),force_t=t)
            for j,t in enumerate([1,30,1000])]
    mech=None if full else mechanism()
    if mech is not None:emit(out,'mechanism.json',mech)
    arms=[]
    for arm,enabled in [('A_matched_no_CN',False),('B_joint_CN',True)]:
        c.sync(device);tick=time.perf_counter();model,model_cfg=make(case,enabled,device,full=full);c.sync(device)
        init=time.perf_counter()-tick
        if device=='cuda':torch.cuda.reset_peak_memory_stats()
        opt=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=1e-12,amsgrad=False)
        history=[];selected=inputs[2:3] if full else inputs
        for j,inp in enumerate(selected):
            print(json.dumps(dict(event='engineering_update_start',arm=arm,index=j,t=inp['t'])),flush=True)
            diag={};row=c.update(model,opt,case,inp,1000,device,diagnostics=diag)
            history.append(row)
            emit(out,arm+'_update_%02d.json'%j,dict(status='passed',arm=arm,**row))
        reports=[diagnostics(model,case,inp,1000) for inp in validations]
        for i,r in enumerate(reports):emit(out,arm+'_diagnostic_%02d.json'%i,dict(status='measured',arm=arm,**r))
        frozen=inputs[2];fam,local=case['space'].encode(frozen['edges']);state=np.zeros(case['space'].total,dtype=np.uint8)
        state[case['space'].offsets[fam]+local]=1
        degree=np.bincount(frozen['edges'].ravel(),minlength=len(case['space'].types));wedges=int((degree*(degree-1)//2).sum())
        print(json.dumps(dict(event='frozen_engineering_step_start',arm=arm,time=30,rounds=rounds,edges=int(state.sum()),wedges=wedges)),flush=True)
        model.eval();result,step=reverse_step(model,case,state,MarginalKernel(case['prior'],c.schedule(1000)),30,rounds,2026101407,
            np.random.default_rng(160530),device,trace=True)
        arrays=step.pop('trace')
        assert np.array_equal(result,arrays['state']) and step['exact_coverage']
        record=dict(status='passed',arm=arm,dimensions=c.dimensions(model),config=model_cfg,
            initialization_seconds=init,update_records=history,post_update_diagnostics=reports,
            frozen_engineering_step=step,frozen_input_edges=int(state.sum()),frozen_input_wedges=wedges,
            engineering_state_sha256=hashlib.sha256(np.packbits(result).tobytes()).hexdigest(),
            timing_caveat='full graph engineering state, detailed trace overhead included; not trained quality or steady formal throughput',
            resources=c.memory(device))
        if len(history)>1:
            stable=[r['synchronized_update_seconds'] for r in history[1:]]
            record['post_first_update_median']=float(np.median(stable));record['post_first_update_P90']=float(np.quantile(stable,.9))
            record['post_first_update_count']=len(stable)
        emit(out,arm+'.json',record);arms.append(record)
        del model,opt,diag,arrays,result
        if device=='cuda':torch.cuda.empty_cache()
    result=dict(status='passed_engineering_only',environment=env,reference=inventory(case),arms=arms,
        mechanism=mech,wall_seconds=time.perf_counter()-start,cpu_seconds=time.process_time()-cpu,
        resources=c.memory(device),formal_training=0,full_scientific_chains=0,new_scientific_graphs=0,
        engineering_optimizer_updates=2 if full else 8,BW_measured=bool(full),
        quality_conclusion='not tested; random weights with finite engineering updates only',
        four_arm_role_guidance='separate t1 role-degree module unchanged; not applied in these fusion checks')
    emit(out,'summary.json',result);return result
