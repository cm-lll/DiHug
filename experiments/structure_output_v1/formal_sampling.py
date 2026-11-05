"""Canonical paired R1 generation using the frozen reverse_step unchanged."""
import copy,hashlib,json,pathlib,time
import numpy as np
import torch
from .formal_runtime import SCI,read,save,sha,now,g
def tensor_sha(v): return g.digest(v)
from probe import rng_get
from worker import atomic_checkpoint
from hsgd.bw_preflight import common
from hsgd.kernel import MarginalKernel
from hsgd.runtime import reverse_step
from evaluation import evaluate
from observers import RecordedGenerator,RecordedPlan,digest
from boundary_transactions import SamplingBoundary
def authenticate_readout(modelrow,case,cfg):
    path=pathlib.Path(modelrow['readout_file']);receipt=read(path.with_suffix('.sha256.json'));actual=sha(path)
    if receipt['sha256']!=actual or (modelrow.get('readout_sha256') and actual!=modelrow['readout_sha256']):raise ValueError('readout SHA256 mismatch')
    z=torch.load(path,map_location='cpu');b=z['binding'];seed=modelrow['train_seed'];configuration=modelrow['configuration']
    if z['updates']!=3000 or z['train_seed']!=seed or z['readout']!='ema_0.99' or z['decay']!=.99:raise ValueError('only correct final3000 EMA.99')
    if b['train_seed']!=seed or b['dataset']!=modelrow['dataset'] or b['config']!=cfg:raise ValueError('readout dataset/seed/config binding')
    if configuration!='STRUCTURE_OUTPUT_late' or b.get('architecture_version')!=g.StructureInputA.architecture_version or b.get('configuration')!=configuration:raise ValueError('candidate architecture binding')
    if set(k for k in z['model'] if k.startswith('structure_'))!={'structure_node_projection.weight','structure_pair_projection.weight'}:raise ValueError('candidate extra state identity')
    from .role_loss import VERSION,LAMBDA
    if b.get('output_objective',{}).get('version')!=VERSION or b['output_objective']['weight']!=LAMBDA:raise ValueError('output objective identity')
    expected_head=776
    if z['model']['joint_head.0.weight'].shape!=(128,expected_head) or ('cn_edge_projection.weight' in z['model']) or ('role_basis' in z['model'])!=configuration.startswith('E'):raise ValueError('configuration state-layout identity')
    final=path.parent/'final.sha256.json'
    if not final.exists() or read(final)['sha256']!=z['checkpoint_sha256']:raise ValueError('final checkpoint receipt binding')
    if modelrow.get('parent_approval_sha256') and b.get('approval_sha256')!=modelrow['parent_approval_sha256']:raise ValueError('protected parent approval binding')
    return z,actual
def sample_one(modelrow,ss,folder,case,cfg,basis,out,check,event,resume_chain=None):
    folder=pathlib.Path(folder);folder.mkdir(parents=True,exist_ok=False);payload,readout_sha=authenticate_readout(modelrow,case,cfg)
    model=g.StructureInputA(case,dict(cfg,seed=modelrow['train_seed']));model.load_state_dict(payload['model'],strict=True)
    expected={k:tensor_sha(v) for k,v in payload['model'].items()};model=model.cuda().eval()
    if any(tensor_sha(v)!=expected[k] for k,v in model.state_dict().items()):raise ValueError('actual loaded readout content')
    del payload
    contract=read(pathlib.Path(out)/'prepared/sampling_contracts.json')[modelrow['dataset']+'_'+str(ss)];contract_by_t={x['t']:x for x in contract['steps']}
    family=np.repeat(np.arange(case['n_families']),case['space'].counts);rng=RecordedGenerator(ss,case['space'].total);state=(rng.random(case['space'].total)<case['prior'][family]).astype(np.uint8)
    initial_rng=copy.deepcopy(rng.bit_generator.state);ih=digest(state)
    if ih!=contract['initial_state_sha256'] or rng.last_sha!=contract['initial_uniform_sha256']:raise ValueError('canonical sampling initial pairing')
    identity=dict(dataset=modelrow['dataset'],train_seed=modelrow['train_seed'],configuration=modelrow['configuration'],model_id=modelrow['id'],updates=3000,readout='EMA0.99',sample_seed=ss,R=1,T=1000,model_sha256=readout_sha,initial_state_sha256=ih,initial_rng=initial_rng)
    save(folder/'identity.json',identity);next_t=1000;began=time.perf_counter();plans=[];original_plan=common.plan_for
    def planned(*args,**kwargs):
        p=RecordedPlan(original_plan(*args,**kwargs));plans.append(p);return p
    kernel=MarginalKernel(case['prior'],common.schedule(1000))
    chain=SamplingBoundary(state,next_t,rng,next(model.parameters()).device.type)
    resumed=None
    if resume_chain is not None:
        parent=pathlib.Path(resume_chain);parent_sha=sha(parent)
        if read(parent.with_suffix('.sha256.json'))['sha256']!=parent_sha:raise ValueError('resume chain SHA256 mismatch')
        restored=torch.load(parent,map_location='cpu');proposed=restored['state']
        if proposed.shape!=state.shape or (proposed[case['prior'][family]==0]!=0).any() or (proposed[case['prior'][family]==1]!=1).any():raise ValueError('resume chain violates fixed support')
        chain.restore(restored,identity)
        state,next_t=chain.state,chain.next_t
        resumed=dict(file=str(parent),sha256=parent_sha,next_t=next_t)
    start_t=chain.next_t
    def snapshot(name):atomic_checkpoint(folder/name,chain.checkpoint(identity))
    common.plan_for=planned
    try:
        snapshot('Xt1000.pt' if resumed is None else 'resumed_t%04d.pt'%chain.next_t)
        for t in range(chain.next_t,0,-1):
            check();plans.clear();calls=rng.calls
            def validate(candidate,record):
                target=contract_by_t[t]
                if len(plans)!=1 or plans[0].blocks!=1 or plans[0].h.hexdigest()!=target['query_plan_sha256'] or rng.calls!=calls+1 or rng.last_sha!=target['uniform_sha256']:raise ValueError('canonical per-step plan/draw pairing')
                if record['blocks']!=1 or record['covered_slots']!=int(np.repeat(case['active'],case['space'].counts).sum()) or not record['exact_coverage']:raise ValueError('complete R1 query coverage')
            rec=chain.advance(lambda frozen,at:reverse_step(model,case,frozen,kernel,at,1,ss,rng,'cuda',check_time=check,trace=False),validate)
            state,next_t=chain.state,chain.next_t
            rec.update(uniform_sha256=rng.last_sha,query_plan_sha256=plans[0].h.hexdigest(),canonical_pairing=True)
            with (folder/'steps.jsonl').open('a') as f:f.write(json.dumps(rec)+'\n')
            if t%100==0 or t==1:snapshot('complete_t%04d.pt'%next_t)
            if t in [1000,750,500,250,100,50,10,1]:
                family_counts=np.bincount(family,weights=state,minlength=case['n_families']).astype(int)
                with (folder/'trajectory.jsonl').open('a') as f:f.write(json.dumps(dict(input_t=t,committed_t=next_t,total_edges=int(state.sum()),family_edges=family_counts.tolist()))+'\n')
            if t%25==0:event(status='sampling',sample_seed=ss,next_t=next_t)
        check();edges,fam=common.state_edges(case,state);np.savez_compressed(folder/'graph.npz',types=case['space'].types,edges=edges,families=fam,schema=np.array(case['space'].pairs))
        save(folder/'metrics.json',evaluate(case,edges,cfg['datasets'][modelrow['dataset']]['type_names'],check))
        result=dict(identity,status='complete',steps=1000,executed_steps=start_t,resumed_from=resumed,graph_sha256=sha(folder/'graph.npz'),metrics_sha256=sha(folder/'metrics.json'),seconds=time.perf_counter()-began,finished=now(),canonical_all1000_plans_and_draws=True)
        save(folder/'binding.json',result);save(folder/'summary.json',result);return result
    except BaseException as exc:
        state,next_t=chain.state,chain.next_t
        try:snapshot('stopped_chain.pt')
        except BaseException as e:save(folder/'save_failure.json',dict(error=repr(e)))
        save(folder/'failure.json',dict(identity,next_t=next_t,error=repr(exc),at=now(),exact_resume_allowed=chain.resumable,recovery=chain.last_failure));raise
    finally:
        common.plan_for=original_plan;del model;torch.cuda.empty_cache()
