"""Adapters only: unchanged capacity model, normal2 update, kernel and sampler."""
import hashlib,json,os,sys,time
from pathlib import Path
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
HERE=Path(__file__).resolve().parent;EXP=HERE.parent
sys.path.insert(0,str(EXP/'role_robust_training_v1'))
import core as base
base.setup()
import numpy as np
import torch
from device_eval import validate_device
from hsgd.computational import FrozenTypedContext
from role_condition.metrics import graph
from role_condition.panel import nx_graph
sys.path.insert(0,str(EXP/'role_unseen_target_validation_v1'))
from execute import chain as original_chain,verify_inputs,target_objects,metadata_case
from analyse import independent_graph,features,stat as old_stat,target_errors,NAMES

RESULT=Path('/data2/lyh/gnn_project/DiHug/results/role_coverage_20261021_v1')
OLD=Path('/data2/lyh/gnn_project/DiHug/results/role_robust_training_6000_20261021_v1/learning_run01')
PANEL=Path('/data2/lyh/gnn_project/DiHug/results/role_late_paired_20261020_run01')
INITIAL=Path('/data2/lyh/gnn_project/DiHug/results/role_specialization_20261020_v1/learning_run01')

def read(p):return json.loads(Path(p).read_text())
def save(p,x):
    p=Path(p);q=p.with_name(p.name+'.tmp');q.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');q.replace(p)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def append(p,x):
    with Path(p).open('a') as f:f.write(json.dumps(x,allow_nan=False)+'\n')
def settings():
    cfg=read(EXP/'role_robust_training_v1/config.json')
    cfg.update(version='role_condition_coverage_v1',arms=['narrow','broad'],updates=6000,
        checkpoint_updates=[1000,3000,6000],validation_updates=[0,250,500,1000,3000,6000],
        expected_updates=36000,expected_chains=528,evaluation_updates=[6000],selection_seed=2026102121,
        graph_seed=2026102122,augmentation_seed=2026102123,
        total_train_graphs_per_arm=48,wide_graphs_per_condition=4,narrow_graphs_per_condition=24,
        validation_graphs_per_condition=2,graph_attempt_cap=2000,
        forbidden=['corruption','extra_features','new_losses','probability_control','matching','repair','approximate_readout'],
        budget_seconds=10800,learning_budget_seconds=10800,training_device='cpu',parallel_workers=6,parallel_seed_workers=6,
        remaining_budget_fraction_gate=.8,cost_safety_factor=1.2,audit_delivery_reserve_seconds=360)
    cfg.pop('perturbation')
    for key in list(cfg):
        if key.startswith('holdout_'):cfg.pop(key)
    return cfg
def check():
    if time.time()>read(RESULT/'activation.json')['deadline_epoch']-30:raise TimeoutError('unique3h deadline; save/exit reserve')
    if (RESULT/'STOP').exists():raise InterruptedError('STOP')
def source_manifest():
    result=base.file_manifest()
    for directory in (HERE,EXP/'role_unseen_target_validation_v1',EXP/'role_target_scalability_v1'):
        for p in directory.glob('*'):
            if p.is_file() and p.suffix in ('.py','.json','.md'):result[str(p.relative_to(EXP))]=sha(p)
    return result
def check_binding():
    f=read(RESULT/'freeze.json')
    for n,h in f['source'].items():assert sha(EXP/n)==h,n
    for n,h in f['inputs'].items():assert sha(RESULT/n)==h,n
    assert f['config']==settings()
    return f
def target(x):return base.RoleTarget(tuple(tuple((tuple(v),int(n)) for v,n in rows) for rows in x['roles']),tuple(x['type_counts']),tuple(map(tuple,x['schema'])))
def load_data():
    p=read(RESULT/'panel.json');targets={n:target(x) for n,x in read(RESULT/'conditions.json')['targets'].items()}
    records=[dict(id=r['id'],condition=r['condition'],split=r['split'],arm=r['arm'],
             case=base.c.make_case(np.asarray(r['types']),np.asarray(r['edges']).T,tuple(map(tuple,p['schema'])))) for r in p['graphs']]
    return records,targets
def make_model(case,cfg,seed):
    base.c.seed_all(seed)
    h=read(EXP/'role_specialization_v1_results/evidence/learning_run01/weight_hashes.json')[f'seed{seed}/capacity_C_initial.pt']
    p=INITIAL/f'seed{seed}/capacity_C_initial.pt';assert sha(p)==h
    model=base.CapacityDenoiser(4,10,family_pairs=case['space'].pairs,prior=case['prior'],T=64,**cfg['model'])
    state=torch.load(p,map_location='cpu',weights_only=True)
    model.load_state_dict({k:v.clone() for k,v in state.items()},strict=True)
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in state.items())
    return model,dict(source=str(p),source_sha256=h,tensors=len(state),strict_tensor_copy=True,
                       parameters=sum(p.numel() for p in model.parameters()))
def inputs_for(records,targets,arm,seed,u,cfg):
    names=read(RESULT/'conditions.json')['training'][arm]
    # Uniform condition draw, shared scalar source; graph/augmentation/noise independent streams.
    selector=np.random.default_rng([cfg['selection_seed'],seed,u])
    uniform=float(selector.random());name=names[min(int(uniform*len(names)),len(names)-1)]
    pool=[r for r in records if r['split']=='train' and r['arm']==arm and r['condition']==name]
    k=int(selector.integers(len(pool)));rec=pool[k]
    case=base.augment(rec['case'],np.random.default_rng([cfg['augmentation_seed'],seed,u]))
    inp=[base.c.train_input(case,64,4,seed,2*u+j,np.random.default_rng([cfg['input_seed'],seed,u,j])) for j in (0,1)]
    query=[]
    for x in inp:
        hh=hashlib.sha256()
        for k in ('ids','f','query'):hh.update(np.ascontiguousarray(x[k]).tobytes())
        query.append(hh.hexdigest())
    return name,case,inp,dict(graph_id=rec['id'],condition_uniform=uniform,
        augmentation_seed=[cfg['augmentation_seed'],seed,u],noise_seeds=[[cfg['input_seed'],seed,u,j] for j in (0,1)],
        t=[x['t'] for x in inp],query_sha256=query)
def stat(v):
    r=old_stat([float(x) if x is not None else None for x in v]);r['values']=v;return r

def error_stats(metrics,targ):
    e=target_errors(metrics,targ)
    for i,name in enumerate(NAMES):
        a=metrics['joint_relation_degree'][name];b={','.join(map(str,v)):n for v,n in targ['roles'][i]}
        e[name]['joint_TV']=sum(abs(a.get(k,0)-b.get(k,0)) for k in a.keys()|b.keys())/(2*targ['type_counts'][i])
    return e
