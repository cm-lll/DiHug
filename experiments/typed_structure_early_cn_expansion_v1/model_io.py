"""Sealed D theta0 migration and paired fresh E early/late initialization."""
import sys,pathlib,hashlib,json
ROOT=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'package'))
import torch
from prepare import load_case
from model_variants import StructuralDenoiser,copy_common
from experiment_contract import reset_training_streams,read_dictionary
from role_dictionary import seed32
from runtime_contract import dictionary_expected,TRAIN_SEEDS,verify_assets,read,sha,PROTOCOL
from hsgd.bw_preflight.common import environment
from early_cn import install
def configure():
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True,warn_only=False)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1 or '2080 Ti' not in torch.cuda.get_device_name(0):raise RuntimeError('one RTX2080Ti required')
    return environment('cuda')
def tensor_hash(x):return hashlib.sha256(x.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
def identity(dataset,arm,seed):
    matches=[t for t in PROTOCOL['tasks'] if (t['dataset'],t['arm'],t['train_seed'])==(dataset,arm,seed)]
    if len(matches)!=1:raise ValueError('unfrozen identity')
    return matches[0]
def sealed(dataset):
    b=read(ROOT/'baseline_bindings.json')['initial'][dataset]
    if sha(b['path'])!=b['sha256']:raise ValueError('sealed D theta0 hash')
    p=torch.load(b['path'],map_location='cpu')
    if p['updates']!=0 or p['ema']['updates']!=0 or p['optimizer']['state']:raise ValueError('theta0 only')
    if (p['binding']['dataset'],p['binding']['arm'],p['binding']['train_seed'],p['binding']['source_sha256'])!=(dataset,'D',b['train_seed'],b['source_sha256']):raise ValueError('sealed D identity/source')
    return p,b
def new_late(dataset,seed,case,cfg,basis):
    # Same construction sequence as reviewed 472e533 model_io.construct(E).
    torch.manual_seed(seed);anchor=StructuralDenoiser(case,cfg,'A')
    module_seed=seed32(seed,dataset=dataset,arm='E',purpose='new_module_initialization')
    torch.manual_seed(module_seed);donor=StructuralDenoiser(case,cfg,'E',basis);copy_common(anchor,donor)
    torch.manual_seed(module_seed);late=StructuralDenoiser(case,cfg,'E',basis);copy_common(donor,late)
    return late,dict(anchor_seed=seed,donor_and_target_module_seed=module_seed,sequence='A anchor -> E donor copied from A -> E target copied from donor')
def construct(dataset,arm,seed):
    task=identity(dataset,arm,seed);verify_assets();case,cfg=load_case(dataset);cfg=dict(cfg,seed=seed)
    if task['model']=='D':
        p,b=sealed(dataset)
        if cfg!=p['binding']['config']:raise ValueError('D scientific config differs')
        torch.manual_seed(seed);model=StructuralDenoiser(case,cfg,'D')
        if not torch.equal(model.prior,p['model']['prior']):raise ValueError('D prior')
        model.load_state_dict(p['model'],strict=True);source=dict(sealed_D=b)
    else:
        basis,receipt=read_dictionary(ROOT/'dictionaries',dataset,dictionary_expected(dataset))
        model,source=new_late(dataset,seed,case,cfg,basis)
        source.update(fresh=True,old_training_weights_used=False,basis_sha256=receipt['basis_sha256'])
    original={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    original_hashes={k:tensor_hash(v) for k,v in original.items()}
    old_count=sum(x.numel() for x in model.parameters())
    if task['fusion']=='early':install(model)
    proof={}
    for k,v in model.state_dict().items():
        ref=original.get(k)
        proof[k]=bool(torch.equal(v.cpu(),ref[:,:739] if k=='joint_head.0.weight' and task['fusion']=='early' else ref)) if ref is not None else bool(k=='cn_edge_projection.weight' and torch.count_nonzero(v)==0)
    if not all(proof.values()):raise ValueError('actual tensor copy mismatch')
    count=sum(x.numel() for x in model.parameters());delta=-2368 if task['fusion']=='early' else 0
    if count-old_count!=delta:raise ValueError('parameter delta')
    if task['model']=='D' and any('role_' in k for k in model.state_dict()):raise ValueError('D must have no role parameters/buffers')
    if task['fusion']=='late' and hasattr(model,'cn_edge_projection'):raise ValueError('late must have no early projection')
    return model,case,cfg,dict(task=task,source=source,late_initial_tensor_sha256=original_hashes,late_initial_content_sha256=hashlib.sha256(json.dumps(original_hashes,sort_keys=True).encode()).hexdigest(),copied_exact=proof,parameters=count,original_parameters=old_count,parameter_delta=delta,tensor_sha256={k:tensor_hash(v) for k,v in model.state_dict().items()},engineering_weights_used=False)
def fresh(dataset,arm,seed,device='cuda'):
    from probe import rng_get,equal
    model,case,cfg,proof=construct(dataset,arm,seed);model=model.to(device)
    opt=torch.optim.AdamW(model.parameters(),**{k:v for k,v in cfg['optimizer'].items() if k!='type'})
    if {id(p) for g in opt.param_groups for p in g['params']}!={id(p) for p in model.parameters() if p.requires_grad}:raise AssertionError('optimizer coverage')
    rng,rr=reset_training_streams(seed,dataset,device)
    if proof['task']['model']=='D':
        old,_=sealed(dataset);now=rng_get(rng,device)
        for key in now:
            if key=='device' and device=='cpu':continue
            if not equal(now[key],old['rng'][key]):raise ValueError('D paired RNG '+key)
    return model,opt,case,cfg,rng,dict(construction=proof,rng_reset=rr,optimizer='new empty AdamW')
