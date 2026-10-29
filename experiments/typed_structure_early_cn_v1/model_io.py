"""Strict theta0 migration from the sealed, non-Dropout E run."""
import sys,pathlib,hashlib
ROOT=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'package'))
import torch
from prepare import load_case
from model_variants import StructuralDenoiser,copy_common
from experiment_contract import reset_training_streams,read_dictionary
from role_dictionary import seed32
from runtime_contract import dictionary_expected,TRAIN_SEEDS,ARMS,verify_assets
from hsgd.bw_preflight.common import environment
from early_cn import install
from runtime_contract import read,sha
def configure():
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True,warn_only=False)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1 or '2080 Ti' not in torch.cuda.get_device_name(0):raise RuntimeError('one RTX2080Ti required')
    return environment('cuda')
def tensor_hash(x):return hashlib.sha256(x.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
def sealed(dataset):
    b=read(ROOT/'baseline_bindings.json')['initial'][dataset]
    if sha(b['path'])!=b['sha256']:raise ValueError('sealed theta0 hash')
    p=torch.load(b['path'],map_location='cpu')
    if p['updates']!=0 or p['ema']['updates']!=0 or p['optimizer']['state']:raise ValueError('theta0 only')
    if (p['binding']['dataset'],p['binding']['arm'],p['binding']['train_seed'],p['binding']['source_sha256'])!=(dataset,'E',TRAIN_SEEDS[dataset],b['source_sha256']):raise ValueError('sealed identity/source')
    return p,b
def construct(dataset,arm,seed):
    if TRAIN_SEEDS.get(dataset)!=seed or arm!='E':raise ValueError('unfrozen initialization identity')
    verify_assets();case,cfg=load_case(dataset);cfg=dict(cfg,seed=seed)
    basis,receipt=read_dictionary(ROOT/'dictionaries',dataset,dictionary_expected(dataset))
    p,b=sealed(dataset)
    if cfg!=p['binding']['config']:raise ValueError('scientific config differs')
    torch.manual_seed(seed);model=StructuralDenoiser(case,cfg,'E',basis)
    for k in ('prior','role_basis','role_offsets'):
        if not torch.equal(model.state_dict()[k],p['model'][k]):raise ValueError('asset tensor '+k)
    model.load_state_dict(p['model'],strict=True)
    old_count=sum(x.numel() for x in model.parameters());install(model)
    proof={}
    for k,v in model.state_dict().items():
        ref=p['model'].get(k)
        proof[k]=bool(torch.equal(v,ref[:,:739] if k=='joint_head.0.weight' else ref)) if ref is not None else bool(k=='cn_edge_projection.weight' and torch.count_nonzero(v)==0)
    if not all(proof.values()):raise ValueError('initial tensor migration mismatch')
    count=sum(x.numel() for x in model.parameters())
    if count-old_count!=-2368:raise ValueError('parameter delta differs')
    return model,case,cfg,dict(sealed_initial=b,copied_exact=proof,parameters=count,original_parameters=old_count,parameter_delta=count-old_count,
        tensor_sha256={k:tensor_hash(v) for k,v in model.state_dict().items()},engineering_weights_used=False,head_columns='original [:,:739]; removed [739:776]')
def fresh(dataset,arm,seed,device='cuda'):
    model,case,cfg,proof=construct(dataset,arm,seed);model=model.to(device)
    opt=torch.optim.AdamW(model.parameters(),**{k:v for k,v in cfg['optimizer'].items() if k!='type'})
    if {id(p) for g in opt.param_groups for p in g['params']}!={id(p) for p in model.parameters() if p.requires_grad}:raise AssertionError('optimizer coverage')
    rng,rr=reset_training_streams(seed,dataset,device)
    from probe import rng_get,equal
    original,_=sealed(dataset);current=rng_get(rng,device)
    for key in current:
        if key=='device' and device=='cpu':continue
        if not equal(current[key],original['rng'][key]):raise ValueError('paired initial RNG '+key)
    return model,opt,case,cfg,rng,dict(construction=proof,rng_reset=rr,optimizer='new empty AdamW',original_training_rng_exact=True)
