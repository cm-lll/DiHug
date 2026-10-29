"""Both loss arms copy the same sealed untrained E-early state; new AdamW."""
import sys,pathlib,hashlib
ROOT=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'package'))
import torch
from prepare import load_case
from model_variants import StructuralDenoiser
from experiment_contract import reset_training_streams,read_dictionary
from runtime_contract import dictionary_expected,verify_assets,read,sha,PROTOCOL
from hsgd.bw_preflight.common import environment
from early_cn import install
def configure():
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True,warn_only=False)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1 or '2080 Ti' not in torch.cuda.get_device_name(0):raise RuntimeError('one RTX2080Ti required')
    return environment('cuda')
def tensor_hash(x):return hashlib.sha256(x.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
def fresh(dataset,arm,seed,device='cuda'):
    from probe import rng_get,equal
    if (dataset,arm,seed) not in [(t['dataset'],t['arm'],t['train_seed']) for t in PROTOCOL['tasks']]:raise ValueError('unfrozen identity')
    verify_assets();case,cfg=load_case(dataset);cfg=dict(cfg,seed=seed)
    b=read(ROOT/'initial_bindings.json')[f'{dataset}_{seed}']
    if sha(b['path'])!=b['sha256']:raise ValueError('theta0 bytes')
    p=torch.load(b['path'],map_location='cpu')
    if p['updates']!=0 or p['ema']['updates']!=0 or p['optimizer']['state']:raise ValueError('only untrained theta0')
    if (p['binding']['dataset'],p['binding']['train_seed'])!=(dataset,seed) or p['binding']['config']!=cfg:raise ValueError('theta0 scientific configuration')
    basis,receipt=read_dictionary(ROOT/'dictionaries',dataset,dictionary_expected(dataset))
    torch.manual_seed(seed);model=install(StructuralDenoiser(case,cfg,'E',basis))
    if not torch.equal(model.prior,p['model']['prior']):raise ValueError('prior')
    model.load_state_dict(p['model'],strict=True)
    if not equal(model.state_dict(),p['model']):raise ValueError('initial tensors')
    model=model.to(device);model.count_aux_weight=0. if arm=='CE' else PROTOCOL['loss']['lambda_count']
    opt=torch.optim.AdamW(model.parameters(),**{k:v for k,v in cfg['optimizer'].items() if k!='type'})
    if {id(q) for g in opt.param_groups for q in g['params']}!={id(q) for q in model.parameters() if q.requires_grad}:raise ValueError('optimizer coverage')
    rng,rr=reset_training_streams(seed,dataset,device)
    now=rng_get(rng,device)
    for k in now:
        if device=='cpu' and k=='device':continue
        if not equal(now[k],p['rng'][k]):raise ValueError('original paired training RNG '+k)
    proof=dict(sealed_theta0=b,parameters=sum(q.numel() for q in model.parameters()),
               tensor_sha256={k:tensor_hash(v) for k,v in model.state_dict().items()},
               copied_exact=True,new_empty_optimizer=True,rng_reset=rr,engineering_weights_used=False)
    return model,opt,case,cfg,rng,proof
