"""Paired fresh A anchor and E feature donor; no trained checkpoint is used."""
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
def configure():
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True,warn_only=False)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1 or '2080 Ti' not in torch.cuda.get_device_name(0):raise RuntimeError('one RTX2080Ti required')
    return environment('cuda')
def tensor_hash(x):return hashlib.sha256(x.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
def construct(dataset,arm,seed):
    if TRAIN_SEEDS.get(dataset)!=seed or arm not in ARMS:raise ValueError('unfrozen initialization identity')
    verify_assets();case,cfg=load_case(dataset)
    basis,receipt=read_dictionary(ROOT/'dictionaries',dataset,dictionary_expected(dataset))
    torch.manual_seed(seed);anchor=StructuralDenoiser(case,cfg,'A')
    donor_seed=seed32(seed,dataset=dataset,arm='E',purpose='new_module_initialization')
    torch.manual_seed(donor_seed);donor=StructuralDenoiser(case,cfg,'E',basis);copy_common(anchor,donor)
    module_seed=seed32(seed,dataset=dataset,arm=arm,purpose='new_module_initialization')
    torch.manual_seed(module_seed);model=StructuralDenoiser(case,cfg,arm,basis if arm in ('C','E') else None)
    copied=copy_common(donor,model)
    proof=dict(seed=seed,module_seed=module_seed,donor_seed=donor_seed,anchor='fresh A',donor='fresh E with A common tensors copied',copied=copied,
        tensor_sha256={k:tensor_hash(v) for k,v in model.state_dict().items()},parameters=sum(p.numel() for p in model.parameters()),
        dictionary_sha256=receipt['basis_sha256'],K=receipt['K'],copied_exact=True,engineering_weights_used=False)
    return model,case,dict(cfg,seed=seed),proof
def fresh(dataset,arm,seed,device='cuda'):
    model,case,cfg,proof=construct(dataset,arm,seed);model=model.to(device)
    rng,rr=reset_training_streams(seed,dataset,device)
    opt=torch.optim.AdamW(model.parameters(),**{k:v for k,v in cfg['optimizer'].items() if k!='type'})
    if {id(p) for g in opt.param_groups for p in g['params']}!={id(p) for p in model.parameters() if p.requires_grad}:raise AssertionError('optimizer coverage')
    return model,opt,case,cfg,rng,dict(construction=proof,rng_reset=rr,optimizer='new empty AdamW')
