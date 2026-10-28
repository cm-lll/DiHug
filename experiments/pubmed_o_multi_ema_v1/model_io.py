"""Only construction/loading glue. Scientific model and initial bytes unchanged."""
import sys,pathlib,time
ROOT=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'package'))
import torch
from prepare import load_case
from model_variants import StructuralDenoiser
from experiment_contract import reset_training_streams,initialization
from runtime_contract import initial_path,asset,OLD_SEEDS,NEW_SEEDS
from probe import equal
from hsgd.bw_preflight.common import environment

def configure():
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True,warn_only=False)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1 or '2080 Ti' not in torch.cuda.get_device_name(0):raise RuntimeError('one RTX2080Ti required')
    return environment('cuda')
def fresh(seed,overrides=None):
    if seed in NEW_SEEDS:
        model,case,cfg,proof=initialization('PubMed','O',seed,None)
        model=model.cuda();rng,rng_receipt=reset_training_streams(seed,'PubMed','cuda')
        opt=torch.optim.AdamW(model.parameters(),**{k:v for k,v in cfg['optimizer'].items() if k!='type'})
        return model,opt,case,dict(cfg,seed=seed),rng,dict(new_scientific_initial=True,construction=proof,rng_reset=rng_receipt,optimizer='new empty AdamW')
    if seed not in OLD_SEEDS:raise ValueError('unfrozen seed')
    path=initial_path(seed,overrides);p=torch.load(path,map_location='cpu');case,cfg=load_case('PubMed')
    if p['updates']!=0 or p['binding']['arm']!='O' or p['binding']['config']['seed']!=seed:raise ValueError('not the bound O scientific initial')
    if p['binding']['dataset']['sha256']!=cfg['datasets']['PubMed']['sha256']:raise ValueError('initial data binding')
    for key in ['model','optimizer','T','rounds','precision','tf32','schedule_offset']:
        if p['binding']['config'][key]!=cfg[key]:raise ValueError('initial scientific config '+key)
    model=StructuralDenoiser(case,cfg,'O');model.load_state_dict(p['model'],strict=True)
    if not equal(model.state_dict(),p['model']):raise ValueError('initial load differs')
    model=model.cuda();rng,rng_receipt=reset_training_streams(seed,'PubMed','cuda')
    opt=torch.optim.AdamW(model.parameters(),**{k:v for k,v in cfg['optimizer'].items() if k!='type'})
    return model,opt,case,dict(cfg,seed=seed),rng,dict(initial_sha256=asset(seed)['initial_sha256_from_sealed_receipt'],rng_reset=rng_receipt,optimizer='new empty AdamW; no historical state')
