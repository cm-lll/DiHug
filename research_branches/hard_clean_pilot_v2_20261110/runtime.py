import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '1'
import sys, json, hashlib, importlib.util
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('hard_clean_science', ROOT/'research_branches/joint_output_projected_review_20261108/runtime/science.py')
g = importlib.util.module_from_spec(spec); spec.loader.exec_module(g)
from ema import cpu_copy
from probe import input_hash
from hsgd.losses import supported_binary_ce
torch, np = g.torch, g.np

def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for x in iter(lambda: f.read(1024*1024), b''): h.update(x)
    return h.hexdigest()

def save(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)); tmp.replace(path)

def checkpoint(path, obj):
    path = Path(path); tmp = path.with_suffix('.tmp'); torch.save(obj, tmp); tmp.replace(path)

def setup():
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    assert not torch.are_deterministic_algorithms_enabled(), 'ordinary historical execution policy required'

def load(row, device='cuda'):
    assert sha(row['checkpoint']) == row['checkpoint_sha256']
    z = torch.load(row['checkpoint'], map_location='cpu'); cfg = z['binding']['config']
    assert z['updates'] == z['ema']['updates'] == 2000
    assert z['binding']['dataset'] == row['dataset'] and z['binding']['train_seed'] == row['seed']
    assert z['binding']['arm'] == 'A' and cfg['inputs_per_update'] == cfg['rounds'] == 1
    assert sha(row['data']) == row['data_sha256'] == cfg['datasets'][row['dataset']]['sha256']
    with np.load(row['data']) as data:
        case = g.common.make_case(data['types'], data['edges'].T, tuple(map(tuple, data['schema'])))
    m = g.StructuralDenoiser(case, cfg, 'A').to(device)
    o = torch.optim.AdamW(m.parameters(), **{k:v for k,v in cfg['optimizer'].items() if k != 'type'})
    e = g.ParameterEMA(m); rng = np.random.default_rng(1)
    b = restore(z, m, o, e, rng, device)
    return z, cfg, case, m, o, e, rng, b

def restore(z, m, o, e, rng, device='cuda'):
    g.restore_payload(cpu_copy(z), m, o, e, z['binding'], lambda x:g.rng_restore(x,rng,device))
    o.zero_grad(set_to_none=True); m.train()
    b = g.TrainingBoundary(m, o, e, rng, device, z['updates'])
    live = b.checkpoint(z['binding'])
    for k in ('model','optimizer','ema','rng','updates'):
        assert g.digest(live[k]) == g.digest(z[k]), 'restore mismatch '+k
    return b

