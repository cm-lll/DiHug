"""Bound output-objective runtime; unchanged inputs, CE coefficient and reverse kernel."""
import os, json, time, hashlib, copy, datetime
from pathlib import Path
from experiments.structure_input_v1 import gpu_engineering as g
torch,np=g.torch,g.np
ROOT=g.ROOT; SCI=g.SCI
ASSETS=Path(os.environ['STRUCTURE_ASSETS']).resolve()
OUT=Path(os.environ['STRUCTURE_RUN']).resolve()
BASE=ASSETS/'results/cn_multiseed_20261104T042546'
CONTROL=ASSETS/'results/structure_input_formal_20261106T045833'
if OUT.parent!=ASSETS/'results' or not OUT.name.startswith('structure_output_formal_'):raise ValueError('isolated output required')
SAMPLES=[2026102821,2026102822,2026110231,2026110232]
TASKS=[f'{ds}_{s}_A_late' for ds,seeds in [('IMDB',[2026110411,2026110412]),('PubMed',[2026110421,2026110422])] for s in seeds]
from worker import atomic_checkpoint,validation
from integration import ema_validation,final_readouts
from ema import cpu_copy
from probe import input_hash
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def now():return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
def save(p,value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix(p.suffix+'.tmp')
    q.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n');q.replace(p)
def check():
    if (OUT/'STOP_SCIENCE').exists() or time.time()>=read(OUT/'ledger.json')['science_stop_epoch']-45:raise TimeoutError('science/save boundary')
def setup():
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.set_float32_matmul_precision('highest')
    if torch.are_deterministic_algorithms_enabled():raise ValueError('do not silently change ordinary compute mode')
def authenticate_sources():
    for p,h in read(OUT/'source_binding.json').items():
        if sha(ROOT/p)!=h:raise ValueError('frozen source changed '+p)
def load(task):
    row=next(x for x in read(OUT/'tasks.json') if x['task_id']==task)
    cfg=row['config'];ds=row['dataset'];entry=cfg['datasets'][ds]
    data=ASSETS/SCI.relative_to(ROOT)/'package'/entry['file']
    if sha(data)!=entry['sha256']:raise ValueError('data hash')
    with np.load(data,allow_pickle=False) as z:
        case=g.common.make_case(z['types'],z['edges'].T,tuple(map(tuple,z['schema'])))
        assert np.array_equal(case['space'].encode(z['edges'].T)[0]+1,z['relations'])
    parent=Path(row['initial']);assert sha(parent)==row['initial_sha256']
    z=torch.load(parent,map_location='cpu');b=z['binding']
    if z['updates']!=0 or z['optimizer']['state'] or z['ema']['updates']!=0:raise ValueError('untrained complete boundary required')
    if b['dataset']!=ds or b['train_seed']!=row['train_seed'] or b['arm']!='A' or b['config']!=cfg:raise ValueError('parent identity/config')
    original=g.StructuralDenoiser(case,cfg,'A');original.load_state_dict(z['model'],strict=True)
    model=g.StructureInputA.from_initial_A(original,case,cfg,completed_updates=0).cuda()
    for n,v in original.state_dict().items():
        if not torch.equal(v.cpu(),model.state_dict()[n].cpu()):raise ValueError('common initial parameter/buffer '+n)
    del original
    opt=torch.optim.AdamW(model.parameters(),**{k:v for k,v in cfg['optimizer'].items() if k!='type'})
    ema=g.ParameterEMA(model);rng=np.random.default_rng(1);g.rng_restore(z['rng'],rng,'cuda')
    if g.digest(g.rng_capture(rng,'cuda'))!=g.digest(z['rng']):raise ValueError('initial complete RNG')
    boundary=g.TrainingBoundary(model,opt,ema,rng,'cuda',0)
    from .role_loss import VERSION,LAMBDA,PROJECTION_SEED,RANDOM_PROJECTIONS
    binding=dict(output_objective=dict(version=VERSION,weight=LAMBDA,projection_seed=PROJECTION_SEED,random_projections=RANDOM_PROJECTIONS),dataset=ds,train_seed=row['train_seed'],arm='STRUCTURE_OUTPUT',configuration='STRUCTURE_OUTPUT_late',
      architecture_version=model.architecture_version,config=cfg,updates=3000,inputs_per_update=1,
      parent_initial_sha256=row['initial_sha256'],approval_sha256=sha(OUT/'approval.json'),source_manifest_sha256=sha(OUT/'source_binding.json'))
    return row,case,cfg,model,opt,ema,rng,boundary,binding
