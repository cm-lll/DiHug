"""Bound output-objective runtime; unchanged inputs, CE coefficient and reverse kernel."""
import os, json, time, hashlib, copy, datetime
from pathlib import Path
from . import science as g
torch,np=g.torch,g.np
SOURCE=g.ROOT; ROOT=Path(__file__).resolve().parents[1]; SCI=g.SCI
ASSETS=Path(os.environ['STRUCTURE_ASSETS']).resolve()
OUT=Path(os.environ['STRUCTURE_RUN']).resolve()
BASE=ASSETS/'results/cn_multiseed_20261104T042546'
CONTROL=ASSETS/'results/pb_degree_formal_20261106T170600'
if OUT.parent!=ASSETS/'results' or not OUT.name.startswith('structure_count_unit_extension_'):raise ValueError('isolated output required')
SAMPLES=[2026102821,2026102822,2026110231,2026110232]
TASKS=[f'{ds}_{s}_{arm}' for ds,s in [('IMDB',2026110811),('PubMed',2026110821)] for arm in ['PB','UNIT']]
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
        if sha(Path(p))!=h:raise ValueError('frozen source changed '+p)
def load(task):
    row=next(x for x in read(OUT/'tasks.json') if x['task_id']==task)
    cfg=row['config'];ds=row['dataset'];entry=cfg['datasets'][ds]
    data=ASSETS/SCI.relative_to(SOURCE)/'package'/entry['file']
    if sha(data)!=entry['sha256']:raise ValueError('data hash')
    with np.load(data,allow_pickle=False) as z:
        case=g.common.make_case(z['types'],z['edges'].T,tuple(map(tuple,z['schema'])))
        assert np.array_equal(case['space'].encode(z['edges'].T)[0]+1,z['relations'])
    parent=Path(row['initial']);assert sha(parent)==row['initial_sha256']
    z=torch.load(parent,map_location='cpu');b=z['binding']
    if z['updates']!=0 or z['optimizer']['state'] or z['ema']['updates']!=0:raise ValueError('untrained complete boundary required')
    if b['dataset']!=ds or b['train_seed']!=row['train_seed'] or b['arm']!='A' or b['config']!=cfg:raise ValueError('parent identity/config')
    original=g.StructuralDenoiser(case,cfg,'A');original.load_state_dict(z['model'],strict=True)
    model=g.StructuralDenoiser(case,cfg,'A').cuda();model.load_state_dict(z['model'],strict=True)
    for n,v in original.state_dict().items():
        if not torch.equal(v.cpu(),model.state_dict()[n].cpu()):raise ValueError('common initial parameter/buffer '+n)
    del original
    opt=torch.optim.AdamW(model.parameters(),**{k:v for k,v in cfg['optimizer'].items() if k!='type'})
    ema=g.ParameterEMA(model);rng=np.random.default_rng(1);g.rng_restore(z['rng'],rng,'cpu')
    if g.digest(g.rng_capture(rng,'cpu'))!=g.digest(z['rng']):raise ValueError('sealed CPU/main-input RNG')
    # New seeds have no historical CUDA state. Explicitly initialize their paired
    # local CUDA stream after construction, then seal all streams at update zero.
    torch.cuda.manual_seed(row['CUDA_stream_seed'])
    boundary=g.TrainingBoundary(model,opt,ema,rng,'cuda',0)
    from .objective import objective_version,PROPOSED_WEIGHT
    from .objective import make_objective
    objective=make_objective(case,cfg,ds,row['arm']);configuration='A_PB_late' if row['arm']=='PB' else 'A_PB_UNIT_late'
    binding=dict(output_objective=dict(version=objective_version(row['arm']),weight=PROPOSED_WEIGHT,target=objective.objective_binding()),dataset=ds,train_seed=row['train_seed'],arm=configuration[:-5],configuration=configuration,
      architecture_version='original_A_late',config=cfg,updates=3000,inputs_per_update=1,
      parent_initial_sha256=row['initial_sha256'],approval_sha256=sha(OUT/'approval.json'),source_manifest_sha256=sha(OUT/'source_binding.json'))
    return row,case,cfg,model,opt,ema,rng,boundary,binding
