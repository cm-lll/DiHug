import pathlib,json,hashlib,os,time
SOURCE=pathlib.Path('/data2/lyh/gnn_project/DiHug/research_branches/count_aux_length_run/experiments/typed_structure_count_aux_length_v1')
PARENT=pathlib.Path('/data2/lyh/gnn_project/DiHug/results/count_aux_length_20261029T231348/run')
OUT=pathlib.Path('/data2/lyh/gnn_project/DiHug/results/imdb_count_aux_u6000_20261030T050209')
HERE=pathlib.Path(__file__).parent
START=1793307729.;SCIENCE=1793320200.;DEADLINE=1793322000.
SOURCE_HASH='c0f5d92db2f891ebde70befb4e0e29adc50bcb632bdeff5866173d79aa56ed16'
CELLS=[(2026102911,'CE'),(2026102911,'CE_COUNT'),(2026102931,'CE'),(2026102931,'CE_COUNT')]
DEVICES=['GPU-e2284395-dd00-1e74-3eed-7f406fa69d5c','GPU-2db6cf6a-a8b5-58af-4e89-22bfc863a0fb','GPU-cddbf2e5-1af7-7219-ff58-38e6f4c44f69','GPU-360156ae-7a75-37f4-eaed-9a5500275409']
def read(p):return json.loads(pathlib.Path(p).read_text())
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def save(p,x):
 p=pathlib.Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(x,indent=2,allow_nan=False));os.replace(tmp,p)
def identity(seed,arm):return f'IMDB_s{seed}_{arm}'
def jobs(stages=(6000,),external=False):
 out=[]
 for i,(seed,arm) in enumerate(CELLS):
  cell=identity(seed,arm)
  for u in stages:
   for z in (2026102821,2026102822):
    d=f'{cell}/u{u}/ema_0.99/sample_{z}'
    if external and u==3000:d=str(PARENT/d)
    out.append(dict(dataset='IMDB',arm=arm,train_seed=seed,slot=i+4,model='E',fusion='early',id=f'{cell}_u{u}_ema_0.99_{z}',mode='sample',readout='ema_0.99',sample_seed=z,checkpoint=u,directory=d,T=1000,depends_on=cell))
 return out
def train_jobs():return [dict(dataset='IMDB',arm=a,train_seed=s,slot=i+4,model='E',fusion='early',id=identity(s,a),mode='train',directory=identity(s,a),updates=6000,resume_updates=3000) for i,(s,a) in enumerate(CELLS)]
def source_binding():
 files={str(p.relative_to(HERE)):sha(p) for p in sorted(HERE.rglob('*')) if p.is_file() and p.suffix in ('.py','.json') and '__pycache__' not in p.parts}
 files['scientific_parent_combined']=SOURCE_HASH
 return files,hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest()
def own_binding():
 f,h=source_binding();return dict(files=f,sha256=h)
def check(science=True):
 if time.time()>=(SCIENCE if science else DEADLINE):raise TimeoutError('IMDB science stop08:30; delivery09:00')
 if (OUT/'STOP').exists():raise InterruptedError('independent IMDB STOP')
